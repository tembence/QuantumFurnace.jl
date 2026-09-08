using Test
using LinearAlgebra
using Random
using QuantumFurnace

# test_helpers.jl is already included by runtests.jl

# Helper to measure allocations in a hard-scoped function context.
# @testset soft-scope can cause variable boxing (176-byte spurious allocations)
# when the function under test destructures NamedTuples with complex type params.
function _measure_matvec_allocs(ws, rho, config, ham)
    # Warmup forward
    apply_lindbladian!(ws, rho, config, ham)
    apply_lindbladian!(ws, rho, config, ham)
    allocs_fwd = @allocated apply_lindbladian!(ws, rho, config, ham)

    # Warmup adjoint
    apply_adjoint_lindbladian!(ws, rho, config, ham)
    apply_adjoint_lindbladian!(ws, rho, config, ham)
    allocs_adj = @allocated apply_adjoint_lindbladian!(ws, rho, config, ham)

    return allocs_fwd, allocs_adj
end

function _b_bohr_reference(ham::HamHam, jumps::Vector{<:JumpOp}, cfg::Config)
    dim = size(ham.data, 1)
    B = zeros(ComplexF64, dim, dim)
    f = QuantumFurnace._pick_f(cfg)
    for nu_2 in keys(ham.bohr_dict), idx in ham.bohr_dict[nu_2]
        i, j = Tuple(idx)
        for jump in jumps
            in_eb = jump.in_eigenbasis
            val = conj(in_eb[i, j])
            for col in 1:dim
                B[j, col] += val * f(ham.bohr_freqs[i, col], nu_2) * in_eb[i, col]
            end
        end
    end
    return B
end

function _make_degenerate_bohr_ham(beta_alg::Real)
    eigvals = [0.0, 1.0, 1.0, 2.0]
    matrix = diagm(ComplexF64.(eigvals))
    bohr_freqs = eigvals .- transpose(eigvals)
    gibbs_weights = exp.(-beta_alg .* eigvals)
    gibbs = Hermitian(diagm(ComplexF64.(gibbs_weights ./ sum(gibbs_weights))))
    return HamHam{Float64}(
        matrix,
        bohr_freqs,
        create_bohr_dict(bohr_freqs),
        Vector{Vector{Matrix{ComplexF64}}}(),
        Float64[],
        nothing,
        nothing,
        eigvals,
        Matrix{ComplexF64}(I, 4, 4),
        1.0,
        0.0,
        1.0,
        true,
        gibbs,
    )
end

function _shuffle_bohr_rows(ham::HamHam{T}; seed::Int=7) where {T}
    rng = MersenneTwister(seed)
    shuffled = Dict{T, Vector{CartesianIndex{2}}}()
    for (nu, indices) in ham.bohr_dict
        shuffled[nu] = shuffle(rng, copy(indices))
    end
    return HamHam{T}(
        ham.data,
        ham.bohr_freqs,
        shuffled,
        ham.base_terms,
        ham.base_coeffs,
        ham.disordering_terms,
        ham.disordering_coeffs,
        ham.eigvals,
        ham.eigvecs,
        ham.nu_min,
        ham.shift,
        ham.rescaling_factor,
        ham.periodic,
        ham.gibbs,
    )
end

# ============================================================================
# Round-trip correctness and allocation tests for Krylov matvec
# Phase 27: Core Matvec Infrastructure
# ============================================================================

@testset "Krylov Matvec" begin

    # ========================================================================
    # Testset 1: Workspace construction
    # ========================================================================
    @testset "Workspace construction" begin
        config_kms = make_config(Lindbladian(),EnergyDomain(); construction=KMS())
        ws = Workspace(config_kms, TEST_HAM, TEST_JUMPS)
        @test ws.G_left != ws.G_right
        @test size(ws.scratch.sandwich_tmp) == (DIM, DIM)
        @test size(ws.scratch.sandwich_out) == (DIM, DIM)
        @test size(ws.scratch.rho_out) == (DIM, DIM)
        @test size(ws.scratch.jump_oft) == (DIM, DIM)
        @test size(ws.scratch.bohr_component_dag) == (DIM, DIM)

        config_gns = make_config(Lindbladian(), EnergyDomain(); construction=GNS())
        ws_gns = Workspace(config_gns, TEST_HAM, TEST_JUMPS)
        @test ws_gns.G_left == ws_gns.G_right
    end

    # ========================================================================
    # Testset 2: Round-trip matvec vs dense (EnergyDomain KMS, no coherent)
    # ========================================================================
    @testset "Round-trip: matvec vs dense (EnergyDomain KMS, no coherent)" begin
        config = make_config(Lindbladian(),EnergyDomain(); construction=GNS())
        L_dense = construct_lindbladian(TEST_JUMPS, config, TEST_HAM)
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)

        # Threshold rationale: matvec is algebraically exact (same computation, different code path).
        # Error is pure FP accumulation: O(n_jumps * DIM^2 * eps) ~ 12 * 256 * 1e-16 ~ 3e-13.
        # Threshold 1e-12 gives ~3x safety margin.
        max_err = 0.0
        for _ in 1:10
            rho = Matrix(random_density_matrix(NUM_QUBITS))
            v_dense = L_dense * vec(rho)
            L_rho = apply_lindbladian!(ws, rho, config, TEST_HAM)
            err = norm(v_dense - vec(L_rho))
            @test err < 1e-12
            max_err = max(max_err, err)
        end
        @info "Matvec round-trip (EnergyDomain GNS, no coherent)" max_error=max_err n_samples=10 threshold=1e-12
    end

    # ========================================================================
    # Testset 3: Round-trip matvec vs dense (EnergyDomain KMS, with coherent)
    # ========================================================================
    @testset "Round-trip: matvec vs dense (EnergyDomain KMS, with coherent)" begin
        config = make_config(Lindbladian(),EnergyDomain(); construction=KMS())
        L_dense = construct_lindbladian(TEST_JUMPS, config, TEST_HAM)
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)

        # Threshold rationale: same as testset 2 -- algebraically exact, FP accumulation only.
        max_err = 0.0
        for _ in 1:10
            rho = Matrix(random_density_matrix(NUM_QUBITS))
            v_dense = L_dense * vec(rho)
            L_rho = apply_lindbladian!(ws, rho, config, TEST_HAM)
            err = norm(v_dense - vec(L_rho))
            @test err < 1e-12
            max_err = max(max_err, err)
        end
        @info "Matvec round-trip (EnergyDomain KMS, with coherent)" max_error=max_err n_samples=10 threshold=1e-12
    end

    # ========================================================================
    # Testset 4: Round-trip matvec vs dense (EnergyDomain GNS)
    # ========================================================================
    @testset "Round-trip: matvec vs dense (EnergyDomain GNS)" begin
        config = make_config(Lindbladian(), EnergyDomain(); construction=GNS())
        L_dense = construct_lindbladian(TEST_JUMPS, config, TEST_HAM)
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)

        # Threshold rationale: same as testset 2 -- algebraically exact, FP accumulation only.
        max_err = 0.0
        for _ in 1:10
            rho = Matrix(random_density_matrix(NUM_QUBITS))
            v_dense = L_dense * vec(rho)
            L_rho = apply_lindbladian!(ws, rho, config, TEST_HAM)
            err = norm(v_dense - vec(L_rho))
            @test err < 1e-12
            max_err = max(max_err, err)
        end
        @info "Matvec round-trip (EnergyDomain GNS)" max_error=max_err n_samples=10 threshold=1e-12
    end

    # ========================================================================
    # Testset 5: Round-trip adjoint matvec vs dense adjoint (EnergyDomain KMS)
    # ========================================================================
    @testset "Round-trip: adjoint matvec vs dense adjoint (EnergyDomain KMS)" begin
        config = make_config(Lindbladian(),EnergyDomain(); construction=KMS())
        L_dense = construct_lindbladian(TEST_JUMPS, config, TEST_HAM)
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)

        # Threshold rationale: adjoint matvec has same FP accumulation as forward matvec.
        max_err = 0.0
        for _ in 1:10
            rho = Matrix(random_density_matrix(NUM_QUBITS))
            v_adj_dense = L_dense' * vec(rho)
            L_adj_rho = apply_adjoint_lindbladian!(ws, rho, config, TEST_HAM)
            err = norm(v_adj_dense - vec(L_adj_rho))
            @test err < 1e-12
            max_err = max(max_err, err)
        end
        @info "Adjoint round-trip (EnergyDomain KMS)" max_error=max_err n_samples=10 threshold=1e-12
    end

    # ========================================================================
    # Testset 6: Adjoint duality check: tr(X' * L(Y)) == tr(L*(X)' * Y)
    # ========================================================================
    @testset "Adjoint duality check: tr(X' * L(Y)) == tr(L*(X)' * Y)" begin
        config = make_config(Lindbladian(),EnergyDomain(); construction=KMS())
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)

        # Threshold rationale: duality identity tr(X'*L(Y)) == tr(L*(X)'*Y) is exact.
        # Error is FP accumulation from two matvecs + two traces: O(DIM^2 * n_jumps * eps) * 2.
        # 1e-11 gives ~30x margin over expected ~3e-13 per-sample error.
        max_err = 0.0
        for _ in 1:5
            X = Matrix(random_density_matrix(NUM_QUBITS))
            Y = Matrix(random_density_matrix(NUM_QUBITS))

            L_Y = copy(apply_lindbladian!(ws, Y, config, TEST_HAM))
            lhs = tr(X' * L_Y)

            Lstar_X = copy(apply_adjoint_lindbladian!(ws, X, config, TEST_HAM))
            rhs = tr(Lstar_X' * Y)

            err = abs(lhs - rhs)
            @test err < 1e-11
            max_err = max(max_err, err)
        end
        @info "Adjoint duality (EnergyDomain KMS)" max_error=max_err n_samples=5 threshold=1e-11
    end

    # ========================================================================
    # Testset 7: Zero allocations in matvec hot path
    # ========================================================================
    # Transition/alpha values are now computed via dispatched 2-arg methods
    # (pick_transition(config, w) / _pick_alpha(config, nu1, nu2)) instead of
    # stored closures, eliminating the Union{Nothing, Function} boxing overhead.
    #
    # qf-in3.4 update: when `Threads.nthreads() > 1` and `n_labels >=
    # OMEGA_THREAD_THRESHOLD`, the matvec dispatches to the threaded ω-loop
    # which has fixed overhead from `Threads.@spawn` Task allocations
    # (~1 kB / spawn). The work_list and per-thread sandwich scratches are
    # pre-allocated, so the only remaining cost is the spawn itself plus the
    # `_partition_range` chunks vector. Budget below covers nthreads ≤ 8 and
    # is the price of the 2–3× wall-time speedup.
    _running_threaded() = Threads.nthreads() > 1
    MATVEC_ALLOC_BUDGET       = _running_threaded() ? 8192 : 0  # bytes (EnergyDomain / BohrDomain)
    MATVEC_ALLOC_BUDGET_NUFFT = _running_threaded() ? 8192 : 0  # bytes (TimeDomain / TrotterDomain)

    @testset "Near-zero allocations in matvec hot path" begin
        config = make_config(Lindbladian(),EnergyDomain(); construction=KMS())
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)
        rho = Matrix(random_density_matrix(NUM_QUBITS))

        # Threshold rationale: serial path must be zero-alloc; threaded path
        # incurs `Threads.@spawn` Task overhead (~1 kB × nthreads).
        allocs, allocs_adj = _measure_matvec_allocs(ws, rho, config, TEST_HAM)
        @test allocs <= MATVEC_ALLOC_BUDGET
        @info "apply_lindbladian! allocations (EnergyDomain)" allocs_bytes=allocs threshold=MATVEC_ALLOC_BUDGET
        @test allocs_adj <= MATVEC_ALLOC_BUDGET
        @info "apply_adjoint_lindbladian! allocations (EnergyDomain)" allocs_bytes=allocs_adj threshold=MATVEC_ALLOC_BUDGET
    end

    # qf-lkb.11.4: EnergyDomain matvec at the production-sweep config
    # (smooth Metropolis a=0, s=0.25) must remain on the zero-alloc fast path.
    # Different (a, s) regime than make_config (a=BETA/30, s=0.4) — pick_transition
    # branches into the same smooth-Metropolis arm (s != 0) but with tighter
    # smoothing; defensively retest the allocation budget.
    @testset "Allocation regression: EnergyDomain CKG @ a=0, s=0.25 (qf-lkb.11.4)" begin
        config = Config(;
            sim = Lindbladian(),
            domain = EnergyDomain(),
            construction = KMS(),
            num_qubits = NUM_QUBITS,
            with_linear_combination = true,
            beta = BETA,
            sigma = SIGMA,
            a = 0.0,
            s = 0.25,
            num_energy_bits = NUM_ENERGY_BITS,
            w0 = W0,
            t0 = T0,
            num_trotter_steps_per_t0 = NUM_TROTTER_STEPS_PER_T0,
        )
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)
        rho = Matrix(random_density_matrix(NUM_QUBITS))
        allocs, allocs_adj = _measure_matvec_allocs(ws, rho, config, TEST_HAM)
        # Budget rationale: serial-path 0 bytes; threaded path (qf-in3.4)
        # has @spawn Task overhead (~1 kB × nthreads).
        budget = MATVEC_ALLOC_BUDGET
        @test allocs <= budget
        @test allocs_adj <= budget
        @info "EnergyDomain CKG (a=0, s=0.25) allocations" forward_bytes=allocs adjoint_bytes=allocs_adj
    end

    # ========================================================================
    # Phase 28: TimeDomain round-trip and allocation tests
    # ========================================================================

    # Testset 8: Round-trip matvec vs dense (TimeDomain KMS, with coherent)
    @testset "Round-trip: matvec vs dense (TimeDomain KMS, with coherent)" begin
        config = make_config(Lindbladian(),TimeDomain(); construction=KMS())
        L_dense = construct_lindbladian(TEST_JUMPS, config, TEST_HAM)
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)

        # Threshold rationale: TimeDomain uses NUFFT but dense reference uses same OFT path.
        # Error is FP accumulation only. Same 1e-12 threshold as EnergyDomain.
        max_err = 0.0
        for _ in 1:10
            rho = Matrix(random_density_matrix(NUM_QUBITS))
            v_dense = L_dense * vec(rho)
            L_rho = apply_lindbladian!(ws, rho, config, TEST_HAM)
            err = norm(v_dense - vec(L_rho))
            @test err < 1e-12
            max_err = max(max_err, err)
        end
        @info "Matvec round-trip (TimeDomain KMS, with coherent)" max_error=max_err n_samples=10 threshold=1e-12
    end

    # Testset 9: Round-trip matvec vs dense (TimeDomain GNS)
    @testset "Round-trip: matvec vs dense (TimeDomain GNS)" begin
        config = make_config(Lindbladian(), TimeDomain(); construction=GNS())
        L_dense = construct_lindbladian(TEST_JUMPS, config, TEST_HAM)
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)

        # Threshold rationale: same as testset 8 -- NUFFT path, FP accumulation only.
        max_err = 0.0
        for _ in 1:10
            rho = Matrix(random_density_matrix(NUM_QUBITS))
            v_dense = L_dense * vec(rho)
            L_rho = apply_lindbladian!(ws, rho, config, TEST_HAM)
            err = norm(v_dense - vec(L_rho))
            @test err < 1e-12
            max_err = max(max_err, err)
        end
        @info "Matvec round-trip (TimeDomain GNS)" max_error=max_err n_samples=10 threshold=1e-12
    end

    # Testset 10: Round-trip adjoint matvec vs dense adjoint (TimeDomain KMS)
    @testset "Round-trip: adjoint matvec vs dense adjoint (TimeDomain KMS)" begin
        config = make_config(Lindbladian(),TimeDomain(); construction=KMS())
        L_dense = construct_lindbladian(TEST_JUMPS, config, TEST_HAM)
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)

        # Threshold rationale: adjoint NUFFT path, same FP accumulation as forward.
        max_err = 0.0
        for _ in 1:10
            rho = Matrix(random_density_matrix(NUM_QUBITS))
            v_adj_dense = L_dense' * vec(rho)
            L_adj_rho = apply_adjoint_lindbladian!(ws, rho, config, TEST_HAM)
            err = norm(v_adj_dense - vec(L_adj_rho))
            @test err < 1e-12
            max_err = max(max_err, err)
        end
        @info "Adjoint round-trip (TimeDomain KMS)" max_error=max_err n_samples=10 threshold=1e-12
    end

    # Testset 11: Near-zero allocations in matvec hot path (TimeDomain)
    @testset "Near-zero allocations in matvec hot path (TimeDomain)" begin
        config = make_config(Lindbladian(),TimeDomain(); construction=KMS())
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)
        rho = Matrix(random_density_matrix(NUM_QUBITS))
        # Threshold rationale: NUFFT hot-path must also be zero-allocation for Krylov performance.
        allocs, allocs_adj = _measure_matvec_allocs(ws, rho, config, TEST_HAM)
        @test allocs <= MATVEC_ALLOC_BUDGET_NUFFT
        @info "apply_lindbladian! allocations (TimeDomain)" allocs_bytes=allocs threshold=MATVEC_ALLOC_BUDGET_NUFFT
        @test allocs_adj <= MATVEC_ALLOC_BUDGET_NUFFT
        @info "apply_adjoint_lindbladian! allocations (TimeDomain)" allocs_bytes=allocs_adj threshold=MATVEC_ALLOC_BUDGET_NUFFT
    end

    # ========================================================================
    # Phase 28: TrotterDomain round-trip and allocation tests
    # ========================================================================

    # Testset 12: Round-trip matvec vs dense (TrotterDomain KMS, with coherent)
    @testset "Round-trip: matvec vs dense (TrotterDomain KMS, with coherent)" begin
        config = make_config(Lindbladian(),TrotterDomain(); construction=KMS())
        L_dense = construct_lindbladian(TEST_TROTTER_JUMPS, config, TEST_HAM; trotter=TEST_TROTTER)
        ws = Workspace(config, TEST_HAM, TEST_TROTTER_JUMPS; trotter=TEST_TROTTER)

        # Threshold rationale: TrotterDomain uses Trotter eigenbasis but same OFT arithmetic.
        # FP accumulation only, same 1e-12 threshold.
        max_err = 0.0
        for _ in 1:10
            rho = Matrix(random_density_matrix(NUM_QUBITS))
            v_dense = L_dense * vec(rho)
            L_rho = apply_lindbladian!(ws, rho, config, TEST_HAM)
            err = norm(v_dense - vec(L_rho))
            @test err < 1e-12
            max_err = max(max_err, err)
        end
        @info "Matvec round-trip (TrotterDomain KMS, with coherent)" max_error=max_err n_samples=10 threshold=1e-12
    end

    # Testset 13: Round-trip matvec vs dense (TrotterDomain GNS)
    @testset "Round-trip: matvec vs dense (TrotterDomain GNS)" begin
        config = make_config(Lindbladian(), TrotterDomain(); construction=GNS())
        L_dense = construct_lindbladian(TEST_TROTTER_JUMPS, config, TEST_HAM; trotter=TEST_TROTTER)
        ws = Workspace(config, TEST_HAM, TEST_TROTTER_JUMPS; trotter=TEST_TROTTER)

        # Threshold rationale: same as testset 12 -- Trotter eigenbasis, FP accumulation only.
        max_err = 0.0
        for _ in 1:10
            rho = Matrix(random_density_matrix(NUM_QUBITS))
            v_dense = L_dense * vec(rho)
            L_rho = apply_lindbladian!(ws, rho, config, TEST_HAM)
            err = norm(v_dense - vec(L_rho))
            @test err < 1e-12
            max_err = max(max_err, err)
        end
        @info "Matvec round-trip (TrotterDomain GNS)" max_error=max_err n_samples=10 threshold=1e-12
    end

    # Testset 14: Round-trip adjoint matvec vs dense adjoint (TrotterDomain KMS)
    @testset "Round-trip: adjoint matvec vs dense adjoint (TrotterDomain KMS)" begin
        config = make_config(Lindbladian(),TrotterDomain(); construction=KMS())
        L_dense = construct_lindbladian(TEST_TROTTER_JUMPS, config, TEST_HAM; trotter=TEST_TROTTER)
        ws = Workspace(config, TEST_HAM, TEST_TROTTER_JUMPS; trotter=TEST_TROTTER)

        # Threshold rationale: adjoint Trotter path, same FP accumulation as forward.
        max_err = 0.0
        for _ in 1:10
            rho = Matrix(random_density_matrix(NUM_QUBITS))
            v_adj_dense = L_dense' * vec(rho)
            L_adj_rho = apply_adjoint_lindbladian!(ws, rho, config, TEST_HAM)
            err = norm(v_adj_dense - vec(L_adj_rho))
            @test err < 1e-12
            max_err = max(max_err, err)
        end
        @info "Adjoint round-trip (TrotterDomain KMS)" max_error=max_err n_samples=10 threshold=1e-12
    end

    # Testset 15: Near-zero allocations in matvec hot path (TrotterDomain)
    @testset "Near-zero allocations in matvec hot path (TrotterDomain)" begin
        config = make_config(Lindbladian(),TrotterDomain(); construction=KMS())
        ws = Workspace(config, TEST_HAM, TEST_TROTTER_JUMPS; trotter=TEST_TROTTER)
        rho = Matrix(random_density_matrix(NUM_QUBITS))
        # Threshold rationale: Trotter NUFFT hot-path must also be zero-allocation.
        allocs, allocs_adj = _measure_matvec_allocs(ws, rho, config, TEST_HAM)
        @test allocs <= MATVEC_ALLOC_BUDGET_NUFFT
        @info "apply_lindbladian! allocations (TrotterDomain)" allocs_bytes=allocs threshold=MATVEC_ALLOC_BUDGET_NUFFT
        @test allocs_adj <= MATVEC_ALLOC_BUDGET_NUFFT
        @info "apply_adjoint_lindbladian! allocations (TrotterDomain)" allocs_bytes=allocs_adj threshold=MATVEC_ALLOC_BUDGET_NUFFT
    end

    # ========================================================================
    # Phase 28: BohrDomain round-trip and duality tests
    # ========================================================================

    # Testset 16: Round-trip matvec vs dense (BohrDomain KMS, with coherent)
    @testset "Round-trip: matvec vs dense (BohrDomain KMS, with coherent)" begin
        config = make_config(Lindbladian(),BohrDomain(); construction=KMS())
        L_dense = construct_lindbladian(TEST_JUMPS, config, TEST_HAM)
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)

        # Threshold rationale: BohrDomain has different loop structure but same algebraic exactness.
        # FP accumulation only, same 1e-12 threshold.
        max_err = 0.0
        for _ in 1:10
            rho = Matrix(random_density_matrix(NUM_QUBITS))
            v_dense = L_dense * vec(rho)
            L_rho = apply_lindbladian!(ws, rho, config, TEST_HAM)
            err = norm(v_dense - vec(L_rho))
            @test err < 1e-12
            max_err = max(max_err, err)
        end
        @info "Matvec round-trip (BohrDomain KMS, with coherent)" max_error=max_err n_samples=10 threshold=1e-12
    end

    # Testset 17: Round-trip matvec vs dense (BohrDomain GNS)
    @testset "Round-trip: matvec vs dense (BohrDomain GNS)" begin
        config = make_config(Lindbladian(), BohrDomain(); construction=GNS())
        L_dense = construct_lindbladian(TEST_JUMPS, config, TEST_HAM)
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)

        # Threshold rationale: same as testset 16 -- BohrDomain, FP accumulation only.
        max_err = 0.0
        for _ in 1:10
            rho = Matrix(random_density_matrix(NUM_QUBITS))
            v_dense = L_dense * vec(rho)
            L_rho = apply_lindbladian!(ws, rho, config, TEST_HAM)
            err = norm(v_dense - vec(L_rho))
            @test err < 1e-12
            max_err = max(max_err, err)
        end
        @info "Matvec round-trip (BohrDomain GNS)" max_error=max_err n_samples=10 threshold=1e-12
    end

    # Testset 18: Round-trip adjoint matvec vs dense adjoint (BohrDomain KMS)
    @testset "Round-trip: adjoint matvec vs dense adjoint (BohrDomain KMS)" begin
        config = make_config(Lindbladian(),BohrDomain(); construction=KMS())
        L_dense = construct_lindbladian(TEST_JUMPS, config, TEST_HAM)
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)

        # Threshold rationale: adjoint BohrDomain path, same FP accumulation as forward.
        max_err = 0.0
        for _ in 1:10
            rho = Matrix(random_density_matrix(NUM_QUBITS))
            v_adj_dense = L_dense' * vec(rho)
            L_adj_rho = apply_adjoint_lindbladian!(ws, rho, config, TEST_HAM)
            err = norm(v_adj_dense - vec(L_adj_rho))
            @test err < 1e-12
            max_err = max(max_err, err)
        end
        @info "Adjoint round-trip (BohrDomain KMS)" max_error=max_err n_samples=10 threshold=1e-12
    end

    # Testset 19: Adjoint duality check (BohrDomain): tr(X' * L(Y)) == tr(L*(X)' * Y)
    @testset "Adjoint duality check (BohrDomain): tr(X' * L(Y)) == tr(L*(X)' * Y)" begin
        config = make_config(Lindbladian(),BohrDomain(); construction=KMS())
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)

        # Threshold rationale: same duality identity as testset 6, BohrDomain loop structure.
        # 1e-11 gives ~30x margin over expected per-sample FP accumulation.
        max_err = 0.0
        for _ in 1:5
            X = Matrix(random_density_matrix(NUM_QUBITS))
            Y = Matrix(random_density_matrix(NUM_QUBITS))

            L_Y = copy(apply_lindbladian!(ws, Y, config, TEST_HAM))
            lhs = tr(X' * L_Y)

            Lstar_X = copy(apply_adjoint_lindbladian!(ws, X, config, TEST_HAM))
            rhs = tr(Lstar_X' * Y)

            err = abs(lhs - rhs)
            @test err < 1e-11
            max_err = max(max_err, err)
        end
        @info "Adjoint duality (BohrDomain KMS)" max_error=max_err n_samples=5 threshold=1e-11
    end

    @testset "Zero allocations and inferred hot loop (BohrDomain KMS/GNS)" begin
        system = make_dll_n3_system(BETA_ALG)
        rho = Matrix(random_density_matrix(3))

        for construction in (KMS(), GNS())
            config = make_config(
                Lindbladian(), BohrDomain(); num_qubits=3,
                construction=construction)
            ws = Workspace(config, system.ham, system.jumps)
            sc = ws.scratch
            alpha = ws.bohr_alpha

            @test length(system.jumps) == 9
            @test alpha !== nothing
            @test isapprox(
                alpha(system.ham.bohr_freqs[2, 3], system.ham.bohr_freqs[4, 5]),
                QuantumFurnace._pick_alpha(
                    config, system.ham.bohr_freqs[2, 3], system.ham.bohr_freqs[4, 5]);
                rtol=eps(Float64), atol=0.0)
            @test all(child -> size(child.bohr_component_dag) == (8, 8),
                      sc.task_scratches)

            for adjoint in (false, true)
                hot_loop_args = Tuple{
                    typeof(sc),
                    typeof(rho),
                    typeof(ws.jump_eigenbases),
                    typeof(system.ham.bohr_freqs),
                    typeof(system.ham.bohr_dict),
                    typeof(alpha),
                    Float64,
                    Val{adjoint},
                }
                hot_loop_return = Core.Compiler.return_type(
                    QuantumFurnace._apply_bohr_dissipator!, hot_loop_args)
                @test hot_loop_return === Matrix{ComplexF64}
            end

            allocs, allocs_adj = _measure_matvec_allocs(
                ws, rho, config, system.ham)
            @test allocs == 0
            @test allocs_adj == 0
            @info "BohrDomain $(typeof(construction)) matvec allocations" forward_bytes=allocs adjoint_bytes=allocs_adj
        end
    end

    # ========================================================================
    # Quick-35: Complex non-Hermitian jump operator round-trip tests
    # Validates that Krylov matvec matches dense kron convention for
    # general complex operators (where conj(J) rho J^T != J rho J').
    # ========================================================================

    # Create a single complex non-Hermitian jump operator for the 4-qubit test system
    let rng = MersenneTwister(42)
        raw_jump = randn(rng, ComplexF64, DIM, DIM) ./ sqrt(DIM)
        jump_in_eigen = TEST_HAM.eigvecs' * raw_jump * TEST_HAM.eigvecs
        # Not orthogonal, not Hermitian
        complex_jump = JumpOp(raw_jump, jump_in_eigen, false, false)
        complex_jumps = JumpOp[complex_jump]

        # Testset 20: Round-trip with complex jump (EnergyDomain forward)
        # qf-bm1 Q1: unpaired non-Hermitian jump — flagged with the
        # `allow_unpaired_nonhermitian` kwarg because this test compares
        # serial dense vs Krylov matvec on the same physics; KMS-DB is
        # not asserted here.
        @testset "Round-trip: complex jump forward (EnergyDomain)" begin
            config = make_config(Lindbladian(),EnergyDomain(); construction=KMS())
            L_dense = construct_lindbladian(complex_jumps, config, TEST_HAM;
                allow_unpaired_nonhermitian=true)
            ws = Workspace(config, TEST_HAM, complex_jumps)
            # Threshold rationale: complex non-Hermitian jump validates conj(J) convention.
            # Same algebraic exactness as real jumps, FP accumulation only.
            max_err = 0.0
            for _ in 1:10
                rho = Matrix(random_density_matrix(NUM_QUBITS))
                v_dense = L_dense * vec(rho)
                L_rho = apply_lindbladian!(ws, rho, config, TEST_HAM)
                err = norm(v_dense - vec(L_rho))
                @test err < 1e-12
                max_err = max(max_err, err)
            end
            @info "Matvec round-trip (complex jump, EnergyDomain fwd)" max_error=max_err n_samples=10 threshold=1e-12
        end

        # Testset 21: Round-trip with complex jump (EnergyDomain adjoint)
        # qf-bm1 Q1 — see preamble of testset 20.
        @testset "Round-trip: complex jump adjoint (EnergyDomain)" begin
            config = make_config(Lindbladian(),EnergyDomain(); construction=KMS())
            L_dense = construct_lindbladian(complex_jumps, config, TEST_HAM;
                allow_unpaired_nonhermitian=true)
            ws = Workspace(config, TEST_HAM, complex_jumps)
            # Threshold rationale: adjoint path for complex jumps, same FP accumulation.
            max_err = 0.0
            for _ in 1:10
                rho = Matrix(random_density_matrix(NUM_QUBITS))
                v_adj_dense = L_dense' * vec(rho)
                L_adj_rho = apply_adjoint_lindbladian!(ws, rho, config, TEST_HAM)
                err = norm(v_adj_dense - vec(L_adj_rho))
                @test err < 1e-12
                max_err = max(max_err, err)
            end
            @info "Adjoint round-trip (complex jump, EnergyDomain)" max_error=max_err n_samples=10 threshold=1e-12
        end

        # Testset 22: Adjoint duality with complex jump (EnergyDomain)
        @testset "Adjoint duality: complex jump (EnergyDomain)" begin
            config = make_config(Lindbladian(),EnergyDomain(); construction=KMS())
            ws = Workspace(config, TEST_HAM, complex_jumps)
            # Threshold rationale: duality identity with complex non-Hermitian jumps.
            # Same 1e-11 threshold as real-jump duality tests (testsets 6, 19).
            max_err = 0.0
            for _ in 1:5
                X = Matrix(random_density_matrix(NUM_QUBITS))
                Y = Matrix(random_density_matrix(NUM_QUBITS))
                L_Y = copy(apply_lindbladian!(ws, Y, config, TEST_HAM))
                lhs = tr(X' * L_Y)
                Lstar_X = copy(apply_adjoint_lindbladian!(ws, X, config, TEST_HAM))
                rhs = tr(Lstar_X' * Y)
                err = abs(lhs - rhs)
                @test err < 1e-11
                max_err = max(max_err, err)
            end
            @info "Adjoint duality (complex jump, EnergyDomain)" max_error=max_err n_samples=5 threshold=1e-11
        end

        # Testset 23: Round-trip with complex jump (TimeDomain forward + adjoint)
        # qf-bm1 Q1 — see preamble of testset 20.
        @testset "Round-trip: complex jump (TimeDomain)" begin
            config_td = make_config(Lindbladian(),TimeDomain(); construction=KMS())
            L_dense_td = construct_lindbladian(complex_jumps, config_td, TEST_HAM;
                allow_unpaired_nonhermitian=true)
            ws_td = Workspace(config_td, TEST_HAM, complex_jumps)
            # Threshold rationale: complex jump + TimeDomain NUFFT path, FP accumulation only.
            max_err_fwd = 0.0
            max_err_adj = 0.0
            for _ in 1:10
                rho = Matrix(random_density_matrix(NUM_QUBITS))
                err_fwd = norm(L_dense_td * vec(rho) - vec(apply_lindbladian!(ws_td, rho, config_td, TEST_HAM)))
                @test err_fwd < 1e-12
                max_err_fwd = max(max_err_fwd, err_fwd)
                err_adj = norm(L_dense_td' * vec(rho) - vec(apply_adjoint_lindbladian!(ws_td, rho, config_td, TEST_HAM)))
                @test err_adj < 1e-12
                max_err_adj = max(max_err_adj, err_adj)
            end
            @info "Matvec round-trip (complex jump, TimeDomain fwd)" max_error=max_err_fwd n_samples=10 threshold=1e-12
            @info "Adjoint round-trip (complex jump, TimeDomain adj)" max_error=max_err_adj n_samples=10 threshold=1e-12
        end
    end

    # ========================================================================
    # include_coherent=false round-trip tests (dissipator-only Lindbladian)
    # ========================================================================

    @testset "include_coherent=false: $dname forward+adjoint" for (dname, cfg, ham, jumps_used) in [
        ("EnergyDomain", make_config(Lindbladian(), EnergyDomain(); construction=KMS()), TEST_HAM, TEST_JUMPS),
        ("TimeDomain",   make_config(Lindbladian(), TimeDomain(); construction=KMS()),   TEST_HAM, TEST_JUMPS),
        ("TrotterDomain", make_config(Lindbladian(), TrotterDomain(); construction=KMS()), TEST_HAM, TEST_TROTTER_JUMPS),
        ("BohrDomain",   make_config(Lindbladian(), BohrDomain(); construction=KMS()),   TEST_HAM, TEST_JUMPS),
    ]
        trotter_kw = cfg.domain isa TrotterDomain ? (; trotter=TEST_TROTTER) : (;)
        L_diss = construct_lindbladian(jumps_used, cfg, ham; include_coherent=false, trotter_kw...)
        ws = Workspace(cfg, ham, jumps_used; trotter_kw...)

        max_fwd = 0.0
        max_adj = 0.0
        for _ in 1:10
            rho = Matrix(random_density_matrix(NUM_QUBITS))
            L_rho = apply_lindbladian!(ws, rho, cfg, ham; include_coherent=false)
            err_fwd = norm(L_diss * vec(rho) - vec(L_rho))
            @test err_fwd < 1e-12
            max_fwd = max(max_fwd, err_fwd)

            L_adj_rho = apply_adjoint_lindbladian!(ws, rho, cfg, ham; include_coherent=false)
            err_adj = norm(L_diss' * vec(rho) - vec(L_adj_rho))
            @test err_adj < 1e-12
            max_adj = max(max_adj, err_adj)
        end
        @info "include_coherent=false ($dname)" max_fwd max_adj
    end

    @testset "B_bohr row-cache correctness" begin
        cfg = make_config(Lindbladian(), BohrDomain(); construction=KMS())
        B = B_bohr(TEST_HAM, TEST_JUMPS, cfg)
        B_ref = _b_bohr_reference(TEST_HAM, TEST_JUMPS, cfg)
        @test norm(B - B_ref) / max(norm(B_ref), 1.0) < 1e-12

        shuffled_ham = _shuffle_bohr_rows(TEST_HAM)
        B_shuffled = B_bohr(shuffled_ham, TEST_JUMPS, cfg)
        B_shuffled_ref = _b_bohr_reference(shuffled_ham, TEST_JUMPS, cfg)
        @test norm(B_shuffled - B_shuffled_ref) / max(norm(B_shuffled_ref), 1.0) < 1e-12
        @test norm(B_shuffled - B) / max(norm(B), 1.0) < 1e-12

        beta_alg = 5.0
        ham_deg = _make_degenerate_bohr_ham(beta_alg)
        positive_zero = Set(ham_deg.bohr_dict[0.0])
        @test positive_zero == Set(vcat(
            CartesianIndex{2}.(1:4, 1:4),
            [CartesianIndex(2, 3), CartesianIndex(3, 2)],
        ))
        @test !haskey(ham_deg.bohr_dict, -0.0)

        jumps_deg = JumpOp[]
        for site in 1:2
            op = Matrix(pad_term([X], 2, site)) / sqrt(2)
            push!(jumps_deg, JumpOp(op, op, false, true))
        end
        cfg_deg = Config(;
            sim=Lindbladian(),
            domain=BohrDomain(),
            construction=KMS(),
            num_qubits=2,
            with_linear_combination=true,
            beta=beta_alg,
            sigma=1 / beta_alg,
            a=beta_alg / 30,
            s=0.4,
            num_energy_bits=8,
            w0=0.05,
        )
        B_deg = B_bohr(ham_deg, jumps_deg, cfg_deg)
        @test norm(B_deg - _b_bohr_reference(ham_deg, jumps_deg, cfg_deg)) /
              max(norm(B_deg), 1.0) < 1e-12

        cfg_gauss = Config(;
            sim=Lindbladian(),
            domain=BohrDomain(),
            construction=KMS(),
            num_qubits=NUM_QUBITS,
            with_linear_combination=false,
            beta=BETA_ALG,
            sigma=SIGMA,
            gaussian_parameters=(BETA_ALG * (SIGMA^2 + 0.5^2) / 2, 0.5),
            num_energy_bits=NUM_ENERGY_BITS,
            w0=W0,
        )
        B_gauss = B_bohr(TEST_HAM, TEST_JUMPS, cfg_gauss)
        @test norm(B_gauss - _b_bohr_reference(TEST_HAM, TEST_JUMPS, cfg_gauss)) /
              max(norm(B_gauss), 1.0) < 1e-12
    end

end  # @testset "Krylov Matvec"
#
