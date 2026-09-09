# test/test_predict_sandbox.jl
#
# Sandbox shadow of test_predict_lindbladian.jl + test_predict_channel.jl
# with small fixtures covering these invariants:
#
#   1. predict_lindbladian_trajectory matches the dense reference
#      exp(t·L)·vec(ρ₀) along the trajectory — the spectral-expansion
#      accuracy invariant (Krylov subspace captures the slow modes).
#   2. predict_channel_trajectory matches run_thermalize byte-for-byte at
#      small δ — the channel byte-identity invariant (closure uses the same
#      _apply_one_dm_substep! kernel as run_thermalize).
#
# Both at n = 3, β = 10. The Lindbladian check uses 5 time points (range
# [0, 5/gap]) at krylovdim = 30 — captures the diagonal-sector dynamics on
# d² = 64 modes ⇒ 1e-7 absolute error matches the heavy test threshold.
# The channel check uses ~50 outer steps at δ = 1e-3 (mixing_time = 0.05)
# at krylovdim = 30 — matches run_thermalize to 1e-10 in trace distance
# along the full save_every-spaced grid.

using LinearAlgebra: I, eigvals, svdvals, norm, tr, Hermitian
using Test
using QuantumFurnace

@testset "Predictor convergence and analytic dephasing" begin
    initial = fill(ComplexF64(0.5), 2, 2)
    # The implemented Gaussian DLL filter has f_hat(0)=1. For H=0, a Z
    # source therefore damps off-diagonal entries at twice its source rate.
    expected = ComplexF64[0.5 0.5exp(-2); 0.5exp(-2) 0.5]
    for rate in (1.0, 1e-20)
        inputs = prepare_gibbs_inputs(zeros(2, 2); beta_phys=0.8, jumps=[Z],
            clock=GeneratorClock{Float64}(:scaled_dephasing, rate, 1.0))
        times = [0.0, inv(rate)]
        for raw in (false, true)
            truncated = predict_lindbladian_trajectory(inputs.config, inputs.hamiltonian,
                inputs.jumps, initial, times; krylovdim=1, raw_reconstruction=raw)
            @test !truncated.all_converged
            @test !truncated.invariant_subspace
            @test norm(truncated.rho_final - expected) > 0.1
            for dimension in (2, 4)
                exact = predict_lindbladian_trajectory(inputs.config, inputs.hamiltonian,
                    inputs.jumps, initial, times; krylovdim=dimension,
                    raw_reconstruction=raw, save_states=true)
                @test exact.all_converged
                @test exact.invariant_subspace
                @test exact.states[1] ≈ initial atol=1e-11
                @test exact.rho_final ≈ expected atol=1e-11
            end
        end
    end
end


# Measure in hard function scope after repeated warm-up. This avoids soft-scope
# boxing and exercises the production workspace-reuse path at fixed matvec cost.
function _measure_channel_predictor_allocs(cfg, ham, jumps, rho_0, k_grid, ws)
    warm1 = predict_channel_trajectory(
        cfg, ham, jumps, rho_0, k_grid; krylovdim=20, workspace=ws)
    warm2 = predict_channel_trajectory(
        cfg, ham, jumps, rho_0, k_grid; krylovdim=20, workspace=ws)
    allocs = @allocated predict_channel_trajectory(
        cfg, ham, jumps, rho_0, k_grid; krylovdim=20, workspace=ws)
    return allocs, warm1.total_matvecs, warm2.total_matvecs
end


@testset "Predictor sandbox shadows" begin

    # -----------------------------------------------------------------------
    # (a) predict_lindbladian_trajectory accuracy vs dense reference.
    # at n=3 with the 3n=9 single-Pauli jump set + KMS-DB
    # the slow spectrum is well-separated; krylovdim=30 captures the entire
    # diagonal-sector dynamics on d²=64 modes. Same threshold as the heavy
    # test (1e-7) — the regime where bi-exp τ_mix extraction is accurate.
    # -----------------------------------------------------------------------
    @testset "(a) predict_lindbladian vs dense @ n=3, β=10" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        ham = sys.ham; jumps = sys.jumps
        d = size(ham.data, 1)

        cfg = Config(
            sim = Lindbladian(),
            domain = EnergyDomain(),
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta, sigma = 1.0 / beta,
            a = 0.0, s = 0.25,
            num_energy_bits = 12, w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
        )

        rho_0 = Matrix{ComplexF64}(I(d) / d)
        sigma_beta = Matrix{ComplexF64}(ham.gibbs)

        # Use Krylov gap to pick t_max (same as heavy test).
        gap_res = krylov_spectral_gap(cfg, ham, jumps;
                                       krylovdim = 30, howmany = 4, tol = 1e-10)
        gap = gap_res.spectral_gap
        @test gap > 0

        # Pass-2 (krylov_spectral_gap) carries operator-side diagnostics
        # (no seeded ρ₀ ⇒ c-side is NaN; R-side is the ρ₀-independent picture).
        gsm = gap_res.spectral_modes
        @test gsm isa SpectralModeDiagnostics
        @test all(isnan, gsm.c_abs2)
        @test all(0.0 .<= gsm.off_diag_weight .<= 1.0)
        @test gsm.off_diag_weight[1] < 1e-6   # fixed_point ≈ σ_β: diagonal

        # 5 time points across [0, 5/gap] — shrunk from heavy test's 21 to
        # keep dense exp(t·L) workload trivial (5 exponentiations of a
        # 64-dim generator).
        t_grid = collect(range(0.0, 5.0 / gap, length = 5))

        # Dense reference trajectory.
        L_dense = construct_lindbladian(jumps, cfg, ham)
        v0 = vec(rho_0)
        distances_dense = Vector{Float64}(undef, length(t_grid))
        for (k, t) in enumerate(t_grid)
            rho_t = reshape(exp(t * L_dense) * v0, d, d)
            rho_t .= (rho_t + rho_t') ./ 2
            distances_dense[k] = sum(svdvals(rho_t .- sigma_beta)) / 2
        end

        # Krylov spectral expansion.
        res_kr = predict_lindbladian_trajectory(cfg, ham, jumps, rho_0, t_grid;
                                                 krylovdim = 30, tol = 1e-10)
        @test length(res_kr.t) == length(t_grid)
        @test size(res_kr.rho_final) == (d, d)
        @test res_kr.total_matvecs <= length(t_grid) * 30

        # per-mode spectral diagnostics are attached to every result.
        sm = res_kr.spectral_modes
        @test sm isa SpectralModeDiagnostics
        @test length(sm.off_diag_weight) == length(res_kr.eigenvalues)
        @test all(0.0 .<= sm.off_diag_weight .<= 1.0)
        @test sm.off_diag_weight[1] < 1e-6   # steady mode ≈ σ_β: diagonal in the energy eigenbasis

        max_abs_err = maximum(abs.(res_kr.distances .- distances_dense))
        @test max_abs_err < 1e-7
        @info "(a) predict_lindbladian sandbox" max_abs_err matvecs=res_kr.total_matvecs

        # Spectral gap from Krylov must match dense (KrylovKit tolerance).
        eigs_dense = sort(eigvals(L_dense); by = v -> abs(real(v)))
        gap_dense = abs(real(eigs_dense[2]))
        @test isapprox(res_kr.spectral_gap, gap_dense; rtol = 1e-9)
    end

    # -----------------------------------------------------------------------
    # (b) predict_channel_trajectory byte-identity vs run_thermalize.
    # 50 outer steps at δ=1e-3 (mixing_time=0.05). The forward closure uses
    # the SAME _apply_one_dm_substep! kernel run_thermalize calls, so the
    # residual is dominated by finite Krylov subspace + dense eigen(H)
    # tolerance only — must be below 1e-10 for the n=3 fixture.
    # -----------------------------------------------------------------------
    @testset "(b) predict_channel byte-identity vs run_thermalize @ n=3, δ=1e-3" begin
        beta = 10.0
        delta = 1e-3
        sys = make_dll_n3_system(beta)
        ham = sys.ham; jumps = sys.jumps
        d = size(ham.data, 1)

        cfg = Config(
            sim = Thermalize(),
            domain = BohrDomain(),
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta, sigma = 1.0 / beta,
            a = 0.0, s = 0.25,
            num_energy_bits = 12, w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
            delta = delta,
            mixing_time = 0.05,        # 50 outer steps
            jump_selection = :sweep,
        )
        rho_0 = Matrix{ComplexF64}(I(d) / d)

        k_step = 10
        k_grid = collect(0:k_step:50)   # [0, 10, …, 50] — 6 save points

        res_kr = predict_channel_trajectory(cfg, ham, jumps, rho_0, k_grid;
                                              krylovdim = 30, tol = 1e-10)
        res_th = run_thermalize(jumps, cfg, ham; initial_dm = copy(rho_0),
                                save_every = k_step)

        @test length(res_kr.t) == length(k_grid)
        @test res_kr.delta_used == delta
        @test res_kr.k_grid == k_grid

        # per-mode spectral diagnostics on the channel predictor too.
        # eigenvalues here are the raw channel μ (μ-units mode_spacing — see the
        # SpectralModeDiagnostics docstring); the steady mode (μ₁≈1) ≈ σ_β so its
        # off_diag_weight is small.
        sm = res_kr.spectral_modes
        @test sm isa SpectralModeDiagnostics
        @test length(sm.off_diag_weight) == length(res_kr.eigenvalues)
        @test all(0.0 .<= sm.off_diag_weight .<= 1.0)
        @test sm.off_diag_weight[1] < 1e-6   # steady mode ≈ σ_β: diagonal in the Trotter eigenbasis

        # CPTP fixed point at mu_1.
        @test abs(abs(res_kr.eigenvalues[1]) - 1.0) < 1e-10

        @assert length(res_kr.distances) == length(res_th.trace_distances)
        max_abs_err = maximum(abs.(res_kr.distances .- res_th.trace_distances))
        @test max_abs_err < 1e-10
        @info "(b) predict_channel sandbox byte-identity" max_abs_err matvecs=res_kr.total_matvecs

        # The deterministic weak-measurement construction is CPTP. These raw
        # trace diagnostics witness that no hidden normalisation is needed.
        @test res_kr.trace_preserving_assumed
        @test res_kr.physical_channel
        @test res_kr.channel_representation === :deterministic_cptp
        @test !res_kr.trace_normalized
        @test maximum(abs.(res_kr.trace_values .- 1)) < 1e-10
        @test res_kr.max_abs_trace_drift < 1e-10
        @test res_th.metadata[:trace_preserving_assumed]
        @test res_th.metadata[:physical_channel]
        @test res_th.metadata[:channel_representation] === :deterministic_cptp
        @test res_th.metadata[:max_abs_trace_drift] < 1e-10

        # Final density matrix agrees too (within hermitisation noise).
        rho_th_final = res_th.final_dm
        @test maximum(abs.(res_kr.rho_final .- rho_th_final)) < 1e-9
    end

    # -----------------------------------------------------------------------
    # (c) run_krylov_spectrum stashes the operator-side diagnostics into
    # KrylovSpectrumResults.metadata[:spectral_modes]. Operator-only (no seeded
    # ρ₀ in the Pass-2 path) ⇒ the c-side fields are all NaN.
    # -----------------------------------------------------------------------
    @testset "(c) run_krylov_spectrum metadata carries spectral_modes" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        ham = sys.ham; jumps = sys.jumps

        cfg = Config(
            sim = Lindbladian(),
            domain = EnergyDomain(),
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta, sigma = 1.0 / beta,
            a = 0.0, s = 0.25,
            num_energy_bits = 12, w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
        )

        res = run_krylov_spectrum(jumps, cfg, ham; krylovdim = 30, howmany = 4, tol = 1e-10)
        @test haskey(res.metadata, :spectral_modes)
        sm = res.metadata[:spectral_modes]
        @test sm isa SpectralModeDiagnostics
        @test all(isnan, sm.c_abs2)            # operator-only: no modal coefficient
        @test all(isnan, sm.modal_hs_weight)
        @test all(0.0 .<= sm.off_diag_weight .<= 1.0)
        @test sm.off_diag_weight[1] < 1e-6     # fixed_point ≈ σ_β: diagonal
    end

    @testset "Predictor workspace reuse and guards" begin
        psi_plus = ones(ComplexF64, N3_DIM) / sqrt(N3_DIM)
        rho_plus_computational = psi_plus * psi_plus'
        rho_0 = Matrix{ComplexF64}(
            N3_HAM.eigvecs' * rho_plus_computational * N3_HAM.eigvecs)

        cfg_L = make_config(Lindbladian(), EnergyDomain(); num_qubits=3)
        t_grid = [0.0, 1.0, 2.0]
        fresh_L = predict_lindbladian_trajectory(
            cfg_L, N3_HAM, N3_JUMPS, rho_0, t_grid; krylovdim=20)
        ws_L = Workspace(cfg_L, N3_HAM, N3_JUMPS)
        reuse_L1 = predict_lindbladian_trajectory(
            cfg_L, N3_HAM, N3_JUMPS, rho_0, t_grid; krylovdim=20, workspace=ws_L)
        reuse_L2 = predict_lindbladian_trajectory(
            cfg_L, N3_HAM, N3_JUMPS, rho_0, t_grid; krylovdim=20, workspace=ws_L)
        @test fresh_L.distances == reuse_L1.distances == reuse_L2.distances
        @test fresh_L.spectral_gap == reuse_L1.spectral_gap == reuse_L2.spectral_gap
        @test fresh_L.eigenvalues == reuse_L1.eigenvalues

        cfg_C = make_config(Thermalize(), EnergyDomain(); num_qubits=3)
        k_grid = [0, 5, 10]
        fresh_C = predict_channel_trajectory(
            cfg_C, N3_HAM, N3_JUMPS, rho_0, k_grid; krylovdim=20)
        ws_C = Workspace(cfg_C, N3_HAM, N3_JUMPS)
        reuse_C1 = predict_channel_trajectory(
            cfg_C, N3_HAM, N3_JUMPS, rho_0, k_grid; krylovdim=20, workspace=ws_C)
        reuse_C2 = predict_channel_trajectory(
            cfg_C, N3_HAM, N3_JUMPS, rho_0, k_grid; krylovdim=20, workspace=ws_C)
        @test fresh_C.distances == reuse_C1.distances == reuse_C2.distances
        @test fresh_C.spectral_gap == reuse_C1.spectral_gap == reuse_C2.spectral_gap
        @test fresh_C.eigenvalues == reuse_C1.eigenvalues

        rho_before = copy(rho_0)
        predictor_allocs, warm1_matvecs, warm2_matvecs =
            _measure_channel_predictor_allocs(
                cfg_C, N3_HAM, N3_JUMPS, rho_0, k_grid, ws_C)
        # Pre-edit Julia 1.12 baseline: 2,720,568 bytes. Passing an Arnoldi
        # column view and reusing `rho_buf` remove 46,080 bytes. Julia 1.10's
        # baseline ranges up to 2,935,056 bytes across the Pkg.test and focused
        # clean environments; both margins remain smaller than either
        # 23,040-byte copy regression.
        predictor_budget = VERSION < v"1.11" ? 2_950_000 : 2_683_000
        @test predictor_allocs <= predictor_budget
        @test warm1_matvecs == warm2_matvecs == 20
        @test rho_0 == rho_before
        @info "Warmed n=3 channel predictor allocations" allocs_bytes=predictor_allocs threshold=predictor_budget matvecs=warm2_matvecs

        # Inject a visible anti-Hermitian perturbation at k=t=0. The predictor
        # reconstruction must use the unbiased pairwise projection, not a
        # sequential overwrite whose second triangle reads modified data.
        rho_asymmetric = copy(rho_0)
        rho_asymmetric[1, 2] += 0.2 + 0.3im
        expected_projection = (rho_asymmetric + rho_asymmetric') / 2
        asymmetric_L = predict_lindbladian_trajectory(
            cfg_L, N3_HAM, N3_JUMPS, rho_asymmetric, [0.0];
            krylovdim=20, workspace=ws_L)
        asymmetric_C = predict_channel_trajectory(
            cfg_C, N3_HAM, N3_JUMPS, rho_asymmetric, [0];
            krylovdim=20, workspace=ws_C)
        @test norm(asymmetric_L.rho_final - expected_projection) < 1e-10
        @test norm(asymmetric_C.rho_final - expected_projection) < 1e-10
        @test norm(asymmetric_L.rho_final - asymmetric_L.rho_final') < 1e-14
        @test norm(asymmetric_C.rho_final - asymmetric_C.rho_final') < 1e-14

        cfg_C_mismatch = make_config(
            Thermalize(), EnergyDomain(); num_qubits=3, delta=2 * TEST_DELTA)
        cfg_C_random = make_config(
            Thermalize(), EnergyDomain(); num_qubits=3, jump_selection=:random)
        @test_throws ArgumentError predict_channel_trajectory(
            cfg_C_random, N3_HAM, N3_JUMPS, rho_0, k_grid; krylovdim=20)
        @test_throws ArgumentError predict_channel_trajectory(
            cfg_C_mismatch, N3_HAM, N3_JUMPS, rho_0, k_grid;
            krylovdim=20, workspace=ws_C)
        @test_throws ArgumentError predict_lindbladian_trajectory(
            cfg_L, N3_HAM, N3_JUMPS, rho_0, t_grid;
            krylovdim=20, workspace=ws_C)
        @test_throws ArgumentError predict_channel_trajectory(
            cfg_C, N3_HAM, N3_JUMPS, rho_0, k_grid;
            krylovdim=20, workspace=ws_L)

        rho_wrong = Matrix{ComplexF64}(I(2 * N3_DIM) / (2 * N3_DIM))
        @test_throws AssertionError predict_lindbladian_trajectory(
            cfg_L, N3_HAM, N3_JUMPS, rho_wrong, t_grid;
            krylovdim=20, workspace=ws_L)
        @test_throws ArgumentError predict_lindbladian_trajectory(
            cfg_L, N3_HAM, N3_JUMPS[1:end-1], rho_0, t_grid;
            krylovdim=20, workspace=ws_L)
    end

    @testset "True-gap pass on a parity-symmetric fixture" begin
        system = make_classical_ising_n3()
        (; ham, jumps) = system
        dim = size(ham.data, 1)
        rho_0 = Matrix{ComplexF64}(I(dim) / dim)

        cfg_L = make_classical_ising_config(Lindbladian(), system)
        dense_eigs = eigvals(construct_lindbladian(jumps, cfg_L, ham))
        gap_dense = abs(real(dense_eigs[sortperm(real.(dense_eigs); by=abs)[2]]))
        pass1_L = predict_lindbladian_trajectory(
            cfg_L, ham, jumps, rho_0, [0.0, 1.0, 2.0]; krylovdim=40)
        pass2_L = predict_lindbladian_trajectory(
            cfg_L, ham, jumps, rho_0, [0.0, 1.0, 2.0];
            krylovdim=40, compute_true_gap=true)
        @test abs(pass1_L.spectral_gap - pass2_L.spectral_gap) / pass2_L.spectral_gap > 0.5
        @test isapprox(pass2_L.spectral_gap, gap_dense; rtol=1e-8)
        @test pass2_L.total_matvecs > pass1_L.total_matvecs

        cfg_C = make_classical_ising_config(Thermalize(), system; delta=1e-3)
        pass1_C = predict_channel_trajectory(
            cfg_C, ham, jumps, rho_0, [0, 5, 10]; krylovdim=40)
        pass2_C = predict_channel_trajectory(
            cfg_C, ham, jumps, rho_0, [0, 5, 10];
            krylovdim=40, compute_true_gap=true)
        @test abs(pass2_C.spectral_gap - gap_dense) / gap_dense < 1e-3
        @test pass2_C.total_matvecs > pass1_C.total_matvecs
    end

end
