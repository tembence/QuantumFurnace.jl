using Test
using Random
using LinearAlgebra

function _serial_rho_jump!(scratch, rho, jump, ham, config::Config{Thermalize, EnergyDomain}, precomp, weight)
    fill!(scratch.rho_jump, 0)
    prefactor = precomp.oft_domain_prefactor * weight
    inv_4sigma2 = 1 / (4 * config.sigma^2)
    if jump.hermitian
        for w_raw in precomp.energy_labels
            w_raw > 1e-12 && continue
            w = abs(w_raw)
            oft!(scratch.jump_oft, jump.in_eigenbasis, ham.bohr_freqs, w, inv_4sigma2)
            mul!(scratch.sandwich_tmp, rho, scratch.jump_oft')
            mul!(scratch.rho_jump, scratch.jump_oft, scratch.sandwich_tmp,
                config.delta * prefactor * precomp.transition(w), 1.0)
            if w > 1e-12
                mul!(scratch.sandwich_tmp, rho, scratch.jump_oft)
                mul!(scratch.rho_jump, scratch.jump_oft', scratch.sandwich_tmp,
                    config.delta * prefactor * precomp.transition(-w), 1.0)
            end
        end
    else
        for w in precomp.energy_labels
            oft!(scratch.jump_oft, jump.in_eigenbasis, ham.bohr_freqs, w, inv_4sigma2)
            mul!(scratch.sandwich_tmp, rho, scratch.jump_oft')
            mul!(scratch.rho_jump, scratch.jump_oft, scratch.sandwich_tmp,
                config.delta * prefactor * precomp.transition(w), 1.0)
        end
    end
    return scratch.rho_jump
end

function _serial_rho_jump!(scratch, rho, jump, ham_or_trott,
                           config::Config{Thermalize, D}, precomp, weight) where {D<:Union{TimeDomain, TrotterDomain}}
    fill!(scratch.rho_jump, 0)
    prefactor = precomp.oft_domain_prefactor * weight
    if jump.hermitian
        for w_raw in precomp.energy_labels
            w_raw > 1e-12 && continue
            w = abs(w_raw)
            nufft = QuantumFurnace._prefactor_view(precomp.oft_nufft_prefactors, w)
            @. scratch.jump_oft = jump.in_eigenbasis * nufft
            mul!(scratch.sandwich_tmp, rho, scratch.jump_oft')
            mul!(scratch.rho_jump, scratch.jump_oft, scratch.sandwich_tmp,
                config.delta * prefactor * precomp.transition(w), 1.0)
            if w > 1e-12
                mul!(scratch.sandwich_tmp, rho, scratch.jump_oft)
                mul!(scratch.rho_jump, scratch.jump_oft', scratch.sandwich_tmp,
                    config.delta * prefactor * precomp.transition(-w), 1.0)
            end
        end
    else
        for w in precomp.energy_labels
            nufft = QuantumFurnace._prefactor_view(precomp.oft_nufft_prefactors, w)
            @. scratch.jump_oft = jump.in_eigenbasis * nufft
            mul!(scratch.sandwich_tmp, rho, scratch.jump_oft')
            mul!(scratch.rho_jump, scratch.jump_oft, scratch.sandwich_tmp,
                config.delta * prefactor * precomp.transition(w), 1.0)
        end
    end
    return scratch.rho_jump
end

function _serial_rho_jump!(scratch, rho, jump, ham,
                           config::Config{Thermalize, BohrDomain}, precomp, weight)
    fill!(scratch.rho_jump, 0)
    dim = size(rho, 1)
    in_eb = jump.in_eigenbasis
    for nu_2 in keys(ham.bohr_dict)
        @. scratch.jump_oft = precomp.alpha(ham.bohr_freqs, nu_2) * in_eb
        fill!(scratch.sandwich_tmp, 0)
        for idx in ham.bohr_dict[nu_2]
            i, j = Tuple(idx)
            v = conj(in_eb[i, j])
            for p in 1:dim
                scratch.sandwich_tmp[p, i] += rho[p, j] * v
            end
        end
        mul!(scratch.rho_jump, scratch.jump_oft, scratch.sandwich_tmp,
            config.delta * weight, 1.0)
    end
    return scratch.rho_jump
end

# ============================================================================
# DM thermalization BLAS policy tests
# ============================================================================

@testset "DM thermalization preserves caller BLAS policy" begin
    saved_blas = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(2)
        caller_blas = BLAS.get_num_threads()
        @test caller_blas == 2

        for (domain, jumps, trott, name) in [
            (EnergyDomain(), N3_JUMPS, nothing, "Energy"),
            (TimeDomain(), N3_JUMPS, nothing, "Time"),
            (TrotterDomain(), N3_TROTTER_JUMPS, N3_TROTTER, "Trotter"),
            (BohrDomain(), N3_JUMPS, nothing, "Bohr"),
        ]
            cfg = make_config(Thermalize(), domain; num_qubits=3,
                delta=0.01, mixing_time=0.1)
            run_thermalize(jumps, cfg, N3_HAM, trott)
            @test BLAS.get_num_threads() == caller_blas
            @info "DM caller BLAS policy preserved" path=name blas=caller_blas
        end
    finally
        BLAS.set_num_threads(saved_blas)
    end
end

# ============================================================================
# Omega-loop threading tests
# ============================================================================

@testset "Omega-loop threading determinism" begin
    if Threads.nthreads() > 1
        for (domain, jumps, trott, name) in [
            (EnergyDomain(), N3_JUMPS, nothing, "Energy"),
            (TimeDomain(), N3_JUMPS, nothing, "Time"),
            (TrotterDomain(), N3_TROTTER_JUMPS, N3_TROTTER, "Trotter"),
            (BohrDomain(), N3_JUMPS, nothing, "Bohr"),
        ]
            cfg = make_config(Thermalize(), domain; num_qubits=3,
                delta=0.01, mixing_time=0.05)
            rng_seed = 42

            result1 = run_thermalize(jumps, cfg, N3_HAM, trott;
                rng=Random.Xoshiro(rng_seed))
            result2 = run_thermalize(jumps, cfg, N3_HAM, trott;
                rng=Random.Xoshiro(rng_seed))

            # Deterministic: same seed, same thread count => identical results
            @test result1.trace_distances == result2.trace_distances
            @test result1.final_dm == result2.final_dm
            @info "Omega-loop threading determinism ($name)" domain=name passed=true
        end
    else
        @info "Skipping omega-loop threading determinism tests (nthreads=$(Threads.nthreads()))"
        @test_skip Threads.nthreads() > 1
    end
end

# ============================================================================
# Lindbladian-matvec ω-loop threading tests
# These exercise `_apply_lindbladian_threaded_frequency!`, comparing its output to
# the serial public path. Threading is correctness-preserving up to floating-
# point summation order, so the threshold for the bit-match check is set
# relative to ‖L(rho)‖.
# ============================================================================

# Helper: run threaded variant directly and return the resulting rho_out.
# Mirrors what `apply_lindbladian!` does internally when the threshold dispatch
# fires, so bypasses the threshold gate.
function _run_threaded_lindbladian!(ws, rho, config, ham; adjoint::Bool=false)
    sc = ws.scratch::QuantumFurnace.KrylovScratch{ComplexF64}
    if adjoint
        G_left, G_right = ws.G_right, ws.G_left
    else
        G_left, G_right = ws.G_left, ws.G_right
    end
    fill!(sc.rho_out, 0)
    mul!(sc.rho_out, G_left, rho)
    mul!(sc.rho_out, rho, G_right, 1.0, 1.0)

    prefactor = ws.oft_domain_prefactor * ws.gamma_norm_factor
    data = config.domain isa EnergyDomain ?
        (ham.bohr_freqs, QuantumFurnace._energy_oft_kernel(config)) : ws.oft_nufft_prefactors
    QuantumFurnace._apply_lindbladian_threaded_frequency!(
        sc, rho, ws.jump_eigenbases, ws.jump_hermitian, ws.energy_labels,
        config, prefactor, data, Val(adjoint))
    return copy(sc.rho_out)
end

@testset "Lindbladian threaded matvec: serial ≡ threaded" begin
    if Threads.nthreads() > 1
        rng = MersenneTwister(123)

        # Build a non-Hermitian complex jump for breadth (covers the
        # `is_herm == false` branch of the work-list builder). KMS detailed
        # balance in production needs the conjugate-paired partner, but a
        # single non-Hermitian jump is fine for a serial vs threaded unit
        # test — both paths run the same (possibly non-physical) physics.
        raw_complex = randn(rng, ComplexF64, DIM, DIM) ./ sqrt(DIM)
        complex_jump = JumpOp(raw_complex,
                              TEST_HAM.eigvecs' * raw_complex * TEST_HAM.eigvecs,
                              false, false)

        for (domain, jumps, trott, name) in [
            (EnergyDomain(),  TEST_JUMPS,                 nothing,        "Energy / Hermitian jumps"),
            (EnergyDomain(),  JumpOp[complex_jump],       nothing,        "Energy / non-Hermitian jump"),
            (TimeDomain(),    TEST_JUMPS,                 nothing,        "Time / Hermitian jumps"),
            (TimeDomain(),    JumpOp[complex_jump],       nothing,        "Time / non-Hermitian jump"),
            (TrotterDomain(), TEST_TROTTER_JUMPS,         TEST_TROTTER,   "Trotter / Hermitian jumps"),
        ]
            config = make_config(Lindbladian(), domain; construction=KMS())
            ws_a = Workspace(config, TEST_HAM, jumps; trotter=trott)
            ws_b = Workspace(config, TEST_HAM, jumps; trotter=trott)

            for adjoint in (false, true)
                Random.seed!(rng, 7)
                rho = Matrix(random_density_matrix(NUM_QUBITS))

                serial = if adjoint
                    copy(apply_adjoint_lindbladian!(ws_a, rho, config, TEST_HAM))
                else
                    copy(apply_lindbladian!(ws_a, rho, config, TEST_HAM))
                end
                threaded = _run_threaded_lindbladian!(ws_b, rho, config, TEST_HAM; adjoint=adjoint)

                # FP-accumulation tolerance scaled to ‖L(rho)‖. Per-jump
                # sandwich GEMMs keep the error in the 1e-13/‖L(rho)‖ regime
                # for any reduction order; the helpers above empirically land
                # at ~1e-16, ten orders below the gate.
                tol = max(norm(serial), 1.0) * 1e-12
                err = norm(threaded - serial)
                @test err < tol
                @info "Lindbladian threaded match" path=name adjoint=adjoint err=err tol=tol
            end
        end
    else
        @info "Skipping Lindbladian threaded matvec test (nthreads=$(Threads.nthreads()))"
        @test_skip Threads.nthreads() > 1
    end
end

@testset "Lindbladian matvecs preserve caller BLAS policy" begin
    if Threads.nthreads() > 1
        saved_blas = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(2)
            caller_blas = BLAS.get_num_threads()
            @test caller_blas == 2

            config = make_config(Lindbladian(), EnergyDomain(); construction=KMS())
            ws = Workspace(config, TEST_HAM, TEST_JUMPS)
            rho = Matrix(random_density_matrix(NUM_QUBITS))

            apply_lindbladian!(ws, rho, config, TEST_HAM)
            @test BLAS.get_num_threads() == caller_blas
            apply_adjoint_lindbladian!(ws, rho, config, TEST_HAM)
            @test BLAS.get_num_threads() == caller_blas
            @info "Lindbladian caller BLAS policy preserved" blas=caller_blas
        finally
            BLAS.set_num_threads(saved_blas)
        end
    else
        @info "Skipping Lindbladian caller BLAS policy test (nthreads=$(Threads.nthreads()))"
        @test_skip Threads.nthreads() > 1
    end
end

@testset "Lindbladian threaded matvec: empty work list short-circuit" begin
    if Threads.nthreads() > 1
        # An all-Hermitian jump set with energy_labels that are all > 1e-12
        # produces an empty work list; the helper must return without
        # touching `sc.rho_out` beyond the coherent terms already there.
        config = make_config(Lindbladian(), EnergyDomain(); construction=KMS())
        ws = Workspace(config, TEST_HAM, TEST_JUMPS)
        rho = Matrix(random_density_matrix(NUM_QUBITS))

        sc = ws.scratch
        prefactor = ws.oft_domain_prefactor * ws.gamma_norm_factor
        inv_4sigma2 = 1.0 / (4 * config.sigma^2)

        # Pre-fill rho_out with a sentinel value
        fill!(sc.rho_out, ComplexF64(7.0))
        # Empty energy labels -> empty work list
        QuantumFurnace._apply_lindbladian_threaded_frequency!(
            sc, rho, ws.jump_eigenbases, ws.jump_hermitian, Float64[],
            config, prefactor, (TEST_HAM.bohr_freqs, inv_4sigma2), Val(false))
        @test all(sc.rho_out .== ComplexF64(7.0))
    else
        @info "Skipping empty work-list test (nthreads=$(Threads.nthreads()))"
        @test_skip Threads.nthreads() > 1
    end
end

# ============================================================================
# Channel frequency accumulation and caller BLAS policy.
# Compare threaded accumulation with the serial kernel in the same process.
# ============================================================================

@testset "Channel frequency accumulation: serial reference ≡ threaded" begin
    if Threads.nthreads() > 1
        rng = MersenneTwister(0x51a1)
        raw = randn(rng, ComplexF64, DIM, DIM) / sqrt(DIM)
        nonhermitian_jump = JumpOp(
            raw,
            TEST_HAM.eigvecs' * raw * TEST_HAM.eigvecs,
            false,
            false,
        )
        cases = [
            (EnergyDomain(), TEST_HAM, TEST_JUMPS[1], "Energy/Hermitian"),
            (EnergyDomain(), TEST_HAM, nonhermitian_jump, "Energy/non-Hermitian"),
            (TimeDomain(), TEST_HAM, TEST_JUMPS[1], "Time"),
            (TimeDomain(), TEST_HAM, nonhermitian_jump, "Time/non-Hermitian"),
            (TrotterDomain(), TEST_TROTTER, TEST_TROTTER_JUMPS[1], "Trotter"),
            (BohrDomain(), TEST_HAM, TEST_JUMPS[1], "Bohr"),
        ]
        rho = Matrix(random_density_matrix(NUM_QUBITS))
        for (domain, ham_or_trott, jump, name) in cases
            config = make_config(Thermalize(), domain; construction=KMS())
            precomp = QuantumFurnace._precompute_data(config, ham_or_trott)
            threaded = QuantumFurnace.ThermalizeScratch(ComplexF64, DIM)
            serial = QuantumFurnace.ThermalizeScratch(ComplexF64, DIM)
            weight = precomp.gamma_norm_factor

            old_blas = BLAS.get_num_threads()
            QuantumFurnace._accumulate_rho_jump!(
                threaded, rho, jump, ham_or_trott, config, precomp;
                jump_weight_scaling=weight,
            )
            @test BLAS.get_num_threads() == old_blas
            _serial_rho_jump!(serial, rho, jump, ham_or_trott, config, precomp, weight)

            err = norm(threaded.rho_jump - serial.rho_jump)
            tol = 1e-12 * max(norm(serial.rho_jump), 1.0)
            @test err < tol
            @info "Channel frequency accumulation" path=name err tol
        end
    else
        @test_skip Threads.nthreads() > 1
    end
end

@testset "Channel actions preserve caller BLAS policy" begin
    if Threads.nthreads() > 1
        saved_blas = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(2)
            caller_blas = BLAS.get_num_threads()
            @test caller_blas == 2

            config = make_config(Thermalize(), EnergyDomain(); construction=KMS())
            ws = Workspace(config, TEST_HAM, TEST_JUMPS)
            rho = Matrix(random_density_matrix(NUM_QUBITS))

            apply_delta_channel!(ws, rho, config, TEST_HAM)
            @test BLAS.get_num_threads() == caller_blas
            apply_adjoint_delta_channel!(ws, rho, config, TEST_HAM)
            @test BLAS.get_num_threads() == caller_blas
            @info "Channel caller BLAS policy preserved" blas=caller_blas
        finally
            BLAS.set_num_threads(saved_blas)
        end
    else
        @info "Skipping channel caller BLAS policy test (nthreads=$(Threads.nthreads()))"
        @test_skip Threads.nthreads() > 1
    end
end

# Construction-time reductions are compared with serial references in the
# same process. This isolates reduction-order error from eigenvector phase
# differences between independently constructed Hamiltonians.
#
# Coverage:
#   • _accumulate_R_total_threaded_energy!   (Workspace EnergyDomain)
#   • _accumulate_R_total_threaded_timetrot! (Workspace Time + TrotterDomain)
#   • _accumulate_R_total_threaded_bohr!     (Workspace BohrDomain)
#   • _accumulate_R_total_dll_chunk!         (Workspace DLL BohrDomain)
#   • shared inner/outer coherent quadrature kernels (B_time, B_trotter)
#   • _B_bohr_threaded                       (BohrDomain coherent precompute)
#
# All threaded paths are entered when `Threads.nthreads() > 1` and `n_work >=
# OMEGA_THREAD_THRESHOLD = 10`; the test fixtures comfortably meet both.

using LinearAlgebra
using Random

@testset "Library source does not mutate BLAS policy" begin
    src_dir = dirname(pathof(QuantumFurnace))
    source_files = sort(filter(name -> endswith(name, ".jl"), readdir(src_dir)))
    @test !isempty(source_files)
    for filename in source_files
        source = read(joinpath(src_dir, filename), String)
        @test !occursin("BLAS.set_num_threads", source)
    end
end

@testset "Shared jump-frequency work-list ordering" begin
    labels = [-1.0, 0.0, 1e-12, nextfloat(1e-12), 1.0]
    work = Tuple{Int, Int}[(99, 99)]
    returned = QuantumFurnace._populate_jump_frequency_work_list!(
        work, [true, false], labels)

    @test returned === work
    @test work == [
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
    ]
end

@testset "Constructors preserve caller BLAS policy" begin
    if Threads.nthreads() > 1
        saved_blas = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(2)
            caller_blas = BLAS.get_num_threads()
            @test caller_blas == 2

            for (domain, jumps, trotter, name) in (
                (EnergyDomain(), TEST_JUMPS, nothing, "Energy"),
                (TimeDomain(), TEST_JUMPS, nothing, "Time"),
                (TrotterDomain(), TEST_TROTTER_JUMPS, TEST_TROTTER, "Trotter"),
                (BohrDomain(), TEST_JUMPS, nothing, "Bohr"),
            )
                lind_cfg = make_config(Lindbladian(), domain; construction=KMS())
                Workspace(lind_cfg, TEST_HAM, jumps; trotter=trotter)
                @test BLAS.get_num_threads() == caller_blas

                channel_cfg = make_config(Thermalize(), domain; construction=KMS())
                Workspace(channel_cfg, TEST_HAM, jumps; trotter=trotter)
                @test BLAS.get_num_threads() == caller_blas
                @info "Constructor BLAS policy preserved" path=name blas=caller_blas
            end

            dll_cfg = Config(;
                sim=Lindbladian(),
                domain=BohrDomain(),
                construction=DLL(),
                num_qubits=NUM_QUBITS,
                with_linear_combination=true,
                beta=BETA,
                sigma=SIGMA,
                a=BETA / 30.0,
                s=0.4,
                num_energy_bits=NUM_ENERGY_BITS,
                t0=T0,
                num_trotter_steps_per_t0=NUM_TROTTER_STEPS_PER_T0,
                filter=DLLGaussianFilter(BETA),
            )
            Workspace(dll_cfg, TEST_HAM, TEST_JUMPS)
            @test BLAS.get_num_threads() == caller_blas
            @info "Constructor BLAS policy preserved" path="DLL Bohr" blas=caller_blas
        finally
            BLAS.set_num_threads(saved_blas)
        end
    else
        @info "Skipping constructor BLAS policy test (nthreads=$(Threads.nthreads()))"
        @test_skip Threads.nthreads() > 1
    end
end

@testset "construction-time threading" begin
    # ------------------------------------------------------------
    # (a) R_total accumulation — EnergyDomain (Hermitian + non-Hermitian)
    # ------------------------------------------------------------
    @testset "(a) R_total threaded vs serial: EnergyDomain" begin
        if Threads.nthreads() > 1
            cfg = make_config(Lindbladian(), EnergyDomain(); construction=KMS())
            precomp = QuantumFurnace._precompute_data(cfg, TEST_HAM)

            for (jumps, name) in [
                (TEST_JUMPS, "Hermitian"),
                (begin
                    rng = MersenneTwister(123)
                    raw = randn(rng, ComplexF64, DIM, DIM) ./ sqrt(DIM)
                    JumpOp[JumpOp(raw, TEST_HAM.eigvecs' * raw * TEST_HAM.eigvecs,
                        false, false)]
                end, "non-Hermitian"),
            ]
                jump_eigenbases = [Matrix{ComplexF64}(j.in_eigenbasis) for j in jumps]
                jump_hermitian = [j.hermitian for j in jumps]

                R_threaded = zeros(ComplexF64, DIM, DIM)
                QuantumFurnace._accumulate_R_total!(R_threaded, jump_eigenbases,
                    jump_hermitian, precomp, cfg, TEST_HAM)

                # Hand-rolled serial reference (replicates the inline serial body
                # of `_accumulate_R_total!` for EnergyDomain).
                R_serial = zeros(ComplexF64, DIM, DIM)
                jump_oft = zeros(ComplexF64, DIM, DIM)
                LdagL = zeros(ComplexF64, DIM, DIM)
                inv_4sigma2 = 1.0 / (4 * cfg.sigma^2)
                pf = precomp.oft_domain_prefactor * precomp.gamma_norm_factor
                for (k, eigenbasis) in enumerate(jump_eigenbases)
                    is_herm = jump_hermitian[k]
                    if is_herm
                        for w_raw in precomp.energy_labels
                            w_raw > 1e-12 && continue
                            w = abs(w_raw)
                            QuantumFurnace.oft!(jump_oft, eigenbasis,
                                TEST_HAM.bohr_freqs, w, inv_4sigma2)
                            r2 = pf * precomp.transition(w)
                            mul!(LdagL, jump_oft', jump_oft)
                            R_serial .+= r2 .* LdagL
                            if w > 1e-12
                                r2n = pf * precomp.transition(-w)
                                mul!(LdagL, jump_oft, jump_oft')
                                R_serial .+= r2n .* LdagL
                            end
                        end
                    else
                        for w in precomp.energy_labels
                            QuantumFurnace.oft!(jump_oft, eigenbasis,
                                TEST_HAM.bohr_freqs, w, inv_4sigma2)
                            r2 = pf * precomp.transition(w)
                            mul!(LdagL, jump_oft', jump_oft)
                            R_serial .+= r2 .* LdagL
                        end
                    end
                end

                err = norm(R_threaded .- R_serial)
                rel = err / max(norm(R_serial), 1.0)
                @test rel < 1e-12
                @info "R_total energy" path=name err=err rel=rel
            end
        else
            @info "Skipping R_total energy test (nthreads=$(Threads.nthreads()))"
            @test_skip Threads.nthreads() > 1
        end
    end

    # ------------------------------------------------------------
    # (b) R_total — TimeDomain (NUFFT lookup path)
    # ------------------------------------------------------------
    @testset "(b) R_total threaded vs serial: TimeDomain" begin
        if Threads.nthreads() > 1
            cfg = make_config(Lindbladian(), TimeDomain(); construction=KMS())
            precomp = QuantumFurnace._precompute_data(cfg, TEST_HAM)
            jump_eigenbases = [Matrix{ComplexF64}(j.in_eigenbasis) for j in TEST_JUMPS]
            jump_hermitian = [j.hermitian for j in TEST_JUMPS]

            R_threaded = zeros(ComplexF64, DIM, DIM)
            QuantumFurnace._accumulate_R_total!(R_threaded, jump_eigenbases,
                jump_hermitian, precomp, cfg, TEST_HAM)

            R_serial = zeros(ComplexF64, DIM, DIM)
            jump_oft = zeros(ComplexF64, DIM, DIM)
            LdagL = zeros(ComplexF64, DIM, DIM)
            pf = precomp.oft_domain_prefactor * precomp.gamma_norm_factor
            for (k, eigenbasis) in enumerate(jump_eigenbases)
                is_herm = jump_hermitian[k]
                if is_herm
                    for w_raw in precomp.energy_labels
                        w_raw > 1e-12 && continue
                        w = abs(w_raw)
                        nufft_pf = QuantumFurnace._prefactor_view(
                            precomp.oft_nufft_prefactors, w)
                        @. jump_oft = eigenbasis * nufft_pf
                        r2 = pf * precomp.transition(w)
                        mul!(LdagL, jump_oft', jump_oft)
                        R_serial .+= r2 .* LdagL
                        if w > 1e-12
                            r2n = pf * precomp.transition(-w)
                            mul!(LdagL, jump_oft, jump_oft')
                            R_serial .+= r2n .* LdagL
                        end
                    end
                else
                    for (li, w) in enumerate(precomp.energy_labels)
                        nufft_pf = @view precomp.oft_nufft_prefactors.data[:, :, li]
                        @. jump_oft = eigenbasis * nufft_pf
                        r2 = pf * precomp.transition(w)
                        mul!(LdagL, jump_oft', jump_oft)
                        R_serial .+= r2 .* LdagL
                    end
                end
            end

            err = norm(R_threaded .- R_serial)
            rel = err / max(norm(R_serial), 1.0)
            @test rel < 1e-12
            @info "R_total time" err=err rel=rel
        else
            @test_skip Threads.nthreads() > 1
        end
    end

    # ------------------------------------------------------------
    # (c) R_total — TrotterDomain (NUFFT path on Trotter eigenbasis)
    # ------------------------------------------------------------
    @testset "(c) R_total threaded vs serial: TrotterDomain" begin
        if Threads.nthreads() > 1
            cfg = make_config(Lindbladian(), TrotterDomain(); construction=KMS())
            precomp = QuantumFurnace._precompute_data(cfg, TEST_TROTTER)
            jump_eigenbases = [Matrix{ComplexF64}(j.in_eigenbasis)
                               for j in TEST_TROTTER_JUMPS]
            jump_hermitian = [j.hermitian for j in TEST_TROTTER_JUMPS]

            R_threaded = zeros(ComplexF64, DIM, DIM)
            QuantumFurnace._accumulate_R_total!(R_threaded, jump_eigenbases,
                jump_hermitian, precomp, cfg, TEST_TROTTER)

            R_serial = zeros(ComplexF64, DIM, DIM)
            jump_oft = zeros(ComplexF64, DIM, DIM)
            LdagL = zeros(ComplexF64, DIM, DIM)
            pf = precomp.oft_domain_prefactor * precomp.gamma_norm_factor
            for (k, eigenbasis) in enumerate(jump_eigenbases)
                is_herm = jump_hermitian[k]
                if is_herm
                    for w_raw in precomp.energy_labels
                        w_raw > 1e-12 && continue
                        w = abs(w_raw)
                        nufft_pf = QuantumFurnace._prefactor_view(
                            precomp.oft_nufft_prefactors, w)
                        @. jump_oft = eigenbasis * nufft_pf
                        r2 = pf * precomp.transition(w)
                        mul!(LdagL, jump_oft', jump_oft)
                        R_serial .+= r2 .* LdagL
                        if w > 1e-12
                            r2n = pf * precomp.transition(-w)
                            mul!(LdagL, jump_oft, jump_oft')
                            R_serial .+= r2n .* LdagL
                        end
                    end
                else
                    for (li, w) in enumerate(precomp.energy_labels)
                        nufft_pf = @view precomp.oft_nufft_prefactors.data[:, :, li]
                        @. jump_oft = eigenbasis * nufft_pf
                        r2 = pf * precomp.transition(w)
                        mul!(LdagL, jump_oft', jump_oft)
                        R_serial .+= r2 .* LdagL
                    end
                end
            end

            err = norm(R_threaded .- R_serial)
            rel = err / max(norm(R_serial), 1.0)
            @test rel < 1e-12
            @info "R_total trot" err=err rel=rel
        else
            @test_skip Threads.nthreads() > 1
        end
    end

    # ------------------------------------------------------------
    # (d) R_total — BohrDomain
    # ------------------------------------------------------------
    @testset "(d) R_total threaded vs serial: BohrDomain" begin
        if Threads.nthreads() > 1
            cfg = make_config(Lindbladian(), BohrDomain(); construction=KMS())
            precomp = QuantumFurnace._precompute_data(cfg, TEST_HAM)
            jump_eigenbases = [Matrix{ComplexF64}(j.in_eigenbasis) for j in TEST_JUMPS]
            jump_hermitian = [j.hermitian for j in TEST_JUMPS]

            R_threaded = zeros(ComplexF64, DIM, DIM)
            QuantumFurnace._accumulate_R_total!(R_threaded, jump_eigenbases,
                jump_hermitian, precomp, cfg, TEST_HAM)

            R_serial = zeros(ComplexF64, DIM, DIM)
            jump_oft = zeros(ComplexF64, DIM, DIM)
            A_nu2_dag = zeros(ComplexF64, DIM, DIM)
            bohr_keys = collect(keys(TEST_HAM.bohr_dict))
            for (k, eigenbasis) in enumerate(jump_eigenbases)
                for nu_2 in bohr_keys
                    @. jump_oft = precomp.alpha(TEST_HAM.bohr_freqs, nu_2) * eigenbasis
                    fill!(A_nu2_dag, 0)
                    indices = TEST_HAM.bohr_dict[nu_2]
                    @inbounds for idx in indices
                        i = idx[1]; j = idx[2]
                        A_nu2_dag[j, i] = conj(eigenbasis[i, j])
                    end
                    mul!(R_serial, A_nu2_dag, jump_oft, precomp.gamma_norm_factor, 1.0)
                end
            end

            err = norm(R_threaded .- R_serial)
            rel = err / max(norm(R_serial), 1.0)
            @test rel < 1e-12
            @info "R_total bohr" err=err rel=rel
        else
            @test_skip Threads.nthreads() > 1
        end
    end

    # ------------------------------------------------------------
    # (e) DLL R_total + per-jump operators — BohrDomain
    # ------------------------------------------------------------
    @testset "(e) DLL R_total + per-jump ordering" begin
        if Threads.nthreads() > 1
            beta = 10.0
            cfg = Config(;
                sim = Lindbladian(),
                domain = BohrDomain(),
                construction = DLL(),
                num_qubits = 3,
                with_linear_combination = true,
                beta = beta,
                sigma = 1.0/beta,
                a = beta/30.0,
                s = 0.4,
                num_energy_bits = 12,
                t0 = 2pi / (2^12 * 0.05),
                num_trotter_steps_per_t0 = 10,
                filter = DLLGaussianFilter(beta),
            )
            sys = make_dll_n3_system(beta)
            precomp = QuantumFurnace._precompute_data(cfg, sys.ham)

            R_threaded = zeros(ComplexF64, N3_DIM, N3_DIM)
            dll_threaded = Vector{Matrix{ComplexF64}}()
            QuantumFurnace._accumulate_R_total_dll!(
                R_threaded, dll_threaded, sys.jumps,
                precomp, cfg, sys.ham)

            R_serial = zeros(ComplexF64, N3_DIM, N3_DIM)
            dll_serial = Vector{Matrix{ComplexF64}}()
            for jump in sys.jumps
                L_or_Ls = QuantumFurnace.dll_lindblad_op_bohr(jump, sys.ham, precomp.filter)
                if L_or_Ls isa AbstractMatrix
                    L_a = Matrix{ComplexF64}(L_or_Ls)
                    push!(dll_serial, L_a)
                    mul!(R_serial, L_a', L_a, 1.0, 1.0)
                else
                    for L_one in L_or_Ls
                        L_a = Matrix{ComplexF64}(L_one)
                        push!(dll_serial, L_a)
                        mul!(R_serial, L_a', L_a, 1.0, 1.0)
                    end
                end
            end

            @test length(dll_threaded) == length(dll_serial)
            for k in eachindex(dll_serial)
                @test isapprox(dll_threaded[k], dll_serial[k]; atol=1e-15)
            end

            err = norm(R_threaded .- R_serial)
            rel = err / max(norm(R_serial), 1.0)
            @test rel < 1e-12
            @info "R_total dll" err=err rel=rel n_ops=length(dll_serial)
        else
            @test_skip Threads.nthreads() > 1
        end
    end

    # ------------------------------------------------------------
    # (f) B_time threaded (inner τ × jumps + outer t)
    # ------------------------------------------------------------
    @testset "(f) B_time threaded vs serial" begin
        if Threads.nthreads() > 1
            cfg = make_config(Lindbladian(), TimeDomain(); construction=KMS())
            precomp = QuantumFurnace._precompute_data(cfg, TEST_HAM)
            B_threaded = QuantumFurnace.B_time(TEST_JUMPS, TEST_HAM,
                precomp.b_minus, precomp.b_plus,
                QuantumFurnace.register_t0_b_minus(cfg),
                QuantumFurnace.register_t0_b_plus(cfg),
                cfg.beta, cfg.sigma)

            # Serial reference: explicit nested loop matching the inline body.
            d = DIM
            CT = ComplexF64
            eigvals = TEST_HAM.eigvals

            b_plus_summand = zeros(CT, d, d)
            diag_u  = Vector{CT}(undef, d)
            diag_u2 = Vector{CT}(undef, d)
            tmp     = Matrix{CT}(undef, d, d)
            M       = Matrix{CT}(undef, d, d)
            for tau in keys(precomp.b_plus)
                t_tau = tau * cfg.beta
                @. diag_u  = exp(1im * eigvals * t_tau)
                @. diag_u2 = exp(-2im * eigvals * t_tau)
                diag_u_row = transpose(diag_u)
                for jump_a in TEST_JUMPS
                    jump_eig = jump_a.in_eigenbasis
                    @. tmp = diag_u2 * jump_eig
                    mul!(M, jump_eig', tmp)
                    b_plus_summand .+= precomp.b_plus[tau] .* diag_u .* M .* diag_u_row
                end
            end
            B_serial = zeros(CT, d, d)
            for t in keys(precomp.b_minus)
                @. diag_u = exp(1im * eigvals * (t / cfg.sigma))
                diag_u_row = transpose(diag_u)
                B_serial .+= precomp.b_minus[t] .* conj.(diag_u) .* b_plus_summand .* diag_u_row
            end
            t0o = QuantumFurnace.register_t0_b_minus(cfg)
            t0i = QuantumFurnace.register_t0_b_plus(cfg)
            B_serial .*= t0o * t0i

            err = norm(B_threaded .- B_serial)
            rel = err / max(norm(B_serial), 1.0)
            @test rel < 1e-12
            @info "B_time threaded" err=err rel=rel
        else
            @test_skip Threads.nthreads() > 1
        end
    end

    # ------------------------------------------------------------
    # (g) B_trotter threaded
    # ------------------------------------------------------------
    @testset "(g) B_trotter threaded vs serial" begin
        if Threads.nthreads() > 1
            cfg = make_config(Lindbladian(), TrotterDomain(); construction=KMS())
            precomp = QuantumFurnace._precompute_data(cfg, TEST_TROTTER)
            t0o = QuantumFurnace.register_t0_b_minus(cfg)
            t0i = QuantumFurnace.register_t0_b_plus(cfg)
            B_threaded = QuantumFurnace.B_trotter(TEST_TROTTER_JUMPS, TEST_TROTTER,
                precomp.b_minus, precomp.b_plus, t0o, t0i, cfg.beta, cfg.sigma)

            d = DIM
            CT = ComplexF64
            # TEST_TROTTER is a single-cache TrottTrott: both legs run against
            # the same eigvals/t0 (canonical KMS coherent uses TrotterTriple).
            eigvals_outer = TEST_TROTTER.eigvals_t0
            eigvals_inner = TEST_TROTTER.eigvals_t0
            t0_step_outer = TEST_TROTTER.t0
            t0_step_inner = TEST_TROTTER.t0

            b_plus_summand = zeros(CT, d, d)
            diag_u  = Vector{CT}(undef, d)
            diag_u2 = Vector{CT}(undef, d)
            tmp     = Matrix{CT}(undef, d, d)
            M       = Matrix{CT}(undef, d, d)
            for (tau, b_tau) in precomp.b_plus
                num_t0_steps = Int(round(tau * cfg.beta / t0_step_inner))
                @. diag_u  = eigvals_inner ^ num_t0_steps
                @. diag_u2 = eigvals_inner ^ (-2 * num_t0_steps)
                diag_u_row = transpose(diag_u)
                for jump_a in TEST_TROTTER_JUMPS
                    jump_a_eig = jump_a.in_eigenbasis
                    @. tmp = diag_u2 * jump_a_eig
                    mul!(M, jump_a_eig', tmp)
                    b_plus_summand .+= b_tau .* diag_u .* M .* diag_u_row
                end
            end
            B_serial = zeros(CT, d, d)
            for (t, b_t) in precomp.b_minus
                num_t0_steps = Int(round(t / (cfg.sigma * t0_step_outer)))
                @. diag_u = eigvals_outer ^ num_t0_steps
                diag_u_row = transpose(diag_u)
                B_serial .+= b_t .* conj.(diag_u) .* b_plus_summand .* diag_u_row
            end
            B_serial .*= t0o * t0i

            err = norm(B_threaded .- B_serial)
            rel = err / max(norm(B_serial), 1.0)
            @test rel < 1e-12
            @info "B_trotter threaded" err=err rel=rel
        else
            @test_skip Threads.nthreads() > 1
        end
    end

end
