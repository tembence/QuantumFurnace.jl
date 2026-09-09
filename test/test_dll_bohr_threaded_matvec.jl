# ============================================================================
# DLL BohrDomain threaded matvec verification
#
# Independent verification of `_apply_lindbladian_threaded_bohr_dll!` and its
# wiring into `apply_lindbladian!` / `apply_adjoint_lindbladian!` for
# `Config{Lindbladian, BohrDomain, DLL}`. The new threaded path parallelizes
# the dissipator `Σ_a L_a ρ L_a†` over the jump index `a` (the
# `ws.dll_lindblads` vector), accumulating per-task partials into the
# (coherent-pre-seeded) `sc.rho_out`.
#
# References used (all INDEPENDENT of the threaded matvec code):
#   - Dense `construct_lindbladian(jumps, cfg, ham)` for DLL Bohr → a full
#     d²×d² Liouvillian built by `_jump_contribution!` / dense vectorization
#     (used as `L_dense * vec(ρ)` and `L_dense' * vec(ρ)`).
#   - An explicit serial sum `G_left·ρ + ρ·G_right + Σ_a L_a·ρ·L_a†` assembled
#     from the workspace matrices directly (bypasses ALL matvec code).
#
# The fixtures mirror the production driver `simulations/qf_edk_dll_gap_nsweep.jl`
# (`build_dll_cfg`): 1D Heisenberg seed-46, Metropolis filter, β_phys-scaled.
#
# This file is the FIRST coverage of the DLL Bohr `apply_lindbladian!` path:
# `test_krylov_matvec.jl` only covers KMS/GNS BohrDomain, and
# `test_dll_dissipator.jl` only checks the dense Liouvillian — neither calls
# `apply_lindbladian!` on a `Workspace{KrylovSpectrum, BohrDomain, DLL}`.
#
# Gate boundary: `OMEGA_THREAD_THRESHOLD = 10`; for 1D Heisenberg |𝒜| = 3n, so
#   n=3 → |dll_lindblads|=9  (< 10 ⇒ SERIAL path)
#   n≥4 → |dll_lindblads|≥12 (≥ 10 ⇒ THREADED path when nthreads > 1)
# Multi-channel filters multiply the count by k channels.
# ============================================================================

using Test
using LinearAlgebra
using Random
using QuantumFurnace
using QuantumFurnace: _parse_hamiltonian_bson, _jumps_in_basis,
                     apply_lindbladian!, apply_adjoint_lindbladian!,
                     _apply_lindbladian_threaded_bohr_dll!, KrylovScratch,
                     OMEGA_THREAD_THRESHOLD

# test_helpers.jl is already included by runtests.jl

# DLL Metropolis configuration with s=0.25, a=0 and σ=1/β_alg. BohrDomain
# register fields are inert.
function _dll_bohr_cfg(n, ham, beta_phys; filter=nothing)
    β_alg = beta_alg(ham, beta_phys)
    f = filter === nothing ? DLLMetropolisFilter(β_alg) : filter
    return Config(;
        sim = Lindbladian(), domain = BohrDomain(), construction = DLL(),
        num_qubits = n, with_linear_combination = true,
        beta = β_alg, beta_phys = beta_phys, sigma = 1.0 / β_alg,
        a = 0.0, s = 0.25, gaussian_parameters = (nothing, nothing),
        num_energy_bits = 12, w0 = 0.05, t0 = 2π / (2^12 * 0.05),
        num_trotter_steps_per_t0 = 10,
        filter = f,
    )
end

function _load_heis(n, beta_phys)
    path = test_hamiltonian_path(n)
    isfile(path) || error("missing 1D fixture $path")
    return HamHam(_parse_hamiltonian_bson(path); beta_phys = beta_phys)
end

# Random Hermitian matrix (the matvec acts on Hermitian ρ in production, but the
# code is C-linear; we also test a generic complex input below).
_rand_herm(rng, d) = (G = randn(rng, ComplexF64, d, d); (G + G') / 2)

@testset "DLL BohrDomain threaded matvec" begin

    _threaded = Threads.nthreads() > 1
    @info "DLL Bohr threaded-matvec test" nthreads=Threads.nthreads() OMEGA_THREAD_THRESHOLD=OMEGA_THREAD_THRESHOLD threaded_runtime=_threaded

    # n=3 (serial gate, |𝒜|=9<10), n=4,5 (threaded gate, |𝒜|≥12).
    # β_phys grid touches both small and large β_alg (the kink width ∝ 1/β_alg).
    _NS = (3, 4, 5)
    _BETAS_PHYS = (0.25, 0.5, 1.0)

    # ---------------------------------------------------------------------
    # (a) Gate boundary: confirm n=3 takes SERIAL, n≥4 takes THREADED path
    #     (when nthreads > 1), via the |dll_lindblads| count vs the threshold.
    # ---------------------------------------------------------------------
    @testset "(a) gate boundary |dll_lindblads| vs OMEGA_THREAD_THRESHOLD" begin
        @test OMEGA_THREAD_THRESHOLD == 10
        for n in _NS
            ham = _load_heis(n, 0.5)
            jumps = _jumps_in_basis(n, ham.eigvecs)
            cfg = _dll_bohr_cfg(n, ham, 0.5)
            validate_config!(cfg, ham)
            ws = Workspace(cfg, ham, jumps)
            ndll = length(ws.dll_lindblads)
            @test ndll == 3 * n                       # X,Y,Z per site, single channel
            takes_threaded = _threaded && ndll >= OMEGA_THREAD_THRESHOLD
            if n == 3
                @test ndll == 9
                @test !(ndll >= OMEGA_THREAD_THRESHOLD)   # serial regardless of nthreads
            elseif n >= 4
                @test ndll >= OMEGA_THREAD_THRESHOLD       # threaded when nthreads > 1
            end
            @info "gate" n=n ndll=ndll takes_threaded_path=takes_threaded
        end
    end

    # ---------------------------------------------------------------------
    # (b) Forward matvec == dense Liouvillian to machine precision.
    #     Covers BOTH the serial (n=3) and threaded (n≥4) branches.
    # ---------------------------------------------------------------------
    @testset "(b) forward apply_lindbladian! == dense L·vec(ρ)" begin
        max_err = 0.0
        for n in _NS, bp in _BETAS_PHYS
            ham = _load_heis(n, bp)
            d = 2^n
            jumps = _jumps_in_basis(n, ham.eigvecs)
            cfg = _dll_bohr_cfg(n, ham, bp)
            validate_config!(cfg, ham)
            ws = Workspace(cfg, ham, jumps)
            L_dense = construct_lindbladian(jumps, cfg, ham)
            rng = MersenneTwister(1000 + 31 * n + round(Int, 100bp))
            for _ in 1:4
                rho = _rand_herm(rng, d)
                out = copy(apply_lindbladian!(ws, copy(rho), cfg, ham))
                ref = reshape(L_dense * vec(rho), d, d)
                # relative error (Frobenius); machine-precision target 1e-12
                relerr = norm(out - ref) / max(norm(ref), 1e-30)
                @test relerr < 1e-12
                max_err = max(max_err, relerr)
            end
        end
        @info "(b) forward vs dense" max_relerr=max_err threshold=1e-12
    end

    # ---------------------------------------------------------------------
    # (c) Adjoint matvec == dense L†·vec(ρ) to machine precision.
    #     The HS-adjoint of L_a·ρ·L_a† is L_a†·ρ·L_a; coherent sign-flips
    #     through G_left_adj = G_right, G_right_adj = G_left.
    # ---------------------------------------------------------------------
    @testset "(c) adjoint apply_adjoint_lindbladian! == dense L'·vec(ρ)" begin
        max_err = 0.0
        for n in _NS, bp in _BETAS_PHYS
            ham = _load_heis(n, bp)
            d = 2^n
            jumps = _jumps_in_basis(n, ham.eigvecs)
            cfg = _dll_bohr_cfg(n, ham, bp)
            validate_config!(cfg, ham)
            ws = Workspace(cfg, ham, jumps)
            L_dense = construct_lindbladian(jumps, cfg, ham)
            rng = MersenneTwister(2000 + 31 * n + round(Int, 100bp))
            for _ in 1:4
                rho = _rand_herm(rng, d)
                out = copy(apply_adjoint_lindbladian!(ws, copy(rho), cfg, ham))
                ref = reshape(L_dense' * vec(rho), d, d)
                relerr = norm(out - ref) / max(norm(ref), 1e-30)
                @test relerr < 1e-12
                max_err = max(max_err, relerr)
            end
        end
        @info "(c) adjoint vs dense" max_relerr=max_err threshold=1e-12
    end

    # ---------------------------------------------------------------------
    # (d) Matvec == explicit serial sum assembled from workspace matrices.
    #     This bypasses ALL matvec code (no apply_*! on the reference side),
    #     so it isolates the threaded reduction from the dense-vectorize path.
    #     forward : G_left·ρ + ρ·G_right + Σ_a L_a·ρ·L_a†
    #     adjoint : G_left_adj·ρ + ρ·G_right_adj + Σ_a L_a†·ρ·L_a
    # ---------------------------------------------------------------------
    @testset "(d) matvec == explicit G·ρ + Σ L_a·ρ·L_a† (bypasses matvec)" begin
        max_f = 0.0; max_a = 0.0
        for n in _NS
            ham = _load_heis(n, 0.5)
            d = 2^n
            jumps = _jumps_in_basis(n, ham.eigvecs)
            cfg = _dll_bohr_cfg(n, ham, 0.5)
            validate_config!(cfg, ham)
            ws = Workspace(cfg, ham, jumps)
            rng = MersenneTwister(3000 + n)
            for _ in 1:3
                rho = _rand_herm(rng, d)
                # explicit forward
                expl_f = ws.G_left * rho + rho * ws.G_right
                for L_a in ws.dll_lindblads
                    expl_f += L_a * rho * L_a'
                end
                out_f = copy(apply_lindbladian!(ws, copy(rho), cfg, ham))
                rf = norm(out_f - expl_f) / max(norm(expl_f), 1e-30)
                @test rf < 1e-12
                max_f = max(max_f, rf)
                # explicit adjoint
                expl_a = ws.G_right * rho + rho * ws.G_left
                for L_a in ws.dll_lindblads
                    expl_a += L_a' * rho * L_a
                end
                out_a = copy(apply_adjoint_lindbladian!(ws, copy(rho), cfg, ham))
                ra = norm(out_a - expl_a) / max(norm(expl_a), 1e-30)
                @test ra < 1e-12
                max_a = max(max_a, ra)
            end
        end
        @info "(d) vs explicit serial sum" max_fwd=max_f max_adj=max_a threshold=1e-12
    end

    # ---------------------------------------------------------------------
    # (e) Adjoint duality: tr(X' · L(Y)) == tr(L*(X)' · Y) (HS adjoint).
    #     Exercises forward and adjoint together on the same workspace
    #     within one iteration (the forward-then-adjoint reuse pattern that a
    #     Krylov solve drives), guarding against per-call state leakage.
    # ---------------------------------------------------------------------
    @testset "(e) adjoint duality tr(X'·L(Y)) == tr(L*(X)'·Y)" begin
        max_err = 0.0
        for n in _NS
            ham = _load_heis(n, 0.5)
            d = 2^n
            jumps = _jumps_in_basis(n, ham.eigvecs)
            cfg = _dll_bohr_cfg(n, ham, 0.5)
            validate_config!(cfg, ham)
            ws = Workspace(cfg, ham, jumps)
            rng = MersenneTwister(4000 + n)
            for _ in 1:4
                X = _rand_herm(rng, d); Y = _rand_herm(rng, d)
                L_Y    = copy(apply_lindbladian!(ws, copy(Y), cfg, ham))
                Lstar_X = copy(apply_adjoint_lindbladian!(ws, copy(X), cfg, ham))
                err = abs(tr(X' * L_Y) - tr(Lstar_X' * Y))
                @test err < 1e-11
                max_err = max(max_err, err)
            end
        end
        @info "(e) adjoint duality" max_err=max_err threshold=1e-11
    end

    # ---------------------------------------------------------------------
    # (f) Determinism + no state leak: repeated and interleaved fwd/adjoint
    #     calls on a SHARED workspace must reproduce the dense reference and
    #     be bit-identical call-to-call (the threaded reduction is order-
    #     deterministic: fixed chunk partition + fixed reduction order).
    # ---------------------------------------------------------------------
    @testset "(f) determinism + interleaved fwd/adjoint state-leak guard" begin
        n = 5  # threaded path
        ham = _load_heis(n, 0.5)
        d = 2^n
        jumps = _jumps_in_basis(n, ham.eigvecs)
        cfg = _dll_bohr_cfg(n, ham, 0.5)
        validate_config!(cfg, ham)
        ws = Workspace(cfg, ham, jumps)
        L_dense = construct_lindbladian(jumps, cfg, ham)
        rng = MersenneTwister(5005)
        rho = _rand_herm(rng, d)
        ref_f = reshape(L_dense * vec(rho), d, d)
        ref_a = reshape(L_dense' * vec(rho), d, d)

        # bit-identical determinism across two consecutive forward calls
        o1 = copy(apply_lindbladian!(ws, copy(rho), cfg, ham))
        o2 = copy(apply_lindbladian!(ws, copy(rho), cfg, ham))
        @test o1 == o2                       # exact equality: deterministic reduction
        @info "(f) determinism" call_to_call_maxabs=maximum(abs.(o1 .- o2))

        # interleave adjoint/forward/adjoint many times; no drift, no leak
        max_f = 0.0; max_a = 0.0
        for _ in 1:30
            oa  = copy(apply_adjoint_lindbladian!(ws, copy(rho), cfg, ham))
            of  = copy(apply_lindbladian!(ws, copy(rho), cfg, ham))
            oa2 = copy(apply_adjoint_lindbladian!(ws, copy(rho), cfg, ham))
            max_f = max(max_f, norm(of - ref_f) / norm(ref_f))
            max_a = max(max_a, norm(oa - ref_a) / norm(ref_a),
                                norm(oa2 - ref_a) / norm(ref_a))
        end
        @test max_f < 1e-12
        @test max_a < 1e-12
        @info "(f) interleaved 30×" max_fwd=max_f max_adj=max_a
    end

    # ---------------------------------------------------------------------
    # (g) Multi-channel filter (k=2): |dll_lindblads| = k·|𝒜| = 2·3n.
    #     The dissipator is a flat sum over all per-channel operators (no
    #     cross terms). Even n=3 with k=2 (=18 ≥ 10) takes the THREADED path,
    #     so this also covers threaded correctness at low n.
    # ---------------------------------------------------------------------
    @testset "(g) multi-channel k=2 forward+adjoint == dense" begin
        max_f = 0.0; max_a = 0.0
        for n in (3, 5)
            ham = _load_heis(n, 0.5)
            d = 2^n
            jumps = _jumps_in_basis(n, ham.eigvecs)
            β_alg = beta_alg(ham, 0.5)
            ch = DLLMetropolisFilter(β_alg; S = 2.0)
            multi = DLLMultiChannelFilter([ch, ch], β_alg)
            cfg = _dll_bohr_cfg(n, ham, 0.5; filter = multi)
            validate_config!(cfg, ham)
            ws = Workspace(cfg, ham, jumps)
            @test length(ws.dll_lindblads) == 2 * 3 * n
            @test length(ws.dll_lindblads) >= OMEGA_THREAD_THRESHOLD  # threaded if nt>1
            L_dense = construct_lindbladian(jumps, cfg, ham)
            rng = MersenneTwister(6000 + n)
            for _ in 1:3
                rho = _rand_herm(rng, d)
                of = copy(apply_lindbladian!(ws, copy(rho), cfg, ham))
                rf = norm(of - reshape(L_dense * vec(rho), d, d)) /
                     norm(reshape(L_dense * vec(rho), d, d))
                @test rf < 1e-12
                max_f = max(max_f, rf)
                oa = copy(apply_adjoint_lindbladian!(ws, copy(rho), cfg, ham))
                ra = norm(oa - reshape(L_dense' * vec(rho), d, d)) /
                     norm(reshape(L_dense' * vec(rho), d, d))
                @test ra < 1e-12
                max_a = max(max_a, ra)
            end
        end
        @info "(g) multi-channel k=2 vs dense" max_fwd=max_f max_adj=max_a threshold=1e-12
    end

    # ---------------------------------------------------------------------
    # (h) Empty-pool (nt<2) serial fallback INSIDE the helper. Build a
    #     KrylovScratch with num_threads=1 so `task_scratches` is empty; call
    #     `_apply_lindbladian_threaded_bohr_dll!` directly. It must hit the
    #     `nt < 2` fallback and accumulate the dissipator into the
    #     coherent-pre-seeded sc.rho_out, matching the dense reference.
    # ---------------------------------------------------------------------
    @testset "(h) nt<2 empty-pool fallback inside helper == dense" begin
        n = 5
        ham = _load_heis(n, 0.5)
        d = 2^n
        jumps = _jumps_in_basis(n, ham.eigvecs)
        cfg = _dll_bohr_cfg(n, ham, 0.5)
        validate_config!(cfg, ham)
        ws = Workspace(cfg, ham, jumps)
        L_dense = construct_lindbladian(jumps, cfg, ham)
        rng = MersenneTwister(7007)
        rho = _rand_herm(rng, d)

        # forward: empty-pool scratch, pre-seed coherent, call helper directly
        sc_f = KrylovScratch(ComplexF64, d; num_threads = 1)
        @test length(sc_f.task_scratches) == 0
        mul!(sc_f.rho_out, ws.G_left, rho)
        mul!(sc_f.rho_out, rho, ws.G_right, 1.0, 1.0)
        _apply_lindbladian_threaded_bohr_dll!(sc_f, rho, ws.dll_lindblads; adjoint = false)
        ref_f = reshape(L_dense * vec(rho), d, d)
        @test norm(sc_f.rho_out - ref_f) / norm(ref_f) < 1e-12

        # adjoint: same, with adjoint=true and the adj coherent seed
        sc_a = KrylovScratch(ComplexF64, d; num_threads = 1)
        mul!(sc_a.rho_out, ws.G_right, rho)
        mul!(sc_a.rho_out, rho, ws.G_left, 1.0, 1.0)
        _apply_lindbladian_threaded_bohr_dll!(sc_a, rho, ws.dll_lindblads; adjoint = true)
        ref_a = reshape(L_dense' * vec(rho), d, d)
        @test norm(sc_a.rho_out - ref_a) / norm(ref_a) < 1e-12
        @info "(h) empty-pool fallback OK"
    end

    # ---------------------------------------------------------------------
    # (i) The matvec leaves the caller-selected BLAS policy unchanged.
    # ---------------------------------------------------------------------
    @testset "(i) caller BLAS policy preservation" begin
        n = 5
        ham = _load_heis(n, 0.5)
        d = 2^n
        jumps = _jumps_in_basis(n, ham.eigvecs)
        cfg = _dll_bohr_cfg(n, ham, 0.5)
        validate_config!(cfg, ham)
        ws = Workspace(cfg, ham, jumps)
        rho = _rand_herm(MersenneTwister(8008), d)
        saved = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(2)
            caller_blas = BLAS.get_num_threads()
            @test caller_blas == 2
            apply_lindbladian!(ws, copy(rho), cfg, ham)
            @test BLAS.get_num_threads() == caller_blas
            apply_adjoint_lindbladian!(ws, copy(rho), cfg, ham)
            @test BLAS.get_num_threads() == caller_blas
        finally
            BLAS.set_num_threads(saved)
        end
    end

    # ---------------------------------------------------------------------
    # (j) Krylov spectral gap (threaded matvec inside the eigsolve) matches
    #     the dense eigenvalue gap — an end-to-end check that the threaded
    #     matvec produces the correct operator spectrum, not just pointwise
    #     correct images. n=4 (threaded), small enough for a dense eigvals.
    # ---------------------------------------------------------------------
    @testset "(j) krylov_spectral_gap (threaded) == dense eigen gap" begin
        n = 4
        ham = _load_heis(n, 0.5)
        jumps = _jumps_in_basis(n, ham.eigvecs)
        cfg = _dll_bohr_cfg(n, ham, 0.5)
        validate_config!(cfg, ham)
        L_dense = construct_lindbladian(jumps, cfg, ham)
        ev = eigvals(L_dense)
        perm = sortperm(ev; by = v -> abs(real(v)))
        gap_dense = abs(real(ev[perm][2]))
        res = krylov_spectral_gap(cfg, ham, jumps; krylovdim = 40, howmany = 6, tol = 1e-10)
        @test isapprox(res.spectral_gap, gap_dense; rtol = 1e-9)
        @info "(j) gap" gap_dense=gap_dense gap_krylov=res.spectral_gap
    end

end  # @testset "DLL BohrDomain threaded matvec"
