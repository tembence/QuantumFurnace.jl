using LinearAlgebra: I, eigvals, svdvals, norm, tr, Hermitian
using Random
using Test
using QuantumFurnace

# Compare the shared channel kernel with thermalization, check cross-domain
# agreement and splitting-error scaling, and verify workspace reuse.
# These n=3 fixtures complement the n=4 byte-parity test in test_predict_channel.jl.

@testset "Faithful apply_delta_channel!" begin

    # -----------------------------------------------------------------------
    # (1) Faithfulness vs run_thermalize :sweep
    #
    # `apply_delta_channel!` per-jump-sweeps the SAME `_apply_one_dm_substep!`
    # kernel `run_thermalize :sweep` calls (`src/furnace.jl:230-249`). Wire
    # both to the same Workspace-built K0s/U_residuals/U_coherents and apply
    # for the same number of steps; the resulting density matrices must be
    # bit-identical to FP rounding. A larger n=4, β=10, δ=1e-3 byte-identity
    # check (k_grid=0:50:1000) is in test_predict_channel.jl testset (a);
    # this is the small-fixture variant for fast CI.
    # -----------------------------------------------------------------------
    @testset "(1) Faithfulness vs run_thermalize :sweep (n=3 BohrDomain)" begin
        beta = 10.0
        delta = 1e-2
        sys = make_dll_n3_system(beta)
        ham = sys.ham
        jumps = sys.jumps
        d = size(ham.data, 1)

        cfg = Config(
            sim = Thermalize(),
            domain = BohrDomain(),
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta, sigma = 1.0 / beta,
            a = 0.0, s = 0.25,
            num_energy_bits = 12,
            w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
            delta = delta,
            mixing_time = 0.5,  # 50 outer steps at delta=1e-2
            jump_selection = :sweep,
        )

        rho_0 = Matrix{ComplexF64}(I(d) / d)
        n_steps = 50

        # Reference: run_thermalize :sweep, save_every=n_steps so trace_distances
        # has length 2 (initial + final). The final dm is what we compare against.
        res_th = run_thermalize(jumps, cfg, ham; initial_dm=copy(rho_0),
                                save_every=n_steps)

        # Predict via apply_delta_channel! sweep (no Krylov reconstruction —
        # we forward-iterate the matvec n_steps times). This is the per-step
        # ground-truth check: Φ_δ^k(rho_0) for the same k both ways.
        ws = Workspace(cfg, ham, jumps)
        rho_pred = copy(rho_0)
        for _ in 1:n_steps
            apply_delta_channel!(ws, rho_pred, cfg, ham)
            rho_pred = copy(ws.scratch.rho_next)
        end

        # both paths sweep the SAME `_apply_one_dm_substep!`
        # kernel n_steps × n_jumps times in the same order, so any drift comes
        # purely from FP accumulation order — ~ O(n_steps · n_jumps · DIM² · eps)
        # ≈ 50 · 9 · 64 · 2.2e-16 ≈ 6e-12. Threshold 1e-10 keeps a ~15× margin.
        diff_F = norm(rho_pred - res_th.final_dm)
        @test diff_F < 1e-10
        @info "(1) Faithfulness vs run_thermalize :sweep" diff_F threshold=1e-10 n_steps=n_steps
    end

    # -----------------------------------------------------------------------
    # (2) Cross-domain agreement: BohrDomain ≡ EnergyDomain
    #
    # The faithful Φ_δ matvec should produce identical density-matrix
    # trajectories in BohrDomain (closed-form α(ν), no quadrature) and
    # EnergyDomain (Eb=12 OFT discretisation), modulo NUFFT/quadrature
    # precision. Both build the dissipator and coherent unitary in the
    # SAME Hamiltonian eigenbasis, so there is no basis-transform
    # contribution to the residual.
    #
    # NOTE: a TrotterDomain ≡ BohrDomain check at this threshold is
    # physically infeasible because TrotterDomain uses Trotterised
    # `e^{iHt}` operators that introduce O(M·δt_0²·β²) splitting error
    # per outer step (~6e-3 over 50 steps at our standard fixture). That
    # splitting is faithful to the implementable algorithm; the BohrDomain
    # is exact. The cross-domain test isolates the predictor pipeline, not
    # the algorithmic Trotter error. TrotterDomain ≡ run_thermalize byte-
    # identity is regression-tested in test_predict_channel.jl testset (b1).
    # -----------------------------------------------------------------------
    @testset "(2) Cross-domain agreement BohrDomain ≡ EnergyDomain (n=3)" begin
        beta = 10.0
        delta = 1e-3
        sys = make_dll_n3_system(beta)
        ham = sys.ham
        jumps = sys.jumps
        d = size(ham.data, 1)

        # BohrDomain config — closed-form α(ν), no quadrature.
        cfg_b = Config(
            sim = Thermalize(),
            domain = BohrDomain(),
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta, sigma = 1.0 / beta,
            a = 0.0, s = 0.25,
            num_energy_bits = 12,
            w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
            delta = delta,
            mixing_time = 0.05,  # 50 steps at δ=1e-3
            jump_selection = :sweep,
        )

        # EnergyDomain config — Eb=12 OFT (4096-point Gaussian quadrature).
        cfg_e = Config(
            sim = Thermalize(),
            domain = EnergyDomain(),
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta, sigma = 1.0 / beta,
            a = 0.0, s = 0.25,
            num_energy_bits = 12,
            w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
            delta = delta,
            mixing_time = 0.05,
            jump_selection = :sweep,
        )

        rho_0 = Matrix{ComplexF64}(I(d) / d)
        n_steps = 50

        # BohrDomain forward iteration.
        ws_b = Workspace(cfg_b, ham, jumps)
        rho_b = copy(rho_0)
        for _ in 1:n_steps
            apply_delta_channel!(ws_b, rho_b, cfg_b, ham)
            rho_b = copy(ws_b.scratch.rho_next)
        end

        # EnergyDomain forward iteration.
        ws_e = Workspace(cfg_e, ham, jumps)
        rho_e = copy(rho_0)
        for _ in 1:n_steps
            apply_delta_channel!(ws_e, rho_e, cfg_e, ham)
            rho_e = copy(ws_e.scratch.rho_next)
        end

        # At Eb=12, w0=0.05 the measured per-matvec quadrature error is ~1e-9.
        # The 1e-5 threshold allows accumulation over 50 steps and 9 jumps.
        diff_F = norm(rho_b - rho_e)
        @test diff_F < 1e-5
        @info "(2) Cross-domain agreement BohrDomain ≡ EnergyDomain" diff_F threshold=1e-5 n_steps=n_steps
    end

    # -----------------------------------------------------------------------
    # (4) Splitting-error scaling slope
    #
    # The faithful per-jump Lie–Trotter Φ_δ is an O(δ) approximation to
    # e^{δ𝓛}; the leading splitting error is O(δ²). Verify by comparing
    # against the Euler step (I + δ·𝓛_dense) — also O(δ) approximant —
    # over δ ∈ {1e-2, 5e-3, 1e-3, 5e-4, 1e-4}: the difference
    # ‖Φ_δ_faithful(ρ) − (I + δ·𝓛)·vec(ρ)‖ scales as δ². Linear-fit the
    # log-log slope; expect ≈ 2 ± 0.1.
    #
    # This is the same Euler closeness check used in test_krylov_eigsolve.jl
    # testset 1 — promoted here to a multi-δ slope sweep so the scaling
    # exponent is verified, not just the order constant.
    # -----------------------------------------------------------------------
    @testset "(4) Splitting-error scaling slope ≈ 2" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        ham = sys.ham
        jumps = sys.jumps
        d = size(ham.data, 1)

        # Lindbladian config (no delta) for dense L construction.
        cfg_L = Config(
            sim = Lindbladian(),
            domain = EnergyDomain(),
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta, sigma = 1.0 / beta,
            a = 0.0, s = 0.25,
            num_energy_bits = 12,
            w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
        )
        L_dense = construct_lindbladian(jumps, cfg_L, ham)

        Random.seed!(0)
        rho = Matrix(random_density_matrix(3))
        I_d2 = Matrix{ComplexF64}(I(d^2))

        deltas = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
        errs = Float64[]
        for delta in deltas
            cfg_T = Config(
                sim = Thermalize(),
                domain = EnergyDomain(),
                construction = KMS(),
                num_qubits = 3,
                with_linear_combination = true,
                beta = beta, sigma = 1.0 / beta,
                a = 0.0, s = 0.25,
                num_energy_bits = 12,
                w0 = 0.05,
                t0 = 2π / (2^12 * 0.05),
                num_trotter_steps_per_t0 = 10,
                delta = delta,
                mixing_time = 1.0,
                jump_selection = :sweep,
            )
            ws = Workspace(cfg_T, ham, jumps)
            rho_in = copy(rho)
            apply_delta_channel!(ws, rho_in, cfg_T, ham)
            rho_chen = copy(ws.scratch.rho_next)

            # Euler step: same O(δ) approximant.
            v_euler = (I_d2 + delta .* L_dense) * vec(rho)
            err = norm(vec(rho_chen) - v_euler)
            push!(errs, err)
        end

        # Linear fit of log(err) vs log(δ): slope ≈ 2 expected.
        log_d = log.(deltas)
        log_e = log.(errs)
        n = length(deltas)
        # Standard least-squares slope: (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²).
        Σx = sum(log_d); Σy = sum(log_e)
        Σxy = sum(log_d .* log_e); Σx2 = sum(log_d .^ 2)
        slope = (n * Σxy - Σx * Σy) / (n * Σx2 - Σx^2)

        # faithful Lie–Trotter on n_jumps substeps composes
        # into Φ_δ = e^{δ𝓛} + O(δ²·∑[𝓛_a, 𝓛_b]); the difference vs Euler
        # I + δ𝓛 is also O(δ²) (Euler omits all higher-order terms). So
        # both share the same O(δ²) leading correction; their difference
        # is also O(δ²). Slope ≈ 2 ± 0.1 covers the linear-fit residual
        # at 5 sample points across two decades of δ.
        @test 1.9 ≤ slope ≤ 2.1
        @info "(4) Splitting-error scaling slope" slope errs deltas n_pts=n
    end

    # -----------------------------------------------------------------------
    # (5) Threading bit-match
    #
    # Two consecutive `apply_delta_channel!` calls on a fresh Workspace with
    # the same input ρ must produce bit-identical results — the threaded
    # `_accumulate_rho_jump_threaded_*!` reduction (chunked sum into
    # `task_scratches[idx].rho_jump`) is deterministic, and the `task_scratches`
    # pool must reset cleanly between calls.
    # Catches any state-leak in the pool or non-determinism in the chunked
    # reduction.
    # -----------------------------------------------------------------------
    @testset "(5) Threading bit-match: deterministic across calls" begin
        beta = 10.0
        delta = 1e-3
        sys = make_dll_n3_system(beta)
        ham = sys.ham
        jumps = sys.jumps
        d = size(ham.data, 1)

        cfg = Config(
            sim = Thermalize(),
            domain = EnergyDomain(),  # threaded path: n_labels = 4096 ≫ OMEGA_THREAD_THRESHOLD=10
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta, sigma = 1.0 / beta,
            a = 0.0, s = 0.25,
            num_energy_bits = 12,
            w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
            delta = delta,
            mixing_time = 1.0,
            jump_selection = :sweep,
        )

        Random.seed!(0)
        rho = Matrix(random_density_matrix(3))

        # Two independent Workspaces, each running one matvec from the same
        # input: the threaded reduction order across `task_scratches` chunks
        # must be deterministic ⇒ bit-identical outputs.
        ws_a = Workspace(cfg, ham, jumps)
        rho_a = copy(rho)
        apply_delta_channel!(ws_a, rho_a, cfg, ham)
        rho_a_out = copy(ws_a.scratch.rho_next)

        ws_b = Workspace(cfg, ham, jumps)
        rho_b = copy(rho)
        apply_delta_channel!(ws_b, rho_b, cfg, ham)
        rho_b_out = copy(ws_b.scratch.rho_next)

        # Bit-identity threshold: any non-zero diff would indicate
        # non-deterministic threading reduction or pool state leak. Threshold
        # 1e-12 · max(1, ‖rho‖) covers FP rounding without tolerating real drift.
        tol = 1e-12 * max(1.0, norm(rho_a_out))
        @test norm(rho_a_out - rho_b_out) ≤ tol
        @info "(5) Threading bit-match across calls" diff=norm(rho_a_out - rho_b_out) tol nthreads=Threads.nthreads()

        # Sanity: also test in-place reuse — same Workspace, two calls from
        # the same input. The matvec writes to ws.scratch.rho_next; nothing
        # in ws should retain rho-dependent state across calls.
        ws_c = Workspace(cfg, ham, jumps)
        rho_c1 = copy(rho)
        apply_delta_channel!(ws_c, rho_c1, cfg, ham)
        out_c1 = copy(ws_c.scratch.rho_next)
        rho_c2 = copy(rho)
        apply_delta_channel!(ws_c, rho_c2, cfg, ham)
        out_c2 = copy(ws_c.scratch.rho_next)
        @test norm(out_c1 - out_c2) ≤ tol
        @info "(5) Threading bit-match same Workspace" diff=norm(out_c1 - out_c2) tol
    end
end
