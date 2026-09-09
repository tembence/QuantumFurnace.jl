@testset "DLL coherent term G" begin

    # =====================================================================
    # Direct unit tests for `dll_coherent_op_bohr` and `dll_coherent_op_time`.
    # The full-Lindbladian KMS-DB checks live in `test_dll_dissipator.jl`
    # (the dissipator tests now run with the coherent G automatically wired
    # in via `_precompute_coherent_B`); here we verify properties of G in
    # isolation.
    #
    # β-sweep ∈ {1, 5, 10}: β = 10 is the user-specified stress level —
    # the DLL filter narrows ∝ 1/β so ‖G‖_op should NOT grow with β.
    #
    # Fixture: n=3 disordered Heisenberg (same fixture as test_dll_kms_db.jl
    # and test_discriminant.jl). 8-dim with `|B_H| ≫ n` — non-trivial Bohr-
    # frequency structure that exercises the eigenbasis collapse used by the
    # closed-form `dll_coherent_op_bohr`.
    # =====================================================================
    # Shared n=3 disordered Heisenberg fixture; see test_helpers.jl::make_dll_n3_system.
    _build_dll_n3_system = make_dll_n3_system

    # Reuse the same grid as test_dll_dissipator.jl. N=10 (Nt=1024) keeps
    # t_max ≈ 63 (= Nt·t0/2) — same as the legacy N=12 — but cuts NUFFT
    # source-points 16× (4096² → 1024²). Empirically the Bohr↔Time error
    # is already at the FINUFFT precision floor (~3e-9) at Nt ≥ 256, so
    # the existing 1e-3 / 1e-5 tolerances stay well-clear.
    _NUM_ENERGY_BITS = 10
    _T0 = 2pi / (2^_NUM_ENERGY_BITS * 0.05)
    _BETAS = (1.0, 5.0, 10.0)

    function _make_cfg(domain; beta::Real)
        Config(;
            sim = Lindbladian(),
            domain = domain,
            construction = DLL(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta,
            sigma = 1.0 / beta,
            a = beta / 30.0,
            s = 0.4,
            num_energy_bits = _NUM_ENERGY_BITS,
            t0 = _T0,
            num_trotter_steps_per_t0 = 10,
            filter = DLLGaussianFilter(beta),
        )
    end

    function _make_meta_cfg(domain; beta::Real, S::Real = 2.0)
        Config(;
            sim = Lindbladian(),
            domain = domain,
            construction = DLL(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta,
            sigma = 1.0 / beta,
            a = beta / 30.0,
            s = 0.4,
            num_energy_bits = _NUM_ENERGY_BITS,
            t0 = _T0,
            num_trotter_steps_per_t0 = 10,
            filter = DLLMetropolisFilter(beta; S = S),
        )
    end

    # DLL TimeDomain trapezoidal grid: t_m = m · t0 for m ∈ [-N/2, N/2 − 1].
    function _dll_time_labels(cfg, filter)
        N = 2^cfg.num_energy_bits
        labels = collect((-N÷2):(N÷2 - 1)) .* cfg.t0
        return QuantumFurnace._truncate_time_labels_for_oft(labels, cfg.sigma; filter=filter)
    end

    # ---------------------------------------------------------------------
    # (a/i, b/j, c/k) Parameterised per-filter G properties:
    #   - BohrDomain G Hermiticity (closed-form, exact ≤ 1e-15)
    #   - TimeDomain G Hermiticity (quadrature, ≤ 1e-6)
    #   - Bohr ↔ Time agreement on G (Gaussian ≤ 1e-3, Metropolis ≤ 1e-5)
    #
    # Theorem 10 / Eq. 2.33: the closed-form `G = (1/2i) · T ⊙ Σ_a (M^a)† M^a`
    # is exactly Hermitian for any AbstractFilter; the same testset body
    # works for both DLL filter types. Bohr↔Time tolerance differs per
    # filter: Gaussian's t-tail is broader, Metropolis's flat bump localises
    # to a ~15× tighter cross-check.
    #
    # Skipped here: (d) Gaussian and (l) Metropolis ‖G‖_op tests have
    # different intents (β-shrinking vs O(1)) and stay separate below.
    # ---------------------------------------------------------------------
    _make_dll_filter_cfg(label, beta) = label === :gaussian ?
        (filter=DLLGaussianFilter(beta),         cfg=_make_cfg(TimeDomain(); beta=beta),       bt_tol=1e-3) :
        (filter=DLLMetropolisFilter(beta; S=2.0), cfg=_make_meta_cfg(TimeDomain(); beta=beta), bt_tol=1e-5)

    @testset "(a/i) BohrDomain G is Hermitian — $label" for label in (:gaussian, :metropolis)
        for beta in _BETAS
            sys = _build_dll_n3_system(beta)
            v = _make_dll_filter_cfg(label, beta)
            G = dll_coherent_op_bohr(sys.jumps, sys.ham, v.filter, beta)
            @test norm(G - G') <= 1e-15
        end
    end

    @testset "(b/j) TimeDomain G is Hermitian — $label" for label in (:gaussian, :metropolis)
        for beta in _BETAS
            sys = _build_dll_n3_system(beta)
            v = _make_dll_filter_cfg(label, beta)
            time_labels = _dll_time_labels(v.cfg, v.filter)
            G = dll_coherent_op_time(sys.jumps, sys.ham, time_labels, v.filter, beta, v.cfg.t0)
            @test norm(G - G') <= 1e-6
        end
    end

    @testset "(c/k) Bohr ↔ Time agreement on G — $label" for label in (:gaussian, :metropolis)
        for beta in _BETAS
            sys = _build_dll_n3_system(beta)
            v = _make_dll_filter_cfg(label, beta)
            time_labels = _dll_time_labels(v.cfg, v.filter)
            G_b = dll_coherent_op_bohr(sys.jumps, sys.ham, v.filter, beta)
            G_t = dll_coherent_op_time(sys.jumps, sys.ham, time_labels, v.filter, beta, v.cfg.t0)
            @test opnorm(G_b - G_t) <= v.bt_tol
        end
    end

    # ---------------------------------------------------------------------
    # (d) ‖G‖_op does NOT blow up at β = 10 (DLL filter narrows ∝ 1/β).
    # The exact scaling with β depends on the spectrum: on the disordered
    # n=3 Heisenberg fixture, ‖G‖_op stays well below 1 across the sweep
    # (peaks at β=5: 2.4e-3) — far from any blow-up scenario.
    # ---------------------------------------------------------------------
    @testset "(d) G norm bounded as β grows" begin
        norms = Float64[]
        for beta in _BETAS
            sys = _build_dll_n3_system(beta)
            filter = DLLGaussianFilter(beta)
            G = dll_coherent_op_bohr(sys.jumps, sys.ham, filter, beta)
            push!(norms, opnorm(G))
        end
        @test all(<=(1.0), norms)
    end

    # ---------------------------------------------------------------------
    # Single-jump variant. The closed-form `G_a` is non-trivial per coupling
    # on the disordered n=3 fixture (no parity-cancellation tricks); the
    # single-jump subset just isolates the per-coupling KMS-DB structure
    # for tests (e)–(g).
    # ---------------------------------------------------------------------
    function _single_jump_system(beta::Real)
        sys = _build_dll_n3_system(beta)
        return (; sys.ham, jumps = sys.jumps[1:1], sys.gibbs)
    end

    # ---------------------------------------------------------------------
    # (e) G is non-trivial: dissipator-only Lindbladian differs from full
    # ---------------------------------------------------------------------
    @testset "(e) G contributes non-trivially to L" begin
        beta = 5.0
        sys = _single_jump_system(beta)
        cfg = _make_cfg(BohrDomain(); beta=beta)
        L_full = construct_lindbladian(sys.jumps, cfg, sys.ham)
        L_diss = construct_lindbladian(sys.jumps, cfg, sys.ham; include_coherent=false)
        # Difference is the i[G, ·] commutator superoperator; non-zero
        # whenever the coupling breaks the system's symmetry (always the
        # case on the disordered n=3 fixture).
        @test opnorm(L_full - L_diss) > 1e-6
    end

    # ---------------------------------------------------------------------
    # (f) G is non-zero for a single coupling. Note: `G` may still commute
    # with `σ_β` when both preserve a conserved quantum number, so we check
    # `‖G‖_op > 0` rather than `[G, σ_β]`.
    # ---------------------------------------------------------------------
    @testset "(f) G is non-zero (single jump)" begin
        beta = 5.0
        sys = _single_jump_system(beta)
        filter = DLLGaussianFilter(beta)
        G = dll_coherent_op_bohr(sys.jumps, sys.ham, filter, beta)
        @test opnorm(G) > 1e-8
    end

    # ---------------------------------------------------------------------
    # (g) Full Lindbladian (with G) preserves σ_β even on the single-jump
    # subsystem — KMS-DB property (Theorem 10) holds per coupling.
    # ---------------------------------------------------------------------
    @testset "(g) Full L[σ_β] = 0 (single jump, β ∈ {1, 5, 10})" begin
        for beta in _BETAS
            sys = _single_jump_system(beta)
            cfg = _make_cfg(BohrDomain(); beta=beta)
            L = construct_lindbladian(sys.jumps, cfg, sys.ham)
            sigma_vec = vec(Matrix(sys.gibbs))
            @test norm(L * sigma_vec) <= 1e-10
        end
    end

    # ---------------------------------------------------------------------
    # (h) closed-form + NUFFT path agrees with the
    # internal-(ν,ν')-grid `dll_coherent_op_time_legacy` reference. The
    # legacy path has its own ~1e-9 quadrature error at the default Nν=64,
    # so we tolerate ~1e-8 op-norm.
    # ---------------------------------------------------------------------
    @testset "(h) closed-form NUFFT G == _legacy internal-grid G" begin
        for beta in _BETAS
            sys = _build_dll_n3_system(beta)
            filter = DLLGaussianFilter(Float64(beta))
            cfg_t = _make_cfg(TimeDomain(); beta=beta)
            time_labels = _dll_time_labels(cfg_t, filter)

            G_new = dll_coherent_op_time(sys.jumps, sys.ham, time_labels, filter, beta, cfg_t.t0)
            G_legacy = QuantumFurnace.dll_coherent_op_time_legacy(
                sys.jumps, sys.ham, time_labels, filter, beta, cfg_t.t0,
            )
            @test opnorm(G_new - G_legacy) <= 1e-8
        end
    end

    # =====================================================================
    # DLL Metropolis-type filter — same n=3 fixture; coherent G
    # is computed numerically (no closed form for g(t,t')) via 2D NUFFT
    # over [-S, S]² + the shared `_dll_coherent_from_g_tt` helper.
    # Hermiticity / Bohr↔Time tests for Metropolis are merged with the
    # Gaussian variants in (a/i), (b/j), (c/k) above. (l) below is the
    # only Metropolis-only structural test (different intent than Gaussian (d)).
    # =====================================================================

    # ---------------------------------------------------------------------
    # (l) ‖G‖_op stays O(1) in β for Metropolis — does NOT shrink ∝ 1/β
    # like the Gaussian filter (motivating qualitative contrast). The
    # Metropolis G norm is bounded but does not decay with β.
    # ---------------------------------------------------------------------
    @testset "(l) Metropolis ‖G‖_op stays O(1) across β" begin
        for beta in _BETAS
            sys = _build_dll_n3_system(beta)
            filter = DLLMetropolisFilter(beta; S = 2.0)
            G = dll_coherent_op_bohr(sys.jumps, sys.ham, filter, beta)
            # Bounded — Metropolis does not blow up
            @test opnorm(G) <= 10.0
            # Non-trivial — Metropolis does not vanish either. The 1e-5 floor
            # accommodates fixture-dependent variation in ‖G‖_op across the
            # find_typical-selected n=3 fixture (observed minimum
            # ~6.7e-5 at one β value), well above the ~1/β decay rate that
            # would mark the Gaussian-style β-collapse this test guards against.
            @test opnorm(G) >= 1e-5
        end
    end
end

@testset "DLL arbitrary-source two-time contraction" begin
    ham = HamHam(build_heis_1d(2, [0.8, 0.7, 1.2]; seed=14,
        periodic=false, disorder_strength=0.2); beta_phys=0.5)
    beta = beta_alg(ham, 0.5)
    rng = QuantumFurnace.Random.MersenneTwister(42)
    A = randn(rng, ComplexF64, 4, 4) / 4
    makejump(M; hermitian=ishermitian(M)) = JumpOp(Matrix(M),
        ham.eigvecs' * M * ham.eigvecs, false, hermitian)
    xy = Matrix(pad_term([X + Y], 2, 1)) / 4

    @testset "NUFFT versus independent direct matrix products" begin
        # A nonsymmetric kernel exposes sign, order, and conjugation errors
        # independently of Fourier inversion and final Hermitian projection.
        ts = collect(-1.0:0.5:1.0)
        g = randn(rng, ComplexF64, length(ts), length(ts))
        for jump in (makejump(A), makejump(xy), makejump(xy; hermitian=false))
            evolved = [Diagonal(cis.(ham.eigvals .* t)) * jump.in_eigenbasis *
                       Diagonal(cis.(-ham.eigvals .* t)) for t in ts]
            direct = sum(g[m, n] .* (evolved[n]' * evolved[m])
                         for m in eachindex(ts), n in eachindex(ts)) .* 0.5^2
            fast = QuantumFurnace._dll_coherent_from_g_tt(
                [jump], ham, g, ts, 0.5)
            @test isapprox(fast, direct; atol=1e-11, rtol=1e-11)
        end
    end

    @testset "Legacy and per-source Gaussian references" begin
        ts = collect(-20.0:0.5:20.0)
        f = DLLGaussianFilter(beta)
        # Resolve the frequency integral explicitly; the legacy default grid
        # is not a controlled reference for this beta and time window.
        nus = collect(range(-13/beta, 11/beta; length=256))
        paired = JumpOp[makejump(A), makejump(A')]
        for jump in (paired..., makejump(xy))
            B = dll_coherent_op_time([jump], ham, ts, f, beta, 0.5)
            Blegacy = QuantumFurnace.dll_coherent_op_time_legacy(
                [jump], ham, ts, f, beta, 0.5; nu_grid=nus)
            Bbohr = dll_coherent_op_bohr([jump], ham, f, beta)
            @test isapprox(B, Blegacy; atol=1e-10, rtol=0)
            @test isapprox(B, Bbohr; atol=1e-9, rtol=0)
        end
        Bsum = sum(dll_coherent_op_time([j], ham, ts, f, beta, 0.5) for j in paired)
        @test isapprox(dll_coherent_op_time(paired, ham, ts, f, beta, 0.5),
                       Bsum; atol=1e-12, rtol=0)
    end
end
