@testset "DLL multi-channel simulator integration" begin
    # =====================================================================
    # End-to-end smoke tests for `Config{Lindbladian, *, DLL}` with a
    # multi-channel filter. Verifies that:
    #   - validate_config! accepts the multi-channel filter,
    #   - construct_lindbladian builds without error,
    #   - the resulting Liouvillian preserves σ_β at machine precision (Bohr)
    #     or quadrature precision (Time),
    #   - k=1 reduction matches the standard single-channel Liouvillian byte
    #     for byte (BohrDomain).
    # =====================================================================
    _BETAS = (1.0, 5.0, 10.0)
    # N=10 (Nt=1024): same t_max ≈ 63 as legacy N=12, 16× less NUFFT memory.
    # Bohr↔Time error already at FINUFFT floor by Nt ≥ 256. Required
    # to keep the k∈{1,2,4} channel sweep below the 3.5 GB sandbox cap.
    _NEB = 10
    _T0_CFG = 2π / (2^_NEB * 0.05)

    function _make_cfg(
        domain,
        beta,
        filter;
        num_energy_bits::Int = _NEB,
        t0::Real = _T0_CFG,
    )
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
            num_energy_bits = num_energy_bits,
            t0 = t0,
            num_trotter_steps_per_t0 = 10,
            filter = filter,
        )
    end

    # ---------------------------------------------------------------------
    # (a) k=1 BohrDomain Liouvillian byte-identity to single-channel filter.
    # ---------------------------------------------------------------------
    @testset "(a) k=1 Bohr Liouvillian == single-channel byte-identity" begin
        for beta in _BETAS
            sys = make_dll_n3_system(beta)
            for ch in (DLLGaussianFilter(beta), DLLMetropolisFilter(beta; S = 2.0))
                multi = DLLMultiChannelFilter([ch], beta)
                cfg_single = _make_cfg(BohrDomain(), beta, ch)
                cfg_multi  = _make_cfg(BohrDomain(), beta, multi)
                L_single = construct_lindbladian(sys.jumps, cfg_single, sys.ham)
                L_multi  = construct_lindbladian(sys.jumps, cfg_multi,  sys.ham)
                @test L_single == L_multi
            end
        end
    end

    # ---------------------------------------------------------------------
    # (b) k=2 identical-channels Bohr Liouvillian = 2 · single-channel L.
    # ---------------------------------------------------------------------
    @testset "(b) k=2 identical channels: L_multi = 2·L_single" begin
        beta = 5.0
        sys = make_dll_n3_system(beta)
        ch = DLLMetropolisFilter(beta; S = 2.0)
        multi = DLLMultiChannelFilter([ch, ch], beta)
        L_single = construct_lindbladian(sys.jumps, _make_cfg(BohrDomain(), beta, ch), sys.ham)
        L_multi  = construct_lindbladian(sys.jumps, _make_cfg(BohrDomain(), beta, multi), sys.ham)
        @test isapprox(L_multi, 2 .* L_single; atol = 1e-12, rtol = 1e-12)
    end

    # ---------------------------------------------------------------------
    # (c) Multi-channel BohrDomain Liouvillian preserves σ_β at machine
    #     precision (Theorem 10 of Ding–Li–Lin holds per channel; the sum
    #     of KMS-DBC Lindbladians is again KMS-DBC).
    # ---------------------------------------------------------------------
    @testset "(c) Multi-channel Bohr: L[σ_β] ≈ 0" begin
        for beta in _BETAS
            sys = make_dll_n3_system(beta)
            ch1 = DLLMetropolisFilter(beta; S = 2.0)
            ch2 = DLLMetropolisFilter(beta; S = 3.0)
            multi = DLLMultiChannelFilter([ch1, ch2], beta)
            L = construct_lindbladian(sys.jumps, _make_cfg(BohrDomain(), beta, multi), sys.ham)
            σ_vec = vec(Matrix(sys.gibbs))
            @test norm(L * σ_vec) <= 1e-10
        end
    end

    # ---------------------------------------------------------------------
    # (d) Multi-channel TimeDomain Liouvillian preserves σ_β within
    #     trapezoidal-quadrature error.
    # ---------------------------------------------------------------------
    @testset "(d) Multi-channel Time: L[σ_β] ≤ quadrature tol" begin
        for beta in _BETAS
            sys = make_dll_n3_system(beta)
            ch1 = DLLMetropolisFilter(beta; S = 2.0)
            ch2 = DLLMetropolisFilter(beta; S = 3.0)
            multi = DLLMultiChannelFilter([ch1, ch2], beta)
            L = construct_lindbladian(sys.jumps, _make_cfg(TimeDomain(), beta, multi), sys.ham)
            σ_vec = vec(Matrix(sys.gibbs))
            @test norm(L * σ_vec) <= 1e-4
        end
    end

    # ---------------------------------------------------------------------
    # (e) Multi-channel BohrDomain Liouvillian is dual trace-preserving.
    # ---------------------------------------------------------------------
    @testset "(e) Multi-channel Bohr: L†[I] ≈ 0" begin
        for beta in _BETAS
            sys = make_dll_n3_system(beta)
            ch1 = DLLMetropolisFilter(beta; S = 2.0)
            ch2 = DLLMetropolisFilter(beta; S = 3.0)
            multi = DLLMultiChannelFilter([ch1, ch2], beta)
            L = construct_lindbladian(sys.jumps, _make_cfg(BohrDomain(), beta, multi), sys.ham)
            Id_vec = ComplexF64.(vec(Matrix(I, 8, 8)))
            @test norm(L' * Id_vec) <= 1e-10
        end
    end

    # ---------------------------------------------------------------------
    # (f) Multi-channel Bohr ↔ Time consistency on the same Liouvillian.
    # ---------------------------------------------------------------------
    @testset "(f) Multi-channel Bohr ↔ Time consistency" begin
        for beta in _BETAS
            sys = make_dll_n3_system(beta)
            ch1 = DLLMetropolisFilter(beta; S = 2.0)
            ch2 = DLLMetropolisFilter(beta; S = 3.0)
            multi = DLLMultiChannelFilter([ch1, ch2], beta)
            L_b = construct_lindbladian(sys.jumps, _make_cfg(BohrDomain(), beta, multi), sys.ham)
            L_t = construct_lindbladian(sys.jumps, _make_cfg(TimeDomain(), beta, multi), sys.ham)
            @test opnorm(L_b - L_t) <= 1e-4
        end
    end

    # ---------------------------------------------------------------------
    # (g.0) **Physics anchor — full KMS detailed balance** (Theorem 10).
    #
    # Theorem 10 of Ding–Li–Lin (2024) characterises σ_β-KMS DBC as:
    #
    #   L(X) = i[G, X] + Σ_j (L_j† X L_j − ½ {L_j† L_j, X}),
    #
    # with each L_j satisfying Δ_{σ_β}^{-1/2} L_j = L_j† (Eq. 2.28 / 3.2)
    # and `G = −i tanh ∘ log(Δ^{1/4}_{σ_β}) · (½ Σ_j L_j† L_j)` (Eq. 2.33).
    #
    # The map V → −i tanh ∘ log(Δ^{1/4}) · V is *linear* (a fixed
    # superoperator depending only on σ_β), so for a flat
    # `{L_j} = {L_a^(ℓ) : a ∈ couplings, ℓ ∈ channels}` the joint G is
    #
    #   G^multi = −i tanh ∘ log(Δ^{1/4}) · ½ Σ_{a, ℓ} (L_a^(ℓ))† L_a^(ℓ)
    #           = Σ_{a, ℓ} G_{a, ℓ},
    #
    # the sum of per-channel coherent operators. This is exactly what
    # `dll_coherent_op_bohr(jumps, ham, ::DLLMultiChannelFilter, β)`
    # returns. The shifted-symmetric `q_ℓ` is real-even, so
    # each channel satisfies q_ℓ(-ν) = q_ℓ(ν)^* = q_ℓ(ν) (Eq. 3.2),
    # giving a valid KMS-DBC L_a^(ℓ).
    #
    # The numerical witness is `verify_detailed_balance`'s
    # `relative_norm = ‖A_anti‖₂ / ‖D‖₂` (anti-Hermitian part of the
    # discriminant relative to the discriminant). KMS-DBC ⇒ `D = D†`
    # ⇒ `‖A_anti‖ = 0`. We assert this for k ∈ {2, 4} on the n=3
    # disordered Heisenberg fixture across β ∈ {1, 5, 10}.
    # ---------------------------------------------------------------------
    @testset "(g.0) KMS-DBC: ‖A_anti‖ / ‖D‖ ≈ 0 for k ∈ {2, 4}" begin
        for beta in _BETAS
            sys = make_dll_n3_system(beta)
            base = DLLMetropolisFilter(beta; S = 2.0)
            # k = 2, 4 with symmetrised translates inside the bump flat-top.
            for centers in ([0.0, 0.5], [0.0, 0.25, 0.5, 0.75])
                multi = dll_multichannel_translates(base; centers = centers)
                cfg = _make_cfg(BohrDomain(), beta, multi)
                L = construct_lindbladian(sys.jumps, cfg, sys.ham)
                res = QuantumFurnace.verify_detailed_balance(
                    L, sys.ham.gibbs; atol = 1e-10)
                @test res.relative_norm <= 1e-10
                @test res.is_kms_db
                # Sanity: the fixed-point residual ‖D·vec(σ^{1/2})‖ is
                # already covered by (c) above; double-check inline.
                @test res.fixed_point_residual <= 1e-10
            end
        end
    end

    # ---------------------------------------------------------------------
    # (g) validate_config! rejects mismatched β at the multi-channel level.
    # ---------------------------------------------------------------------
    @testset "(g) validate_config! mismatched β" begin
        beta = 5.0
        # Construct a DLLMultiChannelFilter at one β but supply Config.beta
        # at a different β: validate_config! must catch it.
        multi = DLLMultiChannelFilter([DLLMetropolisFilter(beta; S = 2.0)], beta)
        cfg_bad = _make_cfg(BohrDomain(), beta + 1, multi)
        try
            validate_config!(cfg_bad)
            @test false
        catch e
            @test e isa ArgumentError
            @test occursin("DLLMultiChannelFilter", e.msg)
        end
    end


    @testset "(h) DLL rejects an ordinary CKG Gaussian filter" begin
        beta_alg = 5.0
        bad_config = _make_cfg(
            BohrDomain(), beta_alg, GaussianFilter(0.5))
        err = try
            validate_config!(bad_config)
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test occursin("not an admissible DLL filter", sprint(showerror, err))
    end


    @testset "(i) factory-translated filters construct end-to-end in TimeDomain" begin
        beta_alg = 5.0
        sys = make_dll_n3_system(beta_alg)
        filter = dll_multichannel_translates(
            DLLMetropolisFilter(beta_alg; S = 2.0);
            centers = [0.0, 0.4],
            weights = [1.0, 0.7],
        )
        config_bohr = _make_cfg(BohrDomain(), beta_alg, filter)
        # Keep the validated spacing while doubling the time horizon. This is
        # the final point of the convergence check in test_dll_multichannel_time.
        config_time = _make_cfg(
            TimeDomain(), beta_alg, filter;
            num_energy_bits = 11,
            t0 = _T0_CFG,
        )
        @test validate_config!(config_time) === nothing
        L_bohr = construct_lindbladian(sys.jumps, config_bohr, sys.ham)
        L_time = @test_logs (:warn, r"OFT Integration Warning") begin
            construct_lindbladian(sys.jumps, config_time, sys.ham)
        end

        verification = verify_detailed_balance(
            L_time, sys.ham.gibbs; atol = 1e-8)
        @test opnorm(L_time - L_bohr) <= 5e-8
        @test norm(L_time * vec(Matrix(sys.gibbs))) <= 5e-9
        @test verification.relative_norm <= 5e-9
        @test verification.fixed_point_residual <= 5e-9
        @test verification.is_kms_db
    end
end
