@testset "DLL multi-channel BohrDomain operators" begin
    # =====================================================================
    # n=3 disordered Heisenberg fixture (matches existing DLL test files).
    # β-sweep ∈ {1, 5, 10}: same regime as the single-channel DLL tests.
    # =====================================================================
    _MR_BETAS = (1.0, 5.0, 10.0)

    # ---------------------------------------------------------------------
    # (a) k=1 reduces to single-channel dll_lindblad_op_bohr at machine
    #     precision (each metric: byte-equality of the lone matrix).
    # ---------------------------------------------------------------------
    @testset "(a) k=1 Lindblad reduces to single-channel" begin
        for beta in _MR_BETAS
            sys = make_dll_n3_system(beta)
            for ch in (DLLGaussianFilter(beta), DLLMetropolisFilter(beta; S = 2.0))
                multi = DLLMultiChannelFilter([ch], beta)
                for jump in sys.jumps
                    L_single = dll_lindblad_op_bohr(jump, sys.ham, ch)
                    Ls_multi = dll_lindblad_op_bohr(jump, sys.ham, multi)
                    @test length(Ls_multi) == 1
                    @test Ls_multi[1] == L_single  # byte-identity (same single call)
                end
            end
        end
    end

    # ---------------------------------------------------------------------
    # (b) k=2 with two identical channels yields [L, L] (per-channel
    #     decomposition — explicitly distinct from a √2-rescaled L).
    # ---------------------------------------------------------------------
    @testset "(b) k=2 identical-channels Lindblad = [L, L]" begin
        beta = 5.0
        sys = make_dll_n3_system(beta)
        ch = DLLMetropolisFilter(beta; S = 2.0)
        multi = DLLMultiChannelFilter([ch, ch], beta)
        L_ref = dll_lindblad_op_bohr(sys.jumps[1], sys.ham, ch)
        Ls_multi = dll_lindblad_op_bohr(sys.jumps[1], sys.ham, multi)
        @test length(Ls_multi) == 2
        @test Ls_multi[1] == L_ref
        @test Ls_multi[2] == L_ref
    end

    # ---------------------------------------------------------------------
    # (c) Multi-channel α^multi = Σ_ℓ α^(ℓ): summing per-channel rank-1
    #     Kossakowskis (via dll_kossakowski_bohr) reproduces the multi-
    #     channel matrix that `assert_kms_skew_symmetric` accepts.
    # ---------------------------------------------------------------------
    @testset "(c) α^multi = sum of per-channel α^(ℓ) is KMS skew-symmetric" begin
        # Use two distinct DLL filters (different shapes) on the same Bohr
        # grid; each per-channel α is KMS-skew-symmetric, so the sum is too.
        for beta in _MR_BETAS
            sys = make_dll_n3_system(beta)
            ch_g = DLLGaussianFilter(beta)
            ch_m = DLLMetropolisFilter(beta; S = 2.0)
            # Symmetric Bohr-frequency grid (so ν_grid is closed under negation).
            νmax = 0.4
            ν_grid = collect(range(-νmax, νmax; length = 9))
            α_g = QuantumFurnace.dll_kossakowski_bohr(ch_g, ν_grid)
            α_m = QuantumFurnace.dll_kossakowski_bohr(ch_m, ν_grid)
            α_multi = α_g .+ α_m
            assert_kms_skew_symmetric(α_multi, ν_grid, beta; atol = 1e-12)
        end
    end

    # ---------------------------------------------------------------------
    # (d) k=1 coherent operator equals single-channel (byte-identity at
    #     all (n=3) entries).
    # ---------------------------------------------------------------------
    @testset "(d) k=1 coherent G reduces to single-channel" begin
        for beta in _MR_BETAS
            sys = make_dll_n3_system(beta)
            for ch in (DLLGaussianFilter(beta), DLLMetropolisFilter(beta; S = 2.0))
                multi = DLLMultiChannelFilter([ch], beta)
                G_single = dll_coherent_op_bohr(sys.jumps, sys.ham, ch, beta)
                G_multi  = dll_coherent_op_bohr(sys.jumps, sys.ham, multi, beta)
                @test G_multi == G_single
            end
        end
    end

    # ---------------------------------------------------------------------
    # (e) k=2 identical channels: G^multi = 2·G^single (additivity).
    # ---------------------------------------------------------------------
    @testset "(e) k=2 identical-channels G = 2·G^single" begin
        beta = 5.0
        sys = make_dll_n3_system(beta)
        ch = DLLGaussianFilter(beta)
        multi = DLLMultiChannelFilter([ch, ch], beta)
        G_single = dll_coherent_op_bohr(sys.jumps, sys.ham, ch, beta)
        G_multi  = dll_coherent_op_bohr(sys.jumps, sys.ham, multi, beta)
        @test isapprox(G_multi, 2 .* G_single; atol = 1e-13, rtol = 1e-13)
    end

    # ---------------------------------------------------------------------
    # (f) k=2 distinct channels (Gaussian + Metropolis): G^multi must
    #     equal the sum of the two single-channel coherent operators.
    # ---------------------------------------------------------------------
    @testset "(f) k=2 distinct channels: G^multi = Σ G^(ℓ)" begin
        beta = 5.0
        sys = make_dll_n3_system(beta)
        ch_g = DLLGaussianFilter(beta)
        ch_m = DLLMetropolisFilter(beta; S = 2.0)
        # Heterogeneous channel vector: store as Vector{AbstractFilter}.
        multi = DLLMultiChannelFilter{Float64, AbstractFilter}(
            AbstractFilter[ch_g, ch_m], beta)
        G_g = dll_coherent_op_bohr(sys.jumps, sys.ham, ch_g, beta)
        G_m = dll_coherent_op_bohr(sys.jumps, sys.ham, ch_m, beta)
        G_multi = dll_coherent_op_bohr(sys.jumps, sys.ham, multi, beta)
        @test isapprox(G_multi, G_g .+ G_m; atol = 1e-13, rtol = 1e-13)
    end

    # ---------------------------------------------------------------------
    # (g) Multi-channel coherent G is Hermitian (Theorem 10 holds per
    #     channel, so the sum is Hermitian).
    # ---------------------------------------------------------------------
    @testset "(g) G^multi Hermitian" begin
        for beta in _MR_BETAS
            sys = make_dll_n3_system(beta)
            ch_g = DLLGaussianFilter(beta)
            ch_m = DLLMetropolisFilter(beta; S = 2.0)
            multi = DLLMultiChannelFilter{Float64, AbstractFilter}(
                AbstractFilter[ch_g, ch_m], beta)
            G_multi = dll_coherent_op_bohr(sys.jumps, sys.ham, multi, beta)
            @test norm(G_multi - G_multi') / max(norm(G_multi), 1e-30) <= 1e-12
        end
    end
end
