@testset "DLL multi-channel TimeDomain operators" begin
    # =====================================================================
    # n=3 disordered Heisenberg fixture with the same trapezoidal grid as
    # `test_dll_coherent.jl`. β-sweep ∈ {1, 5, 10}.
    # =====================================================================
    _MR_BETAS = (1.0, 5.0, 10.0)
    # N=10 (Nt=1024): same t_max ≈ 63 as legacy N=12, 16× less NUFFT memory.
    # Bohr↔Time error already at FINUFFT floor (~3e-9) by Nt=256.
    _NUM_ENERGY_BITS = 10
    _T0 = 2π / (2^_NUM_ENERGY_BITS * 0.05)

    # Mirror of `test_dll_coherent.jl::_dll_time_labels`. Builds the
    # uniform trapezoidal grid then truncates by the filter's
    # `filter_time_cutoff`.
    function _time_labels(beta, filter)
        N = 2^_NUM_ENERGY_BITS
        labels = collect((-N÷2):(N÷2 - 1)) .* _T0
        sigma = 1.0 / beta
        return QuantumFurnace._truncate_time_labels_for_oft(labels, sigma; filter=filter)
    end

    # ---------------------------------------------------------------------
    # (a) k=1 Lindblad time-domain reduces to single-channel byte-identity.
    # ---------------------------------------------------------------------
    @testset "(a) k=1 Lindblad reduces to single-channel" begin
        for beta in _MR_BETAS
            sys = make_dll_n3_system(beta)
            for ch in (DLLGaussianFilter(beta), DLLMetropolisFilter(beta; S = 2.0))
                multi = DLLMultiChannelFilter([ch], beta)
                tl = _time_labels(beta, ch)
                tl_multi = _time_labels(beta, multi)
                # cutoffs derive from the same single channel ⇒ same grid.
                @test tl == tl_multi

                L_single = dll_lindblad_op_time(sys.jumps[1], sys.ham, tl, ch, _T0)
                Ls_multi = dll_lindblad_op_time(sys.jumps[1], sys.ham, tl_multi, multi, _T0)
                @test length(Ls_multi) == 1
                @test Ls_multi[1] == L_single
            end
        end
    end

    # ---------------------------------------------------------------------
    # (b) k=2 identical channels: time-domain Lindblad = [L, L].
    # ---------------------------------------------------------------------
    @testset "(b) k=2 identical-channels Lindblad time = [L, L]" begin
        beta = 5.0
        sys = make_dll_n3_system(beta)
        ch = DLLMetropolisFilter(beta; S = 2.0)
        multi = DLLMultiChannelFilter([ch, ch], beta)
        tl = _time_labels(beta, multi)
        L_ref = dll_lindblad_op_time(sys.jumps[1], sys.ham, tl, ch, _T0)
        Ls_multi = dll_lindblad_op_time(sys.jumps[1], sys.ham, tl, multi, _T0)
        @test length(Ls_multi) == 2
        @test Ls_multi[1] == L_ref
        @test Ls_multi[2] == L_ref
    end

    # ---------------------------------------------------------------------
    # (c) Time-domain Bohr ↔ Time agreement: per-channel L^(ℓ) matches the
    #     Bohr-domain reference at single-channel quadrature precision.
    # ---------------------------------------------------------------------
    @testset "(c) Multi-channel Lindblad time ↔ Bohr agreement" begin
        beta = 5.0
        sys = make_dll_n3_system(beta)
        # Two distinct DLLMetropolis channels (different S keep both within
        # the bump support but with different shapes).
        ch1 = DLLMetropolisFilter(beta; S = 2.0)
        ch2 = DLLMetropolisFilter(beta; S = 3.0)
        multi = DLLMultiChannelFilter([ch1, ch2], beta)
        tl = _time_labels(beta, multi)
        Ls_t = dll_lindblad_op_time(sys.jumps[1], sys.ham, tl, multi, _T0)
        Ls_b = dll_lindblad_op_bohr(sys.jumps[1], sys.ham, multi)
        @test length(Ls_t) == length(Ls_b) == 2
        for ℓ in 1:2
            @test opnorm(Ls_t[ℓ] - Ls_b[ℓ]) <= 1e-5
        end
    end

    # ---------------------------------------------------------------------
    # (d) k=1 coherent time-domain G reduces to single-channel byte-identity.
    # ---------------------------------------------------------------------
    @testset "(d) k=1 coherent G time reduces to single-channel" begin
        for beta in _MR_BETAS
            sys = make_dll_n3_system(beta)
            for ch in (DLLGaussianFilter(beta), DLLMetropolisFilter(beta; S = 2.0))
                multi = DLLMultiChannelFilter([ch], beta)
                tl = _time_labels(beta, ch)
                G_single = dll_coherent_op_time(sys.jumps, sys.ham, tl, ch, beta, _T0)
                G_multi  = dll_coherent_op_time(sys.jumps, sys.ham, tl, multi, beta, _T0)
                @test G_multi == G_single
            end
        end
    end

    # ---------------------------------------------------------------------
    # (e) k=2 identical-channel time-domain coherent G = 2·G^single.
    # ---------------------------------------------------------------------
    @testset "(e) k=2 identical-channels G time = 2·G^single" begin
        beta = 5.0
        sys = make_dll_n3_system(beta)
        ch = DLLGaussianFilter(beta)
        multi = DLLMultiChannelFilter([ch, ch], beta)
        tl = _time_labels(beta, multi)
        G_single = dll_coherent_op_time(sys.jumps, sys.ham, tl, ch, beta, _T0)
        G_multi  = dll_coherent_op_time(sys.jumps, sys.ham, tl, multi, beta, _T0)
        @test isapprox(G_multi, 2 .* G_single; atol = 1e-12, rtol = 1e-12)
    end

    # ---------------------------------------------------------------------
    # (f) Time ↔ Bohr coherent agreement on a multi-channel filter.
    #     Per-channel quadrature precision; sum holds at the same tol.
    # ---------------------------------------------------------------------
    @testset "(f) Multi-channel coherent G: time ↔ Bohr" begin
        beta = 5.0
        sys = make_dll_n3_system(beta)
        ch1 = DLLMetropolisFilter(beta; S = 2.0)
        ch2 = DLLMetropolisFilter(beta; S = 3.0)
        multi = DLLMultiChannelFilter([ch1, ch2], beta)
        tl = _time_labels(beta, multi)
        G_t = dll_coherent_op_time(sys.jumps, sys.ham, tl, multi, beta, _T0)
        G_b = dll_coherent_op_bohr(sys.jumps, sys.ham, multi, beta)
        @test opnorm(G_t - G_b) <= 1e-5
    end


    @testset "(g) shifted coherent TimeDomain path is controlled" begin
        beta_alg = 5.0
        sys = make_dll_n3_system(beta_alg)

        gaussian = ShiftedSymmetricFilter(DLLGaussianFilter(beta_alg), 0.4, 1.0)
        gaussian_labels = collect((-512):(511)) .* _T0
        G_gaussian_bohr = dll_coherent_op_bohr(
            sys.jumps, sys.ham, gaussian, beta_alg)
        G_gaussian_time = dll_coherent_op_time(
            sys.jumps, sys.ham, gaussian_labels, gaussian, beta_alg, _T0;
            nu_grid_size = 128)
        @test opnorm(G_gaussian_time - G_gaussian_bohr) <= 1e-10

        metropolis = ShiftedSymmetricFilter(
            DLLMetropolisFilter(beta_alg; S = 2.0), 0.4, 1.0)
        G_metropolis_bohr = dll_coherent_op_bohr(
            sys.jumps, sys.ham, metropolis, beta_alg)
        errors = Float64[]
        for num_times in (512, 1024, 2048)
            time_labels = collect((-num_times÷2):(num_times÷2 - 1)) .* _T0
            G_metropolis_time = dll_coherent_op_time(
                sys.jumps, sys.ham, time_labels, metropolis, beta_alg, _T0;
                nu_grid_size = 256)
            push!(errors, opnorm(G_metropolis_time - G_metropolis_bohr))
        end
        @test all(diff(errors) .< 0)
        @test last(errors) <= 1e-9
    end
end
