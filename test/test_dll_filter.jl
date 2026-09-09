using QuadGK

@testset "DLL filter" begin

    # -----------------------------------------------------------------------
    # (a) Gaussian time kernel matches the hardcoded form used in production
    # -----------------------------------------------------------------------
    @testset "(a) GaussianFilter time_kernel == exp(-σ² t²)" begin
        for sigma in (0.5, 1.5)
            f = GaussianFilter(sigma)
            for t in (-2.0, -0.5, 0.0, 0.5, 2.0)
                @test isapprox(time_kernel(f, t), exp(-sigma^2 * t^2); atol=1e-15)
            end
            @test time_kernel(f, 0.0) == 1.0
        end
    end

    # -----------------------------------------------------------------------
    # (b) DLL time kernel matches the closed form (Eq. 3.3)
    # -----------------------------------------------------------------------
    @testset "(b) DLLGaussianFilter time_kernel closed form" begin
        for beta in (0.5, 2.0)
            f = DLLGaussianFilter(beta)
            pref = exp(1/8) * sqrt(2/pi) / beta
            for t in (-2.0, -0.5, 0.0, 0.5, 2.0)
                expected = pref * exp(-2 * t^2 / beta^2) * cis(t / beta)
                @test isapprox(time_kernel(f, t), expected; atol=1e-13)
            end
            # f(0) is real and equals the prefactor
            @test isapprox(real(time_kernel(f, 0.0)), pref; atol=1e-15)
            @test abs(imag(time_kernel(f, 0.0))) < 1e-15
        end
    end

    # -----------------------------------------------------------------------
    # (c) Round-trip: numerical FT of time_kernel matches freq_kernel.
    # For Gaussian, the bare FT is √π/σ · freq_kernel (the unnormalised form).
    # For DLL, freq_kernel already encodes the full f̂(ν).
    # -----------------------------------------------------------------------
    @testset "(c) Fourier round-trip via QuadGK" begin
        # Gaussian
        sigma = 0.7
        fG = GaussianFilter(sigma)
        gauss_norm = sqrt(pi) / sigma
        t_max_G = 12.0 / sigma
        for ν in (-2.5, -0.5, 0.0, 1.3)
            integrand = t -> time_kernel(fG, t) * cis(ν * t)
            num, _ = quadgk(integrand, -t_max_G, t_max_G; atol=1e-12, rtol=0)
            ref = gauss_norm * freq_kernel(fG, ν)
            @test isapprox(num, ref; atol=1e-8)
            @test abs(imag(num)) < 1e-8  # real f → real f̂
        end

        # DLL
        beta = 1.5
        fD = DLLGaussianFilter(beta)
        t_max_D = 12.0 * beta
        for ν in (-2.5, -1/beta, 0.0, 1.3)
            integrand = t -> time_kernel(fD, t) * cis(ν * t)
            num, _ = quadgk(integrand, -t_max_D, t_max_D; atol=1e-12, rtol=0)
            ref = freq_kernel(fD, ν)  # DLL freq_kernel is the full f̂
            @test isapprox(num, ref; atol=1e-8)
            @test abs(imag(num)) < 1e-8  # f̂ is real here (Eq. 3.22 is real-valued for real ν)
        end
    end

    # -----------------------------------------------------------------------
    # (d) Symmetry: Gaussian f̂(ν)=f̂(-ν); DLL f̂(-1/β + δ)=f̂(-1/β - δ)
    # -----------------------------------------------------------------------
    @testset "(d) Symmetry of frequency kernels" begin
        sigma = 0.7
        fG = GaussianFilter(sigma)
        for ν in (0.3, 1.7, 4.2)
            @test isapprox(freq_kernel(fG, ν), freq_kernel(fG, -ν); atol=1e-15)
        end

        beta = 2.0
        fD = DLLGaussianFilter(beta)
        nu_star = -1 / beta  # peak of DLL freq_kernel
        for δ in (0.1, 0.7, 2.3)
            @test isapprox(
                freq_kernel(fD, nu_star + δ),
                freq_kernel(fD, nu_star - δ);
                atol=1e-13,
            )
        end
        # DLL peak value is e^{1/8}
        @test isapprox(freq_kernel(fD, nu_star), exp(1/8); atol=1e-13)
    end

    # -----------------------------------------------------------------------
    # (e) filter_time_cutoff: the residual `prefactor · |kernel(tc)|` must be
    # ≤ tol — that is the quantity the original CKG cutoff formula bounds (see
    # `_truncate_time_labels_for_oft`). For Gaussian, `time_kernel` is the bare
    # `exp(-σ² t²)` so the residual is `prefactor_gauss * |kernel|`. For DLL,
    # `time_kernel` already includes the prefactor so the residual is just
    # `|kernel|`. Allow a small numerical slack because `log → √` propagation
    # can lose a few ULPs.
    # -----------------------------------------------------------------------
    @testset "(e) filter_time_cutoff decay" begin
        tol = 1e-12
        slack = 4.0  # absorbs ULP loss in log/sqrt and integer-trip near-equalities

        for sigma in (0.5, 1.5)
            f = GaussianFilter(sigma)
            tc = filter_time_cutoff(f, tol)
            prefactor = QuantumFurnace._time_oft_prefactor_gaussian(f)
            @test prefactor * abs(time_kernel(f, tc)) <= slack * tol
        end

        for beta in (0.5, 2.0)
            f = DLLGaussianFilter(beta)
            tc = filter_time_cutoff(f, tol)
            # DLL kernel already includes the prefactor.
            @test abs(time_kernel(f, tc)) <= slack * tol
        end
    end

    # -----------------------------------------------------------------------
    # (f) NUFFT byte-identity: filter=nothing must produce the same NUFFT
    #     prefactor array as filter=GaussianFilter(σ). This is the load-bearing
    #     CKG-preservation guarantee.
    # -----------------------------------------------------------------------
    @testset "(f) NUFFT byte-identity (CKG default)" begin
        config_default = make_config(Lindbladian(), TimeDomain())
        config_explicit = Config(;
            sim = Lindbladian(),
            domain = TimeDomain(),
            construction = KMS(),
            num_qubits = NUM_QUBITS,
            with_linear_combination = true,
            beta = BETA,
            sigma = SIGMA,
            a = BETA / 30.0,
            s = 0.4,
            num_energy_bits = NUM_ENERGY_BITS,
            w0 = W0,
            t0 = T0,
            num_trotter_steps_per_t0 = NUM_TROTTER_STEPS_PER_T0,
            filter = GaussianFilter(SIGMA),
        )

        pd_default = QuantumFurnace._precompute_data(config_default, TEST_HAM)
        pd_explicit = QuantumFurnace._precompute_data(config_explicit, TEST_HAM)

        # Exact equality, not isapprox: same inputs, same FINUFFT plan (eps=1e-12).
        @test pd_default.oft_nufft_prefactors.data == pd_explicit.oft_nufft_prefactors.data
        @test pd_default.oft_nufft_prefactors.energy_labels == pd_explicit.oft_nufft_prefactors.energy_labels
    end

    # -----------------------------------------------------------------------
    # (f2) Filter families are construction-specific. Mixing a DLL kernel into
    #      KMS/GNS, or changing the CKG Gaussian width independently of sigma,
    #      breaks detailed balance rather than merely changing accuracy.
    # -----------------------------------------------------------------------
    @testset "(f2) construction-specific filter validation" begin
        common = (;
            sim = Lindbladian(), domain = BohrDomain(), num_qubits = NUM_QUBITS,
            with_linear_combination = true, beta = BETA, sigma = SIGMA,
            a = BETA / 30.0, s = 0.4,
        )

        kms_dll = Config(; common..., construction = KMS(),
                         filter = DLLGaussianFilter(BETA))
        gns_dll = Config(; common..., construction = GNS(),
                         filter = DLLMetropolisFilter(BETA))
        kms_wrong_width = Config(; common..., construction = KMS(),
                                 filter = GaussianFilter(2SIGMA))
        dll_ckg = Config(; common..., construction = DLL(),
                         filter = GaussianFilter(SIGMA))

        @test_throws ArgumentError validate_config!(kms_dll)
        @test_throws ArgumentError validate_config!(gns_dll)
        @test_throws ArgumentError validate_config!(kms_wrong_width)
        @test_throws ArgumentError validate_config!(dll_ckg)
        @test validate_config!(Config(; common..., construction = KMS(),
                                     filter = GaussianFilter(SIGMA))) === nothing
        @test validate_config!(Config(; common..., construction = GNS(),
                                     filter = GaussianFilter(SIGMA))) === nothing
        @test validate_config!(Config(; common..., construction = KMS(),
                                     filter = GaussianFilter(nextfloat(SIGMA)))) === nothing

        for bad in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError GaussianFilter(bad)
            @test_throws ArgumentError DLLGaussianFilter(bad)
            @test_throws ArgumentError DLLMetropolisFilter(bad)
        end
        @test_throws ArgumentError DLLMetropolisFilter(BETA; S = 0.0)
        @test_throws ArgumentError DLLMetropolisFilter(BETA; S = -1.0)
        @test_throws ArgumentError DLLMetropolisFilter(BETA; S = Inf)
        @test_throws ArgumentError DLLMetropolisFilter(BETA; S = NaN)
    end

    @testset "(f3) exported DLL helpers reject incompatible filters" begin
        ckg_filter = GaussianFilter(SIGMA)
        dll_filter = DLLGaussianFilter(BETA)
        jump = TEST_JUMPS[1]
        times = [-0.1, 0.0, 0.1]
        bohr_grid = [-0.2, 0.0, 0.2]

        @test_throws ArgumentError dll_lindblad_op_bohr(jump, TEST_HAM, ckg_filter)
        @test_throws ArgumentError dll_lindblad_op_time(
            jump, TEST_HAM, times, ckg_filter, 0.1)
        @test_throws ArgumentError dll_coherent_op_bohr(
            TEST_JUMPS, TEST_HAM, ckg_filter, BETA)
        @test_throws ArgumentError dll_kossakowski_bohr(ckg_filter, bohr_grid)
        @test_throws ArgumentError dll_coherent_op_bohr(
            TEST_JUMPS, TEST_HAM, dll_filter, 2BETA)
        @test_throws ArgumentError dll_coherent_op_time(
            TEST_JUMPS, TEST_HAM, times, dll_filter, 2BETA, 0.1)

        shifted = ShiftedSymmetricFilter(dll_filter, 0.1, 1.0)
        multi = DLLMultiChannelFilter([shifted], BETA)
        @test_throws ArgumentError dll_coherent_op_bohr(
            TEST_JUMPS, TEST_HAM, shifted, 2BETA)
        @test_throws ArgumentError dll_coherent_op_bohr(
            TEST_JUMPS, TEST_HAM, multi, 2BETA)
    end

    # -----------------------------------------------------------------------
    # (g) DLL NUFFT prefactor spot check: the (i,j,k) entry equals the direct
    #     quadrature ∫ time_kernel(filter, t) cis(-ω·t) cis(bohr_freq[i,j]·t) dt.
    #     We pick BETA as both the DLL filter parameter and the Config beta so
    #     validate_config! is happy.
    # -----------------------------------------------------------------------
    @testset "(g) DLL NUFFT prefactor spot check" begin
        # Fresh config with DLL filter; matching beta required by validate_config!.
        config_dll = Config(;
            sim = Lindbladian(),
            domain = TimeDomain(),
            construction = DLL(),
            num_qubits = NUM_QUBITS,
            with_linear_combination = true,
            beta = BETA,
            sigma = SIGMA,
            a = BETA / 30.0,
            s = 0.4,
            num_energy_bits = NUM_ENERGY_BITS,
            w0 = W0,
            t0 = T0,
            num_trotter_steps_per_t0 = NUM_TROTTER_STEPS_PER_T0,
            filter = DLLGaussianFilter(BETA),
        )

        # Sanity: validate_config! must accept this configuration.
        @test_nowarn validate_config!(config_dll)

        # Build a quadrature ground truth for one (i,j) Bohr index and one ω.
        # The NUFFT computes:
        #     P[i,j,k] = ∑_t time_kernel(filter, t) * cis(-ω_k * t) * cis(bohr[i,j] * t)
        # Discrete Riemann sum. Compare to direct sum over the SAME truncated time grid.
        energy_labels = QuantumFurnace._create_energy_labels(NUM_ENERGY_BITS, W0)
        time_labels_full = energy_labels .* (T0 / W0)
        oft_time_labels = QuantumFurnace._truncate_time_labels_for_oft(
            time_labels_full, SIGMA; filter=DLLGaussianFilter(BETA),
        )

        f = DLLGaussianFilter(BETA)
        i, j = 2, 5
        bohr_ij = TEST_HAM.bohr_freqs[i, j]
        ω = energy_labels[length(energy_labels) ÷ 2 + 4]
        prefs = QuantumFurnace._prepare_oft_nufft_prefactors(
            TEST_HAM.bohr_freqs, oft_time_labels, [ω], f; eps=1e-12)

        ref = ComplexF64(0)
        for t in oft_time_labels
            ref += time_kernel(f, t) * cis(-ω * t) * cis(bohr_ij * t)
        end

        nufft_val = ComplexF64(prefs.data[i, j, 1])
        @test isapprox(nufft_val, ref; atol=1e-8, rtol=1e-8)
    end

    # -----------------------------------------------------------------------
    # (h) validate_config! mismatch handling for DLLGaussianFilter
    # -----------------------------------------------------------------------
    @testset "(h) validate_config! DLL beta mismatch" begin
        # Mismatched beta → should throw.
        config_bad = Config(;
            sim = Lindbladian(),
            domain = TimeDomain(),
            construction = DLL(),
            num_qubits = NUM_QUBITS,
            with_linear_combination = true,
            beta = 1.0,
            sigma = SIGMA,
            a = 1.0 / 30.0,
            s = 0.4,
            num_energy_bits = NUM_ENERGY_BITS,
            w0 = W0,
            t0 = T0,
            num_trotter_steps_per_t0 = NUM_TROTTER_STEPS_PER_T0,
            filter = DLLGaussianFilter(2.0),
        )
        try
            validate_config!(config_bad)
            @test false  # should have thrown
        catch e
            @test e isa ArgumentError
            @test occursin("DLLGaussianFilter.beta must match Config.beta", e.msg)
        end

        # Matching beta → no throw.
        config_good = Config(;
            sim = Lindbladian(),
            domain = TimeDomain(),
            construction = DLL(),
            num_qubits = NUM_QUBITS,
            with_linear_combination = true,
            beta = 2.0,
            sigma = SIGMA,
            a = 2.0 / 30.0,
            s = 0.4,
            num_energy_bits = NUM_ENERGY_BITS,
            w0 = W0,
            t0 = T0,
            num_trotter_steps_per_t0 = NUM_TROTTER_STEPS_PER_T0,
            filter = DLLGaussianFilter(2.0),
        )
        @test_nowarn validate_config!(config_good)
    end

    # -----------------------------------------------------------------------
    # (h2) validate_config! mismatch handling for DLLMetropolisFilter
    # -----------------------------------------------------------------------
    @testset "(h2) validate_config! Metropolis beta + S checks" begin
        # Mismatched beta → should throw.
        config_bad_beta = Config(;
            sim = Lindbladian(),
            domain = TimeDomain(),
            construction = DLL(),
            num_qubits = NUM_QUBITS,
            with_linear_combination = true,
            beta = 1.0,
            sigma = SIGMA,
            a = 1.0 / 30.0,
            s = 0.4,
            num_energy_bits = NUM_ENERGY_BITS,
            w0 = W0,
            t0 = T0,
            num_trotter_steps_per_t0 = NUM_TROTTER_STEPS_PER_T0,
            filter = DLLMetropolisFilter(5.0; S = 2.0),
        )
        try
            validate_config!(config_bad_beta)
            @test false  # should have thrown
        catch e
            @test e isa ArgumentError
            @test occursin("DLLMetropolisFilter.beta must match Config.beta", e.msg)
        end

        # Matching beta + positive S → no throw.
        config_good = Config(;
            sim = Lindbladian(),
            domain = TimeDomain(),
            construction = DLL(),
            num_qubits = NUM_QUBITS,
            with_linear_combination = true,
            beta = 2.0,
            sigma = SIGMA,
            a = 2.0 / 30.0,
            s = 0.4,
            num_energy_bits = NUM_ENERGY_BITS,
            w0 = W0,
            t0 = T0,
            num_trotter_steps_per_t0 = NUM_TROTTER_STEPS_PER_T0,
            filter = DLLMetropolisFilter(2.0; S = 2.0),
        )
        @test_nowarn validate_config!(config_good)
    end

    # -----------------------------------------------------------------------
    # (g) Hörmander bump — Ding–Li–Lin 2024 Assumption 15 / Eq. 3.19
    # Properties: w ≡ 1 on |x| ≤ 1/2, w = 0 on |x| ≥ 1, even, smooth on [0,1].
    # -----------------------------------------------------------------------
    @testset "(g) Hörmander bump w(x) sanity" begin
        w = QuantumFurnace._hormander_bump

        # Flat top: w ≡ 1 on |x| ≤ 1/2.
        for x in (-0.5, -0.25, 0.0, 0.25, 0.5)
            @test w(x) == 1.0
        end

        # Outside support: w ≡ 0 on |x| ≥ 1.
        for x in (-2.0, -1.0, 1.0, 1.5)
            @test w(x) == 0.0
        end

        # Evenness on the transition region.
        for x in (0.55, 0.65, 0.75, 0.85, 0.95)
            @test isapprox(w(x), w(-x); atol=1e-15)
        end

        # Monotone decay on the transition region (|x| ∈ (1/2, 1)).
        # Start at 0.55 (not 0.51): for x ≲ 0.53, η(1 - 2(1-|x|)) underflows
        # below the precision of the (a + b)-denominator and φ rounds to 1.0
        # numerically, even though analytically φ < 1. The smooth-decay
        # property is still cleanly visible from x = 0.55 onward.
        xs_trans = collect(0.55:0.05:0.95)
        ws = w.(xs_trans)
        @test all(diff(ws) .< 0)
        @test all(0.0 .< ws .< 1.0)

        # Continuity at the boundaries (matched by limits, not just the special
        # cases): the underflow path must give 0/1 cleanly.
        @test isapprox(w(0.5 + 1e-12), 1.0; atol=1e-9)
        @test isapprox(w(1.0 - 1e-12), 0.0; atol=1e-9)

        # Type stability: returns the same float type as the input.
        @test typeof(w(0.7)) === Float64
        @test typeof(w(0.7f0)) === Float32

        # Integer input promotes to Float64 (does not error).
        @test w(0) === 1.0
        @test w(2) === 0.0
    end

    # -----------------------------------------------------------------------
    # (h) DLLMetropolisFilter struct + q_weight + freq_kernel
    # Ding–Li–Lin 2024 Eq. 3.19–3.20. The bump w(ν/S) enforces compact
    # support; on the flat top |ν| ≤ S/2 we recover the smoothed
    # Metropolis acceptance min{1, exp(-βν/2)}.
    # -----------------------------------------------------------------------
    @testset "(h) DLLMetropolisFilter freq_kernel + q_weight" begin
        # ----- (h.1) Constructor: default S, custom S, type stability -----
        @testset "constructor" begin
            f = DLLMetropolisFilter(5.0)
            @test f.beta == 5.0
            @test f.S == 2.0
            @test eltype(f) === ComplexF64

            f2 = DLLMetropolisFilter(2.0; S = 5.0)
            @test f2.beta == 2.0
            @test f2.S == 5.0

            # Float32 path
            f32 = DLLMetropolisFilter(2.0f0)
            @test f32.beta === 2.0f0
            @test f32.S === 2.0f0
            @test eltype(f32) === ComplexF32

            # S keyword promotes to T
            f3 = DLLMetropolisFilter(1.0; S = 3)
            @test f3.S === 3.0
            @test typeof(f3) === DLLMetropolisFilter{Float64}
        end

        # ----- (h.2) q_weight matches Eq. 3.19 closed form -----
        @testset "q_weight = exp(-√(1+(βν)²)/4) · w(ν/S)" begin
            for (beta, S) in ((5.0, 2.0), (1.0, 3.0), (10.0, 1.5))
                f = DLLMetropolisFilter(beta; S = S)
                w = QuantumFurnace._hormander_bump
                for nu in (-1.5, -0.5, 0.0, 0.4, 1.2, 5.0)
                    expected = exp(-sqrt(1 + (beta * nu)^2) / 4) * w(nu / S)
                    @test isapprox(QuantumFurnace.q_weight(f, nu), expected;
                                   atol=1e-15)
                end
            end
        end

        # ----- (h.3) freq_kernel = q · exp(-βν/4) (Eq. 3.20 RHS) -----
        @testset "freq_kernel = q · e^{-βν/4}" begin
            for (beta, S) in ((5.0, 2.0), (1.0, 3.0), (10.0, 1.5))
                f = DLLMetropolisFilter(beta; S = S)
                w = QuantumFurnace._hormander_bump
                for nu in (-1.5, -0.5, 0.0, 0.4, 1.2, 5.0)
                    q = exp(-sqrt(1 + (beta * nu)^2) / 4) * w(nu / S)
                    expected = q * exp(-beta * nu / 4)
                    @test isapprox(freq_kernel(f, nu), expected; atol=1e-15)
                end
            end
        end

        # ----- (h.4) Compact support: f̂(ν) = 0 strictly outside [-S, S] -----
        @testset "compact support outside [-S, S]" begin
            for (beta, S) in ((1.0, 2.0), (5.0, 1.5), (10.0, 3.0))
                f = DLLMetropolisFilter(beta; S = S)
                for nu in (-3 * S, -1.5 * S, -1.001 * S, -S, S, 1.001 * S,
                          1.5 * S, 3 * S)
                    @test freq_kernel(f, nu) == 0.0
                    @test QuantumFurnace.q_weight(f, nu) == 0.0
                end
            end
        end

        # ----- (h.5) On the flat top |ν| ≤ S/2: bump=1, q is bare u(βν) -----
        @testset "bump invisible on flat top |ν| ≤ S/2" begin
            for (beta, S) in ((5.0, 2.0), (1.0, 3.0), (10.0, 1.5))
                f = DLLMetropolisFilter(beta; S = S)
                # span the flat-top region
                nu_grid = range(-S/2 + 1e-9, S/2 - 1e-9; length = 7)
                for nu in nu_grid
                    expected_q = exp(-sqrt(1 + (beta * nu)^2) / 4)
                    @test isapprox(QuantumFurnace.q_weight(f, nu), expected_q;
                                   atol=1e-15)
                end
            end
        end

        # ----- (h.6) Symmetry: q(ν) = q(-ν) (Eq. 3.11 with q real) -----
        @testset "q symmetry q(ν) = q(-ν)" begin
            for (beta, S) in ((5.0, 2.0), (1.0, 3.0), (10.0, 1.5))
                f = DLLMetropolisFilter(beta; S = S)
                for nu in (0.1, 0.7, 1.3, 1.9, 2.5)
                    @test isapprox(QuantumFurnace.q_weight(f, nu),
                                   QuantumFurnace.q_weight(f, -nu);
                                   atol=1e-15)
                end
            end
        end

        # ----- (h.7) KMS detailed-balance: f̂(ν) = f̂(-ν) · e^{-βν/2} -----
        # f̂ = q · e^{-βν/4} with q even ⇒ f̂(ν)/f̂(-ν) = e^{-βν/2}.
        # Hold within the flat top, where the bump factor cancels.
        @testset "KMS DB ratio f̂(ν)/f̂(-ν) = e^{-βν/2}" begin
            for (beta, S) in ((5.0, 2.0), (1.0, 3.0), (10.0, 1.5))
                f = DLLMetropolisFilter(beta; S = S)
                for nu in (0.1, 0.3, 0.5, 0.7, 0.9) .* (S/2 - 1e-9)
                    ratio = freq_kernel(f, nu) / freq_kernel(f, -nu)
                    @test isapprox(ratio, exp(-beta * nu / 2); atol=1e-13)
                end
            end
        end

        # ----- (h.8) Metropolis asymptote on flat top, |βν| ≫ 1 -----
        # f̂(ν) = exp(-(√(1+(βν)²) + βν)/4) on |ν| ≤ S/2; the asymptote
        # min{1, e^{-βν/2}} (Eq. 3.20) holds with correction O(1/|βν|), so
        # we restrict to |βν| ≥ 20 (β=40, |ν| ≥ 0.5) for clean convergence.
        # ν=0 is the smoothed corner with f̂(0) = e^{-1/4}.
        @testset "Metropolis asymptote (flat top, |βν| ≫ 1)" begin
            f = DLLMetropolisFilter(40.0; S = 2.0)
            # Down-jump (ν < 0): f̂ → 1, correction ≈ 1/(8|βν|)
            for nu in (-1.0, -0.75, -0.5)
                @test isapprox(freq_kernel(f, nu), 1.0; atol=2e-2)
            end
            # Up-jump (ν > 0): f̂ → e^{-βν/2}, relative correction ≈ 1/(8βν)
            for nu in (0.5, 0.75, 1.0)
                @test isapprox(freq_kernel(f, nu), exp(-40.0 * nu / 2);
                               rtol=2e-2)
            end
            # ν = 0: smoothed corner — exact f̂(0) = e^{-1/4} (no asymptote)
            @test isapprox(freq_kernel(f, 0.0), exp(-0.25); atol=1e-15)
        end

        # ----- (h.9) Smoothness: f̂ continuous at the bump boundaries -----
        # At |ν| = S/2 (flat-top edge), q must equal u(βν) (continuity from
        # the inside) and at |ν| = S, q must equal 0 (continuity from outside).
        @testset "boundary continuity" begin
            for (beta, S) in ((5.0, 2.0), (1.0, 3.0), (10.0, 1.5))
                f = DLLMetropolisFilter(beta; S = S)
                # Flat-top boundary
                @test isapprox(QuantumFurnace.q_weight(f, S/2),
                               exp(-sqrt(1 + (beta * S/2)^2) / 4);
                               atol=1e-15)
                @test isapprox(QuantumFurnace.q_weight(f, -S/2),
                               exp(-sqrt(1 + (beta * S/2)^2) / 4);
                               atol=1e-15)
                # Outer support boundary
                @test QuantumFurnace.q_weight(f, S) == 0.0
                @test QuantumFurnace.q_weight(f, -S) == 0.0
            end
        end
    end

    # -----------------------------------------------------------------------
    # (i) DLLMetropolisFilter time_kernel via QuadGK + filter_time_cutoff
    # The time kernel is the inverse Fourier transform of f̂(ν) over the
    # compact support [-S, S]. Verifies (1) Fourier round-trip via separate
    # QuadGK integration in the test, (2) f(0) > 0 and decays as |t| grows,
    # (3) cutoff bounds the kernel correctly.
    # -----------------------------------------------------------------------
    @testset "(i) DLLMetropolisFilter time_kernel + cutoff" begin
        # ----- (i.1) f(0) matches the bare integral of f̂(ν)/2π -----
        @testset "f(0) = (1/2π) ∫ f̂(ν) dν" begin
            for (beta, S) in ((1.0, 2.0), (5.0, 2.0), (10.0, 2.0))
                f = DLLMetropolisFilter(beta; S = S)
                # Independent QuadGK reference (just the integrand without phase).
                ref, _ = quadgk(nu -> freq_kernel(f, nu), -S, S;
                                rtol = 1e-13, atol = 1e-15)
                ref_complex = ComplexF64(ref / (2π))
                @test isapprox(time_kernel(f, 0.0), ref_complex;
                               atol = 1e-12, rtol = 1e-10)
            end
        end

        # ----- (i.2) Fourier round-trip: time_kernel followed by FT
        # recovers freq_kernel. This is a numerical transform check at a
        # practical window; the certified cutoff is tested separately below.
        @testset "Fourier round-trip (time → freq)" begin
            for beta in (1.0, 5.0)
                f = DLLMetropolisFilter(beta; S = 2.0)
                T_cut = 160.0
                for nu in (-0.7, -0.2, 0.0, 0.3, 0.9)
                    integrand = t -> time_kernel(f, t) * cis(nu * t)
                    num, _ = quadgk(integrand, -T_cut, T_cut;
                                    rtol = 1e-10, atol = 1e-12)
                    expected = freq_kernel(f, nu)
                    @test isapprox(num, expected; atol = 1e-6, rtol = 1e-6)
                end
            end
        end

        # ----- (i.3) Time kernel modulus decays for large |t|. -----
        @testset "modulus decay for |t| large" begin
            for beta in (1.0, 2.0, 5.0)
                f = DLLMetropolisFilter(beta; S = 2.0)
                f0 = abs(time_kernel(f, 0.0))
                # Pick t_far well past 8β to ensure decay.
                t_far = 30 * beta
                f_far = abs(time_kernel(f, t_far))
                @test f_far < f0 / 100  # at least 100x smaller
            end
        end

        # ----- (i.4) filter_time_cutoff: the fourth-derivative envelope
        # controls the weighted tail of every centred uniform time lattice. -----
        @testset "filter_time_cutoff certifies the trapezoidal lattice tail" begin
            for tol in (1e-6, 1e-9, 1e-12)
                for beta in (1.0, 2.0, 5.0, 10.0)
                    f = DLLMetropolisFilter(beta; S = 2.0)
                    tc = filter_time_cutoff(f, tol)
                    tail_bound = QuantumFurnace._dll_metropolis_time_tail_bound(f, tc)
                    @test tail_bound <= tol
                    @test isapprox(
                        QuantumFurnace._dll_metropolis_time_tail_bound(f, 2tc),
                        tail_bound / 8;
                        atol = 0,
                        rtol = 100eps(Float64),
                    )
                    @test isfinite(tc) && tc > 0
                end
            end
            f = DLLMetropolisFilter(2.0; S = 2.0)
            tc = filter_time_cutoff(f, 1e-9)
            derivative_bound =
                QuantumFurnace._dll_metropolis_fourier_d4_l1_bound(f)
            # Worst spacing-independent case: tau is just above the cutoff,
            # so every nonzero lattice point is discarded. The exact envelope
            # sum uses zeta(4)=pi^4/90 and remains below the returned bound.
            tau = nextfloat(tc)
            exact_envelope_sum = derivative_bound * (pi^4 / 90) /
                                 (pi * tau^3)
            @test exact_envelope_sum <=
                  QuantumFurnace._dll_metropolis_time_tail_bound(f, tc)

            # Ratios that are representable only after combining powers must
            # not spuriously underflow the certificate to zero.
            extreme = DLLMetropolisFilter(1e-100; S = 1e100)
            @test QuantumFurnace._dll_metropolis_fourier_d4_l1_bound(extreme) > 0
            @test filter_time_cutoff(extreme, 1e-12) > 0
            @test_throws ArgumentError filter_time_cutoff(f, 0.0)
            @test_throws ArgumentError filter_time_cutoff(f, Inf)
        end

        # ----- (i.5) Cutoff scales monotonically: smaller tol ⇒ larger cutoff -----
        @testset "cutoff monotonicity in tol" begin
            f = DLLMetropolisFilter(2.0; S = 2.0)
            tcs = [filter_time_cutoff(f, tol) for tol in (1e-3, 1e-6, 1e-9, 1e-12)]
            @test all(diff(tcs) .>= 0)  # non-decreasing
        end

        # ----- (i.6) Type stability: Float64 input → ComplexF64 result. -----
        @testset "type stability" begin
            f = DLLMetropolisFilter(2.0; S = 2.0)
            @test typeof(time_kernel(f, 1.0)) === ComplexF64
            @test typeof(filter_time_cutoff(f, 1e-9)) === Float64
        end
    end
end
