"""
Tests for the eta-regularized smooth Metropolis variant (a=0, s>0):
1. `_compute_b_plus_metro(t, β, σ, η, 0.0)` reduces to the legacy 4-arg form
2. `_select_b_plus_calculator` returns `_compute_b_plus_metro` with the right
   args for the (a=0, s>0) config
3. `validate_config!` accepts (a=0, s>0, eta>0) and rejects (a=0, s>0, eta=0/missing)
4. `_precompute_data` produces a non-empty `b_plus` Dict for the new path
"""

@testset "Smooth Metropolis eta-regularization (a=0, s>0)" begin

    # Four-argument formula for reference comparison
    function ref_b_plus_metro_legacy(t, beta, sigma, eta)
        if abs(t) < 1e-12
            return complex(1 / (2 * sqrt(2) * pi^2))
        elseif abs(t) <= eta
            numerator = exp(-sigma^2 * beta^2 * (2*t^2 + 1im*t)) + 1im * (2*t + 1im)
        else
            numerator = exp(-sigma^2 * beta^2 * (2*t^2 + 1im*t))
        end
        denominator = t * (2*t + 1im)
        return (1 / (2 * sqrt(2) * pi^2)) * numerator / denominator
    end

    @testset "s=0 reduces to legacy form (σβ=1 case)" begin
        # BETA = 10.0, SIGMA = 0.1 → σβ = 1, so the legacy hard-code matches at t=0
        for t in [-1.0, -0.5, -0.05, 0.05, 0.5, 1.0]
            new = QuantumFurnace._compute_b_plus_metro(t, BETA, SIGMA, 0.05, 0.0)
            legacy = ref_b_plus_metro_legacy(t, BETA, SIGMA, 0.05)
            @test new ≈ legacy atol=TOL_EXACT
        end
        # t=0 case: at σβ=1 the new analytic limit equals the legacy hard-code
        @test QuantumFurnace._compute_b_plus_metro(0.0, BETA, SIGMA, 0.05, 0.0) ≈
              complex(1 / (2 * sqrt(2) * pi^2)) atol=TOL_EXACT
    end

    @testset "s>0 modifies kernel (smoothing factor (1+s))" begin
        # For s > 0, the kernel is smoother but should not blow up; verify
        # that the s=0.4 value differs from the legacy form for nonzero t
        for t in [0.05, 0.5, 1.0]
            with_s = QuantumFurnace._compute_b_plus_metro(t, BETA, SIGMA, 0.05, 0.4)
            without_s = QuantumFurnace._compute_b_plus_metro(t, BETA, SIGMA, 0.05, 0.0)
            @test isfinite(real(with_s))
            @test isfinite(imag(with_s))
            @test with_s !== without_s  # genuinely different
        end
        # t=0 limit: (2 - σ²β²(1+s)) / (2√2 π²); at σβ=1, s=0.4 → (2 - 1.4)/(2√2 π²)
        expected = complex((2 - SIGMA^2 * BETA^2 * 1.4) / (2 * sqrt(2) * pi^2))
        @test QuantumFurnace._compute_b_plus_metro(0.0, BETA, SIGMA, 0.05, 0.4) ≈
              expected atol=TOL_EXACT
    end

    @testset "_select_b_plus_calculator dispatches to metro for a=0, s>0" begin
        config_a0_spos = Config(;
            sim=Thermalize(), domain=TimeDomain(), construction=KMS(),
            num_qubits=NUM_QUBITS, with_linear_combination=true,
            beta=BETA, sigma=SIGMA, a=0.0, s=0.4, eta=0.05,
            num_energy_bits=NUM_ENERGY_BITS, w0=W0, t0=T0,
            num_trotter_steps_per_t0=NUM_TROTTER_STEPS_PER_T0,
            mixing_time=1.0, delta=TEST_DELTA,
        )
        validate_config!(config_a0_spos)  # must NOT throw
        bp_fn, bp_args = QuantumFurnace._select_b_plus_calculator(config_a0_spos)
        @test bp_fn === QuantumFurnace._compute_b_plus_metro
        @test bp_args == (BETA, SIGMA, 0.05, 0.4)
    end

    @testset "validate_config! enforces eta>0 when a=0 in TimeDomain" begin
        # Missing eta with a=0, s>0 in TimeDomain → must throw ArgumentError
        @test_throws ArgumentError Config(;
            sim=Thermalize(), domain=TimeDomain(), construction=KMS(),
            num_qubits=NUM_QUBITS, with_linear_combination=true,
            beta=BETA, sigma=SIGMA, a=0.0, s=0.4,
            num_energy_bits=NUM_ENERGY_BITS, w0=W0, t0=T0,
            num_trotter_steps_per_t0=NUM_TROTTER_STEPS_PER_T0,
            mixing_time=1.0, delta=TEST_DELTA,
        ) |> validate_config!
    end

    @testset "_precompute_data populates b_plus for a=0, s>0" begin
        config_a0_spos = Config(;
            sim=Thermalize(), domain=TimeDomain(), construction=KMS(),
            num_qubits=NUM_QUBITS, with_linear_combination=true,
            beta=BETA, sigma=SIGMA, a=0.0, s=0.4, eta=0.05,
            num_energy_bits=NUM_ENERGY_BITS, w0=W0, t0=T0,
            num_trotter_steps_per_t0=NUM_TROTTER_STEPS_PER_T0,
            mixing_time=1.0, delta=TEST_DELTA,
        )
        pd = QuantumFurnace._precompute_data(config_a0_spos, TEST_HAM)
        @test pd.b_plus !== nothing
        @test !isempty(pd.b_plus)
        # Spot check: every value should be finite
        for (_, v) in pd.b_plus
            @test isfinite(real(v))
            @test isfinite(imag(v))
        end
    end

end

@testset "Metropolis rates and Bohr coefficients against high precision" begin
    using SpecialFunctions: erfc

    # Evaluate the unfactored closed forms in BigFloat, whose exponent range
    # retains the large exponentials and small Gaussian tails separately.
    function reference_rate(w, beta, sigma, a, s, construction)
        w, beta, sigma, a, s = BigFloat.((w, beta, sigma, a, s))
        shifted = construction isa KMS ? w + beta * sigma^2 / 2 : w
        A = sqrt(beta * (4a + 1) / 4)
        B = sqrt(beta / 4) * abs(shifted)
        u = sqrt(beta * sigma^2 * s / 2)
        common = -beta * w / 2 - (construction isa KMS ? beta^2 * sigma^2 / 4 : zero(beta))
        iszero(s) && return exp(common - 2A * B)
        return exp(common - 2A * B) *
            (erfc(A * u - B / u) + exp(4A * B) * erfc(A * u + B / u)) / 2
    end

    function reference_alpha(v, w, beta, sigma, a, s, construction)
        v, w, beta, sigma, a, s = BigFloat.((v, w, beta, sigma, a, s))
        A = sqrt(beta * (4a + 1) / 4)
        shifted = v + w + (construction isa GNS ? beta * sigma^2 / 2 : zero(beta))
        B = sqrt(beta / 16) * abs(shifted)
        u = sqrt(beta * sigma^2 * (1 + s) / 2)
        return exp(a * beta^2 * sigma^2 / 2) * exp(-beta * (v + w) / 4) *
            exp(-(v - w)^2 / (8sigma^2)) * exp(-2A * B) *
            (erfc(A * u - B / u) + exp(4A * B) * erfc(A * u + B / u)) / 2
    end

    setprecision(BigFloat, 256) do
        for construction in (KMS(), GNS()), beta in (0.7, 10.0, 2000.0),
            (a, s) in ((0.0, 0.0), (0.3, 0.0), (0.0, 0.25), (0.3, 0.25))
            sigma = inv(beta)
            cfg = Config(; sim=Lindbladian(), domain=BohrDomain(), construction,
                num_qubits=1, with_linear_combination=true, beta, sigma, a, s)
            rate = pick_transition(cfg)
            alpha = QuantumFurnace._pick_alpha(cfg)
            scalar_alpha = construction isa KMS ? QuantumFurnace.create_alpha : QuantumFurnace.create_alpha_gns
            kink = construction isa KMS ? -beta * sigma^2 / 2 : 0.0
            for w in (-0.45, -0.01, 0.0, 0.3, kink)
                expected = Float64(reference_rate(w, beta, sigma, a, s, construction))
                @test isfinite(pick_transition(cfg, w))
                @test pick_transition(cfg, w) ≈ expected rtol=2e-11 atol=0.0
                @test rate(w) ≈ expected rtol=2e-11 atol=0.0
                if construction isa KMS && s > 0
                    typed = SmoothMetropolisTransition(beta; sigma, a, s)
                    @test transition_value(typed, w) ≈ expected rtol=2e-11 atol=0.0
                end
            end
            for (v, w) in ((-0.45, -0.45), (-0.45, 0.3), (0.0, 0.0), (0.3, 0.3), (-sigma, 0.3sigma))
                expected = Float64(reference_alpha(v, w, beta, sigma, a, s, construction))
                actual = scalar_alpha(v, w, beta, sigma, a, s)
                @test isfinite(actual)
                @test actual ≈ expected rtol=2e-11 atol=0.0
                @test alpha(v, w) ≈ expected rtol=2e-11 atol=0.0
                @test QuantumFurnace._pick_alpha(cfg, v, w) ≈ expected rtol=2e-11 atol=0.0
            end
        end

        # Both erfc terms need scaling when the common exponential is large.
        for construction in (KMS(), GNS())
            beta, sigma, a, s = 100.0, 1.0, 2.0, 0.001
            scalar_alpha = construction isa KMS ? QuantumFurnace.create_alpha : QuantumFurnace.create_alpha_gns
            center = construction isa KMS ? -50.0 : -25.0
            for (v, w) in ((center, center), (center + 1, center))
                expected = Float64(reference_alpha(v, w, beta, sigma, a, s, construction))
                @test isfinite(scalar_alpha(v, w, beta, sigma, a, s))
                @test scalar_alpha(v, w, beta, sigma, a, s) ≈ expected rtol=2e-11 atol=0.0
            end
        end
        beta, sigma, a, s, frequency = 10000.0, 0.004, 2.0, 0.001, -0.08008
        expected = Float64(reference_alpha(frequency, frequency, beta, sigma, a, s, KMS()))
        @test QuantumFurnace.create_alpha(frequency, frequency, beta, sigma, a, s) ≈ expected rtol=2e-11
    end
end

@testset "Energy-domain γ for a=0, s>0 uses smooth Metropolis form (eq:smooth-metro)" begin
    using SpecialFunctions: erfc

    function _make_cfg_eD(construction, a, s)
        Config(;
            sim=Thermalize(), domain=EnergyDomain(), construction=construction,
            num_qubits=NUM_QUBITS, with_linear_combination=true,
            beta=BETA, sigma=SIGMA, a=a, s=s,
            num_energy_bits=NUM_ENERGY_BITS, w0=W0, t0=T0,
            num_trotter_steps_per_t0=NUM_TROTTER_STEPS_PER_T0,
            mixing_time=1.0, delta=TEST_DELTA,
        )
    end

    # Closed-form γ at a=0 from thesis eq:smooth-metro (line 297-303):
    #   γ_M^{(s)}(ω) = γ_M^{(0)}(ω) · (1/2) [erfc(z_-) + e^{β|ω̃|} erfc(z_+)]
    # with ω̃ = ω + βσ²/2 and z_± = βσ√s/(2√2) ± |ω̃|/(σ√(2s)).
    function thesis_gamma_a0_kms(omega, beta, sigma, s)
        omega_tilde = omega + beta * sigma^2 / 2
        gamma_0 = exp(-beta * max(omega_tilde, 0.0))
        s == 0 && return gamma_0
        z_plus  = beta * sigma * sqrt(s) / (2*sqrt(2)) + abs(omega_tilde) / (sigma * sqrt(2*s))
        z_minus = beta * sigma * sqrt(s) / (2*sqrt(2)) - abs(omega_tilde) / (sigma * sqrt(2*s))
        return gamma_0 * (erfc(z_minus) + exp(beta * abs(omega_tilde)) * erfc(z_plus)) / 2
    end

    # GNS un-shifted analog (replace ω̃ → ω in z_± and prefactor)
    function thesis_gamma_a0_gns(omega, beta, sigma, s)
        gamma_0 = exp(-beta * max(omega, 0.0))
        s == 0 && return gamma_0
        z_plus  = beta * sigma * sqrt(s) / (2*sqrt(2)) + abs(omega) / (sigma * sqrt(2*s))
        z_minus = beta * sigma * sqrt(s) / (2*sqrt(2)) - abs(omega) / (sigma * sqrt(2*s))
        return gamma_0 * (erfc(z_minus) + exp(beta * abs(omega)) * erfc(z_plus)) / 2
    end

    test_omegas = [-0.5, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.5]

    @testset "KMS (a=0, s=0.4) γ matches thesis closed form" begin
        cfg = _make_cfg_eD(KMS(), 0.0, 0.4)
        for omega in test_omegas
            @test pick_transition(cfg, omega) ≈
                  thesis_gamma_a0_kms(omega, BETA, SIGMA, 0.4) atol=TOL_EXACT
        end
    end

    @testset "KMS (a=0, s=0.4) closure form matches 2-arg form" begin
        cfg = _make_cfg_eD(KMS(), 0.0, 0.4)
        gamma_fn = pick_transition(cfg)
        for omega in test_omegas
            @test gamma_fn(omega) ≈ pick_transition(cfg, omega) atol=TOL_EXACT
        end
    end

    @testset "KMS (a=0, s=0.4) differs from kinky γ near the kink (ω̃ ≈ 0)" begin
        # ω̃ = 0 ⇒ ω = -βσ²/2 = -0.05 at BETA=10, SIGMA=0.1
        cfg_kinky  = _make_cfg_eD(KMS(), 0.0, 0.0)
        cfg_smooth = _make_cfg_eD(KMS(), 0.0, 0.4)
        omega_kink = -BETA * SIGMA^2 / 2
        for delta_omega in [-0.02, -0.005, 0.005, 0.02]
            omega = omega_kink + delta_omega
            kinky  = pick_transition(cfg_kinky, omega)
            smooth = pick_transition(cfg_smooth, omega)
            @test !isapprox(kinky, smooth; atol=1e-6)
        end
    end

    @testset "KMS (a=0, s→0⁺) recovers kinky γ_M^(0)" begin
        cfg_kinky   = _make_cfg_eD(KMS(), 0.0, 0.0)
        cfg_tiny_s  = _make_cfg_eD(KMS(), 0.0, 1e-8)
        for omega in test_omegas
            @test pick_transition(cfg_tiny_s, omega) ≈
                  pick_transition(cfg_kinky, omega) atol=1e-3
        end
    end

    @testset "GNS (a=0, s=0.4) γ matches un-shifted thesis form" begin
        cfg = _make_cfg_eD(GNS(), 0.0, 0.4)
        for omega in test_omegas
            @test pick_transition(cfg, omega) ≈
                  thesis_gamma_a0_gns(omega, BETA, SIGMA, 0.4) atol=TOL_EXACT
        end
        # Closure form too
        gamma_fn = pick_transition(cfg)
        for omega in test_omegas
            @test gamma_fn(omega) ≈ pick_transition(cfg, omega) atol=TOL_EXACT
        end
    end

    @testset "GNS (a=0, s→0⁺) recovers un-shifted kinky γ" begin
        cfg_kinky  = _make_cfg_eD(GNS(), 0.0, 0.0)
        cfg_tiny_s = _make_cfg_eD(GNS(), 0.0, 1e-8)
        for omega in test_omegas
            @test pick_transition(cfg_tiny_s, omega) ≈
                  pick_transition(cfg_kinky, omega) atol=1e-3
        end
    end
end
