"""
Tests for the GQSP coherent-step configuration fields and validation.

Covers:
- Default `with_gqsp=false` and `gqsp_degree=1` on a fresh `Config`.
- `validate_config!` accepts `with_gqsp=true` for the supported regime
  (`with_coherent(construction)`, Time/TrotterDomain, `1 ≤ gqsp_degree ≤ 100`).
- `validate_config!` rejects:
    * `GNS` (no coherent term),
    * `EnergyDomain` / `BohrDomain`,
    * `gqsp_degree < 1`,
    * `gqsp_degree > 100`.
"""

@testset "GQSP config fields and validation" begin

    @testset "coherent construction trait" begin
        @test with_coherent(KMS())
        @test !with_coherent(GNS())
        @test with_coherent(DLL())
    end

    # Build a Thermalize/KMS/TimeDomain config with overridable kwargs
    function _make_cfg(; construction=KMS(), domain=TimeDomain(), with_gqsp=false, gqsp_degree=1)
        Config(;
            sim=Thermalize(), domain=domain, construction=construction,
            num_qubits=NUM_QUBITS, with_linear_combination=true,
            beta=BETA, sigma=SIGMA, a=BETA / 30.0, s=0.4,
            num_energy_bits=NUM_ENERGY_BITS, w0=W0, t0=T0,
            num_trotter_steps_per_t0=NUM_TROTTER_STEPS_PER_T0,
            mixing_time=1.0, delta=TEST_DELTA,
            with_gqsp=with_gqsp, gqsp_degree=gqsp_degree,
        )
    end

    @testset "Defaults: with_gqsp=false, gqsp_degree=1" begin
        cfg = make_config(Thermalize(), TimeDomain())
        @test cfg.with_gqsp === false
        @test cfg.gqsp_degree === 1
        validate_config!(cfg)  # baseline must not throw
    end

    @testset "Accepts with_gqsp=true in supported regime" begin
        # KMS + TimeDomain + d=1 (default)
        validate_config!(_make_cfg(with_gqsp=true))
        # KMS + TrotterDomain
        validate_config!(_make_cfg(with_gqsp=true, domain=TrotterDomain()))
        # higher degrees within cap
        validate_config!(_make_cfg(with_gqsp=true, gqsp_degree=2))
        validate_config!(_make_cfg(with_gqsp=true, gqsp_degree=100))
    end

    @testset "Rejects with_gqsp=true with GNS (no coherent term)" begin
        @test_throws ArgumentError validate_config!(_make_cfg(with_gqsp=true, construction=GNS()))
    end

    @testset "Rejects with_gqsp=true outside Time/TrotterDomain" begin
        @test_throws ArgumentError validate_config!(_make_cfg(with_gqsp=true, domain=EnergyDomain()))
        @test_throws ArgumentError validate_config!(_make_cfg(with_gqsp=true, domain=BohrDomain()))
    end

    @testset "Rejects with_gqsp=true with DLL (no DLL block-encoding norm yet)" begin
        cfg_dll = Config(;
            sim=Thermalize(), domain=TimeDomain(), construction=DLL(),
            num_qubits=NUM_QUBITS, with_linear_combination=true,
            beta=BETA, sigma=SIGMA, a=BETA / 30.0, s=0.4,
            num_energy_bits=NUM_ENERGY_BITS, w0=W0, t0=T0,
            num_trotter_steps_per_t0=NUM_TROTTER_STEPS_PER_T0,
            mixing_time=1.0, delta=TEST_DELTA,
            filter=DLLGaussianFilter(BETA),
            with_gqsp=true, gqsp_degree=1,
        )
        @test_throws ArgumentError validate_config!(cfg_dll)
    end

    @testset "Rejects gqsp_degree out of [1, 100]" begin
        @test_throws ArgumentError validate_config!(_make_cfg(with_gqsp=true, gqsp_degree=0))
        @test_throws ArgumentError validate_config!(_make_cfg(with_gqsp=true, gqsp_degree=-3))
        @test_throws ArgumentError validate_config!(_make_cfg(with_gqsp=true, gqsp_degree=101))
    end

end
