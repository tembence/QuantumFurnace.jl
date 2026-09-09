using Test
using QuantumFurnace
using QuantumFurnace: register_t0_D, register_w0_D, register_r_D,
    register_t0_b_minus, register_w0_b_minus, register_r_b_minus,
    register_t0_b_plus, register_w0_b_plus, register_r_b_plus,
    validate_config!

# Per-register design: Config carries three independent register
# triples `(t0_X, w0_X, r_X)` for X ∈ {D, b_minus, b_plus}, each obeying its
# own Fourier relation `t0_X · w0_X ≈ 2π / 2^{r_X}`. The legacy single-register
# kwargs `(t0, w0, num_energy_bits)` still work — the helper accessors fall
# back to them when the per-term field is `nothing`.

const _LEGACY_KMS_KW = (
    sim = Lindbladian(),
    domain = TimeDomain(),
    construction = KMS(),
    num_qubits = 3,
    with_linear_combination = true,
    beta = 10.0,
    sigma = 0.1,
    a = 10.0 / 30.0,
    s = 0.4,
    num_energy_bits = 12,
    w0 = 0.05,
    t0 = 2π / (2^12 * 0.05),
    num_trotter_steps_per_t0 = 10,
)

@testset "Per-register validation" begin
    @testset "Helper accessors fall back to legacy fields" begin
        cfg = Config(; _LEGACY_KMS_KW...)
        @test register_t0_D(cfg) == cfg.t0
        @test register_w0_D(cfg) == cfg.w0
        @test register_r_D(cfg) == cfg.num_energy_bits
        @test register_t0_b_minus(cfg) == cfg.t0
        @test register_w0_b_minus(cfg) == cfg.w0
        @test register_r_b_minus(cfg) == cfg.num_energy_bits
        @test register_t0_b_plus(cfg) == cfg.t0
        @test register_w0_b_plus(cfg) == cfg.w0
        @test register_r_b_plus(cfg) == cfg.num_energy_bits
    end

    @testset "Helper accessors prefer per-term fields when set" begin
        N_D = 12; w0_D = 0.05; t0_D = 2π / (2^N_D * w0_D)
        N_bm = 10; w0_bm = 0.2; t0_bm = 2π / (2^N_bm * w0_bm)
        N_bp = 8; w0_bp = 0.5; t0_bp = 2π / (2^N_bp * w0_bp)

        cfg = Config(;
            sim = Lindbladian(), domain = TimeDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = true, beta = 10.0, sigma = 0.1,
            a = 10.0 / 30.0, s = 0.4,
            num_energy_bits_D = N_D, t0_D = t0_D, w0_D = w0_D,
            num_energy_bits_b_minus = N_bm, t0_b_minus = t0_bm, w0_b_minus = w0_bm,
            num_energy_bits_b_plus = N_bp, t0_b_plus = t0_bp, w0_b_plus = w0_bp,
            num_trotter_steps_per_t0 = 10,
        )
        @test register_t0_D(cfg) == t0_D
        @test register_t0_b_minus(cfg) == t0_bm
        @test register_t0_b_plus(cfg) == t0_bp
        @test register_w0_b_minus(cfg) == w0_bm
        @test register_r_b_plus(cfg) == N_bp
    end

    @testset "Legacy KMS TimeDomain validates" begin
        cfg = Config(; _LEGACY_KMS_KW...)
        @test validate_config!(cfg) === nothing
    end

    @testset "Common physical parameter domain" begin
        make_cfg(; kwargs...) = Config(; merge(_LEGACY_KMS_KW, (; kwargs...))...)
        for bad in (
            make_cfg(num_qubits = 0),
            make_cfg(beta = 0.0),
            make_cfg(beta = -1.0),
            make_cfg(beta = Inf),
            make_cfg(sigma = 0.0),
            make_cfg(sigma = -0.1),
            make_cfg(a = nothing),
            make_cfg(s = nothing),
            make_cfg(a = -0.1),
            make_cfg(s = -0.1),
            make_cfg(a = NaN),
            make_cfg(s = NaN),
        )
            @test_throws ArgumentError validate_config!(bad)
        end

        make_thermal(; kwargs...) = Config(; merge(
            _LEGACY_KMS_KW,
            (; sim = Thermalize(), mixing_time = 1.0, delta = 0.1),
            (; kwargs...),
        )...)
        @test validate_config!(make_thermal()) === nothing
        @test_throws ArgumentError validate_config!(make_thermal(mixing_time = -1.0))
        @test_throws ArgumentError validate_config!(make_thermal(delta = 0.0))
        @test_throws ArgumentError validate_config!(make_thermal(delta = 1.01))
    end

    @testset "Independent triples KMS TimeDomain validates" begin
        N_D = 12; w0_D = 0.05; t0_D = 2π / (2^N_D * w0_D)
        N_bm = 10; w0_bm = 0.2; t0_bm = 2π / (2^N_bm * w0_bm)
        N_bp = 8; w0_bp = 0.5; t0_bp = 2π / (2^N_bp * w0_bp)
        cfg = Config(;
            sim = Lindbladian(), domain = TimeDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = true, beta = 10.0, sigma = 0.1,
            a = 10.0 / 30.0, s = 0.4,
            num_energy_bits_D = N_D, t0_D = t0_D, w0_D = w0_D,
            num_energy_bits_b_minus = N_bm, t0_b_minus = t0_bm, w0_b_minus = w0_bm,
            num_energy_bits_b_plus = N_bp, t0_b_plus = t0_bp, w0_b_plus = w0_bp,
            num_trotter_steps_per_t0 = 10,
        )
        @test validate_config!(cfg) === nothing
    end

    @testset "Mismatched Fourier on b_minus is rejected" begin
        N_D = 12; w0_D = 0.05; t0_D = 2π / (2^N_D * w0_D)
        N_bm = 10; w0_bm = 0.2; t0_bm = 2π / (2^N_bm * w0_bm)
        cfg = Config(;
            sim = Lindbladian(), domain = TimeDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = true, beta = 10.0, sigma = 0.1,
            a = 10.0 / 30.0, s = 0.4,
            num_energy_bits_D = N_D, t0_D = t0_D, w0_D = w0_D,
            num_energy_bits_b_minus = N_bm, t0_b_minus = 0.99 * t0_bm, w0_b_minus = w0_bm,
            num_energy_bits_b_plus = N_bm, t0_b_plus = t0_bm, w0_b_plus = w0_bm,
            num_trotter_steps_per_t0 = 10,
        )
        @test_throws ArgumentError validate_config!(cfg)
    end

    @testset "EnergyDomain requires only (r_D, w0_D), no t0_D" begin
        # Legacy form
        cfg_legacy = Config(;
            sim = Lindbladian(), domain = EnergyDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = true, beta = 10.0, sigma = 0.1,
            a = 10.0 / 30.0, s = 0.4,
            num_energy_bits = 12, w0 = 0.05,
        )
        @test validate_config!(cfg_legacy) === nothing
        # Per-term form (no t0 needed)
        cfg_new = Config(;
            sim = Lindbladian(), domain = EnergyDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = true, beta = 10.0, sigma = 0.1,
            a = 10.0 / 30.0, s = 0.4,
            num_energy_bits_D = 12, w0_D = 0.05,
        )
        @test validate_config!(cfg_new) === nothing
    end

    @testset "DLL TimeDomain validates with (r_D, t0_D), no w0_D" begin
        cfg = Config(;
            sim = Lindbladian(), domain = TimeDomain(), construction = DLL(),
            num_qubits = 3, with_linear_combination = false, beta = 10.0, sigma = 0.1,
            gaussian_parameters = (5.0, sqrt(2.0 * 5.0 / 10.0 - 0.1^2)),
            num_energy_bits_D = 12, t0_D = 2π / (2^12 * 0.05),
            filter = DLLGaussianFilter(10.0),
        )
        @test validate_config!(cfg) === nothing
    end

    @testset "Missing per-term r_D is rejected (TimeDomain KMS)" begin
        cfg = Config(;
            sim = Lindbladian(), domain = TimeDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = true, beta = 10.0, sigma = 0.1,
            a = 10.0 / 30.0, s = 0.4,
            t0_D = 2π / (2^12 * 0.05), w0_D = 0.05,
            t0_b_minus = 2π / (2^12 * 0.05), w0_b_minus = 0.05, num_energy_bits_b_minus = 12,
            t0_b_plus  = 2π / (2^12 * 0.05), w0_b_plus  = 0.05, num_energy_bits_b_plus  = 12,
            num_trotter_steps_per_t0 = 10,
        )
        @test_throws ArgumentError validate_config!(cfg)
    end

    @testset "TrotterDomain requires three triples for KMS" begin
        cfg = Config(;
            sim = Lindbladian(), domain = TrotterDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = true, beta = 10.0, sigma = 0.1,
            a = 10.0 / 30.0, s = 0.4,
            num_energy_bits_D = 12, t0_D = 2π / (2^12 * 0.05), w0_D = 0.05,
            num_energy_bits_b_minus = 12, t0_b_minus = 2π / (2^12 * 0.05), w0_b_minus = 0.05,
            num_energy_bits_b_plus = 12, t0_b_plus = 2π / (2^12 * 0.05), w0_b_plus = 0.05,
            num_trotter_steps_per_t0 = 10,
        )
        @test validate_config!(cfg) === nothing

        cfg_per_leg_M = Config(;
            sim = Lindbladian(), domain = TrotterDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = true, beta = 10.0, sigma = 0.1,
            a = 10.0 / 30.0, s = 0.4,
            num_energy_bits_D = 12, t0_D = 2π / (2^12 * 0.05), w0_D = 0.05,
            num_energy_bits_b_minus = 12, t0_b_minus = 2π / (2^12 * 0.05), w0_b_minus = 0.05,
            num_energy_bits_b_plus = 12, t0_b_plus = 2π / (2^12 * 0.05), w0_b_plus = 0.05,
            num_trotter_steps_per_t0_D = 4,
            num_trotter_steps_per_t0_b_minus = 5,
            num_trotter_steps_per_t0_b_plus = 6,
        )
        @test validate_config!(cfg_per_leg_M) === nothing
    end
end
