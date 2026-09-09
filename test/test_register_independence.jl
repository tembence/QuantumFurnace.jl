using Test
using QuantumFurnace
using LinearAlgebra
using QuantumFurnace: register_t0_D, register_w0_D, register_r_D,
    register_t0_b_minus, register_w0_b_minus, register_r_b_minus,
    register_t0_b_plus, register_w0_b_plus, register_r_b_plus

# byte-identical regression + independent-variation Lindbladian
# checks for the per-term register design.
#
# Setting all three triples equal to the legacy `(num_energy_bits, t0, w0)`
# value must reproduce the shared-register Lindbladian exactly, regardless of
# construction (KMS / GNS / DLL) or domain (Bohr / Energy / Time).

const _BETA_REG_INDEP = 5.0
const _SIGMA_REG_INDEP = 1.0 / _BETA_REG_INDEP
const _NUM_QUBITS_REG_INDEP = 3

# Need a fresh n=3 fixture at β = 5 (test_helpers's TEST_HAM is at β=BETA=10).
function _make_reg_indep_fixture()
    ham_path = test_hamiltonian_path(3)
    ham = QuantumFurnace._load_hamiltonian_bson(ham_path, _BETA_REG_INDEP)
    jump_paulis = [[X], [Y], [Z]]
    num_jumps = length(jump_paulis) * _NUM_QUBITS_REG_INDEP
    jump_norm = sqrt(num_jumps)
    jumps = JumpOp[]
    for pauli in jump_paulis
        for site in 1:_NUM_QUBITS_REG_INDEP
            op = Matrix(pad_term(pauli, _NUM_QUBITS_REG_INDEP, site)) ./ jump_norm
            op_eb = ham.eigvecs' * op * ham.eigvecs
            push!(jumps, JumpOp(op, op_eb, op == transpose(op), op == op'))
        end
    end
    return (; ham, jumps)
end

# Build a `Config` at the legacy `(N, w0, t0)` register, then a copy with
# all three explicit triples set to the same values. They should produce
# byte-identical Lindbladians.
function _legacy_kms_cfg(domain; N::Int = 10, w0::Real = 0.05, beta = _BETA_REG_INDEP)
    t0 = 2π / (2^N * w0)
    Config(;
        sim = Lindbladian(),
        domain = domain,
        construction = KMS(),
        num_qubits = _NUM_QUBITS_REG_INDEP,
        with_linear_combination = true,
        beta = float(beta),
        sigma = 1.0 / float(beta),
        a = float(beta) / 30.0,
        s = 0.4,
        num_energy_bits = N,
        w0 = w0,
        t0 = t0,
        num_trotter_steps_per_t0 = 10,
    )
end

function _new_kms_cfg(domain; N::Int = 10, w0::Real = 0.05, beta = _BETA_REG_INDEP)
    t0 = 2π / (2^N * w0)
    Config(;
        sim = Lindbladian(),
        domain = domain,
        construction = KMS(),
        num_qubits = _NUM_QUBITS_REG_INDEP,
        with_linear_combination = true,
        beta = float(beta),
        sigma = 1.0 / float(beta),
        a = float(beta) / 30.0,
        s = 0.4,
        num_energy_bits_D = N, w0_D = w0, t0_D = t0,
        num_energy_bits_b_minus = N, w0_b_minus = w0, t0_b_minus = t0,
        num_energy_bits_b_plus = N, w0_b_plus = w0, t0_b_plus = t0,
        num_trotter_steps_per_t0 = 10,
    )
end

function _legacy_dll_cfg(; N::Int = 10, w0::Real = 0.05, beta = _BETA_REG_INDEP)
    t0 = 2π / (2^N * w0)
    Config(;
        sim = Lindbladian(),
        domain = TimeDomain(),
        construction = DLL(),
        num_qubits = _NUM_QUBITS_REG_INDEP,
        with_linear_combination = false,
        beta = float(beta),
        sigma = 1.0 / float(beta),
        gaussian_parameters = (0.5 * float(beta), sqrt(2.0 * 0.5 - (1.0 / float(beta))^2)),
        num_energy_bits = N,
        t0 = t0,
        filter = DLLGaussianFilter(float(beta)),
    )
end

function _new_dll_cfg(; N::Int = 10, w0::Real = 0.05, beta = _BETA_REG_INDEP)
    t0 = 2π / (2^N * w0)
    Config(;
        sim = Lindbladian(),
        domain = TimeDomain(),
        construction = DLL(),
        num_qubits = _NUM_QUBITS_REG_INDEP,
        with_linear_combination = false,
        beta = float(beta),
        sigma = 1.0 / float(beta),
        gaussian_parameters = (0.5 * float(beta), sqrt(2.0 * 0.5 - (1.0 / float(beta))^2)),
        num_energy_bits_D = N,
        t0_D = t0,
        filter = DLLGaussianFilter(float(beta)),
    )
end

@testset "Per-register independence" begin
    sys = _make_reg_indep_fixture()

    @testset "byte-identical regression: KMS / $(typeof(dom))" for dom in (BohrDomain(), EnergyDomain(), TimeDomain())
        L_legacy = construct_lindbladian(sys.jumps, _legacy_kms_cfg(dom), sys.ham)
        L_new    = construct_lindbladian(sys.jumps, _new_kms_cfg(dom),    sys.ham)
        @test L_legacy == L_new
    end

    @testset "byte-identical regression: DLL TimeDomain" begin
        L_legacy = construct_lindbladian(sys.jumps, _legacy_dll_cfg(), sys.ham)
        L_new    = construct_lindbladian(sys.jumps, _new_dll_cfg(),    sys.ham)
        @test L_legacy == L_new
    end

    @testset "BSON dual-schema: legacy keys auto-promote to per-term registers" begin
        # `_config_to_dict` writes both legacy and per-term keys.
        # `_dict_to_config_kwargs` reads either schema; old caches without
        # per-term keys land in the legacy fields and the helper accessors
        # auto-promote at access time.
        cfg_orig = _new_kms_cfg(TimeDomain())  # explicit per-term triples
        d_new = QuantumFurnace._config_to_dict(cfg_orig)
        @test haskey(d_new, :num_energy_bits_D)
        @test haskey(d_new, :t0_b_minus)
        @test haskey(d_new, :w0_b_plus)
        @test haskey(d_new, :num_energy_bits)  # legacy key still present
        cfg_round = QuantumFurnace._reconstruct_config(d_new)
        @test register_t0_D(cfg_round) == register_t0_D(cfg_orig)
        @test register_t0_b_minus(cfg_round) == register_t0_b_minus(cfg_orig)
        @test register_t0_b_plus(cfg_round) == register_t0_b_plus(cfg_orig)

        # Simulate a legacy-only cache (no per-term keys).
        d_legacy = Dict{Symbol, Any}(
            :config_type => "KMS",
            :config_kind => "lindbladian",
            :domain => "TimeDomain",
            :num_qubits => 3,
            :with_coherent => true,
            :with_linear_combination => true,
            :beta => 5.0,
            :sigma => 0.2,
            :a => 5.0 / 30.0,
            :s => 0.4,
            :num_energy_bits => 12,
            :t0 => 2π / (2^12 * 0.05),
            :w0 => 0.05,
            :num_trotter_steps_per_t0 => 10,
        )
        cfg_legacy = QuantumFurnace._reconstruct_config(d_legacy)
        @test register_t0_D(cfg_legacy) == d_legacy[:t0]
        @test register_t0_b_minus(cfg_legacy) == d_legacy[:t0]
        @test register_t0_b_plus(cfg_legacy) == d_legacy[:t0]
        @test register_w0_D(cfg_legacy) == d_legacy[:w0]
        @test register_r_b_plus(cfg_legacy) == d_legacy[:num_energy_bits]
    end

end
