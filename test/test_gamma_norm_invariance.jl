"""
Grid-independent transition-rate normalization.

Verify continuum suprema, register invariance, cross-domain agreement,
Krylov and simulator parity, and the GQSP block-encoding bound.
"""

using Test
using QuantumFurnace
using LinearAlgebra

@testset "Public KMS/GNS prefactor relations" begin
    beta = 5.0
    sigma = 0.2
    a = 0.1
    s = 0.4
    nu_1, nu_2 = 0.3, -0.7
    gaussian_parameters = (0.4, 0.3)

    @test create_alpha_gns(nu_1, nu_2, beta, sigma, a, s) > 0
    @test create_alpha_gauss(nu_1, nu_2, sigma, gaussian_parameters) > 0
    @test create_alpha_gauss(nu_1, nu_2, sigma, gaussian_parameters) ≈
          create_alpha_gauss(nu_2, nu_1, sigma, gaussian_parameters) atol=1e-15

    f_12 = create_f(nu_1, nu_2, beta, sigma, a, s)
    f_gauss_12 = create_f_gauss(nu_1, nu_2, beta, sigma, gaussian_parameters)
    @test isapprox(real(f_12), 0.0; atol=1e-15)
    @test f_12 ≈ -create_f(nu_2, nu_1, beta, sigma, a, s) atol=1e-15
    @test isapprox(real(f_gauss_12), 0.0; atol=1e-15)
    @test f_gauss_12 ≈
          -create_f_gauss(nu_2, nu_1, beta, sigma, gaussian_parameters) atol=1e-15

    alpha = (x, y) -> create_alpha(x, y, beta, sigma, a, s)
    @test check_alpha_skew_symmetry(alpha, nu_1, nu_2, beta) === nothing
end

@testset "pick_gamma_sup closed-form is correct continuum sup" begin
    N_FINE = 2^16
    BETA = 10.0
    SIGMA = 0.1
    SIGMA_G = 0.5
    KMS_OMEGA_G = BETA * (SIGMA^2 + SIGMA_G^2) / 2  # = 1.3
    GNS_OMEGA_G = BETA * SIGMA_G^2 / 2              # = 1.25

    # Standard wide grid (halfwidth ±10/β = ±1.0). Peaks of γ for KMS lie at
    # ω = -βσ²/2 = -0.05 for kinky / a-reg / smooth, and at ω = -ω_γ = -1.3
    # for Gaussian; we use case-specific grids so the peak is well-resolved.
    function fine_grid_centered(center::Real; halfwidth::Real=1.0)
        # Shift the grid so the analytical peak lies exactly on a sample point.
        return collect(center .+ range(-halfwidth, halfwidth; length=N_FINE))
    end

    @testset "KMS Gaussian (γ ≤ 1)" begin
        cfg = Config(
            sim = Lindbladian(), domain = BohrDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = false,
            beta = BETA, sigma = SIGMA,
            gaussian_parameters = (KMS_OMEGA_G, SIGMA_G),
            num_energy_bits = 8, w0 = 0.05,
        )
        γ = pick_transition(cfg)
        # Closed-form maximiser: ω = -ω_γ.
        @test isapprox(γ(-KMS_OMEGA_G), 1.0; atol=1e-15)
        ω = fine_grid_centered(-KMS_OMEGA_G; halfwidth=2.0)
        @test maximum(γ.(ω)) <= 1.0 + 1e-15
        @test pick_gamma_sup(cfg) == 1.0
    end

    @testset "KMS kinky Metropolis (s=0, a=0)" begin
        cfg = Config(
            sim = Lindbladian(), domain = BohrDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = true,
            beta = BETA, sigma = SIGMA, a = 0.0, s = 0.0,
            num_energy_bits = 8, w0 = 0.05,
        )
        γ = pick_transition(cfg)
        # Closed-form maximiser: γ ≡ 1 on ω ≤ -βσ²/2.
        peak = -BETA * SIGMA^2 / 2
        @test isapprox(γ(peak), 1.0; atol=1e-15)
        @test isapprox(γ(peak - 1.0), 1.0; atol=1e-15)
        ω = fine_grid_centered(peak; halfwidth=2.0)
        @test maximum(γ.(ω)) <= 1.0 + 1e-15
        @test pick_gamma_sup(cfg) == 1.0
    end

    @testset "KMS smooth Metropolis (s=0.25, a=0) — locked thesis fixture" begin
        cfg = Config(
            sim = Lindbladian(), domain = BohrDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = true,
            beta = BETA, sigma = SIGMA, a = 0.0, s = 0.25,
            num_energy_bits = 8, w0 = 0.05,
        )
        γ = pick_transition(cfg)
        # Smooth Metropolis sup is attained as ω → -∞; on any finite grid the
        # sample sup is strictly less than 1, but extending the grid pulls
        # it toward 1. This is exactly the discretisation residue the fix
        # eliminates from `gamma_norm_factor`. The closed-form factor `exp(4·sqrtA·sqrtB)`
        # in the smoothing term overflows for |ω| ≳ 70, so we evaluate at a
        # moderate ω = -10 where γ is already within 1e-4 of the analytical sup.
        ω_grid = collect(range(-10.0, 0.0; length=N_FINE))
        sup_grid = maximum(γ.(ω_grid))
        @test sup_grid <= 1.0 + 1e-12
        @test sup_grid >= 0.999
        @test pick_gamma_sup(cfg) == 1.0
    end

    @testset "GNS Gaussian / kinky / smooth all give pick_gamma_sup = 1.0" begin
        cases = [
            (false, nothing, nothing, (GNS_OMEGA_G, SIGMA_G), -GNS_OMEGA_G),  # Gaussian
            (true,  0.0,     0.0,     (nothing, nothing),   0.0),             # kinky
            (true,  0.0,     0.25,    (nothing, nothing),   nothing),         # smooth
        ]
        for (with_lc, a, s, gp, peak) in cases
            cfg = Config(
                sim = Lindbladian(), domain = BohrDomain(), construction = GNS(),
                num_qubits = 3, with_linear_combination = with_lc,
                beta = BETA, sigma = SIGMA,
                gaussian_parameters = gp,
                a = a, s = s,
                num_energy_bits = 8, w0 = 0.05,
            )
            @test pick_gamma_sup(cfg) == 1.0
            γ = pick_transition(cfg)
            if peak !== nothing
                @test isapprox(γ(peak), 1.0; atol=1e-15)
            else
                # smooth Metro GNS — sup approached as ω → -∞; evaluate at a
                # moderate ω where γ is within 1e-3 of the sup but the
                # `exp(4·sqrtA·sqrtB)` factor has not overflowed.
                @test γ(-10.0) >= 0.999
            end
            # Sample sup never exceeds 1 on a moderate grid.
            ω = collect(range(-10.0, 5.0; length=N_FINE))
            @test maximum(γ.(ω)) <= 1.0 + 1e-12
        end
    end
end

# ---------------------------------------------------------------------------
# `_precompute_data` populates `gamma_norm_factor = 1.0` for every
# CKG branch — i.e. construction is grid-independent.
# ---------------------------------------------------------------------------
@testset "_precompute_data has grid-independent gamma_norm_factor" begin
    n = 3
    ham_path = test_hamiltonian_path(n)
    ham = QuantumFurnace._load_hamiltonian_bson(ham_path, 10.0)

    register_pairs = [(8, 0.05), (10, 0.025)]
    cases = [
        (false, nothing, nothing, (1.3, 0.5), "Gaussian"),
        (true,  0.0,     0.0,     (nothing, nothing), "kinky Metro"),
        (true,  0.0,     0.25,    (nothing, nothing), "smooth Metro a=0 s=0.25"),
    ]

    for (r_D, w0_D) in register_pairs, (with_lc, a, s, gp, label) in cases
        cfg = Config(
            sim = Lindbladian(), domain = BohrDomain(), construction = KMS(),
            num_qubits = n, with_linear_combination = with_lc,
            beta = 10.0, sigma = 0.1,
            gaussian_parameters = gp,
            a = a, s = s,
            num_energy_bits = r_D, w0 = w0_D,
        )
        pd = QuantumFurnace._precompute_data(cfg, ham)
        @test pd.gamma_norm_factor ≈ 1.0 atol=1e-15
    end
end

# ---------------------------------------------------------------------------
# BohrDomain `construct_lindbladian` is byte-identical across
# `(r_D, w0_D)` register choices. Pre-fix, the two builds disagreed by the
# ratio of their respective `1.0 / maximum(transition.(...))` samples.
# ---------------------------------------------------------------------------
@testset "BohrDomain construct_lindbladian register invariance" begin
    n = 3
    ham_path = test_hamiltonian_path(n)
    ham = QuantumFurnace._load_hamiltonian_bson(ham_path, 10.0)

    jump_paulis = [[X], [Y], [Z]]
    jump_norm = sqrt(length(jump_paulis) * n)
    jumps = JumpOp[]
    for pauli in jump_paulis, site in 1:n
        op = Matrix(pad_term(pauli, n, site)) ./ jump_norm
        op_eb = ham.eigvecs' * op * ham.eigvecs
        push!(jumps, JumpOp(op, op_eb, op == transpose(op), op == op'))
    end

    cases = [
        (false, nothing, nothing, (1.3, 0.5), "Gaussian"),
        (true,  0.0,     0.0,     (nothing, nothing), "kinky Metro"),
        (true,  0.0,     0.25,    (nothing, nothing), "smooth Metro"),
    ]
    for (with_lc, a, s, gp, label) in cases
        cfg_a = Config(
            sim = Lindbladian(), domain = BohrDomain(), construction = KMS(),
            num_qubits = n, with_linear_combination = with_lc,
            beta = 10.0, sigma = 0.1,
            gaussian_parameters = gp, a = a, s = s,
            num_energy_bits = 8, w0 = 0.05,
        )
        cfg_b = Config(
            sim = Lindbladian(), domain = BohrDomain(), construction = KMS(),
            num_qubits = n, with_linear_combination = with_lc,
            beta = 10.0, sigma = 0.1,
            gaussian_parameters = gp, a = a, s = s,
            num_energy_bits = 10, w0 = 0.025,
        )
        L_a = construct_lindbladian(jumps, cfg_a, ham)
        L_b = construct_lindbladian(jumps, cfg_b, ham)
        @test isapprox(L_a, L_b; atol=1e-13, rtol=1e-13)
    end
end

# ---------------------------------------------------------------------------
# BohrDomain and EnergyDomain use the same continuum normalization.
# Their difference measures EnergyDomain quadrature error. Both evaluate
# the coherent term with B_bohr on the exact Bohr frequencies.
# ---------------------------------------------------------------------------
@testset "BohrDomain ↔ EnergyDomain agreement" begin
    n = 3
    ham_path = test_hamiltonian_path(n)
    ham = QuantumFurnace._load_hamiltonian_bson(ham_path, 10.0)

    jump_paulis = [[X], [Y], [Z]]
    jump_norm = sqrt(length(jump_paulis) * n)
    jumps = JumpOp[]
    for pauli in jump_paulis, site in 1:n
        op = Matrix(pad_term(pauli, n, site)) ./ jump_norm
        op_eb = ham.eigvecs' * op * ham.eigvecs
        push!(jumps, JumpOp(op, op_eb, op == transpose(op), op == op'))
    end

    # ω-range from the unified principle (`scripts/scratch_dissipative_quadrature.jl`):
    # ω_max = ‖H‖ + 8σ; full range = 2·ω_max.
    # Smooth Metro and Gaussian are already at machine precision at r=12.
    R_REF = 12
    omega_range = 2.0 * (opnorm(ham.data) + 8 * 0.1)
    w0_ref = omega_range / 2^R_REF

    cases = [
        (false, nothing, nothing, (1.3, 0.5), "KMS Gaussian", 1e-12),
        (true,  0.0,     0.25,    (nothing, nothing), "KMS smooth Metro", 1e-12),
    ]
    for (with_lc, a, s, gp, label, tol) in cases
        cfg_bohr = Config(
            sim = Lindbladian(), domain = BohrDomain(), construction = KMS(),
            num_qubits = n, with_linear_combination = with_lc,
            beta = 10.0, sigma = 0.1,
            gaussian_parameters = gp, a = a, s = s,
            num_energy_bits = R_REF, w0 = w0_ref,
        )
        cfg_eng = Config(
            sim = Lindbladian(), domain = EnergyDomain(), construction = KMS(),
            num_qubits = n, with_linear_combination = with_lc,
            beta = 10.0, sigma = 0.1,
            gaussian_parameters = gp, a = a, s = s,
            num_energy_bits = R_REF, w0 = w0_ref,
        )
        L_bohr = construct_lindbladian(jumps, cfg_bohr, ham)
        L_eng  = construct_lindbladian(jumps, cfg_eng,  ham)
        diff_op = opnorm(L_eng - L_bohr)
        @test diff_op <= tol
    end

    # The nonsmooth Metropolis kernel converges more slowly. Demonstrate
    # controllability instead of accepting a loose one-point residual.
    kinky_errors = Float64[]
    for r in (12, 14, 16)
        w0 = omega_range / 2^r
        cfg_b = Config(
            sim=Lindbladian(), domain=BohrDomain(), construction=KMS(),
            num_qubits=n, with_linear_combination=true,
            beta=10.0, sigma=0.1, a=0.0, s=0.0,
            num_energy_bits=r, w0=w0,
        )
        cfg_e = Config(
            sim=Lindbladian(), domain=EnergyDomain(), construction=KMS(),
            num_qubits=n, with_linear_combination=true,
            beta=10.0, sigma=0.1, a=0.0, s=0.0,
            num_energy_bits=r, w0=w0,
        )
        push!(kinky_errors, opnorm(
            construct_lindbladian(jumps, cfg_e, ham) -
            construct_lindbladian(jumps, cfg_b, ham)))
    end
    @test all(diff(kinky_errors) .< 0)
    @test kinky_errors[end] <= 1e-9
end

# ---------------------------------------------------------------------------
# Krylov route — `apply_lindbladian!` matvec parity vs the dense
# `construct_lindbladian * vec(ρ)` (already a property of the codebase, but
# we re-verify it survives the gnf fix), and `krylov_spectral_gap`
# register-invariance for BohrDomain.
# ---------------------------------------------------------------------------
@testset "Krylov route invariance" begin
    n = 3
    ham_path = test_hamiltonian_path(n)
    ham = QuantumFurnace._load_hamiltonian_bson(ham_path, 10.0)

    jump_paulis = [[X], [Y], [Z]]
    jump_norm = sqrt(length(jump_paulis) * n)
    jumps = JumpOp[]
    for pauli in jump_paulis, site in 1:n
        op = Matrix(pad_term(pauli, n, site)) ./ jump_norm
        op_eb = ham.eigvecs' * op * ham.eigvecs
        push!(jumps, JumpOp(op, op_eb, op == transpose(op), op == op'))
    end

    @testset "apply_lindbladian! matvec parity vs construct_lindbladian (BohrDomain, KMS smooth Metro)" begin
        cfg = Config(
            sim = Lindbladian(), domain = BohrDomain(), construction = KMS(),
            num_qubits = n, with_linear_combination = true,
            beta = 10.0, sigma = 0.1, a = 0.0, s = 0.25,
            num_energy_bits = 10, w0 = 0.05,
        )
        L = construct_lindbladian(jumps, cfg, ham)
        dim = size(ham.data, 1)
        ws = Workspace(cfg, ham, jumps)
        # Random Hermitian, trace-normalised input ρ
        rng_seed = 4242
        ρ_init = (m = randn(QuantumFurnace.MersenneTwister(rng_seed), ComplexF64, dim, dim);
                   QuantumFurnace.hermitianize!(m); m ./= tr(m); m)
        out_dense = reshape(L * vec(ρ_init), dim, dim)
        apply_lindbladian!(ws, ρ_init, cfg, ham)
        out_krylov = copy(ws.scratch.rho_out)
        @test isapprox(out_krylov, out_dense; atol=1e-12, rtol=1e-12)
    end

    @testset "krylov_spectral_gap register invariance (BohrDomain, KMS smooth Metro)" begin
        cfg_a = Config(
            sim = Lindbladian(), domain = BohrDomain(), construction = KMS(),
            num_qubits = n, with_linear_combination = true,
            beta = 10.0, sigma = 0.1, a = 0.0, s = 0.25,
            num_energy_bits = 8, w0 = 0.05,
        )
        cfg_b = Config(
            sim = Lindbladian(), domain = BohrDomain(), construction = KMS(),
            num_qubits = n, with_linear_combination = true,
            beta = 10.0, sigma = 0.1, a = 0.0, s = 0.25,
            num_energy_bits = 10, w0 = 0.025,
        )
        res_a = krylov_spectral_gap(cfg_a, ham, jumps; krylovdim=20, tol=1e-12)
        res_b = krylov_spectral_gap(cfg_b, ham, jumps; krylovdim=20, tol=1e-12)
        # BohrDomain has no register-grid dependence; spectral gaps
        # must agree to machine precision.
        @test isapprox(res_a.spectral_gap, res_b.spectral_gap; atol=1e-10, rtol=1e-10)
    end
end

# ---------------------------------------------------------------------------
# Simulator routes (run_thermalize, predict_channel_trajectory)
# register invariance for BohrDomain. Pre-fix, two BohrDomain configs at
# different `(r_D, w0_D)` produced different `gamma_norm_factor` values,
# which fed into `jump_weight_scaling` in the trajectory simulator and
# polluted otherwise grid-independent results. Post-fix the two register
# choices must produce byte-identical final ρ trajectories.
# ---------------------------------------------------------------------------
@testset "Simulator routes register invariance" begin
    n = 3
    ham_path = test_hamiltonian_path(n)
    ham = QuantumFurnace._load_hamiltonian_bson(ham_path, 10.0)

    jump_paulis = [[X], [Y], [Z]]
    jump_norm = sqrt(length(jump_paulis) * n)
    jumps = JumpOp[]
    for pauli in jump_paulis, site in 1:n
        op = Matrix(pad_term(pauli, n, site)) ./ jump_norm
        op_eb = ham.eigvecs' * op * ham.eigvecs
        push!(jumps, JumpOp(op, op_eb, op == transpose(op), op == op'))
    end

    @testset "run_thermalize register invariance (BohrDomain, KMS smooth Metro)" begin
        delta = 0.01
        mixing_time = 0.05
        cfg_a = Config(
            sim = Thermalize(), domain = BohrDomain(), construction = KMS(),
            num_qubits = n, with_linear_combination = true,
            beta = 10.0, sigma = 0.1, a = 0.0, s = 0.25,
            num_energy_bits = 8, w0 = 0.05,
            mixing_time = mixing_time, delta = delta,
        )
        cfg_b = Config(
            sim = Thermalize(), domain = BohrDomain(), construction = KMS(),
            num_qubits = n, with_linear_combination = true,
            beta = 10.0, sigma = 0.1, a = 0.0, s = 0.25,
            num_energy_bits = 10, w0 = 0.025,
            mixing_time = mixing_time, delta = delta,
        )
        res_a = run_thermalize(jumps, cfg_a, ham)
        res_b = run_thermalize(jumps, cfg_b, ham)
        @test isapprox(res_a.final_dm, res_b.final_dm; atol=1e-12, rtol=1e-12)
        @test isapprox(res_a.trace_distances, res_b.trace_distances; atol=1e-10, rtol=1e-10)
    end

    @testset "predict_channel_trajectory register invariance (BohrDomain, KMS smooth Metro)" begin
        delta = 0.01
        cfg_a = Config(
            sim = Thermalize(), domain = BohrDomain(), construction = KMS(),
            num_qubits = n, with_linear_combination = true,
            beta = 10.0, sigma = 0.1, a = 0.0, s = 0.25,
            num_energy_bits = 8, w0 = 0.05,
            mixing_time = 0.5, delta = delta,
        )
        cfg_b = Config(
            sim = Thermalize(), domain = BohrDomain(), construction = KMS(),
            num_qubits = n, with_linear_combination = true,
            beta = 10.0, sigma = 0.1, a = 0.0, s = 0.25,
            num_energy_bits = 10, w0 = 0.025,
            mixing_time = 0.5, delta = delta,
        )
        rho0 = Matrix{ComplexF64}(I(2^n) / 2^n)
        k_grid = collect(0:5:50)
        res_a = predict_channel_trajectory(cfg_a, ham, jumps, rho0, k_grid; krylovdim=20, tol=1e-12)
        res_b = predict_channel_trajectory(cfg_b, ham, jumps, rho0, k_grid; krylovdim=20, tol=1e-12)
        # Distances along the trajectory must match (trace distance to ρ_inf).
        @test isapprox(res_a.distances, res_b.distances; atol=1e-10, rtol=1e-10)
        @test isapprox(res_a.spectral_gap, res_b.spectral_gap; atol=1e-10, rtol=1e-10)
    end
end

# ---------------------------------------------------------------------------
# GQSP block-encoding invariant `‖B_a / α_a‖_op ≤ 1`.
# Both `‖B_a‖_op` and `α_a` inherit the same scalar normalization.
# Check the bound across filter families and polynomial degrees.
# ---------------------------------------------------------------------------
@testset "GQSP α_be block-encoding invariant" begin
    n = 3
    ham_path = test_hamiltonian_path(n)
    ham = QuantumFurnace._load_hamiltonian_bson(ham_path, 10.0)

    jump_paulis = [[X], [Y], [Z]]
    jump_norm = sqrt(length(jump_paulis) * n)
    jumps = JumpOp[]
    for pauli in jump_paulis, site in 1:n
        op = Matrix(pad_term(pauli, n, site)) ./ jump_norm
        op_eb = ham.eigvecs' * op * ham.eigvecs
        push!(jumps, JumpOp(op, op_eb, op == transpose(op), op == op'))
    end

    R = 10
    w0 = 0.05
    t0 = 2π / (2^R * w0)

    # Three KMS filter families on TimeDomain (the GQSP path).
    cases = [
        (false, nothing, nothing, (1.3, 0.5), "KMS Gaussian"),
        (true,  0.0,     0.0,     (nothing, nothing), "KMS kinky Metro"),
        (true,  0.0,     0.25,    (nothing, nothing), "KMS smooth Metro"),
    ]
    for (with_lc, a, s, gp, label) in cases
        for d_gqsp in [1, 2, 4]
            cfg = Config(
                sim = Thermalize(), domain = TimeDomain(), construction = KMS(),
                num_qubits = n, with_linear_combination = with_lc,
                beta = 10.0, sigma = 0.1, a = a, s = s,
                gaussian_parameters = gp,
                num_energy_bits = R, w0 = w0, t0 = t0,
                eta = 1e-3,
                mixing_time = 0.01, delta = 0.01,
                with_gqsp = true, gqsp_degree = d_gqsp,
            )
            pd = QuantumFurnace._precompute_data(cfg, ham)
            t0_outer = register_t0_b_minus(cfg)
            t0_inner = register_t0_b_plus(cfg)
            for jump in jumps
                B = QuantumFurnace.B_time([jump], ham, pd.b_minus, pd.b_plus,
                    t0_outer, t0_inner, cfg.beta, cfg.sigma)
                rmul!(B, pd.gamma_norm_factor)
                α_be = QuantumFurnace._gqsp_block_encoding_alpha(jump,
                    pd.b_minus, pd.b_plus, t0_outer, t0_inner, pd.gamma_norm_factor)
                @test opnorm(B) / α_be <= 1.0 + 1e-10
            end
        end
    end
end

# ---------------------------------------------------------------------------
# validate_config! enforces the (a, s) Metropolis taxonomy.
# Kinky Metropolis is exactly (s = 0, a = 0); smooth Metropolis is
# (s > 0, any a ≥ 0). The (s = 0, a > 0) combination is rejected.
# ---------------------------------------------------------------------------
@testset "validate_config! (a, s) Metropolis taxonomy" begin
    function _taxonomy_cfg(; a, s, construction=KMS(), domain=BohrDomain())
        Config(
            sim = Lindbladian(), domain = domain, construction = construction,
            num_qubits = 3, with_linear_combination = true,
            beta = 10.0, sigma = 0.1, a = a, s = s,
            num_energy_bits = 8, w0 = 0.05,
        )
    end
    # Allowed: kinky (s = a = 0)
    validate_config!(_taxonomy_cfg(a = 0.0, s = 0.0))
    # Allowed: smooth with a = 0, s > 0 (thesis-default branch)
    validate_config!(_taxonomy_cfg(a = 0.0, s = 0.25))
    # Allowed: smooth with a > 0, s > 0
    validate_config!(_taxonomy_cfg(a = 0.333, s = 0.4))
    # Rejected: a-regularised but unsmoothed (s = 0, a > 0)
    @test_throws ArgumentError validate_config!(_taxonomy_cfg(a = 1.0, s = 0.0))
    # Same rejection on GNS construction
    @test_throws ArgumentError validate_config!(_taxonomy_cfg(a = 1.0, s = 0.0, construction = GNS()))
end
