"""
Tests for the TrotterDomain Trotter caches.

The canonical KMS coherent scheme is the `TrotterTriple` (three
independent per-leg Strang caches). The single-cache `TrottTrott` is the
dissipator-only (GNS) cache and the per-leg building block of `TrotterTriple`.

Coverage:
- Single-cache `TrottTrott` constructor basics + single-cache `B_trotter`
  behaviour (both coherent legs share one `t0`).
- `make_trotter_for_config` returns a `TrotterTriple` for KMS coherent and a
  single-cache `TrottTrott` for GNS.
- `TrotterTriple` struct invariants, field aliasing to the `D` leg, builder
  validation, and `B_trotter` slope -2 convergence to the `B_time` reference
  (including σ ≠ 1/β).
"""

using Test
using LinearAlgebra
using QuantumFurnace
using QuantumFurnace: B_trotter, B_time, _compute_b_minus, _compute_b_plus,
    _compute_b_plus_metro, _compute_truncated_func

# test_helpers.jl is included once at the top of runtests.jl; the fixtures
# (BETA, SIGMA, N3_HAM, N3_TROTTER, etc.) are already in scope.

# ---------------------------------------------------------------------------
# Helpers (test-local).
# ---------------------------------------------------------------------------

function _make_b_dicts(beta::Real, sigma::Real; r_b::Int = 10,
                       T_minus::Real = 10.0, T_plus::Real = 5.0,
                       eta::Real = 1e-3)
    N = 2^r_b
    t0_minus = 2 * T_minus / N
    t0_plus  = 2 * T_plus  / N
    grid_minus = collect(-(N÷2):((N÷2)-1)) .* t0_minus
    grid_plus  = collect(-(N÷2):((N÷2)-1)) .* t0_plus
    b_minus = _compute_truncated_func(_compute_b_minus, grid_minus, beta, sigma)
    b_plus  = _compute_truncated_func(_compute_b_plus_metro, grid_plus, beta, sigma, eta, 0.25)
    return (; b_minus, b_plus, t0_minus, t0_plus)
end

function _build_jumps_in_basis(ham, basis_eigvecs, n)
    jumps = JumpOp[]
    for pauli in [[X], [Y], [Z]], site in 1:n
        op = Matrix(pad_term(pauli, n, site)) ./ sqrt(3 * n)
        op_eb = basis_eigvecs' * op * basis_eigvecs
        push!(jumps, JumpOp(op, op_eb, op == transpose(op), op == op'))
    end
    return jumps
end

@testset "Single-cache TrottTrott" begin
    ham = N3_HAM
    n = 3
    beta = BETA  # from test_helpers.jl, locked at 10.0
    sigma = SIGMA

    @testset "Single-cache constructor basics" begin
        trotter = TrottTrott(ham, 0.5, 4)
        @test trotter isa TrottTrott
        @test trotter isa AbstractTrotter
        @test trotter.t0 ≈ 0.5
        @test trotter.num_trotter_steps_per_t0 == 4
        @test trotter.source_hamiltonian === ham
        @test length(trotter.eigvals_t0) == 8
        # Strang eigenbasis is orthonormal and eigenvalues sit on the unit circle.
        @test isapprox(trotter.eigvecs * trotter.eigvecs', I; atol = 1e-10)
        @test all(abs.(abs.(trotter.eigvals_t0) .- 1) .< 1e-10)

        # Public error helper agrees with independently materialised unitaries.
        t = trotter.t0
        U_exact = exp(1im .* Matrix(ham.data) .* t)
        U_trotter = trotter.eigvecs * Diagonal(trotter.eigvals_t0) * trotter.eigvecs'
        @test compute_trotter_error(ham, trotter, t) ≈ norm(U_exact - U_trotter) atol=1e-12
    end

    @testset "General Strang products reverse every noncommuting factor" begin
        base_ham = HamHam(
            Vector{Matrix{ComplexF64}}[[X], [Z]], [0.7, 1.1], 1, 1.0;
            periodic=false)
        disorder_ham = HamHam(
            Vector{Vector{Matrix{ComplexF64}}}(), Float64[],
            Vector{Matrix{ComplexF64}}[[X], [Z]], [[0.7], [1.1]], 1, 1.0;
            periodic=false)
        P = (X + Z) / sqrt(2)
        two_site_ham = HamHam(
            Vector{Matrix{ComplexF64}}[[X, X], [P, P]], [0.7, 1.1], 2, 1.0;
            periodic=false)
        Ms = (10, 20, 40, 80, 160, 320, 640, 1280)

        for test_ham in (base_ham, disorder_ham, two_site_ham)
            exact = exp(1im * Matrix(test_ham.data))
            errors = [opnorm(QuantumFurnace._trotterize2(test_ham, 1.0, M) - exact)
                for M in Ms]
            ratios = errors[1:end-1] ./ errors[2:end]
            @test issorted(errors; rev=true)
            @test all(ratio -> 3.8 < ratio < 4.2, ratios)
            @test errors[end] < 1e-9
        end

        grouped = group_hamiltonian_terms(two_site_ham)
        @test grouped.commuting[1] == two_site_ham.base_terms[1:1]
        @test grouped.noncommuting[1] == two_site_ham.base_terms[2:2]
    end

    @testset "make_trotter_for_config returns single-cache TrottTrott for GNS" begin
        cfg_gns = make_config(Lindbladian(), TrotterDomain(); num_qubits=3, construction=GNS())
        trot_gns = make_trotter_for_config(N3_HAM, cfg_gns)
        @test trot_gns isa TrottTrott
        @test !(trot_gns isa TrotterTriple)

        cfg_gns_D = Config(
            sim=Lindbladian(), domain=TrotterDomain(), construction=GNS(),
            num_qubits=3, with_linear_combination=true,
            beta=BETA, sigma=SIGMA, a=BETA / 30, s=0.4,
            num_energy_bits=NUM_ENERGY_BITS, t0=T0, w0=W0,
            num_trotter_steps_per_t0=10,
            num_trotter_steps_per_t0_D=4,
        )
        trot_gns_D = make_trotter_for_config(N3_HAM, cfg_gns_D)
        @test trot_gns_D.num_trotter_steps_per_t0 == 4
        @test Workspace(cfg_gns_D, N3_HAM,
            _build_jumps_in_basis(N3_HAM, trot_gns_D.eigvecs, n);
            trotter=trot_gns_D) isa Workspace
    end

    @testset "Run thermalize with KMS trotter (smoke)" begin
        cfg = make_config(Thermalize(), TrotterDomain(); num_qubits=3, construction=KMS(),
            delta=0.01, mixing_time=0.05)
        trotter = make_trotter_for_config(N3_HAM, cfg)
        @test trotter isa TrotterTriple
        jumps_t = _build_jumps_in_basis(N3_HAM, trotter.eigvecs, n)
        result = run_thermalize(jumps_t, cfg, N3_HAM, trotter)
        @test all(isfinite, result.final_dm)
        @test all(result.trace_distances .>= 0)
        @test result.trace_distances[end] <= result.trace_distances[1] + 1e-8
    end
end

# ============================================================================
# Independent per-leg Trotter caches (TrotterTriple).
# ============================================================================

@testset "TrotterTriple — independent per-leg caches" begin
    ham = N3_HAM
    n = 3
    beta = BETA
    sigma = SIGMA

    @testset "Struct invariants — rotations are unitary; composition consistent" begin
        triple = TrotterTriple(ham, 0.5, 0.3, 0.1, 4, 2, 1)
        @test triple isa AbstractTrotter
        @test triple isa TrotterTriple
        d = size(ham.data, 1)
        # Each per-leg sub-cache is a single-cache TrottTrott.
        for leg in (triple.D, triple.b_minus, triple.b_plus)
            @test leg isa TrottTrott
            @test isapprox(leg.eigvecs * leg.eigvecs', I; atol = 1e-10)
        end
        # Per-leg t0's recover the input.
        @test triple.D.t0       ≈ 0.5
        @test triple.b_minus.t0 ≈ 0.3
        @test triple.b_plus.t0  ≈ 0.1
        # Per-leg M counts.
        @test triple.D.num_trotter_steps_per_t0       == 4
        @test triple.b_minus.num_trotter_steps_per_t0 == 2
        @test triple.b_plus.num_trotter_steps_per_t0  == 1

        # Inter-basis rotations are unitary.
        @test isapprox(triple.R_bm_in_D  * triple.R_bm_in_D',  I; atol = 1e-10)
        @test isapprox(triple.R_bp_in_D  * triple.R_bp_in_D',  I; atol = 1e-10)
        @test isapprox(triple.R_bm_in_bp * triple.R_bm_in_bp', I; atol = 1e-10)
        # Composition: R_bm_in_D = V_bm' V_D = (V_bm' V_bp)(V_bp' V_D) = R_bm_in_bp · R_bp_in_D.
        @test isapprox(triple.R_bm_in_D, triple.R_bm_in_bp * triple.R_bp_in_D; atol = 1e-10)
    end

    @testset "All-same legs → rotations are identity to machine precision" begin
        triple = TrotterTriple(ham, 0.5, 0.5, 0.5, 4, 4, 4)
        d = size(ham.data, 1)
        @test isapprox(triple.R_bm_in_D,  I(d); atol = 1e-10)
        @test isapprox(triple.R_bp_in_D,  I(d); atol = 1e-10)
        @test isapprox(triple.R_bm_in_bp, I(d); atol = 1e-10)
    end

    @testset "Field aliasing: D-leg fields exposed via getproperty" begin
        triple = TrotterTriple(ham, 0.5, 0.3, 0.1, 4, 2, 1)
        @test triple.t0                       == triple.D.t0
        @test triple.eigvecs                  === triple.D.eigvecs
        @test triple.bohr_freqs               === triple.D.bohr_freqs
        @test triple.eigvals_t0               === triple.D.eigvals_t0
        @test triple.num_trotter_steps_per_t0 == triple.D.num_trotter_steps_per_t0
    end

    @testset "Builder argument validation" begin
        @test_throws ArgumentError TrotterTriple(ham, -0.1, 0.3, 0.1, 1, 1, 1)
        @test_throws ArgumentError TrotterTriple(ham,  0.5, 0.0, 0.1, 1, 1, 1)
        @test_throws ArgumentError TrotterTriple(ham,  0.5, 0.3, 0.1, 0, 1, 1)
        @test_throws ArgumentError TrotterTriple(ham,  0.5, 0.3, 0.1, 1, 0, 1)
        @test_throws ArgumentError TrotterTriple(ham,  0.5, 0.3, 0.1, 1, 1, -2)
    end

    @testset "B_trotter slope -2 in joint M tightening for σ ≠ 1/β" begin
        # The outer b_-(t) loop uses round(t / (σ·t0_step_outer)); the inner
        # b_+(τ) loop uses round(τ·β / t0_step_inner). The two scalings coincide
        # only when σ = 1/β. With σ ≠ 1/β the per-leg natural Trotter steps
        # (t0_bm/σ, β·t0_bp) keep both step counts integer, so the Trotter error
        # must still fall as M^{-2}. Use σ = 0.05 = 1/(2β) at β=10.
        sigma_off = 0.05  # ≠ 1/β = 0.1
        r_D = 8; r_b = 10
        T_minus = 10.0; T_plus = 5.0
        eta = 1e-3
        w0_D = pi / (5 * beta)
        t0_D = 2pi / (2^r_D * w0_D)
        bd = _make_b_dicts(beta, sigma_off; r_b, T_minus, T_plus, eta)
        t0_bm = bd.t0_minus; t0_bp = bd.t0_plus
        b_minus = bd.b_minus; b_plus = bd.b_plus
        t0_bm_evol = t0_bm / sigma_off
        t0_bp_evol = beta * t0_bp

        jumps_h = _build_jumps_in_basis(ham, ham.eigvecs, n)
        B_ref = B_time(jumps_h, ham, b_minus, b_plus, t0_bm, t0_bp, beta, sigma_off)

        errs = Float64[]
        for M in (1, 2, 4, 8, 16, 32, 64)
            triple = TrotterTriple(ham, t0_D, t0_bm_evol, t0_bp_evol, M, M, M)
            jumps_t = _build_jumps_in_basis(ham, triple.eigvecs, n)
            B_t = B_trotter(jumps_t, triple, b_minus, b_plus, t0_bm, t0_bp, beta, sigma_off)
            U = ham.eigvecs' * triple.eigvecs
            push!(errs, opnorm(B_ref - U * B_t * U'))
        end
        @test issorted(errs; rev = true)
        @test all(errs[k] / errs[k+1] > 3 for k in 1:length(errs)-1)
        @test errs[end] < 1e-9
    end

    @testset "Slope -2 in joint M tightening (Strang 2nd order)" begin
        # When all three per-leg M counts grow together, the Trotter error
        # should drop as M^{-2} (Strang 2nd order) in the independent-cache
        # TrotterTriple scheme.
        r_D = 8; r_b = 10
        T_minus = 10.0; T_plus = 5.0
        eta = 1e-3
        w0_D = pi / (5 * beta)
        t0_D = 2pi / (2^r_D * w0_D)
        bd = _make_b_dicts(beta, sigma; r_b, T_minus, T_plus, eta)
        t0_bm = bd.t0_minus; t0_bp = bd.t0_plus
        b_minus = bd.b_minus; b_plus = bd.b_plus
        t0_bm_evol = t0_bm / sigma
        t0_bp_evol = beta * t0_bp

        jumps_h = _build_jumps_in_basis(ham, ham.eigvecs, n)
        B_ref   = B_time(jumps_h, ham, b_minus, b_plus, t0_bm, t0_bp, beta, sigma)

        errs = Float64[]
        for M in (1, 2, 4, 8, 16)
            triple = TrotterTriple(ham, t0_D, t0_bm_evol, t0_bp_evol, M, M, M)
            jumps_t = _build_jumps_in_basis(ham, triple.eigvecs, n)
            B_t = B_trotter(jumps_t, triple, b_minus, b_plus, t0_bm, t0_bp, beta, sigma)
            U = ham.eigvecs' * triple.eigvecs
            push!(errs, opnorm(B_ref - U * B_t * U'))
        end
        @test issorted(errs; rev = true)
        # Each doubling of M should give roughly 4× improvement (Strang -2).
        @test all(errs[k] / errs[k+1] > 3 for k in 1:length(errs)-1)
        @test errs[end] < 1e-9
    end

    @testset "Independent leg-knob tightening (M_b_minus only)" begin
        # Hold M_D = M_b_plus = 1 (coarse) and tighten M_b_minus. The error
        # should fall as the outer-leg substep shrinks, independently of the
        # other legs (the three TrotterTriple legs are fully independent).
        r_D = 8; r_b = 10
        T_minus = 10.0; T_plus = 5.0
        eta = 1e-3
        w0_D = pi / (5 * beta)
        t0_D = 2pi / (2^r_D * w0_D)
        bd = _make_b_dicts(beta, sigma; r_b, T_minus, T_plus, eta)
        t0_bm = bd.t0_minus; t0_bp = bd.t0_plus
        b_minus = bd.b_minus; b_plus = bd.b_plus
        t0_bm_evol = t0_bm / sigma
        t0_bp_evol = beta * t0_bp

        jumps_h = _build_jumps_in_basis(ham, ham.eigvecs, n)
        B_ref   = B_time(jumps_h, ham, b_minus, b_plus, t0_bm, t0_bp, beta, sigma)

        errs = Float64[]
        for M_bm in (1, 2, 4, 8, 16)
            triple = TrotterTriple(ham, t0_D, t0_bm_evol, t0_bp_evol, 1, M_bm, 8)
            jumps_t = _build_jumps_in_basis(ham, triple.eigvecs, n)
            B_t = B_trotter(jumps_t, triple, b_minus, b_plus, t0_bm, t0_bp, beta, sigma)
            U = ham.eigvecs' * triple.eigvecs
            push!(errs, opnorm(B_ref - U * B_t * U'))
        end
        # Errors must strictly decrease as M_b_minus grows.
        @test issorted(errs; rev = true)
        # Aggressive improvement at large M_b_minus.
        @test errs[end] / errs[1] < 0.01
    end

    @testset "make_trotter_for_config returns TrotterTriple for KMS+Trotter" begin
        cfg_kms = make_config(Lindbladian(), TrotterDomain(); num_qubits=3, construction=KMS())
        trotter_kms = make_trotter_for_config(N3_HAM, cfg_kms)
        @test trotter_kms isa TrotterTriple
        @test trotter_kms isa AbstractTrotter
        # Legacy num_trotter_steps_per_t0 = 10 → all three legs at M=10.
        @test trotter_kms.D.num_trotter_steps_per_t0       == 10
        @test trotter_kms.b_minus.num_trotter_steps_per_t0 == 10
        @test trotter_kms.b_plus.num_trotter_steps_per_t0  == 10

        # GNS branch still returns legacy single-cache TrottTrott.
        cfg_gns = make_config(Lindbladian(), TrotterDomain(); num_qubits=3, construction=GNS())
        trotter_gns = make_trotter_for_config(N3_HAM, cfg_gns)
        @test trotter_gns isa TrottTrott
        @test !(trotter_gns isa TrotterTriple)
    end

    @testset "Per-leg M_user fields in Config + accessors" begin
        cfg = Config(
            sim = Lindbladian(),
            domain = TrotterDomain(),
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = BETA,
            sigma = SIGMA,
            s = 0.4,
            a = BETA / 30.0,
            num_energy_bits = NUM_ENERGY_BITS,
            w0 = W0,
            t0 = T0,
            num_trotter_steps_per_t0 = 4,
            num_trotter_steps_per_t0_b_minus = 12,
        )
        # Per-leg M_b_minus overrides legacy; the other two fall back to legacy.
        @test register_M_D(cfg)        == 4
        @test register_M_b_minus(cfg)  == 12
        @test register_M_b_plus(cfg)   == 4

        trotter = make_trotter_for_config(N3_HAM, cfg)
        @test trotter isa TrotterTriple
        @test trotter.D.num_trotter_steps_per_t0       == 4
        @test trotter.b_minus.num_trotter_steps_per_t0 == 12
        @test trotter.b_plus.num_trotter_steps_per_t0  == 4
    end

    @testset "construct_lindbladian with TrotterTriple (smoke)" begin
        cfg = make_config(Lindbladian(), TrotterDomain(); num_qubits=3,
            construction=KMS(), num_trotter_steps_per_t0=40)
        trotter = make_trotter_for_config(N3_HAM, cfg)
        @test trotter isa TrotterTriple
        jumps = _build_jumps_in_basis(N3_HAM, trotter.eigvecs, n)
        L = construct_lindbladian(jumps, cfg, N3_HAM; trotter=trotter)
        @test size(L) == (64, 64)
        # σ_β in V_D
        sigma_beta = trotter.eigvecs' * N3_HAM.eigvecs * N3_HAM.gibbs *
                     N3_HAM.eigvecs' * trotter.eigvecs
        @test norm(L * vec(sigma_beta)) < 1e-9

        # Caches are tied to their construction Hamiltonian and register data.
        cfg_wrong_M = make_config(Lindbladian(), TrotterDomain(); num_qubits=3,
            construction=KMS(), num_trotter_steps_per_t0=41)
        @test_throws ArgumentError construct_lindbladian(
            jumps, cfg_wrong_M, N3_HAM; trotter=trotter)
        @test_throws ArgumentError Workspace(
            cfg_wrong_M, N3_HAM, jumps; trotter=trotter)

        legacy_single = TrottTrott(N3_HAM, T0, NUM_TROTTER_STEPS_PER_T0)
        cfg_single_compatible = make_config(
            Lindbladian(), TrotterDomain(); num_qubits=3, construction=KMS())
        @test QuantumFurnace._validate_trotter_cache!(
            cfg_single_compatible, N3_HAM, legacy_single) === nothing
        cfg_single_incompatible = Config(;
            sim=Lindbladian(), domain=TrotterDomain(), construction=KMS(),
            num_qubits=3, with_linear_combination=true,
            beta=BETA, sigma=SIGMA, a=BETA / 30, s=0.4,
            num_energy_bits=NUM_ENERGY_BITS, t0=T0, w0=W0,
            num_trotter_steps_per_t0=NUM_TROTTER_STEPS_PER_T0,
            num_trotter_steps_per_t0_b_minus=77,
            num_trotter_steps_per_t0_b_plus=88,
        )
        @test_throws ArgumentError QuantumFurnace._validate_trotter_cache!(
            cfg_single_incompatible, N3_HAM, legacy_single)

        other = make_classical_ising_n3()
        cfg_other = Config(
            sim=Lindbladian(), domain=TrotterDomain(), construction=GNS(),
            num_qubits=3, with_linear_combination=true,
            beta=other.beta_alg, beta_phys=other.beta_phys,
            sigma=1 / other.beta_alg, a=0.0, s=0.25,
            num_energy_bits=NUM_ENERGY_BITS, t0=T0, w0=W0,
            num_trotter_steps_per_t0=NUM_TROTTER_STEPS_PER_T0,
        )
        @test_throws ArgumentError Workspace(
            cfg_other, other.ham, N3_TROTTER_JUMPS; trotter=N3_TROTTER)
    end
end
