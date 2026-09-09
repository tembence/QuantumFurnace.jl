using Test
using QuantumFurnace
using LinearAlgebra
using Random
using Printf

# ===========================================================================
# KMS-geometry test suite
# ===========================================================================
#
# Ported from scripts/scratch_kms_geometry.jl. The scratch passes 60/60 at
# machine precision; this file exercises the production library implementation
# (src/kms_geometry.jl) on the same fixtures.
#
# Six fixture/builder helpers live test-side (not library exports) because
# they are config-builders and Davies-toy fixtures specific to this suite.
# Shared `make_dll_n3_system(β)` is reused from test/test_helpers.jl.

# ---------------------------------------------------------------------------
# Fixture/config helpers (test-only, shadow of scratch helpers)
# ---------------------------------------------------------------------------

"""
    make_ckg_n3_system(beta) -> (; ham, jumps, gibbs)

CKG companion to `make_dll_n3_system(beta)`: same Hamiltonian, same jump set,
but used by configs without a DLL filter (`construction = KMS()`).  See
`test/test_helpers.jl::make_dll_n3_system` for the shared physics — the
basis-projected jumps are construction-agnostic.
"""
make_ckg_n3_system(beta::Real) = make_dll_n3_system(beta)

# a=0, s=0.25 matches the CKG smooth-Metropolis defaults in
# `sweep_mixing_times`.
"""
    ckg_smooth_metro_config(beta) -> Config

CKG smooth-Metropolis n=3 EnergyDomain config (thesis-numerics defaults).
"""
function ckg_smooth_metro_config(beta::Real)
    return Config(
        sim = Lindbladian(),
        domain = EnergyDomain(),
        construction = KMS(),
        num_qubits = 3,
        with_linear_combination = true,
        beta = float(beta),
        sigma = 1.0 / float(beta),
        a = 0.0,
        s = 0.25,
        num_energy_bits = 12,
        w0 = 0.05,
        t0 = 2π / (2^12 * 0.05),
        num_trotter_steps_per_t0 = 10,
    )
end

"""
    dll_config(beta, filter) -> Config

DLL BohrDomain n=3 config — Metropolis or Gaussian via the supplied filter.
"""
function dll_config(beta::Real, filter::AbstractFilter)
    return Config(
        sim = Lindbladian(),
        domain = BohrDomain(),
        construction = DLL(),
        num_qubits = 3,
        with_linear_combination = true,
        beta = float(beta),
        sigma = 1.0 / float(beta),
        a = float(beta) / 30.0,
        s = 0.4,
        num_energy_bits = 12,
        w0 = 0.05,
        t0 = 2π / (2^12 * 0.05),
        num_trotter_steps_per_t0 = 10,
        filter = filter,
    )
end

"""
    build_dense_lindbladian(config, ham, jumps) -> (L_super, dim)

Build the dense d²×d² Schrödinger Lindbladian by `apply_lindbladian!`-driven
matvec.  Uses identical sandwich code to `krylov_spectral_gap`, so any
cross-check between a dense routine and the production matrix-free routine
sits on the same physics.
"""
function build_dense_lindbladian(
    config::Config{Lindbladian},
    ham::HamHam,
    jumps::Vector{JumpOp},
)
    dim = size(ham.data, 1)
    ws  = Workspace(config, ham, jumps)
    L_apply! = function (out, X)
        apply_lindbladian!(ws, X, config, ham)
        copyto!(out, ws.scratch.rho_out)
        return out
    end
    L_super = build_dense_superoperator(L_apply!, dim)
    return L_super, dim
end

"""
    dll_lindblad_ops(jumps, ham, filter) -> Vector{Matrix{ComplexF64}}

Convenience: list of per-coupling DLL Lindblad operators in the eigenbasis,
one per jump.  Used by the 1→1 bound on the DLL dissipator.
"""
function dll_lindblad_ops(jumps::Vector{JumpOp}, ham::HamHam, filter::AbstractFilter)
    return [Matrix{ComplexF64}(dll_lindblad_op_bohr(jump, ham, filter)) for jump in jumps]
end

"""
    build_qubit_davies(omega, beta, gamma) -> NamedTuple

Single-qubit Davies thermal generator with `H = -(ω/2) σ_z` and KMS-DB rates
`γ_+ = γ` (de-excitation; ν = -ω) and `γ_- = γ e^{-βω}` (excitation; ν = +ω).
Davies single-qubit eigenvalues:
  `0, -(γ_+ + γ_-), -(γ_+ + γ_-)/2 ± iω`
→ spectral gap (smallest non-zero |Re λ|) = `(γ_+ + γ_-)/2`  (T_2 coherence decay)
→ Λ_max  = `γ_+ + γ_-`                                       (T_1 population decay)
"""
function build_qubit_davies(omega::Real, beta::Real, gamma::Real)
    CT = ComplexF64
    sigma_z = CT[1.0 0.0; 0.0 -1.0]
    sigma_p = CT[0.0 1.0; 0.0 0.0]   # |0⟩⟨1|
    sigma_m = CT[0.0 0.0; 1.0 0.0]   # |1⟩⟨0|
    H       = -(omega / 2) * sigma_z
    Z       = exp(beta * omega / 2) + exp(-beta * omega / 2)
    p0      = exp( beta * omega / 2) / Z
    p1      = exp(-beta * omega / 2) / Z
    sigma_state = Diagonal([p0, p1]) |> Matrix{CT}

    γp = gamma                       # de-excitation rate
    γm = gamma * exp(-beta * omega)  # excitation rate

    L_apply! = function (out, X)
        # Schrödinger Lindbladian: L(X) = -i[H, X]
        #                                + γ_+ (σ_+ X σ_+† - 1/2 {σ_+† σ_+, X})
        #                                + γ_- (σ_- X σ_-† - 1/2 {σ_-† σ_-, X})
        comm = -1im * (H * X - X * H)
        sp_term = γp * (sigma_p * X * sigma_p' .- 0.5 .* (sigma_p' * sigma_p * X .+ X * sigma_p' * sigma_p))
        sm_term = γm * (sigma_m * X * sigma_m' .- 0.5 .* (sigma_m' * sigma_m * X .+ X * sigma_m' * sigma_m))
        copyto!(out, comm .+ sp_term .+ sm_term)
        return out
    end

    L_super = build_dense_superoperator(L_apply!, 2)
    return (; L_super, gibbs = sigma_state, γp, γm, H, sigma_p, sigma_m,
              gap_exact   = (γp + γm) / 2,
              Lambda_exact = γp + γm)
end


# ===========================================================================
# Validation suite — 7 testsets, ported verbatim from scratch
# ===========================================================================

@testset "KMS geometry" begin
    Random.seed!(20260502)

    # -----------------------------------------------------------------------
    # @testset (1) — KMS inner-product algebra
    # -----------------------------------------------------------------------
    @testset "(1) KMS inner-product algebra" begin
        # Use a non-uniform 2-level Gibbs state.
        β = 1.7
        ω = 0.9
        Z = exp(β * ω / 2) + exp(-β * ω / 2)
        sigma = ComplexF64[exp(β*ω/2)/Z 0; 0 exp(-β*ω/2)/Z]
        s12 = sqrt(Hermitian(sigma))

        # Random Hermitian X, Y.
        Xr = randn(ComplexF64, 2, 2);  X = (Xr + Xr') / 2
        Yr = randn(ComplexF64, 2, 2);  Y = (Yr + Yr') / 2

        # Hermitian symmetry: ⟨X, Y⟩ = conj(⟨Y, X⟩)
        @test isapprox(kms_inner_product(X, Y, sigma; sigma_sqrt = s12),
                       conj(kms_inner_product(Y, X, sigma; sigma_sqrt = s12)); atol = 1e-12)

        # Identity-norm: ‖I‖_KMS = √tr(σ) = 1.
        @test isapprox(kms_norm(Matrix{ComplexF64}(I, 2, 2), sigma; sigma_sqrt = s12), 1.0; atol = 1e-12)

        # Identity-variance: Var_KMS(I) = 0.
        @test isapprox(kms_variance(Matrix{ComplexF64}(I, 2, 2), sigma; sigma_sqrt = s12), 0.0; atol = 1e-12)
        @test isapprox(kms_variance(im .* Matrix{ComplexF64}(I, 2, 2), sigma;
                                   sigma_sqrt = s12), 0.0; atol = 1e-12, rtol = 0)

        # Cauchy–Schwarz: |⟨X, Y⟩| ≤ ‖X‖ ‖Y‖.
        cs_lhs = abs(kms_inner_product(X, Y, sigma; sigma_sqrt = s12))
        cs_rhs = kms_norm(X, sigma; sigma_sqrt = s12) * kms_norm(Y, sigma; sigma_sqrt = s12)
        @test cs_lhs ≤ cs_rhs + 1e-12

        # Positivity: ⟨X, X⟩ ≥ 0 (real and non-negative).
        ip_xx = kms_inner_product(X, X, sigma; sigma_sqrt = s12)
        @test isapprox(imag(ip_xx), 0.0; atol = 1e-12)
        @test real(ip_xx) ≥ -1e-12
    end

    # -----------------------------------------------------------------------
    # @testset (2) — Discriminant + KMS-DB witness
    # -----------------------------------------------------------------------
    @testset "(2) Quantum discriminant + KMS-DB diagnostics" begin
        β = 5.0
        sys = make_ckg_n3_system(β)
        cfg = ckg_smooth_metro_config(β)
        L_super, dim = build_dense_lindbladian(cfg, sys.ham, sys.jumps)
        gibbs = Matrix{ComplexF64}(sys.gibbs)

        # Schrödinger fixed-point:  L_super · vec(σ) ≈ 0
        residual_sigma = norm(L_super * vec(gibbs)) / norm(vec(gibbs))
        @test residual_sigma < 1e-10

        # Quantum discriminant D_S = Φ⁻¹ L_S Φ.  For KMS-DBC with the Lamb
        # shift restored, D_S is HS-self-adjoint to numerical precision.
        D_super = materialize_discriminant(L_super, Hermitian(gibbs))
        D_sym   = Hermitian((D_super .+ D_super') ./ 2)
        D_anti  = (D_super .- D_super') ./ 2
        anti_to_sym_ratio = norm(D_anti) / max(norm(D_sym), eps())
        @info "(2) D_S anti-Hermitian / symmetric norm ratio at β=$β" anti_to_sym_ratio
        @test anti_to_sym_ratio < 1e-10   # KMS-DB witness; must be machine-precision

        # D_S has vec(σ^{1/2}) as zero eigenvector.
        v_sh = vec(sqrt(Hermitian(gibbs)))
        @test norm(D_super * v_sh) < 1e-10

        # Eigenvalues of D_sym are real and ≤ 0 (D_S is a similarity transform
        # of L_S so eigenvalues of D_sym match those of L_S which are ≤ 0).
        ev = eigvals(D_sym)
        @test maximum(abs.(imag.(ev))) < 1e-10
        @test maximum(real.(ev)) < 1e-8

        # Diagnostic: removing the Lamb-shift coherent term breaks KMS-DBC
        # (the bare dissipator is *not* KMS-DBC for CKG smooth-Metropolis,
        # contrary to the naive intuition).  Construct the no-coherent variant
        # and confirm:
        #   (a) its discriminant has substantial anti-Hermitian content (non-DBC witness)
        #   (b) Λ_max is preserved at machine precision (dissipator dominates large modes)
        ws_no_coh = Workspace(cfg, sys.ham, sys.jumps)
        # Identity in Workspace constructor: G_left = +i B_T - 0.5 R, G_right = -i B_T - 0.5 R
        # → -(G_left + G_right) = R, so we recover R and rebuild G_left = G_right = -0.5 R.
        R = -(ws_no_coh.G_left .+ ws_no_coh.G_right)
        ws_no_coh.G_left  .= -0.5 .* R
        ws_no_coh.G_right .= -0.5 .* R
        L_apply_no_coh! = function (out, X)
            apply_lindbladian!(ws_no_coh, X, cfg, sys.ham)
            copyto!(out, ws_no_coh.scratch.rho_out)
            return out
        end
        L_super_no_coh = build_dense_superoperator(L_apply_no_coh!, dim)

        # No-coherent discriminant should NOT be HS-self-adjoint (KMS-DBC fails).
        D_nc       = materialize_discriminant(L_super_no_coh, Hermitian(gibbs))
        D_nc_sym   = Hermitian((D_nc .+ D_nc') ./ 2)
        D_nc_anti  = (D_nc .- D_nc') ./ 2
        anti_ratio_nc = norm(D_nc_anti) / max(norm(D_nc_sym), eps())
        @test anti_ratio_nc > 1e-6   # substantial anti-Hermitian piece (KMS-DBC is broken)

        gap_full   = spectral_gap_kms(L_super,        gibbs).gap
        gap_no_coh = spectral_gap_kms(L_super_no_coh, gibbs).gap
        Λ_full     = max_dirichlet_rate_kms(L_super,        gibbs)
        Λ_no_coh   = max_dirichlet_rate_kms(L_super_no_coh, gibbs)

        # Λ_max nearly preserved (dissipator dominates large modes); gap may shift.
        @test isapprox(Λ_full, Λ_no_coh; rtol = 1e-8)

        @info "(2) coherent removal: full KMS-DBC vs bare dissipator (β=$β)" gap_full gap_no_coh Λ_full Λ_no_coh anti_ratio_nc
    end

    # -----------------------------------------------------------------------
    # @testset (3) — Dirichlet form properties
    # -----------------------------------------------------------------------
    @testset "(3) KMS Dirichlet form" begin
        β = 5.0
        sys = make_ckg_n3_system(β)
        cfg = ckg_smooth_metro_config(β)
        L_super, dim = build_dense_lindbladian(cfg, sys.ham, sys.jumps)
        # Heisenberg-picture superoperator = HS-adjoint of Schrödinger.
        L_super_H = Matrix(L_super')
        gibbs = Matrix{ComplexF64}(sys.gibbs)
        s12 = sqrt(Hermitian(gibbs))

        # Random Hermitian X.
        Xraw = randn(ComplexF64, dim, dim)
        X = (Xraw + Xraw') / 2

        # Linearity in L: E_{2L}(X) = 2 E_L(X).
        E_L  = kms_dirichlet_form(L_super_H, X, gibbs; sigma_sqrt = s12)
        E_2L = kms_dirichlet_form(2.0 .* L_super_H, X, gibbs; sigma_sqrt = s12)
        @test isapprox(E_2L, 2 * E_L; rtol = 1e-12)

        # Non-negativity: E_L(X) ≥ 0 for KMS-DBC L on Hermitian X.
        @test E_L ≥ -1e-10

        # Vanishing on constants: E_L(c·I) = 0 — uses unitality of L^H.
        cI = 0.7 * Matrix{ComplexF64}(I, dim, dim)
        E_const = kms_dirichlet_form(L_super_H, cI, gibbs; sigma_sqrt = s12)
        @test isapprox(E_const, 0.0; atol = 1e-10)

        # Cross-check against discriminant route: E_L(X) should equal
        # -Re tr( (ΦX)† D_sym ΦX ) where Φ X = σ^{1/4} X σ^{1/4}.
        s14 = sqrt(s12)
        Φ   = kron(s14, s14)
        D_super = materialize_discriminant(L_super, Hermitian(gibbs))
        D_sym   = Hermitian((D_super .+ D_super') ./ 2)
        ΦX_vec  = Φ * vec(X)
        E_via_D = -real(ΦX_vec' * (D_sym * ΦX_vec))
        @test isapprox(E_via_D, E_L; rtol = 1e-10)

        @info "(3) Dirichlet form sanity at β=$β" E_L E_2L E_const E_via_D
    end

    # -----------------------------------------------------------------------
    # @testset (4) — Spectral gap
    # -----------------------------------------------------------------------
    @testset "(4) Spectral gap λ(L)" begin
        # 4a — qubit Davies closed-form.
        ω = 1.3
        β = 0.8
        γ = 0.4
        davies = build_qubit_davies(ω, β, γ)
        gap_davies_num = spectral_gap_kms(davies.L_super, davies.gibbs).gap
        Λ_davies_num   = max_dirichlet_rate_kms(davies.L_super, davies.gibbs)
        @info "(4a) Davies qubit: numerical vs closed-form" gap_davies_num davies.gap_exact Λ_davies_num davies.Lambda_exact
        @test isapprox(gap_davies_num, davies.gap_exact;    rtol = 1e-10)
        @test isapprox(Λ_davies_num,   davies.Lambda_exact; rtol = 1e-10)

        # 4b — n=3 CKG smooth-Metro vs production krylov_spectral_gap.
        β = 5.0
        sys = make_ckg_n3_system(β)
        cfg = ckg_smooth_metro_config(β)
        L_super, _ = build_dense_lindbladian(cfg, sys.ham, sys.jumps)
        gibbs = Matrix{ComplexF64}(sys.gibbs)

        gap_dense = spectral_gap_kms(L_super, gibbs).gap
        # Production matrix-free.
        kr = krylov_spectral_gap(cfg, sys.ham, sys.jumps; krylovdim = 30,
                                 howmany = 4, tol = 1e-12)
        gap_krylov = kr.spectral_gap
        @info "(4b) n=3 CKG smooth-Metro β=$β: KMS-Poincaré gap vs Krylov gap" gap_dense gap_krylov
        @test isapprox(gap_dense, gap_krylov; rtol = 1e-7)

        # 4c — Schrödinger ↔ Heisenberg (HS-adjoint) similarity sanity.
        ws_h = Workspace(cfg, sys.ham, sys.jumps)
        L_apply_h! = function (out, X)
            apply_adjoint_lindbladian!(ws_h, X, cfg, sys.ham)
            copyto!(out, ws_h.scratch.rho_out)
            return out
        end
        L_super_h = build_dense_superoperator(L_apply_h!, size(sys.ham.data, 1))
        # L_super_h = L_super^† (HS-adjoint) → eigenvalues are conjugates of L_super's
        # → same spectrum since L_super is real-eigenvalued (it's a similarity transform
        # of D_S, which is Hermitian).
        ev_S = sort(real.(eigvals(L_super));   by = abs)
        ev_H = sort(real.(eigvals(L_super_h)); by = abs)
        @info "(4c) Schrödinger ↔ Heisenberg gap" gap_S=abs(ev_S[2]) gap_H=abs(ev_H[2])
        @test isapprox(abs(ev_S[2]), abs(ev_H[2]); rtol = 1e-10)
    end

    # -----------------------------------------------------------------------
    # @testset (5) — Λ_max + intrinsic ratio + scale invariance
    # -----------------------------------------------------------------------
    @testset "(5) Λ_max + intrinsic ratio + scale invariance" begin
        β = 5.0
        sys = make_ckg_n3_system(β)
        cfg = ckg_smooth_metro_config(β)
        L_super, _ = build_dense_lindbladian(cfg, sys.ham, sys.jumps)
        gibbs = Matrix{ComplexF64}(sys.gibbs)

        gap = spectral_gap_kms(L_super, gibbs).gap
        Λ   = max_dirichlet_rate_kms(L_super, gibbs)
        ρ   = intrinsic_mixing_ratio(L_super, gibbs)

        @test Λ ≥ gap - 1e-12
        @test 0.0 < ρ ≤ 1.0 + 1e-12

        # Scale invariance under L → c·L.
        c = 2.7
        gap_c = spectral_gap_kms(c .* L_super, gibbs).gap
        Λ_c   = max_dirichlet_rate_kms(c .* L_super, gibbs)
        ρ_c   = intrinsic_mixing_ratio(c .* L_super, gibbs)

        @test isapprox(gap_c, c * gap; rtol = 1e-10)
        @test isapprox(Λ_c,   c * Λ;   rtol = 1e-10)
        @test isapprox(ρ_c,   ρ;       rtol = 1e-12)

        @info "(5) scale invariance at β=$β" gap Λ ρ gap_c Λ_c ρ_c
    end

    # -----------------------------------------------------------------------
    # @testset (6) — 1→1 norm bound + HS-induced norm
    # -----------------------------------------------------------------------
    @testset "(6) 1→1 norm bound + HS-induced norm" begin
        for β in (5.0, 10.0)
            sys     = make_dll_n3_system(β)
            f_metro = DLLMetropolisFilter(β; S = 2.0)
            cfg     = dll_config(β, f_metro)
            L_super, _ = build_dense_lindbladian(cfg, sys.ham, sys.jumps)
            gibbs   = Matrix{ComplexF64}(sys.gibbs)

            L_a_list = dll_lindblad_ops(sys.jumps, sys.ham, f_metro)
            one_to_one_bound = dissipator_one_to_one_norm_bound(L_a_list)
            hs_norm = hs_operator_norm(L_super)

            gap = spectral_gap_kms(L_super, gibbs).gap

            @test hs_norm > 0
            @test one_to_one_bound > 0

            # Scale invariance of *ratios* gap / one-to-one_bound under L_a → √c L_a.
            c = 1.5
            ratio_orig = gap / one_to_one_bound
            # 1→1 bound scales linearly in c (each L_a → √c L_a → ‖L_a†L_a‖ → c).
            ratio_c = (c * gap) / (c * one_to_one_bound)
            @test isapprox(ratio_orig, ratio_c; rtol = 1e-12)

            @info "(6) DLL Metro β=$β" gap one_to_one_bound hs_norm ratio_orig
        end
    end

    # -----------------------------------------------------------------------
    # @testset (7) — Headline: CKG smooth-Metro vs DLL Metro vs DLL Gauss
    # -----------------------------------------------------------------------
    @testset "(7) headline: CKG vs DLL @ n=3, β ∈ {5,10,20}" begin
        headline = NamedTuple[]   # captured for soft-assertion at end
        for β in (5.0, 10.0, 20.0)
            sys   = make_dll_n3_system(β)   # construction-agnostic ham + jumps + gibbs
            gibbs = Matrix{ComplexF64}(sys.gibbs)

            for (label, cfg, filter_obj) in (
                ("CKG smooth-Metro", ckg_smooth_metro_config(β),         nothing),
                ("DLL Metro",        dll_config(β, DLLMetropolisFilter(β; S = 2.0)),
                                                                          DLLMetropolisFilter(β; S = 2.0)),
                ("DLL Gauss",        dll_config(β, DLLGaussianFilter(β)), DLLGaussianFilter(β)),
            )
                L_super, _ = build_dense_lindbladian(cfg, sys.ham, sys.jumps)

                gap = spectral_gap_kms(L_super, gibbs).gap
                Λ   = max_dirichlet_rate_kms(L_super, gibbs)
                ρ   = intrinsic_mixing_ratio(L_super, gibbs)
                hsn = hs_operator_norm(L_super)

                # 1→1 bound + Tr(α) only meaningful for DLL (where L_a's are
                # explicit and α is per-jump rank-1).  For CKG report NaN.
                if filter_obj !== nothing
                    L_a_list = dll_lindblad_ops(sys.jumps, sys.ham, filter_obj)
                    one2one  = dissipator_one_to_one_norm_bound(L_a_list)
                    tr_alpha = sum(real(tr(L_a' * L_a)) for L_a in L_a_list)
                else
                    one2one  = NaN
                    tr_alpha = NaN
                end

                # Sanity invariants.
                @test gap > 0
                @test Λ ≥ gap - 1e-12
                @test 0 < ρ ≤ 1 + 1e-12

                push!(headline, (; β, label, gap, Λ, ρ, tr_alpha, one2one, hsn))
            end
        end

        # H1 vs H2 sanity: at β=20 the three intrinsic ratios should not be
        # numerically identical, otherwise H2 (diagonal of α drives gap) is
        # falsified before the formal sweep.  Soft assertion only:
        β20 = filter(r -> r.β == 20.0, headline)
        @test length(β20) == 3
        ρs_β20 = [r.ρ for r in β20]
        spread_β20 = (maximum(ρs_β20) - minimum(ρs_β20))
        @info "(7) ρ_intrinsic spread across families at β=20" spread_β20 ρs_β20
    end

    # -----------------------------------------------------------------------
    # @testset (8) — hs_operator_norm_krylov parity
    # -----------------------------------------------------------------------
    # Matrix-free opnorm via Golub–Kahan–Lanczos (KrylovKit.svdsolve) must
    # agree with dense `opnorm` on:
    #   (a) a synthetic random superoperator (no physics, sanity check),
    #   (b) the actual Lindbladian via `apply_lindbladian!` matvec,
    #   (c) the BohrDomain−TimeDomain *difference* superoperator at small
    #       enough r_D that ‖ΔL‖ is well above the matvec noise floor.
    # Operators near the matvec noise floor are excluded here: if numerical
    # cancellation makes the supplied actions fail GKL's adjoint-compatibility
    # check, the wrapper throws rather than returning an uncertified estimate.
    @testset "(8) hs_operator_norm_krylov parity" begin
        Random.seed!(20260506)

        # (a) Synthetic random ComplexF64 superop at d=8 — noise-free path.
        d_synth = 8
        A = randn(ComplexF64, d_synth^2, d_synth^2)
        L_apply_A!  = (out, X) -> (copyto!(out, reshape(A * vec(X), d_synth, d_synth)); out)
        L_apply_A_adj! = (out, X) -> (copyto!(out, reshape(adjoint(A) * vec(X), d_synth, d_synth)); out)
        s_dense  = opnorm(A)
        s_krylov = hs_operator_norm_krylov(L_apply_A!, L_apply_A_adj!, d_synth;
                                            tol = 1e-12, krylovdim = 30)
        @test isapprox(s_dense, s_krylov; rtol = 1e-10)

        # (b) Real CKG smooth-Metro Lindbladian at n=3 — apply_lindbladian!
        # matvec wrapped both densely (build_dense_lindbladian) and via the
        # Krylov closure.  These must agree to machine precision.
        β = 5.0
        sys = make_ckg_n3_system(β)
        cfg = ckg_smooth_metro_config(β)
        L_super, dim_n3 = build_dense_lindbladian(cfg, sys.ham, sys.jumps)
        ws  = Workspace(cfg, sys.ham, sys.jumps)
        L_apply!     = (out, X) -> (apply_lindbladian!(ws, X, cfg, sys.ham);
                                    copyto!(out, ws.scratch.rho_out); out)
        L_apply_adj! = (out, X) -> (apply_adjoint_lindbladian!(ws, X, cfg, sys.ham);
                                    copyto!(out, ws.scratch.rho_out); out)
        s_dense_b  = opnorm(L_super)
        s_krylov_b = hs_operator_norm_krylov(L_apply!, L_apply_adj!, dim_n3;
                                              tol = 1e-12, krylovdim = 30)
        @test isapprox(s_dense_b, s_krylov_b; rtol = 1e-9)
        @info "(8b) full L parity" β s_dense_b s_krylov_b

        # (c) Difference superoperator BohrDomain − TimeDomain (full L,
        # include_coherent=true).  Pick r_D=8 — above the noise floor for
        # smooth-Metropolis at n=3, β=5.
        r_D = 8
        w0_D = π / (5 * β)
        t0_D = 2π / (2^r_D * w0_D)
        cfg_bohr = Config(
            sim = Lindbladian(), domain = BohrDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = true,
            beta = β, sigma = 1.0 / β, a = 0.0, s = 0.25, eta = 1e-3,
            num_energy_bits = r_D, w0 = w0_D, t0 = t0_D,
            num_trotter_steps_per_t0 = 10,
        )
        cfg_time = Config(
            sim = Lindbladian(), domain = TimeDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = true,
            beta = β, sigma = 1.0 / β, a = 0.0, s = 0.25, eta = 1e-3,
            num_energy_bits = r_D, w0 = w0_D, t0 = t0_D,
            num_trotter_steps_per_t0 = 10,
        )
        L_bohr, _ = build_dense_lindbladian(cfg_bohr, sys.ham, sys.jumps)
        L_time, _ = build_dense_lindbladian(cfg_time, sys.ham, sys.jumps)
        err_dense = opnorm(L_bohr .- L_time)

        ws_bohr = Workspace(cfg_bohr, sys.ham, sys.jumps)
        ws_time = Workspace(cfg_time, sys.ham, sys.jumps)
        buf_f = zeros(ComplexF64, dim_n3, dim_n3)
        buf_a = zeros(ComplexF64, dim_n3, dim_n3)
        diff_fwd! = function (out, X)
            apply_lindbladian!(ws_bohr, X, cfg_bohr, sys.ham)
            copyto!(buf_f, ws_bohr.scratch.rho_out)
            apply_lindbladian!(ws_time, X, cfg_time, sys.ham)
            axpby!(-1.0, ws_time.scratch.rho_out, 1.0, buf_f)
            copyto!(out, buf_f)
            return out
        end
        diff_adj! = function (out, X)
            apply_adjoint_lindbladian!(ws_bohr, X, cfg_bohr, sys.ham)
            copyto!(buf_a, ws_bohr.scratch.rho_out)
            apply_adjoint_lindbladian!(ws_time, X, cfg_time, sys.ham)
            axpby!(-1.0, ws_time.scratch.rho_out, 1.0, buf_a)
            copyto!(out, buf_a)
            return out
        end
        err_krylov = hs_operator_norm_krylov(diff_fwd!, diff_adj!, dim_n3;
                                              tol = 1e-12, krylovdim = 30)
        @test isapprox(err_dense, err_krylov; rtol = 1e-8)
        @info "(8c) ΔL parity" err_dense err_krylov rel = abs(err_dense - err_krylov) / err_dense
    end
end
