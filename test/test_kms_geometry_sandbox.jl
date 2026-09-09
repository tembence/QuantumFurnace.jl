# test/test_kms_geometry_sandbox.jl
#
# Sandbox shadow of test_kms_geometry.jl. The NO_SANDBOX heavy
# test exercises the KMS-geometry public surface over multiple (n, β,
# filter) cells; this shadow covers the same public invariants at the smallest
# fixture that still exercises every code path:
#
#   - n = 3, β = 10, single point
#   - BohrDomain CKG, smooth-Metropolis at thesis-numerics defaults
#     (a = 0, s = 0.25). BohrDomain `construct_lindbladian` is closed-form
#     in α(ν), so r_D = 12, w0 = 0.05 are kept only for Config-parity with
#     the heavy test; the dense L_super used here is r_D-independent.
#     Dense Lindbladian d² × d² = 64 × 64 ≈ 64 KiB peak; trivial sandbox.
#
# Public API covered here:
#   - kms_inner_product, kms_norm, kms_variance, kms_dirichlet_form  (testset 1)
#   - materialize_discriminant + KMS-DBC witness                     (testset 2)
#   - spectral_gap_kms (cross-check vs krylov_spectral_gap)          (testset 2/3)
#   - max_dirichlet_rate_kms, intrinsic_mixing_ratio                 (testset 3)
#   - dissipator_one_to_one_norm_bound, hs_operator_norm,
#     hs_operator_norm_krylov                                        (testset 4)
#
# Tolerances mirror the heavy test (≤ 1e-9 absolute / relative for the
# KMS-DBC witness, 1e-10 for the closed-form algebraic identities).
# `build_dense_superoperator` is exercised directly by
# `_build_dense_lindbladian_sandbox`.

using LinearAlgebra: I, eigvals, Hermitian, opnorm, tr, norm
using Random
using Test
using QuantumFurnace


# ---------------------------------------------------------------------------
# Small fixture helpers — shadows of test_kms_geometry.jl::ckg_smooth_metro_config
# at the canonical n=3 sandbox fixture only.
# ---------------------------------------------------------------------------
function _sandbox_ckg_smooth_metro_config(beta::Real)
    return Config(
        sim = Lindbladian(),
        domain = BohrDomain(),
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

function _sandbox_dll_config(beta::Real, filter::AbstractFilter)
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
Build the dense d² × d² Schrödinger Lindbladian via apply_lindbladian!.
Same physics path as the NO_SANDBOX test_kms_geometry.jl helper —
sandbox-version inlined here to avoid loading the heavy file.
"""
function _build_dense_lindbladian_sandbox(
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


@testset "KMS geometry [sandbox shadow]" begin
    Random.seed!(20260512)
    β = 10.0
    sys  = make_dll_n3_system(β)
    cfg  = _sandbox_ckg_smooth_metro_config(β)
    L_super, dim = _build_dense_lindbladian_sandbox(cfg, sys.ham, sys.jumps)
    gibbs = Matrix{ComplexF64}(sys.gibbs)
    s12   = sqrt(Hermitian(gibbs))

    # -----------------------------------------------------------------------
    # (1) KMS inner-product algebra (covers kms_inner_product, kms_norm,
    #     kms_variance — same identities as test_kms_geometry.jl testset (1)
    #     but at the larger d=8 fixture to exercise the n=3 code path).
    # -----------------------------------------------------------------------
    @testset "(1) KMS inner-product algebra" begin
        Xr = randn(ComplexF64, dim, dim); X = (Xr + Xr') / 2
        Yr = randn(ComplexF64, dim, dim); Y = (Yr + Yr') / 2

        # Hermitian symmetry: ⟨X, Y⟩ = conj(⟨Y, X⟩)
        @test isapprox(kms_inner_product(X, Y, gibbs; sigma_sqrt = s12),
                       conj(kms_inner_product(Y, X, gibbs; sigma_sqrt = s12));
                       atol = 1e-10)

        # ‖I‖_KMS = √tr(σ) = 1.
        Id = Matrix{ComplexF64}(I, dim, dim)
        @test isapprox(kms_norm(Id, gibbs; sigma_sqrt = s12), 1.0; atol = 1e-10)

        # Var_KMS(I) = 0.
        @test isapprox(kms_variance(Id, gibbs; sigma_sqrt = s12), 0.0;
                       atol = 1e-10)

        # Complex constants are also centred exactly, and adding any complex
        # constant leaves the variance unchanged.
        @test isapprox(kms_variance(im .* Id, gibbs; sigma_sqrt = s12), 0.0;
                       atol = 1e-10, rtol = 0)
        c = 0.7 - 1.3im
        @test isapprox(
            kms_variance(X .+ c .* Id, gibbs; sigma_sqrt = s12),
            kms_variance(X, gibbs; sigma_sqrt = s12);
            atol = 1e-10, rtol = 0,
        )

        # Cauchy–Schwarz: |⟨X, Y⟩| ≤ ‖X‖ ‖Y‖.
        cs_lhs = abs(kms_inner_product(X, Y, gibbs; sigma_sqrt = s12))
        cs_rhs = kms_norm(X, gibbs; sigma_sqrt = s12) *
                 kms_norm(Y, gibbs; sigma_sqrt = s12)
        @test cs_lhs ≤ cs_rhs + 1e-10

        # Positivity: ⟨X, X⟩ ≥ 0 (real, non-negative).
        ip_xx = kms_inner_product(X, X, gibbs; sigma_sqrt = s12)
        @test isapprox(imag(ip_xx), 0.0; atol = 1e-10)
        @test real(ip_xx) ≥ -1e-10
    end

    # -----------------------------------------------------------------------
    # (2) KMS-DB witness via materialize_discriminant and spectral_gap_kms
    #     cross-check vs krylov_spectral_gap. KMS-DB ⇒ the
    #     anti-Hermitian part of D_S = Φ⁻¹ L_S Φ vanishes; dense and Krylov
    #     gap must coincide because the spectrum is invariant under similarity.
    # -----------------------------------------------------------------------
    @testset "(2) Discriminant KMS-DB witness + dense/Krylov gap" begin
        # Schrödinger fixed-point witness.
        residual_sigma = norm(L_super * vec(gibbs)) / norm(vec(gibbs))
        @test residual_sigma < 1e-9

        # D_S is HS-self-adjoint to numerical precision.
        D_super = materialize_discriminant(L_super, Hermitian(gibbs))
        D_anti  = (D_super .- D_super') ./ 2
        D_sym   = Hermitian((D_super .+ D_super') ./ 2)
        anti_to_sym = norm(D_anti) / max(norm(D_sym), eps())
        @test anti_to_sym < 1e-9       # KMS-DB witness (1e-9 controllable)

        # D_S has vec(σ^{1/2}) as zero eigenvector.
        @test norm(D_super * vec(s12)) < 1e-9

        # Spectrum of D_sym is real and ≤ 0.
        ev = eigvals(D_sym)
        @test maximum(abs.(imag.(ev))) < 1e-9
        @test maximum(real.(ev)) < 1e-8

        # Dense spectral_gap_kms ↔ matrix-free krylov_spectral_gap.
        gap_dense  = spectral_gap_kms(L_super, gibbs).gap
        kr         = krylov_spectral_gap(cfg, sys.ham, sys.jumps;
                                          krylovdim = 30, howmany = 4,
                                          tol = 1e-12)
        gap_krylov = kr.spectral_gap
        @test isapprox(gap_dense, gap_krylov; rtol = 1e-7)
        @info "(2) gap dense vs Krylov" β gap_dense gap_krylov anti_to_sym
    end

    @testset "(2b) stationary-mode exclusion" begin
        Id2 = Matrix{ComplexF64}(I, 2, 2)
        sigma2 = Id2 ./ 2

        # L(rho) = tr(rho) sigma - rho has rates {0, 1, 1, 1}.
        L_depolarising = vec(sigma2) * vec(Id2)' -
            Matrix{ComplexF64}(I, 4, 4)
        for project_constants in (true, false)
            result = spectral_gap_kms(
                L_depolarising, sigma2; project_constants = project_constants)
            @test isapprox(result.gap, 1.0; atol = 1e-12, rtol = 0)
            @test length(result.eigvals) == 3
        end

        # Remove the known stationary witness, not simply the eigenvalue nearest
        # zero: a physical rate can be smaller than a finite stationarity residual.
        pauli_x = ComplexF64[0 1; 1 0]
        pauli_y = ComplexF64[0 -im; im 0]
        pauli_z = ComplexF64[1 0; 0 -1]
        vI = vec(Id2) ./ sqrt(2)
        vX = vec(pauli_x) ./ sqrt(2)
        vY = vec(pauli_y) ./ sqrt(2)
        vZ = vec(pauli_z) ./ sqrt(2)
        stationary_residual = 1e-6
        physical_gap = 1e-8
        positive_parent = stationary_residual .* (vI * vI') .+
            vX * vX' .+ vY * vY' .+ physical_gap .* (vZ * vZ')
        L_near_stationary = -positive_parent
        for project_constants in (true, false)
            result = spectral_gap_kms(
                L_near_stationary, sigma2; project_constants = project_constants)
            @test isapprox(result.gap, physical_gap; atol = 1e-12, rtol = 1e-8)
        end
    end

    # -----------------------------------------------------------------------
    # (3) Dirichlet form + Λ_max + ρ_intrinsic. Covers kms_dirichlet_form,
    #     max_dirichlet_rate_kms, intrinsic_mixing_ratio.
    # -----------------------------------------------------------------------
    @testset "(3) Dirichlet form + Λ_max + ρ_intrinsic" begin
        L_super_H = Matrix(L_super')   # Heisenberg = HS-adjoint of Schrödinger
        Xraw = randn(ComplexF64, dim, dim); X = (Xraw + Xraw') / 2

        # Linearity: E_{2L}(X) = 2 E_L(X).
        E_L  = kms_dirichlet_form(L_super_H, X, gibbs; sigma_sqrt = s12)
        E_2L = kms_dirichlet_form(2.0 .* L_super_H, X, gibbs; sigma_sqrt = s12)
        @test isapprox(E_2L, 2 * E_L; rtol = 1e-10)

        # E_L(X) ≥ 0 for KMS-DB L on Hermitian X.
        @test E_L ≥ -1e-10

        # Vanishing on constants: E_L(c·I) = 0 (unitality of L^H).
        cI = 0.7 * Matrix{ComplexF64}(I, dim, dim)
        E_const = kms_dirichlet_form(L_super_H, cI, gibbs; sigma_sqrt = s12)
        @test isapprox(E_const, 0.0; atol = 1e-10)

        gap = spectral_gap_kms(L_super, gibbs).gap
        Λ   = max_dirichlet_rate_kms(L_super, gibbs)
        ρ   = intrinsic_mixing_ratio(L_super, gibbs)

        # Λ ≥ gap (the modal-Poincaré constant lower-bounds
        # the spectral gap of L for KMS-DB Lindbladians) and ρ = gap/Λ ∈ (0, 1].
        @test Λ ≥ gap - 1e-12
        @test 0.0 < ρ ≤ 1.0 + 1e-12

        # Scale invariance under L → c·L.
        c = 2.7
        @test isapprox(spectral_gap_kms(c .* L_super, gibbs).gap, c * gap;
                       rtol = 1e-10)
        @test isapprox(max_dirichlet_rate_kms(c .* L_super, gibbs), c * Λ;
                       rtol = 1e-10)
        @test isapprox(intrinsic_mixing_ratio(c .* L_super, gibbs), ρ;
                       rtol = 1e-12)

        @info "(3) Dirichlet form + ρ_intrinsic" β E_L E_const Λ gap ρ
    end

    # -----------------------------------------------------------------------
    # (4) 1→1 dissipator bound + HS operator norm + matrix-free parity.
    #     Covers dissipator_one_to_one_norm_bound, hs_operator_norm,
    #     hs_operator_norm_krylov. DLL Metropolis filter exposes the explicit
    #     per-coupling Lindblad operators required by the 1→1 bound.
    # -----------------------------------------------------------------------
    @testset "(4) 1→1 bound + hs_operator_norm[_krylov] parity" begin
        f_metro = DLLMetropolisFilter(β; S = 2.0)
        cfg_dll = _sandbox_dll_config(β, f_metro)
        L_dll, _ = _build_dense_lindbladian_sandbox(cfg_dll, sys.ham, sys.jumps)
        L_a_list = [Matrix{ComplexF64}(dll_lindblad_op_bohr(j, sys.ham, f_metro))
                    for j in sys.jumps]

        bound = dissipator_one_to_one_norm_bound(L_a_list)
        hs    = hs_operator_norm(L_dll)
        @test bound > 0
        @test hs > 0

        # Matrix-free hs_operator_norm_krylov vs dense opnorm on the same L.
        ws  = Workspace(cfg_dll, sys.ham, sys.jumps)
        Lap! = (out, X) -> (apply_lindbladian!(ws, X, cfg_dll, sys.ham);
                             copyto!(out, ws.scratch.rho_out); out)
        Lad! = (out, X) -> (apply_adjoint_lindbladian!(ws, X, cfg_dll, sys.ham);
                             copyto!(out, ws.scratch.rho_out); out)
        s_dense  = opnorm(L_dll)
        s_krylov = hs_operator_norm_krylov(Lap!, Lad!, dim;
                                            tol = 1e-12, krylovdim = 30)
        @test isapprox(s_dense, s_krylov; rtol = 1e-9)

        bad_forward! = (out, X) -> (out .= 2 .* X; out)
        bad_adjoint! = (out, X) -> (out .= 3 .* X; out)
        @test_throws ArgumentError hs_operator_norm_krylov(
            bad_forward!, bad_adjoint!, 2;
            tol = 1e-12, krylovdim = 4, max_retries = 0,
        )
        @info "(4) DLL Metro β=$β" bound hs s_dense s_krylov
    end
end
