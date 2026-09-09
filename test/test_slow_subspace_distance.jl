using LinearAlgebra
using Test
using Random
using QuantumFurnace
using QuantumFurnace: _jumps_in_basis, build_dense_superoperator, trace_distance_nh,
                     make_trotter_for_config, _load_hamiltonian_bson

# slow_subspace_generator_distance — one-scalar slow-subspace
# generator-mismatch ε_slow = ‖⟨L_j | (G_test − G_ref) | R_k⟩‖₂ between a TEST
# propagator and a REFERENCE propagator, projected onto the reference's K slowest
# biorthonormal eigenpairs. Fixture pairing: ref = ideal CKG Lindbladian 𝓛,
# test = faithful δ-channel Φ_δ. Generator-mismatch M = (Φ_δ−I)/δ − 𝓛.
#
# Fixture: n=3 seed-46 disordered Heisenberg at β_phys=0.5, smooth Metropolis.
# s=0.25, a=0, δ=1e-3, r_D=8, M_D=20, r_b±=8, gqsp_degree=1.
# Derive β_alg = β_phys * rescaling_factor at runtime.
#
# LINEARITY: `channel_arm` calls `apply_delta_channel!(...; hermitize=false)`, so
# the channel is complex-linear and acts faithfully on arbitrary operator modes.
# Phase-fixing retained reference modes is only a unitary modal gauge;
# `max_antiherm_frac` describes whether individual modes are phase-Hermitian and
# does not gate the projection. K=1 remains the independently validated
# perturbative gap-shift scalar; K>1 is a fixed-biorthogonal-basis diagnostic.
#
# All checks are NUMERICAL (matrix-free == dense linear algebra ⇒ n=3 suffices):
#   (a) Complex linearity holds and the channel spectrum has no killed modes.
#   (b) K=1 DENSE ground truth — matrix-free ⟨L₂|M|R₂⟩ reproduces the dense PT shift
#       (channel applied to the Hermitian gap eigenvector), AND the exact dense
#       gap_Φ−gap_𝓛 to O(δ²); reduction K=1 ⇒ ε_slow = relative gap shift. Crosses
#       the MISMATCHED bases (𝓛 in ham.eigvecs, Φ_δ in trotter.eigvecs).
#   (c) Large modal anti-Hermiticity is non-gating for complex-linear arms.
#   (d) BASIS-ROTATION + self-pairing canaries ⇒ ε_slow = 0 (rotation-bug guard).
#   (e) TWO-𝓛 quadrature smoke + controllability (error ↓ toward 1e-9 as r_D grows).
#   (f) INPUT validation + include_stationary structure.

@testset "slow_subspace_generator_distance" begin
    n = 3
    d = 2^n
    β_phys = 0.5
    ham_path = test_hamiltonian_path(3)

    resc = _load_hamiltonian_bson(ham_path, 1.0).rescaling_factor
    β_alg = β_phys * resc
    σ_alg = 1.0 / β_alg
    @test isapprox(β_alg, 7.1827637071309782; rtol = 1e-9)   # pins the v4 cell
    ham = _load_hamiltonian_bson(ham_path, β_alg)

    rho_plus(m) = (psi = ones(ComplexF64, 2^m) ./ sqrt(2.0^m); psi * psi')
    rho_0 = rho_plus(n)

    # --- channel arm Φ_δ (canonical TrotterDomain recipe, trotter.eigvecs) ---
    r_bm = 8; t0_bm = 0.029646353064078555
    r_bp = 8; t0_bp = 0.019764235376052371
    cfg_C = Config(
        sim = Thermalize(), domain = TrotterDomain(), construction = KMS(),
        num_qubits = n, beta = β_alg, beta_phys = β_phys, sigma = σ_alg,
        with_linear_combination = true, a = 0.0, s = 0.25,
        gaussian_parameters = (nothing, nothing), eta = 1e-12,
        num_energy_bits_D = 8, t0_D = 2.0089768670761279, w0_D = 0.012217011060904469,
        num_energy_bits_b_minus = r_bm, t0_b_minus = t0_bm, w0_b_minus = 2π / (2.0^r_bm * t0_bm),
        num_energy_bits_b_plus  = r_bp, t0_b_plus  = t0_bp, w0_b_plus  = 2π / (2.0^r_bp * t0_bp),
        num_trotter_steps_per_t0 = 20, delta = 1e-3, mixing_time = 5.0,
        with_gqsp = true, gqsp_degree = 1, jump_selection = :sweep,
    )
    delta = 1e-3
    trotter = make_trotter_for_config(ham, cfg_C)
    jumps_C = _jumps_in_basis(n, trotter.eigvecs)
    armC = channel_arm(cfg_C, ham, jumps_C, trotter; label = "Φ_δ")

    # --- ideal 𝓛 arm (EnergyDomain near-Bohr r_D=8, ham.eigvecs) ---
    H_norm = maximum(abs, ham.eigvals)
    omega_range = 2.0 * (H_norm + 8σ_alg)
    mkL(rD) = (w = omega_range / 2.0^rD; t = 2π / (2.0^rD * w); Config(
        sim = Lindbladian(), domain = EnergyDomain(), construction = KMS(),
        num_qubits = n, with_linear_combination = true, beta = β_alg, beta_phys = β_phys,
        sigma = σ_alg, a = 0.0, s = 0.25,
        num_energy_bits_D = rD, w0_D = w, t0_D = t, num_trotter_steps_per_t0 = 10, filter = nothing))
    jumps_L = _jumps_in_basis(n, ham.eigvecs)
    armL = lindbladian_arm(mkL(8), ham, jumps_L; label = "ideal L")

    VL = Matrix{ComplexF64}(ham.eigvecs)
    VC = Matrix{ComplexF64}(trotter.eigvecs)
    apply_common(arm, V, X) = (o = Matrix{ComplexF64}(undef, d, d); arm.apply!(o, V' * X * V); V * o * V')

    # dense TRUE 𝓛 superoperator in the common basis (armL is ℂ-linear: no hermitize).
    Lsup = let tmp = Matrix{ComplexF64}(undef, d, d)
        build_dense_superoperator((out, X) -> (armL.apply!(tmp, VL' * X * VL); copyto!(out, VL * tmp * VL'); out), d)
    end
    Fr = eigen(Lsup); permL = sortperm(Fr.values; by = v -> abs(real(v)))
    λL = Fr.values[permL]; Rd = Fr.vectors[:, permL]
    Fl = eigen(Lsup')
    match_left(λ) = Fl.vectors[:, argmin(abs.(Fl.values .- conj(λ)))]

    @testset "(a) complex-linear channel spectrum: all |μ|≈1" begin
        Xnh = zeros(ComplexF64, d, d); Xnh[1, 2] = 1; Xnh[3, 1] = -0.4im
        Ynh = zeros(ComplexF64, d, d); Ynh[2, 4] = 0.7 + 0.2im; Ynh[4, 1] = -1
        a = 0.3 + 0.8im
        b = -0.6 + 0.1im
        lhs = apply_common(armC, VC, a .* Xnh .+ b .* Ynh)
        rhs = a .* apply_common(armC, VC, Xnh) .+
            b .* apply_common(armC, VC, Ynh)
        @test norm(lhs - rhs) < 1e-12

        # Independent d²-dimensional representation in an orthonormal
        # Hermitian basis; complex-linearity makes it equivalent to E_ij materialisation.
        B = Matrix{ComplexF64}[]
        for i in 1:d
            E = zeros(ComplexF64, d, d); E[i, i] = 1; push!(B, E)
        end
        for i in 1:d, j in (i + 1):d
            E = zeros(ComplexF64, d, d); E[i, j] = 1 / sqrt(2); E[j, i] = 1 / sqrt(2); push!(B, E)
            F = zeros(ComplexF64, d, d); F[i, j] = im / sqrt(2); F[j, i] = -im / sqrt(2); push!(B, F)
        end
        Phi_true = zeros(ComplexF64, d * d, d * d)
        for (k, Ek) in enumerate(B)
            res = apply_common(armC, VC, Ek)
            for (j, Ej) in enumerate(B); Phi_true[j, k] = dot(vec(Ej), vec(res)); end
        end
        μ = eigvals(Phi_true)
        @test minimum(abs.(μ)) > 0.9                 # NO killed modes (true channel ≈ I+δG)
        @test count(<(0.5), abs.(μ)) == 0
        @test maximum(abs.(μ)) ≤ 1 + 1e-8            # CPTP contraction
        @info "(a) channel spectrum" min_abs_mu=minimum(abs.(μ))
    end

    @testset "(b) K=1 dense ground truth + reduction (mismatched bases)" begin
        # Hermitian gap eigenvector of 𝓛 (idx 2) and its matched left eigenvector.
        r2 = reshape(Rd[:, 2], d, d); r2 = (r2 .+ r2') ./ 2; r2 ./= norm(r2)
        l2 = reshape(match_left(λL[2]), d, d)
        # Apply the channel + 𝓛 DIRECTLY to the Hermitian r2 (uncontaminated):
        Φr2 = apply_common(armC, VC, r2)
        Lr2 = apply_common(armL, VL, r2)
        Mr2 = (Φr2 .- r2) ./ delta .- Lr2
        Δλ_PT_dense = dot(vec(l2), vec(Mr2)) / dot(vec(l2), vec(r2))
        # exact dense gap shift via the TRUE channel generator (Herm-basis eigenvalues).
        Bh = vcat([(E = zeros(ComplexF64, d, d); E[i, i] = 1; E) for i in 1:d],
                  [(E = zeros(ComplexF64, d, d); E[i, j] = 1/sqrt(2); E[j, i] = 1/sqrt(2); E) for i in 1:d for j in (i+1):d],
                  [(F = zeros(ComplexF64, d, d); F[i, j] = im/sqrt(2); F[j, i] = -im/sqrt(2); F) for i in 1:d for j in (i+1):d])
        Phi_true = zeros(ComplexF64, d*d, d*d)
        for (k, Ek) in enumerate(Bh); res = apply_common(armC, VC, Ek); for (j, Ej) in enumerate(Bh); Phi_true[j,k] = dot(vec(Ej), vec(res)); end; end
        gen_eigs = (eigvals(Phi_true) .- 1) ./ delta
        Δλ_exact = gen_eigs[argmin(abs.(gen_eigs .- λL[2]))] - λL[2]

        res = slow_subspace_generator_distance(armL, armC, rho_0; num_slow_modes = 1, krylovdim = 64)

        # DECISIVE: matrix-free PT shift == dense PT shift (validates basis
        # reconciliation 𝓛(ham.eigvecs) ↔ Φ_δ(trotter.eigvecs) + contraction).
        @test abs(res.M[1, 1] - Δλ_PT_dense) < 1e-9
        # Fixture-level agreement with an independently diagonalised dense
        # channel generator; a one-point comparison is not an order claim.
        @test abs(abs(Δλ_PT_dense) - abs(Δλ_exact)) / abs(Δλ_exact) < 1e-3
        # Reduction K=1: ε_slow = |M₁₁|, ε_slow_rel_gap = ε_slow/|λ₂| = rel gap shift.
        @test isapprox(res.eps_slow, abs(res.M[1, 1]); rtol = 1e-12)
        @test isapprox(res.eps_slow_rel_gap, res.eps_slow / res.gap_ref; rtol = 1e-12)
        @test isapprox(res.eps_slow_rel_gap, abs(Δλ_exact) / abs(λL[2]); rtol = 5e-2)
        @test res.max_antiherm_frac < 1e-10          # gap mode is Hermitian ⇒ clean
        @test res.ref_gen_residual < 1e-10           # ⟨L_j|G_ref|R_k⟩ = λ_k δ_jk
        @test res.converged
        @test res.gap_ref > 0
        @info "(b) First-order gap shift" eps_slow=res.eps_slow rel_gap=res.eps_slow_rel_gap M11=res.M[1,1] densePT=Δλ_PT_dense exact=Δλ_exact
    end

    @testset "(c) modal anti-Hermiticity is non-gating" begin
        res1 = @test_logs slow_subspace_generator_distance(armL, armC, rho_0; num_slow_modes = 1, krylovdim = 64)
        @test res1.max_antiherm_frac < 1e-6

        res4_self = @test_logs slow_subspace_generator_distance(
            armL, armL, rho_0; num_slow_modes = 4, krylovdim = 64)
        @test res4_self.max_antiherm_frac > 1e-3
        @test res4_self.eps_slow < 1e-12
        @test res4_self.ref_gen_residual < 1e-10

        res4 = @test_logs slow_subspace_generator_distance(
            armL, armC, rho_0; num_slow_modes = 4, krylovdim = 64)
        @test res4.max_antiherm_frac > 1e-3
        @test isfinite(res4.eps_slow)
        @test res4.ref_gen_residual < 1e-10
        @info "(c) modal diagnostic" af1=res1.max_antiherm_frac af4=res4.max_antiherm_frac eps4=res4.eps_slow
    end

    @testset "(d) basis-rotation + self-pairing canaries ⇒ ε_slow = 0" begin
        Random.seed!(123)
        A = randn(ComplexF64, d, d); U, _ = qr(A); U = Matrix(U)
        buf = Matrix{ComplexF64}(undef, d, d)
        apply_rot! = (out, Y) -> (armL.apply!(buf, U * Matrix{ComplexF64}(Y) * U'); copyto!(out, U' * buf * U); out)
        armL_rot = PropagatorArm(apply_rot!; kind = :lindbladian, basis = VL * U, label = "L rot")
        res_rot = slow_subspace_generator_distance(armL, armL_rot, rho_0; num_slow_modes = 4, krylovdim = 64)
        @test res_rot.eps_slow < 1e-10
        res_id = slow_subspace_generator_distance(armL, armL, rho_0; num_slow_modes = 4, krylovdim = 64)
        @test res_id.eps_slow < 1e-12
        @info "(d) canaries" rotated=res_rot.eps_slow identical=res_id.eps_slow
    end

    @testset "(e) two-𝓛 quadrature smoke + controllability" begin
        armL5 = lindbladian_arm(mkL(5), ham, jumps_L; label = "L(r_D=5)")
        armL6 = lindbladian_arm(mkL(6), ham, jumps_L; label = "L(r_D=6)")
        res5 = slow_subspace_generator_distance(armL, armL5, rho_0; num_slow_modes = 1, krylovdim = 64)
        res6 = slow_subspace_generator_distance(armL, armL6, rho_0; num_slow_modes = 1, krylovdim = 64)
        @test isfinite(res5.eps_slow)
        @test res5.eps_slow > res6.eps_slow
        @test res6.eps_slow < 1e-9
        @test res5.max_antiherm_frac < 1e-10       # both 𝓛 arms ⇒ Hermitian gap mode
        @info "(e) two-L quadrature" rD5=res5.eps_slow rD6=res6.eps_slow
    end

    @testset "(f) input validation + include_stationary" begin
        @test_throws ArgumentError slow_subspace_generator_distance(armL, armC, rho_0; num_slow_modes = 0)
        @test_throws ArgumentError slow_subspace_generator_distance(armL, armC, rho_0; num_slow_modes = 4, krylovdim = 5)
        @test_throws ArgumentError slow_subspace_generator_distance(armL, armC, ComplexF64[1 0 0; 0 0 0]; num_slow_modes = 1)

        res_st = slow_subspace_generator_distance(armL, armC, rho_0;
            num_slow_modes = 1, include_stationary = true, krylovdim = 64)
        @test size(res_st.M) == (2, 2)                       # steady + 1 slow
        @test res_st.include_stationary
        @test abs(res_st.eigenvalues_ref[1]) < 1e-10
        @test res_st.gap_ref > 0

        decomp = QuantumFurnace._krylov_spectral_decomposition(
            QuantumFurnace._arm_generator(armL),
            Matrix{ComplexF64}(VL' * rho_0 * VL), d;
            krylovdim=64, tol=1e-10, sort_mode=:lindbladian,
        )
        @test isapprox(dot(decomp.L_modes[1], decomp.R_modes[1]), 1; atol=1e-12)
    end

    @testset "(g) |λ₂−λ₃| neighbor-spacing PT-validity guard" begin
        # First-order PT for the gap shift needs ε_slow ≪ the gap mode's spacing to its
        # nearest non-stationary neighbour |λ₂−λ₃|, not just ≪|λ₂|. At this n=3 cell the
        # spectrum is crowded (|λ₂−λ₃|≈5e-4) so the dense-validated ratio sits at ≈0.11 —
        # below the default 0.5 (silent), comfortably above machine zero.
        res = slow_subspace_generator_distance(armL, armC, rho_0; num_slow_modes = 1, krylovdim = 64)
        @test isfinite(res.lambda_neighbor_spacing) && res.lambda_neighbor_spacing > 0
        @test isfinite(res.eps_slow_rel_neighbor)
        r = res.eps_slow_rel_neighbor
        @test 0 < r < 0.5                                    # PT valid at default (no warn)
        @test_logs slow_subspace_generator_distance(armL, armC, rho_0;   # default ⇒ NO spacing warn
            num_slow_modes = 1, krylovdim = 64)
        # Threshold logic: a pt_spacing_frac BELOW the actual ratio fires the warn;
        # ABOVE it stays silent. (Tests the guard, not a contrived ill-conditioned arm.)
        @test_logs (:warn, r"neighbor") slow_subspace_generator_distance(armL, armC, rho_0;
            num_slow_modes = 1, krylovdim = 64, pt_spacing_frac = r / 2)
        @test_logs slow_subspace_generator_distance(armL, armC, rho_0;
            num_slow_modes = 1, krylovdim = 64, pt_spacing_frac = min(2r, 0.99))
        @info "(g) neighbor-spacing guard" spacing=res.lambda_neighbor_spacing eps_rel_neighbor=r
    end
end
