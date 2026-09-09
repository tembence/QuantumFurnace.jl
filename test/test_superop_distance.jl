using LinearAlgebra
using Test
using QuantumFurnace
using QuantumFurnace: _jumps_in_basis, build_dense_superoperator, trace_distance_nh,
                     _build_channel_config, _lookup_channel_params,
                     _load_channel_param_table, make_trotter_for_config,
                     _iterate_channel_states

# matrix-free trace-norm distance between two superoperator propagators.
# Decisive correctness checks (all NUMERICAL — matrix-free == dense linear algebra,
# which holds at any n, so n=3 is fine here; physics judgments live in the driver
# at n≥4). The channel arm uses the CANONICAL 3-leg TrotterTriple via
# make_trotter_for_config (NOT a legacy single-cache TrottTrott).
#   (a) DENSE cross-check — matrix-free trajectory (channel = exact iteration, ideal
#       L = Krylov exponentiate) reproduces dense exp(δ·L_super)/Φ_super^k, INCLUDING
#       the basis reconciliation between the Trotter eigenbasis and the H eigenbasis.
#   (b) BASIS-correctness via unitary invariance — the channel arm's common-basis
#       distance-to-Gibbs equals predict_channel_trajectory's Trotter-basis distance.
#   (c) two-Lindbladian path + generator-form fixed-point distance vs dense.
#   (d) cross-DOMAIN L-vs-L (EnergyDomain r_D=7 vs BohrDomain, same ham.eigvecs):
#       agreement to machine precision (1e-9 cross-domain controllability invariant).
#   (e) channel-vs-channel: identical arms ⇒ T ≡ 0 (self-consistency + ordering robustness).
#   (f) input validation: t_grid[1]≠0, empty/unsorted grid, non-commensurate channel
#       t, negative k_grid — all throw ArgumentError.
#   (g) generator-form fixed point at SMALLER δ (1e-4) vs dense — exercises the
#       documented O(1/δ) small-δ conditioning of the (Φ_δ-I)/δ extraction.
#   (h) robust arm_fixed_point (:dense exact, :krylov Gibbs-seeded, :auto
#       residual-gated) + fixed_point_gibbs_distance (½‖σ_Φ-ρ_β‖₁, the thermodynamic
#       error axis read alongside the gap distortion); residual as a stationarity certificate.

@testset "superop_distance" begin
    rho_plus(m) = (psi = ones(ComplexF64, 2^m) ./ sqrt(2.0^m); psi * psi')
    param_table = QuantumFurnace._package_data_path("channel_param_table.bson")

    isfile(param_table) || error("missing required channel parameter table: $param_table")
        n = 3
        d = N3_DIM
        ham = N3_HAM                       # β_alg = BETA = 10
        β = BETA; σ = SIGMA
        rows = _load_channel_param_table(param_table)
        row = _lookup_channel_params(rows, n, β, 1e-3, :smooth_metro)   # s=0.25, a=0 cell
        s = row.s; a = row.a
        delta = row.delta                  # 1e-3

        # Channel arm — canonical TrotterTriple, works in trotter.eigvecs (= D-leg basis).
        cfg_C = _build_channel_config(row, n, β, TrotterDomain(), KMS())
        trotter = make_trotter_for_config(ham, cfg_C)
        jumps_C = _jumps_in_basis(n, trotter.eigvecs)
        armC = channel_arm(cfg_C, ham, jumps_C, trotter; label = "Φ_δ")

        # Ideal L arm — same (β,σ,s,a) generator, near-Bohr r_D, works in ham.eigvecs.
        r_D = 7
        H_norm = maximum(abs, ham.eigvals)
        omega_range = 2.0 * (H_norm + 8σ)
        w0_D = omega_range / 2.0^r_D
        t0_D = 2π / (2.0^r_D * w0_D)
        cfg_L = Config(
            sim = Lindbladian(), domain = EnergyDomain(), construction = KMS(),
            num_qubits = n, with_linear_combination = true,
            beta = β, sigma = σ, a = a, s = s,
            num_energy_bits_D = r_D, w0_D = w0_D, t0_D = t0_D,
            num_trotter_steps_per_t0 = 10, filter = nothing,
        )
        jumps_L = _jumps_in_basis(n, ham.eigvecs)
        armL = lindbladian_arm(cfg_L, ham, jumps_L; label = "e^{δL}")

        @testset "stable Gibbs quarter powers" begin
            sigma_quarter, sigma_inv_quarter =
                QuantumFurnace._gibbs_quarter_powers(ham, β)
            weights = real.(diag(ham.gibbs))
            @test isapprox(sigma_quarter, weights .^ 0.25;
                atol=1e-14, rtol=0)
            @test isapprox(sigma_inv_quarter, weights .^ -0.25;
                atol=1e-12, rtol=0)
        end

        @testset "workspace provenance guards" begin
            stale_L = Workspace(cfg_L, ham, jumps_L)
            changed_L = copy(jumps_L)
            jL = changed_L[1]
            changed_L[1] = JumpOp(2 .* jL.data, 2 .* jL.in_eigenbasis,
                                  jL.orthogonal, jL.hermitian)
            @test_throws ArgumentError lindbladian_arm(
                cfg_L, ham, changed_L; workspace=stale_L)
            @test_throws ArgumentError lindbladian_discriminant_antiherm_norm(
                cfg_L, ham, changed_L; workspace=stale_L)

            stale_C = Workspace(cfg_C, ham, jumps_C; trotter=trotter)
            changed_C = copy(jumps_C)
            jC = changed_C[1]
            changed_C[1] = JumpOp(2 .* jC.data, 2 .* jC.in_eigenbasis,
                                  jC.orthogonal, jC.hermitian)
            @test_throws ArgumentError channel_arm(
                cfg_C, ham, changed_C, trotter; workspace=stale_C)
            @test_throws ArgumentError channel_discriminant_antiherm_norm(
                cfg_C, ham, changed_C, trotter; workspace=stale_C)
        end

        rho_0 = rho_plus(n)
        k_grid = [0, 1, 2, 5, 10, 20]
        t_grid = float.(k_grid) .* delta

        VC = Matrix{ComplexF64}(trotter.eigvecs)
        VL = Matrix{ComplexF64}(ham.eigvecs)
        gibbs_common = VL * Matrix{ComplexF64}(ham.gibbs) * VL'

        @testset "(a) dense cross-check + basis reconciliation" begin
            # hermitize/renormalize OFF so the channel arm matches the raw linear
            # dense map Φ_super^k exactly.
            res = propagator_trace_distance(armC, armL, rho_0, t_grid;
                tol = 1e-12, hermitize = false, renormalize = false, save_states = true)

            @test length(res.trace_distances) == length(k_grid)
            @test res.labels == ("Φ_δ", "e^{δL}")
            @test res.trace_distances[1] < 1e-12     # T_0 = 0 — basis-correctness canary

            Phi  = build_dense_superoperator(armC.apply!, d)
            Lsup = build_dense_superoperator(armL.apply!, d)
            rho_C_w = VC' * rho_0 * VC
            rho_L_w = VL' * rho_0 * VL

            for (i, k) in enumerate(k_grid)
                t = k * delta
                vc = vec(Matrix{ComplexF64}(rho_C_w))
                for _ in 1:k
                    vc = Phi * vc
                end
                rhoC = VC * reshape(vc, d, d) * VC'
                vl = exp(Lsup .* t) * vec(Matrix{ComplexF64}(rho_L_w))
                rhoL = VL * reshape(vl, d, d) * VL'

                # The METRIC is the load-bearing check.
                T_dense = trace_distance_nh(Matrix(rhoC), Matrix(rhoL))
                @test abs(T_dense - res.trace_distances[i]) < 1e-9
                @test maximum(abs.(res.states[1][i] .- rhoC)) < 1e-9
                @test maximum(abs.(res.states[2][i] .- rhoL)) < 1e-9
            end
            @info "(a) dense cross-check" max_T=res.max_distance per_step=res.per_step_distance matvecs=res.matvecs
        end

        @testset "(b) basis-correctness vs predict_channel_trajectory (unitary invariance)" begin
            res = propagator_trace_distance(armC, armL, rho_0, t_grid;
                hermitize = true, renormalize = false, save_states = true)
            chan_states = res.states[1]
            T_chan_gibbs = [trace_distance_nh(chan_states[i], gibbs_common) for i in eachindex(k_grid)]

            rho_init_trotter = VC' * rho_0 * VC
            pr = predict_channel_trajectory(cfg_C, ham, jumps_C, rho_init_trotter, k_grid;
                krylovdim = 40, trotter = trotter)

            # This compares exact channel iteration with a finite spectral
            # reconstruction.  Increasing krylovdim from 40 to 60 leaves the
            # fixture's 3.1e-6 discrepancy unchanged; a basis error is O(1).
            @test maximum(abs.(T_chan_gibbs .- pr.distances)) < 1e-5
            @info "(b) unitary-invariance basis check" max_dev=maximum(abs.(T_chan_gibbs .- pr.distances))
        end

        @testset "(c) two-Lindbladian path + generator-form fixed-point distance" begin
            # Two EnergyDomain Lindbladians at different r_D — both in ham.eigvecs
            # (identical V), measuring the quadrature-induced trajectory difference.
            r_D2 = 5
            w0_D2 = omega_range / 2.0^r_D2
            t0_D2 = 2π / (2.0^r_D2 * w0_D2)
            cfg_L2 = Config(
                sim = Lindbladian(), domain = EnergyDomain(), construction = KMS(),
                num_qubits = n, with_linear_combination = true,
                beta = β, sigma = σ, a = a, s = s,
                num_energy_bits_D = r_D2, w0_D = w0_D2, t0_D = t0_D2,
                num_trotter_steps_per_t0 = 10, filter = nothing,
            )
            armL2 = lindbladian_arm(cfg_L2, ham, jumps_L; label = "L(r_D=5)")

            t_grid_L = [0.0, 0.1, 0.5, 1.0, 2.0]
            res = propagator_trace_distance(armL, armL2, rho_0, t_grid_L; tol = 1e-12)
            @test res.trace_distances[1] < 1e-12          # T_0 = 0
            @test all(isfinite, res.trace_distances)
            @test res.max_distance < 1e-1                 # small quadrature perturbation

            # Fixed-point distances. krylovdim=64 = full Krylov at n=3 (d²=64) ⇒ the
            # matrix-free fixed point is exact, so it can be checked tightly vs dense.
            # The ideal L's steady state IS the Gibbs state.
            fpLL = propagator_fixed_point_distance(armL, armL2; krylovdim = 64)
            @test trace_distance_nh(fpLL.sigma_A, gibbs_common) < 1e-6
            @test fpLL.distance < 1e-2

            # Generator-form channel fixed point must match the DENSE channel fixed
            # point (eigvec of Φ_super @ eigval≈1) — validates the (Φ_δ-I)/δ extraction.
            Phi = build_dense_superoperator(armC.apply!, d)
            F = eigen(Phi); i1 = argmin(abs.(F.values .- 1.0))
            σw = reshape(F.vectors[:, i1], d, d); σw = (σw + σw') / 2; σw ./= real(tr(σw))
            σC_dense = VC * σw * VC'
            T_inf_dense = trace_distance_nh(σC_dense, gibbs_common)

            fpCL = propagator_fixed_point_distance(armC, armL; krylovdim = 64)
            @test trace_distance_nh(fpCL.sigma_B, gibbs_common) < 1e-6   # ideal-L fp = Gibbs
            # Generator-form channel fp matches dense to ~3e-6 on an 8.7e-3 floor
            # (0.04%) — same fixed point, validating the (Φ_δ-I)/δ extraction. The
            # residual is the INHERENT small-δ conditioning: σ amplifies matvec /
            # threaded-reduction noise by ~1/(δ·gap) ≈ 5e3 at δ=1e-3. A method/basis
            # bug would be O(1), not O(1e-6).
            @test trace_distance_nh(fpCL.sigma_A, σC_dense) < 1e-5
            @test abs(fpCL.distance - T_inf_dense) < 1e-5
            @info "(c) fixed-point distances" LL=fpLL.distance CL=fpCL.distance dense=T_inf_dense
        end

        @testset "(h) robust arm_fixed_point + fixed_point_gibbs_distance" begin
            # (h1) DENSE = exact ground truth: kind-aware (μ≈1 for the channel), and the
            # fixed point is the Hermitian eigenvector of the full complex-linear
            # channel. A reference-free residual certifies it.
            fpd = arm_fixed_point(armC; method = :dense)
            @test fpd.method === :dense
            @test abs(abs(fpd.eigval) - 1.0) < 1e-6
            @test fpd.antiherm_frac < 1e-10
            @test fpd.residual < 1e-7
            @test fpd.steady_gap > 1e-9     # unique (gapped) steady state, not degenerate

            # (h2) Gibbs-seeded generator Krylov must reproduce the dense fixed point —
            # krylovdim=64 spans the full operator space at n=3.
            fpk = arm_fixed_point(armC; seed = gibbs_common, method = :krylov, krylovdim = 64)
            @test fpk.method === :krylov
            @test fpk.residual < 1e-6
            @test trace_distance_nh(fpk.sigma, fpd.sigma) < 1e-5

            # (h3) :auto returns a residual-CERTIFIED fixed point (accept Krylov iff its
            # residual passes the gate, else escalate to the exact dense path).
            fpa = arm_fixed_point(armC; method = :auto, krylovdim = 64)
            @test fpa.residual < 1e-6
            @test fpa.method in (:krylov, :dense)
            @test trace_distance_nh(fpa.sigma, fpd.sigma) < 1e-4

            # (h4) DELIVERABLE: ½‖σ_Φ − ρ_β‖₁ vs the explicit Gibbs state. seed=:auto ⇒
            # ρ_β for the :channel arm; matches the dense fixed-point-to-Gibbs distance.
            T_inf_dense = trace_distance_nh(fpd.sigma, gibbs_common)
            gd = fixed_point_gibbs_distance(armC, gibbs_common)
            @test abs(gd.distance - T_inf_dense) < 1e-5
            @test gd.residual < 1e-6
            @test gd.label == "Φ_δ"

            # (h5) Ideal-𝓛 arm: its fixed point IS the Gibbs state. seed=:auto ⇒ I/d
            # (NOT ρ_β, which is 𝓛's exact eigenvector ⇒ would break Arnoldi down).
            gdL = fixed_point_gibbs_distance(armL, gibbs_common; method = :krylov, krylovdim = 64)
            @test gdL.distance < 1e-6
            @test gdL.residual < 1e-8

            # (h6) The residual is a genuine certificate: a NON-stationary input (I/d)
            # scores orders of magnitude larger than the true fixed point.
            r_bad = QuantumFurnace._arm_stationarity_residual(armC, Matrix{ComplexF64}(I(d) / d))
            @test r_bad > 100 * fpd.residual

            # (h7) min_eigval is a PSD certificate. A valid density-matrix
            # fixed point has min eigenvalue ≥ -psd_tol (no warn); the helper fires a
            # @warn on a non-PSD matrix and returns its (negative) min eigenvalue.
            @test fpd.min_eigval ≥ -1e-10            # channel fp is a valid ρ (PSD)
            @test fpk.min_eigval ≥ -1e-10
            @test gd.min_eigval ≥ -1e-10 && gdL.min_eigval ≥ -1e-10
            @test_logs arm_fixed_point(armC; method = :dense)   # clean ⇒ NO PSD warn
            bad = Matrix{ComplexF64}([-0.1 0.0; 0.0 1.1])        # min eigval −0.1 ≪ −psd_tol
            local λbad
            @test_logs (:warn, r"not.*PSD") (λbad = QuantumFurnace._fixed_point_min_eigval(bad, "bad"))
            @test λbad ≈ -0.1 atol = 1e-12
            @info "(h) robust fixed point" dense_resid=fpd.residual krylov_vs_dense=trace_distance_nh(fpk.sigma, fpd.sigma) gibbs_dist=gd.distance idealL=gdL.distance auto_method=fpa.method min_eig=fpd.min_eigval
        end

        @testset "(h8) dense fallback threshold uses superoperator dimension" begin
            local_d = 3
            rates = reshape(ComplexF64.(1:local_d^2), local_d, local_d)
            no_stationary! = (out, X) -> (out .= -rates .* X; out)
            threshold_arm = PropagatorArm(
                no_stationary!;
                kind = :lindbladian,
                basis = Matrix{ComplexF64}(I, local_d, local_d),
                label = "dense-threshold regression",
            )
            seed = Matrix{ComplexF64}(I, local_d, local_d) ./ local_d

            # d fits the threshold while d^2 does not. A zero residual gate
            # forces the fallback decision without materialising the dense map.
            local threshold_result
            @test_logs (:warn, r"superoperator dimension") begin
                threshold_result = arm_fixed_point(
                    threshold_arm;
                    seed = seed,
                    method = :auto,
                    krylovdim = 6,
                    tol = 1e-12,
                    residual_gate = 0.0,
                    dense_max_dim = local_d,
                )
            end
            @test threshold_result.method === :krylov_ungated
            @test threshold_result.residual > 0
        end

        @testset "(i) anti-Hermitian discriminant norm: channel KMS-DB violation" begin
            ws_C = Workspace(cfg_C, ham, jumps_C; trotter = trotter)
            fwd_c!(out, X) = (apply_delta_channel!(ws_C, Matrix{ComplexF64}(X), cfg_C, ham; hermitize = false);
                              copyto!(out, ws_C.scratch.rho_next); out)
            adj_c!(out, X) = (apply_adjoint_delta_channel!(ws_C, Matrix{ComplexF64}(X), cfg_C, ham; hermitize = false);
                              copyto!(out, ws_C.scratch.rho_next); out)
            o1 = Matrix{ComplexF64}(undef, d, d); o2 = similar(o1)

            # (i1) DECISIVE: the adjoint channel is the HS-adjoint of the forward
            # ℂ-linear channel ⇔ its dense superoperator is the conjugate-transpose.
            S_fwd = build_dense_superoperator(fwd_c!, d)
            S_adj = build_dense_superoperator(adj_c!, d)
            @test opnorm(S_adj - S_fwd') / opnorm(S_fwd) < 1e-12

            # (i2) forward hermitize=false ≡ hermitize=true on a Hermitian (density)
            # input — the projection is an exact-arithmetic no-op there.
            Hr = randn(ComplexF64, d, d); Hr = (Hr + Hr') ./ 2
            apply_delta_channel!(ws_C, Hr, cfg_C, ham; hermitize = true);  copyto!(o1, ws_C.scratch.rho_next)
            apply_delta_channel!(ws_C, Hr, cfg_C, ham; hermitize = false); copyto!(o2, ws_C.scratch.rho_next)
            @test norm(o1 - o2) < 1e-12

            # (i3) ℂ-linearity of the hermitize=false channel (fails for the ℝ-linear default).
            X1 = randn(ComplexF64, d, d); X2 = randn(ComplexF64, d, d); αc = 0.7 + 0.3im; βc = -1.2 + 0.9im
            fwd_c!(o1, αc .* X1 .+ βc .* X2)
            r1 = copy(fwd_c!(o2, X1)); r2 = copy(fwd_c!(o2, X2))
            @test norm(o1 - αc .* r1 - βc .* r2) < 1e-10

            # (i4) duality: Φ_δ trace-preserving ⇒ Φ_δ† unital (Φ_δ†(I) = I).
            adj_c!(o1, Matrix{ComplexF64}(I, d, d))
            @test norm(o1 - I) < 1e-6

            # (i5) DECISIVE: matrix-free ‖A[D(G_eff)]‖ reproduces the dense d²×d² value.
            W = Matrix{ComplexF64}(trotter.eigvecs)' * Matrix{ComplexF64}(ham.eigvecs)  # X_tr = W X_ham W'
            t1 = Matrix{ComplexF64}(undef, d, d)
            chan_ham!(out, Xham) = begin
                Xtr = W * Matrix{ComplexF64}(Xham) * W'
                fwd_c!(t1, Xtr)
                out .= W' * t1 * W; out
            end
            Gsup = (build_dense_superoperator(chan_ham!, d) - I) ./ delta
            DG = QuantumFurnace.materialize_discriminant(Gsup, Hermitian(Matrix{ComplexF64}(ham.gibbs)))
            aG_dense = opnorm(Matrix((DG .- DG') ./ 2))
            mfC = channel_discriminant_antiherm_norm(cfg_C, ham, jumps_C, trotter; krylovdim = 30)
            @test abs(mfC.antiherm_norm - aG_dense) / aG_dense < 1e-6
            @test mfC.antiherm_norm > 1e-6                       # genuinely nonzero KMS-DB break
            gd_diag = real.(diag(Matrix(ham.gibbs)))
            @test mfC.conditioning ≈ sqrt(maximum(gd_diag) / minimum(gd_diag)) rtol = 1e-9

            # (i6) The ideal-𝓛 baseline is at the KMS-DB noise floor. Materialise
            # this small fixture for a certified norm: the matrix-free GKL pair
            # can lose its adjoint-compatibility check when cancellation drives A
            # down to matvec roundoff, and now correctly throws in that regime.
            Lsup = build_dense_superoperator(armL.apply!, d)
            DL = QuantumFurnace.materialize_discriminant(Lsup, ham.gibbs)
            aL_dense = opnorm(Matrix((DL .- DL') ./ 2))
            @test aL_dense < 1e-6
            @test mfC.antiherm_norm > 1e3 * aL_dense

            @info "(i) channel KMS-DB violation" mf=mfC.antiherm_norm dense=aG_dense floor=aL_dense κ_σ=mfC.conditioning
        end

        @testset "(d) cross-domain L-vs-L controllability (EnergyDomain vs BohrDomain)" begin
            # Match generator parameters and working bases. With r_D=7 on this
            # fixture, the EnergyDomain quadrature agrees with BohrDomain to 1e-9.
            cfg_B = Config(
                sim = Lindbladian(), domain = BohrDomain(), construction = KMS(),
                num_qubits = n, with_linear_combination = true,
                beta = β, sigma = σ, a = a, s = s, filter = nothing,
            )
            armB = lindbladian_arm(cfg_B, ham, jumps_L; label = "L_Bohr")

            t_grid_LL = [0.0, 0.1, 0.5, 1.0, 2.0]
            resEB = propagator_trace_distance(armL, armB, rho_0, t_grid_LL; tol = 1e-12)
            @test resEB.trace_distances[1] < 1e-12        # T_0 = 0 — basis canary
            @test resEB.max_distance < 1e-9               # r_D=7 ⇒ Energy == Bohr to machine eps

            # Controllability: r_D=5 is visibly worse than r_D=7 — error DECREASES
            # toward 1e-9 as the controlling register grows (not a flat "good-enough"
            # floor). Reuse the r_D=5 generator from cfg_L2's recipe.
            r_D_lo = 5
            w0_lo = omega_range / 2.0^r_D_lo
            t0_lo = 2π / (2.0^r_D_lo * w0_lo)
            cfg_lo = Config(
                sim = Lindbladian(), domain = EnergyDomain(), construction = KMS(),
                num_qubits = n, with_linear_combination = true,
                beta = β, sigma = σ, a = a, s = s,
                num_energy_bits_D = r_D_lo, w0_D = w0_lo, t0_D = t0_lo,
                num_trotter_steps_per_t0 = 10, filter = nothing,
            )
            armE_lo = lindbladian_arm(cfg_lo, ham, jumps_L; label = "L_E(r_D=5)")
            res_lo = propagator_trace_distance(armE_lo, armB, rho_0, t_grid_LL; tol = 1e-12)
            @test res_lo.max_distance > resEB.max_distance   # r_D=7 strictly tighter
            @info "(d) cross-domain controllability" max_T_rD7=resEB.max_distance max_T_rD5=res_lo.max_distance
        end

        @testset "(e) channel-vs-channel: identical arms ⇒ T ≡ 0" begin
            # A channel paired against ITSELF must give T = 0 at every step (same
            # working basis, same matvec, same k-grid). This isolates the channel
            # iteration + posmap reordering from any generator difference. Use a
            # NON-MONOTONE-in-construction but sorted k_grid with a duplicate to
            # exercise the sort/unique/posmap path.
            k2 = [0, 1, 1, 3, 7]
            t2 = float.(k2) .* delta
            armC2 = channel_arm(cfg_C, ham, jumps_C, trotter; label = "Φ_δ (copy)")
            resCC = propagator_trace_distance(armC, armC2, rho_0, t2;
                hermitize = false, renormalize = false, save_states = true)
            @test all(<(1e-12), resCC.trace_distances)
            # duplicate k=1 lands on identical snapshots in BOTH arms
            @test maximum(abs.(resCC.states[1][2] .- resCC.states[1][3])) < 1e-14
            @info "(e) channel-vs-channel self" max_T=resCC.max_distance
        end

        @testset "(f) input validation" begin
            # t_grid[1] != 0 — the arms measure absolute time from 0; the Lindbladian
            # arm's integrator is relative to t_grid[1], so a nonzero start would
            # silently desync the two arms (regression guard for the fix that added
            # this check).
            @test_throws ArgumentError propagator_trace_distance(
                armC, armL, rho_0, [delta, 2delta])
            # empty grid
            @test_throws ArgumentError propagator_trace_distance(
                armC, armL, rho_0, Float64[])
            # unsorted grid
            @test_throws ArgumentError propagator_trace_distance(
                armC, armL, rho_0, [0.0, 2delta, delta])
            # non-square rho_0
            @test_throws ArgumentError propagator_trace_distance(
                armC, armL, ComplexF64[1 0 0; 0 0 0], [0.0, delta])
            # channel t not an integer multiple of δ
            @test_throws ArgumentError propagator_trace_distance(
                armC, armL, rho_0, [0.0, 0.5delta])
            # _iterate_channel_states rejects negative k
            seed_w = VC' * rho_0 * VC
            @test_throws ArgumentError _iterate_channel_states(
                armC.apply!, Matrix{ComplexF64}(seed_w), [-1, 0, 2])
            # fixed-point dimension inference: no basis / seed / d ⇒ clear error
            armC_nb = PropagatorArm(armC.apply!; kind = :channel, delta = delta,
                                    basis = nothing, label = "Φ nobasis")
            armL_nb = PropagatorArm(armL.apply!; kind = :lindbladian, basis = nothing,
                                    label = "L nobasis")
            @test_throws ArgumentError propagator_fixed_point_distance(armC_nb, armL_nb)
        end

        @testset "(g) generator-form fixed point at δ=1e-4 vs dense" begin
            # Re-derive the channel config at a 10x-smaller δ to exercise the small-δ
            # conditioning of the (Φ_δ-I)/δ extraction. The generator-form fp must
            # still land on the DENSE channel fixed point; the deviation is the
            # documented O(1/δ) amplification of matvec noise (~1e-3 at δ=1e-4, ~10x
            # the δ=1e-3 residual of testset (c)) — NOT an O(1) method/basis bug.
            δ_lo = 1e-4
            cfg_C_lo = Config(
                sim = Thermalize(), domain = TrotterDomain(), construction = KMS(),
                num_qubits = n, beta = β, sigma = row.sigma,
                with_linear_combination = row.with_linear_combination,
                a = row.a, s = row.s,
                gaussian_parameters = row.with_linear_combination ? (nothing, nothing) :
                    (row.gaussian_omega, row.gaussian_sigma),
                eta = row.eta,
                num_energy_bits_D = row.r_D, t0_D = row.t0_D, w0_D = row.w0_D,
                num_energy_bits_b_minus = row.r_bm, t0_b_minus = row.t0_bm, w0_b_minus = row.w0_bm,
                num_energy_bits_b_plus = row.r_bp, t0_b_plus = row.t0_bp, w0_b_plus = row.w0_bp,
                num_trotter_steps_per_t0 = row.M_D, delta = δ_lo, mixing_time = 5.0,
                with_gqsp = row.with_gqsp, gqsp_degree = row.gqsp_degree,
                jump_selection = row.jump_selection,
            )
            trotter_lo = make_trotter_for_config(ham, cfg_C_lo)
            jumps_C_lo = _jumps_in_basis(n, trotter_lo.eigvecs)
            armC_lo = channel_arm(cfg_C_lo, ham, jumps_C_lo, trotter_lo; label = "Φ_δ=1e-4")
            VC_lo = Matrix{ComplexF64}(trotter_lo.eigvecs)

            Phi_lo = build_dense_superoperator(armC_lo.apply!, d)
            F = eigen(Phi_lo); i1 = argmin(abs.(F.values .- 1.0))
            σw = reshape(F.vectors[:, i1], d, d); σw = (σw + σw') / 2; σw ./= real(tr(σw))
            σC_dense_lo = VC_lo * σw * VC_lo'

            fp_lo = propagator_fixed_point_distance(armC_lo, armL; krylovdim = 64)
            dev = trace_distance_nh(fp_lo.sigma_A, σC_dense_lo)
            @test dev < 5e-3                 # O(1/δ) conditioning, ≪ O(1)
            @test trace_distance_nh(fp_lo.sigma_B, gibbs_common) < 1e-6   # ideal-L fp = Gibbs
            @info "(g) small-δ fixed point" δ=δ_lo dev_vs_dense=dev
        end
end
