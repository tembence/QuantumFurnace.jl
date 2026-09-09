using LinearAlgebra: I, eigen, eigvals, svdvals, norm, tr, Hermitian
using Test
using QuantumFurnace

# Krylov spectral-expansion trajectory predictor for
# Config{Lindbladian}. Validates the new `predict_lindbladian_trajectory`
# against (i) the existing ODE-based `lindblad_action_integrate` and
# (ii) a dense reference `e^{tL} rho_0` at small n.
#
# The two regression invariants are:
# 1. The reconstructed trajectory matches the dense reference within a
#    Krylov-truncation tolerance set by the captured-vs-missing eigenmode
#    amplitudes; for the standard 3n single-Pauli jump set on the small
#    Heisenberg test fixtures the error is < 1e-7 throughout the trajectory.
# 2. The spectral bisection time reproduces the target trace distance under
#    the independently materialised dense propagator.

@testset "predict_lindbladian_trajectory" begin

    # -----------------------------------------------------------------------
    # (a) n=3, beta=10: dense reference vs ODE vs Krylov spectral
    # -----------------------------------------------------------------------
    @testset "(a) n=3, beta=10 EnergyDomain CKG: dense vs ODE vs Krylov" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        ham = sys.ham
        jumps = sys.jumps
        d = size(ham.data, 1)

        cfg = Config(
            sim = Lindbladian(),
            domain = EnergyDomain(),
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta, sigma = 1.0 / beta,
            a = 0.0, s = 0.25,
            num_energy_bits = 12,
            w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
        )

        rho_0 = Matrix{ComplexF64}(I(d) / d)
        sigma_beta = Matrix{ComplexF64}(ham.gibbs)

        # Spectral gap for setting t_max.
        gap_res = krylov_spectral_gap(cfg, ham, jumps;
                                       krylovdim=30, howmany=4, tol=1e-10)
        gap = gap_res.spectral_gap
        @test gap > 0
        t_grid = collect(range(0.0, 5.0 / gap, length=21))

        # Dense reference: build full Lindbladian and exponentiate at each t.
        L_dense = construct_lindbladian(jumps, cfg, ham)
        v0 = vec(rho_0)
        distances_dense = Vector{Float64}(undef, length(t_grid))
        for (k, t) in enumerate(t_grid)
            rho_t = reshape(exp(t * L_dense) * v0, d, d)
            rho_t .= (rho_t + rho_t') ./ 2
            distances_dense[k] = sum(svdvals(rho_t .- sigma_beta)) / 2
        end

        # ODE-based.
        res_ode = integrate_to_gibbs(cfg, ham, jumps, rho_0, t_grid;
                                      mode=:L, krylovdim=30, tol=1e-10)
        @test maximum(abs.(res_ode.distances .- distances_dense)) < 1e-10

        # Krylov spectral.
        res_kr = predict_lindbladian_trajectory(cfg, ham, jumps, rho_0, t_grid;
                                                 krylovdim=40, tol=1e-10)

        # Check shape matches lindblad_action_integrate.
        @test length(res_kr.t) == length(t_grid)
        @test length(res_kr.distances) == length(t_grid)
        @test size(res_kr.rho_final) == (d, d)
        @test isa(res_kr.total_matvecs, Integer)
        @test res_kr.total_matvecs <= length(t_grid) * 30  # well under ODE budget
        @test isa(res_kr.all_converged, Bool)

        # with the 3n=9 single-Pauli jump set + KMS-DB the slow
        # spectrum is well-separated; krylovdim=40 captures the entire
        # diagonal-sector dynamics on n=3 (d² = 64). 1e-7 covers the bi-exp-fit
        # accuracy regime (slow-tail dynamics) with margin for numerical noise.
        max_abs_err = maximum(abs.(res_kr.distances .- distances_dense))
        @test max_abs_err < 1e-7

        # τ_mix cross-check. The Krylov spectral path uses exact bisection on
        # its eigendecomposition (`eigenmode_mixing_time`), not a bi-exp fit.
        est_kr = eigenmode_mixing_time(res_kr, 1e-3)
        @test est_kr.source === :extrapolated
        @test isfinite(est_kr.mixing_time) && est_kr.mixing_time > 0

        # Gold check (fit-independent): plug the bisection τ back into the DENSE
        # propagator e^{τ L} ⇒ d(τ) ≈ ε. Validates the bisection against the true
        # dynamics, not against another estimator.
        rho_tau = reshape(exp(est_kr.mixing_time * L_dense) * v0, d, d)
        rho_tau .= (rho_tau .+ rho_tau') ./ 2
        @test isapprox(sum(svdvals(rho_tau .- sigma_beta)) / 2, 1e-3; rtol=1e-2)

        # the curve-fit estimator must REJECT a trajectory-predictor
        # result; τ_mix on this path comes only from the bisection above.
        @test_throws ArgumentError estimate_mixing_time(res_kr; model=:biexp,
                                                         target_epsilon=1e-3,
                                                         extrapolate=true)

        @info "(a) n=3 spectral expansion vs dense" max_abs_err matvecs_kr=res_kr.total_matvecs matvecs_ode=res_ode.total_matvecs

        # Spectral gap from Krylov should match dense at machine precision.
        # dense `eigen` gap and Krylov `eigsolve` gap differ
        # only by KrylovKit tolerance (1e-10).
        eigs_dense = sort(eigvals(L_dense); by = v -> abs(real(v)))
        gap_dense = abs(real(eigs_dense[2]))
        @test isapprox(res_kr.spectral_gap, gap_dense; rtol=1e-9)
    end

    # -----------------------------------------------------------------------
    # (b) n=3 BohrDomain: same agreement as EnergyDomain
    # -----------------------------------------------------------------------
    @testset "(b) n=3 BohrDomain CKG: ODE vs Krylov agree" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        cfg = make_config(Lindbladian(), BohrDomain();
                          num_qubits=3, construction=KMS())
        d = size(sys.ham.data, 1)
        rho_0 = Matrix{ComplexF64}(I(d) / d)
        t_grid = collect(range(0.0, 60.0, length=21))

        res_ode = integrate_to_gibbs(cfg, sys.ham, sys.jumps, rho_0, t_grid;
                                      mode=:L, krylovdim=20, tol=1e-10)
        res_kr = predict_lindbladian_trajectory(cfg, sys.ham, sys.jumps, rho_0, t_grid;
                                                 krylovdim=40, tol=1e-10)

        # both are matrix-free using the same apply_lindbladian!,
        # so the residual is dominated by Krylov-truncation in different
        # subspaces (Arnoldi for spectral; KrylovKit.exponentiate for ODE).
        # 1e-6 covers a 64-dim non-normal generator with eigenvalue clusters.
        @test maximum(abs.(res_kr.distances .- res_ode.distances)) < 1e-6
        @info "(b) n=3 BohrDomain CKG" max_diff=maximum(abs.(res_kr.distances .- res_ode.distances))
    end

    # -----------------------------------------------------------------------
    # (c) Output shape (states, rho_final, eigenvalues)
    # -----------------------------------------------------------------------
    @testset "(c) save_states + struct fields" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        cfg = make_config(Lindbladian(), BohrDomain();
                          num_qubits=3, construction=KMS())
        d = size(sys.ham.data, 1)
        rho_0 = Matrix{ComplexF64}(I(d) / d)
        t_grid = collect(range(0.0, 30.0, length=11))

        res = predict_lindbladian_trajectory(cfg, sys.ham, sys.jumps, rho_0, t_grid;
                                              krylovdim=30, save_states=true)
        @test length(res.states) == length(t_grid)
        @test all(size.(res.states) .== ((d, d),))
        # rho(0) should equal rho_0 within hermitisation noise (the spectral
        # expansion projects onto a captured subspace; for the diagonal
        # initial state and the slow-mode-dominated dynamics this is exact
        # to machine precision once enough modes are captured).
        @test maximum(abs.(res.states[1] .- rho_0)) < 1e-6
        # rho_final = rho(t_grid[end])
        @test isapprox(res.rho_final, res.states[end]; atol=1e-12)
        # Steady state stored.
        @test isa(res.rho_inf, AbstractMatrix)
        @test isapprox(real(tr(res.rho_inf)), 1.0; atol=1e-9)
        # Eigenvalues sorted by |Re| ascending; first ~ 0.
        @test abs(real(res.eigenvalues[1])) < 1e-8
    end

    # -----------------------------------------------------------------------
    # (d) DLL path (BohrDomain + DLLGaussianFilter)
    # -----------------------------------------------------------------------
    @testset "(d) DLL BohrDomain @ n=3, β=10 with DLLGaussianFilter" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        cfg = Config(
            sim = Lindbladian(),
            domain = BohrDomain(),
            construction = DLL(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta, sigma = 1.0 / beta,
            a = 0.0, s = 0.25,
            num_energy_bits = 12,
            w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
            filter = DLLGaussianFilter(beta),
        )
        d = size(sys.ham.data, 1)
        rho_0 = Matrix{ComplexF64}(I(d) / d)
        t_grid = collect(range(0.0, 80.0, length=21))

        res_ode = integrate_to_gibbs(cfg, sys.ham, sys.jumps, rho_0, t_grid;
                                      mode=:L, krylovdim=30, tol=1e-10)
        res_kr = predict_lindbladian_trajectory(cfg, sys.ham, sys.jumps, rho_0, t_grid;
                                                 krylovdim=40, tol=1e-10)

        # Same matrix-free path as CKG via apply_lindbladian!; convergence
        # is governed by the captured-mode subspace, same regime as (b).
        @test maximum(abs.(res_kr.distances .- res_ode.distances)) < 1e-6
        @info "(d) DLL BohrDomain Gaussian" max_diff=maximum(abs.(res_kr.distances .- res_ode.distances))
    end

    # -----------------------------------------------------------------------
    # (e0) sweep_mixing_times method=:krylov agrees with method=:ode
    # -----------------------------------------------------------------------
    @testset "(e0) sweep_mixing_times: :ode vs :krylov agreement" begin
        # at n=3 with the standard 3n single-Pauli jump set,
        # the spectral expansion captures the entire diagonal-sector dynamics
        # at krylovdim=60 (out of d²=64 modes available). τ_mix from bi-exp
        # extrapolation must therefore agree with the ODE-based sweep within
        # the 1% bi-exp-fit accuracy tier. Single β=10 cell to keep the test
        # fast — broader scans live in scripts/.
        res_ode = sweep_mixing_times([3], [10.0]; mode=:L, method=:ode,
                                      seeds=[42], use_threads=false, t_max_factor=5.0)
        res_kr = sweep_mixing_times([3], [10.0]; mode=:L, method=:krylov,
                                     seeds=[42], use_threads=false, t_max_factor=5.0)
        @test length(res_ode) == 1 && length(res_kr) == 1
        @test res_ode[1].method == :ode
        @test res_kr[1].method == :krylov
        @test isfinite(res_ode[1].mixing_time)
        @test isfinite(res_kr[1].mixing_time)
        rel = abs(res_ode[1].mixing_time - res_kr[1].mixing_time) /
              res_ode[1].mixing_time
        @test rel < 0.01

        # Krylov should use far fewer matvecs (single Arnoldi, not per-step).
        @test res_kr[1].total_matvecs < res_ode[1].total_matvecs / 5
        @info "(e0) sweep_mixing_times :ode vs :krylov" tau_ode=res_ode[1].mixing_time tau_kr=res_kr[1].mixing_time matvecs_ode=res_ode[1].total_matvecs matvecs_kr=res_kr[1].total_matvecs rel=rel
    end

    # -----------------------------------------------------------------------
    # (e) Trace preservation + Hermiticity + PSD on rho_final
    # -----------------------------------------------------------------------
    @testset "(e) rho_final is a valid density matrix" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        cfg = make_config(Lindbladian(), BohrDomain();
                          num_qubits=3, construction=KMS())
        d = size(sys.ham.data, 1)
        rho_0 = Matrix{ComplexF64}(I(d) / d)
        t_grid = collect(range(0.0, 100.0, length=21))

        res = predict_lindbladian_trajectory(cfg, sys.ham, sys.jumps, rho_0, t_grid;
                                              krylovdim=40)

        # Trace.
        @test isapprox(real(tr(res.rho_final)), 1.0; atol=1e-7)
        # Hermiticity (defensive hermitisation in the integrator).
        @test maximum(abs.(res.rho_final .- res.rho_final')) < 1e-9
        # PSD with small numerical slack.
        evs = eigvals(Hermitian((res.rho_final .+ res.rho_final') ./ 2))
        @test minimum(real.(evs)) > -1e-7
    end

    # -----------------------------------------------------------------------
    # (f) Gap estimation on a parity-symmetric classical Ising fixture.
    # H = Σ Z_i Z_{i+1} commutes with P = X^⊗N. A trajectory from I/d
    # remains parity-even, while the slow magnetisation mode is parity-odd.
    # A separate gap solve must recover the dense-reference gap.
    # -----------------------------------------------------------------------
    @testset "(f) predict_lindbladian_trajectory — symmetric system regression" begin
        n_ising = 3
        terms_zz = Vector{Vector{Matrix{ComplexF64}}}([[Z, Z]])
        coeffs_zz = [1.0]
        base_ham = QuantumFurnace._construct_base_ham(terms_zz, coeffs_zz, n_ising;
                                                      periodic=true)
        rescaling_factor, shift = QuantumFurnace._rescaling_and_shift_factors(base_ham)
        d_ising = 2^n_ising
        rescaled_mat = Matrix(base_ham) ./ rescaling_factor .+ shift * I(d_ising)
        eigvals_rs, eigvecs_rs = eigen(Hermitian(rescaled_mat))

        raw = (
            matrix = rescaled_mat,
            terms = terms_zz,
            base_coeffs = coeffs_zz ./ rescaling_factor,
            disordering_terms = nothing,
            disordering_coeffs = nothing,
            eigvals = eigvals_rs,
            eigvecs = eigvecs_rs,
            nu_min = minimum(diff(eigvals_rs)),
            shift = shift,
            rescaling_factor = rescaling_factor,
            periodic = true,
        )
        β_phys = 0.5
        ham_ising = HamHam(raw; beta_phys=β_phys)
        jumps_ising = QuantumFurnace._jumps_in_basis(n_ising, ham_ising.eigvecs)
        β_alg = beta_alg(ham_ising, β_phys)

        σ = 1.0 / β_alg
        H_norm = maximum(abs, ham_ising.eigvals)
        omega_range = 2.0 * (H_norm + 8 * σ)
        r_D = 7
        w0_D = omega_range / 2.0^r_D
        t0_D = 2π / (2.0^r_D * w0_D)
        cfg = Config(
            sim = Lindbladian(), domain = EnergyDomain(), construction = KMS(),
            num_qubits = n_ising, with_linear_combination = true,
            beta = β_alg, beta_phys = β_phys, sigma = σ,
            a = 0.0, s = 0.25, gaussian_parameters = (nothing, nothing),
            num_energy_bits_D = r_D, w0_D = w0_D, t0_D = t0_D,
            num_trotter_steps_per_t0 = 10, filter = nothing,
        )

        # Dense reference gap from construct_lindbladian
        L_dense = construct_lindbladian(jumps_ising, cfg, ham_ising)
        ev_dense = eigvals(Matrix(L_dense))
        perm = sortperm(real.(ev_dense); by=abs)
        gap_dense = abs(real(ev_dense[perm[2]]))

        # Enable the separate gap solve for the symmetric initial state I/d.
        rho_0 = Matrix{ComplexF64}(I(d_ising) / d_ising)
        t_grid = collect(range(0.0, 100.0, length=21))
        traj = predict_lindbladian_trajectory(cfg, ham_ising, jumps_ising, rho_0, t_grid;
                                              krylovdim=40, compute_true_gap=true)
        # Compare the separate Krylov gap estimate with the dense reference.
        @test isapprox(traj.spectral_gap, gap_dense; rtol=1e-8)
        @info "classical Ising parity regression" n=n_ising β_phys=β_phys gap_predict=traj.spectral_gap gap_dense=gap_dense rel_err=abs(traj.spectral_gap - gap_dense)/gap_dense
    end
end
