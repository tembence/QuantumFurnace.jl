using LinearAlgebra: I, mul!, eigvals, svdvals, norm, tr, Hermitian, Diagonal, kron
using Random: MersenneTwister
using Test
using QuantumFurnace

# ---------------------------------------------------------------------------
# Test-only helpers
# ---------------------------------------------------------------------------

"""
    build_davies_2x2(omega, beta, gamma) -> NamedTuple

Single-qubit Davies thermal Lindbladian with `H = -(omega/2) sigma_z`,
KMS-DB jump rates `gamma_+ = gamma` (de-excitation, `sigma_+ = |0><1|`)
and `gamma_- = gamma * exp(-beta * omega)` (excitation, `sigma_- = |1><0|`).
Returns the full superoperator, the dissipator-only superoperator, the
matrix-free L and K closures, the Gibbs state, fractional-power diagonals,
the two KMS-DB rates, and the analytic decay rate `gamma_+ + gamma_-`.
"""
function build_davies_2x2(omega::T, beta::T, gamma::T) where {T<:Real}
    CT = Complex{T}

    # Hamiltonian and Pauli operators (CT-typed)
    sigma_z = CT[T(1)  T(0); T(0)  T(-1)]
    sigma_p = CT[T(0)  T(1); T(0)   T(0)]   # |0><1|
    sigma_m = CT[T(0)  T(0); T(1)   T(0)]   # |1><0|
    Id2     = Matrix{CT}(I, 2, 2)

    # H = -(omega/2) sigma_z  =>  E_0 = -omega/2 (ground), E_1 = +omega/2 (excited)
    H       = -(omega / 2) * sigma_z

    # Gibbs state (Boltzmann weights). Ground state |0> dominant.
    Z       = exp( beta * omega / 2) + exp(-beta * omega / 2)
    p0      = exp( beta * omega / 2) / Z
    p1      = exp(-beta * omega / 2) / Z
    sigma_beta        = Matrix{CT}(undef, 2, 2)
    sigma_beta       .= 0
    sigma_beta[1, 1]  = p0
    sigma_beta[2, 2]  = p1
    sigma_quarter     = T[p0^T(0.25), p1^T(0.25)]
    sigma_inv_quarter = T[p0^T(-0.25), p1^T(-0.25)]
    sigma_half        = T[p0^T(0.5),  p1^T(0.5)]

    gamma_plus  = gamma
    gamma_minus = gamma * exp(-beta * omega)
    decay_rate  = gamma_plus + gamma_minus

    # ---- Build dense superoperators (column-stacking vec convention). ------
    H_super  = -1im * (kron(Id2, H) - kron(transpose(H), Id2))

    JdJp = sigma_m * sigma_p   # = |1><1|
    D_plus = gamma_plus * (
                  kron(conj(sigma_p), sigma_p)
                - T(0.5) * (kron(Id2, JdJp) + kron(transpose(JdJp), Id2))
              )
    JdJm = sigma_p * sigma_m   # = |0><0|
    D_minus = gamma_minus * (
                  kron(conj(sigma_m), sigma_m)
                - T(0.5) * (kron(Id2, JdJm) + kron(transpose(JdJm), Id2))
              )

    L_super_D = Matrix{CT}(D_plus + D_minus)             # dissipator only
    L_super   = Matrix{CT}(H_super + L_super_D)          # full L = -i[H,.] + L_D

    function L_apply!(out::AbstractMatrix, x::AbstractMatrix)
        mul!(vec(out), L_super, vec(x))
        return out
    end

    function L_D_apply!(out::AbstractMatrix, x::AbstractMatrix)
        mul!(vec(out), L_super_D, vec(x))
        return out
    end

    # K(X) = D(sigma, L_D)(X)
    #     = sigma^{-1/4} L_D(sigma^{1/4} X sigma^{1/4}) sigma^{-1/4}
    work1 = Matrix{CT}(undef, 2, 2)
    work2 = Matrix{CT}(undef, 2, 2)
    function K_apply!(out::AbstractMatrix, x::AbstractMatrix)
        @inbounds for j in 1:2, i in 1:2
            work1[i, j] = sigma_quarter[i] * x[i, j] * sigma_quarter[j]
        end
        L_D_apply!(work2, work1)
        @inbounds for j in 1:2, i in 1:2
            out[i, j] = sigma_inv_quarter[i] * work2[i, j] * sigma_inv_quarter[j]
        end
        return out
    end

    return (
        L_super           = L_super,
        L_super_D         = L_super_D,
        L_apply!          = L_apply!,
        K_apply!          = K_apply!,
        sigma_beta        = sigma_beta,
        sigma_quarter     = sigma_quarter,
        sigma_inv_quarter = sigma_inv_quarter,
        sigma_half        = sigma_half,
        gamma_plus        = gamma_plus,
        gamma_minus       = gamma_minus,
        decay_rate        = decay_rate,
    )
end


"""
    _estimate_tau_mix_slope(t, dist, t_lo, t_hi) -> NamedTuple

Closed-form linear regression of `log(dist) vs t` on the window
`t in [t_lo, t_hi]`. Returns `tau_mix = -1 / slope` and the fit details.
"""
function _estimate_tau_mix_slope(t::AbstractVector{<:Real}, dist::AbstractVector{<:Real},
                                 t_lo::Real, t_hi::Real)
    ix = findall(τ -> t_lo <= τ <= t_hi, t)
    @assert length(ix) >= 5  "fit window too small (need at least 5 points)"
    x = Float64.(@view t[ix])
    y = log.(Float64.(@view dist[ix]))
    @assert all(isfinite, y)  "log(dist) is non-finite in fit window — distance hit floor"
    x_bar = sum(x) / length(x)
    y_bar = sum(y) / length(y)
    cov_xy = sum((x .- x_bar) .* (y .- y_bar))
    var_x  = sum((x .- x_bar) .^ 2)
    slope     = cov_xy / var_x
    intercept = y_bar - slope * x_bar
    return (tau_mix = -1.0 / slope, slope = slope, intercept = intercept,
            ix_lo = first(ix), ix_hi = last(ix))
end


@testset "Lindbladian-action ODE integrator" begin

    # -----------------------------------------------------------------------
    # (a) Davies single-qubit toy (analytic τ_mix)
    # -----------------------------------------------------------------------
    @testset "(a) Davies single-qubit τ_mix matches analytic" begin
        omega = 1.0
        beta  = 2.0
        gamma = 0.5

        toy              = build_davies_2x2(omega, beta, gamma)
        tau_mix_analytic = 1.0 / toy.decay_rate
        t_max            = 10.0 / toy.decay_rate
        n_grid           = 121
        t_grid           = collect(range(0.0, t_max, length = n_grid))

        # |1><1| is diagonal — leading decay rate is exactly
        # gamma_+ + gamma_- (population/T_1 sector). Off-diagonal initial
        # state would bring in slower T_2 modes plus iw rotation.
        rho_0 = Matrix{ComplexF64}(undef, 2, 2)
        rho_0 .= 0
        rho_0[2, 2] = 1.0

        # K-mode initial state: psi_0 = sigma^{-1/4} rho_0 sigma^{-1/4}.
        # For diagonal rho_0 = |1><1|, psi_0 is diagonal too.
        psi_0 = Matrix{ComplexF64}(undef, 2, 2)
        psi_0 .= 0
        psi_0[2, 2] = toy.sigma_inv_quarter[2] * 1.0 * toy.sigma_inv_quarter[2]

        # K-mode equilibrium target: psi_eq = sigma^{-1/4} sigma sigma^{-1/4}
        # = diag(p_i^{1/2}) = sigma^{1/2}.
        psi_eq = Matrix{ComplexF64}(Diagonal(toy.sigma_half))

        res_L = lindblad_action_integrate(
            toy.L_apply!, rho_0, Matrix{ComplexF64}(toy.sigma_beta), t_grid;
            krylovdim = 8, tol = 1e-12, save_states = true,
        )
        res_K = discriminant_action_integrate(
            toy.K_apply!, psi_0, psi_eq, t_grid;
            krylovdim = 8, tol = 1e-12, is_hermitian = true, save_states = true,
        )

        @test res_L.all_converged
        @test res_K.all_converged

        # Slope-fit on the asymptotic window.
        fit_L = _estimate_tau_mix_slope(res_L.t, res_L.distances, t_max / 3, 2 * t_max / 3)
        fit_K = _estimate_tau_mix_slope(res_K.t, res_K.distances, t_max / 3, 2 * t_max / 3)

        @test isapprox(fit_L.tau_mix, tau_mix_analytic; rtol = 1e-3)
        @test isapprox(fit_K.tau_mix, tau_mix_analytic; rtol = 1e-3)
        @info "Davies τ_mix" analytic=tau_mix_analytic L=fit_L.tau_mix K=fit_K.tau_mix

        # Trajectory equivalence: rho^L_i ≈ sigma^{1/4} psi^K_i sigma^{1/4}.
        Sq = Diagonal(toy.sigma_quarter)
        for i in 1:length(t_grid)
            rho_i      = res_L.states[i]
            psi_i      = res_K.states[i]
            rho_from_K = Matrix{ComplexF64}(Sq * psi_i * Sq)
            @test norm(rho_i - rho_from_K) < 1e-10
        end

        # L-mode invariants.
        for i in 1:length(t_grid)
            rho_i = res_L.states[i]
            @test isapprox(real(tr(rho_i)), 1.0; rtol = 1e-10)
            evs = eigvals(Hermitian((rho_i + rho_i') / 2))
            @test minimum(real.(evs)) >= -1e-10
        end

        # Mid-time direct dense-expv cross-check.
        i_mid       = 60
        rho_mid     = res_L.states[i_mid]
        rho_dense_v = exp(t_grid[i_mid] * toy.L_super) * vec(rho_0)
        rho_dense   = reshape(rho_dense_v, 2, 2)
        @test norm(rho_mid - rho_dense) < 1e-9
    end

    # -----------------------------------------------------------------------
    # (b) Trivial sanity: equilibrium initial state and t=0 distance
    # -----------------------------------------------------------------------
    @testset "(b) Trivial sanity: equilibrium and t=0 distance" begin
        omega = 1.0
        beta  = 2.0
        gamma = 0.5
        toy   = build_davies_2x2(omega, beta, gamma)

        t_grid = collect(range(0.0, 1.0, length = 11))

        # ρ_0 = σ_β: Davies preserves σ_β exactly, so trace distance stays 0.
        sigma_b = Matrix{ComplexF64}(toy.sigma_beta)
        res_eq  = lindblad_action_integrate(
            toy.L_apply!, copy(sigma_b), sigma_b, t_grid;
            krylovdim = 8, tol = 1e-12,
        )
        @test all(res_eq.distances .< 1e-12)

        # ρ_0 = |1><1|: distance at t=0 is the trace distance of (ρ_0 - σ_β).
        rho_0 = Matrix{ComplexF64}(undef, 2, 2)
        rho_0 .= 0
        rho_0[2, 2] = 1.0
        res_t0 = lindblad_action_integrate(
            toy.L_apply!, rho_0, sigma_b, t_grid;
            krylovdim = 8, tol = 1e-12,
        )
        expected = 0.5 * sum(svdvals(rho_0 - sigma_b))
        @test isapprox(res_t0.distances[1], expected; atol = 1e-14)
    end

    # -----------------------------------------------------------------------
    # (c) save_states=true shape and save_states=false default
    # -----------------------------------------------------------------------
    @testset "(c) save_states shape" begin
        omega = 1.0
        beta  = 2.0
        gamma = 0.5
        toy   = build_davies_2x2(omega, beta, gamma)
        t_grid = collect(range(0.0, 1.0, length = 11))

        rho_0 = Matrix{ComplexF64}(undef, 2, 2)
        rho_0 .= 0
        rho_0[2, 2] = 1.0
        sigma_b = Matrix{ComplexF64}(toy.sigma_beta)

        psi_0 = Matrix{ComplexF64}(undef, 2, 2)
        psi_0 .= 0
        psi_0[2, 2] = toy.sigma_inv_quarter[2] * 1.0 * toy.sigma_inv_quarter[2]
        psi_eq = Matrix{ComplexF64}(Diagonal(toy.sigma_half))

        # save_states=true
        res_L_t = lindblad_action_integrate(
            toy.L_apply!, rho_0, sigma_b, t_grid;
            krylovdim = 8, tol = 1e-12, save_states = true,
        )
        @test res_L_t.states isa Vector{Matrix{ComplexF64}}
        @test length(res_L_t.states) == 11
        @test res_L_t.states[1] ≈ rho_0

        res_K_t = discriminant_action_integrate(
            toy.K_apply!, psi_0, psi_eq, t_grid;
            krylovdim = 8, tol = 1e-12, save_states = true,
        )
        @test res_K_t.states isa Vector{Matrix{ComplexF64}}
        @test length(res_K_t.states) == 11
        @test res_K_t.states[1] ≈ psi_0

        # save_states=false (default) -> empty vector
        res_L_f = lindblad_action_integrate(
            toy.L_apply!, rho_0, sigma_b, t_grid;
            krylovdim = 8, tol = 1e-12,
        )
        @test res_L_f.states == Matrix{ComplexF64}[]

        res_K_f = discriminant_action_integrate(
            toy.K_apply!, psi_0, psi_eq, t_grid;
            krylovdim = 8, tol = 1e-12,
        )
        @test res_K_f.states == Matrix{ComplexF64}[]
    end

    # -----------------------------------------------------------------------
    # (d) Allocation-free closure path (regression guard)
    # -----------------------------------------------------------------------
    # Spec asks "< 1024" for the body; the integrator wrapper allocates
    # trajectory arrays + per-step Krylov subspace vectors (each
    # `copy(vec(out_buf))` returned to KrylovKit is a fresh allocation, by
    # design, to dodge KrylovKit aliasing). On Julia 1.12 the realistic
    # envelope for a 3-step d=2 run is ~32 KiB. We use 64 KiB as a
    # regression guard against genuine leakage rather than a strict
    # body-only bound.
    @testset "(d) noop closure: bounded allocations" begin
        omega = 1.0
        beta  = 2.0
        gamma = 0.5
        toy   = build_davies_2x2(omega, beta, gamma)

        # Non-allocating identity closure (proves any allocs come from the
        # integrator / KrylovKit, not the user closure).
        noop_action!(o, x) = (copyto!(o, x); o)

        t_grid_short = collect(range(0.0, 0.1, length = 4))   # 3 steps
        rho_0        = Matrix{ComplexF64}(undef, 2, 2)
        rho_0       .= 0
        rho_0[1, 1]  = 0.5
        rho_0[2, 2]  = 0.5
        sigma_b      = Matrix{ComplexF64}(toy.sigma_beta)

        # Warmup -- compile method specialisations.
        lindblad_action_integrate(
            noop_action!, rho_0, sigma_b, t_grid_short;
            krylovdim = 4, tol = 1e-10,
        )

        allocs = @allocated lindblad_action_integrate(
            noop_action!, rho_0, sigma_b, t_grid_short;
            krylovdim = 4, tol = 1e-10,
        )
        @test allocs < 65536
        @info "lindblad_action_integrate noop allocations" allocs
    end

    # -----------------------------------------------------------------------
    # (e) CKG BohrDomain end-to-end @ n=3, β=10 (mode=:L)
    # -----------------------------------------------------------------------
    @testset "(e) CKG BohrDomain end-to-end @ n=3, β=10 (mode=:L)" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        config = make_config(Lindbladian(), BohrDomain();
                             num_qubits=3, construction=KMS())
        d = size(sys.ham.data, 1)
        rho_0 = Matrix{ComplexF64}(I(d) / d)
        # t=120 is about four mixing times on this fixture and resolves
        # the equilibrium tail at the 1e-6 distance threshold.
        t_grid = collect(range(0.0, 120.0, length=121))

        res = integrate_to_gibbs(config, sys.ham, sys.jumps, rho_0, t_grid;
                                  mode=:L, krylovdim=20, tol=1e-10)

        @test res.all_converged
        @test res.distances[end] < 1e-6
        @test isapprox(real(tr(res.rho_final)), 1.0; atol=1e-9)
        evs = eigvals(Hermitian((res.rho_final + res.rho_final') / 2))
        @test minimum(real.(evs)) > -1e-9

        est = estimate_mixing_time(res; model=:biexp,
                                    target_epsilon=1e-3, extrapolate=true)
        @test isfinite(est.mixing_time)
        @test est.mixing_time > 0
        @info "(e) CKG BohrDomain L-mode" tau_mix=est.mixing_time r2=est.r_squared dist_end=res.distances[end]
    end

    # -----------------------------------------------------------------------
    # (f) CKG BohrDomain end-to-end @ n=3, β=10 (mode=:K)
    # -----------------------------------------------------------------------
    @testset "(f) CKG BohrDomain end-to-end @ n=3, β=10 (mode=:K)" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        config = make_config(Lindbladian(), BohrDomain();
                             num_qubits=3, construction=KMS())
        d = size(sys.ham.data, 1)
        rho_0 = Matrix{ComplexF64}(I(d) / d)
        # Use the same mixing-time horizon as in (e).
        t_grid = collect(range(0.0, 120.0, length=121))

        res = integrate_to_gibbs(config, sys.ham, sys.jumps, rho_0, t_grid;
                                  mode=:K, krylovdim=20, tol=1e-10)

        @test res.all_converged
        # KMS-DB ⇒ K is HS-self-adjoint ⇒ chi² is Lyapunov in
        # continuum (monotone non-increasing). Krylov truncation at tol=1e-10
        # introduces O(1e-9) violations of strict monotonicity at machine
        # precision; relax by 1e-9.
        @test all(diff(res.distances) .<= 1e-9)
        @test res.distances[end]^2 < 1e-12
        @info "(f) CKG BohrDomain K-mode" dist_end=res.distances[end] dist_end_sq=res.distances[end]^2 max_diff=maximum(diff(res.distances))
    end

    # -----------------------------------------------------------------------
    # (g) DLL BohrDomain @ n=3, β=10 with DLLGaussianFilter (mode=:L)
    # -----------------------------------------------------------------------
    @testset "(g) DLL BohrDomain end-to-end @ n=3, β=10, DLLGaussianFilter (mode=:L)" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        # make_config does not forward `filter`; build Config directly.
        # Parameter values mirror make_config(Lindbladian(), BohrDomain(); construction=DLL()).
        config = Config(;
            sim = Lindbladian(),
            domain = BohrDomain(),
            construction = DLL(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta,
            sigma = 1.0 / beta,
            a = beta / 30.0,
            s = 0.4,
            num_energy_bits = 12,
            w0 = 0.05,
            t0 = 2pi / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
            filter = DLLGaussianFilter(beta),
        )
        d = size(sys.ham.data, 1)
        rho_0 = Matrix{ComplexF64}(I(d) / d)
        # DLL ν-window is narrower than CKG; thermalisation is
        # slower. t=100 (51 grid points, dt=2.0) gives dist[end] ~5e-10, well
        # below the 1e-7 threshold (sharpened from the plan's provisional 1e-3).
        t_grid = collect(range(0.0, 100.0, length=51))

        res = integrate_to_gibbs(config, sys.ham, sys.jumps, rho_0, t_grid;
                                  mode=:L, krylovdim=20, tol=1e-10)

        @test res.all_converged
        @test res.distances[end] < 1e-7
        @test isapprox(real(tr(res.rho_final)), 1.0; atol=1e-9)
        evs = eigvals(Hermitian((res.rho_final + res.rho_final') / 2))
        @test minimum(real.(evs)) > -1e-9
        @info "(g) DLL Gaussian L-mode" dist_end=res.distances[end]
    end

    # -----------------------------------------------------------------------
    # (h) DLL BohrDomain @ n=3, β=10 with DLLMetropolisFilter (mode=:L)
    # -----------------------------------------------------------------------
    @testset "(h) DLL BohrDomain end-to-end @ n=3, β=10, DLLMetropolisFilter (mode=:L)" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        config = Config(;
            sim = Lindbladian(),
            domain = BohrDomain(),
            construction = DLL(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta,
            sigma = 1.0 / beta,
            a = beta / 30.0,
            s = 0.4,
            num_energy_bits = 12,
            w0 = 0.05,
            t0 = 2pi / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
            filter = DLLMetropolisFilter(beta),
        )
        d = size(sys.ham.data, 1)
        rho_0 = Matrix{ComplexF64}(I(d) / d)
        t_grid = collect(range(0.0, 100.0, length=51))

        res = integrate_to_gibbs(config, sys.ham, sys.jumps, rho_0, t_grid;
                                  mode=:L, krylovdim=20, tol=1e-10)

        @test res.all_converged
        @test res.distances[end] < 1e-7
        @test isapprox(real(tr(res.rho_final)), 1.0; atol=1e-9)
        evs = eigvals(Hermitian((res.rho_final + res.rho_final') / 2))
        @test minimum(real.(evs)) > -1e-9
        @info "(h) DLL Metropolis L-mode" dist_end=res.distances[end]
    end

    # -----------------------------------------------------------------------
    # (i) Sanity benchmark: ODE-integrator τ_mix vs Thermalize-extrapolation τ_mix
    #
    # This file is in the NO_SANDBOX tier because the Trotter-vs-continuous
    # τ_mix agreement at the 10% relative-error level needs δ=0.001 over
    # mixing_time=50 → 50 000
    # run_thermalize steps, ~150 s on the sandbox container. Loosening δ or
    # the horizon would push the rel_err past the 10% physics-check
    # threshold (Trotter weak error scales O(δ·τ_mix)).
    # -----------------------------------------------------------------------
    @testset "(i) Sanity benchmark vs Thermalize @ n=3, β=10 [NO_SANDBOX]" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        d = size(sys.ham.data, 1)
        rho_0 = Matrix{ComplexF64}(I(d) / d)

        # Integrator side (continuous-time, exact L).
        config_int = make_config(Lindbladian(), BohrDomain();
                                  num_qubits=3, construction=KMS())
        t_grid = collect(range(0.0, 60.0, length=61))
        res_int = integrate_to_gibbs(config_int, sys.ham, sys.jumps, rho_0, t_grid;
                                      mode=:L, krylovdim=20, tol=1e-10)
        est_int = estimate_mixing_time(res_int; model=:biexp,
                                        target_epsilon=0.01, extrapolate=true)

        # Thermalize side (discrete CPTP channel, δ-step Trotterization).
        config_therm = make_config(Thermalize(), BohrDomain();
                                    num_qubits=3, mixing_time=50.0, delta=0.001)
        res_therm = run_thermalize(sys.jumps, config_therm, sys.ham;
                                    initial_dm=copy(rho_0))
        est_therm = estimate_mixing_time(res_therm; model=:biexp,
                                          target_epsilon=0.01, extrapolate=true)

        @test isfinite(est_int.mixing_time)
        @test isfinite(est_therm.mixing_time)
        rel_err = abs(est_int.mixing_time - est_therm.mixing_time) / est_therm.mixing_time
        # 10% accommodates Trotter (O(δ·τ_mix) per-step error) +
        # bi-exp fit noise on both sides. Tighter would catch noise; looser
        # would miss factor-of-2+ qualitative bugs.
        @test rel_err < 0.10
        @info "(i) τ_mix sanity benchmark" tau_int=est_int.mixing_time tau_therm=est_therm.mixing_time rel_err=rel_err
    end

    # -----------------------------------------------------------------------
    # (j) Single-point sweep at n=3, β=10
    # -----------------------------------------------------------------------
    @testset "(j) Single-point sweep (:krylov eigenmode schema)" begin
        # Production default for thesis numerics is method=:krylov.
        # Schema: gap_est, mixing_time, mixing_time_source ∈ {:extrapolated,
        # :floor, :nan}, floor_distance. NO fitted_gap / r_squared /
        # converged_fit on this path.
        results = sweep_mixing_times([3], [10.0];
                                      method=:krylov, mode=:L, seeds=[42],
                                      use_threads=false, t_max_factor=5.0)
        @test length(results) == 1
        r = results[1]
        @test r.n == 3
        @test r.beta == 10.0
        @test r.seed == 42
        @test r.method === :krylov
        @test r.mixing_time_source === :extrapolated
        @test isfinite(r.mixing_time) && r.mixing_time > 0
        @test r.gap_est > 0
        # Floor sanity: the asymptotic ‖ρ_inf - σ_β‖_1 / 2 must be below
        # target_eps for extrapolation to be well-defined.
        @test isfinite(r.floor_distance) && r.floor_distance < r.target_epsilon
        @info "(j) Single-point sweep (:krylov)" tau=r.mixing_time gap=r.gap_est floor=r.floor_distance source=r.mixing_time_source
    end

    @testset "(j-ode) Legacy :ode + biexp schema preserved" begin
        # The :ode route still emits the legacy biexp diagnostics — guard
        # against accidental schema regression.
        results = sweep_mixing_times([3], [10.0];
                                      method=:ode, mode=:L, seeds=[42],
                                      use_threads=false, t_max_factor=5.0)
        r = results[1]
        @test r.method === :ode
        @test isfinite(r.mixing_time) && r.mixing_time > 0
        @test r.fitted_gap > 0
        @test r.r_squared > 0.9
        @test r.all_converged == true
        @info "(j-ode) Single-point sweep (:ode)" tau=r.mixing_time gap=r.fitted_gap r2=r.r_squared
    end

    # -----------------------------------------------------------------------
    # (k) Multi-point sweep over β, threaded
    # -----------------------------------------------------------------------
    @testset "(k) Multi-point sweep over beta, threaded" begin
        results = sweep_mixing_times([3], [1.0, 5.0, 10.0]; mode=:L, seeds=[42],
                                      use_threads=true, t_max_factor=5.0)
        @test length(results) == 3
        @test sort([r.beta for r in results]) == [1.0, 5.0, 10.0]
        sorted_by_beta = sort(results; by=r -> r.beta)
        @test all(r -> r.n == 3 && r.seed == 42, sorted_by_beta)
        @test all(r -> isfinite(r.mixing_time) && r.mixing_time > 0, sorted_by_beta)
        @test all(r -> r.gap_est > 0, sorted_by_beta)
        @info "(k) Multi-point sweep" taus=[r.mixing_time for r in sorted_by_beta] betas=[r.beta for r in sorted_by_beta]
    end

    # -----------------------------------------------------------------------
    # (l) Persistence: skip_existing works
    # -----------------------------------------------------------------------
    @testset "(l) Persistence: skip_existing" begin
        mktempdir() do dir
            results1 = sweep_mixing_times([3], [10.0]; mode=:L,
                                           output_dir=dir,
                                           skip_existing=true,
                                           use_threads=false,
                                           t_max_factor=5.0)
            @test length(readdir(dir)) >= 1
            @test isfinite(results1[1].mixing_time)
            results2 = sweep_mixing_times([3], [10.0]; mode=:L,
                                           output_dir=dir,
                                           skip_existing=true,
                                           use_threads=false,
                                           t_max_factor=5.0)
            @test isequal(Dict(pairs(results2[1])), Dict(pairs(results1[1])))
        end
    end

    # -----------------------------------------------------------------------
    # adaptive t_max_factor + observed-mixing fallback
    # -----------------------------------------------------------------------
    @testset "(l1) Adaptive t_max_factor scales with target_epsilon" begin
        # heuristic factor = max(5.0, 1.5 * log10(1/eps))
        # gives 5 at 1e-3 (legacy), 9 at 1e-6, 14 at 1e-9. Verify by
        # inspecting `t_max_factor` field of the returned NamedTuple.
        for (eps, expected_factor) in [(1e-3, 5.0), (1e-6, 9.0), (1e-9, 13.5)]
            res = sweep_mixing_times([3], [10.0];
                                      mode=:L, seeds=[42], use_threads=false,
                                      target_epsilon=eps,
                                      t_max_factor=:auto)
            @test isapprox(res[1].t_max_factor, expected_factor; atol=1e-9)
            @test res[1].t_max ≈ res[1].t_max_factor / res[1].gap_est
        end
    end

    @testset "(l2) Numeric t_max_factor override still works (:ode)" begin
        # Pin to :ode since the {:extrapolated, :observed} enum is the
        # biexp path's source set. The :krylov path uses
        # {:extrapolated, :floor, :nan}.
        res = sweep_mixing_times([3], [10.0];
                                  method=:ode, mode=:L, seeds=[42],
                                  use_threads=false, t_max_factor=5.0)
        @test isapprox(res[1].t_max_factor, 5.0)
        @test res[1].mixing_time_source ∈ (:extrapolated, :observed)
    end

    @testset "(l4) param_table_bson threads ideal-Lindbladian recipe" begin
        # When the ideal-Lindbladian table is provided, CKG / EnergyDomain cells
        # pick (r_D, w0_D, t0_D) from the row matching (n, β, eps, filter_kind)
        # rather than the legacy hardcoded (12, 0.05). Smooth-Metro saturates at
        # machine precision by r_D = 6, so τ_mix at r_D = 7 (recipe) and r_D = 12
        # (legacy) must agree to many digits.
        param_table = QuantumFurnace._package_data_path(
            "ideal_lindbladian_param_table.bson")
        ham_file     = (n) -> "heis_xxx_disordered_periodic_n$(n)_seed46.bson"

        ham_path = test_hamiltonian_path(3)
        isfile(param_table) || error(
            "missing required ideal-Lindbladian parameter table: $param_table")
        isfile(ham_path) || error("missing required Hamiltonian fixture: $ham_path")

        res_legacy = sweep_mixing_times([3], [10.0];
            construction = KMS(), domain = EnergyDomain(), method = :krylov,
            target_epsilon = 1e-3,
            hamiltonian_filename = ham_file,
            seeds = [42], use_threads = false, output_dir = nothing,
        )
        res_table = sweep_mixing_times([3], [10.0];
            construction = KMS(), domain = EnergyDomain(), method = :krylov,
            target_epsilon = 1e-3,
            hamiltonian_filename = ham_file,
            param_table_bson = param_table, filter_kind = :smooth_metro,
            seeds = [42], use_threads = false, output_dir = nothing,
        )
        @test length(res_legacy) == 1 && length(res_table) == 1
        r_legacy, r_table = res_legacy[1], res_table[1]

        # Schema: new fields populated.
        @test r_table.r_D == 7
        @test isapprox(r_table.w0_D, 0.01953125, rtol=1e-12)
        @test r_table.target_epsilon == 1e-3
        @test r_table.filter_kind === :smooth_metro
        @test isfinite(r_table.tau_mix_bound)

        # Legacy fallback still hits the hardcoded defaults.
        @test r_legacy.r_D == 12
        @test r_legacy.w0_D == 0.05

        # τ_mix from the two configurations must agree to better than 1e-5
        # (smooth-Metro is at machine precision in quadrature by r_D = 6).
        @test isapprox(r_legacy.mixing_time, r_table.mixing_time, rtol=1e-5)
        # The :krylov path emits `gap_est`
        # (sourced from the predictor's eigendecomposition) instead of
        # the legacy `fitted_gap`. Both rows are :krylov here.
        @test isapprox(r_legacy.gap_est, r_table.gap_est, rtol=1e-6)
    end

    @testset "(l5) Eigenmode τ_mix consistency vs :ode + biexp on healthy cell" begin
        # n=3, β=10, ε=1e-3 is a clean cell (gap_est ≈ 0.23, fitted_gap from
        # biexp is within ~20% of it on healthy cells). Eigenmode and biexp
        # should agree on τ_mix within fitting noise.
        res_kr = sweep_mixing_times([3], [10.0]; method=:krylov, mode=:L,
                                      seeds=[42], use_threads=false,
                                      target_epsilon=1e-3, t_max_factor=:auto)
        res_ode = sweep_mixing_times([3], [10.0]; method=:ode, mode=:L,
                                      seeds=[42], use_threads=false,
                                      target_epsilon=1e-3, t_max_factor=:auto)
        τ_eig = res_kr[1].mixing_time
        τ_biexp = res_ode[1].mixing_time
        @test isfinite(τ_eig) && τ_eig > 0
        @test isfinite(τ_biexp) && τ_biexp > 0
        @test isapprox(τ_eig, τ_biexp; rtol=0.10)   # 10% tail-fit noise budget
        # gap_est on both paths should agree exactly (both come from
        # Krylov-Arnoldi eigsolve, just routed through different functions).
        @test isapprox(res_kr[1].gap_est, res_ode[1].gap_est; rtol=1e-6)
        @info "(l5) τ_mix krylov vs ode+biexp" τ_eig τ_biexp rel_err=abs(τ_eig - τ_biexp)/τ_biexp
    end

    @testset "(l6) Eigenmode τ_mix on tight-ε / floor-regime cell" begin
        # At ε=1e-6 and the canonical n=3 fixture, the Lindbladian
        # asymptotic floor (‖ρ_inf - σ_β‖_1 / 2 from the captured Krylov
        # subspace) is well below 1e-6 (KMS-DB Lindbladian fixed point
        # equals the Gibbs state up to truncation error). Eigenmode τ_mix
        # should be :extrapolated and finite — which is exactly the regime
        # where biexp on :ode tends to NaN out.
        res_kr = sweep_mixing_times([3], [10.0]; method=:krylov, mode=:L,
                                      seeds=[42], use_threads=false,
                                      target_epsilon=1e-6, t_max_factor=:auto)
        r = res_kr[1]
        @test r.mixing_time_source === :extrapolated
        @test isfinite(r.mixing_time) && r.mixing_time > 0
        @test r.gap_est > 0
        @test r.floor_distance < r.target_epsilon
        @info "(l6) tight-ε eigenmode τ_mix" τ=r.mixing_time gap=r.gap_est floor=r.floor_distance source=r.mixing_time_source
    end

    # -----------------------------------------------------------------------
    # Matrix-free DLL apply_lindbladian!
    # -----------------------------------------------------------------------
    @testset "(m) Matrix-free DLL agreement vs dense Liouvillian" begin
        rng = MersenneTwister(0xCAFE)
        for beta in (1.0, 5.0, 10.0)
            sys = make_dll_n3_system(beta)
            ham = sys.ham; jumps = sys.jumps
            for filter_T in (DLLGaussianFilter, DLLMetropolisFilter)
                filter = filter_T(beta)
                config = Config(
                    sim = Lindbladian(), domain = BohrDomain(), construction = DLL(),
                    num_qubits = 3, with_linear_combination = false,
                    beta = beta, sigma = 1.0 / beta, filter = filter,
                )
                # Dense reference
                L_dense = Matrix{ComplexF64}(construct_lindbladian(jumps, config, ham))
                # Matrix-free workspace
                ws = Workspace(config, ham, jumps)
                d = size(ham.data, 1)
                for trial in 1:5
                    R = randn(rng, ComplexF64, d, d)
                    rho = (R + R') / 2
                    rho ./= tr(rho)
                    out_mf = copy(apply_lindbladian!(ws, Matrix{ComplexF64}(rho), config, ham))
                    out_dense = reshape(L_dense * vec(rho), d, d)
                    rel = norm(out_mf - out_dense) / max(norm(out_dense), 1e-30)
                    @test rel < 1e-10
                end
            end
        end
    end

    @testset "(n) DLL Gibbs fixed-point spot-check (n=3)" begin
        for beta in (1.0, 10.0), filter_T in (DLLGaussianFilter, DLLMetropolisFilter)
            sys = make_dll_n3_system(beta)
            ham = sys.ham; jumps = sys.jumps
            filter = filter_T(beta)
            config = Config(
                sim = Lindbladian(), domain = BohrDomain(), construction = DLL(),
                num_qubits = 3, with_linear_combination = false,
                beta = beta, sigma = 1.0 / beta, filter = filter,
            )
            ws = Workspace(config, ham, jumps)
            gibbs_mat = Matrix{ComplexF64}(ham.gibbs)
            out = copy(apply_lindbladian!(ws, gibbs_mat, config, ham))
            @test norm(out) < 1e-9
        end
    end

    @testset "(o) DLL apply allocation guard" begin
        sys = make_dll_n3_system(10.0)
        ham = sys.ham; jumps = sys.jumps
        filter = DLLGaussianFilter(10.0)
        config = Config(
            sim = Lindbladian(), domain = BohrDomain(), construction = DLL(),
            num_qubits = 3, with_linear_combination = false,
            beta = 10.0, sigma = 0.1, filter = filter,
        )
        ws = Workspace(config, ham, jumps)
        rho = Matrix{ComplexF64}(ham.gibbs)
        # Warmup
        apply_lindbladian!(ws, rho, config, ham)
        allocs = @allocated apply_lindbladian!(ws, rho, config, ham)
        @info "DLL apply_lindbladian! allocations" allocs
        @test allocs < 1024
    end

    @testset "(p) DLL adjoint duality tr(X' L(Y)) == tr(L*(X)' Y)" begin
        rng = MersenneTwister(0xBEEF)
        sys = make_dll_n3_system(10.0)
        ham = sys.ham; jumps = sys.jumps
        filter = DLLGaussianFilter(10.0)
        config = Config(
            sim = Lindbladian(), domain = BohrDomain(), construction = DLL(),
            num_qubits = 3, with_linear_combination = false,
            beta = 10.0, sigma = 0.1, filter = filter,
        )
        ws_fwd = Workspace(config, ham, jumps)
        ws_adj = Workspace(config, ham, jumps)
        d = size(ham.data, 1)
        for trial in 1:3
            X = randn(rng, ComplexF64, d, d)
            Y = randn(rng, ComplexF64, d, d)
            LY = copy(apply_lindbladian!(ws_fwd, Y, config, ham))
            LstarX = copy(apply_adjoint_lindbladian!(ws_adj, X, config, ham))
            lhs = tr(X' * LY)
            rhs = tr(LstarX' * Y)
            @test isapprox(lhs, rhs; rtol=1e-10)
        end
    end

    # -----------------------------------------------------------------------
    # Sweep harness gap-estimate uses matrix-free Krylov (regression guard
    # against the n>5 dense-Liouvillian cliff)
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # CKG EnergyDomain agreement with BohrDomain
    # -----------------------------------------------------------------------
    # Production sweeps switch from BohrDomain (O(d⁴) Bohr-pair scaling) to
    # EnergyDomain (O(N · d²) where N = 2^num_energy_bits is fixed). The two
    # representations of the CKG KMS-DB Lindbladian are equivalent in the
    # continuum; the EnergyDomain Riemann
    # sum has an exponentially small quadrature error at default settings
    # (σ=1/β=0.1, w0=0.05, num_energy_bits=12 ⇒ N=4096). At n=3,4,5 we expect
    # agreement at the FP-accumulation floor (~1e-9 relative on a ||L||~1
    # operator) — orders of magnitude tighter than the physics-meaningful
    # threshold for τ_mix downstream.
    #
    # Use smooth Metropolis with a=0, s=0.25 for this comparison.
    @testset "(q0) CKG EnergyDomain ≈ BohrDomain dense Liouvillian (n=3,4,5)" begin
        rel_threshold = 1e-9
        for n in (3, 4, 5)
            beta = 10.0
            sys = make_test_system(; num_qubits = n)
            ham = sys.hamiltonian
            jumps = sys.jumps
            base_kw = (
                sim = Lindbladian(),
                construction = KMS(),
                num_qubits = n,
                with_linear_combination = true,
                beta = beta,
                sigma = 1.0 / beta,
                a = 0.0,
                s = 0.25,
                num_energy_bits = 12,
                w0 = 0.05,
                t0 = 2π / (2^12 * 0.05),
                num_trotter_steps_per_t0 = 10,
            )
            cfg_b = Config(; domain = BohrDomain(),    base_kw...)
            cfg_e = Config(; domain = EnergyDomain(),  base_kw...)
            L_b = Matrix{ComplexF64}(construct_lindbladian(jumps, cfg_b, ham))
            L_e = Matrix{ComplexF64}(construct_lindbladian(jumps, cfg_e, ham))
            nb = opnorm(L_b)
            rel = opnorm(L_b - L_e) / nb
            @test rel < rel_threshold
            @info "(q0) Bohr ≈ Energy dense ‖·‖_op" n β=beta norm_Bohr=nb rel_diff=rel threshold=rel_threshold
        end
    end

    # -----------------------------------------------------------------------
    # CKG EnergyDomain end-to-end mixing (n=3, β=10)
    # -----------------------------------------------------------------------
    @testset "(q1) CKG EnergyDomain end-to-end @ n=3, β=10 (mode=:L)" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        config = Config(;
            sim = Lindbladian(),
            domain = EnergyDomain(),
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta,
            sigma = 1.0 / beta,
            a = 0.0,
            s = 0.25,
            num_energy_bits = 12,
            w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
        )
        d = size(sys.ham.data, 1)
        rho_0 = Matrix{ComplexF64}(I(d) / d)
        # same horizon as test (e). EnergyDomain CKG should
        # converge to Gibbs to the same precision (matvec-equivalent dense
        # Liouvillians per (q0)).
        t_grid = collect(range(0.0, 60.0, length=61))

        res = integrate_to_gibbs(config, sys.ham, sys.jumps, rho_0, t_grid;
                                  mode=:L, krylovdim=20, tol=1e-10)

        @test res.all_converged
        @test res.distances[end] < 1e-6
        @test isapprox(real(tr(res.rho_final)), 1.0; atol=1e-9)
        evs = eigvals(Hermitian((res.rho_final + res.rho_final') / 2))
        @test minimum(real.(evs)) > -1e-9

        est = estimate_mixing_time(res; model=:biexp,
                                    target_epsilon=1e-3, extrapolate=true)
        @test isfinite(est.mixing_time)
        @test est.mixing_time > 0
        @info "(q1) CKG EnergyDomain L-mode" tau_mix=est.mixing_time r2=est.r_squared dist_end=res.distances[end]
    end

    # -----------------------------------------------------------------------
    # CKG EnergyDomain K-mode
    # -----------------------------------------------------------------------
    @testset "(q2) CKG EnergyDomain end-to-end @ n=3, β=10 (mode=:K)" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        config = Config(;
            sim = Lindbladian(),
            domain = EnergyDomain(),
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta,
            sigma = 1.0 / beta,
            a = 0.0,
            s = 0.25,
            num_energy_bits = 12,
            w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
        )
        d = size(sys.ham.data, 1)
        rho_0 = Matrix{ComplexF64}(I(d) / d)
        t_grid = collect(range(0.0, 60.0, length=61))

        res = integrate_to_gibbs(config, sys.ham, sys.jumps, rho_0, t_grid;
                                  mode=:K, krylovdim=20, tol=1e-10)

        @test res.all_converged
        # KMS-DB ⇒ K is HS-self-adjoint ⇒ chi² is Lyapunov
        # (monotone non-increasing). Same per-step Krylov-tol slack as (f).
        @test all(diff(res.distances) .<= 1e-9)
        @test res.distances[end]^2 < 1e-12
        @info "(q2) CKG EnergyDomain K-mode" dist_end=res.distances[end] dist_end_sq=res.distances[end]^2
    end

    @testset "(q) Krylov gap matches dense H_gap at n=3 (KMS + DLL)" begin
        # KMS: at n=3, β=10 BohrDomain, krylov_spectral_gap should match
        # discriminant_spectrum(L_dense).H_gap. For KMS-DB the two agree to
        # ~1% rel err (similarity transform preserves spectrum). Memory-safe:
        # n=3 dense L is d²=64 → trivial.
        sys = make_dll_n3_system(10.0)
        ham = sys.ham; jumps = sys.jumps

        config_kms = make_config(Lindbladian(), BohrDomain();
                                  num_qubits = 3, construction = KMS())
        L_dense_kms = Matrix{ComplexF64}(construct_lindbladian(jumps, config_kms, ham))
        gap_dense_kms = discriminant_spectrum(L_dense_kms, ham.gibbs).H_gap
        krylov_kms = krylov_spectral_gap(config_kms, ham, jumps;
            krylovdim = 30, howmany = 2, tol = 1e-8)
        rel_err_kms = abs(krylov_kms.spectral_gap - gap_dense_kms) / gap_dense_kms
        @test rel_err_kms < 1e-2

        # DLL Gaussian: same check via the matrix-free DLL apply.
        # make_config does not forward `filter`; build Config directly mirroring
        # the make_config defaults plus filter / DLL construction.
        config_dll = Config(
            sim = Lindbladian(), domain = BohrDomain(), construction = DLL(),
            num_qubits = 3, with_linear_combination = true,
            beta = 10.0, sigma = 0.1, a = 10.0 / 30.0, s = 0.4,
            num_energy_bits = 12, w0 = 0.05, t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
            filter = DLLGaussianFilter(10.0),
        )
        L_dense_dll = Matrix{ComplexF64}(construct_lindbladian(jumps, config_dll, ham))
        gap_dense_dll = discriminant_spectrum(L_dense_dll, ham.gibbs).H_gap
        krylov_dll = krylov_spectral_gap(config_dll, ham, jumps;
            krylovdim = 30, howmany = 2, tol = 1e-8)
        rel_err_dll = abs(krylov_dll.spectral_gap - gap_dense_dll) / gap_dense_dll
        @test rel_err_dll < 1e-2
    end
end
