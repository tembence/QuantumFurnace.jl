using StableRNGs

@testset "Mixing Time Estimation" begin

    # -----------------------------------------------------------------------
    # Helper: build a synthetic ThermalizeResults for testing
    # -----------------------------------------------------------------------
    function _make_synthetic_result(times::Vector{Float64}, dists::Vector{Float64};
                                     num_qubits::Int=3, mixing_time::Float64=50.0)
        config = make_config(Thermalize(), EnergyDomain();
                             num_qubits=num_qubits, mixing_time=mixing_time)
        dim = 2^num_qubits
        final_dm = zeros(ComplexF64, dim, dim)
        metadata = Dict{Symbol, Any}(:test => true)
        return ThermalizeResults(config, final_dm, dists, times, metadata)
    end

    # -----------------------------------------------------------------------
    # Exponential fit on synthetic trace distance curve
    # -----------------------------------------------------------------------
    @testset "synthetic gap recovery" begin
        A_true = 1.5
        gap_true = 0.3
        C_true = 0.001
        times = collect(0.0:0.1:50.0)
        dists = A_true .* exp.(-gap_true .* times) .+ C_true

        result = _make_synthetic_result(times, dists; mixing_time=50.0)
        est = estimate_mixing_time(result; skip_initial=0.1)

        @test isapprox(est.fitted_gap, gap_true; atol=0.01)
        @test est.r_squared > 0.99
        @test est.converged == true
        @test est.amplitude > 0
        @test est.offset >= 0
        @test est.fit_result isa FitResult

        @info "gap recovery" fitted=est.fitted_gap true_gap=gap_true r2=est.r_squared
    end

    # -----------------------------------------------------------------------
    # Extrapolation mode
    # -----------------------------------------------------------------------
    @testset "extrapolation mode" begin
        A_true = 1.5
        gap_true = 0.3
        C_true = 0.001
        times = collect(0.0:0.1:50.0)
        dists = A_true .* exp.(-gap_true .* times) .+ C_true

        result = _make_synthetic_result(times, dists; mixing_time=50.0)
        est = estimate_mixing_time(result; skip_initial=0.1, extrapolate=true, target_epsilon=0.01)

        # Expected extrapolated mixing time from the model:
        # t = -log((epsilon - C) / A) / gap
        t_expected = -log((0.01 - C_true) / A_true) / gap_true

        @test isapprox(est.mixing_time_extrapolated, t_expected; rtol=0.05)
        @test est.mixing_time == est.mixing_time_extrapolated
        @test est.target_epsilon == 0.01

        @info "extrapolation" t_extrap=est.mixing_time_extrapolated t_expected=t_expected
    end

    # -----------------------------------------------------------------------
    # Actual mixing time from data (no extrapolation)
    # -----------------------------------------------------------------------
    @testset "actual mixing time from data" begin
        A_true = 1.5
        gap_true = 0.3
        C_true = 0.001
        target = 0.01
        times = collect(0.0:0.1:50.0)
        dists = A_true .* exp.(-gap_true .* times) .+ C_true

        # The trace distance crosses 0.01 somewhere in this data
        # Find expected crossing: first index where dists <= target
        expected_idx = findfirst(d -> d <= target, dists)
        @assert expected_idx !== nothing "synthetic data should cross target=$target"
        expected_time = times[expected_idx]

        result = _make_synthetic_result(times, dists; mixing_time=50.0)
        est = estimate_mixing_time(result; skip_initial=0.1, target_epsilon=target, extrapolate=false)

        @test est.mixing_time_actual !== nothing
        @test isapprox(est.mixing_time_actual, expected_time; atol=0.15)
        @test est.mixing_time == est.mixing_time_actual
        @test est.mixing_time_extrapolated === nothing

        @info "actual crossing" t_actual=est.mixing_time_actual t_expected=expected_time
    end

    # -----------------------------------------------------------------------
    # skip_initial keyword
    # -----------------------------------------------------------------------
    @testset "skip_initial improves fit" begin
        gap_true = 0.3
        times = collect(0.0:0.05:30.0)
        # Data with fast transient added
        dists = 2.0 .* exp.(-gap_true .* times) .+ 0.5 .+ 3.0 .* exp.(-2.0 .* times)

        result = _make_synthetic_result(times, dists; mixing_time=30.0)
        est_no_skip = estimate_mixing_time(result; skip_initial=0.0)
        est_skip = estimate_mixing_time(result; skip_initial=0.3)

        @test abs(est_skip.fitted_gap - gap_true) < abs(est_no_skip.fitted_gap - gap_true)

        @info "skip_initial" gap_no_skip=est_no_skip.fitted_gap gap_skip=est_skip.fitted_gap true_gap=gap_true
    end

    # -----------------------------------------------------------------------
    # MixingTimeEstimate struct fields
    # -----------------------------------------------------------------------
    @testset "struct fields and types" begin
        times = collect(0.0:0.1:50.0)
        dists = 1.5 .* exp.(-0.3 .* times) .+ 0.001

        result = _make_synthetic_result(times, dists; mixing_time=50.0)
        est = estimate_mixing_time(result; skip_initial=0.1, target_epsilon=0.01, extrapolate=true)

        # Check the returned fields.
        expected_fields = [
            :fitted_gap, :amplitude, :offset, :gap_ci, :gap_se, :r_squared,
            :converged, :mixing_time, :mixing_time_extrapolated, :mixing_time_actual,
            :target_epsilon, :fit_result, :model_used, :biexp_fit_result
        ]
        for f in expected_fields
            @test f in fieldnames(MixingTimeEstimate)
        end

        # Check types
        @test est.fitted_gap isa Float64
        @test est.amplitude isa Float64
        @test est.offset isa Float64
        @test est.gap_ci isa Tuple{Float64, Float64}
        @test est.gap_se isa Float64
        @test est.r_squared isa Float64
        @test est.converged isa Bool
        @test est.mixing_time isa Float64
        @test est.target_epsilon isa Float64
        @test est.fit_result isa FitResult
    end

    # -----------------------------------------------------------------------
    # Quality gate warnings
    # -----------------------------------------------------------------------
    @testset "R-squared warning for bad fit" begin
        times = collect(0.0:0.1:50.0)
        # Use absolute value of sin to keep values positive (trace distances must be >= 0)
        dists = abs.(sin.(times)) .+ 0.01

        result = _make_synthetic_result(times, dists; mixing_time=50.0)
        @test_warn "R-squared" estimate_mixing_time(result; skip_initial=0.0)
    end

    @testset "offset warning for large C" begin
        times = collect(0.0:0.1:50.0)
        # Large offset C=1.0 relative to target_epsilon=0.5
        dists = 1.0 .* exp.(-0.3 .* times) .+ 1.0

        result = _make_synthetic_result(times, dists; mixing_time=50.0)
        @test_warn "offset" estimate_mixing_time(result; skip_initial=0.1, target_epsilon=0.5)
    end

    @testset "quality warnings also apply to bi-exponential fits" begin
        times = collect(0.0:0.1:20.0)
        dists = abs.(sin.(times)) .+ 0.01

        @test_warn "Bi-exponential fit R-squared" estimate_mixing_time(
            times, dists; model=:biexp, skip_initial=0.0)
    end

    # -----------------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------------
    @testset "Edge: extrapolate=true without target_epsilon throws" begin
        times = collect(0.0:0.1:50.0)
        dists = 1.5 .* exp.(-0.3 .* times) .+ 0.001

        result = _make_synthetic_result(times, dists; mixing_time=50.0)
        @test_throws ArgumentError estimate_mixing_time(result; extrapolate=true)
    end

    @testset "Edge: fewer than 10 data points throws" begin
        times = collect(0.0:1.0:5.0)  # 6 points
        dists = 1.5 .* exp.(-0.3 .* times) .+ 0.001

        result = _make_synthetic_result(times, dists; mixing_time=5.0)
        @test_throws ArgumentError estimate_mixing_time(result)
    end

    @testset "Edge: physical distance-grid validation" begin
        times = collect(0.0:0.1:2.0)
        dists = exp.(-times)

        @test_throws ArgumentError estimate_mixing_time(
            times, [dists[1:end-1]; -0.1])
        @test_throws ArgumentError estimate_mixing_time(
            [times[1:end-1]; times[end-1]], dists)
        @test_throws ArgumentError estimate_mixing_time(
            times, dists; target_epsilon=0.0)

        # Distinct input times can collapse at the fitter's Float64 precision.
        precise_times = BigFloat.(0:20)
        precise_times[3] = nextfloat(precise_times[2])
        @test_throws ArgumentError estimate_mixing_time(
            precise_times, exp.(-Float64.(precise_times)); skip_initial=0.0)
    end

    @testset "Edge: target not reached in data" begin
        times = collect(0.0:0.1:5.0)  # short simulation
        dists = 1.5 .* exp.(-0.3 .* times) .+ 0.001  # doesn't reach 0.0001

        result = _make_synthetic_result(times, dists; mixing_time=5.0)
        est = estimate_mixing_time(result; target_epsilon=0.0001, extrapolate=false)

        @test est.mixing_time_actual === nothing
        @test isnan(est.mixing_time)  # NaN because actual not reached

        @info "Edge: target not reached" mixing_time=est.mixing_time actual=est.mixing_time_actual
    end

    # -----------------------------------------------------------------------
    # Integration test: real run_thermalize output
    # -----------------------------------------------------------------------
    @testset "Integration: real run_thermalize" begin
        config = make_config(Thermalize(), EnergyDomain();
                             num_qubits=3, mixing_time=5.0)
        result = run_thermalize(N3_JUMPS, config, N3_HAM)

        # Should NOT throw
        est = estimate_mixing_time(result)

        @test est.fitted_gap > 0
        @test est.r_squared > 0
        @test est.converged isa Bool
        @test est.fit_result isa FitResult

        @info "Integration test" gap=est.fitted_gap r2=est.r_squared converged=est.converged mixing_time=est.mixing_time
    end

    # ===================================================================
    # Bi-exponential mixing time tests
    # ===================================================================

    # -----------------------------------------------------------------------
    # extrapolation accuracy on exact synthetic bi-exp data
    # -----------------------------------------------------------------------
    @testset "biexp extrapolation accuracy" begin
        # Synthetic bi-exponential data mimicking Liouvillian multi-timescale decay
        A1_true = 1.0    # fast mode
        g1_true = 2.0    # fast gap
        A2_true = 0.5    # slow mode
        g2_true = 0.3    # slow gap (spectral gap)
        C_true  = 6.8e-5 # small floor (from coherent unitary)
        target  = 1e-4   # target epsilon, close to floor

        times = collect(0.0:0.1:50.0)
        dists = A1_true .* exp.(-g1_true .* times) .+
                A2_true .* exp.(-g2_true .* times) .+
                C_true

        # True mixing time: solve A1*exp(-g1*t) + A2*exp(-g2*t) + C = target numerically
        # For this data, the fast mode decays quickly so it's mainly A2*exp(-g2*t) + C = target
        # => t_true ~ -log((target - C) / A2) / g2
        # But we want the exact answer from the full bi-exp model
        using Roots
        f_true(t) = A1_true * exp(-g1_true * t) + A2_true * exp(-g2_true * t) + C_true - target
        t_true = Roots.find_zero(f_true, (0.0, 200.0), Roots.Bisection())

        result = _make_synthetic_result(times, dists; mixing_time=50.0)

        # Bi-exponential extrapolation
        est_biexp = estimate_mixing_time(result;
            model=:biexp, skip_initial=0.1,
            target_epsilon=target, extrapolate=true)

        @test est_biexp.model_used == :biexp
        @test est_biexp.biexp_fit_result isa BiexpFitResult
        @test est_biexp.mixing_time_extrapolated !== nothing

        biexp_err = abs(est_biexp.mixing_time_extrapolated - t_true) / t_true
        @test biexp_err < 1e-7

        # Compare with single-exp extrapolation
        est_single = estimate_mixing_time(result;
            model=:single, skip_initial=0.1,
            target_epsilon=target, extrapolate=true)

        @test est_single.mixing_time_extrapolated !== nothing
        @test isfinite(est_single.mixing_time_extrapolated)
        single_err = abs(est_single.mixing_time_extrapolated - t_true) / t_true
        @test biexp_err < single_err
        @info "Biexponential mixing extrapolation" t_true=t_true t_biexp=est_biexp.mixing_time_extrapolated t_single=est_single.mixing_time_extrapolated biexp_err=biexp_err single_err=single_err
    end

    # -----------------------------------------------------------------------
    # BIEXP-MIX edge: invalid model keyword throws
    # -----------------------------------------------------------------------
    @testset "BIEXP-MIX edge: invalid model throws" begin
        times = collect(0.0:0.1:50.0)
        dists = 1.5 .* exp.(-0.3 .* times) .+ 0.001

        result = _make_synthetic_result(times, dists; mixing_time=50.0)
        @test_throws ArgumentError estimate_mixing_time(result; model=:invalid)
    end

    @testset "BIEXP-MIX edge: crossing requires a strict finite bracket" begin
        base = BiexpFitResult(
            0.3, 2.0, 0.5, 1.0, 0.001,
            (0.2, 0.4), 0.01, 1.0, true,
            Float64[], Float64[], Float64[],
        )
        crossing = QuantumFurnace._extrapolate_mixing_time_biexp(base, 0.01)
        @test crossing !== nothing
        @test isfinite(crossing)

        nonfinite = BiexpFitResult(
            base.gap, base.gap_fast, NaN, base.amplitude_fast, base.offset,
            base.gap_ci, base.gap_se, base.r_squared, base.converged,
            base.residuals, base.times_used, base.values_used,
        )
        @test QuantumFurnace._extrapolate_mixing_time_biexp(nonfinite, 0.01) === nothing
        @test QuantumFurnace._extrapolate_mixing_time_biexp(base, NaN) === nothing
        @test QuantumFurnace._extrapolate_mixing_time_biexp(base, base.offset) === nothing
    end

    # ====================================================================
    # Integrator wrapper: vector-method API + NamedTuple forwarder
    # ====================================================================
    @testset "Integrator wrapper" begin

        # ---------------------------------------------------------------
        # (a) Synthetic single-exp matches analytic
        # ---------------------------------------------------------------
        @testset "(a) vector method: single-exp matches analytic" begin
            times      = collect(range(0.0, 10.0; length = 101))
            distances  = 0.5 .* exp.(-0.7 .* times)

            est = estimate_mixing_time(times, distances;
                                       model           = :single,
                                       target_epsilon  = 0.01,
                                       extrapolate     = true)

            # Analytic: 0.5 * exp(-0.7 t) = 0.01  =>  t = log(50) / 0.7
            @test isapprox(est.fitted_gap,   0.7;           rtol = 1e-3)
            @test isapprox(est.mixing_time,  log(50) / 0.7; rtol = 1e-2)
            @test est.model_used === :single

            @info "Single-exponential mixing fit" gap=est.fitted_gap mixing=est.mixing_time
        end

        # ---------------------------------------------------------------
        # (b) Synthetic bi-exp matches analytic
        # ---------------------------------------------------------------
        @testset "(b) vector method: bi-exp recovers slow gap" begin
            # Need t_max >> 1/g_slow ≈ 3.3 to expose the asymptote.
            # Pick t_max = 30 (~10 e-foldings of slow mode), n=301.
            times = collect(range(0.0, 30.0; length = 301))
            dists = 0.4 .* exp.(-2.0 .* times) .+
                    0.3 .* exp.(-0.3 .* times) .+ 1.0e-5

            est = estimate_mixing_time(times, dists;
                                       model           = :biexp,
                                       target_epsilon  = 1.0e-3,
                                       extrapolate     = true)

            @test isapprox(est.fitted_gap, 0.3; rtol = 5.0e-2)
            @test est.mixing_time_extrapolated !== nothing &&
                  isfinite(est.mixing_time_extrapolated)
            @test est.model_used === :biexp
            @test est.biexp_fit_result !== nothing

            @info "Biexponential mixing fit" slow_gap=est.fitted_gap fast_gap=est.biexp_fit_result.gap_fast tmix=est.mixing_time_extrapolated
        end

        # ---------------------------------------------------------------
        # (c) NamedTuple forwarder matches direct vector call
        # ---------------------------------------------------------------
        @testset "(c) NamedTuple forwarder bit-equivalent to vector call" begin
            times = collect(range(0.0, 30.0; length = 301))
            dists = 0.4 .* exp.(-2.0 .* times) .+
                    0.3 .* exp.(-0.3 .* times) .+ 1.0e-5

            # Mock the integrator output shape.
            mock = (
                t              = times,
                distances      = dists,
                total_matvecs  = 0,
                all_converged  = true,
            )

            est_nt  = estimate_mixing_time(mock;
                                           model = :biexp,
                                           target_epsilon = 1.0e-3,
                                           extrapolate    = true)
            est_vec = estimate_mixing_time(times, dists;
                                           model = :biexp,
                                           target_epsilon = 1.0e-3,
                                           extrapolate    = true)

            @test est_nt.mixing_time_extrapolated == est_vec.mixing_time_extrapolated
            @test est_nt.fitted_gap == est_vec.fitted_gap
        end

        # ---------------------------------------------------------------
        # (d) Backward-compat: ThermalizeResults dispatch matches vector dispatch
        # ---------------------------------------------------------------
        @testset "(d) ThermalizeResults regression: delegation parity" begin
            times = collect(0.0:0.1:50.0)
            dists = 1.5 .* exp.(-0.3 .* times) .+ 0.001

            result = _make_synthetic_result(times, dists; mixing_time = 50.0)

            # ThermalizeResults overload (default :single)
            est_old = estimate_mixing_time(result;
                                           target_epsilon = 0.01,
                                           extrapolate    = true)
            # Vector method, model=:single explicit (matches old default)
            est_new = estimate_mixing_time(times, dists;
                                           model = :single,
                                           target_epsilon = 0.01,
                                           extrapolate    = true)

            @test est_old.model_used === :single  # regression: default unchanged
            @test isapprox(est_old.mixing_time, est_new.mixing_time;  atol = 1.0e-12)
            @test isapprox(est_old.fitted_gap,  est_new.fitted_gap;   atol = 1.0e-12)
            @test isapprox(est_old.offset,      est_new.offset;       atol = 1.0e-12)
        end
    end

    # ====================================================================
    # Eigenmode τ_mix: closed-form bisection on the Krylov
    # spectral decomposition. Replaces the bi-exp curve fit on the :krylov
    # route — same answer on healthy cells, finite/correct on cells where
    # LM degenerates.
    # ====================================================================
    @testset "Eigenmode τ_mix" begin

        # Hand-built spectral decomposition: build R_modes, c, eigenvalues
        # such that ρ(t) - σ_β = Σ_i c_i e^{λ_i t} R_i is a known scalar
        # multiple of a fixed (Hermitian, traceless) matrix `M`. Then
        # d(t) = scalar(t) * ‖M‖_1 / 2.
        function _build_toy_decomp(eigenvalues::Vector{ComplexF64},
                                    coeffs::Vector{ComplexF64};
                                    d::Int = 4)
            # Single shared Hermitian traceless mode shape.
            M = ComplexF64[0 1 0 0; 1 0 0 0; 0 0 0 -1; 0 0 -1 0]  # Hermitian, tr=0
            # Replicate M as the right-eigenvector for every non-steady mode;
            # steady mode is a no-op contribution (c[1] = 0 by convention).
            R_modes = Vector{Matrix{ComplexF64}}(undef, length(eigenvalues))
            for i in eachindex(eigenvalues)
                R_modes[i] = copy(M)
            end
            return (R_modes = R_modes, M = M)
        end

        # ---------------------------------------------------------------
        # (a) Hand-built 3-mode toy: synthetic spectral data → analytic τ_mix
        # ---------------------------------------------------------------
        @testset "(a) 3-mode toy recovers analytic τ_mix" begin
            d = 4
            eigenvalues = ComplexF64[0.0, -0.5, -2.0]
            c           = ComplexF64[0.0, 0.3, 0.4]
            toy = _build_toy_decomp(eigenvalues, c; d=d)
            # σ_β and ρ_inf coincide (no floor) so floor_distance = 0.
            sigma_beta = Matrix{ComplexF64}(I, d, d) / d
            rho_inf    = copy(sigma_beta)

            # The residual matrix is (0.3 e^{-0.5 t} + 0.4 e^{-2 t}) * M;
            # ‖M‖_1 = sum of |singular values| = 4 (M has eigenvalues ±1, ±1).
            M_norm = sum(svdvals(toy.M))  # = 4
            # d(t) = (0.3 e^{-0.5 t} + 0.4 e^{-2 t}) * M_norm / 2.
            target = 0.05
            f_true(t) = (0.3 * exp(-0.5 * t) + 0.4 * exp(-2.0 * t)) * M_norm / 2 - target
            t_true = Roots.find_zero(f_true, (0.0, 200.0), Roots.Bisection())

            res = eigenmode_mixing_time(eigenvalues, c, toy.R_modes,
                                          rho_inf, sigma_beta, target;
                                          atol=1e-4)

            @test res.source === :extrapolated
            @test isapprox(res.gap, 0.5; atol=1e-12)
            @test isapprox(res.floor_distance, 0.0; atol=1e-12)
            @test isapprox(res.mixing_time, t_true; atol=1e-3)
            @info "(a) toy τ_mix" τ=res.mixing_time τ_true=t_true gap=res.gap n_evals=res.n_evals
        end

        # ---------------------------------------------------------------
        # (b) Floor-handling: target below ‖ρ_inf - σ_β‖_1 / 2 → :floor
        # ---------------------------------------------------------------
        @testset "(b) target below floor → :floor source" begin
            d = 4
            eigenvalues = ComplexF64[0.0, -0.5]
            c           = ComplexF64[0.0, 0.1]
            toy = _build_toy_decomp(eigenvalues, c; d=d)
            sigma_beta = Matrix{ComplexF64}(I, d, d) / d
            # Build ρ_inf with a known offset from σ_β: 0.01 trace distance.
            # δ = (0.02 / ‖M‖_1) * M is Hermitian and trace-0 → trace dist = 0.01.
            M_norm = sum(svdvals(toy.M))
            rho_inf = sigma_beta .+ (0.02 / M_norm) .* toy.M
            target = 0.001  # below floor=0.01

            res = eigenmode_mixing_time(eigenvalues, c, toy.R_modes,
                                          rho_inf, sigma_beta, target)
            @test res.source === :floor
            @test isinf(res.mixing_time)
            @test isapprox(res.floor_distance, 0.01; atol=1e-12)
            @info "(b) floor branch" floor=res.floor_distance source=res.source
        end

        # ---------------------------------------------------------------
        # (c) Complex-eigenvalue robustness: rotating off-diagonal mode
        # ---------------------------------------------------------------
        @testset "(c) complex conjugate-pair eigenmodes" begin
            d = 4
            # A conjugate pair on |0><1| and |1><0| rotates phase while its
            # singular values decay monotonically as exp(-0.3t).
            eigenvalues = ComplexF64[0.0, -0.3 + 0.5im, -0.3 - 0.5im]
            c           = ComplexF64[0.0, 0.25 + 0.0im, 0.25 + 0.0im]
            sigma_beta = Matrix{ComplexF64}(I, d, d) / d
            rho_inf    = copy(sigma_beta)
            R_steady = copy(sigma_beta)
            R_plus = zeros(ComplexF64, d, d); R_plus[1, 2] = 1
            R_minus = Matrix(R_plus')
            R_modes = [R_steady, R_plus, R_minus]
            target = 0.05
            t_true = log(0.25 / target) / 0.3

            res = eigenmode_mixing_time(eigenvalues, c, R_modes,
                                          rho_inf, sigma_beta, target;
                                          atol=1e-4)
            @test res.source === :extrapolated
            @test isapprox(res.gap, 0.3; atol=1e-12)
            @test isapprox(res.mixing_time, t_true; atol=1e-2)
            @info "(c) complex eigenmodes" τ=res.mixing_time τ_true=t_true
        end

        # ---------------------------------------------------------------
        # (d) Integration with predict_lindbladian_trajectory: eigenmode
        # τ_mix matches a dense fine-grid trajectory crossing within rtol.
        # ---------------------------------------------------------------
        @testset "(d) integrates with predict_lindbladian_trajectory (n=3 β=10)" begin
            # Uses the BETA=10 test fixture from test_helpers.jl; smooth-Metro
            # is the project default for make_config (a=BETA/30, s=0.4).
            cfg = make_config(Lindbladian(), EnergyDomain(); num_qubits=3)
            ham = N3_HAM
            jumps = N3_JUMPS
            d = 2^3
            # Maximally-mixed init (worst-case starting trace distance).
            rho_0 = Matrix{ComplexF64}(I, d, d) / d
            # Coarse grid for predictor; eigenmode formula is grid-independent.
            t_grid = collect(range(0.0, 80.0; length=41))

            pres = predict_lindbladian_trajectory(cfg, ham, jumps, rho_0, t_grid;
                                                    krylovdim=40)
            target = 1e-3
            res_eig = eigenmode_mixing_time(pres.eigenvalues, pres.c, pres.R_modes,
                                              pres.rho_inf, pres.sigma_beta, target;
                                              atol=1e-3)
            @test res_eig.source === :extrapolated
            @test isfinite(res_eig.mixing_time) && res_eig.mixing_time > 0
            @test res_eig.gap > 0

            # the NamedTuple convenience method reproduces the explicit-
            # args call bit-for-bit, and the biexp curve fit must refuse the
            # predictor result (τ_mix on this path is bisection-only).
            res_conv = eigenmode_mixing_time(pres, target)
            @test res_conv.mixing_time == res_eig.mixing_time
            @test res_conv.gap == res_eig.gap
            @test_throws ArgumentError estimate_mixing_time(pres; model=:biexp,
                                                            target_epsilon=target,
                                                            extrapolate=true)

            # Independent dense evolution cross-check at the inferred time.
            L_dense = construct_lindbladian(jumps, cfg, ham)
            rho_dense = reshape(
                exp(res_eig.mixing_time * L_dense) * vec(rho_0), d, d)
            rho_dense .= (rho_dense + rho_dense') / 2
            dense_distance = sum(svdvals(rho_dense .- pres.sigma_beta)) / 2
            @test isapprox(dense_distance, target; rtol=2e-2)
            @info "(d) predict_lindbladian + eigenmode_mixing_time" τ=res_eig.mixing_time gap=res_eig.gap floor=res_eig.floor_distance n_evals=res_eig.n_evals
        end

        # ---------------------------------------------------------------
        # (d') NamedTuple convenience: continuous generator + discrete channel + guards
        # ---------------------------------------------------------------
        @testset "(d') eigenmode_mixing_time(traj::NamedTuple) — Lindbladian + channel" begin
            # Minimal 2-mode toy: steady |0><0| + one slow real mode. d(t) =
            # |c_slow| e^{λ t} ‖R_slow‖₁ / 2 = 0.4 e^{-0.2 t}; floor 0.
            R_steady   = Matrix{ComplexF64}([1.0 0.0; 0.0 0.0])   # trace 1
            R_slow     = Matrix{ComplexF64}([0.5 0.0; 0.0 -0.5])  # traceless Herm
            λ_slow     = -0.2
            eig        = ComplexF64[0.0, λ_slow]
            c          = ComplexF64[0.0, 0.8]                     # steady c ≈ 0
            Rmodes     = [R_steady, R_slow]
            rho_inf    = copy(R_steady)
            sigma_beta = copy(R_steady)                           # floor = 0
            target     = 1e-3

            ref = eigenmode_mixing_time(eig, c, Rmodes, rho_inf, sigma_beta, target)
            @test ref.source === :extrapolated

            # Lindbladian-shape NamedTuple (no :delta_used) — identical extraction.
            trajL = (t = Float64[], distances = Float64[], eigenvalues = eig, c = c,
                     R_modes = Rmodes, rho_inf = rho_inf, sigma_beta = sigma_beta)
            resL = eigenmode_mixing_time(trajL, target)
            @test resL.mixing_time == ref.mixing_time

            # Channel-shape NamedTuple: μ = e^{λδ}. The physical
            # crossing is the first integer step, not a principal-log interpolation.
            δ     = 0.05
            μ     = ComplexF64[exp(0.0 * δ), exp(λ_slow * δ)]
            trajC = (t = Float64[], distances = Float64[], eigenvalues = μ, c = c,
                     R_modes = Rmodes, rho_inf = rho_inf, sigma_beta = sigma_beta,
                     delta_used = δ)
            resC = eigenmode_mixing_time(trajC, target)
            expected_step = ceil(Int, ref.mixing_time / δ)
            @test resC.crossing_kind === :integer_steps
            @test resC.mixing_steps == expected_step
            @test resC.mixing_time == expected_step * δ

            # A negative mode makes the distinction explicit: its sign
            # alternates under physical powers, and the first integer trace-
            # distance crossing is k=5.
            mu_negative = ComplexF64[1.0, -0.8]
            negative_traj = merge(
                trajC, (eigenvalues = mu_negative, delta_used = 1.0))
            negative = eigenmode_mixing_time(negative_traj, 0.15)
            @test negative.source === :extrapolated
            @test negative.crossing_kind === :integer_steps
            @test negative.mixing_steps == 5
            @test negative.mixing_time == 5.0
            @test negative.gap == -log(0.8)

            # A valid unital Pauli channel can be nonmonotone relative to a
            # Gibbs reference distinct from its fixed point. Here lambda_z=-0.8
            # sends |0><0| exactly to sigma at k=1, then away again. A binary
            # search over k=0:9 lands on the later k=7 crossing instead.
            pauli_traj = (
                t = Float64[],
                distances = Float64[],
                eigenvalues = ComplexF64[1.0, -0.8],
                c = ComplexF64[0.0, 0.5],
                R_modes = [Matrix{ComplexF64}(I, 2, 2) / 2,
                           ComplexF64[1 0; 0 -1]],
                rho_inf = Matrix{ComplexF64}(I, 2, 2) / 2,
                sigma_beta = ComplexF64[0.1 0; 0 0.9],
                delta_used = 1.0,
            )
            pauli_crossing = eigenmode_mixing_time(
                pauli_traj, 0.5; k_upper=9)
            @test pauli_crossing.source === :extrapolated
            @test pauli_crossing.mixing_steps == 1
            @test pauli_crossing.mixing_time == 1.0

            # An asymptotic floor is not a no-crossing certificate: this
            # alternating mode hits the distinct Gibbs reference at k=1 even
            # though the limiting state remains 0.4 away from it.
            transient_traj = merge(pauli_traj, (
                c = ComplexF64[0.0, -0.5],
                sigma_beta = ComplexF64[0.9 0; 0 0.1],
            ))
            transient = eigenmode_mixing_time(
                transient_traj, 0.1; k_upper=2)
            @test transient.floor_distance ≈ 0.4 atol=1e-14
            @test transient.source === :extrapolated
            @test transient.mixing_steps == 1
            @test transient.mixing_time == 1.0

            # The k=0 state is chronological too, and must be checked before
            # classifying the same nonzero asymptotic floor.
            initially_mixed = eigenmode_mixing_time(
                merge(transient_traj, (c = ComplexF64[0.0, 0.4],)),
                0.1; k_upper=0)
            @test initially_mixed.floor_distance ≈ 0.4 atol=1e-14
            @test initially_mixed.source === :extrapolated
            @test initially_mixed.mixing_steps == 0
            @test initially_mixed.mixing_time == 0.0

            # A periodic CPTP channel has zero generator-equivalent gap, but a
            # caller-supplied physical horizon can still contain a crossing.
            periodic_traj = merge(pauli_traj, (
                eigenvalues = ComplexF64[1.0, -1.0],
                c = ComplexF64[0.0, 0.5],
                sigma_beta = ComplexF64[0 0; 0 1],
            ))
            periodic = eigenmode_mixing_time(
                periodic_traj, 1e-12; k_upper=1)
            @test periodic.gap == 0.0
            @test periodic.source === :extrapolated
            @test periodic.mixing_steps == 1
            @test periodic.mixing_time == 1.0

            # A hard cap is explicit and does not masquerade as a floor. The
            # dual witness certifies every alternating non-crossing step after
            # one exact decomposition, so even a long chronological horizon
            # remains cheap without weakening first-crossing correctness.
            periodic_no_crossing = eigenmode_mixing_time(
                merge(periodic_traj, (sigma_beta = pauli_traj.rho_inf,)),
                0.1; k_upper=100_000, max_steps=5_000)
            @test periodic_no_crossing.source === :horizon_exhausted
            @test periodic_no_crossing.search_horizon == 5_000
            @test periodic_no_crossing.horizon_capped
            @test periodic_no_crossing.n_evals == 1

            # A decaying tail can certify a genuine asymptotic floor after the
            # finite scan. A nondecaying periodic tail receives the separate
            # certified-no-crossing label rather than being called a floor.
            certified_floor = eigenmode_mixing_time(
                merge(transient_traj, (
                    eigenvalues = ComplexF64[1.0, 0.5],
                    c = ComplexF64[0.0, -0.5],
                )), 0.1; k_upper=10)
            @test certified_floor.source === :floor
            @test certified_floor.mixing_time == Inf
            @test certified_floor.tail_bound <
                certified_floor.floor_distance - 0.1

            certified_periodic = eigenmode_mixing_time(
                merge(transient_traj, (
                    eigenvalues = ComplexF64[1.0, -1.0],
                    c = ComplexF64[0.0, -0.05],
                )), 0.1; k_upper=10)
            @test certified_periodic.source === :certified_no_crossing
            @test certified_periodic.mixing_time == Inf

            # The curve fit refuses both predictor NamedTuples…
            @test_throws ArgumentError estimate_mixing_time(trajL; target_epsilon = target)
            @test_throws ArgumentError estimate_mixing_time(trajC; target_epsilon = target)
            # …and the convenience method needs the spectral fields.
            @test_throws ArgumentError eigenmode_mixing_time(
                (t = Float64[], distances = Float64[]), target)
        end

        # ---------------------------------------------------------------
        # (e) Edge: degenerate input (only steady mode) → :nan
        # ---------------------------------------------------------------
        @testset "(e) only-steady-mode input → :nan" begin
            d = 4
            # Single eigenvalue at zero (steady), no slow modes captured.
            eigenvalues = ComplexF64[0.0]
            c           = ComplexF64[0.0]
            R_modes     = Matrix{ComplexF64}[Matrix{ComplexF64}(I, d, d) / d]
            sigma_beta  = Matrix{ComplexF64}(I, d, d) / d
            rho_inf     = copy(sigma_beta)
            res = eigenmode_mixing_time(eigenvalues, c, R_modes,
                                          rho_inf, sigma_beta, 1e-3)
            @test res.source === :nan
            @test isnan(res.mixing_time)
            @info "(e) degenerate input" source=res.source gap=res.gap
        end

        # ---------------------------------------------------------------
        # (f) Edge: invalid input (mismatched lengths) → ArgumentError
        # ---------------------------------------------------------------
        @testset "(f) mismatched input lengths throw" begin
            d = 4
            eigenvalues = ComplexF64[0.0, -0.5]
            c           = ComplexF64[0.0]   # wrong length
            R_modes     = [Matrix{ComplexF64}(I, d, d) for _ in 1:2]
            sigma_beta  = Matrix{ComplexF64}(I, d, d) / d
            rho_inf     = copy(sigma_beta)
            @test_throws ArgumentError eigenmode_mixing_time(eigenvalues, c,
                R_modes, rho_inf, sigma_beta, 1e-3)
            @test_throws ArgumentError eigenmode_mixing_time(
                ComplexF64[0.0, -0.5], ComplexF64[0.0, 0.1],
                [Matrix{ComplexF64}(I, d, d) for _ in 1:2],
                rho_inf, sigma_beta, -1e-3)
        end
    end

end
