# ============================================================================
# Tests for scaling_fit.jl
# ============================================================================
# Coverage:
#   (a) M0 recovers known (C, x, y) from synthetic power-law data
#   (b) M1 recovers known (C, x, α) from synthetic Arrhenius data
#   (c) AICc prefers M0 on power-law data; M1 on Arrhenius data
#   (d) Input validation (negative τ, mismatched lengths, too few points, bad models)
#   (e) NamedTuple and Vector{NamedTuple} input forms work
#   (f) predict_scaling round-trips through the data
#   (g) formula_string contains the right tokens
#   (h) Aliasing diagnostic: corr(x, slope) is computed and finite

using Test
using QuantumFurnace
using LinearAlgebra
using Random: MersenneTwister

@testset "scaling_fit.jl" begin
    # ------------------------------------------------------------------------
    # (a) M0 exponent recovery: τ = C · n^x · β^y
    # ------------------------------------------------------------------------
    @testset "(a) M0 recovers known (C, x, y)" begin
        rng = MersenneTwister(42)
        C_true, x_true, y_true = 0.5, 2.3, 1.7
        n_grid = [3, 4, 5, 6, 7, 8, 9, 10, 11]
        β_grid = [3.0, 5.0, 8.0, 10.0, 20.0, 30.0]
        n_vals = Int[]
        β_vals = Float64[]
        τ_vals = Float64[]
        for n in n_grid, β in β_grid
            τ_true = C_true * n^x_true * β^y_true
            # ~1% multiplicative noise (log-normal)
            τ_obs = τ_true * exp(0.01 * randn(rng))
            push!(n_vals, n); push!(β_vals, β); push!(τ_vals, τ_obs)
        end

        fits = fit_scaling(n_vals, β_vals, τ_vals; models = (:M0, :M1))
        @test haskey(fits, :M0)
        @test haskey(fits, :M1)

        fit_m0 = fits[:M0]
        @test fit_m0.converged
        @test fit_m0.model === :M0
        @test fit_m0.param_names === (:c, :x, :y)
        @test fit_m0.n_data == length(n_vals)

        # Exponent recovery within ~3σ
        c_fit, x_fit, y_fit = fit_m0.params
        @test isapprox(exp(c_fit), C_true; rtol = 0.05)
        @test isapprox(x_fit, x_true; atol = 0.05)
        @test isapprox(y_fit, y_true; atol = 0.05)

        # Residual stdev should match noise level (~0.01 in log space)
        @test fit_m0.sigma_residual < 0.05

        # Predict matches the data within noise
        for i in eachindex(n_vals)
            τ_pred = predict_scaling(fit_m0, n_vals[i], β_vals[i])
            @test isapprox(log(τ_pred), log(τ_vals[i]); atol = 0.10)
        end
    end

    # ------------------------------------------------------------------------
    # (b) M1 exponent recovery: τ = C · n^x · exp(α·β)
    # ------------------------------------------------------------------------
    @testset "(b) M1 recovers known (C, x, α)" begin
        rng = MersenneTwister(123)
        C_true, x_true, α_true = 0.3, 2.0, 0.15
        n_grid = [3, 4, 5, 6, 7, 8, 9, 10, 11]
        β_grid = [3.0, 5.0, 8.0, 10.0, 20.0, 30.0]
        n_vals = Int[]
        β_vals = Float64[]
        τ_vals = Float64[]
        for n in n_grid, β in β_grid
            τ_true = C_true * n^x_true * exp(α_true * β)
            τ_obs = τ_true * exp(0.01 * randn(rng))
            push!(n_vals, n); push!(β_vals, β); push!(τ_vals, τ_obs)
        end

        fits = fit_scaling(n_vals, β_vals, τ_vals; models = (:M0, :M1))
        fit_m1 = fits[:M1]
        @test fit_m1.converged
        @test fit_m1.model === :M1
        @test fit_m1.param_names === (:c, :x, :α)

        c_fit, x_fit, α_fit = fit_m1.params
        @test isapprox(exp(c_fit), C_true; rtol = 0.10)
        @test isapprox(x_fit, x_true; atol = 0.05)
        @test isapprox(α_fit, α_true; atol = 0.005)

        @test fit_m1.sigma_residual < 0.05
    end

    # ------------------------------------------------------------------------
    # (c) AICc preference: M0 wins on power data, M1 wins on Arrhenius data
    # ------------------------------------------------------------------------
    @testset "(c) AICc model discrimination" begin
        rng = MersenneTwister(2025)
        n_grid = [3, 4, 5, 6, 7, 8, 9, 10, 11]
        β_grid = [3.0, 5.0, 8.0, 10.0, 20.0, 30.0]

        # Build a power-law dataset (M0 ground truth).
        n_vals, β_vals, τ_vals = Int[], Float64[], Float64[]
        for n in n_grid, β in β_grid
            τ = 0.5 * n^2.3 * β^1.7 * exp(0.005 * randn(rng))
            push!(n_vals, n); push!(β_vals, β); push!(τ_vals, τ)
        end
        fits_pow = fit_scaling(n_vals, β_vals, τ_vals)
        cmp_pow = compare_models(fits_pow)
        @test cmp_pow.ranked[1] === :M0
        # Δ-AICc(M1) should be substantial (≫ 2) given the wide β range.
        @test cmp_pow.delta_aicc[2] > 2.0
        # Weights sum to 1.
        @test isapprox(sum(cmp_pow.weights), 1.0; atol = 1e-12)
        @test 0 ≤ cmp_pow.weights[1] ≤ 1

        # Build an Arrhenius dataset (M1 ground truth).
        n_vals2, β_vals2, τ_vals2 = Int[], Float64[], Float64[]
        for n in n_grid, β in β_grid
            τ = 0.3 * n^2.0 * exp(0.15 * β) * exp(0.005 * randn(rng))
            push!(n_vals2, n); push!(β_vals2, β); push!(τ_vals2, τ)
        end
        fits_arr = fit_scaling(n_vals2, β_vals2, τ_vals2)
        cmp_arr = compare_models(fits_arr)
        @test cmp_arr.ranked[1] === :M1
        @test cmp_arr.delta_aicc[2] > 2.0

        # AICc weights matching the ranking
        weights = aicc_weights(fits_arr)
        @test weights[:M1] > weights[:M0]
        @test isapprox(weights[:M0] + weights[:M1], 1.0; atol = 1e-12)
    end

    # ------------------------------------------------------------------------
    # (d) Input validation
    # ------------------------------------------------------------------------
    @testset "(d) Input validation" begin
        # Mismatched length
        @test_throws ArgumentError fit_scaling([3,4,5,6,7,8], [1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        @test_throws ArgumentError fit_scaling([3,4,5,6,7,8], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0])

        # Too few data points (< 6)
        @test_throws ArgumentError fit_scaling([3,4,5,6,7], [5.0, 5.0, 5.0, 5.0, 5.0],
                                                [1.0, 2.0, 3.0, 4.0, 5.0])

        # Non-positive τ_mix
        @test_throws ArgumentError fit_scaling([3,4,5,6,7,8], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                                                [1.0, 2.0, 0.0, 4.0, 5.0, 6.0])
        @test_throws ArgumentError fit_scaling([3,4,5,6,7,8], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                                                [1.0, 2.0, -1.0, 4.0, 5.0, 6.0])

        # Non-finite τ_mix
        @test_throws ArgumentError fit_scaling([3,4,5,6,7,8], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                                                [1.0, 2.0, NaN, 4.0, 5.0, 6.0])

        # Non-finite inverse temperature
        @test_throws ArgumentError fit_scaling([3,4,5,6,7,8],
                                                [5.0, 5.0, Inf, 5.0, 5.0, 5.0],
                                                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Unknown model
        @test_throws ArgumentError fit_scaling([3,4,5,6,7,8], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                                                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; models = (:M0, :M99))

        # Empty models tuple
        @test_throws ArgumentError fit_scaling([3,4,5,6,7,8], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                                                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; models = ())

        # Bad confidence level
        @test_throws ArgumentError fit_scaling([3,4,5,6,7,8], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                                                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; level = 1.5)
    end

    # ------------------------------------------------------------------------
    # (e) NamedTuple and Vector{NamedTuple} convenience methods
    # ------------------------------------------------------------------------
    @testset "(e) Convenience input shapes" begin
        # NamedTuple form
        nt_in = (
            n       = [3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6],
            beta    = [5.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0],
            tau_mix = [1.0, 2.0, 3.5, 5.5, 1.5, 3.0, 5.0, 8.0, 2.5, 5.0, 8.5, 13.0],
        )
        fits = fit_scaling(nt_in; models = (:M0,))
        @test haskey(fits, :M0)
        @test fits[:M0].converged
        @test fits[:M0].n_data == 12

        # Missing fields → ArgumentError
        @test_throws ArgumentError fit_scaling((n = [1,2,3], beta = [1.0, 2.0, 3.0]); models = (:M0,))

        # Vector{NamedTuple} (Lindbladian sweep schema with :mixing_time / :mixing_time_source)
        results = NamedTuple[]
        for n in 3:8, β in (5.0, 10.0, 20.0)
            push!(results, (
                n = n, beta = β,
                mixing_time = 0.5 * n^2.0 * β^1.5,
                mixing_time_source = :extrapolated,
            ))
        end
        # Add a junk cell that should be filtered out.
        push!(results, (n = 5, beta = 10.0, mixing_time = NaN,
                        mixing_time_source = :nan))
        push!(results, (n = 5, beta = 10.0, mixing_time = 1e9,
                        mixing_time_source = :floor))

        fits2 = fit_scaling(results; models = (:M0,))
        @test fits2[:M0].n_data == 18  # 6 n × 3 β, no junk
        # M0 should recover x≈2.0, y≈1.5
        @test isapprox(fits2[:M0].params[2], 2.0; atol = 0.05)
        @test isapprox(fits2[:M0].params[3], 1.5; atol = 0.05)

        # Channel sweep schema also works (:tau_mix / :tau_mix_source)
        channel_results = NamedTuple[]
        for n in 3:8, β in (5.0, 10.0, 20.0)
            push!(channel_results, (
                n = n, beta = β,
                tau_mix = 0.5 * n^2.0 * β^1.5,
                tau_mix_source = :extrapolated,
            ))
        end
        fits3 = fit_scaling(channel_results; models = (:M0,))
        @test fits3[:M0].n_data == 18

        # Empty-after-filter → error
        bad_results = [(n = 3, beta = 5.0, mixing_time = NaN,
                        mixing_time_source = :nan) for _ in 1:5]
        @test_throws ArgumentError fit_scaling(bad_results; models = (:M0,))
    end

    # ------------------------------------------------------------------------
    # (f) Round-trip: predict_scaling matches log_tau_predicted at the data
    # ------------------------------------------------------------------------
    @testset "(f) predict_scaling round-trip" begin
        rng = MersenneTwister(9)
        n_vals, β_vals, τ_vals = Int[], Float64[], Float64[]
        for n in 3:11, β in (5.0, 10.0, 20.0, 30.0)
            push!(n_vals, n); push!(β_vals, β)
            push!(τ_vals, 0.4 * n^2.1 * β^1.4 * exp(0.01 * randn(rng)))
        end
        fits = fit_scaling(n_vals, β_vals, τ_vals)
        for fit in values(fits)
            for i in eachindex(n_vals)
                τ_pred = predict_scaling(fit, n_vals[i], β_vals[i])
                @test isapprox(log(τ_pred), fit.log_tau_predicted[i]; atol = 1e-10)
            end
        end
    end

    # ------------------------------------------------------------------------
    # (g) formula_string contains expected tokens
    # ------------------------------------------------------------------------
    @testset "(g) formula_string formatting" begin
        rng = MersenneTwister(11)
        n_vals, β_vals, τ_vals = Int[], Float64[], Float64[]
        for n in 3:8, β in (5.0, 10.0, 20.0)
            push!(n_vals, n); push!(β_vals, β)
            push!(τ_vals, n^2.0 * β^1.5 * exp(0.005 * randn(rng)))
        end
        fits = fit_scaling(n_vals, β_vals, τ_vals)
        s0 = formula_string(fits[:M0])
        @test occursin("τ_mix", s0)
        @test occursin("n^", s0)
        # formula_string disambiguates β with the explicit kind suffix.
        @test occursin("β_alg^", s0)
        @test occursin("±", s0)

        s1 = formula_string(fits[:M1])
        @test occursin("τ_mix", s1)
        @test occursin("n^", s1)
        @test occursin("exp(", s1)
        @test occursin("·β_alg", s1)

        # explicit beta_kind=:phys tag flips the label in the formula.
        fits_phys = fit_scaling(n_vals, β_vals, τ_vals; beta_kind = :phys)
        s0_phys = formula_string(fits_phys[:M0])
        @test occursin("β_phys^", s0_phys)
    end

    # ------------------------------------------------------------------------
    # (h) Aliasing diagnostic: corr matrix is finite and symmetric
    # ------------------------------------------------------------------------
    @testset "(h) Correlation matrix diagnostics" begin
        rng = MersenneTwister(13)
        n_vals, β_vals, τ_vals = Int[], Float64[], Float64[]
        for n in 3:11, β in (5.0, 10.0, 20.0, 30.0)
            push!(n_vals, n); push!(β_vals, β)
            push!(τ_vals, 0.5 * n^2.3 * β^1.7 * exp(0.005 * randn(rng)))
        end
        fits = fit_scaling(n_vals, β_vals, τ_vals)
        for fit in values(fits)
            @test all(isfinite, fit.corr_matrix)
            @test all(isfinite, fit.cov_matrix)
            # Symmetry up to round-off
            @test maximum(abs.(fit.corr_matrix - fit.corr_matrix')) < 1e-10
            # Diagonal is 1
            for i in 1:3
                @test isapprox(fit.corr_matrix[i, i], 1.0; atol = 1e-10)
            end
            # All off-diagonals are correlations in [-1, 1]
            for i in 1:3, j in 1:3
                @test -1.0 - 1e-10 ≤ fit.corr_matrix[i, j] ≤ 1.0 + 1e-10
            end
        end
    end

    # ------------------------------------------------------------------------
    # (i) scaling_fit_grid evaluates the model on a regular grid
    # ------------------------------------------------------------------------
    @testset "(i) scaling_fit_grid for diagnostic plots" begin
        rng = MersenneTwister(17)
        n_vals, β_vals, τ_vals = Int[], Float64[], Float64[]
        for n in 3:8, β in (5.0, 10.0, 20.0)
            push!(n_vals, n); push!(β_vals, β)
            push!(τ_vals, n^2.0 * β^1.5 * exp(0.005 * randn(rng)))
        end
        fits = fit_scaling(n_vals, β_vals, τ_vals)
        grid = scaling_fit_grid(fits[:M0])
        @test grid.n_grid == collect(3:8)
        @test length(grid.beta_grid) == 21
        @test minimum(grid.beta_grid) ≈ 5.0
        @test maximum(grid.beta_grid) ≈ 20.0
        @test size(grid.tau_predicted) == (length(grid.n_grid), length(grid.beta_grid))
        @test all(>(0), grid.tau_predicted)
        @test length(grid.tau_obs) == length(n_vals)
        @test length(grid.tau_pred_at_obs) == length(n_vals)
        @test length(grid.residuals_log) == length(n_vals)
        for i in eachindex(grid.n_grid), j in eachindex(grid.beta_grid)
            @test grid.tau_predicted[i, j] ==
                predict_scaling(fits[:M0], grid.n_grid[i], grid.beta_grid[j])
        end
        @test grid.tau_pred_at_obs == exp.(fits[:M0].log_tau_predicted)
        @test grid.residuals_log ≈ log.(grid.tau_obs) .- log.(grid.tau_pred_at_obs)

        # Explicit grids honoured
        grid2 = scaling_fit_grid(fits[:M0]; n_grid = [3, 5, 7], beta_grid = [5.0, 10.0])
        @test grid2.n_grid == [3, 5, 7]
        @test grid2.beta_grid ≈ [5.0, 10.0]
        @test size(grid2.tau_predicted) == (3, 2)
        for i in eachindex(grid2.n_grid), j in eachindex(grid2.beta_grid)
            @test grid2.tau_predicted[i, j] ==
                predict_scaling(fits[:M0], grid2.n_grid[i], grid2.beta_grid[j])
        end
    end

    @testset "(j) result rows select physical or algorithmic beta" begin
        rescale = 3.0
        rows_phys = [(
            n=n,
            beta=βp * rescale,
            beta_alg=βp * rescale,
            beta_phys=βp,
            rescaling_factor=rescale,
            mixing_time=Float64(n)^2 * βp^1.5,
            mixing_time_source=:extrapolated,
        ) for n in 3:8 for βp in (1.0, 2.0, 3.0)]
        fit_phys = fit_scaling(rows_phys)[:M0]
        @test fit_phys.beta_kind === :phys
        @test Set(fit_phys.beta_values) == Set((1.0, 2.0, 3.0))

        rows_legacy = [(
            n=n,
            beta=β,
            mixing_time=Float64(n)^2 * β^1.5,
            mixing_time_source=:extrapolated,
        ) for n in 3:8 for β in (5.0, 10.0, 20.0)]
        fit_legacy = fit_scaling(rows_legacy)[:M0]
        @test fit_legacy.beta_kind === :alg
        @test Set(fit_legacy.beta_values) == Set((5.0, 10.0, 20.0))

        fit_alg = fit_scaling(rows_phys; beta_kind=:alg)[:M0]
        @test fit_alg.beta_kind === :alg
        @test Set(fit_alg.beta_values) == Set((3.0, 6.0, 9.0))

        # All redundant metadata are checked, even when one convention is
        # explicitly selected for the fit.
        inconsistent_alias = copy(rows_phys)
        inconsistent_alias[1] = merge(
            inconsistent_alias[1], (beta=inconsistent_alias[1].beta + 0.25,))
        @test_throws ArgumentError fit_scaling(inconsistent_alias; beta_kind=:phys)

        inconsistent_rescaling = copy(rows_phys)
        inconsistent_rescaling[1] = merge(
            inconsistent_rescaling[1], (rescaling_factor=4.0,))
        @test_throws ArgumentError fit_scaling(inconsistent_rescaling; beta_kind=:alg)

        invalid_redundant = copy(rows_phys)
        invalid_redundant[1] = merge(invalid_redundant[1], (beta_phys=NaN,))
        @test_throws ArgumentError fit_scaling(invalid_redundant; beta_kind=:alg)

        # Missing algorithmic beta is derived only when the physical beta and
        # positive rescaling factor are both present.
        rows_phys_only = [(
            n=n,
            beta_phys=βp,
            rescaling_factor=rescale,
            mixing_time=Float64(n)^2 * βp^1.5,
            mixing_time_source=:extrapolated,
        ) for n in 3:8 for βp in (1.0, 2.0, 3.0)]
        fit_derived_alg = fit_scaling(rows_phys_only; beta_kind=:alg)[:M0]
        @test fit_derived_alg.beta_kind === :alg
        @test Set(fit_derived_alg.beta_values) == Set((3.0, 6.0, 9.0))
    end

    # ------------------------------------------------------------------------
    # (k) AICc weights edge cases
    # ------------------------------------------------------------------------
    @testset "(k) AICc weights edge cases" begin
        # Empty dict
        @test isempty(aicc_weights(Dict{Symbol, ScalingFit}()))
        @test_throws ArgumentError compare_models(Dict{Symbol, ScalingFit}())

        metrics = QuantumFurnace._scaling_aic_metrics(0.0, 10, 3)
        @test metrics.aicc == -Inf
        @test metrics.aic == -Inf
        @test metrics.log_likelihood == Inf

        n_vals = [3, 4, 5, 6, 7, 8]
        β_vals = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        τ_vals = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0]
        template = fit_scaling(n_vals, β_vals, τ_vals; models=(:M0,))[:M0]
        function with_aicc(fit::ScalingFit, value::Float64)
            fields = Any[getfield(fit, i) for i in 1:fieldcount(ScalingFit)]
            fields[findfirst(==(:aicc), fieldnames(ScalingFit))] = value
            return ScalingFit(fields...)
        end

        tied_exact = Dict(
            :M0 => with_aicc(template, -Inf),
            :M1 => with_aicc(template, -Inf),
        )
        tied_weights = aicc_weights(tied_exact)
        @test tied_weights == Dict(:M0 => 0.5, :M1 => 0.5)
        tied_comparison = compare_models(tied_exact)
        @test tied_comparison.delta_aicc == [0.0, 0.0]
        @test all(isfinite, tied_comparison.weights)

        exact_and_finite = Dict(
            :M0 => with_aicc(template, -Inf),
            :M1 => with_aicc(template, 3.0),
        )
        @test aicc_weights(exact_and_finite) == Dict(:M0 => 1.0, :M1 => 0.0)

        finite_and_unavailable = Dict(
            :M0 => with_aicc(template, 3.0),
            :M1 => with_aicc(template, Inf),
        )
        @test aicc_weights(finite_and_unavailable) == Dict(:M0 => 1.0, :M1 => 0.0)

        all_unavailable = Dict(
            :M0 => with_aicc(template, Inf),
            :M1 => with_aicc(template, Inf),
        )
        @test aicc_weights(all_unavailable) == Dict(:M0 => 0.5, :M1 => 0.5)
        unavailable_comparison = compare_models(all_unavailable)
        @test unavailable_comparison.delta_aicc == [0.0, 0.0]
        @test all(isfinite, unavailable_comparison.weights)

        @test_throws ArgumentError aicc_weights(Dict(
            :M0 => with_aicc(template, NaN),
        ))
    end
end
