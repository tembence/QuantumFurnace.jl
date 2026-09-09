using StableRNGs

@testset "Fitting" begin

    # -----------------------------------------------------------------------
    # basic exponential decay recovery (clean data)
    # -----------------------------------------------------------------------
    @testset "basic exponential decay recovery" begin
        A_true = 2.0
        gap_true = 0.5
        C_true = 0.3
        times = collect(0.0:0.1:20.0)
        values = A_true .* exp.(-gap_true .* times) .+ C_true

        result = fit_exponential_decay(times, values)

        @test result isa FitResult
        @test result.converged == true
        @test isapprox(result.gap, gap_true; atol=1e-6)
        @test isapprox(result.amplitude, A_true; atol=1e-6)
        @test isapprox(result.offset, C_true; atol=1e-6)
        @test result.r_squared > 0.999
    end

    # -----------------------------------------------------------------------
    # noisy data recovery within CI
    # -----------------------------------------------------------------------
    @testset "noisy data recovery within CI" begin
        rng = StableRNG(42)
        A_true = 2.0
        gap_true = 0.5
        C_true = 0.3
        times = collect(0.0:0.1:20.0)
        values = A_true .* exp.(-gap_true .* times) .+ C_true .+ 0.05 .* randn(rng, length(times))

        result = fit_exponential_decay(times, values)

        @test result.converged == true
        @test result.gap_ci[1] <= gap_true <= result.gap_ci[2]
        @test result.gap_se > 0.0
        @test result.r_squared > 0.9
        @test length(result.residuals) == length(times)
        @test result.times_used == times
        @test result.values_used == values
    end

    # -----------------------------------------------------------------------
    # auto-generated initial guess (log-linear)
    # -----------------------------------------------------------------------
    @testset "auto-generated initial guess (log-linear)" begin
        A_true = 5.0
        gap_true = 1.5
        C_true = -0.5
        times = collect(0.0:0.05:5.0)
        values = A_true .* exp.(-gap_true .* times) .+ C_true

        result = fit_exponential_decay(times, values)

        @test result.converged == true
        @test isapprox(result.gap, gap_true; atol=0.01)

    end

    # -----------------------------------------------------------------------
    # log-linear fallback for difficult data
    # -----------------------------------------------------------------------
    @testset "log-linear fallback for difficult data" begin
        times = collect(0.0:0.1:4.9)
        values = fill(1.0, 50)

        p0 = QuantumFurnace._log_linear_initial_guess(times, values)
        @test length(p0) == 3
        @test all(isfinite, p0)
    end

    # -----------------------------------------------------------------------
    # skip_initial window selection
    # -----------------------------------------------------------------------
    @testset "skip_initial window selection" begin
        gap_true = 0.3
        times = collect(0.0:0.05:10.0)
        # Data with fast-decaying transient added to the slow exponential
        values = 2.0 .* exp.(-gap_true .* times) .+ 0.5 .+ 3.0 .* exp.(-2.0 .* times)

        r1 = fit_exponential_decay(times, values)
        r2 = fit_exponential_decay(times, values; skip_initial=0.3)

        @test abs(r2.gap - gap_true) < abs(r1.gap - gap_true)
        @test length(r2.times_used) < length(r1.times_used)
    end

    # -----------------------------------------------------------------------
    # quality metrics present and correct
    # -----------------------------------------------------------------------
    @testset "quality metrics present and correct" begin
        times = collect(0.0:0.1:10.0)
        values = 2.0 .* exp.(-0.5 .* times) .+ 0.3

        result = fit_exponential_decay(times, values)

        @test result.r_squared isa Float64
        @test result.gap_ci isa Tuple{Float64, Float64}
        @test result.gap_ci[1] < result.gap_ci[2]
        @test result.gap_se isa Float64
        @test result.gap_se >= 0.0
        @test result.converged isa Bool
        @test result.residuals isa Vector{Float64}
    end

    # -----------------------------------------------------------------------
    # non-negative gap enforced via bounds
    # -----------------------------------------------------------------------
    @testset "non-negative gap bound" begin
        times = collect(0.0:0.1:10.0)
        # Negative amplitude: rising exponential that LM might explore negative gap for
        values = -1.5 .* exp.(-0.3 .* times) .+ 2.0

        result = fit_exponential_decay(times, values)

        @test result.gap >= 0.0
    end

    # -----------------------------------------------------------------------
    # R-squared not clamped for bad fits
    # -----------------------------------------------------------------------
    @testset "R-squared not clamped for bad fits" begin
        times = collect(0.0:0.1:10.0)
        values = sin.(times)

        result = fit_exponential_decay(times, values)

        @test result.r_squared < 0.5
    end

    # -----------------------------------------------------------------------
    # Custom p0 override
    # -----------------------------------------------------------------------
    @testset "custom p0 override" begin
        A_true = 2.0
        gap_true = 0.5
        C_true = 0.3
        times = collect(0.0:0.1:20.0)
        values = A_true .* exp.(-gap_true .* times) .+ C_true

        result = fit_exponential_decay(times, values; p0=[2.0, 0.5, 0.3])

        @test result.converged == true
        @test isapprox(result.gap, gap_true; atol=1e-6)
    end

    # ===================================================================
    # Bi-exponential fitting tests
    # ===================================================================

    # -----------------------------------------------------------------------
    # Clean bi-exponential data recovery
    # -----------------------------------------------------------------------
    @testset "clean bi-exp data recovery" begin
        A1_true = 1.0    # fast amplitude
        g1_true = 2.0    # fast gap
        A2_true = 0.5    # slow amplitude
        g2_true = 0.3    # slow gap (spectral gap estimate)
        C_true  = 0.001  # offset

        times = collect(0.0:0.05:30.0)
        values = A1_true .* exp.(-g1_true .* times) .+
                 A2_true .* exp.(-g2_true .* times) .+
                 C_true

        result = fit_biexponential_decay(times, values)

        @test result isa BiexpFitResult
        @test result.converged == true
        @test result.r_squared > 0.999

        # Slow mode (spectral gap)
        @test isapprox(result.gap, g2_true; rtol=1e-7)
        @test isapprox(result.amplitude, A2_true; rtol=1e-7)

        # Fast mode
        @test isapprox(result.gap_fast, g1_true; rtol=1e-7)
        @test isapprox(result.amplitude_fast, A1_true; rtol=1e-7)

        # Offset
        @test isapprox(result.offset, C_true; atol=1e-12)

        # Mode sorting: fast >= slow
        @test result.gap_fast >= result.gap

        @info "Biexponential fit recovery" gap_slow=result.gap gap_fast=result.gap_fast offset=result.offset r2=result.r_squared
    end

    # -----------------------------------------------------------------------
    # Offset accuracy — bi-exp closer to true C than single-exp
    # -----------------------------------------------------------------------
    @testset "offset accuracy vs single-exp" begin
        # This is the key validation: bi-exp should give more accurate offset
        # when data has two timescales
        A1_true = 1.0    # fast
        g1_true = 2.0    # fast gap
        A2_true = 0.5    # slow
        g2_true = 0.3    # slow gap
        C_true  = 6.8e-5 # small offset (like floor from coherent unitary)

        times = collect(0.0:0.1:40.0)
        values = A1_true .* exp.(-g1_true .* times) .+
                 A2_true .* exp.(-g2_true .* times) .+
                 C_true

        single_fit = fit_exponential_decay(times, values; skip_initial=0.2)
        biexp_fit  = fit_biexponential_decay(times, values; skip_initial=0.2)

        single_err = abs(single_fit.offset - C_true)
        biexp_err  = abs(biexp_fit.offset - C_true)

        @test biexp_err < single_err
        @test biexp_err <= 1e-12
        @info "offset comparison" true_C=C_true single_C=single_fit.offset biexp_C=biexp_fit.offset single_err=single_err biexp_err=biexp_err
    end

    # -----------------------------------------------------------------------
    # skip_initial works with bi-exp
    # -----------------------------------------------------------------------
    @testset "skip_initial with bi-exp" begin
        A1_true = 1.0
        g1_true = 2.0
        A2_true = 0.5
        g2_true = 0.3
        C_true  = 0.001

        times = collect(0.0:0.05:30.0)
        values = A1_true .* exp.(-g1_true .* times) .+
                 A2_true .* exp.(-g2_true .* times) .+
                 C_true

        r1 = fit_biexponential_decay(times, values; skip_initial=0.0)
        r2 = fit_biexponential_decay(times, values; skip_initial=0.2)

        @test length(r2.times_used) < length(r1.times_used)
        @test isapprox(r1.gap, g2_true; rtol=1e-7)
        @test isapprox(r2.gap, g2_true; rtol=1e-7)
        @test isapprox(r1.gap_fast, g1_true; rtol=1e-7)
        @test isapprox(r2.gap_fast, g1_true; rtol=1e-7)
    end

    @testset "fitted distance is nonnegative and nonincreasing" begin
        times = collect(range(0.0, 10.0; length=101))
        increasing_data = 0.2 .+ 0.03 .* times
        result = fit_biexponential_decay(
            times, increasing_data; p0=[0.1, 1.0, 0.1, 0.2, 0.2])

        @test result.amplitude >= 0.0
        @test result.amplitude_fast >= 0.0
        @test result.gap >= 0.0
        @test result.gap_fast >= 0.0
        @test result.offset >= 0.0

        fitted = result.amplitude_fast .* exp.(-result.gap_fast .* times) .+
                 result.amplitude .* exp.(-result.gap .* times) .+ result.offset
        @test all(fitted .>= 0.0)
        @test all(diff(fitted) .<= 10eps(Float64))
    end

    @testset "physical input and initial-guess validation" begin
        times = collect(range(0.0, 5.0; length=21))
        values = exp.(-times)

        @test_throws ArgumentError fit_biexponential_decay(
            times, values; p0=[-0.1, 1.0, 0.1, 0.2, 0.0])
        @test_throws ArgumentError fit_biexponential_decay(
            times, values; p0=[0.1, 1.0, 0.1, 0.2])
        @test_throws ArgumentError fit_biexponential_decay(
            times, [values[1:end-1]; -0.1])
        @test_throws ArgumentError fit_biexponential_decay(
            [times[1:end-1]; times[end-1]], values)
        @test_throws ArgumentError fit_biexponential_decay(
            [-1.0; times[2:end]], values)
    end

    # -----------------------------------------------------------------------
    # BIEXP edge case: too few data points
    # -----------------------------------------------------------------------
    @testset "BIEXP: too few data points throws" begin
        times = collect(0.0:1.0:5.0)  # 6 points
        values = exp.(-0.3 .* times)
        @test_throws ArgumentError fit_biexponential_decay(times, values)
    end

    @testset "Singular fits retain parameters without uncertainty estimates" begin
        times = collect(range(0.0, 10.0; length=21))
        result = fit_exponential_decay(times, ones(length(times)); p0=[0.0, 0.0, 1.0])
        @test isapprox(result.offset, 1.0; atol=1e-12)
        @test isinf(result.gap_se)
        @test result.gap_ci[1] == -Inf
        @test result.gap_ci[2] == Inf
    end

    @testset "Biexponential mode ordering preserves slow-gap uncertainty" begin
        times = collect(range(0.0, 30.0; length=301))
        values = 0.8 .* exp.(-1.2 .* times) .+ 1.2 .* exp.(-0.16 .* times) .+
            0.001 .+ 1e-4 .* sin.(times)
        fast_first = fit_biexponential_decay(times, values;
            p0=[0.8, 1.2, 1.2, 0.16, 0.001])
        slow_first = fit_biexponential_decay(times, values;
            p0=[1.2, 0.16, 0.8, 1.2, 0.001])
        for field in (:gap, :gap_fast, :amplitude, :amplitude_fast, :offset, :gap_se)
            @test isapprox(getfield(fast_first, field), getfield(slow_first, field);
                rtol=1e-7, atol=1e-10)
        end
        @test all(isapprox.(fast_first.gap_ci, slow_first.gap_ci; rtol=1e-7, atol=1e-10))
    end

end
