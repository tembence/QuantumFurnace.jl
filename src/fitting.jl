# Exponential decay fits.
# Math: $y(t) = A exp(-Delta t) + C$, with parameters `[A, Delta, C]`.

const _IDX_A   = 1
const _IDX_GAP = 2
const _IDX_C   = 3

_exp_decay_model(t, p) = @. p[_IDX_A] * exp(-p[_IDX_GAP] * t) + p[_IDX_C]

"""
    FitResult

Result of a single-exponential decay fit.

# Fields
- `gap`, `amplitude`, `offset`: Fitted model parameters.
- `gap_ci`, `gap_se`: Confidence interval and standard error for `gap`.
- `r_squared`, `converged`: Fit-quality diagnostics.
- `residuals`, `times_used`, `values_used`: Data retained after preprocessing.
"""
struct FitResult
    gap::Float64
    amplitude::Float64
    offset::Float64
    gap_ci::Tuple{Float64, Float64}
    gap_se::Float64
    r_squared::Float64
    converged::Bool
    residuals::Vector{Float64}
    times_used::Vector{Float64}
    values_used::Vector{Float64}
end

"""
    _log_linear_initial_guess(times, values) -> Vector{Float64}

Estimate `[A, gap, C]` from a log-linear fit after subtracting a tail plateau.
"""
function _log_linear_initial_guess(times::AbstractVector{<:Real}, values::AbstractVector{<:Real})
    n = length(times)

    # Estimate the plateau from the final 20% of samples.
    tail_start = max(1, n - div(n, 5))
    C_guess = sum(values[tail_start:n]) / (n - tail_start + 1)

    y_shifted = values .- C_guess
    mask = y_shifted .> 1e-10

    if sum(mask) < 3
        A_guess = values[1] - values[end]
        gap_guess = 1.0 / max(times[end] - times[1], eps(Float64))
        return [A_guess, gap_guess, C_guess]
    end

    t_valid = times[mask]
    log_y = log.(y_shifted[mask])

    # Math: $log(y-C) = log(A) - Delta t$.
    X = hcat(ones(length(t_valid)), t_valid)
    coeffs = X \ log_y

    A_guess = exp(coeffs[1])
    gap_guess = max(-coeffs[2], 1e-6)  # ensure positive

    return [A_guess, gap_guess, C_guess]
end

"""
    _compute_r_squared(values, residuals) -> Float64

Return `1 - RSS/TSS`; negative values indicate a fit worse than the mean.
"""
function _compute_r_squared(values::AbstractVector{<:Real}, residuals::AbstractVector{<:Real})
    ss_res = sum(residuals .^ 2)
    y_mean = sum(values) / length(values)
    ss_tot = sum((values .- y_mean) .^ 2)
    return 1.0 - ss_res / ss_tot
end

# A singular covariance leaves fitted parameters usable but uncertainty unavailable.
function _gap_uncertainty(fit, index, level)
    try
        se = stderror(fit)
        ci = confint(fit; level=level)
        return se[index], (ci[index][1], ci[index][2])
    catch e
        covariance_singular = e isa LinearAlgebra.SingularException ||
            (e isa LinearAlgebra.LAPACKException && e.info > 0)
        covariance_singular || rethrow(e)
        return Inf, (-Inf, Inf)
    end
end

"""
    fit_exponential_decay(times, values; skip_initial=0.0, p0=nothing, level=0.95) -> FitResult

Fit a non-negative single-exponential decay rate by nonlinear least squares.

# Arguments
- `times`: Sample times.
- `values`: Observations at the corresponding times.

# Keywords
- `skip_initial`: Fraction of leading samples to discard.
- `p0`: Optional initial `[A, gap, C]`; otherwise estimated automatically.
- `level`: Confidence level for the gap interval.

# Returns
A [`FitResult`](@ref) with parameters, uncertainties, and residuals.
"""
function fit_exponential_decay(
    times::AbstractVector{<:Real},
    values::AbstractVector{<:Real};
    skip_initial::Float64 = 0.0,
    p0::Union{Nothing, Vector{Float64}} = nothing,
    level::Float64 = 0.95,
)
    length(times) == length(values) ||
        throw(ArgumentError("times and values must have the same length (got $(length(times)) and $(length(values)))"))
    length(times) >= 4 ||
        throw(ArgumentError("need at least 4 data points for 3-parameter fit (got $(length(times)))"))
    0.0 <= skip_initial < 1.0 ||
        throw(ArgumentError("skip_initial must be in [0, 1) (got $skip_initial)"))

    start_idx = max(1, floor(Int, skip_initial * length(times)) + 1)
    times_fit = Float64.(times[start_idx:end])
    values_fit = Float64.(values[start_idx:end])

    length(times_fit) >= 4 ||
        throw(ArgumentError("fewer than 4 data points remain after skip_initial=$skip_initial (got $(length(times_fit)))"))

    p0_used = if p0 === nothing
        _log_linear_initial_guess(times_fit, values_fit)
    else
        Float64.(p0)
    end

    # Constrain the fitted decay rate to be non-negative.
    lower = [-Inf, 0.0, -Inf]
    upper = [Inf, Inf, Inf]

    # Unweighted fitting preserves LsqFit's covariance interpretation.
    fit = curve_fit(_exp_decay_model, times_fit, values_fit, p0_used;
                    lower=lower, upper=upper)

    params = coef(fit)
    gap     = params[_IDX_GAP]
    A       = params[_IDX_A]
    C       = params[_IDX_C]

    gap_se, gap_ci = _gap_uncertainty(fit, _IDX_GAP, level)

    resid   = residuals(fit)
    conv    = fit.converged

    r2 = _compute_r_squared(values_fit, resid)

    return FitResult(gap, A, C, gap_ci, gap_se, r2, conv, resid, times_fit, values_fit)
end

# Bi-exponential model parameters are `[A1, g1, A2, g2, C]`.
# Math: $y(t) = A_1 exp(-g_1 t) + A_2 exp(-g_2 t) + C$.

const _BIEXP_IDX_A1   = 1
const _BIEXP_IDX_G1   = 2
const _BIEXP_IDX_A2   = 3
const _BIEXP_IDX_G2   = 4
const _BIEXP_IDX_C    = 5

_biexp_decay_model(t, p) = @. p[_BIEXP_IDX_A1] * exp(-p[_BIEXP_IDX_G1] * t) +
                               p[_BIEXP_IDX_A2] * exp(-p[_BIEXP_IDX_G2] * t) +
                               p[_BIEXP_IDX_C]

"""
    BiexpFitResult

Result of a bi-exponential fit, sorted so `gap_fast >= gap`.

# Fields
- `gap`, `amplitude`: Slow-mode rate and amplitude.
- `gap_fast`, `amplitude_fast`: Fast-mode rate and amplitude.
- `offset`: Long-time plateau.
- `gap_ci`, `gap_se`, `r_squared`, `converged`: Fit diagnostics.
- `residuals`, `times_used`, `values_used`: Retained fit data.
"""
struct BiexpFitResult
    gap::Float64
    gap_fast::Float64
    amplitude::Float64
    amplitude_fast::Float64
    offset::Float64
    gap_ci::Tuple{Float64, Float64}
    gap_se::Float64
    r_squared::Float64
    converged::Bool
    residuals::Vector{Float64}
    times_used::Vector{Float64}
    values_used::Vector{Float64}
end

"""
    _biexp_initial_guess(values, single_fit::FitResult) -> Vector{Float64}

Seed `[A1, g1, A2, g2, C]` from a single-exponential fit and its residuals.
"""
function _biexp_initial_guess(
    values::AbstractVector{<:Real},
    single_fit::FitResult,
)
    # Slow mode from single-exp fit
    value_scale = max(maximum(values), eps(Float64))
    C_guess = max(single_fit.offset, 0.0)
    A2_guess = max(single_fit.amplitude, values[1] - C_guess, 0.1 * value_scale)
    g2_guess = max(single_fit.gap, 1e-6)

    # Residuals of single-exp fit
    resids = single_fit.residuals

    # Estimate fast mode from early residuals
    # If residuals show a fast-decaying pattern, use it
    n_early = max(3, div(length(resids), 4))
    early_resids = resids[1:n_early]
    A1_guess = maximum(abs.(early_resids))
    A1_guess = max(A1_guess, 0.1 * A2_guess, eps(value_scale))

    # Estimate fast gap from residual decay rate
    # Expect fast mode ~ 3-10x the slow gap
    g1_guess = max(3.0 * g2_guess, 1.0)

    return [A1_guess, g1_guess, A2_guess, g2_guess, C_guess]
end

"""
    fit_biexponential_decay(times, values; skip_initial=0.0, p0=nothing, level=0.95) -> BiexpFitResult

Fit a two-rate exponential decay by nonlinear least squares.

# Arguments
- `times`: Sample times.
- `values`: Observations at the corresponding times.

# Keywords
- `skip_initial`: Fraction of leading samples to discard.
- `p0`: Optional initial `[A1, g1, A2, g2, C]`.
- `level`: Confidence level for the slow-gap interval.

# Returns
A [`BiexpFitResult`](@ref) with rates sorted into fast and slow modes.
"""
function fit_biexponential_decay(
    times::AbstractVector{<:Real},
    values::AbstractVector{<:Real};
    skip_initial::Float64 = 0.0,
    p0::Union{Nothing, Vector{Float64}} = nothing,
    level::Float64 = 0.95,
)
    # --- Input validation ---
    length(times) == length(values) ||
        throw(ArgumentError("times and values must have the same length (got $(length(times)) and $(length(values)))"))
    length(times) >= 8 ||
        throw(ArgumentError("need at least 8 data points for 5-parameter bi-exponential fit (got $(length(times)))"))
    0.0 <= skip_initial < 1.0 ||
        throw(ArgumentError("skip_initial must be in [0, 1) (got $skip_initial)"))
    all(isfinite, times) || throw(ArgumentError("times must contain only finite values"))
    all(>=(0), times) || throw(ArgumentError("times must be nonnegative"))
    all(isfinite, values) || throw(ArgumentError("values must contain only finite values"))
    all(>=(0), values) || throw(ArgumentError(
        "bi-exponential distance values must be nonnegative"))
    all(diff(times) .> 0) || throw(ArgumentError(
        "times must be strictly increasing for a decay fit"))

    # --- Apply skip_initial ---
    start_idx = max(1, floor(Int, skip_initial * length(times)) + 1)
    times_fit = Float64.(times[start_idx:end])
    values_fit = Float64.(values[start_idx:end])

    length(times_fit) >= 8 ||
        throw(ArgumentError("fewer than 8 data points remain after skip_initial=$skip_initial (got $(length(times_fit)))"))
    # --- Generate initial guess ---
    p0_used = if p0 !== nothing
        Float64.(p0)
    else
        # Fit single-exp first, then use residuals to seed bi-exp
        single_fit = fit_exponential_decay(times_fit, values_fit; skip_initial=0.0, level=level)
        _biexp_initial_guess(values_fit, single_fit)
    end
    length(p0_used) == 5 || throw(ArgumentError(
        "p0 must contain [A1, g1, A2, g2, C]"))
    all(isfinite, p0_used) || throw(ArgumentError("p0 must contain only finite values"))
    all(>=(0.0), p0_used) || throw(ArgumentError(
        "p0 amplitudes, rates, and offset must be nonnegative"))

    # Nonnegative amplitudes, rates, and offset make the fitted distance
    # nonnegative and nonincreasing for all t >= 0.
    lower = zeros(5)
    upper = [Inf, Inf, Inf, Inf, Inf]

    # --- Fit using LsqFit.jl ---
    fit = curve_fit(_biexp_decay_model, times_fit, values_fit, p0_used;
                    lower=lower, upper=upper)

    params = coef(fit)
    # Parameter pairs are [amplitude, gap]; preserve the slow-gap index for its CI.
    fast_index, slow_index = params[_BIEXP_IDX_G1] >= params[_BIEXP_IDX_G2] ?
        (_BIEXP_IDX_G1, _BIEXP_IDX_G2) : (_BIEXP_IDX_G2, _BIEXP_IDX_G1)
    gap_se, gap_ci = _gap_uncertainty(fit, slow_index, level)

    resid = residuals(fit)
    conv  = fit.converged

    # --- R-squared ---
    r2 = _compute_r_squared(values_fit, resid)

    return BiexpFitResult(
        params[slow_index], params[fast_index],
        params[slow_index - 1], params[fast_index - 1], params[_BIEXP_IDX_C],
        gap_ci, gap_se, r2, conv, resid, times_fit, values_fit,
    )
end
