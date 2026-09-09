# Mixing-time post-processing for sampled curves and Krylov spectral data.

"""
    MixingTimeEstimate

Result of fitting a sampled distance-to-equilibrium curve.

# Fields
- `fitted_gap`, `amplitude`, `offset`: Slow fitted mode.
- `gap_ci`, `gap_se`, `r_squared`, `converged`: Fit diagnostics.
- `mixing_time`: Selected actual or extrapolated answer.
- `mixing_time_extrapolated`, `mixing_time_actual`, `target_epsilon`: Crossing details.
- `fit_result`, `biexp_fit_result`, `model_used`: Underlying model results.
"""
struct MixingTimeEstimate
    # Fit parameters (from FitResult)
    fitted_gap::Float64
    amplitude::Float64
    offset::Float64
    gap_ci::Tuple{Float64, Float64}
    gap_se::Float64
    r_squared::Float64
    converged::Bool
    # Mixing time results
    mixing_time::Float64
    mixing_time_extrapolated::Union{Nothing, Float64}
    mixing_time_actual::Union{Nothing, Float64}
    target_epsilon::Union{Nothing, Float64}
    # Full fit for advanced users
    fit_result::FitResult
    # Fitted model and optional two-rate details.
    model_used::Symbol
    biexp_fit_result::Union{Nothing, BiexpFitResult}
end

"""
    _check_fit_quality(fit, target_epsilon; model_label="Fit")

Issue `@warn` messages for quality gate violations. Does not throw.
"""
function _check_fit_quality(
    fit::Union{FitResult, BiexpFitResult},
    target_epsilon::Union{Nothing, Float64};
    model_label::AbstractString="Fit",
)
    if fit.r_squared < 0.95
        @warn "$model_label R-squared = $(fit.r_squared) < 0.95. The fitted decay model may not describe the data well."
    end
    if target_epsilon !== nothing && fit.offset > 0.1 * target_epsilon
        @warn "Fit offset C = $(fit.offset) is large relative to target epsilon = $(target_epsilon). Extrapolation may be unreliable."
    end
    if !fit.converged
        @warn "Levenberg-Marquardt optimization did not converge. Fit results are unreliable."
    end
    if fit.gap_se > 0.5 * fit.gap && isfinite(fit.gap_se)
        @warn "Gap standard error ($(fit.gap_se)) exceeds 50% of fitted gap ($(fit.gap)). Confidence interval is very wide."
    end
    return nothing
end

"""
    _find_actual_mixing_time(times, dists, target_epsilon)

Return the first sampled threshold crossing, or `nothing` if none exists.
"""
function _find_actual_mixing_time(
    times::AbstractVector{<:Real},
    dists::AbstractVector{<:Real},
    target_epsilon::Union{Nothing, Float64},
)
    target_epsilon === nothing && return nothing
    for i in eachindex(dists)
        if dists[i] <= target_epsilon
            return Float64(times[i])
        end
    end
    return nothing  # target not reached in data
end

"""
    _extrapolate_mixing_time(fit::FitResult, target_epsilon)

Solve the single-exponential fit for a target crossing.

Return `nothing` when the parameters do not define a future crossing.
"""
function _extrapolate_mixing_time(fit::FitResult, target_epsilon::Union{Nothing, Float64})
    target_epsilon === nothing && return nothing
    fit.gap <= 0.0 && return nothing
    fit.amplitude <= 0.0 && return nothing

    # Math: $t = -log((epsilon-C)/A) / Delta$.
    effective_target = target_epsilon - fit.offset
    effective_target <= 0.0 && return nothing   # offset exceeds target
    effective_target >= fit.amplitude && return nothing  # already below target at t=0

    return -log(effective_target / fit.amplitude) / fit.gap
end

"""
    _extrapolate_mixing_time_biexp(bifit::BiexpFitResult, target_epsilon)

Solve the bi-exponential fit for a target crossing by bisection.
"""
function _extrapolate_mixing_time_biexp(bifit::BiexpFitResult, target_epsilon::Union{Nothing, Float64})
    target_epsilon === nothing && return nothing
    isfinite(target_epsilon) && target_epsilon > 0.0 || return nothing

    parameters = (
        bifit.amplitude_fast,
        bifit.gap_fast,
        bifit.amplitude,
        bifit.gap,
        bifit.offset,
    )
    all(isfinite, parameters) || return nothing
    all(>=(0.0), parameters) || return nothing

    # A strict positive-to-negative bracket is required below.
    f0 = bifit.amplitude_fast + bifit.amplitude + bifit.offset
    residual_lo = f0 - target_epsilon
    isfinite(residual_lo) && residual_lo > 0.0 || return nothing

    asymptotic_floor = bifit.offset +
        (iszero(bifit.gap_fast) ? bifit.amplitude_fast : 0.0) +
        (iszero(bifit.gap) ? bifit.amplitude : 0.0)
    isfinite(asymptotic_floor) && asymptotic_floor < target_epsilon || return nothing

    # Define the function to find root of: f(t) - epsilon = 0
    function biexp_residual(t)
        return bifit.amplitude_fast * exp(-bifit.gap_fast * t) +
               bifit.amplitude * exp(-bifit.gap * t) +
               bifit.offset - target_epsilon
    end

    positive_rates = Float64[]
    bifit.amplitude_fast > 0.0 && bifit.gap_fast > 0.0 &&
        push!(positive_rates, bifit.gap_fast)
    bifit.amplitude > 0.0 && bifit.gap > 0.0 &&
        push!(positive_rates, bifit.gap)
    isempty(positive_rates) && return nothing

    t_upper = 10.0 / minimum(positive_rates)
    isfinite(t_upper) && t_upper > 0.0 || return nothing
    residual_hi = biexp_residual(t_upper)
    for _ in 1:8
        isfinite(residual_hi) || return nothing
        residual_hi < 0.0 && break
        t_upper *= 2.0
        isfinite(t_upper) || return nothing
        residual_hi = biexp_residual(t_upper)
    end
    isfinite(residual_hi) && residual_hi < 0.0 || return nothing

    try
        t_mix = Roots.find_zero(biexp_residual, (0.0, t_upper), Roots.Bisection())
        return t_mix
    catch
        return nothing
    end
end

"""
    _biexp_to_single_fit_result(bifit::BiexpFitResult) -> FitResult

Project a `BiexpFitResult` onto its slow-mode `FitResult` fields.
"""
function _biexp_to_single_fit_result(bifit::BiexpFitResult)
    return FitResult(
        bifit.gap,          # gap (slow mode)
        bifit.amplitude,    # amplitude (slow mode)
        bifit.offset,       # offset
        bifit.gap_ci,       # gap_ci (slow mode)
        bifit.gap_se,       # gap_se (slow mode)
        bifit.r_squared,    # r_squared (from full bi-exp fit)
        bifit.converged,    # converged
        bifit.residuals,    # residuals (from full bi-exp fit)
        bifit.times_used,   # times_used
        bifit.values_used,  # values_used
    )
end

"""
    estimate_mixing_time(times, distances; kwargs...) -> MixingTimeEstimate

Fit a decay curve and optionally extrapolate its target crossing.

# Arguments
- `times`: Sample times.
- `distances`: Non-negative distance or observable values.

# Keywords
- `skip_initial`: Fraction of leading samples to discard.
- `target_epsilon`: Target value; required for extrapolation.
- `extrapolate`: Solve the fitted model rather than use a sampled crossing.
- `level`: Confidence level for the gap interval.
- `model`: `:single` or `:biexp`.

# Returns
A [`MixingTimeEstimate`](@ref) with crossing and fit diagnostics.
"""
function estimate_mixing_time(
    times::AbstractVector{<:Real},
    distances::AbstractVector{<:Real};
    skip_initial::Real = 0.2,
    target_epsilon::Union{Nothing, Real} = nothing,
    extrapolate::Bool = false,
    level::Real = 0.95,
    model::Symbol = :biexp,
)::MixingTimeEstimate
    length(times) == length(distances) || throw(ArgumentError(
        "times and distances must have the same length (got $(length(times)) and $(length(distances)))"))
    length(times) >= 10 || throw(ArgumentError(
        "Need at least 10 data points for mixing time estimation (got $(length(times)))"))
    all(value -> value isa Real && isfinite(value), times) || throw(ArgumentError(
        "times must contain only finite real values"))
    all(>=(0), times) || throw(ArgumentError("times must be nonnegative"))
    all(value -> value isa Real && isfinite(value) && value >= 0, distances) ||
        throw(ArgumentError("distances must contain only finite nonnegative real values"))
    all(diff(times) .> 0) || throw(ArgumentError(
        "times must be strictly increasing"))

    if extrapolate && target_epsilon === nothing
        throw(ArgumentError("target_epsilon required when extrapolate=true"))
    end

    model in (:single, :biexp) || throw(ArgumentError(
        "model must be :single or :biexp (got :$model)"))

    skip_initial_f = Float64(skip_initial)
    level_f        = Float64(level)
    target_eps_f   = target_epsilon === nothing ? nothing : Float64(target_epsilon)
    target_eps_f === nothing || (isfinite(target_eps_f) && target_eps_f > 0.0) ||
        throw(ArgumentError("target_epsilon must be finite and > 0"))

    t_mix_actual = _find_actual_mixing_time(times, distances, target_eps_f)
    # Fit validation operates at Float64 precision; already-converted inputs need no copy.
    fit_times = eltype(times) === Float64 ? times : Float64.(times)
    fit_distances = eltype(distances) === Float64 ? distances : Float64.(distances)

    fit, bifit, t_mix_extrap = if model == :single
        single_fit = fit_exponential_decay(fit_times, fit_distances;
            skip_initial=skip_initial_f, level=level_f)
        _check_fit_quality(single_fit, target_eps_f)
        crossing = extrapolate ? _extrapolate_mixing_time(single_fit, target_eps_f) : nothing
        single_fit, nothing, crossing
    else
        biexp_fit = fit_biexponential_decay(fit_times, fit_distances;
            skip_initial=skip_initial_f, level=level_f)
        _check_fit_quality(biexp_fit, target_eps_f; model_label="Bi-exponential fit")
        crossing = extrapolate ? _extrapolate_mixing_time_biexp(biexp_fit, target_eps_f) : nothing
        _biexp_to_single_fit_result(biexp_fit), biexp_fit, crossing
    end

    mixing_time = if extrapolate
        t_mix_extrap !== nothing ? t_mix_extrap : NaN
    elseif target_eps_f !== nothing
        t_mix_actual !== nothing ? t_mix_actual : NaN
    else
        Float64(last(times))
    end

    return MixingTimeEstimate(
        fit.gap, fit.amplitude, fit.offset,
        fit.gap_ci, fit.gap_se, fit.r_squared, fit.converged,
        mixing_time, t_mix_extrap, t_mix_actual, target_eps_f,
        fit, model, bifit,
    )
end

"""
    estimate_mixing_time(result::ThermalizeResults; kwargs...) -> MixingTimeEstimate

Estimate mixing from a completed full-density-matrix simulation.

# Arguments
- `result`: Simulation result containing sample times and trace distances.

# Keywords
- `skip_initial`, `target_epsilon`, `extrapolate`, `level`: As in the vector method.
- `model`: Fit model; defaults to `:single` for this overload.

# Returns
A [`MixingTimeEstimate`](@ref). This is post-processing and does not rerun the simulation.
"""
function estimate_mixing_time(
    result::ThermalizeResults;
    skip_initial::Real = 0.2,
    target_epsilon::Union{Nothing, Real} = nothing,
    extrapolate::Bool = false,
    level::Real = 0.95,
    model::Symbol = :single,
)::MixingTimeEstimate
    return estimate_mixing_time(
        result.time_steps, result.trace_distances;
        skip_initial    = skip_initial,
        target_epsilon  = target_epsilon,
        extrapolate     = extrapolate,
        level           = level,
        model           = model,
    )
end

"""
    estimate_mixing_time(integrator_result::NamedTuple; kwargs...) -> MixingTimeEstimate

Estimate mixing from an integrator result containing `t` and `distances`.

Krylov predictor results are rejected because their spectral data should be
passed to [`eigenmode_mixing_time`](@ref) instead of curve fitting.
"""
function estimate_mixing_time(integrator_result::NamedTuple; kwargs...)::MixingTimeEstimate
    # Predictor spectral data require exact subspace bisection, not a curve fit.
    if haskey(integrator_result, :R_modes) || haskey(integrator_result, :eigenvalues)
        throw(ArgumentError(
            "estimate_mixing_time received a Krylov trajectory-predictor result " *
            "(it carries the spectral fields :R_modes / :eigenvalues). The " *
            "(bi)exponential curve fit must not be used on " *
            "predict_lindbladian_trajectory / predict_channel_trajectory output — " *
            "call `eigenmode_mixing_time(traj, target_epsilon)` (exact closed-form " *
            "bisection on the Krylov spectrum) instead. The curve-fit estimator " *
            "remains the right tool for the matrix-free integrator outputs " *
            "(lindblad_action_integrate / discriminant_action_integrate) and " *
            "ThermalizeResults."))
    end
    return estimate_mixing_time(
        integrator_result.t, integrator_result.distances;
        kwargs...,
    )
end

# Exact mixing-time bisection on a captured Krylov spectral expansion.

const _EIGENMODE_ZERO_TOL = 1e-10  # |λ| below this counts as the steady mode

"""
    eigenmode_mixing_time(eigenvalues, c, R_modes, rho_inf, sigma_beta,
                          target_epsilon; t_upper, atol, max_iters,
                          eigenvalue_zero_tol)
        -> NamedTuple

Bisect the trace-distance crossing of a biorthogonal Krylov expansion.

# Arguments
- `eigenvalues`: Lindbladian rates, including the steady mode.
- `c`: Biorthogonal expansion coefficients.
- `R_modes`: Right eigenmodes as matrices.
- `rho_inf`: Captured stationary state.
- `sigma_beta`: Reference Gibbs state in the same basis.
- `target_epsilon`: Trace-distance threshold.

# Keywords
- `t_upper`: Upper bracket; zero selects `max(10/gap, 50)`.
- `atol`: Absolute time-axis tolerance.
- `max_iters`: Maximum bisection iterations.
- `eigenvalue_zero_tol`: Magnitude below which a rate is stationary.

# Returns
A named tuple with `mixing_time`, `gap`, `floor_distance`, `source`, and
`n_evals`. `mixing_time` is `Inf` when the target is below the asymptotic
floor and `NaN` for degenerate input or a failed bracket.
"""
function eigenmode_mixing_time(
    eigenvalues::AbstractVector{<:Complex},
    c::AbstractVector{<:Complex},
    R_modes::AbstractVector{<:AbstractMatrix},
    rho_inf::AbstractMatrix,
    sigma_beta::AbstractMatrix,
    target_epsilon::Real;
    t_upper::Real = 0.0,
    atol::Real = 1e-3,
    max_iters::Int = 64,
    eigenvalue_zero_tol::Real = _EIGENMODE_ZERO_TOL,
)::NamedTuple
    h = length(eigenvalues)
    h == length(c) == length(R_modes) || throw(ArgumentError(
        "eigenvalues, c, R_modes must have the same length (got $h, $(length(c)), $(length(R_modes)))"))
    target_epsilon > 0.0 || throw(ArgumentError(
        "target_epsilon must be positive (got $target_epsilon)"))

    # Math: $d(infinity) = norm(rho_infinity-sigma_beta)_1 / 2$.
    floor_distance = sum(svdvals(rho_inf .- sigma_beta)) / 2

    # Math: $Delta = min_(lambda_i != 0) abs(Re(lambda_i))$.
    gap = Inf
    for i in 1:h
        abs(eigenvalues[i]) < eigenvalue_zero_tol && continue
        gi = abs(real(eigenvalues[i]))
        gi < gap && (gap = gi)
    end

    # Degenerate input: no non-steady modes captured.
    if !isfinite(gap) || gap <= 0.0
        return (
            mixing_time     = NaN,
            gap             = isfinite(gap) ? gap : NaN,
            floor_distance  = floor_distance,
            source          = :nan,
            n_evals         = 0,
        )
    end

    # Floor branch: target below asymptotic floor → no crossing.
    if floor_distance >= target_epsilon
        return (
            mixing_time     = Inf,
            gap             = gap,
            floor_distance  = floor_distance,
            source          = :floor,
            n_evals         = 0,
        )
    end

    d = size(rho_inf, 1)
    T = promote_type(eltype(rho_inf), eltype(sigma_beta), ComplexF64)
    rho_t = Matrix{T}(undef, d, d)
    floor_residual = Matrix{T}(rho_inf .- sigma_beta)
    eval_count = Ref(0)

    function d_at(t::Real)::Float64
        copyto!(rho_t, floor_residual)
        @inbounds for i in 1:h
            abs(eigenvalues[i]) < eigenvalue_zero_tol && continue
            phase = exp(eigenvalues[i] * t)
            rho_t .+= (c[i] * phase) .* R_modes[i]
        end
        # Defensive Hermitisation (mirrors predict_*_trajectory loop).
        hermitianize!(rho_t)
        eval_count[] += 1
        return sum(svdvals(rho_t)) / 2
    end

    # Bisection bracket. The eigenmode formula is monotonically decreasing in
    # t on the slow tail, so [0, t_upper] with d(t_upper) < ε brackets a root.
    t_hi = float(t_upper)
    if t_hi <= 0.0
        t_hi = max(10.0 / gap, 50.0)
    end

    d_hi = d_at(t_hi)
    if d_hi > target_epsilon
        # One 3× expansion. The slow mode decays like e^{-gap * t}, so
        # log(d(t)/d(t_hi)) ≈ -gap (t - t_hi); 3× is enough margin in
        # practice for any well-behaved spectrum.
        t_hi *= 3.0
        d_hi = d_at(t_hi)
        if d_hi > target_epsilon
            return (
                mixing_time     = NaN,
                gap             = gap,
                floor_distance  = floor_distance,
                source          = :nan,
                n_evals         = eval_count[],
            )
        end
    end

    # d(0) ≥ target_epsilon must hold for a crossing on [0, t_hi]; if not,
    # the trajectory is below target at t=0 — degenerate (or trivially mixed).
    d_zero = d_at(0.0)
    if d_zero <= target_epsilon
        return (
            mixing_time     = 0.0,
            gap             = gap,
            floor_distance  = floor_distance,
            source          = :extrapolated,
            n_evals         = eval_count[],
        )
    end

    residual(t::Float64) = d_at(t) - float(target_epsilon)

    t_mix = try
        # `atol` is the t-axis tolerance ⇒ pass as `xatol` to Roots (their
        # `atol` kw governs f(x)=d(t)-ε, which would couple to the slow
        # mode's amplitude and hide stride O(ε / |d'|) ≫ atol).
        Roots.find_zero(residual, (0.0, t_hi), Roots.Bisection();
                         xatol=float(atol), maxiters=max_iters)
    catch
        return (
            mixing_time     = NaN,
            gap             = gap,
            floor_distance  = floor_distance,
            source          = :nan,
            n_evals         = eval_count[],
        )
    end

    return (
        mixing_time     = float(t_mix),
        gap             = gap,
        floor_distance  = floor_distance,
        source          = :extrapolated,
        n_evals         = eval_count[],
    )
end

"""
    _channel_eigenmode_mixing_time(mu, c, R_modes, rho_inf, sigma_beta,
                                   target_epsilon, delta; t_upper=0,
                                   k_upper=nothing, max_steps=100_000,
                                   max_iters=64) -> NamedTuple

Find a channel trace-distance crossing on integer step counts.

The reconstruction uses `rho_k = rho_inf + sum_i c_i * mu_i^k * R_i`
directly. This avoids the unphysical principal-branch interpolation
`exp(k*delta*log(mu)/delta)` between channel applications when a mode is
negative or complex. `t_upper` is converted to an integer ceiling; `k_upper`
may instead provide the horizon directly, and `max_steps` caps either route.
The scan remains chronological but uses trace-norm dual witnesses to certify
steps that cannot cross without performing a dense decomposition at every
integer. `source=:horizon_exhausted` means no crossing was found within the
reported `search_horizon`; `:floor` is returned only when the decaying-mode
tail bound certifies that no later crossing is possible. The legacy
`source=:extrapolated` tag is retained for a crossing, while `crossing_kind`
and `mixing_steps` make the discrete semantics explicit. `atol`, `max_iters`,
and `eigenvalue_zero_tol` remain accepted for keyword compatibility with the
continuous helper; an integer scan has no fractional root tolerance.
"""
function _channel_eigenmode_mixing_time(
    eigenvalues::AbstractVector{<:Complex},
    c::AbstractVector{<:Complex},
    R_modes::AbstractVector{<:AbstractMatrix},
    rho_inf::AbstractMatrix,
    sigma_beta::AbstractMatrix,
    target_epsilon::Real,
    delta::Real;
    t_upper::Real = 0.0,
    k_upper::Union{Nothing, Integer} = nothing,
    max_steps::Integer = 100_000,
    atol::Real = 1e-3,
    max_iters::Int = 64,
    eigenvalue_zero_tol::Real = _EIGENMODE_ZERO_TOL,
)::NamedTuple
    h = length(eigenvalues)
    h == length(c) == length(R_modes) || throw(ArgumentError(
        "eigenvalues, c, R_modes must have the same length (got $h, $(length(c)), $(length(R_modes)))"))
    target_epsilon > 0.0 || throw(ArgumentError(
        "target_epsilon must be positive (got $target_epsilon)"))
    isfinite(delta) && delta > 0 || throw(ArgumentError(
        "delta must be finite and positive (got $delta)"))
    isfinite(t_upper) && t_upper >= 0 || throw(ArgumentError(
        "t_upper must be finite and nonnegative (got $t_upper)"))
    isnothing(k_upper) || k_upper >= 0 || throw(ArgumentError(
        "k_upper must be nonnegative (got $k_upper)"))
    0 <= max_steps < typemax(Int) || throw(ArgumentError(
        "max_steps must be a nonnegative integer below typemax(Int) (got $max_steps)"))
    isfinite(atol) && atol >= 0 || throw(ArgumentError(
        "atol must be finite and nonnegative (got $atol)"))
    max_iters > 0 || throw(ArgumentError("max_iters must be positive (got $max_iters)"))
    isfinite(eigenvalue_zero_tol) && eigenvalue_zero_tol >= 0 ||
        throw(ArgumentError("eigenvalue_zero_tol must be finite and nonnegative."))

    d = size(rho_inf, 1)
    size(rho_inf, 2) == d && size(sigma_beta) == (d, d) || throw(ArgumentError(
        "rho_inf and sigma_beta must be equal-size square matrices."))
    all(size(R) == (d, d) for R in R_modes) || throw(ArgumentError(
        "every channel mode must have the same size as rho_inf."))

    floor_distance = sum(svdvals(rho_inf .- sigma_beta)) / 2
    abs_mu2 = h >= 2 ? abs(eigenvalues[2]) : NaN
    gap = if isnan(abs_mu2)
        NaN
    elseif abs_mu2 == 0
        Inf
    else
        -log(abs_mu2) / float(delta)
    end

    max_steps_int = Int(max_steps)
    requested_horizon = if k_upper !== nothing
        k_upper
    elseif t_upper > 0
        raw_horizon = t_upper / delta
        raw_horizon > max_steps_int ? max_steps_int + 1 : ceil(Int, raw_horizon)
    elseif h < 2 || !(gap > 0 || gap == Inf)
        0
    elseif gap == Inf
        1
    else
        raw_horizon = max(10.0 / gap, 50.0) / delta
        raw_horizon > max_steps_int ? max_steps_int + 1 : ceil(Int, raw_horizon)
    end
    horizon_capped = requested_horizon > max_steps_int
    k_hi = horizon_capped ? max_steps_int : Int(requested_horizon)

    base_result(source, mixing_time, mixing_steps, n_evals;
                tail_bound=NaN) = (
        mixing_time = mixing_time,
        mixing_steps = mixing_steps,
        gap = gap,
        floor_distance = floor_distance,
        source = source,
        crossing_kind = :integer_steps,
        search_horizon = k_hi,
        horizon_capped = horizon_capped,
        tail_bound = tail_bound,
        n_evals = n_evals,
    )

    T = promote_type(
        eltype(eigenvalues), eltype(c), eltype(rho_inf),
        eltype(sigma_beta), ComplexF64)
    mu = Vector{T}(eigenvalues)
    coefficients = Vector{T}(c)
    rho_k = Matrix{T}(undef, d, d)
    floor_residual = Matrix{T}(rho_inf .- sigma_beta)
    witness_floor = Ref{T}(zero(T))
    witness_modes = Vector{T}(undef, h)
    eval_count = Ref(0)

    function fill_difference!()
        copyto!(rho_k, floor_residual)
        @inbounds for i in 1:h
            rho_k .+= coefficients[i] .* R_modes[i]
        end
        return rho_k
    end

    function exact_distance_and_witness!()::Float64
        fill_difference!()
        # Physical channel powers preserve Hermiticity up to numerical error.
        # Form the projection from both untouched entries; sequential in-place
        # overwrites would bias one triangle toward the other.
        rho_k_hermitian = Hermitian((rho_k + adjoint(rho_k)) / 2)
        eig = eigen(rho_k_hermitian)
        distance = sum(abs, eig.values) / 2

        # Trace-norm duality: for Y = sign(A_k), ||Y||_inf <= 1 and hence
        # ||A_j||_1 / 2 >= |tr(Y A_j)| / 2. This witness lets the chronological
        # scan reject most steps using O(h) scalar work while retaining an exact
        # decomposition whenever the certified lower bound approaches epsilon.
        witness = eig.vectors * Diagonal(sign.(eig.values)) * eig.vectors'
        witness_floor[] = dot(witness, floor_residual)
        @inbounds for i in 1:h
            witness_modes[i] = dot(witness, R_modes[i])
        end
        eval_count[] += 1
        return Float64(distance)
    end

    d_zero = exact_distance_and_witness!()
    d_zero <= target_epsilon &&
        return base_result(:extrapolated, 0.0, 0, eval_count[])

    # Distance to rho_inf contracts for a CPTP channel, but distance to the
    # separate Gibbs reference need not: negative or complex modes can cross,
    # recede, and cross again. Scan every integer chronologically. A dual
    # witness may certify a non-crossing step, but it can never certify a
    # crossing; those always receive an exact Hermitian trace-norm evaluation.
    for k in 1:k_hi
        @inbounds for i in 1:h
            coefficients[i] *= mu[i]
        end
        witness_value = witness_floor[]
        witness_scale = abs(witness_floor[])
        @inbounds for i in 1:h
            term = coefficients[i] * witness_modes[i]
            witness_value += term
            witness_scale += abs(term)
        end
        witness_lower = abs(real(witness_value)) / 2
        roundoff_margin = sqrt(eps(Float64)) * max(1.0, witness_scale) / 2
        witness_lower > target_epsilon + roundoff_margin && continue

        if exact_distance_and_witness!() <= target_epsilon
            return base_result(
                :extrapolated, float(k) * float(delta), k, eval_count[])
        end
    end

    # Tail certificate. For A_k = F + sum_i c_i mu_i^k R_i,
    # |D(A_k)-D(F)| <= E(k), where
    # E(k) = sum_i |c_i| |mu_i|^k ||R_i||_1 / 2. If every active mode is
    # nonexpansive, E(j) <= E(k_hi) for all j >= k_hi. A strict lower bound
    # above epsilon therefore certifies that no unscanned step can cross.
    tail_bound = 0.0
    all_nonexpansive = true
    all_decaying = true
    @inbounds for i in 1:h
        radius_i = sum(svdvals(R_modes[i])) / 2
        weight_i = abs(c[i]) * radius_i
        abs_mu_i = abs(eigenvalues[i])
        tail_bound += weight_i * abs_mu_i^k_hi
        if weight_i > 0
            all_nonexpansive &= isfinite(abs_mu_i) && abs_mu_i <= 1
            all_decaying &= isfinite(abs_mu_i) && abs_mu_i < 1
        end
    end
    certificate_margin = sqrt(eps(Float64)) *
        max(1.0, floor_distance, tail_bound)
    if all_nonexpansive &&
       floor_distance - tail_bound > target_epsilon + certificate_margin
        source = all_decaying ? :floor : :certified_no_crossing
        return base_result(
            source, Inf, nothing, eval_count[]; tail_bound=tail_bound)
    end

    return base_result(
        :horizon_exhausted, NaN, nothing, eval_count[];
        tail_bound=tail_bound)
end

"""
    eigenmode_mixing_time(traj::NamedTuple, target_epsilon; kwargs...) -> NamedTuple

Compute mixing time directly from a Krylov predictor result.

Channel predictors are evaluated at integer powers `mu^k`; Lindbladian
eigenvalues are continuous-time rates. Remaining keywords forward to the
matching spectral-data method.
"""
function eigenmode_mixing_time(traj::NamedTuple, target_epsilon::Real; kwargs...)::NamedTuple
    for f in (:eigenvalues, :c, :R_modes, :rho_inf, :sigma_beta)
        haskey(traj, f) || throw(ArgumentError(
            "eigenmode_mixing_time(traj::NamedTuple, …) expects a Krylov " *
            "trajectory-predictor result with fields :eigenvalues, :c, :R_modes, " *
            ":rho_inf, :sigma_beta (missing :$f). Pass the output of " *
            "predict_lindbladian_trajectory / predict_channel_trajectory."))
    end
    if haskey(traj, :physical_channel) && !traj.physical_channel
        throw(ArgumentError(
            "eigenmode_mixing_time requires a physical trace-preserving channel; " *
            "the supplied trajectory is an unscaled GQSP polynomial surrogate."))
    end
    if haskey(traj, :delta_used)
        return _channel_eigenmode_mixing_time(
            ComplexF64.(traj.eigenvalues), traj.c, traj.R_modes,
            traj.rho_inf, traj.sigma_beta, target_epsilon, traj.delta_used;
            kwargs...)
    end
    if get(traj,:raw_reconstruction,false)
        haskey(traj,:stationary_projection_scope) || throw(ArgumentError(
            "Raw all-mode predictor payload needs a stationary projection before mixing-time estimation; use the facade modal_crossing report."))
        # The facade already separated its numerical stationary projection;
        # do not discard genuinely slow retained rates with a fixed cutoff.
        kwargs = merge((;eigenvalue_zero_tol=0.),(;kwargs...))
    end
    return eigenmode_mixing_time(
        ComplexF64.(traj.eigenvalues), traj.c, traj.R_modes,
        traj.rho_inf, traj.sigma_beta,
        target_epsilon; kwargs...)
end
