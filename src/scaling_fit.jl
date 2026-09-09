# Empirical mixing-time scaling fits.
# Math: M0 uses $log tau = c + x log n + y log beta$; M1 uses
# $log tau = c + x log n + alpha beta$.

# Both models store parameters as `[c, x, slope]`.
const _SCALING_IDX_C     = 1
const _SCALING_IDX_X     = 2
const _SCALING_IDX_SLOPE = 3   # y for M0, α for M1

# `xdata` is an N×2 matrix; column 1 is log(n), column 2 is log(β) for M0 or β for M1.
_scaling_model(xdata, p) = @views @. p[_SCALING_IDX_C] +
    p[_SCALING_IDX_X] * xdata[:, 1] + p[_SCALING_IDX_SLOPE] * xdata[:, 2]

"""
    ScalingFit

Result of one empirical mixing-time scaling fit in log space.

- `model`, `param_names`, `params`: Model identity and `[c, x, slope]` values.
- `std_errors`, `cis`, `cov_matrix`, `corr_matrix`: Parameter uncertainty.
- `aicc`, `log_likelihood`, `rss`, `sigma_residual`: Fit-quality metrics.
- `n_values`, `beta_values`, `beta_kind`: Input grid and temperature convention.
- `log_tau_observed`, `log_tau_predicted`, `residuals`: Log-space data and errors.
"""
struct ScalingFit
    model::Symbol
    param_names::NTuple{3, Symbol}
    params::Vector{Float64}
    std_errors::Vector{Float64}
    cis::Vector{Tuple{Float64, Float64}}
    cov_matrix::Matrix{Float64}
    corr_matrix::Matrix{Float64}
    aicc::Float64
    log_likelihood::Float64
    rss::Float64
    sigma_residual::Float64
    n_data::Int
    converged::Bool
    n_values::Vector{Int}
    beta_values::Vector{Float64}
    beta_kind::Symbol  # `:phys` or `:alg`.
    log_tau_observed::Vector{Float64}
    log_tau_predicted::Vector{Float64}
    residuals::Vector{Float64}
end

# Compute AIC, AICc, and Gaussian log-likelihood for a fit with `n_data`
# points and `n_model_params` regression parameters. Treats σ² as an
# additional free parameter (Burnham–Anderson convention for NLS), so
# k = n_model_params + 1 in the AIC penalty.
function _scaling_aic_metrics(rss::Real, n_data::Integer, n_model_params::Integer)
    N = Int(n_data)
    k = n_model_params + 1                          # +1 for σ²
    isfinite(rss) && rss >= 0 || throw(ArgumentError("rss must be finite and nonnegative"))
    if N <= k + 1
        # AICc denominator (N - k - 1) must be positive.
        # 3-param model + σ² ⇒ need N ≥ 6 for finite AICc.
        return (aicc = Inf, aic = Inf, log_likelihood = -Inf)
    end
    if iszero(rss)
        # The Gaussian MLE has σ²=0: log L=+Inf and both information
        # criteria are -Inf. Model weighting handles tied exact fits below.
        return (aicc = -Inf, aic = -Inf, log_likelihood = Inf)
    end
    σ²_mle = rss / N
    log_L = -N / 2 * (log(2π) + log(σ²_mle) + 1.0)
    aic = 2k - 2 * log_L
    aicc = aic + 2k * (k + 1) / (N - k - 1)
    return (aicc = aicc, aic = aic, log_likelihood = log_L)
end

# Build a ScalingFit from a finished `curve_fit` LsqFitResult plus bookkeeping.
function _build_scaling_fit(
    model::Symbol,
    param_names::NTuple{3, Symbol},
    fit,
    n_vals::AbstractVector{<:Integer},
    beta_vals::AbstractVector{<:Real},
    log_τ::AbstractVector{<:Real},
    xdata::AbstractMatrix{<:Real},
    level::Real;
    beta_kind::Symbol = :alg,
)
    beta_kind in (:phys, :alg) || throw(ArgumentError(
        "beta_kind must be :phys or :alg (got :$beta_kind)"))
    p = coef(fit)
    n_param = length(p)

    # SE / CI / covariance can fail for rank-deficient Jacobians (e.g., a
    # collinear (n, β) grid). LsqFit raises `SingularException` for an exactly
    # singular Jacobian and a plain `ErrorException` ("Covariance matrix is
    # negative …") when finite-difference numerics produce a non-PSD
    # covariance. Catch both and report Inf/NaN so the downstream consumer
    # can decide what to do.
    se, ci_vec, covmat = try
        se_v   = stderror(fit)
        ci_raw = confint(fit; level = level)            # already Vector{Tuple{Float64,Float64}}
        covm   = vcov(fit)                              # LsqFit ≥ 0.14; estimate_covar was deprecated
        se_v, Vector{Tuple{Float64, Float64}}(ci_raw), Matrix{Float64}(covm)
    catch e
        (e isa LinearAlgebra.SingularException || e isa ErrorException) || rethrow(e)
        fill(Inf, n_param),
        [(-Inf, Inf) for _ in 1:n_param],
        fill(NaN, n_param, n_param)
    end

    corrmat = if all(isfinite, covmat) && all(d -> d > 0, diag(covmat))
        s = sqrt.(diag(covmat))
        covmat ./ (s * s')
    else
        fill(NaN, n_param, n_param)
    end

    log_τ_pred = _scaling_model(xdata, p)
    # LsqFit.residuals uses model - data.  ScalingFit's public convention is
    # observed - predicted so positive residuals mean the model underpredicts.
    resid = log_τ .- log_τ_pred
    rss = sum(abs2, resid)
    metrics = _scaling_aic_metrics(rss, length(log_τ), n_param)
    σ_resid = length(log_τ) > 0 ? sqrt(rss / length(log_τ)) : NaN

    return ScalingFit(
        model, param_names,
        Vector{Float64}(p), Vector{Float64}(se), ci_vec, covmat, corrmat,
        metrics.aicc, metrics.log_likelihood, rss, σ_resid,
        length(log_τ), fit.converged,
        Vector{Int}(n_vals), Vector{Float64}(beta_vals), beta_kind,
        Vector{Float64}(log_τ), Vector{Float64}(log_τ_pred),
        Vector{Float64}(resid),
    )
end

"""
    fit_scaling(n_vals, beta_vals, tau_vals; models=(:M0, :M1), level=0.95)
        -> Dict{Symbol, ScalingFit}

Fit empirical scaling laws to `(n, beta, tau_mix)` samples in log space.

# Arguments
- `n_vals`: Positive system sizes.
- `beta_vals`: Positive inverse temperatures.
- `tau_vals`: Positive, finite mixing times.

# Keywords
- `models`: Subset of `(:M0, :M1)` to fit.
- `level`: Confidence level for parameter intervals.
- `beta_kind`: Whether temperatures are `:phys` or `:alg`.

# Returns
A dictionary keyed by model symbol. At least six samples are required for
finite AICc values.
"""
function fit_scaling(
    n_vals::AbstractVector{<:Integer},
    beta_vals::AbstractVector{<:Real},
    tau_vals::AbstractVector{<:Real};
    models = (:M0, :M1),
    level::Real = 0.95,
    beta_kind::Symbol = :alg,
)::Dict{Symbol, ScalingFit}
    beta_kind in (:phys, :alg) || throw(ArgumentError(
        "beta_kind must be :phys or :alg (got :$beta_kind)"))

    N = length(n_vals)
    length(beta_vals) == N || throw(ArgumentError(
        "n_vals and beta_vals must have the same length (got $N and $(length(beta_vals)))"))
    length(tau_vals) == N || throw(ArgumentError(
        "n_vals and tau_vals must have the same length (got $N and $(length(tau_vals)))"))
    N ≥ 6 || throw(ArgumentError(
        "need at least 6 data points for 3-parameter fit + σ² + AICc denominator (got $N)"))
    all(n -> n > 0, n_vals)    || throw(ArgumentError("all n must be positive"))
    all(b -> isfinite(b) && b > 0, beta_vals) || throw(ArgumentError(
        "all β must be positive and finite"))
    all(t -> t > 0 && isfinite(t), tau_vals) || throw(ArgumentError(
        "all τ_mix must be positive and finite (cannot take log of zero, negative, or non-finite)"))
    isempty(models) && throw(ArgumentError("models must be non-empty"))
    for m in models
        m in (:M0, :M1) || throw(ArgumentError("unknown model :$m (supported: :M0, :M1)"))
    end
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1) (got $level)"))

    log_n = log.(Float64.(n_vals))
    log_β = log.(Float64.(beta_vals))
    log_τ = log.(Float64.(tau_vals))

    out = Dict{Symbol, ScalingFit}()

    for model in (:M0, :M1)
        model in models || continue
        beta_data, names, p0 = if model === :M0
            log_β, (:c, :x, :y), [0.0, 1.0, 1.0]
        else
            Float64.(beta_vals), (:c, :x, :α), [0.0, 1.0, 0.1]
        end
        xdata = hcat(log_n, beta_data)
        fit = curve_fit(_scaling_model, xdata, log_τ, p0)
        out[model] = _build_scaling_fit(model, names, fit,
            n_vals, beta_vals, log_τ, xdata, level; beta_kind=beta_kind)
    end

    return out
end

"""
    fit_scaling(table::NamedTuple; kwargs...) -> Dict{Symbol, ScalingFit}

NamedTuple convenience wrapper. Expects fields `:n`, `:beta`, and `:tau_mix`.
"""
function fit_scaling(table::NamedTuple; kwargs...)
    haskey(table, :n) && haskey(table, :beta) && haskey(table, :tau_mix) || throw(ArgumentError(
        "table NamedTuple must have fields :n, :beta, :tau_mix (got $(keys(table)))"))
    return fit_scaling(table.n, table.beta, table.tau_mix; kwargs...)
end

"""
    fit_scaling(results::Vector{<:NamedTuple}; source_filter=(:extrapolated,),
                beta_kind=:auto, kwargs...) -> Dict{Symbol, ScalingFit}

Fit scaling laws from mixing-sweep named tuples.

# Keywords
- `source_filter`: Accepted mixing-time source tags; defaults to extrapolated cells.
- `beta_kind`: `:auto`, `:phys`, or `:alg`. Auto prefers `beta_phys`, then
  `beta_alg` or `beta`, and records the chosen convention.

When redundant temperature fields are present, `beta` must agree with
`beta_alg` and `beta_alg = beta_phys * rescaling_factor` must hold.
"""
function fit_scaling(
    results::Vector{<:NamedTuple};
    source_filter = (:extrapolated,),
    beta_kind::Symbol = :auto,
    kwargs...,
)
    beta_kind in (:auto, :phys, :alg) || throw(ArgumentError(
        "beta_kind must be :auto, :phys, or :alg (got :$beta_kind)"))

    # Channel sweeps use :tau_mix / :tau_mix_source; Lindbladian sweeps use
    # :mixing_time / :mixing_time_source. Support both.
    function _get_tau(r)
        if haskey(r, :mixing_time)
            return r.mixing_time
        elseif haskey(r, :tau_mix)
            return r.tau_mix
        end
        return nothing
    end
    function _get_source(r)
        if haskey(r, :mixing_time_source)
            return r.mixing_time_source
        elseif haskey(r, :tau_mix_source)
            return r.tau_mix_source
        end
        return nothing
    end

    function _positive_metadata(r, key::Symbol, row_index::Int)
        (!haskey(r, key) || getproperty(r, key) === nothing) && return nothing
        value = getproperty(r, key)
        value isa Real && isfinite(value) && value > 0 || throw(ArgumentError(
            "row $row_index has invalid $key metadata; expected a finite positive value"))
        return Float64(value)
    end

    function _temperature_metadata(r, row_index::Int)
        β_phys = _positive_metadata(r, :beta_phys, row_index)
        β_alg_tag = _positive_metadata(r, :beta_alg, row_index)
        β_alias = _positive_metadata(r, :beta, row_index)
        rescale = _positive_metadata(r, :rescaling_factor, row_index)

        if β_alg_tag !== nothing && β_alias !== nothing &&
                !isapprox(β_alg_tag, β_alias; rtol=1e-10, atol=0.0)
            throw(ArgumentError(
                "row $row_index has inconsistent beta_alg=$β_alg_tag and beta=$β_alias"))
        end
        β_alg = β_alg_tag === nothing ? β_alias : β_alg_tag

        if β_phys !== nothing && β_alg !== nothing && rescale !== nothing
            expected_alg = β_phys * rescale
            isapprox(β_alg, expected_alg; rtol=1e-10, atol=0.0) || throw(ArgumentError(
                "row $row_index violates beta_alg = beta_phys * rescaling_factor " *
                "($β_alg != $β_phys * $rescale)"))
        end
        return (; β_phys, β_alg, rescale)
    end

    # Select a temperature field and report the convention actually used.
    function _get_beta(r, kind::Symbol, row_index::Int)
        metadata = _temperature_metadata(r, row_index)
        β_phys = metadata.β_phys
        β_alg = metadata.β_alg
        rescale = metadata.rescale

        if kind === :phys
            β_phys !== nothing && return (β_phys, :phys)
            (β_alg !== nothing && rescale !== nothing) && return (β_alg / rescale, :phys)
            return (nothing, :phys)
        elseif kind === :alg
            β_alg !== nothing && return (β_alg, :alg)
            (β_phys !== nothing && rescale !== nothing) && return (β_phys * rescale, :alg)
            return (nothing, :alg)
        else  # :auto — prefer :beta_phys then :beta_alg/beta
            β_phys !== nothing && return (β_phys, :phys)
            β_alg  !== nothing && return (β_alg,  :alg)
            return (nothing, :alg)
        end
    end

    valid_pairs = Tuple{NamedTuple, Float64, Symbol}[]
    for (row_index, r) in enumerate(results)
        haskey(r, :n) || continue
        β, kind = _get_beta(r, beta_kind, row_index)
        (β === nothing || !(β isa Real) || !isfinite(β) || β <= 0) && continue
        τ = _get_tau(r)
        τ === nothing && continue
        (τ isa Real && isfinite(τ) && τ > 0) || continue
        src = _get_source(r)
        if !isempty(source_filter)
            (src !== nothing && src in source_filter) || continue
        end
        push!(valid_pairs, (r, β, kind))
    end
    isempty(valid_pairs) && throw(ArgumentError(
        "no valid entries found (need :n, finite-positive β " *
        "(:beta_phys / :beta_alg / :beta) and finite-positive :mixing_time " *
        "or :tau_mix with :mixing_time_source/:tau_mix_source ∈ $source_filter)"))

    n_vals = [Int(p[1].n) for p in valid_pairs]
    β_vals = [p[2]        for p in valid_pairs]
    τ_vals = Float64[_get_tau(p[1]) for p in valid_pairs]

    # All entries must agree on a single β interpretation; reject mixed input
    # rather than silently fit on an inconsistent scale.
    used_kinds = unique(p[3] for p in valid_pairs)
    if length(used_kinds) > 1
        throw(ArgumentError(
            "fit_scaling: entries disagree on β interpretation (kinds = $used_kinds). " *
            "Pass beta_kind=:phys or :alg to force a single scale."))
    end
    chosen_kind = first(used_kinds)
    return fit_scaling(n_vals, β_vals, τ_vals; beta_kind = chosen_kind, kwargs...)
end

"""
    predict_scaling(fit::ScalingFit, n::Real, β::Real) -> Float64

Predicted τ_mix at `(n, β)` from the fitted model. Returns the linear-scale
prediction `exp(log τ̂)`.
"""
function predict_scaling(fit::ScalingFit, n::Real, β::Real)
    c     = fit.params[_SCALING_IDX_C]
    x     = fit.params[_SCALING_IDX_X]
    slope = fit.params[_SCALING_IDX_SLOPE]
    log_n = log(Float64(n))
    log_τ_pred = if fit.model === :M0
        c + x * log_n + slope * log(Float64(β))
    elseif fit.model === :M1
        c + x * log_n + slope * Float64(β)
    else
        throw(ArgumentError("unknown model :$(fit.model) in predict_scaling"))
    end
    return exp(log_τ_pred)
end

"""
    aicc_weights(fits::Dict{Symbol, ScalingFit}) -> Dict{Symbol, Float64}

Compute normalized AICc model weights. Exact-fit (`-Inf`) winners share all
weight; unavailable (`+Inf`) candidates receive zero weight when any finite
candidate exists, and all-`+Inf` collections receive uniform weights.
"""
function aicc_weights(fits::Dict{Symbol, ScalingFit})
    isempty(fits) && return Dict{Symbol, Float64}()
    any(isnan(f.aicc) for f in values(fits)) && throw(ArgumentError(
        "AICc values must not be NaN"))

    exact_winners = [k for (k, f) in fits if f.aicc == -Inf]
    if !isempty(exact_winners)
        winner_weight = inv(Float64(length(exact_winners)))
        return Dict(k => (k in exact_winners ? winner_weight : 0.0) for k in keys(fits))
    end

    finite_keys = [k for (k, f) in fits if isfinite(f.aicc)]
    if isempty(finite_keys)
        uniform_weight = inv(Float64(length(fits)))
        return Dict(k => uniform_weight for k in keys(fits))
    end

    aicc_min = minimum(fits[k].aicc for k in finite_keys)
    raw = Dict(k => (isfinite(f.aicc) ? exp(-(f.aicc - aicc_min) / 2) : 0.0)
               for (k, f) in fits)
    Z = sum(values(raw))
    return Dict(k => v / Z for (k, v) in raw)
end

"""
    compare_models(fits::Dict{Symbol, ScalingFit}) -> NamedTuple

Rank fitted models by AICc and return a summary table.

# Returns
NamedTuple with fields:
- `ranked::Vector{Symbol}`: model symbols, best AICc first.
- `aicc::Vector{Float64}`: AICc values aligned with `ranked`.
- `delta_aicc::Vector{Float64}`: AICc differences from the best model.
- `weights::Vector{Float64}`: AICc model weights aligned with `ranked`.
"""
function compare_models(fits::Dict{Symbol, ScalingFit})
    isempty(fits) && throw(ArgumentError("fits dictionary is empty"))
    ordered = sort(collect(fits); by = kv -> (kv[2].aicc, String(kv[1])))
    ranked = [kv[1] for kv in ordered]
    aiccs  = [kv[2].aicc for kv in ordered]
    aicc_min = aiccs[1]
    delta = if aicc_min == -Inf
        [aicc == -Inf ? 0.0 : Inf for aicc in aiccs]
    elseif aicc_min == Inf
        zeros(length(aiccs))
    else
        aiccs .- aicc_min
    end
    w_dict = aicc_weights(fits)
    weights = [w_dict[k] for k in ranked]
    return (
        ranked     = ranked,
        aicc       = aiccs,
        delta_aicc = delta,
        weights    = weights,
    )
end

"""
    formula_string(fit::ScalingFit) -> String

Format the fitted scaling law with one-standard-deviation uncertainties.
"""
function formula_string(fit::ScalingFit)
    c     = fit.params[_SCALING_IDX_C]
    x     = fit.params[_SCALING_IDX_X]
    slope = fit.params[_SCALING_IDX_SLOPE]
    σc    = fit.std_errors[_SCALING_IDX_C]
    σx    = fit.std_errors[_SCALING_IDX_X]
    σs    = fit.std_errors[_SCALING_IDX_SLOPE]

    # Propagate σ(c) → σ(C) through C = exp(c): σ_C ≈ C · σ_c (first-order).
    C = exp(c)
    σC = isfinite(σc) ? C * σc : Inf

    β_label = fit.beta_kind === :phys ? "β_phys" :
              fit.beta_kind === :alg  ? "β_alg"  : "β"

    if fit.model === :M0
        return @sprintf("τ_mix = (%.3g ± %.2g) · n^(%.2f ± %.2f) · %s^(%.2f ± %.2f)",
                        C, σC, x, σx, β_label, slope, σs)
    elseif fit.model === :M1
        return @sprintf("τ_mix = (%.3g ± %.2g) · n^(%.2f ± %.2f) · exp((%.4f ± %.4f)·%s)",
                        C, σC, x, σx, slope, σs, β_label)
    else
        return "(unknown model :$(fit.model))"
    end
end

"""
    scaling_fit_grid(fit::ScalingFit; n_grid=nothing, beta_grid=nothing)
        -> NamedTuple

Evaluate a scaling fit on a regular grid for downstream plotting.

# Keywords
- `n_grid`: Explicit sizes; defaults to the observed integer range.
- `beta_grid`: Explicit temperatures; defaults to 21 log-spaced points.

# Returns
A named tuple containing the evaluation grid, predictions, observations, and
log-space residuals.
"""
function scaling_fit_grid(
    fit::ScalingFit;
    n_grid::Union{Nothing, AbstractVector} = nothing,
    beta_grid::Union{Nothing, AbstractVector} = nothing,
)
    n_lo, n_hi = extrema(fit.n_values)
    β_lo, β_hi = extrema(fit.beta_values)

    n_g = n_grid === nothing ? collect(n_lo:n_hi) : collect(Int, n_grid)
    β_g = if beta_grid === nothing
        # Log-spaced 21-point grid; if the data range is a single β, fall
        # back to a single point so the matrix below is still well-defined.
        if β_hi ≈ β_lo
            [Float64(β_lo)]
        else
            exp.(range(log(β_lo), log(β_hi); length = 21))
        end
    else
        collect(Float64, beta_grid)
    end

    τ_pred_mat = Matrix{Float64}(undef, length(n_g), length(β_g))
    @inbounds for (i, nv) in enumerate(n_g), (j, βv) in enumerate(β_g)
        τ_pred_mat[i, j] = predict_scaling(fit, nv, βv)
    end

    # Per-datapoint predictions and residuals (already on the fit).
    tau_obs = exp.(fit.log_tau_observed)
    tau_pred_at_obs = exp.(fit.log_tau_predicted)

    return (
        n_grid          = n_g,
        beta_grid       = β_g,
        tau_predicted   = τ_pred_mat,
        n_obs           = fit.n_values,
        beta_obs        = fit.beta_values,
        tau_obs         = tau_obs,
        tau_pred_at_obs = tau_pred_at_obs,
        residuals_log   = fit.residuals,
    )
end
