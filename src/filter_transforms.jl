# Numerical Fourier preparation. All cache entries belong to one owned callback
# snapshot and one immutable set of controls; rebuilding is cache invalidation.
struct PreparedFilterTransform{T<:AbstractFloat,F,C} <: AbstractFilter
    beta::T
    base::F
    controls::C
    cache::Dict{Tuple{Symbol,T},Tuple{Complex{T},T,Symbol}}
end
Base.eltype(::PreparedFilterTransform{T}) where {T} = Complex{T}

_transform_input(::AbstractFilter) = :frequency
_transform_input(::TimeFilter) = :time
_transform_support(::AbstractFilter) = nothing
_transform_support(f::_UserDLLFilter) = f.metadata.support
_transform_support(f::DLLMetropolisFilter) = f.S
_transform_tail(::AbstractFilter) = nothing
_transform_tail(f::_UserDLLFilter) = f.metadata.tail_bound

"""
    prepare_filter_transform(filter; window=nothing, breakpoints=[],
        rtol=1e-12, atol=1e-13, maxevals=200000, max_panels=4096,
        analytic=true)

Own a deep copy of a filter and prepare the Fourier convention
`F(ν)=∫f(t)exp(iνt)dt`, `f(t)=∫F(ν)exp(-iνt)dν/(2π)`.
A compact input uses its entire declared support. Otherwise an explicit finite
`window` radius is required for numerical integration. Unknown omitted tails
remain unknown; `tail_bound(W)` declares an L1 bound on the *input amplitude*
outside the window, and remains user-supplied evidence. Analytic Gaussian pairs
are preferred. The CKG adapter multiplies its raw frequency shape by `sqrt(π)/σ`.

QuadGK estimates are separate from tails, aliasing and interpolation. Oscillatory
integrals are split into phase-resolving panels, capped by `max_panels`; exceeding
the cap throws rather than returning an underresolved value. Prepared caches
snapshot captured mutable data via `deepcopy`; callbacks must be deterministic
and must not read mutable global/external state. Rebuild to reflect new captures.
Cache filling is serial preparation, never a source matvec operation.
"""
function prepare_filter_transform(f::AbstractFilter; window=nothing,
    breakpoints=Real[], rtol::Real=1e-12, atol::Real=1e-13,
    maxevals::Int=200000, max_panels::Int=4096, analytic::Bool=true)
    T = typeof(real(zero(eltype(f))))
    support = _transform_support(f)
    radius = support === nothing ? window : support
    builtin_pair = f isa Union{GaussianFilter,DLLGaussianFilter}
    radius === nothing && !(analytic && builtin_pair) && throw(ArgumentError(
        "Numerical transforms require declared support or an explicit finite window; omitted tails are not inferred."))
    radius === nothing || (radius isa Real && isfinite(T(radius)) && radius > 0) ||
        throw(ArgumentError("window must be a positive finite radius."))
    all(x -> isfinite(x) && x >= 0, (rtol,atol)) && max(rtol,atol)>0 ||
        throw(ArgumentError("Require nonnegative finite quadrature tolerances, at least one positive."))
    maxevals > 0 && max_panels > 0 || throw(ArgumentError("Quadrature budgets must be positive."))
    points = sort!(unique!(T.(collect(breakpoints))))
    all(isfinite,points) || throw(ArgumentError("Breakpoints must be finite."))
    radius === nothing || all(x -> abs(x)<radius,points) ||
        throw(ArgumentError("Breakpoints must lie strictly inside the integration window."))
    # Include the reflection origin even for arbitrary user specifications.
    points = sort!(unique!(vcat(points,T[0])))
    controls = (;window=radius === nothing ? nothing : T(radius), support,
        breakpoints=Tuple(points),rtol=T(rtol),atol=T(atol),maxevals,max_panels,
        analytic=analytic && builtin_pair, input=_transform_input(f))
    beta = hasproperty(f,:beta) ? T(f.beta) : one(T) # CKG adapter is not DLL admissible.
    return PreparedFilterTransform(beta,deepcopy(f),controls,
        Dict{Tuple{Symbol,T},Tuple{Complex{T},T,Symbol}}())
end

# CKG's legacy kernels are bare shapes, so adapt without changing those methods.
_transform_frequency(f::AbstractFilter,x) = freq_kernel(f,x)
_transform_frequency(f::GaussianFilter,x) = sqrt(typeof(f.sigma)(pi))/f.sigma*freq_kernel(f,x)

function _transform_value(p::PreparedFilterTransform{T}, x::Real, direction::Symbol) where {T}
    direction in (:forward,:inverse) || throw(ArgumentError("direction must be :forward or :inverse."))
    y = T(x)
    isfinite(y) || throw(ArgumentError("Transform target must be finite in working precision."))
    key = (direction,y)
    haskey(p.cache,key) && return p.cache[key]
    c = p.controls
    native = (direction == :forward && c.input == :frequency) ||
             (direction == :inverse && c.input == :time)
    if native || c.analytic
        value = direction == :forward ? _transform_frequency(p.base,y) : time_kernel(p.base,y)
        result = (Complex{T}(value),zero(T),native ? :input_evaluation : :analytic)
    else
        W = c.window
        # At most pi/2 of phase per initial panel; adaptive refinement then
        # resolves the amplitude. maxevals alone cannot detect missed oscillation.
        required = max(one(T),ceil(4W*abs(y)/T(pi)))
        required <= c.max_panels || throw(ArgumentError(
            "Oscillatory transform requires $required panels, exceeding max_panels=$(c.max_panels)."))
        edges = sort!(unique!(vcat(collect(range(-W,W;length=Int(required)+1)),collect(c.breakpoints))))
        length(edges)-1 <= c.max_panels || throw(ArgumentError("Breakpoints exceed max_panels."))
        sign = direction == :forward ? one(T) : -one(T)
        factor = direction == :forward ? one(T) : inv(2T(pi))
        input = direction == :forward ? time_kernel : _transform_frequency
        integrand = t -> Complex{T}(input(p.base,t))*cis(sign*y*t)*factor
        value,err = QuadGK.quadgk(integrand,edges...;rtol=c.rtol,atol=c.atol,maxevals=c.maxevals)
        status = err <= max(c.atol,c.rtol*abs(value)) ? :estimated : :unresolved_quadrature
        result = (Complex{T}(value),T(err),status)
    end
    all(isfinite,(result[1],result[2])) || throw(ArgumentError("Nonfinite transform value/error."))
    p.cache[key] = result
    return result
end
freq_kernel(p::PreparedFilterTransform,x::Real) = first(_transform_value(p,x,:forward))
time_kernel(p::PreparedFilterTransform,x::Real) = first(_transform_value(p,x,:inverse))

"""
    transform_values(prepared, targets; direction=:inverse)

Return owned values and per-target quadrature estimates. `status=:estimated`
means finite-window numerical evidence, never a rigorous full-transform bound.
`tail_status=:unknown` and `aliasing=:not_estimated` are not zero error bars.
"""
function transform_values(p::PreparedFilterTransform{T},targets;direction::Symbol=:inverse,
    window_refinements::Int=1) where {T}
    0 <= window_refinements <= 3 || throw(ArgumentError("window_refinements must be between 0 and 3."))
    targets = collect(targets)
    rows = [_transform_value(p,x,direction) for x in targets]
    numerical = any(r -> r[3] in (:estimated,:unresolved_quadrature),rows)
    provider = _transform_tail(p.base)
    tail = if !numerical
        nothing
    elseif p.controls.support !== nothing
        zero(T)
    elseif provider === nothing
        nothing
    else
        bound = provider(p.controls.window)
        bound isa Real && isfinite(bound) && bound >= 0 || throw(ArgumentError("tail_bound must return finite nonnegative input L1 bounds."))
        T(bound) * (direction == :inverse ? inv(2T(pi)) : one(T))
    end
    tail === nothing || isfinite(tail) || throw(ArgumentError("Tail bound overflows working precision."))
    differences = T[]
    if numerical && p.controls.support === nothing && window_refinements > 0
        previous = Complex{T}[r[1] for r in rows]
        c = p.controls
        for level in 1:window_refinements
            refined = prepare_filter_transform(p.base;window=c.window*T(2)^level,
                breakpoints=collect(c.breakpoints),rtol=c.rtol,atol=c.atol,
                maxevals=c.maxevals,max_panels=c.max_panels,analytic=false)
            values = Complex{T}[first(_transform_value(refined,x,direction)) for x in targets]
            push!(differences,maximum(abs.(values-previous);init=zero(T)))
            previous = values
        end
    end
    return (;values=Complex{T}[r[1] for r in rows],quadrature_estimates=T[r[2] for r in rows],
        status=any(r -> r[3]==:unresolved_quadrature,rows) ? :unresolved : numerical ? :estimated : :evaluated,
        methods=Symbol[r[3] for r in rows],tail_bound=tail,
        tail_status=!numerical ? :not_applicable : p.controls.support !== nothing ? :declared_support : provider === nothing ? :unknown : :user_supplied_unverified,
        window_refinement_differences=differences,window_refinement_scope=:numerical_not_tail_bound,
        aliasing=:not_estimated,interpolation=:not_used,precision=T,
        convention=:forward_plus_inverse_minus_2pi,window=p.controls.window)
end

function filter_evidence(p::PreparedFilterTransform)
    merge(filter_evidence(p.base),(;continuum_transform=:prepared_finite_window,
        transform_controls=p.controls,callback_ownership=:deepcopy_deterministic_captures,
        cached_targets=length(p.cache),implementation_theorem=:not_established))
end
