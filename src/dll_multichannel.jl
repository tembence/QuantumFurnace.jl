# Multi-channel DLL diagnostics. Production simulations normally use a
# single `DLLGaussianFilter` or `DLLMetropolisFilter`.
# Math: $alpha_(nu,nu') = sum_l f_l(nu) conj(f_l(nu'))$, so the dissipator
# must sum channel contributions without cross terms. All channels share beta.

"""
    DLLMultiChannelFilter(channels, beta)

Separate DLL channels at a common inverse temperature. Tuple/vector input is
owned as a concrete tuple; nested families are flattened in order. Repeated
channels are intentional rate additions, never merged amplitudes.
"""
struct DLLMultiChannelFilter{T<:AbstractFloat,C} <: AbstractFilter
    channels::C
    beta::T
    function DLLMultiChannelFilter(channels::Union{Tuple,AbstractVector}, beta::T) where {T<:AbstractFloat}
        flat = _flatten_local_dll_channels(Tuple(channels))
        isempty(flat) && throw(ArgumentError("DLLMultiChannelFilter requires at least one channel."))
        isfinite(beta) && beta > 0 || throw(ArgumentError("DLL beta must be finite and positive."))
        for (k,c) in enumerate(flat)
            c isa AbstractFilter && (_is_admissible_dll_filter(c) || _is_dll_bohr_spec(c)) ||
                throw(ArgumentError("Channel $k is not a supported DLL filter."))
            hasproperty(c,:beta) && isapprox(c.beta,beta;atol=0,rtol=10max(eps(T),eps(typeof(c.beta)))) ||
                throw(ArgumentError("Channel $k beta must match the family beta."))
        end
        return new{T,typeof(flat)}(flat,beta)
    end
end
# Preserve the old explicit vector constructor while resolving concrete storage.
DLLMultiChannelFilter{T,F}(channels::Vector{F},beta::T) where {T<:AbstractFloat,F<:AbstractFilter} =
    DLLMultiChannelFilter(channels,beta)

"""
    DLLSourceFilters(assignments, beta)

One filter or channel family per source, in the supplied jump order. Alternatively
pass integer-index pairs; indices must specify 1:n exactly once. Adjoint partners
must use the same prescription (reuse the same callback/prepared object). Repeated
channels within an assignment add rates; duplicate source indices are errors.
"""
struct DLLSourceFilters{T<:AbstractFloat,C<:Tuple} <: AbstractFilter
    assignments::C
    beta::T
    function DLLSourceFilters(assignments::Union{Tuple,AbstractVector},beta::T) where {T<:AbstractFloat}
        a=Tuple(assignments)
        isempty(a) && throw(ArgumentError("Provide one filter assignment per source."))
        if all(x->x isa Pair,a)
            ids=first.(a)
            all(x->x isa Integer,ids) && sort(collect(ids))==collect(1:length(a)) ||
                throw(ArgumentError("Source indices must specify 1:n exactly once; duplicate assignments are errors."))
            a=Tuple(last(a[findfirst(==(k),ids)]) for k in 1:length(a))
        end
        all(x->x isa AbstractFilter && !(x isa DLLSourceFilters),a) ||
            throw(ArgumentError("Each source assignment must be a DLL filter or channel family."))
        foreach(x->DLLMultiChannelFilter((x,),beta),a)
        return new{T,typeof(a)}(a,beta)
    end
end
Base.eltype(::DLLSourceFilters{T}) where {T}=Complex{T}
_is_dll_bohr_spec(f::DLLSourceFilters)=all(c->_is_admissible_dll_filter(c)||_is_dll_bohr_spec(c),f.assignments)
_dll_time_supported(f::DLLSourceFilters)=all(_dll_time_supported,f.assignments)
filter_evidence(f::DLLSourceFilters)=(;sources=map(filter_evidence,f.assignments),composition=:per_source_separate_dissipators)

# Compare prescriptions, never merely sampled values or callback display names.
_dll_same_prescription(a,b)=typeof(a)===typeof(b) && isequal(a,b)
function _dll_same_prescription(a::DLLMultiChannelFilter,b::DLLMultiChannelFilter)
    length(a.channels)==length(b.channels) || return false
    matched=falses(length(b.channels))
    for x in a.channels
        j=findfirst(k->!matched[k] && _dll_same_prescription(x,b.channels[k]),eachindex(b.channels))
        j===nothing && return false
        matched[j]=true
    end
    return true
end
_dll_same_prescription(a::PreparedFilterTransform,b::PreparedFilterTransform)=
    a === b || (a.controls==b.controls && _dll_same_prescription(a.base,b.base))
function _validate_dll_source_assignments(f::DLLSourceFilters,jumps)
    length(f.assignments)==length(jumps) || throw(ArgumentError("One DLL filter assignment is required per compiled source."))
    validate_jump_pairing(jumps)
    matched=falses(length(jumps)); partners=collect(eachindex(jumps))
    for k in eachindex(jumps)
        (matched[k] || jumps[k].hermitian) && continue
        j=findfirst(eachindex(jumps)) do j
            j!=k && !matched[j] && !jumps[j].hermitian &&
            _jump_matches(jumps[j].data,jumps[k].data',100eps(typeof(f.beta))) &&
            _jump_matches(jumps[j].in_eigenbasis,jumps[k].in_eigenbasis',100eps(typeof(f.beta))) &&
            _dll_same_prescription(f.assignments[k],f.assignments[j])
        end
        j===nothing && throw(ArgumentError("Source $k needs an adjoint partner with identical DLL channel prescriptions and multiplicities; reuse the same callback/prepared filter object."))
        matched[k]=matched[j]=true; partners[k]=j; partners[j]=k
    end
    return Tuple(partners)
end
_validate_dll_source_assignments(::AbstractFilter,jumps)=nothing
_dll_source_data(data,k)=hasproperty(data,:source_data) ? data.source_data[k] : data

# Multi-channel time kernels are sums of per-channel `Complex{T}` kernels;
# `Complex{T}` is the right element type regardless of which sub-filters appear.
Base.eltype(::DLLMultiChannelFilter{T}) where {T} = Complex{T}
_dll_time_supported(f::DLLMultiChannelFilter) = all(_dll_time_supported, f.channels)
filter_evidence(f::DLLMultiChannelFilter) = (;channels=Tuple(filter_evidence(c) for c in f.channels),
    composition=:separate_dissipators,implementation_theorem=:not_established)
function dll_coherent_kernel_bohr(f::DLLMultiChannelFilter, nu::Real, nup::Real)
    _require_admissible_dll_filter(f)
    return sum(c -> dll_coherent_kernel_bohr(c, nu, nup), f.channels)
end
@inline _is_admissible_dll_filter(f::DLLMultiChannelFilter) =
    isfinite(f.beta) && f.beta > zero(f.beta) && !isempty(f.channels) &&
    all(c -> _is_admissible_dll_filter(c) && hasproperty(c, :beta) &&
             isfinite(c.beta) && c.beta > 0 &&
             isapprox(c.beta, f.beta; atol=zero(f.beta), rtol=10eps(typeof(f.beta))),
        f.channels)

"""
    q_weight(filter::DLLMultiChannelFilter, nu) -> Real

Return the diagnostic sum of the channel weights.
"""
@inline function q_weight(f::DLLMultiChannelFilter{T}, nu::Real) where {T}
    s = q_weight(first(f.channels), nu)
    @inbounds for i in 2:length(f.channels)
        s += q_weight(f.channels[i], nu)
    end
    return s
end

"""
    freq_kernel(filter::DLLMultiChannelFilter, nu) -> Real

Return the diagnostic sum of the channel frequency kernels.
"""
@inline function freq_kernel(f::DLLMultiChannelFilter{T}, nu::Real) where {T}
    s = freq_kernel(first(f.channels), nu)
    @inbounds for i in 2:length(f.channels)
        s += freq_kernel(f.channels[i], nu)
    end
    return s
end

_is_dll_bohr_spec(f::DLLMultiChannelFilter) = all(c -> _is_admissible_dll_filter(c) || _is_dll_bohr_spec(c), f.channels)
function _require_admissible_dll_filter(f::DLLMultiChannelFilter; beta=nothing)
    foreach(c -> _require_admissible_dll_filter(c; beta), f.channels)
    return f
end

function _prepare_dll_bohr_filter(f::DLLMultiChannelFilter, eigvals::AbstractVector{T}; beta=f.beta) where {T<:AbstractFloat}
    # Raw specifications are validated by sampling each channel below; they are
    # not structural continuum certificates.
    return DLLMultiChannelFilter(
        map(c -> _prepare_dll_bohr_filter(c, eigvals; beta), f.channels),
        eltype(eigvals)(beta))
end

"""
    time_kernel(filter::DLLMultiChannelFilter, t) -> Complex

Return the diagnostic sum of the channel time kernels.
"""
@inline function time_kernel(f::DLLMultiChannelFilter{T}, t::Real) where {T}
    CT = Complex{T}
    s = zero(CT)
    @inbounds for c in f.channels
        s += CT(time_kernel(c, t))
    end
    return s
end

"""
    filter_time_cutoff(filter::DLLMultiChannelFilter, tol) -> conservative cutoff

Return the largest channel cutoff after dividing `tol` across channels.
"""
@inline function filter_time_cutoff(f::DLLMultiChannelFilter{T}, tol::Real) where {T}
    k = length(f.channels)
    per_tol = T(tol) / T(k)
    tc = zero(T)
    @inbounds for c in f.channels
        tcc = T(filter_time_cutoff(c, per_tol))
        tcc > tc && (tc = tcc)
    end
    return tc
end

"""
    dll_kossakowski_bohr(filter::DLLMultiChannelFilter, bohr_freqs) -> Matrix

Construct the channel-wise Kossakowski sum
`alpha = sum_l v_l * v_l^dagger`. Channels remain distinct Lindblad operators,
so no cross terms between different `l` occur.
"""
function dll_kossakowski_bohr(
    filter::DLLMultiChannelFilter,
    bohr_freqs::AbstractVector{<:Real},
)
    _require_admissible_dll_filter(filter)
    alpha = dll_kossakowski_bohr(filter.channels[1], bohr_freqs)
    @inbounds for channel_index in 2:length(filter.channels)
        alpha .+= dll_kossakowski_bohr(filter.channels[channel_index], bohr_freqs)
    end
    return alpha
end

# Symmetric frequency translates preserve the real-even DLL weight.
# Math: $q_l(nu) = sqrt(w_l/2) [q(nu-nu_l) + q(nu+nu_l)]$.

"""
    ShiftedSymmetricFilter{T<:AbstractFloat, F<:AbstractFilter}(base, shift, weight, beta)

Represent one weighted, symmetric frequency translate of a DLL filter.

# Fields
- `base`: Unshifted DLL filter.
- `shift`: Non-negative centre frequency.
- `weight`: Positive channel weight.
- `beta`: Algorithm-frame inverse temperature inherited from `base`.
"""
struct ShiftedSymmetricFilter{T<:AbstractFloat, F<:AbstractFilter} <: AbstractFilter
    base::F
    shift::T
    weight::T
    beta::T
    function ShiftedSymmetricFilter{T, F}(
        base::F,
        shift::T,
        weight::T,
        beta::T,
    ) where {T<:AbstractFloat, F<:AbstractFilter}
        base isa Union{DLLGaussianFilter, DLLMetropolisFilter} || throw(ArgumentError(
            "ShiftedSymmetricFilter base $(typeof(base)) is not an admissible DLL filter."))
        _is_admissible_dll_filter(base) || throw(ArgumentError(
            "ShiftedSymmetricFilter base $(typeof(base)) is not an admissible DLL filter."))
        isfinite(beta) && beta > zero(T) || throw(ArgumentError(
            "ShiftedSymmetricFilter beta must be finite and > 0."))
        isapprox(beta, base.beta; atol=zero(T), rtol=10eps(T)) || throw(ArgumentError(
            "ShiftedSymmetricFilter beta=$beta must match base beta=$(base.beta)."))
        isfinite(shift) || throw(ArgumentError(
            "ShiftedSymmetricFilter shift must be finite."))
        isfinite(weight) && weight > zero(T) || throw(ArgumentError(
            "ShiftedSymmetricFilter weight must be finite and > 0."))
        return new{T, F}(base, shift, weight, beta)
    end
end

function ShiftedSymmetricFilter(base::F, shift::T, weight::T) where
        {T<:AbstractFloat, F<:AbstractFilter}
    hasproperty(base, :beta) || throw(ArgumentError(
        "ShiftedSymmetricFilter base $(typeof(base)) lacks a beta field."))
    isfinite(base.beta) && base.beta > zero(base.beta) || throw(ArgumentError(
        "ShiftedSymmetricFilter base beta must be finite and > 0."))
    isfinite(shift) || throw(ArgumentError(
        "ShiftedSymmetricFilter shift must be finite."))
    isfinite(weight) && weight > zero(T) || throw(ArgumentError(
        "ShiftedSymmetricFilter weight must be finite and > 0."))
    return ShiftedSymmetricFilter{T, F}(base, shift, weight, T(base.beta))
end

Base.eltype(::ShiftedSymmetricFilter{T}) where {T} = Complex{T}
@inline _is_admissible_dll_filter(f::ShiftedSymmetricFilter) =
    f.base isa Union{DLLGaussianFilter, DLLMetropolisFilter} &&
    _is_admissible_dll_filter(f.base) &&
    isfinite(f.beta) && f.beta > zero(f.beta) &&
    isapprox(f.beta, f.base.beta; atol=zero(f.beta), rtol=10eps(typeof(f.beta))) &&
    isfinite(f.shift) && isfinite(f.weight) && f.weight > zero(f.weight)

"""
    q_weight(filter::ShiftedSymmetricFilter, ν) -> Real

Evaluate the weighted symmetric DLL channel weight.
"""
@inline function q_weight(f::ShiftedSymmetricFilter{T}, nu::Real) where {T}
    if iszero(f.shift)
        return T(sqrt(f.weight)) * T(q_weight(f.base, nu))
    end
    qm = T(q_weight(f.base, T(nu) - f.shift))
    qp = T(q_weight(f.base, T(nu) + f.shift))
    return T(sqrt(f.weight / T(2))) * (qm + qp)
end

"""
    freq_kernel(filter::ShiftedSymmetricFilter, ν) -> Real

Evaluate the shifted frequency kernel with its KMS factor.
"""
@inline function freq_kernel(f::ShiftedSymmetricFilter{T}, nu::Real) where {T}
    return q_weight(f, nu) * exp(-f.beta * T(nu) / T(4))
end

"""
    time_kernel(filter::ShiftedSymmetricFilter, t) -> Complex

Evaluate the shifted time kernel.
"""
@inline function time_kernel(f::ShiftedSymmetricFilter{T}, t::Real) where {T}
    CT = Complex{T}
    fb = CT(time_kernel(f.base, t))
    if iszero(f.shift)
        return T(sqrt(f.weight)) * fb
    end
    z = Complex{T}(f.beta * f.shift / T(4), f.shift * T(t))
    return T(sqrt(f.weight / T(2))) * fb * (T(2) * cosh(z))
end

"""
    filter_time_cutoff(filter::ShiftedSymmetricFilter, tol) -> Real

Return a cutoff that includes the shifted kernel's `cosh` envelope.
"""
@inline function filter_time_cutoff(f::ShiftedSymmetricFilter{T}, tol::Real) where {T}
    if iszero(f.shift)
        return T(filter_time_cutoff(f.base, T(tol) / T(sqrt(f.weight))))
    end
    envelope = T(sqrt(f.weight / T(2))) * T(2) * cosh(f.beta * f.shift / T(4))
    return T(filter_time_cutoff(f.base, T(tol) / envelope))
end

@inline function _shifted_frequency_window(
    f::ShiftedSymmetricFilter{T, F},
) where {T<:AbstractFloat, F<:DLLMetropolisFilter}
    radius = f.base.S + abs(f.shift)
    return (-radius, radius)
end

@inline function _shifted_frequency_window(
    f::ShiftedSymmetricFilter{T, F},
) where {T<:AbstractFloat, F<:DLLGaussianFilter}
    # The two positive Gaussian terms are centred at +/-shift - 1/beta. Their
    # exact total L1 mass is
    #   M = sqrt(w/2) exp(1/8) 2cosh(beta*shift/4) sqrt(8pi)/beta.
    # Extending the window a distance h beyond both centres leaves frequency
    # L1 tail delta <= M*erfc(beta*h/sqrt(8)) <= M*exp(-beta^2*h^2/8).
    # Since |tanh|<=1, the omitted two-frequency coherent kernel has L1 mass
    # at most M*delta. After the inverse-transform factor (2pi)^-2, the choice
    # below bounds its uniform frequency-tail error by 64eps(T).
    beta_shift = f.beta * abs(f.shift) / T(4)
    isfinite(beta_shift) || throw(ArgumentError(
        "Shifted Gaussian beta*shift is non-finite; reduce shift."))
    log_two_cosh = beta_shift + log1p(exp(-T(2) * beta_shift))
    log_l1_mass = (log(f.weight) - log(T(2))) / T(2) + T(1) / T(8) +
                  log_two_cosh + log(T(8) * T(pi)) / T(2) - log(f.beta)
    kernel_tail_tolerance = T(64) * eps(T)
    tail_exponent = max(
        T(2) * log_l1_mass - T(2) * log(T(2) * T(pi)) -
        log(kernel_tail_tolerance),
        one(T),
    )
    isfinite(tail_exponent) || throw(ArgumentError(
        "Shifted Gaussian frequency-tail budget is non-finite; " *
        "reduce shift or weight, or increase beta."))
    tail_units = sqrt(T(8) * tail_exponent)
    isfinite(tail_units) || throw(ArgumentError(
        "Shifted Gaussian frequency window is non-finite; reduce shift or weight."))
    half_width = tail_units / f.beta
    centre = -one(T) / f.beta
    shift_abs = abs(f.shift)
    nu_min = centre - shift_abs - half_width
    nu_max = centre + shift_abs + half_width
    isfinite(nu_min) && isfinite(nu_max) || throw(ArgumentError(
        "Shifted Gaussian frequency window overflowed; increase beta."))
    return (nu_min, nu_max)
end

"""
    dll_coherent_op_time(jumps, hamiltonian, time_labels,
                         filter::ShiftedSymmetricFilter, beta, tau;
                         nu_grid_size=256)

Construct the translated DLL coherent operator from the paper's two-frequency
kernel on the shifted filter's complete compact support, or on a Gaussian
window whose omitted coherent-kernel tail is bounded in `L1`.
"""
function dll_coherent_op_time(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam{T},
    time_labels::AbstractVector{<:Real},
    filter::ShiftedSymmetricFilter{T},
    beta::Real,
    tau::Real;
    nu_grid_size::Int = 256,
) where {T<:AbstractFloat}
    _require_admissible_dll_filter(filter; beta=beta)
    nu_min, nu_max = _shifted_frequency_window(filter)
    return _dll_coherent_op_time_frequency_grid(
        jumps, hamiltonian, time_labels, filter, beta, tau;
        nu_min = nu_min,
        nu_max = nu_max,
        nu_grid_size = nu_grid_size,
    )
end

"""
    dll_multichannel_translates(base::AbstractFilter;
                                 centers::AbstractVector{<:Real} = [0.0],
                                 weights::Union{Nothing, AbstractVector{<:Real}} = nothing)
        -> DLLMultiChannelFilter

Build a multi-channel DLL filter from symmetric translates of `base`.

# Arguments

- `base`: DLL filter with a `beta` field.
- `centers`: Frequency shifts; zero reproduces the base channel.
- `weights`: Positive channel weights; defaults to one per centre.

# Constraints

For a Metropolis base, each centre must satisfy `abs(center) <= S/2`.
"""
function dll_multichannel_translates(
    base::AbstractFilter;
    centers::AbstractVector{<:Real} = [0.0],
    weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    if !(base isa Union{DLLGaussianFilter, DLLMetropolisFilter})
        throw(ArgumentError("base filter $(typeof(base)) is not an admissible DLL " *
                            "filter; use DLLGaussianFilter or DLLMetropolisFilter."))
    end
    isfinite(base.beta) && base.beta > zero(base.beta) ||
        throw(ArgumentError("base filter beta must be finite and > 0."))
    if base isa DLLMetropolisFilter && (!isfinite(base.S) || base.S <= 0)
        throw(ArgumentError("DLLMetropolisFilter.S must be finite and > 0."))
    end
    if isempty(centers)
        throw(ArgumentError("centers must be non-empty."))
    end
    T = typeof(float(base.beta))
    k = length(centers)
    ws = if weights === nothing
        ones(T, k)
    else
        if length(weights) != k
            throw(ArgumentError("length(weights)=$(length(weights)) must equal " *
                                "length(centers)=$k."))
        end
        T.(weights)
    end
    if any(w -> !isfinite(w) || w <= 0, ws)
        throw(ArgumentError("weights must be finite and strictly positive."))
    end
    if any(c -> !isfinite(c), centers)
        throw(ArgumentError("centers must be finite."))
    end
    if base isa DLLMetropolisFilter
        S = base.S
        for (ℓ, c) in enumerate(centers)
            if abs(T(c)) > S / 2
                throw(ArgumentError("centers[$ℓ]=$c lies outside the bump flat-top " *
                                    "[-S/2, S/2] = [-$(S/2), $(S/2)]."))
            end
        end
    end

    channels = ShiftedSymmetricFilter{T, typeof(base)}[]
    sizehint!(channels, k)
    for ℓ in 1:k
        push!(channels, ShiftedSymmetricFilter{T, typeof(base)}(
            base, T(centers[ℓ]), ws[ℓ], T(base.beta)))
    end
    return DLLMultiChannelFilter{T, ShiftedSymmetricFilter{T, typeof(base)}}(
        channels, T(base.beta))
end

"""
    dll_lindblad_op_bohr(jump, hamiltonian, filter::DLLMultiChannelFilter)
        -> Vector{Matrix}

Construct one Bohr-domain Lindblad operator per DLL channel.

# Returns
A vector of operators. Callers must accumulate their dissipators separately;
summing the operators first would introduce cross terms.
"""
function dll_lindblad_op_bohr(
    jump::JumpOp,
    hamiltonian::HamHam{T},
    filter::DLLMultiChannelFilter,
) where {T<:AbstractFloat}
    _require_admissible_dll_filter(filter)
    return [dll_lindblad_op_bohr(jump, hamiltonian, c) for c in filter.channels]
end

"""
    dll_lindblad_op_time(jump, hamiltonian, time_labels, filter::DLLMultiChannelFilter, t0)
        -> Vector{Matrix}

Construct one time-domain Lindblad operator per DLL channel.

# Returns
A vector of operators whose dissipators must be accumulated separately.
"""
function dll_lindblad_op_time(
    jump::JumpOp,
    hamiltonian::HamHam{T},
    time_labels::AbstractVector{<:Real},
    filter::DLLMultiChannelFilter{T},
    t0::Real,
) where {T<:AbstractFloat}
    _require_admissible_dll_filter(filter)
    return [dll_lindblad_op_time(jump, hamiltonian, time_labels, c, t0)
            for c in filter.channels]
end

"""
    dll_coherent_op_bohr(jumps, hamiltonian, filter::DLLMultiChannelFilter, beta) -> Matrix

Construct the sum of the channels' Bohr-domain coherent operators.
"""
function dll_coherent_op_bohr(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam{T},
    filter::DLLMultiChannelFilter,
    beta::Real,
) where {T<:AbstractFloat}
    _require_admissible_dll_filter(filter; beta=beta)
    G = dll_coherent_op_bohr(jumps, hamiltonian, filter.channels[1], beta)
    @inbounds for ℓ in 2:length(filter.channels)
        G .+= dll_coherent_op_bohr(jumps, hamiltonian, filter.channels[ℓ], beta)
    end
    return G
end

"""
    dll_coherent_op_time(jumps, hamiltonian, time_labels,
                         filter::DLLMultiChannelFilter, beta, τ) -> Matrix

Construct the sum of the channels' time-domain coherent operators.
"""
function dll_coherent_op_time(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam{T},
    time_labels::AbstractVector{<:Real},
    filter::DLLMultiChannelFilter{T},
    beta::Real,
    τ::Real;
    kwargs...,
) where {T<:AbstractFloat}
    _require_admissible_dll_filter(filter; beta=beta)
    G = dll_coherent_op_time(jumps, hamiltonian, time_labels,
                              filter.channels[1], beta, τ; kwargs...)
    @inbounds for ℓ in 2:length(filter.channels)
        G .+= dll_coherent_op_time(jumps, hamiltonian, time_labels,
                                    filter.channels[ℓ], beta, τ; kwargs...)
    end
    return G
end

# Sum per-channel dissipators; the multi-channel Kossakowski matrix has no
# cross-channel terms.
@inline function _accumulate_dll_bohr_dissipator!(
    L_target::AbstractMatrix{<:Complex},
    jump::JumpOp,
    hamiltonian::HamHam,
    filter::DLLMultiChannelFilter,
    ws::DenseLindbladianWorkspace,
)
    Ls = dll_lindblad_op_bohr(jump, hamiltonian, filter)
    @inbounds for L_a in Ls
        _vectorize_liouv_diss_and_add!(L_target, L_a, 1.0, ws)
    end
    return L_target
end

# Multi-channel DLL filter: TimeDomain OFT-prefactor enumeration. The
# dissipator path then sums `L^(ℓ) ρ (L^(ℓ))† − …` per channel (no cross
# terms in the multi-channel α).
@inline _filter_channels_for_dll_oft(filter::DLLMultiChannelFilter) = filter.channels
