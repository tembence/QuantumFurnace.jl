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
_dll_same_prescription(a::AbstractFilter,b::DLLMultiChannelFilter)=
    length(b.channels)==1 && _dll_same_prescription(a,only(b.channels))
_dll_same_prescription(a::DLLMultiChannelFilter,b::AbstractFilter)=
    length(a.channels)==1 && _dll_same_prescription(only(a.channels),b)
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

"""
    ckg_to_dll(alpha, frequencies, beta; hamiltonian=nothing, jumps=nothing,
        max_rank=nothing, rank_rtol=0, roundoff_rtol=nothing,
        max_bohr_frequencies=65, max_bytes=64*1024^2)
    ckg_to_dll(kernel::PreparedCKGJointKernel; kwargs...)

Small-system finite-Bohr reference decomposition of a PSD KMS coefficient
matrix. Rows of `alpha` follow the supplied distinct, reflection-closed
`frequencies` (including zero). All inputs use the same energy frame and clock;
no rate or source normalisation is added. Returns `filter`, reconstructed
`alpha`, `frequencies`, and `evidence`. The filter contains separate tabulated
DLL channels and has no continuum/time callback.

The thermal tilt is diagonalised in a basis fixed by `Kz=P*conj(z)`, including
imaginary odd vectors. This handles complex kernels and degenerate eigenvalues.
Hermiticity, reflection and PSD defects exceeding `roundoff_rtol` are rejected.
The default is `min(128*m*eps(T),sqrt(eps(T)))`; a supplied tolerance cannot
exceed `sqrt(eps(T))`. Accepted symmetry repairs and eigenvalues removed within
that tolerance are reported separately from requested rank compression.

`max_rank` and `rank_rtol` discard eigenvalues of the **tilted** matrix. Rank
compression requires `hamiltonian` and `jumps`, so both coefficient spectral
norms and full-generator induced Hilbert–Schmidt norms are retained. Optional
generator evidence uses the supplied source amplitudes, includes the canonical
coherent correction, and is bounded before dense allocations by `max_bytes`.
It is neither a diamond norm nor an implementation/mixing guarantee. A zero
kernel uses one zero channel (`retained_rank=0`). Production CKG is unchanged.
"""
function ckg_to_dll(alpha::AbstractMatrix, frequencies, beta::Real;
    hamiltonian=nothing, jumps=nothing, max_rank::Union{Nothing,Integer}=nothing,
    rank_rtol::Real=0, roundoff_rtol::Union{Nothing,Real}=nothing,
    max_bohr_frequencies::Integer=65, max_bytes::Integer=64*1024^2)
    m=length(frequencies)
    0<m<=max_bohr_frequencies && max_bytes>0 || throw(ArgumentError(
        "Finite-Bohr reference requires 1 <= frequency count <= max_bohr_frequencies and positive max_bytes."))
    size(alpha)==(m,m) || throw(ArgumentError("alpha must have one row/column per supplied frequency."))
    T=promote_type(typeof(float(beta)),typeof(float(real(zero(eltype(alpha))))),float(eltype(frequencies)))
    T in (Float32,Float64) || throw(ArgumentError("Finite-Bohr diagonalisation supports Float32 or Float64 inputs."))
    b=T(beta)
    isfinite(b)&&b>0 && isfinite(rank_rtol)&&0<=rank_rtol<1 || throw(ArgumentError(
        "Require finite positive beta and 0 <= rank_rtol < 1."))
    max_rank===nothing || 1<=max_rank<=m || throw(ArgumentError("max_rank must lie in 1:frequency_count."))
    tol=roundoff_rtol===nothing ? min(T(128)*m*eps(T),sqrt(eps(T))) : T(roundoff_rtol)
    isfinite(tol)&&0<=tol<=sqrt(eps(T)) || throw(ArgumentError(
        "roundoff_rtol must lie between zero and sqrt(eps(T)); material defects cannot be repaired."))
    (hamiltonian===nothing)==(jumps===nothing) || throw(ArgumentError("Supply both hamiltonian and jumps for generator evidence."))
    compression=max_rank!==nothing || rank_rtol>0
    compression && hamiltonian===nothing && throw(ArgumentError(
        "Rank compression requires hamiltonian and jumps to retain the full-generator truncation norm."))
    d=hamiltonian===nothing ? 0 : length(hamiltonian.eigvals)
    # Includes eigensolver/SVD scratch, channel storage and sequential dense
    # generator differences. BigInt arithmetic prevents overflow of the guard.
    bytes=big(sizeof(Complex{T}))*(32big(m)^2+24big(d)^4+8big(m)*big(d)^2)
    bytes<=max_bytes || throw(ArgumentError("Finite-Bohr reference working-set estimate $bytes exceeds max_bytes=$max_bytes."))
    nus=T[iszero(u) ? zero(T) : T(u) for u in frequencies]
    all(isfinite,nus)&&length(unique(nus))==m&&zero(T) in nus&&all(u->-u in nus,nus) ||
        throw(ArgumentError("frequencies must be distinct, finite, reflection-closed and include zero."))
    labels=Dict(u=>i for (i,u) in enumerate(nus)); opposite=[labels[iszero(u) ? zero(T) : -u] for u in nus]
    a=Matrix{Complex{T}}(alpha)
    all(isfinite,a) || throw(ArgumentError("alpha must be finite at working precision."))
    if hamiltonian!==nothing
        hamiltonian isa HamHam && jumps isa AbstractVector{<:JumpOp} || throw(ArgumentError(
            "Generator evidence needs HamHam and a vector of compiled JumpOp sources in the same energy frame."))
        validate_jump_pairing(jumps)
        all(u->haskey(labels,T(u)),keys(hamiltonian.bohr_dict)) || throw(ArgumentError(
            "The supplied frequencies must cover the complete Hamiltonian Bohr set."))
        all(j->size(j.in_eigenbasis)==(d,d),jumps) || throw(ArgumentError("Source dimensions must match the Hamiltonian."))
    end
    exponents=(b/T(4)).*nus
    all(x->isfinite(x)&&abs(x)<log(floatmax(T))/4,exponents) || throw(ArgumentError(
        "Thermal tilt is outside the bounded reference precision; rescale inputs or reduce beta."))
    tilt=exp.(exponents)
    c=(tilt*transpose(tilt)).*a
    all(isfinite,c) || throw(ArgumentError("Thermally tilted alpha overflows; rescale the coefficient clock."))
    scale=opnorm(c)
    hermiticity=opnorm(c-c')
    reflection=opnorm(c-conj(c[opposite,opposite]))
    all(isfinite,(scale,hermiticity,reflection)) || throw(ArgumentError(
        "Tilted coefficient norm overflows working precision; rescale the coefficient clock."))
    hermiticity<=tol*scale && reflection<=tol*scale || throw(ArgumentError(
        "CKG coefficient fails finite-Bohr Hermiticity/KMS: tilted defects $hermiticity, $reflection exceed $(tol*scale)."))
    # U's columns satisfy P*conj(U)=U. Its real basis
    # represents the antiunitary-fixed subspace, also in degenerate eigenspaces.
    U=zeros(Complex{T},m,m); column=0
    for i in 1:m
        j=opposite[i]
        i>j && continue
        column+=1
        if i==j
            U[i,column]=one(T)
        else
            U[i,column]=U[j,column]=inv(sqrt(T(2)))
            column+=1
            U[i,column]=im/sqrt(T(2)); U[j,column]=-im/sqrt(T(2))
        end
    end
    minimum_raw=minimum(eigvals(Hermitian((c+c')/2)))
    minimum_raw>=-tol*scale || throw(ArgumentError(
        "Material PSD defect: minimum tilted eigenvalue $minimum_raw < $(-tol*scale); no clipping performed."))
    real_basis=real.(U'*c*U)
    E=eigen(Symmetric((real_basis+transpose(real_basis))/2))
    minimum(E.values)>=-tol*scale || throw(ArgumentError("Projected tilted kernel has a material PSD defect."))
    positive=findall(>(tol*scale),E.values)
    ordered=reverse(positive)
    retained=filter(k->E.values[k]>T(rank_rtol)*maximum(E.values),ordered)
    max_rank===nothing || resize!(retained,min(length(retained),Int(max_rank)))
    function amplitudes(indices)
        q=U*E.vectors[:,indices]*Diagonal(sqrt.(E.values[indices]))
        return q./tilt
    end
    full=amplitudes(positive); kept=amplitudes(retained)
    repaired=full*full'; reconstructed=kept*kept'
    all(isfinite,repaired)&&all(isfinite,reconstructed) || throw(ArgumentError(
        "Untilted reconstruction overflows working precision; rescale the coefficient clock."))
    channels=isempty(retained) ? (_DLLBohrFilter(b,Dict(u=>zero(Complex{T}) for u in nus)),) :
        Tuple(_DLLBohrFilter(b,Dict(u=>kept[i,k] for (i,u) in enumerate(nus))) for k in axes(kept,2))
    family=DLLMultiChannelFilter(channels,b)
    repair_norm=opnorm(repaired-a); truncation_norm=opnorm(reconstructed-repaired)
    generator_norms=hamiltonian===nothing ? nothing : (
        repair=_ckg_dll_generator_difference_norm(repaired-a,nus,b,hamiltonian,jumps),
        truncation=_ckg_dll_generator_difference_norm(reconstructed-repaired,nus,b,hamiltonian,jumps),
        total=_ckg_dll_generator_difference_norm(reconstructed-a,nus,b,hamiltonian,jumps))
    evidence=(;scope=:finite_bohr_reference,frequency_count=m,channel_count=length(channels),
        retained_rank=length(retained),numerical_rank=length(positive),
        diagonalisation=:antiunitary_fixed_real_basis,roundoff_rtol=tol,
        tilted_hermiticity_defect=hermiticity,tilted_reflection_defect=reflection,
        minimum_tilted_eigenvalue=minimum_raw,tilted_eigenvalues=copy(E.values),
        coefficient_norm=:spectral_2_norm,coefficient_repair_norm=repair_norm,
        coefficient_truncation_norm=truncation_norm,coefficient_total_error_norm=opnorm(reconstructed-a),
        generator_norm=:induced_hilbert_schmidt_full_generator,generator_error_norms=generator_norms,
        max_rank,rank_rtol=T(rank_rtol),estimated_working_bytes=bytes,
        continuum_filter=:not_defined,implementation_theorem=:not_established,
        clock=:supplied_alpha_and_source_amplitudes)
    return (;filter=family,alpha=reconstructed,frequencies=nus,evidence)
end

function ckg_to_dll(kernel::PreparedCKGJointKernel;kwargs...)
    kernel.evidence.status==:pass || throw(ArgumentError("Only a passing prepared CKG kernel may enter DLL equivalence conversion."))
    nus=sort!(collect(keys(kernel.oft.frequencies));by=u->kernel.oft.frequencies[u])
    ckg_to_dll(kernel.alpha,nus,kernel.beta;kwargs...)
end

# Extend the existing dense two-source dissipator machinery for a coefficient
# difference. This reference path never enters a production matvec.
function _ckg_dll_generator_difference_norm(delta,nus,beta,ham,jumps)
    CT=eltype(delta); d=length(ham.eigvals); labels=Dict(u=>i for (i,u) in enumerate(nus))
    L=zeros(CT,d^2,d^2); R=zeros(CT,d,d)
    ws=DenseLindbladianWorkspace(CT,d)
    for jump in jumps
        for (v,indices) in ham.bohr_dict
            right=zeros(CT,d,d)
            for index in indices
                right[index]=jump.in_eigenbasis[index]
            end
            left=CT[delta[labels[ham.eigvals[i]-ham.eigvals[j]],labels[v]]*jump.in_eigenbasis[i,j]
                for i in 1:d,j in 1:d]
            _vectorize_liouv_diss_and_add!(L,left,right',one(real(zero(CT))),ws)
            mul!(R,right',left,one(CT),one(CT))
        end
    end
    B=_dll_coherent_from_loss(R,ham.eigvals,beta)
    _vectorize_liouvillian_coherent!(L,B,ws)
    return opnorm(L)
end
