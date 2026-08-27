# Exact dense positive-parent reference for Bohr-domain DLL generators.

"""
    DenseDLLParentBlock{T}

Exact dense data for one DLL source/channel parent block.

# Fields
- `source_index`, `channel_index`: Position in the source/channel product.
- `Q`: Balance-filtered source `q(ad_H)A`.
- `L`: DLL Lindblad operator `fhat(ad_H)A`.
- `N`: Exact coherent/decay term
  `-sech(beta * ad_H / 4)(L' * L)`.
- `parent`: Positive block `-D`, in QF column-stacking order.
"""
struct DenseDLLParentBlock{T<:AbstractFloat}
    source_index::Int
    channel_index::Int
    Q::Matrix{Complex{T}}
    L::Matrix{Complex{T}}
    N::Matrix{Complex{T}}
    parent::Matrix{Complex{T}}
end

"""
    DLLIrreducibilityResult{T}

Finite-size commutant diagnostic for the actual filtered DLL operators.

# Fields
- `tolerance`: Singular-value tolerance used for the numerical nullspace.
- `commutant_dimension`: Dimension of the joint commutant of every `L_a`,
  `L_a'`, and the canonical coherent operator.
- `singular_values`: Singular values of the stacked commutator map.
- `smallest_nonzero_singular_value`: Smallest value above `tolerance`, or
  `nothing` if the commutator map vanishes.
- `identity_residual`: Residual of the normalised identity in that map.
- `is_irreducible`: Whether the numerical joint commutant is exactly scalar.
- `finite_size_only`: Always `true`; the result must not be extrapolated in
  system size.
"""
struct DLLIrreducibilityResult{T<:AbstractFloat}
    tolerance::T
    commutant_dimension::Int
    singular_values::Vector{T}
    smallest_nonzero_singular_value::Union{Nothing, T}
    identity_residual::T
    is_irreducible::Bool
    finite_size_only::Bool
end

"""
    DenseDLLParent{T,F}

Exact dense DLL parent assembled from separately retained source/channel
blocks.

# Fields
- `filter`: Original single- or multichannel filter, stored once as provenance.
- `blocks`: One block per `(source, channel)`, with no channel cross terms.
- `parent`: Sum of the retained positive blocks.
- `coherent`: Canonical complete DLL coherent operator.
- `irreducibility`: Finite-size actual-filter commutant diagnostic.
- `spectrum`: Complete kernel-aware dense parent spectrum.
"""
struct DenseDLLParent{T<:AbstractFloat, F<:AbstractFilter}
    filter::F
    blocks::Vector{DenseDLLParentBlock{T}}
    parent::Matrix{Complex{T}}
    coherent::Matrix{Complex{T}}
    irreducibility::DLLIrreducibilityResult{T}
    spectrum::KMSParentSpectrum{T}
end

@inline function _dll_parent_precision_epsilon(
    ::Type{T},
    filter::AbstractFilter,
) where {T<:AbstractFloat}
    filter_real_type = typeof(real(zero(eltype(filter))))
    filter_real_type <: AbstractFloat || throw(ArgumentError(
        "DLL filter element type must have an AbstractFloat real component."))
    return max(eps(T), T(eps(filter_real_type)))
end

@inline function _dll_parent_precision_epsilon(
    ::Type{T},
    filter::DLLMultiChannelFilter,
) where {T<:AbstractFloat}
    precision_epsilon = max(eps(T), T(eps(typeof(filter.beta))))
    for channel in filter.channels
        precision_epsilon = max(
            precision_epsilon,
            _dll_parent_precision_epsilon(T, channel),
        )
    end
    return precision_epsilon
end

@inline function _dll_parent_precision_epsilon(
    ::Type{T},
    filter::ShiftedSymmetricFilter,
) where {T<:AbstractFloat}
    return max(
        eps(T),
        T(eps(typeof(filter.beta))),
        _dll_parent_precision_epsilon(T, filter.base),
    )
end

function _validate_dense_dll_parent_filter(
    hamiltonian::HamHam{T},
    filter::AbstractFilter,
) where {T<:AbstractFloat}
    _require_admissible_dll_filter(filter)
    beta = T(getproperty(filter, :beta))
    expected_weights = _gibbs_weights(hamiltonian.eigvals, beta)
    actual_weights = _validated_diagonal_gibbs_state(hamiltonian.gibbs)
    tolerance = T(100) * T(length(expected_weights)) * eps(T)
    isapprox(actual_weights, expected_weights; atol = tolerance, rtol = tolerance) ||
        throw(ArgumentError(
            "filter beta=$(getproperty(filter, :beta)) does not match the " *
            "Gibbs state cached in hamiltonian."))
    return nothing
end

function _validate_dense_dll_parent_jump(
    jump::JumpOp,
    hamiltonian::HamHam{T},
) where {T<:AbstractFloat}
    d = length(hamiltonian.eigvals)
    size(jump.data) == (d, d) || throw(ArgumentError(
        "jump.data must have size ($d, $d), got $(size(jump.data))."))
    size(jump.in_eigenbasis) == (d, d) || throw(ArgumentError(
        "jump.in_eigenbasis must have size ($d, $d), got " *
        "$(size(jump.in_eigenbasis))."))
    jump.hermitian || throw(ArgumentError(
        "Dense DLL parent blocks currently require Hermitian source jumps."))
    tolerance = T(100) * T(d) * eps(T) * max(
        T(norm(jump.data)), T(norm(jump.in_eigenbasis)), one(T))
    isapprox(jump.data, jump.data'; atol = tolerance, rtol = tolerance) ||
        throw(ArgumentError("jump.data is not Hermitian within tolerance $tolerance."))
    isapprox(jump.in_eigenbasis, jump.in_eigenbasis';
             atol = tolerance, rtol = tolerance) || throw(ArgumentError(
        "jump.in_eigenbasis is not Hermitian within tolerance $tolerance."))
    return nothing
end

"""
    dense_dll_parent_block(jump, hamiltonian, filter;
                           source_index=1, channel_index=1)
        -> DenseDLLParentBlock

Construct one exact dense DLL parent block from a Hermitian source and one
admissible channel filter. A [`DLLMultiChannelFilter`](@ref) is rejected here:
use [`dense_dll_parent`](@ref) so its channels remain separate.
"""
function dense_dll_parent_block(
    jump::JumpOp,
    hamiltonian::HamHam{T},
    filter::F;
    source_index::Int = 1,
    channel_index::Int = 1,
) where {T<:AbstractFloat, F<:AbstractFilter}
    source_index >= 1 || throw(ArgumentError("source_index must be >= 1."))
    channel_index >= 1 || throw(ArgumentError("channel_index must be >= 1."))
    filter isa DLLMultiChannelFilter && throw(ArgumentError(
        "dense_dll_parent_block accepts one DLL channel; use dense_dll_parent " *
        "for DLLMultiChannelFilter inputs."))
    _validate_dense_dll_parent_filter(hamiltonian, filter)
    _validate_dense_dll_parent_jump(jump, hamiltonian)

    d = length(hamiltonian.eigvals)
    CT = Complex{T}
    beta = T(getproperty(filter, :beta))
    source = jump.in_eigenbasis

    Q = Matrix{CT}(undef, d, d)
    @inbounds for column in 1:d, row in 1:d
        frequency = hamiltonian.eigvals[row] - hamiltonian.eigvals[column]
        Q[row, column] = T(q_weight(filter, frequency)) * CT(source[row, column])
    end

    # Reuse the authoritative DLL filtered-jump implementation.
    L = Matrix{CT}(dll_lindblad_op_bohr(jump, hamiltonian, filter))
    rate = adjoint(L) * L
    N = Matrix{CT}(undef, d, d)
    @inbounds for column in 1:d, row in 1:d
        frequency = hamiltonian.eigvals[row] - hamiltonian.eigvals[column]
        N[row, column] = -inv(cosh(beta * frequency / T(4))) * rate[row, column]
    end

    identity_d = Matrix{CT}(I, d, d)
    discriminant = kron(conj(Q), Q)
    discriminant .+= (kron(identity_d, N) +
                      kron(transpose(N), identity_d)) / T(2)
    # The direct formula above is the signed discriminant D. Apply K=-D once.
    parent = -discriminant

    return DenseDLLParentBlock{T}(
        source_index, channel_index, Q, L, N, parent)
end

function _dense_dll_coherent_from_channels(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam{T},
    channels::AbstractVector{<:AbstractFilter},
    beta::Real,
) where {T<:AbstractFloat}
    isempty(channels) && throw(ArgumentError("channels must be nonempty."))
    coherent = zeros(Complex{T}, length(hamiltonian.eigvals),
                     length(hamiltonian.eigvals))
    for channel in channels
        coherent .+= dll_coherent_op_bohr(
            jumps, hamiltonian, channel, beta)
    end
    return coherent
end

function _dense_dll_irreducibility(
    filtered_operators::AbstractVector{<:AbstractMatrix{Complex{T}}},
    coherent::AbstractMatrix{Complex{T}};
    atol::Union{Nothing, Real} = nothing,
    rtol::Union{Nothing, Real} = nothing,
    precision_epsilon::Real = eps(T),
) where {T<:AbstractFloat}
    isempty(filtered_operators) && throw(ArgumentError(
        "filtered_operators must be nonempty."))
    d = size(coherent, 1)
    size(coherent, 2) == d || throw(ArgumentError("coherent must be square."))
    all(operator -> size(operator) == (d, d), filtered_operators) ||
        throw(ArgumentError("Every filtered operator must match coherent size ($d, $d)."))

    CT = Complex{T}
    identity_d = Matrix{CT}(I, d, d)
    d2 = d^2
    operator_count = 2 * length(filtered_operators) + 1
    commutators = Matrix{CT}(undef, operator_count * d2, d2)
    row_start = 1
    for filtered in filtered_operators
        for operator in (filtered, adjoint(filtered))
            rows = row_start:(row_start + d2 - 1)
            @views commutators[rows, :] .=
                kron(transpose(operator), identity_d) -
                kron(identity_d, operator)
            row_start += d2
        end
    end
    rows = row_start:(row_start + d2 - 1)
    @views commutators[rows, :] .=
        kron(transpose(coherent), identity_d) -
        kron(identity_d, coherent)

    singular_values = Vector{T}(svdvals(commutators))
    singular_scale = isempty(singular_values) ? zero(T) : maximum(singular_values)
    atol_value = atol === nothing ? zero(T) : T(atol)
    precision_epsilon_value = T(precision_epsilon)
    isfinite(precision_epsilon_value) && precision_epsilon_value > zero(T) ||
        throw(ArgumentError("precision_epsilon must be finite and > 0."))
    rtol_value = rtol === nothing ?
        T(max(size(commutators)...)) * precision_epsilon_value : T(rtol)
    isfinite(atol_value) && atol_value >= zero(T) || throw(ArgumentError(
        "atol must be finite and >= 0."))
    isfinite(rtol_value) && rtol_value >= zero(T) || throw(ArgumentError(
        "rtol must be finite and >= 0."))
    tolerance = max(atol_value, rtol_value * singular_scale)
    commutant_dimension = count(value -> value <= tolerance, singular_values)
    nonzero_values = filter(value -> value > tolerance, singular_values)
    smallest_nonzero = isempty(nonzero_values) ? nothing : minimum(nonzero_values)

    identity_vector = vec(identity_d) / sqrt(T(d))
    identity_residual = T(norm(commutators * identity_vector))
    identity_tolerance = max(
        tolerance, T(100) * eps(T) * max(singular_scale, one(T)))
    is_irreducible = commutant_dimension == 1 &&
                     identity_residual <= identity_tolerance

    return DLLIrreducibilityResult{T}(
        tolerance,
        commutant_dimension,
        singular_values,
        smallest_nonzero,
        identity_residual,
        is_irreducible,
        true,
    )
end

"""
    dense_dll_irreducibility(jumps, hamiltonian, filter;
                              atol=nothing, rtol=nothing)
        -> DLLIrreducibilityResult

Compute the joint commutant of the actual filtered `L_a`, their adjoints, and
the complete canonical DLL coherent operator. A scalar commutant is a
finite-chain irreducibility certificate in this exact, faithful-Gibbs setting;
the returned result makes no claim for other chain lengths.
"""
function dense_dll_irreducibility(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam{T},
    filter::AbstractFilter;
    atol::Union{Nothing, Real} = nothing,
    rtol::Union{Nothing, Real} = nothing,
) where {T<:AbstractFloat}
    isempty(jumps) && throw(ArgumentError("jumps must be nonempty."))
    _validate_dense_dll_parent_filter(hamiltonian, filter)
    channels = collect(_filter_channels_for_dll_oft(filter))
    filtered_operators = Matrix{Complex{T}}[]
    sizehint!(filtered_operators, length(jumps) * length(channels))
    for jump in jumps
        _validate_dense_dll_parent_jump(jump, hamiltonian)
        for channel in channels
            push!(filtered_operators,
                  Matrix{Complex{T}}(dll_lindblad_op_bohr(
                      jump, hamiltonian, channel)))
        end
    end
    coherent = _dense_dll_coherent_from_channels(
        jumps, hamiltonian, channels, getproperty(filter, :beta))
    return _dense_dll_irreducibility(
        filtered_operators,
        coherent;
        atol = atol,
        rtol = rtol,
        precision_epsilon = _dll_parent_precision_epsilon(T, filter),
    )
end

"""
    dense_dll_parent(jumps, hamiltonian, filter;
                     kernel_tolerance=nothing,
                     hermiticity_tolerance=nothing,
                     commutant_atol=nothing,
                     commutant_rtol=nothing) -> DenseDLLParent

Construct the complete exact dense DLL positive parent. Every source/channel
block is formed and retained separately before summation, so multichannel
filters cannot introduce nonexistent cross terms. The coherent correction is
always the complete canonical DLL term.
"""
function dense_dll_parent(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam{T},
    filter::AbstractFilter;
    kernel_tolerance::Union{Nothing, Real} = nothing,
    hermiticity_tolerance::Union{Nothing, Real} = nothing,
    commutant_atol::Union{Nothing, Real} = nothing,
    commutant_rtol::Union{Nothing, Real} = nothing,
) where {T<:AbstractFloat}
    isempty(jumps) && throw(ArgumentError("jumps must be nonempty."))
    _validate_dense_dll_parent_filter(hamiltonian, filter)
    channels = collect(_filter_channels_for_dll_oft(filter))
    blocks = DenseDLLParentBlock{T}[]
    sizehint!(blocks, length(jumps) * length(channels))
    total_parent = zeros(Complex{T}, length(hamiltonian.eigvals)^2,
                         length(hamiltonian.eigvals)^2)

    for (source_index, jump) in pairs(jumps)
        for (channel_index, channel) in pairs(channels)
            block = dense_dll_parent_block(
                jump,
                hamiltonian,
                channel;
                source_index = source_index,
                channel_index = channel_index,
            )
            push!(blocks, block)
            total_parent .+= block.parent
        end
    end

    coherent = _dense_dll_coherent_from_channels(
        jumps, hamiltonian, channels, getproperty(filter, :beta))
    precision_epsilon = _dll_parent_precision_epsilon(T, filter)
    irreducibility = _dense_dll_irreducibility(
        [block.L for block in blocks],
        coherent;
        atol = commutant_atol,
        rtol = commutant_rtol,
        precision_epsilon = precision_epsilon,
    )
    parent_dimension = size(total_parent, 1)
    resolved_kernel_tolerance = if kernel_tolerance === nothing
        T(100) * T(parent_dimension) * precision_epsilon *
        max(T(opnorm(total_parent)), one(T))
    else
        kernel_tolerance
    end
    resolved_hermiticity_tolerance = if hermiticity_tolerance === nothing
        T(100) * T(parent_dimension) * precision_epsilon
    else
        hermiticity_tolerance
    end
    spectrum = kms_parent_spectrum(
        total_parent,
        hamiltonian.gibbs;
        kernel_tolerance = resolved_kernel_tolerance,
        hermiticity_tolerance = resolved_hermiticity_tolerance,
        irreducibility_established = irreducibility.is_irreducible,
    )

    return DenseDLLParent{T, typeof(filter)}(
        filter, blocks, total_parent, coherent, irreducibility, spectrum)
end
