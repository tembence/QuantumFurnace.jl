const _LOCAL_1D_BOUNDARIES = (:open, :periodic)
const _LOCAL_1D_FRAMES = (:physical, :algorithm)
const _LOCAL_1D_SCALE_PROVENANCE = (
    :exact_dense,
    :certified_local_interval_bound,
)
const _LOCAL_1D_SPECTRAL_BOUND_PROVENANCE =
    :weyl_gershgorin_local_term_ranges

@inline function _validate_local_1d_boundary(boundary::Symbol)
    boundary in _LOCAL_1D_BOUNDARIES || throw(ArgumentError(
        "boundary must be :open or :periodic, got $boundary."))
    return boundary
end

function _validate_local_1d_support(
    sites::AbstractVector{<:Integer},
    num_sites::Int,
)
    num_sites > 0 || throw(ArgumentError("num_sites must be > 0."))
    isempty(sites) && throw(ArgumentError("local support must be nonempty."))
    length(sites) <= num_sites || throw(ArgumentError(
        "local support has $(length(sites)) sites but the chain has only $num_sites."))
    all(site -> 1 <= site <= num_sites, sites) || throw(ArgumentError(
        "local support sites must lie in 1:$num_sites, got $(collect(sites))."))
    allunique(sites) || throw(ArgumentError(
        "local support sites must be distinct, got $(collect(sites))."))
    return nothing
end

@inline function _is_forward_contiguous_1d(sites::AbstractVector{<:Integer})
    return all(index -> sites[index + 1] == sites[index] + 1,
               1:(length(sites) - 1))
end

@inline function _is_cyclic_contiguous_1d(
    sites::AbstractVector{<:Integer},
    num_sites::Int,
)
    return all(index -> sites[index + 1] == mod1(sites[index] + 1, num_sites),
               1:(length(sites) - 1))
end

@inline function _wraps_boundary_1d(
    sites::AbstractVector{<:Integer},
    num_sites::Int,
)
    return length(sites) > 1 && _is_cyclic_contiguous_1d(sites, num_sites) &&
           any(index -> sites[index + 1] < sites[index], 1:(length(sites) - 1))
end

function _validate_contiguous_support_1d(
    sites::AbstractVector{<:Integer},
    num_sites::Int,
    boundary::Symbol,
)
    contiguous = boundary == :open ?
        _is_forward_contiguous_1d(sites) :
        _is_cyclic_contiguous_1d(sites, num_sites)
    contiguous || throw(ArgumentError(
        "support $(collect(sites)) is not contiguous for boundary=$boundary."))
    return nothing
end

@inline function _local_float_type(matrix::AbstractMatrix, coefficient::Number)
    matrix_real_type = typeof(float(real(zero(eltype(matrix)))))
    coefficient_real_type = typeof(float(real(coefficient)))
    T = promote_type(matrix_real_type, coefficient_real_type)
    T <: AbstractFloat || throw(ArgumentError(
        "local numeric data must promote to an AbstractFloat type, got $T."))
    return T
end

function _validate_local_matrix_1d(
    matrix::AbstractMatrix{<:Number},
    support_size::Int,
    local_dim::Int;
    require_hermitian::Bool,
)
    local_dim >= 2 || throw(ArgumentError("local_dim must be >= 2."))
    expected_dimension = local_dim^support_size
    size(matrix) == (expected_dimension, expected_dimension) || throw(ArgumentError(
        "local matrix on $support_size sites of dimension $local_dim must have size " *
        "($expected_dimension, $expected_dimension), got $(size(matrix))."))
    all(isfinite, matrix) || throw(ArgumentError(
        "local matrix must contain only finite values."))
    require_hermitian && !ishermitian(matrix) && throw(ArgumentError(
        "local Hamiltonian matrices must be Hermitian."))
    return nothing
end

"""
    LocalTerm1D

A finite-support Hamiltonian term on an ordered one-dimensional chain. The
small `matrix` acts on `sites` in the listed tensor order; `coefficient` is
stored separately. Periodic wrap support is explicit in `wraps_boundary`.
"""
struct LocalTerm1D{T<:AbstractFloat}
    _sites::Vector{Int}
    _matrix::Matrix{Complex{T}}
    coefficient::T
    local_dim::Int
    num_sites::Int
    boundary::Symbol
    wraps_boundary::Bool

    function LocalTerm1D{T}(
        sites::Vector{Int},
        matrix::Matrix{Complex{T}},
        coefficient::T,
        local_dim::Int,
        num_sites::Int,
        boundary::Symbol,
    ) where {T<:AbstractFloat}
        _validate_local_1d_boundary(boundary)
        _validate_local_1d_support(sites, num_sites)
        _validate_contiguous_support_1d(sites, num_sites, boundary)
        _validate_local_matrix_1d(
            matrix, length(sites), local_dim; require_hermitian=true)
        isfinite(coefficient) || throw(ArgumentError(
            "local Hamiltonian coefficient must be finite."))
        wraps_boundary = _wraps_boundary_1d(sites, num_sites)
        boundary == :open && wraps_boundary && throw(ArgumentError(
            "an open-boundary term cannot wrap around the chain."))
        return new{T}(
            copy(sites), copy(matrix), coefficient, local_dim, num_sites,
            boundary, wraps_boundary)
    end
end

@inline _local_sites(term::LocalTerm1D) = getfield(term, :_sites)
@inline _local_matrix(term::LocalTerm1D) = getfield(term, :_matrix)

function Base.getproperty(term::LocalTerm1D, name::Symbol)
    name === :sites && return copy(_local_sites(term))
    name === :matrix && return copy(_local_matrix(term))
    return getfield(term, name)
end

function Base.propertynames(::LocalTerm1D, private::Bool=false)
    public_names = (
        :sites, :matrix, :coefficient, :local_dim, :num_sites, :boundary,
        :wraps_boundary,
    )
    return private ? (public_names..., :_sites, :_matrix) : public_names
end

function LocalTerm1D(
    sites::AbstractVector{<:Integer},
    matrix::AbstractMatrix{<:Number},
    coefficient::Real,
    num_sites::Integer;
    local_dim::Integer=2,
    boundary::Symbol=:open,
)
    T = _local_float_type(matrix, coefficient)
    return LocalTerm1D{T}(
        Int.(sites), Matrix{Complex{T}}(matrix), T(coefficient), Int(local_dim),
        Int(num_sites), boundary)
end

"""
    LocalHamiltonian1D

Backend-neutral local Hamiltonian data in either physical or algorithm
coordinates. `spectral_lower_bound` and `spectral_upper_bound` are the Weyl
enclosure obtained by summing outward-rounded Gershgorin intervals for the
small terms. Public access to nested terms and matrices returns defensive
copies so that the cached enclosure cannot be invalidated accidentally.
"""
struct LocalHamiltonian1D{T<:AbstractFloat}
    _terms::Vector{LocalTerm1D{T}}
    num_sites::Int
    local_dim::Int
    boundary::Symbol
    coordinate_frame::Symbol
    global_shift::T
    rescaling_factor::T
    scale_provenance::Symbol
    spectral_lower_bound::T
    spectral_upper_bound::T
    spectral_width_bound::T
    spectral_bound_provenance::Symbol

    function LocalHamiltonian1D{T}(
        terms::Vector{LocalTerm1D{T}},
        num_sites::Int,
        local_dim::Int,
        boundary::Symbol,
        coordinate_frame::Symbol,
        global_shift::T,
        rescaling_factor::T,
        scale_provenance::Symbol,
        spectral_lower_bound::T,
        spectral_upper_bound::T,
        spectral_bound_provenance::Symbol,
    ) where {T<:AbstractFloat}
        _validate_local_1d_boundary(boundary)
        coordinate_frame in _LOCAL_1D_FRAMES || throw(ArgumentError(
            "coordinate_frame must be :physical or :algorithm, got $coordinate_frame."))
        num_sites > 0 || throw(ArgumentError("num_sites must be > 0."))
        local_dim >= 2 || throw(ArgumentError("local_dim must be >= 2."))
        owned_terms = LocalTerm1D{T}[
            LocalTerm1D{T}(
                _local_sites(term), _local_matrix(term), term.coefficient,
                term.local_dim, term.num_sites, term.boundary,
            )
            for term in terms
        ]
        all(term -> term.num_sites == num_sites &&
                    term.local_dim == local_dim &&
                    term.boundary == boundary,
            owned_terms) || throw(ArgumentError(
            "all local terms must match the Hamiltonian chain length, local dimension, and boundary."))
        isfinite(global_shift) || throw(ArgumentError("global_shift must be finite."))
        isfinite(rescaling_factor) && rescaling_factor > zero(T) || throw(ArgumentError(
            "rescaling_factor must be finite and > 0."))
        if coordinate_frame == :physical
            rescaling_factor == one(T) || throw(ArgumentError(
                "physical-frame local data must use the identity rescaling_factor=1."))
            scale_provenance == :identity_physical_frame || throw(ArgumentError(
                "physical-frame local data must use scale_provenance=:identity_physical_frame."))
        else
            scale_provenance in _LOCAL_1D_SCALE_PROVENANCE || throw(ArgumentError(
                "algorithm-frame rescaling requires scale_provenance=:exact_dense or " *
                ":certified_local_interval_bound."))
        end
        isfinite(spectral_lower_bound) && isfinite(spectral_upper_bound) ||
            throw(ArgumentError("spectral bounds must be finite."))
        spectral_upper_bound >= spectral_lower_bound || throw(ArgumentError(
            "spectral_upper_bound must be >= spectral_lower_bound."))
        spectral_bound_provenance == _LOCAL_1D_SPECTRAL_BOUND_PROVENANCE ||
            throw(ArgumentError(
                "spectral_bound_provenance must be " *
                ":weyl_gershgorin_local_term_ranges."))
        owned_lower, owned_upper =
            _weyl_spectral_interval_1d(owned_terms, global_shift)
        owned_lower == spectral_lower_bound && owned_upper == spectral_upper_bound ||
            throw(ArgumentError(
                "supplied spectral bounds do not match the local Hamiltonian data."))
        if coordinate_frame == :algorithm &&
                scale_provenance == :certified_local_interval_bound
            _validate_certified_algorithm_interval_1d(owned_lower, owned_upper)
        end
        spectral_width_bound = if owned_upper == owned_lower
            zero(T)
        else
            nextfloat(owned_upper - owned_lower)
        end
        return new{T}(
            owned_terms, num_sites, local_dim, boundary, coordinate_frame,
            global_shift, rescaling_factor, scale_provenance,
            owned_lower, owned_upper, spectral_width_bound,
            spectral_bound_provenance)
    end
end


@inline _local_terms(hamiltonian::LocalHamiltonian1D) = getfield(hamiltonian, :_terms)

function Base.getproperty(hamiltonian::LocalHamiltonian1D, name::Symbol)
    name === :terms && return copy(_local_terms(hamiltonian))
    return getfield(hamiltonian, name)
end

function Base.propertynames(::LocalHamiltonian1D, private::Bool=false)
    public_names = (
        :terms, :num_sites, :local_dim, :boundary, :coordinate_frame,
        :global_shift, :rescaling_factor, :scale_provenance,
        :spectral_lower_bound, :spectral_upper_bound, :spectral_width_bound,
        :spectral_bound_provenance,
    )
    return private ? (public_names..., :_terms) : public_names
end

function _weyl_spectral_interval_1d(
    terms::AbstractVector{LocalTerm1D{T}},
    global_shift::T,
) where {T<:AbstractFloat}
    lower = global_shift
    upper = global_shift
    @inbounds for term in terms
        local_matrix = term.coefficient .* _local_matrix(term)
        local_lower = T(Inf)
        local_upper = T(-Inf)
        for row in axes(local_matrix, 1)
            radius = zero(T)
            for column in axes(local_matrix, 2)
                row == column && continue
                magnitude = T(abs(local_matrix[row, column]))
                iszero(magnitude) && continue
                radius = nextfloat(radius + nextfloat(magnitude))
            end
            center = real(local_matrix[row, row])
            local_lower = min(local_lower, prevfloat(center - radius))
            local_upper = max(local_upper, nextfloat(center + radius))
        end
        lower = prevfloat(lower + local_lower)
        upper = nextfloat(upper + local_upper)
    end
    return lower, upper
end

function _validate_certified_algorithm_interval_1d(
    lower::T,
    upper::T,
) where {T<:AbstractFloat}
    target_lower = zero(T)
    target_upper = T(9) / T(20)
    tolerance = T(64) * eps(T) *
                max(one(T), abs(lower), abs(upper), target_upper)
    lower >= target_lower - tolerance || throw(ArgumentError(
        "scale_provenance=:certified_local_interval_bound requires the " *
        "algorithm-frame spectral enclosure to lie in [0, 0.45], but its " *
        "lower endpoint is $lower."))
    upper <= target_upper + tolerance || throw(ArgumentError(
        "scale_provenance=:certified_local_interval_bound requires the " *
        "algorithm-frame spectral enclosure to lie in [0, 0.45], but its " *
        "upper endpoint is $upper."))
    return nothing
end


function _validate_local_hamiltonian_integrity(
    hamiltonian::LocalHamiltonian1D{T},
) where {T<:AbstractFloat}
    terms = _local_terms(hamiltonian)
    for term in terms
        sites = _local_sites(term)
        matrix = _local_matrix(term)
        _validate_local_1d_support(sites, term.num_sites)
        _validate_contiguous_support_1d(sites, term.num_sites, term.boundary)
        _validate_local_matrix_1d(
            matrix, length(sites), term.local_dim; require_hermitian=true)
        isfinite(term.coefficient) || throw(ArgumentError(
            "local Hamiltonian coefficient must be finite."))
        term.num_sites == hamiltonian.num_sites &&
            term.local_dim == hamiltonian.local_dim &&
            term.boundary == hamiltonian.boundary || throw(ArgumentError(
            "local Hamiltonian term metadata no longer matches its parent."))
        _wraps_boundary_1d(sites, term.num_sites) == term.wraps_boundary ||
            throw(ArgumentError("local Hamiltonian wrap metadata is inconsistent."))
    end
    lower, upper = _weyl_spectral_interval_1d(terms, hamiltonian.global_shift)
    lower == hamiltonian.spectral_lower_bound &&
        upper == hamiltonian.spectral_upper_bound || throw(ArgumentError(
        "local Hamiltonian data no longer match the cached spectral enclosure."))
    if hamiltonian.coordinate_frame == :algorithm &&
            hamiltonian.scale_provenance == :certified_local_interval_bound
        _validate_certified_algorithm_interval_1d(lower, upper)
    end
    return nothing
end

function LocalHamiltonian1D(
    terms::Vector{LocalTerm1D{T}};
    num_sites::Integer,
    local_dim::Integer=2,
    boundary::Symbol=:open,
    coordinate_frame::Symbol=:physical,
    global_shift::Real=zero(T),
    rescaling_factor::Real=one(T),
    scale_provenance::Symbol=:identity_physical_frame,
) where {T<:AbstractFloat}
    shift_T = T(global_shift)
    lower, upper = _weyl_spectral_interval_1d(terms, shift_T)
    provenance = _LOCAL_1D_SPECTRAL_BOUND_PROVENANCE
    return LocalHamiltonian1D{T}(
        terms, Int(num_sites), Int(local_dim), boundary, coordinate_frame,
        shift_T, T(rescaling_factor), scale_provenance, lower, upper,
        provenance)
end

"""
    LocalJump1D

A small jump matrix on ordered support, with its scalar amplitude stored once.
Unlike Hamiltonian terms, the representation permits noncontiguous or
non-Hermitian data so that construction-specific validators can reject it.
"""
struct LocalJump1D{T<:AbstractFloat}
    _sites::Vector{Int}
    _matrix::Matrix{Complex{T}}
    coefficient::Complex{T}
    local_dim::Int
    num_sites::Int
    boundary::Symbol

    function LocalJump1D{T}(
        sites::Vector{Int},
        matrix::Matrix{Complex{T}},
        coefficient::Complex{T},
        local_dim::Int,
        num_sites::Int,
        boundary::Symbol,
    ) where {T<:AbstractFloat}
        _validate_local_1d_boundary(boundary)
        _validate_local_1d_support(sites, num_sites)
        _validate_local_matrix_1d(
            matrix, length(sites), local_dim; require_hermitian=false)
        isfinite(coefficient) || throw(ArgumentError(
            "local jump coefficient must be finite."))
        return new{T}(
            copy(sites), copy(matrix), coefficient, local_dim, num_sites,
            boundary)
    end
end


@inline _local_sites(jump::LocalJump1D) = getfield(jump, :_sites)
@inline _local_matrix(jump::LocalJump1D) = getfield(jump, :_matrix)

function Base.getproperty(jump::LocalJump1D, name::Symbol)
    name === :sites && return copy(_local_sites(jump))
    name === :matrix && return copy(_local_matrix(jump))
    return getfield(jump, name)
end

function Base.propertynames(::LocalJump1D, private::Bool=false)
    public_names = (
        :sites, :matrix, :coefficient, :local_dim, :num_sites, :boundary,
    )
    return private ? (public_names..., :_sites, :_matrix) : public_names
end

function _validate_local_jump_integrity(jump::LocalJump1D)
    sites = _local_sites(jump)
    matrix = _local_matrix(jump)
    _validate_local_1d_support(sites, jump.num_sites)
    _validate_local_matrix_1d(
        matrix, length(sites), jump.local_dim; require_hermitian=false)
    isfinite(jump.coefficient) || throw(ArgumentError(
        "local jump coefficient must be finite."))
    return nothing
end

function LocalJump1D(
    sites::AbstractVector{<:Integer},
    matrix::AbstractMatrix{<:Number},
    coefficient::Number,
    num_sites::Integer;
    local_dim::Integer=2,
    boundary::Symbol=:open,
)
    T = _local_float_type(matrix, coefficient)
    return LocalJump1D{T}(
        Int.(sites), Matrix{Complex{T}}(matrix), Complex{T}(coefficient),
        Int(local_dim), Int(num_sites), boundary)
end

@inline function _local_jump_is_hermitian(jump::LocalJump1D)
    return ishermitian(jump.coefficient .* _local_matrix(jump))
end

function _copy_local_jump(jump::LocalJump1D{T}) where {T<:AbstractFloat}
    return LocalJump1D{T}(
        _local_sites(jump), _local_matrix(jump), jump.coefficient,
        jump.local_dim, jump.num_sites, jump.boundary,
    )
end

_flatten_local_dll_channel(channel::DLLMultiChannelFilter) =
    _flatten_local_dll_channels(Tuple(channel.channels))
_flatten_local_dll_channel(channel) = (channel,)
_flatten_local_dll_channels(::Tuple{}) = ()
function _flatten_local_dll_channels(channels::Tuple)
    return (
        _flatten_local_dll_channel(first(channels))...,
        _flatten_local_dll_channels(Base.tail(channels))...,
    )
end

function _validate_local_dll_channels(channels::Tuple)
    isempty(channels) && throw(ArgumentError(
        "LocalDLLBlock1D requires at least one DLL channel."))
    beta_reference = nothing
    for (index, channel) in pairs(channels)
        channel isa AbstractFilter || throw(ArgumentError(
            "LocalDLLBlock1D channel $index is not an AbstractFilter."))
        channel isa DLLMultiChannelFilter && throw(ArgumentError(
            "LocalDLLBlock1D channels must be atomic; expand nested " *
            "DLLMultiChannelFilter values first."))
        _require_admissible_dll_filter(channel)
        if beta_reference === nothing
            beta_reference = channel.beta
        else
            R = promote_type(typeof(float(beta_reference)), typeof(float(channel.beta)))
            isapprox(R(beta_reference), R(channel.beta);
                     atol=zero(R), rtol=R(10) * eps(R)) || throw(ArgumentError(
                "all LocalDLLBlock1D channels must use the same beta."))
        end
        any(previous -> isequal(previous, channel), channels[1:(index - 1)]) &&
            throw(ArgumentError(
                "LocalDLLBlock1D channels must be distinct; channel $index is duplicated."))
    end
    return nothing
end

"""
    LocalDLLBlock1D(source, channels)

One Hermitian local source and one or more distinct admissible DLL channels.
Each channel receives its own parent block and matching coherent correction.
The current MVP accepts only contiguous Hermitian sources.
"""
struct LocalDLLBlock1D{T<:AbstractFloat, C<:Tuple}
    source::LocalJump1D{T}
    channels::C

    function LocalDLLBlock1D{T, C}(
        source::LocalJump1D{T},
        channels::C,
    ) where {T<:AbstractFloat, C<:Tuple}
        isconcretetype(C) || throw(ArgumentError(
            "LocalDLLBlock1D channels must have a concrete tuple type."))
        _validate_local_jump_integrity(source)
        _local_jump_is_hermitian(source) || throw(ArgumentError(
            "LocalDLLBlock1D currently requires a Hermitian source."))
        _validate_contiguous_support_1d(
            _local_sites(source), source.num_sites, source.boundary)
        _validate_local_dll_channels(channels)
        return new{T, C}(_copy_local_jump(source), channels)
    end
end

function LocalDLLBlock1D(source::LocalJump1D{T}, channels::C) where
        {T<:AbstractFloat, C<:Tuple}
    atomic_channels = _flatten_local_dll_channels(channels)
    return LocalDLLBlock1D{T, typeof(atomic_channels)}(source, atomic_channels)
end

LocalDLLBlock1D(source::LocalJump1D, channel::AbstractFilter) =
    LocalDLLBlock1D(source, (channel,))

LocalDLLBlock1D(source::LocalJump1D, filter::DLLMultiChannelFilter) =
    LocalDLLBlock1D(source, (filter,))

function _validate_local_dll_block_integrity(block::LocalDLLBlock1D)
    source = getfield(block, :source)
    _validate_local_jump_integrity(source)
    _local_jump_is_hermitian(source) || throw(ArgumentError(
        "LocalDLLBlock1D currently requires a Hermitian source."))
    _validate_contiguous_support_1d(
        _local_sites(source), source.num_sites, source.boundary)
    _validate_local_dll_channels(getfield(block, :channels))
    return nothing
end

function _checked_local_dimension(local_dim::Int, num_sites::Int)
    dimension = 1
    for _ in 1:num_sites
        dimension = Base.checked_mul(dimension, local_dim)
    end
    return dimension
end

function _embed_local_matrix_1d(
    sites::AbstractVector{<:Integer},
    matrix::AbstractMatrix{Complex{T}},
    num_sites::Int,
    local_dim::Int,
) where {T<:AbstractFloat}
    dimension = _checked_local_dimension(local_dim, num_sites)
    support_size = length(sites)
    support_dimension = _checked_local_dimension(local_dim, support_size)
    global_strides = [local_dim^(num_sites - site) for site in sites]
    local_strides = [local_dim^(support_size - index) for index in 1:support_size]

    rows = Int[]
    cols = Int[]
    values = Complex{T}[]
    sizehint!(rows, dimension * support_dimension)
    sizehint!(cols, dimension * support_dimension)
    sizehint!(values, dimension * support_dimension)

    @inbounds for column_zero in 0:(dimension - 1)
        local_column_zero = 0
        base_row_zero = column_zero
        for index in eachindex(sites)
            digit = (column_zero ÷ global_strides[index]) % local_dim
            local_column_zero += digit * local_strides[index]
            base_row_zero -= digit * global_strides[index]
        end
        for local_row_zero in 0:(support_dimension - 1)
            value = matrix[local_row_zero + 1, local_column_zero + 1]
            iszero(value) && continue
            row_zero = base_row_zero
            for index in eachindex(sites)
                digit = (local_row_zero ÷ local_strides[index]) % local_dim
                row_zero += digit * global_strides[index]
            end
            push!(rows, row_zero + 1)
            push!(cols, column_zero + 1)
            push!(values, value)
        end
    end
    return sparse(rows, cols, values, dimension, dimension)
end

function _materialize_local_terms_1d(
    terms::AbstractVector{LocalTerm1D{T}},
    num_sites::Int,
    local_dim::Int,
) where {T<:AbstractFloat}
    dimension = _checked_local_dimension(local_dim, num_sites)
    hamiltonian = spzeros(Complex{T}, dimension, dimension)
    @inbounds for term in terms
        hamiltonian += term.coefficient .* _embed_local_matrix_1d(
            _local_sites(term), _local_matrix(term), num_sites, local_dim)
    end
    return Hermitian(Matrix(hamiltonian))
end

"""Materialise a local Hamiltonian for dense reference sizes."""
function materialize_local_hamiltonian(hamiltonian::LocalHamiltonian1D{T}) where
        {T<:AbstractFloat}
    _validate_local_hamiltonian_integrity(hamiltonian)
    dense = Matrix(_materialize_local_terms_1d(
        _local_terms(hamiltonian), hamiltonian.num_sites, hamiltonian.local_dim))
    if !iszero(hamiltonian.global_shift)
        @inbounds for index in axes(dense, 1)
            dense[index, index] += hamiltonian.global_shift
        end
    end
    return Hermitian(dense)
end

"""Materialise a backend-neutral local jump for the legacy dense path."""
function materialize_local_jump(jump::LocalJump1D{T}) where {T<:AbstractFloat}
    _validate_local_jump_integrity(jump)
    return jump.coefficient .* _embed_local_matrix_1d(
        _local_sites(jump), _local_matrix(jump), jump.num_sites, jump.local_dim)
end

function _local_matrix_from_factors(
    factors::AbstractVector{<:AbstractMatrix{<:Number}},
)
    matrix = Matrix(first(factors))
    @inbounds for index in 2:length(factors)
        matrix = kron(matrix, factors[index])
    end
    return matrix
end


function _build_local_heis_1d_specification(
    num_qubits::Int,
    coeffs::Vector{Float64};
    seed::Int,
    periodic::Bool=true,
    disordering_terms::Vector{Vector{Matrix{ComplexF64}}}=
        Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
    disorder_strength::Float64=0.1,
)
    num_qubits >= 2 || throw(ArgumentError(
        "build_local_heis_1d requires at least two qubits, got $num_qubits."))
    length(coeffs) == 3 || throw(ArgumentError(
        "build_local_heis_1d requires exactly [J_x, J_y, J_z], got " *
        "$(length(coeffs)) coefficients."))
    all(isfinite, coeffs) || throw(ArgumentError("Heisenberg couplings must be finite."))
    isfinite(disorder_strength) && disorder_strength >= 0 || throw(ArgumentError(
        "disorder_strength must be finite and >= 0."))
    _validate_local_terms(disordering_terms, num_qubits;
        max_support=2, require_involution=true)

    boundary = periodic ? :periodic : :open
    base_patterns = Vector{Matrix{ComplexF64}}[[X, X], [Y, Y], [Z, Z]]
    rng = MersenneTwister(seed)
    sample_coeffs = [zeros(Float64, num_qubits) for _ in disordering_terms]
    for disorder_coefficients in sample_coeffs
        rand!(rng, disorder_coefficients)
        disorder_coefficients .*= disorder_strength
    end

    terms = LocalTerm1D{Float64}[]
    for (factors, coefficient) in zip(base_patterns, coeffs)
        matrix = _local_matrix_from_factors(factors)
        last_start = periodic ? num_qubits : num_qubits - length(factors) + 1
        for start in 1:last_start
            sites = [mod1(start + offset, num_qubits) for offset in 0:(length(factors) - 1)]
            push!(terms, LocalTerm1D(
                sites, matrix, coefficient, num_qubits;
                local_dim=2, boundary=boundary))
        end
    end
    base_term_count = length(terms)

    for (factors, coefficients) in zip(disordering_terms, sample_coeffs)
        matrix = _local_matrix_from_factors(factors)
        last_start = periodic ? num_qubits : num_qubits - length(factors) + 1
        for start in 1:last_start
            sites = [mod1(start + offset, num_qubits) for offset in 0:(length(factors) - 1)]
            push!(terms, LocalTerm1D(
                sites, matrix, coefficients[start], num_qubits;
                local_dim=2, boundary=boundary))
        end
    end

    local_hamiltonian = LocalHamiltonian1D(
        terms;
        num_sites=num_qubits,
        local_dim=2,
        boundary=boundary,
        coordinate_frame=:physical,
        global_shift=0.0,
        rescaling_factor=1.0,
        scale_provenance=:identity_physical_frame,
    )
    return (
        hamiltonian=local_hamiltonian,
        base_patterns=base_patterns,
        base_coeffs=copy(coeffs),
        disordering_terms=[copy(term) for term in disordering_terms],
        disordering_coeffs=sample_coeffs,
        base_term_count=base_term_count,
    )
end

function _to_algorithm_frame(
    hamiltonian::LocalHamiltonian1D{T};
    rescaling_factor::Real,
    shift::Real,
    scale_provenance::Symbol,
) where {T<:AbstractFloat}
    _validate_local_hamiltonian_integrity(hamiltonian)
    hamiltonian.coordinate_frame == :physical || throw(ArgumentError(
        "only physical-frame local Hamiltonians can be converted to algorithm coordinates."))
    R = T(rescaling_factor)
    isfinite(R) && R > zero(T) || throw(ArgumentError(
        "algorithm-frame rescaling_factor must be finite and > 0."))
    scale_provenance in _LOCAL_1D_SCALE_PROVENANCE || throw(ArgumentError(
        "algorithm-frame rescaling requires scale_provenance=:exact_dense or " *
        ":certified_local_interval_bound."))
    shift_T = T(shift)
    isfinite(shift_T) || throw(ArgumentError("algorithm-frame shift must be finite."))
    algorithm_shift = hamiltonian.global_shift / R + shift_T
    isfinite(algorithm_shift) || throw(ArgumentError(
        "transformed algorithm-frame shift must be finite."))
    terms = LocalTerm1D{T}[
        LocalTerm1D(
            _local_sites(term), _local_matrix(term), term.coefficient / R, term.num_sites;
            local_dim=term.local_dim, boundary=term.boundary)
        for term in _local_terms(hamiltonian)
    ]
    return LocalHamiltonian1D(
        terms;
        num_sites=hamiltonian.num_sites,
        local_dim=hamiltonian.local_dim,
        boundary=hamiltonian.boundary,
        coordinate_frame=:algorithm,
        global_shift=algorithm_shift,
        rescaling_factor=R,
        scale_provenance=scale_provenance,
    )
end

"""
    build_local_heis_1d(num_qubits, coeffs; seed, periodic=true, ...)

Build the local, non-dense specification underlying [`build_heis_1d`](@ref).
Physical coordinates are the default. Algorithm coordinates require an
explicit global `rescaling_factor`, `shift`, and scale provenance. The
`:certified_local_interval_bound` provenance is accepted only when the
outward-rounded algorithm-frame enclosure lies in `[0, 0.45]`;
`:exact_dense` records a trusted external dense calculation.
"""
function build_local_heis_1d(
    num_qubits::Int,
    coeffs::Vector{Float64};
    seed::Int,
    periodic::Bool=true,
    disordering_terms::Vector{Vector{Matrix{ComplexF64}}}=
        Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
    disorder_strength::Float64=0.1,
    coordinate_frame::Symbol=:physical,
    rescaling_factor::Union{Nothing, Real}=nothing,
    shift::Union{Nothing, Real}=nothing,
    scale_provenance::Symbol=:identity_physical_frame,
)
    specification = _build_local_heis_1d_specification(
        num_qubits, coeffs;
        seed=seed,
        periodic=periodic,
        disordering_terms=disordering_terms,
        disorder_strength=disorder_strength,
    )
    if coordinate_frame == :physical
        rescaling_factor === nothing || throw(ArgumentError(
            "physical-frame build_local_heis_1d does not accept a rescaling_factor."))
        shift === nothing || throw(ArgumentError(
            "physical-frame build_local_heis_1d does not accept an algorithm-frame shift."))
        scale_provenance == :identity_physical_frame || throw(ArgumentError(
            "physical-frame build_local_heis_1d requires " *
            "scale_provenance=:identity_physical_frame."))
        return specification.hamiltonian
    elseif coordinate_frame == :algorithm
        rescaling_factor === nothing && throw(ArgumentError(
            "algorithm-frame build_local_heis_1d requires an explicit global rescaling_factor."))
        shift === nothing && throw(ArgumentError(
            "algorithm-frame build_local_heis_1d requires an explicit global shift."))
        return _to_algorithm_frame(
            specification.hamiltonian;
            rescaling_factor=rescaling_factor,
            shift=shift,
            scale_provenance=scale_provenance,
        )
    end
    throw(ArgumentError(
        "coordinate_frame must be :physical or :algorithm, got $coordinate_frame."))
end

"""
    local_pauli_jumps_1d(num_sites; boundary=:open)

Return the standard `3N` single-site Pauli sources in `(X,Y,Z)`-major order,
with the global QuantumFurnace amplitude `1/sqrt(3N)` applied exactly once.
"""
function local_pauli_jumps_1d(num_sites::Integer; boundary::Symbol=:open)
    n = Int(num_sites)
    n > 0 || throw(ArgumentError("num_sites must be > 0."))
    _validate_local_1d_boundary(boundary)
    coefficient = inv(sqrt(Float64(3n)))
    jumps = LocalJump1D{Float64}[]
    sizehint!(jumps, 3n)
    for pauli in (X, Y, Z), site in 1:n
        push!(jumps, LocalJump1D(
            [site], pauli, coefficient, n; local_dim=2, boundary=boundary))
    end
    return jumps
end

"""Reject local DLL tensor-network inputs outside the current OBC MVP."""
function validate_local_dll_tensor_network(
    hamiltonian::LocalHamiltonian1D,
    blocks::AbstractVector{<:LocalDLLBlock1D},
)
    _validate_local_hamiltonian_integrity(hamiltonian)
    hamiltonian.boundary == :open || throw(ArgumentError(
        "the one-dimensional DLL tensor-network MVP supports only open boundaries."))
    isempty(blocks) && throw(ArgumentError(
        "at least one LocalDLLBlock1D is required."))
    for (index, block) in pairs(blocks)
        _validate_local_dll_block_integrity(block)
        source = getfield(block, :source)
        source.boundary == :open || throw(ArgumentError(
            "LocalDLLBlock1D $index must use open boundaries."))
        source.num_sites == hamiltonian.num_sites || throw(ArgumentError(
            "LocalDLLBlock1D $index has chain length $(source.num_sites), expected " *
            "$(hamiltonian.num_sites)."))
        source.local_dim == hamiltonian.local_dim || throw(ArgumentError(
            "LocalDLLBlock1D $index has local dimension $(source.local_dim), expected " *
            "$(hamiltonian.local_dim)."))
    end
    return nothing
end
