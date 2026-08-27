const _BOHR_PATCH_OMISSION_RULE = :retain_only_fully_contained_terms
const _BOHR_PATCH_IDENTITY_GAUGE = :omit_global_shift

"""
    local_operator_siteinds(n; local_dim=2)

Create unconstrained local operator indices for a finite one-dimensional spin
chain. The sites carry no symmetry sectors because the first Bohr-MPO backend
must accept Pauli `X`, `Y`, and `Z` sources in one common representation.
"""
function local_operator_siteinds(
    n::Integer;
    local_dim::Integer=2,
)
    n > 0 || throw(ArgumentError("n must be positive."))
    local_dim >= 2 || throw(ArgumentError("local_dim must be at least two."))
    return ITensorMPS.siteinds("Qudit", Int(n); dim=Int(local_dim))
end

function _validate_local_operator_sites(sites, n::Int, local_dim::Int)
    length(sites) == n || throw(DimensionMismatch(
        "received $(length(sites)) operator sites, expected $n."))
    all(site -> ITensors.dim(site) == local_dim, sites) ||
        throw(DimensionMismatch(
            "every operator site must have dimension $local_dim."))
    all(site -> !ITensors.hasqns(site), sites) || throw(ArgumentError(
        "the Gaussian Bohr-MPO backend currently requires unconstrained " *
        "site indices so all Pauli source families share one representation."))
    return collect(sites)
end

@inline function _local_basis_digit(
    linear_index::Int,
    position::Int,
    support_size::Int,
    local_dim::Int,
)
    stride = local_dim^(support_size - position)
    return ((linear_index - 1) ÷ stride) % local_dim + 1
end

function _add_local_matrix_units!(
    operator_sum::ITensorMPS.OpSum,
    matrix::AbstractMatrix{<:Number},
    support::AbstractVector{<:Integer},
    coefficient::Number,
    local_dim::Int,
)
    support_size = length(support)
    local_dimension = local_dim^support_size
    size(matrix) == (local_dimension, local_dimension) ||
        throw(DimensionMismatch(
            "local operator on $support_size sites must have size " *
            "($local_dimension, $local_dimension)."))
    CT = promote_type(typeof(complex(coefficient)), eltype(matrix))
    for column in 1:local_dimension, row in 1:local_dimension
        value = CT(coefficient) * CT(matrix[row, column])
        iszero(value) && continue
        term = Any[value]
        for position in 1:support_size
            row_digit = _local_basis_digit(
                row, position, support_size, local_dim)
            column_digit = _local_basis_digit(
                column, position, support_size, local_dim)
            matrix_unit = zeros(CT, local_dim, local_dim)
            matrix_unit[row_digit, column_digit] = one(CT)
            push!(term, matrix_unit, Int(support[position]))
        end
        ITensorMPS.add!(operator_sum, Tuple(term))
    end
    return operator_sum
end

function _identity_mpo(::Type{T}, sites) where {T<:AbstractFloat}
    local_dim = ITensors.dim(first(sites))
    identity_matrix = Matrix{Complex{T}}(I, local_dim, local_dim)
    return ITensorMPS.MPO(Complex{T}, sites, _ -> identity_matrix)
end

function _opsum_or_zero_mpo(
    operator_sum::ITensorMPS.OpSum,
    ::Type{T},
    sites,
    term_count::Int,
) where {T<:AbstractFloat}
    term_count > 0 && return ITensorMPS.MPO(Complex{T}, operator_sum, sites)
    return zero(T) * _identity_mpo(T, sites)
end

"""
    local_hamiltonian_mpo(hamiltonian; sites=nothing)

Build an MPO directly from the finite local terms of an open-chain
`LocalHamiltonian1D`. No dense global Hamiltonian is formed. The stored scalar
identity shift is deliberately omitted: every use of this MPO in Task 7B is
inside `ad_H`, where that shift cancels exactly.
"""
function local_hamiltonian_mpo(
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T};
    sites=nothing,
) where {T<:AbstractFloat}
    QuantumFurnace._validate_local_hamiltonian_integrity(hamiltonian)
    hamiltonian.boundary == :open || throw(ArgumentError(
        "the Bohr-MPO Hamiltonian builder supports open boundaries only."))
    operator_sites = sites === nothing ?
        local_operator_siteinds(
            hamiltonian.num_sites; local_dim=hamiltonian.local_dim) :
        _validate_local_operator_sites(
            sites, hamiltonian.num_sites, hamiltonian.local_dim)
    operator_sum = ITensorMPS.OpSum{Complex{T}}()
    term_count = 0
    for term in QuantumFurnace._local_terms(hamiltonian)
        before = length(ITensors.terms(operator_sum))
        _add_local_matrix_units!(
            operator_sum,
            QuantumFurnace._local_matrix(term),
            QuantumFurnace._local_sites(term),
            term.coefficient,
            hamiltonian.local_dim,
        )
        term_count += length(ITensors.terms(operator_sum)) - before
    end
    return _opsum_or_zero_mpo(
        operator_sum, T, operator_sites, term_count)
end

"""
    local_jump_mpo(source; sites=nothing)

Build one local source MPO without padding a dense global matrix. The source's
stored coefficient, including QuantumFurnace's `1/sqrt(3N)` convention, is
applied exactly once.
"""
function local_jump_mpo(
    source::QuantumFurnace.LocalJump1D{T};
    sites=nothing,
) where {T<:AbstractFloat}
    QuantumFurnace._validate_local_jump_integrity(source)
    source.boundary == :open || throw(ArgumentError(
        "the Bohr-MPO source builder supports open boundaries only."))
    operator_sites = sites === nothing ?
        local_operator_siteinds(source.num_sites; local_dim=source.local_dim) :
        _validate_local_operator_sites(
            sites, source.num_sites, source.local_dim)
    operator_sum = ITensorMPS.OpSum{Complex{T}}()
    _add_local_matrix_units!(
        operator_sum,
        QuantumFurnace._local_matrix(source),
        QuantumFurnace._local_sites(source),
        source.coefficient,
        source.local_dim,
    )
    return _opsum_or_zero_mpo(
        operator_sum,
        T,
        operator_sites,
        length(ITensors.terms(operator_sum)),
    )
end

function _compact_bohr_patch(
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    source::QuantumFurnace.LocalJump1D{T},
    radius::Int,
) where {T<:AbstractFloat}
    source_sites = QuantumFurnace._local_sites(source)
    first_site = max(1, minimum(source_sites) - radius)
    last_site = min(hamiltonian.num_sites, maximum(source_sites) + radius)
    patch_size = last_site - first_site + 1
    retained_terms = QuantumFurnace.LocalTerm1D{T}[]
    for term in QuantumFurnace._local_terms(hamiltonian)
        term_sites = QuantumFurnace._local_sites(term)
        all(site -> first_site <= site <= last_site, term_sites) || continue
        push!(retained_terms, QuantumFurnace.LocalTerm1D(
            term_sites .- (first_site - 1),
            QuantumFurnace._local_matrix(term),
            term.coefficient,
            patch_size;
            local_dim=hamiltonian.local_dim,
            boundary=:open,
        ))
    end
    compact_hamiltonian = QuantumFurnace.LocalHamiltonian1D(
        retained_terms;
        num_sites=patch_size,
        local_dim=hamiltonian.local_dim,
        boundary=:open,
        coordinate_frame=hamiltonian.coordinate_frame,
        global_shift=zero(T),
        rescaling_factor=hamiltonian.rescaling_factor,
        scale_provenance=hamiltonian.scale_provenance,
    )
    compact_source = QuantumFurnace.LocalJump1D(
        source_sites .- (first_site - 1),
        QuantumFurnace._local_matrix(source),
        source.coefficient,
        patch_size;
        local_dim=source.local_dim,
        boundary=:open,
    )
    return (;
        hamiltonian=compact_hamiltonian,
        source=compact_source,
        first_site,
        last_site,
        source_sites_global=copy(source_sites),
        retained_term_count=length(retained_terms),
        omitted_term_count=length(QuantumFurnace._local_terms(hamiltonian)) -
                           length(retained_terms),
    )
end
