const _DLL_PATCH_TARGET = :finite_patch_bohr_surrogate
const _DLL_PATCH_OMISSION_RULE = :retain_only_fully_contained_terms
const _DLL_PATCH_GAUGES = (:retain_global_shift, :omit_global_shift)

"""
    ExactDLLBohrPatch

Exact dense Gaussian DLL data for one finite OBC patch. This is an independent
reference for a named finite-patch surrogate, not a locality bound for the
global parent. The source keeps its global coefficient (including the QF
`1/sqrt(3N)` convention), and boundary-crossing Hamiltonian terms are omitted
in full.
"""
struct ExactDLLBohrPatch{T<:AbstractFloat, F<:DLLGaussianFilter}
    target_label::Symbol
    first_site::Int
    last_site::Int
    _source_sites_global::Vector{Int}
    coordinate_frame::Symbol
    rescaling_factor::T
    beta_frame::T
    beta_phys::T
    beta_alg::T
    identity_gauge::Symbol
    retained_global_shift::T
    omission_rule::Symbol
    retained_term_count::Int
    omitted_term_count::Int
    spectral_lower_bound::T
    spectral_upper_bound::T
    spectral_width_bound::T
    dimensionless_interval_radius::T
    spectral_bound_provenance::Symbol
    source_global_coefficient::Complex{T}
    filter::F
    _hamiltonian::Matrix{Complex{T}}
    _source::Matrix{Complex{T}}
    _Q::Matrix{Complex{T}}
    _L::Matrix{Complex{T}}
    _N::Matrix{Complex{T}}
    _exact_energies::Vector{T}
    locality_error::ParentErrorBound{T}
end

@inline _dll_patch_source_sites(patch::ExactDLLBohrPatch) =
    getfield(patch, :_source_sites_global)
@inline _dll_patch_hamiltonian(patch::ExactDLLBohrPatch) =
    getfield(patch, :_hamiltonian)
@inline _dll_patch_source(patch::ExactDLLBohrPatch) = getfield(patch, :_source)
@inline _dll_patch_Q(patch::ExactDLLBohrPatch) = getfield(patch, :_Q)
@inline _dll_patch_L(patch::ExactDLLBohrPatch) = getfield(patch, :_L)
@inline _dll_patch_N(patch::ExactDLLBohrPatch) = getfield(patch, :_N)
@inline _dll_patch_energies(patch::ExactDLLBohrPatch) =
    getfield(patch, :_exact_energies)

function Base.getproperty(patch::ExactDLLBohrPatch, name::Symbol)
    name === :source_sites_global && return copy(_dll_patch_source_sites(patch))
    name === :hamiltonian && return copy(_dll_patch_hamiltonian(patch))
    name === :source && return copy(_dll_patch_source(patch))
    name === :Q && return copy(_dll_patch_Q(patch))
    name === :L && return copy(_dll_patch_L(patch))
    name === :N && return copy(_dll_patch_N(patch))
    name === :exact_energies && return copy(_dll_patch_energies(patch))
    return getfield(patch, name)
end

function Base.propertynames(::ExactDLLBohrPatch, private::Bool=false)
    public_names = (
        :target_label, :first_site, :last_site, :source_sites_global,
        :coordinate_frame, :rescaling_factor, :beta_frame, :beta_phys,
        :beta_alg, :identity_gauge,
        :retained_global_shift, :omission_rule, :retained_term_count,
        :omitted_term_count, :spectral_lower_bound, :spectral_upper_bound,
        :spectral_width_bound, :dimensionless_interval_radius,
        :spectral_bound_provenance, :source_global_coefficient, :filter,
        :hamiltonian, :source, :Q, :L, :N, :exact_energies,
        :locality_error,
    )
    private_names = (
        :_source_sites_global, :_hamiltonian, :_source, :_Q, :_L, :_N,
        :_exact_energies,
    )
    return private ? (public_names..., private_names...) : public_names
end

function _resolve_dll_patch_bounds(
    source::LocalJump1D,
    num_sites::Int;
    first_site::Union{Nothing, Integer},
    last_site::Union{Nothing, Integer},
    radius::Union{Nothing, Integer},
)
    source_sites = _local_sites(source)
    if radius !== nothing
        first_site === nothing && last_site === nothing || throw(ArgumentError(
            "specify either radius or explicit first_site/last_site, not both."))
        radius >= 0 || throw(ArgumentError("patch radius must be nonnegative."))
        first = max(1, minimum(source_sites) - Int(radius))
        last = min(num_sites, maximum(source_sites) + Int(radius))
        return first, last
    end
    first_site !== nothing && last_site !== nothing || throw(ArgumentError(
        "explicit patches require both first_site and last_site."))
    first = Int(first_site)
    last = Int(last_site)
    1 <= first <= last <= num_sites || throw(ArgumentError(
        "invalid patch $first:$last for a $num_sites-site chain."))
    return first, last
end

function _materialize_dll_patch_data(
    hamiltonian::LocalHamiltonian1D{T},
    source::LocalJump1D{T},
    first_site::Int,
    last_site::Int,
    identity_gauge::Symbol,
) where {T<:AbstractFloat}
    patch_size = last_site - first_site + 1
    source_sites = _local_sites(source)
    all(site -> first_site <= site <= last_site, source_sites) ||
        throw(ArgumentError(
            "source support $source_sites is not contained in patch " *
            "$first_site:$last_site."))
    retained_terms = LocalTerm1D{T}[]
    for term in _local_terms(hamiltonian)
        term_sites = _local_sites(term)
        all(site -> first_site <= site <= last_site, term_sites) || continue
        push!(retained_terms, LocalTerm1D(
            term_sites .- (first_site - 1),
            _local_matrix(term),
            term.coefficient,
            patch_size;
            local_dim=hamiltonian.local_dim,
            boundary=:open,
        ))
    end
    retained_shift = identity_gauge === :retain_global_shift ?
        hamiltonian.global_shift : zero(T)
    lower, upper = _weyl_spectral_interval_1d(retained_terms, retained_shift)
    width = upper == lower ? zero(T) : nextfloat(upper - lower)
    dimension = _checked_local_dimension(hamiltonian.local_dim, patch_size)
    H = zeros(Complex{T}, dimension, dimension)
    for term in retained_terms
        embedded = _embed_local_matrix_1d(
            _local_sites(term), _local_matrix(term), patch_size,
            hamiltonian.local_dim)
        H .+= term.coefficient .* embedded
    end
    if !iszero(retained_shift)
        for index in 1:dimension
            H[index, index] += retained_shift
        end
    end
    remapped_source_sites = source_sites .- (first_site - 1)
    A = source.coefficient .* Matrix(_embed_local_matrix_1d(
        remapped_source_sites, _local_matrix(source), patch_size,
        source.local_dim))
    return (; H, A, retained_terms, retained_shift, lower, upper, width)
end

function _exact_gaussian_dll_bohr(
    hamiltonian::AbstractMatrix{Complex{T}},
    source::AbstractMatrix{Complex{T}},
    filter::DLLGaussianFilter,
) where {T<:AbstractFloat}
    decomposition = eigen(Hermitian(Matrix{Complex{T}}(hamiltonian)))
    energies = Vector{T}(decomposition.values)
    eigenvectors = decomposition.vectors
    source_eigen = adjoint(eigenvectors) * source * eigenvectors
    Q_eigen = similar(source_eigen)
    L_eigen = similar(source_eigen)
    beta = T(filter.beta)
    for column in axes(source_eigen, 2), row in axes(source_eigen, 1)
        frequency = energies[row] - energies[column]
        Q_eigen[row, column] = T(q_weight(filter, frequency)) *
                               source_eigen[row, column]
        L_eigen[row, column] = T(freq_kernel(filter, frequency)) *
                               source_eigen[row, column]
    end
    rate_eigen = adjoint(L_eigen) * L_eigen
    N_eigen = similar(rate_eigen)
    for column in axes(rate_eigen, 2), row in axes(rate_eigen, 1)
        x = beta * (energies[row] - energies[column])
        N_eigen[row, column] = -T(_sech_quarter_dimensionless(x)) *
                               rate_eigen[row, column]
    end
    rotate(operator) = eigenvectors * operator * adjoint(eigenvectors)
    return (; Q=rotate(Q_eigen), L=rotate(L_eigen), N=rotate(N_eigen),
            energies)
end

"""
    exact_dll_bohr_patch(hamiltonian, source, filter; ...)

Diagonalize only one selected OBC patch and construct exact Gaussian DLL
`Q`, `L`, and signed `N` operators in the Hamiltonian's existing global
coordinate frame. A term is retained iff its complete support lies in the
patch; boundary-crossing terms are omitted rather than truncated.

Use either `radius` around the source support or explicit `first_site` and
`last_site`. `identity_gauge` may retain or omit the global scalar shift,
which leaves every commutator-filtered operator unchanged.
"""
function exact_dll_bohr_patch(
    hamiltonian::LocalHamiltonian1D{T},
    source::LocalJump1D{T},
    filter::F;
    first_site::Union{Nothing, Integer}=nothing,
    last_site::Union{Nothing, Integer}=nothing,
    radius::Union{Nothing, Integer}=nothing,
    identity_gauge::Symbol=:retain_global_shift,
    beta_phys::Union{Nothing, Real}=nothing,
) where {T<:AbstractFloat, F<:DLLGaussianFilter{T}}
    _validate_local_hamiltonian_integrity(hamiltonian)
    _validate_local_jump_integrity(source)
    _local_jump_is_hermitian(source) || throw(ArgumentError(
        "exact DLL patch references require a Hermitian source."))
    _validate_contiguous_support_1d(
        _local_sites(source), source.num_sites, source.boundary)
    hamiltonian.boundary === :open || throw(ArgumentError(
        "exact DLL patch references support open boundaries only."))
    source.boundary === :open || throw(ArgumentError(
        "exact DLL patch sources must use open boundaries."))
    source.num_sites == hamiltonian.num_sites || throw(ArgumentError(
        "source and Hamiltonian chain lengths must match."))
    source.local_dim == hamiltonian.local_dim || throw(ArgumentError(
        "source and Hamiltonian local dimensions must match."))
    identity_gauge in _DLL_PATCH_GAUGES || throw(ArgumentError(
        "identity_gauge must be :retain_global_shift or :omit_global_shift."))
    _require_admissible_dll_filter(filter)
    beta_metadata = _frame_beta_metadata(
        hamiltonian, filter; beta_phys)
    first, last = _resolve_dll_patch_bounds(
        source, hamiltonian.num_sites;
        first_site, last_site, radius)
    patch = _materialize_dll_patch_data(
        hamiltonian, source, first, last, identity_gauge)
    exact = _exact_gaussian_dll_bohr(patch.H, patch.A, filter)
    beta = T(filter.beta)
    interval_radius = beta * patch.width
    !iszero(interval_radius) && (interval_radius = nextfloat(interval_radius))
    locality_error = ParentErrorBound(
        T, :locality_radius;
        evidence=:unmeasured,
        note="successive patch-radius changes are convergence evidence, not " *
             "a summed operator-norm locality-tail bound",
    )
    return ExactDLLBohrPatch{T, F}(
        _DLL_PATCH_TARGET, first, last, copy(_local_sites(source)),
        hamiltonian.coordinate_frame, hamiltonian.rescaling_factor,
        T(beta_metadata.beta_frame), T(beta_metadata.beta_phys),
        T(beta_metadata.beta_alg),
        identity_gauge, patch.retained_shift, _DLL_PATCH_OMISSION_RULE,
        length(patch.retained_terms),
        length(_local_terms(hamiltonian)) - length(patch.retained_terms),
        patch.lower, patch.upper, patch.width, interval_radius,
        hamiltonian.spectral_bound_provenance, source.coefficient, filter,
        copy(patch.H), copy(patch.A), copy(exact.Q), copy(exact.L),
        copy(exact.N), copy(exact.energies),
        locality_error,
    )
end

function exact_dll_bohr_patch(
    ::LocalHamiltonian1D{T},
    ::LocalJump1D{T},
    filter::AbstractFilter;
    kwargs...,
) where {T<:AbstractFloat}
    message = filter isa DLLGaussianFilter ?
        "exact DLL patch references require DLLGaussianFilter{$T} to match " *
        "the local Hamiltonian numeric type." :
        "Task 7A patch references support DLLGaussianFilter only; got " *
        "$(nameof(typeof(filter)))."
    throw(ArgumentError(message))
end

function exact_dll_bohr_patch(
    hamiltonian::LocalHamiltonian1D{T},
    block::LocalDLLBlock1D{T};
    kwargs...,
) where {T<:AbstractFloat}
    length(block.channels) == 1 || throw(ArgumentError(
        "Task 7A patch references accept exactly one DLL channel; construct " *
        "separate references for channelwise accounting."))
    filter = only(block.channels)
    filter isa DLLGaussianFilter || throw(ArgumentError(
        "Task 7A patch references support DLLGaussianFilter only."))
    return exact_dll_bohr_patch(
        hamiltonian, block.source, filter; kwargs...)
end

function dll_gaussian_chebyshev_data(
    patch::ExactDLLBohrPatch;
    kwargs...,
)
    return dll_gaussian_chebyshev_data(
        patch.dimensionless_interval_radius;
        interval_provenance=patch.spectral_bound_provenance,
        kwargs...,
    )
end

"""
Apply the scalar-certified Gaussian DLL polynomials to an exact patch input.
The returned dense operators are floating-point convergence evidence; their
recurrence and matrix-product errors are not included in the scalar bound.
"""
function approximate_dll_bohr_patch(
    patch::ExactDLLBohrPatch;
    kwargs...,
)
    data = dll_gaussian_chebyshev_data(patch; kwargs...)
    approximation = apply_dll_gaussian_chebyshev(
        _dll_patch_hamiltonian(patch), _dll_patch_source(patch),
        patch.filter.beta, data)
    dense_error = ParentErrorBound(
        typeof(patch.beta_frame), :dense_recurrence;
        evidence=:unmeasured,
        note="scalar polynomial bounds exclude floating dense recurrence, " *
             "L' L product, and propagated N errors",
    )
    return (;
        target_label=patch.target_label,
        first_site=patch.first_site,
        last_site=patch.last_site,
        coordinate_frame=patch.coordinate_frame,
        beta_frame=patch.beta_frame,
        beta_phys=patch.beta_phys,
        beta_alg=patch.beta_alg,
        identity_gauge=patch.identity_gauge,
        omission_rule=patch.omission_rule,
        locality_error=patch.locality_error,
        dense_recurrence_error=dense_error,
        approximation...,
        chebyshev_data=data,
    )
end
