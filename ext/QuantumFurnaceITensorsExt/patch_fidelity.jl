"""
    ExactPatchMPOFactorizationControls(; cutoff=1e-14, maxdim=128)

Controls for factoring one-copy exact patch operators into compact MPOs. These
controls are deliberately separate from the Chebyshev and parent-assembly
controls. A reference whose output bond reaches `maxdim` is recorded as
cap-saturated and cannot qualify a fidelity cell.
"""
struct ExactPatchMPOFactorizationControls{T<:AbstractFloat}
    cutoff::T
    maxdim::Int

    function ExactPatchMPOFactorizationControls{T}(
        cutoff::T,
        maxdim::Int,
    ) where {T<:AbstractFloat}
        isfinite(cutoff) && cutoff >= zero(T) || throw(ArgumentError(
            "exact-patch factorization cutoff must be finite and nonnegative."))
        maxdim > 0 || throw(ArgumentError(
            "exact-patch factorization maxdim must be positive."))
        return new{T}(cutoff, maxdim)
    end
end

function ExactPatchMPOFactorizationControls(;
    cutoff::Real=1e-14,
    maxdim::Integer=128,
)
    T = typeof(float(cutoff))
    return ExactPatchMPOFactorizationControls{T}(T(cutoff), Int(maxdim))
end

"""Measured reconstruction telemetry for one compact one-copy MPO."""
struct ExactPatchMPOFactorizationRecord{T<:AbstractFloat}
    operator_label::Symbol
    patch_first_site::Int
    patch_last_site::Int
    matrix_dimension::Int
    requested_cutoff::T
    requested_maxdim::Int
    maximum_output_bond::Int
    cap_saturated::Bool
    absolute_frobenius_error::T
    relative_frobenius_error::T
end

"""Exact analytic `Q`, `L`, and signed `N` represented on one compact patch."""
struct ExactFinitePatchBohrMPO{
    T<:AbstractFloat,
    F<:QuantumFurnace.DLLGaussianFilter,
    M,
    S,
}
    target_label::Symbol
    first_site::Int
    last_site::Int
    source_sites_global::Vector{Int}
    coordinate_frame::Symbol
    beta_frame::T
    beta_phys::T
    beta_alg::T
    identity_gauge::Symbol
    omission_rule::Symbol
    retained_term_count::Int
    omitted_term_count::Int
    sites::S
    Q::M
    L::M
    N::M
    patch::QuantumFurnace.ExactDLLBohrPatch{T, F}
    factorization_records::Vector{ExactPatchMPOFactorizationRecord{T}}
end

"""
    ExactFinitePatchDLLParentReference

Exact analytic finite-patch target assembled without constructing a dense
doubled-patch parent. Only the one-copy `Q`, `L`, and `N` matrices are dense;
they are factored on their compact physical windows and lifted through the
same parent-block constructor as the represented Chebyshev parent.
"""
struct ExactFinitePatchDLLParentReference{
    T<:AbstractFloat,
    F<:QuantumFurnace.DLLGaussianFilter,
    P<:FinitePatchDLLParent,
    C<:ExactPatchMPOFactorizationControls,
}
    parent::P
    exact_patches::Vector{QuantumFurnace.ExactDLLBohrPatch{T, F}}
    factorization_controls::C
    factorization_records::Vector{ExactPatchMPOFactorizationRecord{T}}
    maximum_relative_reconstruction_error::T
    factorization_cap_saturated::Bool
end

function _compact_mpo_to_dense(
    operator::ITensorMPS.MPO,
    sites::Vector{ITensors.Index{Int}},
)
    length(operator) == length(sites) || throw(DimensionMismatch(
        "compact MPO and site collection must have equal length."))
    local_dim = ITensors.dim(first(sites))
    all(site -> ITensors.dim(site) == local_dim, sites) || throw(
        DimensionMismatch("compact MPO sites must have a uniform local dimension."))
    dimension = Base.checked_pow(local_dim, length(sites))
    tensor = reduce(*, operator)
    array = Array(tensor, ITensors.prime.(sites)..., ITensors.dag.(sites)...)
    itensor_matrix = reshape(array, dimension, dimension)
    permutation = _physical_qf_to_itensor_permutation(
        length(sites), local_dim)
    inverse = invperm(permutation)
    return Matrix(itensor_matrix[inverse, inverse])
end

function _physical_qf_to_itensor_permutation(n::Int, local_dim::Int)
    dimension = Base.checked_pow(local_dim, n)
    permutation = Vector{Int}(undef, dimension)
    for qf_index_zero in 0:(dimension - 1)
        itensor_offset = 0
        for site in 1:n
            qf_stride = Base.checked_pow(local_dim, n - site)
            digit = (qf_index_zero ÷ qf_stride) % local_dim
            itensor_offset += digit * Base.checked_pow(local_dim, site - 1)
        end
        permutation[itensor_offset + 1] = qf_index_zero + 1
    end
    return permutation
end

function _factor_exact_patch_operator(
    matrix::AbstractMatrix{Complex{T}},
    sites::Vector{ITensors.Index{Int}},
    controls::ExactPatchMPOFactorizationControls{T},
    label::Symbol,
    first_site::Int,
    last_site::Int,
) where {T<:AbstractFloat}
    local_dim = ITensors.dim(first(sites))
    dimension = Base.checked_pow(local_dim, length(sites))
    size(matrix) == (dimension, dimension) || throw(DimensionMismatch(
        "exact patch matrix has size $(size(matrix)), expected " *
        "($dimension, $dimension)."))
    operator_indices = vcat(ITensors.prime.(sites), ITensors.dag.(sites))
    permutation = _physical_qf_to_itensor_permutation(
        length(sites), local_dim)
    itensor_matrix = matrix[permutation, permutation]
    tensor = ITensors.ITensor(
        reshape(itensor_matrix, ntuple(_ -> local_dim, 2 * length(sites))),
        operator_indices...,
    )
    operator = ITensorMPS.MPO(
        tensor,
        sites;
        cutoff=controls.cutoff,
        maxdim=controls.maxdim,
    )
    reconstruction = Matrix{Complex{T}}(_compact_mpo_to_dense(operator, sites))
    absolute_error = T(norm(reconstruction - matrix))
    relative_error = absolute_error / max(T(norm(matrix)), eps(T))
    maximum_bond = _maximum_mpo_bond(operator)
    record = ExactPatchMPOFactorizationRecord{T}(
        label,
        first_site,
        last_site,
        dimension,
        controls.cutoff,
        controls.maxdim,
        maximum_bond,
        length(sites) > 1 && maximum_bond >= controls.maxdim,
        absolute_error,
        relative_error,
    )
    return operator, record
end

function _exact_patch_bohr_mpo(
    patch::QuantumFurnace.ExactDLLBohrPatch{T, F},
    controls::ExactPatchMPOFactorizationControls{T},
) where {T<:AbstractFloat, F<:QuantumFurnace.DLLGaussianFilter}
    patch_size = patch.last_site - patch.first_site + 1
    sites = Vector{ITensors.Index{Int}}(local_operator_siteinds(patch_size))
    Q, q_record = _factor_exact_patch_operator(
        patch.Q, sites, controls, :Q, patch.first_site, patch.last_site)
    L, l_record = _factor_exact_patch_operator(
        patch.L, sites, controls, :L, patch.first_site, patch.last_site)
    N, n_record = _factor_exact_patch_operator(
        patch.N, sites, controls, :N, patch.first_site, patch.last_site)
    records = ExactPatchMPOFactorizationRecord{T}[
        q_record, l_record, n_record,
    ]
    return ExactFinitePatchBohrMPO{
        T, F, typeof(Q), typeof(sites),
    }(
        :finite_patch_bohr_surrogate,
        patch.first_site,
        patch.last_site,
        patch.source_sites_global,
        patch.coordinate_frame,
        patch.beta_frame,
        patch.beta_phys,
        patch.beta_alg,
        patch.identity_gauge,
        patch.omission_rule,
        patch.retained_term_count,
        patch.omitted_term_count,
        sites,
        Q,
        L,
        N,
        patch,
        records,
    )
end

function _exact_patch_parent_error_ledger(
    records::Vector{ExactPatchMPOFactorizationRecord{T}},
    block_keys::Vector{Tuple{Int, Int}},
    block_compression_bounds::Vector{T},
    total_compression_bound::T,
) where {T<:AbstractFloat}
    maximum_factorization_error = maximum(
        record.relative_frobenius_error for record in records)
    block_entries = Tuple(
        QuantumFurnace.ParentErrorBound(
            T,
            Symbol("exact_patch_parent_block_", source, "_", channel);
            magnitude=block_compression_bounds[index],
            evidence=:floating_point_norm,
            note="measured Frobenius change from final parent-block additions; " *
                 "compact Q/L/N reconstruction is retained separately",
        )
        for (index, (source, channel)) in pairs(block_keys)
    )
    return QuantumFurnace.ParentErrorLedger(
        T;
        spectral_interval=QuantumFurnace.ParentErrorBound(
            T,
            :spectral_interval;
            evidence=:unmeasured,
            note="dense one-copy eigendecomposition is floating-point reference data",
        ),
        scalar_polynomial=QuantumFurnace.ParentErrorBound(
            T,
            :scalar_polynomial;
            magnitude=zero(T),
            evidence=:rigorous_bound,
            note="the exact-patch target evaluates analytic scalar functions without a polynomial approximation",
        ),
        modular_compatibility=QuantumFurnace.ParentErrorBound(
            T,
            :modular_compatibility;
            evidence=:unmeasured,
            note="floating one-copy reconstruction is checked directly, not interval-certified",
        ),
        locality_radius=QuantumFurnace.ParentErrorBound(
            T,
            :locality_radius;
            evidence=:unmeasured,
            note="exact analytic finite-patch target; no global locality bound is implied",
        ),
        mpo_recurrences=(QuantumFurnace.ParentErrorBound(
            T,
            :compact_exact_patch_factorization;
            magnitude=maximum_factorization_error,
            evidence=:floating_point_norm,
            note="maximum measured relative Frobenius reconstruction error over compact Q/L/N MPOs",
        ),),
        mpo_products=(QuantumFurnace.ParentErrorBound(
            T,
            :exact_patch_q_sandwich;
            evidence=:unmeasured,
            note="zip-up construction of conj(Q) tensor Q has no operator-norm certificate",
        ),),
        block_assembly=block_entries,
        total_sum_compression=QuantumFurnace.ParentErrorBound(
            T,
            :total_sum_compression;
            magnitude=total_compression_bound,
            evidence=:floating_point_norm,
            note="sum of measured final-truncation Frobenius differences",
        ),
    )
end

"""
    build_exact_finite_patch_dll_parent(config, hamiltonian, blocks,
                                        assembly_controls;
                                        factorization_controls, ...)

Build the exact analytic finite-patch target for every standard Pauli source.
The result retains a `FinitePatchDLLParent`, so it can be used as an
independently tighter residual target by the Task-10 solver, but it remains a
finite-patch surrogate and never an exact-global parent.
"""
function build_exact_finite_patch_dll_parent(
    config::QuantumFurnace.Config{
        QuantumFurnace.TensorNetworkSpectrum,
        QuantumFurnace.BohrDomain,
        QuantumFurnace.DLL,
        T,
    },
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    blocks::AbstractVector{<:QuantumFurnace.LocalDLLBlock1D{T}},
    assembly_controls::QuantumFurnace.BohrMPOControls{T};
    factorization_controls::ExactPatchMPOFactorizationControls{T}=
        ExactPatchMPOFactorizationControls(
            cutoff=T(1e-14), maxdim=128),
    sites=nothing,
    symmetrize::Bool=false,
    raw_generator_rate::Real=1,
    proposal_normalisation::Symbol=:qf_averaged,
    include_sweep_clock::Bool=false,
    diagnostic_probe_count::Integer=1,
    diagnostic_probe_seed::Integer=0x10f,
) where {T<:AbstractFloat}
    QuantumFurnace.validate_dll_tensor_network(
        config, hamiltonian, blocks, assembly_controls)
    assembly_controls.target_label == :finite_patch_bohr_surrogate ||
        throw(ArgumentError(
            "exact finite-patch references require target_label=" *
            ":finite_patch_bohr_surrogate."))
    radius = something(assembly_controls.patch_radius)
    fused_sites = sites === nothing ?
        fused_siteinds(
            hamiltonian.num_sites; physical_dim=hamiltonian.local_dim) :
        _validate_parent_fused_sites(
            sites, hamiltonian.num_sites, hamiltonian.local_dim)
    fused_sites = _validate_parent_fused_sites(
        fused_sites, hamiltonian.num_sites, hamiltonian.local_dim)
    provenance = QuantumFurnace.dll_parent_provenance(
        config,
        hamiltonian,
        blocks;
        controls=assembly_controls,
        raw_generator_rate,
        proposal_normalisation,
        include_sweep_clock,
    )

    exact_patches = QuantumFurnace.ExactDLLBohrPatch[]
    exact_bohr_blocks = ExactFinitePatchBohrMPO[]
    block_keys = Tuple{Int, Int}[]
    factorization_records = ExactPatchMPOFactorizationRecord{T}[]
    for (source_index, block) in pairs(blocks)
        channels = getfield(block, :channels)
        length(channels) == 1 || throw(ArgumentError(
            "exact finite-patch references currently require one Gaussian channel."))
        patch = QuantumFurnace.exact_dll_bohr_patch(
            hamiltonian,
            block;
            radius,
            identity_gauge=:omit_global_shift,
            beta_phys=config.beta_phys,
        )
        bohr = _exact_patch_bohr_mpo(patch, factorization_controls)
        push!(exact_patches, patch)
        push!(exact_bohr_blocks, bohr)
        append!(factorization_records, bohr.factorization_records)
        push!(block_keys, (source_index, 1))
    end
    typed_patches = Vector{typeof(first(exact_patches))}(exact_patches)
    typed_bohr = Vector{typeof(first(exact_bohr_blocks))}(exact_bohr_blocks)
    assembly = _assemble_parent_components(
        typed_bohr,
        block_keys,
        fused_sites,
        assembly_controls;
        symmetrize,
        diagnostic_probe_count=Int(diagnostic_probe_count),
        diagnostic_probe_seed=Int(diagnostic_probe_seed),
    )
    ledger = _exact_patch_parent_error_ledger(
        factorization_records,
        block_keys,
        assembly.block_compression_bounds,
        assembly.total_compression_bound,
    )
    bundle = QuantumFurnace.DLLParentBundle(
        :finite_patch_bohr_surrogate,
        assembly.total_parent,
        assembly.represented_blocks,
        block_keys,
        true,
        ledger,
    )
    model_specification = (
        hamiltonian=hamiltonian,
        blocks=Tuple(blocks),
    )
    parent = FinitePatchDLLParent{
        T,
        typeof(first(typed_bohr)),
        typeof(bundle),
        typeof(provenance),
        typeof(model_specification),
    }(
        bundle,
        fused_sites,
        typed_bohr,
        provenance,
        model_specification,
        assembly_controls,
        assembly.assembly_records,
        assembly.diagnostics,
    )
    maximum_error = maximum(
        record.relative_frobenius_error for record in factorization_records)
    saturated = any(record -> record.cap_saturated, factorization_records)
    return ExactFinitePatchDLLParentReference{
        T,
        typeof(first(typed_patches).filter),
        typeof(parent),
        typeof(factorization_controls),
    }(
        parent,
        typed_patches,
        factorization_controls,
        factorization_records,
        maximum_error,
        saturated,
    )
end

struct ExactPatchReferenceComparison{T<:AbstractFloat}
    maximum_relative_action_difference::T
    maximum_coarse_reconstruction_error::T
    maximum_tight_reconstruction_error::T
    controls_independent::Bool
    coarse_cap_saturated::Bool
    tight_cap_saturated::Bool
    accepted::Bool
    probe_count::Int
    probe_seed::Int
end

function _validate_exact_reference_pair(
    coarse::ExactFinitePatchDLLParentReference,
    tight::ExactFinitePatchDLLParentReference,
)
    isequal(coarse.parent.model_specification,
            tight.parent.model_specification) || throw(ArgumentError(
        "exact-patch references must represent the same Hamiltonian and sources."))
    _validate_fidelity_provenances(
        coarse.parent.provenance, tight.parent.provenance)
    coarse.parent.controls.patch_radius == tight.parent.controls.patch_radius ||
        throw(ArgumentError("exact-patch references must use the same radius."))
    coarse.parent.sites == tight.parent.sites || throw(ArgumentError(
        "exact-patch references must use identical fused site indices."))
    coarse.parent.bundle.block_keys == tight.parent.bundle.block_keys ||
        throw(ArgumentError("exact-patch reference block accounting differs."))
    return nothing
end

"""Compare two independently controlled exact-patch references by action."""
function compare_exact_patch_references(
    coarse::ExactFinitePatchDLLParentReference{T},
    tight::ExactFinitePatchDLLParentReference{T};
    relative_tolerance::Real,
    probe_count::Integer=3,
    probe_seed::Integer=0x10f5,
) where {T<:AbstractFloat}
    _validate_exact_reference_pair(coarse, tight)
    count = Int(probe_count)
    count > 0 || throw(ArgumentError("probe_count must be positive."))
    tolerance = T(relative_tolerance)
    isfinite(tolerance) && tolerance >= zero(T) || throw(ArgumentError(
        "relative_tolerance must be finite and nonnegative."))
    differences = T[]
    for probe_index in 1:count
        rng = Random.MersenneTwister(Int(probe_seed) + probe_index - 1)
        state = ITensorMPS.random_mps(
            rng, Complex{T}, coarse.parent.sites; linkdims=1)
        ITensorMPS.normalize!(state)
        coarse_image = _apply_without_requested_truncation(
            coarse.parent.bundle.total_parent, state)
        tight_image = _apply_without_requested_truncation(
            tight.parent.bundle.total_parent, state)
        push!(differences,
            _network_difference_norm(coarse_image, tight_image, T) /
            max(T(norm(tight_image)), eps(T)))
    end
    coarse_controls = coarse.factorization_controls
    tight_controls = tight.factorization_controls
    controls_independent =
        tight_controls.cutoff <= coarse_controls.cutoff &&
        tight_controls.maxdim >= coarse_controls.maxdim &&
        (tight_controls.cutoff < coarse_controls.cutoff ||
         tight_controls.maxdim > coarse_controls.maxdim)
    maximum_difference = maximum(differences)
    accepted = controls_independent &&
        !coarse.factorization_cap_saturated &&
        !tight.factorization_cap_saturated &&
        coarse.maximum_relative_reconstruction_error <= tolerance &&
        tight.maximum_relative_reconstruction_error <= tolerance &&
        maximum_difference <= tolerance
    return ExactPatchReferenceComparison{T}(
        maximum_difference,
        coarse.maximum_relative_reconstruction_error,
        tight.maximum_relative_reconstruction_error,
        controls_independent,
        coarse.factorization_cap_saturated,
        tight.factorization_cap_saturated,
        accepted,
        count,
        Int(probe_seed),
    )
end

"""One low-energy layer retained by a global-gap fidelity cell."""
struct DLLPatchFidelityLayer{T<:AbstractFloat}
    label::Symbol
    absolute_levels::Vector{T}
    observed_manifold_dimension::Int
    spacing::T
    ground_baseline::T
    gibbs_residual::Union{Nothing, T}
    eigen_residuals::Vector{T}
end

function DLLPatchFidelityLayer(
    label::Symbol,
    absolute_levels::AbstractVector{<:Real},
    observed_manifold_dimension::Integer;
    gibbs_residual::Union{Nothing, Real}=nothing,
    eigen_residuals::AbstractVector{<:Real}=Float64[],
)
    isempty(absolute_levels) && throw(ArgumentError(
        "a fidelity layer requires at least two absolute levels."))
    residual_type = eltype(eigen_residuals)
    isconcretetype(residual_type) || throw(ArgumentError(
        "eigen_residuals must have a concrete real element type."))
    T = gibbs_residual === nothing ?
        promote_type(typeof(float(first(absolute_levels))),
                     typeof(float(zero(residual_type)))) :
        promote_type(typeof(float(first(absolute_levels))),
                     typeof(float(zero(residual_type))),
                     typeof(float(gibbs_residual)))
    levels = T.(absolute_levels)
    issorted(levels) || throw(ArgumentError(
        "fidelity-layer absolute levels must be sorted."))
    manifold = Int(observed_manifold_dimension)
    1 <= manifold < length(levels) || throw(ArgumentError(
        "observed_manifold_dimension must index an available adjacent spacing."))
    residuals = T.(eigen_residuals)
    isempty(residuals) || length(residuals) == length(levels) ||
        throw(DimensionMismatch(
            "eigen residuals must be empty or match the number of levels."))
    all(value -> isfinite(value) && value >= zero(T), residuals) ||
        throw(ArgumentError("eigen residuals must be finite and nonnegative."))
    gibbs = gibbs_residual === nothing ? nothing : T(gibbs_residual)
    gibbs === nothing || (isfinite(gibbs) && gibbs >= zero(T)) ||
        throw(ArgumentError(
            "Gibbs residual must be finite and nonnegative."))
    spacing = levels[manifold + 1] - levels[manifold]
    spacing > zero(T) || throw(ArgumentError(
        "the selected fidelity-layer spacing must be positive."))
    return DLLPatchFidelityLayer{T}(
        label, levels, manifold, spacing, first(levels), gibbs, residuals)
end

"""Pairwise discrepancy between two independently retained fidelity layers."""
struct DLLPatchLayerDiscrepancy{T<:AbstractFloat}
    left_label::Symbol
    right_label::Symbol
    absolute_spacing_error::T
    relative_spacing_error::T
    absolute_baseline_difference::T
    maximum_common_level_error::T
    slow_subspace_projector_distance::Union{Nothing, T}
end

"""
    DLLPatchFidelityCell

Layered exact-global to Task-10 comparison. The three adjacent discrepancies
are retained separately; `pairwise_discrepancies` additionally contains every
available pair, so cancellations cannot be reported as one aggregate error.
"""
struct DLLPatchFidelityCell{
    T<:AbstractFloat,
    R<:QuantumFurnace.DLLParentProvenance,
}
    model_seed::Int
    num_sites::Int
    beta_phys::T
    beta_alg::T
    radius::Int
    maximum_patch_fraction::T
    geometry_tag::Symbol
    global_reference_source::Symbol
    boundary::Symbol
    site_ordering::Symbol
    provenance::R
    global_layer::DLLPatchFidelityLayer{T}
    exact_patch_layer::DLLPatchFidelityLayer{T}
    represented_layer::DLLPatchFidelityLayer{T}
    task10_layer::Union{Nothing, DLLPatchFidelityLayer{T}}
    locality_discrepancy::DLLPatchLayerDiscrepancy{T}
    representation_discrepancy::DLLPatchLayerDiscrepancy{T}
    eigensolver_discrepancy::Union{Nothing, DLLPatchLayerDiscrepancy{T}}
    pairwise_discrepancies::Vector{DLLPatchLayerDiscrepancy{T}}
    successive_radius_change::Union{Nothing, T}
    exact_patch_maximum_bond::Int
    exact_patch_cap_saturated::Bool
    represented_maximum_bond::Int
    represented_cap_saturated::Bool
    task10_bond_dimensions::Vector{Int}
    task10_residuals::Vector{T}
end

function _fidelity_provenance_signature(
    provenance::QuantumFurnace.DLLParentProvenance,
)
    filters = Tuple(
        (
            filter.family,
            filter.support_radius,
            filter.shift,
            filter.weight,
            filter.coordinate_frame,
        )
        for filter in provenance.filters
    )
    clock_signature(clock) = clock === nothing ? nothing :
        (clock.label, clock.multiplier, clock.rate)
    return (
        provenance.beta_phys,
        provenance.beta_alg,
        provenance.hamiltonian_frame,
        provenance.rescaling_factor,
        provenance.energy_shift,
        filters,
        provenance.raw_generator_rate,
        provenance.proposal_normalisation,
        provenance.proposal_amplitude,
        clock_signature(provenance.sweep_clock),
        clock_signature(provenance.legacy_clock),
    )
end

function _validate_fidelity_provenances(
    reference::QuantumFurnace.DLLParentProvenance,
    others::QuantumFurnace.DLLParentProvenance...,
)
    signature = _fidelity_provenance_signature(reference)
    for (index, provenance) in pairs(others)
        isequal(signature, _fidelity_provenance_signature(provenance)) ||
            throw(ArgumentError(
                "fidelity layer $(index + 1) disagrees in frame, beta, " *
                "filter, source amplitude, or raw-clock provenance."))
    end
    return nothing
end

function _slow_projector_distance(
    left::AbstractMatrix,
    right::AbstractMatrix,
    ::Type{T},
) where {T<:AbstractFloat}
    size(left, 1) == size(right, 1) || throw(DimensionMismatch(
        "slow-subspace bases must live in the same ambient space."))
    size(left, 2) == size(right, 2) || throw(DimensionMismatch(
        "slow-subspace bases must have equal dimensions."))
    left_q = Matrix(qr(Matrix{Complex{T}}(left)).Q[:, 1:size(left, 2)])
    right_q = Matrix(qr(Matrix{Complex{T}}(right)).Q[:, 1:size(right, 2)])
    left_projector = left_q * adjoint(left_q)
    right_projector = right_q * adjoint(right_q)
    return T(opnorm(left_projector - right_projector))
end

function _layer_discrepancy(
    left::DLLPatchFidelityLayer{T},
    right::DLLPatchFidelityLayer{T},
    slow_subspaces::AbstractDict{Symbol, <:AbstractMatrix},
) where {T<:AbstractFloat}
    common_count = min(length(left.absolute_levels), length(right.absolute_levels))
    maximum_level_error = maximum(abs.(
        @view(left.absolute_levels[1:common_count]) .-
        @view(right.absolute_levels[1:common_count])))
    projector_distance = if haskey(slow_subspaces, left.label) &&
                            haskey(slow_subspaces, right.label)
        _slow_projector_distance(
            slow_subspaces[left.label], slow_subspaces[right.label], T)
    else
        nothing
    end
    absolute_spacing_error = abs(right.spacing - left.spacing)
    return DLLPatchLayerDiscrepancy{T}(
        left.label,
        right.label,
        absolute_spacing_error,
        absolute_spacing_error / max(abs(left.spacing), eps(T)),
        abs(right.ground_baseline - left.ground_baseline),
        maximum_level_error,
        projector_distance,
    )
end

function _convert_fidelity_layer(
    layer::DLLPatchFidelityLayer,
    ::Type{T},
) where {T<:AbstractFloat}
    return DLLPatchFidelityLayer{T}(
        layer.label,
        T.(layer.absolute_levels),
        layer.observed_manifold_dimension,
        T(layer.spacing),
        T(layer.ground_baseline),
        layer.gibbs_residual === nothing ? nothing : T(layer.gibbs_residual),
        T.(layer.eigen_residuals),
    )
end

function _fidelity_geometry_tag(
    num_sites::Int,
    radius::Int,
    maximum_patch_fraction::T,
) where {T<:AbstractFloat}
    radius >= num_sites - 1 && return :full_patch
    isapprox(maximum_patch_fraction, one(T); atol=eps(T), rtol=eps(T)) &&
        return :mixed_full_window
    maximum_patch_fraction > T(0.75) && return :near_full_patch
    return :qualifying_local
end

function DLLPatchFidelityCell(;
    model_seed::Integer,
    num_sites::Integer,
    radius::Integer,
    maximum_patch_fraction::Real,
    global_reference_source::Symbol,
    global_provenance::QuantumFurnace.DLLParentProvenance,
    exact_patch_provenance::QuantumFurnace.DLLParentProvenance,
    represented_provenance::QuantumFurnace.DLLParentProvenance,
    task10_provenance::Union{Nothing, QuantumFurnace.DLLParentProvenance}=nothing,
    global_layer::DLLPatchFidelityLayer,
    exact_patch_layer::DLLPatchFidelityLayer,
    represented_layer::DLLPatchFidelityLayer,
    task10_layer::Union{Nothing, DLLPatchFidelityLayer}=nothing,
    slow_subspaces::AbstractDict{Symbol, <:AbstractMatrix}=
        Dict{Symbol, Matrix{ComplexF64}}(),
    successive_radius_change::Union{Nothing, Real}=nothing,
    exact_patch_maximum_bond::Integer,
    exact_patch_cap_saturated::Bool,
    represented_maximum_bond::Integer,
    represented_cap_saturated::Bool,
    task10_bond_dimensions::AbstractVector{<:Integer}=Int[],
    task10_residuals::AbstractVector{<:Real}=Float64[],
    boundary::Symbol=:open,
    site_ordering::Symbol=:qf_fused_ket_fast,
)
    n = Int(num_sites)
    n > 0 || throw(ArgumentError("num_sites must be positive."))
    r = Int(radius)
    r >= 0 || throw(ArgumentError("radius must be nonnegative."))
    boundary == :open || throw(ArgumentError(
        "the first global-fidelity campaign supports OBC only."))
    site_ordering == :qf_fused_ket_fast || throw(ArgumentError(
        "global-fidelity comparisons require QF fused ket-fast ordering."))
    (task10_layer === nothing) == (task10_provenance === nothing) ||
        throw(ArgumentError(
            "task10_layer and task10_provenance must be supplied together."))
    provenances = task10_provenance === nothing ?
        (exact_patch_provenance, represented_provenance) :
        (exact_patch_provenance, represented_provenance, task10_provenance)
    _validate_fidelity_provenances(global_provenance, provenances...)

    T = promote_type(
        typeof(float(maximum_patch_fraction)),
        typeof(global_provenance.beta_phys),
        typeof(global_layer.spacing),
        typeof(exact_patch_layer.spacing),
        typeof(represented_layer.spacing),
        task10_layer === nothing ? Float64 : typeof(task10_layer.spacing),
        typeof(float(zero(eltype(task10_residuals)))),
    )
    fraction = T(maximum_patch_fraction)
    zero(T) < fraction <= one(T) || throw(ArgumentError(
        "maximum_patch_fraction must lie in (0, 1]."))
    geometry_tag = _fidelity_geometry_tag(n, r, fraction)
    global_result = _convert_fidelity_layer(global_layer, T)
    exact = _convert_fidelity_layer(exact_patch_layer, T)
    represented = _convert_fidelity_layer(represented_layer, T)
    task = task10_layer === nothing ? nothing :
        _convert_fidelity_layer(task10_layer, T)
    required_labels = (
        global_result.label == :exact_global,
        exact.label == :exact_patch,
        represented.label == :represented_mpo,
    )
    all(required_labels) || throw(ArgumentError(
        "required fidelity-layer labels are :exact_global, :exact_patch, " *
        "and :represented_mpo."))
    task === nothing || task.label == :task10 || throw(ArgumentError(
        "the optional eigensolver layer must use label=:task10."))

    layers = task === nothing ? [global_result, exact, represented] :
        [global_result, exact, represented, task]
    discrepancies = DLLPatchLayerDiscrepancy{T}[]
    for left_index in 1:(length(layers) - 1)
        for right_index in (left_index + 1):length(layers)
            push!(discrepancies, _layer_discrepancy(
                layers[left_index], layers[right_index], slow_subspaces))
        end
    end
    find_pair(left, right) = only(filter(discrepancies) do discrepancy
        discrepancy.left_label == left && discrepancy.right_label == right
    end)
    locality = find_pair(:exact_global, :exact_patch)
    representation = find_pair(:exact_patch, :represented_mpo)
    eigensolver = task === nothing ? nothing :
        find_pair(:represented_mpo, :task10)
    radius_change = successive_radius_change === nothing ? nothing :
        T(successive_radius_change)
    radius_change === nothing ||
        (isfinite(radius_change) && radius_change >= zero(T)) ||
        throw(ArgumentError(
            "successive_radius_change must be finite and nonnegative."))
    residuals = T.(task10_residuals)
    all(value -> isfinite(value) && value >= zero(T), residuals) ||
        throw(ArgumentError(
            "Task-10 residuals must be finite and nonnegative."))
    exact_patch_maximum_bond > 0 || throw(ArgumentError(
        "exact_patch_maximum_bond must be positive."))
    represented_maximum_bond > 0 || throw(ArgumentError(
        "represented_maximum_bond must be positive."))
    all(>(0), task10_bond_dimensions) || throw(ArgumentError(
        "Task-10 bond dimensions must be positive."))
    return DLLPatchFidelityCell{
        T, typeof(global_provenance),
    }(
        Int(model_seed),
        n,
        T(global_provenance.beta_phys),
        T(global_provenance.beta_alg),
        r,
        fraction,
        geometry_tag,
        global_reference_source,
        boundary,
        site_ordering,
        global_provenance,
        global_result,
        exact,
        represented,
        task,
        locality,
        representation,
        eigensolver,
        discrepancies,
        radius_change,
        Int(exact_patch_maximum_bond),
        exact_patch_cap_saturated,
        Int(represented_maximum_bond),
        represented_cap_saturated,
        Int.(task10_bond_dimensions),
        residuals,
    )
end
