"""
    FinitePatchParentVerification

Optional dense-overlap diagnostics for a represented finite-patch parent.
Every Gibbs quantity is explicitly global: finite patches generally do not
share a common Gibbs zero mode, even when each exact isolated patch block is
positive and frustration free with respect to its own patch purification.
The observed kernel count belongs only to the represented finite-patch MPO.
"""
struct FinitePatchParentVerification{
    T<:AbstractFloat,
    A<:ParentMPOAssemblyDiagnostics{T},
}
    target_label::Symbol
    assembly::A
    dense_overlap_used::Bool
    block_sum_norm_difference::Union{Nothing, T}
    block_minimum_eigenvalues::Union{Nothing, Vector{T}}
    total_minimum_eigenvalue::Union{Nothing, T}
    observed_kernel_count::Union{Nothing, Int}
    kernel_tolerance::Union{Nothing, T}
    global_gibbs_energy::Union{Nothing, T}
    global_gibbs_residual::Union{Nothing, T}
    global_block_gibbs_residuals::Union{Nothing, Vector{T}}
end

function _dense_parent_verification(
    parent::FinitePatchDLLParent{T};
    gibbs_vector::Union{Nothing, AbstractVector}=nothing,
    kernel_tolerance::Union{Nothing, Real}=nothing,
) where {T<:AbstractFloat}
    _validate_fused_sites(parent.sites)
    total = Matrix{Complex{T}}(mpo_to_dense(
        parent.bundle.total_parent, parent.sites))
    blocks = Matrix{Complex{T}}[
        mpo_to_dense(block, parent.sites)
        for block in parent.bundle.block_parents
    ]
    block_sum_difference = T(norm(sum(blocks) - total))
    hermitian_total = Hermitian((total + adjoint(total)) / T(2))
    total_values = eigvals(hermitian_total)
    scale = max(T(norm(total)), one(T))
    tolerance = kernel_tolerance === nothing ?
        max(T(512) * T(size(total, 1)) * eps(T) * scale, T(1e-10)) :
        T(kernel_tolerance)
    QuantumFurnace._require_finite_nonnegative(
        tolerance, "kernel_tolerance")
    block_minimum = T[
        minimum(eigvals(Hermitian((block + adjoint(block)) / T(2))))
        for block in blocks
    ]

    gibbs_energy = nothing
    gibbs_residual = nothing
    block_gibbs_residuals = nothing
    if gibbs_vector !== nothing
        length(gibbs_vector) == size(total, 1) || throw(DimensionMismatch(
            "global Gibbs vector has length $(length(gibbs_vector)), expected " *
            "$(size(total, 1))."))
        vector = Vector{Complex{T}}(gibbs_vector)
        all(isfinite, vector) || throw(ArgumentError(
            "global Gibbs vector must contain only finite values."))
        vector_norm = T(norm(vector))
        isapprox(vector_norm, one(T); atol=T(100) * eps(T), rtol=T(100) * eps(T)) ||
            throw(ArgumentError(
                "global Gibbs vector must be normalized; got norm $vector_norm."))
        total_image = total * vector
        gibbs_energy = T(real(dot(vector, total_image)))
        gibbs_residual = T(norm(total_image))
        block_gibbs_residuals = T[norm(block * vector) for block in blocks]
    end
    return FinitePatchParentVerification{
        T, typeof(parent.assembly_diagnostics)}(
        :finite_patch_bohr_surrogate,
        parent.assembly_diagnostics,
        true,
        block_sum_difference,
        block_minimum,
        T(first(total_values)),
        count(value -> abs(value) <= tolerance, total_values),
        tolerance,
        gibbs_energy,
        gibbs_residual,
        block_gibbs_residuals,
    )
end

"""
    verify_dll_parent(parent::FinitePatchDLLParent; ...)

Return representation diagnostics for a finite-patch parent. With
`dense_overlap=false` (the default), this is a non-dense operation and Gibbs,
kernel, and minimum-energy fields remain unavailable. With
`dense_overlap=true`, the guarded `N <= 4` dense bridge supplies complete
diagnostics for the represented surrogate only. An optional normalized
`gibbs_vector` must already use the parent's fused ordering.
"""
function QuantumFurnace.verify_dll_parent(
    parent::FinitePatchDLLParent{T};
    dense_overlap::Bool=false,
    gibbs_vector::Union{Nothing, AbstractVector}=nothing,
    kernel_tolerance::Union{Nothing, Real}=nothing,
) where {T<:AbstractFloat}
    if !dense_overlap
        gibbs_vector === nothing || throw(ArgumentError(
            "global Gibbs residuals currently require dense_overlap=true; " *
            "Task 9 will add an independently prepared scalable Gibbs MPS."))
        kernel_tolerance === nothing || throw(ArgumentError(
            "kernel_tolerance is meaningful only with dense_overlap=true."))
        return FinitePatchParentVerification{
            T, typeof(parent.assembly_diagnostics)}(
            :finite_patch_bohr_surrogate,
            parent.assembly_diagnostics,
            false,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
        )
    end
    return _dense_parent_verification(
        parent; gibbs_vector, kernel_tolerance)
end

"""
    GibbsParentDiagnostics

Post-preparation diagnostics of an independent Gibbs MPS against a represented
parent. Energies and residuals refer to the represented operator only. In
particular, a finite-patch parent need not annihilate the global Gibbs state.
Expectations remain complex so an anti-Hermitian representation defect is not
silently discarded.

A small residual alone proves neither uniqueness nor a positive gap. For an
unsymmetrized or otherwise nonnormal surrogate, it also does not by itself
bound distance to a spectral subspace. Such conclusions require the relevant
Hermiticity/normality, complete-kernel, and spectral-separation evidence.
"""
struct GibbsParentDiagnostics{T<:AbstractFloat}
    target_label::Symbol
    total_energy::Complex{T}
    total_residual::T
    block_energies::Vector{Complex{T}}
    block_residuals::Vector{T}
    tighter_target_label::Union{Nothing, Symbol}
    tighter_total_energy::Union{Nothing, Complex{T}}
    tighter_total_residual::Union{Nothing, T}
end

"""
    ParentLowEnergyAttempt

One recorded Task-10 optimization attempt. `energy` and `residual` always
refer to the original unpenalised vector of parent-block MPOs. The tighter
quantities use the tighter block vector and its own Rayleigh quotient. The
penalised DMRG objective value is deliberately not reported as a level.
"""
struct ParentLowEnergyAttempt{T<:AbstractFloat}
    stage::Symbol
    level_index::Int
    bond_dimension::Int
    start_index::Int
    start_kind::Symbol
    sector::Symbol
    penalty_factor::Union{Nothing, T}
    penalty_weight::Union{Nothing, T}
    energy::Union{Nothing, T}
    residual::Union{Nothing, T}
    tighter_energy::Union{Nothing, T}
    tighter_residual::Union{Nothing, T}
    maximum_prior_overlap::Union{Nothing, T}
    accepted::Bool
    rejection_reason::Symbol
    failure_message::Union{Nothing, String}
end

"""
    FinitePatchLowEnergyDiagnostics

Convergence record returned as `DLLTensorNetworkResult.metadata` by the
finite-patch Task-10 solver. `absolute_levels` contains one canonical state per
level; repeated starts and penalty scans remain in `attempts` and are never
counted as additional levels. `observed_manifold_spacing` is an adjacent
absolute-level difference, not a shifted parent energy or a certified gap.
"""
struct FinitePatchLowEnergyDiagnostics{T<:AbstractFloat}
    absolute_levels::Vector{T}
    observed_manifold_dimension::Int
    observed_kernel_dimension::Int
    observed_manifold_spacing::T
    zero_mode_tolerance::T
    requested_bond_dimensions::Vector{Int}
    maximum_exact_bond_dimension::Int
    bond_dimensions_scanned::Vector{Int}
    converged_bond_dimensions::Vector{Int}
    absolute_levels_by_bond::Vector{Vector{T}}
    bond_convergence_spreads::Vector{T}
    multi_start_spreads::Vector{T}
    penalty_scales::Vector{T}
    tighter_energies::Vector{T}
    tighter_residuals::Vector{T}
    unrestricted_all_sectors::Bool
    dense_overlap_levels::Union{Nothing, Vector{T}}
    dense_overlap_maximum_error::Union{Nothing, T}
    attempts::Vector{ParentLowEnergyAttempt{T}}
end

@inline _parent_bundle(parent::FinitePatchDLLParent) = parent.bundle
@inline _parent_bundle(parent::ExactDLLParentReference) = parent.bundle
@inline _parent_sites(parent::FinitePatchDLLParent) = parent.sites
@inline _parent_sites(parent::ExactDLLParentReference) = parent.sites

function _gibbs_mpo_energy_residual(
    state::ITensorMPS.MPS,
    operator::ITensorMPS.MPO,
    ::Type{T},
) where {T<:AbstractFloat}
    energy = Complex{T}(ITensorMPS.inner(state', operator, state))
    image = _apply_without_requested_truncation(operator, state)
    residual = T(norm(image))
    isfinite(energy) || error("parent expectation became nonfinite.")
    isfinite(residual) || error("parent residual became nonfinite.")
    return energy, residual
end

function _validate_gibbs_parent_sites(
    preparation::GibbsPurificationMPS,
    parent,
    label::AbstractString,
)
    _parent_sites(parent) == preparation.sites || throw(ArgumentError(
        "$label must use the same fused site indices as the Gibbs " *
        "preparation."))
    return nothing
end

function _validate_tighter_parent(
    parent::FinitePatchDLLParent,
    tighter_parent::FinitePatchDLLParent,
    ;
    require_strict_controls::Bool=true,
)
    provenance = parent.provenance
    tighter_provenance = tighter_parent.provenance
    same_provenance =
        provenance.beta_phys == tighter_provenance.beta_phys &&
        provenance.beta_alg == tighter_provenance.beta_alg &&
        provenance.hamiltonian_frame == tighter_provenance.hamiltonian_frame &&
        provenance.rescaling_factor == tighter_provenance.rescaling_factor &&
        provenance.energy_shift == tighter_provenance.energy_shift &&
        provenance.filters == tighter_provenance.filters &&
        provenance.raw_generator_rate ==
            tighter_provenance.raw_generator_rate &&
        provenance.proposal_normalisation ==
            tighter_provenance.proposal_normalisation &&
        provenance.proposal_amplitude ==
            tighter_provenance.proposal_amplitude &&
        isequal(provenance.sweep_clock, tighter_provenance.sweep_clock) &&
        isequal(provenance.legacy_clock, tighter_provenance.legacy_clock)
    same_provenance || throw(ArgumentError(
        "tighter_parent must preserve all frame, beta, filter, and rate " *
        "provenance."))
    isequal(parent.model_specification,
            tighter_parent.model_specification) || throw(ArgumentError(
        "tighter_parent must use the same local Hamiltonian and DLL source " *
        "specification as parent."))
    parent.assembly_diagnostics.symmetrized ==
        tighter_parent.assembly_diagnostics.symmetrized ||
        throw(ArgumentError(
            "tighter_parent must preserve the parent's symmetrisation choice."))

    controls = parent.controls
    tighter = tighter_parent.controls
    controls.patch_radius == tighter.patch_radius || throw(ArgumentError(
        "tighter_parent must preserve the finite-patch radius."))
    scalar_no_worse = tighter.scalar_tolerance <= controls.scalar_tolerance
    cutoffs_no_worse =
        tighter.recurrence_cutoff <= controls.recurrence_cutoff &&
        tighter.product_cutoff <= controls.product_cutoff &&
        tighter.sum_cutoff <= controls.sum_cutoff
    caps_no_worse =
        tighter.recurrence_maxdim >= controls.recurrence_maxdim &&
        tighter.product_maxdim >= controls.product_maxdim &&
        tighter.sum_maxdim >= controls.sum_maxdim
    any_strict = tighter.scalar_tolerance < controls.scalar_tolerance ||
                 tighter.recurrence_cutoff < controls.recurrence_cutoff ||
                 tighter.product_cutoff < controls.product_cutoff ||
                 tighter.sum_cutoff < controls.sum_cutoff ||
                 tighter.recurrence_maxdim > controls.recurrence_maxdim ||
                 tighter.product_maxdim > controls.product_maxdim ||
                 tighter.sum_maxdim > controls.sum_maxdim
    scalar_no_worse && cutoffs_no_worse && caps_no_worse &&
        (!require_strict_controls || any_strict) ||
        throw(ArgumentError(
            "tighter_parent controls must be no looser in every scalar and " *
            "MPO layer" * (require_strict_controls ?
            ", with at least one control tightened." : ".")))
    return nothing
end

"""
    gibbs_parent_diagnostics(preparation, parent; tighter_parent=nothing)

Measure total and per-block `K|TFD>` residuals and energies without modifying
the independently prepared Gibbs MPS or either parent. An optional tighter
parent must represent the same labelled target and block accounting on the
same fused indices. These diagnostics are not used to establish Gibbs-MPS
correctness. A small residual is not evidence of a unique fixed point, a gap,
or spectral-subspace proximity for an unsymmetrized/nonnormal surrogate.
"""
function _gibbs_parent_diagnostics(
    preparation::GibbsPurificationMPS{T},
    parent::Union{FinitePatchDLLParent, ExactDLLParentReference};
    tighter_parent::Union{
        Nothing, FinitePatchDLLParent, ExactDLLParentReference,
    }=nothing,
    require_strict_tighter_controls::Bool=true,
) where {T<:AbstractFloat}
    _validate_gibbs_parent_sites(preparation, parent, "parent")
    bundle = _parent_bundle(parent)
    total_energy, total_residual = _gibbs_mpo_energy_residual(
        preparation.state, bundle.total_parent, T)
    block_energies = Complex{T}[]
    block_residuals = T[]
    sizehint!(block_energies, length(bundle.block_parents))
    sizehint!(block_residuals, length(bundle.block_parents))
    for block in bundle.block_parents
        energy, residual = _gibbs_mpo_energy_residual(
            preparation.state, block, T)
        push!(block_energies, energy)
        push!(block_residuals, residual)
    end

    tighter_label = nothing
    tighter_energy = nothing
    tighter_residual = nothing
    if tighter_parent !== nothing
        parent isa FinitePatchDLLParent &&
            tighter_parent isa FinitePatchDLLParent || throw(ArgumentError(
                "tighter_parent diagnostics currently compare only two " *
                "finite-patch parents; exact and finite-patch targets must " *
                "remain separate."))
        _validate_gibbs_parent_sites(
            preparation, tighter_parent, "tighter_parent")
        tighter_bundle = _parent_bundle(tighter_parent)
        tighter_bundle.target_label == bundle.target_label ||
            throw(ArgumentError(
                "tighter_parent must use the same target label as parent."))
        tighter_bundle.block_keys == bundle.block_keys || throw(ArgumentError(
            "tighter_parent must preserve parent block accounting."))
        _validate_tighter_parent(
            parent,
            tighter_parent;
            require_strict_controls=require_strict_tighter_controls,
        )
        tighter_energy, tighter_residual = _gibbs_mpo_energy_residual(
            preparation.state, tighter_bundle.total_parent, T)
        tighter_label = tighter_bundle.target_label
    end
    return GibbsParentDiagnostics{T}(
        bundle.target_label,
        total_energy,
        total_residual,
        block_energies,
        block_residuals,
        tighter_label,
        tighter_energy,
        tighter_residual,
    )
end

function gibbs_parent_diagnostics(
    preparation::GibbsPurificationMPS{T},
    parent::Union{FinitePatchDLLParent, ExactDLLParentReference};
    tighter_parent::Union{
        Nothing, FinitePatchDLLParent, ExactDLLParentReference,
    }=nothing,
) where {T<:AbstractFloat}
    return _gibbs_parent_diagnostics(
        preparation,
        parent;
        tighter_parent,
        require_strict_tighter_controls=true,
    )
end
