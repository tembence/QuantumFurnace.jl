"""
    ParentMPOAssemblyDiagnostics

Representation diagnostics recorded while assembling a finite-patch DLL
parent. Hermiticity defects are measured before optional symmetrisation.
Compression magnitudes are floating-point Frobenius-norm evidence, while the
sampled action defects compare the retained block sum with the separately
compressed total MPO on deterministic unit product states.

These diagnostics do not assert that different finite patches share a common
zero mode. In particular, an exact finite-patch sum can be strictly positive
even though every isolated patch block is positive and frustration free with
respect to its own patch Gibbs purification.
"""
struct ParentMPOAssemblyDiagnostics{T<:AbstractFloat}
    presymmetrization_block_hermiticity_defects::Vector{T}
    presymmetrization_total_hermiticity_defect::T
    block_assembly_compression_bounds::Vector{T}
    total_sum_compression_bound::T
    block_sum_action_defects::Vector{T}
    diagnostic_probe_seed::Int
    symmetrized::Bool
    energy_shift_applied::Bool
    psd_projection_applied::Bool
end

"""
    FinitePatchDLLParent

Complete Gaussian `:finite_patch_bohr_surrogate` assembled on a common fused
doubled chain. `bohr_blocks` retains the compact `Q`, `L`, and signed `N` MPOs
and their per-patch provenance. `bundle` retains every separately assembled
source/channel parent block as well as the compressed total parent.
`model_specification` retains the owned local Hamiltonian and source/channel
data so later convergence comparisons can reject a purported tighter parent
that represents a different mathematical input.

The object is a finite-patch surrogate, not an exact global DLL parent. Its
total operator is not assumed to have a zero mode or to annihilate the global
Gibbs purification.
"""
struct FinitePatchDLLParent{
    T<:AbstractFloat,
    B,
    P<:QuantumFurnace.DLLParentBundle,
    R<:QuantumFurnace.DLLParentProvenance,
    M,
}
    bundle::P
    sites::Vector{ITensors.Index{Int}}
    bohr_blocks::Vector{B}
    provenance::R
    model_specification::M
    controls::QuantumFurnace.BohrMPOControls{T}
    assembly_records::Vector{MPOCompressionRecord{T}}
    assembly_diagnostics::ParentMPOAssemblyDiagnostics{T}
end

function _validate_parent_fused_sites(
    sites,
    num_sites::Int,
    physical_dim::Int,
)
    collected = collect(sites)
    length(collected) == num_sites || throw(DimensionMismatch(
        "received $(length(collected)) fused sites, expected $num_sites."))
    fused_dim = Base.checked_mul(physical_dim, physical_dim)
    all(site -> ITensors.dim(site) == fused_dim, collected) ||
        throw(DimensionMismatch(
            "every fused parent site must have dimension $fused_dim."))
    all(site -> !ITensors.hasqns(site), collected) || throw(ArgumentError(
        "the finite-patch DLL parent currently requires unconstrained fused sites."))
    return Vector{ITensors.Index{Int}}(collected)
end

@inline function _typed_identity_tensor(
    ::Type{Complex{T}},
    site::ITensors.Index,
) where {T<:AbstractFloat}
    d = ITensors.dim(site)
    return ITensors.ITensor(
        Matrix{Complex{T}}(I, d, d), ITensors.prime(site), site)
end

function _fresh_embedded_links(
    operator::ITensorMPS.MPO,
    first_site::Int,
    last_site::Int,
    num_sites::Int,
    tag::Symbol,
)
    length(operator) == last_site - first_site + 1 || throw(DimensionMismatch(
        "compact MPO length $(length(operator)) does not match patch " *
        "$first_site:$last_site."))
    links = ITensors.Index{Int}[]
    sizehint!(links, max(num_sites - 1, 0))
    for bond in 1:(num_sites - 1)
        dimension = if first_site <= bond < last_site
            ITensors.dim(ITensorMPS.linkind(operator, bond - first_site + 1))
        else
            1
        end
        push!(links, ITensors.Index(dimension, "Link,$tag,bond=$bond"))
    end
    return links
end

function _transformed_active_tensor(
    tensor::ITensors.ITensor,
    old_site::ITensors.Index,
    active_site::ITensors.Index,
    transform::Symbol,
)
    if transform == :identity
        return ITensors.replaceinds(
            tensor,
            old_site => active_site,
            ITensors.prime(old_site) => ITensors.prime(active_site),
        )
    elseif transform == :conjugate
        mapped = ITensors.replaceinds(
            tensor,
            old_site => active_site,
            ITensors.prime(old_site) => ITensors.prime(active_site),
        )
        return conj(mapped)
    elseif transform == :transpose
        return ITensors.replaceinds(
            tensor,
            old_site => ITensors.prime(active_site),
            ITensors.prime(old_site) => active_site,
        )
    end
    throw(ArgumentError(
        "physical MPO transform must be :identity, :conjugate, or :transpose."))
end

"""
Lift one compact physical MPO into a common global fused chain. The local
combiner order is `(ket, bra)`, so the ket digit is fastest and the result
matches the extension's `ket + d * bra` convention.
"""
function _lift_physical_mpo_to_fused(
    operator::ITensorMPS.MPO,
    operator_sites,
    fused_sites::Vector{ITensors.Index{Int}},
    first_site::Int,
    last_site::Int;
    active_leg::Symbol,
    transform::Symbol=:identity,
    tag::Symbol=:physical_lift,
)
    active_leg in (:ket, :bra) || throw(ArgumentError(
        "active_leg must be :ket or :bra."))
    length(operator_sites) == length(operator) || throw(DimensionMismatch(
        "operator site count must match MPO length."))
    num_sites = length(fused_sites)
    1 <= first_site <= last_site <= num_sites || throw(ArgumentError(
        "invalid compact patch $first_site:$last_site for $num_sites sites."))
    physical_dim = isqrt(ITensors.dim(first(fused_sites)))
    physical_dim^2 == ITensors.dim(first(fused_sites)) || error(
        "validated fused-site dimension is not a perfect square.")
    all(site -> ITensors.dim(site) == physical_dim, operator_sites) ||
        throw(DimensionMismatch(
            "compact physical MPO sites must all have dimension $physical_dim."))

    global_links = _fresh_embedded_links(
        operator, first_site, last_site, num_sites, tag)
    tensors = ITensors.ITensor[]
    sizehint!(tensors, num_sites)
    # `eltype(::ITensor)` is concrete for the supported Float32/Float64 MPOs.
    CT = eltype(first(operator))
    for global_site in 1:num_sites
        ket_site = ITensors.Index(
            physical_dim, "QFParentKet,site=$global_site")
        bra_site = ITensors.Index(
            physical_dim, "QFParentBra,site=$global_site")
        tensor = if first_site <= global_site <= last_site
            local_site = global_site - first_site + 1
            active_site = active_leg == :ket ? ket_site : bra_site
            inactive_site = active_leg == :ket ? bra_site : ket_site
            mapped = _transformed_active_tensor(
                operator[local_site], operator_sites[local_site],
                active_site, transform)
            replacements = Pair{ITensors.Index{Int}, ITensors.Index{Int}}[]
            if local_site > 1
                push!(
                    replacements,
                    ITensorMPS.linkind(operator, local_site - 1) =>
                        global_links[global_site - 1],
                )
            end
            if local_site < length(operator)
                push!(
                    replacements,
                    ITensorMPS.linkind(operator, local_site) =>
                        global_links[global_site],
                )
            end
            !isempty(replacements) &&
                (mapped = ITensors.replaceinds(mapped, replacements))
            mapped * _typed_identity_tensor(CT, inactive_site)
        else
            _typed_identity_tensor(CT, ket_site) *
                _typed_identity_tensor(CT, bra_site)
        end

        scalar = one(CT)
        if global_site > 1 &&
           !ITensors.hasind(tensor, global_links[global_site - 1])
            tensor *= ITensors.ITensor(
                scalar, global_links[global_site - 1])
        end
        if global_site < num_sites &&
           !ITensors.hasind(tensor, global_links[global_site])
            tensor *= ITensors.ITensor(scalar, global_links[global_site])
        end

        input_combiner = ITensors.combiner(ket_site, bra_site)
        output_combiner = ITensors.combiner(
            ITensors.prime(ket_site), ITensors.prime(bra_site))
        tensor = tensor * input_combiner * output_combiner
        tensor = ITensors.replaceinds(
            tensor,
            ITensors.combinedind(input_combiner) => fused_sites[global_site],
            ITensors.combinedind(output_combiner) =>
                ITensors.prime(fused_sites[global_site]),
        )
        push!(tensors, tensor)
    end
    return ITensorMPS.MPO(tensors)
end

@inline _mpo_adjoint(operator::ITensorMPS.MPO) =
    ITensors.swapprime(ITensors.dag(operator), 0 => 1)

function _network_difference_norm(left, right, ::Type{T}) where {T<:AbstractFloat}
    difference = if length(left) == 1
        tensor = left[1] - right[1]
        left isa ITensorMPS.MPO ?
            ITensorMPS.MPO([tensor]) : ITensorMPS.MPS([tensor])
    else
        ITensorMPS.add(left, -right; alg="directsum")
    end
    if length(difference) > 1
        maximum_bond = maximum(
            ITensors.dim(ITensorMPS.linkind(difference, bond))
            for bond in 1:(length(difference) - 1)
        )
        # Canonicalize the direct-sum difference before contracting its norm.
        # Without this zero-cutoff sweep, two nearly equal networks can suffer
        # sqrt(eps)-level cancellation in a raw overlap contraction.
        ITensorMPS.truncate!(
            difference; cutoff=0.0, maxdim=maximum_bond)
    end
    squared_norm = T(real(ITensorMPS.inner(difference, difference)))
    scale = max(
        T(abs(real(ITensorMPS.inner(left, left)))),
        T(abs(real(ITensorMPS.inner(right, right)))),
        one(T),
    )
    roundoff = T(512) * eps(T) * scale
    squared_norm >= -roundoff || error(
        "tensor-network difference norm became negative beyond roundoff: " *
        "$squared_norm.")
    return sqrt(max(zero(T), squared_norm))
end

function _relative_mpo_hermiticity_defect(
    operator::ITensorMPS.MPO,
    ::Type{T},
) where {T<:AbstractFloat}
    denominator = max(T(norm(operator)), eps(T))
    return _network_difference_norm(operator, _mpo_adjoint(operator), T) /
           denominator
end

function _directsum_candidate(
    left::ITensorMPS.MPO,
    right::ITensorMPS.MPO,
)
    return length(left) == 1 ?
        ITensorMPS.MPO([left[1] + right[1]]) :
        ITensorMPS.add(left, right; alg="directsum")
end

function _measured_directsum_add(
    left::ITensorMPS.MPO,
    right::ITensorMPS.MPO,
    records::Vector{MPOCompressionRecord{T}},
    stage::Symbol,
    order::Int,
    cutoff::T,
    maxdim::Int,
) where {T<:AbstractFloat}
    maximum_input_bond = max(
        _maximum_mpo_bond(left), _maximum_mpo_bond(right))
    candidate = _directsum_candidate(left, right)
    before_truncation = copy(candidate)
    _truncate_and_record!(
        candidate,
        records,
        stage,
        :directsum_add,
        order,
        cutoff,
        maxdim,
        zero(T),
        0,
        maximum_input_bond,
        zero(T),
        :not_applicable,
    )
    return candidate, _network_difference_norm(
        before_truncation, candidate, T)
end

@inline function _parent_stage(
    source_index::Int,
    channel_index::Int,
    suffix::AbstractString,
)
    return Symbol("parent_", source_index, "_", channel_index, "_", suffix)
end

function _build_parent_block_mpo(
    bohr,
    fused_sites::Vector{ITensors.Index{Int}},
    controls::QuantumFurnace.BohrMPOControls,
    records::Vector{MPOCompressionRecord{T}},
    source_index::Int,
    channel_index::Int;
    symmetrize::Bool,
) where {T<:AbstractFloat}
    ket_Q = _lift_physical_mpo_to_fused(
        bohr.Q, bohr.sites, fused_sites, bohr.first_site, bohr.last_site;
        active_leg=:ket,
        transform=:identity,
        tag=_parent_stage(source_index, channel_index, "ket_q"),
    )
    bra_conj_Q = _lift_physical_mpo_to_fused(
        bohr.Q, bohr.sites, fused_sites, bohr.first_site, bohr.last_site;
        active_leg=:bra,
        transform=:conjugate,
        tag=_parent_stage(source_index, channel_index, "bra_q"),
    )
    transition = _zipup_product(
        bra_conj_Q,
        ket_Q,
        records,
        _parent_stage(source_index, channel_index, "q_sandwich"),
        0,
        T(controls.product_cutoff),
        controls.product_maxdim,
    )

    ket_N = _lift_physical_mpo_to_fused(
        bohr.N, bohr.sites, fused_sites, bohr.first_site, bohr.last_site;
        active_leg=:ket,
        transform=:identity,
        tag=_parent_stage(source_index, channel_index, "ket_n"),
    )
    bra_transpose_N = _lift_physical_mpo_to_fused(
        bohr.N, bohr.sites, fused_sites, bohr.first_site, bohr.last_site;
        active_leg=:bra,
        transform=:transpose,
        tag=_parent_stage(source_index, channel_index, "bra_n_transpose"),
    )
    anticommutator, anticommutator_error = _measured_directsum_add(
        ket_N,
        bra_transpose_N,
        records,
        _parent_stage(source_index, channel_index, "n_sum"),
        0,
        T(controls.sum_cutoff),
        controls.sum_maxdim,
    )
    discriminant, discriminant_error = _measured_directsum_add(
        transition,
        (T(1) / T(2)) * anticommutator,
        records,
        _parent_stage(source_index, channel_index, "discriminant_sum"),
        0,
        T(controls.sum_cutoff),
        controls.sum_maxdim,
    )
    # Apply the parent sign K=-D exactly once, after all three D terms exist.
    presymmetrized_parent = -discriminant
    hermiticity_defect = _relative_mpo_hermiticity_defect(
        presymmetrized_parent, T)
    compression_bound = anticommutator_error + discriminant_error

    if !symmetrize
        return presymmetrized_parent, presymmetrized_parent,
               hermiticity_defect, compression_bound
    end
    symmetrized_parent, symmetry_error = _measured_directsum_add(
        presymmetrized_parent,
        _mpo_adjoint(presymmetrized_parent),
        records,
        _parent_stage(source_index, channel_index, "symmetrization"),
        0,
        T(controls.sum_cutoff),
        controls.sum_maxdim,
    )
    symmetrized_parent = (T(1) / T(2)) * symmetrized_parent
    return presymmetrized_parent, symmetrized_parent,
           hermiticity_defect, compression_bound + symmetry_error / T(2)
end

function _assemble_parent_components(
    bohr_blocks::Vector{B},
    block_keys::Vector{Tuple{Int, Int}},
    fused_sites::Vector{ITensors.Index{Int}},
    controls::QuantumFurnace.BohrMPOControls{T};
    symmetrize::Bool,
    diagnostic_probe_count::Int,
    diagnostic_probe_seed::Int,
) where {T<:AbstractFloat, B}
    length(bohr_blocks) == length(block_keys) || throw(DimensionMismatch(
        "Bohr blocks and block keys must have equal length."))
    isempty(bohr_blocks) && throw(ArgumentError(
        "finite-patch parent assembly requires at least one block."))

    assembly_records = MPOCompressionRecord{T}[]
    presymmetrized_blocks = ITensorMPS.MPO[]
    represented_blocks = ITensorMPS.MPO[]
    block_hermiticity_defects = T[]
    block_compression_bounds = T[]
    for (index, bohr) in pairs(bohr_blocks)
        source_index, channel_index = block_keys[index]
        presymmetrized, represented, hermiticity, compression =
            _build_parent_block_mpo(
                bohr,
                fused_sites,
                controls,
                assembly_records,
                source_index,
                channel_index;
                symmetrize,
            )
        push!(presymmetrized_blocks, presymmetrized)
        push!(represented_blocks, represented)
        push!(block_hermiticity_defects, hermiticity)
        push!(block_compression_bounds, compression)
    end

    presymmetrized_total, total_parent, total_compression_bound = if symmetrize
        raw_total, _ = _assemble_parent_sum(
            presymmetrized_blocks,
            controls,
            assembly_records,
            :presymmetrized_total_sum,
        )
        represented_total, represented_bound = _assemble_parent_sum(
            represented_blocks,
            controls,
            assembly_records,
            :represented_total_sum,
        )
        raw_total, represented_total, represented_bound
    else
        represented_total, represented_bound = _assemble_parent_sum(
            represented_blocks,
            controls,
            assembly_records,
            :represented_total_sum,
        )
        represented_total, represented_total, represented_bound
    end
    total_hermiticity_defect = _relative_mpo_hermiticity_defect(
        presymmetrized_total, T)
    action_defects = _sample_block_sum_action_defects(
        total_parent,
        represented_blocks,
        fused_sites,
        T,
        diagnostic_probe_count,
        diagnostic_probe_seed,
    )
    diagnostics = ParentMPOAssemblyDiagnostics{T}(
        block_hermiticity_defects,
        total_hermiticity_defect,
        block_compression_bounds,
        total_compression_bound,
        action_defects,
        diagnostic_probe_seed,
        symmetrize,
        false,
        false,
    )
    return (;
        represented_blocks,
        total_parent,
        assembly_records,
        diagnostics,
        block_compression_bounds,
        total_compression_bound,
    )
end

function _assemble_parent_sum(
    blocks::Vector{ITensorMPS.MPO},
    controls::QuantumFurnace.BohrMPOControls,
    records::Vector{MPOCompressionRecord{T}},
    stage_prefix::Symbol,
) where {T<:AbstractFloat}
    total = copy(first(blocks))
    compression_bound = zero(T)
    for index in 2:length(blocks)
        total, defect = _measured_directsum_add(
            total,
            blocks[index],
            records,
            Symbol(stage_prefix, "_", index),
            index,
            T(controls.sum_cutoff),
            controls.sum_maxdim,
        )
        compression_bound += defect
    end
    return total, compression_bound
end

@inline function _maximum_mps_bond(state::ITensorMPS.MPS)
    length(state) <= 1 && return 1
    return maximum(
        ITensors.dim(ITensorMPS.linkind(state, bond))
        for bond in 1:(length(state) - 1)
    )
end

function _apply_without_requested_truncation(
    operator::ITensorMPS.MPO,
    state::ITensorMPS.MPS,
)
    maxdim = Base.checked_mul(
        _maximum_mpo_bond(operator), _maximum_mps_bond(state))
    return ITensorMPS.apply(
        operator, state; cutoff=0.0, maxdim=maxdim)
end

function _directsum_add_states(
    left::ITensorMPS.MPS,
    right::ITensorMPS.MPS,
)
    return length(left) == 1 ?
        ITensorMPS.MPS([left[1] + right[1]]) :
        ITensorMPS.add(left, right; alg="directsum")
end

function _sample_block_sum_action_defects(
    total::ITensorMPS.MPO,
    blocks::Vector{ITensorMPS.MPO},
    sites::Vector{ITensors.Index{Int}},
    ::Type{T},
    count::Int,
    seed::Int,
) where {T<:AbstractFloat}
    count >= 0 || throw(ArgumentError(
        "diagnostic_probe_count must be nonnegative."))
    defects = T[]
    sizehint!(defects, count)
    for probe_index in 1:count
        rng = Random.MersenneTwister(seed + probe_index - 1)
        probe = ITensorMPS.random_mps(
            rng, Complex{T}, sites; linkdims=1)
        normalize!(probe)
        block_sum = _apply_without_requested_truncation(first(blocks), probe)
        for block in @view blocks[2:end]
            block_sum = _directsum_add_states(
                block_sum,
                _apply_without_requested_truncation(block, probe),
            )
        end
        total_image = _apply_without_requested_truncation(total, probe)
        defect = _network_difference_norm(total_image, block_sum, T) /
                 max(T(norm(block_sum)), eps(T))
        push!(defects, defect)
    end
    return defects
end

function _renamed_error_bound(
    entry::QuantumFurnace.ParentErrorBound{T},
    name::Symbol,
    prefix::AbstractString,
) where {T<:AbstractFloat}
    return QuantumFurnace.ParentErrorBound(
        T,
        name;
        magnitude=entry.magnitude,
        evidence=entry.evidence,
        note=string(prefix, entry.note),
    )
end

function _parent_error_ledger(
    bohr_blocks::Vector{<:DLLGaussianBohrMPO{T}},
    block_keys::Vector{Tuple{Int, Int}},
    block_compression_bounds::Vector{T},
    total_compression_bound::T,
) where {T<:AbstractFloat}
    scalar_bound = maximum(
        block.error_ledger.scalar_polynomial.magnitude for block in bohr_blocks)
    modular_bound = maximum(
        block.error_ledger.modular_compatibility.magnitude for block in bohr_blocks)
    recurrence_entries = QuantumFurnace.ParentErrorBound{T}[]
    product_entries = QuantumFurnace.ParentErrorBound{T}[]
    block_entries = QuantumFurnace.ParentErrorBound{T}[]
    for (index, ((source, channel), bohr)) in enumerate(zip(block_keys, bohr_blocks))
        prefix = "source $source channel $channel: "
        for (entry_index, entry) in enumerate(bohr.error_ledger.mpo_recurrences)
            push!(recurrence_entries, _renamed_error_bound(
                entry,
                Symbol("block_", source, "_", channel,
                       "_recurrence_", entry_index),
                prefix,
            ))
        end
        for (entry_index, entry) in enumerate(bohr.error_ledger.mpo_products)
            push!(product_entries, _renamed_error_bound(
                entry,
                Symbol("block_", source, "_", channel,
                       "_bohr_product_", entry_index),
                prefix,
            ))
        end
        push!(product_entries, QuantumFurnace.ParentErrorBound(
            T,
            Symbol("block_", source, "_", channel, "_q_sandwich");
            evidence=:unmeasured,
            note=prefix *
                 "zip-up construction of conj(Q) tensor Q has no operator-norm certificate",
        ))
        push!(block_entries, QuantumFurnace.ParentErrorBound(
            T,
            Symbol("parent_block_", source, "_", channel);
            magnitude=block_compression_bounds[index],
            evidence=:floating_point_norm,
            note=prefix *
                 "one-sided Frobenius bound for measured final block additions; " *
                 "internal zip-up errors remain separate and unmeasured",
        ))
    end
    return QuantumFurnace.ParentErrorLedger(
        T;
        spectral_interval=QuantumFurnace.ParentErrorBound(
            T,
            :spectral_interval;
            magnitude=zero(T),
            evidence=:rigorous_bound,
            note="every compact patch uses its certified local Weyl-Gershgorin interval",
        ),
        scalar_polynomial=QuantumFurnace.ParentErrorBound(
            T,
            :scalar_polynomial;
            magnitude=scalar_bound,
            evidence=:rigorous_bound,
            note="maximum per-block certified scalar error; propagation to the parent is not yet bounded",
        ),
        modular_compatibility=QuantumFurnace.ParentErrorBound(
            T,
            :modular_compatibility;
            magnitude=modular_bound,
            evidence=:rigorous_bound,
            note="maximum per-block certified scalar modular defect",
        ),
        locality_radius=QuantumFurnace.ParentErrorBound(
            T,
            :locality_radius;
            evidence=:unmeasured,
            note="fixed-radius parent is a named surrogate without a summed global locality-tail bound",
        ),
        mpo_recurrences=Tuple(recurrence_entries),
        mpo_products=Tuple(product_entries),
        block_assembly=Tuple(block_entries),
        total_sum_compression=QuantumFurnace.ParentErrorBound(
            T,
            :total_sum_compression;
            magnitude=total_compression_bound,
            evidence=:floating_point_norm,
            note="sum of measured pre/post final-truncation Frobenius differences during total assembly",
        ),
    )
end

"""
    build_dll_parent(config, hamiltonian, blocks, controls; ...)

Assemble all Gaussian finite-patch DLL parent blocks on a common fused doubled
chain without materialising a dense global matrix. Each source/channel block
is formed separately from its own `Q`, `L`, and signed `N`; channel operators
are never summed before their sandwiches are constructed.

The returned target is always `:finite_patch_bohr_surrogate`. Locality error
remains unmeasured, no spectral shift or PSD projection is applied, and the
default leaves the measured anti-Hermitian component untouched. Set
`symmetrize=true` only for a logged numerical symmetrisation.
"""
function QuantumFurnace.build_dll_parent(
    config::QuantumFurnace.Config{
        QuantumFurnace.TensorNetworkSpectrum,
        QuantumFurnace.BohrDomain,
        QuantumFurnace.DLL,
        T,
    },
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    blocks::AbstractVector{<:QuantumFurnace.LocalDLLBlock1D{T}},
    controls::QuantumFurnace.BohrMPOControls{T};
    sites=nothing,
    symmetrize::Bool=false,
    raw_generator_rate::Real=1,
    proposal_normalisation::Symbol=:qf_averaged,
    include_sweep_clock::Bool=false,
    diagnostic_probe_count::Integer=1,
    diagnostic_probe_seed::Integer=0x51a8,
) where {T<:AbstractFloat}
    QuantumFurnace.validate_dll_tensor_network(
        config, hamiltonian, blocks, controls)
    controls.target_label == :finite_patch_bohr_surrogate ||
        throw(ArgumentError(
            "Task 8 assembles only :finite_patch_bohr_surrogate parents."))
    provenance = QuantumFurnace.dll_parent_provenance(
        config,
        hamiltonian,
        blocks;
        controls,
        raw_generator_rate,
        proposal_normalisation,
        include_sweep_clock,
    )
    fused_sites = sites === nothing ?
        fused_siteinds(
            hamiltonian.num_sites; physical_dim=hamiltonian.local_dim) :
        _validate_parent_fused_sites(
            sites, hamiltonian.num_sites, hamiltonian.local_dim)
    fused_sites = _validate_parent_fused_sites(
        fused_sites, hamiltonian.num_sites, hamiltonian.local_dim)

    bohr_blocks = DLLGaussianBohrMPO[]
    block_keys = Tuple{Int, Int}[]
    for (source_index, block) in pairs(blocks)
        channels = getfield(block, :channels)
        for channel_index in eachindex(channels)
            length(channels) == 1 || throw(ArgumentError(
                "Task 8 currently inherits Task 7B's one-Gaussian-channel scope."))
            push!(bohr_blocks, QuantumFurnace.build_dll_bohr_mpo(
                config, hamiltonian, block, controls))
            push!(block_keys, (source_index, channel_index))
        end
    end
    isempty(bohr_blocks) && throw(ArgumentError(
        "finite-patch parent assembly requires at least one block."))
    typed_bohr_blocks = Vector{typeof(first(bohr_blocks))}(bohr_blocks)

    assembly = _assemble_parent_components(
        typed_bohr_blocks,
        block_keys,
        fused_sites,
        controls;
        symmetrize,
        diagnostic_probe_count=Int(diagnostic_probe_count),
        diagnostic_probe_seed=Int(diagnostic_probe_seed),
    )
    ledger = _parent_error_ledger(
        typed_bohr_blocks,
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
    return FinitePatchDLLParent{
        T,
        typeof(first(typed_bohr_blocks)),
        typeof(bundle),
        typeof(provenance),
        typeof(model_specification),
    }(
        bundle,
        fused_sites,
        typed_bohr_blocks,
        provenance,
        model_specification,
        controls,
        assembly.assembly_records,
        assembly.diagnostics,
    )
end
