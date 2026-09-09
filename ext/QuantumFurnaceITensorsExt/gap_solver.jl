# The first method retains the deliberately small exact-overlap validator.
# The finite-patch Task-10 multi-start solver follows it below.

struct _FinitePatchCandidate{T<:AbstractFloat}
    state::Union{Nothing, ITensorMPS.MPS}
    attempt::ParentLowEnergyAttempt{T}
end

function _fp_record_attempt!(
    attempts::Vector{ParentLowEnergyAttempt{T}},
    attempt_sink::Union{Nothing, Vector{ParentLowEnergyAttempt{T}}},
    attempt::ParentLowEnergyAttempt{T},
) where {T<:AbstractFloat}
    push!(attempts, attempt)
    attempt_sink === nothing || push!(attempt_sink, attempt)
    return attempt
end

function _expanded_sweep_schedule(schedule::Tuple, sweeps::Int)
    return [schedule[min(index, length(schedule))] for index in 1:sweeps]
end

function _exact_kernel_states(reference::ExactDLLParentReference)
    return ITensorMPS.MPS[
        dense_to_mps(reference.kernel_vectors_fused[:, index], reference.sites)
        for index in axes(reference.kernel_vectors_fused, 2)
    ]
end

function _exact_state_diagnostics(
    state::ITensorMPS.MPS,
    reference::ExactDLLParentReference{T},
) where {T<:AbstractFloat}
    vector = Vector{Complex{T}}(mps_to_dense(state, reference.sites))
    vector ./= norm(vector)
    image = reference.parent_fused * vector
    energy = T(real(dot(vector, image)))
    residual_vector = image - energy * vector
    residual = T(norm(residual_vector))
    variance = T(real(dot(residual_vector, residual_vector)))
    overlaps = T.(abs.(adjoint(reference.kernel_vectors_fused) * vector))
    mpo_energy = T(real(ITensorMPS.inner(
        state', reference.bundle.total_parent, state)))
    return (; energy, residual, variance, overlaps, mpo_energy)
end

"""
    solve_dll_parent_gap(reference;
                         controls=ParentGapControls(num_starts=1,
                                                    sectors=(:unrestricted,)),
                         penalty_factor=maximum(controls.penalty_factors),
                         start_index=1)

Run one deterministic penalty-state DMRG calculation against an exact dense
DLL parent reference. The penalty is `penalty_factor * exact_gap`. The result
is returned only if it passes independent dense energy, residual, complete-
kernel-overlap, and dense-versus-MPO gates. Low-penalty, underconverged, and
wrong-excitation runs therefore throw instead of returning a gap estimate.

This is the maintained vertical-slice solver, not the production multi-start
solver planned for approximate MPO parents.
"""
function QuantumFurnace.solve_dll_parent_gap(
    reference::ExactDLLParentReference{T};
    controls::QuantumFurnace.ParentGapControls=
        QuantumFurnace.ParentGapControls(
            num_starts=1, sectors=(:unrestricted,)),
    penalty_factor::Real=maximum(controls.penalty_factors),
    start_index::Integer=1,
) where {T<:AbstractFloat}
    start_index > 0 || throw(ArgumentError("start_index must be > 0."))
    controls.sectors == (:unrestricted,) || throw(ArgumentError(
        "the exact-overlap DMRG skeleton supports only " *
        "sectors=(:unrestricted,); sector scans are deferred to Task 10."))
    controls.num_starts == 1 || throw(ArgumentError(
        "the exact-overlap DMRG skeleton runs one deterministic start; " *
        "multi-start orchestration is deferred to Task 10."))
    factor = T(penalty_factor)
    isfinite(factor) && factor > zero(T) || throw(ArgumentError(
        "penalty_factor must be finite and > 0."))
    gap = reference.dense.spectrum.first_positive_eigenvalue
    gap === nothing && throw(ArgumentError(
        "the exact parent has no resolved positive eigenvalue."))
    reference.diagnostics.kernel_complete || throw(ArgumentError(
        "exact overlap DMRG requires a complete dense kernel."))

    kernel_states = _exact_kernel_states(reference)
    initial_linkdim = first(controls.maxdim_schedule)
    rng = Random.MersenneTwister(controls.random_seed + Int(start_index) - 1)
    initial = ITensorMPS.random_mps(
        rng,
        Complex{T},
        reference.sites;
        linkdims=initial_linkdim,
    )
    maxdims = _expanded_sweep_schedule(
        controls.maxdim_schedule, controls.sweeps)
    cutoffs = fill(T(controls.cutoff), controls.sweeps)
    penalty = factor * gap
    _, state = ITensorMPS.dmrg(
        reference.bundle.total_parent,
        kernel_states,
        initial;
        weight=penalty,
        nsweeps=controls.sweeps,
        maxdim=maxdims,
        cutoff=cutoffs,
        eigsolve_tol=min(T(1e-12), controls.residual_tolerance / T(10)),
        eigsolve_krylovdim=16,
        outputlevel=0,
        ishermitian=true,
    )
    ITensorMPS.normalize!(state)
    measured = _exact_state_diagnostics(state, reference)
    energy_ok = isapprox(
        measured.energy,
        gap;
        atol=controls.energy_tolerance,
        rtol=controls.energy_tolerance,
    )
    residual_ok = measured.residual <= controls.residual_tolerance
    overlap_ok = all(
        overlap -> overlap <= controls.overlap_tolerance,
        measured.overlaps,
    )
    mpo_ok = isapprox(
        measured.mpo_energy,
        measured.energy;
        atol=controls.energy_tolerance,
        rtol=controls.energy_tolerance,
    )
    energy_ok && residual_ok && overlap_ok && mpo_ok || error(
        "exact DLL penalty DMRG rejected: " *
        "energy=$(measured.energy), exact_gap=$gap, " *
        "residual=$(measured.residual), " *
        "max_kernel_overlap=$(maximum(measured.overlaps)), " *
        "mpo_energy=$(measured.mpo_energy), penalty_factor=$factor.")

    bond_dimensions = Int[
        ITensors.dim(ITensorMPS.linkind(state, bond))
        for bond in 1:(length(state) - 1)
    ]
    return QuantumFurnace.DLLParentLowEnergyState(
        state;
        energy=measured.energy,
        residual=measured.residual,
        variance=measured.variance,
        kernel_overlaps=measured.overlaps,
        bond_dimensions,
        start_index,
        sector=:unrestricted,
        converged=true,
    )
end

@inline function _fp_state_bond_dimensions(state::ITensorMPS.MPS)
    return Int[
        ITensors.dim(ITensorMPS.linkind(state, bond))
        for bond in 1:(length(state) - 1)
    ]
end

function _fp_solver_schedule(
    controls::QuantumFurnace.ParentGapControls,
    cap::Int,
    ::Type{T},
) where {T<:AbstractFloat}
    maxdims = min.(
        _expanded_sweep_schedule(controls.maxdim_schedule, controls.sweeps),
        cap,
    )
    cutoffs = fill(T(controls.cutoff), controls.sweeps)
    return (; maxdims, cutoffs)
end

function _fp_initial_state(
    preparation::GibbsPurificationMPS{T},
    controls::QuantumFurnace.ParentGapControls,
    cap::Int,
    level_index::Int,
    penalty_index::Int,
    start_index::Int,
) where {T<:AbstractFloat}
    if start_index == 1
        state = copy(preparation.state)
        if length(state) > 1
            ITensorMPS.truncate!(
                state; cutoff=T(controls.cutoff), maxdim=cap)
        end
        ITensorMPS.normalize!(state)
        return state, :independent_gibbs
    end
    seed_offset =
        1_000_003 * level_index + 10_007 * penalty_index +
        101 * cap + start_index
    rng = Random.MersenneTwister(controls.random_seed + seed_offset)
    initial_linkdim = min(cap, first(controls.maxdim_schedule))
    state = ITensorMPS.random_mps(
        rng,
        Complex{T},
        preparation.sites;
        linkdims=initial_linkdim,
    )
    ITensorMPS.normalize!(state)
    return state, :random
end

function _fp_sum_block_action(
    blocks::Vector{ITensorMPS.MPO},
    state::ITensorMPS.MPS,
    ::Type{T},
) where {T<:AbstractFloat}
    isempty(blocks) && throw(ArgumentError(
        "parent block vector must be nonempty."))
    energy = zero(Complex{T})
    image = nothing
    for block in blocks
        energy += Complex{T}(ITensorMPS.inner(state', block, state))
        block_image = _apply_without_requested_truncation(block, state)
        image = image === nothing ? block_image :
            _directsum_add_states(image, block_image)
    end
    residual_state = _directsum_add_states(image, -(energy * state))
    if length(residual_state) > 1
        ITensorMPS.truncate!(
            residual_state;
            cutoff=zero(T),
            maxdim=_maximum_mps_bond(residual_state),
        )
    end
    residual = T(norm(residual_state))
    isfinite(energy) || error("parent Rayleigh quotient became nonfinite.")
    isfinite(residual) || error("parent eigen-residual became nonfinite.")
    return (; energy, residual)
end

function _fp_complete_gram_basis(
    states::Vector{ITensorMPS.MPS},
    controls::QuantumFurnace.ParentGapControls,
    cap::Int,
    ::Type{T},
) where {T<:AbstractFloat}
    isempty(states) && return ITensorMPS.MPS[]
    count = length(states)
    gram = Matrix{Complex{T}}(undef, count, count)
    for column in 1:count, row in 1:count
        gram[row, column] = Complex{T}(
            ITensorMPS.inner(states[row], states[column]))
    end
    gram = (gram + adjoint(gram)) / T(2)
    decomposition = eigen(Hermitian(gram))
    independence_tolerance = max(
        T(controls.overlap_tolerance)^2,
        T(256) * eps(T),
    )
    minimum(decomposition.values) > independence_tolerance || error(
        "observed low-energy states have a singular complete Gram matrix; " *
        "the next penalty target is unresolved.")
    inverse_sqrt = decomposition.vectors *
        Diagonal(inv.(sqrt.(decomposition.values))) *
        adjoint(decomposition.vectors)
    basis = ITensorMPS.MPS[]
    sizehint!(basis, count)
    for column in 1:count
        combined = inverse_sqrt[1, column] * states[1]
        for row in 2:count
            combined = _directsum_add_states(
                combined, inverse_sqrt[row, column] * states[row])
        end
        if length(combined) > 1
            ITensorMPS.truncate!(
                combined;
                cutoff=zero(T),
                maxdim=Base.checked_mul(cap, count),
            )
        end
        ITensorMPS.normalize!(combined)
        push!(basis, combined)
    end
    orthogonal_gram = Matrix{Complex{T}}(undef, count, count)
    for column in 1:count, row in 1:count
        orthogonal_gram[row, column] = Complex{T}(
            ITensorMPS.inner(basis[row], basis[column]))
    end
    defect = T(norm(orthogonal_gram - I))
    defect <= max(T(controls.overlap_tolerance), T(1024) * eps(T)) || error(
        "complete-Gram orthogonalisation defect $defect exceeds the " *
        "declared overlap tolerance $(controls.overlap_tolerance).")
    return basis
end

@inline function _fp_prior_overlaps(
    state::ITensorMPS.MPS,
    prior_states::Vector{ITensorMPS.MPS},
    ::Type{T},
) where {T<:AbstractFloat}
    return T[
        clamp(T(abs(ITensorMPS.inner(prior, state))), zero(T), one(T))
        for prior in prior_states
    ]
end

function _fp_rejection_reason(
    original,
    tighter,
    overlaps::Vector{T},
    controls::QuantumFurnace.ParentGapControls,
    zero_mode_tolerance::T,
) where {T<:AbstractFloat}
    abs(imag(original.energy)) <= controls.energy_tolerance ||
        return :complex_energy
    abs(imag(tighter.energy)) <= controls.energy_tolerance ||
        return :complex_tighter_energy
    real(original.energy) >= -zero_mode_tolerance ||
        return :materially_negative_energy
    original.residual <= controls.residual_tolerance ||
        return :residual
    tighter.residual <= controls.residual_tolerance ||
        return :tighter_residual
    isapprox(
        real(original.energy),
        real(tighter.energy);
        atol=controls.energy_tolerance,
        rtol=controls.energy_tolerance,
    ) || return :tighter_energy_mismatch
    all(overlap -> overlap <= controls.overlap_tolerance, overlaps) ||
        return :prior_state_leakage
    return :accepted
end

function _fp_dmrg_attempt(
    parent::FinitePatchDLLParent{T},
    tighter_parent::FinitePatchDLLParent{T},
    preparation::GibbsPurificationMPS{T},
    controls::QuantumFurnace.ParentGapControls,
    prior_states::Vector{ITensorMPS.MPS},
    cap::Int,
    level_index::Int,
    penalty_index::Int,
    start_index::Int,
    stage::Symbol,
    penalty_factor::Union{Nothing, T},
    penalty_weight::Union{Nothing, T},
    zero_mode_tolerance::T,
) where {T<:AbstractFloat}
    initial, start_kind = _fp_initial_state(
        preparation, controls, cap, level_index, penalty_index, start_index)
    schedule = _fp_solver_schedule(controls, cap, T)
    state = try
        if isempty(prior_states)
            _, candidate = ITensorMPS.dmrg(
                copy(parent.bundle.block_parents),
                initial;
                nsweeps=controls.sweeps,
                maxdim=schedule.maxdims,
                cutoff=schedule.cutoffs,
                eigsolve_tol=min(T(1e-12), controls.residual_tolerance / T(10)),
                eigsolve_krylovdim=16,
                outputlevel=0,
                ishermitian=true,
            )
            candidate
        else
            penalty_weight === nothing && error(
                "an excited-state attempt requires a penalty weight.")
            basis = _fp_complete_gram_basis(
                prior_states, controls, cap, T)
            _, candidate = ITensorMPS.dmrg(
                parent.bundle.total_parent,
                copy(basis),
                initial;
                weight=penalty_weight,
                nsweeps=controls.sweeps,
                maxdim=schedule.maxdims,
                cutoff=schedule.cutoffs,
                eigsolve_tol=min(T(1e-12), controls.residual_tolerance / T(10)),
                eigsolve_krylovdim=16,
                outputlevel=0,
                ishermitian=true,
            )
            candidate
        end
    catch exception
        attempt = ParentLowEnergyAttempt{T}(
            stage, level_index, cap, start_index, start_kind, :unrestricted,
            penalty_factor, penalty_weight, nothing, nothing, nothing,
            nothing, nothing, false, :dmrg_failure,
            sprint(showerror, exception))
        return _FinitePatchCandidate{T}(nothing, attempt)
    end
    ITensorMPS.normalize!(state)
    original = _fp_sum_block_action(
        parent.bundle.block_parents, state, T)
    tighter = _fp_sum_block_action(
        tighter_parent.bundle.block_parents, state, T)
    overlaps = _fp_prior_overlaps(state, prior_states, T)
    reason = _fp_rejection_reason(
        original, tighter, overlaps, controls, zero_mode_tolerance)
    attempt = ParentLowEnergyAttempt{T}(
        stage,
        level_index,
        cap,
        start_index,
        start_kind,
        :unrestricted,
        penalty_factor,
        penalty_weight,
        T(real(original.energy)),
        original.residual,
        T(real(tighter.energy)),
        tighter.residual,
        isempty(overlaps) ? zero(T) : maximum(overlaps),
        reason == :accepted,
        reason,
        nothing,
    )
    return _FinitePatchCandidate{T}(state, attempt)
end

function _fp_select_cluster(
    candidates,
    controls::QuantumFurnace.ParentGapControls,
    label::AbstractString;
    require_random_multistart::Bool,
)
    accepted = filter(candidate -> candidate.attempt.accepted, candidates)
    isempty(accepted) && error("$label produced no accepted DMRG state.")
    T = typeof(controls.energy_tolerance)
    energies = T[
        candidate.attempt.energy::T for candidate in accepted
    ]
    spread = maximum(energies) - minimum(energies)
    isapprox(
        maximum(energies),
        minimum(energies);
        atol=controls.energy_tolerance,
        rtol=controls.energy_tolerance,
    ) || error(
        "$label disagrees across accepted starts or penalties: energy spread " *
        "$spread exceeds $(controls.energy_tolerance).")
    if require_random_multistart
        random_starts = unique(
            candidate.attempt.start_index for candidate in accepted
            if candidate.attempt.start_kind == :random)
        length(random_starts) >= 2 || error(
            "$label requires agreement from at least two unrelated random starts.")
    end
    sort!(accepted; by=candidate -> (
        candidate.attempt.energy,
        candidate.attempt.tighter_residual,
        candidate.attempt.start_index,
    ))
    return first(accepted), typeof(spread)(spread)
end

function _fp_validate_solver_inputs(
    config::QuantumFurnace.Config{
        QuantumFurnace.TensorNetworkSpectrum,
        QuantumFurnace.BohrDomain,
        QuantumFurnace.DLL,
        T,
    },
    parent::FinitePatchDLLParent{T},
    tighter_parent::FinitePatchDLLParent{T},
    preparation::GibbsPurificationMPS{T},
    controls::QuantumFurnace.ParentGapControls,
    max_levels::Int,
    dense_overlap::Bool,
    ;
    require_strict_parent_controls::Bool=true,
) where {T<:AbstractFloat}
    QuantumFurnace.validate_config!(config)
    config.num_qubits == length(parent.sites) || throw(DimensionMismatch(
        "Config and finite-patch parent lengths differ."))
    controls.sectors == (:unrestricted,) || throw(ArgumentError(
        "the current fused sites carry no QNs; Task 10 therefore covers all " *
        "sectors through exactly sectors=(:unrestricted,) and rejects fake " *
        "duplicated sector labels."))
    controls.num_starts >= 3 || throw(ArgumentError(
        "Task 10 requires at least three starts: the independent Gibbs seed " *
        "and at least two unrelated random starts."))
    requested_caps = sort!(unique(Int.(controls.maxdim_schedule)))
    total_dimension = prod(
        site -> ITensors.dim(site), parent.sites; init=1)
    left_dimension = 1
    maximum_exact_bond = 1
    for bond in 1:(length(parent.sites) - 1)
        left_dimension = Base.checked_mul(
            left_dimension, ITensors.dim(parent.sites[bond]))
        right_dimension = total_dimension ÷ left_dimension
        maximum_exact_bond = max(
            maximum_exact_bond, min(left_dimension, right_dimension))
    end
    caps = sort!(unique(min(cap, maximum_exact_bond)
                        for cap in requested_caps))
    minimum_effective_caps = dense_overlap ? 1 : 2
    length(caps) >= minimum_effective_caps || throw(ArgumentError(
        "Task 10 requires at least $minimum_effective_caps distinct effective " *
        "bond dimension$(minimum_effective_caps == 1 ? "" : "s") after " *
        "clamping requested caps $(requested_caps) to the exact MPS ceiling " *
        "$maximum_exact_bond."))
    dense_overlap && last(caps) != maximum_exact_bond && throw(ArgumentError(
        "dense-overlap substitution for a second converged bond control " *
        "requires the tightest effective cap to reach the exact MPS ceiling " *
        "$maximum_exact_bond; got $(last(caps))."))
    factors = T.(controls.penalty_factors)
    minimum(factors) < one(T) < maximum(factors) || throw(ArgumentError(
        "penalty_factors must scan both sides of the adaptive observed-level " *
        "scale, with at least one factor below and above one."))
    max_levels >= 2 || throw(ArgumentError("max_levels must be at least 2."))
    parent.sites == tighter_parent.sites || throw(ArgumentError(
        "parent and tighter_parent must use identical fused site indices."))
    preparation.sites == parent.sites || throw(ArgumentError(
        "the independent Gibbs seed must use the parent's fused site indices."))
    _validate_tighter_parent(
        parent,
        tighter_parent;
        require_strict_controls=require_strict_parent_controls,
    )
    isequal(
        preparation.hamiltonian_specification,
        parent.model_specification.hamiltonian,
    ) || throw(ArgumentError(
        "the independent Gibbs seed must represent the same local Hamiltonian " *
        "as the parent."))
    parent.bundle.block_keys == tighter_parent.bundle.block_keys ||
        throw(ArgumentError(
            "tighter_parent must preserve every source/channel block key."))
    parent.assembly_diagnostics.presymmetrization_total_hermiticity_defect <=
        controls.residual_tolerance || throw(ArgumentError(
        "the represented parent Hermiticity defect exceeds the solver " *
        "residual tolerance."))
    tighter_parent.assembly_diagnostics.
        presymmetrization_total_hermiticity_defect <=
        controls.residual_tolerance || throw(ArgumentError(
        "the tighter parent Hermiticity defect exceeds the solver residual " *
        "tolerance."))
    return requested_caps, caps, factors, maximum_exact_bond
end

@inline _finite_patch_parent(parent::FinitePatchDLLParent) = parent
@inline _finite_patch_parent(
    reference::ExactFinitePatchDLLParentReference,
) = reference.parent

function _validate_exact_patch_solver_reference(
    reference::ExactFinitePatchDLLParentReference{T},
    controls::QuantumFurnace.ParentGapControls,
) where {T<:AbstractFloat}
    reference.factorization_cap_saturated && throw(ArgumentError(
        "the exact-patch residual reference is compact-factorization cap-saturated."))
    reference.maximum_relative_reconstruction_error <=
        max(T(controls.energy_tolerance), T(1024) * eps(T)) ||
        throw(ArgumentError(
            "the exact-patch residual reference reconstruction error " *
            "$(reference.maximum_relative_reconstruction_error) exceeds the " *
            "declared eigensolver accuracy."))
    return nothing
end

"""
    solve_dll_parent_gap(config, parent, preparation; tighter_parent,
                         controls=ParentGapControls(), max_levels=6,
                         zero_mode_tolerance=controls.energy_tolerance,
                         dense_overlap=false)

Run the Task-10 finite-patch low-energy solver. Ground-state DMRG uses the
uncompressed vector of channel-parent MPOs. Excited-state DMRG uses the
separately validated total MPO only for ITensorMPS's penalty-state API, after
orthogonalising all previously retained states through their complete Gram
matrix. Every returned absolute level and residual is recomputed against the
original unpenalised block vector and against a genuinely tighter block-vector
MPO.

The independent Gibbs MPS is one diagnostic seed; at least two unrelated
random starts, unrestricted full-space coverage, and an adaptive penalty scan
on both sides of each observed level scale are required. Outside dense overlap,
two distinct effective bond dimensions must converge. At guarded `N <= 4`
dense overlap, one converged full-rank cap may instead be qualified by the
complete dense represented-parent spectrum; lower caps remain telemetry. The
result is conservatively labelled `:observed_manifold_spacing`. It is not a
global DLL gap, lower bound, or certificate.
"""
function QuantumFurnace.solve_dll_parent_gap(
    config::QuantumFurnace.Config{
        QuantumFurnace.TensorNetworkSpectrum,
        QuantumFurnace.BohrDomain,
        QuantumFurnace.DLL,
        T,
    },
    parent::FinitePatchDLLParent{T},
    preparation::GibbsPurificationMPS{T};
    tighter_parent::Union{
        FinitePatchDLLParent{T},
        ExactFinitePatchDLLParentReference{T},
    },
    controls::QuantumFurnace.ParentGapControls=
        QuantumFurnace.ParentGapControls(),
    max_levels::Integer=6,
    zero_mode_tolerance::Real=controls.energy_tolerance,
    dense_overlap::Bool=false,
    _allow_equal_parent_controls::Bool=false,
    _attempt_sink::Union{
        Nothing, Vector{ParentLowEnergyAttempt{T}}}=nothing,
) where {T<:AbstractFloat}
    level_limit = Int(max_levels)
    zero_tolerance = T(zero_mode_tolerance)
    QuantumFurnace._require_finite_positive(
        zero_tolerance, "zero_mode_tolerance")
    _allow_equal_parent_controls &&
        !(tighter_parent isa ExactFinitePatchDLLParentReference) &&
        throw(ArgumentError(
            "equal parent-assembly controls are reserved for an independently " *
            "validated exact-patch factorization reference."))
    tighter_parent isa ExactFinitePatchDLLParentReference &&
        _validate_exact_patch_solver_reference(tighter_parent, controls)
    tighter = _finite_patch_parent(tighter_parent)
    requested_caps, caps, factors, maximum_exact_bond =
        _fp_validate_solver_inputs(
        config,
        parent,
        tighter,
        preparation,
        controls,
        level_limit,
        dense_overlap;
        require_strict_parent_controls=!_allow_equal_parent_controls,
    )
    required_converged_caps = dense_overlap ? 1 : 2
    attempts = ParentLowEnergyAttempt{T}[]
    states_by_cap = [ITensorMPS.MPS[] for _ in caps]
    measurements_by_cap = [ParentLowEnergyAttempt{T}[] for _ in caps]
    multi_start_spreads = T[]
    penalty_scales = T[]

    # Absolute ground level: the Gibbs state is only a seed, never a presumed
    # zero mode of the finite-patch parent.
    for (cap_index, cap) in pairs(caps)
        candidates = _FinitePatchCandidate{T}[]
        for start_index in 1:controls.num_starts
            candidate = _fp_dmrg_attempt(
                parent, tighter, preparation, controls,
                ITensorMPS.MPS[], cap, 1, 0, start_index, :ground,
                nothing, nothing, zero_tolerance)
            _fp_record_attempt!(attempts, _attempt_sink, candidate.attempt)
            push!(candidates, candidate)
        end
        accepted = filter(candidate -> candidate.attempt.accepted, candidates)
        isempty(accepted) && continue
        selected, spread = _fp_select_cluster(
            candidates, controls, "ground level at bond $cap";
            require_random_multistart=cap == last(caps),
        )
        push!(states_by_cap[cap_index], selected.state)
        push!(measurements_by_cap[cap_index], selected.attempt)
        cap == last(caps) && push!(multi_start_spreads, spread)
    end
    successful_ground_caps = findall(states -> !isempty(states), states_by_cap)
    length(successful_ground_caps) >= required_converged_caps || error(
        "fewer than $required_converged_caps effective bond controls produced " *
        "an accepted ground level.")
    last(successful_ground_caps) == length(caps) || error(
        "the tightest requested bond dimension did not converge.")

    observed_manifold_dimension = 0
    for level_index in 2:level_limit
        tight_cap_index = length(caps)
        tight_prior = states_by_cap[tight_cap_index]
        preliminary_weight = T(2) * max(
            T(norm(parent.bundle.total_parent)), one(T))
        preliminary_candidates = _FinitePatchCandidate{T}[]
        for growth_step in 0:2
            weight = preliminary_weight * T(2)^growth_step
            empty!(preliminary_candidates)
            for start_index in 1:controls.num_starts
                candidate = _fp_dmrg_attempt(
                    parent, tighter, preparation, controls,
                    tight_prior, last(caps), level_index, -growth_step - 1,
                    start_index, :penalty_scale, nothing, weight,
                    zero_tolerance)
                _fp_record_attempt!(attempts, _attempt_sink, candidate.attempt)
                push!(preliminary_candidates, candidate)
            end
            any(candidate -> candidate.attempt.accepted,
                preliminary_candidates) && break
        end
        preliminary, _ = _fp_select_cluster(
            preliminary_candidates,
            controls,
            "adaptive penalty-scale level $level_index";
            require_random_multistart=false,
        )
        ground_energy = measurements_by_cap[tight_cap_index][1].energy
        raw_scale = preliminary.attempt.energy - ground_energy
        resolution = max(
            controls.energy_tolerance,
            preliminary.attempt.residual +
                measurements_by_cap[tight_cap_index][1].residual,
        )
        raw_scale >= -resolution || error(
            "adaptive penalty solve returned an absolute level below the " *
            "ground-level residual interval: separation=$raw_scale, " *
            "resolution=$resolution.")
        # Degenerate manifold directions have zero penalty threshold. Retain
        # an explicit resolution-scale penalty so their Gram-independent
        # partners can still be collected without inventing a nonzero level.
        scale = max(raw_scale, T(10) * resolution)
        push!(penalty_scales, scale)

        for (cap_index, cap) in pairs(caps)
            length(states_by_cap[cap_index]) == level_index - 1 || continue
            candidates = _FinitePatchCandidate{T}[]
            for (factor_index, factor) in pairs(factors)
                weight = factor * scale
                for start_index in 1:controls.num_starts
                    candidate = _fp_dmrg_attempt(
                        parent, tighter, preparation, controls,
                        states_by_cap[cap_index], cap, level_index,
                        factor_index, start_index, :penalty_scan, factor,
                        weight, zero_tolerance)
                    _fp_record_attempt!(attempts, _attempt_sink, candidate.attempt)
                    push!(candidates, candidate)
                end
            end
            high_candidates = filter(candidates) do candidate
                candidate.attempt.penalty_factor !== nothing &&
                candidate.attempt.penalty_factor > one(T)
            end
            accepted_high = filter(
                candidate -> candidate.attempt.accepted, high_candidates)
            isempty(accepted_high) && continue
            selected, spread = _fp_select_cluster(
                high_candidates,
                controls,
                "level $level_index at bond $cap";
                require_random_multistart=cap == last(caps),
            )
            low_scanned = any(candidates) do candidate
                candidate.attempt.penalty_factor !== nothing &&
                candidate.attempt.penalty_factor < one(T)
            end
            low_scanned || error(
                "level $level_index did not scan a penalty below its " *
                "adaptive scale.")
            if cap == last(caps) && raw_scale > resolution
                low_collapse_observed = any(candidates) do candidate
                    candidate.attempt.penalty_factor !== nothing &&
                    candidate.attempt.penalty_factor < one(T) &&
                    candidate.attempt.rejection_reason == :prior_state_leakage
                end
                low_collapse_observed || error(
                    "the below-scale penalty control did not demonstrate " *
                    "collapse into the previously retained low-energy span.")
            end
            push!(states_by_cap[cap_index], selected.state)
            push!(measurements_by_cap[cap_index], selected.attempt)
            cap == last(caps) && push!(multi_start_spreads, spread)
        end
        successful_level_caps = findall(
            states -> length(states) == level_index, states_by_cap)
        length(successful_level_caps) >= required_converged_caps || error(
            "level $level_index converged at fewer than " *
            "$required_converged_caps effective bond controls.")
        last(successful_level_caps) == length(caps) || error(
            "level $level_index did not converge at the tightest bond dimension.")

        tight_energies = T[
            measurement.energy
            for measurement in measurements_by_cap[end]
        ]
        separation = tight_energies[end] - tight_energies[1]
        adjacent_resolution =
            measurements_by_cap[end][end].residual +
            measurements_by_cap[end][1].residual
        if separation > max(controls.energy_tolerance, adjacent_resolution)
            observed_manifold_dimension = level_index - 1
            break
        end
    end
    observed_manifold_dimension > 0 || error(
        "max_levels=$level_limit did not resolve a state outside the observed " *
        "lowest-energy manifold.")

    converged_cap_indices = findall(
        states -> length(states) == observed_manifold_dimension + 1,
        states_by_cap,
    )
    length(converged_cap_indices) >= required_converged_caps || error(
        "the resolved manifold has fewer than $required_converged_caps " *
        "converged effective bond controls.")
    comparison_indices = length(converged_cap_indices) >= 2 ?
        last(converged_cap_indices, 2) : Int[]
    absolute_levels_by_bond = Vector{Vector{T}}()
    converged_caps = Int[]
    for cap_index in converged_cap_indices
        push!(converged_caps, caps[cap_index])
        push!(absolute_levels_by_bond, T[
            measurement.energy
            for measurement in measurements_by_cap[cap_index]
        ])
    end
    bond_spreads = T[]
    if !isempty(comparison_indices)
        for level_index in 1:(observed_manifold_dimension + 1)
            coarse = measurements_by_cap[first(comparison_indices)][level_index].energy
            tight = measurements_by_cap[last(comparison_indices)][level_index].energy
            spread = abs(tight - coarse)
            isapprox(
                tight,
                coarse;
                atol=controls.energy_tolerance,
                rtol=controls.energy_tolerance,
            ) || error(
                "absolute level $level_index is not bond-dimension converged: " *
                "difference $spread exceeds $(controls.energy_tolerance).")
            push!(bond_spreads, spread)
        end
    end

    tight_states = states_by_cap[end]
    tight_measurements = measurements_by_cap[end]
    absolute_levels = T[measurement.energy for measurement in tight_measurements]
    issorted(absolute_levels) || error(
        "retained unpenalised absolute levels are not sorted.")
    spacing = absolute_levels[observed_manifold_dimension + 1] -
              absolute_levels[observed_manifold_dimension]
    spacing_resolution =
        tight_measurements[observed_manifold_dimension + 1].residual +
        tight_measurements[observed_manifold_dimension].residual
    spacing > max(controls.energy_tolerance, spacing_resolution) || error(
        "observed manifold spacing $spacing is not resolved beyond adjacent " *
        "residual intervals $spacing_resolution.")
    tighter_spacing =
        tight_measurements[observed_manifold_dimension + 1].tighter_energy -
        tight_measurements[observed_manifold_dimension].tighter_energy
    tighter_spacing_resolution =
        tight_measurements[observed_manifold_dimension + 1].tighter_residual +
        tight_measurements[observed_manifold_dimension].tighter_residual
    tighter_spacing > max(controls.energy_tolerance, tighter_spacing_resolution) ||
        error(
            "the tighter-MPO manifold spacing $tighter_spacing is not resolved " *
            "beyond its adjacent residual intervals " *
            "$tighter_spacing_resolution.")
    isapprox(
        tighter_spacing,
        spacing;
        atol=controls.energy_tolerance,
        rtol=controls.energy_tolerance,
    ) || error(
        "the represented and tighter-MPO manifold spacings disagree: " *
        "$spacing versus $tighter_spacing.")

    observed_kernel_dimension = count(eachindex(absolute_levels)) do index
        abs(absolute_levels[index]) <= zero_tolerance &&
        tight_measurements[index].residual <= controls.residual_tolerance &&
        tight_measurements[index].tighter_residual <= controls.residual_tolerance
    end
    observed_kernel_dimension <= observed_manifold_dimension || error(
        "a zero mode was observed above the resolved lowest manifold.")

    dense_levels = nothing
    dense_error = nothing
    kernel_complete = false
    kernel_evidence = :observed_only
    if dense_overlap
        verification = QuantumFurnace.verify_dll_parent(
            parent;
            dense_overlap=true,
            kernel_tolerance=zero_tolerance,
        )
        dense_matrix = Matrix{Complex{T}}(mpo_to_dense(
            parent.bundle.total_parent, parent.sites))
        dense_values = T.(eigvals(Hermitian(
            (dense_matrix + adjoint(dense_matrix)) / T(2))))
        dense_levels = dense_values[1:length(absolute_levels)]
        errors = abs.(dense_levels .- absolute_levels)
        dense_error = maximum(errors)
        all(index -> isapprox(
                dense_levels[index], absolute_levels[index];
                atol=controls.energy_tolerance,
                rtol=controls.energy_tolerance,
            ), eachindex(absolute_levels)) || error(
            "DMRG absolute levels disagree with the complete dense represented " *
            "surrogate spectrum; maximum error $dense_error.")
        verification.observed_kernel_count == observed_kernel_dimension ||
            error("DMRG and dense represented-surrogate kernel counts disagree.")
        if observed_kernel_dimension > 0
            kernel_complete = true
            kernel_evidence = :verified_complete
        end
    end

    gibbs = _gibbs_parent_diagnostics(
        preparation,
        parent;
        tighter_parent=tighter,
        require_strict_tighter_controls=!_allow_equal_parent_controls,
    )
    abs(imag(gibbs.total_energy)) <= controls.energy_tolerance || error(
        "global Gibbs expectation has a material imaginary component.")
    tighter_residuals = T[
        measurement.tighter_residual for measurement in tight_measurements
    ]
    tighter_energies = T[
        measurement.tighter_energy for measurement in tight_measurements
    ]
    diagnostics = QuantumFurnace.DLLParentDiagnostics(
        observed_kernel_dimension=observed_kernel_dimension,
        observed_manifold_dimension=observed_manifold_dimension,
        kernel_complete=kernel_complete,
        kernel_tolerance=zero_tolerance,
        kernel_evidence=kernel_evidence,
        primitivity_established=false,
        primitivity_provenance=:unrestricted_multistart_observed,
        hermiticity_defect=parent.assembly_diagnostics.
            presymmetrization_total_hermiticity_defect,
        minimum_energy=first(absolute_levels),
        gibbs_energy=T(real(gibbs.total_energy)),
        gibbs_residual=gibbs.total_residual,
        block_gibbs_residuals=gibbs.block_residuals,
        tighter_parent_residual=maximum(tighter_residuals),
    )

    low_energy_states = QuantumFurnace.DLLParentLowEnergyState{
        ITensorMPS.MPS, T,
    }[]
    sizehint!(low_energy_states, length(tight_states))
    kernel_states = tight_states[1:observed_kernel_dimension]
    for index in eachindex(tight_states)
        overlaps = _fp_prior_overlaps(tight_states[index], kernel_states, T)
        push!(low_energy_states, QuantumFurnace.DLLParentLowEnergyState(
            tight_states[index];
            energy=absolute_levels[index],
            residual=tight_measurements[index].residual,
            variance=tight_measurements[index].residual^2,
            kernel_overlaps=overlaps,
            bond_dimensions=_fp_state_bond_dimensions(tight_states[index]),
            start_index=tight_measurements[index].start_index,
            sector=:unrestricted,
            converged=true,
        ))
    end
    metadata = FinitePatchLowEnergyDiagnostics{T}(
        absolute_levels,
        observed_manifold_dimension,
        observed_kernel_dimension,
        spacing,
        zero_tolerance,
        requested_caps,
        maximum_exact_bond,
        caps,
        converged_caps,
        absolute_levels_by_bond,
        bond_spreads,
        multi_start_spreads,
        penalty_scales,
        tighter_energies,
        tighter_residuals,
        true,
        dense_levels,
        dense_error,
        attempts,
    )
    return QuantumFurnace.DLLTensorNetworkResult(
        config,
        parent.bundle,
        low_energy_states,
        diagnostics,
        parent.provenance;
        bohr_controls=parent.controls,
        gibbs_controls=preparation.controls,
        gap_controls=controls,
        gap_label=:observed_manifold_spacing,
        gap_value=spacing,
        metadata,
    )
end

"""
Solve the exact analytic finite-patch target with two independently controlled,
nonsaturated compact references. This remains an observed finite-patch spacing;
it is not an exact-global gap or a certificate.
"""
function QuantumFurnace.solve_dll_parent_gap(
    config::QuantumFurnace.Config{
        QuantumFurnace.TensorNetworkSpectrum,
        QuantumFurnace.BohrDomain,
        QuantumFurnace.DLL,
        T,
    },
    reference::ExactFinitePatchDLLParentReference{T},
    preparation::GibbsPurificationMPS{T};
    tighter_parent::ExactFinitePatchDLLParentReference{T},
    controls::QuantumFurnace.ParentGapControls=
        QuantumFurnace.ParentGapControls(),
    max_levels::Integer=6,
    zero_mode_tolerance::Real=controls.energy_tolerance,
    dense_overlap::Bool=false,
) where {T<:AbstractFloat}
    comparison = compare_exact_patch_references(
        reference,
        tighter_parent;
        relative_tolerance=max(
            T(controls.residual_tolerance), T(1024) * eps(T)),
        probe_count=2,
        probe_seed=controls.random_seed + 0x10f,
    )
    comparison.accepted || throw(ArgumentError(
        "the two exact-patch references did not pass their independent " *
        "non-saturation and action-agreement gate."))
    return QuantumFurnace.solve_dll_parent_gap(
        config,
        reference.parent,
        preparation;
        tighter_parent,
        controls,
        max_levels,
        zero_mode_tolerance,
        dense_overlap,
        _allow_equal_parent_controls=true,
    )
end
