# Deliberately small exact-overlap DMRG solver. Kernel discovery, multi-start
# logic, sector scans, and surrogate error transfer belong to Task 10.

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
