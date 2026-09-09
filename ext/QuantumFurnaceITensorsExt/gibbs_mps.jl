"""Dense small-system verification of an independently prepared purification."""
struct GibbsDenseVerification{T<:AbstractFloat}
    vector_error::T
    infidelity::T
    gibbs_trace_distance::T
    energy_error::T
end

"""
Convergence evidence obtained by comparing one preparation with an explicitly
stricter Gibbs-MPS preparation. These are floating-point diagnostics, not
rigorous state-norm bounds.
"""
struct GibbsMPSComparison{T<:AbstractFloat}
    infidelity::T
    maximum_one_site_trace_distance::T
    energy_difference::T
    energy_density_difference::T
    maximum_probe_expectation_difference::T
    maximum_probe_correlation_difference::T
end

"""
Diagnostics for a parent-independent Gibbs purification. `energy` includes the
stored scalar identity shift even though that shift is omitted from TDVP, where
it can only change the unnormalised state by a scalar.
"""
struct GibbsPurificationDiagnostics{T<:AbstractFloat}
    beta_phys::T
    beta_alg::T
    beta_frame::T
    hamiltonian_frame::Symbol
    coordinate_rescaling_factor::T
    identity_shift::T
    scale_provenance::Symbol
    preparation_method::Symbol
    requested_time_step::T
    actual_time_step::T
    sweeps::Int
    cutoff::Union{Nothing, T}
    maxdim_schedule::Vector{Int}
    state_norm::T
    maximum_bond_dimension::Int
    energy::T
    energy_density::T
    one_site_reduced_states::Vector{Matrix{Complex{T}}}
    maximum_local_hermiticity_defect::T
    maximum_local_trace_error::T
    probe_expectations::Vector{T}
    nearest_neighbor_probe_correlations::Vector{T}
    dense_verification::Union{Nothing, GibbsDenseVerification{T}}
    stricter_comparison::Union{Nothing, GibbsMPSComparison{T}}
end

"""
    GibbsPurificationMPS

An independently prepared, normalized thermofield purification on the
extension's ket-fast fused doubled chain. It stores the ket-copy Hamiltonian
used by TDVP, but no DLL parent or parent-derived convergence criterion.
"""
struct GibbsPurificationMPS{
    T<:AbstractFloat,
    S,
    H,
    C<:QuantumFurnace.GibbsMPSControls,
}
    state::S
    sites::Vector{ITensors.Index{Int}}
    ket_hamiltonian::H
    hamiltonian_specification::QuantumFurnace.LocalHamiltonian1D{T}
    controls::C
    diagnostics::GibbsPurificationDiagnostics{T}
end

function _validate_gibbs_sites(
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
            "every fused Gibbs site must have dimension $fused_dim."))
    all(site -> !ITensors.hasqns(site), collected) || throw(ArgumentError(
        "Gibbs purification currently requires unconstrained fused sites."))
    return Vector{ITensors.Index{Int}}(collected)
end

function _normalized_fused_bell_mps(
    ::Type{T},
    sites::Vector{ITensors.Index{Int}},
    physical_dim::Int,
) where {T<:AbstractFloat}
    amplitude = inv(sqrt(T(physical_dim)))
    tensors = ITensors.ITensor[]
    sizehint!(tensors, length(sites))
    for site in sites
        tensor = ITensors.ITensor(Complex{T}, site)
        for physical_state in 0:(physical_dim - 1)
            fused_state = physical_state +
                          physical_dim * physical_state + 1
            tensor[site => fused_state] = amplitude
        end
        push!(tensors, tensor)
    end
    state = ITensorMPS.MPS(tensors)
    state_norm = T(norm(state))
    isapprox(state_norm, one(T); atol=T(64) * eps(T), rtol=T(64) * eps(T)) ||
        error("normalized fused Bell construction produced norm $state_norm.")
    return state
end

function _exact_one_site_gibbs_mps(
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    beta_frame::T,
    site::ITensors.Index{Int},
) where {T<:AbstractFloat}
    matrix = Matrix{Complex{T}}(
        QuantumFurnace.materialize_local_hamiltonian(hamiltonian))
    matrix -= hamiltonian.global_shift * I
    decomposition = eigen(Hermitian(matrix))
    shifted_energies = decomposition.values .- minimum(decomposition.values)
    amplitude = decomposition.vectors *
                Diagonal(exp.(-beta_frame .* shifted_energies ./ T(2))) *
                adjoint(decomposition.vectors)
    amplitude ./= T(norm(amplitude))
    fused = matrix_to_fused(
        amplitude, 1; physical_dim=hamiltonian.local_dim)
    tensor = ITensors.ITensor(reshape(fused, length(fused)), site)
    return ITensorMPS.MPS([tensor])
end

function _config_beta_metadata(
    config::QuantumFurnace.Config{
        QuantumFurnace.TensorNetworkSpectrum,
        QuantumFurnace.BohrDomain,
        QuantumFurnace.DLL,
        T,
    },
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
) where {T<:AbstractFloat}
    QuantumFurnace.validate_config!(config)
    config.num_qubits == hamiltonian.num_sites || throw(DimensionMismatch(
        "Config num_qubits=$(config.num_qubits) does not match the local " *
        "Hamiltonian length $(hamiltonian.num_sites)."))
    config.beta_phys === nothing && throw(ArgumentError(
        "frame-checked Gibbs preparation requires Config.beta_phys."))
    beta_phys = T(config.beta_phys)
    beta_alg = T(config.beta)
    QuantumFurnace._require_finite_positive(beta_phys, "beta_phys")
    QuantumFurnace._require_finite_positive(beta_alg, "beta_alg")
    coordinate_rescaling_factor = beta_alg / beta_phys
    QuantumFurnace._require_finite_positive(
        coordinate_rescaling_factor, "coordinate rescaling factor")
    beta_frame = hamiltonian.coordinate_frame == :physical ?
        beta_phys : beta_alg

    hamiltonian.coordinate_frame == :physical && return (;
        beta_phys,
        beta_alg,
        beta_frame,
        coordinate_rescaling_factor,
    )
    expected_beta_alg = beta_phys * hamiltonian.rescaling_factor
    tolerance = T(128) * eps(T) *
                max(one(T), abs(beta_alg), abs(expected_beta_alg))
    isapprox(beta_alg, expected_beta_alg;
             atol=tolerance, rtol=tolerance) || throw(ArgumentError(
        "Gibbs preparation requires beta_alg == beta_phys * " *
        "rescaling_factor for the supplied Hamiltonian frame."))
    isapprox(coordinate_rescaling_factor, hamiltonian.rescaling_factor;
             atol=tolerance, rtol=tolerance) || throw(ArgumentError(
        "Config beta metadata and the algorithm-frame Hamiltonian use " *
        "different coordinate rescaling factors."))
    return (;
        beta_phys,
        beta_alg,
        beta_frame,
        coordinate_rescaling_factor,
    )
end

function _direct_beta_metadata(
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    beta_frame::T,
) where {T<:AbstractFloat}
    QuantumFurnace._require_finite_nonnegative(beta_frame, "beta_frame")
    coordinate_rescaling_factor = hamiltonian.rescaling_factor
    if hamiltonian.coordinate_frame == :physical
        beta_phys = beta_frame
        beta_alg = beta_phys * coordinate_rescaling_factor
    else
        beta_alg = beta_frame
        beta_phys = beta_alg / coordinate_rescaling_factor
    end
    return (;
        beta_phys,
        beta_alg,
        beta_frame,
        coordinate_rescaling_factor,
    )
end

function _ket_hamiltonian_mpo(
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    fused_sites::Vector{ITensors.Index{Int}},
) where {T<:AbstractFloat}
    physical_sites = local_operator_siteinds(
        hamiltonian.num_sites; local_dim=hamiltonian.local_dim)
    physical_hamiltonian = local_hamiltonian_mpo(
        hamiltonian; sites=physical_sites)
    # This is only the convention-preserving physical-to-fused lift shared
    # with parent assembly. No parent operator or DLL datum enters TDVP.
    return _lift_physical_mpo_to_fused(
        physical_hamiltonian,
        physical_sites,
        fused_sites,
        1,
        hamiltonian.num_sites;
        active_leg=:ket,
        transform=:identity,
        tag=:gibbs_ket_hamiltonian,
    )
end

function _ket_local_operator(
    matrix::AbstractMatrix{Complex{T}},
) where {T<:AbstractFloat}
    d = size(matrix, 1)
    size(matrix, 2) == d || throw(DimensionMismatch(
        "local physical observable must be square."))
    identity_d = Matrix{Complex{T}}(I, d, d)
    return Matrix{Complex{T}}(
        superoperator_to_fused(kron(identity_d, matrix), 1;
                               physical_dim=d))
end

function _one_site_reduced_states(
    state::ITensorMPS.MPS,
    physical_dim::Int,
    ::Type{T},
) where {T<:AbstractFloat}
    reduced = [zeros(Complex{T}, physical_dim, physical_dim)
               for _ in 1:length(state)]
    matrix_unit = zeros(Complex{T}, physical_dim, physical_dim)
    for row in 1:physical_dim, column in 1:physical_dim
        fill!(matrix_unit, zero(Complex{T}))
        matrix_unit[column, row] = one(Complex{T})
        values = ITensorMPS.expect(state, _ket_local_operator(matrix_unit))
        for site in eachindex(reduced)
            reduced[site][row, column] = Complex{T}(values[site])
        end
    end
    return reduced
end

function _local_state_defects(
    reduced::Vector{Matrix{Complex{T}}},
) where {T<:AbstractFloat}
    hermiticity = maximum(
        T(norm(rho - adjoint(rho))) / max(T(norm(rho)), eps(T))
        for rho in reduced
    )
    trace_error = maximum(T(abs(tr(rho) - one(T))) for rho in reduced)
    return hermiticity, trace_error
end

function _probe_observable(::Type{T}, physical_dim::Int) where {T<:AbstractFloat}
    values = physical_dim == 1 ? T[zero(T)] :
        collect(range(one(T), -one(T); length=physical_dim))
    return Matrix{Complex{T}}(Diagonal(Complex{T}.(values)))
end

function _probe_diagnostics(
    state::ITensorMPS.MPS,
    physical_dim::Int,
    ::Type{T},
) where {T<:AbstractFloat}
    fused_probe = _ket_local_operator(_probe_observable(T, physical_dim))
    expectations = T.(real.(ITensorMPS.expect(state, fused_probe)))
    if length(state) == 1
        return expectations, T[]
    end
    correlations = ITensorMPS.correlation_matrix(
        state, fused_probe, fused_probe; ishermitian=true)
    connected = T[
        real(correlations[site, site + 1]) -
        expectations[site] * expectations[site + 1]
        for site in 1:(length(state) - 1)
    ]
    return expectations, connected
end

@inline function _trace_distance(matrix::AbstractMatrix{Complex{T}}) where
        {T<:AbstractFloat}
    return T(sum(svdvals(matrix)) / T(2))
end

function _dense_gibbs_verification(
    state::ITensorMPS.MPS,
    sites::Vector{ITensors.Index{Int}},
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    beta_frame::T,
    energy::T,
) where {T<:AbstractFloat}
    hamiltonian.num_sites <= DENSE_BRIDGE_MAX_SITES || return nothing
    matrix = Matrix{Complex{T}}(
        QuantumFurnace.materialize_local_hamiltonian(hamiltonian))
    decomposition = eigen(Hermitian(matrix))
    shifted_energies = decomposition.values .- minimum(decomposition.values)
    square_root_weights = exp.(-beta_frame .* shifted_energies ./ T(2))
    square_root = decomposition.vectors * Diagonal(square_root_weights) *
                  adjoint(decomposition.vectors)
    square_root ./= T(norm(square_root))
    exact_vector = matrix_to_fused(
        square_root,
        hamiltonian.num_sites;
        physical_dim=hamiltonian.local_dim,
    )
    prepared_vector = mps_to_dense(state, sites)
    overlap = dot(exact_vector, prepared_vector)
    phase = iszero(overlap) ? one(Complex{T}) : conj(overlap) / abs(overlap)
    vector_error = T(norm(phase .* prepared_vector - exact_vector))
    infidelity = max(zero(T), one(T) - T(abs2(overlap)))

    exact_weights = abs2.(square_root_weights)
    exact_weights ./= sum(exact_weights)
    exact_gibbs = decomposition.vectors * Diagonal(exact_weights) *
                  adjoint(decomposition.vectors)
    prepared_gibbs = physical_partial_trace(state, sites)
    gibbs_trace_distance = _trace_distance(
        Matrix{Complex{T}}(prepared_gibbs - exact_gibbs))
    exact_energy = T(real(tr(exact_gibbs * matrix)))
    return GibbsDenseVerification{T}(
        vector_error,
        infidelity,
        gibbs_trace_distance,
        T(abs(energy - exact_energy)),
    )
end

function _strictness_check(
    controls::QuantumFurnace.GibbsMPSControls,
    actual_time_step::T,
    reference::GibbsPurificationMPS,
) where {T<:AbstractFloat}
    reference_step = T(reference.diagnostics.actual_time_step)
    current_schedule = collect(controls.maxdim_schedule)
    reference_schedule = collect(reference.controls.maxdim_schedule)
    cap_at_fraction(schedule, sweeps, fraction) = schedule[min(
        max(1, ceil(Int, fraction * sweeps)), length(schedule))]
    comparison_fractions = unique!(sort!(vcat(
        [sweep // controls.sweeps for sweep in 1:controls.sweeps],
        [sweep // reference.controls.sweeps
         for sweep in 1:reference.controls.sweeps],
    )))
    schedule_no_worse = all(comparison_fractions) do fraction
        cap_at_fraction(
            reference_schedule, reference.controls.sweeps, fraction) >=
        cap_at_fraction(current_schedule, controls.sweeps, fraction)
    end
    schedule_strict = any(comparison_fractions) do fraction
        cap_at_fraction(
            reference_schedule, reference.controls.sweeps, fraction) >
        cap_at_fraction(current_schedule, controls.sweeps, fraction)
    end
    no_worse = reference_step <= actual_time_step &&
               reference.controls.cutoff <= controls.cutoff &&
               schedule_no_worse
    strictly_better = reference_step < actual_time_step ||
                      reference.controls.cutoff < controls.cutoff ||
                      schedule_strict
    no_worse && strictly_better || throw(ArgumentError(
        "reference must use no coarser time step or cutoff and no smaller " *
        "bond cap at any normalized imaginary-time sweep, with at least " *
        "one control tightened."))
    return nothing
end

function _validate_gibbs_reference_target(
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    beta_phys::T,
    beta_alg::T,
    beta_frame::T,
    coordinate_rescaling_factor::T,
    reference::GibbsPurificationMPS,
) where {T<:AbstractFloat}
    diagnostics = reference.diagnostics
    tolerance = T(128) * eps(T) * max(
        one(T), abs(beta_phys), abs(beta_alg), abs(beta_frame),
        abs(coordinate_rescaling_factor))
    isapprox(T(diagnostics.beta_phys), beta_phys;
             atol=tolerance, rtol=tolerance) &&
        isapprox(T(diagnostics.beta_alg), beta_alg;
                 atol=tolerance, rtol=tolerance) &&
        isapprox(T(diagnostics.beta_frame), beta_frame;
                 atol=tolerance, rtol=tolerance) &&
        isapprox(T(diagnostics.coordinate_rescaling_factor),
                 coordinate_rescaling_factor;
                 atol=tolerance, rtol=tolerance) || throw(ArgumentError(
            "a stricter Gibbs reference must use the same physical and " *
            "algorithm inverse temperatures and coordinate rescaling."))
    diagnostics.hamiltonian_frame == hamiltonian.coordinate_frame &&
        diagnostics.identity_shift == hamiltonian.global_shift &&
        diagnostics.scale_provenance == hamiltonian.scale_provenance ||
        throw(ArgumentError(
            "a stricter Gibbs reference must preserve the Hamiltonian " *
            "frame, identity gauge, and scale provenance."))
    isequal(reference.hamiltonian_specification, hamiltonian) ||
        throw(ArgumentError(
            "a stricter Gibbs reference must use the same local " *
            "Hamiltonian specification."))
    return nothing
end

function _gibbs_mps_comparison(
    state::ITensorMPS.MPS,
    sites::Vector{ITensors.Index{Int}},
    energy::T,
    energy_density::T,
    reduced::Vector{Matrix{Complex{T}}},
    expectations::Vector{T},
    correlations::Vector{T},
    controls::QuantumFurnace.GibbsMPSControls,
    actual_time_step::T,
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    beta_phys::T,
    beta_alg::T,
    beta_frame::T,
    coordinate_rescaling_factor::T,
    reference::GibbsPurificationMPS,
) where {T<:AbstractFloat}
    reference.sites == sites || throw(ArgumentError(
        "a stricter Gibbs reference must use the same fused site indices; " *
        "pass reference.sites through the sites keyword."))
    _validate_gibbs_reference_target(
        hamiltonian,
        beta_phys,
        beta_alg,
        beta_frame,
        coordinate_rescaling_factor,
        reference,
    )
    _strictness_check(controls, actual_time_step, reference)
    overlap = ITensorMPS.inner(reference.state, state)
    infidelity = max(zero(T), one(T) - T(abs2(overlap)))
    local_distance = maximum(
        _trace_distance(reduced[index] -
                        reference.diagnostics.one_site_reduced_states[index])
        for index in eachindex(reduced)
    )
    expectation_difference = maximum(abs.(
        expectations .- reference.diagnostics.probe_expectations))
    correlation_difference = isempty(correlations) ? zero(T) : maximum(abs.(
        correlations .-
        reference.diagnostics.nearest_neighbor_probe_correlations))
    return GibbsMPSComparison{T}(
        infidelity,
        local_distance,
        T(abs(energy - reference.diagnostics.energy)),
        T(abs(energy_density - reference.diagnostics.energy_density)),
        T(expectation_difference),
        T(correlation_difference),
    )
end

function _expanded_maxdim_schedule(
    schedule::Tuple,
    sweeps::Int,
)
    return Int[schedule[min(sweep, length(schedule))] for sweep in 1:sweeps]
end

function _prepare_gibbs_purification(
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    beta_phys::T,
    beta_alg::T,
    beta_frame::T,
    coordinate_rescaling_factor::T,
    controls::C;
    sites=nothing,
    reference::Union{Nothing, GibbsPurificationMPS}=nothing,
) where {
    T<:AbstractFloat,
    C<:QuantumFurnace.GibbsMPSControls,
}
    QuantumFurnace._validate_local_hamiltonian_integrity(hamiltonian)
    hamiltonian.boundary == :open || throw(ArgumentError(
        "Gibbs-MPS preparation supports open boundaries only."))
    for (value, label) in (
        (beta_phys, "beta_phys"),
        (beta_alg, "beta_alg"),
        (beta_frame, "beta_frame"),
    )
        QuantumFurnace._require_finite_nonnegative(value, label)
    end
    QuantumFurnace._require_finite_positive(
        coordinate_rescaling_factor, "coordinate rescaling factor")
    length(controls.maxdim_schedule) <= controls.sweeps || throw(ArgumentError(
        "maxdim_schedule cannot contain more entries than TDVP sweeps."))

    fused_sites = sites === nothing ?
        fused_siteinds(
            hamiltonian.num_sites; physical_dim=hamiltonian.local_dim) :
        _validate_gibbs_sites(
            sites, hamiltonian.num_sites, hamiltonian.local_dim)
    fused_sites = _validate_gibbs_sites(
        fused_sites, hamiltonian.num_sites, hamiltonian.local_dim)
    ket_hamiltonian = _ket_hamiltonian_mpo(hamiltonian, fused_sites)
    state = _normalized_fused_bell_mps(
        T, fused_sites, hamiltonian.local_dim)

    preparation_method = beta_frame == zero(T) ? :fused_bell :
        hamiltonian.num_sites == 1 ? :exact_one_site : :tdvp
    actual_time_step = zero(T)
    applied_sweeps = 0
    applied_cutoff = nothing
    applied_maxdim_schedule = Int[]
    if beta_frame > zero(T) && hamiltonian.num_sites == 1
        # ITensorMPS intentionally rejects TDVP on a one-site chain. This
        # exact local fallback exponentiates only the d-by-d on-site matrix;
        # it is not a dense many-body preparation path.
        state = _exact_one_site_gibbs_mps(
            hamiltonian, beta_frame, first(fused_sites))
    elseif beta_frame > zero(T)
        actual_time_step = beta_frame / T(2 * controls.sweeps)
        step_tolerance = T(128) * eps(T) *
                         max(one(T), actual_time_step, T(controls.time_step))
        actual_time_step <= T(controls.time_step) + step_tolerance ||
            throw(ArgumentError(
                "sweeps=$(controls.sweeps) gives imaginary-time step " *
                "$actual_time_step, exceeding controls.time_step=" *
                "$(controls.time_step); increase sweeps or the declared step."))
        applied_sweeps = controls.sweeps
        applied_cutoff = T(controls.cutoff)
        applied_maxdim_schedule = _expanded_maxdim_schedule(
            controls.maxdim_schedule, controls.sweeps)
        state = ITensorMPS.tdvp(
            ket_hamiltonian,
            -beta_frame / T(2),
            state;
            time_step=-actual_time_step,
            nsteps=controls.sweeps,
            maxdim=applied_maxdim_schedule,
            cutoff=applied_cutoff,
            normalize=true,
            nsite=min(2, hamiltonian.num_sites),
            outputlevel=0,
        )
        normalize!(state)
    end

    state_norm = T(norm(state))
    energy_without_shift = T(real(ITensorMPS.inner(
        state', ket_hamiltonian, state)))
    energy = energy_without_shift + hamiltonian.global_shift
    energy_density = energy / T(hamiltonian.num_sites)
    reduced = _one_site_reduced_states(
        state, hamiltonian.local_dim, T)
    local_hermiticity, local_trace = _local_state_defects(reduced)
    expectations, correlations = _probe_diagnostics(
        state, hamiltonian.local_dim, T)
    dense = _dense_gibbs_verification(
        state, fused_sites, hamiltonian, beta_frame, energy)
    comparison = reference === nothing ? nothing : _gibbs_mps_comparison(
        state,
        fused_sites,
        energy,
        energy_density,
        reduced,
        expectations,
        correlations,
        controls,
        actual_time_step,
        hamiltonian,
        beta_phys,
        beta_alg,
        beta_frame,
        coordinate_rescaling_factor,
        reference,
    )
    diagnostics = GibbsPurificationDiagnostics{T}(
        beta_phys,
        beta_alg,
        beta_frame,
        hamiltonian.coordinate_frame,
        coordinate_rescaling_factor,
        hamiltonian.global_shift,
        hamiltonian.scale_provenance,
        preparation_method,
        T(controls.time_step),
        actual_time_step,
        applied_sweeps,
        applied_cutoff,
        applied_maxdim_schedule,
        state_norm,
        Int(ITensorMPS.maxlinkdim(state)),
        energy,
        energy_density,
        reduced,
        local_hermiticity,
        local_trace,
        expectations,
        correlations,
        dense,
        comparison,
    )
    return GibbsPurificationMPS{
        T, typeof(state), typeof(ket_hamiltonian), C,
    }(
        state,
        fused_sites,
        ket_hamiltonian,
        hamiltonian,
        controls,
        diagnostics,
    )
end

"""
    prepare_gibbs_purification(hamiltonian, beta_frame, controls; ...)

Prepare `exp(-beta_frame * H / 2)|I>` by imaginary-time TDVP on the ket
copy of a normalized fused Bell-pair MPS. `controls.sweeps` spans the fixed
total imaginary time, so the actual step is `beta_frame/(2*sweeps)` and must
not exceed `controls.time_step`. The Hamiltonian identity shift is omitted
during evolution and restored only in reported energies.

At beta zero the Bell state is returned directly. A one-site chain uses an
exact local fallback because ITensorMPS does not define one-site TDVP. These
non-TDVP paths record zero applied sweeps and timestep, no cutoff, and an empty
applied bond schedule.

Passing `reference` measures convergence against an explicitly stricter
preparation on the same fused site indices. Dense verification is performed
automatically only within the guarded `N <= 4` overlap regime.
"""
function QuantumFurnace.prepare_gibbs_purification(
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    beta_frame::Real,
    controls::QuantumFurnace.GibbsMPSControls;
    sites=nothing,
    reference::Union{Nothing, GibbsPurificationMPS}=nothing,
) where {T<:AbstractFloat}
    metadata = _direct_beta_metadata(hamiltonian, T(beta_frame))
    return _prepare_gibbs_purification(
        hamiltonian,
        metadata.beta_phys,
        metadata.beta_alg,
        metadata.beta_frame,
        metadata.coordinate_rescaling_factor,
        controls;
        sites,
        reference,
    )
end

"""
    prepare_gibbs_purification(config, hamiltonian, controls; ...)

Frame-checked Gibbs-MPS preparation. A physical-frame Hamiltonian uses
`(H_phys, beta_phys)` and an algorithm-frame Hamiltonian uses
`(H_alg, beta_alg)` after validating `beta_alg = R * beta_phys`.
"""
function QuantumFurnace.prepare_gibbs_purification(
    config::QuantumFurnace.Config{
        QuantumFurnace.TensorNetworkSpectrum,
        QuantumFurnace.BohrDomain,
        QuantumFurnace.DLL,
        T,
    },
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    controls::QuantumFurnace.GibbsMPSControls;
    sites=nothing,
    reference::Union{Nothing, GibbsPurificationMPS}=nothing,
) where {T<:AbstractFloat}
    metadata = _config_beta_metadata(config, hamiltonian)
    return _prepare_gibbs_purification(
        hamiltonian,
        metadata.beta_phys,
        metadata.beta_alg,
        metadata.beta_frame,
        metadata.coordinate_rescaling_factor,
        controls;
        sites,
        reference,
    )
end
