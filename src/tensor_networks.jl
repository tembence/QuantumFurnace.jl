# Dependency-free contracts for one-dimensional DLL parent tensor networks.

"""Machine-readable mathematical targets supported by the DLL parent API."""
const DLL_PARENT_TARGET_LABELS = (
    :exact_bohr_dense,
    :bohr_polynomial_full_chain,
    :finite_patch_bohr_surrogate,
)

"""Machine-readable evidence labels for a reported parent spacing or gap."""
const DLL_PARENT_GAP_LABELS = (
    :exact_gap,
    :variational_upper_estimate,
    :observed_manifold_spacing,
    :certified_bracket,
)

"""Evidence classes for numerical or certified kernel completeness."""
const DLL_KERNEL_EVIDENCE_LABELS = (
    :exact_dense_complete,
    :verified_complete,
    :observed_only,
)

"""Evidence classes admitted by [`ParentErrorBound`](@ref)."""
const PARENT_ERROR_EVIDENCE_LABELS = (
    :rigorous_bound,
    :floating_point_norm,
    :sampled_evidence,
    :unmeasured,
)

"""
    build_dll_parent(args...; kwargs...)

Build a complete DLL KMS parent through an optional tensor-network backend.
Core declares the function without importing a backend package.
"""
function build_dll_parent end

"""
    prepare_gibbs_purification(args...; kwargs...)

Prepare an equilibrium purification independently of the DLL parent through
an optional tensor-network backend.
"""
function prepare_gibbs_purification end

"""
    solve_dll_parent_gap(args...; kwargs...)

Solve for the kernel-aware low-energy spectrum of a represented DLL parent
through an optional tensor-network backend.
"""
function solve_dll_parent_gap end

"""
    verify_dll_parent(args...; kwargs...)

Verify fixed-point, kernel, representation, and gap evidence for a represented
DLL parent through an optional tensor-network backend.
"""
function verify_dll_parent end

@inline function _require_tn_label(
    label::Symbol,
    allowed::Tuple,
    kind::AbstractString,
)
    label in allowed || throw(ArgumentError(
        "$kind must be one of $(collect(allowed)), got $label."))
    return label
end

@inline function _require_finite_nonnegative(value::T, name::AbstractString) where
        {T<:AbstractFloat}
    isfinite(value) && value >= zero(T) || throw(ArgumentError(
        "$name must be finite and >= 0, got $value."))
    return value
end

@inline function _require_finite_positive(value::T, name::AbstractString) where
        {T<:AbstractFloat}
    isfinite(value) && value > zero(T) || throw(ArgumentError(
        "$name must be finite and > 0, got $value."))
    return value
end

function _tn_float_type(values::Real...)
    T = promote_type(map(value -> typeof(float(value)), values)...)
    T <: AbstractFloat || throw(ArgumentError(
        "tensor-network numerical controls must promote to AbstractFloat, got $T."))
    return T
end

"""
    BohrMPOControls(; ...)

Backend-neutral polynomial, compression, and spatial controls for a DLL parent
MPO. A finite-patch target requires a positive `patch_radius`; the other
targets reject one.
"""
struct BohrMPOControls{T<:AbstractFloat}
    target_label::Symbol
    scalar_tolerance::T
    recurrence_cutoff::T
    recurrence_maxdim::Int
    product_cutoff::T
    product_maxdim::Int
    sum_cutoff::T
    sum_maxdim::Int
    patch_radius::Union{Nothing, Int}

    function BohrMPOControls{T}(
        target_label::Symbol,
        scalar_tolerance::T,
        recurrence_cutoff::T,
        recurrence_maxdim::Int,
        product_cutoff::T,
        product_maxdim::Int,
        sum_cutoff::T,
        sum_maxdim::Int,
        patch_radius::Union{Nothing, Int},
    ) where {T<:AbstractFloat}
        _require_tn_label(target_label, DLL_PARENT_TARGET_LABELS, "target_label")
        _require_finite_positive(scalar_tolerance, "scalar_tolerance")
        _require_finite_nonnegative(recurrence_cutoff, "recurrence_cutoff")
        _require_finite_nonnegative(product_cutoff, "product_cutoff")
        _require_finite_nonnegative(sum_cutoff, "sum_cutoff")
        recurrence_maxdim > 0 || throw(ArgumentError(
            "recurrence_maxdim must be > 0."))
        product_maxdim > 0 || throw(ArgumentError("product_maxdim must be > 0."))
        sum_maxdim > 0 || throw(ArgumentError("sum_maxdim must be > 0."))
        if target_label == :finite_patch_bohr_surrogate
            patch_radius !== nothing && patch_radius > 0 || throw(ArgumentError(
                "finite_patch_bohr_surrogate requires patch_radius > 0."))
        elseif patch_radius !== nothing
            throw(ArgumentError(
                "patch_radius is only valid for :finite_patch_bohr_surrogate."))
        end
        return new{T}(
            target_label, scalar_tolerance, recurrence_cutoff,
            recurrence_maxdim, product_cutoff, product_maxdim, sum_cutoff,
            sum_maxdim, patch_radius)
    end
end

function BohrMPOControls(;
    target_label::Symbol=:bohr_polynomial_full_chain,
    scalar_tolerance::Real=1e-10,
    recurrence_cutoff::Real=1e-12,
    recurrence_maxdim::Integer=256,
    product_cutoff::Real=1e-12,
    product_maxdim::Integer=512,
    sum_cutoff::Real=1e-12,
    sum_maxdim::Integer=512,
    patch_radius::Union{Nothing, Integer}=nothing,
)
    T = _tn_float_type(
        scalar_tolerance, recurrence_cutoff, product_cutoff, sum_cutoff)
    return BohrMPOControls{T}(
        target_label, T(scalar_tolerance), T(recurrence_cutoff),
        Int(recurrence_maxdim), T(product_cutoff), Int(product_maxdim),
        T(sum_cutoff), Int(sum_maxdim),
        patch_radius === nothing ? nothing : Int(patch_radius))
end

function _validate_bond_schedule(schedule::Tuple, label::AbstractString)
    isempty(schedule) && throw(ArgumentError("$label must be nonempty."))
    all(value -> value isa Integer && value > 0, schedule) || throw(ArgumentError(
        "$label entries must be positive integers."))
    return Tuple(Int(value) for value in schedule)
end

"""
    GibbsMPSControls(; ...)

Controls for the parent-independent imaginary-time preparation of the Gibbs
purification. These controls intentionally contain no DLL filter parameters.
"""
struct GibbsMPSControls{T<:AbstractFloat, D<:Tuple}
    time_step::T
    sweeps::Int
    cutoff::T
    maxdim_schedule::D

    function GibbsMPSControls{T, D}(
        time_step::T,
        sweeps::Int,
        cutoff::T,
        maxdim_schedule::D,
    ) where {T<:AbstractFloat, D<:Tuple}
        _require_finite_positive(time_step, "time_step")
        sweeps > 0 || throw(ArgumentError("sweeps must be > 0."))
        _require_finite_nonnegative(cutoff, "cutoff")
        schedule = _validate_bond_schedule(maxdim_schedule, "maxdim_schedule")
        return new{T, typeof(schedule)}(time_step, sweeps, cutoff, schedule)
    end
end

function GibbsMPSControls(;
    time_step::Real=0.05,
    sweeps::Integer=20,
    cutoff::Real=1e-12,
    maxdim_schedule::Tuple=(32, 64, 128),
)
    T = _tn_float_type(time_step, cutoff)
    schedule = _validate_bond_schedule(maxdim_schedule, "maxdim_schedule")
    return GibbsMPSControls{T, typeof(schedule)}(
        T(time_step), Int(sweeps), T(cutoff), schedule)
end

"""
    ParentGapControls(; ...)

Kernel-aware multi-start eigensolver controls. Penalties and sectors are
explicit tuples so backend methods can specialise without `Any` fields.
"""
struct ParentGapControls{T<:AbstractFloat, D<:Tuple, P<:Tuple, S<:Tuple}
    sweeps::Int
    cutoff::T
    maxdim_schedule::D
    num_starts::Int
    random_seed::Int
    residual_tolerance::T
    overlap_tolerance::T
    energy_tolerance::T
    penalty_factors::P
    sectors::S

    function ParentGapControls{T, D, P, S}(
        sweeps::Int,
        cutoff::T,
        maxdim_schedule::D,
        num_starts::Int,
        random_seed::Int,
        residual_tolerance::T,
        overlap_tolerance::T,
        energy_tolerance::T,
        penalty_factors::P,
        sectors::S,
    ) where {T<:AbstractFloat, D<:Tuple, P<:Tuple, S<:Tuple}
        sweeps > 0 || throw(ArgumentError("sweeps must be > 0."))
        _require_finite_nonnegative(cutoff, "cutoff")
        schedule = _validate_bond_schedule(maxdim_schedule, "maxdim_schedule")
        num_starts > 0 || throw(ArgumentError("num_starts must be > 0."))
        _require_finite_positive(residual_tolerance, "residual_tolerance")
        _require_finite_nonnegative(overlap_tolerance, "overlap_tolerance")
        overlap_tolerance <= one(T) || throw(ArgumentError(
            "overlap_tolerance must be <= 1."))
        _require_finite_positive(energy_tolerance, "energy_tolerance")
        isempty(penalty_factors) && throw(ArgumentError(
            "penalty_factors must be nonempty."))
        all(value -> value isa Real && isfinite(value) && value > 0,
            penalty_factors) || throw(ArgumentError(
                "penalty_factors must contain finite positive values."))
        penalties = Tuple(T(value) for value in penalty_factors)
        isempty(sectors) && throw(ArgumentError("sectors must be nonempty."))
        all(value -> value isa Symbol, sectors) || throw(ArgumentError(
            "sectors must contain Symbol labels."))
        return new{T, typeof(schedule), typeof(penalties), S}(
            sweeps, cutoff, schedule, num_starts, random_seed,
            residual_tolerance, overlap_tolerance, energy_tolerance,
            penalties, sectors)
    end
end

function ParentGapControls(;
    sweeps::Integer=20,
    cutoff::Real=1e-12,
    maxdim_schedule::Tuple=(32, 64, 128),
    num_starts::Integer=4,
    random_seed::Integer=0,
    residual_tolerance::Real=1e-8,
    overlap_tolerance::Real=1e-8,
    energy_tolerance::Real=1e-10,
    penalty_factors::Tuple=(0.5, 1.0, 2.0),
    sectors::Tuple=(:unrestricted,),
)
    T = _tn_float_type(
        cutoff, residual_tolerance, overlap_tolerance, energy_tolerance,
        penalty_factors...)
    schedule = _validate_bond_schedule(maxdim_schedule, "maxdim_schedule")
    penalties = Tuple(T(value) for value in penalty_factors)
    return ParentGapControls{
        T, typeof(schedule), typeof(penalties), typeof(sectors)}(
        Int(sweeps), T(cutoff), schedule, Int(num_starts), Int(random_seed),
        T(residual_tolerance), T(overlap_tolerance), T(energy_tolerance),
        penalties, sectors)
end

"""
    ParentErrorBound{T}(name; magnitude=nothing, evidence=:unmeasured, note="")

One named approximation or solver error entry. `magnitude` is absent exactly
when `evidence == :unmeasured`; numerical evidence is never promoted to a
rigorous bound by the container.
"""
struct ParentErrorBound{T<:AbstractFloat}
    name::Symbol
    magnitude::Union{Nothing, T}
    evidence::Symbol
    note::String

    function ParentErrorBound{T}(
        name::Symbol,
        magnitude::Union{Nothing, T},
        evidence::Symbol,
        note::String,
    ) where {T<:AbstractFloat}
        _require_tn_label(
            evidence, PARENT_ERROR_EVIDENCE_LABELS, "error evidence")
        if evidence == :unmeasured
            magnitude === nothing || throw(ArgumentError(
                "unmeasured error $name must not carry a magnitude."))
        else
            magnitude === nothing && throw(ArgumentError(
                "$evidence error $name requires a magnitude."))
            _require_finite_nonnegative(magnitude, "error magnitude for $name")
        end
        return new{T}(name, magnitude, evidence, note)
    end
end

function ParentErrorBound(
    ::Type{T},
    name::Symbol;
    magnitude::Union{Nothing, Real}=nothing,
    evidence::Symbol=:unmeasured,
    note::AbstractString="",
) where {T<:AbstractFloat}
    value = magnitude === nothing ? nothing : T(magnitude)
    return ParentErrorBound{T}(name, value, evidence, String(note))
end

function _validate_error_entries(entries::Tuple, ::Type{T}, label::AbstractString) where
        {T<:AbstractFloat}
    all(entry -> entry isa ParentErrorBound{T}, entries) || throw(ArgumentError(
        "$label must contain only ParentErrorBound{$T} entries."))
    return entries
end

@inline function _require_error_slot(
    entry::ParentErrorBound,
    expected_name::Symbol,
)
    entry.name == expected_name || throw(ArgumentError(
        "error-ledger slot $expected_name must contain an entry named " *
        "$expected_name, got $(entry.name)."))
    return entry
end

"""
    ParentErrorLedger(T=Float64; ...)

Typed ledger keeping every approximation layer separate. Per-recurrence,
per-product, and per-block entries are tuples whose elements retain their own
names and evidence classes.
"""
struct ParentErrorLedger{
    T<:AbstractFloat,
    R<:Tuple,
    P<:Tuple,
    B<:Tuple,
}
    spectral_interval::ParentErrorBound{T}
    scalar_polynomial::ParentErrorBound{T}
    modular_compatibility::ParentErrorBound{T}
    locality_radius::ParentErrorBound{T}
    mpo_recurrences::R
    mpo_products::P
    block_assembly::B
    total_sum_compression::ParentErrorBound{T}
    gibbs_mps::ParentErrorBound{T}
    eigensolver::ParentErrorBound{T}

    function ParentErrorLedger{T, R, P, B}(
        spectral_interval::ParentErrorBound{T},
        scalar_polynomial::ParentErrorBound{T},
        modular_compatibility::ParentErrorBound{T},
        locality_radius::ParentErrorBound{T},
        mpo_recurrences::R,
        mpo_products::P,
        block_assembly::B,
        total_sum_compression::ParentErrorBound{T},
        gibbs_mps::ParentErrorBound{T},
        eigensolver::ParentErrorBound{T},
    ) where {T<:AbstractFloat, R<:Tuple, P<:Tuple, B<:Tuple}
        _require_error_slot(spectral_interval, :spectral_interval)
        _require_error_slot(scalar_polynomial, :scalar_polynomial)
        _require_error_slot(modular_compatibility, :modular_compatibility)
        _require_error_slot(locality_radius, :locality_radius)
        _validate_error_entries(mpo_recurrences, T, "mpo_recurrences")
        _validate_error_entries(mpo_products, T, "mpo_products")
        _validate_error_entries(block_assembly, T, "block_assembly")
        _require_error_slot(total_sum_compression, :total_sum_compression)
        _require_error_slot(gibbs_mps, :gibbs_mps)
        _require_error_slot(eigensolver, :eigensolver)
        return new{T, R, P, B}(
            spectral_interval, scalar_polynomial, modular_compatibility,
            locality_radius, mpo_recurrences, mpo_products, block_assembly,
            total_sum_compression, gibbs_mps, eigensolver)
    end
end

function ParentErrorLedger(
    ::Type{T}=Float64;
    spectral_interval::ParentErrorBound{T}=
        ParentErrorBound(T, :spectral_interval),
    scalar_polynomial::ParentErrorBound{T}=
        ParentErrorBound(T, :scalar_polynomial),
    modular_compatibility::ParentErrorBound{T}=
        ParentErrorBound(T, :modular_compatibility),
    locality_radius::ParentErrorBound{T}=
        ParentErrorBound(T, :locality_radius),
    mpo_recurrences::Tuple=(),
    mpo_products::Tuple=(),
    block_assembly::Tuple=(),
    total_sum_compression::ParentErrorBound{T}=
        ParentErrorBound(T, :total_sum_compression),
    gibbs_mps::ParentErrorBound{T}=ParentErrorBound(T, :gibbs_mps),
    eigensolver::ParentErrorBound{T}=ParentErrorBound(T, :eigensolver),
) where {T<:AbstractFloat}
    recurrences = _validate_error_entries(mpo_recurrences, T, "mpo_recurrences")
    products = _validate_error_entries(mpo_products, T, "mpo_products")
    assembly = _validate_error_entries(block_assembly, T, "block_assembly")
    return ParentErrorLedger{
        T, typeof(recurrences), typeof(products), typeof(assembly)}(
        spectral_interval, scalar_polynomial, modular_compatibility,
        locality_radius, recurrences, products, assembly,
        total_sum_compression, gibbs_mps, eigensolver)
end

"""Filter support and shift recorded in the Hamiltonian coordinate frame."""
struct DLLFilterFrame{T<:AbstractFloat}
    family::Symbol
    support_radius::Union{Nothing, T}
    shift::T
    weight::T
    coordinate_frame::Symbol

    function DLLFilterFrame{T}(
        family::Symbol,
        support_radius::Union{Nothing, T},
        shift::T,
        weight::T,
        coordinate_frame::Symbol,
    ) where {T<:AbstractFloat}
        coordinate_frame in _LOCAL_1D_FRAMES || throw(ArgumentError(
            "filter coordinate_frame must be :physical or :algorithm."))
        if support_radius !== nothing
            _require_finite_positive(support_radius, "filter support_radius")
        end
        _require_finite_nonnegative(shift, "filter shift")
        _require_finite_positive(weight, "filter weight")
        return new{T}(family, support_radius, shift, weight, coordinate_frame)
    end
end

"""One explicitly named rate clock relative to the raw DLL generator."""
struct GeneratorClock{T<:AbstractFloat}
    label::Symbol
    multiplier::T
    rate::T

    function GeneratorClock{T}(
        label::Symbol,
        multiplier::T,
        rate::T,
    ) where {T<:AbstractFloat}
        label == :raw_generator && throw(ArgumentError(
            "derived clocks must not reuse the :raw_generator label."))
        _require_finite_positive(multiplier, "clock multiplier")
        _require_finite_positive(rate, "clock rate")
        return new{T}(label, multiplier, rate)
    end
end

"""
    DLLParentProvenance

Frame, filter, proposal-normalisation, and generator-clock provenance retained
with every tensor-network result.
"""
struct DLLParentProvenance{T<:AbstractFloat}
    beta_phys::T
    beta_alg::T
    hamiltonian_frame::Symbol
    rescaling_factor::T
    energy_shift::T
    filters::Vector{DLLFilterFrame{T}}
    raw_generator_rate::T
    proposal_normalisation::Symbol
    proposal_amplitude::T
    sweep_clock::Union{Nothing, GeneratorClock{T}}
    legacy_clock::Union{Nothing, GeneratorClock{T}}

    function DLLParentProvenance{T}(
        beta_phys::T,
        beta_alg::T,
        hamiltonian_frame::Symbol,
        rescaling_factor::T,
        energy_shift::T,
        filters::Vector{DLLFilterFrame{T}},
        raw_generator_rate::T,
        proposal_normalisation::Symbol,
        proposal_amplitude::T,
        sweep_clock::Union{Nothing, GeneratorClock{T}},
        legacy_clock::Union{Nothing, GeneratorClock{T}},
    ) where {T<:AbstractFloat}
        _require_finite_positive(beta_phys, "beta_phys")
        _require_finite_positive(beta_alg, "beta_alg")
        hamiltonian_frame in _LOCAL_1D_FRAMES || throw(ArgumentError(
            "hamiltonian_frame must be :physical or :algorithm."))
        _require_finite_positive(rescaling_factor, "rescaling_factor")
        hamiltonian_frame == :physical && rescaling_factor != one(T) &&
            throw(ArgumentError(
                "physical-frame provenance must use the identity " *
                "rescaling_factor=1; use :algorithm for a nontrivial R."))
        isfinite(energy_shift) || throw(ArgumentError(
            "energy_shift must be finite."))
        expected_beta_alg = beta_phys * rescaling_factor
        consistency_rtol = max(T(1e-10), T(100) * eps(T))
        isapprox(beta_alg, expected_beta_alg;
                 atol=zero(T), rtol=consistency_rtol) || throw(ArgumentError(
            "provenance requires beta_alg == beta_phys * rescaling_factor."))
        isempty(filters) && throw(ArgumentError(
            "provenance requires at least one filter record."))
        all(filter -> filter.coordinate_frame == hamiltonian_frame, filters) ||
            throw(ArgumentError(
                "every filter record must use the Hamiltonian coordinate frame."))
        _require_finite_positive(raw_generator_rate, "raw_generator_rate")
        proposal_normalisation in (:qf_averaged, :extensive) ||
            throw(ArgumentError(
                "proposal_normalisation must be :qf_averaged or :extensive."))
        _require_finite_positive(proposal_amplitude, "proposal_amplitude")
        for clock in (sweep_clock, legacy_clock)
            clock === nothing && continue
            expected_rate = raw_generator_rate * clock.multiplier
            isapprox(clock.rate, expected_rate;
                     atol=zero(T), rtol=T(32) * eps(T)) || throw(ArgumentError(
                "clock $(clock.label) rate must equal raw_generator_rate times its multiplier."))
        end
        if sweep_clock !== nothing
            proposal_normalisation == :qf_averaged || throw(ArgumentError(
                "a sweep clock requires :qf_averaged proposal normalisation."))
            sweep_clock.label == :sweep_normalised || throw(ArgumentError(
                "sweep_clock must use label=:sweep_normalised."))
        end
        legacy_clock !== nothing && legacy_clock.label == :sweep_normalised &&
            throw(ArgumentError(
                "legacy_clock must use its own explicit label, not :sweep_normalised."))
        return new{T}(
            beta_phys, beta_alg, hamiltonian_frame, rescaling_factor,
            energy_shift, filters, raw_generator_rate,
            proposal_normalisation, proposal_amplitude, sweep_clock,
            legacy_clock)
    end
end

function _filter_frame(
    filter::DLLGaussianFilter,
    frame::Symbol,
    ::Type{T},
) where {T<:AbstractFloat}
    return DLLFilterFrame{T}(
        :dll_gaussian, nothing, zero(T), one(T), frame)
end

function _filter_frame(
    filter::DLLMetropolisFilter,
    frame::Symbol,
    ::Type{T},
) where {T<:AbstractFloat}
    return DLLFilterFrame{T}(
        :dll_metropolis, T(filter.S), zero(T), one(T), frame)
end

function _filter_frame(
    filter::ShiftedSymmetricFilter,
    frame::Symbol,
    ::Type{T},
) where {T<:AbstractFloat}
    base = _filter_frame(filter.base, frame, T)
    return DLLFilterFrame{T}(
        Symbol("shifted_", base.family), base.support_radius,
        T(filter.shift), T(filter.weight), frame)
end

function _config_dll_channels(config::Config)
    config.filter === nothing && throw(ArgumentError(
        "DLL tensor-network configuration requires an explicit filter."))
    return _flatten_local_dll_channels((config.filter,))
end

function _validate_block_filters_match_config(
    config::Config,
    blocks::AbstractVector{<:LocalDLLBlock1D},
)
    expected = _config_dll_channels(config)
    for (index, block) in pairs(blocks)
        channels = getfield(block, :channels)
        isequal(channels, expected) || throw(ArgumentError(
            "LocalDLLBlock1D $index channels do not match Config.filter. " *
            "The validated Config is the authoritative filter source."))
    end
    return expected
end

function _validate_standard_pauli_sources(
    blocks::AbstractVector{<:LocalDLLBlock1D},
    num_sites::Int,
    normalisation::Symbol,
)
    normalisation in (:qf_averaged, :extensive) || throw(ArgumentError(
        "proposal_normalisation must be :qf_averaged or :extensive."))
    expected = local_pauli_jumps_1d(num_sites; boundary=:open)
    length(blocks) == length(expected) || throw(ArgumentError(
        "$normalisation Pauli proposal requires exactly 3N=$(3num_sites) blocks."))
    expected_amplitude = normalisation == :qf_averaged ?
        inv(sqrt(3 * num_sites)) : 1.0
    for index in eachindex(blocks, expected)
        actual = getfield(blocks[index], :source)
        reference = expected[index]
        _local_sites(actual) == _local_sites(reference) || throw(ArgumentError(
            "block $index does not follow the standard (X,Y,Z)-major Pauli support order."))
        isapprox(_local_matrix(actual), _local_matrix(reference);
                 atol=0, rtol=0) || throw(ArgumentError(
            "block $index is not the expected standard Pauli source."))
        actual_real_type = typeof(real(actual.coefficient))
        expected_real_type = typeof(expected_amplitude)
        R = promote_type(actual_real_type, expected_real_type)
        coefficient_rtol = R(32) * R(max(
            eps(actual_real_type), eps(expected_real_type)))
        isapprox(Complex{R}(actual.coefficient), Complex{R}(expected_amplitude);
                 atol=zero(R), rtol=coefficient_rtol) || throw(ArgumentError(
            "block $index coefficient $(actual.coefficient) is inconsistent with " *
            "proposal_normalisation=$normalisation."))
    end
    return expected_amplitude
end

"""
    validate_dll_tensor_network(config, hamiltonian, blocks, controls;
                                include_coherent=true)

Validate the complete dependency-free input contract for the one-dimensional
DLL parent backend. The scalable polynomial targets currently accept only an
unshifted `DLLGaussianFilter`; exact dense overlap permits every admissible DLL
filter already supported by the dense reference.
"""
function validate_dll_tensor_network(
    config::Config,
    hamiltonian::LocalHamiltonian1D,
    blocks::AbstractVector{<:LocalDLLBlock1D},
    controls::BohrMPOControls;
    include_coherent::Bool=true,
)
    config isa Config{TensorNetworkSpectrum, BohrDomain, DLL} ||
        throw(ArgumentError(
            "DLL tensor networks require Config{TensorNetworkSpectrum,BohrDomain,DLL}."))
    validate_config!(config, hamiltonian)
    validate_local_dll_tensor_network(hamiltonian, blocks)
    include_coherent || throw(ArgumentError(
        "DLL parent construction requires the canonical coherent correction; " *
        "a dissipator-only parent is unsupported."))
    with_coherent(config.construction) || throw(ArgumentError(
        "DLL parent construction requires a coherent detailed-balance construction."))
    channels = _validate_block_filters_match_config(config, blocks)
    all(channel -> !(channel isa ShiftedSymmetricFilter) || channel.shift >= 0,
        channels) || throw(ArgumentError(
            "ShiftedSymmetricFilter centre shifts must be nonnegative."))
    if controls.target_label != :exact_bohr_dense &&
       !(length(channels) == 1 && only(channels) isa DLLGaussianFilter)
        throw(ArgumentError(
            "$(controls.target_label) currently supports only unshifted " *
            "single-channel DLLGaussianFilter blocks; use :exact_bohr_dense for overlap " *
            "validation of other admissible filters."))
    end
    return nothing
end

"""
    dll_parent_provenance(config, hamiltonian, blocks; ...)

Construct validated unit and rate metadata. No Hamiltonian rescaling is ever
converted into a DLL clock automatically. A legacy/global clock is recorded
only when both an explicit label and multiplier are supplied. Physical-frame
local data use the identity active-coordinate scale `R=1`; a nontrivial `R` is
retained on an explicitly converted algorithm-frame Hamiltonian.
"""
function dll_parent_provenance(
    config::Config,
    hamiltonian::LocalHamiltonian1D,
    blocks::AbstractVector{<:LocalDLLBlock1D};
    controls::BohrMPOControls=BohrMPOControls(),
    raw_generator_rate::Real=1,
    proposal_normalisation::Symbol=:qf_averaged,
    include_sweep_clock::Bool=false,
    legacy_clock_label::Union{Nothing, Symbol}=nothing,
    legacy_clock_multiplier::Union{Nothing, Real}=nothing,
)
    validate_dll_tensor_network(config, hamiltonian, blocks, controls)
    amplitude = _validate_standard_pauli_sources(
        blocks, hamiltonian.num_sites, proposal_normalisation)
    T = typeof(config.beta)
    raw_rate = T(raw_generator_rate)
    _require_finite_positive(raw_rate, "raw_generator_rate")
    channels = _config_dll_channels(config)
    filters = DLLFilterFrame{T}[
        _filter_frame(channel, hamiltonian.coordinate_frame, T)
        for channel in channels
    ]
    sweep_clock = if include_sweep_clock
        proposal_normalisation == :qf_averaged || throw(ArgumentError(
            "the 3N sweep clock is defined only for :qf_averaged proposals."))
        multiplier = T(3 * hamiltonian.num_sites)
        GeneratorClock{T}(:sweep_normalised, multiplier, raw_rate * multiplier)
    else
        nothing
    end
    xor(legacy_clock_label === nothing, legacy_clock_multiplier === nothing) &&
        throw(ArgumentError(
            "legacy_clock_label and legacy_clock_multiplier must be supplied together."))
    legacy_clock = if legacy_clock_label === nothing
        nothing
    else
        multiplier = T(legacy_clock_multiplier)
        GeneratorClock{T}(
            legacy_clock_label, multiplier, raw_rate * multiplier)
    end
    return DLLParentProvenance{T}(
        T(config.beta_phys), T(config.beta), hamiltonian.coordinate_frame,
        T(hamiltonian.rescaling_factor), T(hamiltonian.global_shift), filters,
        raw_rate, proposal_normalisation, T(amplitude), sweep_clock,
        legacy_clock)
end

"""A complete parent object and its separately retained channel blocks."""
struct DLLParentBundle{P, L<:ParentErrorLedger}
    target_label::Symbol
    total_parent::P
    block_parents::Vector{P}
    block_keys::Vector{Tuple{Int, Int}}
    coherent_correction_included::Bool
    error_ledger::L

    function DLLParentBundle{P, L}(
        target_label::Symbol,
        total_parent::P,
        block_parents::Vector{P},
        block_keys::Vector{Tuple{Int, Int}},
        coherent_correction_included::Bool,
        error_ledger::L,
    ) where {P, L<:ParentErrorLedger}
        isconcretetype(P) || throw(ArgumentError(
            "backend parent type must be concrete, got $P."))
        _require_tn_label(target_label, DLL_PARENT_TARGET_LABELS, "target_label")
        coherent_correction_included || throw(ArgumentError(
            "a DLLParentBundle must contain the canonical coherent correction."))
        isempty(block_parents) && throw(ArgumentError(
            "a DLLParentBundle requires at least one channel block."))
        length(block_parents) == length(block_keys) || throw(ArgumentError(
            "block_parents and block_keys must have equal length."))
        all(key -> key[1] > 0 && key[2] > 0, block_keys) ||
            throw(ArgumentError(
                "block keys must contain positive source and channel indices."))
        allunique(block_keys) || throw(ArgumentError("block keys must be unique."))
        if target_label != :exact_bohr_dense
            isempty(error_ledger.mpo_recurrences) && throw(ArgumentError(
                "an approximate parent requires explicit per-recurrence error entries."))
            isempty(error_ledger.mpo_products) && throw(ArgumentError(
                "an approximate parent requires explicit per-product error entries."))
            length(error_ledger.block_assembly) == length(block_parents) ||
                throw(ArgumentError(
                    "an approximate parent requires one block-assembly error " *
                    "entry per parent block."))
        end
        return new{P, L}(
            target_label, total_parent, block_parents, block_keys, true,
            error_ledger)
    end
end

function DLLParentBundle(
    target_label::Symbol,
    total_parent::P,
    block_parents::AbstractVector{P},
    block_keys::AbstractVector{<:Tuple{Int, Int}},
    coherent_correction_included::Bool,
    error_ledger::L,
) where {P, L<:ParentErrorLedger}
    keys = Tuple{Int, Int}[Tuple(key) for key in block_keys]
    return DLLParentBundle{P, L}(
        target_label, total_parent, collect(block_parents), keys,
        coherent_correction_included, error_ledger)
end

"""One backend-specific low-energy state with independent residual evidence."""
struct DLLParentLowEnergyState{S, T<:AbstractFloat}
    state::S
    energy::T
    residual::T
    variance::T
    kernel_overlaps::Vector{T}
    bond_dimensions::Vector{Int}
    start_index::Int
    sector::Symbol
    converged::Bool

    function DLLParentLowEnergyState{S, T}(
        state::S,
        energy::T,
        residual::T,
        variance::T,
        kernel_overlaps::Vector{T},
        bond_dimensions::Vector{Int},
        start_index::Int,
        sector::Symbol,
        converged::Bool,
    ) where {S, T<:AbstractFloat}
        isconcretetype(S) || throw(ArgumentError(
            "backend state type must be concrete, got $S."))
        isfinite(energy) || throw(ArgumentError("energy must be finite."))
        _require_finite_nonnegative(residual, "residual")
        _require_finite_nonnegative(variance, "variance")
        all(value -> isfinite(value) && zero(T) <= value <= one(T),
            kernel_overlaps) || throw(ArgumentError(
                "kernel_overlaps must lie in [0, 1]."))
        all(>(0), bond_dimensions) || throw(ArgumentError(
            "bond_dimensions must be positive."))
        start_index > 0 || throw(ArgumentError("start_index must be > 0."))
        return new{S, T}(
            state, energy, residual, variance, kernel_overlaps,
            bond_dimensions, start_index, sector, converged)
    end
end

function DLLParentLowEnergyState(
    state::S;
    energy::Real,
    residual::Real,
    variance::Real,
    kernel_overlaps::AbstractVector{<:Real}=Float64[],
    bond_dimensions::AbstractVector{<:Integer}=Int[],
    start_index::Integer=1,
    sector::Symbol=:unrestricted,
    converged::Bool,
) where {S}
    overlap_eltype = eltype(kernel_overlaps)
    isconcretetype(overlap_eltype) || throw(ArgumentError(
        "kernel_overlaps must have a concrete real element type."))
    T = _tn_float_type(
        energy, residual, variance, float(zero(overlap_eltype)))
    energy_T = T(energy)
    residual_T = T(residual)
    variance_T = T(variance)
    overlaps = T.(kernel_overlaps)
    bonds = Int.(bond_dimensions)
    return DLLParentLowEnergyState{S, T}(
        state, energy_T, residual_T, variance_T, overlaps, bonds,
        Int(start_index), sector, converged)
end

"""Fixed-point, kernel, and structural diagnostics for a represented parent."""
struct DLLParentDiagnostics{T<:AbstractFloat}
    observed_kernel_dimension::Int
    kernel_complete::Bool
    kernel_tolerance::T
    kernel_evidence::Symbol
    primitivity_established::Bool
    primitivity_provenance::Symbol
    hermiticity_defect::T
    minimum_energy::T
    gibbs_energy::T
    gibbs_residual::T
    block_gibbs_residuals::Vector{T}
    tighter_parent_residual::Union{Nothing, T}

    function DLLParentDiagnostics{T}(
        observed_kernel_dimension::Int,
        kernel_complete::Bool,
        kernel_tolerance::T,
        kernel_evidence::Symbol,
        primitivity_established::Bool,
        primitivity_provenance::Symbol,
        hermiticity_defect::T,
        minimum_energy::T,
        gibbs_energy::T,
        gibbs_residual::T,
        block_gibbs_residuals::Vector{T},
        tighter_parent_residual::Union{Nothing, T},
    ) where {T<:AbstractFloat}
        observed_kernel_dimension >= 0 || throw(ArgumentError(
            "observed_kernel_dimension must be >= 0."))
        _require_finite_nonnegative(kernel_tolerance, "kernel_tolerance")
        _require_tn_label(
            kernel_evidence, DLL_KERNEL_EVIDENCE_LABELS, "kernel_evidence")
        kernel_complete && observed_kernel_dimension == 0 && throw(ArgumentError(
            "a complete kernel must contain at least one mode."))
        kernel_complete == (kernel_evidence != :observed_only) ||
            throw(ArgumentError(
                "kernel_complete requires exact or verified completeness evidence; " *
                "incomplete kernels require kernel_evidence=:observed_only."))
        primitivity_established && !kernel_complete && throw(ArgumentError(
            "primitivity cannot be established when the kernel is incomplete."))
        primitivity_established && observed_kernel_dimension != 1 &&
            throw(ArgumentError(
                "a primitive parent must have observed_kernel_dimension == 1."))
        _require_finite_nonnegative(hermiticity_defect, "hermiticity_defect")
        isfinite(minimum_energy) || throw(ArgumentError(
            "minimum_energy must be finite."))
        isfinite(gibbs_energy) || throw(ArgumentError(
            "gibbs_energy must be finite."))
        _require_finite_nonnegative(gibbs_residual, "gibbs_residual")
        all(value -> isfinite(value) && value >= zero(T),
            block_gibbs_residuals) || throw(ArgumentError(
                "block_gibbs_residuals must be finite and >= 0."))
        tighter_parent_residual === nothing || _require_finite_nonnegative(
            tighter_parent_residual, "tighter_parent_residual")
        return new{T}(
            observed_kernel_dimension, kernel_complete, kernel_tolerance,
            kernel_evidence, primitivity_established,
            primitivity_provenance, hermiticity_defect, minimum_energy,
            gibbs_energy, gibbs_residual, block_gibbs_residuals,
            tighter_parent_residual)
    end
end

function DLLParentDiagnostics(;
    observed_kernel_dimension::Integer,
    kernel_complete::Bool,
    kernel_tolerance::Real,
    kernel_evidence::Symbol,
    primitivity_established::Bool,
    primitivity_provenance::Symbol,
    hermiticity_defect::Real,
    minimum_energy::Real,
    gibbs_energy::Real,
    gibbs_residual::Real,
    block_gibbs_residuals::AbstractVector{<:Real},
    tighter_parent_residual::Union{Nothing, Real}=nothing,
)
    residual_eltype = eltype(block_gibbs_residuals)
    isconcretetype(residual_eltype) || throw(ArgumentError(
        "block_gibbs_residuals must have a concrete real element type."))
    type_values = tighter_parent_residual === nothing ?
        (kernel_tolerance, hermiticity_defect, minimum_energy, gibbs_energy,
         gibbs_residual, float(zero(residual_eltype))) :
        (kernel_tolerance, hermiticity_defect, minimum_energy, gibbs_energy,
         gibbs_residual, float(zero(residual_eltype)), tighter_parent_residual)
    T = _tn_float_type(type_values...)
    kernel_tolerance_T = T(kernel_tolerance)
    hermiticity_T = T(hermiticity_defect)
    gibbs_residual_T = T(gibbs_residual)
    minimum_T = T(minimum_energy)
    gibbs_energy_T = T(gibbs_energy)
    block_residuals = T.(block_gibbs_residuals)
    tighter = tighter_parent_residual === nothing ? nothing : T(tighter_parent_residual)
    return DLLParentDiagnostics{T}(
        Int(observed_kernel_dimension), kernel_complete, kernel_tolerance_T,
        kernel_evidence, primitivity_established, primitivity_provenance,
        hermiticity_T, minimum_T, gibbs_energy_T, gibbs_residual_T,
        block_residuals, tighter)
end

function _validate_result_provenance(
    config::Config{TensorNetworkSpectrum, BohrDomain, DLL, T},
    provenance::DLLParentProvenance{T},
) where {T<:AbstractFloat}
    consistency_rtol = max(T(1e-10), T(100) * eps(T))
    isapprox(provenance.beta_alg, config.beta;
             atol=zero(T), rtol=consistency_rtol) || throw(ArgumentError(
        "result provenance beta_alg must match Config.beta."))
    isapprox(provenance.beta_phys, config.beta_phys;
             atol=zero(T), rtol=consistency_rtol) || throw(ArgumentError(
        "result provenance beta_phys must match Config.beta_phys."))
    expected_filters = DLLFilterFrame{T}[
        _filter_frame(channel, provenance.hamiltonian_frame, T)
        for channel in _config_dll_channels(config)
    ]
    isequal(provenance.filters, expected_filters) || throw(ArgumentError(
        "result provenance filters must match Config.filter in the recorded frame."))
    expected_amplitude = provenance.proposal_normalisation == :qf_averaged ?
        inv(sqrt(T(3 * config.num_qubits))) : one(T)
    isapprox(provenance.proposal_amplitude, expected_amplitude;
             atol=zero(T), rtol=T(32) * eps(T)) || throw(ArgumentError(
        "result proposal amplitude is inconsistent with its normalisation label."))
    if provenance.sweep_clock !== nothing
        expected_multiplier = T(3 * config.num_qubits)
        provenance.sweep_clock.multiplier == expected_multiplier ||
            throw(ArgumentError(
                "sweep-clock multiplier must equal 3N=$(3config.num_qubits)."))
    end
    return nothing
end

function _validate_result_block_accounting(
    config::Config,
    parent::DLLParentBundle,
    diagnostics::DLLParentDiagnostics,
)
    num_sources = 3 * config.num_qubits
    num_channels = length(_config_dll_channels(config))
    expected_keys = Tuple{Int, Int}[
        (source_index, channel_index)
        for source_index in 1:num_sources
        for channel_index in 1:num_channels
    ]
    parent.block_keys == expected_keys || throw(ArgumentError(
        "parent block keys must cover every standard Pauli source/channel pair " *
        "in source-major order."))
    length(diagnostics.block_gibbs_residuals) == length(expected_keys) ||
        throw(ArgumentError(
            "block_gibbs_residuals must contain one entry per parent block."))
    return nothing
end

function _matching_gap_states(
    states::AbstractVector{<:DLLParentLowEnergyState{<:Any, T}},
    value::T,
    controls::ParentGapControls,
) where {T<:AbstractFloat}
    return filter(states) do state
        state.converged &&
        state.residual <= controls.residual_tolerance &&
        isapprox(state.energy, value;
                 atol=controls.energy_tolerance,
                 rtol=controls.energy_tolerance)
    end
end

function _require_reported_gap_state(
    states::AbstractVector{<:DLLParentLowEnergyState{<:Any, T}},
    value::T,
    controls::ParentGapControls;
    require_kernel_orthogonality::Bool,
    kernel_dimension::Int,
) where {T<:AbstractFloat}
    isempty(states) && throw(ArgumentError(
        "a gap or spacing result requires retained low-energy states."))
    matching = _matching_gap_states(states, value, controls)
    isempty(matching) && throw(ArgumentError(
        "no converged low-energy state matches gap_value within the solver tolerances."))
    if require_kernel_orthogonality
        any(matching) do state
            length(state.kernel_overlaps) == kernel_dimension &&
            all(overlap -> overlap <= controls.overlap_tolerance,
                state.kernel_overlaps)
        end || throw(ArgumentError(
            "a variational upper estimate requires a matching state with " *
            "small overlap against every complete kernel mode."))
    end
    return nothing
end

function _require_observed_spacing_states(
    states::AbstractVector{<:DLLParentLowEnergyState{<:Any, T}},
    value::T,
    controls::ParentGapControls,
    observed_manifold_dimension::Int,
) where {T<:AbstractFloat}
    observed_manifold_dimension > 0 || throw(ArgumentError(
        "an observed-manifold spacing requires at least one observed manifold mode."))
    reliable_states = filter(states) do state
        state.converged && state.residual <= controls.residual_tolerance
    end
    required_states = observed_manifold_dimension + 1
    length(reliable_states) >= required_states || throw(ArgumentError(
        "an observed manifold of dimension $observed_manifold_dimension " *
        "requires at least $required_states reliable states, including one " *
        "state outside the manifold."))
    sort!(reliable_states; by=state -> state.energy)
    spacing = reliable_states[required_states].energy -
              reliable_states[observed_manifold_dimension].energy
    isapprox(spacing, value;
             atol=controls.energy_tolerance,
             rtol=controls.energy_tolerance) || throw(ArgumentError(
        "gap_value must equal the adjacent spacing from the top of the " *
        "observed low-energy manifold to the lowest reliable outside state, " *
        "not an arbitrary pairwise level difference."))
    return nothing
end

function _require_resolved_gap_value(
    value::T,
    diagnostics::DLLParentDiagnostics{T},
    controls::ParentGapControls,
    label::Symbol,
) where {T<:AbstractFloat}
    resolution = max(diagnostics.kernel_tolerance, controls.energy_tolerance)
    value > resolution || throw(ArgumentError(
        "$label gap_value must exceed both the numerical kernel tolerance " *
        "and the eigensolver energy resolution (effective threshold $resolution)."))
    return nothing
end

function _require_exact_result_diagnostics(
    diagnostics::DLLParentDiagnostics{T},
    controls::ParentGapControls,
) where {T<:AbstractFloat}
    diagnostics.kernel_complete || throw(ArgumentError(
        ":exact_gap requires a complete numerical kernel."))
    diagnostics.kernel_evidence == :exact_dense_complete || throw(ArgumentError(
        ":exact_gap requires kernel_evidence=:exact_dense_complete."))
    diagnostics.hermiticity_defect <= controls.residual_tolerance ||
        throw(ArgumentError(":exact_gap failed the Hermiticity tolerance."))
    diagnostics.minimum_energy >= -diagnostics.kernel_tolerance ||
        throw(ArgumentError(":exact_gap failed the positivity tolerance."))
    abs(diagnostics.gibbs_energy) <= controls.energy_tolerance ||
        throw(ArgumentError(":exact_gap failed the Gibbs-energy tolerance."))
    diagnostics.gibbs_residual <= controls.residual_tolerance ||
        throw(ArgumentError(":exact_gap failed the Gibbs-residual tolerance."))
    all(residual -> residual <= controls.residual_tolerance,
        diagnostics.block_gibbs_residuals) || throw(ArgumentError(
            ":exact_gap failed a per-block Gibbs-residual tolerance."))
    return nothing
end


function _require_variational_parent_structure(
    diagnostics::DLLParentDiagnostics,
    controls::ParentGapControls,
)
    diagnostics.hermiticity_defect <= controls.residual_tolerance ||
        throw(ArgumentError(
            ":variational_upper_estimate requires a Hermitian represented parent."))
    abs(diagnostics.minimum_energy) <= diagnostics.kernel_tolerance ||
        throw(ArgumentError(
            ":variational_upper_estimate uses an absolute Rayleigh quotient " *
            "and therefore requires a zero-energy parent baseline within the " *
            "declared kernel tolerance."))
    return nothing
end

function _require_variational_error_evidence(
    ledger::ParentErrorLedger,
)
    for (label, entry) in (
        (:total_sum_compression, ledger.total_sum_compression),
        (:gibbs_mps, ledger.gibbs_mps),
        (:eigensolver, ledger.eigensolver),
    )
        entry.evidence != :unmeasured || throw(ArgumentError(
            ":variational_upper_estimate requires measured $label evidence."))
    end
    return nothing
end

"""
    DLLTensorNetworkResult

Final dependency-free report. Backend parent/state objects and every numerical
control family remain encoded in type parameters. Gap labels fail closed when
the stored kernel, residual, frame, block-accounting, or bound evidence is
insufficient.
"""
struct DLLTensorNetworkResult{
    T<:AbstractFloat,
    P<:DLLParentBundle,
    S,
    B<:BohrMPOControls,
    G<:GibbsMPSControls,
    C<:ParentGapControls,
    V,
}
    config::Config{TensorNetworkSpectrum, BohrDomain, DLL, T}
    parent::P
    low_energy_states::Vector{S}
    diagnostics::DLLParentDiagnostics{T}
    provenance::DLLParentProvenance{T}
    bohr_controls::B
    gibbs_controls::G
    gap_controls::C
    target_label::Symbol
    gap_label::Symbol
    gap_value::Union{Nothing, T}
    gap_lower::Union{Nothing, T}
    gap_upper::Union{Nothing, T}
    metadata::V

    function DLLTensorNetworkResult{
        T, P, S, B, G, C, V,
    }(
        ::Val{:validated},
        config::Config{TensorNetworkSpectrum, BohrDomain, DLL, T},
        parent::P,
        low_energy_states::Vector{S},
        diagnostics::DLLParentDiagnostics{T},
        provenance::DLLParentProvenance{T},
        bohr_controls::B,
        gibbs_controls::G,
        gap_controls::C,
        target_label::Symbol,
        gap_label::Symbol,
        gap_value::Union{Nothing, T},
        gap_lower::Union{Nothing, T},
        gap_upper::Union{Nothing, T},
        metadata::V,
    ) where {
        T<:AbstractFloat,
        P<:DLLParentBundle,
        S,
        B<:BohrMPOControls,
        G<:GibbsMPSControls,
        C<:ParentGapControls,
        V,
    }
        lower, upper = _validate_dll_tensor_network_result(
            config, parent, low_energy_states, diagnostics, provenance,
            bohr_controls, gap_controls, target_label, gap_label, gap_value,
            gap_lower, gap_upper)
        return new{T, P, S, B, G, C, V}(
            config, parent, low_energy_states, diagnostics, provenance,
            bohr_controls, gibbs_controls, gap_controls, target_label,
            gap_label, gap_value, lower, upper, metadata)
    end
end

function _validate_dll_tensor_network_result(
    config::Config{TensorNetworkSpectrum, BohrDomain, DLL, T},
    parent::P,
    low_energy_states::AbstractVector{S},
    diagnostics::DLLParentDiagnostics{T},
    provenance::DLLParentProvenance{T},
    bohr_controls::BohrMPOControls,
    gap_controls::ParentGapControls,
    target_label::Symbol,
    gap_label::Symbol,
    value::Union{Nothing, T},
    lower::Union{Nothing, T},
    upper::Union{Nothing, T},
) where {T<:AbstractFloat, P<:DLLParentBundle, S}
    validate_config!(config)
    isconcretetype(S) || throw(ArgumentError(
        "low_energy_states must have a concrete backend state record type."))
    S <: DLLParentLowEnergyState || throw(ArgumentError(
        "low_energy_states must contain DLLParentLowEnergyState records."))
    S.parameters[2] == T || throw(ArgumentError(
        "low-energy state precision must match Config precision $T."))
    _require_tn_label(target_label, DLL_PARENT_TARGET_LABELS, "target_label")
    _require_tn_label(gap_label, DLL_PARENT_GAP_LABELS, "gap_label")
    parent.target_label == target_label || throw(ArgumentError(
        "result target_label must match parent.target_label."))
    bohr_controls.target_label == target_label || throw(ArgumentError(
        "bohr_controls.target_label must match the result target."))
    _validate_result_provenance(config, provenance)
    _validate_result_block_accounting(config, parent, diagnostics)
    for (name, candidate) in (("gap_value", value), ("gap_lower", lower),
                              ("gap_upper", upper))
        candidate === nothing || _require_finite_nonnegative(candidate, name)
    end
    if gap_label == :exact_gap
        target_label == :exact_bohr_dense || throw(ArgumentError(
            ":exact_gap requires target_label=:exact_bohr_dense."))
        value !== nothing && value > zero(T) || throw(ArgumentError(
            ":exact_gap requires a positive gap_value."))
        _require_resolved_gap_value(
            value, diagnostics, gap_controls, :exact_gap)
        lower === nothing && upper === nothing || throw(ArgumentError(
            ":exact_gap does not accept certificate bounds; use " *
            ":certified_bracket once a checked certificate object is available."))
        _require_exact_result_diagnostics(diagnostics, gap_controls)
        _require_reported_gap_state(
            low_energy_states, value, gap_controls;
            require_kernel_orthogonality=true,
            kernel_dimension=diagnostics.observed_kernel_dimension)
    elseif gap_label == :variational_upper_estimate
        diagnostics.kernel_complete || throw(ArgumentError(
            ":variational_upper_estimate requires a complete kernel."))
        value !== nothing && value > zero(T) || throw(ArgumentError(
            ":variational_upper_estimate requires a positive gap_value."))
        _require_resolved_gap_value(
            value, diagnostics, gap_controls, :variational_upper_estimate)
        lower === nothing || throw(ArgumentError(
            "a variational upper estimate cannot carry a lower bound."))
        if upper !== nothing
            isapprox(upper, value;
                     atol=gap_controls.energy_tolerance,
                     rtol=gap_controls.energy_tolerance) || throw(ArgumentError(
                "gap_upper must equal gap_value for a variational upper estimate."))
        end
        diagnostics.tighter_parent_residual !== nothing &&
            diagnostics.tighter_parent_residual <=
                gap_controls.residual_tolerance || throw(ArgumentError(
            ":variational_upper_estimate requires a small residual against a tighter parent."))
        _require_variational_parent_structure(diagnostics, gap_controls)
        _require_variational_error_evidence(parent.error_ledger)
        _require_reported_gap_state(
            low_energy_states, value, gap_controls;
            require_kernel_orthogonality=true,
            kernel_dimension=diagnostics.observed_kernel_dimension)
        upper = value
    elseif gap_label == :observed_manifold_spacing
        value !== nothing && value > zero(T) || throw(ArgumentError(
            ":observed_manifold_spacing requires a positive gap_value."))
        _require_resolved_gap_value(
            value, diagnostics, gap_controls, :observed_manifold_spacing)
        lower === nothing && upper === nothing || throw(ArgumentError(
            "an observed-manifold spacing cannot carry gap bounds."))
        _require_observed_spacing_states(
            low_energy_states, value, gap_controls,
            diagnostics.observed_kernel_dimension)
    else
        throw(ArgumentError(
            ":certified_bracket is reserved until a checked lower-certificate " *
            "object is implemented; DMRG and an error ledger alone are insufficient."))
    end
    return lower, upper
end

function DLLTensorNetworkResult(
    config::Config{TensorNetworkSpectrum, BohrDomain, DLL, T},
    parent::P,
    low_energy_states::AbstractVector{S},
    diagnostics::DLLParentDiagnostics{T},
    provenance::DLLParentProvenance{T};
    bohr_controls::B,
    gibbs_controls::G,
    gap_controls::C,
    target_label::Symbol=parent.target_label,
    gap_label::Symbol,
    gap_value::Union{Nothing, Real}=nothing,
    gap_lower::Union{Nothing, Real}=nothing,
    gap_upper::Union{Nothing, Real}=nothing,
    metadata::V=NamedTuple(),
) where {
    T<:AbstractFloat,
    P<:DLLParentBundle,
    S,
    B<:BohrMPOControls,
    G<:GibbsMPSControls,
    C<:ParentGapControls,
    V,
}
    value = gap_value === nothing ? nothing : T(gap_value)
    lower = gap_lower === nothing ? nothing : T(gap_lower)
    upper = gap_upper === nothing ? nothing : T(gap_upper)
    return DLLTensorNetworkResult{T, P, S, B, G, C, V}(
        Val(:validated), config, parent, collect(low_energy_states),
        diagnostics, provenance, bohr_controls, gibbs_controls, gap_controls,
        target_label, gap_label, value, lower, upper, metadata)
end
