"""
    MPOCompressionRecord

Telemetry for one MPO multiplication or addition. ITensor truncation error is
stored as discarded-weight telemetry only; it is not promoted to an operator-
norm bound. ITensorMPS does not expose the discarded weights or pretruncation
bond dimensions of the sitewise zip-up factorizations, so those observations
are labelled explicitly rather than inferred from the returned MPO.
"""
struct MPOCompressionRecord{T<:AbstractFloat}
    stage::Symbol
    operation::Symbol
    order::Int
    requested_cutoff::T
    requested_maxdim::Int
    intermediate_cutoff::T
    intermediate_maxdim::Int
    maximum_input_bond::Int
    maximum_pre_final_bond::Int
    maximum_output_bond::Int
    maximum_internal_sweep_discarded_weight::T
    maximum_final_discarded_weight::T
    internal_factorization_telemetry::Symbol
    final_cap_reached::Bool
    intermediate_cap_status::Symbol
end

"""
    DLLGaussianBohrMPO

Compact MPO representation of Gaussian DLL `Q`, `L`, and signed `N` for one
fixed finite patch. The object targets `:finite_patch_bohr_surrogate`, not the
global exact-Bohr parent. `locality_error` in `error_ledger` therefore remains
unmeasured.
"""
struct DLLGaussianBohrMPO{
    T<:AbstractFloat,
    M,
    D<:QuantumFurnace.DLLGaussianChebyshevData,
    L<:QuantumFurnace.ParentErrorLedger,
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
    chebyshev_data::D
    compression_records::Vector{MPOCompressionRecord{T}}
    error_ledger::L
end

@inline function _maximum_mpo_bond(operator::ITensorMPS.MPO)
    length(operator) <= 1 && return 1
    return maximum(
        ITensors.dim(ITensorMPS.linkind(operator, bond))
        for bond in 1:(length(operator) - 1)
    )
end

function _truncate_and_record!(
    candidate::ITensorMPS.MPO,
    records::Vector{MPOCompressionRecord{T}},
    stage::Symbol,
    operation::Symbol,
    order::Int,
    cutoff::T,
    maxdim::Int,
    intermediate_cutoff::T,
    intermediate_maxdim::Int,
    maximum_input_bond::Int,
    maximum_internal_sweep_discarded_weight::T,
    internal_factorization_telemetry::Symbol,
) where {T<:AbstractFloat}
    has_links = length(candidate) > 1
    maximum_pre_final_bond = _maximum_mpo_bond(candidate)
    maximum_final_discarded_weight = Ref(zero(T))
    callback = function (; link, truncation_error)
        maximum_final_discarded_weight[] = max(
            maximum_final_discarded_weight[], T(truncation_error))
        return nothing
    end
    if has_links
        ITensorMPS.truncate!(
            candidate;
            cutoff,
            maxdim,
            callback,
        )
    end
    maximum_output_bond = _maximum_mpo_bond(candidate)
    intermediate_cap_status = if !has_links || intermediate_maxdim == 0
        :not_applicable
    elseif maximum_pre_final_bond >= intermediate_maxdim
        :cap_reached_after_internal_sweep
    else
        :unmeasured
    end
    push!(records, MPOCompressionRecord{T}(
        stage,
        operation,
        order,
        cutoff,
        maxdim,
        intermediate_cutoff,
        intermediate_maxdim,
        maximum_input_bond,
        maximum_pre_final_bond,
        maximum_output_bond,
        maximum_internal_sweep_discarded_weight,
        maximum_final_discarded_weight[],
        internal_factorization_telemetry,
        has_links && maximum_output_bond >= maxdim,
        intermediate_cap_status,
    ))
    return candidate
end

function _zipup_product(
    left::ITensorMPS.MPO,
    right::ITensorMPS.MPO,
    records::Vector{MPOCompressionRecord{T}},
    stage::Symbol,
    order::Int,
    cutoff::T,
    maxdim::Int,
) where {T<:AbstractFloat}
    intermediate_cutoff = cutoff / T(10)
    intermediate_maxdim = Base.checked_mul(4, maxdim)
    maximum_input_bond = max(
        _maximum_mpo_bond(left), _maximum_mpo_bond(right))
    maximum_internal_sweep_discarded_weight = Ref(zero(T))
    internal_sweep_callback = function (; link, truncation_error)
        maximum_internal_sweep_discarded_weight[] = max(
            maximum_internal_sweep_discarded_weight[], T(truncation_error))
        return nothing
    end
    candidate = ITensorMPS.apply(
        left,
        right;
        alg="zipup",
        cutoff=intermediate_cutoff,
        maxdim=intermediate_maxdim,
        truncate_kwargs=(;
            cutoff=intermediate_cutoff,
            maxdim=intermediate_maxdim,
            callback=internal_sweep_callback,
        ),
    )
    return _truncate_and_record!(
        candidate,
        records,
        stage,
        :zipup_product,
        order,
        cutoff,
        maxdim,
        intermediate_cutoff,
        intermediate_maxdim,
        maximum_input_bond,
        maximum_internal_sweep_discarded_weight[],
        length(left) > 1 ? :unmeasured : :not_applicable,
    )
end

function _directsum_add(
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
    candidate = if length(left) == 1
        ITensorMPS.MPO([left[1] + right[1]])
    else
        ITensorMPS.add(left, right; alg="directsum")
    end
    return _truncate_and_record!(
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
end

function _scaled_mpo_commutator(
    hamiltonian::ITensorMPS.MPO,
    operator::ITensorMPS.MPO,
    beta::T,
    interval_radius::T,
    controls::QuantumFurnace.BohrMPOControls,
    records::Vector{MPOCompressionRecord{T}},
    stage_prefix::Symbol,
    order::Int,
) where {T<:AbstractFloat}
    left = _zipup_product(
        hamiltonian,
        operator,
        records,
        Symbol(stage_prefix, :_left),
        order,
        T(controls.product_cutoff),
        controls.product_maxdim,
    )
    right = _zipup_product(
        operator,
        hamiltonian,
        records,
        Symbol(stage_prefix, :_right),
        order,
        T(controls.product_cutoff),
        controls.product_maxdim,
    )
    commutator = _directsum_add(
        left,
        -right,
        records,
        Symbol(stage_prefix, :_difference),
        order,
        T(controls.recurrence_cutoff),
        controls.recurrence_maxdim,
    )
    return (beta / interval_radius) * commutator
end

@inline function _coefficient(
    approximation::QuantumFurnace.BohrChebyshevApproximation{T},
    order::Int,
) where {T<:AbstractFloat}
    order <= approximation.degree || return zero(T)
    return getfield(approximation, :_coefficients)[order + 1]
end

function _accumulate_chebyshev_term(
    accumulator::ITensorMPS.MPO,
    basis::ITensorMPS.MPO,
    coefficient::T,
    controls::QuantumFurnace.BohrMPOControls,
    records::Vector{MPOCompressionRecord{T}},
    stage::Symbol,
    order::Int,
) where {T<:AbstractFloat}
    iszero(coefficient) && return accumulator
    return _directsum_add(
        accumulator,
        coefficient * basis,
        records,
        stage,
        order,
        T(controls.sum_cutoff),
        controls.sum_maxdim,
    )
end

function _coupled_q_l_mpo_recurrence(
    hamiltonian::ITensorMPS.MPO,
    source::ITensorMPS.MPO,
    beta::T,
    data::QuantumFurnace.DLLGaussianChebyshevData{T},
    controls::QuantumFurnace.BohrMPOControls,
    records::Vector{MPOCompressionRecord{T}},
) where {T<:AbstractFloat}
    q = data.q
    f = data.f
    maximum_degree = max(q.degree, f.degree)
    previous = copy(source)
    Q = (_coefficient(q, 0) / T(2)) * previous
    L = (_coefficient(f, 0) / T(2)) * previous
    maximum_degree == 0 && return Q, L

    current = _scaled_mpo_commutator(
        hamiltonian,
        previous,
        beta,
        data.interval_radius,
        controls,
        records,
        :q_l_commutator,
        1,
    )
    Q = _accumulate_chebyshev_term(
        Q, current, _coefficient(q, 1), controls, records,
        :q_accumulator, 1)
    L = _accumulate_chebyshev_term(
        L, current, _coefficient(f, 1), controls, records,
        :l_accumulator, 1)
    for order in 2:maximum_degree
        action = _scaled_mpo_commutator(
            hamiltonian,
            current,
            beta,
            data.interval_radius,
            controls,
            records,
            :q_l_commutator,
            order,
        )
        next = _directsum_add(
            T(2) * action,
            -previous,
            records,
            :q_l_recurrence,
            order,
            T(controls.recurrence_cutoff),
            controls.recurrence_maxdim,
        )
        Q = _accumulate_chebyshev_term(
            Q, next, _coefficient(q, order), controls, records,
            :q_accumulator, order)
        L = _accumulate_chebyshev_term(
            L, next, _coefficient(f, order), controls, records,
            :l_accumulator, order)
        previous, current = current, next
    end
    return Q, L
end

function _single_mpo_recurrence(
    hamiltonian::ITensorMPS.MPO,
    source::ITensorMPS.MPO,
    beta::T,
    approximation::QuantumFurnace.BohrChebyshevApproximation{T},
    controls::QuantumFurnace.BohrMPOControls,
    records::Vector{MPOCompressionRecord{T}},
) where {T<:AbstractFloat}
    previous = copy(source)
    result = (_coefficient(approximation, 0) / T(2)) * previous
    approximation.degree == 0 && return result
    current = _scaled_mpo_commutator(
        hamiltonian,
        previous,
        beta,
        approximation.interval_radius,
        controls,
        records,
        :sech_commutator,
        1,
    )
    result = _accumulate_chebyshev_term(
        result, current, _coefficient(approximation, 1), controls, records,
        :sech_accumulator, 1)
    for order in 2:approximation.degree
        action = _scaled_mpo_commutator(
            hamiltonian,
            current,
            beta,
            approximation.interval_radius,
            controls,
            records,
            :sech_commutator,
            order,
        )
        next = _directsum_add(
            T(2) * action,
            -previous,
            records,
            :sech_recurrence,
            order,
            T(controls.recurrence_cutoff),
            controls.recurrence_maxdim,
        )
        result = _accumulate_chebyshev_term(
            result,
            next,
            _coefficient(approximation, order),
            controls,
            records,
            :sech_accumulator,
            order,
        )
        previous, current = current, next
    end
    return result
end

function _bohr_mpo_error_ledger(
    data::QuantumFurnace.DLLGaussianChebyshevData{T},
) where {T<:AbstractFloat}
    scalar_bound = maximum((
        data.q.uniform_error_bound,
        data.f.uniform_error_bound,
        data.sech.uniform_error_bound,
    ))
    return QuantumFurnace.ParentErrorLedger(
        T;
        spectral_interval=QuantumFurnace.ParentErrorBound(
            T,
            :spectral_interval;
            magnitude=zero(T),
            evidence=:rigorous_bound,
            note="local Weyl-Gershgorin interval encloses the patch commutator spectrum",
        ),
        scalar_polynomial=QuantumFurnace.ParentErrorBound(
            T,
            :scalar_polynomial;
            magnitude=scalar_bound,
            evidence=:rigorous_bound,
            note="maximum certified exact-polynomial scalar error for q, f, and sech",
        ),
        modular_compatibility=QuantumFurnace.ParentErrorBound(
            T,
            :modular_compatibility;
            magnitude=data.modular_compatibility_bound,
            evidence=:rigorous_bound,
            note="certified scalar modular-compatibility error on the patch interval",
        ),
        locality_radius=QuantumFurnace.ParentErrorBound(
            T,
            :locality_radius;
            evidence=:unmeasured,
            note="fixed-radius patch is a named surrogate; no summed global locality-tail bound is available",
        ),
        mpo_recurrences=(
            QuantumFurnace.ParentErrorBound(
                T, :q_l_mpo_recurrence;
                evidence=:unmeasured,
                note="cutoffs, caps, discarded weights, and bonds are telemetry, not operator-norm bounds",
            ),
            QuantumFurnace.ParentErrorBound(
                T, :sech_mpo_recurrence;
                evidence=:unmeasured,
                note="propagated floating recurrence error is not certified",
            ),
        ),
        mpo_products=(
            QuantumFurnace.ParentErrorBound(
                T, :commutator_mpo_products;
                evidence=:unmeasured,
                note="zip-up product truncation lacks a rigorous operator-norm bound",
            ),
            QuantumFurnace.ParentErrorBound(
                T, :rate_mpo_product;
                evidence=:unmeasured,
                note="L' L product error and its propagation into N are unmeasured",
            ),
        ),
    )
end

function _validate_supplied_bohr_data(
    data::QuantumFurnace.DLLGaussianChebyshevData{T},
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    filter::QuantumFurnace.DLLGaussianFilter,
) where {T<:AbstractFloat}
    required_radius = T(filter.beta) * hamiltonian.spectral_width_bound
    !iszero(required_radius) && (required_radius = nextfloat(required_radius))
    data.interval_radius >= required_radius || throw(ArgumentError(
        "supplied Chebyshev interval $(data.interval_radius) does not enclose " *
        "the certified patch radius $required_radius."))
    return data
end

"""
    build_dll_bohr_mpo(config, hamiltonian, block, controls;
                       chebyshev_data=nothing)

Construct Gaussian DLL `Q`, `L`, and signed `N` MPOs on the compact patch
selected by `controls.patch_radius`. Products use the Task-6b zip-up strategy;
polynomial additions use exact direct sums followed by the requested
truncation. Only radii one through three are accepted because that is the
measured Task-6b viability envelope.

The returned object is always a `:finite_patch_bohr_surrogate`. It is not a
global KMS parent block, does not establish global Gibbs stationarity, and
does not carry a gap claim. The MPO recurrence supports `Float32` and
`Float64`; higher precision remains available for the scalar certificate but
is not supported by the backend's SVD truncations.
"""
function QuantumFurnace.build_dll_bohr_mpo(
    config::QuantumFurnace.Config{
        QuantumFurnace.TensorNetworkSpectrum,
        QuantumFurnace.BohrDomain,
        QuantumFurnace.DLL,
        T,
    },
    hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    block::QuantumFurnace.LocalDLLBlock1D{T},
    controls::QuantumFurnace.BohrMPOControls;
    chebyshev_data::Union{
        Nothing,
        QuantumFurnace.DLLGaussianChebyshevData{T},
    }=nothing,
) where {T<:AbstractFloat}
    T === Float32 || T === Float64 || throw(ArgumentError(
        "the ITensor Bohr-MPO recurrence supports Float32 and Float64; " *
        "got $T. Higher precision is supported only for scalar certificate " *
        "construction before conversion to a LAPACK floating type."))
    QuantumFurnace.validate_dll_tensor_network(
        config, hamiltonian, [block], controls)
    controls.target_label == :finite_patch_bohr_surrogate ||
        throw(ArgumentError(
            "Task 7B implements only :finite_patch_bohr_surrogate; " *
            "the full-chain requalification did not pass its scaling gate."))
    radius = something(controls.patch_radius)
    1 <= radius <= 3 || throw(ArgumentError(
        "Task 6b validated fixed patch radii 1:3; got radius=$radius."))
    channels = getfield(block, :channels)
    length(channels) == 1 || throw(ArgumentError(
        "Task 7B accepts one Gaussian channel per local block."))
    filter = only(channels)
    filter isa QuantumFurnace.DLLGaussianFilter{T} || throw(ArgumentError(
        "Task 7B supports DLLGaussianFilter{$T} only."))
    source = getfield(block, :source)
    patch = _compact_bohr_patch(hamiltonian, source, radius)
    beta_metadata = QuantumFurnace._frame_beta_metadata(
        patch.hamiltonian, filter; beta_phys=config.beta_phys)
    data = chebyshev_data === nothing ?
        QuantumFurnace.dll_gaussian_chebyshev_data(
            patch.hamiltonian,
            filter;
            beta_phys=config.beta_phys,
            tolerance=T(controls.scalar_tolerance),
        ) :
        _validate_supplied_bohr_data(
            chebyshev_data, patch.hamiltonian, filter)
    sites = local_operator_siteinds(
        patch.hamiltonian.num_sites; local_dim=patch.hamiltonian.local_dim)
    hamiltonian_mpo = local_hamiltonian_mpo(
        patch.hamiltonian; sites)
    source_mpo = local_jump_mpo(patch.source; sites)
    records = MPOCompressionRecord{T}[]

    Q, L, N = if iszero(data.interval_radius)
        Q_zero = copy(source_mpo)
        L_zero = copy(source_mpo)
        L_dagger = ITensors.swapprime(ITensors.dag(L_zero), 0 => 1)
        rate = _zipup_product(
            L_dagger,
            L_zero,
            records,
            :rate_product,
            0,
            T(controls.product_cutoff),
            controls.product_maxdim,
        )
        (Q_zero, L_zero, -rate)
    else
        Q_nonzero, L_nonzero = _coupled_q_l_mpo_recurrence(
            hamiltonian_mpo,
            source_mpo,
            T(filter.beta),
            data,
            controls,
            records,
        )
        L_dagger = ITensors.swapprime(ITensors.dag(L_nonzero), 0 => 1)
        rate = _zipup_product(
            L_dagger,
            L_nonzero,
            records,
            :rate_product,
            max(data.q.degree, data.f.degree),
            T(controls.product_cutoff),
            controls.product_maxdim,
        )
        filtered_rate = _single_mpo_recurrence(
            hamiltonian_mpo,
            rate,
            T(filter.beta),
            data.sech,
            controls,
            records,
        )
        (Q_nonzero, L_nonzero, -filtered_rate)
    end
    ledger = _bohr_mpo_error_ledger(data)
    return DLLGaussianBohrMPO{
        T,
        typeof(Q),
        typeof(data),
        typeof(ledger),
        typeof(sites),
    }(
        :finite_patch_bohr_surrogate,
        patch.first_site,
        patch.last_site,
        patch.source_sites_global,
        hamiltonian.coordinate_frame,
        T(beta_metadata.beta_frame),
        T(beta_metadata.beta_phys),
        T(beta_metadata.beta_alg),
        _BOHR_PATCH_IDENTITY_GAUGE,
        _BOHR_PATCH_OMISSION_RULE,
        patch.retained_term_count,
        patch.omitted_term_count,
        sites,
        Q,
        L,
        N,
        data,
        records,
        ledger,
    )
end
