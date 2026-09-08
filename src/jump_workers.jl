"""
    _jump_contribution!(L_target, jump, hamiltonian, config, precomputed_data, ws;
                        coherent_term=nothing)

Accumulate one jump's dense vectorized Liouvillian contribution in place.

Pass `coherent_term` already scaled by `gamma_norm_factor` so cached matrices
remain unchanged.
"""
function _jump_contribution!(
    L_target::AbstractMatrix{<:Complex},
    jump::JumpOp,
    hamiltonian::HamHam,
    config::Config{Lindbladian, BohrDomain},
    precomputed_data,
    ws::DenseLindbladianWorkspace;
    coherent_term::Union{Nothing, AbstractMatrix{<:Complex}} = nothing,
    )
    dim = size(hamiltonian.data, 1)
    unique_freqs = keys(hamiltonian.bohr_dict)
    (; alpha, gamma_norm_factor) = precomputed_data

    B = coherent_term
    if B !== nothing
        _vectorize_liouvillian_coherent!(L_target, B, ws)
    end

    alpha_A_nu1 = ws.scratch.jump_tmp
    for nu_2 in unique_freqs
        @. alpha_A_nu1 = alpha(hamiltonian.bohr_freqs, nu_2) * jump.in_eigenbasis

        indices = hamiltonian.bohr_dict[nu_2]
        A_nu_2_vals = view(jump.in_eigenbasis, indices)

        # swapped for dagger
        rows_dag = getindex.(indices, 2)
        cols_dag = getindex.(indices, 1)

        A_nu_2_dag = sparse(rows_dag, cols_dag, conj.(A_nu_2_vals), dim, dim)

        _vectorize_liouv_diss_and_add!(L_target, alpha_A_nu1, A_nu_2_dag, gamma_norm_factor, ws)
    end
    return L_target
end

function _jump_contribution!(
    L_target::AbstractMatrix{<:Complex},
    jump::JumpOp,
    hamiltonian::HamHam,
    config::Config{Lindbladian, EnergyDomain},
    precomputed_data,
    ws::DenseLindbladianWorkspace;
    coherent_term::Union{Nothing, AbstractMatrix{<:Complex}} = nothing,
    )

    (; transition, gamma_norm_factor, energy_labels) = precomputed_data

    B = coherent_term
    if B !== nothing
        _vectorize_liouvillian_coherent!(L_target, B, ws)
    end

    jump_oft = ws.scratch.jump_tmp
    prefactor = precomputed_data.oft_domain_prefactor * gamma_norm_factor
    inv_4sigma2 = _energy_oft_kernel(config)

    if jump.hermitian && !_is_joint_ckg(config)
        for w_raw in energy_labels
            # iterate only half-grid (w<=0) and mirror manually
            w_raw > 1e-12 && continue
            w = abs(w_raw)
            oft!(jump_oft, jump.in_eigenbasis, hamiltonian.bohr_freqs, w, inv_4sigma2)
            scalar_w = prefactor * transition(w)
            _vectorize_liouv_diss_and_add!(L_target, jump_oft, scalar_w, ws)
            if w > 1e-12
                scalar_negative_w = prefactor * transition(-w)
                _vectorize_liouv_diss_and_add!(L_target, jump_oft', scalar_negative_w, ws)
            end
        end
    else
        for w in energy_labels
            oft!(jump_oft, jump.in_eigenbasis, hamiltonian.bohr_freqs, w, inv_4sigma2)
            scalar_w = prefactor * transition(w)
            _vectorize_liouv_diss_and_add!(L_target, jump_oft, scalar_w, ws)
        end
    end

    return L_target
end

"""
    _jump_contribution!(
        L_target, jump, hamiltonian,
        config::Config{Lindbladian, BohrDomain, DLL}, precomputed_data, ws;
        coherent_term=nothing,
    )

Accumulate one DLL Bohr-domain dissipator.

Multi-channel filters contribute one dissipator per channel; their operators
must not be summed before forming sandwiches because that would add cross terms.
"""
function _jump_contribution!(
    L_target::AbstractMatrix{<:Complex},
    jump::JumpOp,
    hamiltonian::HamHam,
    config::Config{Lindbladian, BohrDomain, DLL},
    precomputed_data,
    ws::DenseLindbladianWorkspace;
    coherent_term::Union{Nothing, AbstractMatrix{<:Complex}} = nothing,
    )
    (; filter) = precomputed_data

    if coherent_term !== nothing
        _vectorize_liouvillian_coherent!(L_target, coherent_term, ws)
    end

    _accumulate_dll_bohr_dissipator!(L_target, jump, hamiltonian, filter, ws)
    return L_target
end

# Single-channel DLL filter: one Lindblad operator per coupling.
@inline function _accumulate_dll_bohr_dissipator!(
    L_target::AbstractMatrix{<:Complex},
    jump::JumpOp,
    hamiltonian::HamHam,
    filter::AbstractFilter,
    ws::DenseLindbladianWorkspace,
)
    L_a = dll_lindblad_op_bohr(jump, hamiltonian, filter)
    _vectorize_liouv_diss_and_add!(L_target, L_a, 1.0, ws)
    return L_target
end

# See `dll_multichannel.jl` for the multi-channel overload.
# `_accumulate_dll_bohr_dissipator!(::DLLMultiChannelFilter, ...)` overload.

"""
    _jump_contribution!(
        L_target, jump, hamiltonian,
        config::Config{Lindbladian, TimeDomain, DLL}, precomputed_data, ws;
        coherent_term=nothing,
    )

Accumulate one DLL time-domain dissipator from precomputed zero-frequency
NUFFT slices.

Each slice represents one channel's trapezoidal Fourier sum, so multi-channel
dissipators are accumulated without cross terms.
"""
function _jump_contribution!(
    L_target::AbstractMatrix{<:Complex},
    jump::JumpOp,
    hamiltonian::HamHam,
    config::Config{Lindbladian, TimeDomain, DLL},
    precomputed_data,
    ws::DenseLindbladianWorkspace;
    coherent_term::Union{Nothing, AbstractMatrix{<:Complex}} = nothing,
    )
    (; t0, oft_nufft_at_zero_list) = precomputed_data

    if coherent_term !== nothing
        _vectorize_liouvillian_coherent!(L_target, coherent_term, ws)
    end

    L_a = ws.scratch.jump_tmp
    @inbounds for nufft_at_zero in oft_nufft_at_zero_list
        @. L_a = jump.in_eigenbasis * nufft_at_zero * t0
        _vectorize_liouv_diss_and_add!(L_target, L_a, 1.0, ws)
    end
    return L_target
end

function _jump_contribution!(
    L_target::AbstractMatrix{<:Complex},
    jump::JumpOp,
    ham_or_trott::Union{HamHam, AbstractTrotter},
    config::Config{Lindbladian, D},
    precomputed_data,
    ws::DenseLindbladianWorkspace;
    coherent_term::Union{Nothing, AbstractMatrix{<:Complex}} = nothing,
    ) where {D<:Union{TimeDomain, TrotterDomain}}

    (; transition, gamma_norm_factor, energy_labels, oft_nufft_prefactors, b_minus, b_plus) = precomputed_data

    B = coherent_term
    if B !== nothing
        _vectorize_liouvillian_coherent!(L_target, B, ws)
    end

    jump_oft = ws.scratch.jump_tmp
    prefactor = precomputed_data.oft_domain_prefactor * gamma_norm_factor

    if jump.hermitian
        for w_raw in energy_labels
            w_raw > 1e-12 && continue
            w = abs(w_raw)
            nufft_prefactor_matrix = _prefactor_view(oft_nufft_prefactors, w)
            @. jump_oft = jump.in_eigenbasis * nufft_prefactor_matrix

            scalar_w = prefactor * transition(w)
            _vectorize_liouv_diss_and_add!(L_target, jump_oft, scalar_w, ws)
            if w > 1e-12
                scalar_negative_w = prefactor * transition(-w)
                _vectorize_liouv_diss_and_add!(L_target, jump_oft', scalar_negative_w, ws)
            end
        end
    else
        for w in energy_labels
            nufft_prefactor_matrix = _prefactor_view(oft_nufft_prefactors, w)
            @. jump_oft = jump.in_eigenbasis * nufft_prefactor_matrix
            scalar_w = prefactor * transition(w)
            _vectorize_liouv_diss_and_add!(L_target, jump_oft, scalar_w, ws)
        end
    end

    return L_target
end

"""
    _apply_coherent_unitary!(evolving_dm, U_B, scratch) -> nothing

Apply `rho -> U_B * rho * U_B'`, or do nothing when `U_B` is `nothing`.
"""
@inline function _apply_coherent_unitary!(
    evolving_dm::Matrix{<:Complex},
    U_B::Union{Nothing,Matrix{<:Complex}},
    scratch::ThermalizeScratch{<:Complex},
)
    U_B === nothing && return nothing
    mul!(scratch.sandwich_tmp, U_B, evolving_dm)
    mul!(scratch.rho_next, scratch.sandwich_tmp, U_B')
    copyto!(evolving_dm, scratch.rho_next)
    return nothing
end

# The Hilbert--Schmidt adjoint maps $X -> U^dagger X U$.
@inline function _apply_adjoint_coherent_unitary!(
    evolving_dm::Matrix{<:Complex},
    U_B::Union{Nothing,Matrix{<:Complex}},
    scratch::ThermalizeScratch{<:Complex},
)
    U_B === nothing && return nothing
    mul!(scratch.sandwich_tmp, U_B', evolving_dm)
    mul!(scratch.rho_next, scratch.sandwich_tmp, U_B)
    copyto!(evolving_dm, scratch.rho_next)
    return nothing
end


"""
    _accumulate_rho_jump!(scratch, evolving_dm, jump, hamiltonian, config::Config{Thermalize, EnergyDomain},
                          precomputed_data; jump_weight_scaling)

Accumulate the energy-domain jump sandwich for a precomputed channel.

Math: \$rho_jump = delta sum_omega rate(omega)^2 L_(a,omega) rho L_(a,omega)^dagger\$.
"""
function _accumulate_rho_jump!(
    scratch::ThermalizeScratch{<:Complex},
    evolving_dm::Matrix{<:Complex},
    jump::JumpOp,
    hamiltonian::HamHam,
    config::Config{Thermalize, EnergyDomain},
    precomputed_data;
    jump_weight_scaling::Real,
)
    (; transition, energy_labels) = precomputed_data

    n_labels = length(energy_labels)
    if Threads.nthreads() > 1 && n_labels >= OMEGA_THREAD_THRESHOLD
        return _accumulate_rho_jump_threaded_energy!(scratch, evolving_dm, jump, hamiltonian, config, precomputed_data;
            jump_weight_scaling=jump_weight_scaling,
            task_scratches=isempty(scratch.task_scratches) ? nothing : scratch.task_scratches)
    end

    base_prefactor = precomputed_data.oft_domain_prefactor * jump_weight_scaling
    inv_4sigma2 = 1.0 / (4 * config.sigma^2)

    fill!(scratch.rho_jump, 0)

    if jump.hermitian
        @inbounds for w_raw in energy_labels
            w_raw > 1e-12 && continue
            w = abs(w_raw)

            oft!(scratch.jump_oft, jump.in_eigenbasis, hamiltonian.bohr_freqs, w, inv_4sigma2)

            rate2_pos = base_prefactor * transition(w)

            # rho_jump += delta * rate^2 * (Aw rho Aw_dag)
            mul!(scratch.sandwich_tmp, evolving_dm, scratch.jump_oft')
            mul!(scratch.rho_jump, scratch.jump_oft, scratch.sandwich_tmp, config.delta * rate2_pos, 1.0)

            if w > 1e-12
                rate2_neg = base_prefactor * transition(-w)

                # Negative-frequency partner uses (Aw)_dag as Lindblad operator.
                mul!(scratch.sandwich_tmp, evolving_dm, scratch.jump_oft)
                mul!(scratch.rho_jump, scratch.jump_oft', scratch.sandwich_tmp, config.delta * rate2_neg, 1.0)
            end
        end
    else
        @inbounds for w in energy_labels
            oft!(scratch.jump_oft, jump.in_eigenbasis, hamiltonian.bohr_freqs, w, inv_4sigma2)

            rate2 = base_prefactor * transition(w)

            mul!(scratch.sandwich_tmp, evolving_dm, scratch.jump_oft')
            mul!(scratch.rho_jump, scratch.jump_oft, scratch.sandwich_tmp, config.delta * rate2, 1.0)
        end
    end

    return nothing
end

"""
    _accumulate_rho_jump!(scratch, evolving_dm, jump, ham_or_trott, config::Config{Thermalize, D},
                          precomputed_data; jump_weight_scaling) where D<:Union{TimeDomain, TrotterDomain}

Accumulate rho_jump for TimeDomain/TrotterDomain. Uses NUFFT prefactors for jump_oft
computation. No R or LdagL accumulation.
"""
function _accumulate_rho_jump!(
    scratch::ThermalizeScratch{<:Complex},
    evolving_dm::Matrix{<:Complex},
    jump::JumpOp,
    ham_or_trott,
    config::Config{Thermalize, D},
    precomputed_data;
    jump_weight_scaling::Real,
) where {D<:Union{TimeDomain, TrotterDomain}}
    (; transition, energy_labels, oft_nufft_prefactors) = precomputed_data

    n_labels = length(energy_labels)
    if Threads.nthreads() > 1 && n_labels >= OMEGA_THREAD_THRESHOLD
        return _accumulate_rho_jump_threaded_timetrot!(scratch, evolving_dm, jump, ham_or_trott, config, precomputed_data;
            jump_weight_scaling=jump_weight_scaling,
            task_scratches=isempty(scratch.task_scratches) ? nothing : scratch.task_scratches)
    end

    base_prefactor = precomputed_data.oft_domain_prefactor * jump_weight_scaling

    fill!(scratch.rho_jump, 0)

    if jump.hermitian
        @inbounds for w_raw in energy_labels
            w_raw > 1e-12 && continue
            w = abs(w_raw)

            nufft_prefactor_matrix = _prefactor_view(oft_nufft_prefactors, w)
            @. scratch.jump_oft = jump.in_eigenbasis * nufft_prefactor_matrix

            rate2_pos = base_prefactor * transition(w)

            mul!(scratch.sandwich_tmp, evolving_dm, scratch.jump_oft')
            mul!(scratch.rho_jump, scratch.jump_oft, scratch.sandwich_tmp, config.delta * rate2_pos, 1.0)

            if w > 1e-12
                rate2_neg = base_prefactor * transition(-w)

                mul!(scratch.sandwich_tmp, evolving_dm, scratch.jump_oft)
                mul!(scratch.rho_jump, scratch.jump_oft', scratch.sandwich_tmp, config.delta * rate2_neg, 1.0)
            end
        end
    else
        for w in energy_labels
            nufft_prefactor_matrix = _prefactor_view(oft_nufft_prefactors, w)
            @. scratch.jump_oft = jump.in_eigenbasis * nufft_prefactor_matrix

            rate2_pos = base_prefactor * transition(w)

            mul!(scratch.sandwich_tmp, evolving_dm, scratch.jump_oft')
            mul!(scratch.rho_jump, scratch.jump_oft, scratch.sandwich_tmp, config.delta * rate2_pos, 1.0)
        end
    end

    return nothing
end

"""
    _accumulate_rho_jump!(scratch, evolving_dm, jump, hamiltonian, config::Config{Thermalize, BohrDomain},
                          precomputed_data; jump_weight_scaling)

Accumulate the Bohr-domain jump sandwich without rebuilding the rate operator.
"""
function _accumulate_rho_jump!(
    scratch::ThermalizeScratch{<:Complex},
    evolving_dm::Matrix{<:Complex},
    jump::JumpOp,
    hamiltonian::HamHam,
    config::Config{Thermalize, BohrDomain},
    precomputed_data;
    jump_weight_scaling::Real,
)
    dim = size(evolving_dm, 1)
    (; alpha) = precomputed_data

    bohr_keys = hasproperty(precomputed_data, :bohr_keys) ? precomputed_data.bohr_keys : collect(keys(hamiltonian.bohr_dict))
    bohr_is   = hasproperty(precomputed_data, :bohr_is)   ? precomputed_data.bohr_is   : nothing
    bohr_js   = hasproperty(precomputed_data, :bohr_js)   ? precomputed_data.bohr_js   : nothing

    n_keys = length(bohr_keys)
    if Threads.nthreads() > 1 && n_keys >= OMEGA_THREAD_THRESHOLD
        return _accumulate_rho_jump_threaded_bohr!(scratch, evolving_dm, jump, hamiltonian, config, precomputed_data, bohr_keys, bohr_is, bohr_js;
            jump_weight_scaling=jump_weight_scaling,
            task_scratches=isempty(scratch.task_scratches) ? nothing : scratch.task_scratches)
    end

    scaled_delta = config.delta * jump_weight_scaling

    fill!(scratch.rho_jump, 0)

    # Keep the hot-loop matrix view concretely typed.
    CT = eltype(scratch.jump_oft)
    in_eb = jump.in_eigenbasis::Matrix{CT}

    @inbounds for (k, nu_2) in pairs(bohr_keys)
        # B_{v2} = sum_{v1} alpha(v1, v2) * A
        @. scratch.jump_oft = alpha(hamiltonian.bohr_freqs, nu_2) * in_eb

        # sandwich_tmp := rho A_{v2}dag
        fill!(scratch.sandwich_tmp, 0)
        if bohr_is !== nothing
            is = bohr_is[k]
            js = bohr_js[k]
            @inbounds for t in eachindex(is)
                i = is[t]
                j = js[t]
                v = conj(in_eb[i, j])
                @inbounds for p in 1:dim
                    scratch.sandwich_tmp[p, i] += evolving_dm[p, j] * v
                end
            end
        else
            indices = hamiltonian.bohr_dict[nu_2]
            @inbounds for idx in indices
                i = idx[1]
                j = idx[2]
                v = conj(in_eb[i, j])
                @inbounds for p in 1:dim
                    scratch.sandwich_tmp[p, i] += evolving_dm[p, j] * v
                end
            end
        end

        # rho_jump += scaled_delta * B_{v2} * (rho A_{v2}dag)
        mul!(scratch.rho_jump, scratch.jump_oft, scratch.sandwich_tmp, scaled_delta, 1.0)
    end

    return nothing
end

function _accumulate_rho_jump_threaded_energy!(
    scratch::ThermalizeScratch{CT},
    evolving_dm::Matrix{CT},
    jump::JumpOp,
    hamiltonian::HamHam,
    config::Config{Thermalize, EnergyDomain},
    precomputed_data;
    jump_weight_scaling::Real,
    task_scratches::Union{Nothing, Vector{ThermalizeScratch{CT}}}=nothing,
) where {CT<:Complex}
    (; transition, energy_labels) = precomputed_data
    base_prefactor = precomputed_data.oft_domain_prefactor * jump_weight_scaling
    inv_4sigma2 = 1.0 / (4 * config.sigma^2)

    # For Hermitian jumps, pre-filter to half-grid indices for balanced partitioning
    if jump.hermitian
        half_indices = [i for i in eachindex(energy_labels) if energy_labels[i] <= 1e-12]
    else
        half_indices = collect(eachindex(energy_labels))
    end

    n_work = length(half_indices)
    nt = min(Threads.nthreads(), n_work)
    chunks = _partition_range(1:n_work, nt)
    dim = size(evolving_dm, 1)

    # Per-task scratch: each needs rho_jump, jump_oft, sandwich_tmp.
    # Reuse a supplied thread-local pool; otherwise allocate one for this call.
    local_pool = if task_scratches === nothing
        [ThermalizeScratch(CT, dim) for _ in 1:length(chunks)]
    else
        @assert length(task_scratches) >= length(chunks)
        task_scratches
    end

    @sync for (idx, chunk) in enumerate(chunks)
        Threads.@spawn _accumulate_rho_jump_chunk_energy!(
            local_pool[idx], evolving_dm, jump, hamiltonian,
            config, precomputed_data, half_indices[chunk];
            base_prefactor=base_prefactor, inv_4sigma2=inv_4sigma2)
    end

    # Reduce: sum per-task rho_jump into scratch.rho_jump
    fill!(scratch.rho_jump, 0)
    for idx in 1:length(chunks)
        scratch.rho_jump .+= local_pool[idx].rho_jump
    end

    return nothing
end

function _accumulate_rho_jump_chunk_energy!(
    scratch::ThermalizeScratch{CT},
    evolving_dm::Matrix{CT},
    jump::JumpOp,
    hamiltonian::HamHam,
    config::Config{Thermalize, EnergyDomain},
    precomputed_data,
    label_indices::AbstractVector{Int};
    base_prefactor::Real,
    inv_4sigma2::Real,
) where {CT<:Complex}
    (; transition, energy_labels) = precomputed_data
    fill!(scratch.rho_jump, 0)

    @inbounds for li in label_indices
        w_raw = energy_labels[li]
        w = jump.hermitian ? abs(w_raw) : w_raw

        oft!(scratch.jump_oft, jump.in_eigenbasis, hamiltonian.bohr_freqs, w, inv_4sigma2)

        rate2_pos = base_prefactor * transition(w)
        mul!(scratch.sandwich_tmp, evolving_dm, scratch.jump_oft')
        mul!(scratch.rho_jump, scratch.jump_oft, scratch.sandwich_tmp, config.delta * rate2_pos, 1.0)

        if jump.hermitian && w > 1e-12
            rate2_neg = base_prefactor * transition(-w)
            mul!(scratch.sandwich_tmp, evolving_dm, scratch.jump_oft)
            mul!(scratch.rho_jump, scratch.jump_oft', scratch.sandwich_tmp, config.delta * rate2_neg, 1.0)
        end
    end

    return nothing
end

function _accumulate_rho_jump_threaded_timetrot!(
    scratch::ThermalizeScratch{CT},
    evolving_dm::Matrix{CT},
    jump::JumpOp,
    ham_or_trott,
    config::Config{Thermalize, D},
    precomputed_data;
    jump_weight_scaling::Real,
    task_scratches::Union{Nothing, Vector{ThermalizeScratch{CT}}}=nothing,
) where {CT<:Complex, D<:Union{TimeDomain, TrotterDomain}}
    (; transition, energy_labels, oft_nufft_prefactors) = precomputed_data
    base_prefactor = precomputed_data.oft_domain_prefactor * jump_weight_scaling

    # For Hermitian jumps, pre-filter to half-grid indices for balanced partitioning
    if jump.hermitian
        half_indices = [i for i in eachindex(energy_labels) if energy_labels[i] <= 1e-12]
    else
        half_indices = collect(eachindex(energy_labels))
    end

    n_work = length(half_indices)
    nt = min(Threads.nthreads(), n_work)
    chunks = _partition_range(1:n_work, nt)
    dim = size(evolving_dm, 1)

    local_pool = if task_scratches === nothing
        [ThermalizeScratch(CT, dim) for _ in 1:length(chunks)]
    else
        @assert length(task_scratches) >= length(chunks)
        task_scratches
    end

    @sync for (idx, chunk) in enumerate(chunks)
        Threads.@spawn _accumulate_rho_jump_chunk_timetrot!(
            local_pool[idx], evolving_dm, jump,
            config, precomputed_data, half_indices[chunk];
            base_prefactor=base_prefactor)
    end

    # Reduce: sum per-task rho_jump into scratch.rho_jump
    fill!(scratch.rho_jump, 0)
    for idx in 1:length(chunks)
        scratch.rho_jump .+= local_pool[idx].rho_jump
    end

    return nothing
end

function _accumulate_rho_jump_chunk_timetrot!(
    scratch::ThermalizeScratch{CT},
    evolving_dm::Matrix{CT},
    jump::JumpOp,
    config::Config{Thermalize, D},
    precomputed_data,
    label_indices::AbstractVector{Int};
    base_prefactor::Real,
) where {CT<:Complex, D<:Union{TimeDomain, TrotterDomain}}
    (; transition, energy_labels, oft_nufft_prefactors) = precomputed_data
    fill!(scratch.rho_jump, 0)

    @inbounds for li in label_indices
        w_raw = energy_labels[li]
        w = jump.hermitian ? abs(w_raw) : w_raw

        nufft_prefactor_matrix = _prefactor_view(oft_nufft_prefactors, w)
        @. scratch.jump_oft = jump.in_eigenbasis * nufft_prefactor_matrix

        rate2_pos = base_prefactor * transition(w)
        mul!(scratch.sandwich_tmp, evolving_dm, scratch.jump_oft')
        mul!(scratch.rho_jump, scratch.jump_oft, scratch.sandwich_tmp, config.delta * rate2_pos, 1.0)

        if jump.hermitian && w > 1e-12
            rate2_neg = base_prefactor * transition(-w)
            mul!(scratch.sandwich_tmp, evolving_dm, scratch.jump_oft)
            mul!(scratch.rho_jump, scratch.jump_oft', scratch.sandwich_tmp, config.delta * rate2_neg, 1.0)
        end
    end

    return nothing
end

function _accumulate_rho_jump_threaded_bohr!(
    scratch::ThermalizeScratch{CT},
    evolving_dm::Matrix{CT},
    jump::JumpOp,
    hamiltonian::HamHam,
    config::Config{Thermalize, BohrDomain},
    precomputed_data,
    bohr_keys::AbstractVector,
    bohr_is::Union{Nothing, Vector{Vector{Int}}},
    bohr_js::Union{Nothing, Vector{Vector{Int}}};
    jump_weight_scaling::Real,
    task_scratches::Union{Nothing, Vector{ThermalizeScratch{CT}}}=nothing,
) where {CT<:Complex}
    dim = size(evolving_dm, 1)
    (; alpha) = precomputed_data
    scaled_delta = config.delta * jump_weight_scaling

    n_keys = length(bohr_keys)
    nt = min(Threads.nthreads(), n_keys)
    chunks = _partition_range(1:n_keys, nt)

    local_pool = if task_scratches === nothing
        [ThermalizeScratch(CT, dim) for _ in 1:length(chunks)]
    else
        @assert length(task_scratches) >= length(chunks)
        task_scratches
    end

    @sync for (idx, chunk) in enumerate(chunks)
        Threads.@spawn _accumulate_rho_jump_chunk_bohr!(
            local_pool[idx], evolving_dm, jump, hamiltonian,
            precomputed_data, bohr_keys, bohr_is, bohr_js, chunk;
            scaled_delta=scaled_delta)
    end

    # Reduce: sum per-task rho_jump into scratch.rho_jump
    fill!(scratch.rho_jump, 0)
    for idx in 1:length(chunks)
        scratch.rho_jump .+= local_pool[idx].rho_jump
    end

    return nothing
end

function _accumulate_rho_jump_chunk_bohr!(
    scratch::ThermalizeScratch{CT},
    evolving_dm::Matrix{CT},
    jump::JumpOp,
    hamiltonian::HamHam,
    precomputed_data,
    bohr_keys::AbstractVector,
    bohr_is::Union{Nothing, Vector{Vector{Int}}},
    bohr_js::Union{Nothing, Vector{Vector{Int}}},
    key_indices::UnitRange{Int};
    scaled_delta::Real,
) where {CT<:Complex}
    dim = size(evolving_dm, 1)
    (; alpha) = precomputed_data
    fill!(scratch.rho_jump, 0)

    # Keep the hot-loop matrix view concretely typed.
    in_eb = jump.in_eigenbasis::Matrix{CT}

    @inbounds for k in key_indices
        nu_2 = bohr_keys[k]

        # B_{v2} = sum_{v1} alpha(v1, v2) * A
        @. scratch.jump_oft = alpha(hamiltonian.bohr_freqs, nu_2) * in_eb

        # sandwich_tmp := rho A_{v2}dag
        fill!(scratch.sandwich_tmp, 0)
        if bohr_is !== nothing
            is = bohr_is[k]
            js = bohr_js[k]
            @inbounds for t in eachindex(is)
                i = is[t]
                j = js[t]
                v = conj(in_eb[i, j])
                @inbounds for p in 1:dim
                    scratch.sandwich_tmp[p, i] += evolving_dm[p, j] * v
                end
            end
        else
            indices = hamiltonian.bohr_dict[nu_2]
            @inbounds for idx in indices
                i = idx[1]
                j = idx[2]
                v = conj(in_eb[i, j])
                @inbounds for p in 1:dim
                    scratch.sandwich_tmp[p, i] += evolving_dm[p, j] * v
                end
            end
        end

        # rho_jump += scaled_delta * B_{v2} * (rho A_{v2}dag)
        mul!(scratch.rho_jump, scratch.jump_oft, scratch.sandwich_tmp, scaled_delta, 1.0)
    end

    return nothing
end
