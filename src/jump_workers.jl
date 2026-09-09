# Shared frequency-domain rate operator: sum_w rate(w) * A(w)'A(w).
# Callers supply source matrices in their working spectral basis.
function _accumulate_frequency_loss!(R::Matrix{T}, bases::Vector{Matrix{T}}, hermitian,
    labels, transition::F, prefactor, data::D; threaded::Bool, buffers=nothing,
) where {T,F,D}
    work = Tuple{Int,Int}[]
    _populate_jump_frequency_work_list!(work, hermitian, labels)
    isempty(work) && return nothing
    if !threaded
        operator, product = buffers === nothing ? (similar(R), similar(R)) : buffers
        _accumulate_frequency_loss_chunk!(R, operator, product, bases,
            hermitian, labels, work, eachindex(work), transition, prefactor, data)
        return nothing
    end
    chunks = _partition_range(1:length(work), min(Threads.nthreads(),length(work)))
    partials = [zero(R) for _ in chunks]
    operators = [similar(R) for _ in chunks]
    products = [similar(R) for _ in chunks]
    @sync for (idx, chunk) in enumerate(chunks)
        Threads.@spawn _accumulate_frequency_loss_chunk!(partials[idx], operators[idx], products[idx],
            bases, hermitian, labels, work, chunk, transition, prefactor, data)
    end
    for partial in partials
        R .+= partial
    end
    return nothing
end

function _accumulate_frequency_loss_chunk!(R, operator, product, bases, hermitian,
    labels, work, chunk, transition::F, prefactor, data::D) where {F,D}
    @inbounds for wi in chunk
        k, li = work[wi]
        folded = hermitian[k]
        w = folded ? abs(labels[li]) : labels[li]
        _frequency_oft!(operator, bases[k], data, w, li, folded)
        rate = prefactor * transition(w)
        mul!(product, operator', operator)
        @. R += rate * product
        if folded && w > 1e-12
            rate_negative = prefactor * transition(-w)
            mul!(product, operator, operator')
            @. R += rate_negative * product
        end
    end
    return nothing
end

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

    if jump.hermitian && !_is_joint_ckg(config)
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
    _accumulate_rho_jump!(scratch, rho, jump, ham, config, precomputed; jump_weight_scaling)

Accumulate the frequency-domain channel gain into `scratch.rho_jump`.
The weight includes the channel step size and the source selection multiplier.
"""
function _accumulate_rho_jump!(scratch::ThermalizeScratch{T}, rho::Matrix{T},
    jump::JumpOp, ham, config::Config{Thermalize,D}, precomputed;
    jump_weight_scaling::Real,
) where {T<:Complex,D<:Union{EnergyDomain,TimeDomain,TrotterDomain}}
    labels = precomputed.energy_labels
    data = _frequency_oft_data(config, ham, precomputed)
    prefactor = precomputed.oft_domain_prefactor * jump_weight_scaling
    if Threads.nthreads() > 1 && length(labels) >= OMEGA_THREAD_THRESHOLD
        indices = jump.hermitian ? findall(w -> w <= 1e-12, labels) : collect(eachindex(labels))
        isempty(indices) && (fill!(scratch.rho_jump, 0); return nothing)
        chunks = _partition_range(1:length(indices), min(Threads.nthreads(),length(indices)))
        pool = isempty(scratch.task_scratches) ?
            [ThermalizeScratch(T,size(rho,1)) for _ in chunks] : scratch.task_scratches
        @assert length(pool) >= length(chunks)
        @sync for (idx, chunk) in enumerate(chunks)
            Threads.@spawn _accumulate_rho_jump_chunk_frequency!(pool[idx], rho, jump,
                config.delta, precomputed.transition, labels, data, view(indices,chunk), prefactor)
        end
        fill!(scratch.rho_jump, 0)
        for idx in eachindex(chunks)
            scratch.rho_jump .+= pool[idx].rho_jump
        end
    else
        _accumulate_rho_jump_chunk_frequency!(scratch, rho, jump,
            config.delta, precomputed.transition, labels, data, eachindex(labels), prefactor)
    end
    return nothing
end

function _accumulate_rho_jump_chunk_frequency!(scratch::ThermalizeScratch{T}, rho::Matrix{T},
    jump, delta, transition::F, labels, data::D, indices, prefactor) where {T,F,D}
    fill!(scratch.rho_jump, 0)
    basis = jump.in_eigenbasis::Matrix{T}
    folded = jump.hermitian
    @inbounds for li in indices
        raw = labels[li]
        folded && raw > 1e-12 && continue
        w = folded ? abs(raw) : raw
        _frequency_oft!(scratch.jump_oft, basis, data, w, li, folded)
        rate = prefactor * transition(w)
        mul!(scratch.sandwich_tmp, rho, scratch.jump_oft')
        mul!(scratch.rho_jump, scratch.jump_oft, scratch.sandwich_tmp, delta*rate, 1.0)
        if folded && w > 1e-12
            rate_negative = prefactor * transition(-w)
            mul!(scratch.sandwich_tmp, rho, scratch.jump_oft)
            mul!(scratch.rho_jump, scratch.jump_oft', scratch.sandwich_tmp, delta*rate_negative, 1.0)
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

    _accumulate_rho_jump_chunk_bohr!(scratch, evolving_dm, jump, hamiltonian,
        precomputed_data, bohr_keys, bohr_is, bohr_js, eachindex(bohr_keys);
        scaled_delta=config.delta*jump_weight_scaling)
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
    key_indices::AbstractUnitRange{Int};
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
