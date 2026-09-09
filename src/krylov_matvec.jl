# Sandwich-only helpers. The anticommutator terms are absorbed into the
# precomputed `G_left`/`G_right`; these helpers only perform the two GEMMs.

# No `Workspace` wrapper: callers already own the path-specific scratch.

@inline function _accumulate_sandwich_scratch!(
    out::Matrix{T},
    L_op::Matrix{T},
    rho::Matrix{T},
    scalar::Real,
    sandwich_tmp::Matrix{T},
    sandwich_out::Matrix{T},
) where {T<:Complex}
    CT = one(T)
    ZT = zero(T)
    BLAS.gemm!('N', 'N', CT, L_op, rho, ZT, sandwich_tmp)           # sandwich_tmp = L * rho
    BLAS.gemm!('N', 'C', CT, sandwich_tmp, L_op, ZT, sandwich_out)  # sandwich_out = L * rho * L'
    BLAS.axpy!(T(scalar), sandwich_out, out)
    return nothing
end

@inline function _accumulate_sandwich_adj_scratch!(
    out::Matrix{T},
    L_op::Matrix{T},
    rho::Matrix{T},
    scalar::Real,
    sandwich_tmp::Matrix{T},
    sandwich_out::Matrix{T},
) where {T<:Complex}
    CT = one(T)
    ZT = zero(T)
    BLAS.gemm!('C', 'N', CT, L_op, rho, ZT, sandwich_tmp)           # sandwich_tmp = L' * rho
    BLAS.gemm!('N', 'N', CT, sandwich_tmp, L_op, ZT, sandwich_out)  # sandwich_out = L' * rho * L
    BLAS.axpy!(T(scalar), sandwich_out, out)
    return nothing
end

"""
    _accumulate_sandwich_2op!(out, A, B_dag, rho, scalar, sc) -> nothing

Accumulate `scalar * A * rho * B_dag` into `out`. BohrDomain two-operator sandwich.
"""
function _accumulate_sandwich_2op!(
    out::Matrix{T},
    A::Matrix{T},
    B_dag::Matrix{T},
    rho::Matrix{T},
    scalar::Real,
    sc::KrylovScratch{T},
) where {T<:Complex}
    CT = one(T)
    ZT = zero(T)
    BLAS.gemm!('N', 'N', CT, A, rho, ZT, sc.sandwich_tmp)
    BLAS.gemm!('N', 'N', CT, sc.sandwich_tmp, B_dag, ZT, sc.sandwich_out)
    BLAS.axpy!(T(scalar), sc.sandwich_out, out)
    return nothing
end

"""
    _accumulate_adjoint_sandwich_2op!(out, A, B_dag, rho, scalar, sc) -> nothing

Accumulate `scalar * A' * rho * B_dag'` into `out`. BohrDomain adjoint two-operator sandwich.
"""
function _accumulate_adjoint_sandwich_2op!(
    out::Matrix{T},
    A::Matrix{T},
    B_dag::Matrix{T},
    rho::Matrix{T},
    scalar::Real,
    sc::KrylovScratch{T},
) where {T<:Complex}
    CT = one(T)
    ZT = zero(T)
    BLAS.gemm!('C', 'N', CT, A, rho, ZT, sc.sandwich_tmp)
    BLAS.gemm!('N', 'C', CT, sc.sandwich_tmp, B_dag, ZT, sc.sandwich_out)
    BLAS.axpy!(T(scalar), sc.sandwich_out, out)
    return nothing
end

function _apply_bohr_dissipator!(
    sc::KrylovScratch{T},
    rho::Matrix{T},
    jump_eigenbases::Vector{Matrix{T}},
    bohr_freqs::Matrix{R},
    bohr_dict::Dict{R, Vector{CartesianIndex{2}}},
    alpha::F,
    gamma_norm_factor::Float64,
    ::Val{ADJOINT},
) where {T<:Complex, R<:AbstractFloat, F, ADJOINT}
    A_nu2_dag = sc.bohr_component_dag

    @inbounds for eigenbasis in jump_eigenbases
        for (nu_2, indices) in bohr_dict
            @. sc.jump_oft = alpha(bohr_freqs, nu_2) * eigenbasis

            fill!(A_nu2_dag, 0)
            for idx in indices
                i = idx[1]
                j = idx[2]
                A_nu2_dag[j, i] = conj(eigenbasis[i, j])
            end

            if ADJOINT
                _accumulate_adjoint_sandwich_2op!(
                    sc.rho_out, sc.jump_oft, A_nu2_dag, rho,
                    gamma_norm_factor, sc)
            else
                _accumulate_sandwich_2op!(
                    sc.rho_out, sc.jump_oft, A_nu2_dag, rho,
                    gamma_norm_factor, sc)
            end
        end
    end

    return sc.rho_out
end


"""
    apply_lindbladian!(ws, rho, config, hamiltonian; include_coherent=true)

Apply the compiled Lindbladian into `ws.scratch.rho_out` without changing `rho`.
Set `include_coherent=false` only for a labelled dissipator-only diagnostic.
"""
function apply_lindbladian!(ws::Workspace{KrylovSpectrum}, rho::Matrix{T},
    config::Config{Lindbladian,D}, ham::HamHam; include_coherent::Bool=true,
) where {T<:Complex,D<:Union{BohrDomain,EnergyDomain,TimeDomain,TrotterDomain}}
    return _apply_lindbladian!(ws, rho, config, ham, Val(false), include_coherent)
end

"""
    apply_adjoint_lindbladian!(ws, rho, config, hamiltonian; include_coherent=true)

Apply the Hilbert--Schmidt adjoint into `ws.scratch.rho_out`.
"""
function apply_adjoint_lindbladian!(ws::Workspace{KrylovSpectrum}, rho::Matrix{T},
    config::Config{Lindbladian,D}, ham::HamHam; include_coherent::Bool=true,
) where {T<:Complex,D<:Union{BohrDomain,EnergyDomain,TimeDomain,TrotterDomain}}
    return _apply_lindbladian!(ws, rho, config, ham, Val(true), include_coherent)
end

function _apply_lindbladian!(ws, rho, config, ham, direction, include_coherent)
    _apply_lindbladian_drift!(ws, rho, direction, include_coherent)
    return _apply_lindbladian_gain!(ws, rho, config, ham, direction)
end

@inline function _apply_lindbladian_drift!(ws, rho::Matrix{T}, ::Val{ADJOINT}, include_coherent) where {T,ADJOINT}
    sc = ws.scratch::KrylovScratch{T}
    left = (ADJOINT ? ws.G_right : ws.G_left)::Matrix{T}
    right = (ADJOINT ? ws.G_left : ws.G_right)::Matrix{T}
    if include_coherent
        BLAS.gemm!('N', 'N', one(T), left, rho, zero(T), sc.rho_out)
        BLAS.gemm!('N', 'N', one(T), rho, right, one(T), sc.rho_out)
    else
        neg_R = sc.sandwich_tmp
        @. neg_R = left + right
        BLAS.gemm!('N', 'N', T(0.5), neg_R, rho, zero(T), sc.rho_out)
        BLAS.gemm!('N', 'N', T(0.5), rho, neg_R, one(T), sc.rho_out)
    end
    return nothing
end

@inline _sandwich_action(::Val{false}) = _accumulate_sandwich_scratch!
@inline _sandwich_action(::Val{true}) = _accumulate_sandwich_adj_scratch!

function _apply_lindbladian_gain!(ws, rho::Matrix{T}, config::Config{Lindbladian,BohrDomain}, ham, direction) where {T}
    sc = ws.scratch::KrylovScratch{T}
    bases = ws.jump_eigenbases::Vector{Matrix{T}}
    alpha = ws.bohr_alpha::Function
    scale = ws.gamma_norm_factor::Float64
    if config.transition_weight === nothing
        kernel = alpha::BohrAlphaKernel{typeof(config.construction),typeof(config.beta),Nothing}
        return _apply_bohr_dissipator!(sc, rho, bases, ham.bohr_freqs, ham.bohr_dict, kernel, scale, direction)
    end
    return _apply_bohr_dissipator!(sc, rho, bases, ham.bohr_freqs, ham.bohr_dict, alpha, scale, direction)
end

function _apply_lindbladian_gain!(ws, rho::Matrix{T}, config::Config{Lindbladian,D}, ham, direction) where {T,D<:Union{EnergyDomain,TimeDomain,TrotterDomain}}
    sc = ws.scratch::KrylovScratch{T}
    bases = ws.jump_eigenbases::Vector{Matrix{T}}
    hermitian = ws.jump_hermitian::Vector{Bool}
    labels = ws.energy_labels::Vector{Float64}
    prefactor = (ws.oft_domain_prefactor::Float64) * (ws.gamma_norm_factor::Float64)
    data = D <: EnergyDomain ? (ham.bohr_freqs, _energy_oft_kernel(config)) :
        ws.oft_nufft_prefactors::NUFFTPrefactors{real(T),Array{T,3}}
    if Threads.nthreads() > 1 && length(labels) >= OMEGA_THREAD_THRESHOLD && !isempty(sc.task_scratches)
        return _apply_lindbladian_threaded_frequency!(sc, rho, bases, hermitian, labels,
            config, prefactor, data, direction)
    end
    _populate_jump_frequency_work_list!(sc.work_list, hermitian, labels)
    _apply_lindbladian_chunk_frequency!(sc, rho, bases, hermitian, labels, sc.work_list,
        eachindex(sc.work_list), config, prefactor, data, direction)
    return sc.rho_out
end

# Concrete domain intersections keep DLL dispatch unambiguous.
_apply_lindbladian_gain!(ws, rho::Matrix{T}, config::Config{Lindbladian,BohrDomain,DLL}, ham, direction) where {T} =
    _apply_dll_gain!(ws, rho, direction)
_apply_lindbladian_gain!(ws, rho::Matrix{T}, config::Config{Lindbladian,TimeDomain,DLL}, ham, direction) where {T} =
    _apply_dll_gain!(ws, rho, direction)

function _apply_dll_gain!(ws, rho::Matrix{T}, direction) where {T}
    sc = ws.scratch::KrylovScratch{T}
    operators = ws.dll_lindblads::Vector{Matrix{T}}
    if Threads.nthreads() > 1 && length(operators) >= OMEGA_THREAD_THRESHOLD
        return _apply_lindbladian_threaded_bohr_dll!(sc, rho, operators; adjoint=direction isa Val{true})
    end
    _apply_lindbladian_chunk_bohr_dll!(sc, rho, operators, eachindex(operators); adjoint=direction isa Val{true})
    return sc.rho_out
end

# Work and task buffers belong to the workspace. Reduction order is deterministic.
function _apply_lindbladian_threaded_frequency!(sc::KrylovScratch{T}, rho, bases, hermitian,
    labels, config, prefactor, data::F, direction::V) where {T,F,V}
    work = sc.work_list
    _populate_jump_frequency_work_list!(work, hermitian, labels)
    isempty(work) && return sc.rho_out
    pool = sc.task_scratches
    chunks = _partition_range(1:length(work), min(Threads.nthreads(), length(work), length(pool)))
    @sync for (idx, chunk) in enumerate(chunks)
        Threads.@spawn begin
            fill!(pool[idx].rho_out, 0)
            _apply_lindbladian_chunk_frequency!(pool[idx], rho, bases, hermitian, labels,
                work, chunk, config, prefactor, data, direction)
        end
    end
    for idx in eachindex(chunks)
        sc.rho_out .+= pool[idx].rho_out
    end
    return sc.rho_out
end

function _apply_lindbladian_chunk_frequency!(sc::KrylovScratch{T}, rho::Matrix{T},
    bases::Vector{Matrix{T}}, hermitian, labels, work, chunk, config, prefactor,
    data::F, ::Val{ADJOINT}) where {T,F,ADJOINT}
    positive = _sandwich_action(Val(ADJOINT))
    negative = _sandwich_action(Val(!ADJOINT))
    @inbounds for wi in chunk
        k, li = work[wi]
        folded = hermitian[k]
        w = folded ? abs(labels[li]) : labels[li]
        _frequency_oft!(sc.jump_oft, bases[k], data, w, li, folded)
        positive(sc.rho_out, sc.jump_oft, rho, prefactor*pick_transition(config,w), sc.sandwich_tmp, sc.sandwich_out)
        if folded && w > 1e-12
            negative(sc.rho_out, sc.jump_oft, rho, prefactor*pick_transition(config,-w), sc.sandwich_tmp, sc.sandwich_out)
        end
    end
    return nothing
end

function _apply_lindbladian_threaded_bohr_dll!(sc::KrylovScratch{T}, rho::Matrix{T},
    operators::Vector{Matrix{T}}; adjoint::Bool) where {T<:Complex}
    isempty(operators) && return sc.rho_out
    pool = sc.task_scratches
    nt = min(Threads.nthreads(), length(operators), length(pool))
    if nt < 2
        _apply_lindbladian_chunk_bohr_dll!(sc, rho, operators, eachindex(operators); adjoint)
        return sc.rho_out
    end
    chunks = _partition_range(1:length(operators), nt)
    @sync for (idx, chunk) in enumerate(chunks)
        Threads.@spawn begin
            fill!(pool[idx].rho_out, 0)
            _apply_lindbladian_chunk_bohr_dll!(pool[idx], rho, operators, chunk; adjoint)
        end
    end
    for idx in eachindex(chunks)
        sc.rho_out .+= pool[idx].rho_out
    end
    return sc.rho_out
end

function _apply_lindbladian_chunk_bohr_dll!(sc::KrylovScratch{T}, rho::Matrix{T},
    operators::Vector{Matrix{T}}, chunk; adjoint::Bool) where {T<:Complex}
    sandwich! = adjoint ? _accumulate_sandwich_adj_scratch! : _accumulate_sandwich_scratch!
    @inbounds for k in chunk
        sandwich!(sc.rho_out, operators[k], rho, 1.0, sc.sandwich_tmp, sc.sandwich_out)
    end
    return nothing
end
