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
    apply_lindbladian!(ws, rho, config, hamiltonian) -> sc.rho_out

Apply the energy-domain Lindbladian and return `ws.scratch.rho_out`.

Set `include_coherent=false` only for an explicitly labelled dissipator-only diagnostic.
"""
function apply_lindbladian!(
    ws::Workspace{KrylovSpectrum},
    rho::Matrix{T},
    config::Config{Lindbladian, EnergyDomain},
    hamiltonian::HamHam;
    include_coherent::Bool = true,
) where {T<:Complex}
    sc = ws.scratch::KrylovScratch{T}
    G_left = ws.G_left::Matrix{T}
    G_right = ws.G_right::Matrix{T}
    jump_eigenbases = ws.jump_eigenbases::Vector{Matrix{T}}
    jump_hermitian = ws.jump_hermitian::Vector{Bool}
    prefactor = (ws.oft_domain_prefactor::Float64) * (ws.gamma_norm_factor::Float64)
    energy_labels = ws.energy_labels::Vector{Float64}
    bohr_freqs = hamiltonian.bohr_freqs
    inv_4sigma2 = 1.0 / (4 * config.sigma^2)

    CT = one(T)
    ZT = zero(T)

    if include_coherent
        BLAS.gemm!('N', 'N', CT, G_left, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', CT, rho, G_right, CT, sc.rho_out)
    else
        neg_R = sc.sandwich_tmp
        @. neg_R = G_left + G_right
        half = T(0.5)
        BLAS.gemm!('N', 'N', half, neg_R, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', half, rho, neg_R, CT, sc.rho_out)
    end

    if Threads.nthreads() > 1 && length(energy_labels) >= OMEGA_THREAD_THRESHOLD
        return _apply_lindbladian_threaded_energy!(
            sc, rho, jump_eigenbases, jump_hermitian, bohr_freqs,
            energy_labels, config, prefactor, inv_4sigma2; adjoint=false)
    end

    for (k, eigenbasis) in enumerate(jump_eigenbases)
        is_herm = jump_hermitian[k]
        if is_herm
            for w_raw in energy_labels
                w_raw > 1e-12 && continue
                w = abs(w_raw)

                oft!(sc.jump_oft, eigenbasis, bohr_freqs, w, inv_4sigma2)

                scalar_w = prefactor * pick_transition(config, w)
                _accumulate_sandwich_scratch!(sc.rho_out, sc.jump_oft, rho, scalar_w, sc.sandwich_tmp, sc.sandwich_out)

                if w > 1e-12
                    scalar_neg = prefactor * pick_transition(config, -w)
                    _accumulate_sandwich_adj_scratch!(sc.rho_out, sc.jump_oft, rho, scalar_neg, sc.sandwich_tmp, sc.sandwich_out)
                end
            end
        else
            for w in energy_labels
                oft!(sc.jump_oft, eigenbasis, bohr_freqs, w, inv_4sigma2)
                scalar_w = prefactor * pick_transition(config, w)
                _accumulate_sandwich_scratch!(sc.rho_out, sc.jump_oft, rho, scalar_w, sc.sandwich_tmp, sc.sandwich_out)
            end
        end
    end

    return sc.rho_out
end

"""
    apply_adjoint_lindbladian!(ws, rho, config, hamiltonian) -> sc.rho_out

Apply the Hilbert--Schmidt adjoint energy-domain Lindbladian.

Set `include_coherent=false` only for an explicitly labelled dissipator-only diagnostic.
"""
function apply_adjoint_lindbladian!(
    ws::Workspace{KrylovSpectrum},
    rho::Matrix{T},
    config::Config{Lindbladian, EnergyDomain},
    hamiltonian::HamHam;
    include_coherent::Bool = true,
) where {T<:Complex}
    sc = ws.scratch::KrylovScratch{T}
    G_left_adj = ws.G_right::Matrix{T}
    G_right_adj = ws.G_left::Matrix{T}
    jump_eigenbases = ws.jump_eigenbases::Vector{Matrix{T}}
    jump_hermitian = ws.jump_hermitian::Vector{Bool}
    prefactor = (ws.oft_domain_prefactor::Float64) * (ws.gamma_norm_factor::Float64)
    energy_labels = ws.energy_labels::Vector{Float64}
    bohr_freqs = hamiltonian.bohr_freqs
    inv_4sigma2 = 1.0 / (4 * config.sigma^2)

    CT = one(T)
    ZT = zero(T)

    if include_coherent
        BLAS.gemm!('N', 'N', CT, G_left_adj, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', CT, rho, G_right_adj, CT, sc.rho_out)
    else
        neg_R = sc.sandwich_tmp
        @. neg_R = G_left_adj + G_right_adj
        half = T(0.5)
        BLAS.gemm!('N', 'N', half, neg_R, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', half, rho, neg_R, CT, sc.rho_out)
    end

    if Threads.nthreads() > 1 && length(energy_labels) >= OMEGA_THREAD_THRESHOLD
        return _apply_lindbladian_threaded_energy!(
            sc, rho, jump_eigenbases, jump_hermitian, bohr_freqs,
            energy_labels, config, prefactor, inv_4sigma2; adjoint=true)
    end

    for (k, eigenbasis) in enumerate(jump_eigenbases)
        is_herm = jump_hermitian[k]
        if is_herm
            for w_raw in energy_labels
                w_raw > 1e-12 && continue
                w = abs(w_raw)

                oft!(sc.jump_oft, eigenbasis, bohr_freqs, w, inv_4sigma2)

                scalar_w = prefactor * pick_transition(config, w)
                _accumulate_sandwich_adj_scratch!(sc.rho_out, sc.jump_oft, rho, scalar_w, sc.sandwich_tmp, sc.sandwich_out)

                if w > 1e-12
                    scalar_neg = prefactor * pick_transition(config, -w)
                    _accumulate_sandwich_scratch!(sc.rho_out, sc.jump_oft, rho, scalar_neg, sc.sandwich_tmp, sc.sandwich_out)
                end
            end
        else
            for w in energy_labels
                oft!(sc.jump_oft, eigenbasis, bohr_freqs, w, inv_4sigma2)
                scalar_w = prefactor * pick_transition(config, w)
                _accumulate_sandwich_adj_scratch!(sc.rho_out, sc.jump_oft, rho, scalar_w, sc.sandwich_tmp, sc.sandwich_out)
            end
        end
    end

    return sc.rho_out
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
    apply_lindbladian!(ws, rho, config, hamiltonian) -> sc.rho_out

Apply the Bohr-domain Lindbladian and return `ws.scratch.rho_out`.
"""
function apply_lindbladian!(
    ws::Workspace{KrylovSpectrum},
    rho::Matrix{T},
    config::Config{Lindbladian, BohrDomain},
    hamiltonian::HamHam;
    include_coherent::Bool = true,
) where {T<:Complex}
    sc = ws.scratch::KrylovScratch{T}
    gamma_norm_factor = ws.gamma_norm_factor::Float64
    G_left = ws.G_left::Matrix{T}
    G_right = ws.G_right::Matrix{T}
    jump_eigenbases = ws.jump_eigenbases::Vector{Matrix{T}}
    alpha = ws.bohr_alpha::Function

    CT = one(T)
    ZT = zero(T)

    if include_coherent
        BLAS.gemm!('N', 'N', CT, G_left, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', CT, rho, G_right, CT, sc.rho_out)
    else
        neg_R = sc.sandwich_tmp
        @. neg_R = G_left + G_right
        half = T(0.5)
        BLAS.gemm!('N', 'N', half, neg_R, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', half, rho, neg_R, CT, sc.rho_out)
    end

    return _apply_bohr_dissipator!(
        sc, rho, jump_eigenbases, hamiltonian.bohr_freqs,
        hamiltonian.bohr_dict, alpha, gamma_norm_factor, Val(false))
end

"""
    apply_adjoint_lindbladian!(ws, rho, config, hamiltonian) -> sc.rho_out

Apply the Hilbert--Schmidt adjoint Bohr-domain Lindbladian.
"""
function apply_adjoint_lindbladian!(
    ws::Workspace{KrylovSpectrum},
    rho::Matrix{T},
    config::Config{Lindbladian, BohrDomain},
    hamiltonian::HamHam;
    include_coherent::Bool = true,
) where {T<:Complex}
    sc = ws.scratch::KrylovScratch{T}
    gamma_norm_factor = ws.gamma_norm_factor::Float64
    G_left_adj = ws.G_right::Matrix{T}
    G_right_adj = ws.G_left::Matrix{T}
    jump_eigenbases = ws.jump_eigenbases::Vector{Matrix{T}}
    alpha = ws.bohr_alpha::Function

    CT = one(T)
    ZT = zero(T)

    if include_coherent
        BLAS.gemm!('N', 'N', CT, G_left_adj, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', CT, rho, G_right_adj, CT, sc.rho_out)
    else
        neg_R = sc.sandwich_tmp
        @. neg_R = G_left_adj + G_right_adj
        half = T(0.5)
        BLAS.gemm!('N', 'N', half, neg_R, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', half, rho, neg_R, CT, sc.rho_out)
    end

    return _apply_bohr_dissipator!(
        sc, rho, jump_eigenbases, hamiltonian.bohr_freqs,
        hamiltonian.bohr_dict, alpha, gamma_norm_factor, Val(true))
end

# Concrete domain entry points avoid ambiguous intersections with the legacy
# Bohr methods; both dispatch into the same retained-matrix function barrier.
for D in (BohrDomain, TimeDomain)
    @eval begin
        function apply_lindbladian!(
            ws::Workspace{KrylovSpectrum}, rho::Matrix{T},
            config::Config{Lindbladian, $D, DLL}, hamiltonian::HamHam;
            include_coherent::Bool=true,
        ) where {T<:Complex}
            return _apply_lindbladian_dll!(ws, rho; include_coherent)
        end

        function apply_adjoint_lindbladian!(
            ws::Workspace{KrylovSpectrum}, rho::Matrix{T},
            config::Config{Lindbladian, $D, DLL}, hamiltonian::HamHam;
            include_coherent::Bool=true,
        ) where {T<:Complex}
            return _apply_adjoint_lindbladian_dll!(ws, rho; include_coherent)
        end
    end
end

# DLL uses one Lindblad matrix per coupling or channel.
# Math: $L(rho) = G_L rho + rho G_R + sum_a L_a rho L_a^dagger$.

"""
    _apply_lindbladian_dll!(ws, rho) -> sc.rho_out

Apply the DLL Bohr- or Time-domain Lindbladian.
"""
function _apply_lindbladian_dll!(
    ws::Workspace{KrylovSpectrum},
    rho::Matrix{T};
    include_coherent::Bool = true,
) where {T<:Complex}
    sc = ws.scratch::KrylovScratch{T}
    G_left  = ws.G_left::Matrix{T}
    G_right = ws.G_right::Matrix{T}
    dll_lindblads = ws.dll_lindblads::Vector{Matrix{T}}

    CT = one(T)
    ZT = zero(T)

    if include_coherent
        BLAS.gemm!('N', 'N', CT, G_left, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', CT, rho, G_right, CT, sc.rho_out)
    else
        neg_R = sc.sandwich_tmp
        @. neg_R = G_left + G_right
        half = T(0.5)
        BLAS.gemm!('N', 'N', half, neg_R, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', half, rho, neg_R, CT, sc.rho_out)
    end

    if Threads.nthreads() > 1 && length(dll_lindblads) >= OMEGA_THREAD_THRESHOLD
        return _apply_lindbladian_threaded_bohr_dll!(sc, rho, dll_lindblads; adjoint=false)
    end

    for L_a in dll_lindblads
        _accumulate_sandwich_scratch!(sc.rho_out, L_a, rho, 1.0,
                                      sc.sandwich_tmp, sc.sandwich_out)
    end

    return sc.rho_out
end

"""
    _apply_adjoint_lindbladian_dll!(ws, rho) -> sc.rho_out

Apply the Hilbert--Schmidt adjoint DLL Bohr- or Time-domain Lindbladian.
"""
function _apply_adjoint_lindbladian_dll!(
    ws::Workspace{KrylovSpectrum},
    rho::Matrix{T};
    include_coherent::Bool = true,
) where {T<:Complex}
    sc = ws.scratch::KrylovScratch{T}
    G_left_adj  = ws.G_right::Matrix{T}
    G_right_adj = ws.G_left::Matrix{T}
    dll_lindblads = ws.dll_lindblads::Vector{Matrix{T}}

    CT = one(T)
    ZT = zero(T)

    if include_coherent
        BLAS.gemm!('N', 'N', CT, G_left_adj, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', CT, rho, G_right_adj, CT, sc.rho_out)
    else
        neg_R = sc.sandwich_tmp
        @. neg_R = G_left_adj + G_right_adj
        half = T(0.5)
        BLAS.gemm!('N', 'N', half, neg_R, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', half, rho, neg_R, CT, sc.rho_out)
    end

    if Threads.nthreads() > 1 && length(dll_lindblads) >= OMEGA_THREAD_THRESHOLD
        return _apply_lindbladian_threaded_bohr_dll!(sc, rho, dll_lindblads; adjoint=true)
    end

    for L_a in dll_lindblads
        _accumulate_sandwich_adj_scratch!(sc.rho_out, L_a, rho, 1.0,
                                          sc.sandwich_tmp, sc.sandwich_out)
    end

    return sc.rho_out
end

"""
    apply_lindbladian!(ws, rho, config, hamiltonian) -> sc.rho_out

Apply a time- or Trotter-domain Lindbladian.
"""
function apply_lindbladian!(
    ws::Workspace{KrylovSpectrum},
    rho::Matrix{T},
    config::Config{Lindbladian, D},
    hamiltonian::HamHam;
    include_coherent::Bool = true,
) where {T<:Complex, D<:Union{TimeDomain, TrotterDomain}}
    sc = ws.scratch::KrylovScratch{T}
    _nufft = ws.oft_nufft_prefactors::NUFFTPrefactors{real(T), Array{T, 3}}
    nufft_data = _nufft.data
    nufft_idx = _nufft.energy_to_index
    G_left = ws.G_left::Matrix{T}
    G_right = ws.G_right::Matrix{T}
    jump_eigenbases = ws.jump_eigenbases::Vector{Matrix{T}}
    jump_hermitian = ws.jump_hermitian::Vector{Bool}
    prefactor = (ws.oft_domain_prefactor::Float64) * (ws.gamma_norm_factor::Float64)
    energy_labels = ws.energy_labels::Vector{Float64}

    CT = one(T)
    ZT = zero(T)

    if include_coherent
        BLAS.gemm!('N', 'N', CT, G_left, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', CT, rho, G_right, CT, sc.rho_out)
    else
        neg_R = sc.sandwich_tmp
        @. neg_R = G_left + G_right
        half = T(0.5)
        BLAS.gemm!('N', 'N', half, neg_R, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', half, rho, neg_R, CT, sc.rho_out)
    end

    if Threads.nthreads() > 1 && length(energy_labels) >= OMEGA_THREAD_THRESHOLD
        return _apply_lindbladian_threaded_timetrot!(
            sc, rho, jump_eigenbases, jump_hermitian,
            nufft_data, nufft_idx, energy_labels, config, prefactor; adjoint=false)
    end

    for (k, eigenbasis) in enumerate(jump_eigenbases)
        is_herm = jump_hermitian[k]
        if is_herm
            for w_raw in energy_labels
                w_raw > 1e-12 && continue
                w = abs(w_raw)

                nufft_prefactor_matrix = @view nufft_data[:, :, nufft_idx[w]]
                @. sc.jump_oft = eigenbasis * nufft_prefactor_matrix

                scalar_w = prefactor * pick_transition(config, w)
                _accumulate_sandwich_scratch!(sc.rho_out, sc.jump_oft, rho, scalar_w, sc.sandwich_tmp, sc.sandwich_out)

                if w > 1e-12
                    scalar_neg = prefactor * pick_transition(config, -w)
                    _accumulate_sandwich_adj_scratch!(sc.rho_out, sc.jump_oft, rho, scalar_neg, sc.sandwich_tmp, sc.sandwich_out)
                end
            end
        else
            for (i, w) in enumerate(energy_labels)
                nufft_prefactor_matrix = @view nufft_data[:, :, i]
                @. sc.jump_oft = eigenbasis * nufft_prefactor_matrix
                scalar_w = prefactor * pick_transition(config, w)
                _accumulate_sandwich_scratch!(sc.rho_out, sc.jump_oft, rho, scalar_w, sc.sandwich_tmp, sc.sandwich_out)
            end
        end
    end

    return sc.rho_out
end

"""
    apply_adjoint_lindbladian!(ws, rho, config, hamiltonian) -> sc.rho_out

Apply the Hilbert--Schmidt adjoint time- or Trotter-domain Lindbladian.
"""
function apply_adjoint_lindbladian!(
    ws::Workspace{KrylovSpectrum},
    rho::Matrix{T},
    config::Config{Lindbladian, D},
    hamiltonian::HamHam;
    include_coherent::Bool = true,
) where {T<:Complex, D<:Union{TimeDomain, TrotterDomain}}
    sc = ws.scratch::KrylovScratch{T}
    _nufft = ws.oft_nufft_prefactors::NUFFTPrefactors{real(T), Array{T, 3}}
    nufft_data = _nufft.data
    nufft_idx = _nufft.energy_to_index
    G_left_adj = ws.G_right::Matrix{T}
    G_right_adj = ws.G_left::Matrix{T}
    jump_eigenbases = ws.jump_eigenbases::Vector{Matrix{T}}
    jump_hermitian = ws.jump_hermitian::Vector{Bool}
    prefactor = (ws.oft_domain_prefactor::Float64) * (ws.gamma_norm_factor::Float64)
    energy_labels = ws.energy_labels::Vector{Float64}

    CT = one(T)
    ZT = zero(T)

    if include_coherent
        BLAS.gemm!('N', 'N', CT, G_left_adj, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', CT, rho, G_right_adj, CT, sc.rho_out)
    else
        neg_R = sc.sandwich_tmp
        @. neg_R = G_left_adj + G_right_adj
        half = T(0.5)
        BLAS.gemm!('N', 'N', half, neg_R, rho, ZT, sc.rho_out)
        BLAS.gemm!('N', 'N', half, rho, neg_R, CT, sc.rho_out)
    end

    if Threads.nthreads() > 1 && length(energy_labels) >= OMEGA_THREAD_THRESHOLD
        return _apply_lindbladian_threaded_timetrot!(
            sc, rho, jump_eigenbases, jump_hermitian,
            nufft_data, nufft_idx, energy_labels, config, prefactor; adjoint=true)
    end

    for (k, eigenbasis) in enumerate(jump_eigenbases)
        is_herm = jump_hermitian[k]
        if is_herm
            for w_raw in energy_labels
                w_raw > 1e-12 && continue
                w = abs(w_raw)

                nufft_prefactor_matrix = @view nufft_data[:, :, nufft_idx[w]]
                @. sc.jump_oft = eigenbasis * nufft_prefactor_matrix

                scalar_w = prefactor * pick_transition(config, w)
                _accumulate_sandwich_adj_scratch!(sc.rho_out, sc.jump_oft, rho, scalar_w, sc.sandwich_tmp, sc.sandwich_out)

                if w > 1e-12
                    scalar_neg = prefactor * pick_transition(config, -w)
                    _accumulate_sandwich_scratch!(sc.rho_out, sc.jump_oft, rho, scalar_neg, sc.sandwich_tmp, sc.sandwich_out)
                end
            end
        else
            for (i, w) in enumerate(energy_labels)
                nufft_prefactor_matrix = @view nufft_data[:, :, i]
                @. sc.jump_oft = eigenbasis * nufft_prefactor_matrix
                scalar_w = prefactor * pick_transition(config, w)
                _accumulate_sandwich_adj_scratch!(sc.rho_out, sc.jump_oft, rho, scalar_w, sc.sandwich_tmp, sc.sandwich_out)
            end
        end
    end

    return sc.rho_out
end

# Threaded frequency loops use a flat `(jump_idx, label_idx)` work list and
# private output matrices before the final deterministic reduction.

function _apply_lindbladian_threaded_energy!(
    sc::KrylovScratch{T},
    rho::Matrix{T},
    jump_eigenbases::Vector{Matrix{T}},
    jump_hermitian::Vector{Bool},
    bohr_freqs::AbstractMatrix{<:Real},
    energy_labels::Vector{Float64},
    config::Config{Lindbladian, EnergyDomain},
    prefactor::Float64,
    inv_4sigma2::Float64;
    adjoint::Bool,
) where {T<:Complex}
    # `work` is the scratch's pre-allocated buffer; the population helper does
    # `empty!` + `push!` which is zero-alloc when the buffer is large enough
    # (the Workspace constructor sized it for the production label set).
    work = sc.work_list
    _populate_jump_frequency_work_list!(work, jump_hermitian, energy_labels)
    n_work = length(work)
    n_work == 0 && return sc.rho_out

    pool = sc.task_scratches
    nt = min(Threads.nthreads(), n_work, length(pool))
    chunks = _partition_range(1:n_work, nt)

    @sync for (idx, chunk) in enumerate(chunks)
        Threads.@spawn _apply_lindbladian_chunk_energy!(
            pool[idx], rho, jump_eigenbases, jump_hermitian,
            bohr_freqs, energy_labels, work, chunk, config,
            prefactor, inv_4sigma2; adjoint=adjoint)
    end

    @inbounds for idx in 1:length(chunks)
        sc.rho_out .+= pool[idx].rho_out
    end

    return sc.rho_out
end

function _apply_lindbladian_chunk_energy!(
    task_sc::KrylovScratch{T},
    rho::Matrix{T},
    jump_eigenbases::Vector{Matrix{T}},
    jump_hermitian::Vector{Bool},
    bohr_freqs::AbstractMatrix{<:Real},
    energy_labels::Vector{Float64},
    work::Vector{Tuple{Int, Int}},
    chunk::UnitRange{Int},
    config::Config{Lindbladian, EnergyDomain},
    prefactor::Float64,
    inv_4sigma2::Float64;
    adjoint::Bool,
) where {T<:Complex}
    fill!(task_sc.rho_out, 0)

    @inbounds for w_idx in chunk
        (k, li) = work[w_idx]
        eigenbasis = jump_eigenbases[k]
        is_herm = jump_hermitian[k]

        w_raw = energy_labels[li]
        # Hermitian fold: only `w_raw <= 1e-12` is queued, OFT and rate use
        # `w = |w_raw|` (matches serial). Non-Hermitian: `w = w_raw` directly,
        # OFT and rate take the signed value.
        w = is_herm ? abs(w_raw) : w_raw

        oft!(task_sc.jump_oft, eigenbasis, bohr_freqs, w, inv_4sigma2)

        scalar_w = prefactor * pick_transition(config, w)
        if adjoint
            _accumulate_sandwich_adj_scratch!(task_sc.rho_out, task_sc.jump_oft, rho, scalar_w,
                                              task_sc.sandwich_tmp, task_sc.sandwich_out)
        else
            _accumulate_sandwich_scratch!(task_sc.rho_out, task_sc.jump_oft, rho, scalar_w,
                                          task_sc.sandwich_tmp, task_sc.sandwich_out)
        end

        if is_herm && w > 1e-12
            scalar_neg = prefactor * pick_transition(config, -w)
            if adjoint
                _accumulate_sandwich_scratch!(task_sc.rho_out, task_sc.jump_oft, rho, scalar_neg,
                                              task_sc.sandwich_tmp, task_sc.sandwich_out)
            else
                _accumulate_sandwich_adj_scratch!(task_sc.rho_out, task_sc.jump_oft, rho, scalar_neg,
                                                  task_sc.sandwich_tmp, task_sc.sandwich_out)
            end
        end
    end

    return nothing
end

# The DLL Bohr dissipator Σ_a L_a ρ L_a† is a flat sum over the per-jump
# operators in `ws.dll_lindblads` (one dense L_a per coupling — or per channel
# for multi-channel filters). Parallelise over the jump index `a`: the direct
# analogue of the EnergyDomain ω-loop, where each (jump, ω) term is itself one
# Lindblad operator. Each task accumulates a private `rho_out`; the coherent
# term (already in `sc.rho_out` when this is called) is preserved by the `.+=`
# reduction.

function _apply_lindbladian_threaded_bohr_dll!(
    sc::KrylovScratch{T},
    rho::Matrix{T},
    dll_lindblads::Vector{Matrix{T}};
    adjoint::Bool,
) where {T<:Complex}
    n_jumps = length(dll_lindblads)
    n_jumps == 0 && return sc.rho_out

    pool = sc.task_scratches
    nt = min(Threads.nthreads(), n_jumps, length(pool))
    if nt < 2
        # Pool unavailable (e.g. nthreads changed since workspace construction)
        # — fall back to the serial sum into sc.rho_out (coherent already there).
        @inbounds for L_a in dll_lindblads
            if adjoint
                _accumulate_sandwich_adj_scratch!(sc.rho_out, L_a, rho, 1.0,
                                                  sc.sandwich_tmp, sc.sandwich_out)
            else
                _accumulate_sandwich_scratch!(sc.rho_out, L_a, rho, 1.0,
                                              sc.sandwich_tmp, sc.sandwich_out)
            end
        end
        return sc.rho_out
    end
    chunks = _partition_range(1:n_jumps, nt)

    @sync for (idx, chunk) in enumerate(chunks)
        Threads.@spawn _apply_lindbladian_chunk_bohr_dll!(
            pool[idx], rho, dll_lindblads, chunk; adjoint=adjoint)
    end

    @inbounds for idx in 1:length(chunks)
        sc.rho_out .+= pool[idx].rho_out
    end

    return sc.rho_out
end

function _apply_lindbladian_chunk_bohr_dll!(
    task_sc::KrylovScratch{T},
    rho::Matrix{T},
    dll_lindblads::Vector{Matrix{T}},
    chunk::UnitRange{Int};
    adjoint::Bool,
) where {T<:Complex}
    fill!(task_sc.rho_out, 0)
    @inbounds for k in chunk
        L_a = dll_lindblads[k]
        if adjoint
            _accumulate_sandwich_adj_scratch!(task_sc.rho_out, L_a, rho, 1.0,
                                              task_sc.sandwich_tmp, task_sc.sandwich_out)
        else
            _accumulate_sandwich_scratch!(task_sc.rho_out, L_a, rho, 1.0,
                                          task_sc.sandwich_tmp, task_sc.sandwich_out)
        end
    end
    return nothing
end

# --- TimeDomain / TrotterDomain threaded variant ---

function _apply_lindbladian_threaded_timetrot!(
    sc::KrylovScratch{T},
    rho::Matrix{T},
    jump_eigenbases::Vector{Matrix{T}},
    jump_hermitian::Vector{Bool},
    nufft_data::AbstractArray{T, 3},
    nufft_idx::AbstractDict,
    energy_labels::Vector{Float64},
    config::Config{Lindbladian, D},
    prefactor::Float64;
    adjoint::Bool,
) where {T<:Complex, D<:Union{TimeDomain, TrotterDomain}}
    work = sc.work_list
    _populate_jump_frequency_work_list!(work, jump_hermitian, energy_labels)
    n_work = length(work)
    n_work == 0 && return sc.rho_out

    pool = sc.task_scratches
    nt = min(Threads.nthreads(), n_work, length(pool))
    chunks = _partition_range(1:n_work, nt)

    @sync for (idx, chunk) in enumerate(chunks)
        Threads.@spawn _apply_lindbladian_chunk_timetrot!(
            pool[idx], rho, jump_eigenbases, jump_hermitian,
            nufft_data, nufft_idx, energy_labels, work, chunk, config,
            prefactor; adjoint=adjoint)
    end

    @inbounds for idx in 1:length(chunks)
        sc.rho_out .+= pool[idx].rho_out
    end

    return sc.rho_out
end

function _apply_lindbladian_chunk_timetrot!(
    task_sc::KrylovScratch{T},
    rho::Matrix{T},
    jump_eigenbases::Vector{Matrix{T}},
    jump_hermitian::Vector{Bool},
    nufft_data::AbstractArray{T, 3},
    nufft_idx::AbstractDict,
    energy_labels::Vector{Float64},
    work::Vector{Tuple{Int, Int}},
    chunk::UnitRange{Int},
    config::Config{Lindbladian, D},
    prefactor::Float64;
    adjoint::Bool,
) where {T<:Complex, D<:Union{TimeDomain, TrotterDomain}}
    fill!(task_sc.rho_out, 0)

    @inbounds for w_idx in chunk
        (k, li) = work[w_idx]
        eigenbasis = jump_eigenbases[k]
        is_herm = jump_hermitian[k]

        w_raw = energy_labels[li]
        # Hermitian fold: only `w_raw <= 1e-12` queued; rate uses `|w_raw|`,
        # NUFFT prefactor index found via `nufft_idx[|w_raw|]`. Non-Hermitian:
        # rate uses signed `w_raw`; prefactor index is the label index `li`.
        w = is_herm ? abs(w_raw) : w_raw
        prefactor_idx = is_herm ? nufft_idx[w] : li
        nufft_prefactor_matrix = @view nufft_data[:, :, prefactor_idx]
        @. task_sc.jump_oft = eigenbasis * nufft_prefactor_matrix

        scalar_w = prefactor * pick_transition(config, w)
        if adjoint
            _accumulate_sandwich_adj_scratch!(task_sc.rho_out, task_sc.jump_oft, rho, scalar_w,
                                              task_sc.sandwich_tmp, task_sc.sandwich_out)
        else
            _accumulate_sandwich_scratch!(task_sc.rho_out, task_sc.jump_oft, rho, scalar_w,
                                          task_sc.sandwich_tmp, task_sc.sandwich_out)
        end

        if is_herm && w > 1e-12
            scalar_neg = prefactor * pick_transition(config, -w)
            if adjoint
                _accumulate_sandwich_scratch!(task_sc.rho_out, task_sc.jump_oft, rho, scalar_neg,
                                              task_sc.sandwich_tmp, task_sc.sandwich_out)
            else
                _accumulate_sandwich_adj_scratch!(task_sc.rho_out, task_sc.jump_oft, rho, scalar_neg,
                                                  task_sc.sandwich_tmp, task_sc.sandwich_out)
            end
        end
    end

    return nothing
end
