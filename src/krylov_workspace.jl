"""
    Workspace(config::Config{Lindbladian}, hamiltonian, jumps; trotter=nothing)

Construct a matrix-free Lindbladian workspace and preallocate its scratch data.

A `Workspace{KrylovSpectrum,D,C,T}` matching the configuration and Hamiltonian.
"""
function Workspace(
    config::Config{Lindbladian},
    hamiltonian::HamHam,
    jumps::Vector{JumpOp};
    trotter::Union{AbstractTrotter, Nothing}=nothing,
)
    validate_config!(config, hamiltonian)

    # Determine ham_or_trott (mirrors construct_lindbladian in furnace.jl)
    ham_or_trott = if config.domain isa TrotterDomain
        trotter === nothing && error("A Trotter object must be provided for the TrotterDomain")
        _validate_trotter_cache!(config, hamiltonian, trotter)
        trotter
    else
        hamiltonian
    end

    # Precompute domain-specific data (transition, energy_labels, gamma_norm_factor, ...)
    precomputed_data = _precompute_data(config, ham_or_trott)

    # Precompute coherent B_total (returns nothing for GNS / with_coherent=false)
    B_total = _precompute_coherent_B(jumps, ham_or_trott, config, precomputed_data)

    # Determine dimensions and element type
    dim = size(hamiltonian.data, 1)
    T = eltype(hamiltonian.eigvals)
    CT = Complex{T}

    # Extract concrete-typed eigenbasis matrices and hermitian flags
    jump_eigenbases = [Matrix{CT}(j.in_eigenbasis) for j in jumps]
    jump_hermitian  = [j.hermitian for j in jumps]

    # Allocate KrylovScratch (no channel_rho_jump for Lindbladian)
    sc = KrylovScratch(CT, dim; with_channel_rho_jump=false)

    # Precompute the anticommutator and coherent left/right factors.
    R_total = zeros(CT, dim, dim)
    _accumulate_R_total!(R_total, jump_eigenbases, jump_hermitian,
                         precomputed_data, config, ham_or_trott)
    hermitianize!(R_total)

    if B_total !== nothing
        B = Matrix{CT}(B_total)
        G_left  = Matrix{CT}(-1im .* B .- 0.5 .* R_total)
        G_right = Matrix{CT}(1im .* B .- 0.5 .* R_total)
    else
        G_left  = Matrix{CT}(-0.5 .* R_total)
        G_right = Matrix{CT}(-0.5 .* R_total)
    end

    # Absorb precomputed_data fields into flat workspace fields
    pd_transition = hasproperty(precomputed_data, :transition) ? precomputed_data.transition : nothing
    pd_gnf = hasproperty(precomputed_data, :gamma_norm_factor) ? precomputed_data.gamma_norm_factor : nothing
    pd_el = hasproperty(precomputed_data, :energy_labels) ? precomputed_data.energy_labels : nothing
    pd_odp = hasproperty(precomputed_data, :oft_domain_prefactor) ? precomputed_data.oft_domain_prefactor : nothing
    pd_nufft = hasproperty(precomputed_data, :oft_nufft_prefactors) ? precomputed_data.oft_nufft_prefactors : nothing
    pd_alpha = hasproperty(precomputed_data, :alpha) ? precomputed_data.alpha : nothing
    pd_bkeys = hasproperty(precomputed_data, :bohr_keys) ? precomputed_data.bohr_keys : nothing
    pd_bis = hasproperty(precomputed_data, :bohr_is) ? precomputed_data.bohr_is : nothing
    pd_bjs = hasproperty(precomputed_data, :bohr_js) ? precomputed_data.bohr_js : nothing
    pd_bminus = hasproperty(precomputed_data, :b_minus) ? precomputed_data.b_minus : nothing
    pd_bplus = hasproperty(precomputed_data, :b_plus) ? precomputed_data.b_plus : nothing

    # Pre-build the `(jump_idx, label_idx)` work list for threaded reductions.
    # ω-loop dispatch in apply_lindbladian! (EnergyDomain / TimeDomain /
    # TrotterDomain). BohrDomain leaves the list empty.
    if pd_el !== nothing
        _populate_jump_frequency_work_list!(sc.work_list, jump_hermitian, pd_el)
    end

    D = typeof(config.domain)
    C = typeof(config.construction)

    return Workspace{KrylovSpectrum, D, C, T}(
        jump_eigenbases, jump_hermitian, copy(jumps),
        nothing,  # dll_lindblads (CKG/GNS path; populated by DLL specialised constructor)
        G_left, G_right,
        pd_transition, pd_gnf, pd_el, pd_odp, pd_nufft,
        pd_alpha, pd_bkeys, pd_bis, pd_bjs, pd_bminus, pd_bplus,
        ham_or_trott, nothing, nothing, nothing,  # source provenance; no channel matrices
        sc,
        config,   # Cached to validate predictor workspace reuse.
    )
end

"""
    _accumulate_R_total!(R, ws, config, hamiltonian) -> nothing

Accumulate the total rate operator during workspace construction.

Math: \$R = sum_(a,omega) rate(omega)^2 L_(a,omega)^dagger L_(a,omega)\$.
"""
function _accumulate_R_total!(
    R::Matrix{T},
    ws_eigenbases::Vector{Matrix{T}},
    ws_hermitian::Vector{Bool},
    precomputed_data,
    config::Config{<:Any, EnergyDomain},
    hamiltonian::HamHam,
) where {T<:Complex}
    (; transition, gamma_norm_factor, energy_labels) = precomputed_data
    bohr_freqs = hamiltonian.bohr_freqs
    inv_4sigma2 = 1.0 / (4 * config.sigma^2)
    prefactor = precomputed_data.oft_domain_prefactor * gamma_norm_factor

    n_labels = length(energy_labels)
    n_jumps  = length(ws_eigenbases)
    if Threads.nthreads() > 1 && n_jumps * n_labels >= OMEGA_THREAD_THRESHOLD
        return _accumulate_R_total_threaded_energy!(
            R, ws_eigenbases, ws_hermitian, bohr_freqs, energy_labels,
            transition, prefactor, inv_4sigma2)
    end

    dim = size(R, 1)
    jump_oft = zeros(T, dim, dim)
    LdagL = zeros(T, dim, dim)

    for (k, eigenbasis) in enumerate(ws_eigenbases)
        is_herm = ws_hermitian[k]
        if is_herm
            for w_raw in energy_labels
                w_raw > 1e-12 && continue
                w = abs(w_raw)
                oft!(jump_oft, eigenbasis, bohr_freqs, w, inv_4sigma2)
                rate2 = prefactor * transition(w)
                mul!(LdagL, jump_oft', jump_oft)
                @. R += rate2 * LdagL
                if w > 1e-12
                    rate2_neg = prefactor * transition(-w)
                    mul!(LdagL, jump_oft, jump_oft')
                    @. R += rate2_neg * LdagL
                end
            end
        else
            for w in energy_labels
                oft!(jump_oft, eigenbasis, bohr_freqs, w, inv_4sigma2)
                rate2 = prefactor * transition(w)
                mul!(LdagL, jump_oft', jump_oft)
                @. R += rate2 * LdagL
            end
        end
    end
    return nothing
end

function _accumulate_R_total!(
    R::Matrix{T},
    ws_eigenbases::Vector{Matrix{T}},
    ws_hermitian::Vector{Bool},
    precomputed_data,
    config::Config{<:Any, D},
    ham_or_trott::Union{HamHam, AbstractTrotter},
) where {T<:Complex, D<:Union{TimeDomain, TrotterDomain}}
    (; transition, gamma_norm_factor, energy_labels, oft_nufft_prefactors) = precomputed_data
    prefactor = precomputed_data.oft_domain_prefactor * gamma_norm_factor

    n_labels = length(energy_labels)
    n_jumps  = length(ws_eigenbases)
    if Threads.nthreads() > 1 && n_jumps * n_labels >= OMEGA_THREAD_THRESHOLD
        return _accumulate_R_total_threaded_timetrot!(
            R, ws_eigenbases, ws_hermitian, oft_nufft_prefactors,
            energy_labels, transition, prefactor)
    end

    dim = size(R, 1)
    jump_oft = zeros(T, dim, dim)
    LdagL = zeros(T, dim, dim)

    for (k, eigenbasis) in enumerate(ws_eigenbases)
        is_herm = ws_hermitian[k]
        if is_herm
            for w_raw in energy_labels
                w_raw > 1e-12 && continue
                w = abs(w_raw)
                nufft_pf = _prefactor_view(oft_nufft_prefactors, w)
                @. jump_oft = eigenbasis * nufft_pf
                rate2 = prefactor * transition(w)
                mul!(LdagL, jump_oft', jump_oft)
                @. R += rate2 * LdagL
                if w > 1e-12
                    rate2_neg = prefactor * transition(-w)
                    mul!(LdagL, jump_oft, jump_oft')
                    @. R += rate2_neg * LdagL
                end
            end
        else
            for w in energy_labels
                nufft_pf = _prefactor_view(oft_nufft_prefactors, w)
                @. jump_oft = eigenbasis * nufft_pf
                rate2 = prefactor * transition(w)
                mul!(LdagL, jump_oft', jump_oft)
                @. R += rate2 * LdagL
            end
        end
    end
    return nothing
end

function _accumulate_R_total!(
    R::Matrix{T},
    ws_eigenbases::Vector{Matrix{T}},
    ws_hermitian::Vector{Bool},
    precomputed_data,
    config::Config{<:Any, BohrDomain},
    hamiltonian::HamHam,
) where {T<:Complex}
    (; alpha, gamma_norm_factor) = precomputed_data
    bohr_keys = collect(keys(hamiltonian.bohr_dict))
    n_jumps = length(ws_eigenbases)
    n_keys  = length(bohr_keys)
    if Threads.nthreads() > 1 && n_jumps * n_keys >= OMEGA_THREAD_THRESHOLD
        return _accumulate_R_total_threaded_bohr!(
            R, ws_eigenbases, alpha, gamma_norm_factor,
            hamiltonian.bohr_freqs, hamiltonian.bohr_dict, bohr_keys)
    end

    dim = size(R, 1)
    jump_oft = zeros(T, dim, dim)
    A_nu2_dag = zeros(T, dim, dim)

    for (k, eigenbasis) in enumerate(ws_eigenbases)
        for nu_2 in bohr_keys
            @. jump_oft = alpha(hamiltonian.bohr_freqs, nu_2) * eigenbasis

            fill!(A_nu2_dag, 0)
            indices = hamiltonian.bohr_dict[nu_2]
            @inbounds for idx in indices
                i = idx[1]; j = idx[2]
                A_nu2_dag[j, i] = conj(eigenbasis[i, j])
            end

            mul!(R, A_nu2_dag, jump_oft, gamma_norm_factor, 1.0)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Threaded rate-operator accumulation mirrors the matvec frequency loop.
# from `src/krylov_matvec.jl`, but for the construction-time additive
# accumulator. Allocates per-task buffers locally (one-time at Workspace
# construction; trivial overhead vs. the BLAS work it covers). Each task
# accumulates a private `R_partial`; final reduction sums into the caller's
# `R` after `@sync`.
# ---------------------------------------------------------------------------

function _accumulate_R_total_threaded_energy!(
    R::Matrix{T},
    ws_eigenbases::Vector{Matrix{T}},
    ws_hermitian::Vector{Bool},
    bohr_freqs::AbstractMatrix{<:Real},
    energy_labels::AbstractVector{Float64},
    transition,
    prefactor::Float64,
    inv_4sigma2::Float64,
) where {T<:Complex}
    work = Tuple{Int, Int}[]
    _populate_jump_frequency_work_list!(work, ws_hermitian, energy_labels)
    n_work = length(work)
    n_work == 0 && return nothing

    dim = size(R, 1)
    nt = min(Threads.nthreads(), n_work)
    chunks = _partition_range(1:n_work, nt)
    n_chunks = length(chunks)

    R_partials = [zeros(T, dim, dim) for _ in 1:n_chunks]
    jump_ofts  = [Matrix{T}(undef, dim, dim) for _ in 1:n_chunks]
    LdagLs     = [Matrix{T}(undef, dim, dim) for _ in 1:n_chunks]

    @sync for (idx, chunk) in enumerate(chunks)
        Threads.@spawn _accumulate_R_total_chunk_energy!(
            R_partials[idx], jump_ofts[idx], LdagLs[idx],
            ws_eigenbases, ws_hermitian, bohr_freqs, energy_labels,
            work, chunk, transition, prefactor, inv_4sigma2)
    end

    @inbounds for idx in 1:n_chunks
        R .+= R_partials[idx]
    end
    return nothing
end

function _accumulate_R_total_chunk_energy!(
    R_partial::Matrix{T},
    jump_oft::Matrix{T},
    LdagL::Matrix{T},
    ws_eigenbases::Vector{Matrix{T}},
    ws_hermitian::Vector{Bool},
    bohr_freqs::AbstractMatrix{<:Real},
    energy_labels::AbstractVector{Float64},
    work::Vector{Tuple{Int, Int}},
    chunk::UnitRange{Int},
    transition,
    prefactor::Float64,
    inv_4sigma2::Float64,
) where {T<:Complex}
    @inbounds for w_idx in chunk
        (k, li) = work[w_idx]
        eigenbasis = ws_eigenbases[k]
        is_herm    = ws_hermitian[k]

        w_raw = energy_labels[li]
        w = is_herm ? abs(w_raw) : w_raw

        oft!(jump_oft, eigenbasis, bohr_freqs, w, inv_4sigma2)
        rate2 = prefactor * transition(w)
        mul!(LdagL, jump_oft', jump_oft)
        @. R_partial += rate2 * LdagL

        if is_herm && w > 1e-12
            rate2_neg = prefactor * transition(-w)
            mul!(LdagL, jump_oft, jump_oft')
            @. R_partial += rate2_neg * LdagL
        end
    end
    return nothing
end

function _accumulate_R_total_threaded_timetrot!(
    R::Matrix{T},
    ws_eigenbases::Vector{Matrix{T}},
    ws_hermitian::Vector{Bool},
    oft_nufft_prefactors,
    energy_labels::AbstractVector{Float64},
    transition,
    prefactor::Float64,
) where {T<:Complex}
    work = Tuple{Int, Int}[]
    _populate_jump_frequency_work_list!(work, ws_hermitian, energy_labels)
    n_work = length(work)
    n_work == 0 && return nothing

    dim = size(R, 1)
    nt = min(Threads.nthreads(), n_work)
    chunks = _partition_range(1:n_work, nt)
    n_chunks = length(chunks)

    R_partials = [zeros(T, dim, dim) for _ in 1:n_chunks]
    jump_ofts  = [Matrix{T}(undef, dim, dim) for _ in 1:n_chunks]
    LdagLs     = [Matrix{T}(undef, dim, dim) for _ in 1:n_chunks]

    @sync for (idx, chunk) in enumerate(chunks)
        Threads.@spawn _accumulate_R_total_chunk_timetrot!(
            R_partials[idx], jump_ofts[idx], LdagLs[idx],
            ws_eigenbases, ws_hermitian, oft_nufft_prefactors,
            energy_labels, work, chunk, transition, prefactor)
    end

    @inbounds for idx in 1:n_chunks
        R .+= R_partials[idx]
    end
    return nothing
end

function _accumulate_R_total_chunk_timetrot!(
    R_partial::Matrix{T},
    jump_oft::Matrix{T},
    LdagL::Matrix{T},
    ws_eigenbases::Vector{Matrix{T}},
    ws_hermitian::Vector{Bool},
    oft_nufft_prefactors,
    energy_labels::AbstractVector{Float64},
    work::Vector{Tuple{Int, Int}},
    chunk::UnitRange{Int},
    transition,
    prefactor::Float64,
) where {T<:Complex}
    @inbounds for w_idx in chunk
        (k, li) = work[w_idx]
        eigenbasis = ws_eigenbases[k]
        is_herm    = ws_hermitian[k]

        w_raw = energy_labels[li]
        # Hermitian fold: only `w_raw <= 1e-12` queued; index NUFFT slice via
        # the |w_raw| key. Non-Hermitian: index by label position `li`.
        w = is_herm ? abs(w_raw) : w_raw
        nufft_pf = is_herm ?
            _prefactor_view(oft_nufft_prefactors, w) :
            (@view oft_nufft_prefactors.data[:, :, li])
        @. jump_oft = eigenbasis * nufft_pf

        rate2 = prefactor * transition(w)
        mul!(LdagL, jump_oft', jump_oft)
        @. R_partial += rate2 * LdagL

        if is_herm && w > 1e-12
            rate2_neg = prefactor * transition(-w)
            mul!(LdagL, jump_oft, jump_oft')
            @. R_partial += rate2_neg * LdagL
        end
    end
    return nothing
end

function _accumulate_R_total_threaded_bohr!(
    R::Matrix{T},
    ws_eigenbases::Vector{Matrix{T}},
    alpha,
    gamma_norm_factor::Real,
    bohr_freqs::AbstractMatrix{<:Real},
    bohr_dict,
    bohr_keys::Vector,
) where {T<:Complex}
    n_jumps = length(ws_eigenbases)
    n_keys  = length(bohr_keys)
    n_work  = n_jumps * n_keys
    n_work == 0 && return nothing

    dim = size(R, 1)
    nt = min(Threads.nthreads(), n_work)
    chunks = _partition_range(1:n_work, nt)
    n_chunks = length(chunks)

    R_partials  = [zeros(T, dim, dim) for _ in 1:n_chunks]
    jump_ofts   = [Matrix{T}(undef, dim, dim) for _ in 1:n_chunks]
    A_nu2_dags  = [zeros(T, dim, dim) for _ in 1:n_chunks]

    @sync for (idx, chunk) in enumerate(chunks)
        Threads.@spawn _accumulate_R_total_chunk_bohr!(
            R_partials[idx], jump_ofts[idx], A_nu2_dags[idx],
            ws_eigenbases, alpha, gamma_norm_factor,
            bohr_freqs, bohr_dict, bohr_keys, n_keys, chunk)
    end

    @inbounds for idx in 1:n_chunks
        R .+= R_partials[idx]
    end
    return nothing
end

function _accumulate_R_total_chunk_bohr!(
    R_partial::Matrix{T},
    jump_oft::Matrix{T},
    A_nu2_dag::Matrix{T},
    ws_eigenbases::Vector{Matrix{T}},
    alpha,
    gamma_norm_factor::Real,
    bohr_freqs::AbstractMatrix{<:Real},
    bohr_dict,
    bohr_keys::Vector,
    n_keys::Int,
    chunk::UnitRange{Int},
) where {T<:Complex}
    @inbounds for w_idx in chunk
        # Linear (k, key_idx) decoding: outer over jumps, inner over keys.
        k       = ((w_idx - 1) ÷ n_keys) + 1
        key_idx = ((w_idx - 1) % n_keys) + 1
        eigenbasis = ws_eigenbases[k]
        nu_2 = bohr_keys[key_idx]

        @. jump_oft = alpha(bohr_freqs, nu_2) * eigenbasis

        fill!(A_nu2_dag, 0)
        indices = bohr_dict[nu_2]
        @inbounds for idx in indices
            i = idx[1]; j = idx[2]
            A_nu2_dag[j, i] = conj(eigenbasis[i, j])
        end

        mul!(R_partial, A_nu2_dag, jump_oft, gamma_norm_factor, 1.0)
    end
    return nothing
end

# Compile once per source, then share retained-channel gain/loss machinery.
_dll_workspace_lindblads(jump, ham, data, ::BohrDomain) =
    dll_lindblad_op_bohr(jump, ham, data.filter)

function _dll_workspace_lindblads(jump, ham, data, ::TimeDomain)
    return [jump.in_eigenbasis .* prefactor .* data.t0
            for prefactor in data.oft_nufft_at_zero_list]
end

"""
    _accumulate_R_total_dll!(R, dll_lindblads_out, jumps, precomputed_data, config, hamiltonian) -> nothing

Accumulate the DLL rate operator and retain each Bohr- or Time-domain Lindblad matrix.
"""
function _accumulate_R_total_dll!(
    R::Matrix{T},
    dll_lindblads_out::Vector{Matrix{T}},
    jumps::Vector{JumpOp},
    precomputed_data,
    config::Config{<:Any, <:Union{BohrDomain, TimeDomain}, DLL},
    hamiltonian::HamHam,
) where {T<:Complex}
    n_jumps = length(jumps)

    # Each task builds its DLL Lindblad operators into private accumulators.
    # operator(s) and accumulates a private R_partial. Reduce after `@sync`.
    # `dll_lindblads_out` retains deterministic per-jump order.
    if Threads.nthreads() > 1 && n_jumps >= 2
        nt = min(Threads.nthreads(), n_jumps)
        chunks = _partition_range(1:n_jumps, nt)
        n_chunks = length(chunks)
        dim = size(R, 1)

        # Each task accumulates into its own R_partial and a per-jump-vector
        # of operator lists, sized to the original jump count for ordered merge.
        R_partials = [zeros(T, dim, dim) for _ in 1:n_chunks]
        per_jump_ops = Vector{Vector{Matrix{T}}}(undef, n_jumps)

        @sync for (idx, chunk) in enumerate(chunks)
            Threads.@spawn _accumulate_R_total_dll_chunk!(
                R_partials[idx], per_jump_ops, jumps, hamiltonian,
                precomputed_data, config.domain, chunk)
        end

        @inbounds for idx in 1:n_chunks
            R .+= R_partials[idx]
        end
        @inbounds for k in 1:n_jumps
            for L_a in per_jump_ops[k]
                push!(dll_lindblads_out, L_a)
            end
        end
        return nothing
    end

    for jump in jumps
        L_or_Ls = _dll_workspace_lindblads(jump, hamiltonian, precomputed_data, config.domain)
        # Single-channel filters return a Matrix; multi-channel filters
        # Multi-channel filters return a vector; flatten it in jump-major order.
        # output `dll_lindblads_out` — the matrix-free hot path
        # `apply_lindbladian!` iterates `for L_a in dll_lindblads` and the
        # dissipator `Σ_a L_a ρ L_a†` is a flat sum over all per-channel
        # operators (no cross terms in the multi-channel α).
        if L_or_Ls isa AbstractMatrix
            L_a = Matrix{T}(L_or_Ls)
            push!(dll_lindblads_out, L_a)
            mul!(R, L_a', L_a, 1.0, 1.0)
        else
            for L_one in L_or_Ls
                L_a = Matrix{T}(L_one)
                push!(dll_lindblads_out, L_a)
                mul!(R, L_a', L_a, 1.0, 1.0)
            end
        end
    end
    return nothing
end

function _accumulate_R_total_dll_chunk!(
    R_partial::Matrix{T},
    per_jump_ops::Vector{Vector{Matrix{T}}},
    jumps::Vector{JumpOp},
    hamiltonian::HamHam,
    precomputed_data,
    domain::D,
    chunk::UnitRange{Int},
) where {T<:Complex, D}
    @inbounds for k in chunk
        L_or_Ls = _dll_workspace_lindblads(jumps[k], hamiltonian, precomputed_data, domain)
        ops = Vector{Matrix{T}}()
        if L_or_Ls isa AbstractMatrix
            L_a = Matrix{T}(L_or_Ls)
            push!(ops, L_a)
            mul!(R_partial, L_a', L_a, 1.0, 1.0)
        else
            for L_one in L_or_Ls
                L_a = Matrix{T}(L_one)
                push!(ops, L_a)
                mul!(R_partial, L_a', L_a, 1.0, 1.0)
            end
        end
        per_jump_ops[k] = ops
    end
    return nothing
end

"""
    Workspace(config::Config{Lindbladian, D, DLL}, hamiltonian, jumps; trotter=nothing)

Construct the DLL Bohr- or Time-domain matrix-free workspace.

It stores per-channel Lindblad matrices, the total rate operator, and the
coherent correction. The left/right factors follow the package's transposed
column-stacking convention.
"""
function Workspace(
    config::Config{Lindbladian, D, DLL},
    hamiltonian::HamHam,
    jumps::Vector{JumpOp};
    trotter::Union{AbstractTrotter, Nothing}=nothing,
) where {D<:Union{BohrDomain, TimeDomain}}
    validate_config!(config, hamiltonian)
    trotter === nothing || throw(ArgumentError("DLL Bohr/Time workspaces do not use a Trotter cache."))

    precomputed_data = _precompute_data(config, hamiltonian)
    (; filter) = precomputed_data

    dim = size(hamiltonian.data, 1)
    T = eltype(hamiltonian.eigvals)
    CT = Complex{T}

    # Per-jump DLL Lindblad operators + accumulated R_total. For
    # Multi-channel filters store `k * length(jumps)` matrices.
    # operators (one per channel per coupling).
    n_channels = filter isa DLLMultiChannelFilter ? length(filter.channels) : 1
    dll_lindblads = Vector{Matrix{CT}}()
    sizehint!(dll_lindblads, length(jumps) * n_channels)
    R_total = zeros(CT, dim, dim)
    _accumulate_R_total_dll!(R_total, dll_lindblads, jumps,
                              precomputed_data, config, hamiltonian)
    hermitianize!(R_total)

    # Keep the requested domain's coherent correction, including Time quadrature.
    G = Matrix{CT}(_precompute_coherent_B(jumps, hamiltonian, config, precomputed_data))

    # Schrödinger coherent action: -i[G,rho].
    G_left  = Matrix{CT}(-1im .* G .- 0.5 .* R_total)
    G_right = Matrix{CT}(1im .* G .- 0.5 .* R_total)
    jump_eigenbases = [Matrix{CT}(j.in_eigenbasis) for j in jumps]
    jump_hermitian  = [j.hermitian for j in jumps]

    sc = KrylovScratch(CT, dim; with_channel_rho_jump=false)

    return Workspace{KrylovSpectrum, D, DLL, T}(
        jump_eigenbases, jump_hermitian, copy(jumps),
        dll_lindblads,
        G_left, G_right,
        nothing, nothing, nothing, nothing, nothing,  # transition/gnf/energy_labels/odp/nufft
        nothing, nothing, nothing, nothing, nothing, nothing,  # bohr_alpha/bohr_keys/bohr_is/bohr_js/b_minus/b_plus
        hamiltonian, nothing, nothing, nothing,  # source provenance; no channel matrices
        sc,
        config,   # Cached to validate predictor workspace reuse.
    )
end

"""
    Workspace(config::Config{Thermalize}, hamiltonian, jumps; trotter=nothing)

Construct a matrix-free workspace for the faithful deterministic channel.

The workspace stores per-jump no-event, residual, and coherent matrices in
sweep order. `jump_selection=:random` is rejected because it does not define a
single deterministic channel map.

A `Workspace{KrylovSpectrum,D,C,T}` for channel application and Krylov solves.
"""
function Workspace(
    config::Config{Thermalize},
    hamiltonian::HamHam,
    jumps::Vector{JumpOp};
    trotter::Union{AbstractTrotter, Nothing}=nothing,
)
    validate_config!(config, hamiltonian)
    isempty(jumps) && throw(ArgumentError(
        "Workspace(::Config{Thermalize}, ...) requires at least one jump operator."))

    # Determine ham_or_trott
    ham_or_trott = if config.domain isa TrotterDomain
        trotter === nothing && error("A Trotter object must be provided for the TrotterDomain")
        _validate_trotter_cache!(config, hamiltonian, trotter)
        trotter
    else
        hamiltonian
    end

    # A Krylov matvec requires a deterministic channel.
    config.jump_selection === :sweep || throw(ArgumentError(
        "Workspace(::Config{Thermalize}, ...) for the Krylov channel matvec " *
        "requires config.jump_selection = :sweep (got :$(config.jump_selection)). " *
        ":random is an expectation map of the per-step jump pick, not a single deterministic channel."))

    # Precompute domain-specific data
    precomputed_data = _precompute_data(config, ham_or_trott)

    # Determine dimensions and element type
    dim = size(hamiltonian.data, 1)
    T = eltype(hamiltonian.eigvals)
    CT = Complex{T}

    # Extract concrete-typed eigenbasis matrices and hermitian flags
    jump_eigenbases = [Matrix{CT}(j.in_eigenbasis) for j in jumps]
    jump_hermitian  = [j.hermitian for j in jumps]

    # Precompute per-jump coherent unitaries, using GQSP when configured.
    # :sweep ⇒ delta_scale = 1.0 (matches run_thermalize at furnace.jl:206-207).
    coh_raw = _precompute_coherent_unitary(
        jumps, hamiltonian, config, precomputed_data;
        trotter=trotter, delta_scale=1.0)
    # Broaden element type to `Vector{Union{Nothing, Matrix{CT}}}` so the field
    # type matches the struct declaration.
    U_coherents = if coh_raw === nothing
        nothing
    else
        Vector{Union{Nothing, Matrix{CT}}}(coh_raw)
    end

    # Reuse the full-DM simulator's per-jump channel construction.
    (; K0s, U_residuals) = _precompute_per_jump_channels(
        jumps, ham_or_trott, config, precomputed_data;
        rescale_by_inv_prob=false,
    )

    # Thread-local accumulators avoid per-jump allocation in channel matvecs.
    sc = ThermalizeScratch(CT, dim;
        with_task_pool = Threads.nthreads() > 1,
        num_threads = Threads.nthreads(),
    )

    # Absorb precomputed_data fields (mirrors the Lindbladian constructor).
    pd_transition = hasproperty(precomputed_data, :transition) ? precomputed_data.transition : nothing
    pd_gnf = hasproperty(precomputed_data, :gamma_norm_factor) ? precomputed_data.gamma_norm_factor : nothing
    pd_el = hasproperty(precomputed_data, :energy_labels) ? precomputed_data.energy_labels : nothing
    pd_odp = hasproperty(precomputed_data, :oft_domain_prefactor) ? precomputed_data.oft_domain_prefactor : nothing
    pd_nufft = hasproperty(precomputed_data, :oft_nufft_prefactors) ? precomputed_data.oft_nufft_prefactors : nothing
    pd_alpha = hasproperty(precomputed_data, :alpha) ? precomputed_data.alpha : nothing
    pd_bkeys = hasproperty(precomputed_data, :bohr_keys) ? precomputed_data.bohr_keys : nothing
    pd_bis = hasproperty(precomputed_data, :bohr_is) ? precomputed_data.bohr_is : nothing
    pd_bjs = hasproperty(precomputed_data, :bohr_js) ? precomputed_data.bohr_js : nothing
    pd_bminus = hasproperty(precomputed_data, :b_minus) ? precomputed_data.b_minus : nothing
    pd_bplus = hasproperty(precomputed_data, :b_plus) ? precomputed_data.b_plus : nothing

    D = typeof(config.domain)
    C = typeof(config.construction)
    return Workspace{KrylovSpectrum, D, C, T}(
        jump_eigenbases, jump_hermitian, copy(jumps),
        nothing,  # dll_lindblads (Thermalize path)
        nothing, nothing,  # G_left/G_right: only used by the Lindbladian matvec
        pd_transition, pd_gnf, pd_el, pd_odp, pd_nufft,
        pd_alpha, pd_bkeys, pd_bis, pd_bjs, pd_bminus, pd_bplus,
        ham_or_trott, K0s, U_residuals, U_coherents,
        sc,
        config,   # Cached to validate predictor workspace reuse.
    )
end
