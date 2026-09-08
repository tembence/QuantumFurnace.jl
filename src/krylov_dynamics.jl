# State-specific Krylov spectral dynamics.
# Math: $rho(t) = rho_infinity + sum_(i != 0) c_i exp(lambda_i t) R_i$.
# The initial-state seed captures exactly the symmetry sectors populated by
# `rho_0`; an optional independent operator-gap solve covers other sectors.

"""
    _arnoldi_factorize(f, x0, m) -> (Q, H, broke)

Build an Arnoldi basis with modified Gram--Schmidt and one reorthogonalisation.

# Returns
`(Q, H, broke)`, truncated if the basis breaks down before `m` steps.
"""
function _arnoldi_factorize(f, x0::AbstractVector{T}, m::Int) where {T}
    N = length(x0)
    Q = zeros(T, N, m + 1)
    H = zeros(T, m + 1, m)
    Q[:, 1] .= x0 ./ norm(x0)
    broke_at = m
    @inbounds for j in 1:m
        w = f(view(Q, :, j))
        # Modified Gram–Schmidt
        for i in 1:j
            H[i, j] = dot(view(Q, :, i), w)
            w .-= H[i, j] .* view(Q, :, i)
        end
        # Reorthogonalisation pass for numerical stability
        for i in 1:j
            corr = dot(view(Q, :, i), w)
            H[i, j] += corr
            w .-= corr .* view(Q, :, i)
        end
        h_jp1 = norm(w)
        H[j + 1, j] = h_jp1
        if h_jp1 < eps(real(T)) * sqrt(N)
            broke_at = j
            break
        end
        Q[:, j + 1] .= w ./ h_jp1
    end
    return Q[:, 1:broke_at], H[1:broke_at, 1:broke_at], broke_at < m
end


"""
    _krylov_spectral_decomposition(forward_apply!, rho_0, dim; krylovdim,
                                   tol, fwd_init, sort_mode,
                                   assume_trace_preserving)

Build a biorthogonal eigendecomposition in the Krylov subspace seeded by
`vec(rho_0)` and project `rho_0 - rho_inf` onto its modes.

The small Hessenberg eigendecomposition is lifted through the same Arnoldi
basis for both left and right modes, preserving biorthogonality. If the
operator and `rho_0` share a symmetry, unpopulated sectors are absent: the
trajectory remains correct, but its state-coupled gap need not be the full
operator gap.

# Returns
A named tuple containing sorted eigenvalues, left and right modes, projection
coefficients, normalized stationary state, matvec count, and breakdown status.
When `assume_trace_preserving=false`, no stationary state is inserted: all
captured modes, including the dominant one, retain their raw powers.
"""
function _krylov_spectral_decomposition(
    forward_apply!::F1,
    rho_0::Matrix{T},
    dim::Integer;
    krylovdim::Integer = 40,
    tol::Real = 1e-10,
    fwd_init::Union{Nothing, AbstractVector} = nothing,
    sort_mode::Symbol = :lindbladian,
    assume_trace_preserving::Bool = true,
) where {T<:Complex, F1}

    dim2 = dim * dim

    # Closure on flat vectors with private scratch.
    rho_buf = Matrix{T}(undef, dim, dim)
    out_buf = Matrix{T}(undef, dim, dim)
    function fwd_vec(v::AbstractVector)
        copyto!(rho_buf, reshape(v, dim, dim))
        forward_apply!(out_buf, rho_buf)
        return copy(vec(out_buf))
    end

    # Seeding with `rho_0` exactly captures its orbit but not unpopulated symmetry sectors.
    x0 = if fwd_init === nothing
        Vector{T}(vec(rho_0))
    else
        Vector{T}(fwd_init)
    end
    m_target = min(Int(krylovdim), dim2)

    Q, H, broke = _arnoldi_factorize(fwd_vec, x0, m_target)
    m = size(H, 1)
    m >= 1 || error("Arnoldi returned no basis vectors")
    matvec_count = m

    # Diagonalise H densely. eigen returns columns of W as right eigvecs;
    # we form V = (W^{-1})' so V' W = I (the biorthogonality condition).
    F = eigen(H)
    Λ = F.values
    W = F.vectors
    W_inv = inv(W)
    V = Matrix{T}(W_inv')  # so V' = W_inv ⇒ V' W = I

    # Sort generator modes from the stationary rate. For a physical channel,
    # select the captured stationary mode independently before ordering the
    # remaining modes by their actual discrete decay factor |mu|. A raw GQSP
    # surrogate retains pure decreasing-modulus order.
    perm = if sort_mode === :lindbladian
        sortperm(Λ; by = v -> abs(real(v)))
    elseif sort_mode === :channel
        _channel_mode_permutation(
            Λ; assume_trace_preserving=assume_trace_preserving)
    else
        throw(ArgumentError("sort_mode must be :lindbladian or :channel (got :$sort_mode)"))
    end
    Λ = Λ[perm]
    W = W[:, perm]
    V = V[:, perm]

    # Lift small-space eigvecs to operator space.
    QW = Q * W            # right eigvecs as flat columns
    QV = Q * V            # left eigvecs as flat columns
    R_modes = [reshape(QW[:, i], dim, dim) for i in 1:m]
    L_modes = [reshape(QV[:, i], dim, dim) for i in 1:m]

    rho_inf, c = if assume_trace_preserving
        # Fix the arbitrary modal phase before Hermitian projection and trace
        # normalisation; otherwise a mode near `im * rho` can collapse.
        rho_stationary = _normalize_stationary_mode!(R_modes[1])
        R_modes[1] = rho_stationary
        steady_overlap = dot(L_modes[1], R_modes[1])
        overlap_scale = norm(L_modes[1]) * norm(R_modes[1])
        overlap_tol = sqrt(eps(real(float(one(T))))) * overlap_scale
        isfinite(abs(steady_overlap)) && isfinite(overlap_scale) &&
            abs(steady_overlap) > overlap_tol ||
            error("normalised steady-state mode is orthogonal to its left eigenvector")
        L_modes[1] ./= conj(steady_overlap)

        # Biorthogonal projection of rho_0-rho_inf. Trace preservation makes
        # the stationary coefficient zero; remove its residual explicitly.
        coeffs = (V') * (Q' * vec(rho_0 .- rho_stationary))
        coeffs[1] = zero(T)
        rho_stationary, coeffs
    else
        # The unscaled GQSP polynomial surrogate is generally not trace
        # preserving and has no stationary eigenvalue at one. Reconstruct the
        # raw linear map from every captured mode without normalising it.
        zeros(T, dim, dim), (V') * (Q' * vec(rho_0))
    end

    return (
        eigenvalues  = Complex{Float64}.(Λ),
        R_modes      = R_modes,
        L_modes      = L_modes,
        c            = Complex{Float64}.(c),
        rho_inf      = rho_inf,
        matvec_count = matvec_count,
        converged    = !broke,
        trace_preserving_assumed = assume_trace_preserving,
        invariant_subspace = broke || m == dim2,
    )
end


"""
    predict_lindbladian_trajectory(config::Config{Lindbladian, <:Union{BohrDomain, EnergyDomain}},
                                   hamiltonian, jumps, rho_0, t_grid;
                                   krylovdim=40, tol=1e-10, save_states=false)

Reconstruct `rho(t) = exp(tL) rho_0` from one state-seeded Arnoldi factorization.

The returned `spectral_gap` is the slowest captured mode coupled to `rho_0`.
Use `compute_true_gap=true` for an independent operator-spectrum solve when
symmetry may hide unpopulated sectors.

# Arguments
- `config`, `hamiltonian`, `jumps`: Lindbladian construction data.
- `rho_0`: Initial density matrix.
- `t_grid`: Times at which to reconstruct the state.

# Keywords
- `krylovdim`: Arnoldi subspace size.
- `compute_true_gap`: Run the independent operator-gap solve.
- `krylovdim_gap_pass`: Subspace size for that optional solve.
- `tol`: Forwarded to the optional gap solve; otherwise reserved.
- `save_states`: Retain every reconstructed state.
- `workspace`: Matching precomputed operator workspace.
- `raw_reconstruction`: Retain every captured mode without repair (facade use);
  a stationary projection is required before passing this raw payload to mixing-time helpers.

# Returns
A named tuple with the time grid, trace distances, optional states, captured
spectrum and modes, stationary state, Gibbs reference, gap, and convergence
metadata. The spectral fields can be passed to [`eigenmode_mixing_time`](@ref).
"""
function predict_lindbladian_trajectory(
    config::Config{Lindbladian, <:Union{BohrDomain, EnergyDomain}},
    hamiltonian::HamHam,
    jumps::Vector{JumpOp},
    rho_0::Matrix{T},
    t_grid::AbstractVector{<:Real};
    krylovdim::Integer = 40,
    krylovdim_gap_pass::Union{Nothing, Integer} = nothing,
    tol::Real = 1e-10,
    save_states::Bool = false,
    allow_unpaired_nonhermitian::Bool = false,
    workspace::Union{Nothing, Workspace{KrylovSpectrum}} = nothing,
    compute_true_gap::Bool = false,
    raw_reconstruction::Bool = false,
    matvec_callback=nothing, work_callback=nothing,
    dense_max_dim::Integer=64, max_dense_bytes::Integer=64*1024^2,
)::NamedTuple where {T<:Complex}
    isempty(t_grid) && throw(ArgumentError("t_grid must be nonempty."))
    work() = work_callback === nothing ? nothing : work_callback()
    work()
    d = size(rho_0, 1)
    @assert size(rho_0, 2) == d  "rho_0 must be square"

    validate_config!(config, hamiltonian)
    validate_jump_pairing(jumps; allow_unpaired_nonhermitian=allow_unpaired_nonhermitian)

    ws = _reuse_or_build_krylov_workspace(
        workspace, config, hamiltonian, jumps, nothing, d;
        caller="predict_lindbladian_trajectory")
    fwd! = let ws = ws, config = config, ham = hamiltonian
        (out::AbstractMatrix, x::AbstractMatrix) -> begin
            matvec_callback === nothing || matvec_callback()
            apply_lindbladian!(ws, x, config, ham)
            copyto!(out, ws.scratch.rho_out)
            return out
        end
    end

    decomp = _krylov_spectral_decomposition(
        fwd!, rho_0, d;
        krylovdim=krylovdim, tol=tol, assume_trace_preserving=!raw_reconstruction,
    )

    sigma_beta = Matrix{T}(hamiltonian.gibbs)
    n_t = length(t_grid)
    distances = Vector{Float64}(undef, n_t)
    states = save_states ? Vector{Matrix{T}}(undef, n_t) : Matrix{T}[]
    rho_t = Matrix{T}(undef, d, d)
    h = length(decomp.eigenvalues)

    raw_checks = NamedTuple[]
    @inbounds for k in 1:n_t
        work()
        t = float(t_grid[k])
        copyto!(rho_t, decomp.rho_inf)
        for i in 1:h
            # Skip the steady-state mode (its c is ~0 by trace preservation).
            !raw_reconstruction && abs(decomp.eigenvalues[i]) < 1e-10 && continue
            phase = exp(decomp.eigenvalues[i] * t)
            rho_t .+= (decomp.c[i] * phase) .* decomp.R_modes[i]
        end
        # Defensive Hermitisation (mirrors lindblad_action_integrate).
        if raw_reconstruction
            push!(raw_checks,state_diagnostics(rho_t;rtol=max(1e-9,100eps(real(T))),dense_max_dim,max_dense_bytes))
        else
            hermitianize!(rho_t)
        end
        distances[k] = sum(svdvals(rho_t .- sigma_beta)) / 2
        save_states && (states[k] = copy(rho_t))
    end

    # The state-coupled gap is free; the optional operator gap needs a second Arnoldi solve.
    if compute_true_gap
        kdim_gp = krylovdim_gap_pass === nothing ? max(30, Int(krylovdim)÷2) : Int(krylovdim_gap_pass)
        # Reuse construction data, but keep the operator Arnoldi factorization independent.
        gap_res = krylov_spectral_gap(
            config, hamiltonian, jumps;
            krylovdim=kdim_gp, howmany=4, tol=tol,
            allow_unpaired_nonhermitian=allow_unpaired_nonhermitian,
            workspace=ws,
        )
        spectral_gap  = gap_res.spectral_gap
        total_matvecs = decomp.matvec_count + gap_res.matvec_count
        all_converged = decomp.converged && gap_res.converged >= 1
    else
        # Pass-1 gap: smallest |Re(λ_i)| over non-steady modes. The decomp
        # sorts steady (|Re| ~ 0) to index 1, so eigenvalues[2] is the
        # slowest non-steady mode by construction.
        spectral_gap  = length(decomp.eigenvalues) >= 2 ?
                        abs(real(decomp.eigenvalues[2])) : 0.0
        total_matvecs = decomp.matvec_count
        all_converged = raw_reconstruction ? decomp.invariant_subspace : decomp.converged
    end

    return (
        t              = collect(t_grid),
        distances      = distances,
        rho_final      = copy(rho_t),
        total_matvecs  = total_matvecs,
        all_converged  = all_converged,
        states         = states,
        eigenvalues    = decomp.eigenvalues,
        c              = decomp.c,
        spectral_gap   = spectral_gap,
        rho_inf        = decomp.rho_inf,
        R_modes        = decomp.R_modes,
        spectral_modes = spectral_mode_diagnostics(decomp.eigenvalues, decomp.R_modes, decomp.c),
        sigma_beta     = sigma_beta,
        raw_checks,
        repair_norms = raw_reconstruction ? zeros(n_t) : fill(NaN,n_t),
        repair_policy = raw_reconstruction ? :none : :legacy_hermitian,
        failure = nothing,
        raw_reconstruction,
        invariant_subspace = decomp.invariant_subspace,
    )
end


# Faithful-channel spectral dynamics.
# Math: $rho_k = rho_infinity + sum_i c_i mu_i^k R_i$.

"""
    predict_channel_trajectory(config::Config{Thermalize, <:Union{BohrDomain,
                                EnergyDomain, TimeDomain}}, hamiltonian, jumps,
                                rho_0, k_grid; krylovdim=40, tol=1e-10,
                                save_states=false)

Reconstruct repeated applications of the implemented faithful channel.

The forward map matches `run_thermalize` in deterministic `:sweep` mode and
therefore includes finite-step, coherent-splitting, Fourier, and Trotter errors.
With `with_gqsp=true`, it applies the raw unscaled Jacobi–Anger polynomial
surrogate, not a certified postselected block or deterministic unitary
completion. The raw state is not renormalised; `trace_values` and
`trace_drift` report the resulting norm loss or gain. Its `spectral_gap` is
`NaN` because a non-channel surrogate has no physical channel gap.

# Arguments
- `config`: Thermalization configuration with `jump_selection=:sweep`.
- `hamiltonian`, `jumps`: Channel construction data.
- `rho_0`: Initial density matrix.
- `k_grid`: Integer channel-step counts; physical time is `k * delta`.

# Keywords
- `krylovdim`: Arnoldi subspace size.
- `compute_true_gap`: Run a separate direct-channel operator solve. This is
  off by default because non-normal channel Arnoldi can select a faster mode.
- `krylovdim_gap_pass`: Subspace size for the optional solve.
- `tol`: Forwarded to the optional solve; otherwise reserved.
- `save_states`: Retain every reconstructed state.
- `trotter`: Required for `TrotterDomain`.
- `workspace`: Matching precomputed channel workspace.

# Returns
A named tuple with physical times, distances, optional states, channel
eigenvalues and modes, basis-aligned Gibbs reference, generator-rate gap,
step size, and convergence metadata.
"""
function predict_channel_trajectory(
    config::Config{Thermalize, <:Union{BohrDomain, EnergyDomain, TimeDomain, TrotterDomain}},
    hamiltonian::HamHam,
    jumps::Vector{JumpOp},
    rho_0::Matrix{T},
    k_grid::AbstractVector{<:Integer};
    krylovdim::Integer = 40,
    krylovdim_gap_pass::Union{Nothing, Integer} = nothing,
    tol::Real = 1e-10,
    save_states::Bool = false,
    trotter::Union{Nothing, AbstractTrotter} = nothing,
    allow_unpaired_nonhermitian::Bool = false,
    workspace::Union{Nothing, Workspace{KrylovSpectrum}} = nothing,
    compute_true_gap::Bool = false,
)::NamedTuple where {T<:Complex}
    d = size(rho_0, 1)
    @assert size(rho_0, 2) == d  "rho_0 must be square"

    validate_config!(config, hamiltonian)
    validate_jump_pairing(jumps; allow_unpaired_nonhermitian=allow_unpaired_nonhermitian)
    config.jump_selection === :sweep || throw(ArgumentError(
        "predict_channel_trajectory requires config.jump_selection = :sweep " *
        "(got :$(config.jump_selection)). The :random selection runs a stochastic " *
        "process whose deterministic Φ_δ matvec is e^{δ𝓛} only in expectation."))
    config.with_gqsp && compute_true_gap && throw(ArgumentError(
        "compute_true_gap is unavailable for the unscaled GQSP polynomial " *
        "surrogate because it is not a trace-preserving channel."))

    # Select the evolution basis used by the channel workspace.
    ham_or_trott = config.domain isa TrotterDomain ? begin
        trotter === nothing && error("TrotterDomain requires a trotter object")
        trotter
    end : hamiltonian

    # Channel construction and application are shared with `run_thermalize`.
    ws = _reuse_or_build_krylov_workspace(
        workspace, config, hamiltonian, jumps, trotter, d;
        caller="predict_channel_trajectory")

    CT = Complex{eltype(hamiltonian.eigvals)}

    # Forward channel matvec: applies one full Φ_δ step on `x`, writes to `out`.
    fwd! = let ws = ws, config = config, ham = hamiltonian
        (out::AbstractMatrix, x::AbstractMatrix) -> begin
            # The spectral operator must remain complex-linear. Physical
            # reconstructions are Hermitianised only after applying mu^k.
            apply_delta_channel!(ws, x, config, ham; hermitize=false)
            copyto!(out, ws.scratch.rho_next)
            return out
        end
    end

    decomp = _krylov_spectral_decomposition(
        fwd!, Matrix{CT}(rho_0), d;
        krylovdim=krylovdim, tol=tol,
        sort_mode=:channel,
        assume_trace_preserving=!config.with_gqsp,
    )

    # Align the Gibbs reference with the basis in which the channel evolves.
    sigma_beta = if config.domain isa TrotterDomain
        @assert trotter !== nothing
        Matrix{CT}(Hermitian(trotter.eigvecs' * hamiltonian.eigvecs *
                              hamiltonian.gibbs *
                              hamiltonian.eigvecs' * trotter.eigvecs))
    else
        Matrix{CT}(hamiltonian.gibbs)
    end

    delta = float(config.delta)
    n_k = length(k_grid)
    t_grid = collect(k_grid) .* delta
    distances = Vector{Float64}(undef, n_k)
    states = save_states ? Vector{Matrix{CT}}(undef, n_k) : Matrix{CT}[]
    trace_values = Vector{CT}(undef, n_k)
    trace_drift = Vector{CT}(undef, n_k)
    rho_k = Matrix{CT}(undef, d, d)
    h = length(decomp.eigenvalues)

    @inbounds for j in 1:n_k
        k = Int(k_grid[j])
        copyto!(rho_k, decomp.rho_inf)
        for i in 1:h
            # Steady state (i=1) is suppressed via c[1] = 0 in the engine,
            # so we can include all modes uniformly.
            mu_pow = decomp.eigenvalues[i]^k
            rho_k .+= (decomp.c[i] * mu_pow) .* decomp.R_modes[i]
        end
        # Defensive Hermitisation.
        hermitianize!(rho_k)
        trace_values[j] = tr(rho_k)
        trace_drift[j] = trace_values[j] - one(CT)
        distances[j] = sum(svdvals(rho_k .- sigma_beta)) / 2
        save_states && (states[j] = copy(rho_k))
    end

    # Math: a physical channel has rate $-log(abs(mu_2)) / delta$. The
    # unscaled polynomial surrogate has no stationary mode at one and hence no
    # channel-gap interpretation.
    if config.with_gqsp
        spectral_gap = NaN
        total_matvecs = decomp.matvec_count
        all_converged = decomp.converged
    elseif compute_true_gap
        kdim_gp = krylovdim_gap_pass === nothing ? max(30, Int(krylovdim)÷2) : Int(krylovdim_gap_pass)
        # Reuse construction data, but keep the operator Arnoldi factorization independent.
        gap_res = krylov_spectral_gap(
            config, hamiltonian, jumps;
            trotter=trotter,
            krylovdim=kdim_gp, howmany=4, tol=tol,
            allow_unpaired_nonhermitian=allow_unpaired_nonhermitian,
            workspace=ws,
        )
        spectral_gap  = gap_res.spectral_gap
        total_matvecs = decomp.matvec_count + gap_res.matvec_count
        all_converged = decomp.converged && gap_res.converged >= 1
    else
        spectral_gap = _channel_spectral_gap(decomp.eigenvalues, delta)
        total_matvecs = decomp.matvec_count
        all_converged = decomp.converged
    end

    return (
        t              = t_grid,
        distances      = distances,
        rho_final      = copy(rho_k),
        total_matvecs  = total_matvecs,
        all_converged  = all_converged,
        states         = states,
        eigenvalues    = decomp.eigenvalues,
        c              = decomp.c,
        spectral_gap   = spectral_gap,
        rho_inf        = decomp.rho_inf,
        R_modes        = decomp.R_modes,
        spectral_modes = spectral_mode_diagnostics(decomp.eigenvalues, decomp.R_modes, decomp.c),
        sigma_beta     = sigma_beta,
        delta_used     = delta,
        k_grid         = collect(k_grid),
        trace_values   = trace_values,
        trace_drift    = trace_drift,
        max_abs_trace_drift = maximum(abs, trace_drift; init=0.0),
        trace_normalized = false,
        trace_preserving_assumed = decomp.trace_preserving_assumed,
        physical_channel = !config.with_gqsp,
        raw_dominant_modulus = maximum(abs, decomp.eigenvalues; init=0.0),
        channel_representation = config.with_gqsp ?
            :unscaled_gqsp_polynomial_surrogate : :deterministic_cptp,
    )
end
