# Matrix-free Krylov eigensolvers for Lindbladians and channels.

"""
    _check_krylov_memory(n_qubits, krylovdim)

Warn when the estimated Krylov storage exceeds 80% of free memory.
"""
function _check_krylov_memory(n_qubits::Int, krylovdim::Int)
    estimated_bytes = krylovdim * (4^n_qubits) * 16 * 1.5
    available = Sys.free_memory()
    if estimated_bytes > 0.8 * available
        est_gb = round(estimated_bytes / 1e9; digits=2)
        avail_gb = round(available / 1e9; digits=2)
        @warn "Krylov memory estimate $(est_gb) GB exceeds 80% of free memory $(avail_gb) GB. " *
              "Consider reducing krylovdim or num_qubits."
    end
    return nothing
end

"""
    _eigsolve_with_retry(f, x0, howmany, which; krylovdim=30, tol=1e-10, maxiter=100, max_retries=3, krylov_kwargs...)

Run `KrylovKit.eigsolve`, enlarging `krylovdim` by 50% after partial convergence.

# Arguments
- `f`, `x0`: Linear map and starting vector.
- `howmany`, `which`: Requested eigenpairs and targeting rule.
- `krylovdim`, `tol`, `maxiter`, `max_retries`: Arnoldi controls.

# Returns
`(vals, vecs, info)` from `KrylovKit.eigsolve`.
"""
function _eigsolve_with_retry(f, x0, howmany::Int, which::Symbol;
    krylovdim::Int=30, tol::Real=1e-10, maxiter::Int=100, max_retries::Int=3,
    accept_invariant_subspace::Bool=false, krylov_kwargs...)

    current_krylovdim = krylovdim
    local vals, vecs, info

    for attempt in 1:(max_retries + 1)
        vals, vecs, info = eigsolve(f, x0, howmany, which,
            Arnoldi(; krylovdim=current_krylovdim, tol=tol, maxiter=maxiter, verbosity=0);
            krylov_kwargs...)

        if info.converged >= howmany ||
            (accept_invariant_subspace && !isempty(vals) && info.converged == length(vals))
            return vals, vecs, info
        end

        if attempt <= max_retries
            new_krylovdim = ceil(Int, current_krylovdim * 1.5)
            @warn "KrylovKit: $(info.converged)/$(howmany) converged. " *
                  "Retrying with krylovdim=$new_krylovdim (attempt $(attempt+1)/$(max_retries+1))"
            current_krylovdim = new_krylovdim
        end
    end

    error("KrylovKit failed to converge: $(info.converged)/$(howmany) eigenvalues " *
          "after $(max_retries + 1) attempts (final krylovdim=$(current_krylovdim))")
end

# A pure maximally mixed seed stays in the trivial symmetry sector. Add a
# deterministic traceless Hermitian perturbation to overlap other sectors.
"""
    _krylov_default_x0(dim) -> Vector{ComplexF64}

Build `vec(I/d + epsilon H)` using a fixed traceless Hermitian random matrix.

The fixed `epsilon = 1e-10` is large enough to survive Arnoldi
orthogonalisation while leaving the trace-one steady-state component dominant.
"""
function _krylov_default_x0(dim::Integer)
    rng = MersenneTwister(0xb8fae9d3)
    G = randn(rng, ComplexF64, dim, dim)
    H = (G + G') / 2
    # Project out the trace: tr(H - tr(H)/d · I) = 0.
    H .-= (tr(H) / dim) .* I(dim)
    nH = opnorm(H)
    if nH > 0
        H ./= nH
    end
    eps_pert = 1e-10
    rho0 = Matrix{ComplexF64}(I(dim) / dim) .+ eps_pert .* H
    return vec(rho0)
end

"""
    apply_delta_channel!(ws, rho, config, hamiltonian) -> ws.scratch.rho_next

Apply one faithful jumpwise Lie--Trotter channel step.

# Arguments
- `ws`: Precomputed thermalization workspace.
- `rho`: Input matrix.
- `config`: Channel configuration, including `delta`.
- `hamiltonian`: Hamiltonian associated with the workspace.

# Keywords
- `hermitize`: Project each substep onto Hermitian matrices. Set `false` when
  materialising the complex-linear channel on arbitrary operators.

# Returns
`ws.scratch.rho_next` containing the channel output.
"""
function apply_delta_channel!(
    ws::Workspace{KrylovSpectrum, D, C, T},
    rho::Matrix{CT},
    config::Config{Thermalize, D},
    hamiltonian::HamHam;
    hermitize::Bool = true,
) where {D, C, T<:AbstractFloat, CT<:Complex}
    sc = ws.scratch::ThermalizeScratch{CT}
    K0s = ws.K0s::Vector{Matrix{CT}}
    U_residuals = ws.U_residuals::Vector{Matrix{CT}}
    U_coherents = ws.U_coherents
    jumps = ws.jumps::Vector{JumpOp}
    ham_or_trott = ws.ham_or_trott
    jws = ws.gamma_norm_factor::Float64

    # Present workspace data through the full-DM substep's shared field surface.
    pd = (
        transition           = ws.transition,
        gamma_norm_factor    = ws.gamma_norm_factor,
        energy_labels        = ws.energy_labels,
        oft_domain_prefactor = ws.oft_domain_prefactor,
        oft_nufft_prefactors = ws.oft_nufft_prefactors,
        alpha                = ws.bohr_alpha,
        bohr_keys            = ws.bohr_keys,
        bohr_is              = ws.bohr_is,
        bohr_js              = ws.bohr_js,
        b_minus              = ws.b_minus,
        b_plus               = ws.b_plus,
    )

    # The evolving input must not alias buffers written by the channel kernel.
    copyto!(sc.rho_work, rho)
    evolving_dm = sc.rho_work

    @inbounds for a in 1:length(jumps)
        U_a = U_coherents === nothing ? nothing : U_coherents[a]
        _apply_one_dm_substep!(
            evolving_dm, sc, jumps[a],
            U_a, K0s[a], U_residuals[a],
            ham_or_trott, config, pd, jws;
            hermitize = hermitize,
        )
    end

    copyto!(sc.rho_next, evolving_dm)
    return sc.rho_next
end

"""
    apply_adjoint_delta_channel!(ws, rho, config, hamiltonian; hermitize=true) -> ws.scratch.rho_next

Apply the Hilbert--Schmidt adjoint of the faithful jumpwise channel.

Substeps run in reverse order with adjointed Kraus operators. Hermitian jumps
allow the dissipator adjoint to reuse the forward kernel with frequencies
negated; non-Hermitian jumps are rejected.

# Keywords
- `hermitize`: Project each substep onto Hermitian matrices. Set `false` for
  the complex-linear adjoint on arbitrary operators.

# Returns
`ws.scratch.rho_next` containing the adjoint-channel output.
"""
function apply_adjoint_delta_channel!(
    ws::Workspace{KrylovSpectrum, D, C, T},
    rho::Matrix{CT},
    config::Config{Thermalize, D},
    hamiltonian::HamHam;
    hermitize::Bool = true,
) where {D, C, T<:AbstractFloat, CT<:Complex}
    sc = ws.scratch::ThermalizeScratch{CT}
    K0s = ws.K0s::Vector{Matrix{CT}}
    U_residuals = ws.U_residuals::Vector{Matrix{CT}}
    U_coherents = ws.U_coherents
    jumps = ws.jumps::Vector{JumpOp}
    ham_or_trott = ws.ham_or_trott
    jws = ws.gamma_norm_factor::Float64

    all(j -> j.hermitian, jumps) || throw(ArgumentError(
        "apply_adjoint_delta_channel! currently supports Hermitian jumps only " *
        "(the rate-flip dissipator-adjoint reuse assumes Aᵥ† = A₋ᵥ). Got a " *
        "non-Hermitian jump in the set."))

    # Math: for Hermitian jumps, the adjoint dissipator uses $gamma(nu) -> gamma(-nu)$.
    transition_adj = let t = ws.transition
        w -> t(-w)
    end
    pd_adj = (
        transition           = transition_adj,
        gamma_norm_factor    = ws.gamma_norm_factor,
        energy_labels        = ws.energy_labels,
        oft_domain_prefactor = ws.oft_domain_prefactor,
        oft_nufft_prefactors = ws.oft_nufft_prefactors,
        alpha                = ws.bohr_alpha,
        bohr_keys            = ws.bohr_keys,
        bohr_is              = ws.bohr_is,
        bohr_js              = ws.bohr_js,
        b_minus              = ws.b_minus,
        b_plus               = ws.b_plus,
    )

    copyto!(sc.rho_work, rho)
    evolving_dm = sc.rho_work

    # The adjoint reverses substep composition order.
    @inbounds for a in length(jumps):-1:1
        U_a = U_coherents === nothing ? nothing : U_coherents[a]
        _apply_one_adjoint_dm_substep!(
            evolving_dm, sc, jumps[a],
            U_a, K0s[a], U_residuals[a],
            ham_or_trott, config, pd_adj, jws;
            hermitize = hermitize,
        )
    end

    copyto!(sc.rho_next, evolving_dm)
    return sc.rho_next
end

# A physical trace-preserving channel has a stationary mode at mu = 1, but a
# peripheral mode such as mu = -1 can tie it in modulus. Select stationarity
# independently, then order only the remaining decay modes by decreasing |mu|.
# Raw non-TP polynomial surrogates have no distinguished stationary mode.
function _channel_mode_permutation(
    eigenvalues::AbstractVector;
    assume_trace_preserving::Bool,
)
    perm = sortperm(eigenvalues; by=abs, rev=true)
    assume_trace_preserving || return perm
    isempty(eigenvalues) && return perm

    stationary_idx = argmin(abs.(eigenvalues .- one(eltype(eigenvalues))))
    filter!(i -> i != stationary_idx, perm)
    pushfirst!(perm, stationary_idx)
    return perm
end

function _channel_spectral_gap(eigenvalues_sorted::AbstractVector, delta::Real)
    length(eigenvalues_sorted) >= 2 || return 0.0
    abs_mu2 = abs(eigenvalues_sorted[2])
    return abs_mu2 > 0 ? -log(abs_mu2) / delta : Inf
end

# Operator diagnostics do not have state-dependent modal coefficients.
function _operator_spectral_modes(eigenvalues_sorted::AbstractVector{<:Complex},
                                  vecs_sorted::AbstractVector, dim::Integer)
    R_modes_diag = [reshape(vecs_sorted[i], dim, dim) for i in eachindex(vecs_sorted)]
    return spectral_mode_diagnostics(eigenvalues_sorted, R_modes_diag)
end

# A trajectory predictor and its optional independent gap solve may share the
# expensive precomputed operator workspace, but never their Krylov
# factorisations. These two dispatches keep path-specific scratch and dimension
# checks in one place for predictors and gap solvers alike.
function _validate_reused_krylov_workspace(
    workspace::Workspace{KrylovSpectrum},
    config::Config,
    hamiltonian::HamHam,
    trotter::Union{Nothing, AbstractTrotter},
    jumps::Vector{JumpOp},
)
    workspace.cached_cfg === nothing && throw(ArgumentError(
        "workspace was built without a cached config (internal API path?). " *
        "Public callers should pass a workspace built via Workspace(config, ham, jumps; trotter=...)."))
    workspace.cached_cfg == config || throw(ArgumentError(
        "workspace.cached_cfg != config — cannot reuse a workspace whose " *
        "construction config differs from the call-site config (β, σ, a, s, δ, " *
        "register triples, with_gqsp, jump_selection, etc. all matter). Rebuild the workspace."))
    source = if config.domain isa TrotterDomain
        trotter === nothing && throw(ArgumentError("TrotterDomain requires a trotter object."))
        _validate_trotter_cache!(config, hamiltonian, trotter)
        trotter
    else
        hamiltonian
    end
    workspace.ham_or_trott === source || throw(ArgumentError(
        "workspace source Hamiltonian/Trotter object differs from the call-site source. " *
        "Rebuild the workspace for this system."))
    length(workspace.jumps) == length(jumps) || throw(ArgumentError(
        "workspace jump count differs from the call-site jump count. Rebuild the workspace."))
    # Workspace reuse treats jump matrices as immutable. The constructor keeps
    # a shallow snapshot, so this O(number of jumps) check accepts a copied
    # vector but rejects replaced operators without scanning every dense matrix.
    @inbounds for k in eachindex(jumps)
        cached = workspace.jumps[k]
        current = jumps[k]
        current.data === cached.data &&
        current.in_eigenbasis === cached.in_eigenbasis &&
        current.orthogonal == cached.orthogonal &&
        current.hermitian == cached.hermitian ||
            throw(ArgumentError(
                "jump $k differs from the operator used to build the workspace. " *
                "Rebuild the workspace; jump matrix contents must not be mutated in place."))
    end
    return workspace
end

function _reuse_or_build_krylov_workspace(
    workspace::Union{Nothing, Workspace{KrylovSpectrum}},
    config::Config{Lindbladian},
    hamiltonian::HamHam,
    jumps::Vector{JumpOp},
    trotter::Union{AbstractTrotter, Nothing},
    dim::Integer;
    caller::AbstractString,
)
    workspace === nothing && return Workspace(config, hamiltonian, jumps; trotter=trotter)
    workspace.scratch isa KrylovScratch || throw(ArgumentError(
        "$caller requires Workspace(::Config{Lindbladian}, ...); " *
        "got workspace with scratch::$(typeof(workspace.scratch))"))
    _validate_reused_krylov_workspace(workspace, config, hamiltonian, trotter, jumps)
    @assert size(workspace.G_left, 1) == dim  "workspace dim mismatch"
    return workspace
end

function _reuse_or_build_krylov_workspace(
    workspace::Union{Nothing, Workspace{KrylovSpectrum}},
    config::Config{Thermalize},
    hamiltonian::HamHam,
    jumps::Vector{JumpOp},
    trotter::Union{AbstractTrotter, Nothing},
    dim::Integer;
    caller::AbstractString,
)
    workspace === nothing && return Workspace(config, hamiltonian, jumps; trotter=trotter)
    workspace.scratch isa ThermalizeScratch || throw(ArgumentError(
        "$caller requires Workspace(::Config{Thermalize}, ...); " *
        "got workspace with scratch::$(typeof(workspace.scratch))"))
    _validate_reused_krylov_workspace(workspace, config, hamiltonian, trotter, jumps)
    @assert size(workspace.scratch.rho_next, 1) == dim  "workspace dim mismatch"
    return workspace
end

"""
    krylov_spectral_gap(config::Config{Lindbladian}, hamiltonian, jumps; kwargs...) -> NamedTuple

Compute the Lindbladian operator spectrum matrix-free with `:LR` targeting.

This solve is independent of any initial state and uses a deterministic
symmetry-broken seed. A converged Arnoldi solve can still select the wrong mode
when the true eigenvector is nearly seed-orthogonal; suspicious gaps should be
checked across Krylov dimensions and structurally different seeds.

# Arguments
- `config`: Lindbladian construction and numerical settings.
- `hamiltonian`: Hamiltonian spectral data.
- `jumps`: Coupling operators.

# Keywords
- `trotter`: Required construction for `TrotterDomain`.
- `krylovdim`, `howmany`, `tol`, `max_retries`: Eigensolver controls.
- `workspace`: Matching precomputed workspace; Krylov factorizations are not reused.
- `allow_unpaired_nonhermitian`: Opt out of adjoint-pair validation.
- `krylov_kwargs...`: Additional `KrylovKit` keywords.

# Returns
A named tuple with eigenvalues, gap, stationary state, gap mode, residuals,
convergence metadata, and spectral-mode diagnostics. `spectral_gap` is the first
resolved rate beyond all detected zero-compatible modes, or `NaN` for unresolved,
all-zero or unstable spectra. It is a captured estimate, never a global-gap
certificate. Inspect `spectrum_diagnostics` and `fixed_point_diagnostics`;
`raw_eigenvectors` are retained without state normalisation/repair. `normres`
contains recomputed relative Ritz residuals (in raw operator units).
"""
function krylov_spectral_gap(
    config::Config{Lindbladian},
    hamiltonian::HamHam,
    jumps::Vector{JumpOp};
    trotter::Union{AbstractTrotter, Nothing}=nothing,
    krylovdim::Int=30,
    howmany::Int=4,
    tol::Real=1e-10,
    max_retries::Int=3,
    allow_unpaired_nonhermitian::Bool=false,
    workspace::Union{Nothing, Workspace{KrylovSpectrum}}=nothing,
    krylov_kwargs...
)
    # Guards
    krylovdim > howmany || error("krylovdim ($krylovdim) must be > howmany ($howmany)")
    _check_krylov_memory(config.num_qubits, krylovdim)
    validate_config!(config, hamiltonian)
    validate_jump_pairing(jumps; allow_unpaired_nonhermitian=allow_unpaired_nonhermitian)
    # Dimensions
    dim = size(hamiltonian.data, 1)

    # Reuse construction data when provided; the Arnoldi factorization is fresh.
    ws = _reuse_or_build_krylov_workspace(
        workspace, config, hamiltonian, jumps, trotter, dim;
        caller="krylov_spectral_gap(::Config{Lindbladian}, ...)")

    # Build matvec closure: Vector{ComplexF64} -> Vector{ComplexF64}
    function lindbladian_matvec(v::AbstractVector)
        rho = reshape(v, dim, dim)
        apply_lindbladian!(ws, rho, config, hamiltonian)
        return copy(vec(ws.scratch.rho_out))  # KrylovKit must own each returned vector.
    end

    # Pure `I/d` can miss non-trivial symmetry sectors.
    x0 = _krylov_default_x0(dim)

    # Eigsolve with retry
    vals, vecs, info = _eigsolve_with_retry(
        lindbladian_matvec, x0, howmany, :LR;
        krylovdim=krylovdim, tol=tol, max_retries=max_retries,
        accept_invariant_subspace=true, krylov_kwargs...)

    # Sort eigenvalues by |Re(lambda)| ascending (steady state first, then gap mode)
    perm = sortperm(vals; by=v -> abs(real(v)))

    eigenvalues_sorted = vals[perm]
    vecs_sorted = vecs[perm]

    # Recompute residuals on owned, unnormalised Ritz modes; preserve raw data.
    normres = [norm(lindbladian_matvec(vecs_sorted[i]) - eigenvalues_sorted[i]*vecs_sorted[i]) /
        norm(vecs_sorted[i]) for i in eachindex(vecs_sorted)]
    scale = maximum((norm(lindbladian_matvec(v))/norm(v) for v in vecs_sorted);init=0.0)
    spectrum_diagnostics = spectral_gap_diagnostics(eigenvalues_sorted,normres;operator_scale=scale)
    fixed_point_diagnostics = _spectral_fixed_point(vecs_sorted,spectrum_diagnostics.stationary_indices,dim)
    fixed_point = fixed_point_diagnostics.state
    gap_index = spectrum_diagnostics.gap_index
    gap_mode = gap_index === nothing ? fill(ComplexF64(NaN),dim,dim) : reshape(copy(vecs_sorted[gap_index]),dim,dim)
    spectral_gap = spectrum_diagnostics.spectral_gap

    spectral_modes = _operator_spectral_modes(eigenvalues_sorted, vecs_sorted, dim)

    return (;
        eigenvalues = Complex{Float64}.(eigenvalues_sorted),
        spectral_gap,
        fixed_point = Complex{Float64}.(fixed_point),
        gap_mode = Complex{Float64}.(gap_mode),
        converged = info.converged,
        matvec_count = info.numops + 2length(vecs_sorted),
        num_restarts = info.numiter,
        normres,
        spectral_modes,
        spectrum_diagnostics,
        fixed_point_diagnostics,
        raw_eigenvectors = vecs_sorted,
        gap_mode_index = gap_index,
        channel_eigenvalues = nothing,
        delta_used = nothing,
    )
end

"""
    krylov_spectral_gap(config::Config{Thermalize}, hamiltonian, jumps; kwargs...) -> NamedTuple

Compute the faithful channel spectrum with `:LM` targeting.

The captured mode closest to `mu=1` is placed first; remaining modes are
ordered by decreasing `abs(mu)`. The reported generator-equivalent gap is
`-log(abs(mu_gap)) / delta` beyond the detected stationary subspace; the `eigenvalues` field retains the legacy
first-order conversion `(mu - 1) / delta` for diagnostics, while
`channel_eigenvalues` contains the raw discrete spectrum.

Direct Arnoldi on a non-normal channel can select a faster mode even after
formal convergence. Prefer the Lindbladian overload when only the robust
operator gap is needed; use this method to inspect the implemented channel's
own eigenvalues.

# Arguments
- `config`: Thermalization configuration, including `delta`.
- `hamiltonian`: Hamiltonian spectral data.
- `jumps`: Coupling operators.

# Keyword Arguments
As for the Lindbladian overload. A supplied workspace must match all
construction inputs.

# Returns
A named tuple with converted rates, gap data, diagnostics, and raw channel
eigenvalues in `channel_eigenvalues`.
"""
function krylov_spectral_gap(
    config::Config{Thermalize},
    hamiltonian::HamHam,
    jumps::Vector{JumpOp};
    trotter::Union{AbstractTrotter, Nothing}=nothing,
    krylovdim::Int=30,
    howmany::Int=4,
    tol::Real=1e-10,
    max_retries::Int=3,
    allow_unpaired_nonhermitian::Bool=false,
    workspace::Union{Nothing, Workspace{KrylovSpectrum}}=nothing,
    krylov_kwargs...
)
    # Guards
    krylovdim > howmany || error("krylovdim ($krylovdim) must be > howmany ($howmany)")
    _check_krylov_memory(config.num_qubits, krylovdim)
    validate_config!(config, hamiltonian)
    validate_jump_pairing(jumps; allow_unpaired_nonhermitian=allow_unpaired_nonhermitian)
    config.with_gqsp && throw(ArgumentError(
        "krylov_spectral_gap does not accept the unscaled GQSP polynomial " *
        "surrogate: it has no trace-preserving fixed point or physical channel gap."))

    # Get delta from config
    delta = config.delta

    # Dimensions
    dim = size(hamiltonian.data, 1)

    # Reuse construction data when provided; the Arnoldi factorization is fresh.
    ws = _reuse_or_build_krylov_workspace(
        workspace, config, hamiltonian, jumps, trotter, dim;
        caller="krylov_spectral_gap(::Config{Thermalize}, ...)")

    # Build the faithful jumpwise channel matvec.
    function channel_matvec(v::AbstractVector)
        rho = reshape(v, dim, dim)
        # Arnoldi acts on arbitrary complex operators. Hermitian projection is
        # only real-linear and would therefore change the channel spectrum.
        apply_delta_channel!(ws, rho, config, hamiltonian; hermitize=false)
        return copy(vec(ws.scratch.rho_next))  # KrylovKit must own each returned vector.
    end

    # Pure `I/d` can miss non-trivial symmetry sectors.
    x0 = _krylov_default_x0(dim)

    # Eigsolve with :LM targeting (channel eigenvalues cluster near 1)
    vals, vecs, info = _eigsolve_with_retry(
        channel_matvec, x0, howmany, :LM;
        krylovdim=krylovdim, tol=tol, max_retries=max_retries,
        accept_invariant_subspace=true, krylov_kwargs...)

    # Store raw channel eigenvalues before conversion
    channel_eigenvalues_raw = Complex{Float64}.(vals)

    # Retain the legacy first-order converted spectrum for diagnostics.
    lindblad_eigenvalues = (vals .- 1) ./ delta

    # A peripheral mode can tie the stationary mode in modulus. Pin the mode
    # closest to mu=1 first, then sort the remaining decay modes by |mu|.
    perm = _channel_mode_permutation(
        channel_eigenvalues_raw; assume_trace_preserving=true)

    eigenvalues_sorted = lindblad_eigenvalues[perm]
    vecs_sorted = vecs[perm]
    channel_eigenvalues_sorted = channel_eigenvalues_raw[perm]

    normres = [norm(channel_matvec(vecs_sorted[i]) - channel_eigenvalues_sorted[i]*vecs_sorted[i]) /
        norm(vecs_sorted[i]) for i in eachindex(vecs_sorted)]
    scale = maximum((norm(channel_matvec(v))/norm(v) for v in vecs_sorted);init=0.0)
    spectrum_diagnostics = spectral_gap_diagnostics(channel_eigenvalues_sorted,normres;
        operator_scale=scale,channel_delta=delta)
    fixed_point_diagnostics = _spectral_fixed_point(vecs_sorted,spectrum_diagnostics.stationary_indices,dim)
    fixed_point = fixed_point_diagnostics.state
    gap_index = spectrum_diagnostics.gap_index
    gap_mode = gap_index === nothing ? fill(ComplexF64(NaN),dim,dim) : reshape(copy(vecs_sorted[gap_index]),dim,dim)
    spectral_gap = spectrum_diagnostics.spectral_gap

    # Diagnostics use converted generator-rate units.
    spectral_modes = _operator_spectral_modes(eigenvalues_sorted, vecs_sorted, dim)

    return (;
        eigenvalues = Complex{Float64}.(eigenvalues_sorted),
        spectral_gap,
        fixed_point = Complex{Float64}.(fixed_point),
        gap_mode = Complex{Float64}.(gap_mode),
        converged = info.converged,
        matvec_count = info.numops + 2length(vecs_sorted),
        num_restarts = info.numiter,
        normres,
        spectral_modes,
        spectrum_diagnostics,
        fixed_point_diagnostics,
        raw_eigenvectors = vecs_sorted,
        gap_mode_index = gap_index,
        channel_eigenvalues = channel_eigenvalues_sorted,
        delta_used = Float64(delta),
        trace_preserving_assumed = true,
        physical_channel = true,
        channel_representation = :deterministic_cptp,
    )
end

"""
    run_krylov_spectrum(jumps, config, hamiltonian, trotter=nothing; krylov_kwargs...) -> KrylovSpectrumResults

Run the matrix-free spectral solver and attach configuration and timing metadata.

# Arguments
- `jumps`: Coupling operators.
- `config`: Lindbladian or thermalization configuration.
- `hamiltonian`: Hamiltonian spectral data.
- `trotter`: Required for `TrotterDomain`.

# Keyword Arguments
- `krylovdim`, `howmany`, `tol`, `max_retries`: Eigensolver controls.
- `allow_unpaired_nonhermitian`: Opt out of adjoint-pair validation.
- `krylov_kwargs...`: Additional `KrylovKit` keywords.

# Returns
A `KrylovSpectrumResults` with spectrum, stationary state, convergence data,
optional channel eigenvalues, configuration, and metadata.
"""
function run_krylov_spectrum(
    jumps::Vector{JumpOp},
    config::Config{S,D,C,T},
    hamiltonian::HamHam,
    trotter::Union{AbstractTrotter, Nothing}=nothing;
    krylovdim::Int=30,
    howmany::Int=4,
    tol::Real=1e-10,
    max_retries::Int=3,
    allow_unpaired_nonhermitian::Bool=false,
    krylov_kwargs...
) where {S<:Union{Lindbladian, Thermalize}, D, C, T}

    t_start = time()

    krylov_result = krylov_spectral_gap(
        config, hamiltonian, jumps;
        trotter=trotter, krylovdim=krylovdim, howmany=howmany,
        tol=tol, max_retries=max_retries,
        allow_unpaired_nonhermitian=allow_unpaired_nonhermitian,
        krylov_kwargs...
    )

    wall_time = time() - t_start
    metadata = _capture_metadata(wall_time_seconds=wall_time)
    metadata[:spectral_modes] = krylov_result.spectral_modes
    metadata[:spectrum_diagnostics] = krylov_result.spectrum_diagnostics
    metadata[:fixed_point_diagnostics] = krylov_result.fixed_point_diagnostics
    metadata[:gap_mode_index] = krylov_result.gap_mode_index
    if config isa Config{Thermalize}
        metadata[:trace_preserving_assumed] = krylov_result.trace_preserving_assumed
        metadata[:physical_channel] = krylov_result.physical_channel
        metadata[:channel_representation] = krylov_result.channel_representation
    end

    return KrylovSpectrumResults{Float64}(
        config,
        krylov_result.eigenvalues,
        krylov_result.spectral_gap,
        krylov_result.fixed_point,
        krylov_result.gap_mode,
        krylov_result.converged,
        krylov_result.matvec_count,
        krylov_result.num_restarts,
        krylov_result.normres,
        krylov_result.channel_eigenvalues,
        krylov_result.delta_used,
        metadata,
    )
end
