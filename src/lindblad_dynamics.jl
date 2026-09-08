# Matrix-free Krylov integration of Lindbladian and discriminant flows.
# Math: $dot(rho) = L(rho)$ and $psi = sigma^(-1/4) rho sigma^(-1/4)$.

function _validate_time_grid(times; require_zero::Bool=false)
    isempty(times) && throw(ArgumentError("times must be nonempty."))
    all(t -> t isa Real && isfinite(t) && t >= 0,times) ||
        throw(ArgumentError("times must be finite nonnegative real generator times."))
    all(i -> times[i] > times[i-1],2:length(times)) ||
        throw(ArgumentError("times must be strictly increasing."))
    !require_zero || iszero(first(times)) || throw(ArgumentError("times must start at zero."))
    return nothing
end

"""
    lindblad_action_integrate(L_apply!, rho_0, sigma_beta, t_grid;
                              krylovdim=30, tol=1e-10, save_states=false)

Integrate a matrix-free Lindblad equation over an ordered time grid.

# Arguments
- `L_apply!`: In-place closure writing `L(input)` into its first argument.
- `rho_0`: Initial density matrix.
- `sigma_beta`: Reference state for trace distance.
- `t_grid`: Ordered sample times.

# Keywords
- `krylovdim`: Arnoldi subspace size.
- `tol`: Per-step Krylov tolerance.
- `save_states`: Retain every propagated state.
- `repair_states`: Legacy Hermitian/trace correction (default true); false retains raw states.
- `record_diagnostics`: Record raw validity checks before correction and every repair norm.
- `matvec_callback`, `work_callback`: Optional cooperative budget checks; exhaustion
  returns completed samples with a failure reason.

# Returns
A named tuple with times, trace distances, final state, matvec count,
convergence status, and optional states.
"""

function lindblad_action_integrate(
    L_apply!::F, rho_0::Matrix{T}, sigma_beta::Matrix{T},
    t_grid::AbstractVector{<:Real}; krylovdim::Int=30, tol::Real=1e-10,
    save_states::Bool=false, repair_states::Bool=true,
    record_diagnostics::Bool=false, dense_max_dim::Integer=64,
    max_dense_bytes::Integer=64*1024^2,
    matvec_callback=nothing, work_callback=nothing,
)::NamedTuple where {T<:Complex,F}
    _validate_time_grid(t_grid)
    d = size(rho_0,1)
    size(rho_0) == size(sigma_beta) == (d,d) || throw(ArgumentError("State dimensions must match."))
    krylovdim > 0 && isfinite(tol) && tol > 0 || throw(ArgumentError("Require positive krylovdim and finite positive tol."))
    rho_buf = Matrix{T}(undef,d,d)
    out_buf = similar(rho_buf)
    count = Ref(0)
    function L_vec_apply(v::AbstractVector)
        matvec_callback === nothing || matvec_callback()
        count[] += 1
        copyto!(rho_buf,reshape(v,d,d))
        L_apply!(out_buf,rho_buf)
        return copy(vec(out_buf))
    end
    work() = work_callback === nothing ? nothing : work_callback()
    rho = copy(rho_0)
    distances = Float64[sum(svdvals(rho-sigma_beta))/2]
    states = save_states ? [copy(rho)] : Matrix{T}[]
    state_rtol = max(1e-9,100eps(real(T)))
    raw_checks = NamedTuple[]
    repair_norms = Float64[0.]
    record_diagnostics && push!(raw_checks,state_diagnostics(rho;
        rtol=state_rtol,dense_max_dim,max_dense_bytes))
    all_converged = true
    failure = nothing
    completed = 1
    for i in 2:length(t_grid)
        try
            work()
            dt = float(t_grid[i]-t_grid[i-1])
            # Form the dimensionless action dt*L before Arnoldi. Calling
            # exponentiate(L,dt,...) can mistake slow-clock L for a zero map
            # under its absolute breakdown threshold, even when dt*L is O(1).
            step_apply(v) = dt .* L_vec_apply(v)
            v_next,info = exponentiate(step_apply,one(dt),vec(rho);
                krylovdim,tol,ishermitian=false)
            work()
            candidate = reshape(copy(v_next),d,d)
            all(isfinite,candidate) || error("Propagation returned a nonfinite state.")
            checks = record_diagnostics ? state_diagnostics(candidate;rtol=state_rtol,dense_max_dim,max_dense_bytes) : nothing
            original = repair_states ? copy(candidate) : nothing
            if repair_states
                hermitianize!(candidate)
                trace_value = real(tr(candidate))
                iszero(trace_value) || (candidate ./= trace_value)
            end
            distance = sum(svdvals(candidate-sigma_beta))/2
            # Publish a point atomically, only after its measurements succeeded.
            push!(distances,distance)
            record_diagnostics && push!(raw_checks,checks)
            push!(repair_norms,repair_states ? norm(candidate-original) : 0.)
            save_states && push!(states,copy(candidate))
            rho = candidate
            completed = i
            if info.converged == 0
                all_converged = false
                failure = (;reason=:nonconvergence,message="Krylov propagation did not converge at the final returned sample.")
                break
            end
        catch err
            err isa InterruptException && rethrow()
            # Existing unbudgeted low-level callers retain their exception semantics.
            matvec_callback === nothing && work_callback === nothing && rethrow()
            all_converged = false
            failure = (;reason=err isa _WorkLimit ? err.reason : :solver_failure,message=sprint(showerror,err))
            break
        end
    end
    return (;t=collect(t_grid[1:completed]),distances,rho_final=copy(rho),
        total_matvecs=count[],all_converged,states,raw_checks,repair_norms,
        repair_policy=repair_states ? :hermitian_trace : :none,failure)
end


"""
    discriminant_action_integrate(K_apply!, psi_0, psi_eq, t_grid;
                                  krylovdim=30, tol=1e-10,
                                  is_hermitian=true, save_states=false)

Integrate a matrix-free discriminant equation over an ordered time grid.

Set `is_hermitian=true` only when KMS detailed balance makes the discriminant
Hilbert--Schmidt self-adjoint; otherwise Arnoldi is required.

# Returns
A named tuple with times, Frobenius distances, final state, matvec count,
convergence status, and optional states.
"""
function discriminant_action_integrate(
    K_apply!::F,
    psi_0::Matrix{T},
    psi_eq::Matrix{T},
    t_grid::AbstractVector{<:Real};
    krylovdim::Int = 30,
    tol::Real = 1e-10,
    is_hermitian::Bool = true,
    save_states::Bool = false,
)::NamedTuple where {T<:Complex, F}
    d = size(psi_0, 1)
    @assert size(psi_0, 2) == d  "psi_0 must be square"

    in_buf  = Matrix{T}(undef, d, d)
    out_buf = Matrix{T}(undef, d, d)

    # KrylovKit may overwrite `out_buf` on the next call, so we
    # `copy(vec(out_buf))` before returning; we also `copyto!` the input
    # into our private buffer in case `v` aliases internal state.
    function K_vec_apply(v::AbstractVector)
        copyto!(in_buf, reshape(v, d, d))
        K_apply!(out_buf, in_buf)
        return copy(vec(out_buf))
    end

    n_steps   = length(t_grid)
    distances = Vector{Float64}(undef, n_steps)
    states    = save_states ? Vector{Matrix{T}}(undef, n_steps) : Matrix{T}[]

    psi = copy(psi_0)
    distances[1] = norm(psi - psi_eq)             # Frobenius distance (chi metric)
    save_states && (states[1] = copy(psi))

    v_psi = copy(vec(psi))
    total_matvecs = 0
    all_converged = true

    @inbounds for i in 1:(n_steps - 1)
        dt = float(t_grid[i + 1] - t_grid[i])

        v_next, info = exponentiate(K_vec_apply, dt, v_psi;
                                    krylovdim = krylovdim,
                                    tol = tol,
                                    ishermitian = is_hermitian)
        total_matvecs += info.numops
        if info.converged == 0
            all_converged = false
            @warn "K-mode exponentiate did not converge at step" i numops=info.numops
        end
        copyto!(v_psi, v_next)
        copyto!(psi, reshape(v_psi, d, d))

        # KMS-DB preserves Hermiticity and $<psi_eq,psi>_F = tr(rho) = 1$;
        # correct accumulated Krylov round-off in both invariants.
        hermitianize!(psi)
        c_now = real(dot(psi_eq, psi))
        if c_now != 0
            psi ./= c_now
        end
        copyto!(v_psi, vec(psi))

        distances[i + 1] = norm(psi - psi_eq)
        save_states && (states[i + 1] = copy(psi))
    end

    return (
        t              = collect(t_grid),
        distances      = distances,
        psi_final      = copy(psi),
        total_matvecs  = total_matvecs,
        all_converged  = all_converged,
        states         = states,
    )
end


"""
    integrate_to_gibbs(config, hamiltonian, jumps, rho_0, t_grid;
                       mode=:L, krylovdim=30, tol=1e-10, save_states=false)

Integrate a configured Lindbladian or its KMS discriminant toward Gibbs.

# Arguments
- `config`: Bohr- or energy-domain Lindbladian configuration.
- `hamiltonian`: Hamiltonian with cached Gibbs state.
- `jumps`: Coupling operators.
- `rho_0`: Initial density matrix.
- `t_grid`: Ordered sample times.

# Keywords
- `mode`: `:L` for density matrices or `:K` for the discriminant representation.
- `krylovdim`, `tol`, `save_states`: Forwarded to the selected integrator.
- `allow_unpaired_nonhermitian`: Opt out of adjoint-pair validation.

# Returns
The selected integrator's named tuple. Configuration and DLL filter constraints
are validated before constructing the workspace.
"""
function integrate_to_gibbs(
    config::Config{Lindbladian, <:Union{BohrDomain, EnergyDomain}},
    hamiltonian::HamHam,
    jumps::Vector{JumpOp},
    rho_0::Matrix{T},
    t_grid::AbstractVector{<:Real};
    mode::Symbol = :L,
    krylovdim::Int = 30,
    tol::Real = 1e-10,
    save_states::Bool = false,
    allow_unpaired_nonhermitian::Bool = false,
)::NamedTuple where {T<:Complex}
    mode in (:L, :K) || throw(ArgumentError("mode must be :L or :K (got :$mode)"))
    d = size(rho_0, 1)
    @assert size(rho_0, 2) == d  "rho_0 must be square"

    # validate_config! is invoked by run_lindblad/run_thermalize; we call it
    # explicitly here since this entry point bypasses those.
    validate_config!(config, hamiltonian)
    validate_jump_pairing(jumps; allow_unpaired_nonhermitian=allow_unpaired_nonhermitian)

    # Build the L_apply!(out, in) closure: matrix-free for both KMS and DLL.
    # `let` scope binds the captured state directly to dodge Box wrapping under
    # Julia 1.11+ closure capture rules.
    ws = Workspace(config, hamiltonian, jumps)
    L_apply! = let ws = ws, config = config, ham = hamiltonian
        (out::AbstractMatrix, x::AbstractMatrix) -> begin
            apply_lindbladian!(ws, x, config, ham)
            copyto!(out, ws.scratch.rho_out)
            return out
        end
    end

    if mode == :L
        sigma_beta = Matrix{T}(hamiltonian.gibbs)
        return lindblad_action_integrate(
            L_apply!, rho_0, sigma_beta, t_grid;
            krylovdim = krylovdim, tol = tol, save_states = save_states,
        )
    else  # mode == :K
        powers = gibbs_fractional_powers(hamiltonian.gibbs)
        sq, sq_inv, sh = powers.sigma_quarter, powers.sigma_inv_quarter, powers.sigma_half

        # psi_0 = sigma^{-1/4} rho_0 sigma^{-1/4} (diagonal multiply, BohrDomain).
        psi_0 = Matrix{T}(undef, d, d)
        @inbounds for j in 1:d, i in 1:d
            psi_0[i, j] = sq_inv[i] * rho_0[i, j] * sq_inv[j]
        end
        # psi_eq = sigma^{1/2} as a full Matrix (the integrator wants a Matrix, not Diagonal).
        psi_eq = Matrix{T}(Diagonal(complex.(sh)))

        bufs = DiscriminantBuffers{T}(d)
        K_apply! = let L = L_apply!, sq = sq, sq_inv = sq_inv, bufs = bufs
            (out::AbstractMatrix, x::AbstractMatrix) -> begin
                apply_discriminant!(out, x, L, sq, sq_inv, bufs)
                return out
            end
        end

        return discriminant_action_integrate(
            K_apply!, psi_0, psi_eq, t_grid;
            krylovdim = krylovdim, tol = tol,
            is_hermitian = true,  # KMS-DB ⇒ K is HS-self-adjoint (Lanczos OK)
            save_states = save_states,
        )
    end
end
