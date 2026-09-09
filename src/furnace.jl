"""
    construct_lindbladian(jumps, config, hamiltonian; trotter=nothing,
                          include_coherent=true, allow_unpaired_nonhermitian=false,
                          verbose=false) -> Matrix

Construct the dense, column-vectorised Lindbladian superoperator.

# Arguments
- `jumps`: Jump operators in the domain's working basis.
- `config`: Lindbladian configuration.
- `hamiltonian`: Hamiltonian and spectral data.

# Keywords
- `trotter`: Required cache for `TrotterDomain`.
- `include_coherent`: Include the coherent correction; `false` constructs the
  dissipator-only diagnostic.
- `allow_unpaired_nonhermitian`: Skip adjoint-closure validation.
- `verbose`: Print construction progress.

# Returns
The dense `d^2 x d^2` generator acting on column-stacked operators.
"""
function construct_lindbladian(jumps::Vector{JumpOp}, config::Config{Lindbladian}, hamiltonian::HamHam;
    trotter::Union{AbstractTrotter, Nothing}=nothing,
    include_coherent::Bool=true,
    allow_unpaired_nonhermitian::Bool=false,
    verbose::Bool=false)

    validate_config!(config, hamiltonian)
    validate_jump_pairing(jumps; allow_unpaired_nonhermitian=allow_unpaired_nonhermitian)

    if verbose
        domain_name = replace(string(typeof(config.domain)), "Domain" => "")
        println("Constructing Liouvillian ($(domain_name))")
    end

    ham_or_trott = if config.domain isa TrotterDomain
        trotter === nothing && error("A Trotter object must be provided for the TrotterDomain")
        _validate_trotter_cache!(config, hamiltonian, trotter)
        trotter
    else # For Bohr, Energy, Time domains
        hamiltonian
    end

    precomputed_data = _precompute_data(config, ham_or_trott, jumps)

    dim = size(hamiltonian.data, 1)
    T = eltype(hamiltonian.eigvals)
    CT = Complex{T}
    total_lindbladian = zeros(CT, dim^2, dim^2)

    ws = DenseLindbladianWorkspace(CT, dim)

    # Precompute all B's for the A's if for KMS DB and with_coherent.
    # `include_coherent=false` skips the coherent contribution even when the
    # construction's `with_coherent` trait is true -- used by detailed-balance
    # diagnostics that need to compare the dissipator-only Lindbladian against
    # the full one (e.g. KMS without the Lamb-shift restoration).
    if include_coherent
        Btot = _precompute_coherent_B(jumps, ham_or_trott, config, precomputed_data)
        if Btot !== nothing
            _vectorize_liouvillian_coherent!(total_lindbladian, Btot, ws)
        end
    end

    # Jumps arrive in the correct basis (trotter.eigvecs for TrotterDomain,
    # hamiltonian.eigvecs for other domains -- basis selection is at the source).

    # Accumulate Liouvillian in-place (no per-jump dim^2 x dim^2 allocations).
    for (k, jump) in pairs(jumps)
        _jump_contribution!(total_lindbladian, jump, ham_or_trott, config, _dll_source_data(precomputed_data,k), ws;
            coherent_term=nothing)
    end

    return total_lindbladian
end

# ============================================================================
# Public entry points
# ============================================================================

"""
    run_lindblad(jumps, config, hamiltonian, trotter=nothing) -> LindbladResults

Construct the dense Lindbladian and extract its steady state and gap mode with
shift-invert Arnoldi.

# Arguments
- `jumps`: jump operators in the construction's working basis.
- `config`: Lindbladian configuration.
- `hamiltonian`: Hamiltonian and eigenbasis data.
- `trotter`: required cache for `TrotterDomain`.

# Returns
A [`LindbladResults`](@ref) with spectral data and metadata.
"""
function run_lindblad(
    jumps::Vector{JumpOp},
    config::Config{Lindbladian,D,C,T},
    hamiltonian::HamHam{T},
    trotter::Union{AbstractTrotter, Nothing}=nothing;
    allow_unpaired_nonhermitian::Bool=false,
    verbose::Bool=false,
) where {D, C, T<:AbstractFloat}

    t_start = time()

    verbose && _print_press(config)

    liouv = construct_lindbladian(jumps, config, hamiltonian; trotter=trotter,
        allow_unpaired_nonhermitian=allow_unpaired_nonhermitian, verbose=verbose)
    verbose && @printf("Done.\n")

    # Arpack shift-invert eigensolver
    shift = 1e-9 * (1 + 1im)
    eigvals_near_zero, eigvecs_near_zero = eigs(liouv, nev=2, sigma=shift, tol=1e-12)
    sorted_permutation_eigen = sortperm(abs.(real.(eigvals_near_zero)))

    ss_index = sorted_permutation_eigen[1]
    gap_index = sorted_permutation_eigen[2]
    spectral_gap = eigvals_near_zero[gap_index]

    steady_state_vec = eigvecs_near_zero[:, ss_index]
    steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
    _normalize_stationary_mode!(steady_state_dm)

    gap_vec = eigvecs_near_zero[:, gap_index]
    gap_mode_op = reshape(gap_vec, size(hamiltonian.data))

    wall_time = time() - t_start
    metadata = _capture_metadata(wall_time_seconds=wall_time)

    return LindbladResults{T}(
        config,
        Complex{T}.(eigvals_near_zero[sorted_permutation_eigen]),
        Complex{T}.(steady_state_dm),
        Complex{T}.(gap_mode_op),
        Complex{T}(spectral_gap),
        metadata,
    )
end

"""
    run_thermalize(jumps, config, hamiltonian, trotter=nothing; initial_dm=nothing, rng, rescale_by_inv_prob, save_every) -> ThermalizeResults

Evolve a density matrix with the retained faithful channel and record its
distance from the Gibbs state.

# Arguments
- `jumps`: jump operators in the channel's working basis.
- `config`: thermalisation configuration, including `mixing_time` and `delta`.
- `hamiltonian`: Hamiltonian and eigenbasis data.
- `trotter`: required cache for `TrotterDomain`.

# Keywords
- `initial_dm`: initial state; defaults to `I/d`.
- `rng`: used only for `jump_selection=:random`.
- `rescale_by_inv_prob`: override the selection-dependent rate rescaling.
- `save_every`: record and check convergence every this many outer steps; always
  record the final completed step.
- `num_steps`: exact nonnegative outer-step count; defaults to `ceil(mixing_time/delta)`.
- `record_steps`: strictly increasing integer recording grid from `0` to `num_steps`;
  overrides `save_every`.
- `save_states`: store copies of recorded states in `metadata[:states]`.
- `hermitize`: preserve the legacy per-substep Hermitian projection; `false` keeps raw states.
- `convergence_cutoff`: stop at a recorded distance below this value; `0` disables it.
- `work_callback`: optional zero-argument callback before each outer step. An internal
  work-limit exception stops cleanly at the last completed step; other errors propagate.
- `observation_callback`: optional `(step, rho)` callback on initial and recorded states.
  Treat `rho` as read-only; copy it if retaining it beyond the callback.
- `allow_unpaired_nonhermitian`: opt out of adjoint-pair validation.
- `verbose`: print configuration and recorded distances.

# Returns
A [`ThermalizeResults`](@ref) with the final state, saved distances and times,
and metadata. When `with_gqsp=true`, the coherent step is the raw, unscaled
Jacobi–Anger polynomial surrogate rather than a certified GQSP unitary block.
It is not guaranteed contractive. No trace renormalisation is applied; the
metadata records `trace_values`, `trace_drift`, `final_trace`, and
`max_abs_trace_drift`.
"""
function run_thermalize(
    jumps::Vector{JumpOp},
    config::Config{Thermalize,D,C,T},
    hamiltonian::HamHam{T},
    trotter::Union{AbstractTrotter, Nothing}=nothing;
    initial_dm::Union{Nothing, Matrix{<:Complex}}=nothing,
    rng::AbstractRNG = Random.default_rng(),
    rescale_by_inv_prob::Union{Bool, Nothing} = nothing,
    save_every::Int = 1,
    num_steps::Union{Nothing,Int} = nothing,
    record_steps = nothing,
    save_states::Bool = false,
    hermitize::Bool = true,
    convergence_cutoff::Real = 1e-5,
    work_callback = nothing,
    observation_callback = nothing,
    allow_unpaired_nonhermitian::Bool = false,
    verbose::Bool = false,
) where {D, C, T<:AbstractFloat}

    dim = size(hamiltonian.data, 1)

    # Default initial_dm: maximally mixed state I/d
    evolving_dm = if initial_dm === nothing
        Matrix{Complex{T}}(I(dim) / dim)
    else
        copy(initial_dm)
    end

    t_start = time()

    validate_config!(config, hamiltonian)
    isempty(jumps) && throw(ArgumentError("run_thermalize requires at least one jump operator."))
    # Float32 basis rotation needs relative working-precision tolerance; an
    # absolute floor would incorrectly accept mismatched tiny sources.
    # Preserve the legacy policy for other working precisions.
    validate_jump_pairing(jumps; allow_unpaired_nonhermitian=allow_unpaired_nonhermitian,
        atol=T===Float32 ? 0 : 1e-12, rtol=T===Float32 ? 100eps(T) : nothing)
    @assert save_every >= 1 "save_every must be >= 1"
    isfinite(convergence_cutoff) && convergence_cutoff >= 0 ||
        throw(ArgumentError("convergence_cutoff must be finite and nonnegative."))
    total_steps = num_steps === nothing ? Int(ceil(config.mixing_time / config.delta)) : num_steps
    total_steps >= 0 || throw(ArgumentError("num_steps must be nonnegative."))
    requested_record_steps = if record_steps === nothing
        nothing
    else
        grid = collect(record_steps)
        !isempty(grid) && all(x -> x isa Integer && !(x isa Bool), grid) ||
            throw(ArgumentError("record_steps must be a nonempty integer grid."))
        first(grid) == 0 && last(grid) == total_steps &&
            all(i -> grid[i] < grid[i + 1], 1:(length(grid) - 1)) ||
            throw(ArgumentError("record_steps must increase strictly from 0 to num_steps."))
        Int.(grid)
    end
    verbose && _print_press(config)

    if config.domain isa TrotterDomain
        @assert trotter !== nothing
        _validate_trotter_cache!(config, hamiltonian, trotter)
        ham_or_trott = trotter
        gibbs = Hermitian(trotter.eigvecs' * hamiltonian.eigvecs * hamiltonian.gibbs *
                          hamiltonian.eigvecs' * trotter.eigvecs)
    else
        ham_or_trott = hamiltonian
        gibbs = hamiltonian.gibbs
    end

    precomputed_data = _precompute_data(config, ham_or_trott)

    # Random selection rescales by the jump count to reproduce the full generator
    # in expectation; a sweep applies every bare-rate subchannel sequentially.
    rescale = rescale_by_inv_prob === nothing ? (config.jump_selection == :random) : rescale_by_inv_prob

    n_jumps = length(jumps)
    p_jump = 1.0 / n_jumps
    coherent_unitaries = _precompute_coherent_unitary(jumps, hamiltonian, config, precomputed_data;
        trotter=trotter, delta_scale = rescale ? (1.0 / p_jump) : 1.0)

    CT = eltype(evolving_dm)
    scratch = ThermalizeScratch(CT, dim)

    # Precompute per-jump CPTP channels (K0, U_residual) -- eliminates eigen() from hot loop
    (; K0s, U_residuals) = _precompute_per_jump_channels(
        jumps, ham_or_trott, config, precomputed_data;
        rescale_by_inv_prob = rescale,
    )

    # Precompute jump_weight_scaling for _accumulate_rho_jump!
    jump_weight_scaling = rescale ? (precomputed_data.gamma_norm_factor / p_jump) : precomputed_data.gamma_norm_factor

    gibbs_matrix = Matrix(gibbs)
    trace_distances = T[]
    trace_values = Complex{T}[]
    recorded_steps = Int[]
    states = save_states ? Matrix{CT}[] : nothing
    function record_state!(step)
        distance = trace_distance_nh(evolving_dm, gibbs_matrix)
        push!(trace_distances, distance)
        push!(trace_values, tr(evolving_dm))
        push!(recorded_steps, step)
        save_states && push!(states, copy(evolving_dm))
        observation_callback === nothing || observation_callback(step, evolving_dm)
        return distance
    end
    record_state!(0)
    completed_steps = 0
    failure = nothing
    next_record = 2

    for step in 1:total_steps
        if work_callback !== nothing
            try
                work_callback()
            catch err
                err isa _WorkLimit || rethrow()
                failure = (; reason=err.reason, message=sprint(showerror, err))
                break
            end
        end
        if config.jump_selection == :sweep
            # Math: $Phi_A = Phi_S compose dots compose Phi_1 approx exp(delta cal(L))$.
            @inbounds for a in 1:n_jumps
                _apply_one_dm_substep!(
                    evolving_dm, scratch, jumps[a],
                    coherent_unitaries === nothing ? nothing : coherent_unitaries[a],
                    K0s[a], U_residuals[a],
                    ham_or_trott, config, precomputed_data, jump_weight_scaling;
                    hermitize=hermitize,
                )
            end
        else  # :random
            idx = rand(rng, 1:n_jumps)
            _apply_one_dm_substep!(
                evolving_dm, scratch, jumps[idx],
                coherent_unitaries === nothing ? nothing : coherent_unitaries[idx],
                K0s[idx], U_residuals[idx],
                ham_or_trott, config, precomputed_data, jump_weight_scaling;
                hermitize=hermitize,
            )
        end

        completed_steps = step
        record_now = requested_record_steps === nothing ? step % save_every == 0 :
            next_record <= length(requested_record_steps) && step == requested_record_steps[next_record]
        if record_now || step == total_steps
            dist = record_state!(step)
            next_record += 1
            verbose && @printf("Dist to Gibbs: %s\n", dist)
            if dist < convergence_cutoff
                break
            end
        end
    end

    # A budget can stop between requested observations. Retain its completed
    # endpoint so final_dm, recorded times, distances, and states stay aligned.
    if last(recorded_steps) != completed_steps
        record_state!(completed_steps)
    end
    time_steps = T.(recorded_steps .* config.delta)

    wall_time = time() - t_start
    metadata = _capture_metadata(wall_time_seconds=wall_time)
    metadata[:save_every] = save_every
    metadata[:requested_steps] = total_steps
    metadata[:completed_steps] = completed_steps
    metadata[:recorded_steps] = recorded_steps
    metadata[:states] = states
    metadata[:hermitized] = hermitize
    metadata[:failure] = failure
    metadata[:convergence_cutoff] = convergence_cutoff
    metadata[:trace_values] = trace_values
    metadata[:trace_drift] = trace_values .- one(Complex{T})
    metadata[:final_trace] = tr(evolving_dm)
    metadata[:final_trace_drift] = metadata[:final_trace] - one(Complex{T})
    metadata[:max_abs_trace_drift] = max(
        maximum(abs, metadata[:trace_drift]; init=zero(T)),
        abs(metadata[:final_trace_drift]),
    )
    metadata[:trace_normalized] = false
    metadata[:trace_preserving_assumed] = !config.with_gqsp
    metadata[:physical_channel] = !config.with_gqsp
    metadata[:channel_representation] = config.with_gqsp ?
        :unscaled_gqsp_polynomial_surrogate : :deterministic_cptp

    return ThermalizeResults{T}(
        config,
        Complex{T}.(evolving_dm),
        T.(trace_distances),
        T.(time_steps),
        metadata,
    )
end
