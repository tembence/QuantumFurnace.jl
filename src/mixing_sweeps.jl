"""
    _make_init_state(init_state::Symbol, d::Integer,
                     sigma_beta::AbstractMatrix, seed::Integer) -> Matrix{ComplexF64}

Build a maximally mixed, seeded random-pure, or Gibbs-perturbed initial state.
"""
function _make_init_state(init_state::Symbol, d::Integer,
                          sigma_beta::AbstractMatrix, seed::Integer)
    if init_state === :maximally_mixed
        return Matrix{ComplexF64}(I(d) / d)
    elseif init_state === :random_pure
        rng = MersenneTwister(seed)
        psi = randn(rng, ComplexF64, d)
        psi ./= norm(psi)
        return psi * psi'
    elseif init_state === :thermal_perturbed
        rng = MersenneTwister(seed)
        # GUE perturbation: H = (G + G') / 2, G ~ ComplexNormal.
        G = randn(rng, ComplexF64, d, d)
        H = (G + G') / 2
        H_norm = max(norm(H), 1e-30)
        H .*= (1e-3 / d) / H_norm
        rho = Matrix{ComplexF64}(sigma_beta) .+ H
        # Re-Hermitise + renormalise trace.
        rho .= (rho + rho') / 2
        tr_now = real(tr(rho))
        tr_now <= 0 && return Matrix{ComplexF64}(I(d) / d)  # fallback
        rho ./= tr_now
        return rho
    else
        throw(ArgumentError(
            "init_state must be :maximally_mixed, :random_pure, or :thermal_perturbed (got :$init_state)"))
    end
end

"""
    _sweep_sidecar_path(output_dir, n, beta, seed, mode, construction_tag, domain_tag;
                        beta_phys=nothing) -> String

Return a collision-resistant Lindbladian-sweep sidecar path.

The filename distinguishes `beta_alg` from `beta_phys`, construction, and domain.
"""
function _sweep_sidecar_path(output_dir::AbstractString, n::Integer, beta::Real,
                             seed::Integer, mode::Symbol,
                             construction_tag::AbstractString,
                             domain_tag::AbstractString;
                             beta_phys::Union{Nothing, Real} = nothing)
    b_value = beta_phys === nothing ? float(beta) : float(beta_phys)
    b_tag   = beta_phys === nothing ? "beta"      : "betaphys"
    beta_str = let s = @sprintf("%.6f", b_value)
        s = rstrip(s, '0')
        s = rstrip(s, '.')
        isempty(s) ? "0" : s
    end
    fname = "sweep_n$(n)_$(b_tag)$(beta_str)_seed$(seed)_$(mode)_$(construction_tag)_$(domain_tag).bson"
    return joinpath(output_dir, fname)
end

"""
    _channel_sweep_sidecar_path(output_dir, n, beta, seed, eps, filter_kind,
                                construction_tag, domain_tag;
                                beta_phys=nothing) -> String

Return a channel-sweep sidecar path keyed by temperature, tolerance, filter,
construction, and domain.
"""
function _channel_sweep_sidecar_path(output_dir::AbstractString, n::Integer,
                                     beta::Real, seed::Integer, eps::Real,
                                     filter_kind::Symbol,
                                     construction_tag::AbstractString,
                                     domain_tag::AbstractString;
                                     beta_phys::Union{Nothing, Real} = nothing)
    b_value = beta_phys === nothing ? float(beta) : float(beta_phys)
    b_tag   = beta_phys === nothing ? "beta"      : "betaphys"
    beta_str = let s = @sprintf("%.6f", b_value)
        s = rstrip(s, '0'); s = rstrip(s, '.')
        isempty(s) ? "0" : s
    end
    eps_str = @sprintf("%.0e", float(eps))            # e.g. "1e-03"
    fname = "channel_n$(n)_$(b_tag)$(beta_str)_seed$(seed)_eps$(eps_str)_$(filter_kind)_$(construction_tag)_$(domain_tag).bson"
    return joinpath(output_dir, fname)
end

# Keep the resource total and its interpretation coupled. Only an achieved
# integer-step crossing is a physical total-to-mixing cost; all no-crossing
# outcomes retain per-step diagnostics but expose no finite total.
function _channel_sweep_cost_fields(
    physical_channel::Bool,
    tau_mix_source::Symbol,
    sim_budget,
)
    if physical_channel && tau_mix_source === :extrapolated
        return (
            n_steps_total=sim_budget.n_steps,
            total_ham_sim_time=sim_budget.total_time,
            cost_interpretation=:physical_mixing_time,
        )
    end

    cost_interpretation = if !physical_channel
        sim_budget.cost_interpretation
    elseif tau_mix_source === :floor
        :unavailable_asymptotic_floor
    elseif tau_mix_source === :certified_no_crossing
        :unavailable_certified_no_crossing
    elseif tau_mix_source === :horizon_exhausted
        :unavailable_horizon_exhausted
    else
        :unavailable_no_certified_crossing
    end
    return (
        n_steps_total=0,
        total_ham_sim_time=NaN,
        cost_interpretation,
    )
end

"""
    sweep_mixing_times(n_values, beta_values; kwargs...) -> Vector{NamedTuple}

Sweep matrix-free Lindbladian mixing estimates over qubit count, temperature,
and seed, with optional threading and resumable BSON sidecars.

# Arguments
- `n_values`: qubit counts with matching Hamiltonian BSON files.
- `beta_values`: algorithmic inverse temperatures. Leave empty when using
  `beta_phys_values`.

# Keywords
- `beta_phys_values`: physical inverse temperatures; converted per Hamiltonian.
- `construction`, `domain`, `filter`: select KMS or DLL construction. DLL is
  restricted to `BohrDomain` and rebuilds its filter for each temperature.
- `mode`: `:L` for density-matrix flow or `:K` for discriminant flow.
- `method`: `:ode` for Krylov exponentiation or `:krylov` for a spectral
  reconstruction; the latter supports only `mode=:L`.
- `seeds`, `init_state`: cell seeds and initial-state family.
- `target_epsilon`, `t_max_factor`, `t_grid_length`: mixing-time grid controls.
- `krylovdim`, `spectral_krylovdim`, `tol`: numerical solver controls.
- `param_table_bson`: optional per-cell Energy-domain register table.
- `output_dir`, `skip_existing`: sidecar persistence controls.
- `use_threads`: parallelise cells; each cell owns its workspace. Keep BLAS at
  one thread when using Julia threads.

# Returns

A vector of named tuples containing cell identifiers, temperature metadata,
gap and mixing estimates, fit diagnostics, matvec counts, and wall time.

`init_state=:maximally_mixed` can miss symmetry-odd slow modes. Use a
symmetry-breaking initial state or request an independent true-gap solve when
the generator preserves a discrete symmetry.
"""
function sweep_mixing_times(
    n_values::AbstractVector{<:Integer},
    beta_values::AbstractVector{<:Real} = Float64[];
    beta_phys_values::Union{Nothing, AbstractVector{<:Real}} = nothing,
    construction::AbstractConstruction = KMS(),
    domain::AbstractDomain = BohrDomain(),
    filter::Union{Nothing, AbstractFilter} = nothing,
    mode::Symbol = :L,
    method::Symbol = :ode,
    seeds::AbstractVector{<:Integer} = [42],
    init_state::Symbol = :maximally_mixed,
    a::Real = 0.0,
    s::Real = 0.25,
    target_epsilon::Real = 1e-3,
    t_max_factor::Union{Real, Symbol} = :auto,
    t_grid_length::Integer = 81,
    krylovdim::Integer = 30,
    spectral_krylovdim::Integer = 60,
    tol::Real = 1e-10,
    output_dir::Union{Nothing, AbstractString} = nothing,
    hamiltonian_dir::AbstractString = _PACKAGED_HAMILTONIAN_DIR,
    hamiltonian_filename::Function = n -> "heis_xxx_disordered_periodic_n$(n)_seed46.bson",
    use_threads::Bool = true,
    skip_existing::Bool = true,
    # An optional table supplies per-cell Energy-domain registers. Its measured
    # range ends at n=6; validate cross-domain error before extrapolating it.
    param_table_bson::Union{Nothing, AbstractString} = nothing,
    filter_kind::Symbol = :smooth_metro,
)::Vector{NamedTuple}
    mode in (:L, :K) || throw(ArgumentError("mode must be :L or :K (got :$mode)"))
    method in (:ode, :krylov) || throw(ArgumentError(
        "method must be :ode or :krylov (got :$method)"))
    method === :krylov && mode === :K && throw(ArgumentError(
        "method=:krylov requires mode=:L (the K-mode discriminant flow needs the K matvec, " *
        "not the L spectral expansion). Run mode=:L if τ_mix is the goal."))
    init_state in (:maximally_mixed, :random_pure, :thermal_perturbed) || throw(
        ArgumentError("init_state ∈ {:maximally_mixed, :random_pure, :thermal_perturbed} (got :$init_state)"))
    domain isa Union{BohrDomain, EnergyDomain} || throw(
        ArgumentError("sweep_mixing_times supports BohrDomain or EnergyDomain only (got $(typeof(domain)))"))
    domain isa EnergyDomain && construction isa DLL && throw(
        ArgumentError("DLL construction is not supported in EnergyDomain (out of scope for DLL-2)"))

    # Exactly one temperature grid is supplied; physical values are converted
    # using each Hamiltonian's rescaling factor and tagged separately on disk.
    mode_phys = beta_phys_values !== nothing
    if mode_phys && !isempty(beta_values)
        throw(ArgumentError(
            "sweep_mixing_times: pass either positional `beta_values` (β_alg list) " *
            "OR kwarg `beta_phys_values` (β_phys list), not both."))
    end
    if !mode_phys && isempty(beta_values)
        throw(ArgumentError(
            "sweep_mixing_times: must specify positional `beta_values` (β_alg list) " *
            "or kwarg `beta_phys_values` (β_phys list)."))
    end

    # Extend tighter-tolerance trajectories far enough to resolve the fit offset.
    # Math: $t_max = max(5, 1.5 log_10(1 / epsilon)) / gap$.
    t_max_factor_resolved::Float64 = if t_max_factor isa Symbol
        t_max_factor === :auto || throw(ArgumentError(
            "t_max_factor must be :auto or a positive number (got :$t_max_factor)"))
        max(5.0, 1.5 * log10(1.0 / target_epsilon))
    else
        Float64(t_max_factor) > 0 || throw(ArgumentError(
            "t_max_factor must be positive (got $t_max_factor)"))
        Float64(t_max_factor)
    end

    construction_tag = construction isa DLL ? "DLL" : "KMS"
    domain_tag = domain isa EnergyDomain ? "Energy" : "Bohr"
    filter_name = if filter === nothing
        "default"
    elseif filter isa DLLGaussianFilter
        "DLLGaussian"
    elseif filter isa DLLMetropolisFilter
        "DLLMetropolis"
    else
        string(typeof(filter).name.name)
    end

    # Load the ideal-Lindbladian parameter table once when threading is requested.
    # Only consumed by the CKG / EnergyDomain branch below; ignored otherwise.
    use_param_table = (param_table_bson !== nothing
                        && construction isa KMS
                        && domain isa EnergyDomain)
    table_rows = use_param_table ? _load_channel_param_table(param_table_bson) : NamedTuple[]
    if use_param_table
        filter_kind in (:gaussian, :smooth_metro, :kinky_metro) || throw(ArgumentError(
            "param_table_bson requires filter_kind ∈ {:gaussian, :smooth_metro, :kinky_metro} (got :$filter_kind)"))
    end

    if output_dir !== nothing && !isdir(output_dir)
        mkpath(output_dir)
    end

    # Flat product of (n, β_unit, seed). `β_unit` is β_phys in mode_phys, β_alg
    # otherwise — the runner resolves the (β_phys, β_alg) pair per-cell once
    # `ham.rescaling_factor` is in hand. Materialised so @threads can index it.
    β_unit_values = mode_phys ? Float64.(beta_phys_values) : Float64.(beta_values)
    points = [(Int(n), Float64(β), Int(s)) for n in n_values
                                            for β in β_unit_values
                                            for s in seeds]
    n_points = length(points)
    results = Vector{NamedTuple}(undef, n_points)
    skipped = falses(n_points)

    # Pre-pass: load and stamp sidecars for skip_existing. Done serially —
    # strictly before the threaded launcher — to dodge concurrent BSON.load
    # contention and any race with per-thread sidecar writes downstream.
    if output_dir !== nothing && skip_existing
        for i in 1:n_points
            n_i, β_unit_i, seed_i = points[i]
            # Sidecars are keyed by the supplied temperature convention.
            sidecar = mode_phys ?
                _sweep_sidecar_path(output_dir, n_i, 0.0, seed_i,
                                    mode, construction_tag, domain_tag;
                                    beta_phys = β_unit_i) :
                _sweep_sidecar_path(output_dir, n_i, β_unit_i, seed_i,
                                    mode, construction_tag, domain_tag)
            if isfile(sidecar)
                try
                    d_loaded = BSON.load(sidecar, @__MODULE__)
                    results[i] = NamedTuple(d_loaded[:result])
                    skipped[i] = true
                catch err
                    @warn "skip_existing: failed to load sidecar; will recompute" sidecar err
                end
            end
        end
    end

    runner = function (i)
        skipped[i] && return
        n_i, β_unit_i, seed_i = points[i]
        ham_path = joinpath(hamiltonian_dir, hamiltonian_filename(n_i))
        if !isfile(ham_path)
            @warn "Hamiltonian file missing; skipping point" n=n_i β_unit=β_unit_i ham_path
            results[i] = (n=n_i, beta=β_unit_i, seed=seed_i, init_state=init_state,
                          mode=mode, method=method, construction=construction_tag,
                          domain=domain_tag,
                          filter_name=filter_name,
                          target_epsilon=float(target_epsilon),
                          filter_kind=filter_kind,
                          beta_phys = mode_phys ? β_unit_i : NaN,
                          beta_alg  = mode_phys ? NaN      : β_unit_i,
                          rescaling_factor=NaN,
                          r_D=0, w0_D=NaN, t0_D=NaN,
                          gap_est=NaN, t_max=NaN,
                          t_max_factor=t_max_factor_resolved,
                          tau_mix_bound=NaN, n_grid=0,
                          total_matvecs=0, all_converged=false,
                          fitted_gap=NaN, mixing_time=NaN,
                          mixing_time_source=:nan, r_squared=NaN,
                          converged_fit=false, wall_time=0.0)
            return
        end

        t0_run = time()
        # Derive the physical/algorithmic pair before constructing `HamHam`.
        ham_raw_nt = _parse_hamiltonian_bson(ham_path)
        rescale = ham_raw_nt.rescaling_factor
        β_phys_i, β_i = mode_phys ?
            (β_unit_i, β_unit_i * rescale) :
            (β_unit_i / rescale, β_unit_i)
        ham = HamHam(ham_raw_nt, β_i)
        jumps = _jumps_in_basis(n_i, ham.eigvecs)

        # Per-(n, β) filter: DLL filter must match β; rebuild from the user's
        # filter type tag if construction is DLL. CKG passes filter through
        # unchanged (typically `nothing`).
        local_filter = if construction isa DLL
            if filter isa DLLGaussianFilter
                DLLGaussianFilter(β_i)
            elseif filter isa DLLMetropolisFilter
                DLLMetropolisFilter(β_i)
            else
                throw(ArgumentError(
                    "DLL sweeps require filter::Union{DLLGaussianFilter, DLLMetropolisFilter}"))
            end
        else
            filter
        end

        # Use conservative defaults unless an Energy-domain table row is supplied.
        cell_r_D, cell_w0_D, cell_t0_D = 12, 0.05, 2π / (2^12 * 0.05)
        cell_a, cell_s = float(a), float(s)
        cell_with_lc = true
        cell_gauss_params = (nothing, nothing)
        if use_param_table
            row = _lookup_channel_params(table_rows, n_i, β_i, target_epsilon, filter_kind)
            cell_r_D, cell_w0_D, cell_t0_D = row.r_D, row.w0_D, row.t0_D
            cell_a, cell_s = row.a, row.s
            cell_with_lc = row.with_linear_combination
            cell_gauss_params = row.with_linear_combination ? (nothing, nothing) :
                (row.gaussian_omega, row.gaussian_sigma)
        end

        config = Config(
            sim = Lindbladian(),
            domain = domain,
            construction = construction,
            num_qubits = n_i,
            with_linear_combination = cell_with_lc,
            beta = β_i,
            beta_phys = β_phys_i,
            sigma = 1.0 / β_i,
            a = cell_a,
            s = cell_s,
            gaussian_parameters = cell_gauss_params,
            num_energy_bits = cell_r_D,
            w0 = cell_w0_D,
            t0 = cell_t0_D,
            num_trotter_steps_per_t0 = 10,
            filter = local_filter,
        )

        d = size(ham.data, 1)
        rho_0 = _make_init_state(init_state, d,
                                  Matrix{ComplexF64}(ham.gibbs), seed_i)

        # `:krylov` estimates the gap and mixing time from one spectral
        # decomposition; `:ode` integrates a grid and fits its tail.

        result = if method === :krylov
            # Math: $rho(t) = rho_inf + sum_i c_i exp(lambda_i t) R_i$.
            # The mixing-time solver widens its initial time bracket as needed.
            t_max_seed = t_max_factor_resolved * float(β_i)
            t_grid = collect(range(0.0, t_max_seed, length=t_grid_length))

            predict_res = predict_lindbladian_trajectory(
                config, ham, jumps, rho_0, t_grid;
                krylovdim=spectral_krylovdim, tol=tol,
            )

            # gap_est and t_max are reconstructed from the predictor's own
            # eigendecomposition — `predict_res.spectral_gap` is the smallest
            # |Re(λ_i)| over non-steady modes (already excludes λ_1 ≈ 0).
            gap_est = predict_res.spectral_gap > 0 ?
                       predict_res.spectral_gap : 1.0 / β_i
            t_max = t_max_factor_resolved / max(gap_est, 1e-12)

            res_eig = eigenmode_mixing_time(
                predict_res.eigenvalues, predict_res.c, predict_res.R_modes,
                predict_res.rho_inf, predict_res.sigma_beta,
                target_epsilon;
                t_upper = max(t_max, t_max_seed),
            )
            mixing_time        = res_eig.mixing_time
            mixing_time_source = res_eig.source

            wall = time() - t0_run
            (
                n                   = n_i,
                beta                = β_i,
                beta_alg            = β_i,
                beta_phys           = β_phys_i,
                rescaling_factor    = rescale,
                seed                = seed_i,
                init_state          = init_state,
                mode                = mode,
                method              = method,
                construction        = construction_tag,
                domain              = domain_tag,
                filter_name         = filter_name,
                target_epsilon      = float(target_epsilon),
                filter_kind         = filter_kind,
                r_D                 = cell_r_D,
                w0_D                = cell_w0_D,
                t0_D                = cell_t0_D,
                gap_est             = gap_est,
                t_max               = t_max,
                t_max_factor        = t_max_factor_resolved,
                tau_mix_bound       = log(size(ham.data, 1) / float(target_epsilon)) /
                                        max(gap_est, 1e-12),
                n_grid              = t_grid_length,
                total_matvecs       = predict_res.total_matvecs,
                all_converged       = predict_res.all_converged,
                mixing_time         = mixing_time,
                mixing_time_source  = mixing_time_source,
                floor_distance      = res_eig.floor_distance,
                wall_time           = wall,
            )
        else  # method === :ode
            # Estimate the gap matrix-free; use 1/beta only if Krylov fails.
            gap_est = try
                krylov_result = krylov_spectral_gap(config, ham, jumps;
                    krylovdim = 30, howmany = 2, tol = 1e-8)
                krylov_result.spectral_gap > 0 ?
                    krylov_result.spectral_gap : 1.0 / β_i
            catch err
                @warn "krylov_spectral_gap failed; falling back to 1/β" n=n_i β=β_i err
                1.0 / β_i
            end
            t_max = t_max_factor_resolved / max(gap_est, 1e-12)
            t_grid = collect(range(0.0, t_max, length=t_grid_length))

            res_int = integrate_to_gibbs(config, ham, jumps, rho_0, t_grid;
                                          mode=mode, krylovdim=krylovdim, tol=tol)
            est = estimate_mixing_time(res_int; model=:biexp,
                                        target_epsilon=target_epsilon,
                                        extrapolate=true)

            # If tail extrapolation fails after the sampled trajectory crossed
            # the target, report the observed crossing explicitly.
            mixing_time_source = if isfinite(est.mixing_time)
                :extrapolated
            elseif est.mixing_time_actual !== nothing && isfinite(est.mixing_time_actual)
                :observed
            else
                :nan
            end
            mixing_time = if mixing_time_source === :extrapolated
                est.mixing_time
            elseif mixing_time_source === :observed
                est.mixing_time_actual::Float64
            else
                NaN
            end

            wall = time() - t0_run
            (
                n                   = n_i,
                beta                = β_i,
                beta_alg            = β_i,
                beta_phys           = β_phys_i,
                rescaling_factor    = rescale,
                seed                = seed_i,
                init_state          = init_state,
                mode                = mode,
                method              = method,
                construction        = construction_tag,
                domain              = domain_tag,
                filter_name         = filter_name,
                target_epsilon      = float(target_epsilon),
                filter_kind         = filter_kind,
                r_D                 = cell_r_D,
                w0_D                = cell_w0_D,
                t0_D                = cell_t0_D,
                gap_est             = gap_est,
                t_max               = t_max,
                t_max_factor        = t_max_factor_resolved,
                tau_mix_bound       = log(size(ham.data, 1) / float(target_epsilon)) /
                                        max(gap_est, 1e-12),
                n_grid              = t_grid_length,
                total_matvecs       = res_int.total_matvecs,
                all_converged       = res_int.all_converged,
                fitted_gap          = est.fitted_gap,
                mixing_time         = mixing_time,
                mixing_time_source  = mixing_time_source,
                r_squared           = est.r_squared,
                converged_fit       = est.converged,
                wall_time           = wall,
            )
        end
        results[i] = result

        if output_dir !== nothing
            sidecar = mode_phys ?
                _sweep_sidecar_path(output_dir, n_i, β_i, seed_i,
                                    mode, construction_tag, domain_tag;
                                    beta_phys = β_phys_i) :
                _sweep_sidecar_path(output_dir, n_i, β_i, seed_i,
                                    mode, construction_tag, domain_tag)
            try
                BSON.bson(sidecar, Dict(:result => Dict(pairs(result))))
            catch err
                @warn "Sidecar write failed (continuing)" sidecar err
            end
        end
        return
    end

    if use_threads
        Threads.@threads for i in 1:n_points
            runner(i)
        end
    else
        for i in 1:n_points
            runner(i)
        end
    end

    return results
end

"""
    _load_channel_param_table(path) -> Vector{NamedTuple}

Load the parameter-table BSON produced by `scripts/numerics_param_table.jl`.
Returns the `:rows` field as a `Vector{NamedTuple}`; throws if the file
is missing or malformed.
"""
function _load_channel_param_table(path::AbstractString)
    isfile(path) || throw(ArgumentError(
        "channel parameter table not found at $path — run `julia --project scripts/numerics_param_table.jl` first"))
    raw = BSON.load(path, @__MODULE__)
    haskey(raw, :rows) || throw(ArgumentError(
        "channel parameter table $path missing :rows entry"))
    return Vector{NamedTuple}(raw[:rows])
end

"""
    _lookup_channel_params(rows, n, β, ε, filter_kind) -> NamedTuple

Find the unique row of the parameter table matching `(n, β, ε, filter_kind)`.
Throws with a clear message if no match (the harness can then skip the cell
or fail loudly depending on the caller's preference).
"""
function _lookup_channel_params(rows::AbstractVector{<:NamedTuple},
                                n::Integer, β::Real, ε::Real, filter_kind::Symbol)
    for r in rows
        r.n == n && r.beta ≈ β && r.eps ≈ ε && r.filter === filter_kind && return r
    end
    throw(ArgumentError(
        "no parameter-table row for (n=$n, β=$β, ε=$ε, filter=$filter_kind). " *
        "Either add the cell to scripts/numerics_param_table.jl or pick a covered cell."))
end

"""
    _build_channel_config(row, n, β, domain, construction; mixing_time_target)
        -> Config{Thermalize, ...}

Construct a `Config{Thermalize, <:Union{TimeDomain, TrotterDomain}, KMS}` from a
parameter-table row. The `mixing_time` field is a placeholder
(`predict_channel_trajectory` only uses `k_grid`); pass any positive number.
"""
function _build_channel_config(row::NamedTuple, n::Integer, β::Real,
                                domain::AbstractDomain,
                                construction::AbstractConstruction;
                                mixing_time_target::Real = 5.0,
                                beta_phys::Union{Nothing, Real} = nothing)
    return Config(
        sim = Thermalize(),
        domain = domain,
        construction = construction,
        num_qubits = n,
        beta = β,
        beta_phys = beta_phys,
        sigma = row.sigma,
        with_linear_combination = row.with_linear_combination,
        a = row.a, s = row.s,
        gaussian_parameters = row.with_linear_combination ? (nothing, nothing) :
            (row.gaussian_omega, row.gaussian_sigma),
        eta = row.eta,
        num_energy_bits_D = row.r_D, t0_D = row.t0_D, w0_D = row.w0_D,
        num_energy_bits_b_minus = row.r_bm, t0_b_minus = row.t0_bm, w0_b_minus = row.w0_bm,
        num_energy_bits_b_plus  = row.r_bp, t0_b_plus  = row.t0_bp,  w0_b_plus  = row.w0_bp,
        num_trotter_steps_per_t0 = row.M_D,
        delta = row.delta,
        mixing_time = mixing_time_target,
        with_gqsp = row.with_gqsp, gqsp_degree = row.gqsp_degree,
        jump_selection = row.jump_selection,
    )
end

"""
    sweep_channel_mixing(n_values, beta_values; kwargs...) -> Vector{NamedTuple}

Sweep channel mixing estimates and Hamiltonian-simulation costs over qubit
count, temperature, target distance, filter, and seed.

Per-cell registers and channel parameters come from `param_table_bson`.
Optional BSON sidecars make the sweep resumable.

# Arguments
- `n_values`: qubit counts with matching Hamiltonian BSON files.
- `beta_values`: algorithmic inverse temperatures. Leave empty when using
  `beta_phys_values`.

# Keywords
- `beta_phys_values`: physical inverse temperatures; converted per Hamiltonian.
- `target_epsilons`, `filter_kinds`, `domain`, `construction`: channel grid.
- `param_table_bson`, `family`, `hamiltonian_dir`: input selection.
- `seeds`, `init_state`: cell seeds and initial-state family.
- `krylovdim`, `k_grid_max_log`, `k_grid_length`: predictor controls.
- `mixing_step_horizon`: hard cap for the chronological integer-step crossing
  search. Exhaustion is reported explicitly; it is not converted to a gap
  proxy.
- `output_dir`, `skip_existing`: sidecar persistence controls.

# Returns
A vector of named tuples containing cell and parameter metadata, mixing source,
gap and floor diagnostics, Krylov work, and simulation-time estimates.

`tau_mix_source` is `:extrapolated`, `:floor`, `:certified_no_crossing`,
`:horizon_exhausted`, `:nan`, or `:nonphysical_surrogate`; `:extrapolated` is
a legacy tag for the exact integer-step spectral crossing. Only that branch
reports a total-to-mixing resource cost. The current `with_gqsp=true` path
applies an unscaled polynomial surrogate, so its rows expose raw trace drift
and per-step costs but report no physical gap, mixing time, or total-to-mixing
cost. Those per-step numbers are labelled
`:formal_unscaled_gqsp_surrogate`, since no contractive rescaling or
complementary-polynomial circuit has been synthesized.

`init_state=:maximally_mixed` can miss symmetry-odd slow modes. Use a
symmetry-breaking state or an independent true-gap solve for symmetric models.
"""
function sweep_channel_mixing(
    n_values::AbstractVector{<:Integer},
    beta_values::AbstractVector{<:Real} = Float64[];
    beta_phys_values::Union{Nothing, AbstractVector{<:Real}} = nothing,
    target_epsilons::AbstractVector{<:Real} = [1e-3],
    filter_kinds::AbstractVector{Symbol} = [:smooth_metro],
    domain::AbstractDomain = TrotterDomain(),
    construction::AbstractConstruction = KMS(),
    param_table_bson::AbstractString = _package_data_path("channel_param_table.bson"),
    family::Symbol = :xxx_disordered,
    seeds::AbstractVector{<:Integer} = [42],
    init_state::Symbol = :maximally_mixed,
    krylovdim::Integer = 30,
    k_grid_max_log::Real = 5,
    k_grid_length::Integer = 50,
    mixing_step_horizon::Integer = 100_000,
    output_dir::Union{Nothing, AbstractString} = nothing,
    skip_existing::Bool = true,
    hamiltonian_dir::AbstractString = _PACKAGED_HAMILTONIAN_DIR,
)::Vector{NamedTuple}
    domain isa Union{TimeDomain, TrotterDomain} || throw(ArgumentError(
        "sweep_channel_mixing supports TimeDomain or TrotterDomain (got $(typeof(domain)))"))
    init_state in (:maximally_mixed, :random_pure, :thermal_perturbed) || throw(
        ArgumentError("init_state ∈ {:maximally_mixed, :random_pure, :thermal_perturbed} (got :$init_state)"))
    all(f -> f in (:gaussian, :smooth_metro, :kinky_metro), filter_kinds) || throw(ArgumentError(
        "filter_kinds must be ⊆ {:gaussian, :smooth_metro, :kinky_metro} (got $filter_kinds)"))
    Float64(k_grid_max_log) > 0 || throw(ArgumentError(
        "k_grid_max_log must be > 0 (got $k_grid_max_log)"))
    k_grid_length >= 10 || throw(ArgumentError(
        "k_grid_length must be ≥ 10 (got $k_grid_length)"))
    0 <= mixing_step_horizon < typemax(Int) || throw(ArgumentError(
        "mixing_step_horizon must be a nonnegative integer below typemax(Int) " *
        "(got $mixing_step_horizon)"))

    # Exactly one temperature grid is supplied; physical values are converted
    # per cell, while parameter-table lookup always uses `beta_alg`.
    mode_phys = beta_phys_values !== nothing
    if mode_phys && !isempty(beta_values)
        throw(ArgumentError(
            "sweep_channel_mixing: pass either positional `beta_values` (β_alg list) " *
            "OR kwarg `beta_phys_values` (β_phys list), not both."))
    end
    if !mode_phys && isempty(beta_values)
        throw(ArgumentError(
            "sweep_channel_mixing: must specify positional `beta_values` (β_alg list) " *
            "or kwarg `beta_phys_values` (β_phys list)."))
    end

    construction_tag = construction isa KMS ? "KMS" : construction isa GNS ? "GNS" : "DLL"
    domain_tag = domain isa TrotterDomain ? "Trotter" : "Time"
    family_str = string(family)
    ham_filename = (n) -> "heis_$(family_str)_periodic_n$(n)_seed46.bson"

    output_dir !== nothing && !isdir(output_dir) && mkpath(output_dir)

    rows_table = _load_channel_param_table(param_table_bson)

    # Flat product of (n, β_unit, ε, filter, seed). β_unit is β_phys in
    # mode_phys, β_alg otherwise; the runner resolves the (β_phys, β_alg)
    # pair per-cell via `ham.rescaling_factor`.
    β_unit_values = mode_phys ? Float64.(beta_phys_values) : Float64.(beta_values)
    points = [(Int(n), Float64(β), Float64(ε), f, Int(s))
              for n in n_values for β in β_unit_values
              for ε in target_epsilons for f in filter_kinds for s in seeds]
    n_points = length(points)
    results = Vector{NamedTuple}(undef, n_points)
    skipped = falses(n_points)

    # Pre-pass: load existing sidecars under skip_existing, before the main loop.
    if output_dir !== nothing && skip_existing
        for i in 1:n_points
            n_i, β_unit_i, ε_i, f_i, seed_i = points[i]
            sidecar = mode_phys ?
                _channel_sweep_sidecar_path(output_dir, n_i, 0.0, seed_i, ε_i,
                                            f_i, construction_tag, domain_tag;
                                            beta_phys = β_unit_i) :
                _channel_sweep_sidecar_path(output_dir, n_i, β_unit_i, seed_i, ε_i,
                                            f_i, construction_tag, domain_tag)
            if isfile(sidecar)
                try
                    d_loaded = BSON.load(sidecar, @__MODULE__)
                    results[i] = NamedTuple(d_loaded[:result])
                    skipped[i] = true
                catch err
                    @warn "skip_existing: failed to load channel sidecar; will recompute" sidecar err
                end
            end
        end
    end

    k_max = round(Int, exp10(Float64(k_grid_max_log)))

    for i in 1:n_points
        skipped[i] && continue
        n_i, β_unit_i, ε_i, f_i, seed_i = points[i]
        ham_path = joinpath(hamiltonian_dir, ham_filename(n_i))
        if !isfile(ham_path)
            @warn "Hamiltonian file missing; skipping channel cell" n=n_i β_unit=β_unit_i ham_path
            continue
        end

        local row, ham, jumps, cfg, trotter, rho_0, rho_init, predict_res, res_eig, sim_budget
        local β_phys_i, β_i, rescale
        t0_run = time()
        try
            ham_raw_nt = _parse_hamiltonian_bson(ham_path)
            rescale = ham_raw_nt.rescaling_factor
            β_phys_i, β_i = mode_phys ?
                (β_unit_i, β_unit_i * rescale) :
                (β_unit_i / rescale, β_unit_i)
            row = _lookup_channel_params(rows_table, n_i, β_i, ε_i, f_i)
            ham = HamHam(ham_raw_nt, β_i)
            cfg = _build_channel_config(row, n_i, β_i, domain, construction;
                                        beta_phys = β_phys_i)

            # KMS coherent construction uses independent per-leg Strang caches.
            trotter = domain isa TrotterDomain ? make_trotter_for_config(ham, cfg) : nothing

            # Build jump set in the right basis. TimeDomain uses Hamiltonian
            # eigenbasis; TrotterDomain uses the Trotter eigenbasis (Kraus path).
            jumps = if domain isa TrotterDomain
                _jumps_in_basis(n_i, trotter.eigvecs)
            else
                _jumps_in_basis(n_i, ham.eigvecs)
            end

            # Initial state in the basis the predictor expects.
            d = size(ham.data, 1)
            rho_0 = _make_init_state(init_state, d, Matrix{ComplexF64}(ham.gibbs), seed_i)
            rho_init = if domain isa TrotterDomain
                Matrix{ComplexF64}(trotter.eigvecs' * rho_0 * trotter.eigvecs)
            else
                Matrix{ComplexF64}(rho_0)
            end

            k_grid = unique(round.(Int, exp10.(range(0, Float64(k_grid_max_log),
                                                       length=k_grid_length))))

            predict_res = predict_channel_trajectory(cfg, ham, jumps, rho_init, k_grid;
                krylovdim=krylovdim, trotter=trotter)

            d_dim = size(ham.data, 1)
            gap_ch = predict_res.spectral_gap
            if predict_res.physical_channel
                # Integer-step search horizon: take the larger of k_max*delta
                # and a generous gap-based estimate `5 log(d/epsilon)/gap`.
                t_upper_ch = if isfinite(gap_ch) && gap_ch > 0
                    max(predict_res.t[end], 5.0 * log(d_dim / ε_i) / gap_ch)
                else
                    predict_res.t[end]
                end
                res_eig = eigenmode_mixing_time(
                    predict_res, ε_i;
                    t_upper=t_upper_ch,
                    max_steps=mixing_step_horizon)
                if res_eig.source === :extrapolated
                    # Pass the exact integer count separately so floating
                    # division cannot add a spurious channel step.
                    sim_budget = compute_simulation_time(
                        cfg, ham, res_eig.mixing_time;
                        n_steps=res_eig.mixing_steps)
                else
                    # Preserve per-step diagnostics, but never price a floor,
                    # failed certificate, or exhausted finite horizon as an
                    # achievable total-to-mixing resource cost.
                    diagnostic_horizon = max(predict_res.t[end], cfg.delta)
                    sim_budget = compute_simulation_time(
                        cfg, ham, diagnostic_horizon)
                end
            else
                # The unscaled polynomial surrogate is not a quantum channel:
                # retain its raw trajectory and per-step resource diagnostics,
                # but do not manufacture a fixed point, gap, mixing time, or
                # total-to-mixing cost.
                res_eig = (
                    mixing_time=NaN,
                    mixing_steps=nothing,
                    gap=NaN,
                    floor_distance=NaN,
                    source=:nonphysical_surrogate,
                    crossing_kind=:not_applicable,
                    search_horizon=0,
                    horizon_capped=false,
                    tail_bound=NaN,
                    n_evals=0,
                )
                diagnostic_horizon = max(predict_res.t[end], cfg.delta)
                sim_budget = compute_simulation_time(
                    cfg, ham, diagnostic_horizon)
            end
        catch err
            @warn "channel cell failed; recording NaN row" n=n_i β=β_i eps=ε_i filter=f_i err
            # `β_i`, `β_phys_i`, `rescale` may be undefined if parsing failed —
            # fall back to NaN / β_unit_i so the result row is still
            # serialisable.
            β_alg_recorded  = @isdefined(β_i)        ? β_i        : (mode_phys ? NaN        : β_unit_i)
            β_phys_recorded = @isdefined(β_phys_i)   ? β_phys_i   : (mode_phys ? β_unit_i  : NaN)
            rescale_recorded = @isdefined(rescale)   ? rescale    : NaN
            results[i] = (
                n=n_i, beta=β_alg_recorded, seed=seed_i, eps=ε_i, filter=f_i,
                beta_alg=β_alg_recorded, beta_phys=β_phys_recorded,
                rescaling_factor=rescale_recorded,
                family=family_str, construction=construction_tag, domain=domain_tag,
                r_D=NaN, w0_D=NaN, t0_D=NaN, r_bm=NaN, w0_bm=NaN, t0_bm=NaN,
                r_bp=NaN, w0_bp=NaN, t0_bp=NaN,
                M_D=NaN, M_bm=NaN, M_bp=NaN, delta=NaN, eta=NaN,
                with_gqsp=false, gqsp_degree=0,
                tau_mix=NaN, tau_mix_source=:nan, lambda_gap_channel=NaN,
                floor_distance=NaN,
                crossing_kind=:unknown, mixing_steps=nothing,
                mixing_search_horizon=0, mixing_horizon_capped=false,
                mixing_tail_bound=NaN, mixing_n_exact_evals=0,
                n_steps_to_target=0, k_max=k_max, t_max=NaN,
                achieved_dist_at_kmax=NaN,
                max_abs_trace_drift=NaN,
                channel_representation=:unknown,
                physical_channel=false,
                interpretation_status=:failed,
                cost_interpretation=:unavailable,
                total_matvecs=0, all_converged_predict=false,
                oft_time_per_step=NaN, b_per_be_per_step=NaN, b_time_per_step=NaN,
                per_step_time=NaN, n_steps_total=0, total_ham_sim_time=NaN,
                wall_time_seconds=time() - t0_run,
                init_state=init_state, family_tag=family_str,
            )
            continue
        end

        # Resolve τ_mix from the eigenmode helper output. Only
        # :extrapolated represents an achieved physical integer-step crossing.
        # Certified no-crossing branches report Inf; exhausted/degenerate
        # searches report NaN. None receives a total-to-mixing resource cost.
        tau_mix_source = res_eig.source
        if tau_mix_source === :nonphysical_surrogate
            @warn "unscaled GQSP polynomial surrogate is not a physical channel; reporting raw trace diagnostics without tau_mix, gap, or total cost" n=n_i β=β_i ε=ε_i filter=f_i max_abs_trace_drift=predict_res.max_abs_trace_drift
        elseif tau_mix_source !== :extrapolated
            @warn "channel cell has no certified mixing-time crossing within its integer-step search; no total-to-mixing cost is reported" n=n_i β=β_i ε=ε_i filter=f_i source=tau_mix_source floor_distance=res_eig.floor_distance search_horizon=res_eig.search_horizon horizon_capped=res_eig.horizon_capped δ_used=predict_res.delta_used
        end
        tau_mix = if tau_mix_source === :extrapolated
            res_eig.mixing_time
        elseif tau_mix_source in (:floor, :certified_no_crossing)
            Inf
        else
            NaN
        end
        # Exact integer crossing from the channel spectral reconstruction.
        n_steps_to_target = res_eig.mixing_steps === nothing ?
            0 : Int(res_eig.mixing_steps)
        cost_fields = _channel_sweep_cost_fields(
            predict_res.physical_channel, tau_mix_source, sim_budget)

        wall = time() - t0_run
        result = (
            n                       = n_i,
            beta                    = β_i,
            beta_alg                = β_i,
            beta_phys               = β_phys_i,
            rescaling_factor        = rescale,
            seed                    = seed_i,
            eps                     = ε_i,
            filter                  = f_i,
            family                  = family_str,
            construction            = construction_tag,
            domain                  = domain_tag,
            # parameters (echoed from the param-table row)
            r_D = row.r_D,  w0_D = row.w0_D,  t0_D = row.t0_D,
            r_bm = row.r_bm, w0_bm = row.w0_bm, t0_bm = row.t0_bm,
            r_bp = row.r_bp, w0_bp = row.w0_bp, t0_bp = row.t0_bp,
            M_D = row.M_D, M_bm = row.M_bm, M_bp = row.M_bp,
            delta = row.delta, eta = row.eta,
            with_gqsp = row.with_gqsp, gqsp_degree = row.gqsp_degree,
            # mixing diagnostics
            tau_mix                 = tau_mix,
            tau_mix_source          = tau_mix_source,
            lambda_gap_channel      = predict_res.spectral_gap,
            floor_distance          = res_eig.floor_distance,
            crossing_kind           = res_eig.crossing_kind,
            mixing_steps            = res_eig.mixing_steps,
            mixing_search_horizon   = res_eig.search_horizon,
            mixing_horizon_capped   = res_eig.horizon_capped,
            mixing_tail_bound       = res_eig.tail_bound,
            mixing_n_exact_evals    = res_eig.n_evals,
            n_steps_to_target       = n_steps_to_target,
            k_max                   = k_max,
            t_max                   = predict_res.t[end],
            achieved_dist_at_kmax   = predict_res.distances[end],
            max_abs_trace_drift     = predict_res.max_abs_trace_drift,
            channel_representation  = predict_res.channel_representation,
            physical_channel        = predict_res.physical_channel,
            interpretation_status   = predict_res.physical_channel ?
                :physical_channel : :nonphysical_surrogate,
            total_matvecs           = predict_res.total_matvecs,
            all_converged_predict   = predict_res.all_converged,
            # Hamiltonian-simulation time
            oft_time_per_step       = sim_budget.oft_time,
            b_per_be_per_step       = sim_budget.b_per_be,
            b_time_per_step         = sim_budget.b_time,
            per_step_time           = sim_budget.per_step_time,
            n_steps_total           = cost_fields.n_steps_total,
            total_ham_sim_time      = cost_fields.total_ham_sim_time,
            cost_interpretation     = cost_fields.cost_interpretation,
            # bookkeeping
            wall_time_seconds       = wall,
            init_state              = init_state,
            family_tag              = family_str,
        )
        results[i] = result

        if output_dir !== nothing
            sidecar = mode_phys ?
                _channel_sweep_sidecar_path(output_dir, n_i, β_i, seed_i, ε_i,
                                            f_i, construction_tag, domain_tag;
                                            beta_phys = β_phys_i) :
                _channel_sweep_sidecar_path(output_dir, n_i, β_i, seed_i, ε_i,
                                            f_i, construction_tag, domain_tag)
            try
                BSON.bson(sidecar, Dict(:result => Dict(pairs(result))))
            catch err
                @warn "Channel sidecar write failed (continuing)" sidecar err
            end
        end
    end

    return results
end

# Materialise the standard normalised local Pauli sources in a chosen basis.
function _jumps_in_basis(num_qubits::Integer, basis_eigvecs::AbstractMatrix)
    jumps = JumpOp[]
    for local_jump in local_pauli_jumps_1d(num_qubits)
        op = Matrix(materialize_local_jump(local_jump))
        op_eb = basis_eigvecs' * op * basis_eigvecs
        push!(jumps, JumpOp(op, op_eb, op == transpose(op), op == op'))
    end
    return jumps
end
