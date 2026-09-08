# DLL-first orchestration. Numerical construction/propagation stay in the
# existing HamHam, Workspace, Krylov and diagnostic implementations.

"""Combined trajectory, independent spectrum, checks and physical-input provenance.
`trajectory.distances` is the half trace norm; `trajectory.trace_norms` is twice it.
No portable save/load schema is provided before the result-persistence task.
"""
struct GibbsSimulationResult{T,S,D,P,C} <: AbstractResults
    trajectory::T
    spectrum::S
    diagnostics::D
    provenance::P
    convergence::C
end
function Base.show(io::IO,r::GibbsSimulationResult)
    print(io,"GibbsSimulationResult(",r.convergence.status,
        ", t=",last(r.trajectory.t),", trace_distance=",last(r.trajectory.distances),
        ", gap_reliability=",r.spectrum.reliability,
        ", uniqueness=",r.diagnostics.uniqueness,
        ", basis=",r.provenance.basis,")")
end

# Preflight intentionally does not call HamHam or rotate/diagonalise any matrix.
function _gibbs_preflight(H; beta_phys=nothing,temperature=nothing,
    filter=DLLGaussianFilter,jumps=:onsite_paulis,rates=1,complete_adjoint::Bool=false,
    basis::Symbol=:computational,domain::AbstractDomain=BohrDomain(),
    construction::AbstractConstruction=DLL(),time_step=nothing,num_energy_bits=nothing,
    clock=nothing,transition_weight=nothing,max_bytes::Integer=256*1024^2)
    construction isa DLL || throw(ArgumentError("The built-in facade currently supports DLL; CKG/GNS use the legacy Config API."))
    domain isa Union{BohrDomain,TimeDomain} || throw(ArgumentError("DLL facade supports BohrDomain and TimeDomain."))
    transition_weight === nothing || throw(ArgumentError("DLL already includes the thermal amplitude; transition_weight must be nothing."))
    basis in (:computational,:eigen) || throw(ArgumentError("basis must be :computational or :eigen."))
    (beta_phys === nothing) != (temperature === nothing) || throw(ArgumentError("Supply exactly one of beta_phys or temperature."))
    beta = temperature === nothing ? beta_phys : inv(temperature)
    beta isa Real && isfinite(beta) && beta > 0 || throw(ArgumentError("Physical inverse temperature must be finite and positive."))
    clock === nothing || clock isa GeneratorClock || throw(ArgumentError("clock must be an explicit GeneratorClock or nothing."))
    max_bytes >= 0 || throw(ArgumentError("max_bytes must be nonnegative."))
    matrix = H isa AbstractMatrix ? H : H isa HamHam ? H.data : H isa NamedTuple && haskey(H,:matrix) ? H.matrix :
        throw(ArgumentError("H must be a physical qubit matrix, raw Hamiltonian tuple or HamHam."))
    d = size(matrix,1)
    d >= 2 && ispow2(d) && size(matrix,2) == d || throw(ArgumentError("H must be a square qubit matrix of dimension at least two."))
    all(isfinite,matrix) && ishermitian(matrix) || throw(ArgumentError("H must be finite and Hermitian."))
    n = trailing_zeros(d)
    count = if jumps isa Symbol
        jumps == :onsite_paulis || throw(ArgumentError("Only the :onsite_paulis source preset is supported."))
        3n
    else
        length(jumps)
    end
    count > 0 || throw(ArgumentError("Provide a nonempty source family."))
    rate_values = rates isa Real ? (rates,) : rates
    (rates isa Real || length(rates)==count) && all(r -> r isa Real && isfinite(r) && r > 0,rate_values) ||
        throw(ArgumentError("Source rates must be positive finite scalars, one per source."))
    physical = filter isa AbstractFilter ? filter : applicable(filter,beta) ? filter(beta) :
        throw(ArgumentError("filter must be a built-in DLL instance or physical-beta factory."))
    physical isa AbstractFilter || throw(ArgumentError("Filter factory must return an AbstractFilter."))
    domain isa TimeDomain && !_dll_time_supported(physical) && throw(ArgumentError(
        "Custom DLL Time requires prepare_filter_transform with numerical controls."))
    errors = String[]
    T = typeof(real(float(zero(eltype(matrix)))))
    _collect_dll_filter_errors!(errors,physical,T(beta);label="physical filter")
    isempty(errors) || throw(ArgumentError(join(errors,"; ")))
    physical = _physical_dll_filter(physical,one(T),T(beta))
    channels = length(_flatten_local_dll_channels((physical,)))
    nt = if domain isa TimeDomain
        time_step isa Real && isfinite(time_step) && time_step > 0 &&
            num_energy_bits isa Integer && 0 < num_energy_bits < 63 ||
            throw(ArgumentError("TimeDomain requires positive time_step and num_energy_bits < 63."))
        big(2)^num_energy_bits
    else
        time_step === nothing && num_energy_bits === nothing || throw(ArgumentError("BohrDomain does not use time registers."))
        big(0)
    end
    source_bound = complete_adjoint && !(jumps isa Symbol) ? 2count : count
    # Includes Hamiltonian spectral copies/Bohr caches, owned input/filtered
    # sources, per-thread matvec storage, and Time pair-grid temporaries.
    coherent_bytes=big(0)
    if domain isa TimeDomain && physical isa PreparedFilterTransform
        c=physical.controls.coherent
        c.backend==:finufft && !(T in (Float32,Float64)) && throw(ArgumentError("FINUFFT uses Float64; select coherent.backend=:direct."))
        if c.method==:time
            dt=c.time_step===nothing ? time_step : c.time_step
            W=c.time_window===nothing ? (nt÷2)*time_step : c.time_window
            points=2ceil(BigInt,W/dt)+1
            largest_time=c.refine ? 2points-1 : points
            largest_freq=c.refine ? 2big(c.frequency_grid_size)-1 : big(c.frequency_grid_size)
            max(largest_time,largest_freq)<=c.max_points || throw(ArgumentError("Coherent controls/refinements exceed max_points=$(c.max_points)."))
            coherent_bytes=big(128)*(largest_time^2+largest_freq^2)
        end
    end
    bytes = big(16)*d^2*(80+12source_bound*channels+20Threads.nthreads()) + big(64)*nt^2 + coherent_bytes
    return (;dimension=d,num_qubits=n,construction=:DLL,domain=Symbol(nameof(typeof(domain))),
        beta_phys=beta,beta_alg=H isa AbstractMatrix ? nothing : beta*H.rescaling_factor,
        rescaling_factor=H isa AbstractMatrix ? nothing : H.rescaling_factor,
        basis,filter=physical,source_count=count,compiled_source_upper_bound=source_bound,
        channel_count=channels,complete_adjoint,rates,
        clock=clock === nothing ? :raw_generator : clock.label,
        generator_multiplier=clock === nothing ? 1. : clock.multiplier,
        estimated_construction_bytes=bytes,max_bytes,permitted=bytes<=max_bytes,
        spectral_preparation=:not_run,
        unresolved=H isa AbstractMatrix ? (:beta_alg,:rescaling_factor,:cached_gibbs,:source_basis_validation) : (:cached_gibbs,:source_basis_validation),
        resource_scope=:conservative_working_set_estimate)
end

"""
    Workspace(H; beta_phys=nothing, temperature=nothing, jumps=:onsite_paulis, ...)

Compile built-in DLL physical inputs once into the existing workspace. A
preflight allocation estimate is checked before Hamiltonian diagonalisation.
`dry_run=true` returns that estimate and resolved physical settings; algorithm
coordinates and source-basis checks requiring diagonalisation remain explicit.
"""
function Workspace(H::Union{AbstractMatrix,NamedTuple,HamHam};dry_run::Bool=false,
    max_bytes::Integer=256*1024^2,transition_weight=nothing,kwargs...)
    preflight = _gibbs_preflight(H;max_bytes,transition_weight,kwargs...)
    dry_run && return preflight
    preflight.permitted || throw(ArgumentError("Construction working-set estimate $(preflight.estimated_construction_bytes) exceeds max_bytes=$max_bytes; inspect dry_run=true."))
    source_basis = get(kwargs,:jumps,:onsite_paulis) isa Symbol ? :computational : preflight.basis
    p = prepare_gibbs_inputs(H;merge((;kwargs...),(;filter=preflight.filter,basis=source_basis))...)
    ws = Workspace(p.config,p.hamiltonian,p.jumps)
    provenance = merge(ws.research_provenance === nothing ? (;) : ws.research_provenance,
        p.provenance,
        (;basis=preflight.basis,preflight))
    return typeof(ws)((getfield(ws,i) for i in 1:fieldcount(typeof(ws))-1)...,provenance)
end

"""
    simulate_gibbs(H; times, beta_phys, diagnostics=:standard, ...)
    simulate_gibbs(ws; times, rho0=nothing, basis=:computational, ...)

Evolve the full DLL generator by matrix-free Krylov exponentiation. Inputs and
outputs use the declared basis; default rho0 is computational |+><+| tensor power.
The returned initial-state-specific threshold refers to trace distance (half
trace norm), never worst-case mixing. `:not_reached_by_horizon` is not a claim of
nonergodicity. Raw states are not repaired by default; `repair_states=true`
explicitly enables recorded Hermitian/trace corrections.

`method=:predictor` uses the existing state-coupled spectral predictor, retaining
all captured modes, with independent full-state propagation spot checks and
separate operator-gap analysis. Its crossing estimate uses its own eigenmodes.
`max_extensions` (default zero) permits doubling the horizon up to `max_time`.
Saved states default off and require `max_saved_bytes`. Construction, trajectory
and diagnostics have explicit estimates/caps; `gap_options` controls the separate
robust spectral budget. `dry_run` performs no Hamiltonian diagonalisation.
"""
function simulate_gibbs(H::Union{AbstractMatrix,NamedTuple,HamHam};times,
    rho0=nothing,basis::Symbol=:computational,diagnostics::Symbol=:standard,
    dry_run::Bool=false,max_bytes::Integer=256*1024^2,
    method::Symbol=:krylov,save_states::Bool=false,max_saved_bytes::Integer=64*1024^2,
    epsilon::Real=1e-3,krylovdim::Int=30,tol::Real=1e-10,
    max_matvecs::Integer=100000,max_seconds::Real=120,
    max_extensions::Int=0,max_time::Real=isempty(times) ? 0. : last(times),repair_states::Bool=false,
    gap_options::NamedTuple=NamedTuple(),kwargs...)
    _validate_time_grid(times;require_zero=true)
    preflight = Workspace(H;basis,dry_run=true,max_bytes,kwargs...)
    controls = _simulation_controls(times,preflight.dimension;diagnostics,method,
        save_states,max_saved_bytes,epsilon,krylovdim,tol,max_matvecs,max_seconds,
        max_extensions,max_time,max_bytes,base_bytes=preflight.estimated_construction_bytes)
    dry_run && return merge(preflight,(;trajectory=controls,diagnostics,permitted=preflight.permitted && controls.permitted,
        planned_checks=diagnostics==:strict ? (:raw_states,:independent_starts,:subspace_refinement,:complete_kms_reference) :
            diagnostics==:standard ? (:raw_states,:independent_starts,:subspace_refinement) : (:raw_states,:one_operator_start)))
    controls.permitted || throw(ArgumentError("Trajectory/state storage working-set estimate exceeds its memory cap; inspect dry_run=true."))
    ws = Workspace(H;basis,max_bytes,merge((;kwargs...),(;filter=preflight.filter))...)
    return simulate_gibbs(ws;times,rho0,basis,diagnostics,max_bytes,method,save_states,
        max_saved_bytes,epsilon,krylovdim,tol,max_matvecs,max_seconds,max_extensions,
        max_time,repair_states,gap_options)
end

function _simulation_controls(times,d;diagnostics,method,save_states,max_saved_bytes,
    epsilon,krylovdim,tol,max_matvecs,max_seconds,max_extensions,max_time,max_bytes,base_bytes)
    diagnostics in (:quick,:standard,:strict) || throw(ArgumentError("Unknown diagnostic level."))
    method in (:krylov,:predictor) || throw(ArgumentError("method must be :krylov or :predictor."))
    isfinite(epsilon) && epsilon > 0 && isfinite(tol) && tol > 0 || throw(ArgumentError("epsilon and tol must be finite and positive."))
    krylovdim > 0 && max_extensions >= 0 && isfinite(max_time) && max_time >= last(times) ||
        throw(ArgumentError("Require positive krylovdim, nonnegative extension count and finite max_time >= requested horizon."))
    max_saved_bytes >= 0 && max_bytes >= 0 || throw(ArgumentError("Memory caps must be nonnegative."))
    _work_budget(max_matvecs,max_seconds)
    points = big(length(times))+max_extensions
    saved = save_states ? big(16)*d^2*points : big(0)
    estimated = base_bytes + 3saved + big(16)*d^2*(8min(krylovdim,d^2)+32) + 4096points
    return (;permitted=saved<=max_saved_bytes && estimated<=max_bytes,
        estimated_bytes=estimated,saved_state_bytes=saved,max_saved_bytes,
        max_bytes,max_matvecs,max_seconds,max_extensions,max_time,method)
end

function _join_trajectories(a,b)
    return merge(a,(;t=vcat(a.t,b.t[2:end]),distances=vcat(a.distances,b.distances[2:end]),
        rho_final=b.rho_final,total_matvecs=a.total_matvecs+b.total_matvecs,
        all_converged=a.all_converged && b.all_converged,
        states=isempty(a.states) ? a.states : vcat(a.states,b.states[2:end]),
        raw_checks=vcat(a.raw_checks,b.raw_checks[2:end]),
        repair_norms=vcat(a.repair_norms,b.repair_norms[2:end]),failure=b.failure))
end

function simulate_gibbs(ws::Workspace{KrylovSpectrum};times,rho0=nothing,
    basis::Symbol=ws.research_provenance === nothing ? :computational : ws.research_provenance.basis,
    diagnostics::Symbol=:standard,method::Symbol=:krylov,dry_run::Bool=false,
    save_states::Bool=false,max_saved_bytes::Integer=64*1024^2,
    epsilon::Real=1e-3,krylovdim::Int=30,tol::Real=1e-10,
    max_matvecs::Integer=100000,max_seconds::Real=120,max_bytes::Integer=256*1024^2,
    max_extensions::Int=0,max_time::Real=isempty(times) ? 0. : last(times),
    repair_states::Bool=false,gap_options::NamedTuple=NamedTuple())
    _validate_time_grid(times;require_zero=true)
    cfg,ham = ws.cached_cfg,ws.ham_or_trott
    cfg isa Config{Lindbladian} && cfg.construction isa DLL && ham isa HamHam ||
        throw(ArgumentError("Built-in DLL Lindbladian workspace required."))
    _validate_reused_krylov_workspace(ws,cfg,ham,nothing,ws.jumps)
    basis in (:computational,:eigen) || throw(ArgumentError("basis must be :computational or :eigen."))
    method == :predictor && cfg.domain isa TimeDomain && throw(ArgumentError("The spectral predictor supports BohrDomain; use method=:krylov for TimeDomain."))
    method == :predictor && repair_states && throw(ArgumentError("Predictor raw reconstruction does not support state repair."))
    d = size(ham.data,1)
    controls = _simulation_controls(times,d;diagnostics,method,save_states,max_saved_bytes,
        epsilon,krylovdim,tol,max_matvecs,max_seconds,max_extensions,max_time,max_bytes,
        base_bytes=Base.summarysize(ws))
    dry_run && return (;provenance=ws.research_provenance,trajectory=controls,basis,diagnostics,spectral_preparation=:already_compiled)
    controls.permitted || throw(ArgumentError("Trajectory/state storage estimate exceeds memory cap."))
    CT = eltype(ws.G_left)
    initial = rho0 === nothing ? fill(one(CT)/d,d,d) : Matrix{CT}(rho0)
    size(initial) == (d,d) || throw(ArgumentError("rho0 must match the Hamiltonian dimension."))
    # Positivity of user input is mandatory, within the preflight allocation cap.
    initial_checks = state_diagnostics(initial;rtol=max(1e-9,100eps(real(CT))),dense_max_dim=d,max_dense_bytes=max_bytes)
    all(c.status==:pass for c in values(initial_checks)) || throw(ArgumentError("rho0 must be finite, Hermitian, unit trace and positive; no repair of invalid input is applied."))
    U = ham.eigvecs
    rho = basis==:computational || rho0===nothing ? Matrix{CT}(U'*initial*U) : initial
    sigma = Matrix{CT}(ham.gibbs)
    budget = _work_budget(max_matvecs,max_seconds)
    action! = (out,x) -> copyto!(out,apply_lindbladian!(ws,x,cfg,ham))
    integrate(r,t;save=save_states) = lindblad_action_integrate(action!,r,sigma,collect(t);
        krylovdim=min(krylovdim,d^2+1),tol,save_states=save,repair_states,record_diagnostics=true,
        dense_max_dim=64,max_dense_bytes=max_bytes,
        matvec_callback=budget.tick,work_callback=budget.check)
    predictor_check = _skipped("Spectral predictor not selected.")
    modal_crossing = nothing
    trajectory = nothing
    if method == :krylov
        trajectory = integrate(rho,times)
    else
        try
            trajectory = predict_lindbladian_trajectory(cfg,ham,ws.jumps,rho,collect(times);
                workspace=ws,krylovdim,tol,save_states,raw_reconstruction=true,
                matvec_callback=budget.tick,work_callback=budget.check,
                max_dense_bytes=max_bytes)
            indices = unique([1,cld(length(times),2),length(times)])
            spot = integrate(rho,times[indices];save=true)
            defects = Float64[]
            for (i,t) in enumerate(spot.t)
                predicted = zeros(CT,d,d)
                for j in eachindex(trajectory.eigenvalues)
                    predicted .+= trajectory.c[j]*exp(trajectory.eigenvalues[j]*t).*trajectory.R_modes[j]
                end
                push!(defects,sum(svdvals(predicted-spot.states[i]))/2)
            end
            passed = spot.all_converged && length(spot.t)==length(indices) && maximum(defects)<=max(100tol,1e-9)
            predictor_check = _diagnostic(passed ? :pass : :inconclusive,
                (;trace_distance_defects=defects,times=spot.t),max(100tol,1e-9),
                :independent_krylov_exponentiation,:sampled_states,
                "Full-state spot checks of the captured predictor; no uniform-in-time error certificate.")
            trajectory = merge(trajectory,(;all_converged=passed,failure=spot.failure,
                total_matvecs=trajectory.total_matvecs+spot.total_matvecs))
            # Use this predictor's own modes and its initial-state stationary
            # projection. Zero resolution scales with the captured operator.
            scale = maximum(abs,trajectory.eigenvalues;init=0.)
            ztol = 10length(trajectory.eigenvalues)*eps(Float64)*scale
            zeroset = findall(x -> abs(x)<=ztol,trajectory.eigenvalues)
            decay = findall(x -> real(x)<-ztol,trajectory.eigenvalues)
            if passed && length(zeroset)+length(decay)==length(trajectory.eigenvalues) && !isempty(decay)
                floor = zeros(CT,d,d)
                for j in zeroset
                    floor .+= trajectory.c[j].*trajectory.R_modes[j]
                end
                budget.check()
                estimate = eigenmode_mixing_time(trajectory.eigenvalues[decay],trajectory.c[decay],
                    trajectory.R_modes[decay],floor,sigma,epsilon;
                    t_upper=max(last(times),1.),eigenvalue_zero_tol=0.)
                budget.check()
                trajectory = merge(trajectory,(;eigenvalues=trajectory.eigenvalues[decay],
                    c=trajectory.c[decay],R_modes=trajectory.R_modes[decay],rho_inf=floor,
                    stationary_projection_scope=:captured_initial_state,
                    spectral_modes=spectral_mode_diagnostics(trajectory.eigenvalues[decay],
                        trajectory.R_modes[decay],trajectory.c[decay])))
                modal_crossing = merge(estimate,(;scope=:initial_state_captured_spectrum,
                    propagated=false,horizon=last(times)))
            end
        catch err
            err isa InterruptException && rethrow()
            # Retain the valid initial state when a predictor cannot be formed.
            trajectory === nothing && (trajectory = integrate(rho,[zero(eltype(times))]))
            trajectory = merge(trajectory,(;all_converged=false,
                failure=(;reason=err isa _WorkLimit ? err.reason : :predictor_failure,message=sprint(showerror,err))))
        end
    end
    extensions = 0
    while trajectory.all_converged && last(trajectory.distances)>epsilon &&
        extensions < max_extensions && last(trajectory.t)<max_time
        start = last(trajectory.t)
        stop = min(max_time,iszero(start) ? 1. : 2start)
        continuation = integrate(trajectory.rho_final,[start,stop])
        trajectory = _join_trajectories(trajectory,continuation)
        extensions += 1
    end
    trajectory_seconds = budget.elapsed()
    # Diagnostic failure cannot discard an already valid trajectory.
    spectrum = robust_spectral_gap(ws;diagnostics,
        merge((;max_bytes=max(big(0),big(max_bytes)-Base.summarysize(trajectory)),max_seconds=60.),gap_options)...)
    spectrum = merge(spectrum,(;basis=:eigen,clock=ws.research_provenance === nothing ? :compiled_generator : ws.research_provenance.clock_label))
    unavailable = _skipped("Independent diagnostics exhausted their budget or failed; inspect spectrum.failures.")
    physical = spectrum.reference === nothing ?
        GibbsDiagnostics((;stationarity=unavailable,trace_preservation=unavailable,
            conditioning=unavailable,state=unavailable,kms=unavailable,kernel=unavailable,tails=unavailable),
            nothing,spectrum.resources,:not_established) : spectrum.reference
    output_rho(A) = basis==:computational ? Matrix(U*A*U') : A
    trajectory = merge(trajectory,(;rho_final=output_rho(trajectory.rho_final),
        states=[output_rho(A) for A in trajectory.states],trace_norms=2 .* trajectory.distances,
        basis,channel_steps=nothing,wall_seconds=trajectory_seconds,
        predictor_check,modal_crossing))
    # Keep every public state/mode in the declared basis, including predictor payloads.
    if haskey(trajectory,:R_modes)
        trajectory = merge(trajectory,(;R_modes=output_rho.(trajectory.R_modes),
            rho_inf=output_rho(trajectory.rho_inf),sigma_beta=output_rho(trajectory.sigma_beta)))
    end
    crossing = _find_actual_mixing_time(trajectory.t,trajectory.distances,Float64(epsilon))
    raw_pass = all(checks -> all(getproperty(checks,k).status==:pass for k in (:finiteness,:trace,:hermiticity)),trajectory.raw_checks)
    validity = raw_pass && all(c.positivity.status==:pass for c in trajectory.raw_checks)
    status = !trajectory.all_converged || !validity ? :inconclusive :
        crossing === nothing ? :not_reached_by_horizon : iszero(crossing) ? :already_within_threshold : :reached_threshold
    convergence = (;status,epsilon,metric=:trace_distance,threshold_time=crossing,
        horizon=last(trajectory.t),requested_horizon=last(times),max_time,extensions,
        initial_state_specific=true,worst_case_mixing=:not_established,
        numerical_floor=nothing,accuracy_evidence=method==:krylov ? :krylov_step_convergence_and_raw_state_checks : :predictor_spot_checks)
    provenance = ws.research_provenance === nothing ?
        (;beta_phys=cfg.beta_phys,beta_alg=cfg.beta,clock_label=:compiled_generator,
            input_preparation=:legacy_workspace) : ws.research_provenance
    provenance = merge(provenance,(;basis,initial_state=rho0===nothing ? :computational_plus_product : :user_supplied,
        method,coherent=true,resources=controls,trajectory_matvecs=budget.count[],
        simulated_time=last(trajectory.t),channel_steps=nothing,wall_seconds=budget.elapsed()))
    return GibbsSimulationResult(trajectory,spectrum,physical,provenance,convergence)
end

function Base.show(io::IO,r::GibbsDiagnostics)
    print(io,"GibbsDiagnostics(stationarity=",r.checks.stationarity.status,
        ", trace_preservation=",r.checks.trace_preservation.status,
        ", kms=",r.checks.kms.status,", uniqueness=",r.uniqueness,")")
end
