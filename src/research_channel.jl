# Finite-channel facade. All physical updates remain in run_thermalize.
function _channel_schedule(times,steps,delta,save_every,max_bytes)
    delta isa Real && isfinite(delta) && 0 < delta <= 1 ||
        throw(ArgumentError("Channel delta must satisfy 0 < delta <= 1 in the implemented channel clock."))
    save_every isa Integer && save_every > 0 || throw(ArgumentError("save_every must be a positive integer."))
    if times !== nothing
        _validate_time_grid(times;require_zero=true)
        save_every == 1 || throw(ArgumentError("Use either times or channel_options.save_every, not both."))
        64big(length(times)) <= max_bytes || throw(ArgumentError("Channel schedule exceeds max_bytes."))
        grid=Int[]
        for t in times
            q=t/delta
            isfinite(q) && 0 <= q < typemax(Int) || throw(ArgumentError("Channel step index overflows Int."))
            k=round(Int,q)
            isapprox(q,k;atol=min(1e-4,32eps(typeof(float(q)))*max(1,abs(q))),rtol=0) ||
                throw(ArgumentError("Every requested time must be an integer multiple of delta; use steps for an exact horizon."))
            push!(grid,k)
        end
        all(diff(grid).>0) || length(grid)==1 || throw(ArgumentError("Requested times map to repeated channel steps."))
        steps===nothing || steps isa Integer && steps==last(grid) ||
            throw(ArgumentError("steps must equal the last requested channel time divided by delta."))
        return last(grid),grid
    end
    steps isa Integer && 0 <= steps < typemax(Int) || throw(ArgumentError("Channel evolution requires nonnegative integer steps or times on the delta grid."))
    64(big(steps)÷save_every+2) <= max_bytes || throw(ArgumentError("Channel schedule exceeds max_bytes; increase save_every."))
    grid=collect(0:Int(save_every):Int(steps))
    last(grid)==steps || push!(grid,Int(steps))
    return Int(steps),grid
end

function _channel_options(options)
    allowed=(:save_every,:jump_selection,:seed,:max_steps,:rescale_by_inv_prob)
    all(k->k in allowed,keys(options)) || throw(ArgumentError("Unknown channel option; supported keys are $allowed."))
    c=merge((;save_every=1,jump_selection=:sweep,seed=0,max_steps=100000,rescale_by_inv_prob=nothing),options)
    c.jump_selection in (:sweep,:random) || throw(ArgumentError("jump_selection must be :sweep or :random."))
    c.seed isa Integer && c.seed>=0 || throw(ArgumentError("seed must be a nonnegative integer."))
    c.max_steps isa Integer && c.max_steps>=0 || throw(ArgumentError("max_steps must be a nonnegative integer."))
    c.rescale_by_inv_prob===nothing || c.rescale_by_inv_prob isa Bool || throw(ArgumentError("rescale_by_inv_prob must be Bool or nothing."))
    return c
end

function _channel_resources(d,n,points;save_states,max_saved_bytes,max_bytes,base_bytes=0)
    max_saved_bytes>=0 && max_bytes>=0 || throw(ArgumentError("Memory caps must be nonnegative."))
    saved=save_states ? big(16)*d^2*(points+1) : big(0) # one interrupted endpoint
    bytes=big(base_bytes)+big(16)*d^2*(64+12n)+3saved+4096big(points+1)
    return (;estimated_bytes=bytes,saved_state_bytes=saved,max_bytes,max_saved_bytes,
        permitted=bytes<=max_bytes && saved<=max_saved_bytes)
end

function _simulate_gibbs_channel(H;times,steps,delta,rho0,basis,diagnostics,dry_run,
    max_bytes,save_states,max_saved_bytes,epsilon,max_seconds,repair_states,channel_options,kwargs...)
    c=_channel_options(channel_options)
    nsteps,grid=_channel_schedule(times,steps,delta,c.save_every,max_bytes)
    construction=get(kwargs,:construction,DLL())
    construction isa KMS || throw(ArgumentError("Physical channel evolution requires explicit construction=KMS(); DLL Thermalize is unavailable."))
    get(kwargs,:domain,BohrDomain()) isa Union{BohrDomain,EnergyDomain} ||
        throw(ArgumentError("Physical channel inputs support Bohr/Energy; use the Config{Thermalize} overload for existing Time/Trotter channels."))
    preflight=_gibbs_preflight(H;basis,max_bytes,kwargs...)
    preflight.transition_weight isa CKGJointKernel && throw(ArgumentError("Custom joint CKG channels are not supported by this facade."))
    resources=_channel_resources(preflight.dimension,preflight.compiled_source_upper_bound,length(grid);
        save_states,max_saved_bytes,max_bytes,base_bytes=preflight.estimated_construction_bytes+
            (get(kwargs,:domain,BohrDomain()) isa EnergyDomain ?
                big(64)*preflight.dimension^2*big(2)^kwargs[:num_energy_bits] : big(0)))
    diagnostics in (:quick,:standard,:strict) || throw(ArgumentError("Unknown diagnostic level."))
    isfinite(epsilon) && epsilon>0 || throw(ArgumentError("epsilon must be finite and positive."))
    _work_budget(c.max_steps,max_seconds)
    dry_run && return merge(preflight,(;evolution=:channel,method=:channel,steps=nsteps,delta,
        trajectory=resources,permitted=preflight.permitted && resources.permitted,
        planned_checks=(:sampled_raw_states,:trace_drift),channel_spectrum=:not_run,
        rate_operator_bound=:checked_during_channel_compilation))
    preflight.permitted && resources.permitted || throw(ArgumentError("Channel working-set estimate exceeds memory cap; inspect dry_run=true."))
    p=prepare_gibbs_inputs(H;merge((;kwargs...),(;basis=get(kwargs,:jumps,:onsite_paulis) isa Symbol ? :computational : basis,
        filter=preflight.filter,transition_weight=preflight.transition_weight))...)
    T=typeof(p.config.beta)
    fields=(;(k=>getfield(p.config,k) for k in fieldnames(typeof(p.config)))...)
    cfg=Config(;merge(fields,(;sim=Thermalize(),delta=T(delta),mixing_time=T(nsteps)*T(delta),jump_selection=c.jump_selection))...)
    result=simulate_gibbs(p.jumps,cfg,p.hamiltonian;steps=nsteps,
        times=times===nothing ? nothing : grid.*cfg.delta,rho0,basis,
        diagnostics,save_states,max_saved_bytes,max_bytes,epsilon,max_seconds,repair_states,
        channel_options=times===nothing ? c : merge(c,(;save_every=1)))
    # Freeze metadata descriptions, never retain arbitrary callbacks for BSON.
    input_provenance=_portable_unpack(_portable_pack(p.provenance;grid=:metadata))
    provenance=merge(input_provenance,result.provenance,(;input_preparation=:physical_inputs,
        clock_label=p.provenance.clock_label,resources,physical_preflight_domain=preflight.domain))
    return GibbsSimulationResult(result.trajectory,result.spectrum,result.diagnostics,provenance,result.convergence)
end

"""
    simulate_gibbs(jumps, config::Config{Thermalize}, ham, trotter=nothing;
                   steps=nothing, times=nothing, rho0=nothing, ...)

Run the existing finite channel with the shared GibbsSimulationResult interface.
Legacy config/jump fields keep algorithm-frame semantics. States use the
computational basis by default; `basis=:eigen` means the backend basis (the
D-register Trotter eigenbasis for TrotterDomain). JumpOp caches must already use
that backend basis, as in run_thermalize. Default state is the computational plus
product state, matching the physical facade, rather than run_thermalize's I/d.

Channel options are save_every, jump_selection, seed, max_steps and
rescale_by_inv_prob. A random run is conditioned on sampled source choices.
No channel spectrum, stationary-state uniqueness or KMS certificate is inferred
from the ideal generator. Saved evidence cannot automatically rebuild a workspace.
"""
function simulate_gibbs(jumps::Vector{JumpOp},cfg::Config{Thermalize},ham::HamHam,
    trotter::Union{AbstractTrotter,Nothing}=nothing;steps=nothing,times=nothing,
    rho0=nothing,basis::Symbol=:computational,diagnostics::Symbol=:standard,
    dry_run::Bool=false,save_states::Bool=false,max_saved_bytes::Integer=64*1024^2,
    max_bytes::Integer=256*1024^2,epsilon::Real=1e-3,max_seconds::Real=120,
    repair_states::Bool=false,channel_options::NamedTuple=(;))
    cfg.construction isa DLL && throw(ArgumentError("DLL Thermalize channels are unavailable."))
    _is_joint_ckg(cfg) && throw(ArgumentError("Custom joint CKG channels are not supported by this facade."))
    c=_channel_options(merge((;jump_selection=cfg.jump_selection),channel_options))
    if times===nothing && steps===nothing
        cfg.mixing_time===nothing && throw(ArgumentError("Provide steps, times or Config.mixing_time."))
        # Require exact grid semantics instead of silently rounding up.
        horizon=iszero(cfg.mixing_time) ? [zero(cfg.mixing_time)] : [zero(cfg.mixing_time),cfg.mixing_time]
        steps=first(_channel_schedule(horizon,nothing,cfg.delta,1,max_bytes))
    end
    nsteps,grid=_channel_schedule(times,steps,cfg.delta,c.save_every,max_bytes)
    T=typeof(cfg.beta)
    fields=(;(k=>getfield(cfg,k) for k in fieldnames(typeof(cfg)))...)
    cfg=Config(;merge(fields,(;mixing_time=T(nsteps)*cfg.delta,jump_selection=c.jump_selection))...)
    validate_config!(cfg,ham)
    basis in (:computational,:eigen) || throw(ArgumentError("basis must be :computational or :eigen."))
    diagnostics in (:quick,:standard,:strict) || throw(ArgumentError("Unknown diagnostic level."))
    isfinite(epsilon) && epsilon>0 || throw(ArgumentError("epsilon must be finite and positive."))
    isempty(jumps) && throw(ArgumentError("Provide a nonempty source family."))
    d=size(ham.data,1)
    register_bytes=big(0)
    if cfg.domain isa Union{EnergyDomain,TimeDomain,TrotterDomain}
        bits=cfg.domain isa EnergyDomain ? (register_r_D(cfg),) :
            (register_r_D(cfg),register_r_b_minus(cfg),register_r_b_plus(cfg))
        all(r->r isa Integer && 0<r<63,bits) || throw(ArgumentError("Channel register sizes must lie between 1 and 62."))
        sizes=big(2) .^ bits
        # Raw quadrature grids, OFT tables, coherent pair grids and transform
        # temporaries exist even when their eventual sampled support is short.
        register_bytes=big(64)*d^2*sum(sizes)
        cfg.domain isa EnergyDomain || (register_bytes+=big(128)*sum(x->x^2,sizes))
    end
    resources=_channel_resources(d,length(jumps),length(grid);save_states,max_saved_bytes,max_bytes,
        base_bytes=Base.summarysize((ham,jumps,trotter))+register_bytes)
    budget=_work_budget(c.max_steps,max_seconds)
    dry_run && return (;evolution=:channel,steps=nsteps,delta=cfg.delta,trajectory=resources,
        permitted=resources.permitted,spectral_preparation=:already_prepared,planned_checks=(:sampled_raw_states,:trace_drift))
    resources.permitted || throw(ArgumentError("Channel working-set/state storage estimate exceeds memory cap."))
    if cfg.domain isa TrotterDomain
        trotter===nothing && throw(ArgumentError("Trotter channel requires a matching Trotter cache."))
        _validate_trotter_cache!(cfg,ham,trotter)
    elseif trotter!==nothing
        throw(ArgumentError("A Trotter cache requires TrotterDomain."))
    end
    U=cfg.domain isa TrotterDomain ? trotter.eigvecs : ham.eigvecs
    CT=eltype(U)
    for jump in jumps
        size(jump.data)==(d,d) && size(jump.in_eigenbasis)==(d,d) || throw(ArgumentError("Source dimensions must match H."))
        isapprox(jump.in_eigenbasis,U'*jump.data*U;atol=100eps(T),rtol=100eps(T)) ||
            throw(ArgumentError("JumpOp cache does not match the channel working basis."))
    end
    initial,rho = _prepare_initial_density(rho0,U,basis,CT,max_bytes)
    raw_checks=NamedTuple[]
    observe(step,A)=push!(raw_checks,state_diagnostics(A;rtol=max(1e-9,100eps(T)),
        dense_max_dim=diagnostics==:strict ? d : 64,max_dense_bytes=max_bytes))
    backend=run_thermalize(jumps,cfg,ham,trotter;initial_dm=rho,rng=MersenneTwister(c.seed),
        rescale_by_inv_prob=c.rescale_by_inv_prob,num_steps=nsteps,record_steps=grid,
        save_states,convergence_cutoff=0,hermitize=repair_states,
        work_callback=budget.tick,observation_callback=observe)
    metadata=backend.metadata
    output(A)=basis==:computational ? Matrix(U*A*U') : copy(A)
    recorded=metadata[:recorded_steps]
    failure=metadata[:failure]
    failure!==nothing && failure.reason==:matvecs && (failure=(;reason=:steps,message="Channel step budget exhausted."))
    trajectory=(;t=backend.time_steps,distances=backend.trace_distances,
        trace_norms=2 .* backend.trace_distances,rho_final=output(backend.final_dm),
        states=save_states ? output.(metadata[:states]) : Matrix{CT}[],basis,
        channel_steps=copy(recorded),completed_steps=metadata[:completed_steps],
        subchannel_applications=metadata[:completed_steps]*(c.jump_selection==:sweep ? length(jumps) : 1),
        raw_checks,trace_values=metadata[:trace_values],failure,
        all_converged=metadata[:failure]===nothing,wall_seconds=budget.elapsed(),total_matvecs=0)
    unavailable=_skipped("Finite channel spectrum, KMS and stationarity were not evaluated; ideal-generator checks do not certify this channel.")
    physical=GibbsDiagnostics((;stationarity=unavailable,trace_preservation=unavailable,
        conditioning=unavailable,state=last(raw_checks),kms=unavailable,kernel=unavailable,tails=unavailable),
        nothing,resources,:not_established)
    spectrum=(;reliability=:not_run,spectral_gap=nothing,coverage=:not_established,
        reference=nothing,target=:finite_channel,clock=:channel_step,failures=(),resources)
    i=findfirst(x->x<=epsilon,trajectory.distances)
    crossing=i===nothing ? nothing : trajectory.t[i]
    valid=all(cs->all(x->x.status==:pass,values(cs)),raw_checks)
    status=!trajectory.all_converged || !valid || cfg.with_gqsp ? :inconclusive :
        i===nothing ? :not_reached_by_horizon : i==1 ? :already_within_threshold : :reached_threshold
    convergence=(;status,epsilon,metric=:trace_distance,threshold_time=crossing,
        threshold_step=i===nothing ? nothing : recorded[i],horizon=last(trajectory.t),
        requested_horizon=nsteps*cfg.delta,initial_state_specific=true,worst_case_mixing=:not_established,
        crossing_scope=:observed_sample,accuracy_evidence=:implemented_channel_and_sampled_state_checks)
    rescale=something(c.rescale_by_inv_prob,c.jump_selection==:random)
    provenance=(;evolution=:channel,method=:channel,basis,coherent=with_coherent(cfg.construction),
        beta_phys=cfg.beta_phys,beta_alg=cfg.beta,clock_label=:compiled_channel,
        delta=cfg.delta,channel_steps=metadata[:completed_steps],requested_steps=nsteps,
        jump_selection=c.jump_selection,rescale_by_inv_prob=rescale,
        selection_rate_multiplier=rescale ? length(jumps) : 1,
        gamma_norm_factor=inv(pick_gamma_sup(cfg)),
        trajectory_kind=c.jump_selection==:sweep ? :deterministic_channel_composition : :source_conditioned_density_matrices,
        channel_representation=metadata[:channel_representation],physical_channel=metadata[:physical_channel],
        hermitized=repair_states,trace_normalized=false,resources,
        initial_state=rho0===nothing ? :computational_plus_product : :user_supplied,
        initial_density_matrix=output(rho),requested_times=grid.*cfg.delta,
        rng=(;algorithm=:MersenneTwister,seed=c.seed,scope=:source_selection),
        persistence=:evidence_only,reconstruction=:requires_original_channel_inputs,
        channel_config=_config_to_dict(cfg),runtime=_runtime_provenance())
    return GibbsSimulationResult(trajectory,spectrum,physical,provenance,convergence)
end
