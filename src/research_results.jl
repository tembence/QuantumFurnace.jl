# Portable evidence and replay recipes. Only tagged data cross the BSON boundary;
# numerical workspaces, transform caches and callbacks never do.
const _FILTER_REGISTRY = Dict{Tuple{Symbol,String},Function}()

"""
    register_filter!(name, version, factory)

Register `factory(beta, parameters)` returning a custom DLL filter in its input
frame. Registration is process-local: register the same version before rebuilding
in a new process. A name/version is a declaration, checked against retained
samples; it is not a proof that callbacks agree away from those samples.
"""
function register_filter!(name, version::AbstractString, factory::Function)
    key = (Symbol(name), String(version))
    isempty(strip(String(name))) || isempty(strip(version)) ?
        throw(ArgumentError("Filter name and version must be nonempty.")) : nothing
    haskey(_FILTER_REGISTRY,key) && throw(ArgumentError("Filter $key is already registered; use a new version."))
    _FILTER_REGISTRY[key] = factory
    return nothing
end

# CRC32c is a deterministic corruption/change check, not an authentication hash.
# Include shape/type and canonical primitive values (never Julia's salted hash).
function _evidence_digest(x)
    io = IOBuffer()
    _digest_write(io,x)
    return "crc32c:" * string(Base._crc32c(take!(io));base=16,pad=8)
end
function _digest_write(io,x)
    if x isa AbstractDict
        print(io,"dict{")
        for k in sort!(collect(keys(x));by=repr)
            _digest_write(io,k); _digest_write(io,x[k])
        end
        print(io,"}")
    elseif x isa AbstractArray || x isa Tuple
        print(io,"array",size(x isa Tuple ? collect(x) : x),"[")
        foreach(y->_digest_write(io,y),x); print(io,"]")
    else
        print(io,typeof(x),":",repr(x),";")
    end
end

_record(tag;kwargs...) = Dict{Symbol,Any}(:qf_tag=>tag,kwargs...)
function _portable_pack(x; grid=nothing)
    x isa BigInt && return _record("bigint";value=string(x))
    x isa BigFloat && return _record("bigfloat";value=string(x),bits=precision(x))
    x isa Complex && !isbitstype(typeof(x)) && return _record("bigcomplex";re=_portable_pack(real(x)),im=_portable_pack(imag(x)))
    x isa Rational && !isbitstype(typeof(x)) && return _record("rational";num=_portable_pack(numerator(x)),den=_portable_pack(denominator(x)))
    x isa Number && !isbitstype(typeof(x)) && throw(ArgumentError("Unsupported nonprimitive number $(typeof(x)); convert to a supported numeric type."))
    x === nothing || x isa Union{Number,Symbol,AbstractString,Bool} || return _portable_pack_container(x;grid)
    return x
end
function _portable_pack_container(x;grid=nothing)
    if x isa NamedTuple
        return _record("namedtuple";names=collect(keys(x)),values=[_portable_pack(v;grid) for v in values(x)])
    elseif x isa Tuple
        return _record("tuple";values=[_portable_pack(v;grid) for v in x])
    elseif x isa AbstractArray{<:Number} && isbitstype(eltype(x))
        return copy(Array(x))
    elseif x isa AbstractArray && eltype(x) in (BigInt,BigFloat,Complex{BigFloat})
        return _record("numeric_array";dims=collect(size(x)),element=string(eltype(x)),values=[_portable_pack(v;grid) for v in x])
    elseif x isa AbstractArray
        return _record("array";dims=collect(size(x)),
            element=eltype(x) in (Symbol,String) ? string(eltype(x)) : nothing,
            values=[_portable_pack(v;grid) for v in x])
    elseif x isa AbstractDict
        return _record("dict";entries=[[_portable_pack(k;grid),_portable_pack(v;grid)] for (k,v) in x])
    elseif x isa DataType && x in (Float32,Float64,BigFloat)
        return _record("precision";name=string(x))
    elseif x isa _UserDLLFilter
        if grid===:metadata
            return _record("filter_description";kind=string(nameof(typeof(x))),beta=_portable_pack(x.beta),
                name=x.metadata.name,version=x.metadata.version,parameters=_portable_pack(x.metadata.parameters;grid),
                support=_portable_pack(x.metadata.support),scope=:metadata_only_replay_has_frozen_samples)
        end
        g = grid === nothing ? collect(range(-one(x.beta),one(x.beta);length=17)) : collect(grid)
        x isa TimeFilter && (g=collect(range(-something(x.metadata.support,one(x.beta)),something(x.metadata.support,one(x.beta));length=17)))
        vals = x isa TimeFilter ? time_kernel.(Ref(x),g) : freq_kernel.(Ref(x),g)
        return _record("custom_filter";kind=string(nameof(typeof(x))),beta=_portable_pack(x.beta),
            name=x.metadata.name,version=x.metadata.version,parameters=_portable_pack(x.metadata.parameters),
            support=_portable_pack(x.metadata.support),has_tail_bound=x.metadata.tail_bound!==nothing,
            portability=haskey(_FILTER_REGISTRY,(x.metadata.name,x.metadata.version)) ? :registered : :requires_callable,
            grid=_portable_pack(g),values=_portable_pack(vals),sampled_hash=_evidence_digest((g,vals)),scope=:sampled_input_kernel)
    elseif x isa PreparedFilterTransform
        return _record("prepared_transform";base=_portable_pack(x.base;grid),controls=_portable_pack(x.controls))
    elseif x isa CKGJointKernel
        if grid===:metadata
            return _record("ckg_description";beta=_portable_pack(x.beta),scope=:metadata_only_replay_has_frozen_samples)
        end
        g = collect(range(first(x.frequency_window),last(x.frequency_window);length=65))
        vals = hcat(_joint_oft.(Ref(x),g),_joint_rate.(Ref(x),g))
        controls = (;(k=>getproperty(x,k) for k in fieldnames(typeof(x)) if !(k in (:oft,:rate)))...)
        return _record("custom_ckg";controls=_portable_pack(controls),grid=g,values=vals,
            sampled_hash=_evidence_digest((g,vals)),portability=:requires_callable,scope=:sampled_oft_and_rate)
    elseif x isa Union{AbstractFilter,AbstractCKGTransition}
        fields = (;(k=>getproperty(x,k) for k in fieldnames(typeof(x)))...)
        return _record("prescription";kind=string(nameof(typeof(x))),fields=_portable_pack(fields;grid))
    elseif x isa Union{DiagnosticCheck,GibbsDiagnostics,SpectralModeDiagnostics,KMSParentSpectrum,DLLFilterFrame,GeneratorClock}
        return _record("evidence";kind=string(nameof(typeof(x))),fields=_portable_pack((;(k=>getproperty(x,k) for k in fieldnames(typeof(x)))...);grid))
    elseif x isa Function
        return _record("unavailable_callable";portability=:requires_callable)
    end
    throw(ArgumentError("Unsupported portable evidence type $(typeof(x)); retain primitive data instead."))
end

function _portable_unpack(x; restore::Bool=false, filters=Dict(), cache=Dict{String,Any}())
    x isa AbstractDict && haskey(x,:qf_tag) || return x
    unpack(y) = _portable_unpack(y;restore,filters,cache)
    tag=x[:qf_tag]
    tag=="bigint" && return parse(BigInt,x[:value])
    tag=="bigfloat" && return setprecision(BigFloat,x[:bits]) do
        parse(BigFloat,x[:value])
    end
    tag=="bigcomplex" && return complex(unpack(x[:re]),unpack(x[:im]))
    tag=="rational" && return unpack(x[:num])//unpack(x[:den])
    if tag=="numeric_array"
        T=Dict("BigInt"=>BigInt,"BigFloat"=>BigFloat,"Complex{BigFloat}"=>Complex{BigFloat})[x[:element]]
        return reshape(T.(unpack.(x[:values])),Tuple(x[:dims]))
    end
    tag=="namedtuple" && return NamedTuple{Tuple(Symbol.(x[:names]))}(Tuple(unpack.(x[:values])))
    tag=="tuple" && return Tuple(unpack.(x[:values]))
    if tag=="array"
        array_values=unpack.(x[:values])
        element=get(x,:element,nothing)
        element===nothing || (array_values=Dict("Symbol"=>Symbol,"String"=>String)[element].(array_values))
        return reshape(array_values,Tuple(x[:dims]))
    end
    tag=="dict" && return Dict(unpack(p[1])=>unpack(p[2]) for p in x[:entries])
    tag=="precision" && return Dict("Float32"=>Float32,"Float64"=>Float64,"BigFloat"=>BigFloat)[x[:name]]
    if tag=="evidence"
        f=unpack(x[:fields]); name=x[:kind]
        name=="DiagnosticCheck" && return DiagnosticCheck(values(f)...)
        name=="GibbsDiagnostics" && return GibbsDiagnostics(values(f)...)
        name=="SpectralModeDiagnostics" && return SpectralModeDiagnostics(values(f)...)
        name=="KMSParentSpectrum" && return KMSParentSpectrum(values(f)...)
        name=="DLLFilterFrame" && return DLLFilterFrame{typeof(f.shift)}(values(f)...)
        name=="GeneratorClock" && return GeneratorClock{typeof(f.multiplier)}(values(f)...)
        throw(ArgumentError("Unknown evidence tag $name."))
    end
    tag in ("custom_filter","custom_ckg","prepared_transform","prescription",
        "unavailable_callable","filter_description","ckg_description","replay") ||
        throw(ArgumentError("Unknown portable evidence tag $tag."))
    restore || return deepcopy(x)
    cache_key=_evidence_digest(x)
    haskey(cache,cache_key) && return cache[cache_key]
    if tag=="custom_filter"
        key=(x[:name],x[:version])
        f = if haskey(filters,key)
            filters[key]
        elseif haskey(_FILTER_REGISTRY,key)
            _FILTER_REGISTRY[key](unpack(x[:beta]),unpack(x[:parameters]))
        else
            throw(ArgumentError("Missing custom filter $key; register its exact version or supply filters=Dict($key => filter)."))
        end
        f isa _UserDLLFilter && string(nameof(typeof(f)))==x[:kind] && f.metadata.name==x[:name] &&
            f.metadata.version==x[:version] && isequal(f.beta,unpack(x[:beta])) &&
            isequal(f.metadata.parameters,unpack(x[:parameters])) && isequal(f.metadata.support,unpack(x[:support])) &&
            (f.metadata.tail_bound!==nothing)==x[:has_tail_bound] ||
            throw(ArgumentError("Custom filter identity/version/parameters/temperature/support mismatch for $key."))
        vals=f isa TimeFilter ? time_kernel.(Ref(f),unpack(x[:grid])) : freq_kernel.(Ref(f),unpack(x[:grid]))
        _evidence_digest((unpack(x[:grid]),vals))==x[:sampled_hash] || throw(ArgumentError("Custom filter $key changed on the saved sample grid; rebuild the simulation from new inputs."))
        return cache[cache_key]=deepcopy(f)
    elseif tag=="custom_ckg"
        haskey(filters,:ckg_joint) || throw(ArgumentError("Custom CKG reconstruction requires filters=Dict(:ckg_joint => original_kernel)."))
        f=filters[:ckg_joint]; f isa CKGJointKernel || throw(ArgumentError("Resupply a CKGJointKernel."))
        current=_portable_pack(f)
        isequal(current[:controls],x[:controls]) && current[:sampled_hash]==x[:sampled_hash] ||
            throw(ArgumentError("CKG callable samples or controls changed; rebuild from new inputs."))
        return cache[cache_key]=deepcopy(f)
    elseif tag=="prepared_transform"
        c=unpack(x[:controls])
        return cache[cache_key]=prepare_filter_transform(unpack(x[:base]);window=c.window,breakpoints=collect(c.breakpoints),
            rtol=c.rtol,atol=c.atol,maxevals=c.maxevals,max_panels=c.max_panels,analytic=c.analytic,coherent=c.coherent)
    elseif tag=="prescription"
        f=unpack(x[:fields]); kind=x[:kind]
        kind=="GaussianFilter" && return GaussianFilter(f.sigma)
        kind=="DLLGaussianFilter" && return DLLGaussianFilter(f.beta)
        kind=="DLLMetropolisFilter" && return DLLMetropolisFilter(f.beta;S=f.S)
        kind=="DLLMultiChannelFilter" && return DLLMultiChannelFilter(f.channels,f.beta)
        kind=="DLLSourceFilters" && return DLLSourceFilters(f.assignments,f.beta)
        kind=="ShiftedSymmetricFilter" && return ShiftedSymmetricFilter{typeof(f.beta),typeof(f.base)}(f.base,f.shift,f.weight,f.beta)
        kind=="_DLLBohrFilter" && return _DLLBohrFilter(f.beta,Dict{typeof(f.beta),Complex{typeof(f.beta)}}(f.values))
        kind=="GaussianTransition" && return GaussianTransition(f.beta;sigma=f.sigma,sigma_gamma=f.sigma_gamma)
        kind=="MetropolisTransition" && return MetropolisTransition(f.beta;sigma=f.sigma)
        kind=="SmoothMetropolisTransition" && return SmoothMetropolisTransition(f.beta;sigma=f.sigma,a=f.a,s=f.s)
        kind=="GaussianMixtureTransition" && return GaussianMixtureTransition(f.beta;sigma=f.sigma,centers=f.centers,weights=f.weights,
            normalization=f.normalization,supremum_bound=f.normalization==:none ? nothing : f.supremum_bound,provenance=f.provenance)
        kind=="TabulatedCKGOFT" && return TabulatedCKGOFT(f.frequencies,f.labels,f.values)
        kind=="PreparedCKGJointKernel" && return PreparedCKGJointKernel(values(f)...)
        throw(ArgumentError("Unsupported prescription tag $kind."))
    end
    throw(ArgumentError("Cannot reconstruct portable tag $tag without its original callable."))
end

function _compiled_replay_evidence(ws)
    transition=ws.cached_cfg.transition_weight
    joint=transition isa PreparedCKGJointKernel ? _portable_pack(transition) : nothing
    return _portable_pack((;G_left=ws.G_left,G_right=ws.G_right,
        dll_lindblads=ws.dll_lindblads,joint))
end

# Reweighting and basis rotation can associate floating-point products differently
# on replay. Input digests remain exact; effective arrays are compared at roundoff.
function _replay_samples_match(a,b)
    if a isa AbstractDict && b isa AbstractDict
        return keys(a)==keys(b) && all(k->_replay_samples_match(a[k],b[k]),keys(a))
    elseif a isa AbstractArray && b isa AbstractArray
        return size(a)==size(b) && all(_replay_samples_match(x,y) for (x,y) in zip(a,b))
    elseif a isa Number && b isa Number && (a isa Union{AbstractFloat,Complex} || b isa Union{AbstractFloat,Complex})
        T=promote_type(typeof(float(real(a))),typeof(float(real(b))))
        return isequal(a,b) || isapprox(a,b;rtol=256eps(T),atol=256eps(T))
    end
    return isequal(a,b)
end

function _research_replay_snapshot(ws, provenance)
    ham=ws.ham_or_trott; cfg=ws.cached_cfg
    raw=(;matrix=copy(ham.data),terms=deepcopy(ham.base_terms),base_coeffs=copy(ham.base_coeffs),
        disordering_terms=deepcopy(ham.disordering_terms),disordering_coeffs=deepcopy(ham.disordering_coeffs),
        eigvals=copy(ham.eigvals),eigvecs=copy(ham.eigvecs),nu_min=ham.nu_min,shift=ham.shift,
        rescaling_factor=ham.rescaling_factor,periodic=ham.periodic)
    p=provenance
    grid=sort!(unique!(vec(ham.bohr_freqs .* ham.rescaling_factor)))
    transition=get(p,:physical_transition,nothing)
    # A joint prescription owns its OFT; never separately persist its callback.
    filter=transition isa CKGJointKernel ? nothing : p.physical_filter
    payload=_record("replay";model=_portable_pack(raw),jumps=_portable_pack([copy(j.data) for j in ws.jumps]),
        source_basis=:computational,weighted_sources=true,beta_phys=_portable_pack(p.beta_phys),
        filter=_portable_pack(filter;grid),transition=_portable_pack(transition;grid),
        domain=string(nameof(typeof(cfg.domain))),construction=cfg.construction isa DLL ? :DLL : :CKG,
        time_step=_portable_pack(get(p,:physical_time_step,nothing)),energy_step=_portable_pack(get(p,:physical_energy_step,nothing)),
        num_energy_bits=cfg.num_energy_bits_D,
        compiled_samples=_compiled_replay_evidence(ws),
        compiled_comparison=:roundoff_256eps_relative_and_absolute,
        hash_scope=:complete_replay_data,hash_algorithm=:crc32c)
    payload[:digest]=_evidence_digest(payload)
    return payload
end

function _runtime_provenance()
    root=dirname(@__DIR__)
    dirty=try
        repo=LibGit2.GitRepo(root)
        try LibGit2.isdirty(repo) finally close(repo) end
    catch
        :unknown
    end
    return (;julia_version=string(VERSION),package_version=string(pkgversion(@__MODULE__)),
        git_revision=_capture_git_hash(),dirty_worktree=dirty,threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),timestamp=string(Dates.now()))
end

_result_type_tag(::GibbsSimulationResult)="gibbs_simulation"
function _result_to_dict(r::GibbsSimulationResult)
    return Dict{Symbol,Any}(:result_type=>"gibbs_simulation",:schema_version=>1,
        :trajectory=>_portable_pack(r.trajectory),:spectrum=>_portable_pack(r.spectrum),
        :diagnostics=>_portable_pack(r.diagnostics),:provenance=>_portable_pack(r.provenance),
        :convergence=>_portable_pack(r.convergence))
end
function _dict_to_gibbs_results(d)
    get(d,:schema_version,0)==1 || throw(ArgumentError("Unsupported Gibbs result schema version."))
    return GibbsSimulationResult((_portable_unpack(d[k]) for k in (:trajectory,:spectrum,:diagnostics,:provenance,:convergence))...)
end
function _write_result_companion_txt(r::GibbsSimulationResult,path::String)
    open(path,"w") do io
        show(io,r); println(io)
        println(io,"Schema: 1; numerical evidence, no certified worst-case mixing claim.")
        if get(r.provenance,:evolution,:lindbladian)==:channel
            println(io,"Channel evidence only; continue with original channel inputs and the saved final density matrix.")
        else
            println(io,"Rebuild with Workspace(result); unregistered callbacks require explicit resupply.")
        end
    end
end

"""
    Workspace(result::GibbsSimulationResult; filters=Dict(), max_bytes=256*1024^2)

Rebuild fresh mutable workspaces from a saved, hash-checked input snapshot.
`filters[(name, version)]` supplies an unregistered DLL filter; `filters[:ckg_joint]`
supplies a custom CKG joint prescription. Saved sample checks detect changes on
those grids only; they do not identify a function globally. Temperature, domain
and transform controls are fixed by the snapshot: use new physical inputs to
change them. No FFT/backend plans or mutable caches are loaded.
"""
function Workspace(r::GibbsSimulationResult;filters=Dict(),max_bytes::Integer=256*1024^2)
    get(r.provenance,:evolution,:lindbladian)==:channel && throw(ArgumentError(
        "Channel evidence cannot rebuild a Lindbladian workspace. Continue with original channel inputs and rho0=result.trajectory.rho_final."))
    haskey(r.provenance,:replay) || throw(ArgumentError("This result has no physical-input replay snapshot; rebuild from original inputs."))
    d=r.provenance.replay
    body=Dict(k=>v for (k,v) in d if k!=:digest)
    _evidence_digest(body)==d[:digest] || throw(ArgumentError("Saved model/source/filter/control digest mismatch; rebuild from original inputs."))
    unpack(x)=_portable_unpack(x;restore=true,filters)
    domain=_string_to_domain(d[:domain])
    ws=Workspace(unpack(d[:model]);beta_phys=unpack(d[:beta_phys]),jumps=unpack(d[:jumps]),
        filter=unpack(d[:filter]),transition_weight=unpack(d[:transition]),
        construction=d[:construction]==:DLL ? DLL() : KMS(),domain,
        basis=:computational,time_step=unpack(d[:time_step]),energy_step=unpack(d[:energy_step]),
        num_energy_bits=domain isa BohrDomain ? nothing : d[:num_energy_bits],max_bytes)
    _replay_samples_match(_compiled_replay_evidence(ws),d[:compiled_samples]) ||
        throw(ArgumentError("Recompiled generator samples differ from the saved run; callbacks, transforms or numerical runtime changed. Start a new simulation to accept these changes."))
    provenance=merge(ws.research_provenance,r.provenance,(;replay=deepcopy(d)))
    provenance=(;(k=>v for (k,v) in pairs(provenance) if k!=:portable_provenance)...)
    provenance=merge(provenance,(;portable_provenance=_portable_pack(provenance;grid=:metadata)))
    return typeof(ws)((getfield(ws,i) for i in 1:fieldcount(typeof(ws))-1)...,provenance)
end

"""
    simulate_gibbs(result::GibbsSimulationResult; times, filters=Dict(), kwargs...)

Continue from the saved final density matrix, rebuilding a fresh workspace.
`times` starts at zero and measures additional time in the original generator
clock. The returned trajectory covers this segment; `provenance.resume_time_origin`
records cumulative prior elapsed simulation time. Solver settings default to the
saved settings and may be overridden. This is state continuation, not restoration
of an interrupted internal Arnoldi iteration.
"""
function simulate_gibbs(r::GibbsSimulationResult;times,filters=Dict(),kwargs...)
    _validate_time_grid(times;require_zero=true)
    controls=get(r.provenance,:solver_controls,(;))
    ws=Workspace(r;filters,max_bytes=get(kwargs,:max_bytes,get(controls,:max_bytes,256*1024^2)))
    # The new segment owns its horizon; extensions are opt-in again.
    controls=merge(controls,(;max_extensions=0,max_time=last(times)),(;kwargs...))
    result=simulate_gibbs(ws;times,rho0=r.trajectory.rho_final,basis=r.trajectory.basis,controls...)
    origin=get(r.provenance,:resume_time_origin,zero(last(r.trajectory.t)))+last(r.trajectory.t)
    provenance=merge(result.provenance,(;resume_time_origin=origin,resume_scope=:final_state_continuation))
    return GibbsSimulationResult(result.trajectory,result.spectrum,result.diagnostics,provenance,result.convergence)
end
