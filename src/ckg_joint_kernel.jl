"""
    CKGJointKernel(beta; oft, rate, frequency_window, panels=32,
        rtol=1e-10, balance_rtol=1e-8, max_bohr_frequencies=65,
        max_bytes=64*1024^2, maxevals=100000, structural_provenance=nothing, time_transform=(;))

Joint CKG prescription in physical coordinates for the research facade.
`oft(x)` is the **full normalised** transform C(x), with integral |C|² = 1;
`rate(w)` is gamma(w), without an implicit supremum normalisation.

Compilation checks every pair in the complete supplied Bohr set against an
independent adaptive integral over the real line. Positive composite Gauss
quadrature retains a PSD Gram matrix. `frequency_window` controls outer tails;
`panels` controls its spacing. EnergyDomain instead uses its explicit energy
register. Both must agree with the adaptive reference within `balance_rtol`.
Error estimates and sampled positivity/balance are numerical evidence, not
functional proofs, tail certificates, or efficient-implementation theorems.
`structural_provenance` is caller-supplied source information, never a bypass.
Callbacks are sampled during preparation only; prepared kernels own their data.
The explicit Bohr-set and byte caps apply before quadratic allocations.

For TimeDomain, `time_transform` supplies positive physical radii
`frequency_window`, `coherent_frequency_window`, `coherent_time_window` and
`coherent_time_step`; frequency grid counts are `frequency_grid_size` and
`coherent_frequency_grid_size` (default 257). `backend=:direct` or `:finufft`
selects Fourier sums/contraction. The facade's `time_step`/`num_energy_bits`
control the independent dissipative time grid. Coarse transforms reject against
all finite Bohr pairs. This finite-window conversion gives no continuum-tail
or quantum-implementation guarantee; custom Trotter remains unsupported.
"""
struct CKGJointKernel{T<:AbstractFloat,F,G,P,Q} <: AbstractCKGTransition
    beta::T
    oft::F
    rate::G
    frequency_window::Tuple{T,T}
    panels::Int
    rtol::T
    balance_rtol::T
    max_bohr_frequencies::Int
    max_bytes::Int
    maxevals::Int
    structural_provenance::P
    time_transform::Q
end
function CKGJointKernel(beta::Real;oft,rate,frequency_window,panels::Integer=32,
    rtol::Real=1e-10,balance_rtol::Real=1e-8,max_bohr_frequencies::Integer=65,
    max_bytes::Integer=64*1024^2,maxevals::Integer=100000,structural_provenance=nothing,time_transform::NamedTuple=(;))
    b=float(beta); T=typeof(b); lo,hi=T.(frequency_window)
    isfinite(b)&&b>0 && isfinite(lo)&&isfinite(hi)&&hi>lo ||
        throw(ArgumentError("Require finite positive beta and finite lower < upper frequency_window."))
    1<=panels<=100000 && 0<rtol<balance_rtol<1 && max_bohr_frequencies>0 && max_bytes>0 && maxevals>0 ||
        throw(ArgumentError("Require positive budgets, 1<=panels<=100000 and 0<rtol<balance_rtol<1."))
    applicable(oft,zero(T)) && applicable(rate,zero(T)) || throw(ArgumentError("oft and rate must accept a real frequency."))
    CKGJointKernel(b,oft,rate,(lo,hi),Int(panels),T(rtol),T(balance_rtol),
        Int(max_bohr_frequencies),Int(max_bytes),Int(maxevals),structural_provenance,_ckg_time_controls(time_transform,T))
end

# Full complex OFT samples, indexed by outer label and Bohr frequency. No
# Gaussian folding and no callback evaluation is permitted in hot actions.
struct TabulatedCKGOFT{T<:AbstractFloat} <: AbstractFilter
    frequencies::Dict{T,Int}
    labels::Dict{T,Int}
    values::Matrix{Complex{T}}
end
struct PreparedCKGJointKernel{T<:AbstractFloat,P,Q} <: AbstractCKGTransition
    beta::T
    oft::TabulatedCKGOFT{T}
    alpha::Matrix{Complex{T}}
    energy_labels::Vector{T}
    energy_rates::Vector{T}
    evidence::P
    time_data::Q
end
transition_alpha(r::PreparedCKGJointKernel,u::Real,v::Real)=
    r.alpha[r.oft.frequencies[iszero(u) ? zero(u) : u],r.oft.frequencies[iszero(v) ? zero(v) : v]]
transition_value(r::PreparedCKGJointKernel,w::Real)=r.energy_rates[r.oft.labels[w]]
_transition_divisor(r::Union{CKGJointKernel,PreparedCKGJointKernel})=one(r.beta)
_is_joint_ckg(cfg)=cfg.transition_weight isa PreparedCKGJointKernel

function _joint_rate(r,w)
    value=r.rate(w)
    value isa Real && isfinite(value) && value>=0 ||
        throw(ArgumentError("CKG rate must be real, finite and nonnegative at every evaluated frequency (failed at $w)."))
    converted=typeof(r.beta)(value)
    isfinite(converted) || throw(ArgumentError("CKG rate overflows the working precision."))
    converted
end
transition_value(r::CKGJointKernel,w::Real)=_joint_rate(r,w)
function _joint_oft(r,w)
    value=r.oft(w)
    value isa Number && isfinite(value) || throw(ArgumentError("CKG OFT must be finite at every evaluated frequency (failed at $w)."))
    converted=Complex{typeof(r.beta)}(value)
    isfinite(converted) || throw(ArgumentError("CKG OFT overflows the working precision."))
    converted
end
function _joint_gauss_rule(r)
    T=typeof(r.beta); x,w=QuadGK.gauss(T,8)
    lo,hi=r.frequency_window; dx=(hi-lo)/r.panels
    nodes=T[]; weights=T[]
    for k in 1:r.panels,j in eachindex(x)
        push!(nodes,lo+(k-T(0.5))*dx+x[j]*dx/2)
        push!(weights,w[j]*dx/2)
    end
    return nodes,weights
end

"""
    compile_ckg_kernel(kernel, frequencies; strict=true, energy_labels=nothing,
        energy_step=nothing)

Bounded finite-Bohr compilation. The frequency set must contain zero and all
negatives. `strict=false` returns failed evidence for inspection (`:not_KMS`
versus `:quadrature_unresolved`); failed kernels cannot enter the KMS facade.
This does not establish identities away from the supplied frequencies.
"""
function compile_ckg_kernel(r::CKGJointKernel,frequencies;strict::Bool=true,
    energy_labels=nothing,energy_step=nothing)
    T=typeof(r.beta)
    # Count before collecting an unbounded/quadratic object.
    length(frequencies)<=r.max_bohr_frequencies || throw(ArgumentError(
        "Complete Bohr set exceeds max_bohr_frequencies=$(r.max_bohr_frequencies); no quadratic kernel allocated."))
    nus=sort!(unique(T[iszero(u) ? zero(T) : T(u) for u in frequencies]))
    !isempty(nus)&&all(isfinite,nus)&&zero(T) in nus&&all(u->-u in nus,nus) ||
        throw(ArgumentError("Supply the complete finite reflection-closed Bohr set including zero."))
    m=length(nus)
    n=energy_labels===nothing ? big(8)*r.panels : big(length(energy_labels))
    bytes=big(128)*(m*m+n*m+n)
    bytes<=r.max_bytes || throw(ArgumentError("Joint-kernel working-set estimate $bytes exceeds max_bytes=$(r.max_bytes)."))
    labels,weights=if energy_labels===nothing
        energy_step===nothing || throw(ArgumentError("energy_step requires energy_labels."))
        _joint_gauss_rule(r)
    else
        energy_step isa Real && isfinite(energy_step)&&energy_step>0 || throw(ArgumentError("Require positive finite energy_step."))
        (T.(energy_labels),fill(T(energy_step),length(energy_labels)))
    end
    !isempty(labels)&&all(isfinite,labels)&&length(unique(labels))==length(labels) ||
        throw(ArgumentError("Outer labels must be nonempty, finite and distinct."))
    # Both transform normalisation and reference integration have a bounded
    # evaluation budget. QuadGK's returned estimates are not rigorous bounds.
    normC,normerr=quadgk(x->abs2(_joint_oft(r,x)),-T(Inf),zero(T),T(Inf);
        rtol=r.rtol,atol=zero(T),maxevals=r.maxevals)
    isfinite(normC)&&isfinite(normerr)&&abs(normC-1)<=r.balance_rtol&&normerr<=r.rtol ||
        throw(ArgumentError("Full OFT must have integral |C|²=1 with resolved quadrature; got $normC (estimate $normerr). Supply its normalisation explicitly."))
    vals=Complex{T}[_joint_oft(r,w-u) for w in labels,u in nus]
    rates=T[_joint_rate(r,w) for w in labels]
    factors=vals .* (sqrt.(weights).*sqrt.(rates))
    all(isfinite,factors) || throw(ArgumentError("Nonfinite Gram factors; rescale rate/energy units explicitly."))
    # alpha_ij = sum F_ki conj(F_kj), not F'F for complex C.
    alpha=transpose(factors)*conj(factors)
    all(isfinite,alpha) || throw(ArgumentError("Nonfinite implemented alpha; rescale rate/energy units explicitly."))
    reference=similar(alpha); qerr=zero(T)
    for j in 1:m,i in 1:j
        value,err=quadgk(w->_joint_rate(r,w)*_joint_oft(r,w-nus[i])*conj(_joint_oft(r,w-nus[j])),
            -T(Inf),zero(T),T(Inf);rtol=r.rtol,atol=zero(T),maxevals=r.maxevals)
        isfinite(value)&&isfinite(err) || throw(ArgumentError("Nonfinite joint reference integral; check integrability and callback scales."))
        reference[i,j]=value; reference[j,i]=conj(value)
        qerr=max(qerr,err)
    end
    scale=maximum(abs,reference)
    scale>0 && isfinite(scale) || throw(ArgumentError("Joint kernel has zero or nonfinite reference scale; balance is unresolved."))
    # PHYSICS CHECK: use a bounded tilt, so low-temperature overflow cannot
    # manufacture a passing identity. Opposite entries use swapped indices.
    function defect(a)
        worst=zero(T)
        for j in 1:m,i in 1:m
            exponent=(r.beta*nus[i]/2)+(r.beta*nus[j]/2)
            isfinite(exponent) || throw(ArgumentError("Thermal exponent exceeds the working precision; rescale inputs."))
            lhs=exp(min(-exponent,zero(T)))*a[m+1-j,m+1-i]
            rhs=exp(min(exponent,zero(T)))*a[i,j]
            worst=max(worst,abs(lhs-rhs))
        end
        worst/scale
    end
    reference_defect=defect(reference); implemented_defect=defect(alpha)
    integration_error=maximum(abs,alpha-reference)/scale
    all(isfinite,(reference_defect,implemented_defect,integration_error,qerr/scale)) ||
        throw(ArgumentError("Nonfinite joint validation metric; no KMS acceptance is possible."))
    resolved=qerr/scale<=r.rtol
    reference_pass=resolved&&reference_defect<=r.balance_rtol
    status=!resolved ? :reference_unresolved : !reference_pass ? :not_KMS :
        integration_error>r.balance_rtol || implemented_defect>r.balance_rtol ? :quadrature_unresolved : :pass
    evidence=(;status,balance=:complete_finite_bohr_numerical_evidence,
        frequency_count=m,balance_defect_metric=:bounded_reflection_maxabs_over_global_alpha_scale,
        maxevals_per_integral=r.maxevals,adaptive_integral_count=1+m*(m+1)÷2,
        reference_balance_defect=reference_defect,
        implemented_balance_defect=implemented_defect,relative_integration_difference=integration_error,
        relative_adaptive_error_estimate=qerr/scale,rtol=r.rtol,balance_rtol=r.balance_rtol,
        oft_norm=normC,oft_norm_error_estimate=normerr,
        positivity=:positive_quadrature_gram,rate_nonnegativity=:evaluated_points_only,
        outer_rule=energy_labels===nothing ? :composite_gauss8 : :explicit_energy_rectangle,
        outer_points=length(labels),frequency_window=r.frequency_window,
        retained_outer_interval=extrema(labels),energy_step,
        estimated_working_bytes=bytes,coherent=:canonical_from_implemented_loss,
        structural_provenance=r.structural_provenance,global_balance=:not_established_by_samples,
        tails=:adaptive_reference_estimate_not_certificate,callback_portability=:samples_only)
    strict&&status!=:pass && throw(ArgumentError(
        "CKG joint kernel $status: reference balance=$reference_defect, implemented balance=$implemented_defect, grid/reference difference=$integration_error, adaptive estimate=$(qerr/scale). Refine frequency_window/panels (Energy: energy_step/num_energy_bits), or correct the OFT/rate pair; inspect with compile_ckg_kernel(...; strict=false)."))
    oft=TabulatedCKGOFT(Dict(u=>i for (i,u) in enumerate(nus)),Dict(w=>i for (i,w) in enumerate(labels)),vals)
    PreparedCKGJointKernel(r.beta,oft,alpha,labels,rates,evidence,nothing)
end

function _prepare_joint_ckg_inputs(base,r,domain,num_energy_bits,energy_step,time_step)
    ham=base.hamiltonian; T=eltype(ham.eigvals); R=T(ham.rescaling_factor)
    # Existing Energy actions are Float64-only; never silently narrow precision.
    T===Float64 || throw(ArgumentError("General CKG Bohr/Energy/Time facade currently requires Float64 Hamiltonian data."))
    algorithm=CKGJointKernel(base.config.beta;oft=x->sqrt(R)*r.oft(R*x),rate=x->r.rate(R*x),
        frequency_window=r.frequency_window./R,panels=r.panels,rtol=r.rtol,balance_rtol=r.balance_rtol,
        max_bohr_frequencies=r.max_bohr_frequencies,max_bytes=r.max_bytes,maxevals=r.maxevals,
        structural_provenance=r.structural_provenance,
        time_transform=_ckg_scale_time_controls(r.time_transform,R))
    step=energy_step===nothing ? nothing : T(energy_step/R)
    # Guard the register before _create_energy_labels allocates it.
    domain isa EnergyDomain && big(2)^num_energy_bits*128>r.max_bytes &&
        throw(ArgumentError("Energy register exceeds joint-kernel max_bytes before allocation."))
    labels=domain isa EnergyDomain ? _create_energy_labels(num_energy_bits,step) : nothing
    compiled=compile_ckg_kernel(algorithm,keys(ham.bohr_dict);energy_labels=labels,energy_step=step)
    domain isa TimeDomain && (compiled=_compile_ckg_time(algorithm,compiled,T(time_step*R),num_energy_bits,length(ham.eigvals)))
    cfg=Config(;sim=Lindbladian(),domain,construction=KMS(),num_qubits=base.config.num_qubits,
        beta=base.config.beta,beta_phys=base.config.beta_phys,sigma=inv(base.config.beta),
        filter=compiled.oft,transition_weight=compiled,with_linear_combination=false,num_energy_bits_D=num_energy_bits,w0_D=step,
        t0_D=time_step===nothing ? nothing : T(time_step*R))
    validate_config!(cfg,ham)
    provenance=merge(base.provenance,(;construction=:CKG_KMS,physical_filter=r.oft,algorithm_filter=compiled.oft,
        physical_transition=r,algorithm_transition=compiled,filter_evidence=(compiled.evidence,),
        physical_filters=((;family=:CKGJointOFT,frame=:physical),),
        algorithm_filters=((;family=:CKGJointOFT,frame=:algorithm),),
        sigma_role=:unused_legacy_placeholder,physical_energy_step=energy_step,algorithm_energy_step=step,
        physical_time_step=time_step,algorithm_time_step=cfg.t0_D,
        rate_normalization=:none,rate_divisor=one(T),ckg_frame_transform=:normalized_oft_measure_cancels))
    (;hamiltonian=ham,config=cfg,jumps=base.jumps,provenance)
end

# Independent physical-grid controls. Missing values reject only when Time is
# requested; Bohr/Energy never invent a time window for a callback.
function _ckg_time_controls(options,::Type{T}) where {T}
    defaults=(;frequency_window=nothing,frequency_grid_size=257,
        coherent_frequency_window=nothing,coherent_frequency_grid_size=257,
        coherent_time_window=nothing,coherent_time_step=nothing,backend=:direct)
    all(k->haskey(defaults,k),keys(options)) || throw(ArgumentError("Unknown CKG time_transform control."))
    c=merge(defaults,options)
    for key in (:frequency_window,:coherent_frequency_window,:coherent_time_window,:coherent_time_step)
        x=getproperty(c,key)
        x===nothing || (x isa Real && isfinite(T(x)) && T(x)>0) || throw(ArgumentError("CKG $key must be positive and finite."))
    end
    for key in (:frequency_grid_size,:coherent_frequency_grid_size)
        x=getproperty(c,key)
        x isa Integer && 3<=x<=100000 || throw(ArgumentError("CKG $key must be an integer in 3:100000."))
    end
    c.backend in (:direct,:finufft) || throw(ArgumentError("CKG transform backend must be :direct or :finufft."))
    c
end
function _ckg_scale_time_controls(c,R)
    scale(x,a)=x===nothing ? nothing : x*a
    merge(c,(;frequency_window=scale(c.frequency_window,inv(R)),
        coherent_frequency_window=scale(c.coherent_frequency_window,inv(R)),
        coherent_time_window=scale(c.coherent_time_window,R),coherent_time_step=scale(c.coherent_time_step,R)))
end
function _ckg_time_bytes(r,bits,n,m,d)
    c=r.time_transform
    all(k->getproperty(c,k)!==nothing,(:frequency_window,:coherent_frequency_window,:coherent_time_window,:coherent_time_step)) ||
        throw(ArgumentError("CKG Time needs explicit time_transform frequency_window, coherent_frequency_window, coherent_time_window and coherent_time_step. All are physical-frame controls in the facade."))
    ratio=big(c.coherent_time_window)/big(c.coherent_time_step)
    ratio<=100000 || throw(ArgumentError("CKG coherent time grid exceeds 200001 points."))
    nt=big(2)^bits; nc=2floor(BigInt,ratio)+1
    nf=big(c.frequency_grid_size); ng=big(c.coherent_frequency_grid_size)
    # Includes frequency/time kernels, matrix products, prefactors and the
    # direct/FINUFFT contraction scratch, before any quadratic allocation.
    big(128)*(nt*nf+n*nt+n*m+n*d*d+ng*ng+nc*ng+nc*nc+n*ng+nc*m+d^3)
end
function _ckg_trapezoid(window,n,::Type{T}) where {T}
    x=collect(range(-T(window),T(window);length=n))
    weights=fill(T(2window/(n-1)),n); weights[1]/=2; weights[end]/=2
    x,weights
end

function _compile_ckg_time(r,reference,step,bits,dimension)
    T=typeof(r.beta); CT=Complex{T}; c=r.time_transform
    nus=sort!(collect(keys(reference.oft.frequencies)))
    labels,weights=_joint_gauss_rule(r); m=length(nus)
    bytes=_ckg_time_bytes(r,bits,length(labels),m,dimension)+reference.evidence.estimated_working_bytes
    bytes<=r.max_bytes || throw(ArgumentError("CKG Time working-set estimate $bytes exceeds max_bytes=$(r.max_bytes); reduce explicit grids or increase the budget."))
    nt=2^bits; times=T.(collect(-nt÷2:nt÷2-1)).*step
    x,wx=_ckg_trapezoid(c.frequency_window,c.frequency_grid_size,T)
    # f=psi/sqrt(2pi): C(x)=integral f(t) exp(-i*x*t)dt.
    ft=fourier_sum(x,CT[_joint_oft(r,u)*w/(2T(pi)) for (u,w) in zip(x,wx)],times;backend=c.backend)
    vals=Matrix{CT}(undef,length(labels),m)
    for (i,w) in enumerate(labels)
        vals[i,:]=fourier_sum(times,ft.*step,w.-nus;sign=-1,backend=c.backend)
    end
    rates=reference.energy_rates.*weights
    factors=vals.*sqrt.(rates)
    alpha=transpose(factors)*conj(factors)
    scale=maximum(abs,reference.alpha)
    dissipative_difference=maximum(abs,alpha-reference.alpha)/scale

    # General two-frequency kernel, including the retained outer omega
    # quadrature. No b_minus/b_plus factorisation is assumed.
    nu,wnu=_ckg_trapezoid(c.coherent_frequency_window,c.coherent_frequency_grid_size,T)
    vf=CT[_joint_oft(r,w-u)*sqrt(q) for (w,q) in zip(labels,rates),u in nu]
    a=transpose(vf)*conj(vf)
    ghat=CT[tanh(r.beta*(v-u)/4)*a[i,j]/(2im) for (i,u) in enumerate(nu),(j,v) in enumerate(nu)]
    nc=floor(Int,c.coherent_time_window/c.coherent_time_step)
    tc=T.(collect(-nc:nc)).*T(c.coherent_time_step)
    phi=CT[cis(-u*t)*w for t in tc,(u,w) in zip(nu,wnu)]
    gtt=(phi*ghat*adjoint(phi))/(2T(pi))^2
    # Full two-dimensional forward transform on every actual Bohr pair.
    # This tests the independent coherent grid against the frequency reference.
    e=CT[cis(u*t)*T(c.coherent_time_step) for u in nus,t in tc]
    recovered=e*gtt*adjoint(e)
    target=CT[tanh(r.beta*(v-u)/4)*reference.alpha[i,j]/(2im) for (i,u) in enumerate(nus),(j,v) in enumerate(nus)]
    coherent_difference=maximum(abs,recovered-target)/scale
    balance_defect=zero(T)
    for j in 1:m,i in 1:m
        exponent=r.beta*(nus[i]+nus[j])/2
        isfinite(exponent) || throw(ArgumentError("Nonfinite thermal exponent in Time validation."))
        lhs=exp(min(-exponent,zero(T)))*alpha[m+1-j,m+1-i]
        rhs=exp(min(exponent,zero(T)))*alpha[i,j]
        balance_defect=max(balance_defect,abs(lhs-rhs)/scale)
    end
    all(isfinite,alpha)&&all(isfinite,gtt)&&all(isfinite,(dissipative_difference,coherent_difference)) ||
        throw(ArgumentError("Nonfinite CKG Time transform; inspect callback scales and grids."))
    max(dissipative_difference,coherent_difference,balance_defect)<=r.balance_rtol || throw(ArgumentError(
        "CKG Time quadrature_unresolved: dissipative alpha difference=$dissipative_difference, coherent kernel difference=$coherent_difference, implemented balance defect=$balance_defect (tolerance $(r.balance_rtol)). Refine dissipative time_step/num_energy_bits and time_transform frequency window/grid independently from coherent time/frequency controls."))
    evidence=merge(reference.evidence,(;coherent=:two_time_joint_kernel,
        frequency_reference_evidence=reference.evidence,
        implemented_balance_defect=balance_defect,
        relative_integration_difference=dissipative_difference,
        integration_comparator=:retained_frequency_gram,
        adaptive_estimate_scope=:frequency_reference_only,
        realization=:numerical_inverse_and_forward_transforms,
        relative_dissipative_time_difference=dissipative_difference,
        relative_coherent_time_difference=coherent_difference,
        time_transform=c,dissipative_time_step=step,dissipative_time_points=nt,
        coherent_time_points=length(tc),estimated_working_bytes=bytes,
        tails=:finite_window_reference_comparison_not_certificate,
        quantum_implementation=:not_established))
    oft=TabulatedCKGOFT(copy(reference.oft.frequencies),copy(reference.oft.labels),vals)
    data=(;g_tt=gtt,times=tc,step=T(c.coherent_time_step),backend=c.backend,weights)
    PreparedCKGJointKernel(r.beta,oft,alpha,labels,copy(reference.energy_rates),evidence,data)
end

function _precompute_ckg_time_data(config,ham)
    r=config.transition_weight
    r.time_data===nothing && throw(ArgumentError("Joint Time kernel has no time compilation."))
    d=length(ham.eigvals); T=eltype(ham.eigvals)
    values=Array{Complex{T}}(undef,d,d,length(r.energy_labels))
    for (k,w) in enumerate(r.energy_labels),j in 1:d,i in 1:d
        u=ham.bohr_freqs[i,j]
        values[i,j,k]=r.oft.values[k,r.oft.frequencies[iszero(u) ? zero(T) : u]]*sqrt(r.time_data.weights[k])
    end
    prefactors=NUFFTPrefactors(values,copy(r.energy_labels),copy(r.oft.labels))
    (;transition=pick_transition(config),gamma_norm_factor=one(T),energy_labels=r.energy_labels,
        oft_nufft_prefactors=prefactors,b_minus=nothing,b_plus=nothing,oft_domain_prefactor=one(T))
end
