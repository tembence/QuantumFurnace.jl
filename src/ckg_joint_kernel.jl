"""
    CKGJointKernel(beta; oft, rate, frequency_window, panels=32,
        rtol=1e-10, balance_rtol=1e-8, max_bohr_frequencies=65,
        max_bytes=64*1024^2, maxevals=100000, structural_provenance=nothing)

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
"""
struct CKGJointKernel{T<:AbstractFloat,F,G,P} <: AbstractCKGTransition
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
end
function CKGJointKernel(beta::Real;oft,rate,frequency_window,panels::Integer=32,
    rtol::Real=1e-10,balance_rtol::Real=1e-8,max_bohr_frequencies::Integer=65,
    max_bytes::Integer=64*1024^2,maxevals::Integer=100000,structural_provenance=nothing)
    b=float(beta); T=typeof(b); lo,hi=T.(frequency_window)
    isfinite(b)&&b>0 && isfinite(lo)&&isfinite(hi)&&hi>lo ||
        throw(ArgumentError("Require finite positive beta and finite lower < upper frequency_window."))
    1<=panels<=100000 && 0<rtol<balance_rtol<1 && max_bohr_frequencies>0 && max_bytes>0 && maxevals>0 ||
        throw(ArgumentError("Require positive budgets, 1<=panels<=100000 and 0<rtol<balance_rtol<1."))
    applicable(oft,zero(T)) && applicable(rate,zero(T)) || throw(ArgumentError("oft and rate must accept a real frequency."))
    CKGJointKernel(b,oft,rate,(lo,hi),Int(panels),T(rtol),T(balance_rtol),
        Int(max_bohr_frequencies),Int(max_bytes),Int(maxevals),structural_provenance)
end

# Full complex OFT samples, indexed by outer label and Bohr frequency. No
# Gaussian folding and no callback evaluation is permitted in hot actions.
struct TabulatedCKGOFT{T<:AbstractFloat} <: AbstractFilter
    frequencies::Dict{T,Int}
    labels::Dict{T,Int}
    values::Matrix{Complex{T}}
end
struct PreparedCKGJointKernel{T<:AbstractFloat,P} <: AbstractCKGTransition
    beta::T
    oft::TabulatedCKGOFT{T}
    alpha::Matrix{Complex{T}}
    energy_labels::Vector{T}
    energy_rates::Vector{T}
    evidence::P
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
    PreparedCKGJointKernel(r.beta,oft,alpha,labels,rates,evidence)
end

function _prepare_joint_ckg_inputs(base,r,domain,num_energy_bits,energy_step)
    ham=base.hamiltonian; T=eltype(ham.eigvals); R=T(ham.rescaling_factor)
    # Existing Energy actions are Float64-only; never silently narrow precision.
    T===Float64 || throw(ArgumentError("General CKG Bohr/Energy facade currently requires Float64 Hamiltonian data."))
    algorithm=CKGJointKernel(base.config.beta;oft=x->sqrt(R)*r.oft(R*x),rate=x->r.rate(R*x),
        frequency_window=r.frequency_window./R,panels=r.panels,rtol=r.rtol,balance_rtol=r.balance_rtol,
        max_bohr_frequencies=r.max_bohr_frequencies,max_bytes=r.max_bytes,maxevals=r.maxevals,
        structural_provenance=r.structural_provenance)
    step=energy_step===nothing ? nothing : T(energy_step/R)
    # Guard the register before _create_energy_labels allocates it.
    domain isa EnergyDomain && big(2)^num_energy_bits*128>r.max_bytes &&
        throw(ArgumentError("Energy register exceeds joint-kernel max_bytes before allocation."))
    labels=domain isa EnergyDomain ? _create_energy_labels(num_energy_bits,step) : nothing
    compiled=compile_ckg_kernel(algorithm,keys(ham.bohr_dict);energy_labels=labels,energy_step=step)
    cfg=Config(;sim=Lindbladian(),domain,construction=KMS(),num_qubits=base.config.num_qubits,
        beta=base.config.beta,beta_phys=base.config.beta_phys,sigma=inv(base.config.beta),
        filter=compiled.oft,transition_weight=compiled,with_linear_combination=false,num_energy_bits_D=num_energy_bits,w0_D=step)
    validate_config!(cfg,ham)
    provenance=merge(base.provenance,(;construction=:CKG_KMS,physical_filter=r.oft,algorithm_filter=compiled.oft,
        physical_transition=r,algorithm_transition=compiled,filter_evidence=(compiled.evidence,),
        physical_filters=((;family=:CKGJointOFT,frame=:physical),),
        algorithm_filters=((;family=:CKGJointOFT,frame=:algorithm),),
        sigma_role=:unused_legacy_placeholder,physical_energy_step=energy_step,algorithm_energy_step=step,
        rate_normalization=:none,rate_divisor=one(T),ckg_frame_transform=:normalized_oft_measure_cancels))
    (;hamiltonian=ham,config=cfg,jumps=base.jumps,provenance)
end
