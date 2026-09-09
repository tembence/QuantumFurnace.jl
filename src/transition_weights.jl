"""A CKG transition prescription tied to a normalised Gaussian OFT."""
abstract type AbstractCKGTransition end

function _check_ckg_scales(beta,sigma)
    all(x->isfinite(x)&&x>0,(beta,sigma,beta^2,sigma^2,beta^2*sigma^2)) ||
        throw(ArgumentError("beta/sigma or their squared scales are not representable at this precision; rescale physical energy units explicitly."))
    nothing
end

"""Gaussian CKG rate. `sigma` is the OFT width; the centre is fixed by KMS."""
struct GaussianTransition{T<:AbstractFloat} <: AbstractCKGTransition
    beta::T
    sigma::T
    sigma_gamma::T
    function GaussianTransition(beta::Real; sigma::Real=inv(beta), sigma_gamma::Real=sigma)
        b,e,g=promote(float(beta),float(sigma),float(sigma_gamma))
        all(x->isfinite(x)&&x>0,(b,e,g)) || throw(ArgumentError("beta, sigma and sigma_gamma must be finite and positive."))
        _check_ckg_scales(b,e)
        all(x->isfinite(x)&&x>0,(g^2,e^2+g^2,b*(e^2+g^2)/2,2g^2)) ||
            throw(ArgumentError("Gaussian variance and centre must be representable at this precision."))
        new{typeof(b)}(b,e,g)
    end
end

"""Shifted CKG Metropolis rate, exp(-beta*max(w+beta*sigma^2/2,0))."""
struct MetropolisTransition{T<:AbstractFloat} <: AbstractCKGTransition
    beta::T
    sigma::T
    function MetropolisTransition(beta::Real; sigma::Real=inv(beta))
        b,e=promote(float(beta),float(sigma))
        all(x->isfinite(x)&&x>0,(b,e)) || throw(ArgumentError("beta and sigma must be finite and positive."))
        _check_ckg_scales(b,e)
        new{typeof(b)}(b,e)
    end
end

"""Legacy smooth-Metropolis CKG rate, with dimensionless a>=0 and s>0."""
struct SmoothMetropolisTransition{T<:AbstractFloat} <: AbstractCKGTransition
    beta::T
    sigma::T
    a::T
    s::T
    function SmoothMetropolisTransition(beta::Real; sigma::Real=inv(beta),a::Real=0,s::Real=0.25)
        b,e,aa,ss=promote(float(beta),float(sigma),float(a),float(s))
        all(isfinite,(b,e,aa,ss)) && b>0 && e>0 && aa>=0 && ss>0 ||
            throw(ArgumentError("Require positive finite beta, sigma, s and finite a>=0."))
        _check_ckg_scales(b,e)
        all(isfinite,(b*(4aa+1),b*e^2*ss)) || throw(ArgumentError("Smooth-Metropolis scales overflow this precision."))
        new{typeof(b)}(b,e,aa,ss)
    end
end

"""
    GaussianMixtureTransition(beta; sigma=1/beta, centers, weights,
        normalization=:none, supremum_bound=nothing)

Finite nonnegative mixture of Gaussian rates centred at `-centers[k]` with
variance `2centers[k]/beta-sigma^2`. CKG Corollary II.3 ties every component
to the same beta and normalised Gaussian OFT. Require strictly positive
variance; zero-width endpoint distributions are unsupported.

`:none` preserves the supplied weights. `:bound` divides the entire generator
by an explicitly supplied, fixed `supremum_bound >= sum(weights)`; this is an
upper bound, generally not the exact supremum. No sampled maximum is used.
"""
struct GaussianMixtureTransition{T<:AbstractFloat,N,P} <: AbstractCKGTransition
    beta::T
    sigma::T
    centers::NTuple{N,T}
    weights::NTuple{N,T}
    normalization::Symbol
    supremum_bound::T
    provenance::P
    function GaussianMixtureTransition(beta::Real; sigma::Real=inv(beta),centers,weights,
        normalization::Symbol=:none,supremum_bound=nothing,provenance=(;kind=:finite_mixture))
        length(centers)==length(weights)>0 || throw(ArgumentError("centers and weights must have equal nonzero length."))
        b,e=promote(float(beta),float(sigma))
        T=typeof(b)
        all(x->isfinite(x)&&x>0,(b,e)) || throw(ArgumentError("beta and sigma must be positive and finite."))
        _check_ckg_scales(b,e)
        xs=Tuple(T(x) for x in centers); ws=Tuple(T(w) for w in weights)
        # Gaussian identity: beta = 2*x/(sigma^2 + variance).
        all(x->isfinite(x)&&x>b*e^2/2 && isfinite(2x/b-e^2) && 2x/b-e^2>0,xs) ||
            throw(ArgumentError("Each centre must exceed beta*sigma^2/2 with finite positive variance."))
        all(w->isfinite(w)&&w>=0,ws) && isfinite(sum(ws)) && sum(ws)>0 ||
            throw(ArgumentError("Weights must be finite and nonnegative with finite positive sum."))
        normalization in (:none,:bound) || throw(ArgumentError("normalization must be :none or :bound."))
        normalization==:none && supremum_bound!==nothing && throw(ArgumentError("supremum_bound requires normalization=:bound."))
        bound=normalization==:none ? one(T) : supremum_bound===nothing ?
            throw(ArgumentError("normalization=:bound requires an explicit fixed supremum_bound.")) : T(supremum_bound)
        isfinite(bound)&&bound>0 && (normalization==:none || bound>=sum(ws)) ||
            throw(ArgumentError("supremum_bound must be finite, positive and at least sum(weights)."))
        new{T,length(xs),typeof(provenance)}(b,e,xs,ws,normalization,bound,provenance)
    end
end

"""Evaluate an unnormalised typed CKG transition rate."""
transition_value(r::GaussianTransition,w::Real)=exp(-(w+r.beta*(r.sigma^2+r.sigma_gamma^2)/2)^2/(2r.sigma_gamma^2))
transition_value(r::MetropolisTransition,w::Real)=exp(-r.beta*max(w+r.beta*r.sigma^2/2,zero(w)))
# Evaluate exp(logweight) * erfc(z) without an overflowing exponential.
@inline _exp_erfc(logweight, z) = z >= 0 ?
    exp(logweight - z^2) * erfcx(z) : exp(logweight) * erfc(z)

@inline function _metropolis_erfc_pair(A, B, u, common)
    # The u=0 limit avoids dividing by zero at an unsmoothed transition.
    iszero(u) && return exp(common - 2A * B)
    zminus, zplus = A * u - B / u, A * u + B / u
    return (_exp_erfc(common - 2A * B, zminus) +
            _exp_erfc(common + 2A * B, zplus)) / 2
end

@inline _metropolis_kink(beta, w, shift) = exp(-beta * max(w + shift, zero(w + shift)))

@inline function _metropolis_transition(beta, sigma, a, s, w, shift)
    shifted = w + shift
    iszero(s) && iszero(a) && return _metropolis_kink(beta, w, shift)
    A = sqrt(beta / 4) * sqrt(4a + 1)
    B = sqrt(beta / 4) * abs(shifted)
    u = sqrt(beta * sigma^2 * s / 2)
    return _metropolis_erfc_pair(A, B, u, -beta * w / 2 - beta * shift / 2)
end

transition_value(r::SmoothMetropolisTransition, w::Real) =
    _metropolis_transition(r.beta, r.sigma, r.a, r.s, w, r.beta * r.sigma^2 / 2)
function transition_value(r::GaussianMixtureTransition,w::Real)
    sum(weight*exp(-(w+x)^2/(2*(2x/r.beta-r.sigma^2))) for (x,weight) in zip(r.centers,r.weights))
end

"""Analytic CKG coefficient with the normalised Gaussian OFT; no outer grid."""
transition_alpha(r::GaussianTransition,u::Real,v::Real)=
    create_alpha_gauss(u,v,r.sigma,(r.beta*(r.sigma^2+r.sigma_gamma^2)/2,r.sigma_gamma))
transition_alpha(r::MetropolisTransition,u::Real,v::Real)=create_alpha(u,v,r.beta,r.sigma,zero(r.beta),zero(r.beta))
transition_alpha(r::SmoothMetropolisTransition,u::Real,v::Real)=create_alpha(u,v,r.beta,r.sigma,r.a,r.s)
function transition_alpha(r::GaussianMixtureTransition,u::Real,v::Real)
    sum(weight*create_alpha_gauss(u,v,r.sigma,(x,sqrt(2x/r.beta-r.sigma^2))) for (x,weight) in zip(r.centers,r.weights))
end
_transition_divisor(r::Union{GaussianTransition,MetropolisTransition,SmoothMetropolisTransition})=one(r.beta)
_transition_divisor(r::GaussianMixtureTransition)=r.supremum_bound

"""
    prepare_gaussian_mixture(beta; sigma=1/beta, density, interval, panels=64,
        regularity, cutoff=nothing, tail_mass_bound=nothing, tail_atol=1e-8, ...)

Positive composite-midpoint mixture quadrature. `regularity=:continuous_nonnegative`
is a caller assumption, not a callback certificate. On compact support the rule
converges for continuous densities; each retained approximation has exact
algebraic PSD/KMS. `panels` controls this integral only, independently of the
outer-frequency register. Reports doubled-panel mass and alpha probe differences
as numerical evidence, not an error bound.

An infinite upper interval requires a finite explicit cutoff and a caller
`tail_mass_bound(cutoff)<=tail_atol`. The supplied integrated mixing-mass bound
also bounds the omitted rate and each alpha entry before normalisation.
The theorem's zero-variance endpoint and infinite-mass measures are rejected.
"""
function prepare_gaussian_mixture(beta::Real; sigma::Real=inv(beta),density,interval,
    panels::Integer=64,regularity::Symbol,cutoff=nothing,tail_mass_bound=nothing,
    tail_atol::Real=1e-8,normalization::Symbol=:none,supremum_bound=nothing)
    regularity==:continuous_nonnegative || throw(ArgumentError("Declare regularity=:continuous_nonnegative; it is a caller assumption."))
    1<=panels<=100000 || throw(ArgumentError("panels must lie in 1:100000."))
    b,e=promote(float(beta),float(sigma)); T=typeof(b)
    lo,hi=T.(interval)
    isfinite(lo)&&lo>b*e^2/2 && hi>lo || throw(ArgumentError("Require beta*sigma^2/2 < lower < upper."))
    isfinite(tail_atol)&&tail_atol>=0 || throw(ArgumentError("tail_atol must be finite and nonnegative."))
    tail=zero(T)
    if isinf(hi)
        cutoff isa Real && isfinite(cutoff)&&cutoff>lo && tail_mass_bound!==nothing ||
            throw(ArgumentError("Infinite support requires explicit finite cutoff and tail_mass_bound(cutoff)."))
        hi=T(cutoff); tail=T(tail_mass_bound(hi))
        isfinite(tail)&&0<=tail<=tail_atol || throw(ArgumentError("Declared integrated tail bound must be nonnegative and <=tail_atol."))
    else
        cutoff===nothing && tail_mass_bound===nothing || throw(ArgumentError("Finite intervals do not use cutoff/tail_mass_bound."))
    end
    function rule(n)
        dx=(hi-lo)/n
        xs=Tuple(lo+(k-T(0.5))*dx for k in 1:n)
        ws=Tuple(begin
            value=density(x)
            value isa Real&&isfinite(value)&&value>=0 || throw(ArgumentError("density must be real, finite and nonnegative at all quadrature nodes."))
            T(value)*dx
        end for x in xs)
        GaussianMixtureTransition(b;sigma=e,centers=xs,weights=ws)
    end
    coarse=rule(panels); fine=rule(2panels)
    probes=(-inv(b),zero(T),inv(b))
    alpha_difference=maximum(abs(transition_alpha(fine,u,v)-transition_alpha(coarse,u,v)) for u in probes,v in probes)
    provenance=(;kind=:positive_midpoint_quadrature,interval=(T(first(interval)),T(last(interval))),
        retained_interval=(lo,hi),panels=2panels,comparison_panels=panels,
        mass_difference=abs(sum(fine.weights)-sum(coarse.weights)),alpha_probe_difference=alpha_difference,
        tail_mass_bound=tail,tail_evidence=isinf(last(interval)) ? :caller_assumption : :compact_support_assumption,
        regularity=:caller_continuity_and_nonnegativity_assumption,
        convergence=:refinement_evidence_not_certificate,outer_frequency_quadrature=:independent,
        callback_portability=:density_and_tail_callbacks_not_retained,
        reconstruction_scope=:retained_finite_mixture_only)
    GaussianMixtureTransition(b;sigma=e,centers=fine.centers,weights=fine.weights,
        normalization,supremum_bound,provenance)
end

function _physical_ckg_transition(r::GaussianTransition,R::T,beta::T) where {T<:AbstractFloat}
    GaussianTransition(beta;sigma=T(r.sigma)/R,sigma_gamma=T(r.sigma_gamma)/R)
end
_physical_ckg_transition(r::MetropolisTransition,R::T,beta::T) where {T<:AbstractFloat}=MetropolisTransition(beta;sigma=T(r.sigma)/R)
_physical_ckg_transition(r::SmoothMetropolisTransition,R::T,beta::T) where {T<:AbstractFloat}=SmoothMetropolisTransition(beta;sigma=T(r.sigma)/R,a=T(r.a),s=T(r.s))
function _physical_ckg_transition(r::GaussianMixtureTransition,R::T,beta::T) where {T<:AbstractFloat}
    GaussianMixtureTransition(beta;sigma=T(r.sigma)/R,centers=Tuple(T(x)/R for x in r.centers),weights=r.weights,
        normalization=r.normalization,supremum_bound=r.normalization==:bound ? r.supremum_bound : nothing,
        provenance=merge(r.provenance,(;input_to_algorithm_scale=R)))
end

"""Convert typed rates to Config parameter fields."""
_ckg_legacy_fields(r::GaussianTransition)=(;with_linear_combination=false,gaussian_parameters=(r.beta*(r.sigma^2+r.sigma_gamma^2)/2,r.sigma_gamma),a=nothing,s=nothing)
_ckg_legacy_fields(r::MetropolisTransition)=(;with_linear_combination=true,gaussian_parameters=(nothing,nothing),a=zero(r.beta),s=zero(r.beta))
_ckg_legacy_fields(r::SmoothMetropolisTransition)=(;with_linear_combination=true,gaussian_parameters=(nothing,nothing),a=r.a,s=r.s)
_ckg_legacy_fields(r::GaussianMixtureTransition)=(;with_linear_combination=false,gaussian_parameters=(r.beta*r.sigma^2,r.sigma),a=nothing,s=nothing)

_ckg_parameter_match(a,b) = a === b
_ckg_parameter_match(a::Real,b::Real) = isapprox(a,b;rtol=100eps(typeof(float(b))),atol=0)
_ckg_parameter_match(a::Tuple,b::Tuple) = length(a)==length(b) && all(_ckg_parameter_match(x,y) for (x,y) in zip(a,b))

function _collect_ckg_transition_errors!(errors,cfg)
    rate=cfg.transition_weight
    rate===nothing && return nothing
    if rate isa PreparedCKGJointKernel
        cfg.construction isa KMS && cfg.sim isa Lindbladian && cfg.domain isa Union{BohrDomain,EnergyDomain,TimeDomain} && !cfg.with_gqsp ||
            push!(errors,"General CKG kernels support KMS Lindbladian Bohr/Energy/Time only; custom Trotter/GQSP evolution has no validated joint-kernel implementation.")
        (cfg.domain isa TimeDomain)==(rate.time_data!==nothing) || push!(errors,"Joint kernel must be compiled for the requested Time versus frequency domain.")
        rate.evidence.status==:pass || push!(errors,"An inspected failed joint kernel cannot enter the standard KMS simulator.")
        _ckg_parameter_match(rate.beta,cfg.beta) || push!(errors,"Compiled joint beta must match Config.beta.")
        cfg.filter === rate.oft || push!(errors,"Compiled joint OFT and transition must be used together.")
        return nothing
    end
    if !(rate isa Union{GaussianTransition,MetropolisTransition,SmoothMetropolisTransition,GaussianMixtureTransition})
        push!(errors,"Unsupported CKG transition type; arbitrary joint kernels require joint-kernel validation.")
        return nothing
    end
    cfg.construction isa KMS || push!(errors,"Typed CKG transitions require construction=KMS().")
    isapprox(rate.beta,cfg.beta;rtol=100eps(typeof(cfg.beta)),atol=0) || push!(errors,"transition_weight.beta must match Config.beta.")
    isapprox(rate.sigma,cfg.sigma;rtol=100eps(typeof(cfg.sigma)),atol=0) || push!(errors,"transition_weight.sigma must match the Gaussian OFT width Config.sigma.")
    if rate isa GaussianMixtureTransition
        cfg.domain isa Union{BohrDomain,EnergyDomain} || push!(errors,"Gaussian mixtures support BohrDomain/EnergyDomain; Time conversion requires an explicit CKGJointKernel with time_transform controls; custom Trotter is unsupported.")
        cfg.with_gqsp && push!(errors,"Gaussian mixtures do not support GQSP.")
    else
        expected=_ckg_legacy_fields(rate)
        for key in keys(expected)
            _ckg_parameter_match(getproperty(cfg,key),getproperty(expected,key)) || push!(errors,"Typed built-in $key must match its typed transition; use prepare_gibbs_inputs or the physical interface.")
        end
    end
    return nothing
end

function _resolve_physical_ckg(beta,filter,transition_weight)
    rate=transition_weight===nothing ? GaussianTransition(beta) :
        transition_weight isa AbstractCKGTransition ? transition_weight :
        applicable(transition_weight,beta) ? transition_weight(beta) :
        throw(ArgumentError("transition_weight must be a typed CKG transition or a physical-beta factory."))
    rate isa Union{GaussianTransition,MetropolisTransition,SmoothMetropolisTransition,GaussianMixtureTransition,CKGJointKernel} ||
        throw(ArgumentError("Use a supported typed CKG transition; arbitrary joint kernels require joint-kernel validation."))
    isapprox(rate.beta,beta;rtol=100eps(typeof(float(beta))),atol=0) || throw(ArgumentError("Transition beta must match beta_phys."))
    if rate isa CKGJointKernel
        filter===nothing || filter===rate.oft || throw(ArgumentError("CKGJointKernel owns its full OFT; do not substitute a separate filter."))
        return rate,rate.oft
    end
    physical_filter=filter===nothing ? GaussianFilter(rate.sigma) : filter
    physical_filter isa GaussianFilter || throw(ArgumentError("Typed CKG rates require GaussianFilter; use CKGJointKernel for a custom OFT."))
    isapprox(physical_filter.sigma,rate.sigma;rtol=100eps(typeof(float(beta))),atol=0) ||
        throw(ArgumentError("GaussianFilter.sigma must equal the transition's physical sigma."))
    return rate,physical_filter
end

function _ckg_grid(domain,time_step,num_energy_bits,energy_step)
    domain isa TrotterDomain && throw(ArgumentError("Custom CKG Trotter requires a retained local Hamiltonian decomposition and a separately validated joint coherent evolution algorithm; it is unsupported even with local terms. Built-in Trotter remains available through Config and make_trotter_for_config."))
    domain isa Union{BohrDomain,EnergyDomain,TimeDomain} || throw(ArgumentError("Unsupported CKG domain."))
    if domain isa TimeDomain
        time_step isa Real && isfinite(time_step) && time_step>0 && num_energy_bits isa Integer && 0<num_energy_bits<63 ||
            throw(ArgumentError("CKG Time requires positive physical time_step and 0<num_energy_bits<63."))
        energy_step===nothing || throw(ArgumentError("CKG Time outer quadrature uses CKGJointKernel.frequency_window/panels, not energy_step."))
        return nothing
    end
    time_step===nothing || throw(ArgumentError("CKG Bohr/Energy does not use time_step; Energy uses physical energy_step."))
    if domain isa BohrDomain
        num_energy_bits===nothing && energy_step===nothing || throw(ArgumentError("BohrDomain uses no outer-frequency grid."))
    else
        num_energy_bits isa Integer && 0<num_energy_bits<63 && energy_step isa Real && isfinite(energy_step)&&energy_step>0 ||
            throw(ArgumentError("EnergyDomain requires positive physical energy_step and 0<num_energy_bits<63."))
    end
end

function _ckg_preflight(H;beta_phys,temperature,filter,jumps,rates,complete_adjoint,basis,
    domain,time_step,num_energy_bits,clock,transition_weight,energy_step,max_bytes)
    _ckg_grid(domain,time_step,num_energy_bits,energy_step)
    # Reuse all model/source/temperature and resource checks without spectral work.
    base=_physical_preflight(H;beta_phys,temperature,jumps,rates,
        complete_adjoint,basis,clock,max_bytes)
    rate,physical_filter=_resolve_physical_ckg(base.beta_phys,filter,transition_weight)
    domain isa TimeDomain && !(rate isa CKGJointKernel) && throw(ArgumentError("CKG Time interface requires CKGJointKernel and explicit time_transform controls; typed built-ins retain Config Time/Trotter support."))
    if rate isa CKGJointKernel
        m=big(base.dimension)^2-base.dimension+1
        n=domain isa EnergyDomain ? big(2)^num_energy_bits : big(8)*rate.panels
        kernel_bytes=big(128)*(m*m+n*m+n)
        domain isa TimeDomain && (kernel_bytes+=_ckg_time_bytes(rate,num_energy_bits,n,m,base.dimension))
        bytes=base.estimated_construction_bytes+kernel_bytes
        return merge(base,(;construction=:CKG_KMS,domain=Symbol(nameof(typeof(domain))),
            filter=physical_filter,transition_weight=rate,channel_count=1,energy_step,
            outer_frequency_points=n,estimated_construction_bytes=bytes,
            permitted=bytes<=max_bytes && kernel_bytes<=rate.max_bytes,
            max_bohr_frequencies=rate.max_bohr_frequencies,bohr_validation=:complete_set_checked_after_diagonalisation,
            rate_normalization=:none,rate_divisor=one(rate.beta),rate_bound_kind=:not_normalized,
            kernel_realization=:positive_quadrature_gram))
    end
    points=domain isa EnergyDomain ? big(2)^num_energy_bits : big(0)
    bytes=base.estimated_construction_bytes+big(32)*base.dimension^2*points
    return merge(base,(;construction=:CKG_KMS,domain=Symbol(nameof(typeof(domain))),
        filter=physical_filter,transition_weight=rate,channel_count=1,
        energy_step,outer_frequency_points=points,estimated_construction_bytes=bytes,permitted=bytes<=max_bytes,
        rate_normalization=rate isa GaussianMixtureTransition ? rate.normalization : :legacy_unit_bound,
        rate_divisor=_transition_divisor(rate),rate_bound_kind=rate isa GaussianMixtureTransition ?
            (rate.normalization==:none ? :not_normalized : :user_fixed_upper_bound) : :legacy_upper_bound,
        kernel_realization=domain isa BohrDomain ? :analytic : :outer_frequency_quadrature))
end

function _prepare_ckg_inputs(H;beta_phys,temperature,filter,jumps,rates,complete_adjoint,
    basis,domain,time_step,num_energy_bits,clock,transition_weight,energy_step)
    _ckg_grid(domain,time_step,num_energy_bits,energy_step)
    # Reuse owned sources, physical temperature, cached-Gibbs checks and explicit clock.
    base=_prepare_physical_inputs(H;beta_phys,temperature,jumps,rates,
        complete_adjoint,basis,clock)
    ham=base.hamiltonian; T=eltype(ham.eigvals)
    physical_rate,physical_filter=_resolve_physical_ckg(base.provenance.beta_phys,filter,transition_weight)
    domain isa TimeDomain && !(physical_rate isa CKGJointKernel) && throw(ArgumentError("CKG Time interface requires CKGJointKernel with explicit time_transform controls; use Config for typed built-in Time."))
    physical_rate isa CKGJointKernel && return _prepare_joint_ckg_inputs(base,physical_rate,domain,num_energy_bits,energy_step,time_step)
    algorithm_rate=_physical_ckg_transition(physical_rate,T(ham.rescaling_factor),base.provenance.beta_alg)
    algorithm_filter=GaussianFilter(algorithm_rate.sigma)
    cfg=Config(;sim=Lindbladian(),domain,construction=KMS(),num_qubits=trailing_zeros(size(ham.data,1)),
        beta=base.provenance.beta_alg,beta_phys=base.provenance.beta_phys,sigma=algorithm_rate.sigma,
        filter=algorithm_filter,transition_weight=algorithm_rate,_ckg_legacy_fields(algorithm_rate)...,
        num_energy_bits_D=num_energy_bits,w0_D=energy_step===nothing ? nothing : T(energy_step/ham.rescaling_factor))
    validate_config!(cfg,ham;atol=100eps(T),rtol=100eps(T))
    # C_alg(w)=sqrt(R)*C_phys(R*w), gamma_alg(w)=gamma_phys(R*w),
    # so d(w_alg)*C_alg*C_alg preserves alpha and needs no extra generator factor.
    evidence=(;balance=:gaussian_oft_structural_kms,positivity=:nonnegative_rate,
        alpha=:analytic,coherent=:canonical_analytic,outer_frequency_quadrature=domain isa EnergyDomain,
        rate_normalization=physical_rate isa GaussianMixtureTransition ? physical_rate.normalization : :legacy_unit_bound,
        rate_divisor=_transition_divisor(physical_rate),
        mixture=physical_rate isa GaussianMixtureTransition ? physical_rate.provenance : nothing)
    provenance=merge(base.provenance,(;construction=:CKG_KMS,filter_input_frame=:physical,
        source_filter_assignments=nothing,physical_time_step=nothing,algorithm_time_step=nothing,
        physical_filter,algorithm_filter,physical_filters=((;family=:CKGGaussian,sigma=physical_filter.sigma,frame=:physical),),
        algorithm_filters=((;family=:CKGGaussian,sigma=algorithm_filter.sigma,frame=:algorithm),),
        physical_transition=physical_rate,algorithm_transition=algorithm_rate,
        filter_evidence=(evidence,),sigma_role=:normalized_gaussian_oft_width,
        physical_energy_step=energy_step,algorithm_energy_step=cfg.w0_D,
        rate_normalization=evidence.rate_normalization,rate_divisor=evidence.rate_divisor,
        ckg_frame_transform=:normalized_oft_measure_cancels))
    return (;hamiltonian=ham,config=cfg,jumps=base.jumps,provenance)
end
