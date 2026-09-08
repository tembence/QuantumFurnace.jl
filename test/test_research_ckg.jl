using Test, QuantumFurnace, LinearAlgebra, QuadGK, BSON

struct UnsupportedCKGTransition <: AbstractCKGTransition end

@testset "T15 typed CKG rates and Gaussian mixtures" begin
    beta=0.8; sigma=0.35
    rates=(GaussianTransition(beta;sigma,sigma_gamma=0.6),
        MetropolisTransition(beta;sigma),SmoothMetropolisTransition(beta;sigma),
        SmoothMetropolisTransition(beta;sigma,a=0.2,s=0.4),
        GaussianMixtureTransition(beta;sigma,centers=(0.2,0.6),weights=(0.3,0.8)))
    @testset "Independent normalized-OFT alpha, PSD and joint KMS" begin
        C(w)=(2pi*sigma^2)^(-1/4)*exp(-w^2/(4sigma^2))
        for r in rates,u in (-0.6,0.,0.4),v in (-0.4,0.,0.7)
            integral,error=quadgk(w->transition_value(r,w)*C(w-u)*C(w-v),
                -Inf,-beta*sigma^2/2,Inf;atol=1e-12,rtol=1e-11)
            @test transition_alpha(r,u,v)≈integral atol=1e-10 rtol=1e-10
            @test transition_alpha(r,u,v)≈exp(-beta*(u+v)/2)*transition_alpha(r,-u,-v) atol=1e-12 rtol=1e-12
        end
        for r in rates
            nus=(-0.8,-0.3,0.,0.3,0.8)
            alpha=[transition_alpha(r,u,v) for u in nus,v in nus]
            @test minimum(eigvals(Hermitian(alpha)))>=-1e-12
            ra=QuantumFurnace._physical_ckg_transition(r,3.2,3.2beta)
            @test transition_alpha(ra,0.3/3.2,-0.7/3.2)≈transition_alpha(r,0.3,-0.7) atol=1e-13
            ra32=QuantumFurnace._physical_ckg_transition(r,3.2f0,Float32(3.2beta))
            @test ra32.beta isa Float32
            @test ra32.sigma isa Float32
        end
        g=first(rates)
        x=g.beta*(g.sigma^2+g.sigma_gamma^2)/2
        @test g.beta≈2x/(g.sigma^2+g.sigma_gamma^2)
        single=GaussianMixtureTransition(beta;sigma,centers=(x,),weights=(1.,))
        @test transition_alpha(single,0.2,-0.3)≈transition_alpha(g,0.2,-0.3)
        metro=rates[2]
        @test transition_value(metro,-beta*sigma^2/2)≈1.
        @test transition_value(metro,0.)≈exp(-beta^2*sigma^2/2)
        @test all(isfinite(transition_value(rates[3],w)) for w in (-1e4,1e4))
    end
    @testset "Bounds, endpoints and positive mixture quadrature" begin
        @test_throws ArgumentError GaussianTransition(0.)
        @test_throws ArgumentError GaussianTransition(Inf)
        @test_throws ArgumentError GaussianTransition(1e200;sigma=1.)
        @test_throws ArgumentError GaussianTransition(beta;sigma=1e-200)
        @test_throws ArgumentError GaussianTransition(beta;sigma_gamma=1e200)
        @test_throws ArgumentError SmoothMetropolisTransition(beta;s=0.)
        @test_throws ArgumentError GaussianMixtureTransition(beta;sigma,centers=(beta*sigma^2/2,),weights=(1.,))
        @test_throws ArgumentError GaussianMixtureTransition(beta;sigma,centers=(0.3,),weights=(-1.,))
        @test_throws ArgumentError GaussianMixtureTransition(beta;sigma,centers=(0.3,),weights=(0.,))
        @test_throws ArgumentError GaussianMixtureTransition(beta;sigma,centers=(0.3,),weights=(1.,),normalization=:bound)
        @test_throws ArgumentError GaussianMixtureTransition(beta;sigma,centers=(0.3,),weights=(1.,),normalization=:bound,supremum_bound=0.5)
        options=(;sigma,density=x->exp(-x),interval=(0.1,2.),regularity=:continuous_nonnegative)
        coarse=prepare_gaussian_mixture(beta;options...,panels=8)
        fine=prepare_gaussian_mixture(beta;options...,panels=64)
        @test fine.provenance.alpha_probe_difference<coarse.provenance.alpha_probe_difference/20
        @test fine.provenance.mass_difference<coarse.provenance.mass_difference/20
        tail=prepare_gaussian_mixture(beta;sigma,density=x->exp(-x),interval=(0.1,Inf),cutoff=24.,
            tail_mass_bound=x->exp(-x),panels=64,regularity=:continuous_nonnegative)
        @test tail.provenance.tail_mass_bound<=1e-8
        @test tail.provenance.tail_evidence==:caller_assumption
        @test_throws ArgumentError prepare_gaussian_mixture(beta;sigma,density=x->1.,interval=(0.1,Inf),regularity=:continuous_nonnegative)
        @test_throws ArgumentError prepare_gaussian_mixture(beta;options...,panels=2,normalization=:bound,supremum_bound=0.1)
        @test_throws ArgumentError prepare_gaussian_mixture(beta;sigma,density=x->-1.,interval=(0.1,2.),regularity=:continuous_nonnegative)
        @test_throws ArgumentError prepare_gaussian_mixture(beta;sigma,density=x->1.,interval=(0.1,2.),regularity=:unknown)
    end
    @testset "Full analytic generator, legacy equality, NH pair and physical facade" begin
        H=Hermitian(ComplexF64[0.2 0.3im 0.1 0.0; -0.3im 0.7 0.2 -0.1im; 0.1 0.2 -0.5 0.3; 0.0 0.1im 0.3 1.1])
        A=ComplexF64[0.2 0.3im 0.1 0.7; 0.1 -0.2 0.4im 0.2; 0.3 0.0 0.2 0.1im; -0.2im 0.1 0.5 0.3]
        sources=[A,Matrix(A')]
        for r in rates
            p=prepare_gibbs_inputs(H;beta_phys=beta,construction=KMS(),transition_weight=r,jumps=sources)
            cfg,ham=p.config,p.hamiltonian
            alpha=QuantumFurnace._pick_alpha(cfg)
            @test @inferred(alpha(0.1,0.2))≈transition_alpha(cfg.transition_weight,0.1,0.2)
            L=construct_lindbladian(p.jumps,cfg,ham)
            @test norm(L*vec(ham.gibbs))<1e-11
            weights=real(diag(ham.gibbs))
            S=Diagonal(vec((weights*weights').^(1/4)))
            parent=S\L*S
            @test norm(parent-parent')<1e-11
            ws=Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=r,jumps=sources)
            rho=fill(0.25+0im,4,4)
            apply_lindbladian!(ws,rho,ws.cached_cfg,ws.ham_or_trott)
            @test vec(ws.scratch.rho_out)≈L*vec(rho) atol=1e-11 rtol=1e-11
            if !(r isa GaussianMixtureTransition)
                legacy=Config(;sim=Lindbladian(),domain=BohrDomain(),construction=KMS(),
                    num_qubits=2,beta=cfg.beta,sigma=cfg.sigma,QuantumFurnace._ckg_legacy_fields(cfg.transition_weight)...)
                @test construct_lindbladian(p.jumps,legacy,ham)≈L atol=1e-12 rtol=1e-12
                @test QuantumFurnace.pick_gamma_sup(cfg)≈QuantumFurnace.pick_gamma_sup(legacy)
            end
        end
        r=last(rates)
        unnormalized=prepare_gibbs_inputs(H;beta_phys=beta,construction=KMS(),transition_weight=r,jumps=sources)
        normalized=GaussianMixtureTransition(beta;sigma,centers=r.centers,weights=r.weights,normalization=:bound,supremum_bound=2.)
        p=prepare_gibbs_inputs(H;beta_phys=beta,construction=KMS(),transition_weight=normalized,jumps=sources)
        L0=construct_lindbladian(unnormalized.jumps,unnormalized.config,unnormalized.hamiltonian)
        Ln=construct_lindbladian(p.jumps,p.config,p.hamiltonian)
        @test Ln≈L0/2 atol=1e-12 rtol=1e-12
        @test p.provenance.rate_divisor≈2.
        restored=QuantumFurnace._reconstruct_config(QuantumFurnace._config_to_dict(p.config))
        @test restored.transition_weight isa GaussianMixtureTransition
        @test construct_lindbladian(p.jumps,restored,p.hamiltonian)≈Ln atol=1e-12 rtol=1e-12
        mktempdir() do dir
            path=joinpath(dir,"mixture.bson")
            BSON.bson(path,Dict(:config=>QuantumFurnace._config_to_dict(p.config)))
            loaded=QuantumFurnace._reconstruct_config(BSON.load(path,QuantumFurnace)[:config])
            @test QuantumFurnace.pick_gamma_sup(loaded)≈2.
            @test construct_lindbladian(p.jumps,loaded,p.hamiltonian)≈Ln atol=1e-12 rtol=1e-12
        end
        for rate in rates
            p32=prepare_gibbs_inputs(Hermitian(ComplexF32.(H));beta_phys=beta,
                construction=KMS(),transition_weight=rate)
            @test p32.config.beta isa Float32
            @test p32.config.sigma isa Float32
        end
        calls=Ref(0)
        factory=b->begin
            calls[]+=1
            GaussianTransition(b;sigma=0.3+calls[]/10)
        end
        resolved=Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=factory)
        @test calls[]==1
        @test resolved.research_provenance.physical_transition===resolved.research_provenance.preflight.transition_weight
        @test_throws ArgumentError Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=r,filter=GaussianFilter(2sigma),dry_run=true)
        @test_throws ArgumentError Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=r,filter=DLLGaussianFilter(beta),dry_run=true)
        @test_throws ArgumentError Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=r,domain=TimeDomain(),dry_run=true)
        @test_throws ArgumentError Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=w->exp(-w),dry_run=true)
        @test_throws ArgumentError Workspace(H;beta_phys=beta,construction=DLL(),transition_weight=r,dry_run=true)
        @test_throws ArgumentError Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=UnsupportedCKGTransition(),dry_run=true)
        pre=simulate_gibbs(H;beta_phys=beta,construction=KMS(),transition_weight=r,times=[0.,0.1],dry_run=true)
        @test pre.construction==:CKG_KMS
        result=simulate_gibbs(H;beta_phys=beta,construction=KMS(),transition_weight=r,times=[0.,0.1],
            diagnostics=:quick,krylovdim=16,gap_options=(;krylovdim=16))
        @test result.provenance.construction==:CKG_KMS
        @test all(isfinite,result.trajectory.distances)
        @testset "Energy grid refinement to analytic Bohr with fixed normalization" begin
            errors=Float64[]
            for (step,bits) in ((0.8,5),(0.4,6),(0.1,8))
                pE=prepare_gibbs_inputs(H;beta_phys=beta,construction=KMS(),transition_weight=normalized,
                    jumps=sources,domain=EnergyDomain(),energy_step=step,num_energy_bits=bits)
                LE=construct_lindbladian(pE.jumps,pE.config,pE.hamiltonian)
                push!(errors,norm(LE-Ln))
                @test QuantumFurnace.pick_gamma_sup(pE.config)≈2.
            end
            @test errors[end]<1e-9
            @test errors[end]<errors[1]/100
        end
    end
    @testset "Typed built-ins retain legacy Time generators" begin
        H=ComplexF64[0.2 0.1im; -0.1im 0.9]
        for r in rates[1:4]
            p=prepare_gibbs_inputs(H;beta_phys=beta,construction=KMS(),transition_weight=r)
            cfg=p.config
            settings=(;sim=Lindbladian(),domain=TimeDomain(),construction=KMS(),num_qubits=1,
                beta=cfg.beta,sigma=cfg.sigma,QuantumFurnace._ckg_legacy_fields(cfg.transition_weight)...,
                num_energy_bits=7,t0=0.4,w0=2pi/(128*0.4),eta=0.05)
            legacy=Config(;settings...)
            typed=Config(;settings...,transition_weight=cfg.transition_weight)
            @test construct_lindbladian(p.jumps,typed,p.hamiltonian)≈
                construct_lindbladian(p.jumps,legacy,p.hamiltonian) atol=1e-11 rtol=1e-11
        end
    end
end
