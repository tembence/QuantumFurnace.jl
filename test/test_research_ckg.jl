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

@testset "T16 general complex CKG joint compiler" begin
    beta_phys=0.8
    # Independent structural family: C=e^(beta*x/4)q, gamma=e^(-beta*w/2)g,
    # q conjugate-reflected, g even nonnegative. Normalisation is explicit.
    normalization=inv(sqrt(first(quadgk(x->exp(beta_phys*x/2-2x^4),-Inf,Inf;rtol=1e-12))))
    C=x->normalization*exp(beta_phys*x/4-x^4)*cis(0.3x)
    gamma=w->exp(-w^2-beta_phys*w/2)
    pair=CKGJointKernel(beta_phys;oft=C,rate=gamma,frequency_window=(-8.,8.))
    nus=[-0.7,-0.2,0.,0.2,0.7]
    compiled=compile_ckg_kernel(pair,nus)
    @test compiled.evidence.status==:pass
    @test compiled.evidence.global_balance==:not_established_by_samples
    @test minimum(eigvals(Hermitian(compiled.alpha)))>=-1e-12
    @test norm(compiled.alpha-compiled.alpha')<1e-12
    @test norm(imag(compiled.alpha))>1e-3
    for u in nus,v in nus
        ref=first(quadgk(w->gamma(w)*C(w-u)*conj(C(w-v)),-Inf,Inf;atol=1e-13,rtol=1e-12))
        @test transition_alpha(compiled,u,v)≈ref atol=1e-10 rtol=1e-10
        @test transition_alpha(compiled,u,v)≈exp(-beta_phys*(u+v)/2)*transition_alpha(compiled,-v,-u) atol=1e-12
    end
    @testset "Invalid pair versus unresolved quadrature; explicit bounds" begin
        bad=CKGJointKernel(beta_phys;oft=x->(2pi)^(-1/4)*exp(-x^2/4),rate=gamma,frequency_window=(-8.,8.))
        @test gamma(.7)/gamma(-.7)≈exp(-beta_phys*.7)
        inspected=compile_ckg_kernel(bad,nus;strict=false)
        @test inspected.evidence.status==:not_KMS
        @test inspected.evidence.reference_balance_defect>0.01
        @test minimum(eigvals(Hermitian(inspected.alpha)))>=-1e-12
        @test_throws ArgumentError compile_ckg_kernel(bad,nus)
        tiny_bad=CKGJointKernel(beta_phys;oft=bad.oft,rate=w->1e-100*gamma(w),frequency_window=(-8.,8.))
        @test compile_ckg_kernel(tiny_bad,nus;strict=false).evidence.status==:not_KMS
        failed_cfg=Config(;sim=Lindbladian(),domain=BohrDomain(),construction=KMS(),
            num_qubits=1,beta=beta_phys,sigma=1.,with_linear_combination=false,
            filter=inspected.oft,transition_weight=inspected)
        @test_throws ArgumentError validate_config!(failed_cfg)
        @test_throws ArgumentError Workspace(ComplexF64[0 0;0 0.7];beta_phys,construction=KMS(),transition_weight=bad)
        for (window,panels) in (((-8.,8.),2),((-0.5,0.5),32))
            coarse=CKGJointKernel(beta_phys;oft=C,rate=gamma,frequency_window=window,panels)
            evidence=compile_ckg_kernel(coarse,nus;strict=false).evidence
            @test evidence.status==:quadrature_unresolved
            @test evidence.reference_balance_defect<1e-10
            @test evidence.relative_integration_difference>1e-6
            @test_throws ArgumentError compile_ckg_kernel(coarse,nus)
        end
        @test_throws ArgumentError compile_ckg_kernel(CKGJointKernel(beta_phys;oft=C,rate=w->-gamma(w),frequency_window=(-8.,8.)),nus)
        @test_throws ArgumentError compile_ckg_kernel(CKGJointKernel(beta_phys;oft=x->2C(x),rate=gamma,frequency_window=(-8.,8.)),nus)
        @test_throws ArgumentError compile_ckg_kernel(CKGJointKernel(beta_phys;oft=C,rate=gamma,frequency_window=(-8.,8.),max_bohr_frequencies=3),nus)
        @test_throws ArgumentError compile_ckg_kernel(CKGJointKernel(beta_phys;oft=C,rate=gamma,frequency_window=(-8.,8.),max_bytes=100),nus)
        overflow=CKGJointKernel(beta_phys;oft=x->(2pi)^(-1/4)*exp(-x^2/4),
            rate=w->1e307,frequency_window=(-10000.,10000.),panels=1)
        @test_throws ArgumentError compile_ckg_kernel(overflow,[0.])
        @test_throws ArgumentError compile_ckg_kernel(pair,[0.,0.7])
        @test_throws ArgumentError compile_ckg_kernel(pair,nus;energy_labels=[0.,0.],energy_step=1.)
        @test_throws ArgumentError CKGJointKernel(beta_phys;oft=C,rate=gamma,frequency_window=(-Inf,Inf))
        @test_throws ArgumentError CKGJointKernel(beta_phys;oft=C,rate=gamma,frequency_window=(-8.,8.),panels=0)
    end
    @testset "Physical continuum full generator; complex NH and Hermitian sources" begin
        H=Hermitian(ComplexF64[0.2 0.3im 0.1 0.; -0.3im 0.7 0.2 -0.1im; 0.1 0.2 -0.5 0.3; 0. 0.1im 0.3 1.1])
        A=ComplexF64[0.2 0.3im 0.1 0.7; 0.1 -0.2 0.4im 0.2; 0.3 0. 0.2 0.1im; -0.2im 0.1 0.5 0.3]
        sources=[A,Matrix(A')]
        p=prepare_gibbs_inputs(H;beta_phys,construction=KMS(),transition_weight=pair,jumps=sources)
        ham=p.hamiltonian; d=4; Id=Matrix{ComplexF64}(I,d,d)
        delta_phys=ham.bohr_freqs*ham.rescaling_factor
        As=[j.in_eigenbasis for j in p.jumps]
        # Matrix-valued physical-frequency integration, independent of the
        # compiler's alpha, its positive rule, and package vectorisation helpers.
        function gain_loss(w)
            Ls=[a .* C.(w .- delta_phys) for a in As]
            R=sum(L' * L for L in Ls)
            gain=sum(kron(conj(L),L) for L in Ls)
            vcat(vec(gamma(w)*gain),vec(gamma(w)*R))
        end
        integral=first(quadgk(gain_loss,-Inf,0.,Inf;atol=1e-12,rtol=1e-12))
        gain=reshape(integral[1:d^4],d^2,d^2); R=reshape(integral[d^4+1:end],d,d)
        B=(im/2).*tanh.(beta_phys.*delta_phys./4).*R
        reference=gain-(kron(Id,R)+kron(transpose(R),Id))/2-im*(kron(Id,B)-kron(transpose(B),Id))
        @test norm(B)>1e-3
        LB=construct_lindbladian(p.jumps,p.config,ham)
        @test LB≈reference atol=1e-9 rtol=1e-9
        @test norm(LB*vec(ham.gibbs))<1e-10
        weights=real(diag(ham.gibbs)); S=Diagonal(vec((weights*weights').^(1/4)))
        @test norm(S\LB*S-(S\LB*S)')<1e-10
        wsB=Workspace(H;beta_phys,construction=KMS(),transition_weight=pair,jumps=sources)
        X=reshape(ComplexF64.(1:16),4,4)./16 .+ im.*Matrix(I,4,4)
        @test vec(apply_lindbladian!(wsB,X,wsB.cached_cfg,wsB.ham_or_trott))≈reference*vec(X) atol=1e-9
        @test vec(apply_adjoint_lindbladian!(wsB,X,wsB.cached_cfg,wsB.ham_or_trott))≈reference'*vec(X) atol=1e-9
        # Equivalent Hermitian source representation must use signed labels too.
        hermitian_sources=[(A+A')/sqrt(2),(A-A')/(sqrt(2)*im)]
        errors=Float64[]
        for (step,bits) in ((0.5,5),(0.25,6),(0.0625,8))
            # Coarse approximations are inspectable, but may not enter strict KMS.
            labels=step.*collect(-2^(bits-1):2^(bits-1)-1)
            k=compile_ckg_kernel(pair,sort!(collect(keys(ham.bohr_dict))).*ham.rescaling_factor;
                strict=false,energy_labels=labels,energy_step=step)
            push!(errors,k.evidence.relative_integration_difference)
        end
        @test errors[end]<1e-9
        @test errors[end]<errors[1]/100
        for jumps in (sources,hermitian_sources)
            ws=Workspace(H;beta_phys,construction=KMS(),transition_weight=pair,jumps,
                domain=EnergyDomain(),energy_step=0.0625,num_energy_bits=8)
            cfg=ws.cached_cfg; hh=ws.ham_or_trott
            LE=construct_lindbladian(ws.jumps,cfg,hh)
            @test LE≈reference atol=1e-9 rtol=1e-9
            @test vec(apply_lindbladian!(ws,X,cfg,hh))≈reference*vec(X) atol=1e-9
            @test vec(apply_adjoint_lindbladian!(ws,X,cfg,hh))≈reference'*vec(X) atol=1e-9
            @test norm(ws.G_left+ws.G_right+R)<1e-9
            @test norm((ws.G_right-ws.G_left)/(2im)-B)<1e-9
            @test ws.research_provenance.filter_evidence[1].status==:pass
            @test ws.research_provenance.rate_divisor≈1.
        end
        @test_throws ArgumentError Workspace(H;beta_phys,construction=KMS(),transition_weight=pair,
            domain=EnergyDomain(),energy_step=0.5,num_energy_bits=5)
        @test_throws ArgumentError Workspace(H;beta_phys,construction=KMS(),transition_weight=pair,
            domain=EnergyDomain(),energy_step=0.1,num_energy_bits=50)
        @test_throws ArgumentError Workspace(H;beta_phys,construction=KMS(),transition_weight=pair,domain=TimeDomain())
        @test_throws ArgumentError Workspace(H;beta_phys,construction=KMS(),transition_weight=pair,domain=TrotterDomain())
        @test_throws ArgumentError Workspace(Hermitian(ComplexF32.(H));beta_phys,construction=KMS(),transition_weight=pair)
        # Model mismatch must fail before a stale tabulated lookup/action.
        @test_throws ArgumentError validate_config!(p.config,HamHam(2Matrix(H);beta_phys))
    end
    @testset "Analytic fast-path differential and owned samples" begin
        H=ComplexF64[0.2 0.1im;-0.1im 0.9]
        analytic=GaussianTransition(beta_phys;sigma=.35,sigma_gamma=.6)
        Cg=x->(2pi*.35^2)^(-1/4)*exp(-x^2/(4*.35^2))
        numerical=CKGJointKernel(beta_phys;oft=Cg,rate=w->transition_value(analytic,w),frequency_window=(-8.,8.))
        pa=prepare_gibbs_inputs(H;beta_phys,construction=KMS(),transition_weight=analytic)
        pn=prepare_gibbs_inputs(H;beta_phys,construction=KMS(),transition_weight=numerical)
        @test construct_lindbladian(pa.jumps,pa.config,pa.hamiltonian)≈construct_lindbladian(pn.jumps,pn.config,pn.hamiltonian) atol=1e-10 rtol=1e-10
        poison=Ref(false)
        owned=CKGJointKernel(beta_phys;oft=x->(poison[] ? error("callback replay") : C(x)),rate=gamma,frequency_window=(-8.,8.))
        for domain in (BohrDomain(),EnergyDomain())
            poison[]=false
            options=domain isa EnergyDomain ? (;energy_step=0.0625,num_energy_bits=8) : (;)
            ws=Workspace(H;beta_phys,construction=KMS(),transition_weight=owned,domain,options...)
            poison[]=true
            @test all(isfinite,apply_lindbladian!(ws,Matrix{ComplexF64}(I,2,2)/2,ws.cached_cfg,ws.ham_or_trott))
            result=simulate_gibbs(ws;times=[0.,0.1],diagnostics=:quick,krylovdim=4,gap_options=(;krylovdim=4,howmany=2))
            @test all(isfinite,result.trajectory.distances)
        end
    end
end

@testset "T17 general CKG Time and capability gate" begin
    beta=.8; H=ComplexF64[-.35 0;0 .35]
    A=ComplexF64[.2 .3im;.4 -.1]; sources=[A,Matrix(A')]
    z=inv(sqrt(first(quadgk(x->exp(beta*x/2-2x^4),-Inf,Inf;rtol=1e-12))))
    # Odd cubic phase gives a genuinely nonfactorising complex joint kernel.
    C=x->z*exp(beta*x/4-x^4)*cis(.1x^3)
    gamma=w->exp(-w^2-beta*w/2)
    fine=(;frequency_window=8.,frequency_grid_size=257,
        coherent_frequency_window=8.,coherent_frequency_grid_size=129,
        coherent_time_window=19.2,coherent_time_step=.15,backend=:direct)
    pair=CKGJointKernel(beta;oft=C,rate=gamma,frequency_window=(-8.,8.),time_transform=fine)
    ws=Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=pair,jumps=sources,
        domain=TimeDomain(),time_step=.15,num_energy_bits=8)
    cfg=ws.cached_cfg; ham=ws.ham_or_trott; r=cfg.transition_weight
    delta=ham.bohr_freqs*ham.rescaling_factor; As=[j.in_eigenbasis for j in ws.jumps]
    # Physical-frequency, matrix-valued adaptive reference uses no compiler,
    # no package coherent/vectorisation helper and no time quadrature.
    integral=first(quadgk(-Inf,0.,Inf;rtol=1e-12,atol=1e-13) do w
        Ls=[a.*C.(w.-delta) for a in As]
        vcat(vec(gamma(w)*sum(kron(conj(L),L) for L in Ls)),
             vec(gamma(w)*sum(L'*L for L in Ls)))
    end)
    gain=reshape(integral[1:16],4,4); R=reshape(integral[17:end],2,2)
    B=(im/2).*tanh.(beta.*delta./4).*R; Id=Matrix{ComplexF64}(I,2,2)
    reference=gain-(kron(Id,R)+kron(transpose(R),Id))/2-im*(kron(Id,B)-kron(transpose(B),Id))
    LT=construct_lindbladian(ws.jumps,cfg,ham)
    @test LT≈reference atol=1e-9 rtol=1e-9
    @test norm(ws.G_left+ws.G_right+R)<1e-9
    @test norm((ws.G_right-ws.G_left)/(2im)-B)<1e-9
    @test norm(B)>1e-4
    @test norm(LT*vec(ham.gibbs))<1e-9
    @test vec(apply_lindbladian!(ws,A,cfg,ham))≈reference*vec(A) atol=1e-9
    @test vec(apply_adjoint_lindbladian!(ws,A,cfg,ham))≈reference'*vec(A) atol=1e-9
    @test all(w->isapprox(transition_value(r,w),gamma(w*ham.rescaling_factor);rtol=1e-14),r.energy_labels)
    @test r.evidence.coherent==:two_time_joint_kernel
    @test r.evidence.relative_dissipative_time_difference<1e-9
    @test r.evidence.relative_coherent_time_difference<1e-9
    @test r.evidence.quantum_implementation==:not_established
    trajectory=simulate_gibbs(ws;times=[0.,.1],diagnostics=:quick)
    @test all(isfinite,trajectory.trajectory.distances)
    @test r.evidence.integration_comparator==:retained_frequency_gram
    @test r.evidence.frequency_reference_evidence.status==:pass
    weights=real(diag(ham.gibbs)); S=Diagonal(vec((weights*weights').^(1/4)))
    @test norm(S\LT*S-(S\LT*S)')<1e-9

    # Independent 2D inverse-transform reference at a sampled time pair.
    # Separate Gauss outer rule and explicit scalar double sum.
    x,q=QuantumFurnace.QuadGK.gauss(Float64,256); outer=8x; ow=8q
    nu=collect(range(-8.,8.;length=129)); dv=nu[2]-nu[1]
    nw=fill(dv,129); nw[1]/=2; nw[end]/=2
    amplitudes=[sqrt(w*gamma(v))*C(v-u) for (v,w) in zip(outer,ow),u in nu]
    alpha=transpose(amplitudes)*conj(amplitudes)
    ti=120; tj=137; scale=ham.rescaling_factor
    t=r.time_data.times[ti]/scale; s=r.time_data.times[tj]/scale
    gref=sum(nw[i]*nw[j]*tanh(beta*(v-u)/4)*alpha[i,j]/(2im)*cis(-u*t+v*s)
        for (i,u) in enumerate(nu),(j,v) in enumerate(nu))/(2pi)^2
    @test r.time_data.g_tt[ti,tj]*scale^2≈gref atol=1e-10 rtol=1e-9

    # Refine coherent time truncation with dissipation held byte-identical.
    coarse=merge(fine,(;coherent_time_window=4.8,coherent_time_step=.3))
    loose=CKGJointKernel(beta;oft=C,rate=gamma,frequency_window=(-8.,8.),
        time_transform=coarse,balance_rtol=.01)
    wc=Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=loose,jumps=sources,
        domain=TimeDomain(),time_step=.15,num_energy_bits=8)
    @test wc.cached_cfg.transition_weight.oft.values≈r.oft.values atol=1e-14
    @test wc.cached_cfg.transition_weight.evidence.relative_coherent_time_difference>
        100r.evidence.relative_coherent_time_difference
    @test norm((wc.G_right-wc.G_left)/(2im)-B)>100norm((ws.G_right-ws.G_left)/(2im)-B)
    strict_coarse=CKGJointKernel(beta;oft=C,rate=gamma,frequency_window=(-8.,8.),time_transform=coarse)
    @test_throws ArgumentError Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=strict_coarse,
        jumps=sources,domain=TimeDomain(),time_step=.15,num_energy_bits=8)
    @test_throws ArgumentError Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=pair,
        jumps=sources,domain=TimeDomain(),time_step=.6,num_energy_bits=4)
    # NH pairing and general complex OFT prohibit Hermitian frequency folding.
    hp=[(A+A')/sqrt(2),(A-A')/(im*sqrt(2))]
    wf=Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=CKGJointKernel(beta;
        oft=C,rate=gamma,frequency_window=(-8.,8.),time_transform=merge(fine,(;backend=:finufft))),
        jumps=hp,domain=TimeDomain(),time_step=.15,num_energy_bits=8)
    @test construct_lindbladian(wf.jumps,wf.cached_cfg,wf.ham_or_trott)≈reference atol=1e-9
    @test vec(apply_lindbladian!(wf,A,wf.cached_cfg,wf.ham_or_trott))≈reference*vec(A) atol=1e-9
    @test_throws ArgumentError Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=pair,
        domain=TrotterDomain(),dry_run=true)
    @test_throws ArgumentError Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=pair,
        domain=TimeDomain(),time_step=.15,num_energy_bits=30,dry_run=false)
    @test_throws ArgumentError CKGJointKernel(beta;oft=C,rate=gamma,frequency_window=(-8.,8.),time_transform=(;unknown=1))
    # A Gaussian full-normalisation anchor has the same physical generator as
    # the retained analytic typed-rate path.
    rate=GaussianTransition(beta;sigma=.7,sigma_gamma=.9)
    Cg=x->(2pi*.7^2)^(-1/4)*exp(-x^2/(4*.7^2))
    gaussian=CKGJointKernel(beta;oft=Cg,rate=w->transition_value(rate,w),frequency_window=(-8.,8.),time_transform=fine)
    wg=Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=gaussian,jumps=sources,
        domain=TimeDomain(),time_step=.15,num_energy_bits=8)
    wa=Workspace(H;beta_phys=beta,construction=KMS(),transition_weight=rate,jumps=sources)
    @test construct_lindbladian(wg.jumps,wg.cached_cfg,wg.ham_or_trott)≈
        construct_lindbladian(wa.jumps,wa.cached_cfg,wa.ham_or_trott) atol=1e-9 rtol=1e-9
end
