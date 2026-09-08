using BSON
using QuadGK: quadgk

@testset "Portable research results and continuation" begin
    QF=QuantumFurnace
    H=Hermitian(.3X+.4Y+.7Z)
    quiet=(;diagnostics=:quick,gap_options=(;max_matvecs=0))
    mktempdir() do dir
        r=simulate_gibbs(H;beta_phys=.8,times=[0.,.2],save_states=true,
            clock=GeneratorClock{Float64}(:saved_clock,2.5,1.),quiet...)
        path=save_result(r,joinpath(dir,"research.bson"))
        s=load_result(path)
        @test s isa GibbsSimulationResult
        @test s.trajectory.rho_final ≈ r.trajectory.rho_final atol=1e-14
        @test s.diagnostics isa GibbsDiagnostics
        @test s.diagnostics.checks.stationarity isa DiagnosticCheck
        @test s.spectrum.resources.exhausted
        @test s.provenance.rng.gap_seed==0x708
        @test s.provenance.runtime.julia_version==string(VERSION)
        @test s.provenance.runtime.dirty_worktree isa Union{Bool,Symbol}
        @test isfile(joinpath(dir,"research.txt"))
        @test isequal(QF._result_to_dict(s),QF._result_to_dict(r))
        # Primitive data tags only: no package objects or callbacks in BSON.
        primitive(x)=x===nothing || x isa Union{Number,Symbol,AbstractString} ||
            x isa AbstractArray && all(primitive,x) ||
            x isa AbstractDict && all(primitive,keys(x)) && all(primitive,values(x))
        @test primitive(BSON.load(path))
        w=Workspace(s)
        @test w.research_provenance.clock_label==:saved_clock
        continued=simulate_gibbs(s;times=[0.,.3])
        reference=simulate_gibbs(H;beta_phys=.8,times=[0.,.5],
            clock=GeneratorClock{Float64}(:saved_clock,2.5,1.),quiet...)
        @test continued.trajectory.rho_final ≈ reference.trajectory.rho_final atol=1e-10
        @test continued.provenance.resume_time_origin ≈ .2
        @test continued.provenance.generator_multiplier ≈ 2.5
        @test_throws ArgumentError simulate_gibbs(s;times=Float64[])
        # Model, temperature, domain and controls are all in the replay digest.
        for key in (:beta_phys,:domain,:time_step)
            broken=deepcopy(s); broken.provenance.replay[key]=key==:domain ? "TimeDomain" : 7.
            @test_throws ArgumentError Workspace(broken)
        end
        # Explicitly saved basis remains the same eigenbasis through continuation.
        eigen=simulate_gibbs(H;beta_phys=.8,times=[0.,.2],basis=:eigen,quiet...)
        save_result(eigen,joinpath(dir,"eigen.bson"))
        resumed=simulate_gibbs(load_result(joinpath(dir,"eigen.bson"));times=[0.,.3])
        full=simulate_gibbs(H;beta_phys=.8,times=[0.,.5],basis=:eigen,quiet...)
        @test resumed.trajectory.rho_final ≈ full.trajectory.rho_final atol=1e-10
        # A separate Julia process has no closures or current-session definitions.
        script=joinpath(dir,"reload.jl")
        write(script,"using QuantumFurnace, LinearAlgebra\nBLAS.set_num_threads(1)\nr=load_result(ARGS[1])\nw=Workspace(r)\n@assert w.research_provenance.clock_label==:saved_clock\n@assert r.provenance.resources.estimated_bytes > 0\n@assert norm(simulate_gibbs(r;times=[0.,.0+0.01]).trajectory.rho_final)>0\n")
        cmd=`$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) --startup-file=no --heap-size-hint=800M $script $path`
        @test success(pipeline(cmd;stdout=devnull,stderr=stderr))
        # Budget exhaustion still saves the valid initial-state checkpoint.
        partial=simulate_gibbs(H;beta_phys=.8,times=[0.,1.],max_matvecs=0,quiet...)
        p=load_result(save_result(partial,joinpath(dir,"partial.bson")))
        @test p.convergence.status==:inconclusive
        @test last(p.trajectory.t) ≈ 0.
        @test Workspace(p) isa Workspace
    end

    @testset "Custom definitions and frozen samples" begin
        capture=Ref(1.)
        f=KMSFilter(.8;q_positive=x->capture[]*exp(-x^2),name=:t19_closure)
        ws=Workspace(H;beta_phys=.8,filter=f)
        r=simulate_gibbs(ws;times=[0.,.1],quiet...)
        mktempdir() do dir
            path=save_result(r,joinpath(dir,"closure.bson"))
            s=load_result(path)
            @test_throws ArgumentError Workspace(s)
            w=Workspace(s;filters=Dict((:t19_closure,"1")=>f))
            @test w.G_left ≈ ws.G_left atol=1e-13
            @test w.dll_lindblads[1] ≈ ws.dll_lindblads[1] atol=1e-13
            capture[]=2.
            # Saving never calls or resamples the original mutable closure.
            saved_again=save_result(r,joinpath(dir,"closure_after_mutation.bson"))
            @test isequal(QF._result_to_dict(load_result(saved_again)),QF._result_to_dict(s))
            @test_throws ArgumentError Workspace(s;filters=Dict((:t19_closure,"1")=>f))
            wrong=KMSFilter(.8;q_positive=x->exp(-x^2),name=:t19_closure,version="2")
            @test_throws ArgumentError Workspace(s;filters=Dict((:t19_closure,"1")=>wrong))
        end
        factory=(beta,p)->KMSFilter(beta;q_positive=x->exp(-p.width*x^2),
            name=:t19_registered,version="3",parameters=p)
        register_filter!(:t19_registered,"3",factory)
        @test_throws ArgumentError register_filter!(:t19_registered,"3",factory)
        registered=factory(.8,(;width=1.))
        r=simulate_gibbs(H;beta_phys=.8,filter=registered,times=[0.,.1],quiet...)
        mktempdir() do dir
            s=load_result(save_result(r,joinpath(dir,"registered.bson")))
            @test Workspace(s) isa Workspace
            delete!(QF._FILTER_REGISTRY,(:t19_registered,"3"))
            @test_throws ArgumentError Workspace(s)
        end
    end

    @testset "Prescription and old-schema compatibility" begin
        beta=.8
        for value in (big(2)^128,BigFloat("0.125"),Complex{BigFloat}(1,2),BigFloat[1 2;3 4],
            big(2)//big(3),Complex{BigInt}(2,3),Symbol[],String[])
            @test isequal(QF._portable_unpack(QF._portable_pack(value)),value)
            @test typeof(QF._portable_unpack(QF._portable_pack(value)))==typeof(value)
        end
        filters=(DLLGaussianFilter(beta),DLLMetropolisFilter(beta;S=3.),
            DLLMultiChannelFilter((DLLGaussianFilter(beta),DLLMetropolisFilter(beta)),beta),
            DLLSourceFilters((DLLGaussianFilter(beta),DLLMetropolisFilter(beta),DLLGaussianFilter(beta)),beta),
            prepare_filter_transform(DLLGaussianFilter(beta)))
        for f in filters
            decoded=QF._portable_unpack(QF._portable_pack(f);restore=true)
            @test typeof(decoded)==typeof(f)
            if f isa PreparedFilterTransform
                @test isempty(decoded.cache)
            elseif f isa Union{DLLGaussianFilter,DLLMetropolisFilter}
                @test freq_kernel(decoded,.3) ≈ freq_kernel(f,.3) atol=1e-14
            end
        end
        r=simulate_gibbs(H;beta_phys=beta,construction=KMS(),
            transition_weight=GaussianTransition(beta),times=[0.,.1],quiet...)
        mktempdir() do dir
            s=load_result(save_result(r,joinpath(dir,"ckg.bson")))
            w=Workspace(s)
            @test w.cached_cfg.construction isa KMS
            @test simulate_gibbs(s;times=[0.,.1]).trajectory.rho_final ≈
                simulate_gibbs(H;beta_phys=beta,construction=KMS(),transition_weight=GaussianTransition(beta),times=[0.,.2],quiet...).trajectory.rho_final atol=1e-10
        end
        # Legacy config objects (pre-portable filter tags) remain accepted.
        cfg=Config(;sim=Lindbladian(),domain=BohrDomain(),construction=DLL(),
            num_qubits=1,with_linear_combination=false,beta,sigma=1.,filter=DLLGaussianFilter(beta))
        d=QF._config_to_dict(cfg)
        d[:filter]=cfg.filter
        @test isequal(QF._reconstruct_config(d).filter,cfg.filter)
    end
    @testset "Strict diagnostics and compiled transform replay" begin
        # Complete parent-spectrum evidence has its own typed codec.
        strict=simulate_gibbs(H;beta_phys=.8,times=[0.,.02],diagnostics=:strict)
        mktempdir() do dir
            s=load_result(save_result(strict,joinpath(dir,"strict.bson")))
            @test s.diagnostics.parent_spectrum isa KMSParentSpectrum
            @test s.diagnostics.parent_spectrum.eigenvalues ≈ strict.diagnostics.parent_spectrum.eigenvalues atol=1e-14
            @test s.provenance.initial_density_matrix ≈ fill(.5+0im,2,2) atol=1e-14
            @test s.provenance.requested_times ≈ [0.,.02] atol=1e-14
        end
        beta=.8; model=ComplexF64[.2 .3im;-.3im -.2]
        A=ComplexF64[.2+.1im .8-.3im;-.2+.7im .4-.1im]; sources=[A,Matrix(A')]
        custom=KMSFilter(beta;q_positive=x->exp(-(beta*x)^2/8)*cis(.2x),name=:t19_time)
        prepared=prepare_filter_transform(custom;window=20.,coherent=(;time_step=.12,time_window=6.,frequency_grid_size=257,policy=:error))
        timed=Workspace(model;beta_phys=beta,filter=prepared,jumps=sources,
            domain=TimeDomain(),time_step=.12,num_energy_bits=7)
        mixture=GaussianMixtureTransition(beta;centers=(1.4,2.2),weights=(.3,.7))
        energy=Workspace(model;beta_phys=beta,construction=KMS(),transition_weight=mixture,
            domain=EnergyDomain(),energy_step=.0625,num_energy_bits=8)
        normalization=inv(sqrt(first(quadgk(x->exp(beta*x/2-2x^4),-Inf,Inf;rtol=1e-12))))
        C=x->normalization*exp(beta*x/4-x^4)*cis(.1x^3)
        gamma=w->exp(-w^2-beta*w/2)
        joint=CKGJointKernel(beta;oft=C,rate=gamma,frequency_window=(-8.,8.),
            time_transform=(;frequency_window=8.,frequency_grid_size=257,
                coherent_frequency_window=8.,coherent_frequency_grid_size=129,
                coherent_time_window=19.2,coherent_time_step=.15,backend=:direct))
        joint_time=Workspace(ComplexF64[-.35 0;0 .35];beta_phys=beta,construction=KMS(),
            transition_weight=joint,jumps=sources,domain=TimeDomain(),time_step=.15,num_energy_bits=8)
        for (ws,definitions) in ((timed,Dict((:t19_time,"1")=>custom)),(energy,Dict()),(joint_time,Dict(:ckg_joint=>joint)))
            r=simulate_gibbs(ws;times=[0.,.01],quiet...)
            mktempdir() do dir
                s=load_result(save_result(r,joinpath(dir,"transform.bson")))
                restored=Workspace(s;filters=definitions)
                probe=ComplexF64[.3 .1im;-.1im .7]
                expected=copy(apply_lindbladian!(ws,probe,ws.cached_cfg,ws.ham_or_trott))
                actual=apply_lindbladian!(restored,probe,restored.cached_cfg,restored.ham_or_trott)
                @test actual ≈ expected atol=1e-11 rtol=1e-11
                @test restored.G_left !== ws.G_left
                continuation=simulate_gibbs(s;times=[0.,.01],filters=definitions)
                full=simulate_gibbs(ws;times=[0.,.02],quiet...)
                @test continuation.trajectory.rho_final ≈ full.trajectory.rho_final atol=1e-10
                if ws===joint_time
                    @test_throws ArgumentError Workspace(s)
                end
            end
        end
    end

    @testset "Mixed source-family replay" begin
        A=ComplexF64[.2+.1im .8-.3im;-.2+.7im .4-.1im]
        f=KMSFilter(.8;q_positive=x->exp(-x^2),name=:t19_source_family)
        family=DLLMultiChannelFilter((f,DLLGaussianFilter(.8)),.8)
        assignment=DLLSourceFilters((family,family),.8)
        w=Workspace(H;beta_phys=.8,jumps=[A,Matrix(A')],filter=assignment,rates=[.7,.7])
        r=simulate_gibbs(w;times=[0.,.01],quiet...)
        mktempdir() do dir
            s=load_result(save_result(r,joinpath(dir,"source_family.bson")))
            restored=Workspace(s;filters=Dict((:t19_source_family,"1")=>f))
            @test restored.G_left ≈ w.G_left atol=1e-12
            @test all(isapprox(a,b;atol=1e-12) for (a,b) in zip(restored.dll_lindblads,w.dll_lindblads))
        end
        @test isempty(QF._portable_unpack(QF._portable_pack(Union{}[])))
    end

end
