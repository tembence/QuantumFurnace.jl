using Test, QuantumFurnace, LinearAlgebra, Random

# No asserted admissibility trait: only the checked finite Bohr compiler admits it.
struct ResearchPhaseProbe{F} <: AbstractFilter
    beta::Float64
    amplitude::F
end
QuantumFurnace.freq_kernel(f::ResearchPhaseProbe, nu::Real) = f.amplitude(nu)

@testset "T10 complex finite Bohr filters" begin
    QF = QuantumFurnace
    rng = MersenneTwister(710)
    H = pauli_hamiltonian(2, [-0.7 => (1=>:Z,), 0.3 => (2=>:Y,)])
    for Hphys in (H, zeros(ComplexF64,4,4))
        ham = HamHam(Hphys; beta_phys=0.8)
        beta = beta_alg(ham, 0.8)
        A = randn(rng,ComplexF64,4,4)/4
        jumps = prepare_jumps([A,A'],ham).jumps
        for q in (x->exp(-x*x), x->im*x*exp(-x*x), x->exp(-x*x)*cis(0.9x))
            calls = Ref(0)
            raw = ResearchPhaseProbe(beta, x->begin calls[]+=1; q(x)*exp(-beta*x/4) end)
            f = QF._prepare_dll_bohr_filter(raw, ham.eigvals)
            count = calls[]
            @test count == length(unique(vec(ham.bohr_freqs)))
            cfg = Config(;sim=Lindbladian(),domain=BohrDomain(),construction=DLL(),
                num_qubits=2,beta,sigma=1.0,filter=f,with_linear_combination=false)
            ws = Workspace(cfg,ham,jumps)
            dense = construct_lindbladian(jumps,cfg,ham)
            Ls = [q.(ham.bohr_freqs).*exp.(-beta.*ham.bohr_freqs./4).*j.in_eigenbasis for j in jumps]
            R = sum(L'*L for L in Ls)
            B = [im/2*tanh(beta*(ham.eigvals[i]-ham.eigvals[j])/4)*R[i,j] for i in 1:4,j in 1:4]
            I4 = Matrix{ComplexF64}(I,4,4)
            ref = sum(kron(conj(L),L) for L in Ls) -
                (kron(I4,R)+kron(transpose(R),I4))/2 -
                im*(kron(I4,B)-kron(transpose(B),I4))
            @test isapprox(dense,ref; atol=2e-13,rtol=2e-13)
            @test isapprox(QF.dll_coherent_op_bohr(jumps,ham,f,beta),B; atol=2e-13)
            @test isapprox(B,B';atol=2e-13)
            X = randn(rng,ComplexF64,4,4)
            @test isapprox(vec(apply_lindbladian!(ws,X,cfg,ham)),ref*vec(X);atol=2e-13)
            @test isapprox(vec(apply_adjoint_lindbladian!(ws,X,cfg,ham)),ref'*vec(X);atol=2e-13)
            @test norm(ref*vec(Matrix(ham.gibbs))) < 2e-13
            weights = sqrt.(diag(ham.gibbs))
            metric = Diagonal(vec(weights*weights'))
            @test norm(ref*metric-metric*ref') < 2e-13
            freqs = sort(unique(vec(ham.bohr_freqs)))
            alpha = dll_kossakowski_bohr(f,freqs)
            @test norm(alpha-alpha') < 2e-13
            @test eigmin(Hermitian(alpha)) > -2e-13
            multi = DLLMultiChannelFilter([f,f],beta)
            @test isapprox(freq_kernel(multi,0.0),2freq_kernel(f,0.0);atol=2e-13)
            @test isapprox(QF.q_weight(multi,0.0),2QF.q_weight(f,0.0);atol=2e-13)
            @test isapprox(dll_kossakowski_bohr(multi,freqs),2alpha;atol=2e-13)
            @test isapprox(QF.dll_coherent_kernel_bohr(multi,first(freqs),0.0),
                2QF.dll_coherent_kernel_bohr(f,first(freqs),0.0);atol=2e-13)
            @test calls[] == count # construction and matvecs only read the snapshot
        end
    end
    @test_throws ArgumentError QF._prepare_dll_bohr_filter(ResearchPhaseProbe(1.0,x->im),[0.0,0.5])
    @test_throws ArgumentError QF._prepare_dll_bohr_filter(ResearchPhaseProbe(1.0,x->1.0),[0.0,0.5])
    @test_throws ArgumentError QF._prepare_dll_bohr_filter(ResearchPhaseProbe(1.0,x->NaN),[0.0,0.5])
end

@testset "T11 user specifications and physical callbacks" begin
    QF = QuantumFurnace
    beta_phys = 0.8
    phase = KMSFilter(beta_phys;q_positive=x->exp(-x*x/0.7^2)*cis(0.4x),
        name=:phase,parameters=(width=0.7,shift=0.4))
    odd = KMSFilter(beta_phys;q_positive=x->im*x*exp(-x*x),name=:odd)
    rate = RateFilter(beta_phys;downward_rate=x->2exp(-x*x),name=:rate)
    raw = FrequencyFilter(beta_phys;amplitude=x->exp(-x*x/0.7^2-beta_phys*x/4)*cis(0.4x),name=:raw)
    for f in (phase,odd,rate,raw), x in (0.0,0.17,0.8,2.3)
        @test isapprox(freq_kernel(f,x),exp(-beta_phys*x/2)*conj(freq_kernel(f,-x));atol=1e-15,rtol=1e-13)
    end
    for x in (0.0,0.2,1.1)
        @test isapprox(abs2(freq_kernel(rate,-x)),2exp(-x*x);rtol=1e-14)
        @test isapprox(abs2(freq_kernel(rate,x)),2exp(-x*x-beta_phys*x);rtol=1e-14)
        @test isreal(freq_kernel(rate,x))
    end
    @test filter_evidence(phase).algebraic_balance == :conjugate_reflection
    @test filter_evidence(raw).algebraic_balance == :unverified
    @test filter_evidence(phase).continuum_transform == :unknown
    @test filter_evidence(phase).implementation_theorem == :not_established
    @test isapprox(dll_kossakowski_bohr(raw,[0.,0.2]),
        dll_kossakowski_bohr(phase,[0.,0.2]);atol=1e-14)
    @test isapprox(QF.dll_coherent_kernel_bohr(raw,0.,0.2),
        QF.dll_coherent_kernel_bohr(phase,0.,0.2);atol=1e-14)
    H = pauli_hamiltonian(2,[-0.6=>(1=>:Z,),0.2=>(2=>:Y,),0.3=>(1=>:X,2=>:X)])
    for f in (phase,odd,rate,raw)
        ws = Workspace(H;beta_phys,filter=f)
        p = ws.research_provenance
        @test p.filter_compilation.balance == :passed
        @test p.filter_compilation.numerical_scope == :complete_finite_bohr_set
        @test p.clock_label == :raw_generator
        @test p.filter_evidence[1].numerical_checks == :not_run # spec vs compilation
        e = ws.ham_or_trott.eigvals
        expected = freq_kernel.(Ref(f),ws.ham_or_trott.rescaling_factor.*(e.-e'))
        @test isapprox(ws.dll_lindblads[1],expected.*ws.jump_eigenbases[1];atol=1e-14)
        result = simulate_gibbs(ws;times=[0.,0.02],diagnostics=:quick)
        @test length(result.trajectory.t) == 2
        @test result.provenance.filter_compilation.balance == :passed
    end
    w1 = Workspace(H;beta_phys,filter=phase)
    w2 = Workspace(H;beta_phys,filter=raw)
    @test isapprox(w1.G_left,w2.G_left;atol=1e-14)
    @test all(isapprox(a,b;atol=1e-14) for (a,b) in zip(w1.dll_lindblads,w2.dll_lindblads))
    for f in (phase,rate,raw)
        p=prepare_gibbs_inputs(H;beta_phys,filter=f)
        w=Workspace(p.config,p.hamiltonian,p.jumps)
        direct=build_dense_superoperator((out,rho)->copyto!(out,apply_lindbladian!(w,rho,p.config,p.hamiltonian)),4)
        @test isapprox(construct_lindbladian(p.jumps,p.config,p.hamiltonian),direct;atol=1e-12)
    end

    @testset "Validation, support and snapshots" begin
        @test_throws ArgumentError KMSFilter(beta_phys;q_positive=x->im,name=:badzero)
        @test_throws ArgumentError KMSFilter(beta_phys;q_positive=x->NaN,name=:bad)
        @test_throws ArgumentError KMSFilter(beta_phys;q_positive=x->1,name="")
        @test_throws ArgumentError KMSFilter(beta_phys;q_positive=x->1,support=-1,name=:bad)
        @test_throws ArgumentError KMSFilter(beta_phys;q_positive=x->1,logabs_q_positive=x->0,name=:twice)
        @test_throws ArgumentError RateFilter(beta_phys;downward_rate=x->-1,name=:negative)
        @test_throws ArgumentError RateFilter(beta_phys;downward_rate=x->Inf,name=:infinite)
        @test_throws ArgumentError freq_kernel(RateFilter(beta_phys;downward_rate=x->x==0 ? 1 : -1,name=:bad),1.)
        @test_throws ArgumentError Workspace(H;beta_phys,filter=FrequencyFilter(beta_phys;amplitude=x->1,name=:unbalanced))
        @test_throws ArgumentError Workspace(H;beta_phys,filter=FrequencyFilter(beta_phys;amplitude=x->freq_kernel(phase,x)*exp(-beta_phys*x/4),name=:twice))
        @test_throws ArgumentError Workspace(H;beta_phys,filter=phase,transition_weight=x->1)
        @test_throws ArgumentError Workspace(H;beta_phys=0.9,filter=phase)
        @test_throws ArgumentError Workspace(H;beta_phys,filter=phase,domain=TimeDomain(),time_step=0.5,num_energy_bits=8)
        @test_throws ArgumentError Workspace(H;beta_phys,filter=phase,jumps=[ComplexF64[0 1;0 0]])
        # Zero support on active transitions is valid balance but warns about connectivity.
        cut=KMSFilter(beta_phys;q_positive=x->1,name=:cut,support=0.01)
        wc=Workspace([1.0 0;0 -1.0];beta_phys,filter=cut)
        @test wc.research_provenance.filter_compilation.ergodicity_warning == :active_bohr_zeros_may_reduce_connectivity
        wd=Workspace(zeros(2,2);beta_phys,filter=odd)
        @test wd.research_provenance.filter_compilation.ergodicity_warning == :active_bohr_zeros_may_reduce_connectivity
        @test all(iszero,wd.G_left)
        @test iszero(freq_kernel(cut,0.02))
        zerospec=RateFilter(beta_phys;downward_rate=x->0,name=:zero)
        @test iszero(freq_kernel(zerospec,-1.))
        calls=Ref(0); factor=Ref(1.0)
        f=KMSFilter(beta_phys;q_positive=x->begin calls[]+=1; factor[]*exp(-x*x) end,name=:mutable)
        wm=Workspace(H;beta_phys,filter=f)
        before=calls[]; factor[]=2
        X=Matrix{ComplexF64}(I,4,4)/4
        result1=copy(apply_lindbladian!(wm,X,wm.cached_cfg,wm.ham_or_trott))
        factor[]=3
        @test isapprox(apply_lindbladian!(wm,X,wm.cached_cfg,wm.ham_or_trott),result1;atol=1e-15)
        @test calls[] == before
        params=(widths=[0.7],)
        fp=KMSFilter(beta_phys;q_positive=x->1,name=:params,parameters=params)
        params.widths[1]=2
        @test filter_evidence(fp).parameters.widths[1] ≈ 0.7
        time=TimeFilter(beta_phys;kernel=t->exp(-t*t),name=:time,support=2.)
        @test time_kernel(time,0.5) ≈ exp(-0.25)
        @test iszero(time_kernel(time,3.))
        @test filter_evidence(time).support_coordinate == :time
        @test_throws ArgumentError Workspace(H;beta_phys,filter=time)
        @test_throws ArgumentError freq_kernel(time,0.)
    end

    @testset "Steep thermal factors" begin
        # Product is representable although exp(-750) alone underflows.
        steep=RateFilter(1500.;downward_rate=x->exp(400.),name=:steep)
        @test isapprox(freq_kernel(steep,1.),exp(-550.);rtol=1e-12)
        @test isapprox(freq_kernel(steep,-1.),exp(200.);rtol=1e-12)
        qsteep=RateFilter(3000.;downward_rate=x->exp(400.),name=:qsteep)
        @test isapprox(QF.q_weight(qsteep,1.),exp(-550.);rtol=1e-12)
        qsnap=QF._prepare_dll_bohr_filter(qsteep,[0.,1.])
        @test isapprox(QF.q_weight(qsnap,1.),exp(-550.);rtol=1e-12)
        @test QF._prepare_dll_bohr_filter(steep,[0.,1.]) isa AbstractFilter
        steepq=KMSFilter(1500.;q_positive=x->exp(-175.),name=:steepq)
        @test isapprox(freq_kernel(steepq,1.),exp(-550.);rtol=1e-12)
        @test QF._prepare_dll_bohr_filter(steepq,[0.,1.]) isa AbstractFilter
        logarithmic=KMSFilter(4000.;logabs_q_positive=x->-x*x*1000,name=:log)
        @test isapprox(freq_kernel(logarithmic,-1.),1.;atol=1e-14)
        @test iszero(freq_kernel(logarithmic,1.))
        @test QF._prepare_dll_bohr_filter(logarithmic,[0.,1.]) isa AbstractFilter
        zeroq=KMSFilter(1e300;q_positive=x->0,name=:zero)
        @test iszero(freq_kernel(zeroq,-1e100))
        bad=KMSFilter(4000.;q_positive=x->1,name=:overflow)
        @test_throws ArgumentError freq_kernel(bad,-1.)
        @test_throws ArgumentError KMSFilter(beta_phys;logabs_q_positive=x->NaN,name=:badlog)
        @test_throws ArgumentError KMSFilter(beta_phys;logabs_q_positive=x->-big"1e400",name=:precisionoverflow)
        @test_throws ArgumentError KMSFilter(beta_phys;logabs_q_positive=x->0,phase_positive=x->2,name=:badphase)
        @test_throws ArgumentError KMSFilter(beta_phys;logabs_q_positive=x->0,phase_positive=x->im,name=:badorigin)
    end
end

@testset "T11 working precision and documented example" begin
    f=KMSFilter(0.8;q_positive=x->exp(-x*x)*cis(x),name=:precision)
    w=Workspace(Float32[1 0;0 -1];beta_phys=0.8,filter=f)
    @test eltype(w.G_left) == ComplexF32
    @test w.research_provenance.filter_compilation.precision == Float32
    beta_phys=0.8
    filter=KMSFilter(beta_phys;
        q_positive=x->exp(-(x/0.7)^2)*cis(0.4x),
        name=:shifted_gaussian,version="1",parameters=(width=0.7,shift=0.4))
    H=pauli_hamiltonian(2,[-0.7=>(1=>:Z,2=>:Z),-0.4=>(1=>:X,)])
    result=simulate_gibbs(H;beta_phys,filter,times=[0.,0.1],diagnostics=:quick)
    @test length(result.trajectory.t) == 2
    @test result.provenance.filter_compilation.balance == :passed
    @test result.provenance.filter_evidence[1].algebraic_balance == :conjugate_reflection
end
