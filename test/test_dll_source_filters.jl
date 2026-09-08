using Test, QuantumFurnace, LinearAlgebra

@testset "T14 source-specific DLL channels" begin
    QF=QuantumFurnace
    beta_phys=.8
    H=ComplexF64[.2 .3im;-.3im -.2]
    A=ComplexF64[.2+.1im .8-.3im;-.2+.7im .4-.1im]
    X=ComplexF64[0 1;1 0]; Z=ComplexF64[1 0;0 -1]
    g=DLLGaussianFilter(beta_phys); m=DLLMetropolisFilter(beta_phys;S=8.)
    q=KMSFilter(beta_phys;q_positive=x->exp(-(beta_phys*x)^2/8)*cis(.2x),name=:source_phase)
    nested=DLLMultiChannelFilter((g,DLLMultiChannelFilter((m,q),beta_phys)),beta_phys)
    @test length(nested.channels)==3
    @test isconcretetype(typeof(nested.channels))
    @test_throws ArgumentError DLLSourceFilters((1=>g,1=>m),beta_phys)
    @test_throws ArgumentError DLLSourceFilters((2=>g,),beta_phys)
    @test_throws ArgumentError Workspace(H;beta_phys,jumps=[A,A'],filter=DLLSourceFilters((g,m),beta_phys))
    @test_throws ArgumentError Workspace(H;beta_phys,jumps=[X,Z],filter=DLLSourceFilters((g,),beta_phys))
    dense(ws)=build_dense_superoperator((out,rho)->copyto!(out,apply_lindbladian!(ws,rho,ws.cached_cfg,ws.ham_or_trott)),2)
    ws=Workspace(H;beta_phys,jumps=[X,Z],filter=DLLSourceFilters((nested,q),beta_phys))
    ham=ws.ham_or_trott; jumps=prepare_jumps([X,Z],ham).jumps
    # Independent physical-amplitude matrices and direct Kronecker GKLS assembly.
    Ls=[freq_kernel.(Ref(f),ham.bohr_freqs.*ham.rescaling_factor).*j.in_eigenbasis
        for (j,fs) in zip(jumps,(nested.channels,(q,))) for f in fs]
    R=sum(L'*L for L in Ls)
    B=[im/2*tanh(ws.cached_cfg.beta*(ham.eigvals[i]-ham.eigvals[j])/4)*R[i,j] for i in 1:2,j in 1:2]
    id=Matrix{ComplexF64}(I,2,2)
    ref=sum(kron(conj(L),L) for L in Ls)-(kron(id,R)+kron(transpose(R),id))/2-im*(kron(id,B)-kron(transpose(B),id))
    @test dense(ws) ≈ ref atol=1e-12
    @test construct_lindbladian(ws.jumps,ws.cached_cfg,ham) ≈ ref atol=1e-12
    @test norm(ref*vec(Matrix(ham.gibbs)))<1e-12
    w=sqrt.(diag(ham.gibbs)); metric=Diagonal(vec(w*w'))
    @test norm(ref*metric-metric*ref')<1e-12
    perm=Workspace(H;beta_phys,jumps=[Z,X],filter=DLLSourceFilters((q,nested),beta_phys))
    @test dense(perm) ≈ ref atol=1e-12
    @test ws.research_provenance.source_filter_assignments.channel_counts==(3,1)
    pair=Workspace(H;beta_phys,jumps=[A,A'],filter=DLLSourceFilters((nested,DLLMultiChannelFilter(reverse(nested.channels),beta_phys)),beta_phys))
    globalpair=Workspace(H;beta_phys,jumps=[A,A'],filter=nested)
    @test dense(pair) ≈ dense(globalpair) atol=1e-12
    singleton=Workspace(H;beta_phys,jumps=[A,A'],filter=DLLSourceFilters((g,DLLMultiChannelFilter((g,),beta_phys)),beta_phys))
    @test dense(singleton) ≈ dense(Workspace(H;beta_phys,jumps=[A,A'],filter=g)) atol=1e-12
    @test_throws ArgumentError Workspace(H;beta_phys,jumps=[A,A'],filter=DLLSourceFilters((g,DLLMultiChannelFilter((g,g),beta_phys)),beta_phys))
    single=Workspace(H;beta_phys,jumps=[X],filter=g)
    twice=Workspace(H;beta_phys,jumps=[X],filter=DLLMultiChannelFilter((g,g),beta_phys))
    @test dense(twice) ≈ 2dense(single) atol=1e-12
    @test norm(dense(twice)-4dense(single))>1e-2 # D[L+L] would quadruple it.
    completed=Workspace(H;beta_phys,jumps=[A],complete_adjoint=true,filter=DLLSourceFilters((q,),beta_phys))
    @test length(completed.dll_lindblads)==2
    @test completed.research_provenance.source_filter_assignments.partners==(2,1)
    # Distinct callbacks with identical names are not silently treated as one prescription.
    other=KMSFilter(beta_phys;q_positive=x->exp(-(beta_phys*x)^2/8),name=:source_phase)
    @test_throws ArgumentError Workspace(H;beta_phys,jumps=[A,A'],filter=DLLSourceFilters((q,other),beta_phys))
    raw=FrequencyFilter(beta_phys;amplitude=x->freq_kernel(g,x),name=:raw_gaussian)
    @test dense(Workspace(H;beta_phys,jumps=[X],filter=DLLMultiChannelFilter((raw,raw),beta_phys))) ≈ dense(twice) atol=1e-12
    invalidraw=FrequencyFilter(beta_phys;amplitude=x->exp(-x^2),name=:invalid_raw)
    @test_throws ArgumentError Workspace(H;beta_phys,jumps=[X],filter=DLLMultiChannelFilter((invalidraw,g),beta_phys))
    # Mixed incoming precision is converted once at the Hamiltonian boundary.
    mixed=DLLMultiChannelFilter((DLLGaussianFilter(Float32(beta_phys)),g),beta_phys)
    @test length(Workspace(H;beta_phys,jumps=[X],filter=mixed).dll_lindblads)==2
    rho=ComplexF64[.3 .4im;.7 .2-.1im]
    @test vec(apply_adjoint_lindbladian!(ws,rho,ws.cached_cfg,ham)) ≈ ref'*vec(rho) atol=1e-12
    # Prepared heterogeneous channels share no amplitudes or loss corrections.
    pg=prepare_filter_transform(g;coherent=(;method=:hybrid))
    pq=prepare_filter_transform(q;window=20.,coherent=(;method=:hybrid))
    timefamily=DLLMultiChannelFilter((pg,pq),beta_phys)
    tw=Workspace(H;beta_phys,jumps=[A,A'],filter=DLLSourceFilters((timefamily,timefamily),beta_phys),
        domain=TimeDomain(),time_step=.12,num_energy_bits=7)
    bw=Workspace(H;beta_phys,jumps=[A,A'],filter=DLLMultiChannelFilter((g,q),beta_phys))
    @test dense(tw) ≈ dense(bw) atol=1e-9
    @test construct_lindbladian(tw.jumps,tw.cached_cfg,tw.ham_or_trott) ≈ dense(tw) atol=1e-11
    @test length(tw.research_provenance.time_compilation.sources)==2
    @test all(c.method==:time_jumps_implemented_loss_correction for s in tw.research_provenance.time_compilation.sources for c in s.channels)
    capture=Ref(1.)
    captured=KMSFilter(beta_phys;q_positive=x->capture[]*exp(-(beta_phys*x)^2/8)*cis(.2x),name=:capture)
    pc=prepare_filter_transform(captured;window=20.,coherent=(;method=:hybrid))
    ordered=DLLMultiChannelFilter((pc,pg),beta_phys)
    reversed=DLLMultiChannelFilter((pg,pc),beta_phys)
    captured_ws=Workspace(H;beta_phys,jumps=[A,A'],filter=DLLSourceFilters((ordered,reversed),beta_phys),
        domain=TimeDomain(),time_step=.12,num_energy_bits=7)
    @test dense(captured_ws) ≈ dense(bw) atol=1e-9
    capture[]=3.
    @test dense(captured_ws) ≈ dense(bw) atol=1e-9
    # Repeated compilation/thread reductions are deterministic, with concrete hot storage.
    repeated=Workspace(H;beta_phys,jumps=[X,Z],filter=DLLSourceFilters((nested,q),beta_phys))
    @test dense(repeated) ≈ ref atol=1e-13
    @test eltype(ws.dll_lindblads)==Matrix{ComplexF64}
    apply_lindbladian!(ws,rho,ws.cached_cfg,ham)
    bytes=@allocated apply_lindbladian!(ws,rho,ws.cached_cfg,ham)
    apply_lindbladian!(single,rho,single.cached_cfg,single.ham_or_trott)
    baseline=@allocated apply_lindbladian!(single,rho,single.cached_cfg,single.ham_or_trott)
    @test bytes<=baseline+4096
end
