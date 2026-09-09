using Test, QuantumFurnace, LinearAlgebra
using QuadGK

@testset "prepared Fourier transforms" begin
    beta=1.3
    gaussian=DLLGaussianFilter(beta)
    targets=[-3.,-0.4,0.,0.7,4.]
    p=prepare_filter_transform(gaussian)
    @test transform_values(p,targets).values ≈ time_kernel.(Ref(gaussian),targets) atol=1e-14
    pn=prepare_filter_transform(gaussian;window=16/beta,analytic=false)
    report=transform_values(pn,targets)
    @test report.values ≈ time_kernel.(Ref(gaussian),targets) atol=1e-12
    @test report.tail_status==:unknown
    @test report.status==:estimated
    @test maximum(report.quadrature_estimates)<1e-11
    for sigma in (0.4,1.7)
        g=GaussianFilter(sigma)
        pg=prepare_filter_transform(g)
        ng=prepare_filter_transform(g;window=20sigma,analytic=false)
        @test transform_values(ng,targets).values ≈ time_kernel.(Ref(g),targets) atol=1e-12
        @test freq_kernel(pg,0.) ≈ sqrt(pi)/sigma atol=1e-14
        # Independently integrated forward transform, not a second adapter.
        for nu in (-1.,0.,0.6)
            v,e=quadgk(t->exp(-sigma^2*t^2)*cis(nu*t),-Inf,Inf;rtol=1e-12)
            @test freq_kernel(pg,nu) ≈ v atol=1e-12
        end
    end
    shift=0.8
    f=KMSFilter(beta;q_positive=x->exp(-(beta*x)^2/8)*cis(shift*x),name=:shift)
    pf=prepare_filter_transform(f;window=16/beta)
    @test transform_values(pf,targets).values ≈ time_kernel.(Ref(gaussian),targets.-shift) atol=1e-12
    tf=TimeFilter(beta;kernel=t->time_kernel(gaussian,t-shift),name=:shift_time)
    pt=prepare_filter_transform(tf;window=10.)
    @test transform_values(pt,targets;direction=:forward).values ≈ freq_kernel.(Ref(f),targets) atol=1e-11
    # Separately enlarge the window; then tighten quadrature on fixed extent.
    small=prepare_filter_transform(f;window=2.)
    @test norm(transform_values(small,targets).values-time_kernel.(Ref(gaussian),targets.-shift))>1e-3
    @test norm(transform_values(pf,targets).values-time_kernel.(Ref(gaussian),targets.-shift))<1e-9
    @test maximum(transform_values(small,targets).window_refinement_differences)>1e-3
    @test maximum(transform_values(pf,targets).window_refinement_differences)<1e-11
    unresolved=prepare_filter_transform(f;window=16/beta,maxevals=1,rtol=1e-15,atol=1e-15)
    @test transform_values(unresolved,[0.];window_refinements=0).status==:unresolved
    @test_throws r"quadrature budget exhausted" time_kernel(unresolved,0.)
    @test transform_values(pf,[0.]).status==:estimated
    metro=DLLMetropolisFilter(beta;S=3.)
    pm=prepare_filter_transform(metro;breakpoints=[-1.,1.])
    @test transform_values(pm,targets).values ≈ time_kernel.(Ref(metro),targets) atol=1e-11
    @test transform_values(pm,targets).tail_status==:declared_support
    @test transform_values(pm,targets).tail_bound ≈ 0 atol=eps()
    # Compact rectangle is intentionally not a DLL balance example.
    box=FrequencyFilter(beta;amplitude=x->1.,support=2.,name=:box)
    pb=prepare_filter_transform(box;breakpoints=[-.3,.3])
    @test time_kernel(pb,0.) ≈ 2/pi atol=1e-12
    @test time_kernel(pb,7.) ≈ sin(14)/(7pi) atol=1e-12
    @test_throws ArgumentError time_kernel(pb,1e8)
    @test_throws ArgumentError prepare_filter_transform(f)
    @test_throws ArgumentError prepare_filter_transform(f;window=-1.)
    @test_throws ArgumentError prepare_filter_transform(box;breakpoints=[3.])
    @test_throws ArgumentError transform_values(pb,[0.];direction=:bad)
    @test_throws ArgumentError transform_values(pb,[Inf])
    declared=FrequencyFilter(beta;amplitude=x->exp(-x^2),name=:tail,
        tail_bound=W->sqrt(pi)*exp(-W^2))
    pd=prepare_filter_transform(declared;window=5.)
    @test transform_values(pd,[0.]).tail_status==:user_supplied_unverified
    @test transform_values(pd,[0.]).tail_bound ≈ exp(-25)/(2sqrt(pi)) atol=1e-20
    # Captured Ref belongs to the copied closure; a rebuilt plan sees changes.
    mutable_filter=let amplitude=Ref(1.)
        filter=FrequencyFilter(beta;amplitude=x->amplitude[]*exp(-x^2),name=:mutable,support=5.)
        (filter,amplitude)
    end
    frozen=prepare_filter_transform(mutable_filter[1])
    value=time_kernel(frozen,0.)
    mutable_filter[2][]=3.
    @test time_kernel(frozen,0.) ≈ value atol=1e-14
    @test time_kernel(frozen,0.2) ≈ value*exp(-.01) atol=1e-11
    @test time_kernel(prepare_filter_transform(mutable_filter[1]),0.) ≈ 3value atol=1e-12
    nodes=collect(range(-8.,8.;length=1001)); dx=step(range(-8.,8.;length=1001))
    weights=exp.(-nodes.^2).*dx
    direct=fourier_sum(nodes,weights,targets)
    @test direct ≈ sqrt(pi).*exp.(-targets.^2/4) atol=1e-12
    fast=fourier_sum(nodes,weights,targets;backend=:finufft)
    @test fast ≈ direct atol=1e-10
    @test fourier_sum(nodes,weights,targets;backend=:finufft) ≈ fast atol=1e-14
    @test fourier_sum(BigFloat.(nodes),Complex{BigFloat}.(weights),BigFloat.(targets)) isa Vector{Complex{BigFloat}}
    @test_throws ArgumentError fourier_sum(BigFloat.(nodes),weights,targets;backend=:finufft)
    @test_throws DimensionMismatch fourier_sum(nodes,[1.],targets)
    @test_throws ArgumentError fourier_sum(nodes,weights,targets;sign=0)
    @test_throws ArgumentError fourier_sum(nodes,weights,targets;backend=:unknown)
end

@testset "independent DLL Time controls" begin
    QF=QuantumFurnace
    H=ComplexF64[.2 .3im;-.3im -.2]; beta=.8
    A=ComplexF64[.2+.1im .8-.3im;-.2+.7im .4-.1im]
    sources=[A,A']
    shift=.2
    custom=KMSFilter(beta;q_positive=x->exp(-(beta*x)^2/8)*cis(shift*x),name=:phase_time)
    controls=(;time_step=.12,time_window=6.,frequency_grid_size=257,policy=:error)
    prepared=prepare_filter_transform(custom;window=20.,coherent=controls)
    @test only(DLLMultiChannelFilter([prepared],beta).channels) === prepared
    @test_throws ArgumentError prepare_filter_transform(DLLMultiChannelFilter([custom,custom],beta);window=20.)
    @test_throws ArgumentError prepare_filter_transform(prepared;window=20.)
    exact=Workspace(H;beta_phys=beta,filter=custom,jumps=sources)
    timed=Workspace(H;beta_phys=beta,filter=prepared,jumps=sources,
        domain=TimeDomain(),time_step=.12,num_energy_bits=7)
    hybrid=Workspace(H;beta_phys=beta,jumps=sources,
        filter=prepare_filter_transform(custom;window=20.,coherent=(;controls...,method=:hybrid)),
        domain=TimeDomain(),time_step=.12,num_energy_bits=7)
    ham=timed.ham_or_trott; cfg=timed.cached_cfg
    # Independent full GKLS reference from physical-frequency amplitudes and
    # source products. No coherent-grid or loss-correction helper is reused.
    jumps=prepare_jumps(sources,ham).jumps
    Ls=[freq_kernel.(Ref(custom),ham.bohr_freqs.*ham.rescaling_factor).*j.in_eigenbasis for j in jumps]
    R=sum(L'*L for L in Ls)
    B=[im/2*tanh(cfg.beta*(ham.eigvals[i]-ham.eigvals[j])/4)*R[i,j] for i in 1:2,j in 1:2]
    id=Matrix{ComplexF64}(I,2,2)
    reference=sum(kron(conj(L),L) for L in Ls)-(kron(id,R)+kron(transpose(R),id))/2-
        im*(kron(id,B)-kron(transpose(B),id))
    dense=build_dense_superoperator((out,rho)->copyto!(out,apply_lindbladian!(timed,rho,cfg,ham)),2)
    @test dense ≈ reference atol=1e-9 rtol=1e-9
    @test dense ≈ construct_lindbladian(timed.jumps,cfg,ham) atol=1e-10
    hd=construct_lindbladian(hybrid.jumps,hybrid.cached_cfg,hybrid.ham_or_trott)
    @test hd ≈ reference atol=1e-9
    @test -(timed.G_left+timed.G_right) ≈ R atol=1e-9
    @test (timed.G_right-timed.G_left)/(2im) ≈ B atol=1e-9
    @test all(norm(x-y)<1e-9 for (x,y) in zip(timed.dll_lindblads,Ls))
    @test timed.G_left ≈ hybrid.G_left atol=1e-9
    @test timed.G_left ≈ exact.G_left atol=1e-9
    @test norm(dense*vec(Matrix(ham.gibbs)))<1e-9
    weights=sqrt.(diag(ham.gibbs)); metric=Diagonal(vec(weights*weights'))
    @test norm(dense*metric-metric*dense')<1e-9
    @test norm(vec(id)'*dense)<1e-12
    @test sort(real.(eigvals(dense))) ≈ sort(real.(eigvals(reference))) atol=1e-9
    x=ComplexF64[.3 .4im;.7 .2-.1im]
    @test vec(apply_adjoint_lindbladian!(timed,x,cfg,ham)) ≈ dense'*vec(x) atol=1e-11
    evidence=timed.research_provenance.time_compilation
    @test evidence.status==:estimated
    @test evidence.method==:full_two_time_quadrature
    @test maximum(values(evidence.coherent_refinements))<1e-9
    @test evidence.hermitian_repair_norm ≈ 0 atol=eps()
    @test evidence.coherent_time_window != .12*64*ham.rescaling_factor
    @test hybrid.research_provenance.time_compilation.method==:time_jumps_implemented_loss_correction
    # Time-input forward compilation agrees, including the rescaling Jacobian.
    gaussian=DLLGaussianFilter(beta)
    timeinput=TimeFilter(beta;kernel=t->time_kernel(gaussian,t-shift),name=:time_input)
    pt=prepare_filter_transform(timeinput;window=8.,coherent=(;controls...,frequency_window=20.))
    tw=Workspace(H;beta_phys=beta,filter=pt,jumps=sources,domain=TimeDomain(),time_step=.12,num_energy_bits=7)
    @test tw.G_left ≈ exact.G_left atol=1e-9
    @test all(norm(x-y)<1e-9 for (x,y) in zip(tw.dll_lindblads,Ls))
    # Compare generic direct double Fourier sums to FINUFFT on the SAME coarse
    # grid; unlike a cross-domain gate this checks only finite-sum evaluation.
    ap=timed.cached_cfg.filter
    labels=collect(-8:8).*.3
    gd=QF._dll_coherent_op_time_frequency_grid(jumps,ham,labels,ap,cfg.beta,.3;
        nu_min=-10.,nu_max=10.,nu_grid_size=49,backend=:direct)
    gn=QF._dll_coherent_op_time_frequency_grid(jumps,ham,labels,ap,cfg.beta,.3;
        nu_min=-10.,nu_max=10.,nu_grid_size=49,backend=:finufft)
    @test gd ≈ gn atol=1e-10
    coarse=prepare_filter_transform(custom;window=20.,coherent=(;time_step=.6,time_window=.6,frequency_grid_size=65,policy=:error))
    @test_throws r"accuracy unresolved" Workspace(H;beta_phys=beta,filter=coarse,jumps=sources,domain=TimeDomain(),time_step=.6,num_energy_bits=3)
    warned=prepare_filter_transform(custom;window=20.,coherent=(;time_step=.6,time_window=.6,frequency_grid_size=65,policy=:warn))
    cw=@test_logs (:warn,r"accuracy unresolved") Workspace(H;beta_phys=beta,filter=warned,jumps=sources,domain=TimeDomain(),time_step=.6,num_energy_bits=3)
    @test cw.research_provenance.time_compilation.status==:unresolved
    @test norm(cw.G_left-exact.G_left)>norm(timed.G_left-exact.G_left)
    @test_throws ArgumentError prepare_filter_transform(custom;window=20.,coherent=(;nonsense=1))
    @test_throws ArgumentError Workspace(H;beta_phys=beta,filter=prepared,domain=TimeDomain(),time_step=.12,num_energy_bits=7,max_bytes=1)
    invalid=prepare_filter_transform(FrequencyFilter(beta;amplitude=x->exp(-x^2),name=:invalid);window=8.,coherent=controls)
    @test_throws r"weighted reflection" Workspace(H;beta_phys=beta,filter=invalid,domain=TimeDomain(),time_step=.12,num_energy_bits=7)
    # The documented facade carries full Time provenance through propagation.
    result=simulate_gibbs(timed;times=[0.,.05],diagnostics=:quick)
    @test result.provenance.time_compilation.method==:full_two_time_quadrature
    @test all(isfinite,result.trajectory.distances)
end

@testset "compact Metropolis convergence and budget gates" begin
    H=ComplexF64[.2 .3im;-.3im -.2]; beta=.8
    A=ComplexF64[.2+.1im .8-.3im;-.2+.7im .4-.1im]; sources=[A,A']
    f=DLLMetropolisFilter(beta;S=8.)
    p=prepare_filter_transform(f;coherent=(;time_step=.12,time_window=48.,
        frequency_grid_size=513,policy=:error))
    timed=Workspace(H;beta_phys=beta,filter=p,jumps=sources,domain=TimeDomain(),
        time_step=.12,num_energy_bits=10,max_bytes=600*1024^2)
    exact=Workspace(H;beta_phys=beta,filter=f,jumps=sources)
    @test all(norm(x-y)<1e-9 for (x,y) in zip(timed.dll_lindblads,exact.dll_lindblads))
    @test timed.G_left ≈ exact.G_left atol=1e-9 rtol=1e-9
    @test timed.G_right ≈ exact.G_right atol=1e-9 rtol=1e-9
    report=timed.research_provenance.time_compilation
    @test report.status==:estimated
    @test maximum(values(report.coherent_refinements))<1e-9
    @test report.dissipative_amplitude_error<1e-9
    h=timed.ham_or_trott; c=timed.cached_cfg
    @test norm(apply_lindbladian!(timed,Matrix(h.gibbs),c,h))<1e-9
    dense=build_dense_superoperator((out,rho)->copyto!(out,apply_lindbladian!(timed,rho,c,h)),2)
    weights=sqrt.(diag(h.gibbs)); metric=Diagonal(vec(weights*weights'))
    @test norm(dense*metric-metric*dense')<1e-9
    ref=construct_lindbladian(exact.jumps,exact.cached_cfg,exact.ham_or_trott)
    @test dense ≈ ref atol=1e-9 rtol=1e-9
    @test sort(real.(eigvals(dense))) ≈ sort(real.(eigvals(ref))) atol=1e-9
    # The previous half-window has accurate coherent quadrature but insufficient
    # dissipative extent: keep coherent controls fixed and refine just that grid.
    @test_throws r"accuracy unresolved" Workspace(H;beta_phys=beta,filter=p,jumps=sources,
        domain=TimeDomain(),time_step=.12,num_energy_bits=9,max_bytes=600*1024^2)
    # Degenerate zero-frequency balance cannot hide exhausted forward quadrature.
    bad=prepare_filter_transform(TimeFilter(beta;kernel=t->exp(-t*t),support=8.,name=:budget);maxevals=1)
    @test_throws r"quadrature budget exhausted" Workspace(zeros(2,2);beta_phys=beta,filter=bad)
    @test transform_values(bad,[0.];direction=:forward).status==:unresolved
    expanded=prepare_filter_transform(FrequencyFilter(beta;amplitude=x->exp(-x^2),name=:expansion);
        window=2.,maxevals=30,rtol=1e-12,atol=1e-13)
    report=transform_values(expanded,[0.];window_refinements=3)
    @test length(report.window_refinement_quadrature_estimates)==3
    @test :unresolved in report.window_refinement_statuses
    @test report.status==:unresolved
end
