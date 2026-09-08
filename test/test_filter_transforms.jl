using Test, QuantumFurnace, LinearAlgebra
using QuadGK

@testset "T12 prepared Fourier transforms" begin
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
