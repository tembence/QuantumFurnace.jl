using Test, QuantumFurnace, LinearAlgebra
BLAS.set_num_threads(1)

@testset "Research physical diagnostics" begin
    p = prepare_gibbs_inputs(Hermitian(.4X+.7Z); beta_phys=.8)
    w = Workspace(p.config,p.hamiltonian,p.jumps)
    report = workspace_diagnostics(w,p.config,p.hamiltonian;rho=Matrix(p.hamiltonian.gibbs))
    @test report.uniqueness == :established
    for name in (:stationarity,:trace_preservation,:conditioning,:kms,:kernel)
        @test getproperty(report.checks,name).status == :pass
    end
    @test report.checks.state.positivity.status == :pass
    @test report.checks.tails.status == :not_run
    @test report.parent_spectrum.kernel_count == 1
    @test !report.parent_spectrum.primitivity_established
    for options in ((;dense_max_dim=1),(;max_dense_bytes=1))
        skip = workspace_diagnostics(w,p.config,p.hamiltonian;options...)
        @test !skip.resources.dense_permitted
        @test skip.parent_spectrum === nothing
        @test skip.checks.kms.status == :not_run
        @test skip.uniqueness == :not_established
        @test skip.checks.stationarity.status == :pass
    end
    @test state_diagnostics(ComplexF64[1 1;0 0]).hermiticity.status == :fail
    @test state_diagnostics(ComplexF64[1 1;0 0]).positivity.status == :not_run
    @test state_diagnostics(diagm([1.2,-.2])).positivity.status == :fail
    @test state_diagnostics(diagm([.2,.3])).trace.status == :fail
    @test state_diagnostics(fill(NaN,2,2)).finiteness.status == :fail
    @test state_diagnostics(Matrix{Float64}(I,2,2)/2;max_dense_bytes=1).positivity.status == :not_run
    @test_throws ArgumentError workspace_diagnostics(w,p.config,p.hamiltonian;rho=ones(1,1))
    @test_throws ArgumentError workspace_diagnostics(w,p.config,p.hamiltonian;probes=0)
    @test_throws ArgumentError state_diagnostics(ones(2,2);rtol=NaN)

    for sources in ([Z],[zeros(2,2)])
        pd = prepare_gibbs_inputs(Hermitian(Z);beta_phys=.8,jumps=sources)
        wd = Workspace(pd.config,pd.hamiltonian,pd.jumps)
        rd = workspace_diagnostics(wd,pd.config,pd.hamiltonian)
        @test rd.uniqueness == :nonunique
        @test rd.parent_spectrum.kernel_count == (iszero(sources[1]) ? 4 : 2)
        @test rd.checks.stationarity.status == :pass
    end
    for beta in (30.,1000.)
        cold = prepare_gibbs_inputs(Hermitian(Z);beta_phys=beta)
        wc = Workspace(cold.config,cold.hamiltonian,cold.jumps)
        rc = workspace_diagnostics(wc,cold.config,cold.hamiltonian)
        @test rc.checks.conditioning.status == (beta==1000 ? :fail : :inconclusive)
        @test rc.checks.kms.status == :not_run
        @test rc.uniqueness == :not_established
    end
    # Corrupt compiled gain/loss balance, then scale the whole implemented map.
    # Absolute-small defects must not become passes in a slower generator clock.
    w.G_left .+= .1Matrix{ComplexF64}(I,2,2)
    for multiplier in (1.,1e-15)
        if multiplier != 1
            w.G_left .*= multiplier; w.G_right .*= multiplier
            for L in w.dll_lindblads
                L .*= sqrt(multiplier)
            end
        end
        bad = workspace_diagnostics(w,p.config,p.hamiltonian)
        @test bad.checks.stationarity.status == :fail
        @test bad.checks.trace_preservation.status == :fail
        @test bad.uniqueness == :not_established
        @test bad.checks.stationarity.quantity.scaled > .01
    end
end

@testset "Coarse DLL Time diagnostic failure" begin
    p = prepare_gibbs_inputs(Hermitian(.4X+.7Z); beta_phys=.8,
        domain=TimeDomain(),time_step=.8,num_energy_bits=3)
    r = workspace_diagnostics(Workspace(p.config,p.hamiltonian,p.jumps),p.config,p.hamiltonian)
    @test r.checks.stationarity.status == :fail
    @test r.checks.kms.status == :fail
    @test r.checks.trace_preservation.status == :pass
    @test r.uniqueness == :not_established
end
