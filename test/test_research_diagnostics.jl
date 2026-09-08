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

@testset "Kernel-aware spectrum evidence" begin
    for multiplier in (1.,1e-14,1e-100)
        values = ComplexF64[0,0,-1,-1].*multiplier
        s = spectral_gap_diagnostics(values,zeros(4);operator_scale=multiplier,complete=true)
        @test s.detected_zero_modes == 2
        @test s.uniqueness == :nonunique
        @test s.spectral_gap ≈ multiplier rtol=1e-12
        @test s.gap_index == 3
        parent = Matrix{ComplexF64}(Diagonal([0.,0.,1.,1.].*multiplier))
        r = kms_parent_spectrum(parent,Matrix{ComplexF64}(I,2,2)/2)
        @test r.kernel_count == 2
        @test r.first_positive_eigenvalue ≈ multiplier rtol=1e-12
        # Deliberately non-Hermitian parent remains invalid in every clock.
        parent[1,2] = .1multiplier
        @test kms_parent_spectrum(parent,Matrix{ComplexF64}(I,2,2)/2).hermiticity_defect > .01
    end
    low = spectral_gap_diagnostics(ComplexF64[0,-1e-12,-1,-2],zeros(4);
        operator_scale=2.,complete=true)
    @test low.detected_zero_modes == 1
    @test low.spectral_gap ≈ 1e-12 rtol=1e-12
    uncertain = spectral_gap_diagnostics(ComplexF64[0,-1e-12,-1], [0.,1e-11,0.];operator_scale=1.)
    @test uncertain.status == :inconclusive
    @test uncertain.zero_classification == :compatible_with_zero
    for values in (ComplexF64[0,.1,-1], ComplexF64[0,.1im,-1])
        s = spectral_gap_diagnostics(values,zeros(3);operator_scale=1.,complete=true)
        @test isnan(s.spectral_gap)
        @test s.status == (real(values[2]) > 0 ? :fail : :inconclusive)
    end
    zero = spectral_gap_diagnostics(zeros(ComplexF64,4),zeros(4);operator_scale=0.,complete=true)
    @test zero.detected_zero_modes == 4
    @test zero.uniqueness == :nonunique
    @test isnan(zero.spectral_gap)
    @test spectral_gap_diagnostics(ComplexF64[],Float64[];operator_scale=0.).status == :inconclusive
    for values in (ComplexF64[1,1,.8,.8],ComplexF64[1,-1,.8],ComplexF64[1,1.1,.8])
        s = spectral_gap_diagnostics(values,zeros(length(values));operator_scale=1.1,channel_delta=.2,complete=true)
        if values[2] == 1
            @test s.detected_zero_modes == 2
            @test s.spectral_gap ≈ -log(.8)/.2
        else
            @test isnan(s.spectral_gap)
        end
    end
    @test_throws ArgumentError spectral_gap_diagnostics([0.],[NaN];operator_scale=1.)
    @test_throws ArgumentError spectral_gap_diagnostics([0.],[0.];operator_scale=-1.)
    # Raw fixed point checks retain non-Hermiticity and negative eigenvalues.
    for A in (ComplexF64[1 1;0 0],ComplexF64[1.2 0;0 -.2])
        original = copy(A)
        f = QuantumFurnace._spectral_fixed_point([vec(A)],[1],2)
        @test !f.valid
        @test f.state ≈ A
        @test A ≈ original
        @test iszero(f.repair_norm)
    end
    @test !QuantumFurnace._spectral_fixed_point([vec(ComplexF64[1 0;0 -1])],[1],2).valid
    A = ComplexF64[.7 0;0 .3]
    f = QuantumFurnace._spectral_fixed_point([vec(3im*A)],[1],2)
    @test f.valid
    @test f.state ≈ A
    # Complete independent dephasing spectrum diag(0,-2,-2,0).
    L = Matrix{ComplexF64}(Diagonal([0.,-2.,-2.,0.]))
    @test extract_leading_eigendata(L;n_modes=1).spectral_gap ≈ 2.
    L[2,2] = .2
    @test isnan(extract_leading_eigendata(L).spectral_gap)
end

@testset "Compiled kernel extraction and paired commutant" begin
    for sources in ([Z],[zeros(2,2)])
        p = prepare_gibbs_inputs(Hermitian(Z);beta_phys=.8,jumps=sources)
        r = krylov_spectral_gap(p.config,p.hamiltonian,p.jumps;krylovdim=5,howmany=3)
        @test r.spectrum_diagnostics.status == :inconclusive
        @test r.spectrum_diagnostics.uniqueness == :not_established
        @test r.spectrum_diagnostics.detected_zero_modes >= 1
        @test r.fixed_point_diagnostics.valid
        @test all(norm(r.raw_eigenvectors[i]) ≈ 1 for i in eachindex(r.raw_eigenvectors))
        if iszero(sources[1])
            @test isnan(r.spectral_gap)
            @test r.gap_mode_index === nothing
        else
            @test r.spectral_gap ≈ 2.
        end
    end
    A = ComplexF64[.2im 1.3+.7im; -.8+.1im .3-.4im]
    H = Hermitian(.3X+.4Y+.7Z)
    p = prepare_gibbs_inputs(H;beta_phys=.8,jumps=[A,A'])
    herm = prepare_gibbs_inputs(H;beta_phys=.8,jumps=[(A+A')/sqrt(2),(A-A')/(im*sqrt(2))])
    pair = dense_dll_irreducibility(p.jumps,p.hamiltonian,p.config.filter)
    hpair = dense_dll_irreducibility(herm.jumps,herm.hamiltonian,herm.config.filter)
    @test pair.commutant_dimension == hpair.commutant_dimension == 1
    @test pair.is_irreducible
    @test_throws ArgumentError dense_dll_irreducibility(p.jumps[1:1],p.hamiltonian,p.config.filter)
    L = construct_lindbladian(p.jumps,p.config,p.hamiltonian)
    @test L ≈ construct_lindbladian(herm.jumps,herm.config,herm.hamiltonian) atol=1e-12
    r = krylov_spectral_gap(p.config,p.hamiltonian,p.jumps;krylovdim=5,howmany=4,tol=1e-13)
    exact = sort(-real.(eigvals(L)))
    @test r.spectral_gap ≈ exact[2] rtol=1e-10
    @test r.spectrum_diagnostics.uniqueness == :not_established
    @test r.fixed_point_diagnostics.valid
    @test r.fixed_point ≈ p.hamiltonian.gibbs atol=1e-11
    @test r.gap_mode ≈ reshape(r.raw_eigenvectors[r.gap_mode_index],2,2)
end

@testset "Strict pair witness validation" begin
    ham = HamHam(Hermitian(Z);beta_phys=.8)
    A = 1e-13*ComplexF64[0 1;0 0]
    jumps = JumpOp[JumpOp(A,ham),JumpOp(A,ham)]
    filter = DLLGaussianFilter(beta_alg(ham,.8))
    @test_throws ArgumentError dense_dll_irreducibility(jumps,ham,filter)
end
