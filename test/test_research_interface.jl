using Test, QuantumFurnace, LinearAlgebra
BLAS.set_num_threads(1)

@testset "Strict state checks above six qubits" begin
    d = 128
    source = Matrix(Diagonal(vcat(ones(d ÷ 2), -ones(d ÷ 2))))
    workspace = Workspace(zeros(d, d); beta_phys=0.8, jumps=[source])
    equilibrium = Matrix{ComplexF64}(I, d, d) / d
    for method in (:krylov, :predictor)
        result = simulate_gibbs(workspace; times=[0.0], rho0=equilibrium,
            diagnostics=:strict, method, gap_options=(; max_matvecs=0))
        @test result.convergence.status == :already_within_threshold
        @test result.trajectory.raw_checks[1].positivity.status == :pass
        @test result.trajectory.rho_final ≈ equilibrium atol=1e-12
    end
end

@testset "DLL research facade" begin
    H = Hermitian(.3X+.4Y+.7Z)
    times = [0.,.2,1.,4.]
    w = Workspace(H;beta_phys=.8)
    @test w isa Workspace{KrylovSpectrum}
    @test w.research_provenance.beta_phys ≈ .8
    r = simulate_gibbs(w;times,diagnostics=:strict,save_states=true)
    @test r isa GibbsSimulationResult
    @test r.provenance.basis == :computational
    @test r.spectrum.basis == :eigen
    @test r.spectrum.reliability == :pass
    @test r.trajectory.trace_norms ≈ 2r.trajectory.distances
    @test r.trajectory.channel_steps === nothing
    @test r.convergence.status == :not_reached_by_horizon
    @test !isempty(sprint(show,r))
    p = prepare_gibbs_inputs(H;beta_phys=.8)
    L = construct_lindbladian(p.jumps,p.config,p.hamiltonian)
    U = p.hamiltonian.eigvecs
    initial = fill(ComplexF64(.5),2,2)
    for (i,t) in enumerate(times)
        reference = U*reshape(exp(t*L)*vec(U'*initial*U),2,2)*U'
        @test r.trajectory.states[i] ≈ reference atol=1e-10
        @test r.trajectory.raw_checks[i].trace.status == :pass
        @test r.trajectory.raw_checks[i].positivity.status == :pass
        @test iszero(r.trajectory.repair_norms[i])
    end
    @test r.trajectory.rho_final ≈ r.trajectory.states[end]
    H3 = pauli_hamiltonian(3,[-1. => (1=>:Z,2=>:Z),-1. => (2=>:Z,3=>:Z),
        -.7 => (1=>:X,),-.7 => (2=>:X,),-.7 => (3=>:X,)])
    example = simulate_gibbs(H3;beta_phys=.8,times=range(0,20;length=101))
    @test length(example.trajectory.t) == 101
    @test example.trajectory.all_converged
    @test example.convergence.status in (:reached_threshold,:not_reached_by_horizon)
    @test example.spectrum.coverage == :not_established
    @test r.provenance.coherent
    @test r.provenance.resources.permitted
    # Same compiled buffers are reused and physical controls are retained.
    original = copy(w.G_left)
    eigen = simulate_gibbs(H;beta_phys=.8,basis=:eigen,times,save_states=true,diagnostics=:quick)
    @test U*eigen.trajectory.rho_final*U' ≈ r.trajectory.rho_final atol=1e-10
    @test w.G_left ≈ original
    # The legacy spectral predictor evaluates arbitrary absolute sample times;
    # only the new facade requires the time origin to be included.
    late = predict_lindbladian_trajectory(p.config,p.hamiltonian,p.jumps,
        Matrix(U'*initial*U),[.2,1.,4.];krylovdim=4)
    @test late.rho_final ≈ U'*r.trajectory.rho_final*U atol=1e-10
    clock = GeneratorClock{Float64}(:test_clock,3.,1.)
    faster = simulate_gibbs(H;beta_phys=.8,times=times/3,clock,diagnostics=:quick)
    @test faster.trajectory.rho_final ≈ r.trajectory.rho_final atol=1e-10
    @test faster.spectrum.clock == :test_clock
    @test faster.spectrum.spectral_gap ≈ 3r.spectrum.spectral_gap rtol=1e-9
    @test isempty(faster.trajectory.states)

    # dry_run stops before Hamiltonian spectral preparation and filter factories
    # are invoked exactly once per call, not again during compilation.
    calls = Ref(0)
    factory = beta -> (calls[] += 1; DLLGaussianFilter(beta))
    dry = simulate_gibbs(H;beta_phys=.8,times,dry_run=true,filter=factory)
    @test dry.spectral_preparation == :not_run
    @test dry.beta_alg === nothing
    @test calls[] == 1
    @test dry.trajectory.permitted
    calls[] = 0
    simulate_gibbs(H;beta_phys=.8,times=[0.],filter=factory,diagnostics=:quick)
    @test calls[] == 1
    @test !simulate_gibbs(H;beta_phys=.8,times,dry_run=true,max_bytes=1).permitted
    @test_throws ArgumentError Workspace(H;beta_phys=.8,max_bytes=1)
    @test_throws ArgumentError simulate_gibbs(H;beta_phys=.8,times,save_states=true,max_saved_bytes=1)
    for grid in (Float64[],[1.,2.],[0.,2.,1.],[0.,0.],[0.,Inf])
        @test_throws ArgumentError simulate_gibbs(H;beta_phys=.8,times=grid)
        @test_throws ArgumentError simulate_gibbs(w;times=grid)
    end
    @test_throws ArgumentError simulate_gibbs(H;beta_phys=.8,times,filter=DLLGaussianFilter(.4),dry_run=true)
    @test_throws ArgumentError simulate_gibbs(H;beta_phys=.8,times,transition_weight=x->1)
    @test simulate_gibbs(H;beta_phys=.8,times,construction=KMS(),dry_run=true).construction==:CKG_KMS
    @test_throws ArgumentError simulate_gibbs(H;beta_phys=.8,times,construction=GNS())
    @test_throws ArgumentError simulate_gibbs(w;times,rho0=ComplexF64[1.2 0;0 -.2])
    @test_throws ArgumentError simulate_gibbs(w;times,rho0=ones(3,3))
    @test_throws ArgumentError simulate_gibbs(w;times,max_time=1.)

    gibbs = Matrix(U*p.hamiltonian.gibbs*U')
    already = simulate_gibbs(w;times,rho0=gibbs,diagnostics=:quick)
    @test already.convergence.status == :already_within_threshold
    @test already.convergence.threshold_time ≈ 0.
    extended = simulate_gibbs(w;times=[0.,.5],max_extensions=6,max_time=16.,diagnostics=:quick)
    @test extended.convergence.status == :reached_threshold
    @test extended.convergence.initial_state_specific
    @test extended.convergence.worst_case_mixing == :not_established
    @test extended.convergence.extensions <= 6
    @test last(extended.trajectory.t) <= 16.
    @test extended.convergence.numerical_floor === nothing
    nonunique = simulate_gibbs(Z;beta_phys=.8,jumps=[Z],times=[0.,5.],diagnostics=:strict)
    @test nonunique.diagnostics.uniqueness == :nonunique
    @test nonunique.convergence.status == :not_reached_by_horizon
    @test nonunique.trajectory.distances[end] > .1
    # Gap work can fail without losing the propagated result.
    failedgap = simulate_gibbs(w;times,gap_options=(;max_matvecs=0))
    @test failedgap.spectrum.resources.exhausted
    @test failedgap.trajectory.rho_final ≈ r.trajectory.rho_final atol=1e-10
    @test failedgap.diagnostics.checks.stationarity.status == :not_run
    for options in ((;max_matvecs=0),(;max_seconds=0.))
        partial = simulate_gibbs(w;times,diagnostics=:quick,options...)
        @test length(partial.trajectory.t) == 1
        @test partial.trajectory.rho_final ≈ initial atol=1e-12
        @test partial.convergence.status == :inconclusive
        @test !partial.trajectory.all_converged
    end

    for model in (H,Z,zeros(2,2))
        pred = simulate_gibbs(model;beta_phys=.8,times,method=:predictor,diagnostics=:quick,save_states=true)
        direct = simulate_gibbs(model;beta_phys=.8,times,diagnostics=:quick,save_states=true)
        @test pred.trajectory.predictor_check.status == :pass
        @test pred.trajectory.rho_final ≈ direct.trajectory.rho_final atol=1e-10
        @test pred.trajectory.distances ≈ direct.trajectory.distances atol=1e-10
        @test all(iszero,pred.trajectory.repair_norms)
    end
    # Nonunique stationary population is retained by raw all-mode reconstruction.
    pn = simulate_gibbs(Z;beta_phys=.8,jumps=[Z],times,method=:predictor,diagnostics=:strict)
    @test pn.trajectory.predictor_check.status == :pass
    @test pn.convergence.status == :not_reached_by_horizon
    stationary_pred = simulate_gibbs(w;times,rho0=gibbs,method=:predictor,diagnostics=:quick)
    @test stationary_pred.convergence.status == :already_within_threshold
    @test stationary_pred.trajectory.predictor_check.status == :pass
    slowclock = GeneratorClock{Float64}(:slow_clock,1e-12,1.)
    slow = simulate_gibbs(H;beta_phys=.8,times=times*1e12,clock=slowclock,method=:predictor,diagnostics=:quick)
    @test slow.trajectory.rho_final ≈ r.trajectory.rho_final atol=1e-9
    @test slow.trajectory.predictor_check.status == :pass
    slow_direct = simulate_gibbs(H;beta_phys=.8,times=times*1e12,clock=slowclock,diagnostics=:quick)
    @test slow_direct.trajectory.rho_final ≈ r.trajectory.rho_final atol=1e-9
    @test eigenmode_mixing_time(slow.trajectory,.01;t_upper=4e12).mixing_time ≈
        eigenmode_mixing_time(simulate_gibbs(w;times,method=:predictor,diagnostics=:quick).trajectory,.01;t_upper=4.).mixing_time*1e12 rtol=1e-3
    spot_limited = simulate_gibbs(w;times,method=:predictor,max_matvecs=4,diagnostics=:quick)
    @test spot_limited.trajectory.failure.reason == :matvecs
    @test spot_limited.convergence.status == :inconclusive

    # Time-domain evolution reuses the existing DLL Time workspace; explicit
    # grid error is diagnosed rather than renamed an exact Bohr calculation.
    time_result = simulate_gibbs(H;beta_phys=.8,times=[0.,.1],domain=TimeDomain(),
        time_step=.8,num_energy_bits=3,diagnostics=:strict)
    @test time_result.diagnostics.checks.stationarity.status == :fail
    @test_throws ArgumentError simulate_gibbs(H;beta_phys=.8,times,domain=TimeDomain(),
        time_step=.8,num_energy_bits=3,method=:predictor)
end

@testset "Raw evolution defects and explicit repairs" begin
    # Deliberately unphysical trace-growing map: retain pre-correction defects.
    action! = (out,x) -> (out .= .2x)
    initial = ComplexF64[.7 0;0 .3]
    repaired = lindblad_action_integrate(action!,initial,initial,[0.,.2];
        record_diagnostics=true,repair_states=true)
    raw = lindblad_action_integrate(action!,initial,initial,[0.,.2];
        record_diagnostics=true,repair_states=false)
    @test repaired.raw_checks[end].trace.status == :fail
    @test repaired.repair_norms[end] > .01
    @test tr(repaired.rho_final) ≈ 1
    @test tr(raw.rho_final) ≈ exp(.04)
    @test iszero(raw.repair_norms[end])
    for multiplier in (1e-20,1.,1e12)
        physical_action! = (out,x) -> (out .= multiplier*(Z*x*Z-x))
        initial_plus = fill(ComplexF64(.5),2,2)
        evolved = lindblad_action_integrate(physical_action!,initial_plus,initial_plus,
            [0.,.2,1.]/multiplier;repair_states=false)
        @test evolved.rho_final ≈ ComplexF64[.5 .5exp(-2);.5exp(-2) .5] atol=1e-11
        @test evolved.all_converged
    end
    @test_throws ArgumentError lindblad_action_integrate(action!,initial,initial,Float64[])
end
