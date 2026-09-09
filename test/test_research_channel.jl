using Test, QuantumFurnace, LinearAlgebra, Random

@testset "Finite channel research facade" begin
    H=Hermitian(.3X+.4Y+.7Z)
    beta=.8; delta=.01; steps=7
    plus=fill(ComplexF64(.5),2,2)
    for domain in (BohrDomain(),EnergyDomain())
        grid=domain isa EnergyDomain ? (;num_energy_bits=5,energy_step=.4) : (;)
        p=prepare_gibbs_inputs(H;beta_phys=beta,construction=KMS(),domain,grid...)
        cfg=Config(;merge((;(k=>getfield(p.config,k) for k in fieldnames(typeof(p.config)))...),
            (;sim=Thermalize(),delta,mixing_time=steps*delta))...)
        r=simulate_gibbs(H;sim=Thermalize(),construction=KMS(),beta_phys=beta,domain,
            steps,delta,save_states=true,channel_options=(;save_every=3),grid...)
        U=p.hamiltonian.eigvecs
        direct=run_thermalize(p.jumps,cfg,p.hamiltonian;initial_dm=Matrix(U'*plus*U),
            num_steps=steps,record_steps=[0,3,6,7],save_states=true,hermitize=false,convergence_cutoff=0)
        @test r.trajectory.rho_final ≈ U*direct.final_dm*U' atol=1e-13
        @test r.trajectory.distances ≈ direct.trace_distances atol=1e-13
        @test r.trajectory.channel_steps == [0,3,6,7]
        @test r.trajectory.subchannel_applications == 3steps
        @test r.trajectory.states[end] ≈ r.trajectory.rho_final atol=1e-14
        @test r.spectrum.reliability == :not_run
        @test r.diagnostics.checks.kms.status == :not_run
        @test r.provenance.evolution == :channel
        @test !r.provenance.hermitized
        @test !r.provenance.trace_normalized
        @test r.provenance.beta_phys ≈ beta
        @test r.trajectory.trace_norms ≈ 2r.trajectory.distances
        @test !isempty(sprint(show,r))
        eigen=simulate_gibbs(p.jumps,cfg,p.hamiltonian;steps,basis=:eigen,rho0=Matrix(U'*plus*U))
        @test eigen.trajectory.rho_final ≈ direct.final_dm atol=1e-13
        @test_throws ArgumentError simulate_gibbs(p.jumps,cfg,p.hamiltonian;steps,rho0=2plus)
        random=simulate_gibbs(p.jumps,cfg,p.hamiltonian;steps,channel_options=(;jump_selection=:random,seed=18))
        reference=run_thermalize(p.jumps,Config(;merge((;(k=>getfield(cfg,k) for k in fieldnames(typeof(cfg)))...),
            (;jump_selection=:random))...),p.hamiltonian;initial_dm=Matrix(U'*plus*U),
            num_steps=steps,hermitize=false,convergence_cutoff=0,rng=MersenneTwister(18))
        @test random.trajectory.rho_final ≈ U*reference.final_dm*U' atol=1e-13
        @test random.provenance.trajectory_kind == :source_conditioned_density_matrices
        @test random.trajectory.subchannel_applications == steps
        # Evidence roundtrip must not silently resume as Lindblad evolution.
        mktempdir() do dir
            loaded=load_result(save_result(r,joinpath(dir,"channel.bson")))
            @test loaded.trajectory.rho_final ≈ r.trajectory.rho_final atol=1e-14
            @test loaded.provenance.evolution == :channel
            @test occursin("Channel evidence only",read(joinpath(dir,"channel.txt"),String))
            @test !occursin("Workspace(result)",read(joinpath(dir,"channel.txt"),String))
            @test_throws ArgumentError Workspace(loaded)
            @test_throws ArgumentError simulate_gibbs(loaded;times=[0.,delta])
        end
    end
    options=(;sim=Thermalize(),construction=KMS(),beta_phys=beta,delta)
    r=simulate_gibbs(H;options...,steps,channel_options=(;save_every=5,max_steps=3),save_states=true)
    @test r.trajectory.channel_steps == [0,3]
    @test r.trajectory.completed_steps == 3
    @test r.trajectory.failure.reason == :steps
    @test r.convergence.status == :inconclusive
    @test r.trajectory.rho_final ≈ r.trajectory.states[end] atol=1e-14
    timed=simulate_gibbs(H;options...,steps=3,max_seconds=0)
    @test timed.trajectory.channel_steps == [0]
    @test timed.trajectory.failure.reason == :time
    zero=simulate_gibbs(H;options...,steps=0)
    @test zero.trajectory.rho_final ≈ plus atol=1e-14
    scheduled=simulate_gibbs(H;options...,times=[0.,.03,.07])
    @test scheduled.trajectory.channel_steps == [0,3,7]
    @test_throws ArgumentError simulate_gibbs(H;options...,times=[0.,.031])
    @test_throws ArgumentError simulate_gibbs(H;options...,times=[0.,.03],steps=4)
    @test_throws ArgumentError simulate_gibbs(H;options...,steps,method=:predictor)
    @test_throws ArgumentError simulate_gibbs(H;options...,steps,gap_options=(;seed=3))
    @test_throws ArgumentError simulate_gibbs(H;options...,steps,channel_options=(;unknown=true))
    @test_throws ArgumentError simulate_gibbs(H;sim=Thermalize(),beta_phys=beta,steps,delta)
    @test_throws ArgumentError simulate_gibbs(H;beta_phys=beta,steps,delta,times=[0.])
    @test_throws ArgumentError simulate_gibbs(H;options...,steps=-1)
    @test_throws ArgumentError simulate_gibbs(H;options...,steps,save_states=true,max_saved_bytes=1)
    dry=simulate_gibbs(H;options...,steps,dry_run=true)
    @test dry.spectral_preparation == :not_run
    @test dry.evolution == :channel
    @test dry.permitted
    @test !simulate_gibbs(H;options...,steps,dry_run=true,max_saved_bytes=1,save_states=true).permitted
    @test !simulate_gibbs(H;options...,domain=EnergyDomain(),num_energy_bits=16,
        energy_step=.4,steps=0,max_bytes=12_000_000,dry_run=true).permitted
    # Clock is already in sources; facade must agree with explicitly prepared backend.
    clock=GeneratorClock{Float64}(:slow,.5,1.)
    slow=simulate_gibbs(H;options...,steps=2,clock)
    @test slow.provenance.clock_label == :slow
    p=prepare_gibbs_inputs(H;beta_phys=beta,construction=KMS(),clock)
    cfg=Config(;merge((;(k=>getfield(p.config,k) for k in fieldnames(typeof(p.config)))...),
        (;sim=Thermalize(),delta,mixing_time=.02))...)
    manual=simulate_gibbs(p.jumps,cfg,p.hamiltonian;steps=2)
    @test slow.trajectory.rho_final ≈ manual.trajectory.rho_final atol=1e-13
    @test_throws ArgumentError QuantumFurnace._channel_schedule([0.,100000000.25],nothing,1.,1,10^9)
    @test_throws ArgumentError simulate_gibbs(H;options...,steps=1,rates=1e6)
    # An already-Gibbs start still executes the explicitly requested horizon.
    p=prepare_gibbs_inputs(H;beta_phys=beta,construction=KMS())
    initial=p.hamiltonian.eigvecs*p.hamiltonian.gibbs*p.hamiltonian.eigvecs'
    ready=simulate_gibbs(H;options...,steps=3,rho0=initial)
    @test ready.trajectory.completed_steps == 3
    @test ready.convergence.status == :already_within_threshold
end

@testset "Legacy Trotter channel basis" begin
    cfg=make_config(Thermalize(),TrotterDomain();num_qubits=3,mixing_time=.01)
    cfg=Config(;merge((;(k=>getfield(cfg,k) for k in fieldnames(typeof(cfg)))...),
        (;num_energy_bits=5,t0=2pi/(2^5*cfg.w0)))...)
    trotter=make_trotter_for_config(N3_HAM,cfg)
    U=trotter.eigvecs
    jumps=JumpOp[JumpOp(j.data,Matrix(U'*j.data*U),j.orthogonal,j.hermitian) for j in N3_JUMPS]
    plus=fill(ComplexF64(1/8),8,8)
    r=simulate_gibbs(jumps,cfg,N3_HAM,trotter;steps=1,save_states=true)
    direct=run_thermalize(jumps,cfg,N3_HAM,trotter;initial_dm=Matrix(U'*plus*U),
        num_steps=1,hermitize=false,convergence_cutoff=0)
    @test r.trajectory.rho_final ≈ U*direct.final_dm*U' atol=1e-12
    eigen=simulate_gibbs(jumps,cfg,N3_HAM,trotter;steps=1,basis=:eigen,rho0=Matrix(U'*plus*U))
    @test eigen.trajectory.rho_final ≈ direct.final_dm atol=1e-12
    @test_throws ArgumentError simulate_gibbs(jumps,cfg,N3_HAM;steps=1)
end

@testset "Legacy Time and config horizon" begin
    cfg=make_config(Thermalize(),TimeDomain();num_qubits=3,mixing_time=.03)
    cfg=Config(;merge((;(k=>getfield(cfg,k) for k in fieldnames(typeof(cfg)))...),
        (;num_energy_bits=5,t0=2pi/(2^5*cfg.w0)))...)
    plus=fill(ComplexF64(1/8),8,8)
    r=simulate_gibbs(N3_JUMPS,cfg,N3_HAM;channel_options=(;save_every=2))
    direct=run_thermalize(N3_JUMPS,cfg,N3_HAM;initial_dm=Matrix(N3_HAM.eigvecs'*plus*N3_HAM.eigvecs),
        num_steps=3,hermitize=false,convergence_cutoff=0)
    @test r.trajectory.channel_steps == [0,2,3]
    @test r.trajectory.rho_final ≈ N3_HAM.eigvecs*direct.final_dm*N3_HAM.eigvecs' atol=1e-12
end

@testset "Legacy channel register resource gate" begin
    small=make_config(Thermalize(),EnergyDomain();num_qubits=3,mixing_time=.01)
    huge=Config(;merge((;(k=>getfield(small,k) for k in fieldnames(typeof(small)))...),
        (;num_energy_bits_D=30))...)
    dry=simulate_gibbs(N3_JUMPS,huge,N3_HAM;steps=1,dry_run=true)
    @test !dry.permitted
    @test dry.trajectory.estimated_bytes > 2^30
    @test_throws ArgumentError simulate_gibbs(N3_JUMPS,huge,N3_HAM;steps=1)
end

@testset "Float32 integer steps avoid a time-grid round trip" begin
    r=simulate_gibbs(ComplexF32.(.3X+.4Y+.7Z);sim=Thermalize(),construction=KMS(),
        beta_phys=.8f0,delta=.01f0,steps=1602,channel_options=(;max_steps=0,save_every=1602))
    H32=ComplexF32.(.3X+.4Y+.7Z)
    single=simulate_gibbs(H32;sim=Thermalize(),construction=KMS(),beta_phys=.8f0,delta=.01f0,steps=2)
    double=simulate_gibbs(ComplexF64.(H32);sim=Thermalize(),construction=KMS(),
        beta_phys=Float64(.8f0),delta=Float64(.01f0),steps=2)
    @test single.trajectory.rho_final ≈ double.trajectory.rho_final atol=5e-6
    @test single.trajectory.completed_steps == 2
    @test r.provenance.requested_steps == 1602
    @test r.trajectory.completed_steps == 0
    @test r.trajectory.failure.reason == :steps
end

@testset "Float32 pairing has no absolute source floor" begin
    A=ComplexF32[0 1f-7; 0 0]
    wrong=JumpOp[JumpOp(A,copy(A),false,true)]
    @test_throws ArgumentError validate_jump_pairing(wrong;atol=0,rtol=100eps(Float32))
    mismatched=JumpOp[JumpOp(A,copy(A),false,false),JumpOp(2Matrix(A'),2Matrix(A'),false,false)]
    @test_throws ArgumentError validate_jump_pairing(mismatched;atol=0,rtol=100eps(Float32))
    matched=JumpOp[JumpOp(A,copy(A),false,false),JumpOp(Matrix(A'),Matrix(A'),false,false)]
    @test validate_jump_pairing(matched;atol=0,rtol=100eps(Float32)) === nothing
end
