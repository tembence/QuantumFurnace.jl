using Test, QuantumFurnace, LinearAlgebra
BLAS.set_num_threads(1)

@testset "Physical DLL preparation" begin
    beta = 0.8
    H = Hermitian(.3X + .4Y + .7Z)
    ham = HamHam(H; beta_phys=beta)
    A = ComplexF64[.2im 1.3+.7im; -.8+.1im .3-.4im]
    filters = (DLLGaussianFilter(beta), DLLMetropolisFilter(beta; S=1.9),
        ShiftedSymmetricFilter(DLLMetropolisFilter(beta; S=1.9), .2, .7),
        DLLMultiChannelFilter([DLLGaussianFilter(beta), DLLMetropolisFilter(beta; S=1.9)],beta))
    for f in filters
        p = prepare_gibbs_inputs(ham; beta_phys=beta, filter=f, jumps=[A,A'])
        cfg, h = p.config, p.hamiltonian
        @test cfg.beta ≈ beta*ham.rescaling_factor
        @test p.provenance.generator_multiplier ≈ 1
        @test p.provenance.clock_label == :raw_generator
        @test p.provenance.derived_clock === nothing
        @test h.gibbs ≈ ham.gibbs
        @test h.eigvals ≈ ham.eigvals
        @test h.eigvals !== ham.eigvals
        @test h.eigvecs !== ham.eigvecs
        for nu in (-.6,-.13,0.,.07,.6)
            @test freq_kernel(cfg.filter,nu) ≈ freq_kernel(f,nu*ham.rescaling_factor) atol=2e-14
        end
        for t in (-1.,0.,.7)
            @test time_kernel(cfg.filter,t) ≈ time_kernel(f,t/ham.rescaling_factor)/ham.rescaling_factor atol=2e-10
        end
        # Independent physical-coordinate assembly, including a generally nonzero B.
        E = eigvals(H); nu = E .- E'
        fs = f isa DLLMultiChannelFilter ? f.channels : [f]
        ls = [freq_kernel.(Ref(fc),nu) .* (ham.eigvecs' * src * ham.eigvecs) for fc in fs for src in (A,A')]
        loss = sum(l' * l for l in ls)
        B = im/2 .* tanh.(beta*nu/4) .* loss
        id = Matrix{ComplexF64}(I,2,2)
        exact = sum(kron(conj(l),l) for l in ls) - (kron(id,loss)+kron(transpose(loss),id))/2 - im*(kron(id,B)-kron(transpose(B),id))
        actual = construct_lindbladian(p.jumps,cfg,h)
        @test actual ≈ exact atol=2e-13
        @test norm(actual*vec(Matrix(h.gibbs))) < 2e-13
        @test dll_coherent_op_bohr(p.jumps,h,cfg.filter,cfg.beta) ≈ B atol=2e-13
        clock = GeneratorClock{Float64}(:declared_clock,2.7,3.4)
        pc = prepare_gibbs_inputs(ham; beta_phys=beta,filter=f,jumps=[A,A'],clock)
        Lc = construct_lindbladian(pc.jumps,pc.config,pc.hamiltonian)
        @test Lc ≈ clock.multiplier*actual atol=2e-13
        @test pc.provenance.time_multiplier ≈ 1/2.7
        @test pc.provenance.clock_rate_evidence == :user_declared
        @test exp(.3/2.7*Lc) ≈ exp(.3*actual) atol=2e-13
        if f isa DLLMetropolisFilter
            @test cfg.filter.S ≈ f.S/ham.rescaling_factor
            @test p.provenance.physical_filters[1].support_radius ≈ f.S
            @test p.provenance.algorithm_filters[1].support_radius ≈ f.S/ham.rescaling_factor
        end
    end
    @test_throws ArgumentError prepare_gibbs_inputs(H)
    @test_throws ArgumentError prepare_gibbs_inputs(H; beta_phys=beta,temperature=1/beta)
    for temp in (0.,Inf,-1.,NaN)
        @test_throws ArgumentError prepare_gibbs_inputs(H;temperature=temp)
    end
    thermal = prepare_gibbs_inputs(H;temperature=1/beta)
    @test thermal.config.beta_phys ≈ beta
    @test thermal.provenance.temperature_input == :temperature
    @test thermal.provenance.temperature_unit == :energy_kB_one
    @test thermal.provenance.input_frame == :physical
    @test prepare_gibbs_inputs(ham;beta_phys=beta).provenance.input_frame == :algorithm
    @test prepare_gibbs_inputs(ham;beta_phys=beta).provenance.filter_input_frame == :physical
    @test_throws ArgumentError prepare_gibbs_inputs(H;beta_phys=beta,filter=42)
    input_channels = [DLLGaussianFilter(beta)]
    mutable_filter = DLLMultiChannelFilter(input_channels,beta)
    snapshot = prepare_gibbs_inputs(ham;beta_phys=beta,filter=mutable_filter)
    push!(input_channels,DLLGaussianFilter(beta))
    @test length(snapshot.provenance.physical_filter.channels) == 1
    @test length(snapshot.config.filter.channels) == 1
    default = prepare_gibbs_inputs(H; beta_phys=beta)
    @test default.config.filter isa DLLGaussianFilter
    @test default.config.num_energy_bits_D === nothing
    @test default.config.a === nothing
    @test default.config.eta === nothing
    @test default.provenance.sources.proposal_amplitude ≈ inv(sqrt(3))
    for temperature in (.1,.8,2.4)
        next = HamHam(ham; beta_phys=temperature)
        expected = exp(-temperature*Matrix(H)); expected ./= tr(expected)
        @test next.eigvecs*next.gibbs*next.eigvecs' ≈ expected atol=2e-14
        @test prepare_gibbs_inputs(next; beta_phys=temperature).config.beta ≈ beta_alg(next,temperature)
        @test ham.gibbs ≈ default.hamiltonian.gibbs
    end
    @test_throws ArgumentError prepare_gibbs_inputs(ham; beta_phys=1.2)
    @test_throws ArgumentError prepare_gibbs_inputs(ham; beta_phys=beta,filter=DLLGaussianFilter(1.2))
    @test_throws ArgumentError prepare_gibbs_inputs(ham; beta_phys=beta,filter=DLLGaussianFilter(beta_alg(ham,beta)))
    for b in (0.,Inf,-1.,NaN)
        @test_throws ArgumentError prepare_gibbs_inputs(H; beta_phys=b)
    end
    for domain in (EnergyDomain(),TrotterDomain())
        @test_throws ArgumentError prepare_gibbs_inputs(H; beta_phys=beta,domain)
    end
    @test_throws ArgumentError prepare_gibbs_inputs(H; beta_phys=beta,construction=KMS())
    @test_throws ArgumentError prepare_gibbs_inputs(H; beta_phys=beta,filter=GaussianFilter(1.))
    @test_throws ArgumentError prepare_gibbs_inputs(H; beta_phys=beta,domain=TimeDomain())
    @test_throws ArgumentError prepare_gibbs_inputs(H; beta_phys=beta,time_step=.1,num_energy_bits=8)
    @test_throws ArgumentError prepare_gibbs_inputs(H; beta_phys=beta,filter=ShiftedSymmetricFilter(DLLGaussianFilter(beta),-.2,1.))
    bad = deepcopy(ham); parent(bad.gibbs)[1,2] = .1
    @test_throws ArgumentError prepare_gibbs_inputs(bad; beta_phys=beta)
    @test_throws ArgumentError validate_config!(default.config,bad)
    @test HamHam(bad; beta_phys=beta).gibbs ≈ ham.gibbs
    bad = deepcopy(ham); bad.eigvecs[1,1] += .1
    @test_throws ArgumentError HamHam(bad; beta_phys=beta)
    bad = deepcopy(ham); bad.bohr_freqs[1,2] += .1
    @test_throws ArgumentError prepare_gibbs_inputs(bad; beta_phys=beta)
    shifted = prepare_gibbs_inputs(H + 100I; beta_phys=beta)
    @test shifted.hamiltonian.rescaling_factor ≈ ham.rescaling_factor atol=1e-12
    @test construct_lindbladian(shifted.jumps,shifted.config,shifted.hamiltonian) ≈ construct_lindbladian(default.jumps,default.config,default.hamiltonian) atol=1e-12
    zero = prepare_gibbs_inputs(zeros(2,2);beta_phys=beta,jumps=[zeros(2,2)],clock=GeneratorClock{Float64}(:declared,2.,1.))
    @test norm(construct_lindbladian(zero.jumps,zero.config,zero.hamiltonian)) < 1e-14
    @test zero.provenance.clock_rate_evidence == :user_declared
    cold = prepare_gibbs_inputs(Diagonal([0.,1.]);beta_phys=1000.)
    @test cold.provenance.gibbs_underflow
    @test !default.provenance.gibbs_underflow
    for T in (Float32,Float64)
        p = prepare_gibbs_inputs(Hermitian(T(.3)*Complex{T}.(X)+T(.7)*Complex{T}.(Z)); beta_phys=T(beta))
        @test p.config.beta isa T
        supplied = prepare_gibbs_inputs(Hermitian(T(.3)*Complex{T}.(X)+T(.7)*Complex{T}.(Z));
            beta_phys=beta,filter=DLLGaussianFilter(beta))
        @test supplied.config.filter.beta ≈ supplied.config.beta
        @test validate_config!(supplied.config,supplied.hamiltonian) === nothing
        @test validate_config!(p.config,p.hamiltonian; atol=100eps(T),rtol=100eps(T)) === nothing
    end
    raw = build_heis_1d(2,[1.,1.,1.];seed=42)
    @test prepare_gibbs_inputs(raw;beta_phys=beta).config.num_qubits == 2
    # Time registers are physical evolution coordinates, independent of generator clock.
    pt = prepare_gibbs_inputs(ham;beta_phys=beta,domain=TimeDomain(),time_step=.2,num_energy_bits=8)
    @test pt.config.t0_D ≈ .2ham.rescaling_factor
    @test pt.config.eta === nothing
    @test pt.provenance.physical_time_step ≈ .2
    # Matched finite quadrature in physical and algorithm coordinates.
    times = collect(-20:19)*.2
    U = ham.eigvecs; E = eigvals(H)
    source = JumpOp(A,ham)
    direct = sum(time_kernel(DLLGaussianFilter(beta),t) .* (exp(im*t*Diagonal(E)) * source.in_eigenbasis * exp(-im*t*Diagonal(E))) for t in times)*.2
    converted = dll_lindblad_op_time(source,ham,times*ham.rescaling_factor,pt.config.filter,pt.config.t0_D)
    @test converted ≈ direct atol=1e-13
    ws = Workspace(pt.config,pt.hamiltonian,pt.jumps)
    @test ws !== nothing
end
