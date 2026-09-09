# regression gates for DLL construction and matrix-free Time evolution.
using Test, QuantumFurnace, LinearAlgebra, Random

@testset "Research contract: DLL baseline regressions" begin
    raw = build_heis_1d(2, [0.8, 0.7, 1.2]; seed=14, periodic=false,
                        disorder_strength=0.2)
    ham = HamHam(raw; beta_phys=0.5)
    beta = beta_alg(ham, 0.5)
    filter = DLLGaussianFilter(beta)
    cfg(domain; a=beta/30) = Config(; sim=Lindbladian(), domain,
        construction=DLL(), num_qubits=2, with_linear_combination=true,
        beta, beta_phys=0.5, sigma=1/beta, a, s=0.25,
        num_energy_bits_D=8, t0_D=0.5, filter)
    makejump(A) = JumpOp(Matrix(A), ham.eigvecs' * A * ham.eigvecs,
                        false, ishermitian(A))
    A = randn(MersenneTwister(42), ComplexF64, 4, 4) / 4
    paired = JumpOp[makejump(A), makejump(A')]
    hermitian = JumpOp[makejump((A+A')/sqrt(2)),
                        makejump((A-A')/(sqrt(2)*im))]
    Lb = construct_lindbladian(paired, cfg(BohrDomain()), ham)
    Lbh = construct_lindbladian(hermitian, cfg(BohrDomain()), ham)
    Lt = construct_lindbladian(paired, cfg(TimeDomain()), ham)
    Lth = construct_lindbladian(hermitian, cfg(TimeDomain()), ham)
    @test isapprox(Lb, Lbh; atol=1e-12, rtol=0)
    # same grid and same linear source span must give the same generator.
    @test isapprox(Lt, Lth; atol=1e-12, rtol=0)
    @info "paired-source baseline" full_error=opnorm(Lt-Lb) hermitian_error=opnorm(Lth-Lbh) gibbs_residual=norm(Lt*vec(Matrix(ham.gibbs)))

    ws = Workspace(cfg(TimeDomain()), ham, hermitian)
    @test ws isa Workspace
    @test validate_config!(cfg(TimeDomain(); a=0.0), ham) === nothing

    @testset "Time forward and adjoint, separate retained channels" begin
        gaussian = DLLGaussianFilter(beta)
        metro = DLLMetropolisFilter(beta; S=2.0)
        multi = DLLMultiChannelFilter([gaussian, metro], beta)
        # Below/above the threaded matvec threshold, and one Hermitian source
        # to exercise the serial construction path even with multiple threads.
        for f in (gaussian, metro, multi), sources in (hermitian[1:1], paired, repeat(paired, 6))
            config = Config(; sim=Lindbladian(), domain=TimeDomain(),
                construction=DLL(), num_qubits=2, with_linear_combination=false,
                beta, sigma=1/beta, filter=f, num_energy_bits_D=8, t0_D=0.5)
            workspace = Workspace(config, ham, sources)
            dense = construct_lindbladian(sources, config, ham)
            nch = f isa DLLMultiChannelFilter ? 2 : 1
            @test length(workspace.dll_lindblads) == nch * length(sources)
            rng = MersenneTwister(76)
            X, Y = randn(rng, ComplexF64, 4, 4), randn(rng, ComplexF64, 4, 4)
            LY = copy(apply_lindbladian!(workspace, Y, config, ham))
            LstarX = copy(apply_adjoint_lindbladian!(workspace, X, config, ham))
            @test isapprox(vec(LY), dense * vec(Y); atol=1e-12, rtol=1e-12)
            @test isapprox(vec(LstarX), dense' * vec(X); atol=1e-12, rtol=1e-12)
            @test isapprox(dot(X, LY), dot(LstarX, Y); atol=1e-12, rtol=1e-12)
            @test norm(apply_adjoint_lindbladian!(workspace, Matrix{ComplexF64}(I, 4, 4), config, ham)) < 1e-12
            @test isapprox(apply_lindbladian!(workspace, Y, config, ham), LY; atol=1e-12, rtol=0)
        end
    end

    @testset "Construction-specific validation and capability gates" begin
        base = (; sim=Lindbladian(), construction=DLL(), num_qubits=2,
            with_linear_combination=false, beta, sigma=1/beta, filter,
            num_energy_bits_D=8, t0_D=0.5)
        # Legacy rate knobs do not change DLL or impose a fictitious rate.
        for domain in (BohrDomain(), TimeDomain()), linear in (false, true)
            config = Config(; base..., domain, with_linear_combination=linear)
            @test validate_config!(config, ham) === nothing
        end
        for domain in (EnergyDomain(), TrotterDomain())
            @test_throws ArgumentError Workspace(Config(; base..., domain), ham, paired)
        end
        @test_throws r"with_gqsp is not supported with DLL" Workspace(
            Config(; base..., domain=TimeDomain(), with_gqsp=true), ham, paired)
        @test_throws r"DLL Thermalize channels are not supported" Workspace(
            Config(; base..., domain=TimeDomain(), sim=Thermalize(), mixing_time=1.0, delta=0.1), ham, paired)
        # The same omissions must still fail for CKG/GNS rate prescriptions.
        for construction in (KMS(), GNS())
            @test_throws r"gaussian_parameters must be set" validate_config!(Config(;
                base..., domain=BohrDomain(), construction, filter=nothing))
            @test_throws r"explicit finite a and s" validate_config!(Config(;
                base..., domain=BohrDomain(), construction, filter=nothing,
                with_linear_combination=true))
        end
        @test_throws r"eta must be > 0" validate_config!(Config(;
            base..., domain=TimeDomain(), construction=KMS(), filter=nothing,
            with_linear_combination=true, a=0.0, s=0.25))
    end
end
