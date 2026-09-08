# Standalone T00/T01 regressions. Deliberately not in runtests.jl until T02
# repairs the two remaining @test_broken cases; then promote them to @test.
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
    # T01: same grid and same linear source span must give the same generator.
    @test isapprox(Lt, Lth; atol=1e-12, rtol=0)
    @info "T00 paired-source baseline" full_error=opnorm(Lt-Lb) hermitian_error=opnorm(Lth-Lbh) gibbs_residual=norm(Lt*vec(Matrix(ham.gibbs)))

    # Capture the precise baseline failure, rather than accepting any exception.
    workspace_result = try
        Workspace(cfg(TimeDomain()), ham, hermitian)
    catch err
        err
    end
    @test_broken workspace_result isa Workspace
    @test workspace_result isa FieldError
    @test occursin("transition", sprint(showerror, workspace_result))

    validation_result = try
        validate_config!(cfg(TimeDomain(); a=0.0), ham)
        nothing
    catch err
        err
    end
    @test_broken validation_result === nothing
    @test validation_result isa ArgumentError
    @test occursin("eta must be > 0", sprint(showerror, validation_result))
end
