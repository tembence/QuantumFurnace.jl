using Test, QuantumFurnace, LinearAlgebra
BLAS.set_num_threads(1)

@testset "Research jump preparation" begin
    ham = HamHam(Hermitian(0.3X + 0.4Y + 0.7Z); beta_phys=0.8)
    A = ComplexF64[0 2+im; 0 0]
    j = JumpOp(A, ham)
    @test !j.hermitian
    @test j.in_eigenbasis ≈ ham.eigvecs' * A * ham.eigvecs
    @test JumpOp(j.in_eigenbasis, ham; basis=:eigen).data ≈ A
    @test JumpOp(Y, ham).hermitian
    @test !JumpOp(Y, ham).orthogonal
    A[1,2] = 9
    @test j.data[1,2] ≈ 2+im
    for bad in (zeros(3,3), fill(Inf,2,2), fill(NaN,2,2))
        @test_throws ArgumentError JumpOp(bad, ham)
    end
    @test_throws ArgumentError JumpOp(X, ham; basis=:unknown)
    @test_throws ArgumentError prepare_jumps([j], ham)
    @test_throws ArgumentError JumpOp(JumpOp(j.data, copy(j.data), false, false), ham)
    @test_throws ArgumentError JumpOp(JumpOp(j.data, copy(j.in_eigenbasis), false, true), ham)
    bad = JumpOp(j.data, ham)
    bad.in_eigenbasis[1,1] += 0.1
    @test_throws ArgumentError prepare_jumps([bad], ham; complete_adjoint=true)
    p = prepare_jumps([j,j,JumpOp(j.data',ham)], ham; complete_adjoint=true)
    @test length(p.jumps) == 4
    @test p.provenance.added_adjoint == [false,false,false,true]
    @test p.provenance.source_indices == [1,2,3,2]
    @test validate_jump_pairing(p.jumps) === nothing
    p2 = prepare_jumps(p.jumps, ham; complete_adjoint=true)
    @test length(p2.jumps) == 4
    @test all(a.data ≈ b.data for (a,b) in zip(p.jumps,p2.jumps))
    @test p.jumps[1].data !== j.data
    for completion in (false,true)
        @test_throws ArgumentError prepare_jumps([j,JumpOp(j.data',ham)], ham;
            rates=[1,2], complete_adjoint=completion)
    end
    for rates in (0,-1,Inf,NaN,[1,0],[1],[-1,1])
        @test_throws ArgumentError prepare_jumps([X,Y], ham; rates)
    end
    @test_throws ArgumentError prepare_jumps(:unknown, ham)
    @test_throws ArgumentError prepare_jumps([j], ham; basis=:eigen)
    @test length(prepare_jumps([zero(X), Matrix{ComplexF64}(I,2,2), Z],ham).jumps) == 3
    @test isempty(prepare_jumps(Matrix{ComplexF64}[],ham).jumps)
    @test_throws ArgumentError prepare_jumps([1e-30j.data],ham)
    @test_throws ArgumentError prepare_jumps([j.data, 2j.data'],ham)
    defaults = prepare_jumps(:onsite_paulis,ham)
    @test length(defaults.jumps) == 3
    @test defaults.provenance.proposal_amplitude ≈ 1/sqrt(3)
    @test defaults.provenance.generator_multiplier ≈ 1
    @test sum(j.data' * j.data for j in defaults.jumps) ≈ I
    @test all(a.data ≈ b/sqrt(3) for (a,b) in zip(defaults.jumps,(X,Y,Z)))
    ham2 = HamHam(pauli_hamiltonian(2, [0.3 => (1=>:Z,)]); beta_phys=0.8)
    @test length(prepare_jumps(:onsite_paulis,ham2).jumps) == 6
    @test sum(j.data' * j.data for j in prepare_jumps(:onsite_paulis,ham2).jumps) ≈ I
    cfg = Config(sim=Lindbladian(),domain=BohrDomain(),construction=DLL(),
        num_qubits=1,beta=beta_alg(ham,0.8),sigma=1.0,with_linear_combination=false,
        filter=DLLGaussianFilter(beta_alg(ham,0.8)))
    base = prepare_jumps([X,Y,Z],ham).jumps
    scaled = prepare_jumps([X,Y,Z],ham; rates=2.7).jumps
    L = construct_lindbladian(base,cfg,ham)
    Lscaled = construct_lindbladian(scaled,cfg,ham)
    @test Lscaled ≈ 2.7L rtol=1e-12
    @test sort(real.(eigvals(Lscaled))) ≈ 2.7sort(real.(eigvals(L))) atol=1e-12
    @test dll_coherent_op_bohr(scaled,ham,cfg.filter,cfg.beta) ≈ 2.7dll_coherent_op_bohr(base,ham,cfg.filter,cfg.beta) atol=1e-12
end
