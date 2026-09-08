using Test, QuantumFurnace, LinearAlgebra, Random
BLAS.set_num_threads(1)

@testset "Bounded independent gap checks" begin
    # Independent complete dense references, clean and disordered noncommuting
    # models, plus a classical Ising parity fixture.
    for H in (Hermitian(.4X+.7Z),
        pauli_hamiltonian(2,[-1. => (1=>:Z,2=>:Z),-.6 => (1=>:X,),-.6 => (2=>:X,)]),
        pauli_hamiltonian(2,[-1. => (1=>:Z,2=>:Z),-.6 => (1=>:X,),-.31 => (2=>:Y,),.23 => (1=>:Z,)]),
        pauli_hamiltonian(2,[-1. => (1=>:Z,2=>:Z)]))
        p = prepare_gibbs_inputs(H;beta_phys=.8)
        ws = Workspace(p.config,p.hamiltonian,p.jumps)
        L = construct_lindbladian(p.jumps,p.config,p.hamiltonian)
        ground_truth = sort(-real.(eigvals(L)))[2]
        r = robust_spectral_gap(ws;diagnostics=:strict,krylovdim=20,tol=1e-13,parent_action=true)
        @test isempty(r.failures)
        @test r.spectral_gap ≈ ground_truth rtol=1e-9
        @test r.reliability == :pass
        @test r.coverage == :complete_small_system
        @test r.parent_action.status == :pass
        @test length(r.runs) == 4
        @test r.runs[end].refinement
        @test r.resources.matvec_count <= r.resources.max_matvecs
        @test r.reference.checks.kernel.status == :pass
        plain = robust_spectral_gap(ws;krylovdim=20,tol=1e-13)
        @test plain.reliability == :inconclusive
        @test plain.coverage == :not_established
        @test plain.spectral_gap ≈ r.spectral_gap rtol=1e-10
        repeat = robust_spectral_gap(ws;krylovdim=20,tol=1e-13)
        @test plain.runs[1].result.eigenvalues ≈ repeat.runs[1].result.eigenvalues
    end

    p = prepare_gibbs_inputs(Hermitian(Z);beta_phys=.8)
    ws = Workspace(p.config,p.hamiltonian,p.jumps)
    for options in ((;max_bytes=1),(;max_matvecs=0),(;max_seconds=0.))
        r = robust_spectral_gap(ws;options...)
        @test r.resources.exhausted
        @test r.reliability == :inconclusive
        @test isempty(r.runs)
        @test isnan(r.spectral_gap)
    end
    first = robust_spectral_gap(ws;diagnostics=:quick)
    partial = robust_spectral_gap(ws;max_matvecs=first.resources.matvec_count+1)
    @test length(partial.runs) == 1
    @test partial.resources.exhausted
    @test partial.resources.matvec_count == first.resources.matvec_count+1
    @test partial.spectral_gap ≈ first.spectral_gap
    skip = robust_spectral_gap(ws;diagnostics=:strict,dense_max_dim=0)
    @test skip.reference.checks.kms.status == :not_run
    @test skip.reliability == :inconclusive
    @test_throws ArgumentError robust_spectral_gap(ws;operator_starts=[zeros(2,2),Z])
    @test_throws ArgumentError robust_spectral_gap(ws;operator_starts=[Z])
    @test_throws ArgumentError krylov_spectral_gap(p.config,p.hamiltonian,p.jumps;operator_start=zeros(4))
    @test_throws ArgumentError robust_spectral_gap(ws;gap_rtol=NaN)
    @test_throws ArgumentError robust_spectral_gap(ws;max_seconds=Inf)

    # The diagonal seed stays in the population sector, missing the slower
    # coherence rate. A generic independent start catches it. No averaging.
    pt = prepare_gibbs_inputs(Hermitian(Z);beta_phys=.8,jumps=[X,Y,.1Z])
    wt = Workspace(pt.config,pt.hamiltonian,pt.jumps)
    trapped = robust_spectral_gap(wt;operator_starts=[Matrix{ComplexF64}(I,2,2),
        ComplexF64[1 .7im; .4 .3]],krylovdim=5,howmany=3,tol=1e-13)
    @test trapped.runs[1].result.spectral_gap > trapped.spectral_gap * 1.01
    @test !trapped.agreement.gap_agreement
    @test trapped.reliability == :inconclusive
    @test trapped.runs[end].refinement

    # Ising global spin-flip even population seed cannot capture odd modes.
    H = pauli_hamiltonian(2,[-1. => (1=>:Z,2=>:Z)])
    pI = prepare_gibbs_inputs(H;beta_phys=2.)
    wI = Workspace(pI.config,pI.hamiltonian,pI.jumps)
    rI = robust_spectral_gap(wI;operator_starts=[Matrix{ComplexF64}(I,4,4),
        randn(MersenneTwister(123),ComplexF64,4,4)],howmany=6,krylovdim=17,tol=1e-13)
    exact = sort(-real.(eigvals(construct_lindbladian(pI.jumps,pI.config,pI.hamiltonian))))[2]
    @test rI.spectral_gap ≈ exact rtol=1e-8
    @test rI.agreement.status == :inconclusive
    @test rI.reliability == :inconclusive

    # A single-vector Arnoldi process need not recover a degenerate eigenspace.
    pd = prepare_gibbs_inputs(zeros(2,2);beta_phys=.8)
    deg = robust_spectral_gap(Workspace(pd.config,pd.hamiltonian,pd.jumps))
    @test deg.spectral_gap ≈ 4/3 rtol=1e-12
    @test deg.agreement.gap_agreement
    @test deg.reliability == :inconclusive
    @test deg.agreement.status == :inconclusive
end
