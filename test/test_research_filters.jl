using Test, QuantumFurnace, LinearAlgebra, Random

# No asserted admissibility trait: only the checked finite Bohr compiler admits it.
struct ResearchPhaseProbe{F} <: AbstractFilter
    beta::Float64
    amplitude::F
end
QuantumFurnace.freq_kernel(f::ResearchPhaseProbe, nu::Real) = f.amplitude(nu)

@testset "T10 complex finite Bohr filters" begin
    QF = QuantumFurnace
    rng = MersenneTwister(710)
    H = pauli_hamiltonian(2, [-0.7 => (1=>:Z,), 0.3 => (2=>:Y,)])
    for Hphys in (H, zeros(ComplexF64,4,4))
        ham = HamHam(Hphys; beta_phys=0.8)
        beta = beta_alg(ham, 0.8)
        A = randn(rng,ComplexF64,4,4)/4
        jumps = prepare_jumps([A,A'],ham).jumps
        for q in (x->exp(-x*x), x->im*x*exp(-x*x), x->exp(-x*x)*cis(0.9x))
            calls = Ref(0)
            raw = ResearchPhaseProbe(beta, x->begin calls[]+=1; q(x)*exp(-beta*x/4) end)
            f = QF._prepare_dll_bohr_filter(raw, ham.eigvals)
            count = calls[]
            @test count == length(unique(vec(ham.bohr_freqs)))
            cfg = Config(;sim=Lindbladian(),domain=BohrDomain(),construction=DLL(),
                num_qubits=2,beta,sigma=1.0,filter=f,with_linear_combination=false)
            ws = Workspace(cfg,ham,jumps)
            dense = construct_lindbladian(jumps,cfg,ham)
            Ls = [q.(ham.bohr_freqs).*exp.(-beta.*ham.bohr_freqs./4).*j.in_eigenbasis for j in jumps]
            R = sum(L'*L for L in Ls)
            B = [im/2*tanh(beta*(ham.eigvals[i]-ham.eigvals[j])/4)*R[i,j] for i in 1:4,j in 1:4]
            I4 = Matrix{ComplexF64}(I,4,4)
            ref = sum(kron(conj(L),L) for L in Ls) -
                (kron(I4,R)+kron(transpose(R),I4))/2 -
                im*(kron(I4,B)-kron(transpose(B),I4))
            @test isapprox(dense,ref; atol=2e-13,rtol=2e-13)
            @test isapprox(QF.dll_coherent_op_bohr(jumps,ham,f,beta),B; atol=2e-13)
            @test isapprox(B,B';atol=2e-13)
            X = randn(rng,ComplexF64,4,4)
            @test isapprox(vec(apply_lindbladian!(ws,X,cfg,ham)),ref*vec(X);atol=2e-13)
            @test isapprox(vec(apply_adjoint_lindbladian!(ws,X,cfg,ham)),ref'*vec(X);atol=2e-13)
            @test norm(ref*vec(Matrix(ham.gibbs))) < 2e-13
            weights = sqrt.(diag(ham.gibbs))
            metric = Diagonal(vec(weights*weights'))
            @test norm(ref*metric-metric*ref') < 2e-13
            freqs = sort(unique(vec(ham.bohr_freqs)))
            alpha = dll_kossakowski_bohr(f,freqs)
            @test norm(alpha-alpha') < 2e-13
            @test eigmin(Hermitian(alpha)) > -2e-13
            multi = DLLMultiChannelFilter([f,f],beta)
            @test isapprox(freq_kernel(multi,0.0),2freq_kernel(f,0.0);atol=2e-13)
            @test isapprox(QF.q_weight(multi,0.0),2QF.q_weight(f,0.0);atol=2e-13)
            @test isapprox(dll_kossakowski_bohr(multi,freqs),2alpha;atol=2e-13)
            @test isapprox(QF.dll_coherent_kernel_bohr(multi,first(freqs),0.0),
                2QF.dll_coherent_kernel_bohr(f,first(freqs),0.0);atol=2e-13)
            @test calls[] == count # construction and matvecs only read the snapshot
        end
    end
    @test_throws ArgumentError QF._prepare_dll_bohr_filter(ResearchPhaseProbe(1.0,x->im),[0.0,0.5])
    @test_throws ArgumentError QF._prepare_dll_bohr_filter(ResearchPhaseProbe(1.0,x->1.0),[0.0,0.5])
    @test_throws ArgumentError QF._prepare_dll_bohr_filter(ResearchPhaseProbe(1.0,x->NaN),[0.0,0.5])
end
