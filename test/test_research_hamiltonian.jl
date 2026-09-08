using Test, QuantumFurnace, LinearAlgebra, SparseArrays, Random

@testset "Explicit-site and matrix Hamiltonians (T03)" begin
    X, Y, Z = pauli_string_to_matrix(["X", "Y", "Z"])
    id = Matrix{ComplexF64}(I, 2, 2)

    @testset "Explicit sites, tensor order and additive terms" begin
        terms = [-0.8 => (3 => :Y, 1 => :X),
                  0.2 => (2 => :Z,), 1.5 => (),
                  0.3 => (1 => :X, 2 => :Z, 3 => :Y),
                  0.4 => (2 => :Z,), 0.0 => (3 => :X,)]
        H = pauli_hamiltonian(3, terms)
        expected = -0.8kron(X, id, Y) + 0.6kron(id, Z, id) +
                   1.5kron(id, id, id) + 0.3kron(X, Z, Y)
        @test H isa Hermitian
        @test parent(H) isa SparseMatrixCSC
        @test isapprox(H, expected; atol=1e-14, rtol=0)
        @test isapprox(pauli_hamiltonian(1, [2 => (1 => :Y,)]), 2Y; atol=1e-14, rtol=0)
        @test isapprox(pauli_hamiltonian(2, [3 => (1 => :I,)]), 3kron(id, id); atol=1e-14, rtol=0)
        @test isapprox(pauli_hamiltonian(2, []), zeros(4, 4); atol=1e-14, rtol=0)
        @test isapprox(pauli_hamiltonian(1, [1 => (1 => :X,), -1 => (1 => :X,)]), zeros(2, 2); atol=1e-14, rtol=0)
        @test eltype(pauli_hamiltonian(1, [0.5f0 => (1 => :X,)])) === ComplexF32
        # Only the requested strings are built; no enumeration of all 4^n strings.
        large = pauli_hamiltonian(12, [1.0 => (1 => :Y, 12 => :X)])
        @test nnz(parent(large)) == 2^12
        @test size(large) == (2^12, 2^12)
    end

    @testset "Invalid explicit-site input" begin
        for n in (0, -1, Sys.WORD_SIZE)
            @test_throws ArgumentError pauli_hamiltonian(n, [])
        end
        for term in (1 => (1 => :X, 1 => :Y), 0 => (1 => :I, 1 => :X),
                     1 => (0 => :X,), 1 => (3 => :X,), 1 => (1.5 => :X,),
                     1 => (1 => :W,), 1 => (1 => "X",), 1 => (1,),
                     1 => :X, NaN => (), Inf => (), (1+im) => (), "a" => ())
            @test_throws ArgumentError pauli_hamiltonian(2, [term])
        end
        @test_throws ArgumentError pauli_hamiltonian(2, [1.0])
        @test_throws ArgumentError pauli_hamiltonian(1, [floatmax(Float64) => (), floatmax(Float64) => ()])
    end

    @testset "Physical Gibbs state and frame round trip" begin
        rng = MersenneTwister(830)
        A = randn(rng, ComplexF64, 4, 4)
        complex_h = Matrix(Hermitian((A + A') / 2))
        for H in (Y, complex_h, Matrix(Diagonal([0.0, 0.0, 2.0, 2.0]))),
            storage in (H, Hermitian(H), sparse(H)), b in (0.25, 1.0)
            ham = HamHam(storage; beta_phys=b)
            @test ham isa HamHam{Float64}
            @test isapprox(ham.rescaling_factor * (ham.data - ham.shift * I), H; atol=2e-13, rtol=1e-13)
            direct = exp(-b * H)
            direct /= tr(direct)
            @test isapprox(ham.eigvecs * ham.gibbs * ham.eigvecs', direct; atol=2e-13, rtol=1e-13)
            @test isapprox(beta_phys(ham, beta_alg(ham, b)), b; atol=1e-14, rtol=0)
            @test isempty(ham.base_terms) && isempty(ham.base_coeffs)
            @test ham.disordering_terms === nothing
            @test ham.disordering_coeffs === nothing
            cfg = Config(; sim=Lindbladian(), domain=BohrDomain(), construction=DLL(),
                num_qubits=trailing_zeros(size(H, 1)), with_linear_combination=false,
                beta=beta_alg(ham, b), beta_phys=b, sigma=1/beta_alg(ham, b),
                filter=DLLGaussianFilter(beta_alg(ham, b)))
            @test validate_config!(cfg, ham) === nothing
        end
        input = copy(complex_h)
        owned = HamHam(input; beta_phys=0.5)
        baseline = copy(owned.data)
        input .= 0
        @test isapprox(owned.data, baseline; atol=1e-14, rtol=0)
        @test HamHam([1 0; 0 -1]; beta_phys=0.5) isa HamHam{Float64}
        H32 = ComplexF32[0 -im; im 0]
        ham32 = HamHam(H32; beta_phys=0.5f0)
        @test ham32 isa HamHam{Float32}
        rho32 = exp(-0.5f0 * H32); rho32 /= tr(rho32)
        @test isapprox(ham32.eigvecs * ham32.gibbs * ham32.eigvecs', rho32; atol=2e-6, rtol=2e-6)
    end

    @testset "Scalar, zero, degeneracy and large shifts" begin
        for d in (2, 4), c in (0.0, 3.0, -7.0, 1e308)
            ham = HamHam(Matrix{Float64}(c * I, d, d); beta_phys=0.8)
            @test isapprox(ham.rescaling_factor, 1.0; atol=1e-14, rtol=0)
            @test isapprox(ham.shift, -c; atol=1e-14, rtol=0)
            @test isapprox(ham.data, zeros(d, d); atol=1e-14, rtol=0)
            @test isapprox(ham.nu_min, 0.0; atol=1e-14, rtol=0)
            @test isapprox(ham.gibbs, Matrix{ComplexF64}(I, d, d) / d; atol=1e-14, rtol=0)
            @test length(ham.bohr_dict) == 1
            @test length(only(values(ham.bohr_dict))) == d^2
        end
        degenerate = HamHam(Diagonal([0., 0., 2., 2.]); beta_phys=0.5)
        @test isapprox(degenerate.nu_min, 0.0; atol=1e-14, rtol=0)
        anchor = HamHam(X; beta_phys=0.5)
        for c in (1e12, 1e16, 1e308)
            shifted = HamHam(X + c*I; beta_phys=0.5)
            @test isapprox(shifted.data, anchor.data; atol=1e-14, rtol=0)
            @test isapprox(shifted.rescaling_factor, anchor.rescaling_factor; atol=1e-14, rtol=0)
            @test isapprox(shifted.eigvecs * shifted.gibbs * shifted.eigvecs',
                anchor.eigvecs * anchor.gibbs * anchor.eigvecs'; atol=1e-14, rtol=0)
            @test isapprox(shifted.rescaling_factor * (shifted.data - shifted.shift * I), X+c*I; atol=1e-14, rtol=2e-15)
        end
        # nu_min=0 must be valid for the actual DLL workspace, including B.
        ham = HamHam(zeros(2, 2); beta_phys=0.5)
        cfg = Config(; sim=Lindbladian(), domain=BohrDomain(), construction=DLL(),
            num_qubits=1, with_linear_combination=false, beta=0.5, sigma=2.0,
            filter=DLLGaussianFilter(0.5))
        jumps = JumpOp[JumpOp(P, ham.eigvecs' * P * ham.eigvecs, false, true) for P in (X,Y,Z)]
        ws = Workspace(cfg, ham, jumps)
        @test norm(apply_lindbladian!(ws, Matrix(ham.gibbs), cfg, ham)) < 1e-12
        @test all(isfinite, ws.G_left)
    end

    @testset "Opaque input never supplies Trotter metadata" begin
        for H in (X, zeros(2, 2), 2id)
            ham = HamHam(H; beta_phys=0.5)
            @test_throws r"requires local-term data" TrottTrott(ham, 0.5, 2)
            @test_throws r"requires local-term data" TrotterTriple(ham, 0.5, 0.5, 0.5, 2, 2, 2)
            @test_throws r"requires local-term data" trotterize(ham, 0.5, 2)
            @test_throws r"requires local-term data" QuantumFurnace._trotterize2(ham, 0.5, 2)
            @test_throws r"requires local-term data" group_hamiltonian_terms(ham)
        end
    end

    @testset "Invalid matrix input" begin
        for H in (zeros(2, 3), zeros(3, 3), zeros(0, 0), ones(1, 1),
                  [0 1; 0 0], ComplexF64[im 0; 0 -im], [Inf 0.; 0. 1.],
                  [NaN 0.; 0. 1.], fill("x", 2, 2))
            @test_throws ArgumentError HamHam(H; beta_phys=0.5)
        end
        for T in (Float16, BigFloat)
            @test_throws r"supports Float32 and Float64" HamHam(T[0 1; 1 0]; beta_phys=0.5)
        end
        @test HamHam(Real[0.0 1.0; 1.0 0.0]; beta_phys=0.5) isa HamHam{Float64}
        for b in (0.0, -0.1, NaN, Inf)
            @test_throws ArgumentError HamHam(X; beta_phys=b)
        end
    end
end
