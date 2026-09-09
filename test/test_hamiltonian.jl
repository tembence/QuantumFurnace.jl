"""
Tests for Heisenberg Hamiltonian builders in `src/hamiltonian.jl`.

Covers local-term construction, disorder, the seed-driven `build_heis_1d`
and `build_tfim_2d` builders, and direct and term-based HamHam constructors.
"""

using LinearAlgebra
using SparseArrays
using Random
using Statistics: median

@testset "Heisenberg Hamiltonian builders" begin

    @testset "Public Pauli conversion and grouping helpers" begin
        @test isapprox(Had' * Had, I(2); atol=1e-15)
        converted = pauli_string_to_matrix(["X", "Y", "Z", "I"])
        @test converted[1] == X
        @test converted[2] == Y
        @test converted[3] == Z
        @test converted[4] == Matrix{ComplexF64}(I, 2, 2)
        @test_throws KeyError pauli_string_to_matrix(["not-a-Pauli"])

        terms = Vector{Matrix{ComplexF64}}[[X, X], [Y, Y], [X, Y], [Z]]
        grouped = group_hamiltonian_terms(HamHam(terms, [1.0, 2.0, 3.0, 4.0], 3, 1.0))
        @test grouped.commuting[1] == terms[1:2]
        @test grouped.noncommuting[1] == terms[3:3]
        @test grouped.one_sites[1] == terms[4:4]
    end

    # _pad_two_site_op smoke tests --------------------------------------------------
    @testset "_pad_two_site_op: adjacent and non-adjacent placements" begin
        # n=3 adjacent at the chain edge: place X at q=1, q=2
        op = QuantumFurnace._pad_two_site_op([X, X], 3, 1, 2)
        @test size(op) == (8, 8)
        @test ishermitian(Matrix(op))
        @test Matrix(op) ≈ kron(X, X, I(2))

        # n=4 adjacent: place X at q=2, q=3
        op = QuantumFurnace._pad_two_site_op([X, X], 4, 2, 3)
        @test Matrix(op) ≈ kron(I(2), X, X, I(2))

        # n=4 non-adjacent: place Z at q=1, q=4 (separation 3)
        op = QuantumFurnace._pad_two_site_op([Z, Z], 4, 1, 4)
        @test Matrix(op) ≈ kron(Z, I(2), I(2), Z)

        # Order-independence for symmetric terms: q1<q2 vs q1>q2
        op_a = QuantumFurnace._pad_two_site_op([Z, Z], 4, 1, 4)
        op_b = QuantumFurnace._pad_two_site_op([Z, Z], 4, 4, 1)
        @test Matrix(op_a) ≈ Matrix(op_b)
    end

    @testset "_pad_two_site_op: argument validation" begin
        @test_throws ArgumentError QuantumFurnace._pad_two_site_op([X], 4, 1, 2)         # 1-site term
        @test_throws ArgumentError QuantumFurnace._pad_two_site_op([X, X], 4, 2, 2)      # q1 == q2
        @test_throws ArgumentError QuantumFurnace._pad_two_site_op([X, X], 4, 0, 2)      # q < 1
        @test_throws ArgumentError QuantumFurnace._pad_two_site_op([X, X], 4, 2, 5)      # q > num_qubits
    end

    # _construct_2d_heisenberg_base ----------------------------------------------------
    @testset "_construct_2d_heisenberg_base: dimension and Hermiticity for several lattices" begin
        for (Lx, Ly) in [(2, 3), (3, 3), (2, 5)]
            n = Lx * Ly
            ham = QuantumFurnace._construct_2d_heisenberg_base(Lx, Ly,
                [[X, X], [Y, Y], [Z, Z]], [1.0, 1.0, 1.5];
                periodic_x=true, periodic_y=true)
            @test size(ham) == (2^n, 2^n)
            @test ishermitian(Matrix(ham))
            # Heisenberg model is traceless
            @test abs(tr(Matrix(ham))) < 1e-10
        end
    end

    @testset "build_heis_1d: returns a valid raw NamedTuple" begin
        raw = build_heis_1d(3, [1.0, 1.0, 1.0]; seed=20260515,
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z]], disorder_strength=1.0)
        local_ham = build_local_heis_1d(3, [1.0, 1.0, 1.0]; seed=20260515,
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z]], disorder_strength=1.0)
        @test raw.nu_min > 0
        @test size(raw.matrix) == (8, 8)
        @test size(raw.eigvecs) == (8, 8)
        @test length(raw.eigvals) == 8
        @test raw.periodic === true
        @test length(raw.disordering_terms) == 1
        @test length(raw.disordering_coeffs) == 1
        @test length(raw.disordering_coeffs[1]) == 3
        @test raw.seed === 20260515
        @test raw.disorder_strength == 1.0
        @test minimum(raw.eigvals) ≥ -1e-10
        @test maximum(raw.eigvals) ≤ 0.45 + 1e-10
        @test isapprox(raw.matrix, raw.matrix'; atol=1e-12)
        @test local_ham isa LocalHamiltonian1D{Float64}
        @test local_ham.coordinate_frame == :physical
        @test !hasfield(typeof(local_ham), :matrix)
        @test !hasfield(typeof(local_ham), :eigvals)
    end

    @testset "build_heis_1d: HamHam wrap end-to-end (extra fields ignored)" begin
        raw = build_heis_1d(3, [1.0, 1.0, 1.0]; seed=20260516, disorder_strength=1.0)
        ham = HamHam(raw, 1.0)
        @test ham isa HamHam{Float64}
        @test size(ham.data) == (8, 8)
        @test ham.nu_min > 0
        @test isapprox(tr(ham.gibbs), 1.0; atol=1e-10)
    end

    @testset "build_heis_1d: same seed gives identical fixture" begin
        raw_a = build_heis_1d(4, [1.0, 1.0, 1.0]; seed=20260517,
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
            disorder_strength=0.5)
        raw_b = build_heis_1d(4, [1.0, 1.0, 1.0]; seed=20260517,
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
            disorder_strength=0.5)
        @test raw_a.eigvals ≈ raw_b.eigvals
        @test isapprox(raw_a.matrix, raw_b.matrix; atol=1e-14)
    end

    @testset "build_heis_1d: different seeds give different fixtures" begin
        raw_a = build_heis_1d(4, [1.0, 1.0, 1.0]; seed=1, disorder_strength=0.5)
        raw_b = build_heis_1d(4, [1.0, 1.0, 1.0]; seed=2, disorder_strength=0.5)
        @test !isapprox(raw_a.matrix, raw_b.matrix; atol=1e-8)
    end

    @testset "build_heis_1d: disorder_strength bounds the per-coefficient magnitude" begin
        raw = build_heis_1d(3, [1.0, 1.0, 1.0]; seed=20260518,
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z]], disorder_strength=1e-2)
        # raw.disordering_coeffs is the *rescaled* version; each rescaled entry is at
        # most `disorder_strength / rescaling_factor` < 1e-2.
        @test all(abs.(raw.disordering_coeffs[1]) .≤ 1e-2)
    end

    @testset "build_tfim_2d: returns a valid raw NamedTuple" begin
        raw = build_tfim_2d(2, 2; J=1.0, h=1.0, seed=20260520,
            periodic_x=true, periodic_y=true,
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z]],
            disorder_strength=1e-2)
        @test raw.nu_min > 0
        @test size(raw.matrix) == (16, 16)
        @test length(raw.eigvals) == 16
        @test raw.periodic === true
        @test raw.Lx == 2
        @test raw.Ly == 2
        @test raw.periodic_x === true
        @test raw.periodic_y === true
        @test raw.J == 1.0
        @test raw.h == 1.0
        @test length(raw.disordering_coeffs[1]) == 4
        @test minimum(raw.eigvals) ≥ -1e-10
        @test maximum(raw.eigvals) ≤ 0.45 + 1e-10
    end

    @testset "build_tfim_2d: HamHam wrap end-to-end" begin
        raw = build_tfim_2d(2, 2; J=1.0, h=1.0, seed=20260521, disorder_strength=1e-3)
        ham = HamHam(raw, 2.0)
        @test ham isa HamHam{Float64}
        @test size(ham.data) == (16, 16)
        @test isapprox(tr(ham.gibbs), 1.0; atol=1e-10)
    end

    @testset "build_tfim_2d: argument validation" begin
        @test_throws ArgumentError build_tfim_2d(0, 2; seed=1)
        @test_throws ArgumentError build_tfim_2d(2, 0; seed=1)
        @test_throws ArgumentError build_tfim_2d(2, 2; J=NaN, seed=1)
        @test_throws ArgumentError build_tfim_2d(2, 2; h=Inf, seed=1)
        @test_throws ArgumentError build_tfim_2d(2, 2; disorder_strength=-0.1, seed=1)
        @test_throws ArgumentError build_tfim_2d(2, 2; seed=1,
            disordering_terms=Vector{Matrix{ComplexF64}}[[X, Y, Z]])

        one_site = build_tfim_2d(1, 1; seed=1, disorder_strength=0.0)
        one_by_two = build_tfim_2d(1, 2; seed=1, disorder_strength=0.0)
        @test size(one_site.matrix) == (2, 2)
        @test size(one_by_two.matrix) == (4, 4)
        @test HamHam(one_site, 1.0) isa HamHam
    end

    @testset "build_heis_1d: argument validation" begin
        @test_throws ArgumentError build_heis_1d(0, [1.0, 1.0, 1.0]; seed=1)
        @test_throws ArgumentError build_heis_1d(1, [1.0, 1.0, 1.0]; seed=1)
        @test_throws ArgumentError build_heis_1d(3, [1.0, 1.0]; seed=1)
        @test_throws ArgumentError build_heis_1d(3, [1.0, NaN, 1.0]; seed=1)
        @test_throws ArgumentError build_heis_1d(3, [1.0, 1.0, 1.0];
            seed=1, disorder_strength=-0.1)
        @test_throws ArgumentError build_heis_1d(3, [1.0, 1.0, 1.0]; seed=1,
            disordering_terms=Vector{Matrix{ComplexF64}}[[X, Y, Z]])
        noninvolution = ComplexF64[1.0 0.0; 0.0 2.0]
        @test_throws ArgumentError build_heis_1d(3, [1.0, 1.0, 1.0]; seed=1,
            disordering_terms=Vector{Matrix{ComplexF64}}[[noninvolution]])
        near_noninvolution = ComplexF64[1.0 0.0; 0.0 1.0 + 1e-10]
        @test_throws ArgumentError build_heis_1d(3, [1.0, 1.0, 1.0]; seed=1,
            disordering_terms=Vector{Matrix{ComplexF64}}[[near_noninvolution]])
    end

    @testset "Stable Gibbs weights and raw Hamiltonian validation" begin
        shifted_weights = QuantumFurnace._gibbs_weights([1000.0, 1001.0], 1000.0)
        @test all(isfinite, shifted_weights)
        @test isapprox(sum(shifted_weights), 1.0; atol=1e-15, rtol=0)
        @test shifted_weights[1] == 1.0
        @test shifted_weights[2] == 0.0
        degenerate_weights = QuantumFurnace._gibbs_weights([-2.0, -2.0, 3.0], 1.0e308)
        @test degenerate_weights == [0.5, 0.5, 0.0]
        hot_weights = QuantumFurnace._gibbs_weights([-2.0, -2.0, 3.0], eps(Float64))
        @test isapprox(hot_weights, fill(1 / 3, 3); atol=1e-15, rtol=0)
        @test_throws ArgumentError QuantumFurnace._gibbs_weights([0.0, 1.0], 0.0)
        @test_throws ArgumentError QuantumFurnace._gibbs_weights([0.0, 1.0], Inf)

        constant_h = Hermitian(Matrix{ComplexF64}(2.0I, 2, 2))
        constant_scale, constant_shift = QuantumFurnace._rescaling_and_shift_factors(constant_h)
        @test isapprox(constant_scale, 1.0; atol=1e-15, rtol=0)
        @test isapprox(constant_shift, -2.0; atol=1e-15, rtol=0)
        empty_ham = HamHam(Vector{Vector{Matrix{ComplexF64}}}(), Float64[], 1, 1.0)
        @test isapprox(empty_ham.gibbs, Matrix{ComplexF64}(I, 2, 2) / 2; atol=1e-15, rtol=0)

        identity_2 = Matrix{ComplexF64}(I, 2, 2)
        offset_ham = HamHam(
            Vector{Matrix{ComplexF64}}[[identity_2], [Z]], [1.0e15, 1.0], 1, 1.0)
        @test isapprox(first(offset_ham.eigvals), 0.0; atol=1e-15, rtol=0)
        @test isapprox(last(offset_ham.eigvals), 0.45; atol=1e-15, rtol=0)
        @test QuantumFurnace._check_1d_trotter_compatible(offset_ham) < 1e-12

        offset_x = Hermitian(ComplexF64[1.0e16 1.0; 1.0 1.0e16])
        rescaled_x, rescaling_x, _ = QuantumFurnace._rescale_hamiltonian(offset_x)
        @test isapprox(rescaling_x, 2 / 0.45; atol=1e-14, rtol=0)
        @test isapprox(Matrix(rescaled_x), 0.225 .* ComplexF64[1 1; 1 1];
            atol=1e-15, rtol=0)
        @test isapprox(LinearAlgebra.eigvals(rescaled_x), [0.0, 0.45];
            atol=1e-15, rtol=0)

        huge_offset_ham = HamHam(
            Vector{Matrix{ComplexF64}}[[identity_2], [X]], [1.0e308, 1.0], 1, 1.0)
        @test isfinite(QuantumFurnace._check_1d_trotter_compatible(huge_offset_ham))
        @test_throws ArgumentError QuantumFurnace._check_1d_trotter_compatible(
            huge_offset_ham; tol=Inf)

        eigvals = [0.1, 0.4]
        raw = (
            matrix=ComplexF64[0.1 0.0; 0.0 0.4],
            terms=Vector{Vector{Matrix{ComplexF64}}}(),
            base_coeffs=Float64[],
            disordering_terms=nothing,
            disordering_coeffs=nothing,
            eigvals=eigvals,
            eigvecs=Matrix{ComplexF64}(I, 2, 2),
            nu_min=0.3,
            shift=0.2,
            rescaling_factor=2.0,
            periodic=false,
        )
        ham = HamHam(raw, 1.0e5)
        @test all(isfinite, ham.gibbs)
        @test isapprox(tr(ham.gibbs), 1.0; atol=1e-15, rtol=0)
        @test ham.gibbs[1, 1] == 1.0

        theta = 0.37
        eigvecs_rotated = ComplexF64[
            cos(theta) -sin(theta)
            sin(theta)  cos(theta)
        ]
        raw_rotated = merge(raw, (
            matrix=eigvecs_rotated * Diagonal(eigvals) * eigvecs_rotated',
            eigvecs=eigvecs_rotated,
        ))
        beta_extreme = 1.0e308
        ham_rotated = HamHam(raw_rotated, beta_extreme)
        rho_eigen = gibbs_state_in_eigen(ham_rotated, beta_extreme)
        rho_computational = gibbs_state(ham_rotated, beta_extreme)
        @test rho_eigen isa Matrix{ComplexF64}
        @test rho_computational isa Matrix{ComplexF64}
        @test isdiag(rho_eigen)
        @test isapprox(tr(rho_eigen), 1.0; atol=1e-15, rtol=0)
        @test isapprox(tr(rho_computational), 1.0; atol=1e-15, rtol=0)
        @test isapprox(
            rho_computational,
            eigvecs_rotated * rho_eigen * eigvecs_rotated';
            atol=1e-15,
            rtol=0,
        )
        @test isapprox(
            eigvecs_rotated' * rho_computational * eigvecs_rotated,
            rho_eigen;
            atol=1e-15,
            rtol=0,
        )
        @test abs(rho_computational[1, 2]) > 0.1
        @test is_density_matrix(rho_computational)
        @test QuantumFurnace._validate_raw_hamiltonian(
            raw; full_spectral_max_dim=1) === nothing
        @test QuantumFurnace._validate_raw_hamiltonian(
            raw; spectral_validation=:full) === nothing
        @test_throws ArgumentError QuantumFurnace._validate_raw_hamiltonian(
            raw; spectral_validation=:unknown)

        raw32 = (
            matrix=ComplexF32[0.1 0.0; 0.0 0.4],
            terms=Vector{Vector{Matrix{ComplexF32}}}(),
            base_coeffs=Float32[],
            disordering_terms=nothing,
            disordering_coeffs=nothing,
            eigvals=Float32[0.1, 0.4],
            eigvecs=Matrix{ComplexF32}(I, 2, 2),
            nu_min=0.3f0,
            shift=0.2f0,
            rescaling_factor=2.0f0,
            periodic=false,
        )
        @test HamHam(raw32, 1.0f0) isa HamHam{Float32}

        dim32 = 128
        eigvals32 = collect(range(0.0f0, 0.45f0; length=dim32))
        eigvecs32 = Matrix{ComplexF32}(I, dim32, dim32)
        raw32_large = (
            matrix=Matrix(Diagonal(ComplexF32.(eigvals32))),
            terms=Vector{Vector{Matrix{ComplexF32}}}(),
            base_coeffs=Float32[],
            disordering_terms=nothing,
            disordering_coeffs=nothing,
            eigvals=eigvals32,
            eigvecs=eigvecs32,
            nu_min=minimum(diff(eigvals32)),
            shift=0.0f0,
            rescaling_factor=1.0f0,
            periodic=false,
        )
        @test QuantumFurnace._validate_raw_hamiltonian(
            raw32_large; spectral_validation=:full) === nothing
        ham32_large = HamHam(raw32_large, 1.0f6; spectral_validation=:full)
        rho_eigen32 = gibbs_state_in_eigen(ham32_large, 1.0f6)
        rho_computational32 = gibbs_state(ham32_large, 1.0f6)
        @test rho_eigen32 isa Matrix{ComplexF32}
        @test rho_computational32 isa Matrix{ComplexF32}
        @test isdiag(rho_eigen32)
        @test isapprox(tr(rho_eigen32), 1.0f0; atol=eps(Float32), rtol=0)
        @test is_density_matrix(rho_computational32)

        scaled_eigvecs32 = copy(eigvecs32)
        scaled_eigvecs32[:, 1] .*= 1.0008f0
        @test_throws ArgumentError QuantumFurnace._validate_raw_hamiltonian(
            merge(raw32_large, (eigvecs=scaled_eigvecs32,)); spectral_validation=:full)

        rotated_eigvecs32 = copy(eigvecs32)
        theta32 = 6.0f-4
        rotated_eigvecs32[:, [1, end]] .= rotated_eigvecs32[:, [1, end]] *
            Float32[cos(theta32) -sin(theta32); sin(theta32) cos(theta32)]
        @test_throws ArgumentError QuantumFurnace._validate_raw_hamiltonian(
            merge(raw32_large, (eigvecs=rotated_eigvecs32,)); spectral_validation=:full)

        rng32 = MersenneTwister(20260805)
        delocalized_eigvecs32 = Matrix(qr(randn(rng32, ComplexF32, dim32, dim32)).Q)
        delocalized_matrix32 = Matrix(Hermitian(
            delocalized_eigvecs32 * Diagonal(eigvals32) * delocalized_eigvecs32'))
        raw32_delocalized = merge(raw32_large, (
            matrix=delocalized_matrix32,
            eigvecs=delocalized_eigvecs32,
        ))
        @test QuantumFurnace._validate_raw_hamiltonian(
            raw32_delocalized; spectral_validation=:full) === nothing

        corrupted_delocalized32 = copy(delocalized_eigvecs32)
        theta_delocalized32 = 1.5f-3
        corrupted_delocalized32[:, [1, end]] .=
            corrupted_delocalized32[:, [1, end]] *
            Float32[
                cos(theta_delocalized32) -sin(theta_delocalized32)
                sin(theta_delocalized32) cos(theta_delocalized32)
            ]
        @test_throws ArgumentError QuantumFurnace._validate_raw_hamiltonian(
            merge(raw32_delocalized, (eigvecs=corrupted_delocalized32,));
            spectral_validation=:full)

        @test_throws ArgumentError HamHam(merge(raw, (matrix=zeros(ComplexF64, 2, 3),)), 1.0)
        @test_throws ArgumentError HamHam(merge(raw, (
            matrix=ComplexF64[0.1 0.2im; 0.0 0.4],)), 1.0)
        @test_throws ArgumentError HamHam(merge(raw, (eigvals=reverse(eigvals),)), 1.0)
        @test_throws ArgumentError HamHam(merge(raw, (
            eigvecs=ComplexF64[2.0 0.0; 0.0 1.0],)), 1.0)
        @test_throws ArgumentError HamHam(merge(raw, (
            matrix=ComplexF64[0.1 0.0; 0.0 0.41],)), 1.0)
        @test_throws ArgumentError HamHam(merge(raw, (nu_min=0.2,)), 1.0)
        @test_throws ArgumentError HamHam(merge(raw, (rescaling_factor=0.0,)), 1.0)
        @test_throws ArgumentError HamHam(merge(raw, (shift=NaN,)), 1.0)
        @test_throws ArgumentError HamHam(merge(raw, (disordering_coeffs=[[0.1]],)), 1.0)
        @test_throws ArgumentError HamHam(merge(raw, (
            matrix=ComplexF64[-0.1 0.0; 0.0 0.4],
            eigvals=[-0.1, 0.4], nu_min=0.5,)), 1.0)
        @test_throws ArgumentError HamHam(merge(raw, (
            matrix=ComplexF64[0.1 0.0; 0.0 0.6],
            eigvals=[0.1, 0.6], nu_min=0.5,)), 1.0)
        @test_throws ArgumentError HamHam(merge(raw, (
            terms=Vector{Matrix{ComplexF64}}[[Z]], base_coeffs=[NaN],)), 1.0)
        @test_throws ArgumentError HamHam(merge(raw, (
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z]],
            disordering_coeffs=[[0.1, 0.2]],)), 1.0)
        @test_throws ArgumentError HamHam(raw, 0.0)
    end


    @testset "HamHam ctor (no disorder): direct coverage" begin
        # Ctor (1) was reachable only transitively via the NamedTuple ctor (3)
        # before this test. Smoke-test the direct path.
        n = 3
        terms = Vector{Matrix{ComplexF64}}[[X, X], [Y, Y], [Z, Z]]
        coeffs = [1.0, 1.0, 1.0]
        h = HamHam(terms, coeffs, n, 1.0)
        @test h.periodic === true
        @test h.disordering_terms === nothing
        @test h.disordering_coeffs === nothing
        @test size(h.data) == (2^n, 2^n)
        @test isapprox(tr(h.gibbs), 1.0; atol=1e-12)
    end

    @testset "HamHam ctor (single-term convenience): wraps to multi-term" begin
        # Ctor (2b): single-term sugar over (2). Verify the wrapped multi-term
        # storage (disordering_terms is a 1-element vector).
        n = 3
        terms = Vector{Matrix{ComplexF64}}[[X, X], [Y, Y], [Z, Z]]
        coeffs = [1.0, 1.0, 1.0]
        dis_term = Matrix{ComplexF64}[Z]           # singular term: Vector{Matrix}
        dis_coeffs = [0.1, -0.1, 0.05]             # singular coeff vector
        h = HamHam(terms, coeffs, dis_term, dis_coeffs, n, 1.0)
        @test h.disordering_terms isa Vector
        @test length(h.disordering_terms) == 1
        @test length(h.disordering_coeffs) == 1
        @test isapprox(tr(h.gibbs), 1.0; atol=1e-12)
    end

    @testset "build_tfim_2d: deterministic disorder and physical conventions" begin
        kwargs = (;
            J=1.0, h=1.0, disordering_terms=Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
            disorder_strength=1e-3,
        )
        raw_a = build_tfim_2d(2, 3; kwargs..., seed=20260516)
        raw_b = build_tfim_2d(2, 3; kwargs..., seed=20260516)
        raw_c = build_tfim_2d(2, 3; kwargs..., seed=20260517)
        @test raw_a.matrix == raw_b.matrix
        @test raw_a.eigvals == raw_b.eigvals
        @test raw_a.disordering_coeffs == raw_b.disordering_coeffs
        @test !isapprox(raw_a.matrix, raw_c.matrix; atol=1e-10)

        Lx = Ly = 2
        n = Lx * Ly
        raw = build_tfim_2d(Lx, Ly; J=1.0, h=1.0, seed=1,
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z]], disorder_strength=0.0)
        H_physical = raw.rescaling_factor * (raw.matrix - raw.shift * I(2^n))
        H_bonds = QuantumFurnace._construct_2d_heisenberg_base(
            Lx, Ly, Vector{Matrix{ComplexF64}}[[Z, Z]], [-1.0];
            periodic_x=true, periodic_y=true)
        H_field = QuantumFurnace._construct_disordering_terms(
            Vector{Matrix{ComplexF64}}[[X]], [fill(-1.0, n)], n)
        @test isapprox(H_physical, Matrix(H_bonds) + Matrix(H_field); atol=1e-10)

        ham_phys = HamHam(raw; beta_phys=0.5)
        ham_alg = HamHam(raw, 0.5 * ham_phys.rescaling_factor)
        @test ham_phys.gibbs == ham_alg.gibbs
    end

    @testset "build_tfim_2d: clean Ising structure and spectrum" begin
        n = 4
        raw = build_tfim_2d(2, 2; J=1.0, h=0.0, seed=1,
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z]], disorder_strength=0.0)
        @test raw.matrix == Diagonal(diag(raw.matrix))

        H_physical = raw.rescaling_factor * (raw.matrix - raw.shift * I(2^n))
        eigs = sort(real.(eigvals(Hermitian(H_physical))))
        @test all(abs.(eigs ./ 4 .- round.(eigs ./ 4)) .< 1e-10)
        @test minimum(eigs) ≈ -8.0 atol=1e-10
        @test maximum(eigs) ≈ 8.0 atol=1e-10
    end

    @testset "build_tfim_2d: two-site disorder follows lattice bonds" begin
        Lx, Ly = 2, 3
        n = Lx * Ly
        coeffs = [Float64.(1:n)]
        H_2d = Matrix(QuantumFurnace._construct_disordering_terms_2d(
            Lx, Ly, Vector{Matrix{ComplexF64}}[[Z, Z]], coeffs;
            periodic_x=true, periodic_y=true))
        H_chain = Matrix(QuantumFurnace._construct_disordering_terms(
            Vector{Matrix{ComplexF64}}[[Z, Z]], coeffs, n; periodic=true))
        @test H_2d != H_chain

        ZZ_34 = Matrix(QuantumFurnace._pad_two_site_op([Z, Z], n, 3, 4))
        @test abs(real(tr(H_chain * ZZ_34)) / 2^n) > 0.1
        @test abs(real(tr(H_2d * ZZ_34)) / 2^n) < 1e-12
    end

    @testset "load_hamiltonian uses packaged data" begin
        path = test_hamiltonian_path(3)
        @test isfile(path)
        @test startswith(path, QuantumFurnace._PACKAGE_DATA_DIR)

        loaded = mktempdir() do tmp
            cd(tmp) do
                load_hamiltonian("heis", 3; beta=10)
            end
        end
        direct = QuantumFurnace._load_hamiltonian_bson(path, 10.0)
        @test loaded.data == direct.data
        @test loaded.eigvals == direct.eigvals
        @test loaded.gibbs == direct.gibbs

        @test_throws ArgumentError load_hamiltonian("tfim", 3; beta=10.0)
        @test_throws ArgumentError load_hamiltonian("heis", 2; beta=10.0)
        @test_throws ArgumentError load_hamiltonian("heis", 3; beta=0.0)
    end
end
