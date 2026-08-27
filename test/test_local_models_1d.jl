using LinearAlgebra
using Random

function _legacy_heis_1d_reference(
    num_qubits::Int,
    coeffs::Vector{Float64};
    seed::Int,
    periodic::Bool,
    disordering_terms::Vector{Vector{Matrix{ComplexF64}}},
    disorder_strength::Float64,
)
    base_terms = Vector{Matrix{ComplexF64}}[[X, X], [Y, Y], [Z, Z]]
    base_hamiltonian = QuantumFurnace._construct_base_ham(
        base_terms, coeffs, num_qubits; periodic=periodic)
    rng = MersenneTwister(seed)
    sample_coeffs = [zeros(Float64, num_qubits) for _ in disordering_terms]
    for coefficients in sample_coeffs
        rand!(rng, coefficients)
        coefficients .*= disorder_strength
    end
    disordering_hamiltonian = QuantumFurnace._construct_disordering_terms(
        disordering_terms, sample_coeffs, num_qubits; periodic=periodic)
    total_hamiltonian = Hermitian(
        Matrix(base_hamiltonian) + Matrix(disordering_hamiltonian))
    rescaled, rescaling_factor, shift =
        QuantumFurnace._rescale_hamiltonian(total_hamiltonian)
    matrix = Matrix(rescaled)
    eigenvalues, eigenvectors = eigen(Hermitian(matrix))
    return (
        matrix=matrix,
        terms=base_terms,
        base_coeffs=coeffs ./ rescaling_factor,
        disordering_terms=disordering_terms,
        disordering_coeffs=[coefficients ./ rescaling_factor for coefficients in sample_coeffs],
        eigvals=eigenvalues,
        eigvecs=eigenvectors,
        nu_min=minimum(diff(eigenvalues)),
        shift=shift,
        rescaling_factor=rescaling_factor,
        periodic=periodic,
        seed=seed,
        disorder_strength=disorder_strength,
    )
end

@testset "Backend-neutral local 1D models" begin
    @testset "Typed term and Hamiltonian invariants" begin
        term = LocalTerm1D([2, 3], kron(X, Y), -0.25, 4; boundary=:open)
        @test isconcretetype(typeof(term))
        @test term.sites == [2, 3]
        @test term.boundary == :open
        @test !term.wraps_boundary
        @test isapprox(term.matrix, kron(X, Y); atol=2e-15, rtol=2e-15)

        wrap = LocalTerm1D([4, 1], kron(X, Z), 0.5, 4; boundary=:periodic)
        @test wrap.wraps_boundary
        @test isapprox(
            Matrix(QuantumFurnace._embed_local_matrix_1d(
                wrap.sites, wrap.matrix, 4, 2)),
            Matrix(QuantumFurnace._pad_two_site_op([X, Z], 4, 4, 1));
            atol=2e-15,
            rtol=2e-15,
        )

        @test_throws ArgumentError LocalTerm1D([1, 1], kron(X, X), 1.0, 3)
        @test_throws ArgumentError LocalTerm1D([0], X, 1.0, 3)
        @test_throws ArgumentError LocalTerm1D([1, 3], kron(X, X), 1.0, 3)
        @test_throws ArgumentError LocalTerm1D(
            [3, 1], kron(X, X), 1.0, 3; boundary=:open)
        @test_throws ArgumentError LocalTerm1D([1], ones(ComplexF64, 4, 4), 1.0, 3)
        @test_throws ArgumentError LocalTerm1D(
            [1], ComplexF64[0 1; 0 0], 1.0, 3)

        hamiltonian = LocalHamiltonian1D(
            LocalTerm1D{Float64}[term]; num_sites=4, boundary=:open)
        @test isconcretetype(typeof(hamiltonian))
        @test hamiltonian.coordinate_frame == :physical
        @test hamiltonian.rescaling_factor == 1.0
        @test hamiltonian.scale_provenance == :identity_physical_frame
        @test hamiltonian.spectral_bound_provenance ==
              :weyl_gershgorin_local_term_ranges
        @test hamiltonian.spectral_width_bound >= 0
        @test !any(fieldtype(typeof(hamiltonian), index) === Any
                   for index in 1:fieldcount(typeof(hamiltonian)))

        original_matrix = Matrix(materialize_local_hamiltonian(hamiltonian))
        original_bounds = (
            hamiltonian.spectral_lower_bound,
            hamiltonian.spectral_upper_bound,
            hamiltonian.spectral_width_bound,
        )
        hamiltonian.terms[1].matrix[1, 1] = 1.0e6
        hamiltonian.terms[1].sites[1] = 4
        @test isapprox(
            Matrix(materialize_local_hamiltonian(hamiltonian)), original_matrix;
            atol=2e-15, rtol=2e-15)
        @test all(isapprox(actual, expected; atol=2e-15, rtol=2e-15)
                  for (actual, expected) in zip((
                      hamiltonian.spectral_lower_bound,
                      hamiltonian.spectral_upper_bound,
                      hamiltonian.spectral_width_bound,
                  ), original_bounds))

        externally_owned_term = LocalTerm1D([1], X, 1.0, 2)
        owned_hamiltonian = LocalHamiltonian1D(
            LocalTerm1D{Float64}[externally_owned_term]; num_sites=2)
        getfield(externally_owned_term, :_matrix)[1, 1] = 1.0e6
        @test isapprox(
            Matrix(materialize_local_hamiltonian(owned_hamiltonian)),
            kron(X, Matrix{ComplexF64}(I, 2, 2));
            atol=2e-15,
            rtol=2e-15,
        )

        stale_hamiltonian = LocalHamiltonian1D(
            LocalTerm1D{Float64}[LocalTerm1D([1], X, 1.0, 2)]; num_sites=2)
        getfield(first(getfield(stale_hamiltonian, :_terms)), :_matrix)[1, 1] = 1.0e6
        @test_throws ArgumentError materialize_local_hamiltonian(stale_hamiltonian)

        shifted_physical = LocalHamiltonian1D(
            LocalTerm1D{Float64}[LocalTerm1D([1], X, 1.0, 2)];
            num_sites=2,
            global_shift=2.0,
        )
        shifted_algorithm = QuantumFurnace._to_algorithm_frame(
            shifted_physical;
            rescaling_factor=4.0,
            shift=0.1,
            scale_provenance=:exact_dense,
        )
        expected_shifted_algorithm =
            Matrix(materialize_local_hamiltonian(shifted_physical)) ./ 4.0
        expected_shifted_algorithm[diagind(expected_shifted_algorithm)] .+= 0.1
        @test isapprox(shifted_algorithm.global_shift, 0.6;
                       atol=2e-15, rtol=2e-15)
        @test isapprox(
            Matrix(materialize_local_hamiltonian(shifted_algorithm)),
            expected_shifted_algorithm;
            atol=2e-15,
            rtol=2e-15,
        )
    end

    @testset "Local Heisenberg specification preserves the dense builder" begin
        disordering_terms = Vector{Matrix{ComplexF64}}[[X], [Z, Z]]
        coefficients = [1.0, -0.75, 0.4]
        for periodic in (false, true), num_qubits in (2, 3, 4), seed in (1, 46)
            kwargs = (;
                seed=seed,
                periodic=periodic,
                disordering_terms=disordering_terms,
                disorder_strength=0.35,
            )
            reference = _legacy_heis_1d_reference(
                num_qubits, coefficients; kwargs...)
            actual = build_heis_1d(num_qubits, coefficients; kwargs...)
            @test keys(actual) == keys(reference)
            @test isapprox(actual.matrix, reference.matrix; atol=2e-15, rtol=2e-15)
            @test all(isapprox(actual_factor, reference_factor;
                               atol=2e-15, rtol=2e-15)
                      for (actual_pattern, reference_pattern) in
                          zip(actual.terms, reference.terms)
                      for (actual_factor, reference_factor) in
                          zip(actual_pattern, reference_pattern))
            @test isapprox(
                actual.base_coeffs, reference.base_coeffs; atol=2e-15, rtol=2e-15)
            @test all(isapprox(actual_factor, reference_factor;
                               atol=2e-15, rtol=2e-15)
                      for (actual_pattern, reference_pattern) in
                          zip(actual.disordering_terms, reference.disordering_terms)
                      for (actual_factor, reference_factor) in
                          zip(actual_pattern, reference_pattern))
            @test all(isapprox(actual_coefficients, reference_coefficients;
                               atol=2e-15, rtol=2e-15)
                      for (actual_coefficients, reference_coefficients) in
                          zip(actual.disordering_coeffs, reference.disordering_coeffs))
            @test isapprox(actual.eigvals, reference.eigvals; atol=2e-15, rtol=2e-15)
            @test isapprox(actual.eigvecs, reference.eigvecs; atol=2e-15, rtol=2e-15)
            @test isapprox(actual.nu_min, reference.nu_min; atol=2e-15, rtol=2e-15)
            @test isapprox(actual.shift, reference.shift; atol=2e-15, rtol=2e-15)
            @test isapprox(actual.rescaling_factor, reference.rescaling_factor;
                           atol=2e-15, rtol=2e-15)
            @test actual.periodic === reference.periodic
            @test actual.seed == reference.seed
            @test isapprox(actual.disorder_strength, reference.disorder_strength;
                           atol=2e-15, rtol=2e-15)

            local_physical = build_local_heis_1d(
                num_qubits, coefficients; kwargs...)
            physical_dense = Matrix(materialize_local_hamiltonian(local_physical))
            reconstructed = physical_dense ./ actual.rescaling_factor
            reconstructed[diagind(reconstructed)] .+= actual.shift
            @test isapprox(reconstructed, actual.matrix; atol=2e-15, rtol=2e-15)

            physical_eigenvalues = eigvals(Hermitian(physical_dense))
            @test first(physical_eigenvalues) >=
                  local_physical.spectral_lower_bound - 1e-12
            @test last(physical_eigenvalues) <=
                  local_physical.spectral_upper_bound + 1e-12
            @test last(physical_eigenvalues) - first(physical_eigenvalues) <=
                  local_physical.spectral_width_bound + 1e-12

            local_algorithm = build_local_heis_1d(
                num_qubits,
                coefficients;
                kwargs...,
                coordinate_frame=:algorithm,
                rescaling_factor=actual.rescaling_factor,
                shift=actual.shift,
                scale_provenance=:exact_dense,
            )
            @test local_algorithm.coordinate_frame == :algorithm
            @test isapprox(local_algorithm.rescaling_factor, actual.rescaling_factor;
                           atol=2e-15, rtol=2e-15)
            @test isapprox(local_algorithm.global_shift, actual.shift;
                           atol=2e-15, rtol=2e-15)
            @test local_algorithm.scale_provenance == :exact_dense
            @test isapprox(
                Matrix(materialize_local_hamiltonian(local_algorithm)),
                actual.matrix;
                atol=2e-15,
                rtol=2e-15,
            )
            @test minimum(actual.eigvals) >=
                  local_algorithm.spectral_lower_bound - 1e-12
            @test maximum(actual.eigvals) <=
                  local_algorithm.spectral_upper_bound + 1e-12
        end
    end

    @testset "O(N) storage, explicit wraps, and frame validation" begin
        num_sites = 200
        open_hamiltonian = build_local_heis_1d(
            num_sites, [1.0, 1.0, 1.0]; seed=7, periodic=false)
        periodic_hamiltonian = build_local_heis_1d(
            num_sites, [1.0, 1.0, 1.0]; seed=7, periodic=true)
        @test length(open_hamiltonian.terms) == 5num_sites - 4
        @test length(periodic_hamiltonian.terms) == 5num_sites
        @test count(term -> term.wraps_boundary, open_hamiltonian.terms) == 0
        @test count(term -> term.wraps_boundary, periodic_hamiltonian.terms) == 4
        @test all(size(term.matrix, 1) <= 4 for term in periodic_hamiltonian.terms)
        @test !hasfield(typeof(open_hamiltonian), :matrix)
        @test !hasfield(typeof(open_hamiltonian), :eigvals)

        @test_throws ArgumentError build_local_heis_1d(
            3, [1.0, 1.0, 1.0]; seed=1, coordinate_frame=:algorithm)
        @test_throws ArgumentError build_local_heis_1d(
            3, [1.0, 1.0, 1.0]; seed=1,
            coordinate_frame=:algorithm, rescaling_factor=10.0, shift=0.0)
        @test_throws ArgumentError build_local_heis_1d(
            3, [1.0, 1.0, 1.0]; seed=1,
            coordinate_frame=:physical, rescaling_factor=10.0)

        physical = build_local_heis_1d(
            4, [1.0, 1.0, 1.0]; seed=1, periodic=false)
        certified_R = 2physical.spectral_width_bound / 0.9
        certified_shift = -physical.spectral_lower_bound / certified_R
        algorithm = build_local_heis_1d(
            4,
            [1.0, 1.0, 1.0];
            seed=1,
            periodic=false,
            coordinate_frame=:algorithm,
            rescaling_factor=certified_R,
            shift=certified_shift,
            scale_provenance=:certified_local_interval_bound,
        )
        @test algorithm.spectral_lower_bound >= -1e-12
        @test algorithm.spectral_upper_bound <= 0.45 + 1e-12
        @test_throws ArgumentError build_local_heis_1d(
            4,
            [1.0, 1.0, 1.0];
            seed=1,
            periodic=false,
            coordinate_frame=:algorithm,
            rescaling_factor=1.0,
            shift=0.0,
            scale_provenance=:certified_local_interval_bound,
        )
    end

    @testset "Local Pauli sources preserve the 1/sqrt(3N) convention" begin
        num_sites = 4
        local_jumps = local_pauli_jumps_1d(num_sites)
        @test length(local_jumps) == 3num_sites
        @test all(isconcretetype(typeof(jump)) for jump in local_jumps)
        @test all(isapprox(jump.coefficient, inv(sqrt(3num_sites));
                           atol=2e-15, rtol=2e-15) for jump in local_jumps)
        @test [jump.sites[1] for jump in local_jumps] ==
              repeat(collect(1:num_sites), 3)

        dense_jumps = Matrix{ComplexF64}[
            Matrix(materialize_local_jump(jump)) for jump in local_jumps
        ]
        source_sum = zeros(ComplexF64, 2^num_sites, 2^num_sites)
        for jump in dense_jumps
            source_sum .+= jump' * jump
        end
        @test isapprox(source_sum, I; atol=2e-15, rtol=2e-15)

        rng = MersenneTwister(0x1D)
        basis, _ = qr(randn(rng, ComplexF64, 2^num_sites, 2^num_sites))
        basis_matrix = Matrix(basis)
        legacy_jumps = QuantumFurnace._jumps_in_basis(num_sites, basis_matrix)
        @test length(legacy_jumps) == length(local_jumps)
        for index in eachindex(local_jumps)
            @test isapprox(legacy_jumps[index].data, dense_jumps[index];
                           atol=2e-15, rtol=2e-15)
            @test isapprox(
                legacy_jumps[index].in_eigenbasis,
                basis_matrix' * dense_jumps[index] * basis_matrix;
                atol=2e-15,
                rtol=2e-15,
            )
            @test legacy_jumps[index].hermitian
        end
        @test !legacy_jumps[num_sites + 1].orthogonal
        @test legacy_jumps[1].orthogonal
    end

    @testset "DLL blocks fail closed" begin
        source = first(local_pauli_jumps_1d(4))
        gaussian = DLLGaussianFilter(2.0)
        block = LocalDLLBlock1D(source, gaussian)
        @test block.channels == (gaussian,)
        @test isconcretetype(typeof(block))

        shifted_1 = ShiftedSymmetricFilter(gaussian, 0.1, 1.0)
        shifted_2 = ShiftedSymmetricFilter(gaussian, 0.2, 1.0)
        shifted_block = LocalDLLBlock1D(source, (shifted_1, shifted_2))
        @test length(shifted_block.channels) == 2
        multi = DLLMultiChannelFilter([shifted_1, shifted_2], gaussian.beta)
        @test LocalDLLBlock1D(source, multi).channels == (shifted_1, shifted_2)
        nested_multi = DLLMultiChannelFilter([multi], gaussian.beta)
        @test LocalDLLBlock1D(source, (nested_multi,)).channels ==
              (shifted_1, shifted_2)

        @test_throws ArgumentError LocalDLLBlock1D(source, ())
        @test_throws ArgumentError LocalDLLBlock1D(source, (gaussian, gaussian))
        @test_throws ArgumentError LocalDLLBlock1D(source, GaussianFilter(0.5))
        @test_throws ArgumentError LocalDLLBlock1D(
            source, (DLLGaussianFilter(2.0), DLLGaussianFilter(3.0)))

        lowering = LocalJump1D(
            [1], ComplexF64[0 1; 0 0], 1.0, 4; boundary=:open)
        @test_throws ArgumentError LocalDLLBlock1D(lowering, gaussian)
        noncontiguous = LocalJump1D(
            [1, 3], kron(X, X), 1.0, 4; boundary=:open)
        @test_throws ArgumentError LocalDLLBlock1D(noncontiguous, gaussian)

        source_matrix = source.matrix
        block.source.matrix[1, 1] = 1.0e6
        block.source.sites[1] = 2
        @test isapprox(block.source.matrix, source_matrix; atol=2e-15, rtol=2e-15)
        @test block.source.sites == source.sites
        getfield(source, :_matrix) .= ComplexF64[0 1; 0 0]
        @test isapprox(block.source.matrix, source_matrix; atol=2e-15, rtol=2e-15)

        open_hamiltonian = build_local_heis_1d(
            4, [1.0, 1.0, 1.0]; seed=1, periodic=false)
        @test isnothing(validate_local_dll_tensor_network(
            open_hamiltonian, [block]))
        periodic_hamiltonian = build_local_heis_1d(
            4, [1.0, 1.0, 1.0]; seed=1, periodic=true)
        @test_throws ArgumentError validate_local_dll_tensor_network(
            periodic_hamiltonian, [block])
        wrong_size_block = LocalDLLBlock1D(
            first(local_pauli_jumps_1d(3)), gaussian)
        @test_throws ArgumentError validate_local_dll_tensor_network(
            open_hamiltonian, [wrong_size_block])

        corrupted_block = LocalDLLBlock1D(
            first(local_pauli_jumps_1d(4)), gaussian)
        getfield(getfield(corrupted_block, :source), :_matrix) .=
            ComplexF64[0 1; 0 0]
        @test_throws ArgumentError validate_local_dll_tensor_network(
            open_hamiltonian, [corrupted_block])

        corrupted_hamiltonian = build_local_heis_1d(
            4, [1.0, 1.0, 1.0]; seed=1, periodic=false)
        getfield(first(getfield(corrupted_hamiltonian, :_terms)), :_sites)[2] = 3
        @test_throws ArgumentError validate_local_dll_tensor_network(
            corrupted_hamiltonian, [block])
    end
end
