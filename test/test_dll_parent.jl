using Random: MersenneTwister

@testset "Exact dense DLL KMS parent" begin
    function _dll_parent_config(n::Int, beta::Float64, filter::AbstractFilter)
        return Config(;
            sim = Lindbladian(),
            domain = BohrDomain(),
            construction = DLL(),
            num_qubits = n,
            with_linear_combination = true,
            beta = beta,
            sigma = inv(beta),
            a = beta / 30,
            s = 0.4,
            num_energy_bits = 8,
            t0 = 2pi / (2^8 * 0.05),
            num_trotter_steps_per_t0 = 10,
            filter = filter,
        )
    end

    function _assert_dense_parent_matches_generator(sys, filter; atol=2e-12)
        beta = Float64(filter.beta)
        config = _dll_parent_config(length(sys.jumps) ÷ 3, beta, filter)
        direct = dense_dll_parent(sys.jumps, sys.ham, filter)
        lindbladian = construct_lindbladian(sys.jumps, config, sys.ham)
        reference = materialize_kms_parent(lindbladian, sys.gibbs)

        @test isapprox(direct.parent, reference; atol=atol, rtol=atol)
        block_sum = zeros(ComplexF64, size(direct.parent))
        for block in direct.blocks
            block_sum .+= block.parent
        end
        @test isapprox(block_sum, direct.parent; atol=atol, rtol=atol)

        spectrum = direct.spectrum
        reference_values = eigvals(Hermitian((reference + reference') / 2))
        reference_kernel_count = count(
            value -> abs(value) <= spectrum.kernel_tolerance,
            reference_values,
        )
        reference_positive = reference_values[
            reference_values .> spectrum.kernel_tolerance]
        @test spectrum.kernel_count == reference_kernel_count
        @test spectrum.first_positive_eigenvalue !== nothing
        @test isapprox(
            spectrum.first_positive_eigenvalue,
            minimum(reference_positive);
            atol=atol,
            rtol=atol,
        )
        @test spectrum.hermiticity_defect <= spectrum.hermiticity_tolerance
        @test spectrum.minimum_eigenvalue >= -spectrum.kernel_tolerance
        @test spectrum.gibbs_residual <= spectrum.kernel_tolerance
        return direct, reference
    end

    beta = 5.0
    sys = make_dll_n3_system(beta)

    @testset "Gaussian and compact Metropolis direct formulas" begin
        for channel in (
            DLLGaussianFilter(beta),
            DLLMetropolisFilter(beta; S=2.0),
        )
            direct, _ = _assert_dense_parent_matches_generator(sys, channel)
            @test length(direct.blocks) == length(sys.jumps)
            @test direct.irreducibility.finite_size_only
            @test direct.irreducibility.commutant_dimension == 1
            @test direct.irreducibility.is_irreducible
            @test direct.spectrum.kernel_count == 1
            @test direct.spectrum.primitivity_established
        end
    end

    @testset "complex Pauli Y source block" begin
        y_jump = sys.jumps[4]
        @test maximum(abs, imag.(y_jump.data)) > 0
        channel = DLLGaussianFilter(beta)
        block = dense_dll_parent_block(y_jump, sys.ham, channel)
        config = _dll_parent_config(3, beta, channel)
        reference = materialize_kms_parent(
            construct_lindbladian(JumpOp[y_jump], config, sys.ham),
            sys.gibbs,
        )
        @test isapprox(block.parent, reference; atol=2e-12, rtol=2e-12)
        @test isapprox(block.Q, block.Q'; atol=2e-12, rtol=2e-12)
        @test isapprox(block.N, block.N'; atol=2e-12, rtol=2e-12)
        @test maximum(abs, imag.(block.Q)) > 0

        # Pure Y is globally imaginary in this real Hamiltonian eigenbasis, so
        # it cannot distinguish kron(conj(Q), Q) from the swapped factor order.
        # X+Y is genuinely complex and locks QF's column-stacking convention.
        x_jump = sys.jumps[1]
        mixed_data = (x_jump.data + y_jump.data) / sqrt(2)
        mixed_eigen = (x_jump.in_eigenbasis + y_jump.in_eigenbasis) / sqrt(2)
        mixed_jump = JumpOp(mixed_data, mixed_eigen, false, true)
        mixed_block = dense_dll_parent_block(mixed_jump, sys.ham, channel)
        mixed_reference = materialize_kms_parent(
            construct_lindbladian(JumpOp[mixed_jump], config, sys.ham),
            sys.gibbs,
        )
        @test isapprox(
            mixed_block.parent, mixed_reference; atol=2e-12, rtol=2e-12)

        rng = MersenneTwister(0xC01157AC)
        probe = randn(rng, ComplexF64, size(mixed_block.Q))
        expected_sandwich = vec(mixed_block.Q * probe * mixed_block.Q')
        column_stacked = kron(conj(mixed_block.Q), mixed_block.Q) * vec(probe)
        swapped = kron(mixed_block.Q, conj(mixed_block.Q)) * vec(probe)
        @test isapprox(column_stacked, expected_sandwich; atol=2e-12, rtol=2e-12)
        @test norm(swapped - expected_sandwich) / norm(expected_sandwich) > 0.1
    end

    @testset "shifted multichannel blocks remain separate" begin
        base = DLLMetropolisFilter(beta; S=2.0)
        multichannel = dll_multichannel_translates(
            base;
            centers=[0.0, 0.35],
            weights=[0.25, 1.0],
        )
        direct, _ = _assert_dense_parent_matches_generator(sys, multichannel)
        @test length(direct.blocks) == 2 * length(sys.jumps)
        @test [(block.source_index, block.channel_index) for block in direct.blocks] ==
              [(source, channel) for source in eachindex(sys.jumps) for channel in 1:2]

        channel_parents = [
            dense_dll_parent(sys.jumps, sys.ham, channel).parent
            for channel in multichannel.channels
        ]
        @test isapprox(
            direct.parent,
            channel_parents[1] + channel_parents[2];
            atol=2e-12,
            rtol=2e-12,
        )
        @test_throws ArgumentError dense_dll_parent_block(
            first(sys.jumps), sys.ham, multichannel)
    end

    @testset "heterogeneous and mixed-precision channels" begin
        heterogeneous = DLLMultiChannelFilter(
            AbstractFilter[
                DLLGaussianFilter(beta),
                DLLMetropolisFilter(beta; S=2.0),
            ],
            beta,
        )
        heterogeneous_parent, _ = _assert_dense_parent_matches_generator(
            sys, heterogeneous)
        @test heterogeneous_parent.filter === heterogeneous
        @test length(heterogeneous_parent.blocks) == 2 * length(sys.jumps)

        beta32 = Float32(beta)
        channels32 = [
            DLLGaussianFilter(beta32),
            DLLGaussianFilter(beta32),
        ]
        multichannel32 = DLLMultiChannelFilter(channels32, beta32)
        mixed_precision_parent, _ = _assert_dense_parent_matches_generator(
            sys, multichannel32; atol=2e-6)
        coherent_sum = dll_coherent_op_bohr(
            sys.jumps, sys.ham, channels32[1], beta)
        coherent_sum .+= dll_coherent_op_bohr(
            sys.jumps, sys.ham, channels32[2], beta)
        @test isapprox(
            mixed_precision_parent.coherent,
            coherent_sum;
            atol=2e-12,
            rtol=2e-12,
        )
        @test length(mixed_precision_parent.blocks) == 2 * length(sys.jumps)

        heterogeneous_precision = DLLMultiChannelFilter(
            AbstractFilter[
                DLLGaussianFilter(beta32),
                DLLMetropolisFilter(beta; S=2.0),
            ],
            beta,
        )
        heterogeneous_precision_parent, _ =
            _assert_dense_parent_matches_generator(
                sys, heterogeneous_precision; atol=2e-6)
        @test heterogeneous_precision_parent.spectrum.kernel_tolerance >=
              100 * size(heterogeneous_precision_parent.parent, 1) * eps(Float32)
        @test heterogeneous_precision_parent.spectrum.kernel_count == 1

        shifted_precision = ShiftedSymmetricFilter(
            DLLGaussianFilter(beta32), 0.35, 1.0)
        shifted_precision_parent, _ = _assert_dense_parent_matches_generator(
            sys, shifted_precision; atol=2e-6)
        @test shifted_precision_parent.spectrum.kernel_tolerance >=
              100 * size(shifted_precision_parent.parent, 1) * eps(Float32)
    end

    @testset "nonprimitive filtered dynamics has a multidimensional kernel" begin
        one_qubit_ham = HamHam(
            Vector{Vector{Matrix{ComplexF64}}}([[Z]]),
            [1.0],
            Int64(1),
            2.0;
            periodic=false,
            hermitian_check=true,
        )
        z_source = Matrix{ComplexF64}(Z)
        z_eigen = one_qubit_ham.eigvecs' * z_source * one_qubit_ham.eigvecs
        z_jump = JumpOp(z_source, z_eigen, true, true)
        channel = DLLGaussianFilter(2.0)
        direct = dense_dll_parent([z_jump], one_qubit_ham, channel)
        reference = materialize_kms_parent(
            construct_lindbladian(
                JumpOp[z_jump],
                _dll_parent_config(1, 2.0, channel),
                one_qubit_ham,
            ),
            one_qubit_ham.gibbs,
        )

        @test isapprox(direct.parent, reference; atol=2e-12, rtol=2e-12)
        @test direct.irreducibility.commutant_dimension == 2
        @test !direct.irreducibility.is_irreducible
        @test direct.spectrum.kernel_count == 2
        @test direct.spectrum.first_positive_eigenvalue !== nothing
        @test abs(direct.spectrum.eigenvalues[2]) <=
              direct.spectrum.kernel_tolerance
        @test !direct.spectrum.primitivity_established
    end

    @testset "input validation" begin
        @test_throws ArgumentError dense_dll_parent(
            sys.jumps, sys.ham, GaussianFilter(0.5))
        @test_throws ArgumentError dense_dll_parent(
            sys.jumps, sys.ham, DLLGaussianFilter(beta + 1))

        source = first(sys.jumps)
        nonhermitian = JumpOp(
            source.data,
            source.in_eigenbasis,
            source.orthogonal,
            false,
        )
        @test_throws ArgumentError dense_dll_parent_block(
            nonhermitian, sys.ham, DLLGaussianFilter(beta))
    end
end
