@testset "Exact DLL parent overlap" begin
    relative_error(actual, expected) =
        norm(actual - expected) / max(norm(expected), eps(Float64))

    function lindbladian_gap(lindbladian, tolerance)
        values = eigvals(lindbladian)
        rates = [-real(value) for value in values if real(value) < -tolerance]
        isempty(rates) && error("fixture has no decaying Lindbladian mode")
        return minimum(rates), count(value -> abs(value) <= tolerance, values),
               maximum(abs, imag.(values))
    end

    function scale_raw_coordinates(raw::NamedTuple, scale::Float64)
        disordering_coeffs = raw.disordering_coeffs === nothing ? nothing :
            [coefficients ./ scale for coefficients in raw.disordering_coeffs]
        return merge(raw, (;
            matrix=raw.matrix ./ scale,
            base_coeffs=raw.base_coeffs ./ scale,
            disordering_coeffs,
            eigvals=raw.eigvals ./ scale,
            nu_min=raw.nu_min / scale,
            shift=raw.shift / scale,
            rescaling_factor=raw.rescaling_factor * scale,
        ))
    end

    @testset "dense generator, direct formula, MPO, and Gibbs state" begin
        for filter_builder in (
            beta -> DLLGaussianFilter(beta),
            beta -> DLLMetropolisFilter(beta; S=2.0),
        )
            fixture = exact_dll_fixture(2, filter_builder)
            reference = build_dll_parent(
                fixture.config,
                fixture.local_hamiltonian,
                fixture.blocks,
                fixture.hamiltonian,
                fixture.jumps,
            )
            diagnostics = verify_dll_parent(reference)

            lindbladian_config = exact_dll_config(
                Lindbladian(),
                2,
                fixture.beta_algorithm,
                fixture.beta_phys,
                fixture.filter,
            )
            lindbladian = construct_lindbladian(
                fixture.jumps, lindbladian_config, fixture.hamiltonian)
            generator_parent = materialize_kms_parent(
                lindbladian, fixture.hamiltonian.gibbs)
            @test isapprox(
                reference.dense.parent,
                generator_parent;
                atol=3e-12,
                rtol=3e-12,
            )

            reconstructed = QFITensors.mpo_to_dense(
                reference.bundle.total_parent, reference.sites)
            @test isapprox(
                reconstructed,
                reference.parent_fused;
                atol=3e-12,
                rtol=3e-12,
            )
            @test isapprox(
                sum(reference.block_parents_fused),
                reference.parent_fused;
                atol=3e-12,
                rtol=3e-12,
            )
            @test maximum(abs.(
                eigvals(Hermitian((reconstructed + reconstructed') / 2)) -
                reference.eigenvalues)) <= 3e-11

            rng = MersenneTwister(0x5a17)
            probe = randn(rng, ComplexF64, size(reference.parent_fused, 1))
            probe_mps = QFITensors.dense_to_mps(probe, reference.sites)
            image_mps = ITensorMPS.apply(
                reference.bundle.total_parent,
                probe_mps;
                cutoff=0.0,
                maxdim=size(reference.parent_fused, 1),
            )
            @test QFITensors.mps_to_dense(image_mps, reference.sites) ≈
                  reference.parent_fused * probe atol=2e-11 rtol=2e-11

            purification = prepare_gibbs_purification(reference)
            purification_vector = QFITensors.mps_to_dense(
                purification, reference.sites)
            @test isapprox(
                purification_vector,
                reference.gibbs_vector_fused;
                atol=3e-12,
                rtol=3e-12,
            )
            gibbs_computational = fixture.hamiltonian.eigvecs *
                Matrix(fixture.hamiltonian.gibbs) *
                adjoint(fixture.hamiltonian.eigvecs)
            @test isapprox(
                QFITensors.physical_partial_trace(
                    purification, reference.sites),
                gibbs_computational;
                atol=3e-12,
                rtol=3e-12,
            )

            gap_lindbladian, kernel_lindbladian, max_imaginary =
                lindbladian_gap(
                    lindbladian, reference.dense.spectrum.kernel_tolerance)
            @test kernel_lindbladian == diagnostics.observed_kernel_dimension
            @test isapprox(
                gap_lindbladian,
                reference.dense.spectrum.first_positive_eigenvalue;
                atol=3e-11,
                rtol=3e-10,
            )
            @test max_imaginary <= 3e-11
            @test diagnostics.kernel_complete
            @test diagnostics.kernel_evidence == :exact_dense_complete
            @test diagnostics.hermiticity_defect <= 3e-12
            @test diagnostics.minimum_energy >=
                  -diagnostics.kernel_tolerance
            @test diagnostics.gibbs_residual <= 3e-11
            @test maximum(diagnostics.block_gibbs_residuals) <= 3e-11
        end
    end

    @testset "multichannel additivity, no cross terms, and Loewner baseline" begin
        fixture = exact_dll_fixture(
            2, beta -> DLLMetropolisFilter(beta; S=2.0))
        epsilon = 0.25
        shift = 0.35
        baseline = dense_dll_parent(
            fixture.jumps, fixture.hamiltonian, fixture.filter)
        flexible_filter = dll_multichannel_translates(
            fixture.filter; centers=[shift], weights=[1.0])
        multichannel_filter = dll_multichannel_translates(
            fixture.filter;
            centers=[0.0, shift],
            weights=[epsilon, 1.0],
        )
        flexible = dense_dll_parent(
            fixture.jumps, fixture.hamiltonian, flexible_filter)
        multichannel_config = exact_dll_config(
            TensorNetworkSpectrum(),
            2,
            fixture.beta_algorithm,
            fixture.beta_phys,
            multichannel_filter,
        )
        reference = build_dll_parent(
            multichannel_config,
            fixture.local_hamiltonian,
            exact_dll_blocks(2, multichannel_filter),
            fixture.hamiltonian,
            fixture.jumps,
        )

        expected = epsilon .* baseline.parent .+ flexible.parent
        @test isapprox(
            reference.dense.parent, expected; atol=4e-12, rtol=4e-12)
        @test isapprox(
            sum(block.parent for block in reference.dense.blocks),
            reference.dense.parent;
            atol=4e-12,
            rtol=4e-12,
        )
        @test length(reference.dense.blocks) == 2 * length(fixture.jumps)
        @test reference.bundle.block_keys == [
            (source, channel)
            for source in eachindex(fixture.jumps)
            for channel in 1:2
        ]

        first_source_blocks = reference.dense.blocks[1:2]
        q_sum = first_source_blocks[1].Q + first_source_blocks[2].Q
        separated_transition =
            kron(conj(first_source_blocks[1].Q), first_source_blocks[1].Q) +
            kron(conj(first_source_blocks[2].Q), first_source_blocks[2].Q)
        crossed_transition = kron(conj(q_sum), q_sum)
        @test relative_error(crossed_transition, separated_transition) > 1e-3

        loewner_remainder = Hermitian(
            (reference.dense.parent - epsilon .* baseline.parent +
             adjoint(reference.dense.parent - epsilon .* baseline.parent)) / 2)
        @test minimum(eigvals(loewner_remainder)) >= -5e-11
        @test reference.dense.spectrum.kernel_count == baseline.spectrum.kernel_count
        @test reference.dense.spectrum.first_positive_eigenvalue + 5e-11 >=
              epsilon * baseline.spectrum.first_positive_eigenvalue
    end

    @testset "source normalisation and Hamiltonian-frame transformations" begin
        fixture = exact_dll_fixture(2, beta -> DLLGaussianFilter(beta))
        normalized = build_dll_parent(
            fixture.config,
            fixture.local_hamiltonian,
            fixture.blocks,
            fixture.hamiltonian,
            fixture.jumps,
        )
        unnormalized_jumps = exact_dll_jumps(
            fixture.hamiltonian, 2; normalized=false)
        unnormalized_blocks = exact_dll_blocks(
            2, fixture.filter; normalized=false)
        unnormalized = build_dll_parent(
            fixture.config,
            fixture.local_hamiltonian,
            unnormalized_blocks,
            fixture.hamiltonian,
            unnormalized_jumps,
        )
        expected_factor = 3 * 2
        @test isapprox(
            unnormalized.dense.parent,
            expected_factor .* normalized.dense.parent;
            atol=4e-11,
            rtol=4e-11,
        )
        @test unnormalized.dense.spectrum.first_positive_eigenvalue /
              normalized.dense.spectrum.first_positive_eigenvalue ≈
              expected_factor atol=3e-9 rtol=3e-10

        coordinate_scale = Float64(fixture.hamiltonian.rescaling_factor)
        scaled_raw = scale_raw_coordinates(fixture.raw, coordinate_scale)
        scaled_beta = fixture.beta_algorithm * coordinate_scale
        scaled_hamiltonian = HamHam(scaled_raw, scaled_beta)
        scaled_jumps = exact_dll_jumps(scaled_hamiltonian, 2)
        scaled_local_hamiltonian = build_local_heis_1d(
            2,
            [1.0, 1.0, 1.0];
            seed=46,
            periodic=false,
            disorder_strength=0.1,
            coordinate_frame=:algorithm,
            rescaling_factor=scaled_hamiltonian.rescaling_factor,
            shift=scaled_hamiltonian.shift,
            scale_provenance=:exact_dense,
        )

        scaled_gaussian = DLLGaussianFilter(scaled_beta)
        scaled_gaussian_config = exact_dll_config(
            TensorNetworkSpectrum(),
            2,
            scaled_beta,
            fixture.beta_phys,
            scaled_gaussian,
        )
        gaussian_scaled_reference = build_dll_parent(
            scaled_gaussian_config,
            scaled_local_hamiltonian,
            exact_dll_blocks(2, scaled_gaussian),
            scaled_hamiltonian,
            scaled_jumps,
        )
        @test relative_error(
            gaussian_scaled_reference.dense.parent,
            normalized.dense.parent,
        ) <= 4e-11

        metropolis_support = 0.6
        metropolis = DLLMetropolisFilter(
            fixture.beta_algorithm; S=metropolis_support)
        metropolis_config = exact_dll_config(
            TensorNetworkSpectrum(),
            2,
            fixture.beta_algorithm,
            fixture.beta_phys,
            metropolis,
        )
        metropolis_reference = build_dll_parent(
            metropolis_config,
            fixture.local_hamiltonian,
            exact_dll_blocks(2, metropolis),
            fixture.hamiltonian,
            fixture.jumps,
        )
        scaled_metropolis = DLLMetropolisFilter(
            scaled_beta; S=metropolis_support / coordinate_scale)
        scaled_metropolis_config = exact_dll_config(
            TensorNetworkSpectrum(),
            2,
            scaled_beta,
            fixture.beta_phys,
            scaled_metropolis,
        )
        scaled_metropolis_reference = build_dll_parent(
            scaled_metropolis_config,
            scaled_local_hamiltonian,
            exact_dll_blocks(2, scaled_metropolis),
            scaled_hamiltonian,
            scaled_jumps,
        )
        @test relative_error(
            scaled_metropolis_reference.dense.parent,
            metropolis_reference.dense.parent,
        ) <= 4e-11

        wrong_support = DLLMetropolisFilter(
            scaled_beta; S=metropolis_support)
        wrong_support_parent = dense_dll_parent(
            scaled_jumps, scaled_hamiltonian, wrong_support)
        @test relative_error(
            wrong_support_parent.parent,
            metropolis_reference.dense.parent,
        ) > 1e-5
    end

    @testset "geometry and local/dense consistency fail closed" begin
        fixture = exact_dll_fixture(2, beta -> DLLGaussianFilter(beta))
        periodic_local = build_local_heis_1d(
            2,
            [1.0, 1.0, 1.0];
            seed=46,
            periodic=true,
            disorder_strength=0.1,
            coordinate_frame=:algorithm,
            rescaling_factor=fixture.hamiltonian.rescaling_factor,
            shift=fixture.hamiltonian.shift,
            scale_provenance=:exact_dense,
        )
        periodic_blocks = [
            LocalDLLBlock1D(source, fixture.filter)
            for source in local_pauli_jumps_1d(2; boundary=:periodic)
        ]
        @test_throws ArgumentError build_dll_parent(
            fixture.config,
            periodic_local,
            periodic_blocks,
            fixture.hamiltonian,
            fixture.jumps,
        )

        wrong_local = build_local_heis_1d(
            2,
            [1.0, 1.0, 1.0];
            seed=47,
            periodic=false,
            disorder_strength=0.1,
            coordinate_frame=:algorithm,
            rescaling_factor=fixture.hamiltonian.rescaling_factor,
            shift=fixture.hamiltonian.shift,
            scale_provenance=:exact_dense,
        )
        @test_throws ArgumentError build_dll_parent(
            fixture.config,
            wrong_local,
            fixture.blocks,
            fixture.hamiltonian,
            fixture.jumps,
        )
    end
end
