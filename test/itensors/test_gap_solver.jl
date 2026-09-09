@testset "Task 10 unrestricted multi-start finite-patch spectrum" begin
    fixture = exact_dll_fixture(
        2, beta -> DLLGaussianFilter(beta); beta_phys=0.5)
    loose_controls = task8_parent_controls(
        1; scalar_tolerance=1e-7, cutoff=1e-14, maxdim=64)
    tight_controls = task8_parent_controls(
        1; scalar_tolerance=1e-9, cutoff=1e-16, maxdim=128)
    sites = QFITensors.fused_siteinds(2)
    parent = build_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        loose_controls;
        sites,
        diagnostic_probe_count=0,
    )
    tighter_reference = QFITensors.build_exact_finite_patch_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        tight_controls;
        sites,
        factorization_controls=
            QFITensors.ExactPatchMPOFactorizationControls(
                cutoff=1e-14, maxdim=32),
        diagnostic_probe_count=0,
    )
    tighter_parent = tighter_reference.parent
    gibbs_controls = task9_controls(
        fixture.beta_algorithm; sweeps=2, maxdim_schedule=(4,))
    preparation = prepare_gibbs_purification(
        fixture.config,
        fixture.local_hamiltonian,
        gibbs_controls;
        sites,
    )
    gap_controls = ParentGapControls(
        sweeps=8,
        cutoff=1e-13,
        maxdim_schedule=(2, 4),
        num_starts=3,
        random_seed=911,
        residual_tolerance=2e-5,
        overlap_tolerance=2e-6,
        energy_tolerance=3e-6,
        penalty_factors=(0.5, 2.0),
        sectors=(:unrestricted,),
    )
    attempt_sink = QFITensors.ParentLowEnergyAttempt{Float64}[]
    result = solve_dll_parent_gap(
        fixture.config,
        parent,
        preparation;
        tighter_parent=tighter_reference,
        controls=gap_controls,
        max_levels=4,
        zero_mode_tolerance=3e-6,
        dense_overlap=true,
        _attempt_sink=attempt_sink,
    )
    exact_nominal_reference =
        QFITensors.build_exact_finite_patch_dll_parent(
            fixture.config,
            fixture.local_hamiltonian,
            fixture.blocks,
            tight_controls;
            sites,
            factorization_controls=
                QFITensors.ExactPatchMPOFactorizationControls(
                    cutoff=1e-12, maxdim=16),
            diagnostic_probe_count=0,
        )
    exact_result = solve_dll_parent_gap(
        fixture.config,
        exact_nominal_reference,
        preparation;
        tighter_parent=tighter_reference,
        controls=gap_controls,
        max_levels=4,
        zero_mode_tolerance=3e-6,
        dense_overlap=true,
    )
    @test exact_result.gap_value ≈ result.gap_value atol=3e-6 rtol=3e-6
    metadata = result.metadata

    @test result.gap_label == :observed_manifold_spacing
    @test result.target_label == :finite_patch_bohr_surrogate
    @test metadata.unrestricted_all_sectors
    @test metadata.observed_kernel_dimension == 1
    @test metadata.observed_manifold_dimension == 1
    @test length(metadata.absolute_levels) == 2
    @test metadata.absolute_levels[1] >= -3e-6
    @test metadata.observed_manifold_spacing ≈
          metadata.absolute_levels[2] - metadata.absolute_levels[1] atol=3e-6
    @test result.gap_value ≈ metadata.observed_manifold_spacing atol=3e-6
    @test metadata.dense_overlap_maximum_error <= 3e-6
    @test maximum(metadata.tighter_residuals) <=
          gap_controls.residual_tolerance
    @test length(metadata.multi_start_spreads) == 2
    @test all(spread -> spread <= gap_controls.energy_tolerance,
              metadata.multi_start_spreads)
    @test metadata.requested_bond_dimensions == [2, 4]
    @test metadata.maximum_exact_bond_dimension == 4
    @test metadata.bond_dimensions_scanned == [2, 4]
    @test metadata.converged_bond_dimensions == [4]
    @test isempty(metadata.bond_convergence_spreads)
    @test attempt_sink == metadata.attempts
    @test all(state -> state.sector == :unrestricted,
              result.low_energy_states)
    @test all(state -> state.converged, result.low_energy_states)
    @test count(attempt -> attempt.stage == :penalty_scan,
                metadata.attempts) >= 6
    @test any(attempt ->
        attempt.stage == :penalty_scan &&
        attempt.penalty_factor < 1 &&
        attempt.rejection_reason == :prior_state_leakage,
        metadata.attempts,
    )
    @test all(attempt ->
        !attempt.accepted ||
        (attempt.energy !== nothing &&
         attempt.residual <= gap_controls.residual_tolerance &&
         attempt.tighter_residual <= gap_controls.residual_tolerance),
        metadata.attempts,
    )

    dense_parent = QFITensors.mpo_to_dense(
        parent.bundle.total_parent, parent.sites)
    dense_levels = eigvals(Hermitian((dense_parent + dense_parent') / 2))
    @test metadata.absolute_levels ≈ dense_levels[1:2] atol=3e-6 rtol=3e-6
    exact = build_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        fixture.hamiltonian,
        fixture.jumps;
        sites,
    )
    @test result.gap_value ≈
          exact.dense.spectrum.first_positive_eigenvalue atol=3e-6 rtol=3e-6
    lindbladian_config = exact_dll_config(
        Lindbladian(),
        2,
        fixture.beta_algorithm,
        fixture.beta_phys,
        fixture.filter,
    )
    krylov = krylov_spectral_gap(
        lindbladian_config,
        fixture.hamiltonian,
        fixture.jumps;
        krylovdim=20,
        howmany=4,
        tol=1e-10,
    )
    @test result.gap_value ≈ krylov.spectral_gap atol=3e-6 rtol=3e-6
    for index in eachindex(result.low_energy_states)
        state = result.low_energy_states[index].state
        nominal = QFITensors._fp_sum_block_action(
            parent.bundle.block_parents, state, Float64)
        tighter = QFITensors._fp_sum_block_action(
            tighter_parent.bundle.block_parents, state, Float64)
        @test real(nominal.energy) ≈ metadata.absolute_levels[index] atol=3e-6
        @test nominal.residual ≈ result.low_energy_states[index].residual atol=1e-10
        @test real(tighter.energy) ≈ metadata.tighter_energies[index] atol=3e-6
        @test tighter.residual ≈ metadata.tighter_residuals[index] atol=1e-10
    end
    excited_vector = QFITensors.mps_to_dense(
        result.low_energy_states[2].state, parent.sites)
    tighter_action_norm = norm(
        QFITensors.mpo_to_dense(
            tighter_parent.bundle.total_parent, tighter_parent.sites) *
        excited_vector,
    )
    @test tighter_action_norm > 100 * metadata.tighter_residuals[2]

    duplicate = result.low_energy_states[1].state
    @test_throws ErrorException QFITensors._fp_complete_gram_basis(
        [duplicate, copy(duplicate)], gap_controls, 8, Float64)
    bad_residual = QFITensors._fp_rejection_reason(
        (energy=ComplexF64(0.2), residual=1e-2),
        (energy=ComplexF64(0.2), residual=1e-2),
        Float64[],
        gap_controls,
        3e-6,
    )
    @test bad_residual == :residual

    @test_throws ArgumentError solve_dll_parent_gap(
        fixture.config, parent, preparation;
        tighter_parent,
        controls=ParentGapControls(
            num_starts=3,
            maxdim_schedule=(2, 4),
            penalty_factors=(0.5, 2.0),
            sectors=(:even, :odd),
        ),
    )
    @test_throws ArgumentError solve_dll_parent_gap(
        fixture.config, parent, preparation;
        tighter_parent,
        controls=ParentGapControls(
            num_starts=2,
            maxdim_schedule=(2, 4),
            penalty_factors=(0.5, 2.0),
        ),
    )
    @test_throws ArgumentError solve_dll_parent_gap(
        fixture.config, parent, preparation;
        tighter_parent,
        controls=ParentGapControls(
            num_starts=3,
            maxdim_schedule=(4, 8),
            penalty_factors=(0.5, 2.0),
        ),
    )
    @test_throws ArgumentError solve_dll_parent_gap(
        fixture.config, parent, preparation;
        tighter_parent,
        controls=ParentGapControls(
            num_starts=3,
            maxdim_schedule=(8,),
            penalty_factors=(0.5, 2.0),
        ),
    )
    @test_throws ArgumentError solve_dll_parent_gap(
        fixture.config, parent, preparation;
        tighter_parent,
        controls=ParentGapControls(
            num_starts=3,
            maxdim_schedule=(2, 4),
            penalty_factors=(1.5, 2.0),
        ),
    )
    @test_throws ArgumentError solve_dll_parent_gap(
        fixture.config, parent, preparation;
        tighter_parent=parent,
        controls=gap_controls,
    )
    low_bond_controls = ParentGapControls(
        sweeps=1,
        cutoff=1e-6,
        maxdim_schedule=(1, 2),
        num_starts=3,
        random_seed=1411,
        residual_tolerance=1e-8,
        overlap_tolerance=1e-8,
        energy_tolerance=1e-8,
        penalty_factors=(0.5, 2.0),
        sectors=(:unrestricted,),
    )
    failure_attempts = QFITensors.ParentLowEnergyAttempt{Float64}[]
    @test_throws ErrorException solve_dll_parent_gap(
        fixture.config, parent, preparation;
        tighter_parent,
        controls=low_bond_controls,
        _attempt_sink=failure_attempts,
    )
    @test !isempty(failure_attempts)
    @test any(attempt -> !attempt.accepted, failure_attempts)
end

@testset "Task 10 positive-baseline surrogate levels" begin
    fixture = exact_dll_fixture(
        3, beta -> DLLGaussianFilter(beta); beta_phys=0.5)
    sites = QFITensors.fused_siteinds(3)
    parent = build_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        task8_parent_controls(
            1; scalar_tolerance=1e-7, cutoff=1e-14, maxdim=64);
        sites,
        diagnostic_probe_count=0,
    )
    tighter_parent = build_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        task8_parent_controls(
            1; scalar_tolerance=1e-9, cutoff=1e-16, maxdim=128);
        sites,
        diagnostic_probe_count=0,
    )
    preparation = prepare_gibbs_purification(
        fixture.config,
        fixture.local_hamiltonian,
        task9_controls(
            fixture.beta_algorithm; sweeps=4, maxdim_schedule=(4,));
        sites,
    )
    controls = ParentGapControls(
        sweeps=8,
        cutoff=1e-13,
        maxdim_schedule=(2, 4),
        num_starts=3,
        random_seed=1231,
        residual_tolerance=3e-5,
        overlap_tolerance=3e-6,
        energy_tolerance=5e-6,
        penalty_factors=(0.5, 2.0),
        sectors=(:unrestricted,),
    )
    result = solve_dll_parent_gap(
        fixture.config,
        parent,
        preparation;
        tighter_parent,
        controls,
        max_levels=3,
        zero_mode_tolerance=5e-6,
        dense_overlap=true,
    )
    metadata = result.metadata

    @test metadata.observed_kernel_dimension == 0
    @test metadata.observed_manifold_dimension == 1
    @test result.diagnostics.observed_kernel_dimension == 0
    @test result.diagnostics.observed_manifold_dimension == 1
    @test metadata.absolute_levels[1] > 5e-6
    @test metadata.absolute_levels ≈
          metadata.dense_overlap_levels atol=5e-6 rtol=5e-6
    @test result.gap_value ≈
          metadata.absolute_levels[2] - metadata.absolute_levels[1] atol=5e-6
    @test result.gap_lower === nothing
    @test result.gap_upper === nothing
    @test result.gap_label == :observed_manifold_spacing
end
