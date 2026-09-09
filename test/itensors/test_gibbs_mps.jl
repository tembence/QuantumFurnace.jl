function task9_controls(
    beta_frame::Real;
    sweeps::Int=4,
    cutoff::Real=1e-13,
    maxdim_schedule::Tuple=(4, 8, 16),
)
    return GibbsMPSControls(;
        time_step=max(float(beta_frame) / (2 * sweeps), 1e-3),
        sweeps=sweeps,
        cutoff=cutoff,
        maxdim_schedule=maxdim_schedule,
    )
end

@testset "Task 9 beta-zero fused Bell purification" begin
    n = 3
    hamiltonian = build_local_heis_1d(
        n,
        [1.0, 1.0, 1.0];
        seed=46,
        periodic=false,
        disorder_strength=0.1,
    )
    controls = GibbsMPSControls(
        time_step=0.1,
        sweeps=2,
        cutoff=1e-13,
        maxdim_schedule=(1,),
    )
    purification = prepare_gibbs_purification(
        hamiltonian, 0.0, controls)
    expected_matrix = Matrix{ComplexF64}(I, 2^n, 2^n) / sqrt(2.0^n)
    expected_vector = QFITensors.matrix_to_fused(expected_matrix, n)
    actual_vector = QFITensors.mps_to_dense(
        purification.state, purification.sites)

    @test purification isa QFITensors.GibbsPurificationMPS
    @test actual_vector ≈ expected_vector atol=3e-14 rtol=3e-14
    @test purification.diagnostics.preparation_method == :fused_bell
    @test purification.diagnostics.beta_phys ≈ 0.0 atol=eps(Float64)
    @test purification.diagnostics.beta_alg ≈ 0.0 atol=eps(Float64)
    @test purification.diagnostics.coordinate_rescaling_factor ≈
          1.0 atol=eps(Float64)
    @test purification.diagnostics.actual_time_step ≈
          0.0 atol=eps(Float64)
    @test purification.diagnostics.sweeps == 0
    @test purification.diagnostics.cutoff === nothing
    @test isempty(purification.diagnostics.maxdim_schedule)
    @test purification.diagnostics.maximum_bond_dimension == 1
    @test purification.diagnostics.state_norm ≈ 1.0 atol=3e-14
    @test purification.diagnostics.dense_verification !== nothing
    @test purification.diagnostics.dense_verification.vector_error <= 3e-14
    @test purification.diagnostics.dense_verification.gibbs_trace_distance <=
          3e-14
    @test purification.diagnostics.maximum_local_hermiticity_defect <= 3e-14
    @test purification.diagnostics.maximum_local_trace_error <= 3e-14
    for rho in purification.diagnostics.one_site_reduced_states
        @test rho ≈ Matrix{ComplexF64}(I, 2, 2) / 2 atol=3e-14 rtol=3e-14
    end
    @test maximum(abs, purification.diagnostics.probe_expectations) <= 3e-14
    @test maximum(
        abs, purification.diagnostics.nearest_neighbor_probe_correlations) <=
          3e-14
end

@testset "Task 9 dense Gibbs, identity gauge, and frame agreement" begin
    fixture = exact_dll_fixture(
        3, beta -> DLLGaussianFilter(beta); beta_phys=0.5)
    physical_hamiltonian = build_local_heis_1d(
        3,
        [1.0, 1.0, 1.0];
        seed=46,
        periodic=false,
        disorder_strength=0.1,
    )
    sites = QFITensors.fused_siteinds(3)
    controls = task9_controls(
        fixture.beta_algorithm;
        sweeps=2,
        maxdim_schedule=(4,),
    )
    physical = prepare_gibbs_purification(
        fixture.config,
        physical_hamiltonian,
        controls;
        sites,
    )
    algorithm = prepare_gibbs_purification(
        fixture.config,
        fixture.local_hamiltonian,
        controls;
        sites,
    )
    algorithm_direct = prepare_gibbs_purification(
        fixture.local_hamiltonian,
        fixture.beta_algorithm,
        controls;
        sites,
    )
    overlap = abs(ITensorMPS.inner(physical.state, algorithm.state))

    @test physical.diagnostics.beta_frame ≈ fixture.beta_phys rtol=2e-14
    @test algorithm.diagnostics.beta_frame ≈ fixture.beta_algorithm rtol=2e-14
    @test physical.diagnostics.beta_phys ≈ fixture.beta_phys rtol=2e-14
    @test physical.diagnostics.beta_alg ≈ fixture.beta_algorithm rtol=2e-14
    @test algorithm.diagnostics.beta_phys ≈ fixture.beta_phys rtol=2e-14
    @test algorithm.diagnostics.beta_alg ≈ fixture.beta_algorithm rtol=2e-14
    @test algorithm_direct.diagnostics.beta_phys ≈ fixture.beta_phys rtol=2e-14
    @test algorithm_direct.diagnostics.beta_alg ≈
          fixture.beta_algorithm rtol=2e-14
    @test physical.diagnostics.coordinate_rescaling_factor ≈
          fixture.local_hamiltonian.rescaling_factor rtol=2e-14
    @test algorithm.diagnostics.coordinate_rescaling_factor ≈
          fixture.local_hamiltonian.rescaling_factor rtol=2e-14
    @test algorithm_direct.diagnostics.coordinate_rescaling_factor ≈
          fixture.local_hamiltonian.rescaling_factor rtol=2e-14
    @test physical.diagnostics.hamiltonian_frame == :physical
    @test algorithm.diagnostics.hamiltonian_frame == :algorithm
    @test all(preparation -> preparation.diagnostics.preparation_method == :tdvp,
              (physical, algorithm, algorithm_direct))
    @test algorithm.diagnostics.sweeps == controls.sweeps
    @test algorithm.diagnostics.cutoff ≈ controls.cutoff rtol=2e-14
    @test algorithm.diagnostics.maxdim_schedule == fill(4, controls.sweeps)
    @test overlap ≈ 1.0 atol=3e-12 rtol=3e-12
    for preparation in (physical, algorithm)
        dense = preparation.diagnostics.dense_verification
        @test dense !== nothing
        @test dense.vector_error <= 3e-12
        @test dense.infidelity <= 3e-12
        @test dense.gibbs_trace_distance <= 3e-12
        @test dense.energy_error <= 3e-11
    end

    shifted_hamiltonian = LocalHamiltonian1D(
        physical_hamiltonian.terms;
        num_sites=physical_hamiltonian.num_sites,
        local_dim=physical_hamiltonian.local_dim,
        boundary=:open,
        coordinate_frame=:physical,
        global_shift=2.75,
        rescaling_factor=1.0,
        scale_provenance=:identity_physical_frame,
    )
    shifted = prepare_gibbs_purification(
        shifted_hamiltonian,
        fixture.beta_phys,
        controls;
        sites,
    )
    @test abs(ITensorMPS.inner(physical.state, shifted.state)) ≈
          1.0 atol=3e-12 rtol=3e-12
    @test shifted.diagnostics.energy - physical.diagnostics.energy ≈
          2.75 atol=3e-11 rtol=3e-11
end

@testset "Task 9 independent MPS control convergence" begin
    n = 5
    beta = 0.8
    hamiltonian = build_local_heis_1d(
        n,
        [1.0, 1.0, 1.0];
        seed=71,
        periodic=false,
        disorder_strength=0.1,
    )
    sites = QFITensors.fused_siteinds(n)
    strict_controls = task9_controls(
        beta;
        sweeps=8,
        cutoff=1e-13,
        maxdim_schedule=(4, 8, 16),
    )
    strict = prepare_gibbs_purification(
        hamiltonian, beta, strict_controls; sites)
    bond_limited_controls = task9_controls(
        beta;
        sweeps=8,
        cutoff=1e-13,
        maxdim_schedule=(2,),
    )
    bond_limited = prepare_gibbs_purification(
        hamiltonian,
        beta,
        bond_limited_controls;
        sites,
        reference=strict,
    )
    step_limited_controls = task9_controls(
        beta;
        sweeps=1,
        cutoff=1e-13,
        maxdim_schedule=(4,),
    )
    step_medium_controls = task9_controls(
        beta;
        sweeps=4,
        cutoff=1e-13,
        maxdim_schedule=(4,),
    )
    step_reference_controls = task9_controls(
        beta;
        sweeps=16,
        cutoff=1e-13,
        maxdim_schedule=(4,),
    )
    step_reference = prepare_gibbs_purification(
        hamiltonian, beta, step_reference_controls; sites)
    step_limited = prepare_gibbs_purification(
        hamiltonian,
        beta,
        step_limited_controls;
        sites,
        reference=step_reference,
    )
    step_medium = prepare_gibbs_purification(
        hamiltonian,
        beta,
        step_medium_controls;
        sites,
        reference=step_reference,
    )
    cutoff_limited_controls = task9_controls(
        beta;
        sweeps=8,
        cutoff=1e-2,
        maxdim_schedule=(4, 8, 16),
    )
    cutoff_limited = prepare_gibbs_purification(
        hamiltonian,
        beta,
        cutoff_limited_controls;
        sites,
        reference=strict,
    )

    @test strict.diagnostics.dense_verification === nothing
    @test strict.diagnostics.maximum_bond_dimension <= 16
    @test bond_limited.diagnostics.maximum_bond_dimension <= 2
    @test bond_limited.diagnostics.stricter_comparison !== nothing
    bond_comparison = bond_limited.diagnostics.stricter_comparison
    @test bond_comparison.infidelity > 1e-8
    @test bond_comparison.maximum_one_site_trace_distance > 1e-7
    @test step_limited.diagnostics.actual_time_step >
          step_medium.diagnostics.actual_time_step >
          step_reference.diagnostics.actual_time_step
    @test step_limited.diagnostics.stricter_comparison !== nothing
    @test step_medium.diagnostics.stricter_comparison !== nothing
    step_limited_comparison = step_limited.diagnostics.stricter_comparison
    step_medium_comparison = step_medium.diagnostics.stricter_comparison
    @test step_limited_comparison.infidelity >
          step_medium_comparison.infidelity
    @test step_limited_comparison.energy_difference >
          step_medium_comparison.energy_difference
    @test step_limited_comparison.maximum_one_site_trace_distance >
          step_medium_comparison.maximum_one_site_trace_distance
    @test cutoff_limited.diagnostics.cutoff > strict.diagnostics.cutoff
    @test cutoff_limited.diagnostics.stricter_comparison !== nothing
    @test cutoff_limited.diagnostics.stricter_comparison.infidelity > 1e-10
    @test strict.diagnostics.maximum_local_hermiticity_defect <= 2e-11
    @test strict.diagnostics.maximum_local_trace_error <= 2e-11
end

@testset "Task 9 Float32 precision preservation" begin
    n = 1
    matrix = ComplexF32[0.3 0.2im; -0.2im -0.1]
    hamiltonian = LocalHamiltonian1D(
        [LocalTerm1D([1], matrix, Float32(0.7), n; boundary=:open)];
        num_sites=n,
        local_dim=2,
        boundary=:open,
        coordinate_frame=:physical,
        global_shift=Float32(0.4),
        rescaling_factor=Float32(1),
        scale_provenance=:identity_physical_frame,
    )
    controls = GibbsMPSControls(
        time_step=Float32(0.1),
        sweeps=4,
        cutoff=Float32(1e-6),
        maxdim_schedule=(2,),
    )
    preparation = prepare_gibbs_purification(
        hamiltonian, Float32(0.6), controls)
    @test preparation isa QFITensors.GibbsPurificationMPS{Float32}
    @test preparation.diagnostics.preparation_method == :exact_one_site
    @test preparation.diagnostics.actual_time_step ≈ 0f0 atol=eps(Float32)
    @test preparation.diagnostics.sweeps == 0
    @test preparation.diagnostics.cutoff === nothing
    @test isempty(preparation.diagnostics.maxdim_schedule)
    @test all(tensor -> eltype(tensor) == ComplexF32, preparation.state)
    @test preparation.diagnostics.dense_verification.vector_error <= 3e-5
    @test preparation.diagnostics.dense_verification.gibbs_trace_distance <=
          3e-5
end

@testset "Task 9 complex Float32 TDVP preservation" begin
    n = 2
    x = ComplexF32[0 1; 1 0]
    y = ComplexF32[0 -im; im 0]
    z = ComplexF32[1 0; 0 -1]
    terms = LocalTerm1D{Float32}[
        LocalTerm1D([1], y, 0.7f0, n; boundary=:open),
        LocalTerm1D([2], x, -0.3f0, n; boundary=:open),
        LocalTerm1D([1, 2], kron(y, z), 0.4f0, n; boundary=:open),
    ]
    hamiltonian = LocalHamiltonian1D(
        terms;
        num_sites=n,
        local_dim=2,
        boundary=:open,
        coordinate_frame=:physical,
        global_shift=0.2f0,
        rescaling_factor=1f0,
        scale_provenance=:identity_physical_frame,
    )
    controls = GibbsMPSControls(
        time_step=0.075f0,
        sweeps=4,
        cutoff=1f-6,
        maxdim_schedule=(4,),
    )
    preparation = prepare_gibbs_purification(
        hamiltonian, 0.6f0, controls)

    @test preparation isa QFITensors.GibbsPurificationMPS{Float32}
    @test preparation.diagnostics.preparation_method == :tdvp
    @test preparation.diagnostics.maxdim_schedule == fill(4, 4)
    @test all(tensor -> eltype(tensor) == ComplexF32, preparation.state)
    @test preparation.diagnostics.dense_verification.vector_error <= 3f-5
    @test preparation.diagnostics.dense_verification.gibbs_trace_distance <=
          3f-5
end

@testset "Task 9 parent diagnostics remain post-preparation" begin
    fixture = exact_dll_fixture(
        2, beta -> DLLGaussianFilter(beta); beta_phys=0.5)
    controls = task9_controls(
        fixture.beta_algorithm;
        sweeps=2,
        maxdim_schedule=(4,),
    )
    sites = QFITensors.fused_siteinds(2)
    purification = prepare_gibbs_purification(
        fixture.config,
        fixture.local_hamiltonian,
        controls;
        sites,
    )
    loose_parent = build_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        task8_parent_controls(
            1; scalar_tolerance=1e-5, cutoff=1e-10, maxdim=32);
        sites,
        diagnostic_probe_count=0,
    )
    tight_parent = build_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        task8_parent_controls(
            1; scalar_tolerance=1e-8, cutoff=1e-16, maxdim=128);
        sites,
        diagnostic_probe_count=0,
    )
    diagnostics = QFITensors.gibbs_parent_diagnostics(
        purification, loose_parent; tighter_parent=tight_parent)
    exact_parent = build_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        fixture.hamiltonian,
        fixture.jumps;
        sites,
    )
    exact_diagnostics = QFITensors.gibbs_parent_diagnostics(
        purification, exact_parent)

    @test diagnostics.target_label == :finite_patch_bohr_surrogate
    @test length(diagnostics.block_residuals) == 6
    @test length(diagnostics.block_energies) == 6
    @test diagnostics.total_energy isa ComplexF64
    @test eltype(diagnostics.block_energies) == ComplexF64
    @test abs(imag(diagnostics.total_energy)) <= diagnostics.total_residual
    @test diagnostics.tighter_target_label == :finite_patch_bohr_surrogate
    @test diagnostics.tighter_total_residual !== nothing
    @test diagnostics.total_residual <= 2e-4
    @test diagnostics.tighter_total_residual <= 2e-6
    @test exact_diagnostics.target_label == :exact_bohr_dense
    @test exact_diagnostics.total_residual <= 3e-11
    @test maximum(exact_diagnostics.block_residuals) <= 3e-11
    @test_throws ArgumentError QFITensors.gibbs_parent_diagnostics(
        purification, loose_parent; tighter_parent=loose_parent)
    @test_throws ArgumentError QFITensors.gibbs_parent_diagnostics(
        purification, loose_parent; tighter_parent=exact_parent)

    different_radius_parent = build_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        task8_parent_controls(
            2; scalar_tolerance=1e-8, cutoff=1e-16, maxdim=128);
        sites,
        diagnostic_probe_count=0,
    )
    @test_throws ArgumentError QFITensors.gibbs_parent_diagnostics(
        purification, loose_parent; tighter_parent=different_radius_parent)

    changed_terms = fixture.local_hamiltonian.terms
    first_term = first(changed_terms)
    changed_terms[1] = LocalTerm1D(
        first_term.sites,
        first_term.matrix,
        first_term.coefficient * 1.01,
        first_term.num_sites;
        local_dim=first_term.local_dim,
        boundary=first_term.boundary,
    )
    changed_hamiltonian = LocalHamiltonian1D(
        changed_terms;
        num_sites=fixture.local_hamiltonian.num_sites,
        local_dim=fixture.local_hamiltonian.local_dim,
        boundary=fixture.local_hamiltonian.boundary,
        coordinate_frame=fixture.local_hamiltonian.coordinate_frame,
        global_shift=fixture.local_hamiltonian.global_shift,
        rescaling_factor=fixture.local_hamiltonian.rescaling_factor,
        scale_provenance=fixture.local_hamiltonian.scale_provenance,
    )
    changed_model_parent = build_dll_parent(
        fixture.config,
        changed_hamiltonian,
        fixture.blocks,
        task8_parent_controls(
            1; scalar_tolerance=1e-8, cutoff=1e-16, maxdim=128);
        sites,
        diagnostic_probe_count=0,
    )
    @test_throws ArgumentError QFITensors.gibbs_parent_diagnostics(
        purification, loose_parent; tighter_parent=changed_model_parent)
end

@testset "Task 9 refusal gates" begin
    hamiltonian = build_local_heis_1d(
        2,
        [1.0, 1.0, 1.0];
        seed=46,
        periodic=false,
        disorder_strength=0.1,
    )
    coarse_step = GibbsMPSControls(
        time_step=0.01,
        sweeps=2,
        cutoff=1e-12,
        maxdim_schedule=(4,),
    )
    @test_throws ArgumentError prepare_gibbs_purification(
        hamiltonian, 0.5, coarse_step)
    too_long_schedule = GibbsMPSControls(
        time_step=0.2,
        sweeps=1,
        cutoff=1e-12,
        maxdim_schedule=(2, 4),
    )
    @test_throws ArgumentError prepare_gibbs_purification(
        hamiltonian, 0.0, too_long_schedule)

    periodic = build_local_heis_1d(
        3,
        [1.0, 1.0, 1.0];
        seed=46,
        periodic=true,
        disorder_strength=0.1,
    )
    @test_throws ArgumentError prepare_gibbs_purification(
        periodic, 0.5, task9_controls(0.5; maxdim_schedule=(4,)))

    controls = task9_controls(0.5; maxdim_schedule=(4,))
    sites = QFITensors.fused_siteinds(2)
    reference = prepare_gibbs_purification(
        hamiltonian, 0.5, controls; sites)
    @test_throws ArgumentError prepare_gibbs_purification(
        hamiltonian, 0.5, controls; sites, reference)
    @test_throws DimensionMismatch prepare_gibbs_purification(
        hamiltonian,
        0.5,
        controls;
        sites=QFITensors.fused_siteinds(3),
    )

    early_coarse_reference_controls = GibbsMPSControls(
        time_step=0.125,
        sweeps=2,
        cutoff=1e-13,
        maxdim_schedule=(4, 16),
    )
    early_coarse_reference = prepare_gibbs_purification(
        hamiltonian, 0.5, early_coarse_reference_controls; sites)
    early_fine_candidate_controls = GibbsMPSControls(
        time_step=0.125,
        sweeps=2,
        cutoff=1e-8,
        maxdim_schedule=(8,),
    )
    @test_throws ArgumentError prepare_gibbs_purification(
        hamiltonian,
        0.5,
        early_fine_candidate_controls;
        sites,
        reference=early_coarse_reference,
    )

    strict_reference_controls = GibbsMPSControls(
        time_step=0.075,
        sweeps=4,
        cutoff=1e-13,
        maxdim_schedule=(4,),
    )
    different_beta_reference = prepare_gibbs_purification(
        hamiltonian, 0.6, strict_reference_controls; sites)
    @test_throws ArgumentError prepare_gibbs_purification(
        hamiltonian,
        0.5,
        controls;
        sites,
        reference=different_beta_reference,
    )

    different_hamiltonian = build_local_heis_1d(
        2,
        [1.0, 1.0, 1.0];
        seed=47,
        periodic=false,
        disorder_strength=0.1,
    )
    different_hamiltonian_reference = prepare_gibbs_purification(
        different_hamiltonian, 0.5, strict_reference_controls; sites)
    @test_throws ArgumentError prepare_gibbs_purification(
        hamiltonian,
        0.5,
        controls;
        sites,
        reference=different_hamiltonian_reference,
    )
end
