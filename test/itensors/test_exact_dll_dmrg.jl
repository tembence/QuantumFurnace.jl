@testset "Exact DLL penalty DMRG vertical slice" begin
    fixture = exact_dll_fixture(
        4, beta -> DLLMetropolisFilter(beta; S=2.0))
    reference = build_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        fixture.hamiltonian,
        fixture.jumps,
    )
    @test reference.dense.spectrum.kernel_count == 1
    @test reference.dense.spectrum.primitivity_established

    lindbladian_config = exact_dll_config(
        Lindbladian(),
        4,
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
        atol=3e-11,
        rtol=3e-11,
    )
    lindbladian_values = eigvals(lindbladian)
    decaying_rates = [
        -real(value) for value in lindbladian_values
        if real(value) < -reference.dense.spectrum.kernel_tolerance
    ]
    exact_gap = reference.dense.spectrum.first_positive_eigenvalue
    @test minimum(decaying_rates) ≈ exact_gap atol=3e-11 rtol=3e-10
    @test count(
        value -> abs(value) <= reference.dense.spectrum.kernel_tolerance,
        lindbladian_values,
    ) == reference.dense.spectrum.kernel_count
    @test maximum(abs, imag.(lindbladian_values)) <= 3e-11

    krylov = krylov_spectral_gap(
        lindbladian_config,
        fixture.hamiltonian,
        fixture.jumps;
        krylovdim=40,
        howmany=6,
        tol=1e-10,
    )
    @test krylov.spectral_gap ≈ exact_gap atol=3e-10 rtol=3e-9

    reconstructed = QFITensors.mpo_to_dense(
        reference.bundle.total_parent, reference.sites)
    @test isapprox(
        reconstructed,
        reference.parent_fused;
        atol=3e-11,
        rtol=3e-11,
    )
    rng = MersenneTwister(0x4d504f)
    probe = randn(rng, ComplexF64, size(reference.parent_fused, 1))
    probe_mps = QFITensors.dense_to_mps(probe, reference.sites)
    image_mps = ITensorMPS.apply(
        reference.bundle.total_parent,
        probe_mps;
        cutoff=0.0,
        maxdim=size(reference.parent_fused, 1),
    )
    @test isapprox(
        QFITensors.mps_to_dense(image_mps, reference.sites),
        reference.parent_fused * probe;
        atol=3e-11,
        rtol=3e-11,
    )
    purification = prepare_gibbs_purification(reference)
    gibbs_computational = fixture.hamiltonian.eigvecs *
        Matrix(fixture.hamiltonian.gibbs) *
        adjoint(fixture.hamiltonian.eigvecs)
    @test isapprox(
        QFITensors.physical_partial_trace(purification, reference.sites),
        gibbs_computational;
        atol=3e-11,
        rtol=3e-11,
    )
    @test reference.diagnostics.gibbs_residual <= 3e-11
    @test maximum(reference.diagnostics.block_gibbs_residuals) <= 3e-11

    default_excited = solve_dll_parent_gap(reference; start_index=4)
    @test default_excited.converged
    @test default_excited.energy ≈ exact_gap atol=1e-9 rtol=1e-9

    controls = ParentGapControls(
        sweeps=10,
        cutoff=1e-13,
        maxdim_schedule=(4, 8, 16, 32, 64, 128),
        num_starts=1,
        random_seed=307,
        residual_tolerance=2e-5,
        overlap_tolerance=2e-8,
        energy_tolerance=3e-7,
        penalty_factors=(2.0,),
        sectors=(:unrestricted,),
    )
    excited = solve_dll_parent_gap(
        reference; controls, penalty_factor=2.0)
    @test excited.converged
    @test excited.energy ≈ exact_gap atol=3e-7 rtol=3e-7
    @test excited.residual <= controls.residual_tolerance
    @test maximum(excited.kernel_overlaps) <= controls.overlap_tolerance
    @test excited.variance ≈ excited.residual^2 atol=2e-12 rtol=2e-12

    local_algorithm = fixture.local_hamiltonian
    @test Matrix(materialize_local_hamiltonian(local_algorithm)) ≈
          fixture.hamiltonian.data atol=2e-13 rtol=2e-13
    blocks = fixture.blocks
    exact_controls = BohrMPOControls(target_label=:exact_bohr_dense)
    provenance = dll_parent_provenance(
        fixture.config,
        local_algorithm,
        blocks;
        controls=exact_controls,
        include_sweep_clock=true,
    )
    result = DLLTensorNetworkResult(
        fixture.config,
        reference.bundle,
        [excited],
        reference.diagnostics,
        provenance;
        bohr_controls=exact_controls,
        gibbs_controls=GibbsMPSControls(),
        gap_controls=controls,
        gap_label=:exact_gap,
        gap_value=exact_gap,
        metadata=(; deterministic_fixture=:obc_heisenberg_n4_seed46),
    )
    @test result.gap_label == :exact_gap
    @test result.gap_value ≈ exact_gap rtol=1e-14
    @test result.provenance.sweep_clock.multiplier == 12

    @test_throws ErrorException solve_dll_parent_gap(
        reference; controls, penalty_factor=0.25, start_index=2)

    sector_controls = ParentGapControls(
        num_starts=1, sectors=(:even, :odd))
    @test_throws ArgumentError solve_dll_parent_gap(
        reference; controls=sector_controls)
    multistart_controls = ParentGapControls(
        num_starts=2, sectors=(:unrestricted,))
    @test_throws ArgumentError solve_dll_parent_gap(
        reference; controls=multistart_controls)

    underconverged_controls = ParentGapControls(
        sweeps=1,
        cutoff=1e-4,
        maxdim_schedule=(1,),
        num_starts=1,
        random_seed=617,
        residual_tolerance=2e-5,
        overlap_tolerance=2e-8,
        energy_tolerance=3e-7,
        penalty_factors=(2.0,),
        sectors=(:unrestricted,),
    )
    @test_throws ErrorException solve_dll_parent_gap(
        reference;
        controls=underconverged_controls,
        penalty_factor=2.0,
        start_index=3,
    )
end
