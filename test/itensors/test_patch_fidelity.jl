function task10f_factor_controls(; cutoff=1e-13, maxdim=32)
    return QFITensors.ExactPatchMPOFactorizationControls(;
        cutoff, maxdim)
end

function task10f_dense_levels(parent)
    matrix = QFITensors.mpo_to_dense(
        parent.bundle.total_parent, parent.sites)
    decomposition = eigen(Hermitian((matrix + matrix') / 2))
    return decomposition.values, decomposition.vectors
end

function task10f_maximum_represented_bond(parent)
    bohr_records = Iterators.flatten(
        block.compression_records for block in parent.bohr_blocks)
    values = Int[record.maximum_output_bond for record in bohr_records]
    append!(values,
        record.maximum_output_bond for record in parent.assembly_records)
    return isempty(values) ? 1 : maximum(values)
end

@testset "Task 10F exact analytic patch-MPO reference" begin
    fixture = exact_dll_fixture(
        2, beta -> DLLGaussianFilter(beta); beta_phys=0.5)
    sites = QFITensors.fused_siteinds(2)
    assembly_controls = task8_parent_controls(
        1; scalar_tolerance=1e-10, cutoff=1e-16, maxdim=128)
    coarse = QFITensors.build_exact_finite_patch_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        assembly_controls;
        sites,
        factorization_controls=task10f_factor_controls(
            cutoff=1e-12, maxdim=16),
        diagnostic_probe_count=0,
    )
    tight = QFITensors.build_exact_finite_patch_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        assembly_controls;
        sites,
        factorization_controls=task10f_factor_controls(
            cutoff=1e-14, maxdim=32),
        diagnostic_probe_count=0,
    )
    exact_global = build_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        fixture.hamiltonian,
        fixture.jumps;
        sites,
    )

    @test coarse.parent isa QFITensors.FinitePatchDLLParent
    @test coarse.parent.bundle.target_label == :finite_patch_bohr_surrogate
    @test length(coarse.exact_patches) == 6
    @test length(coarse.factorization_records) == 18
    @test !coarse.factorization_cap_saturated
    @test !tight.factorization_cap_saturated
    @test coarse.maximum_relative_reconstruction_error <= 2e-11
    @test tight.maximum_relative_reconstruction_error <= 2e-13
    @test all(record -> record.operator_label in (:Q, :L, :N),
              coarse.factorization_records)
    @test any(patch -> !isreal(patch.source), coarse.exact_patches)
    @test relative_operator_error(
        QFITensors.mpo_to_dense(
            tight.parent.bundle.total_parent, tight.parent.sites),
        exact_global.parent_fused,
    ) <= 3e-12
    comparison = QFITensors.compare_exact_patch_references(
        coarse,
        tight;
        relative_tolerance=3e-11,
        probe_count=2,
        probe_seed=771,
    )
    @test comparison.controls_independent
    @test comparison.accepted
    @test comparison.maximum_relative_action_difference <= 3e-11

    saturated = QFITensors.build_exact_finite_patch_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        assembly_controls;
        sites,
        factorization_controls=task10f_factor_controls(
            cutoff=1e-2, maxdim=1),
        diagnostic_probe_count=0,
    )
    rejected = QFITensors.compare_exact_patch_references(
        saturated,
        tight;
        relative_tolerance=1.0,
        probe_count=1,
    )
    @test saturated.factorization_cap_saturated
    @test !rejected.accepted

    fixture4 = exact_dll_fixture(
        4, beta -> DLLGaussianFilter(beta); beta_phys=0.5)
    source_y_bulk = fixture4.blocks[3 * (2 - 1) + 2].source
    patch_y_bulk = exact_dll_bohr_patch(
        fixture4.local_hamiltonian,
        source_y_bulk,
        fixture4.filter;
        radius=3,
        identity_gauge=:omit_global_shift,
        beta_phys=fixture4.beta_phys,
    )
    compact_y = QFITensors._exact_patch_bohr_mpo(
        patch_y_bulk, task10f_factor_controls(cutoff=1e-14, maxdim=64))
    @test patch_y_bulk.first_site == 1
    @test patch_y_bulk.last_site == 4
    @test isapprox(
        QFITensors._compact_mpo_to_dense(compact_y.Q, compact_y.sites),
        patch_y_bulk.Q;
        atol=2e-12,
        rtol=2e-12,
    )
end

@testset "Task 10F layered fidelity diagnostics" begin
    fixture = exact_dll_fixture(
        3, beta -> DLLGaussianFilter(beta); beta_phys=0.5)
    sites = QFITensors.fused_siteinds(3)
    assembly_controls = task8_parent_controls(
        1; scalar_tolerance=1e-10, cutoff=1e-16, maxdim=128)
    exact_patch = QFITensors.build_exact_finite_patch_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        assembly_controls;
        sites,
        factorization_controls=task10f_factor_controls(
            cutoff=1e-14, maxdim=32),
        diagnostic_probe_count=0,
    )
    represented = build_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        task8_parent_controls(
            1; scalar_tolerance=1e-9, cutoff=1e-16, maxdim=128);
        sites,
        diagnostic_probe_count=0,
    )
    exact_global = build_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        fixture.hamiltonian,
        fixture.jumps;
        sites,
    )
    exact_patch_levels, _ = task10f_dense_levels(exact_patch.parent)
    represented_levels, _ = task10f_dense_levels(represented)
    global_levels = exact_global.eigenvalues
    global_manifold = exact_global.dense.spectrum.kernel_count
    global_layer = QFITensors.DLLPatchFidelityLayer(
        :exact_global,
        global_levels[1:(global_manifold + 1)],
        global_manifold;
        gibbs_residual=exact_global.diagnostics.gibbs_residual,
    )
    exact_layer = QFITensors.DLLPatchFidelityLayer(
        :exact_patch, exact_patch_levels[1:2], 1;
        gibbs_residual=0.02)
    represented_layer = QFITensors.DLLPatchFidelityLayer(
        :represented_mpo, represented_levels[1:2], 1;
        gibbs_residual=0.02)
    task10_layer = QFITensors.DLLPatchFidelityLayer(
        :task10, represented_levels[1:2], 1;
        gibbs_residual=0.02,
        eigen_residuals=zeros(2),
    )
    rng = MersenneTwister(81)
    basis = Matrix(qr(randn(rng, ComplexF64, 8, 2)).Q[:, 1:2])
    rotation = Matrix(qr(randn(rng, ComplexF64, 2, 2)).Q)
    slow_subspaces = Dict(
        :exact_global => basis,
        :exact_patch => basis * rotation,
        :represented_mpo => basis,
        :task10 => basis * rotation,
    )
    cell = QFITensors.DLLPatchFidelityCell(;
        model_seed=46,
        num_sites=3,
        radius=1,
        maximum_patch_fraction=1.0,
        global_reference_source=:exact_dense_parent,
        global_provenance=represented.provenance,
        exact_patch_provenance=exact_patch.parent.provenance,
        represented_provenance=represented.provenance,
        task10_provenance=represented.provenance,
        global_layer,
        exact_patch_layer=exact_layer,
        represented_layer,
        task10_layer,
        slow_subspaces,
        exact_patch_maximum_bond=maximum(
            record.maximum_output_bond
            for record in exact_patch.factorization_records),
        exact_patch_cap_saturated=exact_patch.factorization_cap_saturated,
        represented_maximum_bond=task10f_maximum_represented_bond(represented),
        represented_cap_saturated=any(
            record -> record.final_cap_reached,
            represented.assembly_records),
        task10_bond_dimensions=[8, 16],
        task10_residuals=zeros(2),
    )
    @test cell.geometry_tag == :mixed_full_window
    @test QFITensors._fidelity_geometry_tag(4, 3, 1.0) == :full_patch
    @test QFITensors._fidelity_geometry_tag(4, 2, 1.0) == :mixed_full_window
    @test QFITensors._fidelity_geometry_tag(8, 3, 7 / 8) == :near_full_patch
    @test QFITensors._fidelity_geometry_tag(8, 2, 5 / 8) == :qualifying_local
    @test length(cell.pairwise_discrepancies) == 6
    @test cell.locality_discrepancy.relative_spacing_error > 0.1
    @test cell.representation_discrepancy.relative_spacing_error <= 2e-7
    @test cell.eigensolver_discrepancy.relative_spacing_error <= 5e-15
    @test cell.locality_discrepancy.slow_subspace_projector_distance <= 2e-14
    @test cell.global_layer.ground_baseline ≈ first(global_levels) atol=1e-14
    @test cell.exact_patch_layer.ground_baseline > 1e-6

    mismatched = dll_parent_provenance(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks;
        controls=represented.controls,
        raw_generator_rate=2.0,
    )
    @test_throws ArgumentError QFITensors.DLLPatchFidelityCell(;
        model_seed=46,
        num_sites=3,
        radius=1,
        maximum_patch_fraction=1.0,
        global_reference_source=:exact_dense_parent,
        global_provenance=mismatched,
        exact_patch_provenance=exact_patch.parent.provenance,
        represented_provenance=represented.provenance,
        global_layer,
        exact_patch_layer=exact_layer,
        represented_layer,
        exact_patch_maximum_bond=8,
        exact_patch_cap_saturated=false,
        represented_maximum_bond=8,
        represented_cap_saturated=false,
    )
end
