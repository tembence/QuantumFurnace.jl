function task8_parent_controls(
    radius::Int;
    scalar_tolerance::Real=1e-8,
    cutoff::Real=1e-16,
    maxdim::Int=128,
)
    return BohrMPOControls(;
        target_label=:finite_patch_bohr_surrogate,
        scalar_tolerance,
        recurrence_cutoff=cutoff,
        recurrence_maxdim=maxdim,
        product_cutoff=cutoff,
        product_maxdim=maxdim,
        sum_cutoff=cutoff,
        sum_maxdim=maxdim,
        patch_radius=radius,
    )
end

function task8_embed_patch_operator(
    matrix::AbstractMatrix{Complex{T}},
    first_site::Int,
    last_site::Int,
    num_sites::Int,
) where {T<:AbstractFloat}
    source = LocalJump1D(
        collect(first_site:last_site),
        matrix,
        one(T),
        num_sites;
        boundary=:open,
    )
    return Matrix{Complex{T}}(materialize_local_jump(source))
end

function task8_exact_embedded_patch_block(
    hamiltonian::LocalHamiltonian1D{T},
    source::LocalJump1D{T},
    filter::DLLGaussianFilter{T},
    radius::Int,
) where {T<:AbstractFloat}
    patch = exact_dll_bohr_patch(
        hamiltonian,
        source,
        filter;
        radius,
        identity_gauge=:omit_global_shift,
        beta_phys=filter.beta / hamiltonian.rescaling_factor,
    )
    Q = task8_embed_patch_operator(
        patch.Q, patch.first_site, patch.last_site, hamiltonian.num_sites)
    N = task8_embed_patch_operator(
        patch.N, patch.first_site, patch.last_site, hamiltonian.num_sites)
    d = size(Q, 1)
    identity_d = Matrix{Complex{T}}(I, d, d)
    parent = -(
        kron(conj(Q), Q) +
        (kron(identity_d, N) + kron(transpose(N), identity_d)) / T(2)
    )
    return (; patch, Q, N, parent)
end

function task8_global_tfd(
    hamiltonian::LocalHamiltonian1D{T},
    beta::T,
) where {T<:AbstractFloat}
    matrix = Matrix{Complex{T}}(materialize_local_hamiltonian(hamiltonian))
    decomposition = eigen(Hermitian(matrix))
    weights = exp.(-beta .* decomposition.values)
    square_root = decomposition.vectors *
                  Diagonal(sqrt.(weights ./ sum(weights))) *
                  adjoint(decomposition.vectors)
    return QFITensors.matrix_to_fused(
        square_root, hamiltonian.num_sites;
        physical_dim=hamiltonian.local_dim)
end

function task8_exact_patch_fixed_point_residual(patch)
    T = typeof(patch.beta_frame)
    d = size(patch.Q, 1)
    identity_d = Matrix{Complex{T}}(I, d, d)
    parent = -(
        kron(conj(patch.Q), patch.Q) +
        (kron(identity_d, patch.N) +
         kron(transpose(patch.N), identity_d)) / T(2)
    )
    decomposition = eigen(Hermitian(patch.hamiltonian))
    weights = exp.(-patch.beta_frame .* decomposition.values)
    square_root = decomposition.vectors *
                  Diagonal(sqrt.(weights ./ sum(weights))) *
                  adjoint(decomposition.vectors)
    residual = norm(parent * vec(square_root))
    minimum_energy = minimum(eigvals(Hermitian(
        (parent + adjoint(parent)) / T(2))))
    return (; residual, minimum_energy)
end

@testset "Task 8 fused physical-layer convention" begin
    n = 2
    physical_sites = QFITensors.local_operator_siteinds(n)
    fused_sites = QFITensors.fused_siteinds(n)
    local_matrix = ComplexF64[
        0.3+0.1im  -0.7+0.4im
        0.2-0.6im   0.5-0.2im
    ]
    source = LocalJump1D(
        [1], local_matrix, 0.73, n; boundary=:open)
    operator = QFITensors.local_jump_mpo(source; sites=physical_sites)
    dense_operator = Matrix(materialize_local_jump(source))
    identity_d = Matrix{ComplexF64}(I, size(dense_operator, 1),
                                    size(dense_operator, 1))

    cases = (
        (:ket, :identity, kron(identity_d, dense_operator)),
        (:bra, :conjugate, kron(conj(dense_operator), identity_d)),
        (:bra, :transpose, kron(transpose(dense_operator), identity_d)),
    )
    for (active_leg, transform, expected) in cases
        lifted = QFITensors._lift_physical_mpo_to_fused(
            operator,
            physical_sites,
            fused_sites,
            1,
            n;
            active_leg,
            transform,
            tag=Symbol(active_leg, "_", transform),
        )
        @test isapprox(
            QFITensors.mpo_to_dense(lifted, fused_sites),
            QFITensors.superoperator_to_fused(expected, n);
            atol=5e-13,
            rtol=5e-13,
        )
    end

    physical_sites32 = QFITensors.local_operator_siteinds(1)
    fused_sites32 = QFITensors.fused_siteinds(1)
    source32 = LocalJump1D(
        [1], ComplexF32.(local_matrix), Float32(0.73), 1;
        boundary=:open)
    operator32 = QFITensors.local_jump_mpo(
        source32; sites=physical_sites32)
    lifted32 = QFITensors._lift_physical_mpo_to_fused(
        operator32,
        physical_sites32,
        fused_sites32,
        1,
        1;
        active_leg=:bra,
        transform=:conjugate,
        tag=:float32_conjugate,
    )
    @test all(tensor -> eltype(tensor) == ComplexF32, lifted32)
end

@testset "Task 8 zero-Hamiltonian sign and channel blocks" begin
    n = 1
    T = Float64
    hamiltonian = LocalHamiltonian1D(
        LocalTerm1D{T}[];
        num_sites=n,
        local_dim=2,
        boundary=:open,
        coordinate_frame=:physical,
        global_shift=3.0,
        rescaling_factor=1.0,
        scale_provenance=:identity_physical_frame,
    )
    beta = T(0.5)
    filter = DLLGaussianFilter(beta)
    blocks = [
        LocalDLLBlock1D(source, filter)
        for source in local_pauli_jumps_1d(n; boundary=:open)
    ]
    config = exact_dll_config(
        TensorNetworkSpectrum(), n, beta, beta, filter)
    parent = build_dll_parent(
        config,
        hamiltonian,
        blocks,
        task8_parent_controls(1; scalar_tolerance=1e-10);
        diagnostic_probe_count=1,
    )

    @test parent isa QFITensors.FinitePatchDLLParent
    @test parent.bundle.target_label == :finite_patch_bohr_surrogate
    @test parent.bundle.block_keys == [(1, 1), (2, 1), (3, 1)]
    @test length(parent.bohr_blocks) == 3
    @test all(block -> block.target_label == :finite_patch_bohr_surrogate,
              parent.bohr_blocks)
    @test parent.bundle.coherent_correction_included
    @test !parent.assembly_diagnostics.symmetrized
    @test !parent.assembly_diagnostics.energy_shift_applied
    @test !parent.assembly_diagnostics.psd_projection_applied
    @test maximum(parent.assembly_diagnostics.block_sum_action_defects) <= 2e-11
    @test parent.bundle.error_ledger.locality_radius.evidence == :unmeasured
    @test length(parent.bundle.error_ledger.block_assembly) == 3
    @test all(entry -> entry.evidence == :floating_point_norm,
              parent.bundle.error_ledger.block_assembly)

    represented_blocks = [
        QFITensors.mpo_to_dense(block, parent.sites)
        for block in parent.bundle.block_parents
    ]
    identity_2 = Matrix{ComplexF64}(I, 2, 2)
    for index in eachindex(blocks)
        A = Matrix(materialize_local_jump(blocks[index].source))
        N = -adjoint(A) * A
        expected_qf = -(
            kron(conj(A), A) +
            (kron(identity_2, N) + kron(transpose(N), identity_2)) / 2
        )
        expected = QFITensors.superoperator_to_fused(expected_qf, n)
        @test isapprox(
            represented_blocks[index], expected;
            atol=3e-10,
            rtol=3e-10,
        )
        @test minimum(eigvals(Hermitian(
            (represented_blocks[index] + represented_blocks[index]') / 2))) >=
              -3e-10
    end

    verification = verify_dll_parent(parent; dense_overlap=true)
    @test verification.dense_overlap_used
    @test verification.block_sum_norm_difference <= 3e-12
    @test verification.total_minimum_eigenvalue >= -5e-10
    @test verify_dll_parent(parent).total_minimum_eigenvalue === nothing
end

@testset "Task 8 complete exact and matrix-free overlap" begin
    fixture = exact_dll_fixture(
        2, beta -> DLLGaussianFilter(beta); beta_phys=0.5)
    controls = task8_parent_controls(
        1; scalar_tolerance=1e-8, cutoff=1e-16, maxdim=128)
    parent = build_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        controls;
        diagnostic_probe_count=1,
    )
    exact = build_dll_parent(
        fixture.config,
        fixture.local_hamiltonian,
        fixture.blocks,
        fixture.hamiltonian,
        fixture.jumps;
        sites=parent.sites,
    )

    represented = QFITensors.mpo_to_dense(
        parent.bundle.total_parent, parent.sites)
    @test relative_operator_error(represented, exact.parent_fused) <= 2e-7
    @test maximum(
        relative_operator_error(
            QFITensors.mpo_to_dense(parent.bundle.block_parents[index],
                                    parent.sites),
            exact.block_parents_fused[index],
        )
        for index in eachindex(parent.bundle.block_parents)
    ) <= 2e-7
    @test maximum(parent.assembly_diagnostics.block_sum_action_defects) <= 2e-10

    verification = verify_dll_parent(
        parent;
        dense_overlap=true,
        gibbs_vector=exact.gibbs_vector_fused,
        kernel_tolerance=1e-7,
    )
    @test verification.block_sum_norm_difference <= 3e-11
    @test verification.total_minimum_eigenvalue >= -2e-7
    @test verification.observed_kernel_count == exact.dense.spectrum.kernel_count
    @test verification.global_gibbs_residual <= 2e-7
    @test maximum(verification.global_block_gibbs_residuals) <= 2e-7

    lindbladian_config = exact_dll_config(
        Lindbladian(),
        2,
        fixture.beta_algorithm,
        fixture.beta_phys,
        fixture.filter,
    )
    lindbladian = construct_lindbladian(
        fixture.jumps, lindbladian_config, fixture.hamiltonian)
    powers = gibbs_fractional_powers(fixture.hamiltonian.gibbs)
    function lindblad_action!(output, input)
        mul!(vec(output), lindbladian, vec(input))
        return output
    end
    rng = MersenneTwister(0x8a11)
    input_eigen = randn(rng, ComplexF64, 4, 4)
    output_eigen = similar(input_eigen)
    apply_kms_parent!(
        output_eigen,
        input_eigen,
        lindblad_action!,
        powers.sigma_quarter,
        powers.sigma_inv_quarter,
        DiscriminantBuffers(4),
    )
    U = fixture.hamiltonian.eigvecs
    input_fused = QFITensors.matrix_to_fused(
        U * input_eigen * adjoint(U), 2)
    output_fused = QFITensors.matrix_to_fused(
        U * output_eigen * adjoint(U), 2)
    input_mps = QFITensors.dense_to_mps(input_fused, parent.sites)
    output_mps = ITensorMPS.apply(
        parent.bundle.total_parent,
        input_mps;
        cutoff=0.0,
        maxdim=256,
    )
    represented_output = QFITensors.mps_to_dense(output_mps, parent.sites)
    @test norm(represented_output - output_fused) /
          max(norm(output_fused), eps()) <= 2e-7

    Q1_patch = bohr_mpo_matrices(parent.bohr_blocks[1]).Q
    Q2_patch = bohr_mpo_matrices(parent.bohr_blocks[2]).Q
    Q1 = task8_embed_patch_operator(Q1_patch, 1, 2, 2)
    Q2 = task8_embed_patch_operator(Q2_patch, 1, 2, 2)
    separated = kron(conj(Q1), Q1) + kron(conj(Q2), Q2)
    crossed = kron(conj(Q1 + Q2), Q1 + Q2)
    @test relative_operator_error(crossed, separated) > 1e-3
end

@testset "Task 8 partial-patch target remains distinct from global" begin
    n = 3
    beta = 0.5
    hamiltonian = build_local_heis_1d(
        n,
        [1.0, 1.0, 1.0];
        seed=46,
        periodic=false,
        disorder_strength=0.1,
    )
    filter = DLLGaussianFilter(beta)
    sources = local_pauli_jumps_1d(n; boundary=:open)
    blocks = [LocalDLLBlock1D(source, filter) for source in sources]
    config = exact_dll_config(
        TensorNetworkSpectrum(), n, beta, beta, filter)
    parent = build_dll_parent(
        config,
        hamiltonian,
        blocks,
        task8_parent_controls(
            1; scalar_tolerance=1e-7, cutoff=1e-15, maxdim=96);
        diagnostic_probe_count=0,
    )
    represented = QFITensors.mpo_to_dense(
        parent.bundle.total_parent, parent.sites)
    exact_patch_blocks = [
        task8_exact_embedded_patch_block(
            hamiltonian, source, filter, 1).parent
        for source in sources
    ]
    exact_patch_data = [
        exact_dll_bohr_patch(
            hamiltonian,
            source,
            filter;
            radius=1,
            identity_gauge=:omit_global_shift,
            beta_phys=beta,
        )
        for source in sources
    ]
    fixed_point_checks = task8_exact_patch_fixed_point_residual.(
        exact_patch_data)
    @test maximum(check.residual for check in fixed_point_checks) <= 3e-13
    @test minimum(check.minimum_energy for check in fixed_point_checks) >= -3e-13
    exact_patch_sum_fused = QFITensors.superoperator_to_fused(
        sum(exact_patch_blocks), n)
    @test relative_operator_error(
        represented, exact_patch_sum_fused) <= 3e-6

    exact_global_blocks = [
        task8_exact_embedded_patch_block(
            hamiltonian, source, filter, 3).parent
        for source in sources
    ]
    exact_global_fused = QFITensors.superoperator_to_fused(
        sum(exact_global_blocks), n)
    @test relative_operator_error(
        exact_patch_sum_fused, exact_global_fused) > 1e-3

    global_tfd = task8_global_tfd(hamiltonian, beta)
    verification = verify_dll_parent(
        parent;
        dense_overlap=true,
        gibbs_vector=global_tfd,
        kernel_tolerance=1e-7,
    )
    @test verification.global_gibbs_residual > 1e-5
    @test verification.global_gibbs_energy > 1e-7
    @test verification.total_minimum_eigenvalue > -3e-6
    @test parent.bundle.error_ledger.locality_radius.evidence == :unmeasured
end

@testset "Task 8 logged symmetrisation and refusal gates" begin
    n = 1
    beta = 0.5
    Z = ComplexF64[1 0; 0 -1]
    hamiltonian = LocalHamiltonian1D(
        [LocalTerm1D([1], Z, 0.7, n; boundary=:open)];
        num_sites=n,
        local_dim=2,
        boundary=:open,
        coordinate_frame=:physical,
        global_shift=0.0,
        rescaling_factor=1.0,
        scale_provenance=:identity_physical_frame,
    )
    filter = DLLGaussianFilter(beta)
    config = exact_dll_config(
        TensorNetworkSpectrum(), n, beta, beta, filter)
    blocks = [
        LocalDLLBlock1D(source, filter)
        for source in local_pauli_jumps_1d(1; boundary=:open)
    ]
    parent = build_dll_parent(
        config,
        hamiltonian,
        blocks,
        task8_parent_controls(
            1; scalar_tolerance=1e-5, cutoff=1e-8, maxdim=8);
        symmetrize=true,
        diagnostic_probe_count=0,
    )
    @test parent.assembly_diagnostics.symmetrized
    @test !parent.assembly_diagnostics.energy_shift_applied
    @test !parent.assembly_diagnostics.psd_projection_applied
    @test any(
        record -> occursin("symmetrization", String(record.stage)),
        parent.assembly_records,
    )
    represented = QFITensors.mpo_to_dense(
        parent.bundle.total_parent, parent.sites)
    @test norm(represented - represented') /
          max(norm(represented), eps()) <= 2e-12

    full_chain_controls = BohrMPOControls(
        target_label=:bohr_polynomial_full_chain)
    @test_throws ArgumentError build_dll_parent(
        config,
        hamiltonian,
        blocks,
        full_chain_controls,
    )
    @test_throws ArgumentError verify_dll_parent(
        parent; gibbs_vector=ones(ComplexF64, 4))
end
