function physical_mpo_to_dense(operator::ITensorMPS.MPO, sites)
    n = length(sites)
    local_dim = ITensors.dim(first(sites))
    dimension = local_dim^n
    full_tensor = reduce(*, operator)
    itensor_matrix = reshape(
        Array(
            full_tensor,
            ITensors.prime.(sites)...,
            ITensors.dag.(sites)...,
        ),
        dimension,
        dimension,
    )
    permutation = Vector{Int}(undef, dimension)
    for qf_index_zero in 0:(dimension - 1)
        itensor_offset = 0
        for site in 1:n
            qf_stride = local_dim^(n - site)
            digit = (qf_index_zero ÷ qf_stride) % local_dim
            itensor_offset += digit * local_dim^(site - 1)
        end
        permutation[itensor_offset + 1] = qf_index_zero + 1
    end
    inverse = invperm(permutation)
    return Matrix(itensor_matrix[inverse, inverse])
end

@inline function relative_operator_error(actual, expected)
    return opnorm(actual - expected) / max(opnorm(expected), eps(Float64))
end

function physical_bohr_fixture(
    n::Int;
    beta_phys::Float64=0.5,
    source_index::Int=n + cld(n, 2),
)
    hamiltonian = build_local_heis_1d(
        n,
        [1.0, 1.0, 1.0];
        seed=46,
        periodic=false,
        disorder_strength=0.1,
    )
    filter = DLLGaussianFilter(beta_phys)
    source = local_pauli_jumps_1d(n; boundary=:open)[source_index]
    block = LocalDLLBlock1D(source, filter)
    config = exact_dll_config(
        TensorNetworkSpectrum(), n, beta_phys, beta_phys, filter)
    return (; hamiltonian, filter, source, block, config)
end

function finite_patch_controls(
    radius::Int;
    scalar_tolerance::Real=1e-9,
    cutoff::Real=1e-18,
    maxdim::Int=64,
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

function bohr_mpo_matrices(result)
    return (
        Q=physical_mpo_to_dense(result.Q, result.sites),
        L=physical_mpo_to_dense(result.L, result.sites),
        N=physical_mpo_to_dense(result.N, result.sites),
    )
end

@testset "Task 7B local operator MPO builders" begin
    fixture = physical_bohr_fixture(3)
    sites = QFITensors.local_operator_siteinds(3)
    hamiltonian_mpo = QFITensors.local_hamiltonian_mpo(
        fixture.hamiltonian; sites)
    dense_hamiltonian = physical_mpo_to_dense(hamiltonian_mpo, sites)
    expected_hamiltonian = Matrix(materialize_local_hamiltonian(
        fixture.hamiltonian))
    @test relative_operator_error(
        dense_hamiltonian, expected_hamiltonian) <= 2e-14

    shifted = LocalHamiltonian1D(
        fixture.hamiltonian.terms;
        num_sites=fixture.hamiltonian.num_sites,
        local_dim=fixture.hamiltonian.local_dim,
        boundary=:open,
        coordinate_frame=:physical,
        global_shift=3.25,
        rescaling_factor=1.0,
        scale_provenance=:identity_physical_frame,
    )
    shifted_mpo = QFITensors.local_hamiltonian_mpo(shifted; sites)
    @test relative_operator_error(
        physical_mpo_to_dense(shifted_mpo, sites), dense_hamiltonian) <= 2e-14

    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    two_site_source = LocalJump1D(
        [1, 2], kron(Y, X), 0.37, 3; boundary=:open)
    source_mpo = QFITensors.local_jump_mpo(two_site_source; sites)
    @test relative_operator_error(
        physical_mpo_to_dense(source_mpo, sites),
        Matrix(materialize_local_jump(two_site_source)),
    ) <= 2e-14

    Z32 = ComplexF32[1 0; 0 -1]
    term32 = LocalTerm1D(
        [1], Z32, Float32(0.75), 2; boundary=:open)
    hamiltonian32 = LocalHamiltonian1D(
        [term32];
        num_sites=2,
        local_dim=2,
        boundary=:open,
        coordinate_frame=:physical,
        global_shift=Float32(0),
        rescaling_factor=Float32(1),
        scale_provenance=:identity_physical_frame,
    )
    @test eltype(QFITensors.local_hamiltonian_mpo(hamiltonian32)[1]) ==
          ComplexF32

    X32 = ComplexF32[0 1; 1 0]
    source32 = LocalJump1D(
        [1], X32, Float32(0.5), 2; boundary=:open)
    filter32 = DLLGaussianFilter(Float32(0.25))
    block32 = LocalDLLBlock1D(source32, filter32)
    config32 = exact_dll_config(
        TensorNetworkSpectrum(), 2, Float32(0.25), Float32(0.25), filter32)
    controls32 = finite_patch_controls(
        1;
        scalar_tolerance=Float32(1e-6),
        cutoff=Float32(1e-8),
        maxdim=16,
    )
    result32 = build_dll_bohr_mpo(
        config32, hamiltonian32, block32, controls32)
    @test all(
        operator -> all(tensor -> eltype(tensor) == ComplexF32, operator),
        (result32.Q, result32.L, result32.N),
    )
    exact32 = exact_dll_bohr_patch(
        hamiltonian32,
        source32,
        filter32;
        radius=1,
        identity_gauge=:omit_global_shift,
        beta_phys=Float32(0.25),
    )
    matrices32 = bohr_mpo_matrices(result32)
    @test maximum((
        relative_operator_error(matrices32.Q, exact32.Q),
        relative_operator_error(matrices32.L, exact32.L),
        relative_operator_error(matrices32.N, exact32.N),
    )) <= 2e-5

    Zbig = Matrix{Complex{BigFloat}}(Z32)
    Xbig = Matrix{Complex{BigFloat}}(X32)
    term_big = LocalTerm1D(
        [1], Zbig, BigFloat("0.75"), 2; boundary=:open)
    hamiltonian_big = LocalHamiltonian1D(
        [term_big];
        num_sites=2,
        local_dim=2,
        boundary=:open,
        coordinate_frame=:physical,
        global_shift=BigFloat(0),
        rescaling_factor=BigFloat(1),
        scale_provenance=:identity_physical_frame,
    )
    source_big = LocalJump1D(
        [1], Xbig, BigFloat("0.5"), 2; boundary=:open)
    filter_big = DLLGaussianFilter(BigFloat("0.25"))
    block_big = LocalDLLBlock1D(source_big, filter_big)
    config_big = exact_dll_config(
        TensorNetworkSpectrum(),
        2,
        BigFloat("0.25"),
        BigFloat("0.25"),
        filter_big,
    )
    @test_throws ArgumentError build_dll_bohr_mpo(
        config_big, hamiltonian_big, block_big, finite_patch_controls(1))

    @test_throws ArgumentError QFITensors.local_hamiltonian_mpo(
        build_local_heis_1d(
            3, [1.0, 1.0, 1.0]; seed=46, periodic=true))
end

@testset "Task 7B exact patch overlap and provenance" begin
    fixture = physical_bohr_fixture(4)
    controls = finite_patch_controls(1)
    result = build_dll_bohr_mpo(
        fixture.config,
        fixture.hamiltonian,
        fixture.block,
        controls,
    )
    exact = exact_dll_bohr_patch(
        fixture.hamiltonian,
        fixture.source,
        fixture.filter;
        radius=1,
        identity_gauge=:omit_global_shift,
        beta_phys=fixture.config.beta_phys,
    )
    matrices = bohr_mpo_matrices(result)
    errors = (
        relative_operator_error(matrices.Q, exact.Q),
        relative_operator_error(matrices.L, exact.L),
        relative_operator_error(matrices.N, exact.N),
    )
    @test maximum(errors) <= 2e-8
    @test opnorm(matrices.Q - adjoint(matrices.Q)) /
          max(opnorm(matrices.Q), eps()) <= 2e-10
    @test opnorm(matrices.N - adjoint(matrices.N)) /
          max(opnorm(matrices.N), eps()) <= 2e-10
    @test result isa QFITensors.DLLGaussianBohrMPO
    @test result.target_label == :finite_patch_bohr_surrogate
    @test result.identity_gauge == :omit_global_shift
    @test result.omission_rule == :retain_only_fully_contained_terms
    @test result.source_sites_global == fixture.source.sites
    @test result.first_site == 1
    @test result.last_site == 3
    @test result.beta_phys == fixture.config.beta_phys
    @test result.beta_alg == fixture.config.beta
    @test result.error_ledger.scalar_polynomial.evidence == :rigorous_bound
    @test result.error_ledger.modular_compatibility.evidence == :rigorous_bound
    @test result.error_ledger.locality_radius.evidence == :unmeasured
    @test all(entry -> entry.evidence == :unmeasured,
              result.error_ledger.mpo_recurrences)
    @test !isempty(result.compression_records)
    @test any(record -> record.operation == :zipup_product,
              result.compression_records)
    @test any(record -> record.operation == :directsum_add,
              result.compression_records)
    zipup_records = filter(
        record -> record.operation == :zipup_product,
        result.compression_records,
    )
    addition_records = filter(
        record -> record.operation == :directsum_add,
        result.compression_records,
    )
    @test all(record -> record.internal_factorization_telemetry == :unmeasured,
              zipup_records)
    @test all(record -> record.intermediate_cap_status in
                        (:unmeasured, :cap_reached_after_internal_sweep),
              zipup_records)
    @test all(record -> record.internal_factorization_telemetry ==
                        :not_applicable,
              addition_records)
    @test all(record -> record.intermediate_cap_status == :not_applicable,
              addition_records)
    @test all(record -> !record.final_cap_reached,
              result.compression_records)

    shifted = LocalHamiltonian1D(
        fixture.hamiltonian.terms;
        num_sites=fixture.hamiltonian.num_sites,
        local_dim=fixture.hamiltonian.local_dim,
        boundary=:open,
        coordinate_frame=:physical,
        global_shift=-7.0,
        rescaling_factor=1.0,
        scale_provenance=:identity_physical_frame,
    )
    shifted_result = build_dll_bohr_mpo(
        fixture.config, shifted, fixture.block, controls;
        chebyshev_data=result.chebyshev_data)
    shifted_matrices = bohr_mpo_matrices(shifted_result)
    @test maximum((
        relative_operator_error(shifted_matrices.Q, matrices.Q),
        relative_operator_error(shifted_matrices.L, matrices.L),
        relative_operator_error(shifted_matrices.N, matrices.N),
    )) <= 2e-13

    algorithm = exact_dll_fixture(
        4, beta -> DLLGaussianFilter(beta); beta_phys=0.5)
    source_index = 4 + cld(4, 2)
    algorithm_result = build_dll_bohr_mpo(
        algorithm.config,
        algorithm.local_hamiltonian,
        algorithm.blocks[source_index],
        controls,
    )
    algorithm_matrices = bohr_mpo_matrices(algorithm_result)
    @test maximum((
        relative_operator_error(algorithm_matrices.Q, matrices.Q),
        relative_operator_error(algorithm_matrices.L, matrices.L),
        relative_operator_error(algorithm_matrices.N, matrices.N),
    )) <= 2e-8
end

@testset "Task 7B independent approximation controls" begin
    fixture = physical_bohr_fixture(4)
    exact = exact_dll_bohr_patch(
        fixture.hamiltonian,
        fixture.source,
        fixture.filter;
        radius=1,
        identity_gauge=:omit_global_shift,
        beta_phys=fixture.config.beta_phys,
    )
    polynomial_errors = Float64[]
    polynomial_degrees = Int[]
    for tolerance in (1e-3, 1e-6, 1e-9)
        data = dll_gaussian_chebyshev_data(exact; tolerance)
        result = build_dll_bohr_mpo(
            fixture.config,
            fixture.hamiltonian,
            fixture.block,
            finite_patch_controls(1; scalar_tolerance=tolerance);
            chebyshev_data=data,
        )
        matrices = bohr_mpo_matrices(result)
        push!(polynomial_errors, maximum((
            relative_operator_error(matrices.Q, exact.Q),
            relative_operator_error(matrices.L, exact.L),
            relative_operator_error(matrices.N, exact.N),
        )))
        push!(polynomial_degrees, maximum((
            data.q.degree, data.f.degree, data.sech.degree)))
    end
    @test issorted(polynomial_degrees)
    @test polynomial_errors[2] <= polynomial_errors[1]
    @test polynomial_errors[3] <= polynomial_errors[2]
    @test last(polynomial_errors) <= 2e-8

    compression_fixture = physical_bohr_fixture(5)
    compression_exact = exact_dll_bohr_patch(
        compression_fixture.hamiltonian,
        compression_fixture.source,
        compression_fixture.filter;
        radius=2,
        identity_gauge=:omit_global_shift,
        beta_phys=compression_fixture.config.beta_phys,
    )
    fixed_data = dll_gaussian_chebyshev_data(
        compression_exact; tolerance=1e-9)
    compression_errors = Float64[]
    for (cutoff, maxdim) in ((1e-8, 4), (1e-16, 16), (1e-24, 64))
        result = build_dll_bohr_mpo(
            compression_fixture.config,
            compression_fixture.hamiltonian,
            compression_fixture.block,
            finite_patch_controls(
                2; scalar_tolerance=1e-9, cutoff, maxdim);
            chebyshev_data=fixed_data,
        )
        matrices = bohr_mpo_matrices(result)
        push!(compression_errors, maximum((
            relative_operator_error(matrices.Q, compression_exact.Q),
            relative_operator_error(matrices.L, compression_exact.L),
            relative_operator_error(matrices.N, compression_exact.N),
        )))
    end
    @test compression_errors[2] <= compression_errors[1]
    @test compression_errors[3] <= compression_errors[2]
    @test last(compression_errors) <= 2e-8

    radius_three_fixture = physical_bohr_fixture(4)
    radius_three_exact = exact_dll_bohr_patch(
        radius_three_fixture.hamiltonian,
        radius_three_fixture.source,
        radius_three_fixture.filter;
        radius=3,
        identity_gauge=:omit_global_shift,
        beta_phys=radius_three_fixture.config.beta_phys,
    )
    radius_three_result = build_dll_bohr_mpo(
        radius_three_fixture.config,
        radius_three_fixture.hamiltonian,
        radius_three_fixture.block,
        finite_patch_controls(3),
    )
    radius_three_matrices = bohr_mpo_matrices(radius_three_result)
    @test maximum((
        relative_operator_error(radius_three_matrices.Q, radius_three_exact.Q),
        relative_operator_error(radius_three_matrices.L, radius_three_exact.L),
        relative_operator_error(radius_three_matrices.N, radius_three_exact.N),
    )) <= 2e-8
end

@testset "Task 7B scalable patch boundary and refusal gates" begin
    fixture = physical_bohr_fixture(12; source_index=3 * 12 - 5)
    result = build_dll_bohr_mpo(
        fixture.config,
        fixture.hamiltonian,
        fixture.block,
        finite_patch_controls(1),
    )
    @test length(result.sites) == 3
    @test result.source_sites_global == [7]
    @test result.first_site == 6
    @test result.last_site == 8
    @test isapprox(
        fixture.source.coefficient,
        inv(sqrt(3 * 12));
        rtol=eps(Float64),
        atol=0.0,
    )
    result_matrices = bohr_mpo_matrices(result)
    exact = exact_dll_bohr_patch(
        fixture.hamiltonian,
        fixture.source,
        fixture.filter;
        radius=1,
        identity_gauge=:omit_global_shift,
        beta_phys=fixture.config.beta_phys,
    )
    @test maximum((
        relative_operator_error(result_matrices.Q, exact.Q),
        relative_operator_error(result_matrices.L, exact.L),
        relative_operator_error(result_matrices.N, exact.N),
    )) <= 2e-8

    @test_throws ArgumentError build_dll_bohr_mpo(
        fixture.config,
        fixture.hamiltonian,
        fixture.block,
        BohrMPOControls(target_label=:bohr_polynomial_full_chain),
    )
    @test_throws ArgumentError build_dll_bohr_mpo(
        fixture.config,
        fixture.hamiltonian,
        fixture.block,
        finite_patch_controls(4),
    )
    undersized_data = dll_gaussian_chebyshev_data(0.1; tolerance=1e-9)
    @test_throws ArgumentError build_dll_bohr_mpo(
        fixture.config,
        fixture.hamiltonian,
        fixture.block,
        finite_patch_controls(1);
        chebyshev_data=undersized_data,
    )

    empty_hamiltonian = LocalHamiltonian1D(
        LocalTerm1D{Float64}[];
        num_sites=3,
        local_dim=2,
        boundary=:open,
        coordinate_frame=:physical,
        global_shift=4.0,
        rescaling_factor=1.0,
        scale_provenance=:identity_physical_frame,
    )
    zero_filter = DLLGaussianFilter(0.5)
    zero_source = local_pauli_jumps_1d(3; boundary=:open)[5]
    zero_block = LocalDLLBlock1D(zero_source, zero_filter)
    zero_config = exact_dll_config(
        TensorNetworkSpectrum(), 3, 0.5, 0.5, zero_filter)
    zero_result = build_dll_bohr_mpo(
        zero_config,
        empty_hamiltonian,
        zero_block,
        finite_patch_controls(1),
    )
    zero_matrices = bohr_mpo_matrices(zero_result)
    zero_exact_source = Matrix(materialize_local_jump(LocalJump1D(
        [2], zero_source.matrix, zero_source.coefficient, 3;
        boundary=:open)))
    # The compact patch is the full three-site chain for site two and radius one.
    @test relative_operator_error(zero_matrices.Q, zero_exact_source) <= 2e-14
    @test relative_operator_error(zero_matrices.L, zero_exact_source) <= 2e-14
    @test relative_operator_error(
        zero_matrices.N, -adjoint(zero_exact_source) * zero_exact_source) <= 2e-14

    Z = ComplexF64[1 0; 0 -1]
    X = ComplexF64[0 1; 1 0]
    one_site_hamiltonian = LocalHamiltonian1D(
        [LocalTerm1D([1], Z, 0.75, 1; boundary=:open)];
        num_sites=1,
        local_dim=2,
        boundary=:open,
        coordinate_frame=:physical,
        global_shift=2.0,
        rescaling_factor=1.0,
        scale_provenance=:identity_physical_frame,
    )
    one_site_source = LocalJump1D(
        [1], X, 1.0, 1; boundary=:open)
    one_site_filter = DLLGaussianFilter(0.5)
    one_site_block = LocalDLLBlock1D(one_site_source, one_site_filter)
    one_site_config = exact_dll_config(
        TensorNetworkSpectrum(), 1, 0.5, 0.5, one_site_filter)
    one_site_result = build_dll_bohr_mpo(
        one_site_config,
        one_site_hamiltonian,
        one_site_block,
        finite_patch_controls(1),
    )
    one_site_exact = exact_dll_bohr_patch(
        one_site_hamiltonian,
        one_site_source,
        one_site_filter;
        radius=1,
        identity_gauge=:omit_global_shift,
        beta_phys=0.5,
    )
    one_site_matrices = bohr_mpo_matrices(one_site_result)
    @test maximum((
        relative_operator_error(one_site_matrices.Q, one_site_exact.Q),
        relative_operator_error(one_site_matrices.L, one_site_exact.L),
        relative_operator_error(one_site_matrices.N, one_site_exact.N),
    )) <= 2e-8
    @test all(
        record -> record.intermediate_cap_status == :not_applicable &&
                  !record.final_cap_reached,
        one_site_result.compression_records,
    )
end
