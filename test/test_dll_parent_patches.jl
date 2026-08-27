using Test
using LinearAlgebra

function task7_patch_fixture(; algorithm::Bool=false, scale::Float64=4.0)
    n = 4
    onsite_coefficients = (0.1, 0.2, -0.3, 0.4)
    bond_coefficients = (0.7, -0.8, 0.9)
    divisor = algorithm ? scale : 1.0
    terms = LocalTerm1D{Float64}[]
    for site in 1:n
        push!(terms, LocalTerm1D(
            [site], Z, onsite_coefficients[site] / divisor, n;
            boundary=:open))
    end
    for site in 1:(n - 1)
        push!(terms, LocalTerm1D(
            [site, site + 1], kron(X, X), bond_coefficients[site] / divisor,
            n; boundary=:open))
    end
    hamiltonian = LocalHamiltonian1D(
        terms;
        num_sites=n,
        boundary=:open,
        coordinate_frame=algorithm ? :algorithm : :physical,
        global_shift=algorithm ? 0.35 : 2.0,
        rescaling_factor=algorithm ? scale : 1.0,
        scale_provenance=algorithm ? :exact_dense : :identity_physical_frame,
    )
    source = LocalJump1D(
        [2], Y, inv(sqrt(3n)), n; boundary=:open)
    return (; hamiltonian, source)
end

@testset "Exact finite-patch Gaussian DLL reference" begin
    beta_phys = 0.5
    fixture = task7_patch_fixture()
    filter = DLLGaussianFilter(beta_phys)

    radius_zero = exact_dll_bohr_patch(
        fixture.hamiltonian, fixture.source, filter; radius=0)
    @test radius_zero.target_label == :finite_patch_bohr_surrogate
    @test radius_zero.first_site == 2
    @test radius_zero.last_site == 2
    @test radius_zero.retained_term_count == 1
    @test radius_zero.omitted_term_count == 6
    @test radius_zero.omission_rule == :retain_only_fully_contained_terms
    @test radius_zero.source_global_coefficient == inv(sqrt(12))
    @test radius_zero.beta_frame == beta_phys
    @test radius_zero.beta_phys == beta_phys
    @test radius_zero.beta_alg == beta_phys
    @test radius_zero.source ≈ inv(sqrt(12)) .* Y atol=0 rtol=0
    @test radius_zero.hamiltonian ≈ 0.2 .* Z + 2.0I atol=2e-15 rtol=0
    @test radius_zero.locality_error.evidence == :unmeasured
    @test radius_zero.locality_error.magnitude === nothing

    returned_sites = radius_zero.source_sites_global
    returned_sites[1] = 4
    returned_source = radius_zero.source
    returned_source[1, 1] = 99
    returned_Q = radius_zero.Q
    returned_Q[1, 1] = 99
    @test fixture.source.sites == [2]
    @test radius_zero.source_sites_global == [2]
    @test radius_zero.source ≈ inv(sqrt(12)) .* Y atol=0 rtol=0
    @test radius_zero.Q[1, 1] != 99

    omitted_shift = exact_dll_bohr_patch(
        fixture.hamiltonian, fixture.source, filter;
        radius=0, identity_gauge=:omit_global_shift)
    @test omitted_shift.retained_global_shift == 0.0
    @test radius_zero.hamiltonian - omitted_shift.hamiltonian ≈ 2.0I atol=2e-15 rtol=0
    @test radius_zero.Q ≈ omitted_shift.Q atol=2e-15 rtol=2e-15
    @test radius_zero.L ≈ omitted_shift.L atol=2e-15 rtol=2e-15
    @test radius_zero.N ≈ omitted_shift.N atol=3e-15 rtol=3e-15

    radius_one = exact_dll_bohr_patch(
        fixture.hamiltonian, fixture.source, filter; radius=1)
    @test (radius_one.first_site, radius_one.last_site) == (1, 3)
    @test radius_one.retained_term_count == 5
    @test radius_one.omitted_term_count == 2

    full_patch = exact_dll_bohr_patch(
        fixture.hamiltonian, fixture.source, filter;
        first_site=1, last_site=4)
    @test full_patch.hamiltonian ≈
          Matrix(materialize_local_hamiltonian(fixture.hamiltonian)) atol=2e-15 rtol=0
    @test full_patch.source ≈
          Matrix(materialize_local_jump(fixture.source)) atol=0 rtol=0
    @test full_patch.retained_term_count == 7
    @test full_patch.omitted_term_count == 0

    polynomial = approximate_dll_bohr_patch(
        full_patch; tolerance=1e-9, max_degree=256)
    @test norm(polynomial.Q - full_patch.Q) <= 3e-9
    @test norm(polynomial.L - full_patch.L) <= 3e-9
    @test norm(polynomial.N - full_patch.N) <= 5e-9
    @test polynomial.chebyshev_data.interval_provenance ==
          fixture.hamiltonian.spectral_bound_provenance
    @test polynomial.target_label == :finite_patch_bohr_surrogate
    @test polynomial.locality_error.evidence == :unmeasured
    @test polynomial.dense_recurrence_error.evidence == :unmeasured

    truncated_polynomial = approximate_dll_bohr_patch(
        radius_one; tolerance=1e-9, max_degree=256)
    @test norm(truncated_polynomial.Q - radius_one.Q) <= 3e-9
    @test norm(truncated_polynomial.L - radius_one.L) <= 3e-9
    @test norm(truncated_polynomial.N - radius_one.N) <= 5e-9
    omitted_polynomial = approximate_dll_bohr_patch(
        omitted_shift; tolerance=1e-9, max_degree=256)
    @test norm(omitted_polynomial.Q - omitted_shift.Q) <= 3e-9
    @test norm(omitted_polynomial.L - omitted_shift.L) <= 3e-9
    @test norm(omitted_polynomial.N - omitted_shift.N) <= 5e-9

    block = LocalDLLBlock1D(fixture.source, filter)
    block_patch = exact_dll_bohr_patch(
        fixture.hamiltonian, block; radius=1)
    @test block_patch.Q ≈ radius_one.Q atol=0 rtol=0
    @test_throws ArgumentError exact_dll_bohr_patch(
        fixture.hamiltonian, fixture.source,
        DLLMetropolisFilter(beta_phys; S=4.0); radius=1)
    @test_throws ArgumentError exact_dll_bohr_patch(
        fixture.hamiltonian, fixture.source,
        DLLGaussianFilter(Float32(beta_phys)); radius=1)
    @test_throws ArgumentError exact_dll_bohr_patch(
        fixture.hamiltonian, fixture.source, filter;
        radius=1, first_site=1, last_site=3)
    @test_throws ArgumentError exact_dll_bohr_patch(
        fixture.hamiltonian, fixture.source, filter;
        first_site=3, last_site=4)

    nonhermitian = LocalJump1D(
        [2], ComplexF64[0 1; 0 0], 1.0, 4; boundary=:open)
    @test_throws ArgumentError exact_dll_bohr_patch(
        fixture.hamiltonian, nonhermitian, filter; radius=1)
    noncontiguous = LocalJump1D(
        [1, 3], kron(X, X), 1.0, 4; boundary=:open)
    @test_throws ArgumentError exact_dll_bohr_patch(
        fixture.hamiltonian, noncontiguous, filter;
        first_site=1, last_site=3)
end

@testset "Zero-Hamiltonian patch reference" begin
    hamiltonian = LocalHamiltonian1D(
        LocalTerm1D{Float64}[];
        num_sites=2,
        boundary=:open,
        coordinate_frame=:physical,
        global_shift=0.0,
        rescaling_factor=1.0,
        scale_provenance=:identity_physical_frame,
    )
    source = LocalJump1D([1], X, 0.3, 2; boundary=:open)
    patch = exact_dll_bohr_patch(
        hamiltonian, source, DLLGaussianFilter(0.7); radius=0)
    @test patch.retained_term_count == 0
    @test patch.spectral_width_bound == 0.0
    @test patch.dimensionless_interval_radius == 0.0
    @test patch.Q ≈ patch.source atol=0 rtol=0
    @test patch.L ≈ patch.source atol=0 rtol=0
    @test patch.N ≈ -(patch.source' * patch.source) atol=2e-15 rtol=2e-15
end

@testset "Patch global-frame invariance" begin
    beta_phys = 0.5
    scale = 4.0
    physical = task7_patch_fixture()
    algorithm = task7_patch_fixture(; algorithm=true, scale)
    physical_patch = exact_dll_bohr_patch(
        physical.hamiltonian, physical.source,
        DLLGaussianFilter(beta_phys); radius=1,
        identity_gauge=:omit_global_shift)
    algorithm_patch = exact_dll_bohr_patch(
        algorithm.hamiltonian, algorithm.source,
        DLLGaussianFilter(scale * beta_phys); radius=1,
        identity_gauge=:omit_global_shift, beta_phys)
    @test physical_patch.coordinate_frame == :physical
    @test algorithm_patch.coordinate_frame == :algorithm
    @test algorithm_patch.rescaling_factor == scale
    @test algorithm_patch.beta_frame == scale * beta_phys
    @test algorithm_patch.beta_phys == beta_phys
    @test algorithm_patch.beta_alg == scale * beta_phys
    @test algorithm_patch.hamiltonian ≈ physical_patch.hamiltonian / scale atol=2e-15 rtol=2e-15
    @test physical_patch.Q ≈ algorithm_patch.Q atol=3e-14 rtol=3e-14
    @test physical_patch.L ≈ algorithm_patch.L atol=3e-14 rtol=3e-14
    @test physical_patch.N ≈ algorithm_patch.N atol=4e-14 rtol=4e-14
    @test physical_patch.source_global_coefficient ==
          algorithm_patch.source_global_coefficient == inv(sqrt(12))
    physical_data = dll_gaussian_chebyshev_data(
        physical.hamiltonian, DLLGaussianFilter(beta_phys);
        tolerance=1e-8, max_degree=256)
    algorithm_data = dll_gaussian_chebyshev_data(
        algorithm.hamiltonian, DLLGaussianFilter(scale * beta_phys);
        beta_phys, tolerance=1e-8, max_degree=256)
    @test physical_data.interval_provenance ==
          physical.hamiltonian.spectral_bound_provenance
    @test algorithm_data.interval_provenance ==
          algorithm.hamiltonian.spectral_bound_provenance
    @test_throws ArgumentError exact_dll_bohr_patch(
        algorithm.hamiltonian, algorithm.source,
        DLLGaussianFilter(scale * beta_phys); radius=1)
    @test_throws ArgumentError exact_dll_bohr_patch(
        algorithm.hamiltonian, algorithm.source,
        DLLGaussianFilter(scale * beta_phys); radius=1,
        beta_phys=0.6)
    @test_throws ArgumentError dll_gaussian_chebyshev_data(
        algorithm.hamiltonian, DLLGaussianFilter(scale * beta_phys);
        tolerance=1e-8, max_degree=256)

    algorithm_polynomial = approximate_dll_bohr_patch(
        algorithm_patch; tolerance=1e-9, max_degree=256)
    @test norm(algorithm_polynomial.Q - algorithm_patch.Q) <= 3e-9
    @test norm(algorithm_polynomial.L - algorithm_patch.L) <= 3e-9
    @test norm(algorithm_polynomial.N - algorithm_patch.N) <= 5e-9
end
