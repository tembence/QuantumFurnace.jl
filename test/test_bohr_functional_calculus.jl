using Test
using LinearAlgebra

@testset "Certified Bohr Chebyshev scalar data" begin
    q(x) = exp(-(x * x) / 8)
    f(x) = exp(-(x * x) / 8 - x / 4)
    sech_quarter(x) = begin
        decay = exp(-abs(x) / 4)
        2 * decay / (1 + decay * decay)
    end

    zero_data = dll_gaussian_chebyshev_data(0.0; tolerance=1e-12)
    @test zero_data.q.degree == 0
    @test zero_data.f.degree == 0
    @test zero_data.sech.degree == 0
    @test evaluate_bohr_chebyshev(zero_data.q, 0.0) == 1.0
    @test evaluate_bohr_chebyshev(zero_data.f, 0.0) == 1.0
    @test evaluate_bohr_chebyshev(zero_data.sech, 0.0) == 1.0
    @test zero_data.modular_compatibility_bound == 0.0

    data = dll_gaussian_chebyshev_data(
        10.0; tolerance=1e-9, max_degree=128, working_precision=256)
    @test data.interval_provenance == :user_supplied
    @test data.q.bound_method == :bernstein_ellipse_interpolant
    @test data.q.parity == :even
    @test data.f.parity == :none
    @test data.sech.parity == :even
    @test all(iszero, data.q.coefficients[2:2:end])
    @test all(iszero, data.sech.coefficients[2:2:end])
    @test data.modular_compatibility_bound <= 1e-9

    grid = range(-10.0, 10.0; length=4001)
    q_error = maximum(abs(evaluate_bohr_chebyshev(data.q, x) - q(x))
                      for x in grid)
    f_error = maximum(abs(evaluate_bohr_chebyshev(data.f, x) - f(x))
                      for x in grid)
    sech_error = maximum(
        abs(evaluate_bohr_chebyshev(data.sech, x) - sech_quarter(x))
        for x in grid)
    modular_error = maximum(
        abs(evaluate_bohr_chebyshev(data.f, x) -
            exp(-x / 4) * evaluate_bohr_chebyshev(data.q, x))
        for x in grid)
    @test q_error <= data.q.uniform_error_bound
    @test f_error <= data.f.uniform_error_bound
    @test sech_error <= data.sech.uniform_error_bound
    @test modular_error <= data.modular_compatibility_bound

    # Public coefficient access is defensive: mutating a returned vector must
    # not invalidate the stored certificate.
    coefficients = data.q.coefficients
    coefficients[1] = 99.0
    @test data.q.coefficients[1] != 99.0

    wide = dll_gaussian_chebyshev_data(
        20.0; tolerance=1e-8, max_degree=256, working_precision=256)
    @test maximum((wide.q.degree, wide.f.degree, wide.sech.degree)) <= 256
    @test wide.modular_compatibility_bound <= 1e-8
    @test isapprox(evaluate_bohr_chebyshev(wide.q, 20.0), q(20.0);
                   atol=wide.q.uniform_error_bound, rtol=0)

    data32 = dll_gaussian_chebyshev_data(
        6.0f0; tolerance=1.0f-5, max_degree=128)
    @test data32 isa DLLGaussianChebyshevData{Float32}
    @test data32.q.coefficients isa Vector{Float32}
    @test data32.modular_compatibility_bound <= 1.0f-5

    # Combined exponents and the abs-based sech formula avoid intermediate
    # overflow; BigFloat retains tails that underflow in Float64.
    @test isfinite(QuantumFurnace._gaussian_f_dimensionless(-4000.0))
    @test isfinite(QuantumFurnace._sech_quarter_dimensionless(4000.0))
    @test QuantumFurnace._sech_quarter_dimensionless(big"4000") > 0

    @test_throws DomainError evaluate_bohr_chebyshev(data.q, 10.1)
    @test_throws DomainError evaluate_bohr_chebyshev(data.q, nextfloat(10.0))
    @test_throws ArgumentError dll_gaussian_chebyshev_data(
        10.0; tolerance=1e-14, max_degree=2)
end

function exact_dimensionless_filter(H, A, beta, scalar_function)
    decomposition = eigen(Hermitian(H))
    source_eigen = decomposition.vectors' * A * decomposition.vectors
    filtered = similar(source_eigen)
    for column in axes(filtered, 2), row in axes(filtered, 1)
        x = beta * (decomposition.values[row] - decomposition.values[column])
        filtered[row, column] = scalar_function(x) * source_eigen[row, column]
    end
    return decomposition.vectors * filtered * decomposition.vectors'
end

@testset "Dense rotating Bohr commutator recurrence" begin
    q(x) = exp(-(x * x) / 8)
    f(x) = exp(-(x * x) / 8 - x / 4)
    sech_quarter(x) = inv(cosh(x / 4))

    H = ComplexF64[
        0 0 0;
        0 0 0;
        0 0 1.7
    ]
    A = ComplexF64[
        0.2 0.1+0.3im -0.4im;
        -0.2+0.1im -0.3 0.7;
        0.5+0.2im -0.1im 0.4
    ]
    beta = 0.8
    radius = nextfloat(beta * 1.7)
    data = dll_gaussian_chebyshev_data(
        radius; tolerance=1e-10, max_degree=128)
    approximation = @inferred apply_dll_gaussian_chebyshev(H, A, beta, data)
    Q_exact = exact_dimensionless_filter(H, A, beta, q)
    L_exact = exact_dimensionless_filter(H, A, beta, f)
    rate_exact = L_exact' * L_exact
    N_exact = -exact_dimensionless_filter(H, rate_exact, beta, sech_quarter)
    @test norm(approximation.Q - Q_exact) <=
          2data.q.uniform_error_bound * norm(A)
    @test norm(approximation.L - L_exact) <=
          2data.f.uniform_error_bound * norm(A)
    @test norm(approximation.N - N_exact) <= 2e-9

    shifted = apply_dll_gaussian_chebyshev(
        H + 3.25I, A, beta, data)
    @test shifted.Q ≈ approximation.Q atol=2e-14 rtol=2e-14
    @test shifted.L ≈ approximation.L atol=2e-14 rtol=2e-14
    @test shifted.N ≈ approximation.N atol=3e-14 rtol=3e-14

    scale = 4.0
    algorithm = apply_dll_gaussian_chebyshev(
        H / scale + 0.2I, A, scale * beta, data)
    @test algorithm.Q ≈ approximation.Q atol=3e-14 rtol=3e-14
    @test algorithm.L ≈ approximation.L atol=3e-14 rtol=3e-14
    @test algorithm.N ≈ approximation.N atol=4e-14 rtol=4e-14

    diagonal_source = Diagonal(ComplexF64[0.2, -0.1, 0.7]) |> Matrix
    commuting = apply_dll_gaussian_chebyshev(
        H, diagonal_source, beta, data)
    @test commuting.Q ≈ diagonal_source atol=3e-14 rtol=3e-14
    @test commuting.L ≈ diagonal_source atol=3e-14 rtol=3e-14
    @test commuting.N ≈ -(diagonal_source' * diagonal_source) atol=3e-14 rtol=3e-14

    # Directed transition check fixes the commutator sign: rows carry the
    # outgoing energy in the repository convention [H,A_nu]=nu A_nu.
    H2 = ComplexF64[0 0; 0 2]
    A2 = ComplexF64[0 1; 1 0]
    beta2 = 0.6
    data2 = dll_gaussian_chebyshev_data(
        nextfloat(2beta2); tolerance=1e-11, max_degree=128)
    result2 = apply_dll_gaussian_chebyshev(H2, A2, beta2, data2)
    @test abs(result2.L[2, 1] / result2.L[1, 2] - exp(-beta2)) <= 2e-10

    narrow = dll_gaussian_chebyshev_data(
        beta; tolerance=1e-9, max_degree=128)
    @test_throws ArgumentError apply_dll_gaussian_chebyshev(
        H, A, beta, narrow)
    unit_data = dll_gaussian_chebyshev_data(
        1.0; tolerance=1e-9, max_degree=128)
    @test_throws ArgumentError apply_dll_gaussian_chebyshev(
        Diagonal(ComplexF64[0, nextfloat(1.0)]) |> Matrix,
        ComplexF64[0 1; 1 0], 1.0, unit_data)

    # LAPACK rounds this spectral width just below the exact value 2sqrt(2).
    # Dense interval validation must use an outward certificate, not accept
    # the inward-rounded eigensystem result as a proof of containment.
    inward_hamiltonian = Float64[0 1; 1 2]
    inward_energies = eigvals(Hermitian(inward_hamiltonian))
    inward_radius = maximum(inward_energies) - minimum(inward_energies)
    exact_radius = setprecision(BigFloat, 256) do
        2sqrt(BigFloat(2))
    end
    @test BigFloat(inward_radius) < exact_radius
    certified_radius = @inferred QuantumFurnace._dense_gershgorin_bohr_radius_upper(
        inward_hamiltonian, 1.0)
    @test certified_radius >= exact_radius
    inward_data = dll_gaussian_chebyshev_data(
        inward_radius; tolerance=1e-9, max_degree=128)
    @test_throws ArgumentError apply_dll_gaussian_chebyshev(
        inward_hamiltonian, Float64[0 1; 1 0], 1.0, inward_data)
    certified_data = dll_gaussian_chebyshev_data(
        4.0; tolerance=1e-9, max_degree=128)
    certified_result = apply_dll_gaussian_chebyshev(
        inward_hamiltonian, Float64[0 1; 1 0], 1.0, certified_data)
    certified_exact = exact_dimensionless_filter(
        inward_hamiltonian, Float64[0 1; 1 0], 1.0, q)
    @test norm(certified_result.Q - certified_exact) <=
          2certified_data.q.uniform_error_bound * sqrt(2.0)

    empty_data = dll_gaussian_chebyshev_data(0.0; tolerance=1e-9)
    @test_throws ArgumentError apply_dll_gaussian_chebyshev(
        zeros(0, 0), zeros(0, 0), 1.0, empty_data)
    @test_throws ArgumentError apply_dll_gaussian_chebyshev(
        H + 0.1im * I, A, beta, data)
    bad_source = copy(A)
    bad_source[1, 1] = NaN
    @test_throws ArgumentError apply_dll_gaussian_chebyshev(
        H, bad_source, beta, data)
    big_data = dll_gaussian_chebyshev_data(
        big"1.0"; tolerance=big"1e-20", max_degree=128)
    @test_throws ArgumentError apply_dll_gaussian_chebyshev(
        BigFloat[0 0; 0 1], BigFloat[0 1; 1 0], big"1.0", big_data)
end
