"""
Regression tests with frozen BSON reference data.

DM regression tests compare fresh DM computations against frozen reference density matrices
stored in test/reference/*.bson. Any numerical drift from code changes will cause failures.

Always runs as part of Pkg.test() (fast: load BSON + recompute + compare).
"""

using BSON
using LinearAlgebra: Diagonal, adjoint, norm

# Path resolution for Pkg.test() compatibility
source_root = dirname(@__DIR__)
ref_dir = joinpath(source_root, "test", "reference")

@testset "Regression tests" begin

    @testset "pairwise Hermitian projection and stationary phase" begin
        raw = ComplexF64[
            1 + 2im   2 + 3im
            5 - 7im   4 - 6im
        ]
        expected = (raw + adjoint(raw)) / 2
        projected = copy(raw)
        hermitianize!(projected)

        @test norm(projected - expected) < 1e-14
        @test norm(projected - adjoint(projected)) < 1e-14

        # The canonical projection is used in hot reconstruction loops.
        hermitianize!(projected) # compile before measuring
        allocations = @allocated hermitianize!(projected)
        @test allocations == 0
        @test_throws DimensionMismatch hermitianize!(zeros(ComplexF64, 2, 3))

        rho = ComplexF64[0.7 0.1im; -0.1im 0.3]
        phase_rotated = im .* rho
        QuantumFurnace._normalize_stationary_mode!(phase_rotated)
        @test norm(phase_rotated - rho) < 1e-14
        @test_throws ArgumentError QuantumFurnace._normalize_stationary_mode!(
            zeros(ComplexF64, 2, 2))
        @test_throws ArgumentError QuantumFurnace._normalize_stationary_mode!(
            Matrix(Diagonal(ComplexF64[1, -1 + 1e-15])))
    end

    # Shared initial state for all regression tests
    psi0 = fill(ComplexF64(1.0), N3_DIM) / sqrt(N3_DIM)
    rho0 = psi0 * psi0'

    # ------------------------------------------------------------------
    # DM regression: EnergyDomain
    # ------------------------------------------------------------------
    @testset "DM regression: EnergyDomain" begin
        ref_data = BSON.load(joinpath(ref_dir, "energy_dm_reference.bson"))
        rho_ref = ref_data[:rho]
        delta = ref_data[:delta]

        liouv_config = make_config(Lindbladian(), EnergyDomain(); num_qubits=3, construction=GNS())
        L = construct_lindbladian(N3_JUMPS, liouv_config, N3_HAM)
        rho_fresh = reshape(exp(delta * L) * vec(rho0), N3_DIM, N3_DIM)
        rho_fresh = (rho_fresh + rho_fresh') / 2

        max_err = maximum(abs.(rho_fresh - rho_ref))
        @test isapprox(rho_fresh, rho_ref; atol=1e-10)  # Deterministic DM: exp(delta*L) matches to machine precision; 1e-10 allows FP accumulation in matrix exponential (DIM^2 * eps ~ 64 * 1e-16 ~ 6e-15)
        @info "DM regression (EnergyDomain)" max_element_error=max_err threshold_atol=1e-10
    end

    # ------------------------------------------------------------------
    # DM regression: TrotterDomain (coherent)
    # ------------------------------------------------------------------
    @testset "DM regression: TrotterDomain (coherent)" begin
        ref_data = BSON.load(joinpath(ref_dir, "trotter_coherent_dm_reference.bson"))
        rho_ref = ref_data[:rho]
        delta = ref_data[:delta]

        liouv_config = make_config(Lindbladian(), TrotterDomain(); num_qubits=3, construction=KMS())
        L = construct_lindbladian(N3_TROTTER_JUMPS, liouv_config, N3_HAM; trotter=N3_TROTTER)
        rho_fresh = reshape(exp(delta * L) * vec(rho0), N3_DIM, N3_DIM)
        rho_fresh = (rho_fresh + rho_fresh') / 2

        max_err = maximum(abs.(rho_fresh - rho_ref))
        @test isapprox(rho_fresh, rho_ref; atol=1e-10)  # Deterministic DM with coherent term: same rationale as EnergyDomain DM regression
        @info "DM regression (TrotterDomain, coherent)" max_element_error=max_err threshold_atol=1e-10
    end

    @testset "run_lindblad dense spectral API" begin
        config = make_config(Lindbladian(), BohrDomain(); num_qubits=3, construction=KMS())
        result = run_lindblad(N3_JUMPS, config, N3_HAM)
        L = Matrix{ComplexF64}(construct_lindbladian(N3_JUMPS, config, N3_HAM))
        dense_gap = sort(abs.(real.(eigvals(L))))[2]

        @test result isa LindbladResults
        @test trace_distance_nh(result.fixed_point, Matrix{ComplexF64}(N3_HAM.gibbs)) < 1e-10
        @test isapprox(abs(real(result.spectral_gap)), dense_gap; rtol=1e-8)
    end

end
