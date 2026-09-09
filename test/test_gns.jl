"""
GNS fixed-point and channel-construction validation tests.

Verifies that the GNS Lindbladian fixed point is a valid density matrix and is
distinct from the exact Gibbs state, and that the retained per-jump channel is
CPTP without a coherent correction.

The GNS (approximate detailed balance) code path uses unshifted transition weights
and omits the coherent B term. Its fixed point approximates but does not equal the
exact Gibbs state -- the approximation gap is documented here as a baseline.
"""

@testset "Lindbladian fixed point (TrotterDomain)" begin
    config = make_config(Lindbladian(), TrotterDomain(); num_qubits=3, construction=GNS())
    liouv = construct_lindbladian(N3_TROTTER_JUMPS, config, N3_HAM; trotter=N3_TROTTER)

    # Full eigendecomposition (64x64 dense matrix)
    eig = eigen(liouv)

    # Extract fixed point: eigenvalue with smallest |Re(lambda)|
    ss_idx = argmin(abs.(real.(eig.values)))
    ss_vec = eig.vectors[:, ss_idx]
    ss_dm = reshape(ss_vec, N3_DIM, N3_DIM)
    ss_dm = (ss_dm + ss_dm') / 2   # Hermitianize
    ss_dm ./= tr(ss_dm)            # Normalize

    # Validate fixed point is a valid density matrix (structural checks, atol=1e-12 is machine precision)
    @test isapprox(tr(ss_dm), 1.0, atol=1e-12)
    @test isapprox(ss_dm, ss_dm', atol=1e-12)   # Hermitian
    @test all(eigvals(Hermitian(ss_dm)) .>= -1e-12)  # PSD

    # GNS fixed point is NOT the exact Gibbs state -- measure the approximation gap.
    # The Lindbladian is built in the Trotter eigenbasis, so transform the fixed point
    # to the energy eigenbasis before comparing with N3_GIBBS.
    U_t2e = N3_HAM.eigvecs' * N3_TROTTER.eigvecs  # Trotter-to-energy change of basis
    ss_dm_energy = U_t2e * ss_dm * U_t2e'
    ss_dm_energy = (ss_dm_energy + ss_dm_energy') / 2  # re-Hermitianize after basis change
    gap = trace_distance_h(Hermitian(ss_dm_energy), N3_GIBBS)
    @info "GNS fixed point to Gibbs trace distance (approximation gap)" gap

    # GNS approximate detailed balance: gap is strictly positive because GNS omits the
    # coherent B term and uses unshifted weights. Gap magnitude is system-dependent.
    @test gap > 1e-6     # Strictly positive (GNS does not reproduce exact Gibbs)
    @info "Gap lower bound" gap lower_bound=1e-6

    # Sanity bound: gap should be moderate, not wildly wrong. Empirically ~0.08 for this system.
    @test gap < 0.5      # Sanity bound (should be ~0.08, close to EnergyDomain)
    @info "Gap upper bound" gap upper_bound=0.5
end

@testset "CPTP completeness (TrotterDomain)" begin
    config = make_config(Thermalize(), TrotterDomain(); num_qubits=3, construction=GNS(), delta=0.01)
    precomputed_data = QuantumFurnace._precompute_data(config, N3_TROTTER)
    (; K0s, U_residuals) = QuantumFurnace._precompute_per_jump_channels(
        N3_TROTTER_JUMPS, N3_TROTTER, config, precomputed_data;
        rescale_by_inv_prob=false,
    )
    U_coherents = QuantumFurnace._precompute_coherent_unitary(
        N3_TROTTER_JUMPS, N3_HAM, config, precomputed_data;
        trotter=N3_TROTTER,
    )

    n_jumps = length(N3_TROTTER_JUMPS)
    @test length(K0s) == n_jumps
    @test U_coherents === nothing

    identity = Matrix{ComplexF64}(I, N3_DIM, N3_DIM)
    builder_scratch = QuantumFurnace.ThermalizeScratch(ComplexF64, N3_DIM)
    # CPTP completeness: K0'K0 + delta*R + U'U = I (algebraic identity)
    # atol=1e-10 allows for FP accumulation across DIM^2 matrix entries
    max_completeness_err = 0.0
    for a in 1:n_jumps
        R_a = copy(QuantumFurnace._precompute_R(
            [N3_TROTTER_JUMPS[a]], N3_TROTTER, config,
            precomputed_data, builder_scratch,
        ))
        completeness = K0s[a]' * K0s[a] + config.delta * R_a +
            U_residuals[a]' * U_residuals[a]
        err = norm(completeness - identity)
        max_completeness_err = max(max_completeness_err, err)
        @test isapprox(completeness, identity; atol=1e-10)
    end
    @info "CPTP completeness (TrotterDomain, GNS)" n_jumps max_error=max_completeness_err threshold_atol=1e-10
end

@testset "BohrDomain detailed balance" begin
    config = make_config(Lindbladian(), BohrDomain(); num_qubits=3, construction=GNS())
    liouv = construct_lindbladian(N3_JUMPS, config, N3_HAM)

    # Full eigendecomposition
    eig = eigen(liouv)
    ss_idx = argmin(abs.(real.(eig.values)))
    ss_vec = eig.vectors[:, ss_idx]
    ss_dm_bohr = reshape(ss_vec, N3_DIM, N3_DIM)
    ss_dm_bohr = (ss_dm_bohr + ss_dm_bohr') / 2
    ss_dm_bohr ./= tr(ss_dm_bohr)

    # Validate fixed point is a valid density matrix (structural checks, machine precision)
    @test isapprox(tr(ss_dm_bohr), 1.0, atol=1e-12)
    @test isapprox(ss_dm_bohr, ss_dm_bohr', atol=1e-12)   # Hermitian
    @test all(eigvals(Hermitian(ss_dm_bohr)) .>= -1e-12)   # PSD

    # GNS fixed point should be distinct from Gibbs
    # GNS approximate detailed balance: strictly positive gap, system-dependent magnitude
    gap_bohr = trace_distance_h(Hermitian(ss_dm_bohr), N3_GIBBS)
    @info "Bohr: GNS fixed point to Gibbs distance" gap_bohr
    @test gap_bohr > 1e-6  # Strictly positive
    @info "Bohr: Gap lower bound" gap_bohr lower_bound=1e-6
end
