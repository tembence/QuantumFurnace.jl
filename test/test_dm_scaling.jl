using Test
using LinearAlgebra

# Coherent term B consistency across domains
# B_bohr (exact, Bohr/Energy domain) vs B_time (time quadrature) vs B_trotter (Trotter + time quadrature)
# Expected: B_bohr ~ B_time within TOL_QUADRATURE; B_trotter has additional Trotter error.

@testset "Coherent term B consistency" begin
    jump = TEST_JUMPS[1]  # X on site 1

    # B_bohr (exact, in Hamiltonian eigenbasis)
    config_bohr = make_config(Lindbladian(),BohrDomain())
    precomputed_bohr = QuantumFurnace._precompute_data(config_bohr, TEST_HAM)
    B_bohr_val = B_bohr(TEST_HAM, JumpOp[jump], config_bohr)
    rmul!(B_bohr_val, precomputed_bohr.gamma_norm_factor)

    # B_time (time quadrature, in Hamiltonian eigenbasis)
    config_time = make_config(Lindbladian(),TimeDomain())
    precomputed_time = QuantumFurnace._precompute_data(config_time, TEST_HAM)
    B_time_val = B_time(JumpOp[jump], TEST_HAM, precomputed_time.b_minus, precomputed_time.b_plus,
                        T0, BETA, SIGMA)
    rmul!(B_time_val, precomputed_time.gamma_norm_factor)

    # B_trotter (Trotter + time quadrature, in Trotter eigenbasis)
    # Use pre-built trotter-basis jump (TEST_TROTTER_JUMPS[1] corresponds to TEST_JUMPS[1])
    config_trott = make_config(Lindbladian(),TrotterDomain())
    precomputed_trott = QuantumFurnace._precompute_data(config_trott, TEST_TROTTER)
    trotter_jump = TEST_TROTTER_JUMPS[1]
    B_trott = B_trotter(JumpOp[trotter_jump], TEST_TROTTER, precomputed_trott.b_minus, precomputed_trott.b_plus,
                         BETA, SIGMA)
    rmul!(B_trott, precomputed_trott.gamma_norm_factor)
    # Transform from Trotter eigenbasis to Hamiltonian eigenbasis
    U_t2e = TEST_TROTTER.eigvecs' * TEST_HAM.eigvecs  # Trotter eigenbasis <- H eigenbasis
    B_trott_in_eigen = U_t2e' * B_trott * U_t2e

    # Compute distances
    dist_bohr_time = norm(B_bohr_val - B_time_val)
    dist_bohr_trott = norm(B_bohr_val - B_trott_in_eigen)
    dist_time_trott = norm(B_time_val - B_trott_in_eigen)

    # B_bohr and B_time agree up to time quadrature tolerance
    # Time quadrature error: O(1/N_time_points) with N=4096, well below 1e-6
    @test dist_bohr_time < TOL_QUADRATURE
    @info "B_bohr vs B_time" distance=dist_bohr_time threshold=TOL_QUADRATURE

    # Trotter error on B term (tightened after basis fix in B_trotter)
    # Measured ~1e-8; threshold 1e-5 gives 1000x margin for system-size variation
    @test dist_bohr_trott < 1e-5
    @info "B Trotter total error" distance=dist_bohr_trott threshold=1e-5

    @info "Cross-domain distances" dist_bohr_time dist_bohr_trott dist_time_trott
end

# NUFFT OFT consistency vs analytical Energy-domain reference.
# Verifies the NUFFT-accelerated OFT matches the analytical `oft!` to within
# time-quadrature tolerance (Time domain) and Trotter-error tolerance
# (Trotter domain).

@testset "NUFFT OFT consistency" begin
    jump = TEST_JUMPS[1]  # X on site 1
    w = -3 * W0           # Test energy value (must be on energy grid)

    # --- Analytical OFT (reference) ---
    energy_oft_prefactor = 1 / sqrt(SIGMA * sqrt(2 * pi))
    A_energy = Matrix{ComplexF64}(undef, DIM, DIM)
    oft!(A_energy, jump.in_eigenbasis, TEST_HAM.bohr_freqs, w, 1.0 / (4 * SIGMA^2))
    A_energy .*= energy_oft_prefactor

    time_oft_prefactor = T0 * sqrt(SIGMA * sqrt(2 / pi) / (2 * pi))

    # === Time NUFFT OFT ===
    config_time = make_config(Lindbladian(),TimeDomain())
    precomputed_time = QuantumFurnace._precompute_data(config_time, TEST_HAM)

    # Sanity: test energy must be on the NUFFT grid (structural check, no @info)
    @test haskey(precomputed_time.oft_nufft_prefactors.energy_to_index, w)

    nufft_pf_time = QuantumFurnace._prefactor_view(precomputed_time.oft_nufft_prefactors, w)
    A_nufft_time = jump.in_eigenbasis .* nufft_pf_time
    A_nufft_time .*= time_oft_prefactor

    # === Trotter NUFFT OFT ===
    config_trott = make_config(Lindbladian(),TrotterDomain())
    precomputed_trott = QuantumFurnace._precompute_data(config_trott, TEST_TROTTER)

    # Sanity: test energy must be on the NUFFT grid (structural check, no @info)
    @test haskey(precomputed_trott.oft_nufft_prefactors.energy_to_index, w)

    jump_trott = TEST_TROTTER_JUMPS[1]
    U_t2e = TEST_TROTTER.eigvecs' * TEST_HAM.eigvecs

    nufft_pf_trott = QuantumFurnace._prefactor_view(precomputed_trott.oft_nufft_prefactors, w)
    A_nufft_trott = jump_trott.in_eigenbasis .* nufft_pf_trott
    A_nufft_trott .*= time_oft_prefactor
    A_nufft_trott_in_eigen = U_t2e' * A_nufft_trott * U_t2e

    # Compute distances
    dist_nufft_time_vs_energy = norm(A_nufft_time - A_energy)
    dist_nufft_trott_vs_energy = norm(A_nufft_trott_in_eigen - A_energy)

    # NUFFT time OFT matches analytical within time-quadrature error
    # (O(1/N_time_points) with N=4096, well below TOL_QUADRATURE).
    @test dist_nufft_time_vs_energy < TOL_QUADRATURE
    @info "NUFFT time vs analytical" distance=dist_nufft_time_vs_energy threshold=TOL_QUADRATURE

    # NUFFT Trotter OFT error (measured ~1.5e-8; threshold 1e-5 gives ~1000x
    # margin for system-size variation; Trotter error sits on top of quadrature error).
    @test dist_nufft_trott_vs_energy < 1e-5
    @info "NUFFT trotter vs analytical" distance=dist_nufft_trott_vs_energy threshold=1e-5
end
