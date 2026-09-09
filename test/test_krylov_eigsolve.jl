using Test
using LinearAlgebra
using Random
using QuantumFurnace

# test_helpers.jl is already included by runtests.jl

# ============================================================================
# Spectral-gap accuracy, channel conversion, guard rails, symmetry breaking,
# and workspace ownership for the matrix-free eigensolver.
# ============================================================================

@testset "Krylov Eigsolve" begin

    # ========================================================================
    # Testset 2: krylov_spectral_gap result fields
    # ========================================================================
    @testset "krylov_spectral_gap result fields" begin
        config_kms = make_config(Lindbladian(),EnergyDomain(); construction=KMS())
        result = krylov_spectral_gap(config_kms, TEST_HAM, TEST_JUMPS;
            krylovdim=30, howmany=4)

        @test result isa NamedTuple
        @test length(result.eigenvalues) >= 2
        @test result.spectral_gap > 0
        @test size(result.fixed_point) == (DIM, DIM)
        @test size(result.gap_mode) == (DIM, DIM)
        @test result.converged >= 2
        @test result.matvec_count > 0
        @test result.channel_eigenvalues === nothing
        @test result.delta_used === nothing
        @info "krylov_spectral_gap result fields" spectral_gap=result.spectral_gap n_eigenvalues=length(result.eigenvalues) converged=result.converged matvec_count=result.matvec_count
    end

    # ========================================================================
    # Testset 3: Lindbladian eigsolve accuracy (EnergyDomain KMS)
    # ========================================================================
    @testset "Lindbladian eigsolve accuracy (EnergyDomain KMS)" begin
        config = make_config(Lindbladian(),EnergyDomain(); construction=KMS())
        L_dense = construct_lindbladian(TEST_JUMPS, config, TEST_HAM)
        dense_result = extract_leading_eigendata(L_dense; n_modes=4)

        result = krylov_spectral_gap(config, TEST_HAM, TEST_JUMPS;
            krylovdim=30, howmany=4)

        gap_err = abs(result.spectral_gap - dense_result.spectral_gap) / dense_result.spectral_gap
        @test isapprox(result.spectral_gap, dense_result.spectral_gap; rtol=1e-8)
        @info "Eigsolve gap accuracy (EnergyDomain KMS)" krylov_gap=result.spectral_gap dense_gap=dense_result.spectral_gap relative_error=gap_err rtol=1e-8

        td = trace_distance_h(Hermitian(result.fixed_point), TEST_GIBBS)
        @test td < 1e-8
        @info "Fixed point trace distance to Gibbs" trace_distance=td threshold=1e-8

        ss_err = abs(result.eigenvalues[1])
        @test ss_err < 1e-10
        @info "Steady-state eigenvalue magnitude" abs_lambda1=ss_err threshold=1e-10
    end

    # ========================================================================
    # Testset 4: Lindbladian eigsolve accuracy (EnergyDomain GNS)
    # ========================================================================
    @testset "Lindbladian eigsolve accuracy (EnergyDomain GNS)" begin
        config = make_config(Lindbladian(), EnergyDomain(); construction=GNS())
        L_dense = construct_lindbladian(TEST_JUMPS, config, TEST_HAM)
        dense_result = extract_leading_eigendata(L_dense; n_modes=4)

        result = krylov_spectral_gap(config, TEST_HAM, TEST_JUMPS;
            krylovdim=30, howmany=4)

        gap_err = abs(result.spectral_gap - dense_result.spectral_gap) / dense_result.spectral_gap
        @test isapprox(result.spectral_gap, dense_result.spectral_gap; rtol=1e-8)
        @info "Eigsolve gap accuracy (EnergyDomain GNS)" krylov_gap=result.spectral_gap dense_gap=dense_result.spectral_gap relative_error=gap_err rtol=1e-8

        trace_err = abs(tr(result.fixed_point) - 1.0)
        @test isapprox(tr(result.fixed_point), 1.0; atol=1e-10)
        @test norm(result.fixed_point - result.fixed_point') < 1e-10
        @info "GNS fixed point normalization" trace=real(tr(result.fixed_point)) trace_error=trace_err atol=1e-10

        ss_err = abs(result.eigenvalues[1])
        @test ss_err < 1e-10
        @info "Steady-state eigenvalue magnitude (GNS)" abs_lambda1=ss_err threshold=1e-10
    end

    # ========================================================================
    # Testset 7: Guard rails
    # ========================================================================
    @testset "Guard rails" begin
        config = make_config(Lindbladian(),EnergyDomain(); construction=KMS())

        # krylovdim <= howmany must error
        @test_throws ErrorException krylov_spectral_gap(config, TEST_HAM, TEST_JUMPS;
            krylovdim=2, howmany=4)

        # Memory guard should not throw for small n=4 system
        result = krylov_spectral_gap(config, TEST_HAM, TEST_JUMPS;
            krylovdim=50, howmany=2)
        @test result isa NamedTuple
    end

    # ========================================================================
    # Testset 8: Eigenvalue sorting and conversion
    # ========================================================================
    @testset "Eigenvalue sorting and conversion" begin
        # Lindbladian path: eigenvalues sorted by |Re(lambda)| ascending (structural check)
        config = make_config(Lindbladian(),EnergyDomain(); construction=KMS())
        result_liouv = krylov_spectral_gap(config, TEST_HAM, TEST_JUMPS;
            krylovdim=30, howmany=4)

        @test abs(real(result_liouv.eigenvalues[1])) <= abs(real(result_liouv.eigenvalues[2]))
        @info "Eigenvalue sorting" abs_Re_lambda1=abs(real(result_liouv.eigenvalues[1])) abs_Re_lambda2=abs(real(result_liouv.eigenvalues[2]))

        # Channel path: the stationary mode is pinned first, remaining raw
        # modes are ordered by decreasing |mu|, the reported gap is the exact
        # discrete log-rate, and the legacy converted spectrum remains
        # algebraically consistent.
        config_therm = make_config(Thermalize(),EnergyDomain();
            construction=KMS(), delta=0.01)
        result_chan = krylov_spectral_gap(config_therm, TEST_HAM, TEST_JUMPS;
            krylovdim=30, howmany=4)
        @test result_chan.channel_eigenvalues !== nothing
        @test result_chan.delta_used == config_therm.delta
        @test abs(abs(result_chan.channel_eigenvalues[1]) - 1.0) < 1e-10
        @test argmin(abs.(result_chan.channel_eigenvalues .- 1)) == 1
        @test issorted(abs.(result_chan.channel_eigenvalues[2:end]); rev=true)
        mu2_abs = abs(result_chan.channel_eigenvalues[2])
        expected_channel_gap = mu2_abs > 0 ?
            -log(mu2_abs) / result_chan.delta_used : Inf
        @test isapprox(result_chan.spectral_gap, expected_channel_gap;
            atol=1e-12, rtol=1e-12)
        @test result_chan.trace_preserving_assumed
        @test result_chan.physical_channel
        @test result_chan.channel_representation === :deterministic_cptp

        # Threshold rationale (atol=1e-10): conversion mu = 1 + delta * lambda_L is algebraically
        # exact (just arithmetic). Error is FP rounding only: O(eps * |mu|) ~ 1e-16.
        # Threshold 1e-10 gives >1e6 margin.
        reconstructed_mu = 1.0 .+ result_chan.delta_used .* result_chan.eigenvalues
        conversion_err = maximum(abs.(reconstructed_mu .- result_chan.channel_eigenvalues))
        @test isapprox(reconstructed_mu, result_chan.channel_eigenvalues; atol=1e-10)
        @info "Eigenvalue conversion consistency" max_conversion_error=conversion_err atol=1e-10
    end

    @testset "Channel ordering uses decreasing modulus" begin
        # A generic complex-linear map with an amplifying mode distinguishes
        # decreasing |mu| from the old distance-to-one ordering.
        D = Diagonal(ComplexF64[1.0, 0.99, 1.02, 0.8im])
        apply_diagonal!(out, X) =
            (copyto!(out, reshape(D * vec(X), 2, 2)); out)
        rho_seed = ComplexF64[1 2 + im; 3 - im 4]
        decomp = QuantumFurnace._krylov_spectral_decomposition(
            apply_diagonal!, rho_seed, 2;
            krylovdim=4, sort_mode=:channel,
            assume_trace_preserving=false)
        @test issorted(abs.(decomp.eigenvalues); rev=true)
        @test isapprox(abs(decomp.eigenvalues[1]), 1.02; atol=1e-12, rtol=0)
        @test !decomp.trace_preserving_assumed
    end

    @testset "Arnoldi basis views are isolated from forward-operator mutation" begin
        # `_arnoldi_factorize` passes a basis-column view into `fwd_vec`, which
        # copies it into a private matrix buffer before invoking the operator.
        # Deliberately destroying that buffer must therefore leave Q intact.
        D = Diagonal(ComplexF64[1.02, 0.99, 0.8im, 0.5])
        function apply_destructive_diagonal!(out, X)
            copyto!(out, reshape(D * vec(X), 2, 2))
            fill!(X, ComplexF64(NaN, NaN))
            return out
        end
        rho_seed = ComplexF64[1 2 + im; 3 - im 4]
        rho_before = copy(rho_seed)
        decomp = QuantumFurnace._krylov_spectral_decomposition(
            apply_destructive_diagonal!, rho_seed, 2;
            krylovdim=4, sort_mode=:channel,
            assume_trace_preserving=false)

        @test rho_seed == rho_before
        @test isapprox(decomp.eigenvalues, diag(D); atol=1e-12, rtol=1e-12)
        @test decomp.matvec_count == 4
    end

    @testset "Periodic CPTP channel pins the stationary mode" begin
        # Unitary bit-flip conjugation Phi(rho) = X*rho*X is CPTP and has a
        # period-two mode mu=-1. Its orbit from this diagonal state spans the
        # stationary I/2 component and exactly one alternating component.
        bit_flip = ComplexF64[0 1; 1 0]
        apply_bit_flip!(out, rho) =
            (copyto!(out, bit_flip * rho * bit_flip); out)
        rho_seed = ComplexF64[0.9 0; 0 0.1]
        decomp = QuantumFurnace._krylov_spectral_decomposition(
            apply_bit_flip!, rho_seed, 2;
            krylovdim=4, sort_mode=:channel,
            assume_trace_preserving=true)

        @test isapprox(decomp.eigenvalues[1], 1; atol=1e-12, rtol=0)
        @test isapprox(decomp.eigenvalues[2], -1; atol=1e-12, rtol=0)
        @test isapprox(decomp.rho_inf, Matrix{ComplexF64}(I(2) / 2);
            atol=1e-12, rtol=0)
        @test isapprox(
            QuantumFurnace._channel_spectral_gap(decomp.eigenvalues, 0.37),
            0.0; atol=1e-12, rtol=0)
    end

    # ========================================================================
    # Classical Ising is invariant under translation and global spin flip.
    # Arnoldi seeded with I/d stays in the symmetric sector and can miss
    # the slow magnetisation mode. The default traceless random perturbation
    # gives the seed overlap with other symmetry sectors. Compare its gap
    # estimate against the full dense spectrum on this small fixture.
    # ========================================================================
    @testset "krylov_spectral_gap — symmetric system regression" begin
        system = make_classical_ising_n3()
        (; ham, jumps) = system
        cfg_e = make_classical_ising_config(Lindbladian(), system)

        # Dense reference: build L explicitly, take the |Re|-second smallest.
        L_dense = construct_lindbladian(jumps, cfg_e, ham)
        ev_dense = eigvals(L_dense)
        perm = sortperm(real.(ev_dense); by=abs)
        gap_dense = abs(real(ev_dense[perm[2]]))

        # Krylov must match the independent dense spectrum.
        res = krylov_spectral_gap(cfg_e, ham, jumps;
                                  krylovdim=40, howmany=4)
        @test isapprox(res.spectral_gap, gap_dense; rtol=1e-8)
        @info "classical Ising regression" n=3 beta_phys=system.beta_phys gap_krylov=res.spectral_gap gap_dense=gap_dense rel_err=abs(res.spectral_gap - gap_dense)/gap_dense

        # Also exercise BohrDomain — same physical Lindbladian, different domain wiring.
        cfg_b = make_classical_ising_config(Lindbladian(), system; domain=BohrDomain())
        res_b = krylov_spectral_gap(cfg_b, ham, jumps;
                                    krylovdim=40, howmany=4)
        @test isapprox(res_b.spectral_gap, gap_dense; rtol=1e-8)
        # Energy ≡ Bohr to machine precision for classical Ising (no quadrature error).
        @test isapprox(res.spectral_gap, res_b.spectral_gap; rtol=1e-10)
        @info "Energy ≡ Bohr cross-check" gap_E=res.spectral_gap gap_B=res_b.spectral_gap
    end

    # ========================================================================
    # krylov_spectral_gap `workspace=` reuse.
    #
    # The two numerical pipelines (trajectory/mixing-time vs spectral-gap/
    # spectrum) are decoupled, sharing only the one expensive resource — the
    # O(d^3.x) Workspace operator build. krylov_spectral_gap now accepts a
    # pre-built Workspace so the compute_true_gap=true trajectory path can
    # forward its own and skip the redundant second build. This testset asserts:
    #   (a) reuse is bit-identical to a fresh build (Lindbladian + channel);
    #   (b) the supplied Workspace is actually CONSULTED (a mismatched config /
    #       scratch type / dim throws cleanly — proving no silent fresh rebuild,
    #       i.e. the double-build is gone);
    #   (c) end-to-end, predict_*_trajectory(compute_true_gap=true) with a
    #       forwarded Workspace equals the self-build path bitwise.
    # n=3 throughout — sandbox-cheap.
    # ========================================================================
    @testset "krylov_spectral_gap workspace= reuse" begin
        d3 = 2^3

        # -- (a) Lindbladian: workspace= reuse is bit-identical to fresh build --
        @testset "(a) Lindbladian reuse byte-equality" begin
            cfg = make_config(Lindbladian(), EnergyDomain(); num_qubits=3, construction=KMS())
            r_fresh = krylov_spectral_gap(cfg, N3_HAM, N3_JUMPS;
                                          krylovdim=30, howmany=4)
            ws = Workspace(cfg, N3_HAM, N3_JUMPS)
            r_reuse1 = krylov_spectral_gap(cfg, N3_HAM, N3_JUMPS;
                                           krylovdim=30, howmany=4, workspace=ws)
            # Reuse a second time on the SAME ws — no scratch contamination.
            r_reuse2 = krylov_spectral_gap(cfg, N3_HAM, N3_JUMPS;
                                           krylovdim=30, howmany=4, workspace=ws)
            r_copy = krylov_spectral_gap(cfg, N3_HAM, copy(N3_JUMPS);
                                         krylovdim=30, howmany=4, workspace=ws)
            @test r_fresh.spectral_gap == r_reuse1.spectral_gap
            @test r_reuse1.spectral_gap == r_reuse2.spectral_gap
            @test r_reuse2.spectral_gap == r_copy.spectral_gap
            @test r_fresh.eigenvalues == r_reuse1.eigenvalues
            @test r_fresh.fixed_point == r_reuse1.fixed_point
            @test r_fresh.gap_mode == r_reuse1.gap_mode
            @test r_fresh.matvec_count == r_reuse1.matvec_count
            @test r_reuse1.spectral_modes.off_diag_weight ==
                  r_fresh.spectral_modes.off_diag_weight
        end

        # -- (a') Channel: workspace= reuse is bit-identical to fresh build --
        @testset "(a') Channel reuse byte-equality" begin
            cfg_ch = make_config(Thermalize(), EnergyDomain();
                                 num_qubits=3, construction=KMS(), delta=0.01)
            rc_fresh = krylov_spectral_gap(cfg_ch, N3_HAM, N3_JUMPS;
                                           krylovdim=30, howmany=4)
            ws_ch = Workspace(cfg_ch, N3_HAM, N3_JUMPS)
            rc_reuse1 = krylov_spectral_gap(cfg_ch, N3_HAM, N3_JUMPS;
                                            krylovdim=30, howmany=4, workspace=ws_ch)
            rc_reuse2 = krylov_spectral_gap(cfg_ch, N3_HAM, N3_JUMPS;
                                            krylovdim=30, howmany=4, workspace=ws_ch)
            @test rc_fresh.spectral_gap == rc_reuse1.spectral_gap
            @test rc_reuse1.spectral_gap == rc_reuse2.spectral_gap
            @test rc_fresh.eigenvalues == rc_reuse1.eigenvalues
            @test rc_fresh.channel_eigenvalues == rc_reuse1.channel_eigenvalues
            @test rc_fresh.matvec_count == rc_reuse1.matvec_count
        end

        # -- (b) The supplied Workspace is CONSULTED, not silently rebuilt.
        #        A mismatched config / scratch type / dim must throw — if the
        #        method ignored `workspace=` and built a fresh one, none of
        #        these would throw. This is the "no double-build" witness: the
        #        forwarded ws is on the live path.
        @testset "(b) reuse guards (workspace is consulted)" begin
            cfg = make_config(Lindbladian(), EnergyDomain(); num_qubits=3, construction=KMS())
            cfg_ch = make_config(Thermalize(), EnergyDomain();
                                 num_qubits=3, construction=KMS(), delta=0.01)
            cfg_ch2 = make_config(Thermalize(), EnergyDomain();
                                  num_qubits=3, construction=KMS(), delta=0.02)  # delta differs

            ws_L  = Workspace(cfg, N3_HAM, N3_JUMPS)
            ws_ch = Workspace(cfg_ch, N3_HAM, N3_JUMPS)

            # Config mismatch (channel ws built at δ=0.02, called at δ=0.01).
            @test_throws ArgumentError krylov_spectral_gap(
                cfg_ch, N3_HAM, N3_JUMPS;
                krylovdim=30, howmany=4, workspace=Workspace(cfg_ch2, N3_HAM, N3_JUMPS))
            # Cross-simulation: channel ws into the Lindbladian method.
            @test_throws ArgumentError krylov_spectral_gap(
                cfg, N3_HAM, N3_JUMPS; krylovdim=30, howmany=4, workspace=ws_ch)
            # Cross-simulation: Lindbladian ws into the channel method.
            @test_throws ArgumentError krylov_spectral_gap(
                cfg_ch, N3_HAM, N3_JUMPS; krylovdim=30, howmany=4, workspace=ws_L)

            # Same dimension/config is insufficient: cached physics must come
            # from the exact Hamiltonian and jump vector supplied at the call.
            ham_other = QuantumFurnace._load_hamiltonian_bson(
                test_hamiltonian_path(3), BETA)
            @test_throws ArgumentError krylov_spectral_gap(
                cfg, ham_other, N3_JUMPS; krylovdim=30, howmany=4, workspace=ws_L)

            jumps_other = copy(N3_JUMPS)
            j1 = jumps_other[1]
            jumps_other[1] = JumpOp(2 .* j1.data, 2 .* j1.in_eigenbasis,
                                    j1.orthogonal, j1.hermitian)
            @test_throws ArgumentError krylov_spectral_gap(
                cfg, N3_HAM, jumps_other; krylovdim=30, howmany=4, workspace=ws_L)

            cfg_t = make_config(Lindbladian(), TrotterDomain();
                                num_qubits=3, construction=GNS())
            ws_t = Workspace(cfg_t, N3_HAM, N3_TROTTER_JUMPS;
                             trotter=N3_TROTTER)
            other = make_classical_ising_n3()
            cfg_other = Config(
                sim=Lindbladian(), domain=TrotterDomain(), construction=GNS(),
                num_qubits=3, with_linear_combination=true,
                beta=other.beta_alg, beta_phys=other.beta_phys,
                sigma=1 / other.beta_alg, a=0.0, s=0.25,
                num_energy_bits=NUM_ENERGY_BITS, t0=T0, w0=W0,
                num_trotter_steps_per_t0=NUM_TROTTER_STEPS_PER_T0,
            )
            @test_throws ArgumentError krylov_spectral_gap(
                cfg_other, other.ham, N3_TROTTER_JUMPS;
                trotter=N3_TROTTER, krylovdim=30, howmany=4, workspace=ws_t)
        end

        # -- (c) End-to-end: predict_*_trajectory(compute_true_gap=true) with a
        #        forwarded Workspace == the self-build path, bitwise. This is the
        #        production wiring (krylov_dynamics.jl Pass-2 forwards ws); proves
        #        the double-build elimination changed cost, not the number.
        @testset "(c) predict_* compute_true_gap forwarded ws == self-build" begin
            psi = ones(ComplexF64, d3) ./ sqrt(2.0^3)
            rho_0 = psi * psi'

            cfg = make_config(Lindbladian(), EnergyDomain(); num_qubits=3, construction=KMS())
            t_grid = collect(range(0.0, 1.0, length=3))
            # Self-build path (workspace defaulted to nothing inside predict_*).
            tj_self = predict_lindbladian_trajectory(
                cfg, N3_HAM, N3_JUMPS, rho_0, t_grid;
                krylovdim=30, compute_true_gap=true)
            # Forwarded-workspace path: predict_* reuses this ws for BOTH Pass-1
            # and the internal Pass-2 krylov_spectral_gap.
            ws = Workspace(cfg, N3_HAM, N3_JUMPS)
            tj_ws = predict_lindbladian_trajectory(
                cfg, N3_HAM, N3_JUMPS, rho_0, t_grid;
                krylovdim=30, compute_true_gap=true, workspace=ws)
            @test tj_self.spectral_gap == tj_ws.spectral_gap
            @test tj_self.total_matvecs == tj_ws.total_matvecs
            @test tj_self.all_converged == tj_ws.all_converged

            cfg_ch = make_config(Thermalize(), EnergyDomain(); num_qubits=3, construction=KMS())
            k_grid = collect(0:5:20)
            rho_init = Matrix{ComplexF64}(rho_0)
            tc_self = predict_channel_trajectory(
                cfg_ch, N3_HAM, N3_JUMPS, rho_init, k_grid;
                krylovdim=30, compute_true_gap=true)
            ws_ch = Workspace(cfg_ch, N3_HAM, N3_JUMPS)
            tc_ws = predict_channel_trajectory(
                cfg_ch, N3_HAM, N3_JUMPS, rho_init, k_grid;
                krylovdim=30, compute_true_gap=true, workspace=ws_ch)
            @test tc_self.spectral_gap == tc_ws.spectral_gap
            @test tc_self.total_matvecs == tc_ws.total_matvecs
            @test tc_self.all_converged == tc_ws.all_converged
        end
    end

end  # @testset "Krylov Eigsolve"
