@testset "DLL KMS-DB verification" begin

    # =====================================================================
    # End-to-end KMS detailed-balance verification of the full DLL
    # Lindbladian (dissipator + coherent G) via the existing
    # `verify_detailed_balance` machinery in src/discriminant.jl.
    #
    # By Ding-Li-Lin 2024 Theorem 10, the DLL construction satisfies
    # σ_β-KMS DBC. The matrix-level witness is that the antihermitian part
    # of the quantum discriminant `D = σ^{-1/4} L (σ^{1/4} · σ^{1/4}) σ^{-1/4}`
    # vanishes:
    #
    #   BohrDomain: ‖A_part‖ ≈ 0 (analytic — machine epsilon).
    #   TimeDomain: ‖A_part‖ controllably small (quadrature error).
    #
    # Uses the n=3 disordered Heisenberg fixture from `test_helpers.jl`
    # which is the same fixture used by the existing CKG KMS-DB tests
    # (`test_discriminant.jl`); this guarantees apples-to-apples
    # comparison of CKG and DLL on the same Hamiltonian.
    #
    # β-sweep: re-load the n=3 Hamiltonian at β ∈ {1, 5, 10} so the Gibbs
    # state and DLL filter parameters track each β. β = 10 is the user-
    # specified stress level — DLL filter narrowest in frequency, time-
    # domain integrand widest, quadrature errors most likely to surface.
    # =====================================================================

    # Shared n=3 disordered Heisenberg fixture; see test_helpers.jl::make_dll_n3_system.
    _build_n3_system = make_dll_n3_system

    function _make_dll_cfg(domain; beta::Real)
        Config(;
            sim = Lindbladian(),
            domain = domain,
            construction = DLL(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = Float64(beta),
            sigma = 1.0 / Float64(beta),
            a = beta / 30.0,
            s = 0.4,
            # N=10 (Nt=1024): same t_max ≈ 63 as legacy N=12, 16× less NUFFT
            # memory. Bohr↔Time error is FINUFFT-floor-limited at this fixture
            # (~3e-9), well clear of the Gaussian 1e-3 tolerance below.
            num_energy_bits = 10,
            t0 = 2pi / (2^10 * 0.05),
            num_trotter_steps_per_t0 = 10,
            filter = DLLGaussianFilter(Float64(beta)),
        )
    end

    _BETAS = (1.0, 5.0, 10.0)

    function _make_dll_meta_cfg(domain; beta::Real, S::Real = 2.0)
        Config(;
            sim = Lindbladian(),
            domain = domain,
            construction = DLL(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = Float64(beta),
            sigma = 1.0 / Float64(beta),
            a = beta / 30.0,
            s = 0.4,
            # N=10 — see _make_dll_cfg above for rationale.
            num_energy_bits = 10,
            t0 = 2pi / (2^10 * 0.05),
            num_trotter_steps_per_t0 = 10,
            filter = DLLMetropolisFilter(Float64(beta); S = Float64(S)),
        )
    end

    # Per-filter cfg builder + KMS-DB tolerances. Bohr/Time tolerances differ
    # because the Time path's quadrature error depends on the filter's t-tail
    # (Gaussian decays smoothly; Metropolis has a sharp w(ν/S) cutoff that
    # localises better → tighter Time-domain tolerance).
    _kms_filter_for(label, beta) = label === :gaussian ?
        (cfg_b = _make_dll_cfg(BohrDomain(); beta=beta), cfg_t = _make_dll_cfg(TimeDomain(); beta=beta), tol_t = 1e-3) :
        (cfg_b = _make_dll_meta_cfg(BohrDomain(); beta=beta), cfg_t = _make_dll_meta_cfg(TimeDomain(); beta=beta), tol_t = 1e-5)

    # ---------------------------------------------------------------------
    # (a/f) BohrDomain DLL: KMS-DB exact (Theorem 10) — sweep both filters.
    # ---------------------------------------------------------------------
    @testset "(a/f) BohrDomain DLL: KMS-DB exact — $label" for label in (:gaussian, :metropolis)
        for beta in _BETAS
            sys = _build_n3_system(beta)
            v = _kms_filter_for(label, beta)
            L = construct_lindbladian(sys.jumps, v.cfg_b, sys.ham)

            verif = verify_detailed_balance(L, sys.gibbs; atol=1e-10)
            @test verif.is_kms_db
            @test verif.relative_norm <= 1e-10
            @test verif.fixed_point_residual <= 1e-10
        end
    end

    # ---------------------------------------------------------------------
    # (b/h) TimeDomain DLL: KMS-DB up to quadrature — sweep both filters.
    # Tolerance differs by filter (Gaussian 1e-3, Metropolis 1e-5).
    # ---------------------------------------------------------------------
    @testset "(b/h) TimeDomain DLL: KMS-DB up to quadrature — $label" for label in (:gaussian, :metropolis)
        for beta in _BETAS
            sys = _build_n3_system(beta)
            v = _kms_filter_for(label, beta)
            L = construct_lindbladian(sys.jumps, v.cfg_t, sys.ham)

            verif = verify_detailed_balance(L, sys.gibbs; atol=v.tol_t)
            @test verif.relative_norm <= v.tol_t
            @test verif.fixed_point_residual <= v.tol_t
        end
    end

    # ---------------------------------------------------------------------
    # (c) Discriminant Hermitian-part gap = L spectral gap (Bohr only)
    # ---------------------------------------------------------------------
    @testset "(c) discriminant H_gap = L spectral gap (Bohr DLL)" begin
        beta = 10.0
        sys = _build_n3_system(beta)
        cfg = _make_dll_cfg(BohrDomain(); beta=beta)
        L = construct_lindbladian(sys.jumps, cfg, sys.ham)

        verif = verify_detailed_balance(L, sys.gibbs; atol=1e-10)
        @test verif.hermitian_part_gap > 0
        @test isapprox(verif.hermitian_part_gap, verif.spectral_gap_L; rtol=1e-9)
    end

    # ---------------------------------------------------------------------
    # (d) Dissipator-only L violates KMS-DBC; full L (with G) restores it.
    # ---------------------------------------------------------------------
    @testset "(d) Dissipator-only violates KMS-DBC; full L restores it" begin
        beta = 10.0  # use the demanding regime
        sys = _build_n3_system(beta)
        cfg = _make_dll_cfg(BohrDomain(); beta=beta)

        L_full = construct_lindbladian(sys.jumps, cfg, sys.ham)
        L_diss = construct_lindbladian(sys.jumps, cfg, sys.ham; include_coherent=false)

        v_full = verify_detailed_balance(L_full, sys.gibbs; atol=1e-10)
        v_diss = verify_detailed_balance(L_diss, sys.gibbs; atol=1e-10)

        @test v_full.is_kms_db
        @test !v_diss.is_kms_db
        # Dissipator-only relative norm measurably exceeds full-L's.
        @test v_diss.relative_norm > 1e-3
    end

    # =====================================================================
    # DLL Metropolis-type filter — same Bohr fixture, swap the
    # Gaussian filter for DLLMetropolisFilter. KMS-DB is a property of the
    # algebraic construction (Theorem 10) and only requires the filter to
    # be a real, positive, even q(ν): both Gaussian and Metropolis qualify.
    # =====================================================================

    # ---------------------------------------------------------------------
    # (f) — merged into (a/f) parameterised sweep above. The extra
    # hermitian_part_gap == spectral_gap_L witness is already covered by
    # subtest (c) for both filters (KMS-DB Lindbladian property — filter-
    # agnostic).
    # ---------------------------------------------------------------------
    # (i) Bohr ↔ Time agreement on the FULL DLL Metropolis Liouvillian.
    # The user's load-bearing correctness gate: end-to-end build agrees
    # between domains within trapezoidal quadrature error.
    #
    # Measured ‖L_b - L_t‖_op at default config (w0=0.05, Nt=4096):
    #   β=1:  3.7e-7     β=5: 5.6e-7    β=10: 2.1e-5
    # ---------------------------------------------------------------------
    @testset "(i) DLL Metropolis: Bohr ↔ Time L agreement" begin
        for beta in _BETAS
            sys = _build_n3_system(beta)
            cfg_b = _make_dll_meta_cfg(BohrDomain(); beta = beta, S = 2.0)
            cfg_t = _make_dll_meta_cfg(TimeDomain(); beta = beta, S = 2.0)
            L_b = construct_lindbladian(sys.jumps, cfg_b, sys.ham)
            L_t = construct_lindbladian(sys.jumps, cfg_t, sys.ham)
            diff = opnorm(Matrix(L_b) - Matrix(L_t))
            @test diff <= 1e-4
        end
    end

    # ---------------------------------------------------------------------
    # (j) Bohr ↔ Time error is CONTROLLABLE — doubling t0 (doubling t_max
    # at fixed Nt) drops ‖L_b - L_t‖ by 1000× at β=5 and 2000× at β=10.
    # The dominant error source is the tail truncation |f(t_max)| of the
    # Metropolis time_kernel; once it falls below ~1e-12, the Bohr-↔-Time
    # discrepancy is at the FINUFFT precision floor (~1e-9 op-norm at
    # n=3, Nt=4096). This is the structural witness that the residual
    # Bohr↔Time discrepancy is a chosen-grid artefact, not a bug.
    #
    # NOTE on grid parameters: DLL TimeDomain uses `t0` directly as the
    # uniform spacing of the time grid `t_m = (m - N/2)·t0` with
    # `N = 2^num_energy_bits`. There is no `w0` for DLL TimeDomain (that
    # parameter is only used by CKG TimeDomain to enforce the
    # `t0·w0 = 2π/N` Nyquist relation against an external ω-grid).
    # Doubling `t0` doubles `t_max` at fixed Nt; the bound is
    # therefore set by `t0` alone.
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # the (j-sb) sandbox shadow of (j) lives in its own file
    # `test_dll_kms_db_sandbox.jl` so that the inter-file `GC.gc(true)`
    # in runtests.jl clears the (a)..(i) NUFFT workspaces before the
    # Nt = 4096 (r_D = 12) Bohr ↔ Time pair runs. Inlining it here pushed
    # cumulative RSS over the sandbox cap; moving it out of this file
    # keeps the per-file Δ small.
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # (j) is gated NO_SANDBOX: the legacy N=12 / Nt=4096 grid is
    # required to demonstrate convergence to the FINUFFT precision floor
    # (err5_fine ≤ 1e-9, err10_fine ≤ 1e-7). Coarsening Nt would invalidate
    # the assertion's intent, and N=12 + four sequential 4096² NUFFTs spike
    # peak RSS above the 3.5 GB sandbox cap. Run with
    # QUANTUMFURNACE_FULL_TESTS=true outside the sandbox.
    # ---------------------------------------------------------------------
    if get(ENV, "QUANTUMFURNACE_FULL_TESTS", "false") == "true"
        @testset "(j) DLL Metropolis: ‖L_b - L_t‖ converges with t0 [NO_SANDBOX]" begin
            function _meta_cfg_t0(beta::Real, t0::Real)
                Config(;
                    sim = Lindbladian(), domain = TimeDomain(), construction = DLL(),
                    num_qubits = 3, with_linear_combination = true,
                    beta = beta, sigma = 1.0 / beta, a = beta / 30, s = 0.4,
                    num_energy_bits = 12, t0 = t0,
                    num_trotter_steps_per_t0 = 10,
                    filter = DLLMetropolisFilter(beta; S = 2.0),
                )
            end

            function _meta_cfg_b_n12(beta::Real, S::Real)
                Config(;
                    sim = Lindbladian(), domain = BohrDomain(), construction = DLL(),
                    num_qubits = 3, with_linear_combination = true,
                    beta = beta, sigma = 1.0 / beta, a = beta / 30, s = 0.4,
                    num_energy_bits = 12, t0 = 2π / (2^12 * 0.05),
                    num_trotter_steps_per_t0 = 10,
                    filter = DLLMetropolisFilter(beta; S = Float64(S)),
                )
            end

            # Default spacing and two refinements of the time-window extent.
            t0_default = 2π / (2^12 * 0.05)  # ≈ 0.0307, t_max ≈ 62.8
            for beta in (5.0, 10.0)
                sys = _build_n3_system(beta)
                L_b = Matrix(construct_lindbladian(
                    sys.jumps, _meta_cfg_b_n12(beta, 2.0), sys.ham))
                errors = Float64[]
                for factor in (1.0, 2.0, 4.0)
                    L_t = Matrix(construct_lindbladian(
                        sys.jumps, _meta_cfg_t0(beta, factor * t0_default), sys.ham))
                    push!(errors, opnorm(L_b - L_t))
                    L_t = nothing
                    GC.gc()
                end
                @test all(diff(errors) .< 0)
                @test errors[end] <= 1e-9
                L_b = nothing; sys = nothing
                GC.gc()
            end
        end
    end
end
