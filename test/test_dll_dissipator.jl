@testset "DLL dissipator" begin

    # =====================================================================
    # The full DLL Lindbladian L = D + i[G, ·] preserves σ_β in BohrDomain.
    # Test a three-qubit disordered Heisenberg fixture at β ∈ {1, 5, 10}.
    # The DLL time kernel broadens with β, requiring a wider quadrature grid.
    # =====================================================================
    # Shared n=3 disordered Heisenberg fixture; see test_helpers.jl::make_dll_n3_system.
    _build_dll_n3_system = make_dll_n3_system

    # All Configs share these grid parameters. Half-grid t_max = 62.83,
    # which exceeds the DLL filter cutoff at β=10 (~35.5) by ~2x — sufficient
    # margin for trapezoidal quadrature to converge to ≤1e-4 against the
    # exact Bohr decomposition. N=10 (Nt=1024) reaches the FINUFFT precision
    # floor for Bohr↔Time at this fixture (~3e-9) — bumping to N=12
    # gains nothing, but uses 16× more NUFFT memory.
    _DLL_NUM_ENERGY_BITS = 10
    _DLL_W0 = 0.05
    _DLL_T0 = 2pi / (2^_DLL_NUM_ENERGY_BITS * _DLL_W0)
    _DLL_BETAS = (1.0, 5.0, 10.0)

    function _make_dll_config(domain; beta::Real)
        Config(;
            sim = Lindbladian(),
            domain = domain,
            construction = DLL(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta,
            sigma = 1.0 / beta,
            a = beta / 30.0,
            s = 0.4,
            num_energy_bits = _DLL_NUM_ENERGY_BITS,
            t0 = _DLL_T0,
            num_trotter_steps_per_t0 = 10,
            filter = DLLGaussianFilter(beta),
        )
    end

    # ---------------------------------------------------------------------
    # (c) Bohr ↔ Time agreement: trapezoidal quadrature should reproduce
    # the Bohr-decomposition Liouvillian within quadrature error.
    # ---------------------------------------------------------------------
    @testset "(c) Bohr ↔ Time consistency" begin
        for beta in _DLL_BETAS
            sys = _build_dll_n3_system(beta)
            liouv_b = construct_lindbladian(sys.jumps, _make_dll_config(BohrDomain(); beta=beta), sys.ham)
            liouv_t = construct_lindbladian(sys.jumps, _make_dll_config(TimeDomain(); beta=beta), sys.ham)
            @test opnorm(liouv_b - liouv_t) <= 1e-4
            @test norm(liouv_b - liouv_t) <= 1e-4
        end
    end

    # ---------------------------------------------------------------------
    # (d) Trace preservation in dual: L†[I] = 0 by construction
    # ---------------------------------------------------------------------
    @testset "(d) Dual trace preservation L†[I] ≈ 0" begin
        Id_vec = ComplexF64.(vec(Matrix(I, 8, 8)))     # n=3 ⇒ 2^3 = 8
        for beta in _DLL_BETAS
            sys = _build_dll_n3_system(beta)

            liouv_b = construct_lindbladian(sys.jumps, _make_dll_config(BohrDomain(); beta=beta), sys.ham)
            @test norm(liouv_b' * Id_vec) <= 1e-10

            liouv_t = construct_lindbladian(sys.jumps, _make_dll_config(TimeDomain(); beta=beta), sys.ham)
            @test norm(liouv_t' * Id_vec) <= 1e-4
        end
    end

    # ---------------------------------------------------------------------
    # (e) validate_config!: DLL without an explicit filter must throw.
    # ---------------------------------------------------------------------
    @testset "(e) validate_config! rejects DLL with filter=nothing" begin
        bad = Config(;
            sim = Lindbladian(),
            domain = BohrDomain(),
            construction = DLL(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = 1.0,
            sigma = 1.0,
            a = 1.0 / 30.0,
            s = 0.4,
            num_energy_bits = _DLL_NUM_ENERGY_BITS,
            filter = nothing,
        )
        try
            validate_config!(bad)
            @test false  # should have thrown
        catch e
            @test e isa ArgumentError
            @test occursin("DLL construction requires an explicit AbstractFilter", e.msg)
        end
    end

    # ---------------------------------------------------------------------
    # (f) validate_config!: TrotterDomain DLL is unsupported.
    # ---------------------------------------------------------------------
    @testset "(f) validate_config! rejects TrotterDomain DLL" begin
        bad = Config(;
            sim = Lindbladian(),
            domain = TrotterDomain(),
            construction = DLL(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = 1.0,
            sigma = 1.0,
            a = 1.0 / 30.0,
            s = 0.4,
            num_energy_bits = _DLL_NUM_ENERGY_BITS,
            w0 = _DLL_W0,
            t0 = _DLL_T0,
            num_trotter_steps_per_t0 = 10,
            filter = DLLGaussianFilter(1.0),
        )
        try
            validate_config!(bad)
            @test false  # should have thrown
        catch e
            @test e isa ArgumentError
            @test occursin("TrotterDomain", e.msg)
            @test occursin("deferred", e.msg)
        end
    end

    # ---------------------------------------------------------------------
    # (g) validate_config!: EnergyDomain DLL is unsupported.
    # ---------------------------------------------------------------------
    @testset "(g) validate_config! rejects EnergyDomain DLL" begin
        bad = Config(;
            sim = Lindbladian(),
            domain = EnergyDomain(),
            construction = DLL(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = 1.0,
            sigma = 1.0,
            a = 1.0 / 30.0,
            s = 0.4,
            num_energy_bits = _DLL_NUM_ENERGY_BITS,
            w0 = _DLL_W0,
            filter = DLLGaussianFilter(1.0),
        )
        try
            validate_config!(bad)
            @test false  # should have thrown
        catch e
            @test e isa ArgumentError
            @test occursin("EnergyDomain", e.msg)
        end
    end

    # ---------------------------------------------------------------------
    # (h) NUFFT path agrees with explicit `dll_lindblad_op_time` Riemann sum
    # to the FINUFFT precision floor. Uses the n=3
    # disordered Heisenberg fixture (|B_H| ≫ n; non-trivial Bohr structure).
    #
    # Tolerance 1e-11: both paths evaluate the SAME ω=0 OFT integral
    # — `L_explicit` by a direct O(N·d²) Riemann sum over `time_labels`,
    # `L_nufft` via the FINUFFT slice in `_precompute_data`. The residual is the
    # FINUFFT-vs-direct-sum round-trip floor, NOT a quadrature error: it is
    # INVARIANT under `num_energy_bits` ∈ {10, 11, 12} (verified — the worst
    # β=1 op-norm difference stays 2.06e-12 regardless of grid size), so it
    # cannot be tightened by enlarging the time grid. On the prior find_typical
    # n=3 draw the worst case happened to sit just under 1e-12; the build_heis_1d
    # draw places the β=1 cell at 2.06e-12 op-norm (5.7e-12 relative to
    # ‖L‖≈0.36) while β∈{5,10} stay at ~2e-13. 1e-12 was therefore over-tight
    # for this internal-consistency check. 1e-11 matches the codebase's standard
    # FINUFFT-floor tolerance (cf. the adjoint-duality tests in
    # test_krylov_matvec.jl) and stays 100× tighter than the documented Bohr↔Time
    # FINUFFT floor (~1e-9), so it retains full
    # power to catch any real prefactor / index-map bug. Margin over the observed
    # 2.06e-12 worst cell is ~5×.
    # ---------------------------------------------------------------------
    @testset "(h) NUFFT slice == explicit Riemann sum (n=3, FINUFFT floor)" begin
        ham_path = test_hamiltonian_path(3)
        for beta in _DLL_BETAS
            ham = _load_test_hamiltonian(ham_path, Float64(beta))
            jump_paulis = [[X], [Y], [Z]]
            num_jumps = length(jump_paulis) * 3
            jump_norm = sqrt(num_jumps)
            jumps = JumpOp[]
            for pauli in jump_paulis
                for site in 1:3
                    op = Matrix(pad_term(pauli, 3, site)) ./ jump_norm
                    op_eb = ham.eigvecs' * op * ham.eigvecs
                    push!(jumps, JumpOp(op, op_eb, op == transpose(op), op == op'))
                end
            end

            cfg = Config(;
                sim = Lindbladian(),
                domain = TimeDomain(),
                construction = DLL(),
                num_qubits = 3,
                with_linear_combination = true,
                beta = Float64(beta),
                sigma = 1.0 / Float64(beta),
                a = beta / 30.0,
                s = 0.4,
                num_energy_bits = _DLL_NUM_ENERGY_BITS,
                t0 = _DLL_T0,
                num_trotter_steps_per_t0 = 10,
                filter = DLLGaussianFilter(Float64(beta)),
            )
            pre = QuantumFurnace._precompute_data(cfg, ham)

            for jump in jumps
                # Explicit Riemann sum (reference).
                L_explicit = QuantumFurnace.dll_lindblad_op_time(
                    jump, ham, pre.time_labels, pre.filter, pre.t0,
                )
                # NUFFT path: elementwise multiply against the prefactor at ω=0.
                # Single-channel filter ⇒ length-1 list.
                L_nufft = jump.in_eigenbasis .* pre.oft_nufft_at_zero_list[1] .* pre.t0
                @test opnorm(L_explicit - L_nufft) <= 1e-11
            end
        end
    end
end
