@testset "Simulation time and resource estimates" begin

    # ---- QPE grid info ----
    @testset "QPE grid info" begin
        for r in [4, 8, 12, 16]
            grid = QuantumFurnace._qpe_grid_info(r, W0)
            @test grid.N == 2^r
            # Fourier relation: t0 * w0 * N = 2π
            @test grid.t0 * W0 * grid.N ≈ 2π atol=TOL_EXACT
            # Energy range endpoints
            @test grid.energy_range[1] == -grid.N÷2 * W0
            @test grid.energy_range[2] == (grid.N÷2 - 1) * W0
        end
    end

    # ---- SimulationTimeBudget struct ----
    @testset "SimulationTimeBudget struct" begin
        # New shape: per-term register triples + (with_gqsp, gqsp_degree).
        # Field order matches the struct definition in src/simulation_time.jl.
        budget = SimulationTimeBudget(
            # cost
            100.0, 5.0, 10.0, 210.0, 1000, 210000.0,
            # dissipative register
            12, 4096, 0.05, 2π/(4096*0.05), (-102.4, 102.35),
            # outer coherent (b_-)
            6, 64, 0.314, 2π/(64*0.314),
            # inner coherent (b_+)
            14, 16384, 0.628, 2π/(16384*0.628),
            # GQSP
            true, 2, :formal_unscaled_gqsp_surrogate,
            # physics
            10.0, 0.1, 0.01, :KMS, 4, 1.5, 10.0,
            # filter
            :smooth_metropolis, Dict{Symbol,Float64}(:a => 0.333, :s => 0.4),
        )
        @test budget.oft_time == 100.0
        @test budget.b_per_be == 5.0
        @test budget.b_time == 10.0
        @test budget.per_step_time == 210.0
        @test budget.n_steps == 1000
        @test budget.total_time == 210000.0
        @test budget.r_D == 12
        @test budget.N_D == 4096
        @test budget.r_bm == 6
        @test budget.N_bm == 64
        @test budget.r_bp == 14
        @test budget.N_bp == 16384
        @test budget.with_gqsp === true
        @test budget.gqsp_degree == 2
        @test budget.cost_interpretation === :formal_unscaled_gqsp_surrogate
        @test budget.construction == :KMS
        @test budget.filter_type == :smooth_metropolis
        @test budget.T == 10.0

        # Compact show
        compact = sprint(show, budget)
        @test occursin("SimulationTimeBudget", compact)
        @test occursin("r_D=12", compact)
        @test occursin("gqsp d=2", compact)
        @test occursin("formal surrogate", compact)

        # Verbose show
        verbose = sprint(show, MIME"text/plain"(), budget)
        @test occursin("Dissipative", verbose)
        @test occursin("Outer coh.", verbose)
        @test occursin("Inner coh.", verbose)
        @test occursin("OFT time", verbose)
        @test occursin("B per BE", verbose)
        @test occursin("B time", verbose)
        @test occursin("formal_unscaled_gqsp_surrogate", verbose)
        @test occursin("not a synthesised GQSP circuit", verbose)
        @test occursin("Per step", verbose)
        @test occursin("Total", verbose)
    end

    # ---- OFT time — closed-form validation ----
    @testset "OFT time — closed-form" begin
        for r in [4, 8, 10, 12]
            N = 2^r
            t0 = 2π / (N * W0)
            expected = t0 * N^2 / 4
            result = QuantumFurnace._oft_hamiltonian_time(r, W0, ones(N))
            @test result ≈ expected rtol=1e-10
        end
    end

    @testset "OFT time — weighted" begin
        r = 8
        N = 2^r
        t0 = 2π / (N * W0)
        unweighted = t0 * N^2 / 4

        # Zero weights → 0
        @test QuantumFurnace._oft_hamiltonian_time(r, W0, zeros(N)) == 0.0

        # Uniform 2.0 → 2× closed form
        @test QuantumFurnace._oft_hamiltonian_time(r, W0, fill(2.0, N)) ≈ 2.0 * unweighted atol=TOL_EXACT

        # Transition weights (≤ 1) → less than unweighted
        config = make_config(Thermalize(), TimeDomain())
        energy_labels = QuantumFurnace._create_energy_labels(r, W0)
        tw = Float64[pick_transition(config, w) for w in energy_labels]
        weighted = QuantumFurnace._oft_hamiltonian_time(r, W0, tw)
        @test 0.0 < weighted < unweighted
    end

    # ---- B time ----
    @testset "B time — GNS returns 0" begin
        @test QuantumFurnace._b_hamiltonian_time(nothing, nothing, 10.0, 0.1, 0.01) == 0.0
        @test QuantumFurnace._b_hamiltonian_time(Dict(), Dict(), 10.0, 0.1, 0.01) == 0.0
        @test QuantumFurnace._b_hamiltonian_time(nothing, Dict(1.0 => 0.5+0im), 10.0, 0.1, 0.01) == 0.0
    end

    @testset "B time — KMS positive" begin
        config = make_config(Thermalize(), TimeDomain())
        energy_labels = QuantumFurnace._create_energy_labels(NUM_ENERGY_BITS, W0)
        time_labels = energy_labels .* (T0 / W0)
        bm = QuantumFurnace._compute_truncated_func(
            QuantumFurnace._compute_b_minus, time_labels, BETA, SIGMA)
        bp_fn, bp_args = QuantumFurnace._select_b_plus_calculator(config)
        bp = QuantumFurnace._compute_truncated_func(bp_fn, time_labels, bp_args...)
        bt = QuantumFurnace._b_hamiltonian_time(bm, bp, BETA, SIGMA, T0)
        @test bt > 0.0
        @test isfinite(bt)
    end

    # ---- compute_simulation_time ----
    @testset "compute_simulation_time — KMS" begin
        config = make_config(Thermalize(), TimeDomain())
        budget = compute_simulation_time(config, TEST_HAM, 10.0)
        # Formula: per_step = 2×OFT + B
        @test budget.per_step_time ≈ 2.0 * budget.oft_time + budget.b_time
        @test budget.n_steps == ceil(Int, 10.0 / TEST_DELTA)
        @test budget.total_time ≈ budget.n_steps * budget.per_step_time
        @test budget.construction == :KMS
        @test budget.n_qubits == NUM_QUBITS
        @test budget.rescaling_factor == TEST_HAM.rescaling_factor
        @test budget.oft_time > 0.0
        @test budget.b_per_be > 0.0
        @test budget.b_time > 0.0
        @test isfinite(budget.total_time)
        @test budget.T == 10.0
        # Default config: with_gqsp=false → b_time === b_per_be (no multiplier).
        @test budget.with_gqsp === false
        @test budget.b_time ≈ budget.b_per_be rtol=TOL_EXACT
        @test budget.cost_interpretation === :physical_channel
    end

    @testset "compute_simulation_time — exact nonnegative step count" begin
        config = make_config(Thermalize(), TimeDomain())

        # The exact count owns both resource accounting and time metadata. This
        # avoids ceil((k·δ)/δ) adding a step through floating-point roundoff.
        k = 7
        T_exact = k * config.delta
        budget = compute_simulation_time(
            config, TEST_HAM, T_exact; n_steps=k)
        @test budget.n_steps == k
        @test budget.T ≈ T_exact atol=0 rtol=8eps(Float64)
        @test budget.total_time ≈ k * budget.per_step_time rtol=TOL_EXACT

        zero = compute_simulation_time(
            config, TEST_HAM, 0.0; n_steps=0)
        @test zero.n_steps == 0
        @test zero.T == 0.0
        @test zero.total_time == 0.0

        @test_throws ArgumentError compute_simulation_time(
            config, TEST_HAM, 1.0; n_steps=k)
        @test_throws ArgumentError compute_simulation_time(
            config, TEST_HAM, 0.0)
        @test_throws ArgumentError compute_simulation_time(
            config, TEST_HAM, 0.0; n_steps=-1)
    end

    @testset "compute_simulation_time — GNS" begin
        config = make_config(Thermalize(), TimeDomain(); construction=GNS())
        budget = compute_simulation_time(config, TEST_HAM, 10.0)
        @test budget.b_per_be == 0.0
        @test budget.b_time == 0.0
        @test budget.per_step_time ≈ 2.0 * budget.oft_time
        @test budget.construction == :GNS
    end

    # ---- filter_type tag honours the (a, s) taxonomy ----
    # Smooth Metropolis is exactly `s > 0` (any a ≥ 0); kinky Metropolis is
    # `s = a = 0`. The thesis-default `a = 0, s = 0.25` must tag as smooth, not
    # kinky. (s = 0, a > 0) is rejected upstream by validate_config!.
    @testset "compute_simulation_time — filter_type taxonomy" begin
        function _filter_cfg(; a, s, construction=KMS(), eta=0.05)
            Config(;
                sim = Thermalize(), domain = TimeDomain(), construction = construction,
                num_qubits = NUM_QUBITS, with_linear_combination = true,
                beta = BETA, sigma = SIGMA, a = a, s = s,
                num_energy_bits = NUM_ENERGY_BITS, w0 = W0, t0 = T0,
                num_trotter_steps_per_t0 = NUM_TROTTER_STEPS_PER_T0,
                mixing_time = 1.0, delta = TEST_DELTA,
                eta = eta,
            )
        end

        # Thesis-default: a = 0, s > 0 → smooth Metropolis (a omitted from params Dict only when 0 is irrelevant)
        bgt_thesis = compute_simulation_time(_filter_cfg(a = 0.0, s = 0.25), TEST_HAM, 10.0)
        @test bgt_thesis.filter_type == :smooth_metropolis
        @test bgt_thesis.filter_params[:s] == 0.25
        @test bgt_thesis.filter_params[:a] == 0.0

        # a-regularised smooth: a > 0, s > 0 → smooth Metropolis
        bgt_areg = compute_simulation_time(_filter_cfg(a = BETA / 30.0, s = 0.4), TEST_HAM, 10.0)
        @test bgt_areg.filter_type == :smooth_metropolis
        @test bgt_areg.filter_params[:s] == 0.4

        # Kinky: a = 0, s = 0 → kinky Metropolis (eta still required on TimeDomain)
        bgt_kinky = compute_simulation_time(_filter_cfg(a = 0.0, s = 0.0), TEST_HAM, 10.0)
        @test bgt_kinky.filter_type == :kinky_metropolis
        @test isempty(bgt_kinky.filter_params)
    end

    @testset "compute_simulation_time — TrotterDomain" begin
        config = make_config(Thermalize(), TrotterDomain())
        budget = compute_simulation_time(config, TEST_HAM, 10.0)
        @test budget.per_step_time ≈ 2.0 * budget.oft_time + budget.b_time
        @test budget.n_steps == ceil(Int, 10.0 / TEST_DELTA)
        @test budget.total_time > 0.0
        @test isfinite(budget.total_time)
    end

    # ---- GQSP cost-model multiplier (Form B = MW Eq. 46) ----
    # With Form B: 2d block-encoding queries per CoherentStep ⇒ b_time = 2·d·b_per_be.
    @testset "compute_simulation_time — GQSP multiplier (Form B)" begin
        # Build matched configs at d ∈ {1, 2, 3} with all other params identical.
        # `make_config` doesn't expose with_gqsp/gqsp_degree, so use Config(...) directly.
        function _gqsp_cfg(; with_gqsp::Bool, gqsp_degree::Int=1)
            Config(;
                sim = Thermalize(),
                domain = TimeDomain(),
                construction = KMS(),
                num_qubits = NUM_QUBITS,
                with_linear_combination = true,
                beta = BETA, sigma = SIGMA,
                a = BETA / 30.0, s = 0.4,
                num_energy_bits = NUM_ENERGY_BITS,
                w0 = W0, t0 = T0,
                num_trotter_steps_per_t0 = NUM_TROTTER_STEPS_PER_T0,
                mixing_time = 1.0, delta = TEST_DELTA,
                with_gqsp = with_gqsp, gqsp_degree = gqsp_degree,
            )
        end

        b0 = compute_simulation_time(_gqsp_cfg(with_gqsp=false),               TEST_HAM, 10.0)
        b1 = compute_simulation_time(_gqsp_cfg(with_gqsp=true,  gqsp_degree=1), TEST_HAM, 10.0)
        b2 = compute_simulation_time(_gqsp_cfg(with_gqsp=true,  gqsp_degree=2), TEST_HAM, 10.0)
        b3 = compute_simulation_time(_gqsp_cfg(with_gqsp=true,  gqsp_degree=3), TEST_HAM, 10.0)

        # `b_per_be` is the GQSP-blind per-BE cost — independent of the GQSP flag and degree
        # for matched configs (same registers, same b_± grids).
        @test b0.b_per_be ≈ b1.b_per_be rtol=TOL_EXACT
        @test b1.b_per_be ≈ b2.b_per_be rtol=TOL_EXACT
        @test b1.b_per_be ≈ b3.b_per_be rtol=TOL_EXACT

        # B-time multiplier (Form B): 2d × b_per_be.
        @test b0.b_time ≈ b0.b_per_be          rtol=TOL_EXACT  # no GQSP
        @test b1.b_time ≈ 2.0 * b1.b_per_be    rtol=TOL_EXACT  # 2·1
        @test b2.b_time ≈ 4.0 * b2.b_per_be    rtol=TOL_EXACT  # 2·2
        @test b3.b_time ≈ 6.0 * b3.b_per_be    rtol=TOL_EXACT  # 2·3

        # GQSP fields recorded in budget.
        @test b0.with_gqsp === false
        @test b1.with_gqsp === true && b1.gqsp_degree == 1
        @test b2.with_gqsp === true && b2.gqsp_degree == 2
        @test b3.with_gqsp === true && b3.gqsp_degree == 3
        @test b0.cost_interpretation === :physical_channel
        @test all(b -> b.cost_interpretation ===
                       :formal_unscaled_gqsp_surrogate, (b1, b2, b3))
        @test occursin("formal surrogate", sprint(show, b1))
        @test occursin("formal_unscaled_gqsp_surrogate",
            sprint(show, MIME"text/plain"(), b1))

        # GNS path: with_gqsp is config-rejected (validate_config!), but if a budget were
        # computed for a GNS config, b_time stays 0 regardless of multiplier branch.
        cfg_gns = make_config(Thermalize(), TimeDomain(); construction=GNS())
        bg = compute_simulation_time(cfg_gns, TEST_HAM, 10.0)
        @test bg.b_per_be == 0.0
        @test bg.b_time == 0.0
    end

    # ---- Per-term registers honoured end-to-end ----
    # Vary the b_+ register independently and confirm (i) the budget records the
    # per-term values, (ii) the OFT (D-register) cost is unchanged, (iii) the
    # B-cost responds to the b_+ spacing change (factored-double-sum scaling).
    @testset "compute_simulation_time — per-term registers" begin
        function _per_term_cfg(; rbp::Int)
            # Use legacy single-register defaults for D and b_- (auto-promoted via
            # register_*_X fallback), and override only the b_+ triple.
            w0_bp = W0
            t0_bp = 2π / (2^rbp * w0_bp)
            Config(;
                sim = Thermalize(),
                domain = TimeDomain(),
                construction = KMS(),
                num_qubits = NUM_QUBITS,
                with_linear_combination = true,
                beta = BETA, sigma = SIGMA,
                a = BETA / 30.0, s = 0.4,
                num_energy_bits = NUM_ENERGY_BITS, w0 = W0, t0 = T0,
                num_energy_bits_b_plus = rbp, w0_b_plus = w0_bp, t0_b_plus = t0_bp,
                num_trotter_steps_per_t0 = NUM_TROTTER_STEPS_PER_T0,
                mixing_time = 1.0, delta = TEST_DELTA,
            )
        end

        ba = compute_simulation_time(_per_term_cfg(rbp=8),  TEST_HAM, 10.0)
        bb = compute_simulation_time(_per_term_cfg(rbp=10), TEST_HAM, 10.0)

        # Budget records the per-term r_b+.
        @test ba.r_bp == 8
        @test bb.r_bp == 10
        @test ba.N_bp == 256
        @test bb.N_bp == 1024

        # D-register quantities unchanged across the two configs.
        @test ba.r_D == bb.r_D == NUM_ENERGY_BITS
        @test ba.w0_D ≈ bb.w0_D rtol=TOL_EXACT
        @test ba.oft_time ≈ bb.oft_time rtol=TOL_EXACT

        # b_- register also unchanged (we only varied b_+).
        @test ba.r_bm == bb.r_bm

        # B per BE responds to the b_+ spacing change (factored sum has t0_outer · t0_inner).
        @test ba.b_per_be != bb.b_per_be
    end

    # ---- Trotter-step (gate-level) accounting ----
    @testset "count_trotter_steps — KMS component formulas" begin
        config = make_config(Thermalize(), TrotterDomain())
        cnt = count_trotter_steps(config, TEST_HAM, 10.0)
        N = 2^NUM_ENERGY_BITS
        M = NUM_TROTTER_STEPS_PER_T0

        # QPE-ladder counting: (N−1)·M per OFT pass, two passes per δ-step.
        @test cnt.oft_substeps_per_pass == (N - 1) * M
        @test cnt.oft_substeps_per_step == 2 * cnt.oft_substeps_per_pass

        # B block encoding: outer 2 ladder passes, inner 4 ladder-pass equivalents
        # (legacy single-register config ⇒ all three legs share N and M).
        @test cnt.b_outer_substeps_per_be == 2 * (N - 1) * M
        @test cnt.b_inner_substeps_per_be == 4 * (N - 1) * M
        @test cnt.b_substeps_per_be == cnt.b_outer_substeps_per_be + cnt.b_inner_substeps_per_be

        # Default config: with_gqsp = false ⇒ one direct BE query.
        @test cnt.n_be_queries == 1
        @test cnt.b_substeps_per_step == cnt.b_substeps_per_be

        # Assembly.
        @test cnt.substeps_per_step == cnt.oft_substeps_per_step + cnt.b_substeps_per_step
        @test cnt.n_steps == ceil(Int, 10.0 / TEST_DELTA)
        @test cnt.total_substeps == cnt.n_steps * cnt.substeps_per_step

        # Ladder rungs: 2·r_D + 1·(2·r_bm + 3·r_bp), all r equal here.
        @test cnt.blocks_per_step == 2 * NUM_ENERGY_BITS + (2 + 3) * NUM_ENERGY_BITS
        @test cnt.total_blocks == cnt.n_steps * cnt.blocks_per_step

        # Metadata mirrors the config / HamHam.
        @test cnt.construction == :KMS
        @test cnt.n_qubits == NUM_QUBITS
        @test cnt.rescaling_factor == TEST_HAM.rescaling_factor
        @test cnt.T == 10.0
        @test cnt.cost_interpretation === :physical_channel
    end

    @testset "count_trotter_steps — GNS has no coherent substeps" begin
        config = make_config(Thermalize(), TrotterDomain(); construction=GNS())
        cnt = count_trotter_steps(config, TEST_HAM, 10.0)
        @test cnt.n_be_queries == 0
        @test cnt.b_substeps_per_be == 0
        @test cnt.b_substeps_per_step == 0
        @test cnt.substeps_per_step == cnt.oft_substeps_per_step
        @test cnt.blocks_per_step == 2 * NUM_ENERGY_BITS
        @test cnt.construction == :GNS
    end

    @testset "count_trotter_steps — GQSP Form B multiplier" begin
        function _gqsp_count_cfg(; with_gqsp::Bool, gqsp_degree::Int=1)
            Config(;
                sim = Thermalize(), domain = TrotterDomain(), construction = KMS(),
                num_qubits = NUM_QUBITS, with_linear_combination = true,
                beta = BETA, sigma = SIGMA, a = BETA / 30.0, s = 0.4,
                num_energy_bits = NUM_ENERGY_BITS, w0 = W0, t0 = T0,
                num_trotter_steps_per_t0 = NUM_TROTTER_STEPS_PER_T0,
                mixing_time = 1.0, delta = TEST_DELTA,
                with_gqsp = with_gqsp, gqsp_degree = gqsp_degree,
            )
        end
        base = count_trotter_steps(_gqsp_count_cfg(with_gqsp=false), TEST_HAM, 10.0)
        for d in (1, 2, 3)
            cnt = count_trotter_steps(_gqsp_count_cfg(with_gqsp=true, gqsp_degree=d), TEST_HAM, 10.0)
            @test cnt.n_be_queries == 2 * d
            @test cnt.b_substeps_per_step == 2 * d * cnt.b_substeps_per_be
            @test cnt.b_substeps_per_be == base.b_substeps_per_be
            # OFT leg blind to GQSP.
            @test cnt.oft_substeps_per_step == base.oft_substeps_per_step
            @test cnt.cost_interpretation === :formal_unscaled_gqsp_surrogate
            @test occursin("formal surrogate", sprint(show, cnt))
            @test occursin("formal_unscaled_gqsp_surrogate",
                sprint(show, MIME"text/plain"(), cnt))
        end
    end

    # Cross-check: substep counts × per-leg
    # substep duration t0_X/M_X reproduce the unweighted ladder durations, and
    # the counter agrees with an independently-built SimulationTimeBudget on
    # every shared register/step quantity.
    @testset "count_trotter_steps — ladder-duration identity vs SimulationTimeBudget" begin
        config = make_config(Thermalize(), TrotterDomain())
        cnt = count_trotter_steps(config, TEST_HAM, 10.0)
        budget = compute_simulation_time(config, TEST_HAM, 10.0)

        # Shared quantities agree.
        @test cnt.n_steps == budget.n_steps
        @test cnt.r_D == budget.r_D && cnt.N_D == budget.N_D
        @test cnt.t0_D ≈ budget.t0_D rtol=TOL_EXACT
        @test cnt.r_bm == budget.r_bm && cnt.r_bp == budget.r_bp
        @test cnt.with_gqsp == budget.with_gqsp
        @test cnt.delta == budget.delta

        # Ladder-duration identities (per-leg substep duration × count).
        @test cnt.oft_substeps_per_pass * (cnt.t0_D / cnt.M_D) ≈
              (cnt.N_D - 1) * budget.t0_D rtol=TOL_EXACT
        @test cnt.b_outer_substeps_per_be * ((cnt.t0_bm / SIGMA) / cnt.M_bm) ≈
              2 * (cnt.N_bm - 1) * budget.t0_bm / SIGMA rtol=TOL_EXACT
        @test cnt.b_inner_substeps_per_be * ((BETA * cnt.t0_bp) / cnt.M_bp) ≈
              4 * (cnt.N_bp - 1) * BETA * budget.t0_bp rtol=TOL_EXACT

        # Documented model difference: the physical ladder duration is far below
        # the unweighted expected-time sum Σ_k |k·t0| = t0·N²/4 (γ ≡ 1) — the
        # hardware ladder is implemented once, not per grid point.
        unweighted_oft = QuantumFurnace._oft_hamiltonian_time(cnt.r_D, W0, ones(cnt.N_D))
        @test (cnt.N_D - 1) * cnt.t0_D < unweighted_oft
    end

    @testset "count_trotter_steps — per-leg registers and M (TrotterTriple)" begin
        rbp, Mbp = 8, 3
        w0_bp = W0
        t0_bp = 2π / (2^rbp * w0_bp)
        config = Config(;
            sim = Thermalize(), domain = TrotterDomain(), construction = KMS(),
            num_qubits = NUM_QUBITS, with_linear_combination = true,
            beta = BETA, sigma = SIGMA, a = BETA / 30.0, s = 0.4,
            num_energy_bits = NUM_ENERGY_BITS, w0 = W0, t0 = T0,
            num_energy_bits_b_plus = rbp, w0_b_plus = w0_bp, t0_b_plus = t0_bp,
            num_trotter_steps_per_t0 = NUM_TROTTER_STEPS_PER_T0,
            num_trotter_steps_per_t0_b_plus = Mbp,
            mixing_time = 1.0, delta = TEST_DELTA,
        )
        cnt = count_trotter_steps(config, TEST_HAM, 10.0)
        # b_+ leg picks up its own register and substep count …
        @test cnt.r_bp == rbp && cnt.N_bp == 2^rbp && cnt.M_bp == Mbp
        @test cnt.b_inner_substeps_per_be == 4 * (2^rbp - 1) * Mbp
        # … while D and b_- stay on the legacy fallback.
        @test cnt.M_D == NUM_TROTTER_STEPS_PER_T0
        @test cnt.b_outer_substeps_per_be == 2 * (2^NUM_ENERGY_BITS - 1) * NUM_TROTTER_STEPS_PER_T0
    end

    # ---- RXX estimator ----
    @testset "estimate_rxx_count — hand-computed n=3 cell" begin
        # r = 3 (N = 8), M = 2 on all legs, GQSP d = 1 (2 Form-B BE queries):
        #   OFT/pass = 7·2 = 14, ×2 = 28/step
        #   B/BE = 2·7·2 + 4·7·2 = 84, ×2 queries = 168/step
        #   substeps/step = 196; T/δ = 0.05/0.01 ⇒ 5 steps ⇒ 980 total
        #   blocks/step = 2·3 + 2·(2·3 + 3·3) = 36 ⇒ 180 total
        #   slope 20, intercept 3 ⇒ rxx = 20·980 + 3·180 = 20140
        r = 3
        cfg = Config(;
            sim = Thermalize(), domain = TrotterDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = true,
            beta = BETA, sigma = SIGMA, a = BETA / 30.0, s = 0.4,
            num_energy_bits = r, w0 = W0, t0 = 2π / (2^r * W0),
            num_trotter_steps_per_t0 = 2,
            mixing_time = 0.05, delta = 0.01,
            with_gqsp = true, gqsp_degree = 1,
        )
        table = Dict(("heis1d_test", 3) => (;
            geometry = "-", rxx_slope_per_substep = 20.0, rxx_intercept = 3.0,
            rxx_L1 = 23.0, rxx_fit_max_abs_dev = 0.0, f_ctrl1 = 3.0, f_ctrl2 = 6.0,
            qiskit_version = "2.4.1"))
        bgt = estimate_rxx_count(cfg, N3_HAM, 0.05; rxx_table = table, hamiltonian = "heis1d_test")
        @test bgt.steps.substeps_per_step == 196
        @test bgt.steps.total_substeps == 980
        @test bgt.steps.total_blocks == 180
        @test bgt.rxx_substep_part == 20.0 * 980
        @test bgt.rxx_boundary_part == 3.0 * 180
        @test bgt.rxx_total == 20140.0
        @test bgt.f_ctrl1 == 3.0 && bgt.f_ctrl2 == 6.0
        @test bgt.hamiltonian == "heis1d_test"
        @test bgt.steps.cost_interpretation ===
            :formal_unscaled_gqsp_surrogate
        @test occursin("formal unscaled-GQSP surrogate", sprint(show, bgt))
        @test occursin("formal_unscaled_gqsp_surrogate",
            sprint(show, MIME"text/plain"(), bgt))

        # Per-pass controlled count, hand-computed. OFT/pass = (N_D−1)·M_D =
        # 7·2 = 14, ×2 = 28/step; coherent b/step = 168. Blocks: fwd r_D=3,
        # bwd r_D=3, coherent 180/5−2·3 per step → use steps fields directly.
        s = bgt.steps
        @test s.oft_substeps_per_pass == 14
        @test s.b_substeps_per_step == 168
        b_coh = s.blocks_per_step - 2 * s.r_D
        rxx_ctrl_step = 20.0 * (3.0 * 14 + 6.0 * 14 + 6.0 * 168) +
                        3.0 * (3.0 * 3 + 6.0 * 3 + 6.0 * b_coh)
        @test bgt.rxx_total_controlled ≈ s.n_steps * rxx_ctrl_step rtol = TOL_EXACT
        # Controlled > plain (every pass gets ≥1 control).
        @test bgt.rxx_total_controlled > bgt.rxx_total

        # Missing table entry → clear error.
        @test_throws ArgumentError estimate_rxx_count(cfg, N3_HAM, 0.05;
            rxx_table = table, hamiltonian = "nope")
    end

    @testset "load_rxx_table — TSV parsing (10-col + legacy 9-col)" begin
        mktempdir() do dir
            # Current 10-column schema (f_ctrl1, f_ctrl2).
            path = joinpath(dir, "rxx.tsv")
            open(path, "w") do io
                println(io, "name\tn\tgeometry\trxx_slope_per_substep\trxx_intercept\trxx_L1\trxx_fit_max_abs_dev\tf_ctrl1\tf_ctrl2\tqiskit_version")
                println(io, "heis1d\t3\t-\t20.0\t3.0\t23\t0.0\t2.91\t6.70\t2.4.1")
                println(io, "heis1d\t4\t-\t25.0\t6.0\t31\t0.0\tnan\tnan\t2.4.1")
            end
            tbl = load_rxx_table(path)
            @test length(tbl) == 2
            e = tbl[("heis1d", 3)]
            @test e.rxx_slope_per_substep == 20.0
            @test e.rxx_intercept == 3.0
            @test e.f_ctrl1 == 2.91
            @test e.f_ctrl2 == 6.70
            @test isnan(tbl[("heis1d", 4)].f_ctrl1)
            @test isnan(tbl[("heis1d", 4)].f_ctrl2)

            # Legacy 9-column schema → f_ctrl1 from f_ctrl, f_ctrl2 = NaN.
            legacy = joinpath(dir, "legacy.tsv")
            open(legacy, "w") do io
                println(io, "name\tn\tgeometry\trxx_slope_per_substep\trxx_intercept\trxx_L1\trxx_fit_max_abs_dev\tf_ctrl\tqiskit_version")
                println(io, "heis1d\t3\t-\t20.0\t3.0\t23\t0.0\t2.91\t2.4.1")
            end
            tl = load_rxx_table(legacy)
            @test tl[("heis1d", 3)].f_ctrl1 == 2.91
            @test isnan(tl[("heis1d", 3)].f_ctrl2)
        end
        @test_throws ArgumentError load_rxx_table("/nonexistent/rxx.tsv")
    end

    @testset "load_rxx_table — measured RXX counts" begin
        # The default path is the committed measurement table; spot-check the
        # contract the estimator relies on (keys + positive slopes). Scope is
        # 1D Heisenberg only; the 2D TFIM is out of scope for the
        # RXX plot (user decision 2026-06-08).
        tbl = load_rxx_table()
        for n in 3:9
            @test haskey(tbl, ("heis1d_xxx_disordered_periodic_seed46", n))
        end
        @test all(e.rxx_slope_per_substep > 0 for e in values(tbl))
        @test all(e.rxx_fit_max_abs_dev == 0 for e in values(tbl))
        # Control factors present and ordered f_ctrl2 > f_ctrl1 > 1.
        @test all(1 < e.f_ctrl1 < e.f_ctrl2 for e in values(tbl))
    end

    @testset "count_trotter_steps — validation" begin
        config_no_M = Config(;
            sim = Thermalize(), domain = TrotterDomain(), construction = KMS(),
            num_qubits = NUM_QUBITS, with_linear_combination = true,
            beta = BETA, sigma = SIGMA, a = BETA / 30.0, s = 0.4,
            num_energy_bits = NUM_ENERGY_BITS, w0 = W0, t0 = T0,
            mixing_time = 1.0, delta = TEST_DELTA,
        )
        @test_throws ArgumentError count_trotter_steps(config_no_M, TEST_HAM, 10.0)
        config_ok = make_config(Thermalize(), TrotterDomain())
        @test_throws ArgumentError count_trotter_steps(config_ok, TEST_HAM, -1.0)
        @test_throws ArgumentError count_trotter_steps(config_ok, TEST_HAM, Inf)
        @test_throws ArgumentError count_trotter_steps(config_ok, TEST_HAM, NaN)
        @test_throws ArgumentError count_trotter_steps(
            make_config(Thermalize(), TrotterDomain(); num_qubits = NUM_QUBITS + 1),
            TEST_HAM,
            10.0,
        )
        @test_throws ArgumentError compute_simulation_time(
            make_config(Thermalize(), TimeDomain(); num_qubits = NUM_QUBITS + 1),
            TEST_HAM,
            10.0,
        )
        @test_throws ArgumentError compute_simulation_time(config_ok, TEST_HAM, Inf)
    end

    # ========================================================================
    # Gate-count edge cases: independent per-leg steps, duration weights,
    # controlled partitions, missing table entries and validation fallbacks.
    # ========================================================================

    # ---- count_trotter_steps: distinct per-leg M on ALL THREE legs ----
    # The existing ladder-identity test only varies M on one leg. This pins the
    # M-cancellation independently on each leg with three DIFFERENT M values, so
    # a leg accidentally reading another leg's M would break an identity.
    @testset "count_trotter_steps — three-leg distinct M ladder identity" begin
        rD, rbm, rbp = 6, 7, 8
        MD, Mbm, Mbp = 5, 4, 3   # all distinct
        cfg = Config(;
            sim = Thermalize(), domain = TrotterDomain(), construction = KMS(),
            num_qubits = NUM_QUBITS, with_linear_combination = true,
            beta = BETA, sigma = SIGMA, a = BETA / 30.0, s = 0.4,
            num_energy_bits_D = rD,        w0_D = W0,        t0_D = 2π / (2^rD * W0),
            num_energy_bits_b_minus = rbm, w0_b_minus = W0,  t0_b_minus = 2π / (2^rbm * W0),
            num_energy_bits_b_plus = rbp,  w0_b_plus = W0,   t0_b_plus = 2π / (2^rbp * W0),
            num_trotter_steps_per_t0_D = MD,
            num_trotter_steps_per_t0_b_minus = Mbm,
            num_trotter_steps_per_t0_b_plus = Mbp,
            num_trotter_steps_per_t0 = 99,   # legacy fallback — MUST NOT be used by any leg
            mixing_time = 1.0, delta = TEST_DELTA, with_gqsp = true, gqsp_degree = 1,
        )
        cnt = count_trotter_steps(cfg, TEST_HAM, 10.0)
        bud = compute_simulation_time(cfg, TEST_HAM, 10.0)

        # Each leg picks up its OWN M (never the legacy 99 nor another leg's M).
        @test cnt.M_D == MD && cnt.M_bm == Mbm && cnt.M_bp == Mbp
        @test cnt.oft_substeps_per_pass == (2^rD - 1) * MD
        @test cnt.b_outer_substeps_per_be == 2 * (2^rbm - 1) * Mbm
        @test cnt.b_inner_substeps_per_be == 4 * (2^rbp - 1) * Mbp

        # Ladder-duration identities with the leg's OWN substep duration t0_X/M_X.
        @test cnt.oft_substeps_per_pass * (cnt.t0_D / cnt.M_D) ≈
              (cnt.N_D - 1) * bud.t0_D rtol = TOL_EXACT
        @test cnt.b_outer_substeps_per_be * ((cnt.t0_bm / SIGMA) / cnt.M_bm) ≈
              2 * (cnt.N_bm - 1) * bud.t0_bm / SIGMA rtol = TOL_EXACT
        @test cnt.b_inner_substeps_per_be * ((BETA * cnt.t0_bp) / cnt.M_bp) ≈
              4 * (cnt.N_bp - 1) * BETA * bud.t0_bp rtol = TOL_EXACT
    end

    # ---- count_trotter_steps: substep (2/4 duration) vs block (2/3 evolution) ----
    # The inner coherent leg runs 3 evolutions e^{iHτβ}·e^{−2iHτβ}·e^{iHτβ}: the
    # middle one is TWICE the duration (verified vs B_trotter's
    # eigvals^(-2·num_t0_steps), src/coherent.jl:364), so it weighs 2 in the
    # SUBSTEP count (1+2+1=4) but 1 in the BLOCK count (3 contiguous ladders).
    # The outer leg has 2 evolutions e^{∓iHt/σ} (weight 2 both). Distinct r per
    # leg disentangles the 2·r_bm and 3·r_bp block contributions.
    @testset "count_trotter_steps — substep/block weight distinction" begin
        rD, rbm, rbp, M = 5, 6, 7, 3
        cfg = Config(;
            sim = Thermalize(), domain = TrotterDomain(), construction = KMS(),
            num_qubits = NUM_QUBITS, with_linear_combination = true,
            beta = BETA, sigma = SIGMA, a = BETA / 30.0, s = 0.4,
            num_energy_bits_D = rD,        w0_D = W0,       t0_D = 2π / (2^rD * W0),
            num_energy_bits_b_minus = rbm, w0_b_minus = W0, t0_b_minus = 2π / (2^rbm * W0),
            num_energy_bits_b_plus = rbp,  w0_b_plus = W0,  t0_b_plus = 2π / (2^rbp * W0),
            num_trotter_steps_per_t0 = M,
            mixing_time = 1.0, delta = TEST_DELTA, with_gqsp = true, gqsp_degree = 1,
        )
        cnt = count_trotter_steps(cfg, TEST_HAM, 10.0)
        # Duration-weighted substeps: outer ×2, inner ×4.
        @test cnt.b_outer_substeps_per_be == 2 * (2^rbm - 1) * M
        @test cnt.b_inner_substeps_per_be == 4 * (2^rbp - 1) * M
        # Evolution-count-weighted blocks: outer 2·r_bm rungs, inner 3·r_bp rungs.
        @test cnt.blocks_per_step == 2 * rD + cnt.n_be_queries * (2 * rbm + 3 * rbp)
        # The block weight on the inner leg is 3 (NOT the substep weight 4) and
        # on the outer leg 2 — pin both coefficients via distinct r.
        coherent_blocks = cnt.blocks_per_step - 2 * cnt.r_D
        @test coherent_blocks == cnt.n_be_queries * (2 * rbm + 3 * rbp)
    end

    # ---- estimate_rxx_count: controlled-formula partition completeness ----
    # The controlled count partitions every substep into (S_fwd, S_bwd, S_coh)
    # and every block into (B_fwd, B_bwd, B_coh). Those partitions MUST cover the
    # whole circuit exactly (sum to substeps_per_step / blocks_per_step), else a
    # pass is silently dropped or double-counted. Also confirms the per-step
    # b_coh = blocks_per_step − 2·r_D (NOT total_blocks − 2·r_D — the RxxBudget
    # docstring TEXT says total_blocks, which would be wrong by an n_steps factor;
    # the CODE correctly uses the per-step quantity, asserted here).
    @testset "estimate_rxx_count — controlled partition completeness" begin
        function _cfg(; with_gqsp, d, nq = 3, r = 3, M = 2)
            Config(;
                sim = Thermalize(), domain = TrotterDomain(), construction = KMS(),
                num_qubits = nq, with_linear_combination = true,
                beta = BETA, sigma = SIGMA, a = BETA / 30.0, s = 0.4,
                num_energy_bits = r, w0 = W0, t0 = 2π / (2^r * W0),
                num_trotter_steps_per_t0 = M, mixing_time = 0.05, delta = 0.01,
                with_gqsp = with_gqsp, gqsp_degree = d,
            )
        end
        tab = Dict(("h", 3) => (;
            geometry = "-", rxx_slope_per_substep = 20.0, rxx_intercept = 3.0,
            rxx_L1 = 23.0, rxx_fit_max_abs_dev = 0.0, f_ctrl1 = 3.0, f_ctrl2 = 6.0,
            qiskit_version = "2.4.1"))
        for (with_gqsp, d) in [(true, 1), (true, 2), (false, 1)]
            bgt = estimate_rxx_count(_cfg(; with_gqsp, d), N3_HAM, 0.05;
                                     rxx_table = tab, hamiltonian = "h")
            s = bgt.steps
            # Substep partition.
            S_fwd = s.oft_substeps_per_pass
            S_bwd = s.oft_substeps_per_pass
            S_coh = s.b_substeps_per_step
            @test S_fwd + S_bwd + S_coh == s.substeps_per_step
            # Block partition — per-step quantities.
            B_fwd = s.r_D
            B_bwd = s.r_D
            B_coh = s.blocks_per_step - 2 * s.r_D
            @test B_fwd + B_bwd + B_coh == s.blocks_per_step
            @test B_coh == s.n_be_queries * (2 * s.r_bm + 3 * s.r_bp)
            # Independent re-derivation of rxx_total_controlled from the partition.
            f1, f2 = bgt.f_ctrl1, bgt.f_ctrl2
            slope, intercept = bgt.rxx_per_substep, bgt.rxx_intercept
            per_step = slope * (f1 * S_fwd + f2 * S_bwd + f2 * S_coh) +
                       intercept * (f1 * B_fwd + f2 * B_bwd + f2 * B_coh)
            @test bgt.rxx_total_controlled ≈ s.n_steps * per_step rtol = TOL_EXACT
            # b_coh must be the PER-STEP block count, not total_blocks − 2·r_D
            # (the only way the two coincide is n_steps == 1).
            @test (B_coh == s.total_blocks - 2 * s.r_D) == (s.n_steps == 1)
            # Controlled ≥ plain when both control factors ≥ 1.
            @test bgt.rxx_total_controlled > bgt.rxx_total
        end
    end

    # ---- estimate_rxx_count: GNS controlled collapse + NaN propagation ----
    @testset "estimate_rxx_count — GNS controlled + legacy NaN" begin
        tab_full = Dict(("h", 3) => (;
            geometry = "-", rxx_slope_per_substep = 20.0, rxx_intercept = 3.0,
            rxx_L1 = 23.0, rxx_fit_max_abs_dev = 0.0, f_ctrl1 = 3.0, f_ctrl2 = 6.0,
            qiskit_version = "2.4.1"))
        # GNS: no coherent term ⇒ S_coh = 0, B_coh = blocks_per_step − 2·r_D = 0.
        cfg_gns = Config(;
            sim = Thermalize(), domain = TrotterDomain(), construction = GNS(),
            num_qubits = 3, with_linear_combination = true,
            beta = BETA, sigma = SIGMA, a = BETA / 30.0, s = 0.4,
            num_energy_bits = 3, w0 = W0, t0 = 2π / (2^3 * W0),
            num_trotter_steps_per_t0 = 2, mixing_time = 0.05, delta = 0.01,
        )
        bg = estimate_rxx_count(cfg_gns, N3_HAM, 0.05; rxx_table = tab_full, hamiltonian = "h")
        s = bg.steps
        @test s.b_substeps_per_step == 0
        @test s.blocks_per_step - 2 * s.r_D == 0
        # Controlled collapses to f1·OFT-fwd + f2·OFT-bwd only.
        per_step = 20.0 * (3.0 * s.oft_substeps_per_pass + 6.0 * s.oft_substeps_per_pass) +
                   3.0 * (3.0 * s.r_D + 6.0 * s.r_D)
        @test bg.rxx_total_controlled ≈ s.n_steps * per_step rtol = TOL_EXACT
        @test isfinite(bg.rxx_total) && isfinite(bg.rxx_total_controlled)

        # Legacy table (f_ctrl2 = NaN): plain stays finite, controlled is NaN.
        cfg_kms = Config(;
            sim = Thermalize(), domain = TrotterDomain(), construction = KMS(),
            num_qubits = 3, with_linear_combination = true,
            beta = BETA, sigma = SIGMA, a = BETA / 30.0, s = 0.4,
            num_energy_bits = 3, w0 = W0, t0 = 2π / (2^3 * W0),
            num_trotter_steps_per_t0 = 2, mixing_time = 0.05, delta = 0.01,
            with_gqsp = true, gqsp_degree = 1,
        )
        tab_legacy = Dict(("h", 3) => (;
            geometry = "-", rxx_slope_per_substep = 20.0, rxx_intercept = 3.0,
            rxx_L1 = 23.0, rxx_fit_max_abs_dev = 0.0, f_ctrl1 = 3.0, f_ctrl2 = NaN,
            qiskit_version = "2.4.1"))
        bl = estimate_rxx_count(cfg_kms, N3_HAM, 0.05; rxx_table = tab_legacy, hamiltonian = "h")
        @test isfinite(bl.rxx_total)        # plain never uses f1/f2
        @test isnan(bl.rxx_total_controlled)
        @test bl.rxx_total == 20140.0       # unchanged from the f-aware case
    end

    # ---- count_trotter_steps: DLL is coherent (with_coherent(DLL)=true) ----
    # DLL Trotter dynamics is not implemented: KMS-like resource
    # arithmetic is available only as an explicitly labelled hypothetical.
    @testset "count_trotter_steps — DLL coherent handling" begin
        function _dll_trotter_resource_config(filter)
            Config(;
                sim = Thermalize(), domain = TrotterDomain(), construction = DLL(),
                num_qubits = NUM_QUBITS, with_linear_combination = true,
                beta = BETA, sigma = SIGMA, a = BETA / 30.0, s = 0.4,
                num_energy_bits = NUM_ENERGY_BITS, w0 = W0, t0 = T0,
                num_trotter_steps_per_t0 = NUM_TROTTER_STEPS_PER_T0,
                mixing_time = 1.0, delta = TEST_DELTA,
                filter = filter,
            )
        end

        # Hypothetical opt-in skips only the unsupported-domain rejection; all
        # other configuration contracts, including the DLL filter, still run.
        @test_throws ArgumentError count_trotter_steps(
            _dll_trotter_resource_config(nothing), TEST_HAM, 10.0;
            allow_hypothetical = true)

        cfg = _dll_trotter_resource_config(DLLGaussianFilter(BETA))
        default_err = try
            count_trotter_steps(cfg, TEST_HAM, 10.0)
            nothing
        catch err
            err
        end
        @test default_err isa ArgumentError
        @test occursin("allow_hypothetical=true", sprint(showerror, default_err))
        @test_throws ArgumentError compute_simulation_time(cfg, TEST_HAM, 10.0)

        cnt = count_trotter_steps(
            cfg, TEST_HAM, 10.0; allow_hypothetical = true)
        @test cnt.construction == :DLL
        @test cnt.cost_interpretation === :hypothetical_unimplemented_dll_trotter
        @test cnt.n_be_queries == 1                       # Direct exp(-iδB) requires one query.
        @test cnt.b_substeps_per_be > 0
        @test cnt.b_substeps_per_step == cnt.b_substeps_per_be
        @test cnt.blocks_per_step == 2 * NUM_ENERGY_BITS + (2 + 3) * NUM_ENERGY_BITS
        @test occursin("hypothetical unimplemented DLL Trotter model", sprint(show, cnt))
        @test occursin("construction is not implemented",
            sprint(show, MIME"text/plain"(), cnt))

        table = Dict(("dll", NUM_QUBITS) => (;
            geometry = "-", rxx_slope_per_substep = 20.0,
            rxx_intercept = 3.0, rxx_L1 = 23.0,
            rxx_fit_max_abs_dev = 0.0, f_ctrl1 = 3.0, f_ctrl2 = 6.0,
            qiskit_version = "test"))
        @test_throws ArgumentError estimate_rxx_count(
            cfg, TEST_HAM, 10.0; rxx_table = table, hamiltonian = "dll")
        rxx = estimate_rxx_count(
            cfg, TEST_HAM, 10.0;
            rxx_table = table,
            hamiltonian = "dll",
            allow_hypothetical = true,
        )
        @test rxx.steps.cost_interpretation ===
              :hypothetical_unimplemented_dll_trotter
        @test occursin("hypothetical unimplemented DLL Trotter model", sprint(show, rxx))
    end

    # ---- count_trotter_steps: coherent-leg M/r validation via fallback ----
    # If a coherent leg's M (or r) is unset AND the legacy fallback is also unset,
    # the per-leg accessor returns nothing and the counter must throw — not
    # silently build a ladder from `nothing`.
    @testset "count_trotter_steps — coherent leg M/r fallback validation" begin
        # M_D set per-leg, but b-leg M and legacy num_trotter_steps_per_t0 unset.
        cfg_no_bM = Config(;
            sim = Thermalize(), domain = TrotterDomain(), construction = KMS(),
            num_qubits = NUM_QUBITS, with_linear_combination = true,
            beta = BETA, sigma = SIGMA, a = BETA / 30.0, s = 0.4,
            num_energy_bits = NUM_ENERGY_BITS, w0 = W0, t0 = T0,
            num_trotter_steps_per_t0_D = 5,
            mixing_time = 1.0, delta = TEST_DELTA,
        )
        @test register_M_b_minus(cfg_no_bM) === nothing
        @test_throws ArgumentError count_trotter_steps(cfg_no_bM, TEST_HAM, 10.0)

        # b-register r unset (only D registers set, no legacy num_energy_bits).
        cfg_no_brm = Config(;
            sim = Thermalize(), domain = TrotterDomain(), construction = KMS(),
            num_qubits = NUM_QUBITS, with_linear_combination = true,
            beta = BETA, sigma = SIGMA, a = BETA / 30.0, s = 0.4,
            num_energy_bits_D = 6, w0_D = W0, t0_D = 2π / (2^6 * W0),
            num_trotter_steps_per_t0 = 5,
            mixing_time = 1.0, delta = TEST_DELTA,
        )
        @test register_r_b_minus(cfg_no_brm) === nothing
        @test_throws ArgumentError count_trotter_steps(cfg_no_brm, TEST_HAM, 10.0)
    end

    # ---- load_rxx_table: malformed column counts + blank-line skipping ----
    @testset "load_rxx_table — malformed rows + blank lines" begin
        mktempdir() do dir
            hdr10 = "name\tn\tgeometry\trxx_slope_per_substep\trxx_intercept\trxx_L1\trxx_fit_max_abs_dev\tf_ctrl1\tf_ctrl2\tqiskit_version"
            # 11 columns → ArgumentError.
            p11 = joinpath(dir, "p11.tsv")
            open(p11, "w") do io
                println(io, hdr10 * "\textra")
                println(io, "h\t3\t-\t20.0\t3.0\t23\t0.0\t2.9\t6.7\t2.4.1\tEXTRA")
            end
            @test_throws ArgumentError load_rxx_table(p11)

            # 8 columns → ArgumentError.
            p8 = joinpath(dir, "p8.tsv")
            open(p8, "w") do io
                println(io, "name\tn\tgeometry\trxx_slope_per_substep\trxx_intercept\trxx_L1\trxx_fit_max_abs_dev\tf_ctrl1")
                println(io, "h\t3\t-\t20.0\t3.0\t23\t0.0\t2.9")
            end
            @test_throws ArgumentError load_rxx_table(p8)

            # Blank / whitespace-only lines are skipped, not parsed.
            pblank = joinpath(dir, "pblank.tsv")
            open(pblank, "w") do io
                println(io, hdr10)
                println(io, "h\t3\t-\t20.0\t3.0\t23\t0.0\t2.9\t6.7\t2.4.1")
                println(io, "   ")
                println(io, "")
            end
            tb = load_rxx_table(pblank)
            @test length(tb) == 1
            @test tb[("h", 3)].f_ctrl2 == 6.7

        end
    end
end
