@testset "sweep_channel_mixing harness" begin
    using LinearAlgebra
    using BSON
    using QuantumFurnace: predict_channel_trajectory, _load_hamiltonian_bson,
        _load_channel_param_table, _lookup_channel_params, _build_channel_config,
        _jumps_in_basis, _channel_sweep_sidecar_path,
        _channel_sweep_cost_fields

    # The smoke cell for P0b: n=3, β=10, ε=1e-3, smooth-Metro KMS, TimeDomain.
    # Downstream sweeps may pick TrotterDomain (canonical KMS coherent uses the
    # TrotterTriple — three independent per-leg Strang caches).
    param_table = QuantumFurnace._package_data_path("channel_param_table.bson")
    ham_path = test_hamiltonian_path(3)

    isfile(param_table) || error("missing required channel parameter table: $param_table")
    isfile(ham_path) || error("missing required Hamiltonian fixture: $ham_path")
        @testset "smoke: matches direct predict_channel_trajectory" begin
            results = sweep_channel_mixing(
                [3], [10.0];
                target_epsilons = [1e-3],
                filter_kinds = [:smooth_metro],
                domain = TimeDomain(),
                construction = KMS(),
                seeds = [42],
                krylovdim = 30,
                k_grid_max_log = 4,
                k_grid_length = 40,
                output_dir = nothing,
            )
            @test length(results) == 1
            r = results[1]
            @test r.n == 3
            @test r.beta == 10.0
            @test r.eps == 1e-3
            @test r.filter === :smooth_metro
            @test r.construction == "KMS"
            @test r.domain == "Time"
            @test r.with_gqsp === true
            @test r.gqsp_degree == 1
            # The parameter table selects the current raw GQSP polynomial
            # surrogate. It may be diagnosed, but it is not a physical channel
            # and therefore has no channel gap or mixing-time resource total.
            @test r.tau_mix_source === :nonphysical_surrogate
            @test isnan(r.tau_mix)
            @test isnan(r.lambda_gap_channel)
            @test isnan(r.floor_distance)
            @test r.crossing_kind === :not_applicable
            @test r.mixing_steps === nothing
            @test r.n_steps_to_target == 0
            @test r.channel_representation ===
                :unscaled_gqsp_polynomial_surrogate
            @test !r.physical_channel
            @test r.interpretation_status === :nonphysical_surrogate
            @test isfinite(r.max_abs_trace_drift) && r.max_abs_trace_drift >= 0
            @test isfinite(r.oft_time_per_step) && r.oft_time_per_step > 0
            @test isfinite(r.b_time_per_step)
            @test r.per_step_time ≈ 2.0 * r.oft_time_per_step + r.b_time_per_step rtol=1e-12
            @test r.n_steps_total == 0
            @test isnan(r.total_ham_sim_time)
            @test r.cost_interpretation ===
                :formal_unscaled_gqsp_surrogate

            # Independent run with the same param-table row reproduces the gap and τ_mix exactly.
            rows = _load_channel_param_table(param_table)
            row  = _lookup_channel_params(rows, 3, 10.0, 1e-3, :smooth_metro)
            ham  = _load_hamiltonian_bson(ham_path, 10.0)
            cfg  = _build_channel_config(row, 3, 10.0, TimeDomain(), KMS())
            jumps = _jumps_in_basis(3, ham.eigvecs)
            d = size(ham.data, 1)
            rho_0 = Matrix{ComplexF64}(I(d) ./ d)
            k_grid = unique(round.(Int, exp10.(range(0, 4, length=40))))
            res_direct = predict_channel_trajectory(cfg, ham, jumps, rho_0, k_grid;
                                                     krylovdim=30)

            @test !res_direct.physical_channel
            @test isnan(res_direct.spectral_gap)
            @test_throws ArgumentError eigenmode_mixing_time(res_direct, 1e-3)

            # Exact channel crossings must bypass floating-point ceil(T/delta)
            # in resource accounting. This k deliberately hits a known binary
            # rounding case for delta=1e-3.
            exact_step_probe = 1001
            exact_budget = compute_simulation_time(
                cfg, ham, exact_step_probe * cfg.delta;
                n_steps=exact_step_probe)
            @test exact_budget.n_steps == exact_step_probe
            @test exact_budget.cost_interpretation ===
                :formal_unscaled_gqsp_surrogate
            @test isapprox(r.achieved_dist_at_kmax, res_direct.distances[end], rtol=1e-12)
            @test isapprox(r.max_abs_trace_drift,
                res_direct.max_abs_trace_drift; atol=0, rtol=1e-12)
        end

        @testset "non-crossing rows never receive physical totals" begin
            diagnostic_budget = (
                n_steps=37,
                total_time=12.5,
                cost_interpretation=:physical_deterministic_channel,
            )
            expected_labels = (
                floor=:unavailable_asymptotic_floor,
                certified_no_crossing=:unavailable_certified_no_crossing,
                horizon_exhausted=:unavailable_horizon_exhausted,
            )
            for source in keys(expected_labels)
                fields = _channel_sweep_cost_fields(
                    true, source, diagnostic_budget)
                @test fields.n_steps_total == 0
                @test isnan(fields.total_ham_sim_time)
                @test fields.cost_interpretation === expected_labels[source]
                @test fields.cost_interpretation !== :physical_mixing_time
            end
        end

        @testset "physical sweep preserves the exact integer crossing budget" begin
            rows = _load_channel_param_table(param_table)
            template = _lookup_channel_params(
                rows, 3, 10.0, 1e-3, :smooth_metro)

            # On this deterministic direct-coherent channel, the distance at
            # k=6 is above 0.472 and at k=7 is below it. The product 7*0.01 is
            # also the binary-rounding case that previously became eight steps
            # after ceil(T/delta) inside the resource estimator.
            physical_row = merge(template, (
                eps = 0.472,
                delta = 0.01,
                with_gqsp = false,
            ))
            zero_step_row = merge(physical_row, (eps = 0.49,))
            exhausted_row = merge(physical_row, (eps = 1e-3,))
            mktempdir() do tmp
                physical_table = joinpath(tmp, "physical_channel_table.bson")
                BSON.bson(physical_table,
                    Dict(:rows => [physical_row, zero_step_row, exhausted_row]))
                results = sweep_channel_mixing(
                    [3], [10.0];
                    target_epsilons = [0.472, 0.49, 1e-3],
                    filter_kinds = [:smooth_metro],
                    domain = TimeDomain(),
                    construction = KMS(),
                    param_table_bson = physical_table,
                    seeds = [42],
                    krylovdim = 30,
                    k_grid_max_log = 1,
                    k_grid_length = 10,
                    mixing_step_horizon = 7,
                    output_dir = nothing,
                )
                @test length(results) == 3
                r = only(filter(x -> x.eps == 0.472, results))
                @test r.physical_channel
                @test r.channel_representation === :deterministic_cptp
                @test r.tau_mix_source === :extrapolated
                @test r.crossing_kind === :integer_steps
                @test r.mixing_steps == 7
                @test r.n_steps_to_target == 7
                @test r.n_steps_total == 7
                @test r.tau_mix ≈ 0.07 atol=0 rtol=8eps(Float64)
                @test r.total_ham_sim_time ≈
                    7 * r.per_step_time rtol=1e-12
                @test r.cost_interpretation === :physical_mixing_time

                at_start = only(filter(x -> x.eps == 0.49, results))
                @test at_start.tau_mix_source === :extrapolated
                @test at_start.mixing_steps == 0
                @test at_start.n_steps_to_target == 0
                @test at_start.n_steps_total == 0
                @test at_start.tau_mix == 0.0
                @test at_start.total_ham_sim_time == 0.0

                exhausted = only(filter(x -> x.eps == 1e-3, results))
                @test exhausted.tau_mix_source === :horizon_exhausted
                @test exhausted.mixing_search_horizon == 7
                @test exhausted.mixing_horizon_capped
                @test exhausted.mixing_steps === nothing
                @test isnan(exhausted.tau_mix)
                @test exhausted.n_steps_total == 0
                @test isnan(exhausted.total_ham_sim_time)
                @test exhausted.cost_interpretation ===
                    :unavailable_horizon_exhausted
            end
        end

        @testset "BSON sidecars + skip_existing" begin
            mktempdir() do tmp
                # First run: writes the sidecar.
                r1 = sweep_channel_mixing(
                    [3], [10.0];
                    target_epsilons = [1e-3],
                    filter_kinds = [:smooth_metro],
                    domain = TimeDomain(),
                    construction = KMS(),
                    seeds = [42],
                    krylovdim = 30, k_grid_max_log = 4, k_grid_length = 40,
                    output_dir = tmp,
                )
                @test length(r1) == 1
                sidecar = _channel_sweep_sidecar_path(tmp, 3, 10.0, 42, 1e-3,
                                                       :smooth_metro, "KMS", "Time")
                @test isfile(sidecar)

                # Second run with skip_existing=true: should hit the cache.
                r2 = sweep_channel_mixing(
                    [3], [10.0];
                    target_epsilons = [1e-3],
                    filter_kinds = [:smooth_metro],
                    domain = TimeDomain(),
                    construction = KMS(),
                    seeds = [42],
                    krylovdim = 30, k_grid_max_log = 4, k_grid_length = 40,
                    output_dir = tmp, skip_existing = true,
                )
                @test length(r2) == 1
                @test isequal(Dict(pairs(r2[1])), Dict(pairs(r1[1])))
            end
        end

        @testset "(z) surrogate floor and gap are not reported as physical" begin
            results = sweep_channel_mixing(
                [3], [10.0];
                target_epsilons = [1e-3],
                filter_kinds = [:smooth_metro],
                domain = TimeDomain(),
                construction = KMS(),
                seeds = [42],
                krylovdim = 30, k_grid_max_log = 4, k_grid_length = 40,
                output_dir = nothing,
            )
            r = results[1]
            @test r.tau_mix_source === :nonphysical_surrogate
            @test isnan(r.floor_distance)
            @test isnan(r.lambda_gap_channel)
            @test isnan(r.tau_mix)
            @test isfinite(r.max_abs_trace_drift)
        end

        @testset "(zz) Gaussian surrogate is guarded identically" begin
            results = sweep_channel_mixing(
                [3], [10.0];
                target_epsilons = [1e-3],   # below floor at this fixture
                filter_kinds = [:gaussian],
                domain = TimeDomain(),
                construction = KMS(),
                seeds = [42],
                krylovdim = 30, k_grid_max_log = 4, k_grid_length = 40,
                output_dir = nothing,
            )
            r = results[1]
            @test r.tau_mix_source === :nonphysical_surrogate
            @test !r.physical_channel
            @test isnan(r.tau_mix)
            @test isnan(r.lambda_gap_channel)
            @test isnan(r.total_ham_sim_time)
        end

        @testset "multi-cell expansion" begin
            # Both filters remain explicitly nonphysical until contractive
            # GQSP synthesis is implemented.
            results = sweep_channel_mixing(
                [3], [10.0];
                target_epsilons = [1e-3],
                filter_kinds = [:smooth_metro, :gaussian],
                domain = TimeDomain(),
                construction = KMS(),
                seeds = [42],
                krylovdim = 30, k_grid_max_log = 4, k_grid_length = 40,
                output_dir = nothing,
            )
            @test length(results) == 2
            for r in results
                @test r.n == 3
                @test r.beta == 10.0
                @test r.eps == 1e-3
                @test r.tau_mix_source === :nonphysical_surrogate
                @test isnan(r.lambda_gap_channel)
                @test isnan(r.tau_mix)
                @test !r.physical_channel
            end
            @test results[1].filter === :smooth_metro
            @test results[2].filter === :gaussian
        end
end
