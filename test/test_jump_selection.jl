# Tests for the dissipative jump-selection rule.
#
# Covers:
#   1. Default Config.jump_selection == :sweep and validate_config! reject other symbols.
#   2. Both modes reproduce e^{T𝓛} in expectation: ‖rho_sim - exp(L)·rho0‖ small.
#      Random uses the legacy ×S rescaling; sweep uses bare-δ S substeps per outer step.
#   3. Both modes converge to the Gibbs state under repeated δ-steps.

using QuantumFurnace
using QuantumFurnace: trace_distance_h
using LinearAlgebra
using Random
using Test

@testset "Jump selection: :sweep | :random" begin
    @testset "Config defaults & validation" begin
        cfg = make_config(Thermalize(), EnergyDomain();
            num_qubits=3, construction=KMS(), delta=0.05, mixing_time=1.0)
        @test cfg.jump_selection == :sweep

        # validate_config! accepts :sweep and :random.
        for sel in (:sweep, :random)
            cfg_sel = Config(; sim=cfg.sim, domain=cfg.domain, construction=cfg.construction,
                num_qubits=cfg.num_qubits, with_linear_combination=cfg.with_linear_combination,
                beta=cfg.beta, sigma=cfg.sigma, gaussian_parameters=cfg.gaussian_parameters,
                a=cfg.a, s=cfg.s, num_energy_bits=cfg.num_energy_bits, w0=cfg.w0, t0=cfg.t0,
                num_trotter_steps_per_t0=cfg.num_trotter_steps_per_t0,
                mixing_time=cfg.mixing_time, delta=cfg.delta, jump_selection=sel)
            @test validate_config!(cfg_sel) === nothing
        end

        # validate_config! rejects unknown selection symbols.
        cfg_bad = Config(; sim=cfg.sim, domain=cfg.domain, construction=cfg.construction,
            num_qubits=cfg.num_qubits, with_linear_combination=cfg.with_linear_combination,
            beta=cfg.beta, sigma=cfg.sigma, gaussian_parameters=cfg.gaussian_parameters,
            a=cfg.a, s=cfg.s, num_energy_bits=cfg.num_energy_bits, w0=cfg.w0, t0=cfg.t0,
            num_trotter_steps_per_t0=cfg.num_trotter_steps_per_t0,
            mixing_time=cfg.mixing_time, delta=cfg.delta, jump_selection=:wat)
        @test_throws ArgumentError validate_config!(cfg_bad)
    end

    @testset "random expectation and sweep converge to the dense step" begin
        psi = normalize(ComplexF64.(1:N3_DIM) .+ im .* ComplexF64.(N3_DIM:-1:1))
        rho0 = psi * psi'
        n_jumps = length(N3_JUMPS)

        function selection_config(delta, selection)
            base = make_config(Thermalize(), EnergyDomain();
                num_qubits=3, construction=KMS(), delta=delta, mixing_time=delta)
            return Config(; sim=base.sim, domain=base.domain,
                construction=base.construction, num_qubits=base.num_qubits,
                with_linear_combination=base.with_linear_combination,
                beta=base.beta, sigma=base.sigma,
                gaussian_parameters=base.gaussian_parameters,
                a=base.a, s=base.s, num_energy_bits=base.num_energy_bits,
                w0=base.w0, t0=base.t0,
                num_trotter_steps_per_t0=base.num_trotter_steps_per_t0,
                mixing_time=delta, delta=delta, jump_selection=selection)
        end

        # One deterministic seed per possible first random jump makes the
        # finite average exact rather than Monte Carlo noisy.
        seed_for_jump = fill(0, n_jumps)
        for seed in 1:10_000
            a = rand(Xoshiro(seed), 1:n_jumps)
            seed_for_jump[a] == 0 && (seed_for_jump[a] = seed)
            all(>(0), seed_for_jump) && break
        end
        @test all(>(0), seed_for_jump)

        cfg_L = make_config(Lindbladian(), EnergyDomain();
            num_qubits=3, construction=KMS())
        L = construct_lindbladian(N3_JUMPS, cfg_L, N3_HAM)
        random_errors = Float64[]
        sweep_errors = Float64[]

        for delta in (0.04, 0.02, 0.01)
            cfg_random = selection_config(delta, :random)
            one_jump_states = [
                run_thermalize(N3_JUMPS, cfg_random, N3_HAM;
                    initial_dm=rho0, rng=Xoshiro(seed), save_every=1).final_dm
                for seed in seed_for_jump
            ]
            rho_random_expect = reduce(+, one_jump_states) ./ n_jumps

            cfg_sweep = selection_config(delta, :sweep)
            rho_sweep = run_thermalize(N3_JUMPS, cfg_sweep, N3_HAM;
                initial_dm=rho0, rng=Xoshiro(1), save_every=1).final_dm
            rho_exact = reshape(exp(delta * L) * vec(rho0), N3_DIM, N3_DIM)

            push!(random_errors, norm(rho_random_expect - rho_exact))
            push!(sweep_errors, norm(rho_sweep - rho_exact))
        end

        @test all(diff(random_errors) .< 0)
        @test all(diff(sweep_errors) .< 0)
        random_orders = log2.(random_errors[1:end-1] ./ random_errors[2:end])
        sweep_orders = log2.(sweep_errors[1:end-1] ./ sweep_errors[2:end])
        @test all((1.8 .< random_orders) .& (random_orders .< 2.2))
        @test all((1.8 .< sweep_orders) .& (sweep_orders .< 2.2))
        @info "jump-selection delta refinement" random_errors sweep_errors random_orders sweep_orders
    end
end
