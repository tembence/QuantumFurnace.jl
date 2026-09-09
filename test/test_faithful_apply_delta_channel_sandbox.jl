# test/test_faithful_apply_delta_channel_sandbox.jl
#
# Sandbox shadow of test_faithful_apply_delta_channel.jl. The
# heavy test runs 5 testsets including a 5-δ slope sweep and a 4096-OFT
# threading bit-match. This shadow keeps the two invariants that survive a
# physics-meaningful tightening:
#
#   (1) Bohr ↔ Energy faithful-channel cross-domain agreement at n=3,
#       β=10, 10 outer steps at δ=1e-3 — the cross-domain invariant for
#       apply_delta_channel!.
#   (2) Splitting-error slope ≈ 2 at 3 δ-points (not 5) to cap wall time
#       while still nailing the O(δ²) leading correction.
#
# The threading bit-match (heavy testset 5) is too expensive to shadow —
# stays NO_SANDBOX. The faithfulness-vs-run_thermalize check (heavy
# testset 1) is already covered in sandbox by test_predict_sandbox.jl::(b)
# byte-identity over 50 outer steps.

using LinearAlgebra: I, norm
using Random
using Test
using QuantumFurnace


@testset "Faithful apply_delta_channel! [sandbox shadow]" begin

    # -----------------------------------------------------------------------
    # (2-sb) Bohr ↔ Energy cross-domain agreement
    #
    # At Eb=12, w0=0.05 the measured single-step quadrature error is ~1e-9.
    # Ten outer steps give ~1e-10 accumulated error on this fixture; use 1e-8
    # tolerance to allow floating-point accumulation across jumps.
    @testset "(2-sb) Bohr ≡ Energy faithful Φ_δ @ n=3, β=10, 10 steps" begin
        beta = 10.0
        delta = 1e-3
        sys = make_dll_n3_system(beta)
        ham = sys.ham; jumps = sys.jumps
        d = size(ham.data, 1)

        # Shared kwargs across the two configs (delta = 1e-3, 10 outer steps).
        base_kw = (
            sim = Thermalize(),
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta, sigma = 1.0 / beta,
            a = 0.0, s = 0.25,
            num_energy_bits = 12, w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
            delta = delta,
            mixing_time = 0.01,   # 10 outer steps at δ=1e-3
            jump_selection = :sweep,
        )
        cfg_b = Config(; domain = BohrDomain(),   base_kw...)
        cfg_e = Config(; domain = EnergyDomain(), base_kw...)

        rho_0 = Matrix{ComplexF64}(I(d) / d)
        n_steps = 10

        ws_b = Workspace(cfg_b, ham, jumps)
        rho_b = copy(rho_0)
        for _ in 1:n_steps
            apply_delta_channel!(ws_b, rho_b, cfg_b, ham)
            rho_b = copy(ws_b.scratch.rho_next)
        end

        ws_e = Workspace(cfg_e, ham, jumps)
        rho_e = copy(rho_0)
        for _ in 1:n_steps
            apply_delta_channel!(ws_e, rho_e, cfg_e, ham)
            rho_e = copy(ws_e.scratch.rho_next)
        end

        diff_F = norm(rho_b - rho_e)
        @test diff_F < 1e-8
        @info "(2-sb) sandbox Bohr ≡ Energy" diff_F threshold=1e-8 n_steps
    end

    # -----------------------------------------------------------------------
    # (4-sb) Splitting-error slope ≈ 2 at 3 δ-points.
    #
    # Faithful per-jump Lie–Trotter Φ_δ vs Euler (I + δ·L) — both are O(δ)
    # approximants of e^{δL}; their difference is O(δ²). Linear-fit log-log
    # slope on {1e-2, 1e-3, 1e-4} (3 points spanning two decades): slope ≈
    # 2 ± 0.15. Wider tolerance than the heavy test's ±0.1 because 3-point
    # linear regression has higher residual noise than 5-point.
    # -----------------------------------------------------------------------
    @testset "(4-sb) Splitting-error slope ≈ 2 (3 δ-points)" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        ham = sys.ham; jumps = sys.jumps
        d = size(ham.data, 1)

        # Lindbladian config for dense L construction (no δ).
        cfg_L = Config(
            sim = Lindbladian(),
            domain = EnergyDomain(),
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta, sigma = 1.0 / beta,
            a = 0.0, s = 0.25,
            num_energy_bits = 12, w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
        )
        L_dense = construct_lindbladian(jumps, cfg_L, ham)

        Random.seed!(0)
        rho = Matrix(random_density_matrix(3))
        I_d2 = Matrix{ComplexF64}(I(d^2))

        deltas = [1e-2, 1e-3, 1e-4]
        errs = Float64[]
        for delta in deltas
            cfg_T = Config(
                sim = Thermalize(),
                domain = EnergyDomain(),
                construction = KMS(),
                num_qubits = 3,
                with_linear_combination = true,
                beta = beta, sigma = 1.0 / beta,
                a = 0.0, s = 0.25,
                num_energy_bits = 12, w0 = 0.05,
                t0 = 2π / (2^12 * 0.05),
                num_trotter_steps_per_t0 = 10,
                delta = delta,
                mixing_time = 1.0,
                jump_selection = :sweep,
            )
            ws = Workspace(cfg_T, ham, jumps)
            rho_in = copy(rho)
            apply_delta_channel!(ws, rho_in, cfg_T, ham)
            rho_chen = copy(ws.scratch.rho_next)

            v_euler = (I_d2 + delta .* L_dense) * vec(rho)
            push!(errs, norm(vec(rho_chen) - v_euler))
        end

        log_d = log.(deltas)
        log_e = log.(errs)
        n = length(deltas)
        Σx = sum(log_d); Σy = sum(log_e)
        Σxy = sum(log_d .* log_e); Σx2 = sum(log_d .^ 2)
        slope = (n * Σxy - Σx * Σy) / (n * Σx2 - Σx^2)

        @test 1.85 ≤ slope ≤ 2.15
        @info "(4-sb) splitting-error slope" slope deltas errs
    end
end
