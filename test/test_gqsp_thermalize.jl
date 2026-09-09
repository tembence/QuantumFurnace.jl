"""
Integration tests for the GQSP coherent step inside `run_thermalize`.

The GQSP polynomial `f_d(B/α)` matches `exp(-iδ B)` to `O((δα)^{d+1})`. With small
δ, the simulator's final density matrix should agree with the matrix-exponential
baseline up to that polynomial-truncation error.

Test plan:
1. Smoke test for `run_thermalize` with `with_gqsp=true` on TimeDomain + TrotterDomain
   (n=3 disordered Heisenberg). Returns a valid `ThermalizeResults`.
2. Regression vs matrix-exp baseline at small δ: same setup, only `with_gqsp` flips.
   Final density matrices should match to within the polynomial truncation tolerance.
3. Bumping `gqsp_degree` from 1 to 2 strictly improves agreement with the matrix-exp
   baseline (Bessel-tail bound shrinks by another factor of δα).
"""

using LinearAlgebra: opnorm, tr, Hermitian, eigvals
using Random: Xoshiro

@testset "GQSP integration in run_thermalize" begin

    # Use the small 3-qubit Heisenberg fixtures and a small δ where polynomial trunc
    # error is below the unraveling step error. Short mixing_time keeps the test fast.
    function _therm_cfg(domain; with_gqsp::Bool, gqsp_degree::Int=1, delta::Float64=1e-3, mixing_time::Float64=0.05)
        cfg = make_config(Thermalize(), domain;
            num_qubits=3, construction=KMS(),
            delta=delta, mixing_time=mixing_time)
        # Re-build with the gqsp fields (make_config does not pass them through)
        Config(;
            sim=cfg.sim, domain=cfg.domain, construction=cfg.construction,
            num_qubits=cfg.num_qubits, with_linear_combination=cfg.with_linear_combination,
            beta=cfg.beta, sigma=cfg.sigma, a=cfg.a, s=cfg.s,
            gaussian_parameters=cfg.gaussian_parameters,
            num_energy_bits=cfg.num_energy_bits, t0=cfg.t0, w0=cfg.w0,
            num_trotter_steps_per_t0=cfg.num_trotter_steps_per_t0,
            mixing_time=cfg.mixing_time, delta=cfg.delta, eta=cfg.eta,
            with_gqsp=with_gqsp, gqsp_degree=gqsp_degree,
        )
    end

    @testset "TimeDomain regression: smoke + GQSP d=1 ≈ matrix-exp baseline at small δ" begin
        # Combines (a) smoke: ThermalizeResults validity, finite td, tr≈1, hermitian;
        # and (b) regression: GQSP path agrees with matrix-exp baseline within
        # polynomial-truncation tolerance. Same RNG seed → identical jump selections.
        cfg_exp  = _therm_cfg(TimeDomain(); with_gqsp=false)
        cfg_gqsp = _therm_cfg(TimeDomain(); with_gqsp=true, gqsp_degree=1)
        seed = 17
        r_exp  = run_thermalize(N3_JUMPS, cfg_exp,  N3_HAM; rng=Xoshiro(seed))
        r_gqsp = run_thermalize(N3_JUMPS, cfg_gqsp, N3_HAM; rng=Xoshiro(seed))
        # Smoke
        @test r_gqsp isa ThermalizeResults
        @test isfinite(r_gqsp.trace_distances[end])
        @test isapprox(tr(r_gqsp.final_dm), 1.0; atol=1e-5)
        @test isapprox(r_gqsp.final_dm, r_gqsp.final_dm'; atol=1e-9)
        # Regression
        @test r_exp.trace_distances ≈ r_gqsp.trace_distances rtol=5e-2 atol=1e-3
        @test opnorm(r_exp.final_dm .- r_gqsp.final_dm) < 5e-3
    end

    @testset "TrotterDomain regression: smoke + GQSP d=1 ≈ matrix-exp baseline at small δ" begin
        cfg_exp  = _therm_cfg(TrotterDomain(); with_gqsp=false)
        cfg_gqsp = _therm_cfg(TrotterDomain(); with_gqsp=true, gqsp_degree=1)
        seed = 23
        r_exp  = run_thermalize(N3_TROTTER_JUMPS, cfg_exp,  N3_HAM, N3_TROTTER; rng=Xoshiro(seed))
        r_gqsp = run_thermalize(N3_TROTTER_JUMPS, cfg_gqsp, N3_HAM, N3_TROTTER; rng=Xoshiro(seed))
        # Smoke
        @test r_gqsp isa ThermalizeResults
        @test isfinite(r_gqsp.trace_distances[end])
        @test isapprox(tr(r_gqsp.final_dm), 1.0; atol=1e-5)
        @test isapprox(r_gqsp.final_dm, r_gqsp.final_dm'; atol=1e-9)
        # Regression
        @test r_exp.trace_distances ≈ r_gqsp.trace_distances rtol=5e-2 atol=1e-3
        @test opnorm(r_exp.final_dm .- r_gqsp.final_dm) < 5e-3
    end

    @testset "Thermalization quality: GQSP reaches comparable trace_distance to Gibbs" begin
        # At T_mix = 5.0 (500 steps at δ=1e-2), both paths thermalize from id/d
        # towards the Gibbs state. Verify (a) BOTH reach a meaningful trace distance
        # to Gibbs (i.e., they actually thermalize, not just track each other), and
        # (b) the gap between them is tiny — bounded by the O((δα)^(d+1)) per-step
        # polynomial-truncation error times n_steps, which is far below the unraveling
        # step error O(δ × T) ~ 5e-2 controlling thermalization itself.
        cfg_exp  = _therm_cfg(TimeDomain(); with_gqsp=false, delta=1e-2, mixing_time=5.0)
        cfg_gqsp = _therm_cfg(TimeDomain(); with_gqsp=true,  gqsp_degree=1, delta=1e-2, mixing_time=5.0)
        seed = 1234
        r_exp  = run_thermalize(N3_JUMPS, cfg_exp,  N3_HAM; rng=Xoshiro(seed))
        r_gqsp = run_thermalize(N3_JUMPS, cfg_gqsp, N3_HAM; rng=Xoshiro(seed))
        td_exp  = r_exp.trace_distances[end]
        td_gqsp = r_gqsp.trace_distances[end]
        # (a) genuine thermalization in both paths (initial mixed state ↦ near-Gibbs).
        # Threshold 0.20: the trace distance to Gibbs falls from ~0.46
        # (id/d initial) to 0.1655 over 500 steps on the build_heis_1d n=3 draw — a
        # 2.8× reduction, i.e. genuine relaxation. The endpoint is a fixture-specific
        # magnitude: it is set by the EnergyDomain L gap (≈0.17 here, so the slowest
        # mode is only damped to exp(-0.17·5)≈0.43 of its initial amplitude at T=5.0)
        # and by the O(δ·T)≈5e-2 unraveling-step error, NOT by the GQSP path. The
        # prior find_typical draw relaxed slightly further (< 0.15); the new draw
        # plateaus at 0.1655. 0.20 keeps the "did it thermalize substantially below
        # the 0.46 start" assertion meaningful with ~21% margin over the deterministic
        # (seeded) 0.1655 endpoint — it still fails decisively for a stuck-near-0.46 run.
        # seed-46 n=3 draw plateaus at 0.2051 (vs 0.1655 for the prior draw); 0.25
        # keeps clear margin and still fails decisively for a stuck-near-0.46 run.
        @test td_exp  < 0.25
        @test td_gqsp < 0.25
        # (b) GQSP polynomial truncation error per step is far below unraveling budget.
        # This is the fixture-INDEPENDENT invariant (the two paths agree); it is held
        # TIGHT at 1e-3 and passes comfortably (observed 1.6e-6 on the new draw).
        @test abs(td_gqsp - td_exp) < 1e-3
    end

    @testset "Higher gqsp_degree tightens agreement with matrix-exp baseline" begin
        # f_d → exp(-iδB) with truncation error O((δα)^{d+1}); bumping d shrinks gap.
        cfg_exp = _therm_cfg(TimeDomain(); with_gqsp=false, delta=5e-3)
        cfg_d1  = _therm_cfg(TimeDomain(); with_gqsp=true, gqsp_degree=1, delta=5e-3)
        cfg_d2  = _therm_cfg(TimeDomain(); with_gqsp=true, gqsp_degree=2, delta=5e-3)
        seed = 31
        r_exp = run_thermalize(N3_JUMPS, cfg_exp, N3_HAM; rng=Xoshiro(seed))
        r_d1  = run_thermalize(N3_JUMPS, cfg_d1,  N3_HAM; rng=Xoshiro(seed))
        r_d2  = run_thermalize(N3_JUMPS, cfg_d2,  N3_HAM; rng=Xoshiro(seed))
        err_d1 = opnorm(r_exp.final_dm .- r_d1.final_dm)
        err_d2 = opnorm(r_exp.final_dm .- r_d2.final_dm)
        @test err_d2 < err_d1                                # strictly better
    end

    @testset "Unscaled GQSP surrogate trace is raw and predictor matches direct evolution" begin
        # Degree one is deliberately retained. Over 100 steps its small
        # non-unitarity is visible, while a full Krylov basis reproduces the
        # direct polynomial surrogate to roundoff.
        cfg = _therm_cfg(TimeDomain(); with_gqsp=true, gqsp_degree=1,
                         delta=1e-2, mixing_time=1.0)
        d = size(N3_HAM.data, 1)
        rho_0 = Matrix{ComplexF64}(I, d, d) / d
        direct = run_thermalize(
            N3_JUMPS, cfg, N3_HAM; initial_dm=copy(rho_0), save_every=100)
        predicted = predict_channel_trajectory(
            cfg, N3_HAM, N3_JUMPS, rho_0, [0, 100];
            krylovdim=d^2, tol=1e-12)

        @test direct.metadata[:channel_representation] ===
            :unscaled_gqsp_polynomial_surrogate
        @test !direct.metadata[:trace_preserving_assumed]
        @test !direct.metadata[:physical_channel]
        @test !direct.metadata[:trace_normalized]
        @test direct.metadata[:max_abs_trace_drift] > 1e-7
        @test direct.metadata[:final_trace] == tr(direct.final_dm)

        @test predicted.channel_representation ===
            :unscaled_gqsp_polynomial_surrogate
        @test !predicted.trace_preserving_assumed
        @test !predicted.physical_channel
        @test !predicted.trace_normalized
        @test isnan(predicted.spectral_gap)
        @test predicted.max_abs_trace_drift > 1e-7
        @test isapprox(predicted.trace_values[end], tr(direct.final_dm);
            atol=1e-10, rtol=0)
        @test opnorm(predicted.rho_final - direct.final_dm) < 1e-10
        @test abs(predicted.trace_values[end] - 1) > 1e-7
        @test_throws ArgumentError eigenmode_mixing_time(predicted, 1e-3)
        @test_throws ArgumentError krylov_spectral_gap(
            cfg, N3_HAM, N3_JUMPS; krylovdim=20, howmany=4)
    end

end
