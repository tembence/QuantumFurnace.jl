# test/test_lindblad_action_sandbox.jl
#
# Sandbox shadow of test_lindblad_action.jl. The heavy test
# sweeps n ∈ {3, 4, 5}, β = 10 for the (q0) Energy ↔ Bohr dense-L
# 1e-9 cross-check plus several end-to-end integrator runs. This shadow
# keeps the canonical 1e-9 cross-domain invariant — the strongest
# regression check for CKG smooth-Metropolis register sizing — at the
# n = 3 cell only, plus a single integrator end-to-end run.
#
# Fixture: n = 3, β = 10, CKG smooth-Metropolis (thesis defaults
# a = 0, s = 0.25, r_D = 12).
# Dense L_super is 64 × 64 ⇒ ≪ 1 MiB. Sandbox-safe.

using LinearAlgebra: I, Hermitian, dot, eigvals, norm, tr, opnorm
using Test
using QuantumFurnace

@testset "Discriminant dephasing in equivalent generator clocks" begin
    psi_eq = Matrix{ComplexF64}(I, 2, 2) / sqrt(2)
    psi_0 = fill(ComplexF64(inv(sqrt(2))), 2, 2)
    elapsed = [0.0, 0.2, 1.0]
    for rate in (1e-20, 1.0, 1e12)
        action! = (out, x) -> (out .= rate .* (Z * x * Z - x))
        result = discriminant_action_integrate(action!, psi_0, psi_eq, elapsed ./ rate;
            krylovdim=4, tol=1e-12, save_states=true)
        @test result.all_converged
        for (t, state) in zip(elapsed, result.states)
            expected = ComplexF64[1 exp(-2t); exp(-2t) 1] / sqrt(2)
            @test state ≈ expected atol=1e-11
        end
        @test result.distances ≈ exp.(-2 .* elapsed) atol=1e-11
    end

    action! = (out, x) -> (out .= Z * x * Z - x)
    for times in (Float64[], [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, Inf], [0.0, NaN])
        @test_throws ArgumentError discriminant_action_integrate(action!, psi_0, psi_eq, times)
    end
    @test_throws ArgumentError discriminant_action_integrate(action!, psi_0, zeros(ComplexF64, 3, 3), [0.0])
    @test_throws ArgumentError discriminant_action_integrate(action!, zeros(ComplexF64, 2, 3), psi_eq, [0.0])
    for controls in ((; krylovdim=0), (; tol=0.0), (; tol=Inf), (; tol=NaN))
        @test_throws ArgumentError discriminant_action_integrate(action!, psi_0, psi_eq, [0.0]; controls...)
    end
end


@testset "Lindbladian-action integrator [sandbox shadow]" begin

    @testset "pairwise Hermitian correction under asymmetric Krylov drift" begin
        rates = ComplexF64[0 log(2); log(4) 0]
        asymmetric_action! = function (out, X)
            @. out = rates * X
            return out
        end
        rho_0 = ComplexF64[0.6 0.1 + 0.05im; 0.1 - 0.05im 0.4]
        t_grid = [0.0, 1.0]
        raw_final = exp.(rates) .* rho_0
        expected_rho = (raw_final + raw_final') / 2

        res_L = lindblad_action_integrate(
            asymmetric_action!, rho_0, zeros(ComplexF64, 2, 2), t_grid;
            krylovdim=4, tol=1e-12, save_states=true,
        )
        @test res_L.all_converged
        @test norm(res_L.rho_final - expected_rho) < 1e-12
        @test norm(res_L.rho_final - res_L.rho_final') < 1e-14

        psi_eq = Matrix{ComplexF64}(I, 2, 2) / sqrt(2)
        psi_0 = sqrt(2) .* rho_0
        expected_psi = sqrt(2) .* expected_rho
        res_K = discriminant_action_integrate(
            asymmetric_action!, psi_0, psi_eq, t_grid;
            krylovdim=4, tol=1e-12, save_states=true,
        )
        @test res_K.all_converged
        @test norm(res_K.psi_final - expected_psi) < 1e-12
        @test norm(res_K.psi_final - res_K.psi_final') < 1e-14
    end

    # -----------------------------------------------------------------------
    # (q0) Energy ↔ Bohr dense Liouvillian agreement at 1e-9 (n=3 only).
    #
    # At Eb=12, w0=0.05 the EnergyDomain Riemann sum reaches the
    # floating-point accumulation floor against the closed-form BohrDomain.
    # -----------------------------------------------------------------------
    @testset "(q0) Bohr ≈ Energy dense L at 1e-9 (n=3, β=10)" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        base_kw = (
            sim = Lindbladian(),
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = beta,
            sigma = 1.0 / beta,
            a = 0.0,
            s = 0.25,
            num_energy_bits = 12,
            w0 = 0.05,
            t0 = 2π / (2^12 * 0.05),
            num_trotter_steps_per_t0 = 10,
        )
        cfg_b = Config(; domain = BohrDomain(),   base_kw...)
        cfg_e = Config(; domain = EnergyDomain(), base_kw...)
        L_b = Matrix{ComplexF64}(construct_lindbladian(sys.jumps, cfg_b, sys.ham))
        L_e = Matrix{ComplexF64}(construct_lindbladian(sys.jumps, cfg_e, sys.ham))
        rel = opnorm(L_b - L_e) / opnorm(L_b)
        @test rel < 1e-9
        @info "(q0) sandbox Bohr ≈ Energy" beta rel threshold=1e-9
    end

    @testset "Discriminant integration analytic projector (mode=:K)" begin
        psi_eq = Matrix{ComplexF64}(I, 2, 2) / sqrt(2)
        psi_0 = sqrt(2) .* ComplexF64[1 0; 0 0]
        t_grid = [0.0, 0.25, 0.5, 1.0]
        K_apply! = function (out, X)
            equilibrium_component = dot(psi_eq, X)
            @. out = -(X - equilibrium_component * psi_eq)
            return out
        end

        res = discriminant_action_integrate(
            K_apply!, psi_0, psi_eq, t_grid;
            krylovdim=4, tol=1e-12, save_states=true,
        )
        initial_distance = norm(psi_0 - psi_eq)
        exact_distances = initial_distance .* exp.(-t_grid)
        exact_final = psi_eq + exp(-t_grid[end]) .* (psi_0 - psi_eq)

        @test res.all_converged
        @test maximum(abs.(res.distances - exact_distances)) < 1e-10
        @test norm(res.psi_final - exact_final) < 1e-10
        @test length(res.states) == length(t_grid)
        @test all(abs(dot(psi_eq, psi) - 1) < 1e-12 for psi in res.states)
    end

    # -----------------------------------------------------------------------
    # (e) Integrator end-to-end smoke: BohrDomain CKG, n=3, β=10.
    # Same horizon as the heavy testset (e) (t=120 ≈ 4·τ_mix) so the 1e-6
    # equilibrium-tail floor is genuinely resolved. Shrunk from 121 → 61
    # grid points to halve the number of Krylov-exp evaluations.
    # -----------------------------------------------------------------------
    @testset "(e) Integrator end-to-end @ n=3, β=10 (mode=:L)" begin
        beta = 10.0
        sys = make_dll_n3_system(beta)
        config = make_config(Lindbladian(), BohrDomain();
                              num_qubits = 3, construction = KMS())
        d = size(sys.ham.data, 1)
        rho_0 = Matrix{ComplexF64}(I(d) / d)
        t_grid = collect(range(0.0, 120.0, length = 61))

        res = integrate_to_gibbs(config, sys.ham, sys.jumps, rho_0, t_grid;
                                  mode = :L, krylovdim = 20, tol = 1e-10)

        @test res.all_converged
        @test res.distances[end] < 1e-6
        @test isapprox(real(tr(res.rho_final)), 1.0; atol = 1e-9)
        evs = eigvals(Hermitian((res.rho_final + res.rho_final') / 2))
        @test minimum(real.(evs)) > -1e-9

        est = estimate_mixing_time(res; model = :biexp,
                                    target_epsilon = 1e-3, extrapolate = true)
        @test isfinite(est.mixing_time)
        @test est.mixing_time > 0
        @info "(e) sandbox integrator" tau_mix=est.mixing_time dist_end=res.distances[end]
    end
end
