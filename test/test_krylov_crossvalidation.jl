using Test
using LinearAlgebra
using Printf
using QuantumFurnace

# test_helpers.jl is already included by runtests.jl

# ============================================================================
# Compare Krylov and dense spectra for four-qubit KMS and GNS fixtures
# across all four domains, then test channel-to-Lindbladian convergence.
# ============================================================================

# ---------------------------------------------------------------------------
# Helper: compare Krylov and dense eigsolve on the same system
# ---------------------------------------------------------------------------

"""
    compare_krylov_dense(config_liouv, hamiltonian, jumps; kwargs...) -> NamedTuple

Run both Krylov matrix-free eigsolve and dense eigen() on the same Lindbladian,
returning structured comparison results.

Returns `(; krylov_result, dense_result, L_dense)`.
"""
function compare_krylov_dense(config_liouv, hamiltonian, jumps;
    trotter=nothing,
    krylovdim=30,
    howmany=4,
    tol=1e-10,
    n_dense_modes=4,
)
    # Dense reference: build full Lindbladian and extract leading eigendata
    L_dense = construct_lindbladian(jumps, config_liouv, hamiltonian; trotter=trotter)
    dense_result = extract_leading_eigendata(L_dense; n_modes=n_dense_modes)

    # Krylov result via matrix-free eigsolve
    krylov_result = krylov_spectral_gap(config_liouv, hamiltonian, jumps;
        trotter=trotter, krylovdim=krylovdim, howmany=howmany, tol=tol)

    return (; krylov_result, dense_result, L_dense)
end

# ---------------------------------------------------------------------------
# Diagnostic printing helpers
# ---------------------------------------------------------------------------

"""
    print_gap_summary(domain_name, balance_name, krylov_gap, dense_gap)

Print one-line summary: domain, balance, gap values, absolute and relative error.
Always printed on both success and failure for a record of achieved accuracy.
"""
function print_gap_summary(domain_name, balance_name, krylov_gap, dense_gap)
    err = abs(krylov_gap - dense_gap)
    rel_err = err / max(dense_gap, 1e-30)
    @printf("  %-15s %-5s | gap_krylov=%.8e  gap_dense=%.8e  err=%.2e  rel=%.2e\n",
            domain_name, balance_name, krylov_gap, dense_gap, err, rel_err)
end

"""
    print_eigenvalue_table(krylov_eigs, dense_eigs; top_k=6)

Print formatted comparison table of top-k eigenvalue real parts from Krylov vs dense.
"""
function print_eigenvalue_table(krylov_eigs, dense_eigs; top_k=6)
    n = min(top_k, length(krylov_eigs), length(dense_eigs))
    println("  Top-$n eigenvalue comparison:")
    println("  idx | Re(krylov)        | Re(dense)         | abs_err      | rel_err")
    println("  " * "-"^75)
    for k in 1:n
        kr = real(krylov_eigs[k])
        dr = real(dense_eigs[k])
        ae = abs(kr - dr)
        re = ae / max(abs(dr), 1e-30)
        @printf("  %3d | %+.10e | %+.10e | %.2e | %.2e\n", k, kr, dr, ae, re)
    end
end

"""
    on_failure_diagnostics(krylov_result, dense_result)

Print diagnostic information on test failure: top-k eigenvalue table and
Krylov convergence metadata.
"""
function on_failure_diagnostics(krylov_result, dense_result)
    println("  FAILURE DIAGNOSTICS:")
    print_eigenvalue_table(krylov_result.eigenvalues, dense_result.eigenvalues; top_k=6)
    @printf("  Krylov converged: %d, matvec_count: %d, restarts: %d\n",
            krylov_result.converged, krylov_result.matvec_count, krylov_result.num_restarts)
end

# ---------------------------------------------------------------------------
# L-vs-E convergence analysis helper
# ---------------------------------------------------------------------------

"""
    _dense_phi_delta_gap(config_therm, hamiltonian, jumps; trotter) -> Float64

Dense channel operator gap: materialise the faithful Φ_δ as a d²×d² superoperator
(column-by-column via `apply_delta_channel!`, the same kernel `run_thermalize`
uses), take its second-largest-|μ| non-stationary eigenvalue μ₂, and return the
Lindbladian-unit gap `-log|μ₂|/δ` (the exact log-rate of |μ| under iteration). At n=4 (DIM=16) the
d²×d² eigendecomposition is trivial and is the SAME ground-truth tier already used
for the dense 𝓛 reference (`extract_leading_eigendata`).

The dense channel spectrum supplies an independent reference at this small
dimension. As δ decreases, channel eigenvalues cluster near one; a partial
Krylov spectrum can miss the slowest mode and distort convergence orders.

    run_le_convergence(domain, hamiltonian, jumps; kwargs...) -> NamedTuple

Compute L-vs-E convergence: compare Lindbladian spectral gap (delta-independent)
against the DENSE channel-derived gap at multiple delta values.

The faithful Chen channel has mu = exp(delta*lambda_L) + O(delta^2), so the
first-order conversion (via `-log|mu_2|/delta`) introduces O(delta) error.

Returns `(; gap_L, rows, orders)` where rows is a vector of
`(; delta, gap_from_E, error)` named tuples and orders is a vector of
convergence orders from consecutive error pairs.
"""
function _dense_phi_delta_gap(config_therm, hamiltonian, jumps; trotter=nothing)
    d = size(hamiltonian.data, 1)
    delta = float(config_therm.delta)
    ws = Workspace(config_therm, hamiltonian, jumps; trotter=trotter)
    Phi = zeros(ComplexF64, d * d, d * d)
    E = zeros(ComplexF64, d, d)
    for col in 1:(d * d)
        fill!(E, 0); E[col] = 1.0
        QuantumFurnace.apply_delta_channel!(
            ws, E, config_therm, hamiltonian; hermitize=false)
        @views Phi[:, col] .= vec(ws.scratch.rho_next)
    end
    mu = eigvals(Phi)
    stationary_idx = argmin(abs.(mu .- 1))
    nonstationary = collect(eachindex(mu))
    filter!(i -> i != stationary_idx, nonstationary)
    sort!(nonstationary; by=i -> abs(mu[i]), rev=true)
    mu2 = mu[first(nonstationary)]
    return -log(abs(mu2)) / delta           # Lindbladian-unit gap (log-rate of |μ|)
end

function run_le_convergence(domain, hamiltonian, jumps;
    deltas=[0.1, 0.01, 0.001],
    trotter=nothing,
    krylovdim=30,
    tol=1e-10,
)
    # Delta-independent Lindbladian reference gap from the matrix-free solver.
    config_liouv = make_config(Lindbladian(),domain; construction=KMS())
    gap_L = krylov_spectral_gap(config_liouv, hamiltonian, jumps;
        trotter=trotter, krylovdim=krylovdim, howmany=4, tol=tol).spectral_gap

    rows = NamedTuple{(:delta, :gap_from_E, :error), Tuple{Float64, Float64, Float64}}[]
    for delta in deltas
        config_therm = make_config(Thermalize(),domain; construction=KMS(), delta=delta)
        # dense Φ_δ gap (NOT the fragile channel-Arnoldi) — see the helper
        # docstring above. This is what reliably exhibits the O(δ) channel→𝓛
        # convergence on the build_heis_1d stiff fixture.
        gap_from_E = _dense_phi_delta_gap(config_therm, hamiltonian, jumps; trotter=trotter)
        error_val = abs(gap_L - gap_from_E)
        push!(rows, (; delta, gap_from_E, error=error_val))
    end

    # Compute convergence orders from consecutive error pairs
    orders = Float64[]
    for i in 2:length(rows)
        if rows[i-1].error > 0 && rows[i].error > 0
            ratio = log(rows[i-1].error / rows[i].error) / log(rows[i-1].delta / rows[i].delta)
            push!(orders, ratio)
        end
    end

    # Print formatted convergence table
    domain_name = replace(string(typeof(domain)), "Domain" => "")
    println("  L-vs-E convergence ($domain_name, KMS):")
    @printf("  %-10s | %-18s | %-18s | %-12s | %s\n",
            "delta", "gap_L", "gap_from_E", "error", "order")
    println("  " * "-"^75)
    for (idx, row) in enumerate(rows)
        order_str = idx == 1 ? "  --" : @sprintf("%.2f", orders[idx-1])
        @printf("  %.2e | %.12e | %.12e | %.2e | %s\n",
                row.delta, gap_L, row.gap_from_E, row.error, order_str)
    end

    return (; gap_L, rows, orders)
end

# ============================================================================
# Test suites
# ============================================================================

@testset "Krylov Cross-Validation" begin

    # ========================================================================
    # n=4 KMS cross-validation across all 4 domains
    # Tolerance: atol=1e-8 (KrylovKit tol=1e-10 provides margin)
    # ========================================================================
    @testset "n=4 KMS (all domains)" begin

        # Threshold rationale (atol=1e-8): KrylovKit tol=1e-10 bounds eigenvalue error.
        # atol=1e-8 gives 100x margin for iterative convergence variability on n=4 (DIM=16).
        @testset "EnergyDomain" begin
            comp = compare_krylov_dense(
                make_config(Lindbladian(),EnergyDomain(); construction=KMS()),
                TEST_HAM, TEST_JUMPS)
            print_gap_summary("EnergyDomain", "KMS",
                comp.krylov_result.spectral_gap, comp.dense_result.spectral_gap)
            gap_match = isapprox(comp.krylov_result.spectral_gap,
                comp.dense_result.spectral_gap; atol=1e-8)
            if !gap_match
                on_failure_diagnostics(comp.krylov_result, comp.dense_result)
            end
            @test gap_match
            @info "gap (EnergyDomain KMS)" error=abs(comp.krylov_result.spectral_gap - comp.dense_result.spectral_gap) atol=1e-8
        end

        @testset "TimeDomain" begin
            comp = compare_krylov_dense(
                make_config(Lindbladian(),TimeDomain(); construction=KMS()),
                TEST_HAM, TEST_JUMPS)
            print_gap_summary("TimeDomain", "KMS",
                comp.krylov_result.spectral_gap, comp.dense_result.spectral_gap)
            gap_match = isapprox(comp.krylov_result.spectral_gap,
                comp.dense_result.spectral_gap; atol=1e-8)
            if !gap_match
                on_failure_diagnostics(comp.krylov_result, comp.dense_result)
            end
            @test gap_match
            @info "gap (TimeDomain KMS)" error=abs(comp.krylov_result.spectral_gap - comp.dense_result.spectral_gap) atol=1e-8
        end

        @testset "TrotterDomain" begin
            comp = compare_krylov_dense(
                make_config(Lindbladian(),TrotterDomain(); construction=KMS()),
                TEST_HAM, TEST_TROTTER_JUMPS;
                trotter=TEST_TROTTER)
            print_gap_summary("TrotterDomain", "KMS",
                comp.krylov_result.spectral_gap, comp.dense_result.spectral_gap)
            gap_match = isapprox(comp.krylov_result.spectral_gap,
                comp.dense_result.spectral_gap; atol=1e-8)
            if !gap_match
                on_failure_diagnostics(comp.krylov_result, comp.dense_result)
            end
            @test gap_match
            @info "gap (TrotterDomain KMS)" error=abs(comp.krylov_result.spectral_gap - comp.dense_result.spectral_gap) atol=1e-8
        end

        @testset "BohrDomain" begin
            comp = compare_krylov_dense(
                make_config(Lindbladian(),BohrDomain(); construction=KMS()),
                TEST_HAM, TEST_JUMPS)
            print_gap_summary("BohrDomain", "KMS",
                comp.krylov_result.spectral_gap, comp.dense_result.spectral_gap)
            gap_match = isapprox(comp.krylov_result.spectral_gap,
                comp.dense_result.spectral_gap; atol=1e-8)
            if !gap_match
                on_failure_diagnostics(comp.krylov_result, comp.dense_result)
            end
            @test gap_match
            @info "gap (BohrDomain KMS)" error=abs(comp.krylov_result.spectral_gap - comp.dense_result.spectral_gap) atol=1e-8
        end

    end  # n=4 KMS

    # ========================================================================
    # n=4 GNS cross-validation across all 4 domains
    # GNS uses make_config(Lindbladian(), ...; construction=GNS()) (with_coherent=false)
    # Tolerance: atol=1e-8
    # ========================================================================
    @testset "n=4 GNS (all domains)" begin

        # Threshold rationale (atol=1e-8): same as KMS -- KrylovKit tol=1e-10, 100x margin.
        @testset "EnergyDomain" begin
            comp = compare_krylov_dense(
                make_config(Lindbladian(), EnergyDomain(); construction=GNS()),
                TEST_HAM, TEST_JUMPS)
            print_gap_summary("EnergyDomain", "GNS",
                comp.krylov_result.spectral_gap, comp.dense_result.spectral_gap)
            gap_match = isapprox(comp.krylov_result.spectral_gap,
                comp.dense_result.spectral_gap; atol=1e-8)
            if !gap_match
                on_failure_diagnostics(comp.krylov_result, comp.dense_result)
            end
            @test gap_match
            @info "gap (EnergyDomain GNS)" error=abs(comp.krylov_result.spectral_gap - comp.dense_result.spectral_gap) atol=1e-8
        end

        @testset "TimeDomain" begin
            comp = compare_krylov_dense(
                make_config(Lindbladian(), TimeDomain(); construction=GNS()),
                TEST_HAM, TEST_JUMPS)
            print_gap_summary("TimeDomain", "GNS",
                comp.krylov_result.spectral_gap, comp.dense_result.spectral_gap)
            gap_match = isapprox(comp.krylov_result.spectral_gap,
                comp.dense_result.spectral_gap; atol=1e-8)
            if !gap_match
                on_failure_diagnostics(comp.krylov_result, comp.dense_result)
            end
            @test gap_match
            @info "gap (TimeDomain GNS)" error=abs(comp.krylov_result.spectral_gap - comp.dense_result.spectral_gap) atol=1e-8
        end

        @testset "TrotterDomain" begin
            comp = compare_krylov_dense(
                make_config(Lindbladian(), TrotterDomain(); construction=GNS()),
                TEST_HAM, TEST_TROTTER_JUMPS;
                trotter=TEST_TROTTER)
            print_gap_summary("TrotterDomain", "GNS",
                comp.krylov_result.spectral_gap, comp.dense_result.spectral_gap)
            gap_match = isapprox(comp.krylov_result.spectral_gap,
                comp.dense_result.spectral_gap; atol=1e-8)
            if !gap_match
                on_failure_diagnostics(comp.krylov_result, comp.dense_result)
            end
            @test gap_match
            @info "gap (TrotterDomain GNS)" error=abs(comp.krylov_result.spectral_gap - comp.dense_result.spectral_gap) atol=1e-8
        end

        @testset "BohrDomain" begin
            comp = compare_krylov_dense(
                make_config(Lindbladian(), BohrDomain(); construction=GNS()),
                TEST_HAM, TEST_JUMPS)
            print_gap_summary("BohrDomain", "GNS",
                comp.krylov_result.spectral_gap, comp.dense_result.spectral_gap)
            gap_match = isapprox(comp.krylov_result.spectral_gap,
                comp.dense_result.spectral_gap; atol=1e-8)
            if !gap_match
                on_failure_diagnostics(comp.krylov_result, comp.dense_result)
            end
            @test gap_match
            @info "gap (BohrDomain GNS)" error=abs(comp.krylov_result.spectral_gap - comp.dense_result.spectral_gap) atol=1e-8
        end

    end  # n=4 GNS

    # ========================================================================
    # L-vs-E convergence (KMS)
    # Tests that channel-to-Lindbladian gap mapping converges with O(delta).
    # The faithful jumpwise Φ_δ gives mu = exp(delta*lambda_L) + O(delta^2),
    # so (mu-1)/delta has first-order error. Deltas: [0.1, 0.01, 0.001].
    #
    # The dense complex-linear channel removes Arnoldi mode-selection noise,
    # so every consecutive pair must show the predicted first-order rate.
    # ========================================================================
    @testset "L-vs-E convergence (KMS)" begin

        for domain in (EnergyDomain(), TimeDomain(), TrotterDomain(), BohrDomain())
            jumps = domain isa TrotterDomain ? TEST_TROTTER_JUMPS : TEST_JUMPS
            trotter = domain isa TrotterDomain ? TEST_TROTTER : nothing
            result = run_le_convergence(
                domain, TEST_HAM, jumps; trotter=trotter)
            errors = getproperty.(result.rows, :error)
            @test all(diff(errors) .< 0)
            @test all(0.8 .<= result.orders .<= 1.2)
            @info "channel-to-generator convergence" domain=typeof(domain) errors orders=result.orders
        end

    end  # L-vs-E convergence

end  # @testset "Krylov Cross-Validation"
