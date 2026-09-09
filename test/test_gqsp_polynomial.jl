"""
Tests for the GQSP polynomial approximation `_gqsp_apply_polynomial` and the
block-encoding-norm helper `_gqsp_block_encoding_alpha`.

`_gqsp_apply_polynomial(B, α, δ, d)` computes the raw Jacobi–Anger
Chebyshev truncation

    f_d(B/α) = J_0(δα) I + Σ_{k=1}^{d} 2 (-i)^k J_k(δα) T_k(B/α)

obtained from qubitization + Jacobi-Anger truncation. To `O((δα)^{d+1})`
this approximates `exp(-iδ B)`, but without contractive rescaling it is not
a certified postselected GQSP block.

Test plan:
1. Closed forms at d=1 and d=2 vs hand-derived formulas.
2. Reduces to identity at δ=0 (`J_0(0)=1`, `J_k(0)=0` for k≥1).
3. Slope-(d+1) δ-scaling on a fixed random Hermitian B, ‖B‖=α=1.
4. α-scaling: doubling α at fixed δ doubles δα → expect 2^(d+1) error blow-up.
5. `_gqsp_block_encoding_alpha` against a synthetic precomputed_data.
6. Operator/circuit equivalence: `_gqsp_apply_polynomial(B, α, δ, d)` matches the
   anc=|0⟩ block of `L_d(W)` extracted via the 1-ancilla qubitization recipe (the
   well-known qubitization+Jacobi-Anger identity, verified end-to-end).
7. n=3 Heisenberg anchor: slope-2 against the actual B_a from `B_time` + the kernels
   produced by `_compute_b_minus`/`_compute_b_plus` (the simulator's path).
"""

using LinearAlgebra: Hermitian, eigen, Diagonal, opnorm, ishermitian, mul!
using SpecialFunctions: besselj
using Random: MersenneTwister

# ---------------------------------------------------------------------------
# 1-ancilla qubitization (Babbush-style, ‖H‖ ≤ α).
# Direct port of `block_encoding_one_anc` + `build_walk` from
# `scripts/scratch_gqsp_random_h.jl` -- self-contained for the test.
# ---------------------------------------------------------------------------
function _qubit_block_encoding(H::AbstractMatrix, alpha::Real)
    @assert ishermitian(H) "H must be Hermitian"
    Hsym = Hermitian(Matrix(H))
    @assert opnorm(Matrix(Hsym)) ≤ alpha + 1e-10 "‖H‖ must be ≤ α (got $(opnorm(Matrix(Hsym))) > $alpha)"
    n = size(H, 1)
    H_over_alpha = Matrix(Hsym) ./ alpha
    F = eigen(Hermitian(H_over_alpha))
    sqrtIH = F.vectors * Diagonal(sqrt.(max.(1 .- F.values .^ 2, 0))) * F.vectors'
    UH = zeros(ComplexF64, 2n, 2n)
    UH[1:n, 1:n]         .= H_over_alpha
    UH[1:n, n+1:end]     .= sqrtIH
    UH[n+1:end, 1:n]     .= sqrtIH
    UH[n+1:end, n+1:end] .= -H_over_alpha
    return UH
end

function _qubit_walk(UH::AbstractMatrix)
    n2 = size(UH, 1)
    n = n2 ÷ 2
    Z_anc_I = zeros(ComplexF64, n2, n2)
    Z_anc_I[1:n, 1:n]         .= Matrix{ComplexF64}(I, n, n)
    Z_anc_I[n+1:end, n+1:end] .= -Matrix{ComplexF64}(I, n, n)
    return Z_anc_I * UH
end

function _laurent_apply(W::AbstractMatrix, delta_alpha::Real, d::Int)
    n2 = size(W, 1)
    Lop = zeros(ComplexF64, n2, n2)
    Wp = Matrix{ComplexF64}(I, n2, n2)
    Lop .+= besselj(0, delta_alpha) .* Wp
    for k in 1:d
        Wp = Wp * W
        Lop .+= cis(-π/2 * k) * besselj(k, delta_alpha) .* Wp
    end
    Wm = Matrix{ComplexF64}(I, n2, n2)
    Wdag = adjoint(W)
    for k in 1:d
        Wm = Wm * Wdag
        Lop .+= cis(π/2 * k) * besselj(-k, delta_alpha) .* Wm
    end
    return Lop
end

@testset "GQSP polynomial _gqsp_apply_polynomial" begin

    @testset "d=1 closed form: J_0(δα) I − 2i J_1(δα) (B/α)" begin
        rng = MersenneTwister(0xc0ffee)
        n = 6
        # Random Hermitian B with α = ‖B‖
        Hraw = randn(rng, ComplexF64, n, n)
        B = (Hraw + Hraw') / 2
        α = opnorm(B)
        for δ in (1e-1, 1e-2, 1e-3)
            f1 = QuantumFurnace._gqsp_apply_polynomial(B, α, δ, 1)
            ref = besselj(0, δ * α) .* Matrix{ComplexF64}(I, n, n) .-
                  2im * besselj(1, δ * α) .* (B ./ α)
            @test opnorm(f1 .- ref) < 1e-13
        end
    end

    @testset "Raw truncation is not automatically a postselected contraction" begin
        # At the spectral endpoint x=1, the degree-one truncation has norm
        # sqrt(J0(z)^2 + 4J1(z)^2) > 1. This pins the reason simulator
        # metadata labels the current path a nonphysical polynomial surrogate.
        z = 1e-2
        raw = QuantumFurnace._gqsp_apply_polynomial(
            Matrix{ComplexF64}(I, 2, 2), 1.0, z, 1)
        @test opnorm(raw) > 1
        @test opnorm(raw' * raw - I(2)) > 1e-6
    end

    @testset "d=2 closed form: (J_0+2J_2) I − 2i J_1 (B/α) − 4 J_2 (B/α)²" begin
        rng = MersenneTwister(42)
        n = 5
        Hraw = randn(rng, ComplexF64, n, n)
        B = (Hraw + Hraw') / 2
        α = opnorm(B)
        x = B ./ α
        for δ in (1e-1, 5e-2, 1e-2)
            f2 = QuantumFurnace._gqsp_apply_polynomial(B, α, δ, 2)
            j0 = besselj(0, δ * α)
            j1 = besselj(1, δ * α)
            j2 = besselj(2, δ * α)
            ref = (j0 + 2 * j2) .* Matrix{ComplexF64}(I, n, n) .-
                  2im * j1 .* x .-
                  4 * j2 .* (x * x)
            @test opnorm(f2 .- ref) < 1e-13
        end
    end

    @testset "Reduces to identity at δ=0 for any d, α" begin
        rng = MersenneTwister(1)
        n = 4
        B = let H = randn(rng, ComplexF64, n, n); (H + H') / 2 end
        α = max(opnorm(B), 1.0)
        Id = Matrix{ComplexF64}(I, n, n)
        for d in (1, 2, 3, 5)
            f0 = QuantumFurnace._gqsp_apply_polynomial(B, α, 0.0, d)
            @test opnorm(f0 .- Id) < 1e-13
        end
    end

    @testset "Bessel-tail scaling on random Hermitian, ‖B‖ = 1" begin
        # Bessel-tail bound: ‖f_d(B/α) − e^{-iδB}‖ ≤ 2 (e·δα / (2(d+1)))^{d+1}.
        # Two parametrisations of the same scaling law:
        #   axis = :delta — fix α=1, vary δ; observed slope d log(err)/d log(δ) ≈ d+1.
        #   axis = :alpha — fix δ, double α; observed ratio err(2α)/err(α) ≈ 2^(d+1).
        rng = MersenneTwister(7)
        n = 8
        Hraw = randn(rng, ComplexF64, n, n)
        B = (Hraw + Hraw') / 2
        B ./= opnorm(B)                              # ‖B‖ = 1
        for d in (1, 2, 3)
            # Axis = δ
            α = 1.0
            δs = (1e-1, 1e-2, 1e-3)
            errs = [opnorm(QuantumFurnace._gqsp_apply_polynomial(B, α, δ, d)
                           .- exp(-1im * δ .* Hermitian(B))) for δ in δs]
            slope = log(errs[2] / errs[1]) / log(δs[2] / δs[1])
            @test isapprox(slope, d + 1; atol=0.2)

            # Axis = α at fixed δ — error multiplies by ≈ 2^(d+1) when α doubles
            δ = 1e-2
            target = exp(-1im * δ .* Hermitian(B))
            e1 = opnorm(QuantumFurnace._gqsp_apply_polynomial(B, 1.0, δ, d) .- target)
            e2 = opnorm(QuantumFurnace._gqsp_apply_polynomial(B, 2.0, δ, d) .- target)
            ratio = e2 / e1
            @test 0.4 * 2^(d + 1) < ratio < 2.5 * 2^(d + 1)
        end
    end

    @testset "Joint (B,α,δ)-invariance: f_d depends only on (B/α, δα)" begin
        # f_d(B/α) = J_0(δα) I + Σ 2(-i)^k J_k(δα) T_k(B/α) is fully parameterized by
        # the pair (B/α, δα). Any rescaling that preserves both must give the same
        # operator. Verify this for (B, α, δ) → (cB, cα, δ/c).
        rng = MersenneTwister(99)
        n = 6
        Hraw = randn(rng, ComplexF64, n, n)
        B = (Hraw + Hraw') / 2
        α0 = opnorm(B)
        δ = 1e-2
        for d in (1, 2, 3)
            f_a = QuantumFurnace._gqsp_apply_polynomial(B,        α0,        δ,        d)
            f_b = QuantumFurnace._gqsp_apply_polynomial(2 .* B,   2 * α0,    δ / 2,    d)
            f_c = QuantumFurnace._gqsp_apply_polynomial(B ./ 3,   α0 / 3,    3 * δ,    d)
            @test opnorm(f_a .- f_b) < 1e-12
            @test opnorm(f_a .- f_c) < 1e-12
        end
    end

end

@testset "GQSP block-encoding norm _gqsp_block_encoding_alpha" begin
    # Synthetic dicts mirroring the truncated-func Dict format from _compute_truncated_func
    b_minus = Dict(0.0 => complex(0.5), 0.1 => complex(0.3), -0.1 => complex(0.2))
    b_plus  = Dict(0.0 => complex(1.0, 0.0), 0.05 => complex(0.0, 0.5))
    t0_sim = 0.25
    γ_nf = 2.0
    l1_minus_expected = abs(0.5) + abs(0.3) + abs(0.2)            # 1.0
    l1_plus_expected  = abs(1.0) + abs(complex(0.0, 0.5))         # 1.5

    # JumpOp with a Pauli-X jump (op-norm = 1)
    Aop = ComplexF64[0 1; 1 0]
    jump = JumpOp(Aop, Aop, false, true)  # data, in_eigenbasis, orthogonal, hermitian

    α = QuantumFurnace._gqsp_block_encoding_alpha(jump, b_minus, b_plus, t0_sim, γ_nf)
    expected = γ_nf * t0_sim^2 * l1_minus_expected * l1_plus_expected * 1.0
    @test α ≈ expected atol=1e-14

    # Sanity: scaling jump by 2 → ‖A‖² × 4
    A2 = 2 .* Aop
    jump2 = JumpOp(A2, A2, false, true)
    α2 = QuantumFurnace._gqsp_block_encoding_alpha(jump2, b_minus, b_plus, t0_sim, γ_nf)
    @test α2 ≈ 4 * α atol=1e-14
end

@testset "Jacobi–Anger operator/Laurent equivalence" begin
    # Applying the raw Laurent truncation L_d to the qubitization walk and
    # extracting its ancilla block gives the Chebyshev expansion f_d(B/α).
    # This is an algebraic identity, not a claim that L_d is itself unitary.
    rng = MersenneTwister(2026)
    n = 4
    Hraw = randn(rng, ComplexF64, n, n)
    B = (Hraw + Hraw') / 2
    α = opnorm(B) * 1.5     # leave margin so ‖B/α‖ < 1 strictly
    UH = _qubit_block_encoding(B, α)
    W  = _qubit_walk(UH)

    @test opnorm(W * W' - I(2n)) < 1e-12     # walk is unitary

    for d in (1, 2, 3)
        for δ in (1e-1, 1e-2)
            f_d_op = QuantumFurnace._gqsp_apply_polynomial(B, α, δ, d)
            L_d = _laurent_apply(W, δ * α, d)
            f_d_circ = L_d[1:n, 1:n]                 # anc=|0⟩ block
            @test opnorm(f_d_op .- f_d_circ) < 1e-10
        end
    end
end

@testset "n=3 Heisenberg slope-2 anchor" begin
    # End-to-end check: build B_a via B_time on the actual n=3 disordered Heisenberg
    # test system, compute α via the helper using the same _compute_b_minus / _compute_b_plus
    # kernels the simulator will use, and verify f_1(B_a/α) → exp(-iδ B_a) with slope 2.
    n = 3
    ham = N3_HAM
    jump = N3_JUMPS[1]   # X on site 1, normalized

    # Time grid (mirrors NUM_ENERGY_BITS=12 grid in test_helpers)
    grid = collect(range(-T0 * 2^(NUM_ENERGY_BITS - 1), T0 * (2^(NUM_ENERGY_BITS - 1) - 1); length=2^NUM_ENERGY_BITS))
    b_minus = QuantumFurnace._compute_truncated_func(QuantumFurnace._compute_b_minus, grid, BETA, SIGMA)
    # Use a Gaussian for b_plus consistent with the BETA/SIGMA choice (KMS line)
    w_gamma = BETA * (SIGMA^2 + SIGMA^2) / 2  # solve β = 2 ω_γ / (σ² + σ_γ²) with σ_γ = σ
    σ_γ = SIGMA
    b_plus = QuantumFurnace._compute_truncated_func(QuantumFurnace._compute_b_plus, grid, BETA, w_gamma, σ_γ)

    γ_nf = 1.0
    B_a = QuantumFurnace.B_time([jump], ham, b_minus, b_plus, T0, BETA, SIGMA)
    rmul!(B_a, γ_nf)
    α = QuantumFurnace._gqsp_block_encoding_alpha(jump, b_minus, b_plus, T0, γ_nf)

    @test opnorm(B_a) / α < 1.0 + 1e-10        # block encoding norm bounds B_a

    target(δ) = exp(-1im * δ .* Hermitian(B_a))
    errs = Float64[]
    δs = (1e-1, 1e-2, 1e-3)
    for δ in δs
        f1 = QuantumFurnace._gqsp_apply_polynomial(B_a, α, δ, 1)
        push!(errs, opnorm(f1 .- target(δ)))
    end
    slope = log(errs[2] / errs[1]) / log(δs[2] / δs[1])
    @test isapprox(slope, 2.0; atol=0.2)
end

@testset "Clenshaw at d ∈ {3,4,5,6} matches naive Chebyshev sum" begin
    # Check the Clenshaw branch at d > 2 against the forward recurrence.
    # Reference: build the explicit Chebyshev expansion
    #     f_d(x) = J_0(δα) I + Σ_{k=1}^d 2 (-i)^k J_k(δα) T_k(x)
    # via the forward Chebyshev recurrence T_0 = I, T_1 = x, T_{k+1} = 2 x T_k − T_{k-1}
    # (Motlagh & Wiebe 2024, Eq. 62 + Theorem 7). The Clenshaw recurrence is
    # mathematically equivalent; agreement at ≤1e-12 catches sign, (-i)^k phase,
    # and buffer-rotation bugs at higher d that the d=2 closed form cannot reach.
    rng = MersenneTwister(8675309)
    n = 6
    Hraw = randn(rng, ComplexF64, n, n)
    B = (Hraw + Hraw') / 2
    α = opnorm(B) * 1.2     # margin so ‖B/α‖ < 1 strictly
    x_mat = Matrix(B ./ α)
    Idn = Matrix{ComplexF64}(I, n, n)

    function naive_chebyshev_sum(d::Int, δα::Real)
        T_prev = copy(Idn)
        T_curr = copy(x_mat)
        f = besselj(0, δα) .* Idn
        f .+= 2 * cis(-π/2) * besselj(1, δα) .* T_curr           # k = 1
        for k in 2:d
            T_next = 2 .* (x_mat * T_curr) .- T_prev
            f .+= 2 * cis(-π/2 * k) * besselj(k, δα) .* T_next
            T_prev, T_curr = T_curr, T_next
        end
        return f
    end

    for d in (3, 4, 5, 6), δ in (1e-1, 1e-2)
        f_clenshaw = QuantumFurnace._gqsp_apply_polynomial(B, α, δ, d)
        f_naive    = naive_chebyshev_sum(d, δ * α)
        @test opnorm(f_clenshaw .- f_naive) < 1e-12
    end
end

@testset "Per-d residual shrinkage to exp(-iδB)" begin
    # Bessel-tail bound (Motlagh & Wiebe 2024, Eq. 63): J_n(t) ∈ Θ((t/2)^n / n!),
    # so the leading-order residual ratio between consecutive d is
    #     err_d / err_{d-1} ≈ (δα) / (2(d+1)),
    # i.e. each Clenshaw degree gains one factor of (δα). This quantitatively
    # ties the Clenshaw output to the asymptotic Jacobi-Anger truncation theory
    # — a check the existing slope-(d+1) test does not enforce per-step.
    rng = MersenneTwister(2718)
    n = 6
    Hraw = randn(rng, ComplexF64, n, n)
    B = (Hraw + Hraw') / 2
    B ./= opnorm(B)                 # ‖B‖ = α = 1
    δ = 0.5                         # δα = 0.5: keeps err_d ≫ machine ε up to d=5
    target = exp(-1im * δ .* Hermitian(B))
    errs = [opnorm(QuantumFurnace._gqsp_apply_polynomial(B, 1.0, δ, d) .- target) for d in 1:5]
    for d in 2:5
        ratio = errs[d] / errs[d-1]
        # Analytic leading-order ratio is ≈ (δα) / (2(d+1)). Use 2× slack to
        # absorb random-matrix variance (T_{d+1}(x) is not strictly bounded by 1
        # in operator norm for ‖x‖ < 1, only |T_{d+1}(x)| ≤ 1 on the spectrum).
        @test ratio < 2 * δ / (d + 1)
    end
end
