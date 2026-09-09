"""
    _precompute_coherent_B(
        jumps,
        hamiltonian,
        config,
        precomputed_data;
        trotter=nothing,
    ) -> Union{Nothing, Matrix{<:Complex}}

Return the scaled coherent operator, or `nothing` for a construction without one.
"""
function _precompute_coherent_B(
    jumps::AbstractVector{<:JumpOp},
    ham_or_trott::Union{HamHam, AbstractTrotter},
    config::Config,
    precomputed_data;
    )

    with_coherent(config.construction) || return nothing
    if _is_joint_ckg(config)
        if config.domain isa TimeDomain
            data=config.transition_weight.time_data
            return _dll_coherent_from_g_tt(jumps,ham_or_trott,data.g_tt,data.times,data.step;backend=data.backend)
        end
        CT=Complex{eltype(ham_or_trott.eigvals)}
        R=zeros(CT,size(ham_or_trott.data))
        _accumulate_R_total!(R,[Matrix{CT}(j.in_eigenbasis) for j in jumps],
            fill(false,length(jumps)),precomputed_data,config,ham_or_trott)
        return _dll_coherent_from_loss(R,ham_or_trott.eigvals,config.beta)
    end

    if config.domain isa TimeDomain
        (; b_minus, b_plus, gamma_norm_factor) = precomputed_data
        # Outer and inner kernels use independent register spacings.
        B = B_time(jumps, ham_or_trott, b_minus, b_plus,
            register_t0_b_minus(config), register_t0_b_plus(config),
            config.beta, config.sigma)

    elseif config.domain isa TrotterDomain
        (; b_minus, b_plus, gamma_norm_factor) = precomputed_data
        @assert ham_or_trott !== nothing
        B = B_trotter(jumps, ham_or_trott, b_minus, b_plus,
            register_t0_b_minus(config), register_t0_b_plus(config),
            config.beta, config.sigma)

    else
        # BohrDomain / EnergyDomain
        (; gamma_norm_factor) = precomputed_data
        B = B_bohr(ham_or_trott, jumps, config)
    end

    rmul!(B, gamma_norm_factor)
    return B
end

"""
    _precompute_coherent_B(jumps, ham_or_trott, config::Config{<:Any, BohrDomain, DLL}, precomputed_data)
    _precompute_coherent_B(jumps, ham_or_trott, config::Config{<:Any, TimeDomain, DLL}, precomputed_data)

Return the DLL coherent operator without additional rate normalisation.

The DLL frequency filter already contains the KMS factor.
"""
function _precompute_coherent_B(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam,
    config::Config{<:AbstractSimulation, BohrDomain, DLL},
    precomputed_data,
    )
    if hasproperty(precomputed_data,:source_data)
        return sum(k->_precompute_coherent_B(JumpOp[jumps[k]],hamiltonian,config,precomputed_data.source_data[k]),eachindex(jumps))
    end
    (; filter) = precomputed_data
    return dll_coherent_op_bohr(jumps, hamiltonian, filter, config.beta)
end

function _precompute_coherent_B(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam,
    config::Config{<:AbstractSimulation, TimeDomain, DLL},
    precomputed_data,
    )
    return _dll_time_compilation(jumps,hamiltonian,config,precomputed_data).B
end

"""
    _precompute_coherent_unitary(
        jumps::AbstractVector{<:JumpOp},
        hamiltonian::HamHam,
        config::Config{Thermalize},
        precomputed_data;
        trotter::Union{Nothing, AbstractTrotter}=nothing,
    ) -> Union{Nothing, Vector{Matrix{<:Complex}}}

Precompute the per-jump coherent channel unitaries.

# Returns
Matrices `\$U_k = exp(-i delta B_k)\$`, or `nothing` when the construction has
no coherent term.
"""
function _precompute_coherent_unitary(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam,
    config::Config{Thermalize},
    precomputed_data;
    trotter::Union{Nothing, AbstractTrotter}=nothing,
    delta_scale::Real = 1.0
    )

    with_coherent(config.construction) || return nothing

    delta = delta_scale * config.delta
    CT = Complex{eltype(hamiltonian.eigvals)}
    U_terms = Vector{Matrix{CT}}(undef, length(jumps))

    if config.domain isa TimeDomain
        (; b_minus, b_plus, gamma_norm_factor) = precomputed_data
        # Independent outer and inner integration grids.
        t0_outer = register_t0_b_minus(config)
        t0_inner = register_t0_b_plus(config)
        @inbounds for (k, jump) in pairs(jumps)
            B = B_time([jump], hamiltonian, b_minus, b_plus, t0_outer, t0_inner, config.beta, config.sigma)
            rmul!(B, gamma_norm_factor)
            U_terms[k] = _coherent_unitary_step(jump, B, precomputed_data,
                t0_outer, t0_inner, delta, config.with_gqsp, config.gqsp_degree)
        end

    elseif config.domain isa TrotterDomain
        (; b_minus, b_plus, gamma_norm_factor) = precomputed_data
        @assert trotter !== nothing
        # The integration grids are independent of the Trotter substep.
        t0_outer = register_t0_b_minus(config)
        t0_inner = register_t0_b_plus(config)
        @inbounds for (k, jump) in pairs(jumps)
            B = B_trotter([jump], trotter, b_minus, b_plus, t0_outer, t0_inner, config.beta, config.sigma)
            rmul!(B, gamma_norm_factor)
            U_terms[k] = _coherent_unitary_step(jump, B, precomputed_data,
                t0_outer, t0_inner, delta, config.with_gqsp, config.gqsp_degree)
        end

    else
        # BohrDomain / EnergyDomain — validate_config! prevents with_gqsp here
        (; gamma_norm_factor) = precomputed_data
        @inbounds for (k, jump) in pairs(jumps)
            B = B_bohr(hamiltonian, [jump], config)
            rmul!(B, gamma_norm_factor)
            U_terms[k] = exp(-1im * delta * Hermitian(B))
        end
    end
    return U_terms
end

"""
    B_time(jumps, hamiltonian, b_minus, b_plus, t0_outer, t0_inner, beta, sigma)

Construct the KMS coherent correction by nested time quadrature.

# Arguments
- `jumps`, `hamiltonian`: Operators and spectral data.
- `b_minus`, `b_plus`: Truncated outer and inner kernels.
- `t0_outer`, `t0_inner`: Independent Riemann-sum spacings.
- `beta`, `sigma`: Algorithm-side temperature and Gaussian width.

# Returns
The coherent matrix in the Hamiltonian eigenbasis. Conjugate-paired inner
samples and real outer weights preserve Hermiticity.
"""
function B_time(jumps::AbstractVector{<:JumpOp}, hamiltonian::HamHam,
        b_minus, b_plus, t0_outer::Real, t0_inner::Real, beta, sigma)

    inner = _coherent_inner_sum(jumps, hamiltonian.eigvals, b_plus, beta, nothing)
    B = _coherent_outer_sum(hamiltonian.eigvals, inner, b_minus, sigma, nothing)
    return B .* (t0_outer * t0_inner)
end

# Each task owns its matrices; serial execution uses the same quadrature kernel.
function _coherent_sum(kernel!::F, n, d, ::Type{CT}, args...) where {F,CT}
    result = zeros(CT, d, d)
    n == 0 && return result
    if Threads.nthreads() == 1 || n < OMEGA_THREAD_THRESHOLD
        kernel!(result, 1:n, args...)
        return result
    end
    chunks = _partition_range(1:n, Threads.nthreads())
    partials = [zeros(CT, d, d) for _ in chunks]
    @sync for (i, chunk) in enumerate(chunks)
        Threads.@spawn kernel!(partials[i], chunk, args...)
    end
    for partial in partials
        result .+= partial
    end
    return result
end

function _coherent_inner_sum(jumps, eigvals, b_plus, beta, step)
    labels = collect(keys(b_plus))
    CT = Complex{real(eltype(eigvals))}
    return _coherent_sum(_coherent_inner_chunk!, length(labels)*length(jumps),
        length(eigvals), CT, jumps, eigvals, b_plus, labels, beta, step)
end

function _coherent_inner_chunk!(partial::Matrix{CT}, chunk, jumps, eigvals,
    b_plus, labels, beta, step) where {CT}
    d = size(partial, 1)
    u, u2 = Vector{CT}(undef, d), Vector{CT}(undef, d)
    tmp, product = similar(partial), similar(partial)
    last_label = 0
    n_jumps = length(jumps)
    @inbounds for index in chunk
        label = (index - 1) ÷ n_jumps + 1
        jump = jumps[(index - 1) % n_jumps + 1].in_eigenbasis
        tau = labels[label]
        if label != last_label
            if step === nothing
                time = tau * beta
                @. u = exp(1im * eigvals * time)
                @. u2 = exp(-2im * eigvals * time)
            else
                steps = Int(round(tau * beta / step))
                @. u = eigvals ^ steps
                @. u2 = eigvals ^ (-2 * steps)
            end
            last_label = label
        end
        # U(t) A† U(-2t) A U(t); the right phase is not conjugated.
        @. tmp = u2 * jump
        mul!(product, jump', tmp)
        partial .+= b_plus[tau] .* u .* product .* transpose(u)
    end
    return nothing
end

function _coherent_outer_sum(eigvals, inner, b_minus, sigma, step)
    labels = collect(keys(b_minus))
    return _coherent_sum(_coherent_outer_chunk!, length(labels), length(eigvals),
        eltype(inner), eigvals, inner, b_minus, labels, sigma, step)
end

function _coherent_outer_chunk!(partial::Matrix{CT}, chunk, eigvals, inner,
    b_minus, labels, sigma, step) where {CT}
    u = Vector{CT}(undef, length(eigvals))
    @inbounds for index in chunk
        t = labels[index]
        if step === nothing
            @. u = exp(1im * eigvals * (t / sigma))
        else
            steps = Int(round(t / (sigma * step)))
            @. u = eigvals ^ steps
        end
        # U(-t) inner U(t).
        partial .+= b_minus[t] .* conj.(u) .* inner .* transpose(u)
    end
    return nothing
end

# Compatibility overload using one spacing for both quadratures.
B_time(jumps::AbstractVector{<:JumpOp}, hamiltonian::HamHam, b_minus, b_plus,
    t0::Real, beta::Real, sigma::Real) =
    B_time(jumps, hamiltonian, b_minus, b_plus, t0, t0, beta, sigma)

"""
    B_trotter(jumps, trotter, b_minus, b_plus, t0_outer, t0_inner, beta, sigma)

Construct the KMS coherent correction with Trotterised time evolutions.

The `TrottTrott` method reuses one evolution cache for both quadratures; the
`TrotterTriple` method uses independent caches and bases. The result is in the
dissipative construction basis.
"""
function B_trotter(jumps::AbstractVector{<:JumpOp}, trotter::TrottTrott,
        b_minus, b_plus, t0_outer::Real, t0_inner::Real, beta, sigma)

    inner = _coherent_inner_sum(jumps, trotter.eigvals_t0, b_plus, beta, trotter.t0)
    B = _coherent_outer_sum(trotter.eigvals_t0, inner, b_minus, sigma, trotter.t0)
    return B .* (t0_outer * t0_inner)
end

# Compatibility overload using the Trotter step for both quadratures.
B_trotter(jumps::AbstractVector{<:JumpOp}, trotter::TrottTrott, b_minus, b_plus,
    beta::Real, sigma::Real) =
    B_trotter(jumps, trotter, b_minus, b_plus, trotter.t0, trotter.t0, beta, sigma)

# Basis convention: $R_(Y <- X) = V_Y^dagger V_X$ and
# $M_Y = R_(Y <- X) M_X R_(Y <- X)^dagger$.
function B_trotter(
    jumps::AbstractVector{<:JumpOp},
    triple::TrotterTriple,
    b_minus, b_plus,
    t0_outer::Real, t0_inner::Real,
    beta, sigma,
)
    d  = size(triple.D.eigvecs, 1)
    CT = Complex{eltype(triple.D.bohr_freqs)}

    # Rotate jumps from the dissipative basis to the inner-kernel basis.
    R_bp_in_D = triple.R_bp_in_D
    jumps_bp  = Vector{JumpOp}(undef, length(jumps))
    @inbounds for (k, j) in pairs(jumps)
        j_bp = Matrix{CT}(R_bp_in_D * j.in_eigenbasis * R_bp_in_D')
        jumps_bp[k] = JumpOp(j.data, j_bp, j.orthogonal, j.hermitian)
    end

    b_plus_summand_bp = _coherent_inner_sum(jumps_bp, triple.b_plus.eigvals_t0,
        b_plus, beta, triple.b_plus.t0)

    # Rotate the inner sum to the outer-kernel basis.
    R_bm_in_bp        = triple.R_bm_in_bp
    b_plus_summand_bm = Matrix{CT}(R_bm_in_bp * b_plus_summand_bp * R_bm_in_bp')

    B_bm = _coherent_outer_sum(triple.b_minus.eigvals_t0, b_plus_summand_bm,
        b_minus, sigma, triple.b_minus.t0)

    # Return to the dissipative basis.
    R_bm_in_D = triple.R_bm_in_D
    B_D = Matrix{CT}(R_bm_in_D' * B_bm * R_bm_in_D)

    return B_D .* (t0_outer * t0_inner)  # B in V_D
end

# Compatibility overload using the dissipative step for both quadratures.
B_trotter(jumps::AbstractVector{<:JumpOp}, triple::TrotterTriple, b_minus, b_plus,
    beta::Real, sigma::Real) =
    B_trotter(jumps, triple, b_minus, b_plus, triple.D.t0, triple.D.t0, beta, sigma)

"""
    _coherent_unitary_step(jump, B, precomputed_data, t0_outer, t0_inner, delta_eff,
                           with_gqsp, gqsp_degree) -> Matrix{<:Complex}

Return one coherent channel step from an already normalised Hermitian `B`.

The exact branch evaluates `exp(-i delta_eff B)`; the GQSP branch evaluates a
degree-`gqsp_degree` Jacobi–Anger truncation of the block-encoded operator.
This raw truncation is not rescaled to satisfy the GQSP unit-supremum condition
and is therefore a polynomial surrogate, not a certified postselected block.
Callers apply it without renormalising and must report trace loss or gain.
"""
function _coherent_unitary_step(
    jump::JumpOp,
    B::AbstractMatrix{<:Complex},
    precomputed_data,
    t0_outer::Real,
    t0_inner::Real,
    delta_eff::Real,
    with_gqsp::Bool,
    gqsp_degree::Int,
)
    hermitianize!(B)
    if with_gqsp
        α_be = _gqsp_block_encoding_alpha(jump,
            precomputed_data.b_minus, precomputed_data.b_plus,
            t0_outer, t0_inner, precomputed_data.gamma_norm_factor)
        return _gqsp_apply_polynomial(B, α_be, delta_eff, gqsp_degree)
    else
        return exp(-1im * delta_eff * Hermitian(B))
    end
end

# Compatibility overload using one spacing for both quadratures.
_coherent_unitary_step(jump, B, precomputed_data, t0_sim::Real, delta_eff::Real,
    with_gqsp::Bool, gqsp_degree::Int) =
    _coherent_unitary_step(jump, B, precomputed_data, t0_sim, t0_sim, delta_eff,
        with_gqsp, gqsp_degree)

"""
    _gqsp_block_encoding_alpha(jump, b_minus, b_plus, t0_outer, t0_inner,
                               gamma_norm_factor) -> Real

Return an upper bound on the coherent block-encoding norm.

Math: `alpha_a = gamma_norm_factor t0_outer t0_inner
norm(b_minus)_1 norm(b_plus)_1 norm(A_a)_op^2`, which ensures
`norm(B_a / alpha_a)_op <= 1`.
"""
function _gqsp_block_encoding_alpha(
    jump::JumpOp,
    b_minus,
    b_plus,
    t0_outer::Real,
    t0_inner::Real,
    gamma_norm_factor::Real,
)
    l1_minus = sum(abs, values(b_minus))
    l1_plus  = sum(abs, values(b_plus))
    A_norm_sq = opnorm(jump.data)^2
    return gamma_norm_factor * t0_outer * t0_inner * l1_minus * l1_plus * A_norm_sq
end

# Compatibility overload using one spacing for both quadratures.
_gqsp_block_encoding_alpha(jump, b_minus, b_plus, t0_sim::Real, gamma_norm_factor::Real) =
    _gqsp_block_encoding_alpha(jump, b_minus, b_plus, t0_sim, t0_sim, gamma_norm_factor)

"""
    _gqsp_apply_polynomial(B::AbstractMatrix{<:Complex}, alpha::Real, delta::Real, d::Int)

Evaluate the raw degree-`d` Jacobi–Anger polynomial by Clenshaw recurrence.

Math: `f_d(B/alpha) = J_0(delta alpha) I +
sum_(k=1)^d 2 (-i)^k J_k(delta alpha) T_k(B/alpha)`.

# Returns
A matrix approximating `exp(-i delta B)` with Bessel-tail error
`O((delta alpha)^(d+1))`. The truncation is generally neither Hermitian nor
unitary and can have norm above one. Without contractive rescaling and a
complementary polynomial it is not the postselected block of a GQSP unitary.
"""
function _gqsp_apply_polynomial(
    B::AbstractMatrix{<:Complex},
    alpha::Real,
    delta::Real,
    d::Int,
)
    @assert d ≥ 1 "gqsp_degree must be ≥ 1 (got $d)"
    n = LinearAlgebra.checksquare(B)
    CT = eltype(B)

    delta_alpha = delta * alpha
    a0 = CT(besselj(0, delta_alpha))

    # Degree one needs no matrix multiplication.
    if d == 1
        a1_over_alpha = CT(-2im * besselj(1, delta_alpha) / alpha)
        result = Matrix{CT}(undef, n, n)
        @inbounds for j in 1:n, i in 1:n
            result[i, j] = a1_over_alpha * B[i, j]
        end
        @inbounds for i in 1:n
            result[i, i] += a0
        end
        return result
    end

    # Math: Clenshaw uses $b_k = a_k I + 2 x b_(k+1) - b_(k+2)$.
    inv_alpha = inv(alpha)
    x = Matrix{CT}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        x[i, j] = B[i, j] * inv_alpha
    end

    b_kp2 = zeros(CT, n, n)
    b_kp1 = zeros(CT, n, n)
    b_k   = Matrix{CT}(undef, n, n)
    tmp   = Matrix{CT}(undef, n, n)

    @inbounds for k in d:-1:1
        ak = CT(2 * cis(-π/2 * k) * besselj(k, delta_alpha))
        mul!(tmp, x, b_kp1)
        @. b_k = 2 * tmp - b_kp2
        for i in 1:n
            b_k[i, i] += ak
        end
        # Rotate the three recurrence buffers without allocating.
        b_kp2, b_kp1, b_k = b_kp1, b_k, b_kp2
    end

    result = Matrix{CT}(undef, n, n)
    mul!(result, x, b_kp1)
    @. result = result - b_kp2
    @inbounds for i in 1:n
        result[i, i] += a0
    end
    return result
end

function _compute_b_minus(t::Real, beta::Real, sigma::Real)  # 2pi sqrt(pi) * f_minus(t / sigma_E)
    f1(t) = 1 / cosh(2 * pi * t / (beta * sigma))
    f2(t) = sin(-t * beta * sigma) * exp(-2 * t^2)
    return 2 * sqrt(pi) * exp(beta^2 * sigma^2 / 8) * _convolute(f1, f2, t) / (beta * sigma)
end

function _compute_b_plus(t::Real, beta::Real, w_gamma::Real, sigma_gamma::Real)  # f_plus(t * beta) / (2pi sqrt(pi))
    return beta * sigma_gamma * exp(- 2 * beta * w_gamma * (2 * t^2 + im * t)) / sqrt(pi^3)
end

function _compute_b_plus_metro(t::Real, beta::Real, sigma::Real, eta::Real, s::Real=0.0)
    if abs(t) < 1e-12  # Handle t=0 (L'Hopital limit; reduces to 1/(2√2 π²) at σβ=1, s=0)
        return complex((2 - sigma^2 * beta^2 * (1 + s)) / (2 * sqrt(2) * pi^2))
    elseif abs(t) <= eta
        numerator = exp(- sigma^2 * beta^2 * (2 * t^2 + 1im * t) * (1 + s)) + 1im * (2 * t + 1im)
    else
        numerator = exp(- sigma^2 * beta^2 * (2 * t^2 + 1im * t) * (1 + s))
    end
    denominator = t * (2 * t + 1im)
    return (1 / (2 * sqrt(2) * pi^2)) * numerator / denominator
end

function _compute_b_plus_smooth(t::Real, beta::Real, sigma::Real, a::Real, s::Real)
    b_vals = exp(- a * s / 2) * exp(- sigma^2 * beta^2 * t * (2 * t + 1im) * (1 + s)) / (4 * t^2 + a + 2im * t)
    return sqrt(4 * a + 1) * b_vals / (sqrt(2) * pi^2)
end

function _compute_truncated_func(target_func::Function, time_labels::AbstractVector{<:Real}, fixed_args...; atol::Real = 1e-12)
    f_vals = Vector{ComplexF64}(target_func.(time_labels, fixed_args...))
    indices_to_keep = _get_truncated_indices(f_vals; atol=atol)
    return Dict(zip(time_labels[indices_to_keep], f_vals[indices_to_keep]))
end

function _get_truncated_indices(fvals::AbstractVector{<:Number}; atol::Real = 1e-12)
    return findall(abs.(fvals) .>= atol)
end

function _convolute(f::Function, g::Function, t::Real; atol=1e-12, rtol=1e-12)
    integrand(s) = f(s) * g(t - s)
    result, _ = quadgk(integrand, -Inf, Inf; atol=atol, rtol=rtol)
    return result
end
