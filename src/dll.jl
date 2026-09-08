# Ding–Li–Lin jump and coherent operators.
# Math: $L_a = sum_(nu in B_H) hat(f)(nu) A_nu^a$.

"""
    dll_lindblad_op_bohr(jump, hamiltonian, filter) -> Matrix

Construct one DLL Lindblad operator in the Hamiltonian eigenbasis.

# Arguments
- `jump`: Coupling operator in the Hamiltonian eigenbasis.
- `hamiltonian`: Spectral data.
- `filter`: Full DLL frequency filter, including the KMS factor.

# Returns
The matrix with entries `hat(f)(lambda_i-lambda_j) A_ij`.
"""
function dll_lindblad_op_bohr(
    jump::JumpOp,
    hamiltonian::HamHam{T},
    filter::AbstractFilter,
) where {T<:AbstractFloat}
    if _is_dll_bohr_spec(filter)
        filter = _prepare_dll_bohr_filter(filter, hamiltonian.eigvals)
    end
    _require_admissible_dll_filter(filter)
    eigvals = hamiltonian.eigvals
    A_eb = jump.in_eigenbasis
    n = length(eigvals)
    @assert size(A_eb) == (n, n)
    CT = Complex{T}
    L = zeros(CT, n, n)
    @inbounds for j in 1:n, i in 1:n
        ν_ij = eigvals[i] - eigvals[j]
        L[i, j] = freq_kernel(filter, ν_ij) * A_eb[i, j]
    end
    return L
end

"""
    dll_lindblad_op_time(jump, hamiltonian, time_labels, filter, t0) -> Matrix

Construct one DLL Lindblad operator by direct time quadrature.

# Arguments
- `jump`, `hamiltonian`: Coupling operator and spectral data.
- `time_labels`: Uniform integration grid.
- `filter`: Full DLL time kernel.
- `t0`: Grid spacing.

# Returns
The Hamiltonian-eigenbasis operator. Production workspaces amortise this sum
through a precomputed NUFFT slice.
"""
function dll_lindblad_op_time(
    jump::JumpOp,
    hamiltonian::HamHam{T},
    time_labels::AbstractVector{<:Real},
    filter::AbstractFilter,
    t0::Real,
) where {T<:AbstractFloat}
    _require_admissible_dll_filter(filter)
    eigvals = hamiltonian.eigvals
    A_eb = jump.in_eigenbasis
    n = length(eigvals)
    @assert size(A_eb) == (n, n)
    CT = Complex{T}
    L = zeros(CT, n, n)
    @inbounds for t in time_labels
        ft = time_kernel(filter, t)
        weight = ft * t0
        for j in 1:n, i in 1:n
            phase = cis((eigvals[i] - eigvals[j]) * t)
            L[i, j] += weight * phase * A_eb[i, j]
        end
    end
    return L
end

# DLL coherent kernel. The time-domain operator order is `A(t')' A(t)`, as
# obtained by substituting the Bohr decomposition into the inverse transform.

"""
    dll_coherent_kernel_bohr(filter, ν, νp) -> Complex

Evaluate the DLL coherent frequency kernel for two Bohr frequencies.
"""
@inline function dll_coherent_kernel_bohr(
    filter::AbstractFilter,
    ν::Real,
    νp::Real,
)
    if _is_dll_bohr_spec(filter)
        T = typeof(float(filter.beta))
        samples = unique!(T[0, ν, -ν, νp, -νp])
        filter = _sample_dll_bohr_filter(filter, samples, T(filter.beta))
    end
    _require_admissible_dll_filter(filter)
    T = typeof(float(filter.beta))
    β = filter.beta
    pref = one(T) / (2im)
    th = tanh(β * (νp - ν) / 4)
    fkν = freq_kernel(filter, ν)
    fkνp = freq_kernel(filter, νp)
    return Complex{T}(pref) * Complex{T}(th) * fkν * conj(fkνp)
end

"""
    dll_coherent_op_bohr(jumps, hamiltonian, filter, beta) -> Matrix

Construct the DLL coherent operator in the Hamiltonian eigenbasis.

# Arguments
- `jumps`, `hamiltonian`: Couplings and spectral data.
- `filter`: DLL frequency filter.
- `beta`: Inverse temperature, validated against the filter upstream.

# Returns
The `O(n^3)` Hermitian matrix
`G = (1/(2i)) T hadamard sum_a M_a^dagger M_a`, with
`T_ij = tanh(beta (lambda_j-lambda_i)/4)`.
"""
function dll_coherent_op_bohr(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam{T},
    filter::AbstractFilter,
    beta::Real,
) where {T<:AbstractFloat}
    if _is_dll_bohr_spec(filter)
        filter = _prepare_dll_bohr_filter(filter, hamiltonian.eigvals; beta)
    end
    _require_admissible_dll_filter(filter; beta=beta)
    eigvals = hamiltonian.eigvals
    n = length(eigvals)
    CT = Complex{T}

    f_mat = Matrix{CT}(undef, n, n)
    @inbounds for j in 1:n, k in 1:n
        f_mat[k, j] = CT(freq_kernel(filter, eigvals[k] - eigvals[j]))
    end

    # Math: $T_(i j) = tanh(beta (lambda_j-lambda_i)/4)$ is antisymmetric.
    tanh_mat = Matrix{T}(undef, n, n)
    βT = T(beta)
    @inbounds for j in 1:n, i in 1:n
        tanh_mat[i, j] = tanh(βT * (eigvals[j] - eigvals[i]) / 4)
    end

    G = zeros(CT, n, n)
    M = Matrix{CT}(undef, n, n)
    MdM = Matrix{CT}(undef, n, n)
    @inbounds for jump in jumps
        @. M = f_mat * jump.in_eigenbasis
        mul!(MdM, M', M)
        @. G += MdM
    end
    pref = CT(1) / (2im)
    @. G = pref * tanh_mat * G
    return G
end

# Production Bohr correction uses precisely the retained implemented loss.
# The standalone constructor above remains an independent source-product path.
function _dll_coherent_from_loss(R::AbstractMatrix{CT}, eigvals, beta) where {CT<:Complex}
    B = similar(R)
    @inbounds for j in axes(R, 2), i in axes(R, 1)
        B[i,j] = (im / 2) * tanh((beta / 4) * (eigvals[i]-eigvals[j])) * R[i,j]
    end
    return B
end

# The Gaussian DLL kernel separates after the variables
# $u = nu'-nu$ and $v = (nu+nu')/2$, leaving one scalar quadrature `J(s)`.
# The Gaussian tail is truncated below floating-point precision.
function _dll_J_quadrature(beta::T, s::Real; rtol::Real=1e-12, atol::Real=1e-14) where {T<:AbstractFloat}
    integrand(u) = tanh(beta * u / 4) * exp(-(beta * u)^2 / 16) * sin(u * s / 2)
    cutoff = T(24) / beta
    val, _ = quadgk(integrand, T(0), cutoff; rtol=rtol, atol=atol)
    return T(2) * val
end

# Math: $g(t,t') = I_v(t'-t) J(t+t') / (8 pi^2)$.
@inline function _dll_g_closed_form(beta::T, t::Real, tp::Real, J_val::Real) where {T<:AbstractFloat}
    δ = T(tp) - T(t)
    factor_diff = (T(2) * sqrt(T(π)) / beta) * exp(T(1) / 4 - im * δ / beta - δ^2 / beta^2)
    pref = inv(T(8) * T(π)^2)
    return Complex{T}(pref) * factor_diff * J_val
end

# Tabulate `J` once for each distinct pairwise time sum.
function _dll_J_table(beta::T, time_labels::AbstractVector{<:Real}; rtol::Real=1e-12, atol::Real=1e-14) where {T<:AbstractFloat}
    Nt = length(time_labels)
    sums_set = Set{T}()
    @inbounds for nidx in 1:Nt, m in 1:Nt
        push!(sums_set, T(time_labels[m]) + T(time_labels[nidx]))
    end
    sums_vec = sort!(collect(sums_set))
    J_vals = Vector{T}(undef, length(sums_vec))
    @inbounds for (i, s) in enumerate(sums_vec)
        J_vals[i] = _dll_J_quadrature(beta, s; rtol=rtol, atol=atol)
    end
    sum_to_index = Dict{T, Int}(s => i for (i, s) in enumerate(sums_vec))
    return J_vals, sum_to_index
end

"""
    dll_coherent_op_time(jumps, hamiltonian, time_labels, filter, beta, tau)

Construct the Gaussian-filter DLL coherent operator by time quadrature.

# Arguments
- `jumps`, `hamiltonian`: Couplings and spectral data.
- `time_labels`: Uniform quadrature grid.
- `filter`: DLL Gaussian filter.
- `beta`: Inverse temperature, validated against the filter upstream.
- `tau`: Quadrature spacing.

# Returns
The Hamiltonian-eigenbasis coherent matrix. A closed-form Gaussian kernel and
2D type-3 NUFFT evaluate `A(t')' A(t)`, or `A(t') A(t)` for Hermitian sources.
"""
function dll_coherent_op_time(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam{T},
    time_labels::AbstractVector{<:Real},
    filter::DLLGaussianFilter,
    beta::Real,
    τ::Real,
) where {T<:AbstractFloat}
    _require_admissible_dll_filter(filter; beta=beta)
    Nt = length(time_labels)
    CT = Complex{T}
    βT = T(beta)

    # Tabulate the Gaussian two-time kernel on the quadrature grid.
    J_vals, sum_to_index = _dll_J_table(βT, time_labels)
    g_tt = Matrix{CT}(undef, Nt, Nt)
    @inbounds for nidx in 1:Nt, m in 1:Nt
        s = T(time_labels[m]) + T(time_labels[nidx])
        J_val = J_vals[sum_to_index[s]]
        g_tt[m, nidx] = _dll_g_closed_form(βT, time_labels[m], time_labels[nidx], J_val)
    end

    # Contract the sampled kernel with the ordered products $A(t')^dagger A(t)$.
    return _dll_coherent_from_g_tt(jumps, hamiltonian, g_tt, time_labels, τ)
end

"""
    _dll_coherent_op_time_frequency_grid(jumps, hamiltonian, time_labels,
                                         filter, beta, tau;
                                         nu_min, nu_max, nu_grid_size)

Construct a DLL coherent operator from a controlled frequency window.

The two-dimensional trapezoidal rule samples the paper's coherent kernel
`(2i)^(-1) tanh(beta*(nu'-nu)/4) fhat(nu) conj(fhat(nu'))`, transforms it to
the time grid with a type-3 NUFFT, and contracts the ordered products
`A(t')' A(t)`.
"""
function _dll_coherent_op_time_frequency_grid(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam{T},
    time_labels::AbstractVector{<:Real},
    filter::AbstractFilter,
    beta::Real,
    τ::Real;
    nu_min::Real,
    nu_max::Real,
    nu_grid_size::Int = 256,
    backend::Symbol = :finufft,
) where {T<:AbstractFloat}
    _require_admissible_dll_filter(filter; beta=beta)
    backend in (:direct,:finufft) || throw(ArgumentError("Unknown coherent backend."))
    backend == :finufft && !(T in (Float32,Float64)) && throw(ArgumentError("Coherent FINUFFT uses Float64; select :direct for higher precision."))
    nu_grid_size >= 2 || throw(ArgumentError("nu_grid_size must be >= 2."))
    nu_lo = T(nu_min)
    nu_hi = T(nu_max)
    isfinite(nu_lo) && isfinite(nu_hi) && nu_lo < nu_hi ||
        throw(ArgumentError("Require finite nu_min < nu_max."))

    Nt = length(time_labels)
    CT = Complex{T}
    βT = T(beta)

    # Math: $hat(g)(nu,nu') = tanh(beta(nu'-nu)/4)
    # hat(f)(nu) conj(hat(f)(nu'))/(2i)$.
    Nν = nu_grid_size
    nu = collect(range(nu_lo, nu_hi; length = Nν))
    Δν = (nu_hi - nu_lo) / T(Nν - 1)
    f_vec = CT[CT(freq_kernel(filter, ν)) for ν in nu]
    g_hat = Matrix{CT}(undef, Nν, Nν)
    pref_g = CT(one(T) / (2im))
    @inbounds for q in 1:Nν, p in 1:Nν
        th = tanh(βT * (nu[q] - nu[p]) / 4)
        g_hat[p, q] = pref_g * th * f_vec[p] * conj(f_vec[q])
    end

    if backend == :direct
        weights = fill(Δν,Nν); weights[1]/=2; weights[end]/=2
        phi = CT[cis(-v*t)*w for t in time_labels for (v,w) in zip(nu,weights)]
        phi = transpose(reshape(phi,Nν,Nt))
        psi = CT[cis(v*t)*w for t in time_labels for (v,w) in zip(nu,weights)]
        psi = transpose(reshape(psi,Nν,Nt))
        g_tt = (phi*g_hat*transpose(psi))/(2T(pi))^2
        return _dll_coherent_from_g_tt(jumps,hamiltonian,g_tt,time_labels,τ;backend=:direct)
    end

    # FINUFFT uses $exp(i(s_x t_x + s_y t_y))$; negate the first source axis.
    Nsrc = Nν * Nν
    src_x = Vector{Float64}(undef, Nsrc)
    src_y = Vector{Float64}(undef, Nsrc)
    src_c = Vector{ComplexF64}(undef, Nsrc)
    idx = 0
    @inbounds for q in 1:Nν, p in 1:Nν
        idx += 1
        src_x[idx] = -Float64(nu[p])
        src_y[idx] = Float64(nu[q])
        wp = (p == 1 || p == Nν) ? T(1) / T(2) : one(T)
        wq = (q == 1 || q == Nν) ? T(1) / T(2) : one(T)
        src_c[idx] = ComplexF64(wp * wq * g_hat[p, q])
    end

    Ntgt = Nt * Nt
    tgt_x = Vector{Float64}(undef, Ntgt)
    tgt_y = Vector{Float64}(undef, Ntgt)
    @inbounds for nn in 1:Nt, m in 1:Nt
        idx_t = (nn - 1) * Nt + m
        tgt_x[idx_t] = Float64(time_labels[m])
        tgt_y[idx_t] = Float64(time_labels[nn])
    end

    plan = FINUFFT.finufft_makeplan(3, 2, +1, 1, 1e-12; dtype = Float64, nthreads = 1)
    FINUFFT.finufft_setpts!(plan, src_x, src_y, Float64[], tgt_x, tgt_y, Float64[])
    out_g = Vector{ComplexF64}(undef, Ntgt)
    FINUFFT.finufft_exec!(plan, src_c, out_g)
    FINUFFT.finufft_destroy!(plan)

    norm_factor = (Float64(Δν) / (2π))^2
    g_tt = Matrix{CT}(undef, Nt, Nt)
    @inbounds for nn in 1:Nt, m in 1:Nt
        idx_t = (nn - 1) * Nt + m
        g_tt[m, nn] = CT(out_g[idx_t] * norm_factor)
    end

    # Contract the sampled kernel with the ordered products $A(t')^dagger A(t)$.
    return _dll_coherent_from_g_tt(jumps, hamiltonian, g_tt, time_labels, τ)
end

"""
    dll_coherent_op_time(jumps, hamiltonian, time_labels,
                         filter::DLLMetropolisFilter, beta, tau;
                         nu_grid_size=256)

Construct the Metropolis-filter DLL coherent operator by time quadrature on
the exact support `[-S,S]`.
"""
function dll_coherent_op_time(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam{T},
    time_labels::AbstractVector{<:Real},
    filter::DLLMetropolisFilter{T},
    beta::Real,
    τ::Real;
    nu_grid_size::Int = 256,
) where {T<:AbstractFloat}
    _require_admissible_dll_filter(filter; beta=beta)
    return _dll_coherent_op_time_frequency_grid(
        jumps, hamiltonian, time_labels, filter, beta, τ;
        nu_min = -filter.S,
        nu_max = filter.S,
        nu_grid_size = nu_grid_size,
    )
end

"""
    _dll_coherent_from_g_tt(jumps, hamiltonian, g_tt, time_labels, tau)

Contract a sampled two-time coherent kernel into the DLL operator by NUFFT.
"""
function _dll_coherent_from_g_tt(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam{T},
    g_tt::AbstractMatrix{<:Complex},
    time_labels::AbstractVector{<:Real},
    τ::Real;
    backend::Symbol=:finufft,
) where {T<:AbstractFloat}
    eigvals = hamiltonian.eigvals
    n = length(eigvals)
    Nt = length(time_labels)
    CT = Complex{T}
    @assert size(g_tt) == (Nt, Nt)

    if backend == :direct
        G=zeros(CT,n,n)
        for k in 1:n,j in 1:n,i in 1:n
            left=CT[cis((eigvals[k]-eigvals[j])*t) for t in time_labels]
            right=CT[cis((eigvals[i]-eigvals[k])*t) for t in time_labels]
            q=sum(left .* (g_tt*right))*T(τ)^2
            for jump in jumps
                A=jump.in_eigenbasis
                G[i,j] += conj(A[k,i])*A[k,j]*q
            end
        end
        return G
    end

    # Math: $Q_(i j k) = tau^2 sum_(m,n) g_(m n)
    # exp(i[(lambda_i-lambda_k)t_n + (lambda_k-lambda_j)t_m])$.
    src_x = Vector{Float64}(undef, Nt * Nt)  # paired with γ = λ_k − λ_j
    src_y = Vector{Float64}(undef, Nt * Nt)  # paired with α = λ_i − λ_k
    src_c = Vector{ComplexF64}(undef, Nt * Nt)
    idx = 0
    @inbounds for nidx in 1:Nt, m in 1:Nt
        idx += 1
        src_x[idx] = Float64(time_labels[m])
        src_y[idx] = Float64(time_labels[nidx])
        src_c[idx] = ComplexF64(g_tt[m, nidx])
    end

    n_targets = n^3
    tgt_s = Vector{Float64}(undef, n_targets)
    tgt_t = Vector{Float64}(undef, n_targets)
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        idx_t = ((i - 1) + (j - 1) * n + (k - 1) * n^2) + 1
        tgt_s[idx_t] = Float64(eigvals[k] - eigvals[j])
        tgt_t[idx_t] = Float64(eigvals[i] - eigvals[k])
    end

    plan = FINUFFT.finufft_makeplan(3, 2, +1, 1, 1e-12; dtype=Float64, nthreads=1)
    FINUFFT.finufft_setpts!(plan, src_x, src_y, Float64[], tgt_s, tgt_t, Float64[])
    out_q = Vector{ComplexF64}(undef, n_targets)
    FINUFFT.finufft_exec!(plan, src_c, out_q)
    FINUFFT.finufft_destroy!(plan)

    Q_ijk = reshape(out_q, n, n, n) .* (τ^2)

    # Math: $G_(i j) = sum_(a,k) conj(A^a_(k i)) A^a_(k j) Q_(i j k)$.
    # Adjoint evolution has the same (lambda_i-lambda_k) phase; only the
    # source coefficient changes. Retain the Hermitian fast path.
    G = zeros(CT, n, n)
    @inbounds for jump in jumps
        A_eb = jump.in_eigenbasis
        A_left = jump.hermitian ? A_eb : adjoint(A_eb)
        for j in 1:n, i in 1:n
            acc = CT(0)
            for k in 1:n
                acc += A_left[i, k] * A_eb[k, j] * Q_ijk[i, j, k]
            end
            G[i, j] += acc
        end
    end
    return G
end

"""
    dll_coherent_op_time_legacy(jumps, hamiltonian, time_labels, filter, beta, tau;
                                nu_grid=nothing)

Reference implementation using frequency tabulation and an explicit
`O(Nt^2 n^3)` time contraction.

# Keywords
- `nu_grid`: Optional frequency grid; otherwise an adaptive Gaussian range is used.
"""
function dll_coherent_op_time_legacy(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam{T},
    time_labels::AbstractVector{<:Real},
    filter::AbstractFilter,
    beta::Real,
    τ::Real;
    nu_grid::Union{Nothing, AbstractVector{<:Real}} = nothing,
) where {T<:AbstractFloat}
    _require_admissible_dll_filter(filter; beta=beta)
    eigvals = hamiltonian.eigvals
    n = length(eigvals)
    CT = Complex{T}

    νs = if nu_grid === nothing
        ν_centre = -one(T) / T(beta)
        ν_half = T(12) / T(beta)
        Δν_target = T(beta) / 16
        Nν_pre = 2 * ceil(Int, ν_half / Δν_target)
        Nν = max(Nν_pre, 64)
        collect(range(ν_centre - ν_half, ν_centre + ν_half; length=Nν))
    else
        collect(nu_grid)
    end
    Nν = length(νs)
    Δν = νs[2] - νs[1]

    G_hat = Matrix{CT}(undef, Nν, Nν)
    @inbounds for q in 1:Nν, p in 1:Nν
        G_hat[p, q] = dll_coherent_kernel_bohr(filter, νs[p], νs[q])
    end

    Nt = length(time_labels)
    pref_g = CT(Δν * Δν / (2 * T(π))^2)

    Φ = Matrix{CT}(undef, Nt, Nν)
    Ψ = Matrix{CT}(undef, Nt, Nν)
    @inbounds for p in 1:Nν, m in 1:Nt
        Φ[m, p] = cis(-νs[p] * time_labels[m])
    end
    @inbounds for q in 1:Nν, n in 1:Nt
        Ψ[n, q] = cis(νs[q] * time_labels[n])
    end

    tmp_FH = Matrix{CT}(undef, Nt, Nν)
    g_tt = Matrix{CT}(undef, Nt, Nt)
    mul!(tmp_FH, Φ, G_hat)
    mul!(g_tt, tmp_FH, transpose(Ψ))
    rmul!(g_tt, pref_g)

    G = zeros(CT, n, n)
    weight_outer = τ^2

    phases_t = Matrix{CT}(undef, Nt, n)
    @inbounds for k in 1:n, m in 1:Nt
        phases_t[m, k] = cis(eigvals[k] * time_labels[m])
    end

    Atm = Matrix{CT}(undef, n, n)
    Atn = Matrix{CT}(undef, n, n)
    prod_buf = Matrix{CT}(undef, n, n)

    @inbounds for jump in jumps
        A_eb = jump.in_eigenbasis
        for nidx in 1:Nt
            for j in 1:n, i in 1:n
                Atn[i, j] = phases_t[nidx, i] * conj(phases_t[nidx, j]) * A_eb[i, j]
            end
            for m in 1:Nt
                for j in 1:n, i in 1:n
                    Atm[i, j] = phases_t[m, i] * conj(phases_t[m, j]) * A_eb[i, j]
                end
                mul!(prod_buf, jump.hermitian ? Atn : adjoint(Atn), Atm)
                w = g_tt[m, nidx] * weight_outer
                for j in 1:n, i in 1:n
                    G[i, j] += w * prod_buf[i, j]
                end
            end
        end
    end
    return G
end

# DLL uses one rank-one Kossakowski matrix per coupling.

"""
    dll_kossakowski_bohr(filter, bohr_freqs::AbstractVector{<:Real}) -> Matrix

Construct the rank-one DLL Kossakowski matrix on a Bohr-frequency grid.

# Arguments
- `filter`: Frequency filter used to weight each Bohr component.
- `bohr_freqs`: Ordered Bohr frequencies that label rows and columns.

# Returns
The matrix `alpha = v * v^dagger`, where `v[k] = freq_kernel(filter, bohr_freqs[k])`.
"""
function dll_kossakowski_bohr(
    filter::AbstractFilter,
    bohr_freqs::AbstractVector{<:Real},
)
    if _is_dll_bohr_spec(filter)
        T = typeof(float(filter.beta))
        samples = unique!(vcat(T[0], T.(bohr_freqs), -T.(bohr_freqs)))
        filter = _sample_dll_bohr_filter(filter, samples, T(filter.beta))
    end
    _require_admissible_dll_filter(filter)
    K = length(bohr_freqs)
    v = [freq_kernel(filter, ν) for ν in bohr_freqs]
    α = Matrix{eltype(v)}(undef, K, K)
    @inbounds for q in 1:K, p in 1:K
        α[p, q] = v[p] * conj(v[q])
    end
    return α
end

"""
    dll_kossakowski_bohr(filter, hamiltonian::HamHam) -> (alpha, bohr_freqs)

Construct the DLL Kossakowski matrix on the Hamiltonian's unique Bohr grid.

# Returns
`(alpha, bohr_freqs)` with matching matrix and sorted frequency labels.
"""
function dll_kossakowski_bohr(
    filter::AbstractFilter,
    hamiltonian::HamHam{T},
) where {T<:AbstractFloat}
    bohr_freqs = sort!(collect(keys(hamiltonian.bohr_dict)))
    α = dll_kossakowski_bohr(filter, bohr_freqs)
    return α, bohr_freqs
end

# Explicit finite-grid time representation and an algebraic hybrid reference.
# Refinement differences are measurements, not global generator error bounds.
function _prepared_dll_coherent(jumps,ham,p::PreparedFilterTransform,labels,step;
    loss=nothing)
    T=eltype(ham.eigvals); c=p.controls.coherent
    dt=c.time_step===nothing ? T(step) : T(c.time_step)
    Wt=c.time_window===nothing ? maximum(abs,labels) : T(c.time_window)
    Wnu=c.frequency_window
    if Wnu===nothing
        Wnu=p.controls.input==:frequency ? p.controls.window : nothing
        Wnu===nothing && p.base isa DLLGaussianFilter && (Wnu=T(16)/p.beta)
    end
    # Time-kernel input gives no frequency-tail estimate automatically.
    c.method==:hybrid || Wnu!==nothing || throw(ArgumentError(
        "A two-time kernel needs coherent.frequency_window for time-input filters."))
    R=loss===nothing ? zeros(Complex{T},length(ham.eigvals),length(ham.eigvals)) : loss
    if loss===nothing
        for jump in jumps
            A=dll_lindblad_op_time(jump,ham,labels,p,step)
            R .+= A'*A
        end
    end
    hybrid=_dll_coherent_from_loss(R,ham.eigvals,p.beta)
    bohr=_prepare_dll_bohr_filter(p,ham.eigvals)
    # Direct amplitude discrepancy exposes dissipative window/spacing error;
    # it is finite-spectrum evidence and does not bound coherent tails.
    implemented=fourier_sum(collect(labels),time_kernel.(Ref(p),labels).*step,
        vec(ham.bohr_freqs);backend=c.backend)
    exact=freq_kernel.(Ref(bohr),vec(ham.bohr_freqs))
    dissipative_error=norm(implemented-exact)/max(norm(exact),eps(T))
    function assemble(twindow,tstep,nwindow,ncount)
        half=ceil(Int,twindow/tstep)
        2half+1 <= c.max_points && ncount <= c.max_points || throw(ArgumentError(
            "Coherent refinement exceeds max_points=$(c.max_points); reduce controls or explicitly increase budget."))
        grid=collect(-half:half).*tstep
        return _dll_coherent_op_time_frequency_grid(jumps,ham,grid,p,p.beta,tstep;
            nu_min=-nwindow,nu_max=nwindow,nu_grid_size=ncount,backend=c.backend)
    end
    B=c.method==:hybrid ? hybrid : assemble(Wt,dt,Wnu,c.frequency_grid_size)
    scale=max(norm(B),norm(R),eps(T))
    differences=if c.method==:time && c.refine
        # Preserve frequency spacing when enlarging its window.
        (;frequency_window=norm(assemble(Wt,dt,2Wnu,2c.frequency_grid_size-1)-B)/scale,
          frequency_grid=norm(assemble(Wt,dt,Wnu,2c.frequency_grid_size-1)-B)/scale,
          time_window=norm(assemble(2Wt,dt,Wnu,c.frequency_grid_size)-B)/scale,
          time_grid=norm(assemble(Wt,dt/2,Wnu,c.frequency_grid_size)-B)/scale)
    else
        nothing
    end
    any(row->row[3]==:unresolved_quadrature,values(p.cache)) && throw(ArgumentError(
        "Coherent Fourier quadrature budget exhausted; tighten transform controls."))
    defect=norm(B-B')/scale
    hybrid_error=norm(B-hybrid)/scale
    unresolved=dissipative_error>c.tolerance || defect>c.tolerance || hybrid_error>c.tolerance ||
        (differences!==nothing && maximum(values(differences))>c.tolerance)
    if unresolved
        message="DLL Time accuracy unresolved: dissipative amplitude error=$dissipative_error, coherent refinements=$differences, implemented-loss comparison=$hybrid_error, Hermiticity defect=$defect. Tighten independent windows/grids."
        c.policy==:error ? throw(ArgumentError(message)) : (@warn message)
    end
    # A materially non-Hermitian B would cease to define the advertised GKLS
    # generator. Never repair it silently, even under the warning policy.
    defect <= max(T(1e-11),T(128)*eps(T)) || throw(ArgumentError(
        "Coherent quadrature is not Hermitian at roundoff; repair is not a KMS correction."))
    repair_size=c.repair ? norm((B+B')/2-B) : zero(T)
    c.repair && (B=(B+B')/2)
    evidence=(;method=c.method==:hybrid ? :time_jumps_implemented_loss_correction : :full_two_time_quadrature,
        status=unresolved ? :unresolved : c.refine || c.method==:hybrid ? :estimated : :not_checked,
        dissipative_amplitude_error=dissipative_error,coherent_refinements=differences,
        coherent_refinement_scope=:numerical_differences_not_bounds,
        coherent_time_step=dt,coherent_time_window=ceil(Wt/dt)*dt,
        requested_coherent_time_window=Wt,coherent_frequency_window=Wnu,
        coherent_frequency_grid_size=c.frequency_grid_size,
        frequency_tail=:not_certified,time_tail=:not_certified,
        transform_quadrature_estimate=maximum((row[2] for row in values(p.cache));init=zero(T)),
        transform_estimate_scope=:individual_kernel_evaluations_not_operator_bound,
        aliasing=:covered_only_by_grid_refinement,interpolation=:not_used,
        hybrid_difference=hybrid_error,
        hermiticity_defect_before_repair=defect,hermitian_repair_norm=repair_size,
        backend=c.backend,transform_precision=c.backend==:finufft ? Float64 : T,
        tolerance=c.tolerance,callbacks_in_matvec=false,
        kms_restoration=:not_claimed)
    return (;B,evidence)
end

function dll_coherent_op_time(jumps::AbstractVector{<:JumpOp},ham::HamHam,
    labels::AbstractVector{<:Real},p::PreparedFilterTransform,beta::Real,step::Real)
    _require_admissible_dll_filter(p;beta)
    return _prepared_dll_coherent(jumps,ham,p,labels,step).B
end
