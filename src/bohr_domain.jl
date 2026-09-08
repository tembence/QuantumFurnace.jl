"""
    B_bohr(hamiltonian, jumps, config) -> Matrix

Construct the KMS coherent correction in the Hamiltonian eigenbasis.

# Arguments
- `hamiltonian`: Spectral data and grouped Bohr-frequency indices.
- `jumps`: Jump operators in the Hamiltonian eigenbasis.
- `config`: KMS rate parameters.

# Returns
The coherent-correction matrix `B`.
"""
function B_bohr(hamiltonian::HamHam{T}, jumps::AbstractVector{<:JumpOp}, config::Config{<:Any, <:Any, KMS}) where {T<:AbstractFloat}

    dim = size(hamiltonian.data, 1)
    CT = Complex{T}
    unique_freqs = collect(keys(hamiltonian.bohr_dict))

    f = _pick_f(config)  # Picks rates for B in Bohr domain

    n_jumps = length(jumps)
    n_freqs = length(unique_freqs)
    bohr_freqs = hamiltonian.bohr_freqs
    # Concrete views keep the inner loop type-stable.
    in_ebs = [(jump.in_eigenbasis::Matrix{CT}) for jump in jumps]

    if Threads.nthreads() > 1 && n_freqs >= OMEGA_THREAD_THRESHOLD
        return _B_bohr_threaded(hamiltonian, in_ebs, f, unique_freqs, bohr_freqs, dim, n_jumps, CT)
    end

    B = zeros(CT, dim, dim)
    f_row = Vector{CT}(undef, dim)

    for nu_2 in unique_freqs
        indices = hamiltonian.bohr_dict[nu_2]
        last_i = 0
        @inbounds for idx in indices
            i = idx[1]; j = idx[2]
            if i != last_i
                for col in 1:dim
                    f_row[col] = f(bohr_freqs[i, col], nu_2)
                end
                last_i = i
            end
            for jump_idx in 1:n_jumps
                in_eb = in_ebs[jump_idx]
                val = conj(in_eb[i, j])
                # Math: $B_(j k) += conj(A_(i j)) f(Delta_(i k), nu_2) A_(i k)$.
                @inbounds for col in 1:dim
                    B[j, col] += val * f_row[col] * in_eb[i, col]
                end
            end
        end
    end
    return B
end

# Each task owns its partial matrix and row scratch; reduction occurs after
# all frequency chunks complete.
function _B_bohr_threaded(
    hamiltonian::HamHam{T},
    in_ebs::Vector{Matrix{CT}},
    f,
    unique_freqs::Vector,
    bohr_freqs::AbstractMatrix{<:Real},
    dim::Int,
    n_jumps::Int,
    ::Type{CT},
) where {T<:AbstractFloat, CT}
    n_freqs = length(unique_freqs)
    nt = min(Threads.nthreads(), n_freqs)
    chunks = _partition_range(1:n_freqs, nt)
    n_chunks = length(chunks)

    B_partials = [zeros(CT, dim, dim) for _ in 1:n_chunks]

    @sync for (cidx, chunk) in enumerate(chunks)
        Threads.@spawn _B_bohr_chunk!(
            B_partials[cidx], hamiltonian, in_ebs, f, unique_freqs,
            bohr_freqs, dim, n_jumps, chunk, CT)
    end

    B = zeros(CT, dim, dim)
    @inbounds for cidx in 1:n_chunks
        B .+= B_partials[cidx]
    end
    return B
end

function _B_bohr_chunk!(
    B_partial::Matrix{CT},
    hamiltonian::HamHam,
    in_ebs::Vector{Matrix{CT}},
    f,
    unique_freqs::Vector,
    bohr_freqs::AbstractMatrix{<:Real},
    dim::Int,
    n_jumps::Int,
    chunk::UnitRange{Int},
    ::Type{CT},
) where {CT}
    f_row = Vector{CT}(undef, dim)
    @inbounds for k in chunk
        nu_2 = unique_freqs[k]
        indices = hamiltonian.bohr_dict[nu_2]
        last_i = 0
        for idx in indices
            i = idx[1]; j = idx[2]
            if i != last_i
                for col in 1:dim
                    f_row[col] = f(bohr_freqs[i, col], nu_2)
                end
                last_i = i
            end
            for jump_idx in 1:n_jumps
                in_eb = in_ebs[jump_idx]
                val = conj(in_eb[i, j])
                @inbounds for col in 1:dim
                    B_partial[j, col] += val * f_row[col] * in_eb[i, col]
                end
            end
        end
    end
    return nothing
end

function _pick_f(config::Config{<:Any, <:Any, KMS})
    if config.transition_weight !== nothing
        alpha = _pick_alpha(config)
        return (u,v) -> tanh(-config.beta*(u-v)/4)*alpha(u,v)/(2im)
    end

    beta = config.beta
    sigma = config.sigma
    if config.with_linear_combination
        a = config.a
        s = config.s
        return (nu_1, nu_2) -> create_f(nu_1, nu_2, beta, sigma, a, s)
    else
        gaussian_parameters = config.gaussian_parameters
        return (nu_1, nu_2) -> create_f_gauss(nu_1, nu_2, beta, sigma, gaussian_parameters)
    end
end

"""
    create_f(nu_1, nu_2, beta, sigma, a, s) -> Complex

Return the smooth-Metropolis KMS coherent kernel for two Bohr frequencies.

# Arguments
- `nu_1`, `nu_2`: Bohr frequencies.
- `beta`, `sigma`, `a`, `s`: Algorithm-side rate parameters.

# Returns
The complex coherent-kernel coefficient.
"""
function create_f(nu_1::Real, nu_2::Real, beta::Real, sigma::Real, a::Real, s::Real)
    alpha = create_alpha(nu_1, nu_2, beta, sigma, a, s)
    # Math: $f(nu_1, nu_2) = tanh(-beta (nu_1 - nu_2) / 4) alpha(nu_1, nu_2) / (2 i)$.
    return tanh(-beta * (nu_1 - nu_2) / 4) * alpha / (2im)
end

"""
    create_f_gauss(nu_1, nu_2, beta, sigma, gaussian_parameters) -> Number

Return the Gaussian KMS coherent kernel at two Bohr frequencies.
"""
function create_f_gauss(nu_1::Real, nu_2::Real, beta::Real, sigma::Real,
    gaussian_parameters::Union{Tuple{<:Real, <:Real}, Tuple{Nothing, Nothing}})
    alpha_nu1_nu2 = create_alpha_gauss(nu_1, nu_2, sigma, gaussian_parameters)
    return tanh(-beta * (nu_1 - nu_2) / 4) * alpha_nu1_nu2 / (2im)
end

_pick_alpha(config::Config{<:Any, <:Any, KMS}) = _pick_alpha_kms(config)
_pick_alpha(config::Config{<:Any, <:Any, GNS}) = _pick_alpha_gns(config)

# 2-arg forms: compute alpha directly via dispatch (zero allocation on hot path)
function _pick_alpha(config::Config{<:Any, <:Any, KMS}, nu_1::Real, nu_2::Real)
    config.transition_weight === nothing || return transition_alpha(config.transition_weight,nu_1,nu_2)
    if config.with_linear_combination
        return create_alpha(nu_1, nu_2, config.beta, config.sigma, config.a, config.s)
    else
        return create_alpha_gauss(nu_1, nu_2, config.sigma, config.gaussian_parameters)
    end
end

function _pick_alpha(config::Config{<:Any, <:Any, GNS}, nu_1::Real, nu_2::Real)
    if config.with_linear_combination
        return create_alpha_gns(nu_1, nu_2, config.beta, config.sigma, config.a, config.s)
    else
        return create_alpha_gauss(nu_1, nu_2, config.sigma, config.gaussian_parameters)
    end
end

function _pick_alpha_kms(config::Config{<:Any, <:Any, KMS})
    if config.transition_weight !== nothing
        T=typeof(config.beta)
        return BohrAlphaKernel{KMS,T}(false,config.beta,config.sigma,zero(T),zero(T),
            (zero(T),zero(T)),config.transition_weight)
    end
    if config.with_linear_combination
        return BohrAlphaKernel{KMS, typeof(config.beta)}(
            true, config.beta, config.sigma,
            config.a::typeof(config.beta), config.s::typeof(config.beta),
            (zero(config.beta), zero(config.beta)))
    else
        gaussian_parameters = config.gaussian_parameters::Tuple{typeof(config.beta), typeof(config.beta)}
        return BohrAlphaKernel{KMS, typeof(config.beta)}(
            false, config.beta, config.sigma,
            zero(config.beta), zero(config.beta), gaussian_parameters)
    end
end

@inline function (alpha::BohrAlphaKernel{KMS})(nu_1::Real, nu_2::Real)
    alpha.transition_weight === nothing || return transition_alpha(alpha.transition_weight,nu_1,nu_2)
    if alpha.with_linear_combination
        return create_alpha(nu_1, nu_2, alpha.beta, alpha.sigma, alpha.a, alpha.s)
    else
        return create_alpha_gauss(nu_1, nu_2, alpha.sigma, alpha.gaussian_parameters)
    end
end

"""
    create_alpha(nu_1, nu_2, beta, sigma, a, s) -> Real

Return the smooth-Metropolis KMS Kossakowski coefficient.

# Arguments
- `nu_1`, `nu_2`: Bohr frequencies.
- `beta`, `sigma`, `a`, `s`: Algorithm-side rate parameters.

# Returns
The real coefficient coupling the two Bohr components.
"""
function create_alpha(nu_1::Real, nu_2::Real, beta::Real, sigma::Real, a::Real, s::Real)

    sqrtA = sqrt(beta * (4 * a + 1) / 4)
    sqrtB = sqrt(beta / 16) * abs(nu_1 + nu_2)
    C = beta * (nu_1 + nu_2) / 4
    prefactor = exp(a * beta^2 * sigma^2 / 2) / 2
    u_min = sqrt(beta * sigma^2 * (1 + s) / 2)
    # Math: $z_+ = sqrt(A) u_min + sqrt(B) / u_min$ and
    # $z_- = sqrt(A) u_min - sqrt(B) / u_min$.
    z_plus = sqrtA * u_min + sqrtB / u_min
    z_minus = sqrtA * u_min - sqrtB / u_min

    alpha_nu_1 = (prefactor * exp(-C) * exp(-(nu_1 - nu_2)^2 / (8 * sigma^2)) * exp(- 2 * sqrtA * sqrtB) *
                    (erfc(z_minus) + exp(4 * sqrtA * sqrtB) * erfc(z_plus)))

    return alpha_nu_1
end

"""
    default_smooth_s(beta, sigma) -> Real

Return the diagnostic smoothing parameter with fixed width `sigma sqrt(s) = 0.05`.

# Arguments
- `beta`: Algorithm-side inverse temperature; retained for call-site clarity.
- `sigma`: Gaussian energy width.

# Returns
`(0.05 / sigma)^2`. Production sweeps use fixed `s = 0.25` unless explicitly
testing constant-absolute-width smoothing.
"""
const SMOOTH_S_REF_WIDTH = 0.05
default_smooth_s(beta::Real, sigma::Real) = (SMOOTH_S_REF_WIDTH / sigma)^2

function _pick_alpha_gns(config::Config{<:Any, <:Any, GNS})
    if config.with_linear_combination
        return BohrAlphaKernel{GNS, typeof(config.beta)}(
            true, config.beta, config.sigma,
            config.a::typeof(config.beta), config.s::typeof(config.beta),
            (zero(config.beta), zero(config.beta)))
    else
        gaussian_parameters = config.gaussian_parameters::Tuple{typeof(config.beta), typeof(config.beta)}
        return BohrAlphaKernel{GNS, typeof(config.beta)}(
            false, config.beta, config.sigma,
            zero(config.beta), zero(config.beta), gaussian_parameters)
    end
end

@inline function (alpha::BohrAlphaKernel{GNS})(nu_1::Real, nu_2::Real)
    if alpha.with_linear_combination
        return create_alpha_gns(
            nu_1, nu_2, alpha.beta, alpha.sigma, alpha.a, alpha.s)
    else
        return create_alpha_gauss(nu_1, nu_2, alpha.sigma, alpha.gaussian_parameters)
    end
end

"""
    create_alpha_gns(nu_1, nu_2, beta, sigma, a, s) -> Real

Return the GNS Kossakowski coefficient for two Bohr frequencies.

The GNS construction shifts the sum to
`\$abs(nu_1 + nu_2 + beta sigma^2 / 2)\$` and has no coherent correction.
"""
function create_alpha_gns(nu_1::Real, nu_2::Real, beta::Real, sigma::Real, a::Real, s::Real)
    sqrtA = sqrt(beta * (4 * a + 1) / 4)
    sqrtB = sqrt(beta / 16) * abs(nu_1 + nu_2 + beta * sigma^2 / 2)
    C = beta * (nu_1 + nu_2) / 4
    prefactor = exp(a * beta^2 * sigma^2 / 2) / 2
    u_min = sqrt(beta * sigma^2 * (1 + s) / 2)
    z_plus = sqrtA * u_min + sqrtB / u_min
    z_minus = sqrtA * u_min - sqrtB / u_min

    alpha_nu_1 = (prefactor * exp(-C) * exp(-(nu_1 - nu_2)^2 / (8 * sigma^2)) * exp(- 2 * sqrtA * sqrtB) *
                    (erfc(z_minus) + exp(4 * sqrtA * sqrtB) * erfc(z_plus)))

    return alpha_nu_1
end

"""
    create_alpha_gauss(nu_1, nu_2, sigma, gaussian_parameters) -> Real

Return the Gaussian Kossakowski coefficient for two Bohr frequencies.

# Arguments
- `nu_1`, `nu_2`: Bohr frequencies.
- `sigma`: Energy-filter width.
- `gaussian_parameters`: Transition centre and width `(w_gamma, sigma_gamma)`.

# Returns
The real coefficient coupling the two Bohr components.
"""
function create_alpha_gauss(
    nu_1::Real,
    nu_2::Real,
    sigma::Real,
    gaussian_parameters::Union{Tuple{<:Real, <:Real}, Tuple{Nothing, Nothing}})

    (w_gamma, sigma_gamma) = gaussian_parameters
    combined_sigma = sigma^2 + sigma_gamma^2
    prefactor = sigma_gamma / sqrt(combined_sigma)
    alpha_fn(nu_1) = prefactor * (exp(-(nu_1 + nu_2 + 2 * w_gamma)^2 / (8 * combined_sigma))
                                    * exp(-(nu_1 - nu_2)^2 / (8 * sigma^2)))
    return alpha_fn(nu_1)
end

"""
    create_bohr_dict(bohr_freqs) -> Dict

Group matrix indices by exact Bohr frequency, including all zero-frequency
diagonal indices.
"""
function create_bohr_dict(bohr_freqs::Matrix{T}) where {T<:AbstractFloat}
    bohr_dict = DefaultDict{T, Vector{CartesianIndex{2}}}(() -> CartesianIndex{2}[])
    dim = size(bohr_freqs, 1)
    bohr_dict[zero(T)] = CartesianIndex{2}.(1:dim, 1:dim)
    for j in 1:dim
        for i in 1:(j - 1)
            nu = bohr_freqs[i, j]
            # Dict distinguishes +0.0 and -0.0; degeneracies belong to the
            # same physical zero-frequency sector as the diagonal entries.
            forward = iszero(nu) ? zero(T) : nu
            reverse = iszero(nu) ? zero(T) : -nu
            push!(bohr_dict[forward], CartesianIndex{2}(i, j))
            push!(bohr_dict[reverse], CartesianIndex{2}(j, i))
        end
    end
    return bohr_dict
end

"""
    check_alpha_skew_symmetry(alpha, nu_1, nu_2, beta) -> nothing

Assert the KMS skew-symmetry relation for a Kossakowski kernel.
"""
function check_alpha_skew_symmetry(alpha::Function, nu_1::Real, nu_2::Real, beta::Real)
    @assert norm(alpha(nu_1, nu_2) - alpha(-nu_2, -nu_1) * exp(-beta * (nu_1 + nu_2) / 2)) < 1e-14
end
