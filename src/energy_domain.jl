"""
    pick_transition(config[, w]) -> Function or Real

Return the configured KMS or GNS transition rate, or evaluate it at `w`.

# Arguments
- `config`: Construction and rate parameters.
- `w`: Optional Bohr frequency at which to evaluate the rate.

# Returns
A scalar callable when `w` is omitted, otherwise the scalar transition rate.
"""
pick_transition(config::Config{<:Any, <:Any, KMS}) = _pick_transition_kms(config)
pick_transition(config::Config{<:Any, <:Any, GNS}) = _pick_transition_gns(config)

# 2-arg forms: compute transition value directly via dispatch (zero allocation on hot path)
function pick_transition(config::Config{<:Any, <:Any, KMS}, w::Real)
    if config.transition_weight !== nothing
        # Keep the open Config field from erasing scalar inference in the
        # existing frequency loops, including configurations with no typed rate.
        T = promote_type(typeof(config.beta),typeof(w))
        return T(transition_value(config.transition_weight,w))::T
    end
    if !(config.with_linear_combination)
        return exp(-(w + config.gaussian_parameters[1])^2 / (2 * config.gaussian_parameters[2]^2))
    end
    sqrtA = sqrt(config.beta / 4) * sqrt(4 * config.a + 1)
    sqrtB = sqrt(config.beta / 4) * abs(w + config.beta * config.sigma^2 / 2)
    if config.s == 0 && config.a == 0
        return exp(-config.beta * max(w + config.beta * config.sigma^2 / 2, 0.0))
    else
        # Math: at $a = 0$, smooth Metro is
        # $gamma_M^0 (erfc(z_-) + exp(beta abs(tilde(omega))) erfc(z_+)) / 2$.
        u_min = sqrt(config.beta * config.sigma^2 * config.s / 2)
        transition_b0 = exp(-2 * sqrtA * sqrtB - config.beta * w / 2 - config.beta^2 * config.sigma^2 / 4)
        return transition_b0 * (erfc(sqrtA * u_min - sqrtB / u_min) + exp(4 * sqrtA * sqrtB) * erfc(sqrtA * u_min + sqrtB / u_min)) / 2
    end
end

function pick_transition(config::Config{<:Any, <:Any, GNS}, w::Real)
    if !(config.with_linear_combination)
        w_gamma = config.gaussian_parameters[1]
        sigma_gamma = config.gaussian_parameters[2]
        return exp(-(w + w_gamma)^2 / (2 * sigma_gamma^2))
    end
    sqrtA = sqrt(config.beta / 4) * sqrt(4 * config.a + 1)
    sqrtB = sqrt(config.beta / 4) * abs(w)
    if config.s == 0 && config.a == 0
        return exp(-config.beta * max(w, 0.0))
    else
        # The GNS form uses the unshifted frequency `w`.
        u_min = sqrt(config.beta * config.sigma^2 * config.s / 2)
        transition_b0 = exp(-2 * sqrtA * sqrtB - config.beta * w / 2)
        return transition_b0 * (erfc(sqrtA * u_min - sqrtB / u_min) + exp(4 * sqrtA * sqrtB) * erfc(sqrtA * u_min + sqrtB / u_min)) / 2
    end
end


function _pick_transition_kms(config::Config{<:Any, <:Any, KMS})
    config.transition_weight === nothing || return w -> transition_value(config.transition_weight,w)

    if !(config.with_linear_combination)
        return w -> begin
            return exp(-(w + config.gaussian_parameters[1])^2 /(2 * config.gaussian_parameters[2]^2))
        end
    end

    sqrtA = sqrt(config.beta / 4) * sqrt(4 * config.a + 1)
    if (config.s == 0 && config.a == 0)
        return w -> exp(-config.beta * max(w + config.beta * config.sigma^2 / 2, 0.0))
    else
        return w -> begin
            sqrtB = sqrt(config.beta / 4) * abs(w + config.beta * config.sigma^2 / 2)
            u_min = sqrt(config.beta * config.sigma^2 * config.s / 2)
            transition_b0 = exp((- 2 * sqrtA * sqrtB - config.beta * w / 2 - config.beta^2 * config.sigma^2 / 4))
            return (transition_b0 * (erfc(sqrtA * u_min - sqrtB / u_min)
                + exp(4 * sqrtA * sqrtB) * erfc(sqrtA * u_min + sqrtB / u_min)) / 2)
        end
    end
end

"""
    _pick_transition_gns(config) -> Function

Return the unshifted GNS detailed-balance transition function.

# Arguments
- `config`: GNS construction parameters.

# Returns
A scalar function satisfying
`\$tilde(gamma)(omega) = tilde(gamma)(-omega) exp(-beta omega)\$`.
"""
function _pick_transition_gns(config::Config{<:Any, <:Any, GNS})

    if !(config.with_linear_combination)
        return w -> begin
            w_gamma = config.gaussian_parameters[1]
            sigma_gamma = config.gaussian_parameters[2]
            return exp(-(w + w_gamma)^2 / (2 * sigma_gamma^2))
        end
    end

    sqrtA = sqrt(config.beta / 4) * sqrt(4 * config.a + 1)
    if (config.s == 0 && config.a == 0)
        return w -> exp(-config.beta * max(w, 0.0))
    else
        return w -> begin
            sqrtB = sqrt(config.beta / 4) * abs(w)
            u_min = sqrt(config.beta * config.sigma^2 * config.s / 2)
            transition_b0 = exp((-2 * sqrtA * sqrtB - config.beta * w / 2))
            return (transition_b0 * (erfc(sqrtA * u_min - sqrtB / u_min)
                + exp(4 * sqrtA * sqrtB) * erfc(sqrtA * u_min + sqrtB / u_min)) / 2)
        end
    end
end


"""
    pick_gamma_sup(config::Config) -> Real

Return the fixed divisor used to normalise the configured transition rate.

Legacy Gaussian and Metropolis families retain divisor one. Typed Gaussian
mixtures use one for `normalization=:none`, or their explicit fixed upper
bound for `:bound`; the latter need not equal the exact supremum. No sampled
grid maximum enters this divisor.
"""
pick_gamma_sup(config::Config{<:Any, <:Any, KMS}) =
    config.transition_weight === nothing ? 1.0 : _transition_divisor(config.transition_weight)
pick_gamma_sup(config::Config{<:Any, <:Any, GNS}) = 1.0


function _create_energy_labels(num_energy_bits::Integer, w0::Real)
    N = 2^(num_energy_bits)
    N_labels = [-Int(N/2):1:Int(N/2)-1;]
    energy_labels = w0 * N_labels
    return energy_labels
end

function _truncate_energy_labels(
    energy_labels::AbstractVector{<:Real},
    config::Config;
    cutoff::Real=1e-12
    )

    # Custom mixtures have no legacy Metropolis/Gaussian grid-tail heuristic.
    # Retain the entire explicitly requested outer grid.
    config.transition_weight isa GaussianMixtureTransition && return energy_labels
    transition = pick_transition(config)
    gaussfilter(w, nu) = exp(- (w - nu)^2 / (4 * config.sigma^2)) * sqrt(1 / (config.sigma * sqrt(2 * pi)))
    integrand_lb(w, nu1, nu2) = transition(w) * gaussfilter(w, nu1) * gaussfilter(w, nu2)
    integrand_ub(w, nu1, nu2) = transition(w) * gaussfilter(w, nu1) * gaussfilter(w, nu2)

    candidate_nus = filter(w -> -0.45 <= w <= (-config.beta * config.sigma^2 / 2), [-0.45:0.05:0.0;])

    start_index = length(energy_labels) + 1
    for (nu1_candidate, nu2_candidate) in Iterators.product(candidate_nus, candidate_nus)
        found_index = findfirst(w -> abs(integrand_lb(w, nu1_candidate, nu2_candidate)) >= cutoff, energy_labels)
        if found_index !== nothing
            start_index = min(start_index, found_index)
        end
    end
    
    if start_index  > length(energy_labels)
        @warn "Lower bound cutoff not found for energies, using default range."
        return energy_labels[abs.(energy_labels) .<= 2.0]
    end

    candidate_nus = Iterators.reverse(filter(w -> (-config.beta * config.sigma^2 / 2) <= w <= 0.45, [-0.1:0.05:0.45;]))
    end_index = 0
    for (nu1_candidate, nu2_candidate) in Iterators.product(candidate_nus, candidate_nus)
        found_index = findlast(w -> abs(integrand_ub(w, nu1_candidate, nu2_candidate)) >= cutoff, energy_labels)
        if found_index !== nothing
            end_index = max(end_index, found_index)
        end
    end

    if end_index === 0
        @warn "Upper bound cutoff not found for energies, using default range."
        return energy_labels[abs.(energy_labels) .<= 2.0]
    end

    if start_index == 1 || end_index == length(energy_labels)
        @warn "No truncation was done, might want more estimating energy range."
    end

    # Keep the retained grid symmetric about zero.
    sym_limit = max(abs(energy_labels[start_index]), abs(energy_labels[end_index]))
    return energy_labels[abs.(energy_labels) .<= sym_limit]
end
