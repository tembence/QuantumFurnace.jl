"""
    pick_transition(config[, w]) -> Function or Real

Return the configured KMS or GNS transition rate, or evaluate it at `w`.

# Arguments
- `config`: Construction and rate parameters.
- `w`: Optional Bohr frequency at which to evaluate the rate.

# Returns
A scalar callable when `w` is omitted, otherwise the scalar transition rate.
"""
function pick_transition(config::Config{<:Any, <:Any, <:Union{KMS, GNS}})
    if config.construction isa KMS && config.transition_weight !== nothing
        return w -> transition_value(config.transition_weight, w)
    end
    if config.with_linear_combination && iszero(config.a) && iszero(config.s)
        beta = config.beta
        shift = config.construction isa KMS ? beta * config.sigma^2 / 2 : zero(beta)
        return w -> _metropolis_kink(beta, w, shift)
    end
    return w -> pick_transition(config, w)
end

@inline function pick_transition(config::Config{<:Any, <:Any, KMS}, w::Real)
    if config.transition_weight !== nothing
        # The Config boundary permits several rate types; scalar loops retain
        # their concrete promoted return type.
        T = promote_type(typeof(config.beta), typeof(w))
        return T(transition_value(config.transition_weight, w))::T
    end
    return _parameter_transition(config, w, config.beta * config.sigma^2 / 2)
end

@inline pick_transition(config::Config{<:Any, <:Any, GNS}, w::Real) =
    _parameter_transition(config, w, zero(config.beta))

@inline function _parameter_transition(config, w, shift)
    if !config.with_linear_combination
        centre, width = config.gaussian_parameters
        return exp(-(w + centre)^2 / (2 * width^2))
    end
    return _metropolis_transition(config.beta, config.sigma, config.a, config.s, w, shift)
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
    integrand(w, nu1, nu2) = transition(w) * gaussfilter(w, nu1) * gaussfilter(w, nu2)

    candidate_nus = filter(w -> -0.45 <= w <= (-config.beta * config.sigma^2 / 2), [-0.45:0.05:0.0;])

    start_index = length(energy_labels) + 1
    for (nu1_candidate, nu2_candidate) in Iterators.product(candidate_nus, candidate_nus)
        found_index = findfirst(w -> abs(integrand(w, nu1_candidate, nu2_candidate)) >= cutoff, energy_labels)
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
        found_index = findlast(w -> abs(integrand(w, nu1_candidate, nu2_candidate)) >= cutoff, energy_labels)
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
