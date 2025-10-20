function precompute_labels(::BohrPicture, config::Union{LiouvConfig, ThermalizeConfig})
    return ()# Bohr needs no labels
end

function precompute_labels(::EnergyPicture, config::Union{LiouvConfig, ThermalizeConfig})
    energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
    truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta, config.a, config.b, 
    config.with_linear_combination)
    return (truncated_energy_labels,)  # Energy labels
end

function precompute_labels(::Union{TimePicture, TrotterPicture}, config::Union{LiouvConfig, ThermalizeConfig})
    energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
    truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta, config.a, config.b, 
    config.with_linear_combination)
    time_labels = energy_labels .* (config.t0 / config.w0)
    return (truncated_energy_labels, time_labels) # Energy and time labels
end  

function precompute_data(::BohrPicture, config::Union{LiouvConfig, ThermalizeConfig})
    alpha = pick_alpha(config)
    return (
        alpha = alpha
    )
end

function precompute_data(::EnergyPicture, config::Union{LiouvConfig, ThermalizeConfig})
    energy_labels, = precompute_labels(config.picture, config)
    w0 = abs(energy_labels[2] - energy_labels[1])
    transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)
    return (
        w0 = w0,
        transition = transition,
        energy_labels = energy_labels
    )
end

function precompute_data(::Union{TimePicture, TrotterPicture}, config::Union{LiouvConfig, ThermalizeConfig})
    energy_labels, time_labels = precompute_labels(config.picture, config)
    oft_time_labels = truncate_time_labels_for_oft(time_labels, config.beta)
    w0 = abs(energy_labels[2] - energy_labels[1])
    t0 = abs(time_labels[2] - time_labels[1])
    transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)

    # f_minus, f_plus = if config.with_coherent
    #     _f_minus = compute_truncated_func(compute_f_minus, time_labels, config.beta)
    #     chosen_f_plus, f_plus_args = select_f_plus_calculator(config)
    #     _f_plus = compute_truncated_func(chosen_f_plus, time_labels, config.beta, f_plus_args...)
    #     (_f_minus, _f_plus)
    b_minus, b_plus = if config.with_coherent
        _b_minus = compute_truncated_func(compute_b_minus, time_labels, config.beta)
        chosen_b_plus, b_plus_args = select_b_plus_calculator(config)
        _b_plus = compute_truncated_func(chosen_b_plus, time_labels, config.beta, b_plus_args...)
        (_b_minus, _b_plus)
    else
        (nothing, nothing)
    end
    return (
        w0 = w0,
        t0 = t0,
        transition = transition,
        energy_labels = energy_labels,
        oft_time_labels = oft_time_labels,
        b_minus = b_minus,
        b_plus = b_plus,
    )
end

function select_f_plus_calculator(config::Union{LiouvConfig, ThermalizeConfig})
    if !config.with_linear_combination
        # Gaussian
        return (compute_f_plus, ())
    else
        if config.a != 0.0
            # Improved Metro / Glauber
            return (compute_f_plus_eh, (config.a, config.b))
        else
            # Metro
            return (compute_f_plus_metro, (config.eta,))
        end
    end
end

function select_b_plus_calculator(config::Union{LiouvConfig, ThermalizeConfig})
    if !config.with_linear_combination
        # Gaussian
        return (compute_b_plus, ())
    else
        if config.a != 0.0
            # Improved Metro / Glauber
            return (compute_b_plus_eh, (config.a, config.b))
        else
            # Metro
            return (compute_b_plus_metro, (config.eta,))
        end
    end
end