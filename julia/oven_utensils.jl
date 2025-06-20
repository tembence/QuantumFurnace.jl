
function precompute_labels(::BohrPicture, config)
    return () # Bohr needs no labels
end

function precompute_labels(::EnergyPicture, config)
    energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
    truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta, config.a, config.b, 
    config.with_linear_combination)
    return (truncated_energy_labels,)  # Energy labels
end

function precompute_labels(::Union{TimePicture, TrotterPicture}, config)
    energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
    truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta, config.a, config.b, 
    config.with_linear_combination)
    time_labels = energy_labels .* (config.t0 / config.w0)
    return (truncated_energy_labels, time_labels) # Energy and time labels
end  
