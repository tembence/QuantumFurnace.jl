using LinearAlgebra
using Printf
using BenchmarkTools

include("hamiltonian.jl")
include("qi_tools.jl")
include("structs.jl")
include("bohr_picture.jl")
include("energy_picture.jl")
include("time_picture.jl")
include("trotter_picture.jl")
include("misc_tools.jl")

function construct_liouvillian(jumps::Vector{JumpOp}, config::LiouvConfig;
    hamiltonian::Union{HamHam, Nothing} = nothing,
    trotter::Union{TrottTrott, Nothing} = nothing)

    if (config.picture == TROTTER && trotter === nothing)
        error("For TROTTER picture, a trotterization needs to be provided")
    elseif (config.picture != TROTTER && hamiltonian === nothing)
        error("For NON - TROTTER picture, a hamiltonian needs to be provided")
    end

    if !(is_config_valid(config))
        error("Invalid parameter combination")
    end

    print_press(config)

    if config.picture==BOHR
        if config.with_linear_combination
            return construct_liouvillian_bohr(jumps, hamiltonian, config.with_coherent, config.beta, config.a, config.b)
        else
            return construct_liouvillian_bohr_gauss(jumps, hamiltonian, config.with_coherent, config.beta)
        end
    end
    if config.picture==ENERGY
        energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
        truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta,
            config.a, config.b, config.with_linear_combination)

        if config.with_linear_combination
            return construct_liouvillian_energy(jumps, hamiltonian, truncated_energy_labels, 
                config.with_coherent, config.beta, config.a, config.b)
        else
            return construct_liouvillian_energy_gauss(jumps, hamiltonian, truncated_energy_labels,
                config.with_coherent, config.beta)
        end
    end
    if config.picture==TIME
        energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
        truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta,
        config.a, config.b, config.with_linear_combination)
        time_labels = energy_labels .* (config.t0 / config.w0)

        if config.with_linear_combination
            return construct_liouvillian_time(jumps, hamiltonian, time_labels, truncated_energy_labels, 
                config.with_coherent, config.beta, config.a, config.b)
        else
            return construct_liouvillian_time_gauss(jumps, hamiltonian, time_labels, truncated_energy_labels, 
                config.with_coherent, config.beta)
        end
    end
    if config.picture == TROTTER
        energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
        truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta,
        config.a, config.b, config.with_linear_combination)
        time_labels = energy_labels .* (config.t0 / config.w0)

        if config.with_linear_combination
            return construct_liouvillian_trotter(jumps, trotter, time_labels, truncated_energy_labels,
            config.with_coherent, config.beta, config.a, config.b)
        else
            return construct_liouvillian_trotter_gauss(jumps, trotter, time_labels, truncated_energy_labels,
                config.with_coherent, config.beta)
        end
    end
end

function thermalize(jumps::Vector{JumpOp}, config::ThermalizeConfig, initial_dm::Matrix{ComplexF64};
    hamiltonian::Union{HamHam, Nothing} = nothing,
    trotter::Union{TrottTrott, Nothing} = nothing)

    if (config.picture == TROTTER && trotter === nothing)
        error("For TROTTER picture, a trotterization needs to be provided")
    elseif (config.picture != TROTTER && hamiltonian === nothing)
        error("For NON - TROTTER picture, a hamiltonian needs to be provided")
    end

    if !(is_config_valid(config))
        error("Invalid parameter combination")
    end

    print_press(config)

    if config.picture==BOHR
        if config.with_linear_combination
            return thermalize_bohr(jumps, hamiltonian, initial_dm, config.with_coherent, config.beta, config.a, config.b,
            config.mixing_time, config.delta, config.unravel)
        else
            return thermalize_bohr_gauss(jumps, hamiltonian, initial_dm, config.with_coherent, config.beta, 
                config.mixing_time, config.delta, config.unravel)
        end
    end
    if config.picture==ENERGY
        energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
        truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta,
            config.a, config.b, config.with_linear_combination)

        if config.with_linear_combination
            return thermalize_energy(jumps, hamiltonian, initial_dm, truncated_energy_labels, 
            config.with_coherent, config.beta, config.a, config.b, config.mixing_time, config.delta, config.unravel)
        else
            return thermalize_energy_gauss(jumps, hamiltonian, initial_dm, truncated_energy_labels, 
            config.with_coherent, config.beta, config.mixing_time, config.delta, config.unravel)
        end
    end
    if config.picture==TIME
        energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
        truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta,
        config.a, config.b, config.with_linear_combination)
        time_labels = energy_labels .* (config.t0 / config.w0)

        if config.with_linear_combination
            return thermalize_time(jumps, hamiltonian, initial_dm, time_labels, truncated_energy_labels, 
                config.with_coherent, config.beta, config.a, config.b, config.mixing_time, config.delta, config.unravel)
        else
            return thermalize_time_gauss(jumps, hamiltonian, initial_dm, time_labels, truncated_energy_labels, 
            config.with_coherent, config.beta, config.mixing_time, config.delta, config.unravel)
        end
    end
    if config.picture == TROTTER
        energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
        truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta,
        config.a, config.b, config.with_linear_combination)
        time_labels = energy_labels .* (config.t0 / config.w0)

        if config.with_linear_combination
            return thermalize_trotter(jumps, trotter, initial_dm, time_labels, truncated_energy_labels,
            config.with_coherent, config.beta, config.a, config.b, config.mixing_time, config.delta, config.unravel)
        else
            return thermalize_trotter_gauss(jumps, trotter, initial_dm, time_labels, truncated_energy_labels,
            config.with_coherent, config.beta, config.mixing_time, config.delta, config.unravel)
        end
    end

end
