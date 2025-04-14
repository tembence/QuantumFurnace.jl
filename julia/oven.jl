using LinearAlgebra
using Printf
using BenchmarkTools

include("hamiltonian.jl")
include("qi_tools.jl")
include("structs.jl")
include("bohr_picture.jl")
include("energy_picture.jl")
include("time_picture.jl")
include("ofts.jl")
include("coherent.jl")
include("misc_tools.jl")

function construct_liouvillian(jumps::Vector{JumpOp}, config::LiouvConfig;
    hamiltonian::Union{HamHam, Nothing} = nothing,
    trotter::Union{TrottTrott, Nothing} = nothing)

    if (hamiltonian === nothing && trotter === nothing) || (hamiltonian !== nothing && trotter !== nothing)
        error("Either a `hamiltonian` or a `trotter` scheme must be provided")
    end

    if !(is_config_valid(config))
        error("Invalid parameter combination")
    end

    if picture==BOHR
        if config.with_linear_combination
            return construct_liouvillian_bohr(jumps, hamiltonian, config.with_coherent, config.beta, config.a, config.b)
        else
            return construct_liouvillian_bohr_gauss(jumps, hamiltonian, config.with_coherent, config.beta)
        end
    end
    if picture==ENERGY
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
    if picture==TIME
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
    # if picture==TROTTER  #TODO:
    #     energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
    #     truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta,
            # config.a, config.b, config.with_linear_combination)
    #     time_labels = energy_labels .* (config.t0 / config.w0)
        # if config.with_linear_combination
    #     return construct_liouvillian_trotter()
        # else
        # return construct_liouvillian_trotter_gauss()
    # end
end
