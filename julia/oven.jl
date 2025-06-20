using LinearAlgebra
using Printf
using BenchmarkTools
using ProgressMeter

include("hamiltonian.jl")
include("qi_tools.jl")
include("structs.jl")
include("bohr_picture.jl")
include("energy_picture.jl")
include("time_picture.jl")
include("trotter_picture.jl")
include("misc_tools.jl")
include("jump_workers.jl")
include("oven_utensils.jl")

function run_liouvillian(jumps::Vector{JumpOp}, config::LiouvConfig;
    hamiltonian::Union{HamHam, Nothing}=nothing, trotter::Union{TrottTrott, Nothing}=nothing)

    liouv = construct_liouvillian(jumps, config; 
    hamiltonian=hamiltonian, trotter=trotter)

    liouv_eigvals, liouv_eigvecs = eigen(liouv) 
    steady_state_vec = liouv_eigvecs[:, end]
    steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
    steady_state_dm /= tr(steady_state_dm)

    result = HotSpectralResults(
        data = liouv,
        fixed_point = steady_state_dm,
        lambda_2 = liouv_eigvals[end-1],
        lambda_end = liouv_eigvals[1],
        hamiltonian = hamiltonian,
        trotter = trotter, 
        config = config
    )
    return result
end

function construct_liouvillian(jumps::Vector{JumpOp}, config::LiouvConfig;
    hamiltonian::Union{HamHam, Nothing}=nothing, trotter::Union{TrottTrott, Nothing}=nothing)

    print_press(config)
    
    picture_name = replace(string(typeof(config.picture)), "Picture" => "")
    progress_desc = "Constructing Liouvillian ($(picture_name))"

    ham_or_trott = if config.picture isa TrotterPicture
        trotter === nothing && error("A Trotter object must be provided for the TrotterPicture")
        trotter
    else # For Bohr, Energy, Time pictures
        hamiltonian === nothing && error("A Hamiltonian must be provided for the $(typeof(config.picture))")
        hamiltonian
    end

    labels = precompute_labels(config.picture, config)

    p = Progress(Int(length(jumps)), desc=progress_desc)
    total_liouv = @showprogress dt=0.01 progress_desc @distributed (+) for jump in jumps
        jump_contribution(config.picture, jump, ham_or_trott, config, labels...)
    end

    return total_liouv
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



