using Distributed
using LinearAlgebra
using Printf
using BenchmarkTools
using ProgressMeter
using Arpack

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

function run_liouvillian(jumps::Vector{JumpOp}, config::LiouvConfig, hamiltonian::HamHam; 
    trotter::Union{TrottTrott, Nothing}=nothing)

    validate_config!(config)
    print_press(config)

    liouv = construct_liouvillian(jumps, config, hamiltonian, trotter=trotter)

    # Full eigen
    # liouv_eigvals, liouv_eigvecs = eigen(liouv) 
    # steady_state_vec = liouv_eigvecs[:, end]
    # steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
    # steady_state_dm /= tr(steady_state_dm)

    # Arpack eigs
    eigvals_near_zero, eigvecs_near_zero = eigs(liouv, nev=2, which=:SM, tol=1e-14)
    ss_index = findmin(abs.(eigvals_near_zero))[2]
    gap_index = (ss_index == 1) ? 2 : 1
    steady_state_eigval = eigvals_near_zero[ss_index]
    lambda_2 = eigvals_near_zero[gap_index] # This is the spectral gap eigenvalue

    steady_state_vec = eigvecs_near_zero[:, ss_index]
    steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
    steady_state_dm ./= tr(steady_state_dm) # Normalize

    eigvals_end, _ = eigs(liouv, nev=1, which=:LM)
    lambda_end = eigvals_end[1]

    result = HotSpectralResults(
        data = liouv,
        fixed_point = steady_state_dm,
        lambda_2 = lambda_2,
        lambda_end = lambda_end,
        hamiltonian = hamiltonian,
        trotter = trotter, 
        config = config
    )
    return result
end

function construct_liouvillian(jumps::Vector{JumpOp}, config::LiouvConfig, hamiltonian::HamHam;
    trotter::Union{TrottTrott, Nothing}=nothing)
    
    picture_name = replace(string(typeof(config.picture)), "Picture" => "")
    println("Constructing Liouvillian ($(picture_name))")

    ham_or_trott = if config.picture isa TrotterPicture
        trotter === nothing && error("A Trotter object must be provided for the TrotterPicture")
        trotter
    else # For Bohr, Energy, Time pictures
        hamiltonian
    end

    labels = precompute_labels(config.picture, config)

    total_liouv = @distributed (+) for jump in jumps
        jump_contribution(config.picture, jump, ham_or_trott, config, labels...)
    end

    return total_liouv
end

function run_thermalization(jumps::Vector{JumpOp}, config::ThermalizeConfig, evolving_dm::Matrix{ComplexF64},
    hamiltonian::HamHam;
    trotter::Union{TrottTrott, Nothing}=nothing)

    validate_config!(config)
    print_press(config)
    picture_name = replace(string(typeof(config.picture)), "Picture" => "")
    println("Thermalizing ($(picture_name))")

    convergence_cutoff = 1e-6

    if config.picture isa TrotterPicture
        @assert trotter !== nothing "A Trotter object must be provided for the TrotterPicture"
        ham_or_trott = trotter
        gibbs = Hermitian(trotter.eigvecs' * hamiltonian.eigvecs * hamiltonian.gibbs * hamiltonian.eigvecs' * trotter.eigvecs)
    else
        ham_or_trott = hamiltonian
        gibbs = hamiltonian.gibbs
    end

    num_liouv_steps = Int(ceil(mixing_time / delta))
    energy_labels, time_labels = precompute_labels(config.picture, config)

    # Transition rate gamma
    @everywhere transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)
    # Functions for B
    f_minus, f_plus = if config.with_coherent && config.picture isa Union{TimePicture, TrotterPicture}
        _f_minus = compute_truncated_f(compute_f_minus, time_labels, config.beta)

        f_plus_calculator, f_plus_args = select_f_plus_calculator(config)
        _f_plus = compute_truncated_f(f_plus_calculator, time_labels, config.beta, f_plus_args...)
        
        (_f_minus, _f_plus)
    else
        (nothing, nothing)
    end

    # Broadcast to all workers
    @everywhere begin
        global const energy_labels = $energy_labels
        global const time_labels = $time_labels
        global const f_minus = $f_minus
        global const f_plus = $f_plus
    end

    distances_to_gibbs = [trace_distance_h(Hermitian(evolving_dm), gibbs)]
    for step in 1:num_liouv_steps
        update_dm = zeros(size(evolving_dm))
        update_dm = @distributed (+) for jump in jumps
            jump_contribution(config.picture, evolving_dm, jump, ham_or_trott, config)
        end

        evolving_dm .+= update_dm

        dist = trace_distance_h(Hermitian(evolving_dm), gibbs)
        push!(distances_to_gibbs, dist)
        if dist < convergence_cutoff
            num_liouv_steps = step  # Save the actual number of taken steps
            break
        end
    end
    
    time_steps = [0.0:delta:(num_liouv_steps * delta);]
    return HotAlgorithmResults(evolving_dm, distances_to_gibbs, time_steps, hamiltonian, trotter, config)
end

# function run_thermalization_slow(jumps::Vector{JumpOp}, config::ThermalizeConfig, evolving_dm::Matrix{ComplexF64},
#     hamiltonian::HamHam;
#     trotter::Union{TrottTrott, Nothing}=nothing)

#     validate_config!(config)
#     print_press(config)
#     picture_name = replace(string(typeof(config.picture)), "Picture" => "")
#     println("Thermalizing ($(picture_name))")

#     if config.picture isa TrotterPicture
#         @assert trotter !== nothing "A Trotter object must be provided for the TrotterPicture"
#         ham_or_trott = trotter
#         gibbs = Hermitian(trotter.eigvecs' * hamiltonian.gibbs * trotter.eigvecs)
#     else
#         ham_or_trott = hamiltonian
#         gibbs = hamiltonian.gibbs
#     end

#     num_liouv_steps = Int(ceil(mixing_time / delta))
#     energy_labels, time_labels = precompute_labels(config.picture, config)

#     # Transition rate gamma
#     @everywhere transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)
#     # Functions for B
#     f_minus, f_plus = if config.with_coherent && config.picture isa Union{TimePicture, TrotterPicture}
#         _f_minus = compute_truncated_f(compute_f_minus, time_labels, config.beta)

#         f_plus_calculator, f_plus_args = select_f_plus_calculator(config)
#         _f_plus = compute_truncated_f(f_plus_calculator, time_labels, config.beta, f_plus_args...)
        
#         (_f_minus, _f_plus)
#     else
#         (nothing, nothing)
#     end

#     # Broadcast to all workers
#     @everywhere begin
#         global const energy_labels = $energy_labels
#         global const time_labels = $time_labels
#         global const f_minus = $f_minus
#         global const f_plus = $f_plus
#     end

#     distances_to_gibbs = [trace_distance_h(Hermitian(evolving_dm), gibbs)]
#     for step in 1:num_liouv_steps
#         update_dm = zeros(size(evolving_dm))
#         update_dm = @distributed (+) for jump in jumps
#             jump_contribution_slow(config.picture, evolving_dm, jump, ham_or_trott, config)
#         end

#         evolving_dm .+= update_dm

#         dist = trace_distance_h(Hermitian(evolving_dm), gibbs)
#         push!(distances_to_gibbs, dist)
#         if dist < convergence_cutoff
#             num_liouv_steps = step  # Save the actual number of taken steps
#             break
#         end
#     end
    
#     time_steps = [0.0:delta:(num_liouv_steps * delta);]
#     return HotAlgorithmResults(evolving_dm, distances_to_gibbs, time_steps, hamiltonian, trotter, config)
# end


# function thermalize(jumps::Vector{JumpOp}, config::ThermalizeConfig, initial_dm::Matrix{ComplexF64};
#     hamiltonian::Union{HamHam, Nothing} = nothing,
#     trotter::Union{TrottTrott, Nothing} = nothing)

#     if (config.picture == TROTTER && trotter === nothing)
#         error("For TROTTER picture, a trotterization needs to be provided")
#     elseif (config.picture != TROTTER && hamiltonian === nothing)
#         error("For NON - TROTTER picture, a hamiltonian needs to be provided")
#     end

#     if !(is_config_valid(config))
#         error("Invalid parameter combination")
#     end

#     print_press(config)

#     if config.picture==BOHR
#         if config.with_linear_combination
#             return thermalize_bohr(jumps, hamiltonian, initial_dm, config.with_coherent, config.beta, config.a, config.b,
#             config.mixing_time, config.delta, config.unravel)
#         else
#             return thermalize_bohr_gauss(jumps, hamiltonian, initial_dm, config.with_coherent, config.beta, 
#                 config.mixing_time, config.delta, config.unravel)
#         end
#     end
#     if config.picture==ENERGY
#         energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
#         truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta,
#             config.a, config.b, config.with_linear_combination)

#         if config.with_linear_combination
#             return thermalize_energy(jumps, hamiltonian, initial_dm, truncated_energy_labels, 
#             config.with_coherent, config.beta, config.a, config.b, config.mixing_time, config.delta, config.unravel)
#         else
#             return thermalize_energy_gauss(jumps, hamiltonian, initial_dm, truncated_energy_labels, 
#             config.with_coherent, config.beta, config.mixing_time, config.delta, config.unravel)
#         end
#     end
#     if config.picture==TIME
#         energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
#         truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta,
#         config.a, config.b, config.with_linear_combination)
#         time_labels = energy_labels .* (config.t0 / config.w0)

#         if config.with_linear_combination
#             return thermalize_time(jumps, hamiltonian, initial_dm, time_labels, truncated_energy_labels, 
#                 config.with_coherent, config.beta, config.a, config.b, config.mixing_time, config.delta, config.unravel)
#         else
#             return thermalize_time_gauss(jumps, hamiltonian, initial_dm, time_labels, truncated_energy_labels, 
#             config.with_coherent, config.beta, config.mixing_time, config.delta, config.unravel)
#         end
#     end
#     if config.picture == TROTTER
#         energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
#         truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta,
#         config.a, config.b, config.with_linear_combination)
#         time_labels = energy_labels .* (config.t0 / config.w0)

#         if config.with_linear_combination
#             return thermalize_trotter(jumps, trotter, initial_dm, time_labels, truncated_energy_labels,
#             config.with_coherent, config.beta, config.a, config.b, config.mixing_time, config.delta, config.unravel)
#         else
#             return thermalize_trotter_gauss(jumps, trotter, initial_dm, time_labels, truncated_energy_labels,
#             config.with_coherent, config.beta, config.mixing_time, config.delta, config.unravel)
#         end
#     end
# end



