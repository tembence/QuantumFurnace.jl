using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using BenchmarkTools
using Roots
using QuadGK

include("hamiltonian.jl")
include("ofts.jl")
include("qi_tools.jl")
include("coherent.jl")

#* Linear Combinations
function construct_liouvillian_trotter(jumps::Vector{JumpOp}, trotter::TrottTrott, time_labels::Vector{Float64},
    energy_labels::Vector{Float64}, config::LiouvConfig)

    dim = size(trotter.eigvecs, 1)
    w0 = energy_labels[2] - energy_labels[1]
    oft_time_labels = truncate_time_labels_for_oft(time_labels, config.beta)

    transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)

    if config.with_coherent
        f_minus = compute_truncated_f(compute_f_minus, time_labels, config.beta)
        if config.with_linear_combination  
            if config.a != 0.0  # Improved Metro / Glauber
                f_plus = compute_truncated_f(compute_f_plus_eh, time_labels, config.beta, config.a, config.b)
            else  # Metro
                f_plus = compute_truncated_f(compute_f_plus_metro, time_labels, config.beta, config.eta)
            end
        else  # Gaussian
            f_plus = compute_truncated_f(compute_f_plus, time_labels, config.beta)
        end
    end

    total_liouv_coherent_part = zeros(ComplexF64, dim^2, dim^2)
    total_liouv_diss_part = zeros(ComplexF64, dim^2, dim^2)
    p = Progress(Int(length(jumps) * length(energy_labels)), desc="Liouvillian (TROTTER)...")
    for jump in jumps
        if config.with_coherent  # There is no energy formulation of the coherent term, only Bohr and time.
            coherent_term = coherent_term_trotter(jump, trotter, f_minus, f_plus)
            coherent_term = trotter.trafo_from_eigen_to_trotter' * coherent_term * trotter.trafo_from_eigen_to_trotter
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = trotter_oft(jump, w, trotter, oft_time_labels, config.beta) # t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
            jump_oft = trotter.trafo_from_eigen_to_trotter' * jump_oft * trotter.trafo_from_eigen_to_trotter

            total_liouv_diss_part .+= transition(w) * vectorize_liouvillian_diss(jump_oft)
            next!(p)
        end
    end
    
    prefactor = w0 * trotter.t0^2 * (sqrt(2 / pi) / config.beta) / (2 * pi)
    return total_liouv_coherent_part .+ prefactor * total_liouv_diss_part  # L in energy basis
end

function thermalize_trotter(jumps::Vector{JumpOp}, trotter::TrottTrott, evolving_dm::Matrix{ComplexF64}, 
    time_labels::Vector{Float64}, energy_labels::Vector{Float64}, with_coherent::Bool, beta::Float64, a::Float64, b::Float64, 
    mixing_time::Float64, delta::Float64, unravel::Bool)
    """In Trotter basis"""

    dim = size(hamiltonian.data, 1)
    w0 = energy_labels[2] - energy_labels[1]
    oft_time_labels = truncate_time_labels_for_oft(time_labels, beta)

    # Working in Trotter basis
    gibbs = Hermitian(trotter.trafo_from_eigen_to_trotter * gibbs_state_in_eigen(hamiltonian, beta)
                                    * trotter.trafo_from_eigen_to_trotter')
    evolving_dm = trotter.trafo_from_eigen_to_trotter * evolving_dm * trotter.trafo_from_eigen_to_trotter'

    distances_to_gibbs = [trace_distance_h(Hermitian(evolving_dm), gibbs)]

    oft_prefactor = (sqrt(2 / pi) / beta) / (2 * pi)

    transition = pick_transition(beta, a, b)

    if with_coherent
        f_minus = compute_truncated_f_minus(time_labels, beta)

        if a != 0.0
            f_plus = compute_truncated_f_plus_eh(time_labels, beta, a, b)
        else
            f_plus = compute_truncated_f_plus_metro(time_labels, beta, eta)
        end
    end

    num_liouv_steps = Int(ceil(mixing_time / delta))
    if unravel
        @printf("Unraveling => actual_num_liouv_steps = num_jumps * num_liouv_steps = %i\n", length(jumps) * num_liouv_steps)
        @printf("Mixing time thus also becomes longer: %f\n", mixing_time * length(jumps))
        time_steps = [0.0:delta:(length(jumps) * num_liouv_steps * delta);]
    else
        time_steps = [0.0:delta:(num_liouv_steps * delta);]
    end

    p = Progress(Int(num_liouv_steps * length(jumps) * length(energy_labels)), desc="Thermalize (TROTTER)...")
    for step in 1:num_liouv_steps
        step_coherent = zeros(ComplexF64, dim, dim)
        step_dissipative = zeros(ComplexF64, dim, dim)

        for jump in jumps
            jump_coherent = zeros(ComplexF64, dim, dim)
            jump_dissipative = zeros(ComplexF64, dim, dim)

            # Coherent part
            if with_coherent
                coherent_term = coherent_term_trotter(jump, trotter, f_minus, f_plus)
                jump_coherent .+= - 1im * (coherent_term * evolving_dm - evolving_dm * coherent_term)
            end

            # Dissipative part
            for w in energy_labels
                jump_oft = trotter_oft(jump, w, trotter, oft_time_labels, beta) # t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
                jump_dag_jump = jump_oft' * jump_oft
                jump_dissipative .+= transition(w) * (
                    jump_oft * evolving_dm * jump_oft' - 0.5 * (jump_dag_jump * evolving_dm + evolving_dm * jump_dag_jump))
                next!(p)
            end

            if !(unravel)  # Accumulate
                step_coherent .+= jump_coherent
                step_dissipative .+= jump_dissipative
            else # Apply immediately
                evolving_dm .+= delta * (jump_coherent + w0 * trotter.t0^2 * oft_prefactor * jump_dissipative)
                dist = trace_distance_h(Hermitian(evolving_dm), gibbs)
                push!(distances_to_gibbs, dist)
            end
        end
        
        if !(unravel)
            evolving_dm .+= delta * (step_coherent + w0 * t0^2 * oft_prefactor * step_dissipative)
            dist = trace_distance_h(Hermitian(evolving_dm), gibbs)
            push!(distances_to_gibbs, dist)
        end
    end
    return HotAlgorithmResults(evolving_dm, distances_to_gibbs, time_steps)
end

#* GAUSS
function construct_liouvillian_trotter_gauss(jumps::Vector{JumpOp}, trotter::TrottTrott, time_labels::Vector{Float64},
    energy_labels::Vector{Float64}, config::LiouvConfig)

    dim = size(trotter.eigvecs, 1)
    w0 = energy_labels[2] - energy_labels[1]
    oft_time_labels = truncate_time_labels_for_oft(time_labels, config.beta)

    transition_gauss(w) = exp(-config.beta^2 * (w + 1 / config.beta)^2 /2)

    if config.with_coherent  # Steup for coherent term in time domain
        f_minus = compute_truncated_f_minus(time_labels, config.beta)
        f_plus = compute_truncated_f_plus(time_labels, config.beta)
    end

    total_liouv_coherent_part = zeros(ComplexF64, dim^2, dim^2)
    total_liouv_diss_part = zeros(ComplexF64, dim^2, dim^2)
    p = Progress(Int(length(jumps) * length(energy_labels)), desc="Liouvillian (TROTTER GAUSS)...")
    for jump in jumps
        if config.with_coherent  # There is no energy formulation of the coherent term, only Bohr and time.
            coherent_term = coherent_term_trotter(jump, trotter, f_minus, f_plus)
            coherent_term = trotter.trafo_from_eigen_to_trotter' * coherent_term * trotter.trafo_from_eigen_to_trotter
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = trotter_oft(jump, w, trotter, oft_time_labels, config.beta) # t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
            jump_oft = trotter.trafo_from_eigen_to_trotter' * jump_oft * trotter.trafo_from_eigen_to_trotter
            # jump_oft_w = oft(jump, w, hamiltonian, beta) * sqrt(beta / sqrt(2 * pi))
            # @assert (norm(jump_oft_w - jump_oft * t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))) < 1e-7)

            total_liouv_diss_part .+= transition_gauss(w) * vectorize_liouvillian_diss(jump_oft)
            next!(p)
        end
    end
    
    prefactor = w0 * t0^2 * (sqrt(2 / pi)/config.beta) / (2 * pi)
    return total_liouv_coherent_part .+ prefactor * total_liouv_diss_part  # L in energy basis
end

function thermalize_trotter_gauss(jumps::Vector{JumpOp}, trotter::TrottTrott, evolving_dm::Matrix{ComplexF64}, 
    time_labels::Vector{Float64}, energy_labels::Vector{Float64}, with_coherent::Bool, beta::Float64, 
    mixing_time::Float64, delta::Float64, unravel::Bool)
    """In Trotter basis"""

    dim = size(hamiltonian.data, 1)
    w0 = energy_labels[2] - energy_labels[1]
    oft_time_labels = truncate_time_labels_for_oft(time_labels, beta)

    # Working in Trotter basis
    gibbs = Hermitian(trotter.trafo_from_eigen_to_trotter * gibbs_state_in_eigen(hamiltonian, beta)
                                    * trotter.trafo_from_eigen_to_trotter')
    evolving_dm = trotter.trafo_from_eigen_to_trotter * evolving_dm * trotter.trafo_from_eigen_to_trotter'

    distances_to_gibbs = [trace_distance_h(Hermitian(evolving_dm), gibbs)]

    oft_prefactor = (sqrt(2 / pi) / beta) / (2 * pi)

    transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)

    if with_coherent  # Steup for coherent term in time domain
        f_minus = compute_truncated_f_minus(time_labels, beta)
        f_plus = compute_truncated_f_plus(time_labels, beta)
    end

    num_liouv_steps = Int(ceil(mixing_time / delta))
    if unravel
        @printf("Unraveling => actual_num_liouv_steps = num_jumps * num_liouv_steps = %i\n", length(jumps) * num_liouv_steps)
        @printf("Mixing time thus also becomes longer: %f\n", mixing_time * length(jumps))
        time_steps = [0.0:delta:(length(jumps) * num_liouv_steps * delta);]
    else
        time_steps = [0.0:delta:(num_liouv_steps * delta);]
    end

    p = Progress(Int(num_liouv_steps * length(jumps) * length(energy_labels)), desc="Thermalize (TROTTER GAUSS)...")
    for step in 1:num_liouv_steps
        step_coherent = zeros(ComplexF64, dim, dim)
        step_dissipative = zeros(ComplexF64, dim, dim)

        for jump in jumps
            jump_coherent = zeros(ComplexF64, dim, dim)
            jump_dissipative = zeros(ComplexF64, dim, dim)

            # Coherent part
            if with_coherent
                coherent_term = coherent_term_trotter(jump, trotter, f_minus, f_plus)
                jump_coherent .+= - 1im * (coherent_term * evolving_dm - evolving_dm * coherent_term)
            end

            # Dissipative part
            for w in energy_labels
                jump_oft = trotter_oft(jump, w, trotter, oft_time_labels, beta) # t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
                jump_dag_jump = jump_oft' * jump_oft
                jump_dissipative .+= transition_gauss(w) * (
                    jump_oft * evolving_dm * jump_oft' - 0.5 * (jump_dag_jump * evolving_dm + evolving_dm * jump_dag_jump))
                next!(p)
            end

            if !(unravel)  # Accumulate
                step_coherent .+= jump_coherent
                step_dissipative .+= jump_dissipative
            else # Apply immediately
                evolving_dm .+= delta * (jump_coherent + w0 * trotter.t0^2 * oft_prefactor * jump_dissipative)
                dist = trace_distance_h(Hermitian(evolving_dm), gibbs)
                push!(distances_to_gibbs, dist)
            end
        end
        
        if !(unravel)
            evolving_dm .+= delta * (step_coherent + w0 * t0^2 * oft_prefactor * step_dissipative)
            dist = trace_distance_h(Hermitian(evolving_dm), gibbs)
            push!(distances_to_gibbs, dist)
        end
    end
    return HotAlgorithmResults(evolving_dm, distances_to_gibbs, time_steps)
end