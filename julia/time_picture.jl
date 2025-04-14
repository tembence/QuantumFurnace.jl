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

#* Linear combinations -----------------------------------------------------------------------------------------------------------------------
function construct_liouvillian_time(jumps::Vector{JumpOp}, hamiltonian::HamHam, time_labels::Vector{Float64},
    energy_labels::Vector{Float64}, with_coherent::Bool, beta::Float64, a::Float64, b::Float64)

    dim = size(hamiltonian.data, 1)
    w0 = energy_labels[2] - energy_labels[1]
    t0 = time_labels[2] - time_labels[1]

    transition = pick_transition(beta, a, b)

    if with_coherent
        f_minus = compute_truncated_f_minus(time_labels, beta)

        if a != 0.0
            f_plus = compute_truncated_f_plus_eh(time_labels, beta, a, b)
        else
            f_plus = compute_truncated_f_plus_metro(time_labels, beta, eta)
        end
    end

    total_liouv_coherent_part = zeros(ComplexF64, dim^2, dim^2)
    total_liouv_diss_part = zeros(ComplexF64, dim^2, dim^2)
    p = Progress(Int(length(jumps) * length(energy_labels)), desc="Liouvillian (TIME)...")
    for jump in jumps
        if with_coherent 
            coherent_term = coherent_term_time(jump, hamiltonian, f_minus, f_plus, t0)  
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = time_oft(w, jump, hamiltonian, time_labels, beta) # subnorm = t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
            total_liouv_diss_part .+= transition(w) * vectorize_liouvillian_diss(jump_oft)
            next!(p)
        end
    end
    prefactor = w0 * t0^2 * (sqrt(2 / pi) / beta) / (2 * pi)  # time ints t0^2, energy int w0, OFT time norm^2, Fourier
    return total_liouv_coherent_part .+ prefactor * total_liouv_diss_part
end

#* GAUSS --------------------------------------------------------------------------------------------------------------------
function construct_liouvillian_time_gauss(jumps::Vector{JumpOp}, hamiltonian::HamHam, time_labels::Vector{Float64},
    energy_labels::Vector{Float64}, with_coherent::Bool, beta::Float64)

    dim = size(hamiltonian.data, 1)
    w0 = energy_labels[2] - energy_labels[1]
    t0 = time_labels[2] - time_labels[1]
    transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)

    if with_coherent  # Steup for coherent term in time domain
        f_minus = compute_truncated_f_minus(time_labels, beta)
        f_plus = compute_truncated_f_plus(time_labels, beta)
    end

    total_liouv_coherent_part = zeros(ComplexF64, dim^2, dim^2)
    total_liouv_diss_part = zeros(ComplexF64, dim^2, dim^2)
    p = Progress(Int(length(jumps) * length(energy_labels)), desc="Liouvillian (TIME GAUSS)...")
    for jump in jumps
        if with_coherent
            coherent_term = coherent_term_time(jump, hamiltonian, f_minus, f_plus, t0)
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = time_oft(w, jump, hamiltonian, time_labels, beta) # t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
            total_liouv_diss_part .+= transition_gauss(w) * vectorize_liouvillian_diss(jump_oft)
            next!(p)
        end
    end
    prefactor = w0 * t0^2 * (sqrt(2 / pi)/beta) / (2 * pi)  # time ints t0^2, energy int w0, OFT time norm^2, Fourier
    return total_liouv_coherent_part .+ prefactor * total_liouv_diss_part
end

function thermalize_gauss_time(jumps::Vector{JumpOp}, hamiltonian::HamHam, initial_dm::Matrix{ComplexF64},
    energy_labels::Vector{Float64}, time_labels::Vector{Float64}, with_coherent::Bool, 
    delta::Float64, mixing_time::Float64, beta::Float64)

    w0 = energy_labels[2] - energy_labels[1]
    t0 = time_labels[2] - time_labels[1]
    dim = size(hamiltonian.data, 1)
    num_liouv_steps = Int(round(mixing_time / delta, digits=0))
    gibbs = Hermitian(gibbs_state_in_eigen(hamiltonian, beta))
    transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)

    if with_coherent  # Steup for coherent term in time domain
        b1 = compute_truncated_b1(time_labels)
        b2 = compute_truncated_b2(time_labels)
    end

    distances_to_gibbs = [trace_distance_h(Hermitian(initial_dm), gibbs)]
    time_steps = [0.0:delta:(mixing_time);]
    evolved_dm = copy(initial_dm)
    # This implementation applies all jumps at once for one Liouvillian step.
    @showprogress dt=1 desc="Thermalize (Time)..." for step in 1:num_liouv_steps
        coherent_dm_part = zeros(ComplexF64, dim, dim)
        dissipative_dm_part = zeros(ComplexF64, dim, dim)

        for jump in jumps
            # Coherent part
            if with_coherent
                coherent_term = coherent_term_time_b(jump, hamiltonian, b1, b2, t0, beta)
                coherent_dm_part .+= - 1im * (coherent_term * evolved_dm - evolved_dm * coherent_term)
            end

            # Dissipative part
            for w in energy_labels
                jump_oft = time_oft(jump, w, hamiltonian, time_labels, beta)
                jump_dag_jump = jump_oft' * jump_oft
                dissipative_dm_part .+= transition_gauss(w) * (jump_oft * evolved_dm * jump_oft' 
                                                                - 0.5 * (jump_dag_jump * evolved_dm 
                                                                        + evolved_dm * jump_dag_jump))
            end
        end
        prefactor = w0 * t0^2 * (sqrt(2 / pi)/beta) / (2 * pi)  # time ints t0^2, energy int w0, OFT time norm^2, Fourier
        evolved_dm .+= delta * (coherent_dm_part + prefactor * dissipative_dm_part)
        dist = trace_distance_h(Hermitian(evolved_dm), gibbs)
        push!(distances_to_gibbs, dist)
    end
    return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
end

#! Gauss with b's somehow gave slightly closer results to the Bohr picture.
# function construct_liouvillian_time_gauss(jumps::Vector{JumpOp}, hamiltonian::HamHam, time_labels::Vector{Float64},
#     energy_labels::Vector{Float64}, with_coherent::Bool, beta::Float64)

#     dim = size(hamiltonian.data, 1)
#     w0 = energy_labels[2] - energy_labels[1]
#     t0 = time_labels[2] - time_labels[1]
#     transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)

#     if with_coherent  # Steup for coherent term in time domain
#         b1 = compute_truncated_b1(time_labels)
#         b2 = compute_truncated_b2(time_labels)
#     end

#     total_liouv_coherent_part = zeros(ComplexF64, dim^2, dim^2)
#     total_liouv_diss_part = zeros(ComplexF64, dim^2, dim^2)
#     p = Progress(Int(length(jumps) * length(energy_labels)), desc="Liouvillian (TIME GAUSS)...")
#     for jump in jumps
#         if with_coherent
#             coherent_term = coherent_term_time_b(jump, hamiltonian, b1, b2, t0, beta)
#             total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
#         end

#         for w in energy_labels
#             jump_oft = time_oft(w, jump, hamiltonian, time_labels, beta) # t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
#             total_liouv_diss_part .+= transition_gauss(w) * vectorize_liouvillian_diss(jump_oft)
#             next!(p)
#         end
#     end
#     prefactor = w0 * t0^2 * (sqrt(2 / pi)/beta) / (2 * pi)  # time ints t0^2, energy int w0, OFT time norm^2, Fourier
#     return total_liouv_coherent_part .+ prefactor * total_liouv_diss_part
# end