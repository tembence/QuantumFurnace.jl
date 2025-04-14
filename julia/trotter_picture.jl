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

function construct_liouvillian_gauss_trotter(jumps::Vector{JumpOp}, trotter::TrottTrott, time_labels::Vector{Float64},
    energy_labels::Vector{Float64}, with_coherent::Bool, beta::Float64)

    dim = size(trotter.eigvecs, 1)
    w0 = energy_labels[2] - energy_labels[1]
    transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)

    if with_coherent  # Steup for coherent term in time domain
        b1 = compute_truncated_b1(time_labels)
        b2 = compute_truncated_b2(time_labels)
    end

    total_liouv_coherent_part = zeros(ComplexF64, dim^2, dim^2)
    total_liouv_diss_part = zeros(ComplexF64, dim^2, dim^2)
    @showprogress desc="Liouvillian (Trotter)..." for jump in jumps
        if with_coherent  # There is no energy formulation of the coherent term, only Bohr and time.
            coherent_term = coherent_term_trotter(jump, trotter, b1, b2, beta)
            coherent_term = trotter.trafo_from_eigen_to_trotter' * coherent_term * trotter.trafo_from_eigen_to_trotter  #!
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = trotter_oft(jump, w, trotter, time_labels, beta) # t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
            jump_oft = trotter.trafo_from_eigen_to_trotter' * jump_oft * trotter.trafo_from_eigen_to_trotter #!
            # jump_oft_actually = oft(jump, w, hamiltonian, beta) * sqrt(beta / sqrt(2 * pi))
            # @printf("Jump oft norm: %s\n", norm(jump_oft - jump_oft_actually))
            total_liouv_diss_part .+= transition_gauss(w) * vectorize_liouvillian_diss(jump_oft)
        end
    end
    
    time_oft_norm_squared = (sqrt(2 / pi)/beta) / (2 * pi)  # ft and fourier norms
    return total_liouv_coherent_part .+ w0 * trotter.t0^2 * time_oft_norm_squared * total_liouv_diss_part
end

function thermalize_gauss_trotter(jumps::Vector{JumpOp}, trotter::TrottTrott, initial_dm::Matrix{ComplexF64},
    energy_labels::Vector{Float64}, time_labels::Vector{Float64}, with_coherent::Bool, 
    delta::Float64, mixing_time::Float64, beta::Float64)

    w0 = energy_labels[2] - energy_labels[1]
    dim = size(hamiltonian.data, 1)
    num_liouv_steps = Int(round(mixing_time / delta, digits=0))
    gibbs_in_trotter = Hermitian(trotter.trafo_from_eigen_to_trotter * gibbs_state_in_eigen(hamiltonian, beta)
                                    * trotter.trafo_from_eigen_to_trotter')
    transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)

    if with_coherent  # Steup for coherent term in time domain
        b1 = compute_truncated_b1(time_labels)
        b2 = compute_truncated_b2(time_labels)
    end

    @printf("Initial dm is Hermitian: %s\n", norm(initial_dm - initial_dm'))
    distances_to_gibbs = [trace_distance_h(Hermitian(initial_dm), gibbs_in_trotter)]
    time_steps = [0.0:delta:(mixing_time);]
    evolved_dm = copy(initial_dm)
    # This implementation applies all jumps at once for one Liouvillian step.
    @showprogress dt=1 desc="Thermalize (Time)..." for step in 1:num_liouv_steps
        coherent_dm_part = zeros(ComplexF64, dim, dim)
        dissipative_dm_part = zeros(ComplexF64, dim, dim)

        for jump in jumps
            # Coherent part
            if with_coherent
                coherent_term = coherent_term_trotter(jump, trotter, b1, b2, beta)
                coherent_dm_part .+= - 1im * (coherent_term * evolved_dm - evolved_dm * coherent_term)
            end

            # Dissipative part
            for w in energy_labels
                jump_oft = trotter_oft(jump, w, trotter, time_labels, beta)
                jump_dag_jump = jump_oft' * jump_oft
                dissipative_dm_part .+= transition_gauss(w) * (jump_oft * evolved_dm * jump_oft' 
                                                                - 0.5 * (jump_dag_jump * evolved_dm 
                                                                        + evolved_dm * jump_dag_jump))
            end
        end
        prefactor = w0 * trotter.t0^2 * (sqrt(2 / pi)/beta) / (2 * pi)  # time ints t0^2, energy int w0, OFT time norm^2, Fourier
        evolved_dm .+= delta * (coherent_dm_part + prefactor * dissipative_dm_part)
        @printf("In Trotter thermalization the new dm is Hermitian: %s\n", norm(evolved_dm - evolved_dm'))
        dist = trace_distance_h(Hermitian(evolved_dm), gibbs)  #FIXME: I think the Trotter doesn't keep evolved dm Hermitian?
        push!(distances_to_gibbs, dist)
    end
    evolved_dm_in_eigen = trotter.trafo_from_trotter_to_eigen' * evolved_dm * trotter.trafo_from_trotter_to_eigen
    return HotAlgorithmResults(evolved_dm_in_eigen, distances_to_gibbs, time_steps)
end