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
include("qi_tools.jl")
include("bohr_picture.jl")
include("ofts.jl")

#* Linear Combinations -----------------------------------------------------------------------------------------------------------------------
function construct_liouvillian_energy(jumps::Vector{JumpOp}, hamiltonian::HamHam, energy_labels::Vector{Float64}, 
    with_coherent::Bool, beta::Float64, a::Float64, b::Float64)

    dim = size(hamiltonian.data, 1)
    w0 = energy_labels[2] - energy_labels[1]
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)

    transition = pick_transition(beta, a, b)

    total_liouv_coherent_part = zeros(ComplexF64, dim^2, dim^2)
    total_liouv_diss_part = zeros(ComplexF64, dim^2, dim^2)

    @showprogress desc="Liouvillian (Energy)..." for jump in jumps
        if with_coherent  # There is no energy formulation of the coherent term, only Bohr and time.
            coherent_term = coherent_bohr(hamiltonian, bohr_dict, jump, beta, a, b)
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = oft(jump, w, hamiltonian, beta)
            total_liouv_diss_part .+= transition(w) * vectorize_liouvillian_diss(jump_oft)
        end
    end
    oft_norm_squared = beta / sqrt(2 * pi)
    return total_liouv_coherent_part .+ w0 * oft_norm_squared * total_liouv_diss_part
end

function thermalize_energy(jumps::Vector{JumpOp}, hamiltonian::HamHam, evolving_dm::Matrix{ComplexF64},
    energy_labels::Vector{Float64}, with_coherent::Bool, beta::Float64, a::Float64, b::Float64, 
    mixing_time::Float64, delta::Float64, unravel::Bool)

    w0 = energy_labels[2] - energy_labels[1]
    dim = size(hamiltonian.data, 1)
    gibbs = Hermitian(gibbs_state_in_eigen(hamiltonian, beta))
    oft_prefactor = beta / sqrt(2 * pi)  # discrete sum w0 + OFT normalization^2 + Fourier factor

    transition = pick_transition(beta, a, b)

    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)

    num_liouv_steps = Int(ceil(mixing_time / delta))
    if unravel
        @printf("Unraveling => actual_num_liouv_steps = num_jumps * num_liouv_steps = %i\n", length(jumps) * num_liouv_steps)
        @printf("Mixing time thus also becomes longer: %f\n", mixing_time * length(jumps))
        time_steps = [0.0:delta:(length(jumps) * num_liouv_steps * delta);]
    else
        time_steps = [0.0:delta:(num_liouv_steps * delta);]
    end

    distances_to_gibbs = [trace_distance_h(Hermitian(evolving_dm), gibbs)]

    p = Progress(Int(num_liouv_steps * length(jumps) * length(energy_labels)), desc="Thermalize (ENERGY)...")
    for step in 1:num_liouv_steps
        step_coherent = zeros(ComplexF64, dim, dim)
        step_dissipative = zeros(ComplexF64, dim, dim)

        for jump in jumps
            jump_coherent = zeros(ComplexF64, dim, dim)
            jump_dissipative = zeros(ComplexF64, dim, dim)

            # Coherent part
            if with_coherent
                coherent_term = coherent_bohr(hamiltonian, bohr_dict, jump, beta, a, b)
                jump_coherent .+= - 1im * (coherent_term * evolving_dm - evolving_dm * coherent_term)
            end

            # Dissipative part
            for w in energy_labels
                jump_oft = oft(jump, w, hamiltonian, beta)
                jump_dag_jump = jump_oft' * jump_oft
                jump_dissipative .+= transition(w) * (
                    jump_oft * evolving_dm * jump_oft' - 0.5 * (jump_dag_jump * evolving_dm + evolving_dm * jump_dag_jump))
                next!(p)
            end

            if !(unravel)  # Accumulate
                step_coherent .+= jump_coherent
                step_dissipative .+= jump_dissipative
            else # Apply immediately
                evolving_dm .+= delta * (jump_coherent + w0 * oft_prefactor * jump_dissipative)
                dist = trace_distance_h(Hermitian(evolving_dm), gibbs)
                push!(distances_to_gibbs, dist)
            end
        end
        
        if !(unravel)
            evolving_dm .+= delta * (step_coherent + w0 * oft_prefactor * step_dissipative)
            dist = trace_distance_h(Hermitian(evolving_dm), gibbs)
            push!(distances_to_gibbs, dist)
        end
    end
    return HotAlgorithmResults(evolving_dm, distances_to_gibbs, time_steps)
end

#* GAUSS --------------------------------------------------------------------------------------------------------------------
function construct_liouvillian_energy_gauss(jumps::Vector{JumpOp}, hamiltonian::HamHam, energy_labels::Vector{Float64}, 
    with_coherent::Bool, beta::Float64)

    dim = size(hamiltonian.data, 1)
    w0 = energy_labels[2] - energy_labels[1]
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
    transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)

    total_liouv_coherent_part = zeros(ComplexF64, dim^2, dim^2)
    total_liouv_diss_part = zeros(ComplexF64, dim^2, dim^2)
    p = Progress(Int(length(jumps) * length(energy_labels)), desc="Liouvillian (ENERGY GAUSS)...")
    for jump in jumps
        if with_coherent  # There is no energy formulation of the coherent term, only Bohr and time.
            coherent_term = coherent_bohr_gauss(hamiltonian, bohr_dict, jump, beta)
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = oft(jump, w, hamiltonian, beta)
            total_liouv_diss_part .+= transition_gauss(w) * vectorize_liouvillian_diss(jump_oft)
            next!(p)
        end
    end
    oft_norm_squared = beta / sqrt(2 * pi)
    return total_liouv_coherent_part .+ w0 * oft_norm_squared * total_liouv_diss_part
end

function thermalize_energy_gauss(jumps::Vector{JumpOp}, hamiltonian::HamHam, evolving_dm::Matrix{ComplexF64},
    energy_labels::Vector{Float64}, with_coherent::Bool, beta::Float64, mixing_time::Float64, delta::Float64, unravel::Bool)

    w0 = energy_labels[2] - energy_labels[1]
    dim = size(hamiltonian.data, 1)
    gibbs = Hermitian(gibbs_state_in_eigen(hamiltonian, beta))
    oft_prefactor = beta / sqrt(2 * pi)  # discrete sum w0 + OFT normalization^2 + Fourier factor
    transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)

    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)

    num_liouv_steps = Int(ceil(mixing_time / delta))
    if unravel
        @printf("Unraveling => actual_num_liouv_steps = num_jumps * num_liouv_steps = %i\n", length(jumps) * num_liouv_steps)
        @printf("Mixing time thus also becomes longer: %f\n", mixing_time * length(jumps))
        time_steps = [0.0:delta:(length(jumps) * num_liouv_steps * delta);]
    else
        time_steps = [0.0:delta:(num_liouv_steps * delta);]
    end

    distances_to_gibbs = [trace_distance_h(Hermitian(evolving_dm), gibbs)]

    p = Progress(Int(num_liouv_steps * length(jumps) * length(energy_labels)), desc="Thermalize (ENERGY GAUSS)...")
    for step in 1:num_liouv_steps
        step_coherent = zeros(ComplexF64, dim, dim)
        step_dissipative = zeros(ComplexF64, dim, dim)

        for jump in jumps
            jump_coherent = zeros(ComplexF64, dim, dim)
            jump_dissipative = zeros(ComplexF64, dim, dim)

            # Coherent part
            if with_coherent
                coherent_term = coherent_bohr_gauss(hamiltonian, bohr_dict, jump, beta)
                jump_coherent .+= - 1im * (coherent_term * evolving_dm - evolving_dm * coherent_term)
            end

            # Dissipative part
            for w in energy_labels
                jump_oft = oft(jump, w, hamiltonian, beta)
                jump_dag_jump = jump_oft' * jump_oft
                jump_dissipative .+= transition_gauss(w) * (
                    jump_oft * evolving_dm * jump_oft' - 0.5 * (jump_dag_jump * evolving_dm + evolving_dm * jump_dag_jump))
                next!(p)
            end

            if !(unravel)  # Accumulate
                step_coherent .+= jump_coherent
                step_dissipative .+= jump_dissipative
            else # Apply immediately
                evolving_dm .+= delta * (jump_coherent + w0 * oft_prefactor * jump_dissipative)
                dist = trace_distance_h(Hermitian(evolving_dm), gibbs)
                push!(distances_to_gibbs, dist)
            end
        end

        if !(unravel)
            evolving_dm .+= delta * (step_coherent + w0 * oft_prefactor * step_dissipative)
            dist = trace_distance_h(Hermitian(evolving_dm), gibbs)
            push!(distances_to_gibbs, dist)
        end
    end
    return HotAlgorithmResults(evolving_dm, distances_to_gibbs, time_steps)
end

function transition_gauss_vectorized(jumps::Vector{JumpOp}, hamiltonian::HamHam, energy_labels::Vector{Float64}, 
    beta::Float64)

    dim = size(hamiltonian.data, 1)
    w0 = energy_labels[2] - energy_labels[1]
    transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)

    T = zeros(ComplexF64, dim^2, dim^2)
    for jump in jumps
        for w in energy_labels
            jump_oft = oft(jump, w, hamiltonian, beta)
            T .+= transition_gauss(w) * kron(jump_oft, conj(jump_oft))
        end
    end
    return w0 * beta * T / sqrt(2 * pi)  # with OFT normalizations
end

function transition_bohr_gauss_from_energy_vectorized(jumps::Vector{JumpOp}, hamiltonian::HamHam, 
    energy_labels::Vector{Float64}, beta::Float64)

    dim = size(hamiltonian.data, 1)
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
    
    T = zeros(ComplexF64, dim^2, dim^2)
    for jump in jumps
        for nu_1 in keys(bohr_dict)
            for nu_2 in keys(bohr_dict)
                gaussians(w) = exp(-beta^2 * (w + 1/beta)^2 /2) * exp(-beta^2 * (w - nu_1)^2 / 4) * exp(-beta^2 * (w - nu_2)^2 / 4)
                for w in energy_labels
                    A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                    A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                    A_nu_1[bohr_dict[nu_1]] .= jump.in_eigenbasis[bohr_dict[nu_1]]
                    A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]

                    T .+= gaussians(w) * kron(A_nu_1, conj(A_nu_2))
                end
            end
        end
    end
    return beta * w0 * T / sqrt(2 * pi)
end

function create_alpha_from_gaussians_as_for_real(energy_labels::Vector{Float64}, hamiltonian::HamHam, beta::Float64)

    w0 = energy_labels[2] - energy_labels[1]

    gaussian_on_A_nu1(w) = exp.(-beta^2 * (w .- hamiltonian.bohr_freqs).^2 / 4)                  # A_ij
    gaussian_on_A_nu2_dagger(w) = adjoint(exp.(-beta^2 * (w .- hamiltonian.bohr_freqs).^2 / 4))  # (A_ik)^dagger

    alpha = zeros(Float64, size(hamiltonian.data))
    for w in energy_labels
        # alpha_kj \sim sum_i  alpha_ij_ik
        alpha .+=  (exp(-beta^2 * (w + 1/beta)^2 /2)) * gaussian_on_A_nu2_dagger(w) * gaussian_on_A_nu1(w) 
    end
    return beta * w0 * alpha / sqrt(8 * pi)
end

function create_alpha_from_gaussians_as_for_real_but_integrated(energy_labels::Vector{Float64}, 
    hamiltonian::HamHam, beta::Float64)

    gaussian_on_A_nu1(w) = exp.(-beta^2 * (w .- hamiltonian.bohr_freqs).^2 / 4)                  # A_ij
    gaussian_on_A_nu2_dagger(w) = adjoint(exp.(-beta^2 * (w .- hamiltonian.bohr_freqs).^2 / 4))  # (A_ik)^dagger

    alpha_integrand(w) = (exp(-beta^2 * (w + 1/beta)^2 /2)) * gaussian_on_A_nu2_dagger(w) * gaussian_on_A_nu1(w)
    alpha, _ = quadgk(alpha_integrand, minimum(energy_labels), maximum(energy_labels); atol=1e-12, rtol=1e-12)
    return beta * alpha / sqrt(8 * pi)
end

function create_alpha_nu1_from_gaussians(nu_2::Float64, hamiltonian::HamHam, energy_labels::Vector{Float64}, beta::Float64)

    w0 = energy_labels[2] - energy_labels[1]
    alpha_nu1_matrix = zeros(Float64, size(hamiltonian.data))
    for w in energy_labels
        alpha_nu1_matrix .+= beta * exp.(-beta^2 * (w .- hamiltonian.bohr_freqs).^2 / 4) * (exp(-beta^2 * (w + 1/beta)^2 /2)
                                    * exp(-beta^2 * (w - nu_2)^2 / 4) / sqrt(8 * pi))
    end
    return w0 * alpha_nu1_matrix
end

function create_alpha_from_gaussians(nu_1::Float64, nu_2::Float64, energy_labels::Vector{Float64}, beta::Float64;
    energy_cutoff_epsilon::Float64 = 1e-3)  # Truncating with this epsilon is good, any larger and it actually has an effect.

    w0 = energy_labels[2] - energy_labels[1]
    alpha = 0.0
    for w in energy_labels
        alpha +=  (exp(-beta^2 * (w + 1/beta)^2 / 2)
                           * exp(-beta^2 * (w - nu_1)^2 / 4) * exp(-beta^2 * (w - nu_2)^2 / 4))
    end
    return beta * w0 * alpha / sqrt(8 * pi)
end

function create_alpha_from_gaussians_integrated(nu_1::Float64, nu_2::Float64, num_energy_bits::Int64, 
    w0::Float64, beta::Float64)

    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = w0 * N_labels
    energy_domain = (minimum(energy_labels), maximum(energy_labels))
    alpha_integrand(w) = beta * (exp(-beta^2 * (w + 1/beta)^2 / 2)
                            * exp(-beta^2 * (w - nu_1)^2 / 4) * exp(-beta^2 * (w - nu_2)^2 / 4) / sqrt(8 * pi))

    alpha_nu1_nu2, _ = quadgk(alpha_integrand, energy_domain[1], energy_domain[2]; atol=1e-12, rtol=1e-12)

    return alpha_nu1_nu2
end

#* TOOLS --------------------------------------------------------------------------------------------------------------------
function create_energy_labels(num_energy_bits::Int64, w0::Float64)
    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = w0 * N_labels
    @assert maximum(energy_labels) >= 2.0  # For good results
    return energy_labels
end

function truncate_energy_labels(energy_labels::Vector{Float64}, beta::Float64, a::Float64, b::Float64,
    with_linear_combination::Bool; cutoff::Float64=1e-12)

    if with_linear_combination
        transition = pick_transition(beta, a, b)  # Linear combination of Gaussians
    else
        transition = w -> exp(-beta^2 * (w + 1/beta)^2 /2)  # Single Gaussian
    end

    gaussfilter(w, nu, beta) = exp(-beta^2 * (w - nu)^2 / 4) * sqrt(beta / sqrt(2 * pi))
    integrand_lb(w) = transition(w) * gaussfilter(w, -0.45, beta)^2
    integrand_ub(w) = transition(w) * gaussfilter(w, 0.45, beta)^2

    min_label_for_lb = Inf
    max_label_for_ub = -Inf
    for w in energy_labels
        # Finding LB
        integrand_lb_val = integrand_lb(w)
        if abs(integrand_lb_val) >= cutoff
            min_label_for_lb = min(min_label_for_lb, w)
        end
        # Finding UB
        integrand_ub_val = integrand_ub(w)
        if abs(integrand_ub_val) >= cutoff
            max_label_for_ub = max(max_label_for_ub, w)
        end
    end

    return energy_labels[min_label_for_lb .<= energy_labels .<= max_label_for_ub]
end
#* --------------------------------------------------------------------------------------------------------------------------
#* --------------------------------------------------------------------------------------------------------------------------

# function integrate_gamma_M(nu_1::Float64, nu_2::Float64, energy_labels::Vector{Float64}, beta::Float64)

#     transition_metro(w) = exp(-beta * max(w + 1/(2 * beta), 0.0))
#     integrand(w) = transition_metro(w) * exp(-beta^2 * (w - nu_1)^2 / 4) * exp(-beta^2 * (w - nu_2)^2 / 4)
#     # integrand(w) = exp(-beta^2 * (w - nu_1)^2 / 4) * exp(-beta^2 * (w - nu_2)^2 / 4)
#     w0 = energy_labels[2] - energy_labels[1]

#     resulting_alpha_M = 0.0
#     for w in energy_labels
#         integrand_w = integrand(w)
#         resulting_alpha_M += integrand_w
#     end

#     return w0 * beta * resulting_alpha_M / sqrt(2*pi)
# end

# function integrate_gamma_gauss(nu_1::Float64, nu_2::Float64, energy_labels::Vector{Float64}, beta::Float64)

#     transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)
#     integrand(w) = transition_gauss(w) * exp(-beta^2 * (w - nu_1)^2 / 4) * exp(-beta^2 * (w - nu_2)^2 / 4)
#     w0 = energy_labels[2] - energy_labels[1]

#     resulting_alpha_gauss = 0.0
#     for w in energy_labels
#         integrand_w = integrand(w)
#         resulting_alpha_gauss += integrand_w
#     end

#     return w0 * beta * resulting_alpha_gauss / sqrt(2*pi)
# end