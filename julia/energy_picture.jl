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
include("structs.jl")
include("bohr_picture.jl")
include("ofts.jl")


#* BOHR 
function construct_liouvillian_gauss(jumps::Vector{JumpOp}, hamiltonian::HamHam, energy_labels::Vector{Float64}, 
    with_coherent::Bool, beta::Float64)

    dim = size(hamiltonian.data, 1)
    w0 = energy_labels[2] - energy_labels[1]
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
    transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)

    total_liouv_coherent_part = zeros(ComplexF64, dim^2, dim^2)
    total_liouv_diss_part = zeros(ComplexF64, dim^2, dim^2)

    @showprogress desc="Liouvillian (Energy)..." for jump in jumps
        if with_coherent  # There is no energy formulation of the coherent term, only Bohr and time.
            coherent_term = coherent_gaussian_bohr(hamiltonian, bohr_dict, jump, beta)
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = oft(jump, w, hamiltonian, beta)
            total_liouv_diss_part .+= transition_gauss(w) * vectorize_liouvillian_diss(jump_oft)
        end
    end
    oft_norm_squared = beta / sqrt(2 * pi)  #! sqrt(8 * pi) -> sqrt(2 * pi)
    return total_liouv_coherent_part .+ w0 * oft_norm_squared * total_liouv_diss_part
end

function thermalize_gauss(jumps::Vector{JumpOp}, hamiltonian::HamHam, initial_dm::Matrix{ComplexF64},
    energy_labels::Vector{Float64}, with_coherent::Bool, delta::Float64, mixing_time::Float64, beta::Float64)

    w0 = energy_labels[2] - energy_labels[1]
    dim = size(hamiltonian.data, 1)
    num_liouv_steps = Int(round(mixing_time / delta, digits=0))
    gibbs = Hermitian(gibbs_state_in_eigen(hamiltonian, beta))
    transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)

    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)

    distances_to_gibbs = [trace_distance_h(Hermitian(initial_dm), gibbs)]
    time_steps = [0.0:delta:(mixing_time);]
    evolved_dm = copy(initial_dm)
    # This implementation applies all jumps at once for one Liouvillian step.
    @showprogress dt=1 desc="Thermalize (Gaussian)..." for step in 1:num_liouv_steps
        coherent_dm_part = zeros(ComplexF64, dim, dim)
        dissipative_dm_part = zeros(ComplexF64, dim, dim)

        for jump in jumps
            # Coherent part
            if with_coherent
                coherent_term = coherent_gaussian_bohr(hamiltonian, bohr_dict, jump, beta)
                coherent_dm_part .+= - 1im * (coherent_term * evolved_dm - evolved_dm * coherent_term)
            end

            # Dissipative part
            for w in energy_labels
                jump_oft = oft(jump, w, hamiltonian, beta)
                jump_dag_jump = jump_oft' * jump_oft
                dissipative_dm_part .+= transition_gauss(w) * (jump_oft * evolved_dm * jump_oft' 
                                                                - 0.5 * (jump_dag_jump * evolved_dm 
                                                                        + evolved_dm * jump_dag_jump))
            end
        end
        oft_prefactor = w0 * beta / sqrt(2 * pi)  # discrete sum w0 + OFT normalization^2 + Fourier factor
        evolved_dm .+= delta * (coherent_dm_part + oft_prefactor * dissipative_dm_part)
        dist = trace_distance_h(Hermitian(evolved_dm), gibbs)
        push!(distances_to_gibbs, dist)
    end
    return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
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

#* METRO
function construct_liouvillian_metro(jumps::Vector{JumpOp}, hamiltonian::HamHam, energy_labels::Vector{Float64}, 
    with_coherent::Bool, beta::Float64)

    dim = size(hamiltonian.data, 1)
    w0 = energy_labels[2] - energy_labels[1]
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
    transition_metro(w) = exp(-beta * max(w + 1/(2*beta), 0.0))

    total_liouv_coherent_part = zeros(ComplexF64, dim^2, dim^2)
    total_liouv_diss_part = zeros(ComplexF64, dim^2, dim^2)

    @showprogress desc="Liouvillian (Energy)..." for jump in jumps
        if with_coherent  # There is no energy formulation of the coherent term, only Bohr and time.
            coherent_term = coherent_metro_bohr(hamiltonian, bohr_dict, jump, beta)
            @printf("Is B METRO Hermitian?:%s\n", norm(coherent_term - coherent_term'))
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = oft(jump, w, hamiltonian, beta)
            total_liouv_diss_part .+= transition_metro(w) * vectorize_liouvillian_diss(jump_oft)
        end
    end
    oft_norm_squared = beta / sqrt(2 * pi) #! sqrt(8 * pi) -> sqrt(2 * pi)
    return total_liouv_coherent_part .+ w0 * oft_norm_squared * total_liouv_diss_part
end

function transition_metro(jumps::Vector{JumpOp}, hamiltonian::HamHam, energy_labels::Vector{Float64}, 
    beta::Float64)

    dim = size(hamiltonian.data, 1)
    w0 = energy_labels[2] - energy_labels[1]
    transition_metro(w) = exp(-beta * max(w + 1/(2 * beta), 0.0))

    total_liouv_transition = zeros(ComplexF64, dim^2, dim^2)
    for jump in jumps
        for w in energy_labels
            jump_oft = oft(jump, w, hamiltonian, beta)

            vectorized_transition = transition_metro(w) * kron(jump_oft, conj(jump_oft))
            total_liouv_transition .+=  vectorized_transition 
        end
    end
    return w0 * beta * total_liouv_transition / sqrt(2 * pi)
end

function integrate_gamma_M(nu_1::Float64, nu_2::Float64, energy_labels::Vector{Float64}, beta::Float64)

    transition_metro(w) = exp(-beta * max(w + 1/(2 * beta), 0.0))
    integrand(w) = transition_metro(w) * exp(-beta^2 * (w - nu_1)^2 / 4) * exp(-beta^2 * (w - nu_2)^2 / 4)
    # integrand(w) = exp(-beta^2 * (w - nu_1)^2 / 4) * exp(-beta^2 * (w - nu_2)^2 / 4)
    w0 = energy_labels[2] - energy_labels[1]

    resulting_alpha_M = 0.0
    for w in energy_labels
        integrand_w = integrand(w)
        resulting_alpha_M += integrand_w
    end

    return w0 * beta * resulting_alpha_M / sqrt(2*pi)
end

function integrate_gamma_gauss(nu_1::Float64, nu_2::Float64, energy_labels::Vector{Float64}, beta::Float64)

    transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)
    integrand(w) = transition_gauss(w) * exp(-beta^2 * (w - nu_1)^2 / 4) * exp(-beta^2 * (w - nu_2)^2 / 4)
    w0 = energy_labels[2] - energy_labels[1]

    resulting_alpha_gauss = 0.0
    for w in energy_labels
        integrand_w = integrand(w)
        resulting_alpha_gauss += integrand_w
    end

    return w0 * beta * resulting_alpha_gauss / sqrt(2*pi)
end