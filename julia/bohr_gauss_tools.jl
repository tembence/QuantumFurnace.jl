using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using BenchmarkTools
using Roots

include("hamiltonian_tools.jl")
include("jump_op_tools.jl")
include("qi_tools.jl")
include("structs.jl")

function construct_liouvillian_bohr_gauss(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, beta::Float64)

    dim = size(hamiltonian.data, 1)

    # Bohr dictionary
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)

    liouv = zeros(ComplexF64, dim^2, dim^2)
    @showprogress desc="Liouvillian (Bohr)..." for jump in jumps
        # Coherent part
        if with_coherent
            coherent_term = coherent_gaussian_bohr(hamiltonian, bohr_dict, jump, beta)
            liouv .+= vectorize_liouvillian_coherent(coherent_term)
        end

        # Dissipative part
        for nu_2 in keys(bohr_dict)
            alpha_nu1_matrix = create_alpha_nu1_matrix(hamiltonian.bohr_freqs, nu_2, beta)

            A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
            A_nu_2_dagger::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)

            A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
            A_nu_2_dagger .= A_nu_2'

            liouv .+= vectorize_liouvillian_diss(alpha_nu1_matrix .* jump.in_eigenbasis, A_nu_2_dagger)
        end
    end
    return liouv
end

function thermalize_bohr_gauss_vectorized(jumps::Vector{JumpOp}, hamiltonian::HamHam, initial_dm::Matrix{ComplexF64}, 
    delta::Float64, mixing_time::Float64, beta::Float64)
    
    dim = size(hamiltonian.data, 1)
    num_liouv_steps = Int(round(mixing_time / delta, digits=0))
    gibbs = gibbs_state_in_eigen(hamiltonian, beta)
    gibbs_vec = vec(gibbs)

    # Bohr dictionary
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
    
    time_steps = [0.0:delta:(mixing_time);]
    evolved_dm_vec = vec(copy(initial_dm))
    distances_to_gibbs = [norm(evolved_dm_vec - gibbs_vec)]

    # This implementation applies all jumps at once for one Liouvillian step.
    @showprogress dt=1 desc="Thermalize (Bohr Gaussian)..." for step in 1:num_liouv_steps

        liouv_matrix_for_step = zeros(ComplexF64, dim^2, dim^2)
        for jump in jumps
            # Coherent part
            if with_coherent
                coherent_term = coherent_gaussian_bohr(hamiltonian, bohr_dict, jump, beta)
                liouv_matrix_for_step .+= vectorize_liouvillian_coherent(coherent_term)
            end

            # Dissipative part
            for nu_2 in keys(bohr_dict)
                A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
                A_nu_2_dagger::SparseMatrixCSC{ComplexF64} = A_nu_2'

                alpha_nu1_matrix = create_alpha_nu1_matrix(hamiltonian.bohr_freqs, nu_2, beta)

                liouv_matrix_for_step .+= vectorize_liouvillian_diss(alpha_nu1_matrix .* jump.in_eigenbasis, A_nu_2_dagger)
            end
        end

        # evolved_dm_vec = exp(delta * liouv_matrix_for_step) * evolved_dm_vec # Perfect Liouvillian evolution
        evolved_dm_vec = evolved_dm_vec + delta * liouv_matrix_for_step * evolved_dm_vec # Trotterized Liouvillian evolution
        dist = norm(evolved_dm_vec - gibbs_vec)
        push!(distances_to_gibbs, dist)
    end
    return HotAlgorithmResults(reshape(evolved_dm_vec, size(hamiltonian.data)), distances_to_gibbs, time_steps)
end

function thermalize_bohr_gauss(jumps::Vector{JumpOp}, hamiltonian::HamHam, initial_dm::Matrix{ComplexF64}, 
    delta::Float64, mixing_time::Float64, beta::Float64)
    
    dim = size(hamiltonian.data, 1)
    num_liouv_steps = Int(round(mixing_time / delta, digits=0))
    gibbs = Hermitian(gibbs_state_in_eigen(hamiltonian, beta))

    # Bohr dictionary
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)

    distances_to_gibbs = [trace_distance_h(Hermitian(initial_dm), gibbs)]
    time_steps = [0.0:delta:(mixing_time);]
    evolved_dm = copy(initial_dm)
    # This implementation applies all jumps at once for one Liouvillian step.
    @showprogress dt=1 desc="Thermalize (Bohr Gaussian)..." for step in 1:num_liouv_steps
        coherent_dm_part = zeros(ComplexF64, dim, dim)
        dissipative_dm_part = zeros(ComplexF64, dim, dim)

        for jump in jumps
            # Coherent part
            if with_coherent
                coherent_term = coherent_gaussian_bohr(hamiltonian, bohr_dict, jump, beta)
                coherent_dm_part .+= - 1im * (coherent_term * evolved_dm - evolved_dm * coherent_term)
            end

            # Dissipative part
            for nu_2 in keys(bohr_dict)
                A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]

                alpha_nu1_matrix = create_alpha_nu1_matrix(hamiltonian.bohr_freqs, nu_2, beta)

                dissipative_dm_part .+= ((alpha_nu1_matrix .* jump.in_eigenbasis) * evolved_dm * A_nu_2' 
                                        - 0.5 * (A_nu_2' * (alpha_nu1_matrix .* jump.in_eigenbasis) * evolved_dm 
                                                + evolved_dm * A_nu_2' * (alpha_nu1_matrix .* jump.in_eigenbasis)))
            end
        end
        #! In other thermalizing fns I evolve with coherent earlier, which is wrong!

        evolved_dm .+= delta * (coherent_dm_part + dissipative_dm_part)
        dist = trace_distance_h(Hermitian(evolved_dm), gibbs)
        push!(distances_to_gibbs, dist)
    end
    return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
end

function create_bohr_dict(hamiltonian::HamHam)
    """Creates a dictionary, where the keys are the Bohr frequencies, and the values are a list of their sparse indices 
    in the Bohr matrix. (With special care on the diagonal elements, that are identically 0.)"""

    bohr_dict = Dict{Float64, Vector{CartesianIndex{2}}}()
    dim = size(hamiltonian.data, 1)
    bohr_dict[0.0] = CartesianIndex{2}.(1:dim, 1:dim) # nu = 0.0 is the diagonal
    for j in 1:dim
        for i in 1:(j - 1)
            bohr_dict[hamiltonian.bohr_freqs[i, j]] = [CartesianIndex{2}(i, j)]
            bohr_dict[-hamiltonian.bohr_freqs[i, j]] = [CartesianIndex{2}(j, i)]
        end
    end
    return bohr_dict
end

function coherent_gaussian_bohr(hamiltonian::HamHam, 
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}}, jump::JumpOp, beta::Float64)

    dim = size(hamiltonian.data, 1)

    B = zeros(ComplexF64, dim, dim)
    for nu_2 in keys(bohr_dict)
        A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
        A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
        f_nu1_matrix = create_f_nu1_matrix(hamiltonian.bohr_freqs, nu_2, beta)

        B .+= A_nu_2' * (f_nu1_matrix .* jump.in_eigenbasis)
    end
    return B
end

function create_alpha_nu1_matrix(bohr_freqs::Matrix{Float64}, nu_2::Float64, beta::Float64)
    """Gaussian parameters = 1/β"""
    alpha_fn(nu_1) = exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2/16)exp(-beta^2 * (nu_1 - nu_2)^2/8) / sqrt(8)
    return alpha_fn.(bohr_freqs)
end

function create_f_nu1_matrix(bohr_freqs::Matrix{Float64}, nu_2::Float64, beta::Float64)
    """Tanh * alpha. Gaussian parameters = 1/β"""
    f_fn(nu_1) = (tanh(-beta * (nu_1 - nu_2) / 4) * exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2/16)exp(-beta^2 * (nu_1 - nu_2)^2/8) / (sqrt(8) * (2 * im)))
    return f_fn.(bohr_freqs)
end

function check_alpha_skew_symmetry(alpha::Function, nu_1::Float64, nu_2::Float64, beta::Float64)
    @assert norm(alpha(nu_1, nu_2) - alpha(-nu_2, -nu_1) * exp(-beta * (nu_1 + nu_2) / 2)) < 1e-14
end

#TODO:
function transition_bohr_gauss()
    #! Gibbsing it
    A_nu_1 = gibbs^(-0.5) * A_nu_1 * gibbs^(0.5)
    A_nu_2 = gibbs^(0.5) * A_nu_2 * gibbs^(-0.5)
end
