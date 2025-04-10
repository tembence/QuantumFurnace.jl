using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using BenchmarkTools
using Roots
using DataStructures
using SpecialFunctions: erfc

include("hamiltonian.jl")
include("qi_tools.jl")
include("structs.jl")

#* GAUSS --------------------------------------------------------------------------------------------------------------------
function construct_liouvillian_bohr_gauss(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, beta::Float64)

    dim = size(hamiltonian.data, 1)

    # Bohr dictionary
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
    unique_freqs = keys(bohr_dict)

    liouv = zeros(ComplexF64, dim^2, dim^2)
    @showprogress desc="Liouvillian (Bohr Gauss)..." for jump in jumps
        # Coherent part
        if with_coherent
            coherent_term = coherent_gaussian_bohr(hamiltonian, bohr_dict, jump, beta)
            liouv .+= vectorize_liouvillian_coherent(coherent_term)
        end

        # Dissipative part
        for nu_2 in unique_freqs
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

        for jump in jumps  # Apply the sum of jumps at once
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

        evolved_dm .+= delta * (coherent_dm_part + dissipative_dm_part)
        dist = trace_distance_h(Hermitian(evolved_dm), gibbs)
        push!(distances_to_gibbs, dist)
    end
    return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
end

function coherent_gaussian_bohr(hamiltonian::HamHam, 
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}}, jump::JumpOp, beta::Float64)

    dim = size(hamiltonian.data, 1)
    unique_freqs = keys(bohr_dict)

    B = zeros(ComplexF64, dim, dim)
    for nu_2 in unique_freqs
        A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
        A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
        f_nu1_matrix = create_f_nu1_matrix(hamiltonian.bohr_freqs, nu_2, beta)

        B .+= A_nu_2' * (f_nu1_matrix .* jump.in_eigenbasis)
    end
    return B
end

function transition_bohr_gauss_vectorized(jumps::Vector{JumpOp}, hamiltonian::HamHam, beta::Float64; do_adjoint::Bool=false)

    dim = size(hamiltonian.data, 1)
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)

    T = zeros(ComplexF64, dim^2, dim^2)
    for jump in jumps
        for nu_2 in keys(bohr_dict)
            alpha_nu1_matrix = create_alpha_nu1_matrix(hamiltonian.bohr_freqs, nu_2, beta)

            A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
            A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
            A_nu_2_dagger::SparseMatrixCSC{ComplexF64} = A_nu_2'

            if !(do_adjoint)
                T .+= kron(alpha_nu1_matrix .* jump.in_eigenbasis, transpose(A_nu_2_dagger))
            else
                T .+= kron(adjoint(alpha_nu1_matrix .* jump.in_eigenbasis), transpose(A_nu_2))
            end
        end
    end
    return T
end

function transition_bohr_vectorized_slow(jumps::Vector{JumpOp}, hamiltonian::HamHam, beta::Float64; do_adjoint::Bool=false)
    dim = size(hamiltonian.data, 1)
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
    alpha(nu_1, nu_2) = exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2/16) * exp(-beta^2 * (nu_1 - nu_2)^2/8) / sqrt(8)

    T = zeros(ComplexF64, dim^2, dim^2)
    for jump in jumps
        for nu_1 in keys(bohr_dict)
            for nu_2 in keys(bohr_dict)
                A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                A_nu_1[bohr_dict[nu_1]] .= jump.in_eigenbasis[bohr_dict[nu_1]]
                A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
                if !(do_adjoint)
                    A_nu_2_dagger::SparseMatrixCSC{ComplexF64} = A_nu_2'
                    T .+= alpha(nu_1, nu_2) * kron(A_nu_1, transpose(A_nu_2_dagger))
                else
                    A_nu_1_dagger::SparseMatrixCSC{ComplexF64} = A_nu_1'
                    T .+= alpha(nu_1, nu_2) * kron(A_nu_1_dagger, transpose(A_nu_2))
                end
            end
        end
    end
    return T
end

function transition_bohr_gauss_gibbsed_vectorized(jumps::Vector{JumpOp}, hamiltonian::HamHam, beta::Float64)

    dim = size(hamiltonian.data, 1)
    gibbs = gibbs_state_in_eigen(hamiltonian, beta)
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)

    T = zeros(ComplexF64, dim^2, dim^2)
    for jump in jumps
        for nu_2 in keys(bohr_dict)
            alpha_nu1_matrix = create_alpha_nu1_matrix(hamiltonian.bohr_freqs, nu_2, beta)

            A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
            A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
            A_nu_2_dagger::SparseMatrixCSC{ComplexF64} = A_nu_2'

            A_nu1s_gibbsed = gibbs^(-1/2) * (alpha_nu1_matrix .* jump.in_eigenbasis) * gibbs^(1/2)
            A_nu_2_dagger_gibbsed = gibbs^(1/2) * A_nu_2_dagger * gibbs^(-1/2)

            T .+= kron(A_nu1s_gibbsed, transpose(A_nu_2_dagger_gibbsed))
        end
    end
    return T
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

function create_alpha_nu1_matrix(bohr_freqs::Matrix{Float64}, nu_2::Float64, beta::Float64)
    """Gaussian parameters = 1/β"""
    alpha_fn(nu_1) = exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2/16) * exp(-beta^2 * (nu_1 - nu_2)^2/8) / sqrt(2) #! sqrt(8)->sqrt(2)
    return alpha_fn.(bohr_freqs)
end

function create_alpha_gauss(nu_1::Float64, nu_2::Float64, beta::Float64)
    """Gaussian parameters = 1/β"""
    alpha_fn(nu_1) = exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2/16) * exp(-beta^2 * (nu_1 - nu_2)^2/8) / sqrt(2) #! sqrt(8)->sqrt(2)
    return alpha_fn(nu_1)
end

function create_f_nu1_matrix(bohr_freqs::Matrix{Float64}, nu_2::Float64, beta::Float64)
    tanh_matrix = tanh.(-beta * (bohr_freqs .- nu_2) / 4) / (2im)
    alpha_matrix = create_alpha_nu1_matrix(bohr_freqs, nu_2, beta)
    return tanh_matrix .* alpha_matrix  # f
end

function create_f_gauss(nu_1::Float64, nu_2::Float64, beta::Float64)
    """Tanh * alpha. Gaussian parameters = 1/β"""
    alpha_nu1_nu2 = create_alpha_gauss(nu_1, nu_2, beta)
    return tanh(-beta * (nu_1 - nu_2) / 4) * alpha_nu1_nu2 / (2im)
end

function B_nu_gauss(nu::Float64, nu_2::Float64, hamiltonian::HamHam, bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}}, 
    jump::JumpOp, beta::Float64)

    dim = size(hamiltonian.data, 1)

    B_nu = zeros(ComplexF64, dim, dim)
    nu_1 = nu + nu_2
    
    A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
    A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
    A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
    A_nu_1[bohr_dict[nu_1]] .= jump.in_eigenbasis[bohr_dict[nu_1]]
    f_nu1_nu2 = create_f_gauss(nu_1, nu_2, beta)

    B_nu .= f_nu1_nu2 * A_nu_2' * (A_nu_1)

    return B_nu
end

function R_nu_gauss(nu::Float64, nu_2::Float64, hamiltonian::HamHam, bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}}, 
    jump::JumpOp, beta::Float64)

    dim = size(hamiltonian.data, 1)

    R_nu = zeros(ComplexF64, dim, dim)
    nu_1 = nu + nu_2
    
    A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
    A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
    A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
    A_nu_1[bohr_dict[nu_1]] .= jump.in_eigenbasis[bohr_dict[nu_1]]
    f_nu1_nu2 = create_alpha_gauss(nu_1, nu_2, beta)

    R_nu .= f_nu1_nu2 * A_nu_2' * (A_nu_1)

    return R_nu
end

#* METRO --------------------------------------------------------------------------------------------------------------------
function construct_liouvillian_bohr_metro(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, beta::Float64)

    dim = size(hamiltonian.data, 1)

    # Bohr dictionary
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
    unique_freqs = keys(bohr_dict)

    liouv = zeros(ComplexF64, dim^2, dim^2)
    @showprogress desc="Liouvillian (Bohr Metro)..." for jump in jumps
        # Coherent part
        if with_coherent
            coherent_term = coherent_bohr_metro(hamiltonian, bohr_dict, jump, beta)
            liouv .+= vectorize_liouvillian_coherent(coherent_term)
        end

        # Dissipative part
        for nu_2 in unique_freqs
            alpha_nu1_matrix = create_alpha_nu1_matrix_metro(hamiltonian.bohr_freqs, nu_2, beta)

            A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
            A_nu_2_dagger::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)

            A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
            A_nu_2_dagger .= A_nu_2'

            liouv .+= vectorize_liouvillian_diss(alpha_nu1_matrix .* jump.in_eigenbasis, A_nu_2_dagger)
        end
    end
    return liouv
end

function coherent_bohr_metro(hamiltonian::HamHam, 
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}}, jump::JumpOp, beta::Float64)

    dim = size(hamiltonian.data, 1)
    unique_freqs = keys(bohr_dict)

    B = zeros(ComplexF64, dim, dim)
    for nu_2 in unique_freqs

        A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
        A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
        f_nu1_matrix = create_f_nu1_matrix_metro(hamiltonian.bohr_freqs, nu_2, beta)

        B .+= A_nu_2' * (f_nu1_matrix .* jump.in_eigenbasis)
    end
    return B
end

function B_nu_metro(nu::Float64, nu_2::Float64, hamiltonian::HamHam, bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}}, 
    jump::JumpOp, beta::Float64)

    dim = size(hamiltonian.data, 1)

    B_nu = zeros(ComplexF64, dim, dim)
    nu_1 = nu + nu_2
    
    A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
    A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
    A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
    A_nu_1[bohr_dict[nu_1]] .= jump.in_eigenbasis[bohr_dict[nu_1]]
    f_nu1_nu2 = create_f_metro(nu_1, nu_2, beta)

    B_nu .= f_nu1_nu2 * A_nu_2' * (A_nu_1)

    return B_nu
end

function R_nu_metro(nu::Float64, nu_2::Float64, hamiltonian::HamHam, bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}}, 
    jump::JumpOp, beta::Float64)

    dim = size(hamiltonian.data, 1)

    R_nu = zeros(ComplexF64, dim, dim)
    nu_1 = nu + nu_2
    
    A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
    A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
    A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
    A_nu_1[bohr_dict[nu_1]] .= jump.in_eigenbasis[bohr_dict[nu_1]]
    f_nu1_nu2 = create_alpha_metro(nu_1, nu_2, beta)

    R_nu .= f_nu1_nu2 * A_nu_2' * (A_nu_1)

    return R_nu
end

function transition_bohr_metro(jumps::Vector{JumpOp}, hamiltonian::HamHam, beta::Float64; do_adjoint::Bool=false)

    dim = size(hamiltonian.data, 1)
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
    unique_freqs = keys(bohr_dict)

    T = zeros(ComplexF64, dim^2, dim^2)
    for jump in jumps
        for nu_2 in unique_freqs
            alpha_nu1_matrix = create_alpha_nu1_matrix_metro(hamiltonian.bohr_freqs, nu_2, beta)

            A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
            A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
            A_nu_2_dagger::SparseMatrixCSC{ComplexF64} = A_nu_2'

            if !(do_adjoint)
                T .+= kron(alpha_nu1_matrix .* jump.in_eigenbasis, transpose(A_nu_2_dagger))
            else
                T .+= kron(adjoint(alpha_nu1_matrix .* jump.in_eigenbasis), transpose(A_nu_2))
            end
        end
    end
    return T
end

function transition_bohr_metro_gibbsed(jumps::Vector{JumpOp}, hamiltonian::HamHam, beta::Float64)

    dim = size(hamiltonian.data, 1)
    gibbs = gibbs_state_in_eigen(hamiltonian, beta)
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
    unique_freqs = keys(bohr_dict)

    T = zeros(ComplexF64, dim^2, dim^2)
    for jump in jumps
        for nu_2 in unique_freqs
            alpha_nu1_matrix = create_alpha_nu1_matrix_metro(hamiltonian.bohr_freqs, nu_2, beta)

            A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
            A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
            A_nu_2_dagger::SparseMatrixCSC{ComplexF64} = A_nu_2'

            A_nu1s_gibbsed = gibbs^(-1/2) * (alpha_nu1_matrix .* jump.in_eigenbasis) * gibbs^(1/2)
            A_nu_2_dagger_gibbsed = gibbs^(1/2) * A_nu_2_dagger * gibbs^(-1/2)

            T .+= kron(A_nu1s_gibbsed, transpose(A_nu_2_dagger_gibbsed))
        end
    end
    return T
end

function create_alpha_nu1_matrix_metro(bohr_freqs::Matrix{Float64}, nu_2::Float64, beta::Float64)
    alpha_fn(nu_1) = exp(-beta^2 * (nu_1 - nu_2)^2 / 8) * (erfc((1 + beta * (nu_1 + nu_2)) / sqrt(8))
    + exp(-beta * (nu_1 + nu_2) / 2) * erfc((1 - beta * (nu_1 + nu_2)) / sqrt(8))) / 2

    return alpha_fn.(bohr_freqs)
end

function create_alpha_metro(nu_1::Float64, nu_2::Float64, beta::Float64)
    alpha_fn(nu_1) = exp(-beta^2 * (nu_1 - nu_2)^2 / 8) * (erfc((1 + beta * (nu_1 + nu_2)) / sqrt(8)) 
    + exp(-beta * (nu_1 + nu_2) / 2) * erfc((1 - beta * (nu_1 + nu_2)) / sqrt(8))) / 2
    return alpha_fn(nu_1)
end

function create_f_nu1_matrix_metro(bohr_freqs::Matrix{Float64}, nu_2::Float64, beta::Float64)
    """Tanh * alpha. Gaussian parameters = 1/β"""
    tanh_matrix = tanh.(-beta * (bohr_freqs .- nu_2) / 4) / (2im)
    alpha_matrix = create_alpha_nu1_matrix_metro(bohr_freqs, nu_2, beta)
    return tanh_matrix .* alpha_matrix  # f
end

function create_f_metro(nu_1::Float64, nu_2::Float64, beta::Float64)
    """Tanh * alpha. Gaussian parameters = 1/β"""
    alpha_nu1_nu2 = create_alpha_metro(nu_1, nu_2, beta)
    return tanh(-beta * (nu_1 - nu_2) / 4) * alpha_nu1_nu2 / (2im)
end

#* BETTER LINEAR COMBINATION ------------------------------------------------------------------------------------------------
function construct_liouvillian_bohr_eh(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, 
    beta::Float64, a::Float64, b::Float64)

    dim = size(hamiltonian.data, 1)

    # Bohr dictionary
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
    unique_freqs = keys(bohr_dict)

    liouv = zeros(ComplexF64, dim^2, dim^2)
    @showprogress desc="Liouvillian (Bohr Eh)..." for jump in jumps
        # Coherent part
        if with_coherent
            coherent_term = coherent_bohr_eh(hamiltonian, bohr_dict, jump, beta, a, b)
            liouv .+= vectorize_liouvillian_coherent(coherent_term)
        end

        # Dissipative part
        for nu_2 in unique_freqs
            alpha_nu1_matrix = create_alpha_eh.(hamiltonian.bohr_freqs, nu_2, beta, a, b)

            A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
            A_nu_2_dagger::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)

            A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
            A_nu_2_dagger .= A_nu_2'

            liouv .+= vectorize_liouvillian_diss(alpha_nu1_matrix .* jump.in_eigenbasis, A_nu_2_dagger)
        end
    end
    return liouv
end

function coherent_bohr_eh(hamiltonian::HamHam, 
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}}, jump::JumpOp, beta::Float64, a::Float64, b::Float64)

    dim = size(hamiltonian.data, 1)
    unique_freqs = keys(bohr_dict)

    B = zeros(ComplexF64, dim, dim)
    for nu_2 in unique_freqs

        A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
        A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
        f_nu1_matrix = create_f_eh.(hamiltonian.bohr_freqs, nu_2, beta, a, b)

        B .+= A_nu_2' * (f_nu1_matrix .* jump.in_eigenbasis)
    end
    return B
end

function create_f_eh(nu_1::Float64, nu_2::Float64, beta::Float64, a::Float64, b::Float64)
    alpha_eh = create_alpha_eh(nu_1, nu_2, beta, a, b)
    return tanh(-beta * (nu_1 - nu_2) / 4) * alpha_eh / (2im)
end

function create_alpha_eh(nu_1::Float64, nu_2::Float64, beta::Float64, a::Float64, b::Float64)
    """For the parallel way use create_alpha_eh.(bohr_freqs, nu_2, beta, a)"""
    
    eh = sqrt(4 * a / beta + 1)
    nu = nu_1 + nu_2

    alpha_nu_1 = (exp(-beta^2 * (nu_1 - nu_2)^2 / 8) 
        * exp(a / (2 * beta)) * exp(-beta * nu * (1 + eh) / 4) * (
        erfc((eh * (1 + b) - beta * nu) / sqrt(8 * (1 + b))) 
        + exp(beta * nu * eh / 2) * erfc((eh * (1 + b) + beta * nu) / sqrt(8 * (1 + b)))) / 2)
    return alpha_nu_1
end

#* TOOLS --------------------------------------------------------------------------------------------------------------------
function check_alpha_skew_symmetry(alpha::Function, nu_1::Float64, nu_2::Float64, beta::Float64)
    @assert norm(alpha(nu_1, nu_2) - alpha(-nu_2, -nu_1) * exp(-beta * (nu_1 + nu_2) / 2)) < 1e-14
end

function create_bohr_dict(hamiltonian::HamHam)
    """Creates a dictionary, where the keys are the Bohr frequencies, and the values are a list of their sparse indices 
    in the Bohr matrix. (With special care on the diagonal elements, that are identically 0.)"""

    bohr_dict = DefaultDict{Float64, Vector{CartesianIndex{2}}}(() -> CartesianIndex{2}[])
    dim = size(hamiltonian.data, 1)
    bohr_dict[0.0] = CartesianIndex{2}.(1:dim, 1:dim) # nu = 0.0 is the diagonal and might be other offdiags
    for j in 1:dim
        for i in 1:(j - 1)
            push!(bohr_dict[hamiltonian.bohr_freqs[i, j]], CartesianIndex{2}(i, j))
            push!(bohr_dict[-hamiltonian.bohr_freqs[i, j]], CartesianIndex{2}(j, i))
        end
    end
    return bohr_dict
end

function find_all_nu1s_to_nu2(nu_2::Float64, nu::Float64, unique_freqs::Set{Float64})
    good_nu1s::Set{Float64} = Set()
    for nu_1 in unique_freqs
        # if round(nu_1 - nu_2, digits=15) == nu
        if (nu_1 - nu_2 == nu)
            push!(good_nu1s, nu_1)
        else
            continue
        end
    end
    return good_nu1s
end
#* --------------------------------------------------------------------------------------------------------------------------
#* --------------------------------------------------------------------------------------------------------------------------

#* Some good code techniques to remember, but are not used
# function exact_mask(bohr_freqs::Matrix{Float64}, nu_2::Float64, unique_freqs::Set{Float64})
#     nu1s_minus_nu_2 = bohr_freqs .- nu_2
#     return in.(nu1s_minus_nu_2, Ref(unique_freqs)) .+ 0
# end

# function approx_mask(bohr_freqs::Matrix{Float64}, nu_2::Float64, unique_freqs::Set{Float64})
#     eps = 1e-14
#     nu1s_minus_nu_2 = bohr_freqs .- nu_2
#     return map(x -> any(abs.(unique_freqs .- x) .< eps), nu1s_minus_nu_2) .+ 0
# end


# function create_alpha_nu1_matrix_metro_integrated(bohr_freqs::Matrix{Float64}, nu_2::Float64, beta::Float64)
#     """Gaussian parameters = 1/β, but w_gamma=x is a linear combination, i.e. integral."""
#     alpha_fn(nu_1, x) = exp(-beta^2 * (nu_1 + nu_2 + 2*x)^2/16) * exp(-beta^2 * (nu_1 - nu_2)^2/8) / sqrt(2)
#     inverse_weight(x) = 1 / sqrt(2 * pi * (2*x/beta - 1/beta^2))
#     alpha_nu_1_matrix(x) = alpha_fn.(bohr_freqs, x) * inverse_weight(x)
#     alpha_metro_nu1_matrix = quadgk(alpha_nu_1_matrix, 1/(2*beta), Inf)[1]
#     return alpha_metro_nu1_matrix
# end

# function get_metropolis_weight_integrated(energy::Float64, beta::Float64)
#     weighed_gaussian(x) = exp(-beta^2*(energy + x)^2 / (4*x*beta - 2)) / sqrt(2 * pi * (2*x/beta - 1/beta^2))
#     return quadgk(weighed_gaussian, 1/(2*beta), Inf)[1]
# end