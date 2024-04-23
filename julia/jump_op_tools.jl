using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using TensorOperations
include("hamiltonian_tools.jl")


mutable struct JumpOp
    data::Matrix{ComplexF64}
    in_eigenbasis::Matrix{ComplexF64}
    bohr_decomp::Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}
    unique_freqs::Vector{Float64}
end

function oft(jump::JumpOp, energy::Float64, hamiltonian::HamHam, num_energy_bits::Int64, sigma::Float64, beta::Float64)

    #! I have to do this for each energy, maybe it's faster if it's down somewhere higehr level with energy array
    gaussian_weighted_freqs = add_gaussian_weight(jump.unique_freqs, energy, sigma)  

    #! This assumes that we can store all A_nus in a dict
    #! If needed we can rewrite such that we take 1 nu, get A_nu, and add it to the OFT matrix, this way we store much less
    jump_oft = Matrix(sum([gaussian_weighted_freqs[i] * jump.bohr_decomp[jump.unique_freqs[i]] for i in 1:length(jump.unique_freqs)]))

    return jump_oft
end

function find_unique_jump_freqs(jump::JumpOp, hamiltonian::HamHam)
    
    # Nonzero entry indices of jump operators
    jump_indices = findall(!iszero, jump.in_eigenbasis)
    jump_freqs = hamiltonian.bohr_freqs[jump_indices]  # Bohr freqs are already rounded
    jump.unique_freqs = unique(jump_freqs)
end

#TODO: Test this
#TODO: Also there are multiple keys of the value 0... which is no bueno
function construct_A_nus(jump::JumpOp, hamiltonian::HamHam, num_energy_bits::Int64)
"""Constructrs jump.bohr_decomp = Dict {bohr energy (nu): A_nu}"""

    N = 2^num_energy_bits
    oft_precision = ceil(Int, abs(log10(N^(-1))))

    # A'_ij is the entry for the jump between eigenenergies j -> i
    # jump.in_eigenbasis

    # Matrix of all energy jumps in Hamiltonian, B_ij = E_i - E_j and i,j are ordered from smallest to largest energy
    jump.unique_freqs = unique(hamiltonian.bohr_freqs)  #! Not correct

    jump_bohr_indices = get_jump_bohr_indices(hamiltonian.bohr_freqs)
    for (bohr_freq, indices) in jump_bohr_indices
        jump_op_nu = spzeros(ComplexF64, size(jump.data))
        for (i, j) in indices
            jump_op_nu[i, j] = jump.in_eigenbasis[i, j]
        end
        jump.bohr_decomp[bohr_freq] = jump_op_nu
    end
end

function get_jump_bohr_indices(bohr_freqs::Matrix{Float64})

    jump_bohr_indices = Dict{Float64, Vector{Tuple{Int64, Int64}}}()  # Dict {bohr energy (nu): [all contributing (i,j)]}
    nrows, ncols = size(bohr_freqs)
    for j in 1:ncols, i in 1:nrows
        value = bohr_freqs[i, j]
        if haskey(jump_bohr_indices, value)
            push!(jump_bohr_indices[value], (i, j))
        else
            jump_bohr_indices[value] = [(i, j)]
        end
    end
    return jump_bohr_indices
end

function add_gaussian_weight(freqs::Vector{Float64}, energy::Float64, sigma_t::Float64)
    return exp.(-(energy .- freqs).^2 * sigma_t^2)
end

function entry_wise_oft(jump::JumpOp, energy::Float64, hamiltonian::HamHam, sigma::Float64, beta::Float64)

    # The for loop version is the same but a bit slower than broadcasting
    # for j in 1:ncols, i in 1:nrows
    #     jump_oft[i, j] = jump_op_in_eigenbasis[i, j] * exp(-(energy - bohr_freqs[i, j])^2 * sigma^2)
    # end
    
    return jump.in_eigenbasis .* exp.(-(energy .- hamiltonian.bohr_freqs).^2 * sigma^2)
end

function entry_wise_oft(jump::Tuple{JumpOp}, energies::Vector{Float64}, hamiltonian::Tuple{HamHam}, 
    sigma::Float64, beta::Float64)

    return jump[1].in_eigenbasis .* exp.(-(energies .- hamiltonian[1].bohr_freqs).^2 * sigma^2)
end

#* ---------- Test ----------

#* Parameters

# @printf("Number of threads: %d\n", Threads.nthreads())
# num_qubits = 4
# num_energy_bits = 6
# N = 2^num_energy_bits
# oft_precision = ceil(Int, abs(log10(N^(-1))))
# delta = 0.01
# sigma = 5.
# bohr_bound = 0.
# beta = 1.
# eig_index = 2
# jump_site_index = 1

# #* Hamiltonian
# coeffs = fill(1.0, 3)
# hamiltonian = find_ideal_heisenberg(num_qubits, coeffs, batch_size=1)
# initial_state = hamiltonian.eigvecs[:, eig_index]
# hamiltonian.bohr_freqs = round.(hamiltonian.eigvals .- transpose(hamiltonian.eigvals), digits=oft_precision+3)

# #* Jump operators
# sigmax::Matrix{ComplexF64} = [0 1; 1 0]
# jump_op = Matrix(pad_term([sigmax], num_qubits, jump_site_index))
# jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs

# # For full Liouvillian dynamics:
# # all_x_jump_ops = []
# # for q in 1:num_qubits
# #     padded_x = pad_term([jump_op], q, num_qubits)
# #     push!(all_x_jump_ops, padded_x)
# # end

# #* Fourier labels
# N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
# time_labels = N_labels
# energy_labels = 2 * pi * N_labels / N

# phase = -0.44 * N / (2 * pi)
# energy = 2 * pi * phase / N
# @printf("\nEnergy: %f\n", energy)

# jump = JumpOp(jump_op,
#         jump_op_in_eigenbasis,
#         Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
#         zeros(0))

# find_unique_jump_freqs(jump, hamiltonian)


# @time construct_A_nus(jump, hamiltonian, num_energy_bits)

# @time oft_op = oft(jump, energy, hamiltonian, num_energy_bits, sigma, beta)

# @time entry_wise_oft_op = entry_wise_oft(jump, energy, hamiltonian, sigma, beta)

# @printf("Are they the same?: %s\n", norm(oft_op - entry_wise_oft_op) < 1e-10)