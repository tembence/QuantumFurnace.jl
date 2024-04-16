using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
include("hamiltonian_tools.jl")


#* Parameters
num_qubits = 12
num_energy_bits = 6
N = 2^num_energy_bits
oft_precision = ceil(Int, abs(log10(N^(-1))))
delta = 0.01
sigma = 5.
bohr_bound = 0.
beta = 1.
eig_index = 2
jump_site_index = 1

#* Hamiltonian
coeffs = fill(1.0, 3)
hamiltonian = find_ideal_heisenberg(num_qubits, coeffs, batch_size=1)
initial_state = hamiltonian.eigvecs[:, eig_index]

#* Jump operators
sigmax::Matrix{ComplexF64} = [0 1; 1 0]
jump_op = Hermitian(Matrix(pad_term([sigmax], num_qubits, jump_site_index)))
jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs

# For full Liouvillian dynamics:
# all_x_jump_ops = []
# for q in 1:num_qubits
#     padded_x = pad_term(jump_op, q, num_qubits)
#     push!(all_x_jump_ops, padded_x)
# end

#* Fourier labels
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
time_labels = N_labels
energy_labels = 2 * pi * N_labels / N

phase = -0.44 * N / (2 * pi)
energy = 2 * pi * phase / N
@printf("\nEnergy: %f\n", energy)

# Matrix of all energy jumps in Hamiltonian, B_ij = E_i - E_j and i,j are ordered from smallest to largest energy
bohr_freqs = round.(hamiltonian.eigvals .- transpose(hamiltonian.eigvals), digits=oft_precision+2)
unique_bohr_freqs = unique(bohr_freqs)

function add_gaussian_weight(freqs::Vector{Float64}, energy::Float64, sigma_t::Float64)
    return exp.(-(energy .- freqs).^2 * sigma_t^2)
end

@printf("Adding Gaussian weights to Bohr frequencies of length %d\n", length(unique_bohr_freqs))
@time begin
gaussian_weighed_freqs = add_gaussian_weight(unique_bohr_freqs, energy, sigma)
end



# println("\nBohr frequencies")
# display(bohr_freqs)

# Nonzero entries of jump op in eigenbasis
# println("\nJump operator in eigenbasis")
# display(round.(Real.(jump_op_in_eigenbasis), digits=4))
# println("\n")

function get_jump_bohr_indices(bohr_freqs::Matrix{Float64})
    jump_bohr_indices = Dict{Float64, Vector{Tuple{Int64, Int64}}}()
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

jump_bohr_indices = get_jump_bohr_indices(bohr_freqs)

# Construct the sparse parts of the jump operator that contribute to the same Bohr frequency
jump_op_nus = Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}()
for (bohr_freq, indices) in jump_bohr_indices
    jump_op_nu = spzeros(ComplexF64, size(jump_op_in_eigenbasis))
    for (i, j) in indices
        jump_op_nu[i, j] = jump_op_in_eigenbasis[i, j]
    end
    jump_op_nus[bohr_freq] = jump_op_nu
end

#TODO: Test this
#TODO: Also there multiple keys of the value 0... which is no bueno

# number of keys in jump_bohr_indices
# println("\nNumber of keys in jump_bohr_indices")
# display(length(keys(jump_bohr_indices)))
# display(jump_op_nus)




# nonzeros_indicies_in_jump = findall(!iszero, jump_op_in_eigenbasis)
# println("\nNonzero indicies in jump operator")
# display(nonzeros)

# # Find Bohr frequencies of the jump
# jump_bohr_freqs = bohr_freqs[nonzeros_indicies_in_jump]
# println("\nJump Bohr frequencies")
# display(jump_bohr_freqs)

# # Find indices of jump operator that contribute to the same bohr frequency
# # jump_bohr_freqs_indices = Dict(Float64, Vector)
# for (i, freq) in enumerate(jump_bohr_freqs)
#     @printf("\nJump bohr frequency: %f\n", freq)
#     @printf("%f", i)
# end



