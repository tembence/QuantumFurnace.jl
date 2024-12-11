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
include("bohr_gauss_tools.jl")
include("energy_gauss_tools.jl")

num_qubits = 4
dim = 2^num_qubits
num_energy_bits = 6
beta = 10.
Random.seed!(666)
with_coherent = true

#* Hamiltonian
hamiltonian_terms = [["X", "X"], ["Z"]]
hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

N = 2^(num_energy_bits)
w0 = 2 / N
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
energy_labels = w0 * N_labels
maximum(energy_labels)
get_energy_cutoff_for_alpha(beta, nu_max, eps) = nu_max + sqrt(4 * log(1/eps) / beta^2)
# nu_max = 0.45
energy_cutoff_epsilon = 1e-4
energy_cutoff_for_alpha = get_energy_cutoff_for_alpha(beta, 0.45, energy_cutoff_epsilon)
energy_labels = energy_labels[abs.(energy_labels) .<= energy_cutoff_for_alpha]
maximum(energy_labels)

#* Jumps
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
H::Matrix{ComplexF64} = [1 1; 1 -1] / sqrt(2)
jump_paulis = [[X], [Y], [Z]]

# All jumps once
all_jumps_generated::Vector{JumpOp} = []
for pauli in jump_paulis
    for site in 1:num_qubits
    jump_op = Matrix(pad_term(pauli, num_qubits, site))
    jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
    jump_in_trotter_basis = zeros(0, 0)
    orthogonal = (jump_op == adjoint(jump_op))
    jump = JumpOp(jump_op,
            jump_op_in_eigenbasis,
            Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
            zeros(0),
            jump_in_trotter_basis,
            orthogonal) 
    push!(all_jumps_generated, jump)
    end
end

# bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
# the_jump = all_jumps_generated[1]
# round.(real.(the_jump.in_eigenbasis), digits=4)
# bohr_dict[0.0]
# nu_2 = hamiltonian.bohr_freqs[1, 2]
# alpha_fn(nu_1, nu_2) = exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2/16) * exp(-beta^2 * (nu_1 - nu_2)^2/8) / sqrt(8)
# alpha_nu1_matrix = create_alpha_nu1_matrix(hamiltonian.bohr_freqs, nu_2, beta)

# A_nu1s = alpha_nu1_matrix .* the_jump.in_eigenbasis
# A_nu1s_constructed = zeros(ComplexF64, size(hamiltonian.data))
# for nu_1 in keys(bohr_dict)
#     @printf("nu_1: %s\n", nu_1)
#     @printf("Bohr dict nu_1: %s\n", bohr_dict[nu_1])
#     A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
#     A_nu_1[bohr_dict[nu_1]] .= the_jump.in_eigenbasis[bohr_dict[nu_1]]
#     A_nu1s_constructed .+= alpha_fn(nu_1, nu_2) * A_nu_1
# end
# norm(A_nu1s - A_nu1s_constructed)


# for nu_2 in keys(bohr_dict)
#     alpha_nu1_matrix = create_alpha_nu1_matrix(hamiltonian.bohr_freqs, nu_2, beta)
#     A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
#     A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
#     A_nu_2_dagger::SparseMatrixCSC{ComplexF64} = A_nu_2'
#     A_nu1s = alpha_nu1_matrix .* the_jump.in_eigenbasis
# end

#* Transition part of Liouvillian
T_energy = transition_gauss_vectorized(all_jumps_generated, hamiltonian, energy_labels, beta)
T_bohr = transition_bohr_gauss_vectorized(all_jumps_generated, hamiltonian, beta)
norm(T_bohr - T_energy)

#* Other comparison for integral
# Energy side
# transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)
# energy_side = zeros(ComplexF64, size(hamiltonian.data))
# for jump in all_jumps_generated
#     for w in energy_labels
#         jump_oft = oft(jump, w, hamiltonian, beta)
#         energy_side .+= transition_gauss(w) * adjoint(jump_oft) * jump_oft
#     end
# end
# energy_side = w0 * beta * energy_side / sqrt(8 * pi)

# # Bohr side
# bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
# bohr_side = zeros(ComplexF64, size(hamiltonian.data))
# for jump in all_jumps_generated
#     for nu_2 in keys(bohr_dict)
#         alpha_nu1_matrix = create_alpha_nu1_matrix(hamiltonian.bohr_freqs, nu_2, beta)

#         A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
#         A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
#         bohr_side .+= adjoint(A_nu_2) * (alpha_nu1_matrix .* jump.in_eigenbasis)
#     end
# end

# norm(energy_side - bohr_side)

#* Alpha from Gaussians with sum (as for real)
#! This jump agnostic match works
# alpha_fn(nu_1, nu_2) = exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2/16) * exp(-beta^2 * (nu_1 - nu_2)^2/8) / sqrt(8)
# alpha_from_gaussians_as_for_real = create_alpha_from_gaussians_as_for_real(energy_labels, hamiltonian, beta)
# total_error_for_real = 0.0
# @showprogress dt=1 desc="Building alpha..." for k in 1:dim
#     for j in 1:dim
#         an_entry_kj = alpha_from_gaussians_as_for_real[k, j]

#         nu1s = hamiltonian.bohr_freqs[:, j]
#         nu2s = hamiltonian.bohr_freqs[:, k]
#         alpha = 0.0
#         for (nu1, nu2) in zip(nu1s, nu2s)
#             alpha += alpha_fn(nu1, nu2)
#         end

#         total_error_for_real += norm(an_entry_kj - alpha)
#     end
# end

# @printf("Total error between alphas (as for real): %s\n", total_error_for_real)

#* Alpha nu1 matrix
# total_error_matricized = 0.0
# @printf("Starting to build alpha...\n")
# @time begin
#     @showprogress dt=1 desc="Building alpha..." for nu_2 in hamiltonian.bohr_freqs
#         alpha_nu_1_matrix_from_gaussian = create_alpha_nu1_from_gaussians(hamiltonian, nu_2, energy_labels, beta)
#         alpha_nu_1 = create_alpha_nu1_matrix(hamiltonian.bohr_freqs, nu_2, beta)
#         total_error_matricized += norm(alpha_nu_1_matrix_from_gaussian - alpha_nu_1)
#     end
# end
# @printf("Total error between alphas (matricized): %s\n", total_error_matricized)

#* Alpha match nu by nu
# alpha_fn(nu_1, nu_2) = exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2/16) * exp(-beta^2 * (nu_1 - nu_2)^2/8) / sqrt(8)
# max_deviation = 0.0
# for nu_1 in hamiltonian.bohr_freqs
#     for nu_2 in hamiltonian.bohr_freqs
#         alpha = alpha_fn(nu_1, nu_2)
#         alpha_from_energy = create_alpha_from_gaussians(nu_1, nu_2, energy_labels, beta)
#         deviation = norm(alpha - alpha_from_energy)
#         if deviation > max_deviation
#             max_deviation = deviation
#         end
#     end
# end

# @printf("Max deviation between alphas (nu by nu): %s\n", max_deviation)

