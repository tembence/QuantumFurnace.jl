using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using BenchmarkTools
using Roots

include("hamiltonian.jl")
include("qi_tools.jl")
include("structs.jl")
include("bohr_gauss.jl")
include("energy_gauss.jl")

#* Config
num_qubits = 3
dim = 2^num_qubits
num_energy_bits = 6
beta = 10.
Random.seed!(666)
with_coherent = true
@printf("B was added: %s\n", with_coherent)

# Config for algorithmic thermalization
mixing_time = 15.0
delta = 0.01

#* Hamiltonian
# hamiltonian_terms = [["X", "X"], ["Z"]]
# hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
# hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits)
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=100)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
initial_dm = Matrix{ComplexF64}(I(dim) / dim)
@assert norm(real(tr(initial_dm)) - 1.) < 1e-15 "Trace is not 1.0"
@assert norm(initial_dm - initial_dm') < 1e-15 "Not Hermitian"

N = 2^(num_energy_bits)
# w0 = 4/N
w0 = 0.04
# w0 = hamiltonian.nu_min
@printf("Smallest Bohr frequency: %s\n", hamiltonian.nu_min)
@printf("Chosen w0: %s\n", w0)
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
energy_labels = w0 * N_labels
(maximum(energy_labels) - minimum(energy_labels) )/ (N - 1) == w0
maximum(energy_labels)

# Energy labels truncation
alpha_cutoff(beta, nu_max, eps) = (-(1/beta - nu_max) + sqrt((1/beta - nu_max)^2 
                                                - 4 * (1/(2*beta^2) + nu_max^2/2 - log(beta/(sqrt(2*pi)*eps))/beta^2))) / 2
gaussians_cutoff_epsilon = 1e-16  # Makes the result only worse sometimes at 1e-14 from 1e-16
energy_cutoff_for_alpha = alpha_cutoff(beta, 0.45, gaussians_cutoff_epsilon)

# check_alpha_fn(w, beta, nu_max) = beta * exp(-beta^2*(w + 1/beta)^2 / 2) * exp(-beta^2 * (w - nu_max)^2 / 4)^2 / sqrt(2 * pi)
# alpha_at_energy_cutoff = check_alpha_fn(energy_cutoff_for_alpha, beta, 0.45)
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

#* Thermalization
results_bohr = @time thermalize_bohr_gauss(all_jumps_generated, hamiltonian, initial_dm, delta, mixing_time, beta)
results = @time thermalize_gauss(all_jumps_generated, hamiltonian, initial_dm, energy_labels, with_coherent, 
delta, mixing_time, beta)

@printf("Last distance to Gibbs in BOHR: %s\n", results_bohr.distances_to_gibbs[end])
@printf("Last distance to Gibbs: %s\n", results.distances_to_gibbs[end])

#* Full Liouvillian match
# liouv_energy = @time construct_liouvillian_gauss(all_jumps_generated, hamiltonian, energy_labels, with_coherent, beta)
# liouv_bohr = @time construct_liouvillian_bohr_gauss(all_jumps_generated, hamiltonian, with_coherent, beta)
# @printf("Deviation between Liouvillians (Bohr - Energy): %s\n", norm(liouv_bohr - liouv_energy))

# # Energy
# liouv_eigvals, liouv_eigvecs = eigen(liouv_energy) 
# steady_state_vec = liouv_eigvecs[:, end]
# steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
# steady_state_dm /= tr(steady_state_dm)

# lambda2 = liouv_eigvals[end] - liouv_eigvals[end-1]
# @printf("Lambda2: %s\n", lambda2)

# @printf("Steady state closeness to Gibbs for Liouvillian (Energy): %s\n", norm(steady_state_dm - gibbs))

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
# T_energy = transition_gauss_vectorized(all_jumps_generated, hamiltonian, energy_labels, beta)
# T_bohr = transition_bohr_gauss_vectorized(all_jumps_generated, hamiltonian, beta)
# norm(T_bohr - T_energy)