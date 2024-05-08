using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using TensorOperations
using JLD
using Plots

include("hamiltonian_tools.jl")
include("qi_tools.jl")
include("jump_op_tools.jl")

num_qubits = 2
sigma = 5.
beta = 1.
jump_site_index = 1
eig_index = 1

#* Hamiltonian
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3), batch_size=1)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
initial_state = hamiltonian.eigvecs[:, eig_index]
initial_state_in_eigenbasis = hamiltonian.eigvecs' * initial_state
initial_energy = hamiltonian.eigvals[eig_index]
println("\nEigenenergies:")
display(hamiltonian.eigvals)
println("\nBohr frequencies:")
display(hamiltonian.bohr_freqs)
display(hamiltonian.bohr_freqs[2, 1])
println("\nHamiltonian:")
display(hamiltonian.data)

#* Jump operators
sigmax::Matrix{ComplexF64} = [0 1; 1 0]
jump_op = Matrix(pad_term([sigmax], num_qubits, jump_site_index))
jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
jump = JumpOp(jump_op,
        jump_op_in_eigenbasis,
        Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
        zeros(0))
construct_A_nus(jump, hamiltonian)

#* Labels
num_energy_bits = ceil(Int64, log2((0.45 * 2) / hamiltonian.w0)) + 1
N = 2^(num_energy_bits)
t0 = 2 * pi / (N * hamiltonian.w0)

N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
time_labels = t0 * N_labels
energy_labels = hamiltonian.w0 * N_labels

some_integer = Int(floor(hamiltonian.bohr_freqs[2, 1] / hamiltonian.w0))
an_energy = hamiltonian.w0 * some_integer
phase = an_energy * N / (2 * pi)
a_time = (N / 2 + 3) * t0

@printf("Number of qubits: %d\n", num_qubits)
@printf("Number of energy bits: %d\n", num_energy_bits)
@printf("Energy unit: %e\n", hamiltonian.w0)
@printf("Time unit: %e\n", t0)
@printf("\nEnergy: %f\n", an_energy)

#* Hamiltonian check.
# sigmax::Matrix{ComplexF64} = [0 1; 1 0]
# sigmay::Matrix{ComplexF64} = [0 -im; im 0]
# sigmaz::Matrix{ComplexF64} = [1 0; 0 -1]
# XX = kron(sigmax, sigmax)
# YY = kron(sigmay, sigmay)
# ZZ = kron(sigmaz, sigmaz)

# hamiltonian_check = 2 * hamiltonian.base_coeffs[1]*(XX + YY + ZZ) + 
#                     hamiltonian.symbreak_coeffs[1]*kron(sigmaz, I(2)) +
#                     hamiltonian.symbreak_coeffs[2]*kron(I(2), sigmaz) +
#                     hamiltonian.shift * I(4)

# display(eigvals(hamiltonian_check))
# @printf("Distance between Hamiltonians: %f\n", frobenius_norm(hamiltonian.data - hamiltonian_check))

#* Diagonalizaion check.
# diag_ham = hamiltonian.eigvecs' * hamiltonian.data * hamiltonian.eigvecs
# isapprox(diag_ham, Diagonal(hamiltonian.eigvals), atol=1e-15)

#* A_nus check.
# display(collect(keys(jump.bohr_decomp)))
# display(jump.bohr_decomp[hamiltonian.bohr_freqs[3, 2]])

# Does all the A_nus add up to A?
# reconstr_jump_op = zeros(ComplexF64, size(jump.data))
# for (bohr_freq, jump_op_nu) in jump.bohr_decomp
#     reconstr_jump_op += jump_op_nu
# end
# frobenius_norm(reconstr_jump_op - jump.in_eigenbasis)

# t = 10 * t0
# heis_A = exp(1im * hamiltonian.data * t) * jump.data * exp(-1im * hamiltonian.data * t)
# heis_A_in_eigenbasis = hamiltonian.eigvecs' * heis_A * hamiltonian.eigvecs
# @printf("Num of unique freqs %d\n", length(jump.unique_freqs))
# A_nu_sum = zeros(ComplexF64, size(jump.data))
# for (nu, A_nu) in jump.bohr_decomp
#     A_nu_sum += exp(1im * nu * t) * A_nu
# end
# println("Distance between Heisenberg and A_nu sum:")
# display(frobenius_norm(heis_A_in_eigenbasis - A_nu_sum))

#* Energy jumps check.
# println("\nJump operator in eigenbasis:")
# display(jump.in_eigenbasis)
# jumped_state = jump.in_eigenbasis * initial_state_in_eigenbasis
# new_energy = jumped_state' * Diagonal(hamiltonian.eigvals) * jumped_state
# new_energy2 = jumped_state[2]^2 * hamiltonian.bohr_freqs[2, 1] + jumped_state[4]^2 * hamiltonian.bohr_freqs[4, 1]
# display(norm(new_energy - new_energy2))

# jump_op_engineered_in_eigenbasis = zeros(ComplexF64, size(jump.in_eigenbasis))
# jump_op_engineered_in_eigenbasis[2, 1] = 1
# jumped_state_eng = jump_op_engineered_in_eigenbasis * initial_state_in_eigenbasis
# new_energy_eng = jumped_state_eng' * Diagonal(hamiltonian.eigvals) * jumped_state_eng

#* Fourier sum and trafo check.
ft = exp.(-time_labels.^2 / (4 * sigma^2))
ft_norm = sqrt(sum(ft.^2))
Fw = exp.(- sigma^2 * energy_labels.^2)
Fw_norm = sqrt(sum(Fw.^2))

fourier_sum = 0.
nu = 0.405
for t in time_labels
    fourier_sum += exp(-1im * t * (an_energy - nu)) * exp(-t^2 / (4 * sigma^2)) / sqrt(2*pi)
end
fourier_sum /= ft_norm

fourier_trafod = exp(-(an_energy - nu)^2 * sigma^2) / Fw_norm
#diff
norm(fourier_sum - fourier_trafod)