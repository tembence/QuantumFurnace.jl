using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using JLD

include("jump_op_tools.jl")
include("hamiltonian_tools.jl")
include("liouvillian_tools.jl")

um_qubits = 4
num_energy_bits = 8
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
initial_dm = initial_state * initial_state'
hamiltonian.bohr_freqs = round.(hamiltonian.eigvals .- transpose(hamiltonian.eigvals), digits=oft_precision+3)

#* Jump operators
sigmax::Matrix{ComplexF64} = [0 1; 1 0]
jump_op = Matrix(pad_term([sigmax], num_qubits, jump_site_index))
jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
jump = JumpOp(jump_op,
        jump_op_in_eigenbasis,
        Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
        zeros(0))
find_unique_jump_freqs(jump, hamiltonian)

#* /// RUN ///

