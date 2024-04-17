using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
include("jump_op_tools.jl")
include("hamiltonian_tools.jl")
include("liouvillian_tools.jl")

um_qubits = 4
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
jump_op = Matrix(pad_term([sigmax], num_qubits, jump_site_index))

JUMP = JumpOp(jump_op,
        Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
        zeros(0))


