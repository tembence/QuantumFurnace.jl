using LinearAlgebra
using Arpack
using SparseArrays
include("jl_classical_tools.jl")

# Gibbs state
beta = 1.
num_qubits = 4
N = 2^num_qubits
sigmax::Matrix{ComplexF64} = [0 1; 1 0]
sigmay::Matrix{ComplexF64} = [0.0 -im; im 0.0]
sigmaz::Matrix{ComplexF64} = [1 0; 0 -1]
terms = [[sigmax, sigmax], [sigmay, sigmay], [sigmaz, sigmaz]]

hamiltonian = construct_base_ham(terms, [1.0, 1.0, 1.0], num_qubits)

# gibbs with diagonalization
eigvals, eigvecs = eigen(hamiltonian)
gibbs_state = eigvecs * Diagonal(exp.(-beta * eigvals)) * eigvecs' / sum(exp.(-beta * eigvals))

