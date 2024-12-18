using LinearAlgebra
using Random
using Printf
using ProgressMeter
using JLD

include("hamiltonian.jl")
include("ofts.jl")
include("qi_tools.jl")
include("bohr_metro.jl")

#* Parameters
num_qubits = 5
beta = 10.
eta = 0.02  # Just don't make it smaller than 0.018
Random.seed!(666)

with_coherent = true

#* Hamiltonian
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=10)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

#* Gibbs
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
gibbs_vec = vec(gibbs)
gibbs_largest_eigval = real(eigen(gibbs).values[1])

#* Jump operators
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
H::Matrix{ComplexF64} = [1 1; 1 -1] / sqrt(2)
jump_paulis = [[X], [Y], [Z], [H]]

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

#* The Press
@printf("Number of qubits: %d\n", num_qubits)
@printf("Mixing time: %s\n", mixing_time)
@printf("Delta: %s\n", delta)


jump = all_jumps_generated[1]
nu_2 = hamiltonian.bohr_freqs[1, 7]

alpha_nu1_matrix_metro = @time create_alpha_nu1_matrix_metro(hamiltonian.bohr_freqs, nu_2, beta)
