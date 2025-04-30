using LinearAlgebra
using SparseArrays
using Random
using Printf

include("hamiltonian.jl")

function commutator(A::AbstractMatrix, B::AbstractMatrix)
    return A * B - B * A
end

#* Config
num_qubits = 10

#* Hamiltonian
hamiltonian_terms = [["X", "X"], ["Y", "Y"], ["Z", "Z"]]
hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits; periodic=true)

#* Jumps
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
H::Matrix{ComplexF64} = [1 1; 1 -1] / sqrt(2)
id::Matrix{ComplexF64} = I(2)
jump_paulis = [[X], [Y], [Z], [H]]

jumps::Vector{JumpOp} = []
for pauli in jump_paulis
    for site in 1:num_qubits
    jump_op = Matrix(pad_term(pauli, num_qubits, site))
    jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
    # jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
    orthogonal = (jump_op == transpose(jump_op))
    jump = JumpOp(jump_op,
            jump_op_in_eigenbasis,
            Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
            zeros(0),
            zeros(0,0),
            orthogonal) 
    push!(jumps, jump)
    end
end

#FIXME: It feels off, as the correlation falls off well, but the time evolutions are still not so close to each other.
mid = Int(ceil(num_qubits/2))  # X at mid
L = 2
jump = jumps[mid]
B = jumps[mid + num_qubits + L]  # Y at L = 2

norm(commutator(jump.data, B.data))

T = 0.01
U_T = Diagonal(exp.(-1im * hamiltonian.eigvals * T))
evolved_jump = hamiltonian.eigvecs * U_T' * jump.in_eigenbasis * U_T * hamiltonian.eigvecs'  # In computational basis

l2_correl = norm(commutator(evolved_jump, B.data))

# Make (2*L + some)-site Hamiltonian centred around middle
site_radius = 5
cutoff_hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, site_radius; periodic=false)
U_cutoff_T = Diagonal(exp.(-1im * cutoff_hamiltonian.eigvals * T))

jump_cutoff = pad_term([X], site_radius, Int(ceil(site_radius/2)))
jump_cutoff_in_cutoff_basis = cutoff_hamiltonian.eigvecs' * jump_cutoff * cutoff_hamiltonian.eigvecs
# padded_cutoff_hamiltonian = pad_term([cutoff_hamiltonian.data], 8, 4)
# cutoff_eigvals, cutoff_eigvecs = eigen(Matrix{ComplexF64}(padded_cutoff_hamiltonian))


cutoff_evolved_jump = U_cutoff_T' * jump_cutoff_in_cutoff_basis * U_cutoff_T
cutoff_evolved_jump_in_eigen =  pad_term([cutoff_hamiltonian.eigvecs * cutoff_evolved_jump * cutoff_hamiltonian.eigvecs'], 6, 3)

norm(cutoff_evolved_jump_in_eigen - evolved_jump)
