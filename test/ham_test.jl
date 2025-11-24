using Revise

if !isdefined(Main, :QuantumFurnace)
    includet("../src/QuantumFurnace.jl")
end

using .QuantumFurnace
using LinearAlgebra

num_qubits = 4
dim = 2^num_qubits
beta = 10.  # 5, 10, 30

# X::Matrix{ComplexF64} = [0 1; 1 0]
# Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
# Z::Matrix{ComplexF64} = [1 0; 0 -1]
# id::Matrix{ComplexF64} = I(2)

hamiltonian_terms = [[X, X], [Y, Y], [Z, Z]]
hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
disordering_term = [Z]
disordering_coeffs = rand(num_qubits)
hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, disordering_term, disordering_coeffs, num_qubits)

h1 = @time find_ideal_heisenberg(num_qubits, hamiltonian_coeffs)
h2 = @time find_ideal_heisenberg_new(num_qubits, hamiltonian_coeffs)

norm(h1.data - h2.data)
