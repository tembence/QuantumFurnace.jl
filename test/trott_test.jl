using Revise

if !isdefined(Main, :QuantumFurnace)
    includet("../src/QuantumFurnace.jl")
end

using .QuantumFurnace
using LinearAlgebra

num_qubits = 4
dim = 2^num_qubits
beta = 10.  # 5, 10, 30

num_energy_bits = 10
w0 = 0.05
max_E = w0 * 2^num_energy_bits / 2
t0 = 2pi / (2^num_energy_bits * w0)
num_trotter_steps_per_t0 = 100

X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
id::Matrix{ComplexF64} = I(2)

hamiltonian_terms = [[X, X], [Y, Y], [Z, Z]]
hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
disordering_term = [Z]
disordering_coeffs = rand(num_qubits)
hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, disordering_term, disordering_coeffs, num_qubits)

trotter = create_trotter(hamiltonian, t0, num_trotter_steps_per_t0)

trotter_error_T = compute_trotter_error(hamiltonian, trotter, 2^num_energy_bits * t0 / 2)
# gibbs_in_trotter = Hermitian(trotter.eigvecs' * gibbs_state(hamiltonian, beta) * trotter.eigvecs)
# @printf("Trotter is created.\n")