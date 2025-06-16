using LinearAlgebra
using Random
using Printf
using JLD

include("hamiltonian.jl")
include("qi_tools.jl")
include("misc_tools.jl")
include("structs.jl")
include("timelike_tools.jl")

#* Config
num_qubits = 5
dim = 2^num_qubits
T = 2.0
num_trotter_steps = 1

#* Hamiltonian
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=100)
# hamiltonian_terms = [["X", "X"], ["Y", "Y"], ["Z", "Z"]]
# hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
# hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits)

#* Trotter
trotter = create_trotter(hamiltonian, T, num_trotter_steps)
trotter_error_T = compute_trotter_error(hamiltonian, trotter, T)

exact_U = exp(1im * hamiltonian.data * T)
trotter_U = trotterize22(hamiltonian, T, num_trotter_steps)
@printf("Deviation between exact vs trotter: %s\n", norm(exact_U - trotter_U))

#* Expm Pauli check
# X::Matrix{ComplexF64} = [0 1; 1 0]
# Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
# Z::Matrix{ComplexF64} = [1 0; 0 -1]
# H::Matrix{ComplexF64} = [1 1; 1 -1] / sqrt(2)
# id::Matrix{ComplexF64} = I(2)

# coeff = 1.0
# term = ["X", "X"]
# term_2 = ["Y", "Y"]
# delta_t = 0.1
# paaasition = 1
# pauli_term = pauli_string_to_matrix(term)
# padded_term = Matrix(pad_term(pauli_term, num_qubits, paaasition)) # Padding checked: correct

# expm_pauli = expm_pauli_padded(term, delta_t * coeff / 2, num_qubits, paaasition)  # Pauli expm checked: correct

# num_trotter_steps = 100000
# res = I(2^num_qubits)
# @time for step in num_trotter_steps
#     res *= expm_pauli
# end

# @time expm_pauli^num_trotter_steps

# expm_pauli_2 = expm_pauli_padded(term_2, delta_t * coeff / 2, num_qubits, paaasition)
# # expm_pauli_byhand = exp(1im * delta_t * coeff * padded_term / 2)
# # norm(expm_pauli - expm_pauli_byhand)

# norm(expm_pauli * expm_pauli_2 - expm_pauli_2 * expm_pauli)  # Commutation checked: correct