using LinearAlgebra
using Random
using Printf
using JLD2

include("../src/hamiltonian.jl")
include("../src/qi_tools.jl")
include("../src/structs.jl")
include("../src/bohr_picture.jl")
include("../src/energy_picture.jl")
include("../src/time_picture.jl")
include("../src/ofts.jl")
include("../src/coherent.jl")
include("../src/misc_tools.jl")

#* Config
num_qubits = 5
dim = 2^num_qubits
T = 2.0
num_t0_steps = 100
num_trotter_steps_per_t0 = 1
num_trotter_steps = num_t0_steps * num_trotter_steps_per_t0
t0 = T / num_trotter_steps

#* Hamiltonian
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=100)
# hamiltonian_terms = [["X", "X"], ["Y", "Y"], ["Z", "Z"]]
# hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
# hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits)

exact_U = exp(1im * hamiltonian.data * T)
eigva, eigve = eigen(exact_U)
exact_U_from_eigvals = eigve * Diagonal(eigva) * eigve'
norm(exact_U - exact_U_from_eigvals)

#* Trotter
trotter_full_time = create_trotter(hamiltonian, T, num_trotter_steps)
trotter_t0 = create_trotter(hamiltonian, t0, num_trotter_steps_per_t0)
trotter_error_T = compute_trotter_error(hamiltonian, trotter_full_time, T)

exact_U = exp(1im * hamiltonian.data * T)
trotter_U = trotterize2(hamiltonian, T, num_trotter_steps)
trotter_U_t0 = trotterize2(hamiltonian, t0, num_trotter_steps_per_t0)
norm(trotter_U - trotter_U_t0^num_t0_steps)  #* Correct if we don't exponentiate the eigvals.
eigva, eigve = eigen(trotter_U_t0)
eigva
trotter_U_from_t0 = eigve * Diagonal(eigva .^ num_t0_steps) * eigve'  # Basis trafo to computational basis
norm(trotter_U - trotter_U_from_t0)
@printf("Deviation between exact vs trotter: %s\n", norm(exact_U - trotter_U))

reconstructed_t0 = eigve * Diagonal(eigva) * eigve'
@printf("Eigen decomposition reconstruction error: %e\n", norm(trotter_U_t0 - reconstructed_t0))



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