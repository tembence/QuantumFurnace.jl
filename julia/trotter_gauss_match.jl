using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using BenchmarkTools
using Roots
using QuadGK

include("hamiltonian.jl")
include("qi_tools.jl")
include("structs.jl")
include("bohr_gauss.jl")
include("energy_gauss.jl")
include("time_gauss.jl")
include("trotter_gauss.jl")
include("ofts.jl")

num_qubits = 4
dim = 2^num_qubits
num_energy_bits = 10
beta = 10.
Random.seed!(666)
with_coherent = false

#* Hamiltonian
# hamiltonian_terms = [["X", "X"], ["Z"]]
# hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
# hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits)
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=100)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

N = 2^(num_energy_bits)
w0 = 32 / N
t0 = 2 * pi / (N * w0)
@printf("Smallest Bohr frequency: %s\n", hamiltonian.nu_min)
@printf("Chosen w0: %s\n", w0)
# w0 = hamiltonian.nu_min
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
energy_labels = w0 * N_labels
time_labels = t0 * N_labels  #? Truncating energy domain should concern time domain?
maximum(time_labels)
maximum(energy_labels)

get_energy_cutoff_for_alpha(beta, nu_max, eps) = nu_max + sqrt(4 * log(1/eps) / beta^2)
# nu_max = 0.45
energy_cutoff_epsilon = 1e-4
energy_cutoff_for_alpha = get_energy_cutoff_for_alpha(beta, 0.45, energy_cutoff_epsilon)
energy_labels = energy_labels[abs.(energy_labels) .<= energy_cutoff_for_alpha]
maximum(energy_labels)

#* Trotter
num_trotter_steps_per_t0 = 1000
trotter = create_trotter(hamiltonian, t0, num_trotter_steps_per_t0)
trotter_error_T = compute_trotter_error(hamiltonian, trotter, 2^num_energy_bits * t0)
gibbs_in_trotter = trotter.eigvecs' * gibbs_state(hamiltonian, beta) * trotter.eigvecs
@printf("Num trotter steps / t0: %d\n", num_trotter_steps_per_t0)
@printf("Max order Trotter error on an OFT: %s\n", trotter_error_T)

#* Jumps
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
H::Matrix{ComplexF64} = [1 1; 1 -1] / sqrt(2)
jump_paulis = [[X], [Y], [Z]]

# All jumps once
all_jumps_generated::Vector{JumpOp} = []
for pauli in jump_paulis
    for site in 1:num_qubits
    jump_op = Matrix(pad_term(pauli, num_qubits, site))
    jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
    jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
    orthogonal = (jump_op == transpose(jump_op))
    jump = JumpOp(jump_op,
            jump_op_in_eigenbasis,
            Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
            zeros(0),
            jump_in_trotter_basis,
            orthogonal) 
    push!(all_jumps_generated, jump)
    end
end

#* Liouvillians
#! They are different bases!!!
liouv_bohr = construct_liouvillian_bohr_gauss(all_jumps_generated, hamiltonian, with_coherent, beta)
liouv_time = construct_liouvillian_gauss_time(all_jumps_generated, hamiltonian, time_labels, energy_labels, 
    with_coherent, beta)
liouv_trotter = construct_liouvillian_gauss_trotter(all_jumps_generated, trotter, time_labels, energy_labels, 
    with_coherent, beta)

@printf("Deviation Time - Bohr Liouvillian: %s\n", norm(liouv_time - liouv_bohr))
@printf("Deviation Trotter - Time Liouvillian: %s\n", norm(liouv_trotter - liouv_time))
@printf("Deviation Trotter - Bohr Liouvillian: %s\n", norm(liouv_trotter - liouv_bohr))

#* A(t)
# jump = all_jumps_generated[2]
# oft_t_norm = t0 * sqrt(sqrt(2 / pi)/beta) / sqrt(2 * pi)
# total_oft_error = 0.0
# for w in energy_labels[2]
#     # @printf("Energy: %s\n", w)
#     jump_oft_trotter = trotter_oft(jump, w, trotter, time_labels, beta)
#     jump_oft_trotter_in_eigenbasis = hamiltonian.eigvecs' * trotter.eigvecs * jump_oft_trotter * trotter.eigvecs' * hamiltonian.eigvecs
#     jump_oft_time = time_oft(jump, w, hamiltonian, time_labels, beta)
#     # jump_oft_energy = oft(jump, w, hamiltonian, beta) / Fw_norm
#     # jump_oft_time = time_oft(jump, w, hamiltonian, time_labels, beta) / (ft_norm * sqrt(length(time_labels)))
#     err = norm(jump_oft_trotter_in_eigenbasis - jump_oft_time)
#     total_oft_error += err
#     # @printf("OFT energy vs time deviation: %s\n", err)
# end
# @printf("Total error: %s\n", total_oft_error)