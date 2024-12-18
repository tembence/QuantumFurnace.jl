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

#* Configs
num_qubits = 3
dim = 2^num_qubits
num_energy_bits = 6
beta = 10.
Random.seed!(666)
with_coherent = true

# Config for algorithmic thermalization
delta = 0.01
mixing_time = 15.0

#* Hamiltonian
# hamiltonian_terms = [["X", "X"], ["Z"]]
# hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
# hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits)
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=100)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
initial_dm = Matrix{ComplexF64}(I(dim) / dim)
@assert norm(real(tr(initial_dm)) - 1.) < 1e-15 "Trace is not 1.0"
@assert norm(initial_dm - initial_dm') < 1e-15 "Not Hermitian"

#* Fourier labels
N = 2^(num_energy_bits)
w0 = 32 / N
t0 = 2 * pi / (N * w0)
@printf("Smallest Bohr frequency: %s\n", hamiltonian.nu_min)
@printf("Chosen w0: %s\n", w0)
@printf("t0: %s\n", t0)
# w0 = hamiltonian.nu_min
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
energy_labels = w0 * N_labels
time_labels = t0 * N_labels  #? Truncating energy domain should concern time domain?

# Energy labels truncation
alpha_cutoff(beta, nu_max, eps) = (-(1/beta - nu_max) + sqrt((1/beta - nu_max)^2 
                                                - 4 * (1/(2*beta^2) + nu_max^2/2 - log(beta/(sqrt(2*pi)*eps))/beta^2))) / 2
gaussians_cutoff_epsilon = 1e-16  # Makes the result only worse sometimes at 1e-14
energy_cutoff_for_alpha = alpha_cutoff(beta, 0.45, gaussians_cutoff_epsilon)
energy_labels = energy_labels[abs.(energy_labels) .<= energy_cutoff_for_alpha]
energy_labels = energy_labels[abs.(energy_labels) .<= energy_cutoff_for_alpha]

#* Trotter
num_trotter_steps_per_t0 = 10
trotter = create_trotter(hamiltonian, t0, num_trotter_steps_per_t0)
trotter_error_T = compute_trotter_error(hamiltonian, trotter, 2^num_energy_bits * t0)
gibbs_in_trotter = trotter.eigvecs' * gibbs_state(hamiltonian, beta) * trotter.eigvecs
@printf("Num trotter steps / t0: %d\n", num_trotter_steps_per_t0)
@printf("Max order Trotter error on an OFT: %s\n", trotter_error_T)

initial_dm_in_trotter = trotter.trafo_from_eigen_to_trotter * initial_dm * trotter.trafo_from_eigen_to_trotter'

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

#* Thermalization
# results = @time thermalize_gauss(all_jumps_generated, hamiltonian, initial_dm, energy_labels, with_coherent,
# delta, mixing_time, beta)
# results_time = @time thermalize_gauss_time(all_jumps_generated, hamiltonian, initial_dm, energy_labels, time_labels,
# with_coherent, delta, mixing_time, beta)
results_trotter = @time thermalize_gauss_trotter(all_jumps_generated, trotter, initial_dm, energy_labels, time_labels,
with_coherent, delta, mixing_time, beta)

evolved_dm = results.evolved_dm
evolved_dm_time = results_time.evolved_dm
evolved_dm_trotter = results_trotter.evolved_dm
@printf("Difference between evolved dms (TIME vs ENERGY): %s\n", norm(evolved_dm - evolved_dm_time))
@printf("Difference between evolved dms (TIME vs TROTTER): %s\n", norm(evolved_dm_trotter - evolved_dm_time))

#* Liouvillians
# liouv_bohr = construct_liouvillian_bohr_gauss(all_jumps_generated, hamiltonian, with_coherent, beta)
# liouv_time = construct_liouvillian_gauss_time(all_jumps_generated, hamiltonian, time_labels, energy_labels, 
#     with_coherent, beta)
# liouv_trotter = construct_liouvillian_gauss_trotter(all_jumps_generated, trotter, time_labels, energy_labels, 
#     with_coherent, beta)

# @printf("Deviation Time - Bohr Liouvillian: %s\n", norm(liouv_time - liouv_bohr))
# @printf("Deviation Trotter - Time Liouvillian: %s\n", norm(liouv_trotter - liouv_time))
# @printf("Deviation Trotter - Bohr Liouvillian: %s\n", norm(liouv_trotter - liouv_bohr))
# norm(liouv_time - liouv_bohr)


# liouv_eigvals, liouv_eigvecs = eigen(liouv_trotter) 
# steady_state_vec = liouv_eigvecs[:, end]
# steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
# steady_state_dm /= tr(steady_state_dm)

# @printf("Steady state closeness to Gibbs for Liouvillian (Energy): %s\n", norm(steady_state_dm - gibbs))

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