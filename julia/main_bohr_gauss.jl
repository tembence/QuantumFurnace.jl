using LinearAlgebra
using Random
using Printf
using ProgressMeter
using JLD

include("hamiltonian.jl")
include("ofts.jl")
include("qi_tools.jl")
include("bohr_picture.jl")

#* Parameters
num_qubits = 3
delta = 0.001
mixing_time = 10.
num_liouv_steps = Int(round(mixing_time / delta, digits=0))
beta = 10.
eta = 0.02  # Just don't make it smaller than 0.018
eig_index = 3
Random.seed!(666)

save_it = false
with_coherent = true

#* Hamiltonian
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=10)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

# initial_dm_OG = Matrix{ComplexF64}(diagm([0.25, 0.75]))

maxmixed = hamiltonian.eigvecs' * I(2^num_qubits) / 2^num_qubits * hamiltonian.eigvecs
maxmixed /= tr(maxmixed)
initial_dm_OG = maxmixed
# random_dm::Matrix{ComplexF64} = random_density_matrix(num_qubits)
# initial_dm_OG = random_dm
# ones_dm = ones(ComplexF64, 2^num_qubits, 2^num_qubits)
# ones /= tr(ones)
# initial_dm = ones

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

#* Thermalize
# results = thermalize_bohr_gauss(all_jumps_generated, hamiltonian, initial_dm_OG, delta, mixing_time, beta)
# plot(results.time_steps, results.distances_to_gibbs, 
#     label="Distance to Gibbs", xlabel="Time", ylabel="Distance", title="Distance to Gibbs over time")

# results.distances_to_gibbs[end]

#* Construct Bohr Liouvillian
@time liouv_matrix = construct_liouvillian_bohr_gauss(all_jumps_generated, hamiltonian, with_coherent, beta)
liouv_eigvals, liouv_eigvecs = eigen(liouv_matrix) 
steady_state_vec = liouv_eigvecs[:, end]
steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
steady_state_dm /= tr(steady_state_dm)
steady_state_vec = vec(steady_state_dm)

liouvillian_evolved_vec = exp(mixing_time * liouv_matrix) * vec(initial_dm_OG)  # This is indeed the Gibbs
liouvillian_evolved_dm = reshape(liouvillian_evolved_vec, size(hamiltonian.data))

#* Difference between perfect Liouvillian evolved dm vs Alg evolved dm
# norm(results.evolved_dm - liouvillian_evolved_dm)
# norm(results.evolved_dm - gibbs)
# norm(liouvillian_evolved_dm - gibbs)

#* Steady state, Gibbs?
norm(gibbs_vec - steady_state_vec)