using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using QuantumOptics
using BenchmarkTools

include("hamiltonian_tools.jl")
include("jump_op_tools.jl")
include("trotter.jl")
include("qi_tools.jl")
include("liouvillian_tools.jl")
include("coherent.jl")

#* Parameters
num_qubits = 5
mixing_time = 30.0
delta = 0.1
num_liouv_steps = Int(mixing_time / delta)
sigma = 5.
beta = 1.
eig_index = 8

#* Hamiltonian
hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n5.jld")["ideal_ham"]
initial_dm = zeros(ComplexF64, size(hamiltonian.data))
initial_dm[eig_index, eig_index] = 1.0  # In eigenbasis
# Random.seed!(666)
# initial_dm = random_density_matrix(num_qubits)

hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

#* Jump operators
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
jump_paulis = [X, Y, Z]

#* Fourier labels
# num_energy_bits = ceil(Int64, log2((0.45 * 2) / hamiltonian.w0)) + 1 # paper (above 3.7.), later will be β dependent
# num_energy_bits = 9

num_energy_bits = ceil(Int64, log2((0.45 * 4 + 2/beta) / hamiltonian.w0))# Under Fig. 5. with secular approx.
N = 2^(num_energy_bits)
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

t0 = 2 * pi / (N * hamiltonian.w0)
time_labels = t0 * N_labels
energy_labels = hamiltonian.w0 * N_labels

@printf("Number of qubits: %d\n", num_qubits)
@printf("Number of energy bits: %d\n", num_energy_bits)
@printf("Energy unit: %e\n", hamiltonian.w0)
@printf("Time unit: %e\n", t0)

#* Gibbs
b = SpinBasis(1//2)^num_qubits
gibbs = gibbs_state(hamiltonian, beta)
evolved_dm = copy(initial_dm)
distances_to_gibbs = [tracedistance_nh(Operator(b, evolved_dm), Operator(b, gibbs))]

#* Pregenerate all random jumps
all_random_jumps_generated = []
Random.seed!(666)
for _ in 1:num_liouv_steps
    random_site = rand(1:num_qubits)
    random_pauli = rand(jump_paulis)
    jump_op = Matrix(pad_term([random_pauli], num_qubits, random_site))
    jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
    # jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
    jump = JumpOp(jump_op,
            jump_op_in_eigenbasis,
            Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
            zeros(0),
            zeros(0, 0)) #jump_in_trotter_basis)
    push!(all_random_jumps_generated, jump)
end

#* Coherent term Metropolis
b1_vals, b1_times = compute_truncated_b1(time_labels)
# Smallest possible eta, to approximate B_metropolis the best
eta = exp(-8 * sqrt(2 * pi) / 5)
# eta = 1. # Ran into same local stationary state, so same distance
b2_vals_metro, b2_times_metro = compute_truncated_b2_metro(time_labels, eta)

coherent_terms_metro::Vector{Matrix{ComplexF64}} = coherent_term_timedomain.(all_random_jumps_generated,
Ref(hamiltonian), Ref(b1_vals), Ref(b1_times), Ref(b2_vals_metro), Ref(b2_times_metro), Ref(beta))

tspan =[0.0:delta:mixing_time;]

#* Alg exact DB
evolved_dm = copy(initial_dm)
p = Progress(length(num_liouv_steps))
@showprogress dt=1 desc="Algorithmic converging to Gibbs..." for delta_step in 1:num_liouv_steps
    # Random jump
    jump_delta = all_random_jumps_generated[delta_step]
    # Corresponding coherent term
    coherent_delta = coherent_terms_metro[delta_step]

    # Evolve by delta time steps
    evolved_dm += liouvillian_delta_trajectory_metro_exact_db(jump_delta, hamiltonian, coherent_delta,
    energy_labels, evolved_dm, delta, beta)
    evolved_dm /= tr(evolved_dm)
    dist = tracedistance_nh(Operator(b, evolved_dm), Operator(b, gibbs))
    @printf("Distance to Gibbs: %f\n", dist)
    # next!(p, showvalues = [(:dist, dist)])
    push!(distances_to_gibbs, dist)
    # if dist < 0.05
    #     @printf("Converged to Gibbs state at step %d\n", delta_step)
    #     break
    # end
end

#* Alg approx DB
# evolved_dm = copy(initial_dm)
# p = Progress(length(num_liouv_steps))
# @showprogress dt=1 desc="Algorithmic converging to Gibbs..." for delta_step in 1:num_liouv_steps
#     # Random jump
#     jump_delta = all_random_jumps_generated[delta_step]

#     # Evolve by delta time steps
#     evolved_dm += liouvillian_delta_trajectory(jump_delta, hamiltonian, energy_labels, evolved_dm, delta, sigma, beta)
#     evolved_dm /= tr(evolved_dm)
#     dist = tracedistance_nh(Operator(b, evolved_dm), Operator(b, gibbs))
#     @printf("Distance to Gibbs: %f\n", dist)
#     # next!(p, showvalues = [(:dist, dist)])
#     push!(distances_to_gibbs, dist)
# end