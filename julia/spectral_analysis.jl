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
include("db_tools.jl")

#* Parameters
num_qubits = 4
mixing_time = 50.
delta = 0.1
num_liouv_steps = Int(mixing_time / delta)
# num_liouv_steps = 3 * num_qubits
# num_liouv_steps = 2
beta = 10.
eig_index = 3
Random.seed!(666)

#* Hamiltonian
hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n4.jld")["ideal_ham"]
initial_dm = zeros(ComplexF64, size(hamiltonian.data))
initial_dm[eig_index, eig_index] = 1.0  # In eigenbasis

hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

#* Gibbs 
b = SpinBasis(1//2)^num_qubits
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
gibbs_largest_eigval = real(eigen(gibbs).values[1])
evolved_dm = copy(initial_dm)
best_evolved_dm = copy(evolved_dm)

#* Fourier labels
#! Coherent terms only become significant if we take r + 1 at least.
num_energy_bits = ceil(Int64, log2((0.45 * 4 + 2/beta) / hamiltonian.w0)) + 2 # Under Fig. 5. with secular approx.
N = 2^(num_energy_bits)
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

t0 = 2 * pi / (N * hamiltonian.w0)
time_labels = t0 * N_labels
energy_labels = hamiltonian.w0 * N_labels
energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]  # Energies outside are impossible in a QC

# OFT normalizations for energy basis
Fw = exp.(- beta^2 * (energy_labels).^2 / 4)
Fw_norm = sqrt(sum(Fw.^2))

# Transition weights in the liouv
transition_gaussian(energy) = exp(-(beta * energy + 1)^2 / 2)
transition_weights = transition_gaussian.(energy_labels)

#* Jump operators
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
jump_paulis = [X, Y, Z]
all_jumps_generated = []

#* X JUMP
# sites = [1, 2, 3]
# for site in sites
# jump_op = Matrix(pad_term([X], num_qubits, site))
#     jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
#     # jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
#     jump = JumpOp(jump_op,
#             jump_op_in_eigenbasis,
#             Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
#             zeros(0),
#             zeros(0, 0)) #jump_in_trotter_basis
#     push!(all_jumps_generated, jump)
# end

# All jumps but only once
for pauli in jump_paulis
    for site in 1:num_qubits
    jump_op = Matrix(pad_term([pauli], num_qubits, site))
    jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
    # jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
    jump = JumpOp(jump_op,
            jump_op_in_eigenbasis,
            Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
            zeros(0),
            zeros(0, 0)) #jump_in_trotter_basis
    push!(all_jumps_generated, jump)
    end
end

# Random jumps
all_random_jumps_generated = []
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

#* Coherent terms
b1_vals, b1_times = compute_truncated_b1(time_labels)
b2_vals, b2_times = compute_truncated_b2(time_labels)
@printf("Number of coherent terms: %d\n", length(b1_vals) * length(b2_vals))
# coherent_terms::Vector{Matrix{ComplexF64}} = coherent_term_from_timedomain.(all_jumps_generated, 
# Ref(hamiltonian), Ref(b1_vals), Ref(b1_times), Ref(b2_vals), Ref(b2_times), Ref(beta))

#* The Press
@printf("Number of qubits: %d\n", num_qubits)
@printf("Number of energy bits: %d\n", num_energy_bits)
@printf("Energy unit: %e\n", hamiltonian.w0)
@printf("Time unit: %e\n", t0)

#* Algorithm
tspan =[0.0:delta:mixing_time;]
p = Progress(length(num_liouv_steps))

evolved_dm = copy(initial_dm)
distances_to_gibbs = [tracedistance_nh(Operator(b, evolved_dm), Operator(b, gibbs))]
@showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps

    # Jump
    jump = all_random_jumps_generated[step]

    # Coherent term
    #FIXME: For the cases when coherent terms become more significant, they slightly drive the fixed point from Gibbs...
    coherent_term = coherent_term_from_timedomain(jump, hamiltonian, b1_vals, b1_times, b2_vals, b2_times, beta)
    # @printf("Trace norm of coherent term: %s\n", tracenorm_nh(Operator(b, coherent_term)))
    evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)

    for (i, w) in enumerate(energy_labels)
        # oft_matrix = explicit_oft_exact_db(jump, hamiltonian, w, time_labels, beta)
        oft_matrix = sqrt(transition_weights[i]) * entry_wise_oft_exact_db(jump, w, hamiltonian, beta) / Fw_norm
        oft_matrix_dag = oft_matrix'
        
        evolved_dm .+= delta * (oft_matrix * evolved_dm * oft_matrix_dag
                    - 0.5 * oft_matrix_dag * oft_matrix * evolved_dm
                    - 0.5 * evolved_dm * oft_matrix_dag * oft_matrix)
    end

    # @printf("Trace: %s\n", tr(evolved_dm))
    # evolved_dm /= tr(evolved_dm)
    
    dist = tracedistance_nh(Operator(b, evolved_dm), Operator(b, gibbs))
    if dist < minimum(distances_to_gibbs)
        best_evolved_dm = copy(evolved_dm)
    end
    # @printf("\nDistance to Gibbs: %f\n", dist)
    push!(distances_to_gibbs, dist)
end

#* Liouv construction
liouv = zeros(ComplexF64, 4^num_qubits, 4^num_qubits)
@showprogress dt=1 desc="Liouvillian..." for step in eachindex(all_jumps_generated)
    # Jump
    jump = all_jumps_generated[step]

    # Coherent term
    coherent_term = coherent_term_from_timedomain(jump, hamiltonian, b1_vals, b1_times, b2_vals, b2_times, beta)
    liouv .+= construct_liouvillian_coherent(coherent_term)

    for (i, w) in enumerate(energy_labels)
        # oft_matrix = explicit_oft_exact_db(jump, hamiltonian, w, time_labels, beta)
        oft_matrix = sqrt(transition_weights[i]) * entry_wise_oft_exact_db(jump, w, hamiltonian, beta) / Fw_norm
        oft_matrix_dag = oft_matrix'

        #* Liouv
        liouv .+= construct_liouvillian_diss([oft_matrix])
    end
end

#* Algorithm analysis
min_dist = minimum(distances_to_gibbs)
max_mixed_dm = Matrix(I, 2^num_qubits, 2^num_qubits) / 2^num_qubits
dist_to_maxmixed = tracedistance_nh(Operator(b, best_evolved_dm), Operator(b, max_mixed_dm))
@printf("Last distance to Gibbs: %s\n", distances_to_gibbs[end])
@printf("Minimum distance to Gibbs: %s\n", min_dist)
@printf("Variance of last 10 distances to Gibbs: %s\n", var(distances_to_gibbs[end-10:end]))
@printf("Distance to max mixed: %s\n", dist_to_maxmixed)

#TODO: Not correct... is it eigenspace or something else?
# Longer times, or shorter deltas dont make a difference for this discrepancy. And it can't be the eigenbasis. Vectorization
# has been compared to the QI package version too.
#* Spectral analysis
eigvals_liouv, eigvecs_liouv = eigen(liouv)
spectral_gap = real(eigvals_liouv[end] - eigvals_liouv[end - 1])

#FIXME: It really seems extracting the SS is in total disagreement with everything else I observe. Is there a mistake here?
stationary_vec = eigvecs_liouv[:, end]
stationary_dm = reshape(stationary_vec, 2^num_qubits, 2^num_qubits)
gibbs_vec = vec(gibbs)

# dist_ss_gibbs = tracedistance_nh(Operator(b, stationary_dm), Operator(b, gibbs))
dist_evolved_dm_gibbs_vec = norm(vec(best_evolved_dm) - gibbs_vec)
dist_ss_gibbs_vec = norm(stationary_vec - gibbs_vec)
dist_ss_evolved_dm_vec = norm(stationary_vec - vec(best_evolved_dm))
@printf("Distance evolved DM to Gibbs vec: %f\n", dist_evolved_dm_gibbs_vec)
@printf("Distance SS to Gibbs: %f\n", dist_ss_gibbs_vec)
@printf("Distance evolved DM to SS: %f\n", dist_ss_evolved_dm_vec)

rand_dm = random_density_matrix(num_qubits)
rand_dm_vec = vec(rand_dm)

#* Is Gibbs a steady state?
is_it_zero = liouv * gibbs_vec
@printf("Is Gibbs a steady state? L(π) = %f\n", norm(is_it_zero))
is_evolved_zero = liouv * vec(best_evolved_dm)
@printf("Is evolved a steady state? L(ρ) = %f\n", norm(is_evolved_zero))
is_random_zero = liouv * rand_dm_vec
@printf("Is random a steady state? L(r) = %f\n", norm(is_random_zero))
# diff temp gibbs
# betas = range(9.5, 10.5, length=10)
# gibbs_states_diff_temp = [vec(gibbs_state_in_eigen(hamiltonian, beta)) for beta in betas]
# are_they_zero = [norm(liouv * gibbs) for gibbs in gibbs_states_diff_temp]
# index_of_smallest = argmin(are_they_zero)
# @printf("Temp thats closest to steady state: %f\n", betas[index_of_smallest])


# norm_liouv_check = norm((I(num_qubits^4) + delta * liouv)^(num_liouv_steps / 12) * vec(initial_dm) - vec(best_evolved_dm))
# norm_liouv_check = norm(exp(mixing_time * liouv) * vec(initial_dm) - vec(best_evolved_dm))

#* Mixing time
# mix_time_at_least = log(2 / sqrt(gibbs_largest_eigval)) / spectral_gap