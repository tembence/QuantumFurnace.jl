using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using QuantumOptics
using BenchmarkTools
using Roots

include("hamiltonian_tools.jl")
include("jump_op_tools.jl")
include("trotter.jl")
include("qi_tools.jl")
include("liouvillian_tools.jl")
include("coherent.jl")
include("db_tools.jl")
#TODO: Plot 1 step durations with system size (and correspinding ideal energy bits or less...) to see how much time it takes later.


#* Parameters
num_qubits = 7
mixing_time = 1.
delta = 1.
num_liouv_steps = Int(mixing_time / delta)
# num_liouv_steps = 3 * num_qubits
# num_liouv_steps = 2
beta = 10.
eig_index = 3
Random.seed!(666)
with_coherent = false

#* Hamiltonian
hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n7.jld")["ideal_ham"]
initial_dm = zeros(ComplexF64, size(hamiltonian.data))
initial_dm[eig_index, eig_index] = 1.0  # In eigenbasis
initial_dm[eig_index + 1, eig_index + 1] = 1.0
initial_dm /= tr(initial_dm)
tr(initial_dm)

hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

#* Gibbs 
b = SpinBasis(1//2)^num_qubits
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
gibbs_largest_eigval = real(eigen(gibbs).values[1])
evolved_dm = initial_dm
best_evolved_dm = evolved_dm

#* Fourier labels
# Coherent terms only become significant if we take r + 1 at least.
num_energy_bits = ceil(Int64, log2((0.45 * 4 + 2/beta) / hamiltonian.w0)) + 3 # Under Fig. 5. with secular approx.
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

# Truncating energy labels based on the Gaussian transition function: 
transition_cutoff = 1e-4  #!
energy_cutoff = find_zero(x -> transition_gaussian(x) - transition_cutoff, 0)
energy_labels = energy_labels[energy_labels .<= energy_cutoff]


transition_weights = transition_gaussian.(energy_labels)

#TODO: Do we need all 3 jumps, maybe 2 is good? 2-site jumps, still Hermitians? XX YY ZZ?
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

#! Check if we can get faster converge with same number but random jumps.
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
# all_random_jumps_generated = []
# for _ in 1:num_liouv_steps
#     random_site = rand(1:num_qubits)
#     random_pauli = rand(jump_paulis)
#     jump_op = Matrix(pad_term([random_pauli], num_qubits, random_site))
#     jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
#     # jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
#     jump = JumpOp(jump_op,
#             jump_op_in_eigenbasis,
#             Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
#             zeros(0),
#             zeros(0, 0)) #jump_in_trotter_basis)
#     push!(all_random_jumps_generated, jump)
# end

#* The Press
@printf("Number of qubits: %d\n", num_qubits)
@printf("Number of energy bits: %d\n", num_energy_bits)
@printf("Energy unit: %e\n", hamiltonian.w0)
@printf("Time unit: %e\n", t0)
@printf("Mixing time: %s\n", mixing_time)
@printf("Delta: %s\n", delta)

#* Coherent terms
if with_coherent
    atol = 1e-12
    b1 = compute_truncated_b1(time_labels, atol)
    b2 = compute_truncated_b2(time_labels, atol)
    @printf("Number of b1 terms: %d\n", length(keys(b1)))
    @printf("Number of b2 terms: %d\n", length(keys(b2)))
else
    @printf("Not adding coherent terms! \n")
end

#* Algorithm
p = Progress(length(num_liouv_steps))
t_coh_total = 0.0
t_diss_total = 0.0

# trotterized_liouv_evol = zeros(ComplexF64, 4^num_qubits, 4^num_qubits)
distances_to_gibbs = [tracedistance_nh(Operator(b, evolved_dm), Operator(b, gibbs))]
min_distance = distances_to_gibbs[1]
@showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps

    # jump = all_random_jumps_generated[step]
    # Each jump once with delta
    for jump in all_jumps_generated
        # Coherent term
        t_coh = @elapsed begin
            if with_coherent
                coherent_term = coherent_term_from_timedomain(jump, hamiltonian, b1, b2, beta)
                # coherent_term = coherent_term_timedomain_integrated(jump, hamiltonian, beta)
                # @printf("Hermitian B? %s\n", norm(coherent_term - coherent_term'))
                # @printf("Trace norm of coherent term: %s\n", tracenorm_nh(Operator(b, coherent_term)))
                evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)
            end
        end

        if with_coherent && step == 1 && jump == all_jumps_generated[1]
            check_B(coherent_term, jump, hamiltonian, beta)
        end

        t_diss = @elapsed begin
            dissipative_dm_part = zeros(ComplexF64, size(initial_dm))
            for (i, w) in enumerate(energy_labels)
                # oft_matrix = explicit_oft_exact_db(jump, hamiltonian, w, time_labels, beta)
                oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
                oft_matrix_dag = oft_matrix'
                oft_dag_oft = oft_matrix_dag * oft_matrix
                
                dissipative_dm_part .+= transition_weights[i] * (oft_matrix * evolved_dm * oft_matrix_dag
                                    - 0.5 * oft_dag_oft * evolved_dm
                                    - 0.5 * evolved_dm * oft_dag_oft) 
            end
            evolved_dm .+= delta * dissipative_dm_part / Fw_norm^2
        end 
        # @printf("Trace: %s\n", tr(evolved_dm))
        # evolved_dm /= tr(evolved_dm)
        
        dist = tracedistance_nh(Operator(b, evolved_dm), Operator(b, gibbs))
        if dist < min_distance
            min_distance = dist
            best_evolved_dm = evolved_dm
        end
        # @printf("\nDistance to Gibbs: %f\n", dist)
        push!(distances_to_gibbs, dist)

        t_coh_total += t_coh
        t_diss_total += t_diss
    end
end

println("Time for coherent terms: ", t_coh_total)
println("Time for dissipative terms: ", t_diss_total)
println("----Total time: ", t_coh_total + t_diss_total)

#* Liouv construction
# liouv = zeros(ComplexF64, 4^num_qubits, 4^num_qubits)
# @showprogress dt=1 desc="Liouvillian..." for step in eachindex(all_jumps_generated)
#     # Jump
#     jump = all_jumps_generated[step]

#     # Coherent term
#     if with_coherent
#         coherent_term = coherent_term_from_timedomain(jump, hamiltonian, b1, b2, beta)
#         # coherent_term = coherent_term_timedomain_integrated(jump, hamiltonian, beta)
#         liouv .+= construct_liouvillian_coherent(coherent_term)
#     end

#     for (i, w) in enumerate(energy_labels)
#         # oft_matrix = explicit_oft_exact_db(jump, hamiltonian, w, time_labels, beta)
#         oft_matrix = sqrt(transition_weights[i]) * entry_wise_oft_exact_db(jump, w, hamiltonian, beta) / Fw_norm
#         oft_matrix_dag = oft_matrix'

#         #* Liouv
#         liouv .+= construct_liouvillian_diss([oft_matrix])
#     end
# end

#* Spectral analysis
# eigvals_liouv, eigvecs_liouv = eigen(liouv)
# spectral_gap = real(eigvals_liouv[end] - eigvals_liouv[end - 1])
# # @printf("eigvals of L:\n")
# # display(eigvals_liouv[end - 5:end])

# # Stationary DM / vectorized
# stationary_vec = eigvecs_liouv[:, end]
# stationary_dm = reshape(stationary_vec, 2^num_qubits, 2^num_qubits)
# stationary_dm /= tr(stationary_dm)
# stationary_vec = vec(stationary_dm)

# # Gibbs and evolved vectorized
gibbs_vec = vec(gibbs)
best_evolved_vec = vec(best_evolved_dm)
evolved_vec = vec(evolved_dm)

# rand_dm = random_density_matrix(num_qubits)
# rand_dm_vec = vec(rand_dm)

# HS distances
dist_evolved_dm_gibbs_vec = norm(evolved_vec - gibbs_vec)
dist_best_evolved_dm_gibbs_vec = norm(best_evolved_vec - gibbs_vec)
# dist_ss_gibbs_vec = norm(stationary_vec - gibbs_vec)
# dist_ss_evolved_dm_vec = norm(stationary_vec - evolved_vec)
# dist_ss_best_evolved_dm_vec = norm(stationary_vec - best_evolved_vec)   

println(" //////// HS distances: ////////")
@printf("HS norm evolved DM to Gibbs vec: %s\n", dist_evolved_dm_gibbs_vec)
@printf("HS norm best evolved DM to Gibbs vec: %s\n", dist_best_evolved_dm_gibbs_vec)
# @printf("--HS norm SS to Gibbs: %s\n", dist_ss_gibbs_vec)
# @printf("HS norm evolved DM to SS: %s\n", dist_ss_evolved_dm_vec)
# @printf("HS norm best evolved DM to SS: %s\n", dist_ss_best_evolved_dm_vec)

# Trace distances
println(" //////// Trace distances: ////////")
min_dist = minimum(distances_to_gibbs)
# max_mixed_dm = Matrix(I, 2^num_qubits, 2^num_qubits) / 2^num_qubits
# dist_to_maxmixed = tracedistance_nh(Operator(b, best_evolved_dm), Operator(b, max_mixed_dm))
# trdist_ss_evolved_dm = tracedistance_nh(Operator(b, evolved_dm), Operator(b, stationary_dm))

# trdist_ss_rand = tracedistance_nh(Operator(b, rand_dm), Operator(b, stationary_dm))
@printf("Last distance to Gibbs: %s\n", distances_to_gibbs[end])
@printf("Minimum distance to Gibbs: %s\n", min_dist)
# @printf("Variance of last 10 distances to Gibbs: %s\n", var(distances_to_gibbs[end-10:end]))
# @printf("Distance to max mixed: %s\n", dist_to_maxmixed)
# @printf("Trace distance evolved DM to SS: %s\n", trdist_ss_evolved_dm)
# @printf("Trace distance random to SS: %s\n", trdist_ss_rand)

# println("How good is our SS?:")
# trdist_ss_gibbs = tracedistance_nh(Operator(b, gibbs), Operator(b, stationary_dm))  # Is trdist small to Gibbs?
# @printf("Trace distance Gibbs to SS: %s\n", trdist_ss_gibbs)
# is_it_zero = liouv * gibbs_vec # Is Gibbs a steady state?
# @printf("Is Gibbs a steady state? L(π) = %s\n", norm(is_it_zero))

# is_evolved_zero = liouv * evolved_vec
# @printf("Is evolved a steady state? L(ρ) = %s\n", norm(is_evolved_zero))
# is_random_zero = liouv * rand_dm_vec
# @printf("Is random a steady state? L(r) = %s\n", norm(is_random_zero))
 

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

# Norms
# rand_matrix1 = random_density_matrix(num_qubits)
# rand_matrix2 = random_density_matrix(num_qubits)
# # Trace norms
# dist_rand1_rand2 = tracedistance_nh(Operator(b, rand_matrix1), Operator(b, rand_matrix2))
# my_trdist = 0.5*tr(sqrt((rand_matrix1 - rand_matrix2)' * (rand_matrix1 - rand_matrix2)))
# # Frobeniuses
# my_frobenius = sqrt(tr((rand_matrix1 - rand_matrix2)' * (rand_matrix1 - rand_matrix2)))
# frob_jl = norm(rand_matrix1 - rand_matrix2, 2)

# Frobenius norm = Euclidean norm of vectorized matrices
# norm(rand_matrix1 - rand_matrix2, 2)
# norm(vec(rand_matrix1) - vec(rand_matrix2), 2)
# norm(stationary_dm - gibbs, 2)
# norm(stationary_vec - gibbs_vec, 2)

