using LinearAlgebra
using Random
using Printf
using ProgressMeter
using QuantumOptics
using JLD

include("hamiltonian_tools.jl")
include("jump_op_tools.jl")
include("trotter.jl")
include("qi_tools.jl")
include("thermalizing_tools.jl")
include("coherent.jl")
include("spectral_analysis_tools.jl")

#TODO: memory map parallelization

@enum FurnaceType GAUSS = 1 METRO = 2 TROTT_GAUSS = 3 TROTT_METRO = 4 TIME_GAUSS = 5

#* Parameters
num_qubits = 3
mixing_time = 5.
delta = 0.05
num_liouv_steps = Int(round(mixing_time / delta, digits=0))
beta = 10.
eta = 0.02  # Just don't make it smaller than 0.018
eig_index = 3
Random.seed!(666)
save_it = false

furnace = GAUSS
with_coherent = true  #TODO: Check how close B is between Trotter and ideal

#* Hamiltonian
ham_filename(n) = @sprintf("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n%d.jld", n)
hamiltonian = load(ham_filename(num_qubits))["ideal_ham"]
# hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n3.jld")["ideal_ham"]
initial_dm = zeros(ComplexF64, size(hamiltonian.data))
initial_dm[eig_index, eig_index] = 1.0  # In eigenbasis
initial_dm[eig_index + 1, eig_index + 1] = 1.0
initial_dm /= tr(initial_dm)

hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

#* Gibbs 
b = SpinBasis(1//2)^num_qubits
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
gibbs_largest_eigval = real(eigen(gibbs).values[1])

#* Fourier labels
# Coherent terms only become significant if we take r + 1 at least.
num_energy_bits = ceil(Int64, log2((0.45 * 4 + 2/beta) / hamiltonian.w0)) + 2 # Under Fig. 5. with secular approx.

# Transition weights in the liouv // Jump rate squared
if furnace == GAUSS || furnace == TROTT_GAUSS || furnace == TIME_GAUSS
    transition(energy) = exp(-(beta * energy + 1)^2 / 2)  # Calling this again and again is as fast as storing it.
elseif furnace == METRO || furnace == TROTT_METRO
    transition(energy) = exp(-beta * maximum([energy + 1/(2*beta), 0]))  # Can't really be truncated
else
    println("Furnace type hasn't been chosen")
end

#* Trotter
if furnace == TROTT_GAUSS || furnace == TROTT_METRO 
    filter_gauss_t(t) = exp(- t^2 / beta^2)
    t0 = 2 * pi / (hamiltonian.w0 * 2^num_energy_bits)
    num_trotter_steps_per_t0 = 1000
    trotter = create_trotter(hamiltonian, t0, num_trotter_steps_per_t0)
    trotter_error_T = compute_trotter_error(hamiltonian, trotter, 2^num_energy_bits * t0)
    gibbs = trotter.eigvecs' * gibbs_state(hamiltonian, beta) * trotter.eigvecs
    @printf("Num trotter steps / t0: %d\n", num_trotter_steps_per_t0)
    @printf("Max order Trotter error on an OFT: %s\n", trotter_error_T)
elseif furnace == GAUSS || furnace == METRO
    # OFT normalizations for energy basis
    filter_gauss_w(energy) = exp.(- beta^2 * (energy).^2 / 4)
elseif furnace == TIME_GAUSS
    filter_gauss_t(t) = exp(- t^2 / beta^2)
end

#* Jump operators
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
jump_paulis = [[X], [Y], [Z]]

# All jumps once
all_jumps_generated::Vector{JumpOp} = []
for pauli in jump_paulis
    for site in 1:num_qubits
    jump_op = Matrix(pad_term(pauli, num_qubits, site))
    jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
    if furnace == TROTT_GAUSS || furnace == TROTT_METRO
        jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
    else
        jump_in_trotter_basis = zeros(0, 0)
    end
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

# Random jumps
# all_random_jumps_generated::Vector{JumpOp} = []
# for _ in 1:(num_liouv_steps * 3 * num_qubits + 1)
#     random_site = rand(1:num_qubits)
#     random_pauli = rand(jump_paulis)
#     jump_op = Matrix(pad_term(random_pauli, num_qubits, random_site))
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
@printf("Mixing time: %s\n", mixing_time)
@printf("Delta: %s\n", delta)

#* ////////////////////////// Algorithm //////////////////////////
if furnace == GAUSS
    therm_results = thermalize_gaussian(all_jumps_generated, hamiltonian, with_coherent, initial_dm, num_energy_bits,
    filter_gauss_w, transition, delta, mixing_time, beta)
elseif furnace == METRO
    therm_results = thermalize_metro(all_jumps_generated, hamiltonian, with_coherent, initial_dm,
    num_energy_bits, filter_gauss_w, transition, eta, delta, mixing_time, beta)
elseif furnace == TROTT_GAUSS
    therm_results = thermalize_gaussian_trotter(all_jumps_generated, hamiltonian, trotter, with_coherent, initial_dm,
    num_energy_bits, filter_gauss_t, transition, delta, mixing_time, beta)
elseif furnace == TROTT_METRO
    therm_results = thermalize_metro_trotter(all_jumps_generated, hamiltonian, trotter, with_coherent, initial_dm,
    num_energy_bits, filter_gauss_t, transition, eta, delta, mixing_time, beta)
elseif furnace == TIME_GAUSS
    therm_results = thermalize_gaussian_ideal_time(all_jumps_generated, hamiltonian, with_coherent, initial_dm, num_energy_bits,
    filter_gauss_t, transition, delta, mixing_time, beta)
end

evolved_dm = therm_results.evolved_dm
distances_to_gibbs = therm_results.distances_to_gibbs

#* ////////////////////////// Liouv construction //////////////////////////
if furnace == GAUSS
    liouv_matrix = construct_liouvillian_gauss(all_jumps_generated, hamiltonian, with_coherent, num_energy_bits, 
    filter_gauss_w, transition, beta)
elseif furnace == METRO
    liouv_matrix = construct_liouvillian_metro(all_jumps_generated, hamiltonian, with_coherent, num_energy_bits, 
    filter_gauss_w, transition, eta, beta)
elseif furnace == TROTT_GAUSS
    liouv_matrix = construct_liouvillian_gauss_trotter(all_jumps_generated, hamiltonian, trotter, with_coherent, num_energy_bits, 
    filter_gauss_t, transition, beta)
elseif furnace == TROTT_METRO
    liouv_matrix = construct_liouvillian_metro_trotter(all_jumps_generated, hamiltonian, trotter, with_coherent, num_energy_bits, 
    filter_gauss_t, transition, eta, beta)
elseif furnace == TIME_GAUSS
    liouv_matrix = construct_liouvillian_gauss_ideal_time(all_jumps_generated, hamiltonian, with_coherent, num_energy_bits, 
    filter_gauss_t, transition, beta)
end

liouv = LiouvLiouv(liouv_matrix, zeros(0, 0), 0.0, 0.0)
trdist_eps = 1e-3
mixing_time_bound(liouv, trdist_eps)
@printf("spectral gap Liouv: %s\n", liouv.spectral_gap)
@printf("Mixing time bound: %s\n", liouv.mixing_time_bound)

#* Spectral analysis
# Stationary DM / vectorized
stationary_dm = liouv.steady_state
stationary_vec = vec(stationary_dm)

# # Gibbs and evolved vectorized
gibbs_vec = vec(gibbs)
evolved_vec = vec(evolved_dm)

# rand_dm = random_density_matrix(num_qubits)
# rand_dm_vec = vec(rand_dm)

# HS distances
dist_evolved_dm_gibbs_vec = norm(evolved_vec - gibbs_vec)
dist_ss_gibbs_vec = norm(stationary_vec - gibbs_vec)
dist_ss_evolved_dm_vec = norm(stationary_vec - evolved_vec)  

println("\nHS distances:")
@printf("HS norm evolved DM to Gibbs vec: %s\n", dist_evolved_dm_gibbs_vec)
@printf("--HS norm SS to Gibbs: %s\n", dist_ss_gibbs_vec)
@printf("HS norm evolved DM to SS: %s\n", dist_ss_evolved_dm_vec)

# Trace distances
println("\nTrace distances:")
min_dist = minimum(distances_to_gibbs)
# max_mixed_dm = Matrix(I, 2^num_qubits, 2^num_qubits) / 2^num_qubits
# dist_to_maxmixed = tracedistance_nh(Operator(b, best_evolved_dm), Operator(b, max_mixed_dm))
trdist_ss_evolved_dm = tracedistance_nh(Operator(b, evolved_dm), Operator(b, stationary_dm))

# trdist_ss_rand = tracedistance_nh(Operator(b, rand_dm), Operator(b, stationary_dm))
@printf("Last distance to Gibbs: %s\n", distances_to_gibbs[end])
@printf("Minimum distance to Gibbs: %s\n", min_dist)
# @printf("Variance of last 10 distances to Gibbs: %s\n", var(distances_to_gibbs[end-10:end]))
# @printf("Distance to max mixed: %s\n", dist_to_maxmixed)
@printf("Trace distance evolved DM to SS: %s\n", trdist_ss_evolved_dm)
# @printf("Trace distance random to SS: %s\n", trdist_ss_rand)


trdist_ss_gibbs = tracedistance_nh(Operator(b, gibbs), Operator(b, stationary_dm))  # Is trdist small to Gibbs?
@printf("Trace distance Gibbs to SS: %s\n", trdist_ss_gibbs)
println("\nStationarity checks:")
is_it_zero = liouv.data * gibbs_vec # Is Gibbs a steady state?
@printf("Is Gibbs a steady state? L(π) = %s\n", norm(is_it_zero))

is_evolved_zero = liouv.data * evolved_vec
@printf("Is evolved a steady state? L(ρ) = %s\n", norm(is_evolved_zero))
# is_random_zero = liouv * rand_dm_vec
# @printf("Is random a steady state? L(r) = %s\n", norm(is_random_zero))

if save_it == true
    if with_coherent == false
        filename(furnace, n, r) = @sprintf("/Users/bence/code/liouvillian_metro/julia/data/%s_n%d_r%d.jld", furnace, n, r)
    else
        filename(furnace, n, r) = @sprintf("/Users/bence/code/liouvillian_metro/julia/data/%s_n%d_r%d_B_100.jld", furnace, n, r)
    end
    # Save objects with jld
    save(filename(furnace, num_qubits, num_energy_bits), "alg_results", therm_results, "liouv", liouv)
end
# lll = load(filename(furnace, num_qubits, num_energy_bits))["liouv"]
# rrr = load(filename(furnace, num_qubits, num_energy_bits))["alg_results"]

plot!(therm_results.time_steps, therm_results.distances_to_gibbs, label="Distance to Gibbs", xlabel="Time", ylabel="Distance", title="Distance to Gibbs over time")
