using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using QuantumOptics

include("jump_op_tools.jl")
include("hamiltonian_tools.jl")
include("qi_tools.jl")


function liouvillian_delta_trajectory(jump::JumpOp, hamiltonian::HamHam, energy_labels::Vector{Float64},
    initial_dm::Matrix{ComplexF64}, delta::Float64, sigma::Float64, beta::Float64)
    """Computes δ * L[rho_0]"""

    Fw = exp.(- sigma^2 * (energy_labels).^2)
    Fw_norm = sqrt(sum(Fw.^2))
    boltzmann_factor(energy) = min(1, exp(-beta * energy))

    @printf("Length of energy labels in DELTA: %d\n", length(energy_labels))
    evolved_dm = copy(initial_dm)
    for w in energy_labels
        oft_matrix = entry_wise_oft(jump, w, hamiltonian, sigma, beta) / Fw_norm
        oft_matrix_dag = oft_matrix'
        
        evolved_dm += delta * boltzmann_factor(w) *
                    (oft_matrix * initial_dm * oft_matrix_dag
                    - 0.5 * oft_matrix_dag * oft_matrix * initial_dm
                    - 0.5 * initial_dm * oft_matrix_dag * oft_matrix)
        println("TRACE")
        println(tr(evolved_dm))
        evolved_dm /= tr(evolved_dm)
    end
    return evolved_dm
end

function exact_liouvillian_step(jump::JumpOp, hamiltonian::HamHam, initial_dm::Matrix{ComplexF64}, 
    energy_labels::Vector{Float64}, delta::Float64, sigma::Float64, beta::Float64)

    num_qubits = Int(log2(size(hamiltonian.data)[1]))
    b = SpinBasis(1//2)^num_qubits

    initial_dm = Operator(b, initial_dm)
    evolution_hamiltonian = Operator(b, spzeros(ComplexF64, 2^num_qubits, 2^num_qubits))
    Fw = exp.(- sigma^2 * (energy_labels).^2)
    Fw_norm = sqrt(sum(Fw.^2))
    boltzmann_factor(energy) = min(1, exp(-beta * energy))
    # All jumps
    all_jumps = [Operator(b, sqrt(boltzmann_factor(w)) * 
                                entry_wise_oft(jump, w, hamiltonian, sigma, beta) / Fw_norm) for w in energy_labels]
    
    @printf("Length of all jumps in EXACT: %d\n", length(all_jumps))

    tout, evolved_dms = timeevolution.master([0.0, delta], initial_dm, evolution_hamiltonian, all_jumps) 
    # Trace is already = 1
    return evolved_dms[2].data
end

function liouvillian_step(jump::JumpOp, hamiltonian::HamHam, initial_dm::Matrix{ComplexF64}, 
    delta::Float64, sigma::Float64, beta::Float64)

    ideal_num_estimating_bits = ceil(Int64, log2(1 / hamiltonian.w0))
end

#TODO: Probably a depricable function
function liouvillian_step(jump::JumpOp, hamiltonian::HamHam, initial_dm::Matrix{ComplexF64}, 
    energy_labels::Vector{Float64}, delta::Float64, sigma::Float64, beta::Float64)
    """This version of the function uses the w0_by_hand parameter, but it is not gonna be the ideal w0 that is set
    by the smallest Hamiltonian Bohr frequency"""

    # boltzmann_factors = exp.(-beta * energy_labels)
    boltzmann_factor(w) = exp(-beta * w)
    evolved_dm = deepcopy(initial_dm)

    # evolving_dm = @distributed (+) for w = energy_labels
    #     oft_matrix = entry_wise_oft(jump, w, hamiltonian, sigma, beta)
    #     oft_matrix_dag = oft_matrix'
        
    #     delta * boltzmann_factor(w) *
    #                 (oft_matrix * initial_dm * oft_matrix_dag
    #                  - 0.5 * (oft_matrix_dag * oft_matrix * initial_dm
    #                  + initial_dm * oft_matrix_dag * oft_matrix))
    # end
    # evolved_dm += evolving_dm

    for w in energy_labels
        oft_matrix = entry_wise_oft(jump, w, hamiltonian, sigma, beta)
        oft_matrix_dag = oft_matrix'
        
        evolved_dm += delta * boltzmann_factor(w)*
                    (oft_matrix * initial_dm * oft_matrix_dag
                     - 0.5 * (oft_matrix_dag * oft_matrix * initial_dm
                     + initial_dm * oft_matrix_dag * oft_matrix))
    end


#     oft_matrices = entry_wise_oft.((jump,), energy_labels, (hamiltonian,), sigma, beta)
#     oft_matrices_dag = conj.(oft_matrices)
#     @printf("Length of oft_matrices: %d\n", length(oft_matrices))
#     @printf("Length of boltzmann factors: %d\n", length(boltzmann_factors))
#     @printf("Length of dags matrices: %d\n", length(oft_matrices_dag))

#     evolved_dm = sum(boltzmann_factors .* delta .*
#                 (oft_matrices * initial_dm .* oft_matrices_dag
#                     .- 0.5 * (oft_matrices_dag .* oft_matrices * initial_dm
#                     .+ initial_dm * oft_matrices_dag .* oft_matrices)))


    return evolved_dm / tr(evolved_dm)
end

function truncate_energy_labels(energy_labels::Vector{Float64}, sigma::Float64)
    """Finds ideal number of energy labels based on w0"""
    ideal_N = 2^(ceil(Int64, log2(1 / w0)) + 1) #! Added one for needed precision
    N_labels = [0:1:Int(ideal_N/2)-1; -Int(ideal_N/2):1:-1]
    energy_labels = w0 * N_labels
    energy_bounds = [-0.45 0.45]

    # Truncate all energies that our out of bound  #! Uncomment later
    # energy_labels = energy_labels[energy_labels .> energy_bounds[1]]
    # energy_labels = energy_labels[energy_labels .< energy_bounds[2]]

    #TODO: gaussian truncate possibly here

    return energy_labels
end

function gaussian_truncate_fourier_labels(unique_freqs::Vector{Float64}, energy_labels::Vector{Float64}, 
    sigma::Float64, beta::Float64)

    cut_off_eps = 1e-4
    @time begin
        ft_gauss_amplitudes = exp.( - sigma^2 * energy_labels.^2)
        normalization = sum(ft_gauss_amplitudes)
    end
    @printf("Normalization: %f\n", normalization)

    cut_off_dist = sqrt(abs(log(normalization * cut_off_eps)) / sigma^2)
    truncation_radius = Int(ceil(cut_off_dist * length(energy_labels) / (2 * pi)))
    @printf("Truncation radius: %d\n", truncation_radius)
    
    # Truncate the whole energy axis with this radius around the unique frequencies of the jump
    truncated_energy_axis = 0
    return truncated_energy_axis
end


#* ---------------------- TESTING ---------------------- *#
#! 68s (q, r) = (8, 16)

#* Parameters
num_qubits = 4
mixing_time = 20
delta = 0.1
num_liouv_steps = Int(mixing_time / delta)
sigma = 5.
beta = 1.
eig_index = 8
jump_site_index = 1

#* Hamiltonian
hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n4.jld")["ideal_ham"]
# hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3), batch_size=1)
initial_state = hamiltonian.eigvecs[:, eig_index]
initial_dm = initial_state * initial_state'
# initial_dm = zeros(ComplexF64, size(hamiltonian.data))
# initial_dm[eig_index, eig_index] = 1.0  # In eigenbasis
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
display(hamiltonian.bohr_freqs)

#* Jump operators
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]

# jump_op = Matrix(pad_term([X], num_qubits, jump_site_index))
# jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
# jump = JumpOp(jump_op,
#         jump_op_in_eigenbasis,
#         Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
#         zeros(0))

# For full Liouvillian dynamics:
# all_x_jump_ops = []
# for q in 1:num_qubits
#     padded_x = pad_term([jump_op], q, num_qubits)
#     push!(all_x_jump_ops, padded_x)
# end

#* Fourier labels
num_energy_bits = ceil(Int64, log2((0.45 * 2) / hamiltonian.w0)) + 1  # paper (above 3.7.), later will be β dependent
N = 2^(num_energy_bits)
t0 = 2 * pi / (N * hamiltonian.w0)

N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

time_labels = t0 * N_labels
energy_labels = hamiltonian.w0 * N_labels
energy_labels = [energy_labels[10]]
# ft = exp.(-time_labels.^2 / (4 * sigma^2))
# ft_norm = sqrt(sum(ft.^2))
# Fw = exp.(- sigma^2 * (energy_labels).^2)
# Fw_norm = sqrt(sum(Fw.^2))

# isapprox(sqrt(length(time_labels)) * ft_norm, Fw_norm)

@printf("Number of qubits: %d\n", num_qubits)
@printf("Number of energy bits: %d\n", num_energy_bits)
@printf("Energy unit: %e\n", hamiltonian.w0)
@printf("Time unit: %e\n", t0)

#* Exact Liouvillian evolution
gibbs = hamiltonian.eigvecs * gibbs_state(hamiltonian, beta) * hamiltonian.eigvecs'

jump_paulis = [X, Y, Z]
evolved_dm = copy(initial_dm)
Random.seed!(666)
random_site = rand(1:num_qubits)
random_pauli = rand(jump_paulis)

@printf("Random site: %d\n", random_site)
@printf("Random Pauli: %s\n", random_pauli)
jump_op = Matrix(pad_term([random_pauli], num_qubits, random_site))
jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
jump = JumpOp(jump_op,
        jump_op_in_eigenbasis,
        Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
        zeros(0))

evolved_dm_exact = exact_liouvillian_step(jump, hamiltonian, initial_dm, energy_labels, delta, sigma, beta)
evolved_dm_alg = liouvillian_delta_trajectory(jump, hamiltonian, energy_labels, initial_dm, delta, sigma, beta)

println("Eigvals of evolved dm alg")
println(eigvals(evolved_dm_alg))
trace_distance(Hermitian(evolved_dm_alg), Hermitian(gibbs))
# Compare
@printf("Distance to each other: %f\n", norm(Hermitian(evolved_dm_exact) - Hermitian(evolved_dm_alg)))
# printf("Trace distance: %f\n", trace_distance(Hermitian(evolved_dm_exact), Hermitian(evolved_dm_alg)))


#* Many steps and convergence:
# distances_to_gibbs = [trace_distance(Hermitian(evolved_dm), gibbs)]
# fidelities = [fidelity(Hermitian(evolved_dm), gibbs)]
# for step in 1:num_liouv_steps
#     random_site = rand(1:num_qubits)
#     random_pauli = rand(jump_paulis)
#     @printf("Random site: %d\n", random_site)
#     @printf("Random Pauli: %s\n", random_pauli)
#     jump_op = Matrix(pad_term([random_pauli], num_qubits, random_site))
#     jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
#     jump = JumpOp(jump_op,
#             jump_op_in_eigenbasis,
#             Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
#             zeros(0))

#     evolved_dm = exact_liouvillian_step(jump, hamiltonian, evolved_dm, energy_labels, delta, sigma, beta)
#     # dist = trace_distance(Hermitian(evolved_dm), gibbs)
#     # @printf("Distance to Gibbs: %f\n", dist)
#     # push!(distances_to_gibbs, dist)

#     fid = fidelity(Hermitian(evolved_dm), gibbs)
#     @printf("Fidelity to Gibbs: %f\n", fid)
#     push!(fidelities, fid)
# end

# # Plot
# tspan =[0.0:delta:mixing_time;]
# # plot(tspan, distances_to_gibbs, ylims=(0, 1),
# #     label="Trace distance to Gibbs", xlabel="Time", ylabel="Distance", 
# #     title="Liouvillian dynamics")

# # @printf("Final distance to Gibbs: %f\n", distances_to_gibbs[end])

# plot(tspan, fidelities, ylims=(0, 1),
#     label="Fidelity to Gibbs", xlabel="Time", ylabel="Fidelity", 
#     title="Liouvillian dynamics")
# @printf("Final fidelity to Gibbs: %f\n", fidelities[end])

# b = SpinBasis(1//2)^num_qubits

# initial_dm = Operator(b, initial_dm)
# evolution_hamiltonian = Operator(b, spzeros(ComplexF64, 2^num_qubits, 2^num_qubits))

# # All jumps
# Fw = exp.(- sigma^2 * (energy_labels).^2)
# Fw_norm = sqrt(sum(Fw.^2))
# all_jumps = [Operator(b, entry_wise_oft(jump, w, hamiltonian, sigma, beta) / Fw_norm) for w in energy_labels]

# function fout(t, rho)
#     return trace_distance(Hermitian(rho.data / tr(rho.data)), gibbs)
# end

# #! This uses only one jump at one site atm, and the above for loop somehow doesnt work yet.
# tout, distances = timeevolution.master(tspan, initial_dm, evolution_hamiltonian, all_jumps; fout=fout)

# plot(tout, distances, ylims=(0, 1),
#     label="Trace distance to Gibbs", xlabel="Time", ylabel="Distance", 
#     title="Liouvillian dynamics")