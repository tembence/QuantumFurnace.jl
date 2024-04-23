using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
include("jump_op_tools.jl")
include("hamiltonian_tools.jl")

function liouvillian_step(jump::JumpOp, hamiltonian::HamHam, initial_dm::Matrix{ComplexF64}, 
    delta::Float64, sigma::Float64, beta::Float64)



end

function liouvillian_step(jump::JumpOp, hamiltonian::HamHam, initial_dm::Matrix{ComplexF64}, N::Int64, 
    w0_by_hand::Float64, delta::Float64, sigma::Float64, beta::Float64)
    """This version of the function uses the w0_by_hand parameter, but it is not gonna be the ideal w0 that is set
    by the smallest Hamiltonian Bohr frequency"""
    
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = w0_by_hand * N_labels
    energy_bounds = [-0.45 0.45]
    # Truncate all energies that our out of bound
    energy_labels = energy_labels[energy_labels .> energy_bounds[1]]
    energy_labels = energy_labels[energy_labels .< energy_bounds[2]]

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
num_qubits = 8
num_energy_bits = 16
N = 2^num_energy_bits
oft_precision = ceil(Int, abs(log10(N^(-1))))
delta = 0.01
sigma = 5.
bohr_bound = 0.
beta = 1.
eig_index = 2
jump_site_index = 1

#* Hamiltonian
coeffs = fill(1.0, 3)
println("Finding ideal Heisenberg Hamiltonian...")
@time hamiltonian = find_ideal_heisenberg(num_qubits, coeffs, batch_size=1)
initial_state = hamiltonian.eigvecs[:, eig_index]
initial_dm = initial_state * initial_state'
hamiltonian.bohr_freqs = round.(hamiltonian.eigvals .- transpose(hamiltonian.eigvals), digits=oft_precision+3)

#* Jump operators
sigmax::Matrix{ComplexF64} = [0 1; 1 0]
jump_op = Matrix(pad_term([sigmax], num_qubits, jump_site_index))
jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
jump = JumpOp(jump_op,
        jump_op_in_eigenbasis,
        Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
        zeros(0))

find_unique_jump_freqs(jump, hamiltonian)
    
phase = -0.44 * N / (2 * pi)
energy = 2 * pi * phase / N
@printf("\nEnergy: %f\n", energy)

# @printf("Number of unique freqs: %d\n", length(jump.unique_freqs))
# truncated_energies = gaussian_truncate_fourier_labels(jump.unique_freqs, energy_labels, sigma, beta)

#* w0 by hand
w0 = 2 * pi / N
@time evolved_dm = liouvillian_step(jump, hamiltonian, initial_dm, N, w0, delta, sigma, beta)

