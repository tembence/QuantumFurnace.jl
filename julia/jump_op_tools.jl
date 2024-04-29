using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using TensorOperations
using JLD

include("hamiltonian_tools.jl")


mutable struct JumpOp
    data::Matrix{ComplexF64}
    in_eigenbasis::Matrix{ComplexF64}
    bohr_decomp::Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}
    unique_freqs::Vector{Float64}
end

function entry_wise_oft(jump::JumpOp, energy::Float64, hamiltonian::HamHam, sigma::Float64, beta::Float64)
    """Uses the already Fourier transformed Gaussian filter fn."""
    # The for loop version is the same but a bit slower than broadcasting
    # for j in 1:ncols, i in 1:nrows
    #     jump_oft[i, j] = jump_op_in_eigenbasis[i, j] * exp(-(energy - bohr_freqs[i, j])^2 * sigma^2)
    # end
    
    return jump.in_eigenbasis .* exp.(-(energy .- hamiltonian.bohr_freqs).^2 * sigma^2)
end

function explicit_oft(jump::JumpOp, hamiltonian::HamHam, energy::Float64, time_labels::Vector{Float64},
    sigma::Float64, beta::Float64)
    """Fourier transforms by summing over the time labels. Both time_labels and the energy should be in t0, w0 units"""

    # Check if t_0 and w_0 satisfy the Fourier condition
    if !isapprox(t0 * hamiltonian.w0, 2 * pi / length(time_labels))
        error("t0 * w0 != 2 * pi / N")
    end

    time_gaussian_factors = exp.(- time_labels.^2 / (4 * sigma^2))
    normalized_time_gaussian_factors = time_gaussian_factors / sqrt(sum(time_gaussian_factors.^2))
    fourier_phase_factors = exp.(-1im * energy * time_labels) / sqrt(length(time_labels))
    diag_exponentiate(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))

    @time begin
        oft_op = zeros(ComplexF64, size(jump.data))
        @showprogress for t in 1:length(time_labels)
            oft_op += fourier_phase_factors[t] * normalized_time_gaussian_factors[t] * 
            diag_exponentiate(t) * jump.in_eigenbasis * diag_exponentiate(-t)
        end
    end

    return oft_op
end

function oft(jump::JumpOp, energy::Float64, hamiltonian::HamHam, num_energy_bits::Int64, sigma::Float64, beta::Float64)

    #! I have to do this for each energy, maybe it's faster if it's down somewhere higehr level with energy array
    gaussian_weighted_freqs = add_gaussian_weight(jump.unique_freqs, energy, sigma)  

    #! This assumes that we can store all A_nus in a dict
    #! If needed we can rewrite such that we take 1 nu, get A_nu, and add it to the OFT matrix, this way we store much less
    jump_oft = Matrix(sum([gaussian_weighted_freqs[i] * jump.bohr_decomp[jump.unique_freqs[i]] for i in 1:length(jump.unique_freqs)]))

    return jump_oft
end

function find_unique_jump_freqs(jump::JumpOp, hamiltonian::HamHam)
    
    # Nonzero entry indices of jump operators
    jump_indices = findall(!iszero, jump.in_eigenbasis)
    jump_freqs = hamiltonian.bohr_freqs[jump_indices]  # Bohr freqs are already rounded
    jump.unique_freqs = unique(jump_freqs)
end

#TODO: Test this
#TODO: Also there are multiple keys of the value 0... which is no bueno
function construct_A_nus(jump::JumpOp, hamiltonian::HamHam, num_energy_bits::Int64)
"""Constructrs jump.bohr_decomp = Dict {bohr energy (nu): A_nu}"""

    N = 2^num_energy_bits
    oft_precision = ceil(Int, abs(log10(N^(-1))))

    # A'_ij is the entry for the jump between eigenenergies j -> i
    # jump.in_eigenbasis

    # Matrix of all energy jumps in Hamiltonian, B_ij = E_i - E_j and i,j are ordered from smallest to largest energy
    jump.unique_freqs = unique(hamiltonian.bohr_freqs)  #! Not correct

    jump_bohr_indices = get_jump_bohr_indices(hamiltonian.bohr_freqs)
    for (bohr_freq, indices) in jump_bohr_indices
        jump_op_nu = spzeros(ComplexF64, size(jump.data))
        for (i, j) in indices
            jump_op_nu[i, j] = jump.in_eigenbasis[i, j]
        end
        jump.bohr_decomp[bohr_freq] = jump_op_nu
    end
end

function get_jump_bohr_indices(bohr_freqs::Matrix{Float64})

    jump_bohr_indices = Dict{Float64, Vector{Tuple{Int64, Int64}}}()  # Dict {bohr energy (nu): [all contributing (i,j)]}
    nrows, ncols = size(bohr_freqs)
    for j in 1:ncols, i in 1:nrows
        value = bohr_freqs[i, j]
        if haskey(jump_bohr_indices, value)
            push!(jump_bohr_indices[value], (i, j))
        else
            jump_bohr_indices[value] = [(i, j)]
        end
    end
    return jump_bohr_indices
end

function add_gaussian_weight(freqs::Vector{Float64}, energy::Float64, sigma_t::Float64)
    return exp.(-(energy .- freqs).^2 * sigma_t^2)
end

# Unfinished parallel way, but for that would have to store a lot of matrices
# function entry_wise_oft(jump::Tuple{JumpOp}, energies::Vector{Float64}, hamiltonian::Tuple{HamHam}, 
#     sigma::Float64, beta::Float64)

#     return jump[1].in_eigenbasis .* exp.(-(energies .- hamiltonian[1].bohr_freqs).^2 * sigma^2)
# end

#* ---------- Test ----------

#* Parameters
num_qubits = 8
delta = 0.01
sigma = 5.
beta = 1.
eig_index = 8
jump_site_index = 1

#* Hamiltonian
hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n8.jld")["ideal_ham"]
initial_state = hamiltonian.eigvecs[:, eig_index]

#* Jump operators
sigmax::Matrix{ComplexF64} = [0 1; 1 0]
jump_op = Matrix(pad_term([sigmax], num_qubits, jump_site_index))
jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
jump = JumpOp(jump_op,
        jump_op_in_eigenbasis,
        Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
        zeros(0))

# For full Liouvillian dynamics:
# all_x_jump_ops = []
# for q in 1:num_qubits
#     padded_x = pad_term([jump_op], q, num_qubits)
#     push!(all_x_jump_ops, padded_x)
# end

#* Fourier labels
num_energy_bits = ceil(Int64, log2(1 / hamiltonian.w0))
@printf("Number of qubits: %d\n", num_qubits)
@printf("Number of energy bits: %d\n", num_energy_bits)
N = 2^num_energy_bits
t0 = 2 * pi / (N * hamiltonian.w0)
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

time_labels = t0 * N_labels
energy_labels = hamiltonian.w0 * N_labels
normalization_gaussian_energy = sqrt(sum(exp.(- sigma^2 * energy_labels.^2).^2))

# This has always the same form independent of w0, since in the Fourier phase we always have w0t0 = 2pi / N
phase = -0.40 * N / (2 * pi)
energy = 2 * pi * phase / N
@printf("\nEnergy: %f\n", energy)

oft_precision = ceil(Int, abs(log10(N^(-1))))
hamiltonian.bohr_freqs = round.(hamiltonian.eigvals .- transpose(hamiltonian.eigvals), digits=oft_precision+3)

#* -------------------------------------------- *#
oft_expl = explicit_oft(jump, hamiltonian, energy, time_labels, sigma, beta)
oft_entry = entry_wise_oft(jump, energy, hamiltonian, sigma, beta) / (normalization_gaussian_energy * sqrt(N))

@printf("Distance: %f\n", norm(oft_expl - oft_entry))




# if !isapprox(t0 * hamiltonian.w0, 2 * pi / length(time_labels))
#     error("t0 * w0 != 2 * pi / N")
# end

# time_gaussian_factors = exp.(- time_labels.^2 / (4 * sigma^2))
# normalized_time_gaussian_factors = time_gaussian_factors / sqrt(sum(time_gaussian_factors.^2))
# fourier_phase_factors = exp.(-1im * energy * time_labels) / sqrt(length(time_labels))
# # time_evo_phasevector_t = exp.(1im * transpose(Diagonal(hamiltonian.eigvals)) .* time_labels)
# diag_exponentiate(t) = exp(1im * Diagonal(hamiltonian.eigvals) * t)

# prefactors = normalized_time_gaussian_factors .* fourier_phase_factors

# @time begin
#     oft_op = zeros(ComplexF64, size(jump.data))
#     @showprogress for t in 1:length(time_labels)
#         oft_op += fourier_phase_factors[t] * normalized_time_gaussian_factors[t] * 
#         Diagonal(time_evo_phasevector_t[t, :]) * jump.in_eigenbasis * Diagonal(-(time_evo_phasevector_t[t, :]))
#     end
# end
# time_evolutions = diag_exponentiate.(time_labels)
# time_evolutions_dag = diag_exponentiate.(-time_labels)

# Sum over all time labels, t
# @time @tensor oft_op[j, n] := prefactors[t] *
#                 (hamiltonian.eigvecs[j, a] * time_evo_phasevector_t[t, a]) * 
#                 jump.in_eigenbasis[a, b] * 
#                 (adjoint(time_evo_phasevector_t)[t, b] * adjoint(hamiltonian.eigvecs)[b, n])


# @time @tensor begin
#     oft_op[j, n] := normalized_time_gaussian_factors[t] * fourier_phase_factors[t] *
#                 hamiltonian.eigvecs[j, a] * time_evo_phase_for_all_times[t, a] * adjoint(hamiltonian.eigvecs)[a, k] *
#                 jump.data[k, m] *
#                 view(hamiltonian.eigvecs)[m, b] * adjoint(view(time_evo_phase_for_all_times))[t, b] * adjoint(view(hamiltonian.eigvecs))[b, n]
# end


# find_unique_jump_freqs(jump, hamiltonian)


# @time construct_A_nus(jump, hamiltonian, num_energy_bits)

# @time oft_op = oft(jump, energy, hamiltonian, num_energy_bits, sigma, beta)

# @time entry_wise_oft_op = entry_wise_oft(jump, energy, hamiltonian, sigma, beta)

# @printf("Are they the same?: %s\n", norm(oft_op - entry_wise_oft_op) < 1e-10)