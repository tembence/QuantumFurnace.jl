using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using TensorOperations
using JLD
using Plots

include("hamiltonian.jl")
include("qi_tools.jl")

mutable struct JumpOp
    data::Matrix{ComplexF64}
    in_eigenbasis::Matrix{ComplexF64}
    bohr_decomp::Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}
    unique_freqs::Vector{Float64}
end

function explicit_oft(jump::JumpOp, hamiltonian::HamHam, energy::Float64, time_labels::Vector{Float64},
    sigma::Float64, beta::Float64)

    oft_op = zeros(ComplexF64, size(jump.data))
    @showprogress "Explicit OFT..." for t in time_labels
        oft_op += exp(-1im * energy * t) *
                  exp(-t^2 / (4 * sigma^2)) *
                  Diagonal(exp.(1im * hamiltonian.eigvals * t)) *
                  jump.in_eigenbasis *
                  Diagonal(exp.(-1im * hamiltonian.eigvals * t))
    end

    return oft_op

end

function entry_wise_oft(jump::JumpOp, energy::Float64, hamiltonian::HamHam, sigma::Float64, beta::Float64)
    return jump.in_eigenbasis .* exp.(-((energy .- hamiltonian.bohr_freqs)).^2 * sigma^2)
end

function bohr_decomp_oft(jump::JumpOp, energy::Float64, hamiltonian::HamHam, 
    num_energy_bits::Int64, sigma::Float64, beta::Float64)

    jump_oft = zeros(ComplexF64, size(jump.data))
    for (bohr_freq, jump_op_nu) in jump.bohr_decomp
        jump_oft += jump_op_nu * exp(-((energy - bohr_freq)^2) * sigma^2)
    end

    return jump_oft
end

function construct_A_nus(jump::JumpOp, hamiltonian::HamHam)
    """Constructrs jump.bohr_decomp = Dict {bohr energy (nu): A_nu}"""

    jump_bohr_indices_dict = get_jump_bohr_indices(hamiltonian.bohr_freqs)
    jump.unique_freqs = collect(keys(jump_bohr_indices_dict))

    for (bohr_freq, indices) in jump_bohr_indices_dict
        jump_op_nu = spzeros(ComplexF64, size(jump.data))
        for (i, j) in indices
            jump_op_nu[i, j] = jump.in_eigenbasis[i, j]
        end
        jump.bohr_decomp[bohr_freq] = jump_op_nu
    end
end

function get_jump_bohr_indices(bohr_freqs::Matrix{Float64})
    """Return: Dict {bohr energy (nu): [all contributing (i,j)]}"""

    jump_nnz_indices = findall(!isapprox(0.0, atol=1e-15), jump.in_eigenbasis) # Some entries are truncated
    jump_bohr_indices_dict = Dict{Float64, Vector{Tuple{Int64, Int64}}}()
    for index in jump_nnz_indices
        bohr_freq = bohr_freqs[index]
        if haskey(jump_bohr_indices_dict, bohr_freq)
            push!(jump_bohr_indices_dict[bohr_freq], index)
        else
            jump_bohr_indices_dict[bohr_freq] = [index]
        end
    end

    return jump_bohr_indices_dict
end

#* Parameters
num_qubits = 5
delta = 0.01
sigma = 5.
beta = 1.
eig_index = 8
jump_site_index = 1

#* Hamiltonian
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3), batch_size=1)
initial_state = hamiltonian.eigvecs[:, eig_index]
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

#* Jump operators
sigmax::Matrix{ComplexF64} = [0 1; 1 0]
jump_op = Matrix(pad_term([sigmax], num_qubits, jump_site_index))
jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
max_index = argmax(abs.(jump_op_in_eigenbasis))
MS_jump_bohr_freq = hamiltonian.bohr_freqs[max_index]

jump = JumpOp(jump_op,
        jump_op_in_eigenbasis,
        Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
        zeros(0))
construct_A_nus(jump, hamiltonian)

#* Fourier labels
num_energy_bits = ceil(Int64, log2((0.45 * 2) / hamiltonian.nu_min)) + 1
N = 2^(num_energy_bits)
t0 = 2 * pi / (N * hamiltonian.nu_min)

N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
time_labels = t0 * N_labels
energy_labels = hamiltonian.nu_min * N_labels
some_integer = 90
an_energy = hamiltonian.nu_min * some_integer
phase = an_energy * N / (2 * pi)
a_time = (N / 2 + 3) * t0

@printf("Number of qubits: %d\n", num_qubits)
@printf("Number of energy bits: %d\n", num_energy_bits)
@printf("Energy unit: %e\n", hamiltonian.nu_min)
@printf("Time unit: %e\n", t0)
@printf("\nEnergy: %f\n", an_energy)

normalization_gaussian_energy = sqrt(sum(exp.(- sigma^2 * energy_labels.^2).^2))
prefactor = normalization_gaussian_energy * sqrt(N)
#* OFTs
@time oft_entry = entry_wise_oft(jump, an_energy, hamiltonian, sigma, beta) / prefactor
@time oft_bohr = bohr_decomp_oft(jump, an_energy, hamiltonian, num_energy_bits, sigma, beta) / prefactor
@time oft_explicit = explicit_oft(jump, hamiltonian, an_energy, time_labels, sigma, beta) / prefactor
@printf("Distance Dream - Bohr: %e\n", frobenius_norm(oft_entry - oft_bohr))
@printf("Distance Dream - Explicit: %e\n", frobenius_norm(oft_entry - oft_explicit))
# println(round.(oft_entry, digits=15) == zeros(ComplexF64, size(oft_entry)))

println(oft_entry' == entry_wise_oft(jump, -an_energy, hamiltonian, sigma, beta) / prefactor)
println(oft_bohr' == bohr_decomp_oft(jump, -an_energy, hamiltonian, num_energy_bits, sigma, beta) / prefactor)
println(oft_explicit' == explicit_oft(jump, hamiltonian, -an_energy, time_labels, sigma, beta) / prefactor)