using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using BenchmarkTools
using Roots

include("hamiltonian_tools.jl")
include("jump_op_tools.jl")
include("qi_tools.jl")
include("structs.jl")

#! We shouldn't have truncated the energy labels. Or at least shouldn't have done it so drastically to 0.45.
function create_alpha_from_gaussians(nu_1::Float64, nu_2::Float64, num_energy_bits::Int64, w0::Float64, beta::Float64;
    energy_cutoff_epsilon::Float64 = 1e-3)  # Truncating with this epsilon is good, any larger and it actually has an effect.

    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = w0 * N_labels
    energy_cutoff_for_alpha(beta, nu_max, eps) = nu_max + sqrt(4 * log(1/eps) / beta^2)
    # nu_max = 0.45
    energy_cutoff_for_alpha = energy_cutoff_for_alpha(beta, 0.45, energy_cutoff_epsilon)
    energy_labels = energy_labels[abs.(energy_labels) .<= energy_cutoff_for_alpha]

    alpha = 0.0
    for w in energy_labels
        alpha += beta * (exp(-beta^2 * (w + 1/beta)^2 / 2)
                           * exp(-beta^2 * (w - nu_1)^2 / 4) * exp(-beta^2 * (w - nu_2)^2 / 4) / sqrt(8 * pi))
    end
    return w0 * alpha
end

#! The error becomes larger and larger and seemingly can't get it low with more energy bits or no truncation. 
num_qubits = 5
num_energy_bits = 9
beta = 10.
Random.seed!(666)
with_coherent = true

#* Hamiltonian
ham_filename(n) = @sprintf("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n%d.jld", n)
hamiltonian = load(ham_filename(num_qubits))["ideal_ham"]
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

N = 2^(num_energy_bits)
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
energy_labels = hamiltonian.w0 * N_labels
energy_labels = energy_labels[abs.(energy_labels) .<= 0.5]

alpha_fn(nu_1, nu_2) = exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2/16)exp(-beta^2 * (nu_1 - nu_2)^2/8) / sqrt(8)
total_error = 0.0
for nu_1 in hamiltonian.bohr_freqs
    for nu_2 in hamiltonian.bohr_freqs
        alpha_from_gaussians = create_alpha_from_gaussians(nu_1, nu_2, num_energy_bits, hamiltonian.w0, beta)
        alpha = alpha_fn(nu_1, nu_2)
        total_error += norm(alpha_from_gaussians - alpha)
    end
end

@printf("Total error between alphas: %s\n", total_error)