using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using BenchmarkTools
using Roots
using QuadGK

include("hamiltonian.jl")
include("ofts.jl")
include("qi_tools.jl")
include("structs.jl")
include("coherent.jl")

function construct_liouvillian_gauss_trotter(jumps::Vector{JumpOp}, trotter::TrottTrott, time_labels::Vector{Float64},
    energy_labels::Vector{Float64}, with_coherent::Bool, beta::Float64)

    dim = size(trotter.eigvecs, 1)
    w0 = energy_labels[2] - energy_labels[1]
    transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)

    if with_coherent  # Steup for coherent term in time domain
        b1 = compute_truncated_b1(time_labels)
        b2 = compute_truncated_b2(time_labels)
    end

    total_liouv_coherent_part = zeros(ComplexF64, dim^2, dim^2)
    total_liouv_diss_part = zeros(ComplexF64, dim^2, dim^2)
    @showprogress desc="Liouvillian (Trotter)..." for jump in jumps
        if with_coherent  # There is no energy formulation of the coherent term, only Bohr and time.
            coherent_term = coherent_term_trotter(jump, trotter, b1, b2, beta)
            coherent_term = hamiltonian.eigvecs' * trotter.eigvecs * coherent_term * trotter.eigvecs' * hamiltonian.eigvecs  #!
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = trotter_oft(jump, w, trotter, time_labels, beta) # t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
            jump_oft = hamiltonian.eigvecs' * trotter.eigvecs * jump_oft * trotter.eigvecs' * hamiltonian.eigvecs  #!
            # jump_oft_actually = oft(jump, w, hamiltonian, beta) * sqrt(beta / sqrt(2 * pi))
            # @printf("Jump oft norm: %s\n", norm(jump_oft - jump_oft_actually))
            total_liouv_diss_part .+= transition_gauss(w) * vectorize_liouvillian_diss(jump_oft)
        end
    end
    time_oft_norm_squared = (sqrt(2 / pi)/beta) / (2 * pi)  # ft and fourier norms
    return total_liouv_coherent_part .+ w0 * trotter.t0^2 * time_oft_norm_squared * total_liouv_diss_part
end