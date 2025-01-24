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
include("qi_tools.jl")
include("structs.jl")
include("bohr.jl")
include("ofts.jl")

function construct_liouvillian_metro(jumps::Vector{JumpOp}, hamiltonian::HamHam, energy_labels::Vector{Float64}, 
    with_coherent::Bool, beta::Float64)

    dim = size(hamiltonian.data, 1)
    w0 = energy_labels[2] - energy_labels[1]
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
    transition_metro(w) = exp(-beta * maximum([w + 1/(2*beta), 0]))

    total_liouv_coherent_part = zeros(ComplexF64, dim^2, dim^2)
    total_liouv_diss_part = zeros(ComplexF64, dim^2, dim^2)

    @showprogress desc="Liouvillian (Energy)..." for jump in jumps
        if with_coherent  # There is no energy formulation of the coherent term, only Bohr and time.
            coherent_term = coherent_metro_bohr(hamiltonian, bohr_dict, jump, beta)
            @printf("Is B METRO Hermitian?:%s\n", norm(coherent_term - coherent_term'))
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = oft(jump, w, hamiltonian, beta)
            total_liouv_diss_part .+= transition_metro(w) * vectorize_liouvillian_diss(jump_oft)
        end
    end
    oft_norm_squared = beta / sqrt(2 * pi)  #! sqrt(8 * pi) -> sqrt(2 * pi)
    return total_liouv_coherent_part .+ w0 * oft_norm_squared * total_liouv_diss_part
end
