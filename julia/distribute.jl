using Distributed
using LinearAlgebra

include("jump_workers.jl")

function construct_liouvillian_bohr_distributed(jumps::Vector{JumpOp}, hamiltonian::HamHam, 
    config::LiouvConfig)
    total_liouv = @distributed (+) for jump in jumps
        jump_contribution_bohr(jump, hamiltonian, config)
    end
    return total_liouv
end

function construct_liouvillian_energy_distributed(jumps::Vector{JumpOp}, hamiltonian::HamHam, 
    truncated_energy_labels::Vector{Float64}, config::LiouvConfig)
    total_liouv = @distributed (+) for jump in jumps
        jump_contribution_energy(jump, hamiltonian, config)
    end
    return total_liouv
end

function construct_liouvillian_time_distributed(jumps::Vector{JumpOp}, hamiltonian::HamHam, 
    time_labels::Vector{Float64}, truncated_energy_labels::Vector{Float64}, config::LiouvConfig)
end

function construct_liouvillian_trotter_distributed(jumps::Vector{JumpOp}, trotter::TrottTrott, 
    time_labels::Vector{Float64}, truncated_energy_labels::Vector{Float64}, config::LiouvConfig)
end