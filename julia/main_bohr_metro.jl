using LinearAlgebra
using Random
using Printf
using ProgressMeter
using JLD2

include("hamiltonian.jl")
include("ofts.jl")
include("qi_tools.jl")
include("bohr_picture.jl")

#* Parameters
num_qubits = 4
dim = 2^num_qubits
beta = 10.
Random.seed!(666)

with_coherent = true

#* Hamiltonian
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=10)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

#* Gibbs
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
gibbs_vec = vec(gibbs)
gibbs_largest_eigval = real(eigen(gibbs).values[1])

#* Jump operators
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
H::Matrix{ComplexF64} = [1 1; 1 -1] / sqrt(2)
jump_paulis = [[X], [Y], [Z], [H]]

# All jumps once
all_jumps_generated::Vector{JumpOp} = []
for pauli in jump_paulis
    for site in 1:num_qubits
    jump_op = Matrix(pad_term(pauli, num_qubits, site))
    jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
    jump_in_trotter_basis = zeros(0, 0)
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

#* The Press
@printf("Number of qubits: %d\n", num_qubits)
# --- 

#* Construct Bohr Liouvillian
@time liouv_matrix = construct_liouvillian_bohr_metro(all_jumps_generated, hamiltonian, with_coherent, beta)
# @time liouv_matrix = construct_liouvillian_bohr_gauss(all_jumps_generated, hamiltonian, with_coherent, beta)
liouv_eigvals, liouv_eigvecs = eigen(liouv_matrix) 
steady_state_vec = liouv_eigvecs[:, end]
steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
steady_state_dm /= tr(steady_state_dm)
steady_state_vec = vec(steady_state_dm)

norm(gibbs_vec - steady_state_vec)

#* DB of transition part CHECK
# T_dagger = transition_bohr_metro(all_jumps_generated, hamiltonian, beta, do_adjoint=true)
# T_gibbsed = transition_bohr_metro_gibbsed(all_jumps_generated, hamiltonian, beta)
# norm(T_dagger - T_gibbsed)

#* B and R relation
#! I get that we could get out of bound and thats a problem but in bound how can it be that v1-v2 is not a Bohr freq??
#! Ah for larger systems that error smoothens out as the grid gets finer. And the out of bound is the biggest error source.
# hamiltonian.bohr_freqs
# bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
# nus = collect(keys(bohr_dict))
# for nu_1 in nus
#     for nu_2 in nus
#         nu = nu_1 - nu_2
#         closest_nu = nus[argmin(abs.(nus .- nu).^2)]
#         println(nu, " ", closest_nu)
#     end
# end

# nus_failed = []
# for nu in keys(bohr_dict)
#     for nu_2 in keys(bohr_dict)
#         jump = all_jumps_generated[2]
#         B_nu = B_nu_gauss(nu, nu_2, hamiltonian, bohr_dict, jump, beta)
#         R_nu = R_nu_gauss(nu, nu_2, hamiltonian, bohr_dict, jump, beta)

#         # display(norm(B_nu))
#         display(norm(B_nu - R_nu))
#         # if norm(B_nu - R_nu) > 1e-16
#         #     push!(nus_failed, nu)
#         # end
#     end
# end
# display(nus_failed)

#* Forcing v1-v2 to be at least in bound, even if not another nu from B
#! I think it only has to be forced upon B (f) and not R (alpha)!
# nu_2 = hamiltonian.bohr_freqs[4, 3]
# nus_diff = hamiltonian.bohr_freqs .- nu_2
# mask = heaviside_mask_matrix(hamiltonian.bohr_freqs, nu_2)
# masked_nus = nus_diff .* mask
# masked_diff = nus_diff - masked_nus
# masked_diff[findall(!iszero, masked_diff)]

# Define your matrix and list of frequency values
# println(sort(collect(keys(bohr_dict))))
# println()
# unique_freqs = Set(keys(bohr_dict))
# display(nus_diff)
# println()
# freq_set = Set(frequencies)
# @time exactly_filtered_nus_diff = nus_diff .* in.(nus_diff, Ref(unique_freqs))  # .= helps later
# @time exact_mask = in.(nus_diff, Ref(unique_freqs)) .+ 0
# display(exactly_filtered_nus_diff)
# println()
# eps = 1e-14
# @time approx_filtered_nus_diff =  nus_diff .* map(x -> any(abs.(unique_freqs .- x) .< eps), nus_diff)
# display(approx_filtered_nus_diff)

#* Skew symmetry for metro approach
# jump = all_jumps_generated[1]
# nu_1_indices = CartesianIndex{2}(3, 4)
# nu_2_indices = CartesianIndex{2}(1, 7)

# nu_2 = hamiltonian.bohr_freqs[nu_2_indices]
# nu_1 = hamiltonian.bohr_freqs[nu_1_indices]

# for i in 1:8
#     for j in 1:8
#         nu_1_indices = CartesianIndex{2}(i, j)
#         minus_nu_1_indices = CartesianIndex{2}(j, i)
#         nu_1 = hamiltonian.bohr_freqs[nu_1_indices]
#     #     alpha_nu_1_matrix_gauss = create_alpha_nu1_matrix(hamiltonian.bohr_freqs, nu_2, beta)
#     #     alpha_minus_nu_2_matrix_gauss = create_alpha_nu1_matrix(hamiltonian.bohr_freqs, -nu_1, beta)

#     #    @assert (norm(alpha_nu_1_matrix_gauss[nu_1_indices] 
#     #             - alpha_minus_nu_2_matrix_gauss[minus_nu_2_indices] * exp(-beta * (nu_1 + nu_2) / 2)) < 1e-14)


#         alpha_nu_1_matrix_metro = @time create_alpha_nu1_matrix_metro(hamiltonian.bohr_freqs, nu_2, beta)
#         alpha_minus_nu_1_matrix_metro = @time create_alpha_nu1_matrix_metro(hamiltonian.bohr_freqs, -nu_2, beta)


#         println(norm(alpha_nu_1_matrix_metro[nu_1_indices] 
#                 - alpha_minus_nu_1_matrix_metro[minus_nu_1_indices] * exp(-beta * (nu_1 + nu_2) / 2)) < 1e-14)

#     end
# end
# @printf("donezo\n")

# #* Metropolis weight itself
# energy = 0.22
# get_metropolis_weight_analytically(energy) = exp(-beta * (energy + 1/(2*beta)))

# metropolis_weight_numerical = get_metropolis_weight_numerically(energy, beta)
# metropolis_weight_analytical = get_metropolis_weight_analytically(energy)
# for energy in 0:0.1:0.45
#     @assert norm(metropolis_weight_numerical - metropolis_weight_analytical) < 1e-14
# end