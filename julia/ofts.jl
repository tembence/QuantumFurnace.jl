using LinearAlgebra
using SparseArrays
using Random
using Printf
using Plots

include("hamiltonian.jl")
include("trotter.jl")
include("qi_tools.jl")
include("structs.jl")

function oft(jump::JumpOp, energy::Float64, hamiltonian::HamHam, beta::Float64)
    """sigma_E = 1 / beta. Subnormalized, multiply by sqrt(beta / sqrt(2 * pi))"""
    return jump.in_eigenbasis .* exp.(-beta^2 * (energy .- hamiltonian.bohr_freqs).^2 / 4) 
end

function trotter_oft(jump::JumpOp, energy::Float64, trotter::TrottTrott, time_labels::Vector{Float64}, beta::Float64)

    prefactors = @fastmath exp.(- time_labels.^2 / beta^2 .- 1im * energy * time_labels) # Gauss and Fourier factors

    mid_point = Int(length(time_labels) / 2) # Up to positive times
    N = length(time_labels)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    num_t0_steps = abs.(N_labels)
    eigvals_t0_diag = Diagonal(trotter.eigvals_t0)

    jump_oft = zeros(ComplexF64, size(jump.data))
    if jump.orthogonal # Orthogonal symmetry that halves time labels
        # t = 0.0
        trott_U_plus = eigvals_t0_diag^num_t0_steps[1]
        jump_oft .+= @fastmath (prefactors[1] * trott_U_plus * jump.in_trotter_basis * adjoint(trott_U_plus))

        # t > 0.0
        for i in range(2, mid_point)
            trott_U_plus = eigvals_t0_diag^num_t0_steps[i]

            temp_op = trott_U_plus * jump.in_trotter_basis * adjoint(trott_U_plus)
            jump_oft .+=  @fastmath (prefactors[i] * temp_op)
            jump_oft .+=  @fastmath (conj(prefactors[i]) * transpose(temp_op))
        end
    else
        for i in eachindex(time_labels)
            # trott_U = eigvals_t0_diag^num_t0_steps[i]
            # jump_oft .+= @fastmath (prefactors[i] * trott_U * jump.in_eigenbasis * adjoint(trott_U))

            trott_U_plus = eigvals_t0_diag^num_t0_steps[i]
            trott_U_minus = adjoint(trott_U_plus)

            # # less allocs this way:
            temp_op = (i <= mid_point) ? trott_U_plus * jump.in_trotter_basis * trott_U_minus : 
                                        trott_U_minus * jump.in_trotter_basis * trott_U_plus
            jump_oft .+= @fastmath (prefactors[i] * temp_op)
        end
    end

    # Return in Trotter basis
    return jump_oft
end

function time_oft(jump::JumpOp, energy::Float64, hamiltonian::HamHam, time_labels::Vector{Float64}, beta::Float64)
    """sigma_E = 1 / beta, subnormalized OFT: multiply by: t0 * sqrt(sqrt(2 / pi)/beta) / sqrt(2 * pi)"""

    prefactors = @fastmath exp.(- time_labels.^2 / beta^2 .- 1im * energy * time_labels) # Gauss and Fourier factors
    mid_point = Int(length(time_labels) / 2) # Up to positive times

    diag_exponentiate(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))

    jump_oft = zeros(ComplexF64, size(jump.data))
    if jump.orthogonal # Orthogonal symmetry that halves time labels
        # t = 0.0
        U_plus = diag_exponentiate(time_labels[1])
        jump_oft .+= @fastmath (prefactors[1] * U_plus * jump.in_eigenbasis * adjoint(U_plus))

        # t > 0.0
        for i in range(2, mid_point)
            U_plus = diag_exponentiate(time_labels[i])

            temp_op = U_plus * jump.in_eigenbasis * adjoint(U_plus)
            jump_oft .+=  @fastmath (prefactors[i] * temp_op)
            jump_oft .+=  @fastmath (conj(prefactors[i]) * transpose(temp_op))
        end
    else  # For non-orthogonal jumps
        for i in eachindex(time_labels)
            U = diag_exponentiate(time_labels[i])
            jump_oft .+= @fastmath (prefactors[i] * U * jump.in_eigenbasis * adjoint(U))

            # U_plus = diag_exponentiate(time_labels[i])
            # U_minus = adjoint(U_plus)

            # # less allocs this way:
            # temp_op = (i <= mid_point) ? U_plus * jump.in_eigenbasis * U_minus : 
            #                             U_minus * jump.in_eigenbasis * U_plus
            # jump_oft .+= @fastmath (prefactors[i] * temp_op)
        end
    end
    
    # jump_oft = zeros(ComplexF64, size(jump.data))
    # @showprogress "Explicit OFT..." for t in eachindex(time_labels)
    #     jump_oft .+= fourier_phase_factors[t] * normalized_time_gaussian_factors[t] * 
    #     diag_exponentiate(time_labels[t]) * jump.in_eigenbasis * diag_exponentiate(-time_labels[t])
    # end

    return jump_oft
end

#TODO: Check if this matches with time_oft
function time_oft_integrated(energy::Float64, jump::JumpOp, hamiltonian::HamHam, beta::Float64)

    diag_exponentiate(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))
    integrand(t) = exp.(-t^2 / beta^2 - 1im * energy * t) * diag_exponentiate(t) * jump.in_eigenbasis * diag_exponentiate(-t)
    jump_oft, _ = quadgk(integrand, -Inf, Inf)[1] / sqrt(2 * pi) * sqrt(sqrt(2 / pi) / beta)
    return jump_oft
end

#* ---------- Test ----------

#* Parameters
# num_qubits = 4
# beta = 10.
# eig_index = 8
# jump_site_index = 1

# #* Hamiltonian
# # hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n4.jld")["ideal_ham"]
# hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=100)
# initial_state = hamiltonian.eigvecs[:, eig_index]
# hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

# num_energy_bits = ceil(Int64, log2((0.45 * 4 + 2/beta) / hamiltonian.nu_min)) + 3  # Under Fig. 5. with secular approx.
# N = 2^(num_energy_bits)
# N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

# t0 = 2 * pi / (N * hamiltonian.nu_min)
# time_labels = t0 * N_labels
# energy_labels = hamiltonian.nu_min * N_labels
# # OFT normalizations for energy basis
# Fw = exp.(- beta^2 * (energy_labels).^2 / 4)
# Fw_norm = sqrt(sum(Fw.^2))
# ft = exp.(- time_labels.^2 / beta^2)
# ft_norm = sqrt(sum(ft.^2))

# # energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]  # Energies outside are impossible in a QC

# rand_energy = 0.43
# phase = rand_energy * N / (2 * pi)

# # Transition weights in the liouv
# transition_gaussian(energy) = exp(-(beta * energy + 1)^2 / 2)

# #* Trotter
# num_trotter_steps_per_t0 = 10
# trotter = create_trotter(hamiltonian, t0, num_trotter_steps_per_t0)
# trotter_error_T = compute_trotter_error(hamiltonian, trotter, N*t0)
# trotter_error_t0 = compute_trotter_error(hamiltonian, trotter, t0)
# @printf("Trotter error T: %e\n", trotter_error_T)
# @printf("Trotter error t0: %e\n", trotter_error_t0)

# #* Jump operators
# X::Matrix{ComplexF64} = [0 1; 1 0]
# Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
# Z::Matrix{ComplexF64} = [1 0; 0 -1]
# jump_paulis = [X, Y, Z]
# all_jumps_generated = []

# # #* X JUMP
# sites = [1, 2, 3]
# for site in sites
# jump_op = Matrix(pad_term([X], num_qubits, site))
#     jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
#     jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
#     orthogonal = (jump_op == adjoint(jump_op))
#     jump = JumpOp(jump_op,
#             jump_op_in_eigenbasis,
#             Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
#             zeros(0),
#             jump_in_trotter_basis,
#             orthogonal) #jump_in_trotter_basis
#     push!(all_jumps_generated, jump)
# end

# # #! Uncomment for bohr oft
# # # construct_A_nus(jump, hamiltonian)
# # # println(jump.unique_freqs[abs.(jump.unique_freqs) .< 0.1])
# # # println(jump.bohr_decomp)
# # # for (key, value) in jump.bohr_decomp
# # #     println(key)
# # #     println(value)
# # # end

# # # For full Liouvillian dynamics:
# # # all_x_jump_ops = []
# # # for q in 1:num_qubits
# # #     padded_x = pad_term([jump_op], q, num_qubits)
# # #     push!(all_x_jump_ops, padded_x)
# # # end

# # # @printf("Number of qubits: %d\n", num_qubits)
# # # @printf("Number of energy bits: %d\n", num_energy_bits)
# # # @printf("Energy unit: %e\n", hamiltonian.nu_min)
# # # @printf("Time unit: %e\n", t0)
# # # @printf("Energy")

# # #* -------------------------------------------- *#

# jump = all_jumps_generated[jump_site_index]
# total_err = 0.0
# for w in energy_labels
#     oft_trotter = trotter_oft(jump, w, trotter, time_labels, beta) #/ (ft_norm * sqrt(length(time_labels)))
#     oft_trotter_in_eigenbasis = hamiltonian.eigvecs' * trotter.eigvecs * oft_trotter * trotter.eigvecs' * hamiltonian.eigvecs
#     # oft = entry_wise_oft_exact_db(jump, rand_energy, hamiltonian, beta) / Fw_norm
#     oft_expl = time_oft(jump, w, hamiltonian, time_labels, beta) #/ (ft_norm * sqrt(length(time_labels)))
#     # @printf("Distance Entrywise OFT - Trotter: %e\n", frobenius_norm(oft_trotter_in_eigenbasis - oft))
#     err = frobenius_norm(oft_expl - oft_trotter_in_eigenbasis)
#     total_err += err
#     # @printf("Distance Explicit - Trotter: %e\n", err)
#     # @printf("Distance Explicit - Entrywise: %e\n", frobenius_norm(oft_expl - oft))
# end
# @printf("Total error: %e\n", total_err)
# #* Heisenberg is weighted sum of A_nus
# # t = 3 * t0
# # heis_A = exp(1im * hamiltonian.data * t) * jump.data * exp(-1im * hamiltonian.data * t)
# # heis_A_in_eigenbasis = hamiltonian.eigvecs' * heis_A * hamiltonian.eigvecs
# # @printf("Num of unique freqs %d\n", length(jump.unique_freqs))
# # A_nu_sum = zeros(ComplexF64, size(jump.data))
# # for (nu, A_nu) in jump.bohr_decomp
# #     A_nu_sum += exp(1im * nu * t) * A_nu
# # end
# # println("Distance between Heisenberg and A_nu sum:")
# # display(frobenius_norm(heis_A_in_eigenbasis - A_nu_sum))

