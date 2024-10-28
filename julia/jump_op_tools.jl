using LinearAlgebra
using SparseArrays
using Random
using Printf
using Plots

include("hamiltonian_tools.jl")
include("trotter.jl")
include("qi_tools.jl")
include("trotter.jl")
include("structs.jl")

function add_jump_in_trotter_basis(jump::JumpOp, trotter::TrottTrott)
    """Adds a jump operator in the Trotter basis"""
    jump.in_trotter_basis = trotter.eigvecs' * jump.data * trotter.eigvecs
end

function entry_wise_oft(jump::JumpOp, energy::Float64, hamiltonian::HamHam, sigma::Float64, beta::Float64)
    """Uses the already Fourier transformed Gaussian filter fn."""
    # The for loop version is the same but a bit slower than broadcasting
    # nrows, ncols = size(jump.data)
    # jump_oft = zeros(ComplexF64, nrows, ncols)
    # for j in 1:ncols, i in 1:nrows
    #     jump_oft[i, j] = jump.in_eigenbasis[i, j] * exp(-(energy - hamiltonian.bohr_freqs[i, j])^2 * sigma^2)
    # end

    return jump.in_eigenbasis .* exp.(-((energy .- hamiltonian.bohr_freqs)).^2 * sigma^2)
end

function entry_wise_oft_exact_db(jump::JumpOp, energy::Float64, hamiltonian::HamHam, beta::Float64)
    """sigma_E = 1 / beta. """
    return jump.in_eigenbasis .* exp.(-((energy .- hamiltonian.bohr_freqs)).^2 * beta^2 / 4) #/ sqrt(2 * sqrt(2 * pi) / beta)
end

function trotter_oft(jump::JumpOp, trotter::TrottTrott, 
    energy::Float64, time_labels::Vector{Float64}, beta::Float64)

    prefactors = @fastmath exp.(- time_labels.^2 / beta^2 - 1im * energy * time_labels) # Gauss and Fourier factors

    mid_point = Int(length(time_labels) / 2) # Up to positive times
    num_t0_steps = ceil.((abs.(time_labels) ./ trotter.t0))
    eigvals_t0_diag = Diagonal(trotter.eigvals_t0)

    oft_op = zeros(ComplexF64, size(jump.data))
    if jump.orthogonal # Orthogonal symmetry that halves time labels
        # t = 0.0
        trott_U_plus = eigvals_t0_diag^num_t0_steps[1]
        oft_op .+= @fastmath (prefactors[1] * trott_U_plus * jump.in_trotter_basis * adjoint(trott_U_plus))

        # t > 0.0
        for i in range(2, mid_point)
            trott_U_plus = eigvals_t0_diag^num_t0_steps[i]

            temp_op = trott_U_plus * jump.in_trotter_basis * adjoint(trott_U_plus)
            oft_op .+= @fastmath (prefactors[i] * temp_op)
            oft_op .+= @fastmath (conj(prefactors[i]) * transpose(temp_op))
        end
    else
        for i in eachindex(time_labels)
            trott_U_plus = eigvals_t0_diag^num_t0_steps[i]
            trott_U_minus = adjoint(trott_U_plus)

            # less allocs this way:
            temp_op = (i <= mid_point) ? trott_U_plus * jump.in_trotter_basis * trott_U_minus : 
                                        trott_U_minus * jump.in_trotter_basis * trott_U_plus
            oft_op .+= @fastmath (prefactors[i] * temp_op)
        end
    end

    # Return in Trotter basis
    return oft_op
end

function explicit_oft_exact_db(jump::JumpOp, hamiltonian::HamHam, energy::Float64, time_labels::Vector{Float64},
    beta::Float64)
    """sigma_E = 1 / beta"""

    time_gaussian_factors = exp.(- time_labels.^2 / beta^2)
    # normalization is converging to sqrt(beta * sqrt(pi / 2)) for large N
    normalized_time_gaussian_factors = time_gaussian_factors / sqrt(sum(time_gaussian_factors.^2))

    fourier_phase_factors = exp.(-1im * energy * time_labels) / sqrt(length(time_labels))
    diag_exponentiate(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))
    
    oft_op = zeros(ComplexF64, size(jump.data))
    @showprogress "Explicit OFT..." for t in eachindex(time_labels)
        oft_op .+= fourier_phase_factors[t] * normalized_time_gaussian_factors[t] * 
        diag_exponentiate(time_labels[t]) * jump.in_eigenbasis * diag_exponentiate(-time_labels[t])
    end

    return oft_op
end

function explicit_oft(jump::JumpOp, hamiltonian::HamHam, energy::Float64, time_labels::Vector{Float64},
    sigma::Float64, beta::Float64)
    """Fourier transforms by summing over the time labels. Both time_labels and the energy should be in t0, w0 units"""

    time_gaussian_factors = exp.(- time_labels.^2 / (4 * sigma^2))
    normalized_time_gaussian_factors = time_gaussian_factors / sqrt(sum(time_gaussian_factors.^2))
    
    fourier_phase_factors = exp.(-1im * energy * time_labels) / sqrt(length(time_labels))
    diag_exponentiate(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))

    
    # With full Hamiltonians for debugging
    # time_gaussian(t) = exp(- t^2 / (4 * sigma^2))
    # normalization_gaussian = sqrt(sum(time_gaussian.(time_labels).^2))
    # oft_op = zeros(ComplexF64, size(jump.data))
    # @showprogress for t in time_labels
    #     oft_op += time_gaussian(t) * exp(-1im * energy * t) * exp(1im * t * hamiltonian.data) * 
    #     jump.data * exp(-1im * t * hamiltonian.data) / (normalization_gaussian * sqrt(length(time_labels)))
    # end

    # In eigenbasis
    # return hamiltonian.eigvecs' * oft_op * hamiltonian.eigvecs

    oft_op = zeros(ComplexF64, size(jump.data))
    @showprogress "Explicit OFT..." for t in eachindex(time_labels)
        oft_op .+= fourier_phase_factors[t] * normalized_time_gaussian_factors[t] * 
        diag_exponentiate(time_labels[t]) * jump.in_eigenbasis * diag_exponentiate(-time_labels[t])
    end

    return oft_op
end

function explicit_trotter_oft(jump::JumpOp, trotter::TrottTrott, 
    energy::Float64, time_labels::Vector{Float64}, sigma::Float64, beta::Float64)
    """Fourier transforms by summing over the time labels. Both time_labels and the energy should be in t0, w0 units.
    Time evolution is done with Trotterization. Computes stuff in the Trotter basis."""

    time_gaussian_factors = exp.(- time_labels.^2 / (4 * sigma^2))
    normalized_time_gaussian_factors = time_gaussian_factors / sqrt(sum(time_gaussian_factors.^2))
    
    fourier_phase_factors = exp.(-1im * energy * time_labels) / sqrt(length(time_labels))

    # Do a t0 Trotter with steps s.t. it is precise for the longest time evolution in the algorithm (N * t0 or N/2 * t0)

    num_t0_steps = Int.(round.(abs.(time_labels) ./ trotter.t0, digits=0))
    oft_op = zeros(ComplexF64, size(jump.data))
    @showprogress "Explicit Trotter OFT..." for i in eachindex(time_labels)
        t = time_labels[i]
        trott_U_plus = Diagonal(trotter.eigvals_t0)^(num_t0_steps[i])
        trott_U_minus = adjoint(trott_U_plus)

        if i <= length(time_labels) / 2
            oft_op .+= fourier_phase_factors[i] * normalized_time_gaussian_factors[i] * 
                    trott_U_plus * jump.in_trotter_basis * trott_U_minus
        else
            oft_op .+= fourier_phase_factors[i] * normalized_time_gaussian_factors[i] * 
                    trott_U_minus * jump.in_trotter_basis * trott_U_plus
        end
    end

    # Return in energy basis
    # hamiltonian.eigvecs' * trotter.eigvecs * oft_op * trotter.eigvecs' * hamiltonian.eigvecs

    # Return in Trotter basis
    return oft_op
end

# Sometimes gives Inf values in matrix...
function bohr_decomp_oft(jump::JumpOp, energy::Float64, hamiltonian::HamHam, num_energy_bits::Int64, sigma::Float64, beta::Float64)
    #! This assumes that we can store all A_nus in a dict

    jump_oft = zeros(ComplexF64, size(jump.data))
    for (bohr_freq, jump_op_nu) in jump.bohr_decomp
        jump_oft += jump_op_nu * exp(-((energy - bohr_freq)^2) * sigma^2)
    end

    return jump_oft
end

function construct_A_nus(jump::JumpOp, hamiltonian::HamHam)
"""Constructrs jump.bohr_decomp = Dict {bohr energy (nu): A_nu}"""
    # A'_ij is the entry for the jump between eigenenergies j -> i
    # jump.in_eigenbasis
    # Matrix of all energy jumps in Hamiltonian, B_ij = E_i - E_j and i,j are ordered from smallest to largest energy


    jump_bohr_indices_dict = get_jump_bohr_indices(hamiltonian.bohr_freqs)
    jump.unique_freqs = collect(keys(jump_bohr_indices_dict))

    for (bohr_freq, indices) in jump_bohr_indices_dict
        jump_op_nu = spzeros(ComplexF64, size(jump.data))
        for (i, j) in indices
            jump_op_nu[i, j] = jump.in_eigenbasis[i, j]
        end
        jump.bohr_decomp[bohr_freq] = jump_op_nu
    end

    # Does all the A_nus add up to A?
    # reconstr_jump_op = zeros(ComplexF64, size(jump.data))
    # for (bohr_freq, jump_op_nu) in jump.bohr_decomp
    #     reconstr_jump_op += jump_op_nu
    # end
    # @assert norm(reconstr_jump_op - jump.in_eigenbasis) < 1e-10
    # @printf("All A_nus add up to A\n")
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

#* ---------- Test ----------

#* Parameters
# num_qubits = 4
# beta = 10.
# eig_index = 8
# jump_site_index = 1

# #* Hamiltonian
# hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n4.jld")["ideal_ham"]
# initial_state = hamiltonian.eigvecs[:, eig_index]
# hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

# num_energy_bits = ceil(Int64, log2((0.45 * 4 + 2/beta) / hamiltonian.w0)) + 3  # Under Fig. 5. with secular approx.
# N = 2^(num_energy_bits)
# N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

# t0 = 2 * pi / (N * hamiltonian.w0)
# time_labels = t0 * N_labels
# energy_labels = hamiltonian.w0 * N_labels
# energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]  # Energies outside are impossible in a QC
# Random.seed!(666)
# rand_energy = rand(energy_labels)
# phase = rand_energy * N / (2 * pi)

# # OFT normalizations for energy basis
# Fw = exp.(- beta^2 * (energy_labels).^2 / 4)
# Fw_norm = sqrt(sum(Fw.^2))
# ft = exp.(- time_labels.^2 / beta^2)
# ft_norm = sqrt(sum(ft.^2))

# # Transition weights in the liouv
# transition_gaussian(energy) = exp(-(beta * energy + 1)^2 / 2)

# #* Trotter
# num_trotter_steps_per_t0 = Int(10)
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

# #* X JUMP
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

# # # #! Uncomment for bohr oft
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

# @printf("Number of qubits: %d\n", num_qubits)
# @printf("Number of energy bits: %d\n", num_energy_bits)
# @printf("Energy unit: %e\n", hamiltonian.w0)
# @printf("Time unit: %e\n", t0)
# @printf("Energy")

# #* -------------------------------------------- *#
# #* Orthogonal A reduces trotter_oft!!

# @time oft_trotter = trotter_oft(jump, trotter, rand_energy, time_labels, beta) / (ft_norm * sqrt(length(time_labels)))
# oft_trotter_in_eigenbasis = hamiltonian.eigvecs' * trotter.eigvecs * oft_trotter * trotter.eigvecs' * hamiltonian.eigvecs
# oft = entry_wise_oft_exact_db(jump, rand_energy, hamiltonian, beta) / Fw_norm
# @printf("Distance Entrywise OFT - Trotter: %e\n", frobenius_norm(oft_trotter_in_eigenbasis - oft))


# # # # # is it real?
# # # # # display(isapprox(imag(oft_expl), zeros(ComplexF64, size(jump.data))))
# # # @time oft_entry = entry_wise_oft(jump, energy, hamiltonian, sigma, beta) / (Fw_norm)
# # # @printf("Distance Expl Trotter - Dream: %e\n", frobenius_norm(oft_expl - oft_entry))
# # # # # display(isapprox(imag(oft_entry), zeros(ComplexF64, size(jump.data))))
# # # # # expl_entries = collect(Iterators.flatten(oft_expl))
# # # # # expl_entries[(imag(expl_entries)) .> 1e-18]

# # # # @time oft_bohr = bohr_decomp_oft(jump, energy, hamiltonian, num_energy_bits, sigma, beta) / Fw_norm

# # # # @printf("Distance Expl - Bohr: %e\n", frobenius_norm(oft_expl - oft_bohr))
# # # # @printf("Distance Dream - Bohr: %e\n", frobenius_norm(oft_entry - oft_bohr))
# # # # @printf("Distance Expl - Dream: %e\n", frobenius_norm(oft_expl - oft_entry))
# # # # display(isapprox(oft_expl - oft_entry, zeros(ComplexF64, size(jump.data))))
# # # # display(norm(oft_expl - oft_entry))
# # # # @printf("Are they the same?: %s\n", norm(oft_expl - oft_entry) < 1e-10)


# # # # if !isapprox(t0 * hamiltonian.w0, 2 * pi / length(time_labels))
# # # #     error("t0 * w0 != 2 * pi / N")
# # # # end

# # # # time_gaussian_factors = exp.(- time_labels.^2 / (4 * sigma^2))
# # # # normalized_time_gaussian_factors = time_gaussian_factors / sqrt(sum(time_gaussian_factors.^2))
# # # # fourier_phase_factors = exp.(-1im * energy * time_labels) / sqrt(length(time_labels))
# # # # # time_evo_phasevector_t = exp.(1im * transpose(Diagonal(hamiltonian.eigvals)) .* time_labels)
# # # # diag_exponentiate(t) = exp(1im * Diagonal(hamiltonian.eigvals) * t)

# # # # prefactors = normalized_time_gaussian_factors .* fourier_phase_factors

# # # # @time begin
# # # #     oft_op = zeros(ComplexF64, size(jump.data))
# # # #     @showprogress for t in 1:length(time_labels)
# # # #         oft_op += fourier_phase_factors[t] * normalized_time_gaussian_factors[t] * 
# # # #         Diagonal(time_evo_phasevector_t[t, :]) * jump.in_eigenbasis * Diagonal(-(time_evo_phasevector_t[t, :]))
# # # #     end
# # # # end
# # # # time_evolutions = diag_exponentiate.(time_labels)
# # # # time_evolutions_dag = diag_exponentiate.(-time_labels)

# # # # Sum over all time labels, t
# # # # @time @tensor oft_op[j, n] := prefactors[t] *
# # # #                 (hamiltonian.eigvecs[j, a] * time_evo_phasevector_t[t, a]) * 
# # # #                 jump.in_eigenbasis[a, b] * 
# # # #                 (adjoint(time_evo_phasevector_t)[t, b] * adjoint(hamiltonian.eigvecs)[b, n])


# # # # @time @tensor begin
# # # #     oft_op[j, n] := normalized_time_gaussian_factors[t] * fourier_phase_factors[t] *
# # # #                 hamiltonian.eigvecs[j, a] * time_evo_phase_for_all_times[t, a] * adjoint(hamiltonian.eigvecs)[a, k] *
# # # #                 jump.data[k, m] *
# # # #                 view(hamiltonian.eigvecs)[m, b] * adjoint(view(time_evo_phase_for_all_times))[t, b] * adjoint(view(hamiltonian.eigvecs))[b, n]
# # # # end


# # # # find_unique_jump_freqs(jump, hamiltonian)


# # # # @time construct_A_nus(jump, hamiltonian, num_energy_bits)

# # # # @time oft_op = oft(jump, energy, hamiltonian, num_energy_bits, sigma, beta)

# # # # @time entry_wise_oft_op = entry_wise_oft(jump, energy, hamiltonian, sigma, beta)

# # # # @printf("Are they the same?: %s\n", norm(oft_op - entry_wise_oft_op) < 1e-10)


# # # #* Heisenberg is weighted sum of A_nus
# # # # t = 3 * t0
# # # # heis_A = exp(1im * hamiltonian.data * t) * jump.data * exp(-1im * hamiltonian.data * t)
# # # # heis_A_in_eigenbasis = hamiltonian.eigvecs' * heis_A * hamiltonian.eigvecs
# # # # @printf("Num of unique freqs %d\n", length(jump.unique_freqs))
# # # # A_nu_sum = zeros(ComplexF64, size(jump.data))
# # # # for (nu, A_nu) in jump.bohr_decomp
# # # #     A_nu_sum += exp(1im * nu * t) * A_nu
# # # # end
# # # # println("Distance between Heisenberg and A_nu sum:")
# # # # display(frobenius_norm(heis_A_in_eigenbasis - A_nu_sum))

