using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed

include("hamiltonian.jl")
include("qi_tools.jl")

#! Maybe 1 Trotter step / t0 is still too good, because it deviates from OFT's by 1e-3 - 1e-4
function create_trotter(hamiltonian::HamHam, t0::Float64, num_trotter_steps_per_t0::Int64)

    trott2_1step_of_t0 = trotterize2(hamiltonian, t0/num_trotter_steps_per_t0, 1)
    trott_eigvals, trott_eigvecs = eigen(trott2_1step_of_t0)
    trott2_t0_eigvals = trott_eigvals.^num_trotter_steps_per_t0
    unitary_from_eigen_to_trotter = trott_eigvecs' * hamiltonian.eigvecs
    return TrottTrott(t0, num_trotter_steps_per_t0, trott2_t0_eigvals, trott_eigvecs, unitary_from_eigen_to_trotter)
end

function trotttrotterize2(trotter::TrottTrott, T::Float64)
    num_t0_steps = Int(ceil(T / trotter.t0))
    return trotter.eigvecs * Diagonal(trotter.eigvals_t0.^num_t0_steps) * trotter.eigvecs'
end

function compute_trotter_error(hamiltonian::HamHam, trotter::TrottTrott, T::Float64)
    
    num_t0_steps = Int(ceil(T / trotter.t0))
    exact_time_evolution = Diagonal(exp.(1im * hamiltonian.eigvals * T))  # In energy eigenbasis
    trotter_time_evolution = Diagonal(trotter.eigvals_t0.^num_t0_steps)
    trotter_time_evolution = ( hamiltonian.eigvecs' * trotter.eigvecs 
                                * trotter_time_evolution * trotter.eigvecs' * hamiltonian.eigvecs)
    return norm(exact_time_evolution - trotter_time_evolution)
end

function expm_pauli_padded(paulistring::Vector{String}, coeff::Float64, num_qubits::Int64, position::Int64)
    """Arg e.g. NN terms: ["X", "X"], and it pads it with identities in the rest of the sites. Then creates the expm."""

    pauli_term = pauli_string_to_matrix(paulistring)
    padded_term = pad_term(pauli_term, num_qubits, position)
    expm = cos(coeff) * I(2^num_qubits) + im * sin(coeff) * padded_term
    return expm
end

function expm_pauli(paulistring::Vector{String}, coeff::Float64)
    """Arg e.g. ["X", "X", "Z", "I"]"""

    num_qubits = length(paulistring)
    sigmax::Matrix{ComplexF64} = [0 1; 1 0]
    sigmay::Matrix{ComplexF64} = [0.0 -im; im 0.0]
    sigmaz::Matrix{ComplexF64} = [1 0; 0 -1]
    id::Matrix{ComplexF64} = [1 0; 0 1]

    pauli_matrices::Vector{SparseMatrixCSC{ComplexF64}} = []
    pauli_dict = Dict("X" => sparse(sigmax), "Y" => sparse(sigmay), "Z" => sparse(sigmaz), "I" => sparse(id))
    for pauli_str in paulistring
        push!(pauli_matrices, pauli_dict[pauli_str])
    end

    expm = cos(coeff) * sparse(I(2^num_qubits)) + im * sin(coeff) * kron(pauli_matrices...)
    return expm
end

function pauli_string_to_matrix(paulistring::Vector{String})
    sigmax::Matrix{ComplexF64} = [0 1; 1 0]
    sigmay::Matrix{ComplexF64} = [0.0 -im; im 0.0]
    sigmaz::Matrix{ComplexF64} = [1 0; 0 -1]

    pauli_matrices::Vector{Matrix{ComplexF64}} = []
    pauli_dict = Dict("X" => sigmax, "Y" => sigmay, "Z" => sigmaz, "I" => Matrix{ComplexF64}(I(2)))
    for pauli_str in paulistring
        push!(pauli_matrices, pauli_dict[pauli_str])
    end
    return pauli_matrices
end

function trotterize(hamiltonian::HamHam, T::Float64, num_trotter_steps::Int64)
    """1st order Trotter, periodic"""

    timestep::Float64 = T / num_trotter_steps
    num_qubits::Int64 = Int(log2(size(hamiltonian.data)[1]))

    U::Matrix{ComplexF64} = exp(im * T * hamiltonian.shift) * I(2^num_qubits)  # Shift
    p = Progress(num_trotter_steps)
    @showprogress dt=1 desc="Trotterizing (1st order)..." for step in 1:num_trotter_steps
        # Base Hamiltonian
        for q in 1:num_qubits
            for (i, term) in enumerate(hamiltonian.base_terms)
                    expm_pauli_term = expm_pauli_padded(term, timestep * hamiltonian.base_coeffs[i], num_qubits, q)
                    U *= expm_pauli_term
            end

        # Symbreak
            if typeof(hamiltonian.symbreak_terms) != Nothing
                expm_symbreak_pauli_term = expm_pauli_padded(hamiltonian.symbreak_terms, 
                                                            timestep * hamiltonian.symbreak_coeffs[q], 
                                                            num_qubits, q)
                U *= expm_symbreak_pauli_term
            end
        end
    end
    return U
end

function trotterize2(hamiltonian::HamHam, T::Float64, num_trotter_steps::Int64)
    """2nd order Trotter"""
    timestep::Float64 = T / num_trotter_steps
    num_qubits::Int64 = Int(log2(size(hamiltonian.data)[1]))

    U::Matrix{ComplexF64} = exp(im * T * hamiltonian.shift) * I(2^num_qubits)  # Shift
    p = Progress(num_trotter_steps)
    @showprogress dt=1 desc="Trotterizing (2nd order)..." for step in 1:num_trotter_steps
        ## 1st part
        for q in 1:num_qubits
            for (i, term) in enumerate(hamiltonian.base_terms)
                    expm_pauli_term = expm_pauli_padded(term, timestep * hamiltonian.base_coeffs[i] / 2, num_qubits, q)
                    U *= expm_pauli_term
            end

            # Symbreak
            if typeof(hamiltonian.symbreak_terms) != Nothing
                expm_symbreak_pauli_term = expm_pauli_padded(hamiltonian.symbreak_terms, 
                                                            timestep * hamiltonian.symbreak_coeffs[q] / 2, 
                                                            num_qubits, q)
                U *= expm_symbreak_pauli_term
            end
        end

        ## 2nd part
        reversed_base_terms = reverse(hamiltonian.base_terms)
        reversed_base_coeffs = reverse(hamiltonian.base_coeffs)

        for q in num_qubits:-1:1
            # Symbreak, terms are not reversed as we just assume there is only 1 term
            if typeof(hamiltonian.symbreak_terms) != Nothing
                expm_symbreak_pauli_term = expm_pauli_padded(hamiltonian.symbreak_terms, 
                                                            timestep * hamiltonian.symbreak_coeffs[q] / 2, 
                                                            num_qubits, q)
                U *= expm_symbreak_pauli_term
            end

            # Base Hamiltonian
            for (i, term) in enumerate(reversed_base_terms)
                    expm_pauli_term = expm_pauli_padded(term, timestep * reversed_base_coeffs[i] / 2, num_qubits, q)
                    U *= expm_pauli_term
            end
        end
    end
    return U
end

function trotter_diag(hamiltonian::HamHam, T::Float64, num_trotter_steps::Int64)
    """1st order Trotter with diagonalization"""
    timestep::Float64 = T / num_trotter_steps

    trott_1step = trotterize(hamiltonian, timestep, 1)
    eig_vals, eig_vecs = eigen(trott_1step)
    return eig_vecs * Diagonal(eig_vals)^num_trotter_steps * eig_vecs'
end

function trotter2_diag(hamiltonian::HamHam, T::Float64, num_trotter_steps::Int64)
    """2nd order Trotter with diagonalization"""
    timestep::Float64 = T / num_trotter_steps

    trott2_1step = trotterize2(hamiltonian, timestep, 1)
    eig_vals, eig_vecs = eigen(trott2_1step)
    return eig_vecs * Diagonal(eig_vals)^num_trotter_steps * eig_vecs'
end

function trotter2_t0_multiple(int_multiple::Int64, trott_t0_eigvals::Vector{ComplexF64})
    """Trotter 2nd order with integer mutiples of t0 Trotter, to avoid diagonalization too many times.
    NOTE: it's in Trotter eigenbasis"""
    return Diagonal(trott_t0_eigvals)^int_multiple
end

#*#*#* TESTING *#*#*#

# Pauli matrix exponentiation
# sigmax::Matrix{ComplexF64} = [0 1; 1 0]
# sigmay::Matrix{ComplexF64} = [0.0 -im; im 0.0]
# sigmaz::Matrix{ComplexF64} = [1 0; 0 -1]
# id::Matrix{ComplexF64} = [1 0; 0 1]

# paulistring = ["X", "X", "I", "I", "I", "I", "I", "I", "I", "I", "I", "I"]
# coeff = 3.9
# @time expm_myway = expm_pauli(paulistring, coeff)
# pauli_matrices = [sigmax, sigmax, id, id, id, id, id, id, id, id, id, id]

# @time expm_normally = exp(im * coeff * kron(pauli_matrices...))

# paulistring_nonpadded = ["X", "X"]

# @time expm_myway_padded = expm_pauli_padded(paulistring_nonpadded, coeff, 12, 1)

# @printf("Norm: %f\n", frobenius_norm(expm_myway - expm_normally))
# @printf("Norm padded: %f\n", frobenius_norm(expm_normally - expm_myway_padded))


#* Small Hamiltonian time evolution Test
# sigmax::Matrix{ComplexF64} = [0 1; 1 0]
# sigmay::Matrix{ComplexF64} = [0.0 -im; im 0.0]
# sigmaz::Matrix{ComplexF64} = [1 0; 0 -1]

# num_qubits = 5
# T = 1.
# num_trotter_steps = 1

# terms = [["X", "Y"], ["Z", "X"]]
# symbreak = ["Z"]
# coeffs = [2.2, 3.9]
# symbreak_coeffs = rand(num_qubits)
# hamiltonian = create_hamham(terms, coeffs, symbreak, symbreak_coeffs, num_qubits)

# # Exact
# exact_U = exp(im * T * hamiltonian.data)
# # Trotter
# trotter2_U = trotterize2(hamiltonian, T, num_trotter_steps)
# trotter_U = trotterize(hamiltonian, T, num_trotter_steps)

# dist = norm(trotter_U - exact_U)
# dist2 = norm(trotter2_U - exact_U)

# @printf("Distance: %s\n", dist)
# @printf("Distance 2nd order: %s\n", dist2)

#* Parameters
# num_qubits = 3
# mixing_time = 40.
# delta = 1.
# num_liouv_steps = Int(mixing_time / delta)
# beta = 10.
# eig_index = 3

# #* Hamiltonian
# hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n3.jld")["ideal_ham"]
# random_dm = random_density_matrix(num_qubits)
# num_energy_bits = ceil(Int64, log2((0.45 * 4 + 2/beta) / hamiltonian.nu_min)) - 3 # Under Fig. 5. with secular approx.
# N = 2^(num_energy_bits)
# N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
# t0 = 2 * pi / (N * hamiltonian.nu_min)
# factor = -4 
# num_steps = 10

# U = exp(im * factor * t0 * hamiltonian.data)
# trotter = trotterize2(hamiltonian, factor * t0, abs(factor) * num_steps)
# dist = norm(trotter - U)

# evolved_dm = U * random_dm * U'
# trotter_evolved_dm = trotter * random_dm * trotter'
# b = SpinBasis(1//2)^num_qubits
# trdist = tracedistance_nh(Operator(b, evolved_dm), Operator(b, trotter_evolved_dm))
# @printf("After -4t0 evolution: %f\n", trdist)
# factor = 3
# U = exp(im * factor * t0 * hamiltonian.data)
# trotter = trotterize2(hamiltonian, factor * t0, abs(factor) * num_steps)
# evolved_again_dm = U * evolved_dm * U'
# trotter_evolved_again_dm = trotter * trotter_evolved_dm * trotter'

# trdist2 = tracedistance_nh(Operator(b, evolved_again_dm), Operator(b, trotter_evolved_again_dm))
# @printf("After 3t0 evolution: %f\n", trdist2)

# # #* Fourier labels
# # num_energy_bits = ceil(Int64, log2((0.45 * 2) / hamiltonian.nu_min)) + 1  # paper (above 3.7.), later will be β dependent
# # N = 2^(num_energy_bits)
# # N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

# # t0 = 2 * pi / (N * hamiltonian.nu_min)
# # time_labels = t0 * N_labels
# # energy_labels = hamiltonian.nu_min * N_labels

# # int_multiple = Int(N / 2)
# # T_plus = int_multiple * t0
# # T_minus = - T_plus

# # #* Trotter
# # # @time trott = trotter(hamiltonian, T, num_trotter_steps)
# # # @time begin
# # #     trott2_1step = trotter2(hamiltonian, T/num_trotter_steps, 1)
# # #     eig_vals, eig_vecs = eigen(trott2_1step)
# # #     diag_trott2 = Diagonal(eig_vals)
# # #     trott2_with_diag = eig_vecs * diag_trott2^num_trotter_steps * eig_vecs'
# # # end
# # # @time trott2 = trotter2(hamiltonian, T, num_trotter_steps)

# # # dist_trotts = norm(trott2 - trott2_with_diag)

# # #* Exact time evolution 
# # @time U_exact_plus = exp(im * T_plus * hamiltonian.data)
# # U_exact_minus = exp(im * T_minus * hamiltonian.data)

# # # dist = norm(trott - U_exact)
# # # dist2 = norm(trott2 - U_exact)
# # # dist2_diag = norm(trott2_with_diag - U_exact)
# # # @printf("Distance: %f\n", dist)
# # # @printf("Distance 2nd order: %s\n", dist2)
# # # @printf("Distance 2nd order with diag: %s\n", dist2_diag)

# # #* Testing fix number of trotter steps / t0, for different T
# # @printf("w0: %f\n", hamiltonian.nu_min)
# # @printf("T: %f\n", T_plus)
# # @printf("t0: %f\n", t0)
# # num_trotter_steps_per_t0 = 10

# # num_trotter_steps = int_multiple * num_trotter_steps_per_t0

# # @time trotter_plus = create_trotter(hamiltonian, t0, num_trotter_steps_per_t0)
# # @time trotter_minus = create_trotter(hamiltonian, -t0, num_trotter_steps_per_t0)

# # # How are negative and positive trotter eigvals related
# # conj_dist = norm(trotter_plus.eigvals_t0 - conj.(trotter_minus.eigvals_t0))
# # @printf("Conjugating the eigvals of + gets us - ... dist: %s\n", conj_dist)
# # minus_norm = norm(trotter_plus.eigvals_t0 + trotter_minus.eigvals_t0)

# # @time trott_error_plus = compute_trotter_error(hamiltonian, trotter_plus, T_plus)
# # @time trott_error_minus = compute_trotter_error(hamiltonian, trotter_minus, T_minus)
# # @printf("Trotter error +: %s\n", trott_error_plus)
# # @printf("Trotter error -: %s\n", trott_error_minus)

# # trott2_to_T_plus = trotttrotterize2(trotter_plus, T_plus)
# # trott2_to_T_minus = trotttrotterize2(trotter_minus, T_minus)

# # adjoint_trott2_to_T_plus = trott2_to_T_plus'
# # adj_dist = norm(trott2_to_T_minus - adjoint_trott2_to_T_plus)

# # rewinded_trott = trott2_to_T_plus * trott2_to_T_minus
# # rewinded_exact = U_exact_plus * U_exact_minus
# # dist_to_id_trott = norm(rewinded_trott - I(2^num_qubits))
# # @printf("Distance to identity Trott rewind: %s\n", dist_to_id_trott)
# # dist_to_id_exact = norm(rewinded_exact - I(2^num_qubits))
# # @printf("Distance to identity Exact rewind: %s\n", dist_to_id_exact)

# # #* With jump
# # jump_site_index = 1
# # sigmax::Matrix{ComplexF64} = [0 1; 1 0]
# # jump_op = Matrix(pad_term([sigmax], num_qubits, jump_site_index))

# # oft_core_trotter = trott2_to_T_plus * jump_op * trott2_to_T_minus
# # oft_core_exact = U_exact_plus * jump_op * U_exact_minus

# # dist_oft_cores = norm(oft_core_trotter - oft_core_exact)
# # @printf("Distance of OFT Trotter vs Exact: %s\n", dist_oft_cores)