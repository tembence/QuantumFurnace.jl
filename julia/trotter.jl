using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using QuantumOptics

include("hamiltonian_tools.jl")
include("qi_tools.jl")


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

function trotter(hamiltonian::HamHam, T::Float64, num_trotter_steps::Int64)
    """1st order Trotter, periodic"""

    timestep::Float64 = T / num_trotter_steps
    num_qubits::Int64 = Int(log2(size(hamiltonian.data)[1]))

    U::Matrix{ComplexF64} = exp(im * T * hamiltonian.shift) * I(2^num_qubits)  # Shift
    p = Progress(num_trotter_steps)
    @showprogress dt=1 desc="Trotterizing (1st order)..." for step in 1:num_trotter_steps
        # println("---Step: ", step)

        # Base Hamiltonian
        for q in 1:num_qubits
            # println("Qubit: ", q)
            for (i, term) in enumerate(hamiltonian.base_terms)
                # println("Term:")
                # display(term)
                    expm_pauli_term = expm_pauli_padded(term, timestep * hamiltonian.base_coeffs[i], num_qubits, q)
                    U *= expm_pauli_term
            end

        # Symbreak
            if typeof(hamiltonian.symbreak_terms) != Nothing
                # println("Symbreak term:")
                # display(hamiltonian.symbreak_terms)
                # println("Qubit: ", q)
                expm_symbreak_pauli_term = expm_pauli_padded(hamiltonian.symbreak_terms, 
                                                            timestep * hamiltonian.symbreak_coeffs[q], 
                                                            num_qubits, q)
                U *= expm_symbreak_pauli_term
            end
        end
    end
    return U
end

function trotter2(hamiltonian::HamHam, T::Float64, num_trotter_steps::Int64)
    """2nd order Trotter"""
    timestep::Float64 = T / num_trotter_steps
    num_qubits::Int64 = Int(log2(size(hamiltonian.data)[1]))

    U::Matrix{ComplexF64} = exp(im * T * hamiltonian.shift) * I(2^num_qubits)  # Shift
    p = Progress(num_trotter_steps)
    @showprogress dt=1 desc="Trotterizing (2nd order)..." for step in 1:num_trotter_steps
        # println("---Step: ", step)

        ## 1st part
        for q in 1:num_qubits
            # println("Qubit: ", q)
            # Base Hamiltonian
            for (i, term) in enumerate(hamiltonian.base_terms)
                # println("Term:")
                # display(term)
                    expm_pauli_term = expm_pauli_padded(term, timestep * hamiltonian.base_coeffs[i] / 2, num_qubits, q)
                    U *= expm_pauli_term
            end

            # Symbreak
            if typeof(hamiltonian.symbreak_terms) != Nothing
                # println("Symbreak term:")
                # display(hamiltonian.symbreak_terms)
                # println("Qubit: ", q)
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
            # println("Qubit: ", q)

            # Symbreak, terms are not reversed as we just assume there is only 1 term
            if typeof(hamiltonian.symbreak_terms) != Nothing
                # println("Symbreak term:")
                # display(hamiltonian.symbreak_terms)
                # println("Qubit: ", q)
                # println("Coeff reversed: ", hamiltonian.symbreak_coeffs[q])
                expm_symbreak_pauli_term = expm_pauli_padded(hamiltonian.symbreak_terms, 
                                                            timestep * hamiltonian.symbreak_coeffs[q] / 2, 
                                                            num_qubits, q)
                U *= expm_symbreak_pauli_term
            end

            # Base Hamiltonian
            for (i, term) in enumerate(reversed_base_terms)
                # println("Term:")
                # display(term)
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

    trott_1step = trotter(hamiltonian, T/num_trotter_steps, 1)
    eig_vals, eig_vecs = eigen(trott_1step)
    return eig_vecs * Diagonal(eig_vals)^num_trotter_steps * eig_vecs'
end

function trotter2_diag(hamiltonian::HamHam, T::Float64, num_trotter_steps::Int64)
    """2nd order Trotter with diagonalization"""
    timestep::Float64 = T / num_trotter_steps

    trott2_1step = trotter2(hamiltonian, T/num_trotter_steps, 1)
    eig_vals, eig_vecs = eigen(trott2_1step)
    return eig_vecs * Diagonal(eig_vals)^num_trotter_steps * eig_vecs'
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
# T = 100.
# num_trotter_steps = 20

# terms = [["X", "Y"], ["Z", "X"]]
# symbreak = ["Z"]
# coeffs = [2.2, 3.9]
# symbreak_coeffs = rand(num_qubits)
# hamiltonian = create_hamham(terms, coeffs, symbreak, symbreak_coeffs, num_qubits)

# # Exact
# exact_U = exp(im * T * hamiltonian.data)
# # Trotter
# trotter2_U = trotter2(hamiltonian, T, num_trotter_steps)
# trotter_U = trotter(hamiltonian, T, 2*num_trotter_steps)

# dist = norm(trotter_U - exact_U)
# dist2 = norm(trotter2_U - exact_U)

# @printf("Distance: %f\n", dist)
# @printf("Distance 2nd order: %f\n", dist2)

#* Parameters
num_qubits = 5
mixing_time = 1
beta = 1.
num_trotter_steps = 100

#* Hamiltonian
hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n5.jld")["ideal_ham"]
# hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3), batch_size=1)

#* Fourier labels
num_energy_bits = ceil(Int64, log2((0.45 * 2) / hamiltonian.w0)) + 1  # paper (above 3.7.), later will be β dependent
N = 2^(num_energy_bits)
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

t0 = 2 * pi / (N * hamiltonian.w0)
time_labels = t0 * N_labels
T = minimum(time_labels)
T = 1.
# T = time_labels[Int(N/4 + 2)]
# T = 100.
@printf("T: %f\n", T)
energy_labels = hamiltonian.w0 * N_labels

#* Trotter
# @time trott = trotter(hamiltonian, T, num_trotter_steps)
@printf("Trotterizing with diag...\n")
@time begin
    trott2_1step = trotter2(hamiltonian, T/num_trotter_steps, 1)
    eig_vals, eig_vecs = eigen(trott2_1step)
    diag_trott2 = Diagonal(eig_vals)
    trott2_with_diag = eig_vecs * diag_trott2^num_trotter_steps * eig_vecs'
end
@printf("Trotterizing without diag...\n")
# @time trott2 = trotter2(hamiltonian, T, num_trotter_steps)

# dist_trotts = norm(trott2 - trott2_with_diag)

#* Exact time evolution 
@time U_exact = exp(im * T * hamiltonian.data)

# dist = norm(trott - U_exact)
# dist2 = norm(trott2 - U_exact)
dist2_diag = norm(trott2_with_diag - U_exact)
# @printf("Distance: %f\n", dist)
# @printf("Distance 2nd order: %s\n", dist2)
@printf("Distance 2nd order with diag: %s\n", dist2_diag)
