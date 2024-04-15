using LinearAlgebra
using Debugger
using SparseArrays
using Random

struct HamHam
    H::Matrix{ComplexF64}
    shift::Float64
    rescaling_factor::Float64
    eigvals::Vector{Float64}
    eigvecs::Matrix{ComplexF64}
end

# function find_ideal_heisenberg()
#     ...
# end

function construct_base_ham(terms::Vector{Vector{Matrix{ComplexF64}}}, coeffs::Vector{Float64},
    num_qubits::Int64)
    
    if length(terms) != length(coeffs)
        throw(ArgumentError("The number of terms and coefficients must be equal"))
    end

    hamiltonian::SparseMatrixCSC{ComplexF64} = spzeros(2^num_qubits, 2^num_qubits)
    for (coeff_i, term) in enumerate(terms)
        for q in 1:num_qubits
            padded_term = pad_term(term, num_qubits, q)
            hamiltonian += coeffs[coeff_i] * padded_term
        end
    end
    
    return Hermitian(Matrix(hamiltonian))
end

function construct_symbreak_terms(symbreak_term::Vector{Matrix{ComplexF64}}, 
    num_qubits::Int64, rand_seed::Int64 = 666)

    # set random seed
    Random.seed!(rand_seed)
    symbreak_coeffs = [rand(-1.:1., length(num_qubits)) for term in eachindex(symbreak_term)]
    println(symbreak_coeffs)

    symbreak_hamiltonian::SparseMatrixCSC{ComplexF64} = spzeros(2^num_qubits, 2^num_qubits)
    for q in 1:num_qubits
        padded_term = pad_term(symbreak_term, num_qubits, q)
        symbreak_hamiltonian += symbreak_coeffs[q] * padded_term
    end

    return Hermitian(Matrix(symbreak_hamiltonian))
end

function construct_symbreak_terms(symbreak_term::Vector{Matrix{ComplexF64}}, 
    symbreak_coeffs::Vector{Vector{Float64}}, num_qubits::Int64)

    println(symbreak_coeffs)

    if length(symbreak_coeffs) != num_qubits
        throw(ArgumentError("The number of symmetry breaking coeffs must be equal to the number of qubits"))
    end

    symbreak_hamiltonian::SparseMatrixCSC{ComplexF64} = spzeros(2^num_qubits, 2^num_qubits)
    for q in 1:num_qubits
        padded_term = pad_term(symbreak_term, num_qubits, q)
        symbreak_hamiltonian += symbreak_coeffs[q] * padded_term
    end

    return Hermitian(Matrix(symbreak_hamiltonian))
end

function pad_term(terms::Vector{Matrix{ComplexF64}}, num_qubits::Int64, position::Int)
    
    term_length = length(terms)
    #turn terms into sparse
    terms = [sparse(term) for term in terms]
    last_position = position + term_length - 1
    if last_position <= num_qubits
        id_before = sparse(I, 2^(position - 1), 2^(position - 1))
        id_after = sparse(I, 2^(num_qubits - last_position), 2^(num_qubits - last_position))
        padded_tensor_list = [id_before, terms..., id_after]
    else
        id_between = sparse(I, 2^(num_qubits - term_length), 2^(num_qubits - term_length))
        not_overflown_terms = terms[1:num_qubits - position + 1]
        overflown_terms = terms[num_qubits - position + 2:end]
        padded_tensor_list = [overflown_terms..., id_between, not_overflown_terms...]
    end

    padded_term::SparseMatrixCSC{ComplexF64} = kron(padded_tensor_list...)
    return padded_term
end

function rescale_shift_factors(hamiltonian::Hermitian{ComplexF64, Matrix{ComplexF64}})
    
    eigenergies = eigvals(hamiltonian)
    println(eigenergies)

end

num_qubits = 8

sigmax::Matrix{ComplexF64} = [0 1; 1 0]
sigmay::Matrix{ComplexF64} = [0.0 -im; im 0.0]
sigmaz::Matrix{ComplexF64} = [1 0; 0 -1]
terms = [[sigmax, sigmax], [sigmay, sigmay], [sigmaz, sigmaz]]

coeffs = [1.0, 1.0, 1.0]

hamiltonian = construct_base_ham(terms, coeffs, num_qubits)

symbreak_ham = construct_symbreak_terms([sigmax], num_qubits)

symbroken_ham = hamiltonian + symbreak_ham

rescale_shift_factors(symbroken_ham)