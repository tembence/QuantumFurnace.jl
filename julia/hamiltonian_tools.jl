using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Debugger

mutable struct HamHam
    data::Matrix{ComplexF64}
    base_coeffs::Vector{Float64}
    symbreak_coeffs::Vector{Float64}
    eigvals::Vector{Float64}
    eigvecs::Matrix{ComplexF64}
    w0::Float64  # Smallest bohr frequency
    shift::Float64
    rescaling_factor::Float64
end

function find_ideal_heisenberg(num_qubits::Int64; batch_size::Int64 = 100)

end


function find_ideal_heisenberg(num_qubits::Int64,
    fixed_base_coeffs::Vector{Float64}; batch_size::Int64 = 100)
    #? Could add bohr_bound option, to optimize until it gets above the bohr bound

    sigmax::Matrix{ComplexF64} = [0 1; 1 0]
    sigmay::Matrix{ComplexF64} = [0.0 -im; im 0.0]
    sigmaz::Matrix{ComplexF64} = [1 0; 0 -1]
    terms = [[sigmax, sigmax], [sigmay, sigmay], [sigmaz, sigmaz]]

    base_hamiltonian = construct_base_ham(terms, fixed_base_coeffs, num_qubits)

    # Create a bacth of random seeds
    seeds = rand(1:batch_size, batch_size)

    # Find best config for smallest bohr frequency
    best_smallest_bohr_freq = 0.
    hamiltonian = HamHam(zeros(0,0), zeros(0), zeros(0), zeros(0), zeros(0,0), 0.0, 0.0, 0.0)

    @showprogress dt=1 desc="Finding ideal hamiltonian..." for seed in seeds
        Random.seed!(seed)
        symbreak_coeffs = 2.0 .* rand(num_qubits) .- 1.0
        symbreak_ham = construct_symbreak_terms([sigmaz], symbreak_coeffs, num_qubits)

        symbroken_ham = base_hamiltonian + symbreak_ham
        rescaling_factor, shift = rescaling_and_shift_factors(symbroken_ham)

        rescaled_hamiltonian::Hermitian{ComplexF64, Matrix{ComplexF64}} = symbroken_ham / rescaling_factor + 
                                                                                        shift * I(2^num_qubits)

        rescaled_eigvals, rescaled_eigvecs = eigen(rescaled_hamiltonian)
        rescaled_base_coeffs = fixed_base_coeffs / rescaling_factor
        rescaled_symbreak_coeffs = symbreak_coeffs / rescaling_factor

        # Check all differences between consecutive eigenvalues
        smallest_bohr_freq = minimum(diff(rescaled_eigvals))
        if smallest_bohr_freq > best_smallest_bohr_freq
            best_smallest_bohr_freq = smallest_bohr_freq
            hamiltonian::HamHam = HamHam(rescaled_hamiltonian, rescaled_base_coeffs, rescaled_symbreak_coeffs,
                rescaled_eigvals, rescaled_eigvecs, smallest_bohr_freq, shift, rescaling_factor)
        end
    end

    @printf("Smallest bohr frequency: %f\n", hamiltonian.w0)
    # println("Spectrum is:")
    # println(hamiltonian.eigvals)
    return hamiltonian
end

function construct_base_ham(terms::Vector{Vector{Matrix{ComplexF64}}}, coeffs::Vector{Float64},
    num_qubits::Int64)
    
    if length(terms) != length(coeffs)
        throw(ArgumentError("The number of terms and coefficients must be equal"))
    end

    hamiltonian::SparseMatrixCSC{ComplexF64} = spzeros(2^num_qubits, 2^num_qubits)
    for (i, term) in enumerate(terms)
        for q in 1:num_qubits
            padded_term = pad_term(term, num_qubits, q)
            hamiltonian += coeffs[i] * padded_term
        end
    end
    
    return Hermitian(Matrix(hamiltonian))
end

function construct_symbreak_terms(symbreak_term::Vector{Matrix{ComplexF64}}, 
    num_qubits::Int64, rand_seed::Int64 = 666)

    # set random seed
    Random.seed!(rand_seed)
    symbreak_coeffs = 2.0 .* rand(num_qubits) .- 1.0

    symbreak_hamiltonian::SparseMatrixCSC{ComplexF64} = spzeros(2^num_qubits, 2^num_qubits)

    for q in 1:num_qubits
        padded_term = pad_term(symbreak_term, num_qubits, q)
        symbreak_hamiltonian += symbreak_coeffs[q] * padded_term
    end

    return Hermitian(Matrix(symbreak_hamiltonian))
end

function construct_symbreak_terms(symbreak_term::Vector{Matrix{ComplexF64}}, 
    symbreak_coeffs::Vector{Float64}, num_qubits::Int64)

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

function rescaling_and_shift_factors(hamiltonian::Hermitian{ComplexF64, Matrix{ComplexF64}})
    """Computes rescaling and shifting factors for a Hamiltonian, s.t. the spectrum is in ``[0, 0.5*(1-ϵ)]`` """

    eps = 0.1  # to avoid 0.5 ~ 0.0 in algorithm
    eigenergies = eigvals(hamiltonian)
    smallest_eigval = minimum(eigenergies)
    largest_eigval = maximum(eigenergies)

    rescaling_factor = (largest_eigval - smallest_eigval) * (2 / (1 - eps))
    shift = - (largest_eigval - smallest_eigval * eps) / (2 * (largest_eigval - smallest_eigval)) + 0.5
    return rescaling_factor, shift

end

# --- Testing
# num_qubits = 8

# sigmax::Matrix{ComplexF64} = [0 1; 1 0]
# sigmay::Matrix{ComplexF64} = [0.0 -im; im 0.0]
# sigmaz::Matrix{ComplexF64} = [1 0; 0 -1]
# terms = [[sigmax, sigmax], [sigmay, sigmay], [sigmaz, sigmaz]]

# coeffs = [1.0, 1.0, 1.0]

# hamiltonian = construct_base_ham(terms, coeffs, num_qubits)

# symbreak_ham = construct_symbreak_terms([sigmaz], fill(1.0, num_qubits), num_qubits)

# symbroken_ham = hamiltonian + symbreak_ham

# rescaling_factor, shift = rescaling_and_shift_factors(symbroken_ham)

# ideal_ham::HamHam = find_ideal_heisenberg(num_qubits, coeffs, batch_size=1000)
# display(ideal_ham.w0)