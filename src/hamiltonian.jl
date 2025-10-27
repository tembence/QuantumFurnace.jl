function create_hamham(terms::Vector{Vector{Matrix{ComplexF64}}}, coeffs::Vector{Float64}, num_qubits::Int64; 
    periodic::Bool = true, hermitian_check = false)
    """Creates a HamHam object from terms and coefficients."""

    hamiltonian_matrix = construct_base_ham(terms, coeffs, num_qubits; periodic=periodic)

    rescaling_factor, shift = rescaling_and_shift_factors(hamiltonian_matrix)
    rescaled_hamiltonian::Hermitian{ComplexF64, Matrix{ComplexF64}} = hamiltonian_matrix / rescaling_factor + 
                                                                                    shift * I(2^num_qubits)

    rescaled_eigvals, rescaled_eigvecs = eigen(rescaled_hamiltonian)
    rescaled_base_coeffs = coeffs / rescaling_factor
    smallest_bohr_freq = minimum(diff(rescaled_eigvals))

    if hermitian_check
        @assert ishermitian(rescaled_hamiltonian) "The resulting matrix is not Hermitian!"
    end

    hamiltonian = HamHam(
        rescaled_hamiltonian,
        nothing,  # bohr_freqs is added later
        nothing,
        terms,
        rescaled_base_coeffs,
        nothing,  # disordering_terms absent
        nothing,  # disordering coeffs absent
        rescaled_eigvals,
        rescaled_eigvecs,
        smallest_bohr_freq,
        shift,
        rescaling_factor,
        periodic,
        Hermitian(zeros(ComplexF64, 1, 1))
   )

    return hamiltonian
end

function create_hamham(terms::Vector{Vector{Matrix{ComplexF64}}}, coeffs::Vector{Float64}, 
    disordering_terms::Vector{Matrix{ComplexF64}}, disordering_coeffs::Vector{Float64}, 
    num_qubits::Int64; periodic::Bool = true, hermitian_check = false)
    """Creates a HamHam object from terms and coefficients."""

    base_hamiltonian = construct_base_ham(terms, coeffs, num_qubits)
    disordering_hamiltonian = construct_disordering_terms(disordering_terms, disordering_coeffs, num_qubits)
    disordered_ham = base_hamiltonian + disordering_hamiltonian

    rescaling_factor, shift = rescaling_and_shift_factors(disordered_ham)
    rescaled_hamiltonian::Hermitian{ComplexF64, Matrix{ComplexF64}} = disordered_ham / rescaling_factor + 
                                                                                    shift * I(2^num_qubits)

    rescaled_eigvals, rescaled_eigvecs = eigen(rescaled_hamiltonian)
    rescaled_base_coeffs = coeffs / rescaling_factor
    rescaled_disordering_coeffs = disordering_coeffs / rescaling_factor
    smallest_bohr_freq = minimum(diff(rescaled_eigvals))

    if hermitian_check
        @assert ishermitian(rescaled_hamiltonian) "The resulting matrix is not Hermitian!"
    end

    hamiltonian = HamHam(
        rescaled_hamiltonian,
        nothing,  # bohr_freqs is added later
        nothing,
        terms,
        rescaled_base_coeffs,
        disordering_terms,
        rescaled_disordering_coeffs,
        rescaled_eigvals,
        rescaled_eigvecs,
        smallest_bohr_freq,
        shift,
        rescaling_factor,
        periodic,
        Hermitian(zeros(ComplexF64, 1, 1))
   )

    return hamiltonian
end

function zeros_hamham(num_qubits::Int64)
    """Creates a HamHam object with zero data"""
    hamiltonian = HamHam(
        zeros(0, 0),
        nothing,
        [[""]],
        zeros(0),
        nothing,
        nothing,
        zeros(0),
        zeros(0, 0),
        0.0,
        0.0,
        0.0,
        true
    )
    return hamiltonian
end

function find_ideal_heisenberg(num_qubits::Int64,
    coeffs::Vector{Float64}; batch_size::Int64 = 1, periodic::Bool = true)
    """Periodic Heisenberg 1D chain"""

    X::Matrix{ComplexF64} = [0 1; 1 0]
    Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
    Z::Matrix{ComplexF64} = [1 0; 0 -1]

    terms = [[X, X], [Y, Y], [Z, Z]]
    disordering_term = [Z]

    base_hamiltonian = construct_base_ham(terms, coeffs, num_qubits; periodic=periodic)

    # Create a bacth of random seeds
    seeds = rand(1:batch_size, batch_size)

    # Find best config for smallest bohr frequency
    best_smallest_bohr_freq = -1.0
    # initialize undef HamHam object
    hamiltonian = HamHam(zeros(0, 0), nothing, nothing, [[""]], zeros(0), [""], zeros(0), zeros(0), zeros(0, 0), 
    0.0, 0.0, 0.0, periodic, Hermitian(zeros(0, 0)))

    p = Progress(length(seeds))
    @showprogress dt=1 desc="Finding ideal hamiltonian..." for seed in seeds
        Random.seed!(seed)
        disordering_coeffs = rand(num_qubits)
        disordering_ham = construct_disordering_terms(disordering_term, disordering_coeffs, num_qubits)

        disordered_ham = base_hamiltonian + disordering_ham
        rescaling_factor, shift = rescaling_and_shift_factors(disordered_ham)

        rescaled_hamiltonian::Hermitian{ComplexF64, Matrix{ComplexF64}} = disordered_ham / rescaling_factor + 
                                                                                        shift * I(2^num_qubits)

        rescaled_eigvals, rescaled_eigvecs = eigen(rescaled_hamiltonian)
        rescaled_base_coeffs = coeffs / rescaling_factor
        rescaled_disordering_coeffs = disordering_coeffs / rescaling_factor
        # Check all differences between consecutive eigenvalues
        smallest_bohr_freq = minimum(diff(rescaled_eigvals))
        if smallest_bohr_freq > best_smallest_bohr_freq
            best_smallest_bohr_freq = smallest_bohr_freq
            hamiltonian.data = rescaled_hamiltonian
            hamiltonian.base_terms = terms_str
            hamiltonian.base_coeffs = rescaled_base_coeffs
            hamiltonian.disordering_term = disordering_term
            hamiltonian.disordering_coeffs = rescaled_disordering_coeffs
            hamiltonian.eigvals = rescaled_eigvals
            hamiltonian.eigvecs = rescaled_eigvecs
            hamiltonian.nu_min = smallest_bohr_freq
            hamiltonian.shift = shift
            hamiltonian.rescaling_factor = rescaling_factor

            next!(p, showvalues = [(:nu_min, hamiltonian.nu_min)])
        end
        symbroken_ham = nothing
    end
    # println("\nBest Bohr frequency:")
    # println(hamiltonian.nu_min)
    return hamiltonian
end

function construct_base_ham(terms::Vector{Vector{Matrix{ComplexF64}}}, coeffs::Vector{Float64},
    num_qubits::Int64; periodic::Bool = true)
    
    if length(terms) != length(coeffs)
        throw(ArgumentError("The number of terms and coefficients must be equal"))
    end
    
    hamiltonian::SparseMatrixCSC{ComplexF64} = spzeros(2^num_qubits, 2^num_qubits)
    for (i, term) in enumerate(terms)
        for q in 1:num_qubits
            padded_term = pad_term(term, num_qubits, q; periodic=periodic)  # e.g. term = XX
            hamiltonian += coeffs[i] * padded_term
        end
    end

    return Hermitian(Matrix(hamiltonian))
end

function construct_disordering_terms(term::Vector{Matrix{ComplexF64}}, 
    coeffs::Vector{Float64}, num_qubits::Int64)

    if length(coeffs) != num_qubits
        throw(ArgumentError("The number of disordering coeffs must be equal to the number of qubits"))
    end

    disordering_hamiltonian::SparseMatrixCSC{ComplexF64} = spzeros(2^num_qubits, 2^num_qubits)
    for q in 1:num_qubits
        disordering_hamiltonian += coeffs[q] * pad_term(term, num_qubits, q)
    end

    return Hermitian(Matrix(disordering_hamiltonian))
end

function pad_term(terms::Vector{Matrix{ComplexF64}}, num_qubits::Int64, position::Int; periodic::Bool = true)
    
    term_length = length(terms)
    terms = [sparse(term) for term in terms]
    last_position = position + term_length - 1
    # Drop boundary overstepping terms for aperiodic boundary condition 
    if (!(periodic) && last_position > num_qubits)
        return zeros(2^num_qubits, 2^num_qubits)
    end

    if last_position <= num_qubits
        id_before = sparse(I, 2^(position - 1), 2^(position - 1))
        id_after = sparse(I, 2^(num_qubits - last_position), 2^(num_qubits - last_position))
        padded_tensor_list = [id_before, terms..., id_after]
    else
        id_between = sparse(I, 2^(num_qubits - term_length), 2^(num_qubits - term_length))
        not_overflown_terms = terms[1:num_qubits - position + 1]
        overflown_terms = terms[num_qubits - position + 2:end]
        # println("Overflown terms:")
        # display(overflown_terms)
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


# function create_hamham(terms::Vector{Vector{String}}, coeffs::Vector{Float64}, num_qubits::Int64; 
#     periodic::Bool = true)
#     """Creates a HamHam object from terms and coefficients. Only for NN terms for now."""

#     sigmax::Matrix{ComplexF64} = [0 1; 1 0]
#     sigmay::Matrix{ComplexF64} = [0.0 -im; im 0.0]
#     sigmaz::Matrix{ComplexF64} = [1 0; 0 -1]
#     pauli_dict = Dict("X" => sigmax, "Y" => sigmay, "Z" => sigmaz, "I" => Matrix(I(2)))

#     terms_matrices::Vector{Vector{Matrix{ComplexF64}}} = []
#     for term in terms
#         push!(terms_matrices, [])
#         for pauli_str in term
#             push!(terms_matrices[end], pauli_dict[pauli_str])
#         end
#     end

#     base_hamiltonian = construct_base_ham(terms_matrices, coeffs, num_qubits; periodic=periodic)

#     rescaling_factor, shift = rescaling_and_shift_factors(base_hamiltonian)
#     rescaled_hamiltonian::Hermitian{ComplexF64, Matrix{ComplexF64}} = base_hamiltonian / rescaling_factor + 
#                                                                                     shift * I(2^num_qubits)

#     rescaled_eigvals, rescaled_eigvecs = eigen(rescaled_hamiltonian)
#     rescaled_base_coeffs = coeffs / rescaling_factor
#     smallest_bohr_freq = minimum(diff(rescaled_eigvals))

#    hamiltonian = HamHam(
#     rescaled_hamiltonian,
#     nothing,  # bohr_freqs is added later
#     nothing,
#     terms,
#     rescaled_base_coeffs,
#     nothing,  # disordering_terms absent
#     nothing,  # disordering coeffs absent
#     rescaled_eigvals,
#     rescaled_eigvecs,
#     smallest_bohr_freq,
#     shift,
#     rescaling_factor,
#     periodic,
#     Hermitian(zeros(ComplexF64, 1, 1))
#    )

#     return hamiltonian
# end

#* --- Testing
# hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=100)
# hamiltonian_terms = [["X", "X"], ["Y", "Y"], ["Z", "Z"]]
# hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
# hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits)

# num_qubits = 11

# sigmax::Matrix{ComplexF64} = [0 1; 1 0]
# sigmay::Matrix{ComplexF64} = [0 -im; im 0]
# sigmaz::Matrix{ComplexF64} = [1 0; 0 -1]

# terms = [[sigmax, sigmax], [sigmay, sigmay], [sigmaz, sigmaz]]
# coeffs = fill(1.0, 3)
# hamiltonian = construct_base_ham(terms, coeffs, num_qubits)

# disordering_ham = construct_disordering_terms([sigmaz], fill(1.0, num_qubits), num_qubits)

# symbroken_ham = hamiltonian + disordering_ham

# rescaling_factor, shift = rescaling_and_shift_factors(symbroken_ham)

# @time begin
# ideal_ham::HamHam = find_ideal_heisenberg(num_qubits, coeffs; batch_size=100)
# end

# @save "/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n11.jld" ideal_ham

# load jld
# ideal_ham = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n11.jld")["ideal_ham"]
# display(ideal_ham.nu_min)

# ideal_r = ceil(Int64, log2(1 / ideal_ham.nu_min))
# @printf("Ideal r: %d\n", ideal_r)