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

function find_ideal_heisenberg(num_qubits::Int64,
    coeffs::Vector{Float64}; batch_size::Int64 = 1, periodic::Bool = true)
    """Periodic Heisenberg 1D chain"""

    dim = 2^num_qubits
    terms = [[X, X], [Y, Y], [Z, Z]]
    disordering_term = [Z]

    base_hamiltonian = construct_base_ham(terms, coeffs, num_qubits; periodic=periodic)

    # Find best config for smallest bohr frequency
    best_nu_min = -1.0
    best_ham_matrix = Matrix{ComplexF64}(undef, 0, 0)
    best_eigvals = Float64[]
    best_eigvecs = Matrix{ComplexF64}(undef, 0, 0)
    best_shift = 0.0
    best_rescaling_factor = 1.0
    best_disordering_coeffs = Float64[]
    disordering_coeffs = zeros(Float64, num_qubits)

    p = Progress(batch_size; desc="Optimizing Heisenberg Hamiltonian...")
    for _ in 1:batch_size
        rand!(disordering_coeffs)
        disordering_ham = construct_disordering_terms(disordering_term, disordering_coeffs, num_qubits)

        total_ham = base_hamiltonian + disordering_ham
        rescaling_factor, shift = rescaling_and_shift_factors(total_ham)

        rescaled_ham = (total_ham ./ rescaling_factor) + shift * I 

        rescaled_eigvals, rescaled_eigvecs = eigen(Hermitian(rescaled_ham))
        # Check all differences between consecutive eigenvalues
        nu_min = minimum(diff(rescaled_eigvals))
        if nu_min > best_nu_min
            best_nu_min = nu_min
            best_ham_matrix = copy(rescaled_ham)
            best_disordering_coeffs = copy(disordering_coeffs)
            best_eigvals = rescaled_eigvals
            best_eigvecs = rescaled_eigvecs
            best_shift = shift
            best_rescaling_factor = rescaling_factor

            next!(p, showvalues = [(:nu_min, best_nu_min)])
        else
            next!(p)
        end
    end

    if best_nu_min < 0
        error("Optimization failed to find a valid Hamiltonian")
    end

    best_bohr_freqs = best_eigvals .- transpose(best_eigvals)

    return HamHam(
        best_ham_matrix,
        best_bohr_freqs,
        create_bohr_dict(best_bohr_freqs),
        terms,
        coeffs ./ best_rescaling_factor,
        disordering_term,
        best_disordering_coeffs ./ best_rescaling_factor,
        best_eigvals,
        best_eigvecs,
        best_nu_min,
        best_shift,
        best_rescaling_factor,
        periodic,
        nothing
    )
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

function add_gibbs_to_hamham(hamiltonian::HamHam, beta::Float64)::HamHam
    gibbs_in_eigen = Hermitian(gibbs_state_in_eigen(hamiltonian, beta))

    # Create a new HamHam with gibbs, it copies only the pointers to previous fields, not the data themselves.
    return HamHam(
        hamiltonian.data,
        hamiltonian.bohr_freqs,
        hamiltonian.bohr_dict,
        hamiltonian.base_terms,
        hamiltonian.base_coeffs,
        hamiltonian.disordering_term,
        hamiltonian.disordering_coeffs,
        hamiltonian.eigvals,
        hamiltonian.eigvecs,
        hamiltonian.nu_min,
        hamiltonian.shift,
        hamiltonian.rescaling_factor,
        hamiltonian.periodic,
        gibbs_in_eigen
    )
    
end

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