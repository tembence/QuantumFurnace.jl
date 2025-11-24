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
        nothing,
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
        nothing
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
        nothing,
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
        nothing
   )

    return hamiltonian
end

"""find_ideal_heisenberg(num_qubits::Int, coeffs::Vector{Float64}; 
    batch_size::Int=1, periodic::Bool=true) -> HamHam

    Constructs and optimizes a disordered 1D Heisenberg Hamiltonian to maximize the minimum level spacing (smallest Bohr frequency).

    The function generates `batch_size` random realizations of a disordering ``Z``-field. For each realization, it constructs the Hamiltonian:
    ```math
    H = H_{base} + H_{disorder}
    ```
    where ``H_{base}`` is the Heisenberg chain defined by `coeffs` (XX, YY, ZZ interaction strengths) and ``H_{disorder}`` is a site-dependent ``Z`` term with random coefficients.

    The Hamiltonian is rescaled and shifted to ensure the spectrum fits within specific bounds.

    # Arguments
    - `num_qubits`: The number of sites on the spin chain.
    - `coeffs`: A vector of the uniform interaction strengths for ``\\sigma_x \\sigma_x``, ``\\sigma_y \\sigma_y``, and ``\\sigma_z \\sigma_z`` terms respectively.

    # Keywords
    - `batch_size`: The number of random disorder configurations to sample (default: 1).
    - `periodic`: If `true`, applies periodic boundary conditions to the chain.

    # Returns
    - `HamHam`: A container holding the optimized Hamiltonian data, spectral decomposition, and Bohr frequencies, etc.
    **Note**: The `gibbs` field of the returned struct is set to `nothing`. Use [`add_gibbs_to_hamham`](@ref) to calculate the thermal state once the temperature is decided.
"""
function find_ideal_heisenberg(num_qubits::Int64,
    coeffs::Vector{Float64}; batch_size::Int64 = 1, periodic::Bool = true)


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

    return HamHam(
        best_ham_matrix,
        nothing,
        nothing,
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
"""
    Creates a new HamHam struct from an old one with `bohr_freqs`, `bohr_dict`, and `gibbs`
    It copies only the pointers to previous fields, not the data themselves
"""
function finalize_hamham(hamiltonian::HamHam, beta::Float64)::HamHam

    gibbs_in_eigen = Hermitian(gibbs_state_in_eigen(hamiltonian, beta))

    bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
    
    return HamHam(
        hamiltonian.data,
        bohr_freqs,
        create_bohr_dict(bohr_freqs),
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