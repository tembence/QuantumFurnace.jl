"""
    pauli_hamiltonian(n, terms) -> Hermitian

Build a sparse physical Hamiltonian on `n` qubits. Each term is
`coefficient => (site => :Pauli, ...)`, with labels `:I`, `:X`, `:Y`, `:Z`.
Site 1 is the leftmost Kronecker factor; sites need not be contiguous or sorted.
An empty site tuple denotes the identity. Repeated terms add, but repeated
sites within a term are invalid. Coefficients must be finite real numbers.
An empty term collection gives zero. No rescaling or diagonalisation is done.

```julia
H = pauli_hamiltonian(3, [-1.0 => (1 => :Z, 3 => :Z),
                          0.7 => (2 => :X,), 2.0 => ()])
```
"""
function pauli_hamiltonian(n::Integer, terms)
    0 < n < Sys.WORD_SIZE - 1 || throw(ArgumentError(
        "n must be positive and 2^n must fit in Int."))
    entries = collect(terms)
    # Validate even zero-coefficient terms before allocating register matrices.
    for entry in entries
        entry isa Pair || throw(ArgumentError("Each term must be coefficient => site tuple."))
        coefficient, sites = entry
        coefficient isa Real && isfinite(coefficient) || throw(ArgumentError(
            "Pauli coefficients must be finite real numbers."))
        sites isa Union{Tuple, AbstractVector} || throw(ArgumentError(
            "Term support must be a tuple or vector of site => Pauli pairs."))
        seen = Set{Int}()
        for factor in sites
            factor isa Pair || throw(ArgumentError("Each factor must be site => Pauli."))
            site, label = factor
            site isa Integer && 1 <= site <= n || throw(ArgumentError(
                "Pauli sites must be integers in 1:$n."))
            label isa Symbol && label in (:I, :X, :Y, :Z) || throw(ArgumentError(
                "Pauli labels must be :I, :X, :Y or :Z."))
            site in seen && throw(ArgumentError("Duplicate site $site in a Pauli term."))
            push!(seen, site)
        end
    end
    T = isempty(entries) ? Float64 : float(promote_type(map(e -> typeof(first(e)), entries)...))
    H = spzeros(Complex{T}, 2^Int(n), 2^Int(n))
    for (coefficient, sites) in entries
        iszero(coefficient) && continue
        labels = fill("I", Int(n))
        for (site, label) in sites
            labels[site] = String(label)
        end
        # Full-register support lets the existing contiguous embedding express
        # arbitrary sites without introducing periodic translations.
        term = pad_term(pauli_string_to_matrix(labels), Int(n), 1; periodic=false)
        H += T(coefficient) .* SparseMatrixCSC{Complex{T}, Int}(term)
    end
    all(isfinite, H) || throw(ArgumentError("Pauli Hamiltonian coefficients overflowed during assembly."))
    return Hermitian(H)
end

"""
    pauli_string_to_matrix(paulistring) -> Vector{Matrix{ComplexF64}}

Convert labels from `"I"`, `"X"`, `"Y"`, and `"Z"` to single-qubit matrices.

# Arguments
- `paulistring`: Ordered Pauli labels.

# Returns
One matrix per label; the Kronecker product is not formed.
"""
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

"""
    expm_pauli_padded(pauli_list, coeff, num_qubits, position; periodic=true) -> Matrix

Embed a Pauli string and return its exponential.

# Arguments
- `pauli_list`: Consecutive single-qubit Pauli factors.
- `coeff`: Rotation coefficient.
- `num_qubits`: Register size.
- `position`: First site, using one-based indexing.

# Keywords
- `periodic`: Wrap the support across the boundary. For an absent open-boundary
  term, return the identity.

# Returns
The dense unitary on the full register.
"""
function expm_pauli_padded(pauli_list::Vector{Matrix{ComplexF64}}, coeff::Float64, num_qubits::Int64, position::Int64; periodic::Bool=true)
    term_length = length(pauli_list)
    last_position = position + term_length - 1
    if !periodic && last_position > num_qubits
        return Matrix{ComplexF64}(I, 2^num_qubits, 2^num_qubits)
    end

    padded_term = pad_term(pauli_list, num_qubits, position; periodic=periodic)
    # Math: $exp(i theta P) = cos(theta) I + i sin(theta) P$ because $P^2 = I$.
    expm = cos(coeff) * I(2^num_qubits) + 1im * sin(coeff) * padded_term
    return expm
end

"""
    pad_term(terms, num_qubits, position; periodic=true) -> SparseMatrixCSC

Embed consecutive local operators into a qubit register.

# Arguments
- `terms`: Ordered local factors.
- `num_qubits`: Register size.
- `position`: First site, using one-based indexing.

# Keywords
- `periodic`: Wrap the support; an absent open-boundary term returns zero.

# Returns
The sparse full-register operator.
"""
function pad_term(terms::Vector{Matrix{ComplexF64}}, num_qubits::Int64, position::Int; periodic::Bool = true)

    term_length = length(terms)
    terms = [sparse(term) for term in terms]
    last_position = position + term_length - 1
    # Drop boundary overstepping terms for aperiodic boundary condition
    if (!(periodic) && last_position > num_qubits)
        return spzeros(ComplexF64, 2^num_qubits, 2^num_qubits)
    end

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
