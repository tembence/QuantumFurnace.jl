using LinearAlgebra
using SparseArrays
using Random
using Printf

"""
    hermitianize!(A::AbstractMatrix) -> A

Replace `A` by its Hermitian part and return it.
"""
function hermitianize!(A::AbstractMatrix)
    axes(A, 1) == axes(A, 2) ||
        throw(DimensionMismatch("Hermitian projection requires a square matrix."))

    # Read each off-diagonal pair before writing either member. Besides avoiding
    # a temporary matrix, this prevents the second triangle from averaging
    # against a value that has already been projected.
    inds = axes(A, 1)
    @inbounds for j in inds
        A[j, j] = real(A[j, j])
        for i in first(inds):(j - 1)
            a_ij = A[i, j]
            a_ji = A[j, i]
            projected = (a_ij + conj(a_ji)) / 2
            A[i, j] = projected
            A[j, i] = conj(projected)
        end
    end
    return A
end

# Fix the arbitrary global phase of a stationary eigenmode by making its trace
# positive real. Hermitianising first can annihilate a valid mode proportional
# to `im * rho`.
function _phase_align_trace!(A::AbstractMatrix)
    trace_value = tr(A)
    trace_magnitude = abs(trace_value)
    matrix_norm = norm(A)
    roundoff_scale = sqrt(eps(float(one(trace_magnitude)))) *
        sqrt(float(size(A, 1))) * matrix_norm
    isfinite(trace_magnitude) && isfinite(matrix_norm) && matrix_norm > 0 &&
        trace_magnitude > roundoff_scale ||
        throw(ArgumentError(
            "stationary eigenmode trace is zero or too small for stable normalisation."))
    A .*= conj(trace_value) / trace_magnitude
    return A
end

function _normalize_stationary_mode!(A::AbstractMatrix)
    _phase_align_trace!(A)
    hermitianize!(A)
    trace_value = real(tr(A))
    isfinite(trace_value) && trace_value > zero(trace_value) ||
        throw(ArgumentError(
            "phase-aligned stationary eigenmode must have positive finite trace."))
    A ./= trace_value
    return A
end

"""
    _kron!(C, A, B, alpha) -> C

Accumulate `alpha * kron(A, B)` into `C` without materialising the product.
"""
function _kron!(
    C::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    alpha::Number
)
    m_a, n_a = size(A)
    m_b, n_b = size(B)
    
    for j in 1:n_a
        for i in 1:m_a
            a_ij = A[i, j]
            iszero(a_ij) && continue

            c_row_offset = (i - 1) * m_b
            c_col_offset = (j - 1) * n_b

            val = alpha * a_ij
            for l in 1:n_b
                for k in 1:m_b
                    @inbounds C[c_row_offset + k, c_col_offset + l] += val * B[k, l]
                end
            end
        end
    end
    return C
end

"""
    _vectorize_liouv_diss_and_add!(L_target, jump, scalar, ws) -> L_target

Add a single-jump dissipator to a column-stacked Liouvillian.
"""
function _vectorize_liouv_diss_and_add!(
    L_target::AbstractMatrix{<:Complex},
    jump::AbstractMatrix{<:Complex},
    scalar::Number,
    ws::DenseLindbladianWorkspace,
)
    Id = ws.Id

    jump_conj = ws.scratch.jump_conj
    jump_dag_jump = ws.scratch.jump_dag_jump

    # Math: $vec(J rho J^dagger) = (conj(J) tensor J) vec(rho)$.
    @. jump_conj = conj(jump)
    _kron!(L_target, jump_conj, jump, scalar)

    mul!(jump_dag_jump, jump', jump)
    _kron!(L_target, Id, jump_dag_jump, -0.5 * scalar)
    _kron!(L_target, transpose(jump_dag_jump), Id, -0.5 * scalar)

    return L_target
end

"""
    _vectorize_liouv_diss_and_add!(L_target, jump_1, jump_2, scalar, ws)

Add a two-jump dissipator to a column-stacked Liouvillian.
"""
function _vectorize_liouv_diss_and_add!(
    L_target::AbstractMatrix{<:Complex},
    jump_1::AbstractMatrix{<:Complex},
    jump_2::AbstractMatrix{<:Complex},
    scalar::Number,
    ws::DenseLindbladianWorkspace,
)
    Id = ws.Id
    jump2_jump1 = ws.scratch.jump2_jump1

    # Math: $vec(J_1 rho J_2) = (J_2^T tensor J_1) vec(rho)$.
    _kron!(L_target, transpose(jump_2), jump_1, scalar)

    mul!(jump2_jump1, jump_2, jump_1)
    _kron!(L_target, Id, jump2_jump1, -0.5 * scalar)
    _kron!(L_target, transpose(jump2_jump1), Id, -0.5 * scalar)

    return L_target
end

function _vectorize_liouvillian_coherent!(
    L_target::AbstractMatrix{<:Complex},
    coherent_term::AbstractMatrix{<:Complex},
    ws::DenseLindbladianWorkspace,
)
    Id = ws.Id
    # Math: vec(-i[B,rho]) = [-i(I tensor B) + i(B^T tensor I)] vec(rho).
    _kron!(L_target, Id, coherent_term, -1im)
    _kron!(L_target, transpose(coherent_term), Id, +1im)
    return L_target
end

"""Return the trace distance between Hermitian matrices."""
function trace_distance_h(rho::Union{Hermitian{<:Real}, Hermitian{<:Complex}}, 
    sigma::Union{Hermitian{<:Real}, Hermitian{<:Complex}})
    return sum(abs.(eigvals(rho - sigma))) / 2
end

"""Return the trace distance between general matrices using singular values."""
function trace_distance_nh(rho::Union{Matrix{<:Real}, Matrix{<:Complex}}, 
    sigma::Union{Matrix{<:Real}, Matrix{<:Complex}})
    return sum(svdvals(rho - sigma)) / 2
end

"""Return the trace norm of a Hermitian matrix from its eigenvalues."""
function trace_norm_h(rho::Union{Hermitian{<:Real}, Hermitian{<:Complex}})
    return sum(abs.(eigvals(rho)))
end

"""Return the trace norm of a general matrix from its singular values."""
function trace_norm_nh(rho::Union{Matrix{<:Real}, Matrix{<:Complex}})
    return sum(svdvals(rho))
end

"""
    fidelity(rho, sigma; validate=true, atol=nothing, rtol=nothing) -> Real

Return the squared quantum-state fidelity, optionally validating both inputs.
"""
function fidelity(
    rho::AbstractMatrix{<:Number},
    sigma::AbstractMatrix{<:Number};
    validate::Bool = true,
    atol::Union{Nothing, Real} = nothing,
    rtol::Union{Nothing, Real} = nothing,
)
    size(rho, 1) == size(rho, 2) || throw(ArgumentError("rho must be square."))
    size(sigma, 1) == size(sigma, 2) || throw(ArgumentError("sigma must be square."))
    size(rho, 1) > 0 || throw(ArgumentError("rho and sigma must be nonempty."))
    size(rho) == size(sigma) || throw(ArgumentError(
        "rho and sigma must have the same dimensions."))
    all(isfinite, rho) || throw(ArgumentError("rho must contain only finite values."))
    all(isfinite, sigma) || throw(ArgumentError("sigma must contain only finite values."))
    atol_value, rtol_value = _density_tolerances(rho, sigma; atol=atol, rtol=rtol)
    isapprox(rho, adjoint(rho); atol=atol_value, rtol=rtol_value) ||
        throw(ArgumentError("rho must be Hermitian."))
    isapprox(sigma, adjoint(sigma); atol=atol_value, rtol=rtol_value) ||
        throw(ArgumentError("sigma must be Hermitian."))
    if validate
        is_density_matrix(rho; atol=atol_value, rtol=rtol_value)
        is_density_matrix(sigma; atol=atol_value, rtol=rtol_value)
    end

    rho_matrix = Matrix(rho)
    sigma_matrix = Matrix(sigma)
    rho_h = Hermitian((rho_matrix + adjoint(rho_matrix)) / 2)
    sigma_h = Hermitian((sigma_matrix + adjoint(sigma_matrix)) / 2)
    rho_eigen = eigen(rho_h)
    rho_values = _clamp_psd_eigenvalues(
        rho_eigen.values, atol_value, rtol_value, "rho")
    sigma_eigen = eigen(sigma_h)
    sigma_values = _clamp_psd_eigenvalues(
        sigma_eigen.values, atol_value, rtol_value, "sigma")
    if validate
        rho_values ./= sum(rho_values)
        sigma_values ./= sum(sigma_values)
    end
    sqrt_rho = rho_eigen.vectors * Diagonal(sqrt.(rho_values)) * adjoint(rho_eigen.vectors)
    sqrt_sigma = sigma_eigen.vectors * Diagonal(sqrt.(sigma_values)) * adjoint(sigma_eigen.vectors)
    result = real(sum(svdvals(sqrt_rho * sqrt_sigma))^2)

    if validate
        bound_tolerance = atol_value + rtol_value
        result <= 1 + bound_tolerance || throw(ArgumentError(
            "Computed fidelity exceeds one beyond numerical tolerance: $result."))
        return clamp(result, zero(result), one(result))
    end
    return result
end

"""
    is_density_matrix(rho; atol=nothing, rtol=nothing) -> true

Validate Hermiticity, positive semidefiniteness, and unit trace.

Throws `ArgumentError` when an invariant fails.
"""
function is_density_matrix(
    rho::AbstractMatrix{<:Number};
    atol::Union{Nothing, Real} = nothing,
    rtol::Union{Nothing, Real} = nothing,
)
    size(rho, 1) == size(rho, 2) || throw(ArgumentError("Density matrix must be square."))
    size(rho, 1) > 0 || throw(ArgumentError("Density matrix must be nonempty."))
    all(isfinite, rho) || throw(ArgumentError("Density matrix must contain only finite values."))
    atol_value, rtol_value = _density_tolerances(rho; atol=atol, rtol=rtol)
    isapprox(rho, adjoint(rho); atol=atol_value, rtol=rtol_value) ||
        throw(ArgumentError("Density matrix must be Hermitian."))

    trace_value = tr(rho)
    isfinite(trace_value) && isapprox(trace_value, one(real(trace_value));
        atol=atol_value, rtol=rtol_value) ||
        throw(ArgumentError("Density matrix must have unit trace."))

    rho_matrix = Matrix(rho)
    rho_h = Hermitian((rho_matrix + adjoint(rho_matrix)) / 2)
    _clamp_psd_eigenvalues(eigvals(rho_h), atol_value, rtol_value, "Density matrix")
    return true
end

function _density_tolerances(
    matrices::AbstractMatrix{<:Number}...;
    atol::Union{Nothing, Real},
    rtol::Union{Nothing, Real},
)
    RT = promote_type((typeof(float(real(zero(eltype(matrix))))) for matrix in matrices)...)
    dim = maximum(max(size(matrix)...) for matrix in matrices)
    default_tolerance = RT(10 * dim) * eps(RT)
    atol_value = isnothing(atol) ? default_tolerance : RT(atol)
    rtol_value = isnothing(rtol) ? default_tolerance : RT(rtol)
    isfinite(atol_value) && atol_value >= 0 || throw(ArgumentError(
        "atol must be finite and >= 0."))
    isfinite(rtol_value) && rtol_value >= 0 || throw(ArgumentError(
        "rtol must be finite and >= 0."))
    return atol_value, rtol_value
end

function _clamp_psd_eigenvalues(
    values::AbstractVector{T},
    atol::Real,
    rtol::Real,
    label::AbstractString,
) where {T<:Real}
    all(isfinite, values) || throw(ArgumentError("$label has non-finite eigenvalues."))
    scale = max(maximum(abs, values), one(T))
    tolerance = T(atol) + T(rtol) * scale
    minimum(values) >= -tolerance || throw(ArgumentError(
        "$label has a negative eigenvalue below tolerance: $(minimum(values))."))
    negative_mass = sum(value -> max(-value, zero(T)), values)
    negative_mass <= tolerance || throw(ArgumentError(
        "$label has cumulative negative spectral weight above tolerance: $negative_mass."))
    return max.(values, zero(T))
end

"""
    gibbs_state(hamiltonian, beta) -> Matrix

Return `\$rho_beta = exp(-beta H) / Z\$` in the computational basis.
"""
function gibbs_state(hamiltonian::HamHam{T}, beta::Real) where {T<:AbstractFloat}
    rho_eigen = _gibbs_in_eigen(hamiltonian.eigvals, T(beta))
    return Matrix{Complex{T}}(
        hamiltonian.eigvecs * rho_eigen * adjoint(hamiltonian.eigvecs))
end

"""
    gibbs_state_in_eigen(hamiltonian, beta) -> Matrix

Return `\$rho_beta = exp(-beta H) / Z\$` in the Hamiltonian eigenbasis.
"""
function gibbs_state_in_eigen(hamiltonian::HamHam{T}, beta::Real) where {T<:AbstractFloat}
    return _gibbs_in_eigen(hamiltonian.eigvals, T(beta))
end

"""Return a random `num_qubits`-qubit density matrix from a Ginibre draw."""
function random_density_matrix(num_qubits::Int)
    A = randn(ComplexF64, 2^num_qubits, 2^num_qubits)
    ρ = A * A'
    ρ /= tr(ρ)

    return Hermitian(ρ)
end

"""
    validate_jump_pairing(jumps; allow_unpaired_nonhermitian=false, atol=1e-12)

Validate that stored jump operators form an adjoint-closed multiset.

# Keywords
- `allow_unpaired_nonhermitian`: Skip closure checks for non-physical diagnostics.
- `atol`: Absolute tolerance in both stored bases.

# Returns
`nothing`; throws `ArgumentError` on invalid Hermitian flags or missing adjoints.

KMS detailed balance requires
`\$alpha(omega_1, omega_2) = alpha(-omega_2, -omega_1) exp(-beta(omega_1 + omega_2)/2)\$`.
"""
function validate_jump_pairing(jumps::AbstractVector{<:JumpOp};
                                allow_unpaired_nonhermitian::Bool = false,
                                atol::Real = 1e-12)
    invalid_hermitian_indices = Int[]
    for k in eachindex(jumps)
        jumps[k].hermitian || continue
        data_ok = isapprox(jumps[k].data, jumps[k].data'; atol=atol)
        eigenbasis_ok = isapprox(
            jumps[k].in_eigenbasis, jumps[k].in_eigenbasis'; atol=atol)
        (data_ok && eigenbasis_ok) || push!(invalid_hermitian_indices, k)
    end
    isempty(invalid_hermitian_indices) || throw(ArgumentError(
        "Jump(s) marked hermitian=true are not Hermitian in both stored bases " *
        "at index/indices $(invalid_hermitian_indices)."))

    allow_unpaired_nonhermitian && return nothing

    matched = falses(length(jumps))
    unpaired_indices = Int[]
    for k in eachindex(jumps)
        (jumps[k].hermitian || matched[k]) && continue
        data_adjoint = jumps[k].data'
        eigenbasis_adjoint = jumps[k].in_eigenbasis'
        partner = nothing
        for j in eachindex(jumps)
            (j == k || matched[j] || jumps[j].hermitian) && continue
            data_ok = size(jumps[j].data) == size(data_adjoint) &&
                isapprox(jumps[j].data, data_adjoint; atol=atol)
            eigenbasis_ok = size(jumps[j].in_eigenbasis) == size(eigenbasis_adjoint) &&
                isapprox(jumps[j].in_eigenbasis, eigenbasis_adjoint; atol=atol)
            if data_ok && eigenbasis_ok
                partner = j
                break
            end
        end
        if partner === nothing
            push!(unpaired_indices, k)
        else
            matched[k] = true
            matched[partner] = true
        end
    end

    isempty(unpaired_indices) && return nothing

    throw(ArgumentError(
        "KMS detailed balance requires (A, A†) pairs for non-Hermitian jumps. " *
        "Found $(length(unpaired_indices)) unpaired non-Hermitian jump(s) at " *
        "index/indices $(unpaired_indices). Pass " *
        "`allow_unpaired_nonhermitian=true` for unit-test fixtures only."))
end

# Relative comparisons keep arbitrarily small sources from being mistaken for zero.
_jump_matches(A, B, rtol) = size(A) == size(B) && isapprox(A, B; atol=0, rtol)

"""
    JumpOp(A::AbstractMatrix, ham::HamHam; basis=:computational)

Snapshot a finite source, rotate it into the other basis, and derive the flags.
`basis=:eigen` means the eigenvectors stored in `ham`. No amplitude normalisation
or adjoint completion is performed by this single-source constructor.
"""
function JumpOp(A::AbstractMatrix, ham::HamHam{T}; basis::Symbol=:computational) where {T}
    basis in (:computational, :eigen) || throw(ArgumentError(
        "basis must be :computational or :eigen."))
    size(A) == size(ham.data) || throw(ArgumentError("Jump dimensions must match ham."))
    eltype(A) <: Number && all(isfinite, A) || throw(ArgumentError(
        "Jump entries must be finite numbers."))
    source = Matrix{Complex{T}}(A)
    all(isfinite, source) || throw(ArgumentError("Jump overflows Hamiltonian precision."))
    U = ham.eigvecs
    data, eb = basis == :computational ? (source, U' * source * U) :
        (U * source * U', source)
    rtol = 100eps(T)
    return JumpOp(data, eb, _jump_matches(data, transpose(data), rtol),
        _jump_matches(data, data', rtol) && _jump_matches(eb, eb', rtol))
end

"""
    JumpOp(jump::JumpOp, ham::HamHam)

Validate both stored bases and flags against `ham`, then return owned matrices.
Use this at preparation boundaries to detect stale or mutated source data.
"""
function JumpOp(jump::JumpOp, ham::HamHam{T}) where {T}
    fresh = JumpOp(jump.data, ham)
    all(isfinite, jump.in_eigenbasis) &&
        _jump_matches(jump.in_eigenbasis, fresh.in_eigenbasis, 100eps(T)) ||
        throw(ArgumentError("Stored jump eigenbasis is inconsistent with ham; rebuild JumpOp."))
    jump.hermitian == fresh.hermitian && jump.orthogonal == fresh.orthogonal ||
        throw(ArgumentError("Stored jump flags are inconsistent with its matrices."))
    return fresh
end

"""
    prepare_jumps(sources, ham; basis=:computational, rates=1, complete_adjoint=false)

Return `(jumps, provenance)` with owned `JumpOp` matrices. `sources` is a vector
of matrices/JumpOps or `:onsite_paulis` (all `3n` Paulis, amplitude `1/sqrt(3n)`).
Supplied amplitudes and multiplicities are retained. Finite positive `rates`
(one scalar or one per input) multiply amplitudes by their square roots.

By default non-Hermitian sources require adjoint partners with equal rates.
Explicit completion appends only missing partners with the same rate; an
existing partner with an unequal rate is an error even with completion enabled.
Completion is idempotent on the returned jumps with default rates. Provenance
records original indices, appended partners, rates, and the proposal amplitude.
Adjoint closure does not establish uniqueness or convergence to Gibbs.
"""
function prepare_jumps(sources, ham::HamHam{T}; basis::Symbol=:computational,
                       rates=one(T), complete_adjoint::Bool=false) where {T}
    basis in (:computational, :eigen) || throw(ArgumentError(
        "basis must be :computational or :eigen."))
    onsite = sources === :onsite_paulis
    if onsite
        basis == :computational || throw(ArgumentError(
            "The onsite Pauli preset is defined in the computational basis."))
        n = trailing_zeros(size(ham.data, 1))
        inputs = _jumps_in_basis(n, ham.eigvecs)
    else
        sources isa AbstractVector || throw(ArgumentError(
            "sources must be a vector or :onsite_paulis."))
        inputs = sources
    end
    owned = JumpOp{Matrix{Complex{T}}}[]
    for source in inputs
        if source isa JumpOp
            basis == :computational || throw(ArgumentError(
                "JumpOp already stores both bases; do not specify basis=:eigen."))
            push!(owned, JumpOp(source, ham))
        elseif source isa AbstractMatrix
            push!(owned, JumpOp(source, ham; basis))
        else
            throw(ArgumentError("Every source must be a matrix or JumpOp."))
        end
    end
    raw_rates = rates isa Real ? fill(rates, length(owned)) : collect(rates)
    length(raw_rates) == length(owned) || throw(ArgumentError("One rate is required per source."))
    all(r -> r isa Real && isfinite(r) && r > 0, raw_rates) ||
        throw(ArgumentError("Source rates must be finite and positive; use a zero matrix for a zero source."))
    rates isa Real && !(isfinite(rates) && rates > 0) &&
        throw(ArgumentError("Source rates must be finite and positive."))
    resolved_rates = T.(raw_rates)
    all(r -> isfinite(r) && r > 0, resolved_rates) || throw(ArgumentError(
        "Source rates are not representable at Hamiltonian precision."))
    origins = collect(eachindex(owned))
    added = falses(length(owned))
    matched = falses(length(owned))
    rtol = 100eps(T)
    for k in eachindex(inputs)
        (owned[k].hermitian || matched[k]) && continue
        candidates = [j for j in eachindex(owned) if j != k && !matched[j] &&
            !owned[j].hermitian && _jump_matches(owned[j].data, owned[k].data', rtol)]
        partner = findfirst(j -> isapprox(resolved_rates[j], resolved_rates[k];
            atol=0, rtol), candidates)
        if partner === nothing
            isempty(candidates) || throw(ArgumentError("Adjoint partners must have equal source rates."))
            complete_adjoint || throw(ArgumentError(
                "Missing adjoint partner for source $k; set complete_adjoint=true to append it."))
            push!(owned, JumpOp(owned[k].data', ham))
            push!(resolved_rates, resolved_rates[k])
            push!(origins, k)
            push!(added, true)
            push!(matched, true)
        else
            matched[candidates[partner]] = true
        end
        matched[k] = true
    end
    jumps = JumpOp[JumpOp(sqrt(rate) .* jump.data, ham) for (jump, rate) in zip(owned, resolved_rates)]
    provenance = (; input_basis=basis, source_indices=origins, added_adjoint=added,
        rates=copy(resolved_rates), input_count=length(inputs), source_count=length(jumps),
        proposal_normalisation=onsite ? :qf_averaged : :user_amplitudes,
        proposal_amplitude=onsite ? inv(sqrt(T(length(inputs)))) : one(T),
        clock_label=:raw_generator, generator_multiplier=one(T))
    return (; jumps, provenance)
end
