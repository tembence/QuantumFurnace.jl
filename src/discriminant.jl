# KMS quantum discriminant.
# Math: $D(X) = sigma^(-1/4) L(sigma^(1/4) X sigma^(1/4)) sigma^(-1/4)$.

"""
    DiscriminantBuffers{T<:Complex}

Three reusable matrices for `apply_discriminant!`.

# Fields
- `work1`, `work2`, `work3`: `Matrix{T}` of size `dim x dim`.
"""
struct DiscriminantBuffers{T<:Complex}
    work1::Matrix{T}
    work2::Matrix{T}
    work3::Matrix{T}
end

"""
    DiscriminantBuffers{T}(dim::Int) where {T<:Complex}

Allocate three uninitialised `dim × dim` buffers of element type `T`.
"""
function DiscriminantBuffers{T}(dim::Int) where {T<:Complex}
    dim > 0 || throw(ArgumentError("dim must be > 0."))
    return DiscriminantBuffers{T}(
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
    )
end

"""
    DiscriminantBuffers(dim::Int)

Allocate `ComplexF64` discriminant buffers.
"""
DiscriminantBuffers(dim::Int) = DiscriminantBuffers{ComplexF64}(dim)

function _validated_diagonal_gibbs_state(
    gibbs::AbstractMatrix{<:Number};
    atol::Union{Nothing, Real} = nothing,
    rtol::Union{Nothing, Real} = nothing,
)
    d = size(gibbs, 1)
    size(gibbs, 2) == d || throw(ArgumentError("Gibbs state must be square."))
    d > 0 || throw(ArgumentError("Gibbs state must be nonempty."))
    all(isfinite, gibbs) || throw(ArgumentError(
        "Gibbs state must contain only finite values."))

    atol_value, rtol_value = _density_tolerances(gibbs; atol = atol, rtol = rtol)
    gibbs_matrix = Matrix(gibbs)
    isapprox(gibbs_matrix, adjoint(gibbs_matrix);
        atol = atol_value, rtol = rtol_value) || throw(ArgumentError(
        "Gibbs state must be Hermitian."))

    RT = typeof(atol_value)
    scale = max(norm(gibbs_matrix, Inf), one(RT))
    tolerance = atol_value + rtol_value * scale
    max_offdiag = zero(RT)
    max_diag_imag = zero(RT)
    diag_real = Vector{RT}(undef, d)
    @inbounds for j in 1:d, i in 1:d
        value = gibbs_matrix[i, j]
        if i == j
            diag_real[i] = RT(real(value))
            max_diag_imag = max(max_diag_imag, RT(abs(imag(value))))
        else
            max_offdiag = max(max_offdiag, RT(abs(value)))
        end
    end
    max_offdiag <= tolerance || throw(ArgumentError(
        "Gibbs state must be diagonal in the declared working basis " *
        "(maximum off-diagonal magnitude $max_offdiag exceeds tolerance $tolerance)."))
    max_diag_imag <= tolerance || throw(ArgumentError(
        "Gibbs-state diagonal must be real."))

    trace_value = tr(gibbs_matrix)
    isapprox(trace_value, one(trace_value); atol = atol_value, rtol = rtol_value) ||
        throw(ArgumentError("Gibbs state must have unit trace."))
    minimum(diag_real) > zero(RT) || throw(ArgumentError(
        "Gibbs state must be positive definite; every diagonal weight must be > 0."))
    return diag_real
end

"""
    gibbs_fractional_powers(gibbs; atol=nothing, rtol=nothing)

Return diagonal vectors for `sigma^(1/4)`, `sigma^(-1/4)`, and `sigma^(1/2)`.

# Keywords
- `atol`, `rtol`: Validation tolerances for Hermiticity, diagonality, and trace.

The input must be a finite, normalised, positive-definite Gibbs state diagonal
in the declared working basis. Every strictly positive weight is retained
exactly; changing small weights would change the discriminant similarity
transform.
"""
function gibbs_fractional_powers(
    gibbs::AbstractMatrix{<:Number};
    atol::Union{Nothing, Real} = nothing,
    rtol::Union{Nothing, Real} = nothing,
)
    diag_real = _validated_diagonal_gibbs_state(gibbs; atol = atol, rtol = rtol)
    T = eltype(diag_real)
    return (
        sigma_quarter     = diag_real .^ T(0.25),
        sigma_inv_quarter = diag_real .^ T(-0.25),
        sigma_half        = diag_real .^ T(0.5),
    )
end

"""
    apply_discriminant!(out, X, lindblad_action!, sigma_quarter, sigma_inv_quarter, buffers)

Apply the KMS quantum discriminant without materialising a superoperator.

# Arguments
- `out`: Destination matrix.
- `X::AbstractMatrix{T}`: input operator.
- `lindblad_action!`: In-place callable `L(out, X)`.
- `sigma_quarter`, `sigma_inv_quarter`: Diagonal fractional powers.
- `buffers`: Reusable scratch matrices.

# Returns
`out`.
"""
function apply_discriminant!(
    out::AbstractMatrix{T},
    X::AbstractMatrix{T},
    lindblad_action!::F,
    sigma_quarter::AbstractVector{<:Real},
    sigma_inv_quarter::AbstractVector{<:Real},
    buffers::DiscriminantBuffers{T},
) where {T<:Complex, F}
    work1 = buffers.work1
    work2 = buffers.work2
    d = size(X, 1)
    size(X, 2) == d || throw(ArgumentError("X must be square."))
    size(out) == (d, d) || throw(ArgumentError(
        "out must have size ($d, $d), got $(size(out))."))
    length(sigma_quarter) == d || throw(ArgumentError(
        "sigma_quarter length $(length(sigma_quarter)) does not match dimension $d."))
    length(sigma_inv_quarter) == d || throw(ArgumentError(
        "sigma_inv_quarter length $(length(sigma_inv_quarter)) does not match dimension $d."))
    all(value -> isfinite(value) && value > 0, sigma_quarter) || throw(ArgumentError(
        "sigma_quarter must contain finite positive values."))
    all(value -> isfinite(value) && value > 0, sigma_inv_quarter) || throw(ArgumentError(
        "sigma_inv_quarter must contain finite positive values."))
    size(work1) == (d, d) && size(work2) == (d, d) &&
        size(buffers.work3) == (d, d) || throw(ArgumentError(
        "Discriminant buffers must have size ($d, $d)."))

    # Math: $work1 = sigma^(1/4) X sigma^(1/4)$.
    @inbounds for j in 1:d, i in 1:d
        work1[i, j] = sigma_quarter[i] * X[i, j] * sigma_quarter[j]
    end

    lindblad_action!(work2, work1)

    # Math: $out = sigma^(-1/4) L(work1) sigma^(-1/4)$.
    @inbounds for j in 1:d, i in 1:d
        out[i, j] = sigma_inv_quarter[i] * work2[i, j] * sigma_inv_quarter[j]
    end

    return out
end

"""
    apply_kms_parent!(out, X, lindblad_action!, sigma_quarter,
                      sigma_inv_quarter, buffers)

Apply the positive KMS parent `K = -D` without materialising a
superoperator. The implementation delegates to [`apply_discriminant!`](@ref)
and applies one global minus sign to its output.

# Returns
`out`.
"""
function apply_kms_parent!(
    out::AbstractMatrix{T},
    X::AbstractMatrix{T},
    lindblad_action!::F,
    sigma_quarter::AbstractVector{<:Real},
    sigma_inv_quarter::AbstractVector{<:Real},
    buffers::DiscriminantBuffers{T},
) where {T<:Complex, F}
    apply_discriminant!(
        out, X, lindblad_action!, sigma_quarter, sigma_inv_quarter, buffers)
    @inbounds for index in eachindex(out)
        out[index] = -out[index]
    end
    return out
end

"""
    materialize_discriminant!(D, L, sigma_quarter, sigma_inv_quarter)
    materialize_discriminant!(D, L, gibbs)

Materialise the KMS quantum discriminant in `D`.

Column stacking gives the diagonal similarity transform
`D = (sigma^(-1/4) tensor sigma^(-1/4)) L
(sigma^(1/4) tensor sigma^(1/4))`; Kronecker factors are not allocated.

# Returns
`D`.
"""
function materialize_discriminant!(
    D::AbstractMatrix{<:Complex},
    L::AbstractMatrix{<:Complex},
    sigma_quarter::AbstractVector{<:Real},
    sigma_inv_quarter::AbstractVector{<:Real},
)
    d = length(sigma_quarter)
    d > 0 || throw(ArgumentError("sigma_quarter must be nonempty."))
    length(sigma_inv_quarter) == d || throw(ArgumentError(
        "sigma_quarter and sigma_inv_quarter must have the same length."))
    all(value -> isfinite(value) && value > 0, sigma_quarter) || throw(ArgumentError(
        "sigma_quarter must contain finite positive values."))
    all(value -> isfinite(value) && value > 0, sigma_inv_quarter) || throw(ArgumentError(
        "sigma_inv_quarter must contain finite positive values."))
    d2 = d * d
    size(L) == (d2, d2) || throw(ArgumentError(
        "L must have size ($d2, $d2) for Gibbs dimension $d, got $(size(L))."))
    size(D) == size(L) || throw(ArgumentError(
        "D must have the same dimensions as L, got $(size(D)) and $(size(L))."))

    @inbounds for l in 1:d2
        c2 = ((l - 1) ÷ d) + 1
        r2 = ((l - 1) % d) + 1
        scale_right = sigma_quarter[r2] * sigma_quarter[c2]
        for k in 1:d2
            c1 = ((k - 1) ÷ d) + 1
            r1 = ((k - 1) % d) + 1
            scale_left = sigma_inv_quarter[r1] * sigma_inv_quarter[c1]
            D[k, l] = scale_left * L[k, l] * scale_right
        end
    end
    return D
end

function materialize_discriminant!(
    D::AbstractMatrix{<:Complex},
    L::AbstractMatrix{<:Complex},
    gibbs::AbstractMatrix{<:Number};
    atol::Union{Nothing, Real} = nothing,
    rtol::Union{Nothing, Real} = nothing,
)
    powers = gibbs_fractional_powers(gibbs; atol = atol, rtol = rtol)
    return materialize_discriminant!(D, L, powers.sigma_quarter, powers.sigma_inv_quarter)
end

"""
    materialize_discriminant(L, gibbs; atol=nothing, rtol=nothing) -> Matrix

Allocate and return the dense KMS quantum discriminant.
"""
function materialize_discriminant(
    L::AbstractMatrix{T},
    gibbs::AbstractMatrix{<:Number};
    atol::Union{Nothing, Real} = nothing,
    rtol::Union{Nothing, Real} = nothing,
) where {T<:Complex}
    D = similar(L)
    return materialize_discriminant!(
        D, L, gibbs; atol = atol, rtol = rtol)
end

"""
    materialize_kms_parent(L, gibbs; atol=nothing, rtol=nothing) -> Matrix

Allocate and return the positive KMS parent `K = -D` associated with a dense
Lindbladian. This is a sign-safe wrapper around
[`materialize_discriminant`](@ref); it does not independently reimplement the
Gibbs similarity transform.
"""
function materialize_kms_parent(
    L::AbstractMatrix{T},
    gibbs::AbstractMatrix{<:Number};
    atol::Union{Nothing, Real} = nothing,
    rtol::Union{Nothing, Real} = nothing,
) where {T<:Complex}
    parent = materialize_discriminant(L, gibbs; atol = atol, rtol = rtol)
    @inbounds for index in eachindex(parent)
        parent[index] = -parent[index]
    end
    return parent
end

"""
    hermitian_antihermitian_split(D::AbstractMatrix) -> (H, A)
    hermitian_antihermitian_split!(H, A, D)

Split `D` into Hermitian and anti-Hermitian parts.

Math: `\$H = (D + D^dagger)/2\$` and `\$A = (D-D^dagger)/2\$`. KMS detailed
balance implies `A = 0`.
"""
function hermitian_antihermitian_split(D::AbstractMatrix)
    H = (D + D') / 2
    A = (D - D') / 2
    return (H, A)
end

"""
    hermitian_antihermitian_split!(H, A, D) -> (H, A)

Write the Hermitian and anti-Hermitian parts of `D` into preallocated matrices.

# Arguments
- `H`, `A`: Destination matrices.
- `D`: Input square matrix.

# Returns
The tuple `(H, A)`.
"""
function hermitian_antihermitian_split!(
    H::AbstractMatrix{T},
    A::AbstractMatrix{T},
    D::AbstractMatrix{T},
) where {T<:Complex}
    @inbounds for j in axes(D, 2), i in axes(D, 1)
        d_ij = D[i, j]
        d_ji_conj = conj(D[j, i])
        H[i, j] = (d_ij + d_ji_conj) / 2
        A[i, j] = (d_ij - d_ji_conj) / 2
    end
    return (H, A)
end

"""
    DiscriminantSpectrum

Leading eigenvalues of the Hermitian discriminant part.

# Fields
- `H_eigenvalues::Vector{Float64}`: leading `n_modes` eigenvalues of the
  Hermitian part, sorted ascending by `|λ|` (steady state at index 1).
- `H_gap::Float64`: `|H_eigenvalues[2]|`, the parent-Hamiltonian gap.
- `n_modes::Int`: number of leading eigenvalues retained.
"""
struct DiscriminantSpectrum
    H_eigenvalues::Vector{Float64}
    H_gap::Float64
    n_modes::Int
end

"""
    KMSParentSpectrum{T<:AbstractFloat}

Complete dense diagnostics for a positive KMS parent.

# Fields
- `eigenvalues`: All eigenvalues of the Hermitian part, in ascending order.
- `kernel_tolerance`: Absolute tolerance used to identify the numerical kernel.
- `hermiticity_tolerance`: Dimensionless tolerance for `hermiticity_defect`.
- `kernel_count`: Complete numerical kernel dimension at that tolerance.
- `first_positive_eigenvalue`: Smallest eigenvalue above the kernel tolerance,
  or `nothing` when no positive eigenvalue is present.
- `hermiticity_defect`: Relative operator-norm defect
  `opnorm(K-K') / (2opnorm(K))`.
- `minimum_eigenvalue`: Smallest eigenvalue of the Hermitian part.
- `gibbs_residual`: `norm(K * vec(sqrt(gibbs)))`.
- `primitivity_established`: Whether an independent irreducibility witness was
  supplied and the dense kernel, positivity, Hermiticity, and Gibbs-residual
  gates all pass.
"""
struct KMSParentSpectrum{T<:AbstractFloat}
    eigenvalues::Vector{T}
    kernel_tolerance::T
    hermiticity_tolerance::T
    kernel_count::Int
    first_positive_eigenvalue::Union{Nothing, T}
    hermiticity_defect::T
    minimum_eigenvalue::T
    gibbs_residual::T
    primitivity_established::Bool
end

"""
    kms_parent_spectrum(parent, gibbs;
                        kernel_tolerance=nothing,
                        hermiticity_tolerance=nothing,
                        irreducibility_established=false) -> KMSParentSpectrum

Compute the complete small-system spectrum and kernel diagnostics of a dense
KMS parent. `irreducibility_established=true` must come from an independent
finite-size certificate, such as [`dense_dll_irreducibility`](@ref); the flag
alone cannot override a degenerate numerical kernel or failed structural gate.
"""
function kms_parent_spectrum(
    parent::AbstractMatrix{Complex{T}},
    gibbs::AbstractMatrix{<:Number};
    kernel_tolerance::Union{Nothing, Real} = nothing,
    hermiticity_tolerance::Union{Nothing, Real} = nothing,
    irreducibility_established::Bool = false,
) where {T<:AbstractFloat}
    n = size(parent, 1)
    size(parent, 2) == n || throw(ArgumentError("parent must be square."))
    d = size(gibbs, 1)
    size(gibbs, 2) == d || throw(ArgumentError("Gibbs state must be square."))
    n == d^2 || throw(ArgumentError(
        "parent must have size ($(d^2), $(d^2)) for Gibbs dimension $d, " *
        "got $(size(parent))."))
    all(isfinite, parent) || throw(ArgumentError(
        "parent must contain only finite values."))

    powers = gibbs_fractional_powers(gibbs)
    parent_norm = T(opnorm(parent))
    scale = parent_norm
    tolerance = if kernel_tolerance === nothing
        T(100) * T(n) * eps(T) * scale
    else
        value = T(kernel_tolerance)
        isfinite(value) && value >= zero(T) || throw(ArgumentError(
            "kernel_tolerance must be finite and >= 0."))
        value
    end
    hermiticity_tol = if hermiticity_tolerance === nothing
        T(100) * T(n) * eps(T)
    else
        value = T(hermiticity_tolerance)
        isfinite(value) && value >= zero(T) || throw(ArgumentError(
            "hermiticity_tolerance must be finite and >= 0."))
        value
    end

    hermiticity_defect = T(_scaled_residual(opnorm(parent - parent'), T(2) * parent_norm))
    hermitian_parent = Hermitian((parent + parent') / T(2))
    eigenvalues = Vector{T}(eigvals(hermitian_parent))
    kernel_count = count(value -> abs(value) <= tolerance, eigenvalues)
    positive_indices = findall(value -> value > tolerance, eigenvalues)
    first_positive = isempty(positive_indices) ? nothing :
                     eigenvalues[first(positive_indices)]

    witness = zeros(Complex{T}, n)
    @inbounds for index in 1:d
        witness[(index - 1) * d + index] = T(powers.sigma_half[index])
    end
    gibbs_residual = T(norm(parent * witness))
    minimum_eigenvalue = first(eigenvalues)
    primitivity_established = irreducibility_established &&
                              kernel_count == 1 &&
                              hermiticity_defect <= hermiticity_tol &&
                              minimum_eigenvalue >= -tolerance &&
                              gibbs_residual <= tolerance

    return KMSParentSpectrum{T}(
        eigenvalues,
        tolerance,
        hermiticity_tol,
        kernel_count,
        first_positive,
        hermiticity_defect,
        minimum_eigenvalue,
        gibbs_residual,
        primitivity_established,
    )
end

"""
    discriminant_spectrum(L, gibbs; n_modes=20, atol=nothing, rtol=nothing) -> DiscriminantSpectrum

Compute the dense Hermitian-discriminant spectrum.

# Keywords
- `n_modes`: Number of eigenvalues nearest zero to retain.
- `atol`, `rtol`: Gibbs-state validation tolerances.

# Returns
A `DiscriminantSpectrum`. Dense eigendecomposition costs `O(d^6)`.
Its legacy gap field uses the second mode and assumes a unique resolved kernel;
use `kms_parent_spectrum` for complete kernel-aware parent evidence.
"""
function discriminant_spectrum(
    L::AbstractMatrix{<:Complex},
    gibbs::AbstractMatrix{<:Number};
    n_modes::Int = 20,
    atol::Union{Nothing, Real} = nothing,
    rtol::Union{Nothing, Real} = nothing,
)
    n_modes >= 1 || throw(ArgumentError("n_modes must be >= 1."))
    D = materialize_discriminant(L, gibbs; atol = atol, rtol = rtol)
    H, _ = hermitian_antihermitian_split(D)

    # Remove roundoff asymmetry before the Hermitian eigensolve.
    H_clean = Hermitian((H + H') / 2)
    H_eigs = eigvals(H_clean)

    # Sort by distance to the stationary eigenvalue.
    perm = sortperm(abs.(H_eigs))
    n = min(n_modes, length(H_eigs))
    leading = H_eigs[perm[1:n]]

    gap = n >= 2 ? abs(leading[2]) : 0.0

    return DiscriminantSpectrum(leading, gap, n)
end

"""
    DBVerificationResult

Diagnostics returned by `verify_detailed_balance`.

# Fields
- `antihermitian_norm::Float64`: `||A||_{2→2}`, the operator-2 norm of the
  anti-Hermitian part of the discriminant.  KMS-DB iff this is zero.
- `discriminant_norm::Float64`: `||D||_{2→2}`, for normalisation.
- `relative_norm::Float64`: `antihermitian_norm / discriminant_norm`.
- `fixed_point_residual::Float64`: `||D · vec(σ^{1/2})||_2`.  Should be
  ≈ 0 because `σ^{1/2}` is the zero eigenvector of D for any Lindbladian
  with σ as its steady state (independent of detailed balance).
- `hermitian_part_gap::Float64`: smallest |λ| of the Hermitian part above
  the steady-state zero -- the parent-Hamiltonian gap.
- `spectral_gap_L::Float64`: `min |Re(λ)|` over nonzero eigenvalues of L.
  For a KMS-DB Lindbladian this equals `hermitian_part_gap` (similarity
  transform preserves spectrum).
- `is_kms_db::Bool`: `relative_norm < atol`.
- `atol::Float64`: the threshold used for `is_kms_db`.
"""
struct DBVerificationResult
    antihermitian_norm::Float64
    discriminant_norm::Float64
    relative_norm::Float64
    fixed_point_residual::Float64
    hermitian_part_gap::Float64
    spectral_gap_L::Float64
    is_kms_db::Bool
    atol::Float64
end

"""
    verify_detailed_balance(L, gibbs; atol=1e-10) -> DBVerificationResult

Verify KMS detailed balance from the dense quantum discriminant.

# Keywords
- `atol`: Threshold for the relative anti-Hermitian norm.

# Returns
A `DBVerificationResult` containing the defect, fixed-point residual, and
dense gap cross-check. The calculation costs `O(d^6)`. The legacy gap fields
use the second mode, assuming a unique resolved stationary kernel. They are
not reliable for nonunique or unresolved kernels; use `workspace_diagnostics`
and `spectral_gap_diagnostics` for that classification.
"""
function verify_detailed_balance(
    L::AbstractMatrix{<:Complex},
    gibbs::AbstractMatrix{<:Number};
    atol::Float64 = 1e-10,
)
    D = materialize_discriminant(L, gibbs)
    H_part, A_part = hermitian_antihermitian_split(D)

    A_norm   = opnorm(A_part)
    D_norm   = opnorm(D)
    rel_norm = A_norm / max(D_norm, 1e-30)

    # Math: the stationary discriminant vector is $vec(sigma^(1/2))$.
    powers = gibbs_fractional_powers(gibbs)
    sigma_half = powers.sigma_half
    d = length(sigma_half)
    vec_sh = zeros(eltype(D), d * d)
    @inbounds for i in 1:d
        vec_sh[(i - 1) * d + i] = sigma_half[i]
    end
    fp_residual = norm(D * vec_sh)

    H_clean = Hermitian((H_part + H_part') / 2)
    H_eigs  = eigvals(H_clean)
    H_gap   = sort(abs.(H_eigs))[2]

    L_eigs = eigvals(Matrix(L))
    L_gap  = sort(abs.(real.(L_eigs)))[2]

    is_kms_db = rel_norm < atol

    return DBVerificationResult(
        A_norm, D_norm, rel_norm, fp_residual,
        H_gap, L_gap, is_kms_db, atol,
    )
end
