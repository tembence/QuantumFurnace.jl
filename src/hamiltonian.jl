"""
    HamHam{T<:AbstractFloat}

Hamiltonian, spectral data, Bohr-frequency groups, and Gibbs state.

Real fields use `T`; matrix fields use `Complex{T}`. `data` and `eigvals`
contain the spectrum rescaled to `[0, 0.45]`, while `rescaling_factor` converts
between physical and algorithm units. `gibbs` is stored in the eigenbasis.
"""
struct HamHam{T<:AbstractFloat}
    data::Matrix{Complex{T}}
    bohr_freqs::Matrix{T}
    bohr_dict::Dict{T, Vector{CartesianIndex{2}}}
    base_terms::Vector{Vector{Matrix{Complex{T}}}}
    base_coeffs::Vector{T}
    disordering_terms::Union{Vector{Vector{Matrix{Complex{T}}}}, Nothing}
    disordering_coeffs::Union{Vector{Vector{T}}, Nothing}
    eigvals::Vector{T}
    eigvecs::Matrix{Complex{T}}
    nu_min::T  # Smallest bohr frequency
    shift::T
    rescaling_factor::T
    periodic::Bool
    gibbs::Hermitian{Complex{T}, Matrix{Complex{T}}}
end

"""
    _gibbs_in_eigen(eigvals, beta) -> Matrix

Return the diagonal eigenbasis state `\$rho_i = exp(-beta E_i) / Z\$`.
"""
function _gibbs_in_eigen(eigvals::Vector{T}, beta::T) where {T<:AbstractFloat}
    weights = _gibbs_weights(eigvals, beta)
    return Matrix{Complex{T}}(Diagonal(weights))
end

"""Return normalised Gibbs weights after shifting the ground energy to zero."""
function _gibbs_weights(eigvals::AbstractVector{T}, beta::Real) where {T<:AbstractFloat}
    isempty(eigvals) && throw(ArgumentError("eigvals must be nonempty."))
    all(isfinite, eigvals) || throw(ArgumentError("eigvals must be finite."))
    beta_T = T(beta)
    isfinite(beta_T) && beta_T > zero(T) ||
        throw(ArgumentError("beta must be finite and > 0."))

    shifted_energies = eigvals .- minimum(eigvals)
    weights = exp.(-beta_T .* shifted_energies)
    partition = sum(weights)
    isfinite(partition) && partition > zero(T) ||
        throw(ArgumentError("Gibbs partition function is not finite and positive."))
    weights ./= partition
    return weights
end

const _FULL_RAW_SPECTRAL_VALIDATION_MAX_DIM = 128

function _validate_local_terms(
    terms::AbstractVector,
    num_qubits::Int;
    max_support::Int = num_qubits,
    require_support_fits::Bool = true,
    require_involution::Bool = false,
)
    num_qubits > 0 || throw(ArgumentError("num_qubits must be > 0."))
    for term in terms
        term isa AbstractVector || throw(ArgumentError(
            "Each Hamiltonian term must be a vector of local factors."))
        isempty(term) && throw(ArgumentError("Hamiltonian terms must have nonempty support."))
        length(term) <= max_support || throw(ArgumentError(
            "Hamiltonian term support $(length(term)) exceeds the supported maximum $max_support."))
        if require_support_fits && length(term) > num_qubits
            throw(ArgumentError(
                "Hamiltonian term support $(length(term)) exceeds num_qubits=$num_qubits."))
        end
        for factor in term
            factor isa AbstractMatrix{<:Number} || throw(ArgumentError(
                "Each local Hamiltonian factor must be a numeric matrix."))
            size(factor) == (2, 2) || throw(ArgumentError(
                "Each local Hamiltonian factor must be 2 x 2, got $(size(factor))."))
            all(isfinite, factor) || throw(ArgumentError(
                "Local Hamiltonian factors must contain only finite values."))
            ishermitian(factor) || throw(ArgumentError(
                "Local Hamiltonian factors must be Hermitian."))
            if require_involution
                involution_tolerance = 100 * eps(Float64)
                isapprox(adjoint(factor) * factor, I;
                    atol=involution_tolerance, rtol=involution_tolerance) ||
                    throw(ArgumentError(
                        "Builder-supplied local factors must be Hermitian involutions."))
            end
        end
    end
    return nothing
end

function _validate_raw_hamiltonian(
    raw::NamedTuple;
    atol::Union{Nothing, Real} = nothing,
    rtol::Union{Nothing, Real} = nothing,
    spectral_validation::Symbol = :auto,
    full_spectral_max_dim::Int = _FULL_RAW_SPECTRAL_VALIDATION_MAX_DIM,
)
    spectral_validation in (:auto, :full, :probed) || throw(ArgumentError(
        "spectral_validation must be :auto, :full, or :probed."))
    full_spectral_max_dim >= 0 || throw(ArgumentError(
        "full_spectral_max_dim must be >= 0."))
    required = (
        :matrix, :terms, :base_coeffs, :eigvals, :eigvecs, :nu_min,
        :shift, :rescaling_factor, :periodic,
    )
    missing = filter(name -> !haskey(raw, name), required)
    isempty(missing) || throw(ArgumentError(
        "raw Hamiltonian is missing required fields: $(join(string.(missing), ", "))."))

    matrix = raw.matrix
    eigvals = raw.eigvals
    eigvecs = raw.eigvecs
    matrix isa AbstractMatrix || throw(ArgumentError("raw.matrix must be a matrix."))
    eigvals isa AbstractVector || throw(ArgumentError("raw.eigvals must be a vector."))
    eigvecs isa AbstractMatrix || throw(ArgumentError("raw.eigvecs must be a matrix."))
    eltype(matrix) <: Number || throw(ArgumentError("raw.matrix must be numeric."))
    eltype(eigvecs) <: Number || throw(ArgumentError("raw.eigvecs must be numeric."))
    T = eltype(eigvals)
    T <: AbstractFloat || throw(ArgumentError(
        "raw.eigvals must have an AbstractFloat element type, got $T."))

    dim = size(matrix, 1)
    size(matrix, 2) == dim || throw(ArgumentError("raw.matrix must be square."))
    dim >= 2 && ispow2(dim) || throw(ArgumentError(
        "raw.matrix dimension must be a positive-qubit power of two, got $dim."))
    length(eigvals) == dim || throw(ArgumentError(
        "raw.eigvals length $(length(eigvals)) does not match matrix dimension $dim."))
    size(eigvecs) == (dim, dim) || throw(ArgumentError(
        "raw.eigvecs must have size ($dim, $dim), got $(size(eigvecs))."))
    use_default_unitarity_tolerance = isnothing(atol) && isnothing(rtol)
    default_tolerance = T(10 * dim) * eps(T)
    # LAPACK orthogonality error scales with matrix dimension and varies by
    # implementation, especially for clustered eigenvalues.  Keep this
    # allowance separate so the eigenpair and other cache checks stay strict.
    default_unitarity_tolerance = min(T(64 * dim) * eps(T), sqrt(eps(T)))
    atol = isnothing(atol) ? default_tolerance : T(atol)
    rtol = isnothing(rtol) ? default_tolerance : T(rtol)
    isfinite(atol) && atol >= 0 || throw(ArgumentError("atol must be finite and >= 0."))
    isfinite(rtol) && rtol >= 0 || throw(ArgumentError("rtol must be finite and >= 0."))
    unitarity_tolerance = use_default_unitarity_tolerance ?
        default_unitarity_tolerance : max(atol, rtol)

    all(isfinite, matrix) || throw(ArgumentError("raw.matrix must contain only finite values."))
    all(isfinite, eigvals) || throw(ArgumentError("raw.eigvals must contain only finite values."))
    all(isfinite, eigvecs) || throw(ArgumentError("raw.eigvecs must contain only finite values."))
    hermiticity_error = norm(matrix - adjoint(matrix)) / max(norm(matrix), one(T))
    hermiticity_error <= atol + rtol ||
        throw(ArgumentError("raw.matrix must be Hermitian."))
    issorted(eigvals) || throw(ArgumentError("raw.eigvals must be sorted in nondecreasing order."))

    width = last(eigvals) - first(eigvals)
    isfinite(width) && width > zero(T) || throw(ArgumentError(
        "raw.eigvals must have positive finite spectral width."))
    window_tolerance = T(atol) + T(rtol) * max(abs(first(eigvals)), abs(last(eigvals)), one(T))
    first(eigvals) >= -window_tolerance || throw(ArgumentError(
        "raw.eigvals must lie above the algorithmic lower bound 0."))
    last(eigvals) <= T(0.45) + window_tolerance || throw(ArgumentError(
        "raw.eigvals must lie below the algorithmic upper bound 0.45."))
    raw.rescaling_factor isa Real && isfinite(raw.rescaling_factor) &&
        raw.rescaling_factor > 0 ||
        throw(ArgumentError("raw.rescaling_factor must be finite and > 0."))
    raw.shift isa Real && isfinite(raw.shift) ||
        throw(ArgumentError("raw.shift must be finite and real."))
    raw.periodic isa Bool || throw(ArgumentError("raw.periodic must be Bool."))

    has_Lx = haskey(raw, :Lx)
    has_Ly = haskey(raw, :Ly)
    has_periodic_x = haskey(raw, :periodic_x)
    has_periodic_y = haskey(raw, :periodic_y)
    has_Lx == has_Ly || throw(ArgumentError(
        "raw 2D geometry must provide both Lx and Ly."))
    has_periodic_x == has_periodic_y || throw(ArgumentError(
        "raw 2D geometry must provide both periodic_x and periodic_y."))
    if has_periodic_x
        has_Lx || throw(ArgumentError(
            "raw periodic_x/periodic_y metadata requires Lx/Ly."))
        raw.Lx isa Int && raw.Lx >= 1 || throw(ArgumentError(
            "raw.Lx must be a positive Int."))
        raw.Ly isa Int && raw.Ly >= 1 || throw(ArgumentError(
            "raw.Ly must be a positive Int."))
        raw.Lx * raw.Ly == Int(log2(dim)) || throw(ArgumentError(
            "raw.Lx * raw.Ly must match the Hamiltonian qubit count."))
        raw.periodic_x isa Bool || throw(ArgumentError("raw.periodic_x must be Bool."))
        raw.periodic_y isa Bool || throw(ArgumentError("raw.periodic_y must be Bool."))
        raw.periodic == (raw.periodic_x && raw.periodic_y) || throw(ArgumentError(
            "raw.periodic must equal periodic_x && periodic_y."))
    elseif has_Lx
        # Compatibility with older 2D caches, which retained Lx/Ly but not the
        # independent boundaries. Do not infer a cylinder's missing axis data.
        raw.Lx isa Int && raw.Lx >= 1 || throw(ArgumentError(
            "raw.Lx must be a positive Int."))
        raw.Ly isa Int && raw.Ly >= 1 || throw(ArgumentError(
            "raw.Ly must be a positive Int."))
        raw.Lx * raw.Ly == Int(log2(dim)) || throw(ArgumentError(
            "raw.Lx * raw.Ly must match the Hamiltonian qubit count."))
    end

    expected_nu_min = minimum(diff(eigvals))
    nu_tolerance = atol + rtol * max(abs(expected_nu_min), one(T))
    raw.nu_min isa Real && isfinite(raw.nu_min) &&
        abs(raw.nu_min - expected_nu_min) <= nu_tolerance ||
        throw(ArgumentError(
            "raw.nu_min=$(raw.nu_min) is inconsistent with the cached spectrum " *
            "(expected $expected_nu_min)."))

    raw.terms isa AbstractVector || throw(ArgumentError("raw.terms must be a vector."))
    raw.base_coeffs isa AbstractVector || throw(ArgumentError(
        "raw.base_coeffs must be a vector."))
    length(raw.terms) == length(raw.base_coeffs) || throw(ArgumentError(
        "raw.terms and raw.base_coeffs must have the same length."))
    all(value -> value isa Real && isfinite(value), raw.base_coeffs) ||
        throw(ArgumentError("raw.base_coeffs must contain only finite real values."))
    num_qubits = trailing_zeros(dim)
    _validate_local_terms(raw.terms, num_qubits;
        max_support=max(num_qubits, 2), require_support_fits=false)
    has_disordering_terms = haskey(raw, :disordering_terms) && raw.disordering_terms !== nothing
    has_disordering_coeffs = haskey(raw, :disordering_coeffs) && raw.disordering_coeffs !== nothing
    has_disordering_terms == has_disordering_coeffs || throw(ArgumentError(
        "raw.disordering_terms and raw.disordering_coeffs must either both be present or both be nothing."))
    if has_disordering_terms
        raw.disordering_terms isa AbstractVector || throw(ArgumentError(
            "raw.disordering_terms must be a vector when present."))
        raw.disordering_coeffs isa AbstractVector || throw(ArgumentError(
            "raw.disordering_coeffs must be a vector when present."))
        length(raw.disordering_terms) == length(raw.disordering_coeffs) ||
            throw(ArgumentError(
                "raw.disordering_terms and raw.disordering_coeffs must have the same length."))
        _validate_local_terms(raw.disordering_terms, num_qubits;
            max_support=max(num_qubits, 2), require_support_fits=false)
        for coeffs in raw.disordering_coeffs
            coeffs isa AbstractVector || throw(ArgumentError(
                "Each raw disordering coefficient entry must be a vector."))
            length(coeffs) == num_qubits || throw(ArgumentError(
                "Each raw disordering coefficient vector must have length $num_qubits."))
            all(value -> value isa Real && isfinite(value), coeffs) || throw(ArgumentError(
                "raw.disordering_coeffs must contain only finite real values."))
        end
    end

    use_full_spectral_validation = spectral_validation == :full ||
        (spectral_validation == :auto && dim <= full_spectral_max_dim)
    if use_full_spectral_validation
        gram = adjoint(eigvecs) * eigvecs
        unitarity_error = norm(gram - I, Inf)
        unitarity_error <= unitarity_tolerance ||
            throw(ArgumentError("raw.eigvecs must be unitary."))
        residual = matrix * eigvecs - eigvecs * Diagonal(eigvals)
        eigenpair_error = maximum(j -> norm(@view(residual[:, j])), axes(residual, 2))
        eigenpair_error <= max(atol, rtol) || throw(ArgumentError(
            "raw eigvals/eigvecs are inconsistent with raw.matrix."))
    else
        # This O(d^2) mode is an integrity probe for large trusted caches, not a
        # global certificate of V'V=I and HV=VD. Use `:full` for untrusted data.
        inv_sqrt_dim = inv(sqrt(T(dim)))
        probes = (
            fill(Complex{T}(inv_sqrt_dim), dim),
            Complex{T}.((collect(T, 1:dim) .- T(dim + 1) / 2)) |> normalize,
        )
        for probe in probes
            unitary_residual = adjoint(eigvecs) * (eigvecs * probe) - probe
            norm(unitary_residual) / norm(probe) <= unitarity_tolerance ||
                throw(ArgumentError("raw.eigvecs failed the large-fixture unitarity probe."))

            lhs = matrix * (eigvecs * probe)
            rhs = eigvecs * (eigvals .* probe)
            residual_scale = max(norm(lhs), norm(rhs), one(T))
            norm(lhs - rhs) / residual_scale <= max(atol, rtol) || throw(ArgumentError(
                "raw eigvals/eigvecs failed the large-fixture eigenpair probe."))
        end
    end
    return nothing
end

function HamHam(terms::Vector{Vector{Matrix{ComplexF64}}}, coeffs::Vector{Float64},
    num_qubits::Int64, beta::Float64;
    periodic::Bool = true, hermitian_check = false,
    precision::Type{T} = Float64) where {T<:AbstractFloat}
    # Downward precision conversion is rejected; upward promotion is allowed.
    if T !== Float64 && T <: Union{Float16, Float32}
        throw(ArgumentError(
            "Expected $(Complex{T}) term data, got ComplexF64. " *
            "Reconstruct with $(Complex{T}) inputs or use default Float64 precision."))
    end
    isfinite(beta) && beta > 0 || throw(ArgumentError("beta must be finite and > 0."))

    hamiltonian_matrix = _construct_base_ham(terms, coeffs, num_qubits; periodic=periodic)

    rescaled_hamiltonian, rescaling_factor, shift = _rescale_hamiltonian(hamiltonian_matrix)

    rescaled_eigvals, rescaled_eigvecs = eigen(rescaled_hamiltonian)
    rescaled_base_coeffs = coeffs / rescaling_factor
    smallest_bohr_freq = minimum(diff(rescaled_eigvals))

    if hermitian_check
        @assert ishermitian(rescaled_hamiltonian) "The resulting matrix is not Hermitian!"
    end

    bohr_freqs = rescaled_eigvals .- transpose(rescaled_eigvals)
    bohr_dict = create_bohr_dict(bohr_freqs)
    gibbs = Hermitian(_gibbs_in_eigen(rescaled_eigvals, beta))

    return HamHam{T}(
        Matrix(rescaled_hamiltonian),
        bohr_freqs,
        bohr_dict,
        terms,
        rescaled_base_coeffs,
        nothing,  # disordering_terms absent
        nothing,  # disordering_coeffs absent
        rescaled_eigvals,
        rescaled_eigvecs,
        smallest_bohr_freq,
        shift,
        rescaling_factor,
        periodic,
        gibbs,
    )
end

function HamHam(terms::Vector{Vector{Matrix{ComplexF64}}}, coeffs::Vector{Float64},
    disordering_terms::Vector{Vector{Matrix{ComplexF64}}}, disordering_coeffs::Vector{Vector{Float64}},
    num_qubits::Int64, beta::Float64;
    periodic::Bool = true, hermitian_check = false,
    precision::Type{T} = Float64) where {T<:AbstractFloat}
    # Downward precision conversion is rejected; upward promotion is allowed.
    if T !== Float64 && T <: Union{Float16, Float32}
        throw(ArgumentError(
            "Expected $(Complex{T}) term data, got ComplexF64. " *
            "Reconstruct with $(Complex{T}) inputs or use default Float64 precision."))
    end
    isfinite(beta) && beta > 0 || throw(ArgumentError("beta must be finite and > 0."))

    if length(disordering_terms) != length(disordering_coeffs)
        throw(ArgumentError("Number of disordering terms must match number of coefficient vectors"))
    end

    base_hamiltonian = _construct_base_ham(terms, coeffs, num_qubits; periodic=periodic)
    disordering_hamiltonian = _construct_disordering_terms(disordering_terms, disordering_coeffs, num_qubits;
        periodic=periodic)
    disordered_ham = base_hamiltonian + disordering_hamiltonian

    rescaled_hamiltonian, rescaling_factor, shift = _rescale_hamiltonian(disordered_ham)

    rescaled_eigvals, rescaled_eigvecs = eigen(rescaled_hamiltonian)
    rescaled_base_coeffs = coeffs / rescaling_factor
    rescaled_disordering_coeffs = [dc / rescaling_factor for dc in disordering_coeffs]
    smallest_bohr_freq = minimum(diff(rescaled_eigvals))

    if hermitian_check
        @assert ishermitian(rescaled_hamiltonian) "The resulting matrix is not Hermitian!"
    end

    bohr_freqs = rescaled_eigvals .- transpose(rescaled_eigvals)
    bohr_dict = create_bohr_dict(bohr_freqs)
    gibbs = Hermitian(_gibbs_in_eigen(rescaled_eigvals, beta))

    return HamHam{T}(
        Matrix(rescaled_hamiltonian),
        bohr_freqs,
        bohr_dict,
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
        gibbs,
    )
end

# Wrap a single disorder term in the multi-term representation.
function HamHam(terms::Vector{Vector{Matrix{ComplexF64}}}, coeffs::Vector{Float64},
    disordering_term::Vector{Matrix{ComplexF64}}, disordering_coeffs::Vector{Float64},
    num_qubits::Int64, beta::Float64;
    periodic::Bool = true, hermitian_check = false,
    precision::Type{T} = Float64) where {T<:AbstractFloat}

    return HamHam(terms, coeffs, [disordering_term], [disordering_coeffs],
        num_qubits, beta; periodic=periodic, hermitian_check=hermitian_check, precision=precision)
end

"""
    HamHam(raw::NamedTuple, beta; spectral_validation=:auto) -> HamHam{T}

Construct a fully initialised Hamiltonian from builder output.

# Arguments
- `raw`: Named tuple returned by `build_heis_1d` or `build_tfim_2d`.
- `beta`: Algorithm-side inverse temperature for the rescaled spectrum.
- `spectral_validation`: `:full` checks `V'V` and `HV=V*Diagonal(E)`;
  `:probed` is an `O(d^2)` integrity check for trusted caches; `:auto` uses
  `:full` through dimension 128 and `:probed` above it.

# Returns
A `HamHam` with derived Bohr data and Gibbs state.
"""
function HamHam(raw::NamedTuple, beta::Real; spectral_validation::Symbol = :auto)
    _validate_raw_hamiltonian(raw; spectral_validation=spectral_validation)
    T = eltype(raw.eigvals)
    beta_T = T(beta)
    eigvals = Vector{T}(raw.eigvals)
    eigvecs = Matrix{Complex{T}}(raw.eigvecs)
    bohr_freqs = eigvals .- transpose(eigvals)
    bohr_dict = create_bohr_dict(bohr_freqs)
    gibbs = Hermitian(_gibbs_in_eigen(eigvals, beta_T))

    # Accept scalar and multi-term disorder schemas.
    dis_terms, dis_coeffs = _unpack_disordering_fields(raw, T)

    return HamHam{T}(
        Matrix{Complex{T}}(raw.matrix),
        bohr_freqs,
        bohr_dict,
        Vector{Vector{Matrix{Complex{T}}}}(raw.terms),
        Vector{T}(raw.base_coeffs),
        dis_terms,
        dis_coeffs,
        eigvals,
        eigvecs,
        T(raw.nu_min),
        T(raw.shift),
        T(raw.rescaling_factor),
        raw.periodic,
        gibbs,
    )
end

"""
    HamHam(raw::NamedTuple; beta_phys::Real) -> HamHam{T}

Construct a Hamiltonian at physical inverse temperature `beta_phys`.

The constructor sets `\$beta_alg = beta_phys dot raw.rescaling_factor\$` before
forming the Gibbs state. The positional constructor instead accepts `beta_alg`.
"""
function HamHam(
    raw::NamedTuple;
    beta_phys::Real,
    spectral_validation::Symbol = :auto,
)
    rescale = raw.rescaling_factor
    return HamHam(raw, beta_phys * rescale; spectral_validation=spectral_validation)
end

"""
    beta_alg(ham::HamHam{T}, beta_phys::Real)   where T<:AbstractFloat -> T
    beta_phys(ham::HamHam{T}, beta_alg::Real)   where T<:AbstractFloat -> T

Convert inverse temperature between the physical and rescaled Hamiltonians.

The convention is `\$beta_alg = beta_phys dot ham.rescaling_factor\$`. Use these
helpers whenever a sweep crosses the physical/algorithm boundary.
"""
beta_alg(ham::HamHam{T}, beta_phys::Real) where {T<:AbstractFloat} = T(beta_phys * ham.rescaling_factor)
beta_phys(ham::HamHam{T}, beta_alg::Real) where {T<:AbstractFloat} = T(beta_alg / ham.rescaling_factor)

"""
    _unpack_disordering_fields(raw::NamedTuple, T) -> (terms, coeffs)

Return typed disorder terms and coefficients, or `(nothing, nothing)`.
"""
function _unpack_disordering_fields(raw::NamedTuple, ::Type{T}) where {T}
    if !haskey(raw, :disordering_terms) || raw.disordering_terms === nothing
        return (nothing, nothing)
    end
    terms = [Vector{Matrix{Complex{T}}}(t) for t in raw.disordering_terms]
    coeffs = [Vector{T}(c) for c in raw.disordering_coeffs]
    return (terms, coeffs)
end


"""
    build_heis_1d(num_qubits, coeffs; seed, periodic=true,
        disordering_terms=[[Z], [Z,Z]], disorder_strength=0.1) -> NamedTuple

Build one reproducible disordered 1D Heisenberg Hamiltonian.

# Arguments
- `num_qubits`: chain length.
- `coeffs`: `[J_x, J_y, J_z]` uniform exchange couplings.

# Keywords
- `seed`: MersenneTwister seed for the per-site disorder draw.
- `periodic`: Apply periodic boundaries to the base and two-site disorder.
- `disordering_terms`: Pauli terms for disorder, e.g. `[[Z], [Z, Z]]` (default).
- `disorder_strength`: per-coefficient amplitude `c ∈ [0, disorder_strength)`.

# Returns
A named tuple accepted by either `HamHam` constructor.
"""
function build_heis_1d(num_qubits::Int, coeffs::Vector{Float64};
        seed::Int,
        periodic::Bool=true,
        disordering_terms::Vector{Vector{Matrix{ComplexF64}}}=Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
        disorder_strength::Float64=0.1)
    specification = _build_local_heis_1d_specification(
        num_qubits,
        coeffs;
        seed=seed,
        periodic=periodic,
        disordering_terms=disordering_terms,
        disorder_strength=disorder_strength,
    )
    local_hamiltonian = specification.hamiltonian
    base_term_count = specification.base_term_count
    local_terms = _local_terms(local_hamiltonian)
    base_hamiltonian = _materialize_local_terms_1d(
        local_terms[1:base_term_count], num_qubits, 2)
    disordering_ham = _materialize_local_terms_1d(
        local_terms[(base_term_count + 1):end], num_qubits, 2)

    total_ham = Hermitian(Matrix(base_hamiltonian) + Matrix(disordering_ham))
    rescaled_hamiltonian, rescaling_factor, shift = _rescale_hamiltonian(total_ham)
    rescaled_ham = Matrix(rescaled_hamiltonian)
    rescaled_eigvals, rescaled_eigvecs = eigen(Hermitian(rescaled_ham))
    nu_min = minimum(diff(rescaled_eigvals))

    return (
        matrix = rescaled_ham,
        terms = specification.base_patterns,
        base_coeffs = coeffs ./ rescaling_factor,
        disordering_terms = specification.disordering_terms,
        disordering_coeffs = [dc ./ rescaling_factor for dc in specification.disordering_coeffs],
        eigvals = rescaled_eigvals,
        eigvecs = rescaled_eigvecs,
        nu_min = nu_min,
        shift = shift,
        rescaling_factor = rescaling_factor,
        periodic = periodic,
        seed = seed,
        disorder_strength = disorder_strength,
    )
end

"""
    build_tfim_2d(Lx, Ly; J=1.0, h=1.0, seed, periodic_x=true, periodic_y=true,
        disordering_terms=[[Z], [Z,Z]], disorder_strength=1e-3) -> NamedTuple

Build one reproducible disordered 2D transverse-field Ising Hamiltonian.

The model is `\$H = -J sum_(angle.l i,j angle.r) Z_i Z_j - h sum_i X_i + H_dis\$`.
Two-site disorder follows nearest-neighbour lattice bonds.

# Arguments
- `Lx`, `Ly`: Lattice dimensions.

# Keywords
- `J`: Nearest-neighbour `ZZ` coupling.
- `h`: Transverse-field magnitude.
- `seed`: MersenneTwister seed.
- `periodic_x`, `periodic_y`: per-direction BCs.
- `disordering_terms`, `disorder_strength`: as in [`build_heis_1d`], placed on
   the 2D lattice via the 2D builder.

# Returns
A named tuple accepted by either `HamHam` constructor.

Zero disorder can retain exact symmetries and make gap computations
ill-conditioned; the default weak disorder breaks those degeneracies.
"""
function build_tfim_2d(Lx::Int, Ly::Int;
        J::Float64=1.0, h::Float64=1.0,
        seed::Int,
        periodic_x::Bool=true, periodic_y::Bool=true,
        disordering_terms::Vector{Vector{Matrix{ComplexF64}}}=Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
        disorder_strength::Float64=1e-3)

    if Lx < 1 || Ly < 1
        throw(ArgumentError("Lx and Ly must be at least 1; got Lx=$Lx, Ly=$Ly"))
    end
    isfinite(J) || throw(ArgumentError("J must be finite."))
    isfinite(h) || throw(ArgumentError("h must be finite."))
    isfinite(disorder_strength) && disorder_strength >= 0 || throw(ArgumentError(
        "disorder_strength must be finite and >= 0."))

    num_qubits = Lx * Ly
    _validate_local_terms(disordering_terms, num_qubits;
        max_support=2, require_support_fits=false, require_involution=true)
    H_bond = _construct_2d_heisenberg_base(Lx, Ly,
        Vector{Matrix{ComplexF64}}[[Z, Z]], [-J];
        periodic_x=periodic_x, periodic_y=periodic_y)
    # Math: the transverse-field term is $-h sum_i X_i$.
    field_coeffs = fill(-h, num_qubits)
    H_field = _construct_disordering_terms(
        Vector{Matrix{ComplexF64}}[[X]], [field_coeffs], num_qubits)
    base_clean = Hermitian(Matrix(H_bond) + Matrix(H_field))

    rng = MersenneTwister(seed)
    sample_coeffs = [zeros(Float64, num_qubits) for _ in disordering_terms]
    for dc in sample_coeffs
        rand!(rng, dc)
        dc .*= disorder_strength
    end
    disordering_ham = _construct_disordering_terms_2d(Lx, Ly,
        disordering_terms, sample_coeffs;
        periodic_x=periodic_x, periodic_y=periodic_y)

    total_ham = Hermitian(Matrix(base_clean) + Matrix(disordering_ham))
    rescaled_hamiltonian, rescaling_factor, shift = _rescale_hamiltonian(total_ham)
    rescaled_ham = Matrix(rescaled_hamiltonian)
    rescaled_eigvals, rescaled_eigvecs = eigen(Hermitian(rescaled_ham))
    nu_min = minimum(diff(rescaled_eigvals))

    return (
        matrix = rescaled_ham,
        terms = Vector{Matrix{ComplexF64}}[[Z, Z], [X]],
        base_coeffs = [-J / rescaling_factor, -h / rescaling_factor],
        disordering_terms = disordering_terms,
        disordering_coeffs = [dc ./ rescaling_factor for dc in sample_coeffs],
        eigvals = rescaled_eigvals,
        eigvecs = rescaled_eigvecs,
        nu_min = nu_min,
        shift = shift,
        rescaling_factor = rescaling_factor,
        periodic = periodic_x && periodic_y,
        periodic_x = periodic_x,
        periodic_y = periodic_y,
        seed = seed,
        disorder_strength = disorder_strength,
        Lx = Lx, Ly = Ly,
        J = J, h = h,
    )
end


function _construct_base_ham(terms::Vector{Vector{Matrix{ComplexF64}}}, coeffs::Vector{Float64},
    num_qubits::Int64; periodic::Bool = true)

    _validate_local_terms(terms, num_qubits)
    if length(terms) != length(coeffs)
        throw(ArgumentError("The number of terms and coefficients must be equal"))
    end
    all(isfinite, coeffs) || throw(ArgumentError("Hamiltonian coefficients must be finite."))

    hamiltonian::SparseMatrixCSC{ComplexF64} = spzeros(2^num_qubits, 2^num_qubits)
    for (i, term) in enumerate(terms)
        for q in 1:num_qubits
            padded_term = pad_term(term, num_qubits, q; periodic=periodic)  # e.g. term = XX
            hamiltonian += coeffs[i] * padded_term
        end
    end

    return Hermitian(Matrix(hamiltonian))
end

"""
    _pad_two_site_op(term, num_qubits, q1, q2) -> SparseMatrixCSC{ComplexF64}

Place a two-site operator at distinct qubit indices and pad with identities.

The returned sparse matrix follows the left-to-right tensor order of `kron`.
"""
function _pad_two_site_op(term::Vector{Matrix{ComplexF64}}, num_qubits::Int, q1::Int, q2::Int)
    if length(term) != 2
        throw(ArgumentError("_pad_two_site_op expects a 2-site term, got length $(length(term))"))
    end
    if q1 == q2
        throw(ArgumentError("q1 and q2 must be distinct, got q1=q2=$q1"))
    end
    if !(1 <= q1 <= num_qubits) || !(1 <= q2 <= num_qubits)
        throw(ArgumentError("q1 and q2 must be in 1:$num_qubits, got q1=$q1, q2=$q2"))
    end

    # Preserve operator order after sorting sites into tensor order.
    if q1 < q2
        a, b = q1, q2
        op_a, op_b = term[1], term[2]
    else
        a, b = q2, q1
        op_a, op_b = term[2], term[1]
    end

    id_before  = sparse(I, 2^(a - 1), 2^(a - 1))
    id_between = sparse(I, 2^(b - a - 1), 2^(b - a - 1))
    id_after   = sparse(I, 2^(num_qubits - b), 2^(num_qubits - b))

    return kron(id_before, sparse(op_a), id_between, sparse(op_b), id_after)
end

"""
    _construct_2d_heisenberg_base(Lx, Ly, terms, coeffs; periodic_x=true, periodic_y=true)
        -> Hermitian{ComplexF64, Matrix{ComplexF64}}

Build a dense nearest-neighbour Hamiltonian on an `Lx × Ly` square lattice.

Each site contributes right and upward bonds. Periodic directions wrap unless
their length is one; length-two periodic directions retain the package's
double-bond convention.
"""
function _construct_2d_heisenberg_base(Lx::Int64, Ly::Int64,
    terms::Vector{Vector{Matrix{ComplexF64}}}, coeffs::Vector{Float64};
    periodic_x::Bool = true, periodic_y::Bool = true)

    Lx > 0 && Ly > 0 || throw(ArgumentError(
        "Lx and Ly must be > 0, got Lx=$Lx, Ly=$Ly."))
    num_qubits = Lx * Ly
    _validate_local_terms(terms, num_qubits; max_support=2, require_support_fits=false)
    all(length(term) == 2 for term in terms) || throw(ArgumentError(
        "Nearest-neighbour base terms must act on exactly two sites."))
    if length(terms) != length(coeffs)
        throw(ArgumentError("The number of terms and coefficients must be equal"))
    end
    all(isfinite, coeffs) || throw(ArgumentError("Hamiltonian coefficients must be finite."))

    site_index(i, j) = (i - 1) * Ly + (j - 1) + 1

    hamiltonian::SparseMatrixCSC{ComplexF64} = spzeros(2^num_qubits, 2^num_qubits)
    for (k, term) in enumerate(terms)
        for i in 1:Lx, j in 1:Ly
            # Right neighbour (x-direction): (i, j) -> (i+1, j)
            if i < Lx
                hamiltonian += coeffs[k] * _pad_two_site_op(term, num_qubits,
                    site_index(i, j), site_index(i + 1, j))
            elseif periodic_x && Lx > 1
                hamiltonian += coeffs[k] * _pad_two_site_op(term, num_qubits,
                    site_index(Lx, j), site_index(1, j))
            end

            # Up neighbour (y-direction): (i, j) -> (i, j+1)
            if j < Ly
                hamiltonian += coeffs[k] * _pad_two_site_op(term, num_qubits,
                    site_index(i, j), site_index(i, j + 1))
            elseif periodic_y && Ly > 1
                hamiltonian += coeffs[k] * _pad_two_site_op(term, num_qubits,
                    site_index(i, Ly), site_index(i, 1))
            end
        end
    end

    return Hermitian(Matrix(hamiltonian))
end

"""
    _construct_disordering_terms(terms, coeffs, num_qubits; periodic=true)
        -> Hermitian{ComplexF64, Matrix{ComplexF64}}

Build site and nearest-neighbour disorder on a 1D chain.

`periodic` controls the final two-site wrap bond. Use
`_construct_disordering_terms_2d` for lattice bonds in two dimensions.
"""
function _construct_disordering_terms(terms::Vector{Vector{Matrix{ComplexF64}}},
    coeffs::Vector{Vector{Float64}}, num_qubits::Int64; periodic::Bool=true)

    _validate_local_terms(terms, num_qubits)
    length(terms) == length(coeffs) || throw(ArgumentError(
        "The number of disordering terms and coefficient vectors must be equal."))
    disordering_hamiltonian::SparseMatrixCSC{ComplexF64} = spzeros(2^num_qubits, 2^num_qubits)
    for (term, term_coeffs) in zip(terms, coeffs)
        if length(term_coeffs) != num_qubits
            throw(ArgumentError("Each disordering coefficient vector must have length num_qubits ($num_qubits), got $(length(term_coeffs))"))
        end
        all(isfinite, term_coeffs) || throw(ArgumentError(
            "Disordering coefficients must contain only finite values."))
        for q in 1:num_qubits
            disordering_hamiltonian += term_coeffs[q] * pad_term(term, num_qubits, q; periodic=periodic)
        end
    end

    return Hermitian(Matrix(disordering_hamiltonian))
end

"""
    _construct_disordering_terms_2d(Lx, Ly, terms, coeffs; periodic_x=true, periodic_y=true)
        -> Hermitian{ComplexF64, Matrix{ComplexF64}}

Build site and nearest-neighbour disorder on an `Lx × Ly` lattice.

Each site's two-site coefficient is shared by its right and upward bonds, so
the two bond directions are correlated. Terms must act on one or two sites.
"""
function _construct_disordering_terms_2d(Lx::Int64, Ly::Int64,
    terms::Vector{Vector{Matrix{ComplexF64}}},
    coeffs::Vector{Vector{Float64}};
    periodic_x::Bool=true, periodic_y::Bool=true)

    Lx > 0 && Ly > 0 || throw(ArgumentError(
        "Lx and Ly must be > 0, got Lx=$Lx, Ly=$Ly."))
    num_qubits = Lx * Ly
    _validate_local_terms(terms, num_qubits; max_support=2, require_support_fits=false)
    length(terms) == length(coeffs) || throw(ArgumentError(
        "The number of disordering terms and coefficient vectors must be equal."))
    site_index(i, j) = (i - 1) * Ly + (j - 1) + 1

    disordering_hamiltonian::SparseMatrixCSC{ComplexF64} = spzeros(2^num_qubits, 2^num_qubits)
    for (term, term_coeffs) in zip(terms, coeffs)
        if length(term_coeffs) != num_qubits
            throw(ArgumentError("Each disordering coefficient vector must have length num_qubits ($num_qubits), got $(length(term_coeffs))"))
        end
        all(isfinite, term_coeffs) || throw(ArgumentError(
            "Disordering coefficients must contain only finite values."))
        if length(term) == 1
            # Site-local term.
            for i in 1:Lx, j in 1:Ly
                q = site_index(i, j)
                disordering_hamiltonian += term_coeffs[q] * pad_term(term, num_qubits, q; periodic=true)
            end
        elseif length(term) == 2
            # Share each site coefficient between its right and upward bonds.
            for i in 1:Lx, j in 1:Ly
                c = term_coeffs[site_index(i, j)]
                # Right neighbour (x-direction): (i, j) -> (i+1, j)
                if i < Lx
                    disordering_hamiltonian += c * _pad_two_site_op(term, num_qubits,
                        site_index(i, j), site_index(i + 1, j))
                elseif periodic_x && Lx > 1
                    disordering_hamiltonian += c * _pad_two_site_op(term, num_qubits,
                        site_index(Lx, j), site_index(1, j))
                end
                # Up neighbour (y-direction): (i, j) -> (i, j+1)
                if j < Ly
                    disordering_hamiltonian += c * _pad_two_site_op(term, num_qubits,
                        site_index(i, j), site_index(i, j + 1))
                elseif periodic_y && Ly > 1
                    disordering_hamiltonian += c * _pad_two_site_op(term, num_qubits,
                        site_index(i, Ly), site_index(i, 1))
                end
            end
        else
            throw(ArgumentError(
                "_construct_disordering_terms_2d only supports 1- or 2-site terms, " *
                "got length-$(length(term))"))
        end
    end

    return Hermitian(Matrix(disordering_hamiltonian))
end

function _rescaling_data(hamiltonian::Hermitian)
    size(hamiltonian, 1) == size(hamiltonian, 2) ||
        throw(ArgumentError("Hamiltonian must be square."))
    all(isfinite, hamiltonian) || throw(ArgumentError(
        "Hamiltonian must contain only finite values."))

    # Remove a representable scalar gauge before diagonalising. Otherwise a
    # large identity offset can erase a smaller off-diagonal spectral width.
    scalar_gauge = real(hamiltonian[begin, begin])
    centered = Matrix(hamiltonian)
    @inbounds for i in axes(centered, 1)
        centered[i, i] -= scalar_gauge
    end
    centered_hamiltonian = Hermitian(centered)

    margin = 0.1  # Keep the upper endpoint below the algorithmic wrap at 0.5.
    eigenergies = eigvals(centered_hamiltonian)
    smallest_eigval = minimum(eigenergies)
    largest_eigval = maximum(eigenergies)
    spectral_width = largest_eigval - smallest_eigval
    isfinite(spectral_width) && spectral_width > 0 || throw(ArgumentError(
        "Hamiltonian must have positive finite spectral width before rescaling."))

    rescaling_factor = spectral_width * (2 / (1 - margin))
    shift = -scalar_gauge / rescaling_factor - smallest_eigval / rescaling_factor
    isfinite(rescaling_factor) && rescaling_factor > 0 || throw(ArgumentError(
        "Hamiltonian rescaling factor must be finite and > 0."))
    isfinite(shift) || throw(ArgumentError("Hamiltonian shift must be finite."))
    return centered_hamiltonian, smallest_eigval, rescaling_factor, shift
end

"""Return affine factors that map a Hamiltonian spectrum to `[0, 0.45]`."""
function _rescaling_and_shift_factors(hamiltonian::Hermitian)
    _, _, rescaling_factor, shift = _rescaling_data(hamiltonian)
    return rescaling_factor, shift
end

"""Shift before division to rescale a Hamiltonian without scalar-offset cancellation."""
function _rescale_hamiltonian(hamiltonian::Hermitian)
    centered, smallest_eigval, rescaling_factor, shift = _rescaling_data(hamiltonian)
    rescaled = Matrix(centered)
    @inbounds for i in axes(rescaled, 1)
        rescaled[i, i] -= smallest_eigval
    end
    rescaled ./= rescaling_factor
    return Hermitian(rescaled), rescaling_factor, shift
end
