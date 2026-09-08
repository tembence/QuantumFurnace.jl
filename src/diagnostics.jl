# Dense exact diagnostics for small systems.

"""
    EigenDecompositionResult

Leading eigenvalues and biorthogonal left/right eigenvectors.

# Fields
- `eigenvalues`: Modes sorted by `abs(real(lambda))`.
- `right_eigenvectors`, `left_eigenvectors`: Biorthogonal mode columns.
- `spectral_gap`: First resolved rate above the detected stationary space;
  `NaN` when no relaxation rate resolves or an unstable mode is found.
- `im_re_ratios`: Oscillation-to-decay ratio per mode.
"""
struct EigenDecompositionResult
    eigenvalues::Vector{ComplexF64}
    right_eigenvectors::Matrix{ComplexF64}
    left_eigenvectors::Matrix{ComplexF64}
    spectral_gap::Float64
    im_re_ratios::Vector{Float64}
end

"""
    FixedPointResult

Normalised Lindbladian fixed point and its Gibbs-state trace distance.

# Fields
- `fixed_point`: Normalized density matrix from lambda_1 eigenvector.
- `trace_distance`: Trace distance to Gibbs state.
"""
struct FixedPointResult
    fixed_point::Matrix{ComplexF64}
    trace_distance::Float64
end

"""
    DefectResult

Anti-Hermitian defect of the KMS quantum discriminant.

# Fields
- `A_norm`: Operator 2-norm of anti-Hermitian part.
- `H_gap`: Gap of Hermitian part of similarity-transformed Lindbladian.
- `defect_ratio`: A_norm / H_gap.
- `warning`: true if defect_ratio > threshold (advisory only, does NOT gate anything).
- `threshold`: The warning threshold used.
"""
struct DefectResult
    A_norm::Float64
    H_gap::Float64
    defect_ratio::Float64
    warning::Bool
    threshold::Float64
end

"""
    OverlapResult

Observable overlaps with Lindbladian eigenmodes.

# Fields
- `coefficients`: n_obs x n_modes overlap coefficients.
- `observable_names`: Names of observables.
- `initial_state_name`: Name of initial state used.
- `gap_mode_overlap`: legacy |c_2| per observable. This means the second
  retained mode, which need not be a relaxation mode for nonunique kernels.
"""
struct OverlapResult
    coefficients::Matrix{ComplexF64}
    observable_names::Vector{String}
    initial_state_name::String
    gap_mode_overlap::Vector{Float64}
end

"""
    SzSectorLabel

Dominant `Delta_Sz` sector and its weight for one eigenmode.

# Fields
- `delta_sz`: Dominant Delta_Sz quantum number.
- `purity`: Fraction of weight in dominant sector.
- `is_pure`: purity > 0.95.
- `sector_weights`: All Delta_Sz weights.
"""
struct SzSectorLabel
    delta_sz::Float64
    purity::Float64
    is_pure::Bool
    sector_weights::Dict{Float64, Float64}
end

"""
    MultipletGroup

Group of near-degenerate Lindbladian eigenvalues.

# Fields
- `eigenvalue_indices`: Indices into eigenvalue array.
- `mean_eigenvalue`: Mean eigenvalue of the group.
- `sz_labels`: Per-eigenvector SzSectorLabel (may be empty if not yet computed).
"""
struct MultipletGroup
    eigenvalue_indices::Vector{Int}
    mean_eigenvalue::ComplexF64
    sz_labels::Vector{SzSectorLabel}
end

"""
    SpectralModeDiagnostics

Per-mode properties of a biorthogonal Krylov decomposition.

# Fields
- `off_diag_weight`: Hilbert–Schmidt fraction outside the diagonal.
- `c_abs2`: Squared initial-state modal coefficient, or `NaN`.
- `modal_hs_weight`: Scale-invariant value `abs2(c[k]) * norm(R[k])^2`, or `NaN`.
- `mode_spacing`: Complex-plane distance to the next captured eigenvalue.

Channel predictors report spacing in channel-eigenvalue units; gap solvers
first convert channel eigenvalues to generator rates.
"""
struct SpectralModeDiagnostics
    off_diag_weight::Vector{Float64}
    c_abs2::Vector{Float64}
    modal_hs_weight::Vector{Float64}
    mode_spacing::Vector{Float64}
end

"""
    spectral_mode_diagnostics(eigenvalues, R_modes, c=nothing)

Compute coherence, amplitude, and spacing diagnostics for captured modes.

# Arguments
- `eigenvalues`: Captured generator or channel eigenvalues.
- `R_modes`: Corresponding right modes as matrices.
- `c`: Optional initial-state coefficients.

# Returns
A `SpectralModeDiagnostics`; coefficient-dependent fields are `NaN` when
`c === nothing`.
"""
function spectral_mode_diagnostics(
    eigenvalues::AbstractVector{<:Complex},
    R_modes::AbstractVector{<:AbstractMatrix},
    c::Union{Nothing, AbstractVector{<:Complex}} = nothing,
)
    m = length(R_modes)
    length(eigenvalues) == m || throw(ArgumentError(
        "eigenvalues and R_modes must have equal length (got $(length(eigenvalues)), $m)"))
    c === nothing || length(c) == m || throw(ArgumentError(
        "c must match R_modes length when provided (got $(length(c)), $m)"))
    odw = Vector{Float64}(undef, m)
    ca2 = Vector{Float64}(undef, m)
    mhw = Vector{Float64}(undef, m)
    spc = Vector{Float64}(undef, m)
    @inbounds for k in 1:m
        R = R_modes[k]
        hs2 = sum(abs2, R)
        diag2 = 0.0
        for i in 1:min(size(R, 1), size(R, 2))
            diag2 += abs2(R[i, i])
        end
        odw[k] = hs2 > 0 ? clamp(1.0 - diag2 / hs2, 0.0, 1.0) : 0.0
        if c === nothing
            ca2[k] = NaN
            mhw[k] = NaN
        else
            ca2[k] = abs2(c[k])
            mhw[k] = ca2[k] * hs2
        end
        spc[k] = k < m ? abs(eigenvalues[k] - eigenvalues[k + 1]) : Inf
    end
    return SpectralModeDiagnostics(odw, ca2, mhw, spc)
end

"""
    ExactDiagnosticsResult

Combined result from `run_exact_diagnostics`.

# Fields
- `eigen`: Dense eigendecomposition.
- `fixed_point`: Fixed-point comparison.
- `defect`: Detailed-balance defect.
- `overlaps`: One observable-overlap result per initial state.
- `sz_labels`: One symmetry label per mode.
- `multiplets`: Vector{MultipletGroup}, grouped near-degenerate modes.
"""
struct ExactDiagnosticsResult
    eigen::EigenDecompositionResult
    fixed_point::FixedPointResult
    defect::DefectResult
    overlaps::Vector{OverlapResult}
    sz_labels::Vector{SzSectorLabel}
    multiplets::Vector{MultipletGroup}
end

"""
    extract_leading_eigendata(L::Matrix{ComplexF64}; n_modes::Int=20) -> EigenDecompositionResult

Extract leading left and right modes from a dense Lindbladian.

# Keywords
- `n_modes`: Maximum number of modes to retain.

# Returns
An `EigenDecompositionResult` sorted by `abs(real(lambda))`, with left modes
constructed so `\$V_L^dagger V_R = I\$`.
"""
function extract_leading_eigendata(L::Matrix{ComplexF64}; n_modes::Int=20)
    d2 = size(L, 1)
    n_modes = min(n_modes, d2)

    F = eigen(L)

    # Sort modes by decay rate, placing the stationary mode first.
    perm = sortperm(abs.(real.(F.values)))
    eigenvalues = F.values[perm[1:n_modes]]

    V_full = F.vectors[:, perm]
    V_right = V_full[:, 1:n_modes]

    # Rows of `inv(V_full)` are the biorthogonal left eigenvectors.
    V_inv = inv(V_full)
    V_left = V_inv[1:n_modes, :]'

    residuals = [norm(L*F.vectors[:,i]-F.values[i]*F.vectors[:,i]) /
        norm(F.vectors[:,i]) for i in eachindex(F.values)]
    spectrum = spectral_gap_diagnostics(F.values,residuals;
        operator_scale=opnorm(L),complete=true)
    spectral_gap = spectrum.spectral_gap

    im_re_ratios = Vector{Float64}(undef, n_modes)
    im_re_ratios[1] = 0.0
    for k in 2:n_modes
        im_re_ratios[k] = abs(imag(eigenvalues[k])) / max(abs(real(eigenvalues[k])), 1e-30)
    end

    return EigenDecompositionResult(eigenvalues, V_right, V_left, spectral_gap, im_re_ratios)
end

"""
    compute_fixed_point_distance(eigen_result::EigenDecompositionResult, gibbs::Hermitian) -> FixedPointResult

Legacy normalisation of the first right mode, including Hermitian repair,
and comparison with Gibbs. This helper assumes a unique physical stationary
mode; use `krylov_spectral_gap(...).fixed_point_diagnostics` for raw validity
and kernel-aware evidence. It is not a validity check for nonunique/invalid maps.

# Returns
A `FixedPointResult` containing the density matrix and trace distance.
"""
function compute_fixed_point_distance(eigen_result::EigenDecompositionResult, gibbs::Hermitian)
    fp_vec = eigen_result.right_eigenvectors[:, 1]
    dim = isqrt(length(fp_vec))
    fp_dm = reshape(copy(fp_vec), dim, dim)

    # Fix the arbitrary eigenvector phase before Hermitian projection.
    _normalize_stationary_mode!(fp_dm)

    dist = trace_distance_h(Hermitian(fp_dm), gibbs)
    return FixedPointResult(fp_dm, dist)
end

"""
    compute_anti_hermitian_defect(L::Matrix{ComplexF64}, gibbs::Hermitian) -> DefectResult

Compute the anti-Hermitian defect of the KMS quantum discriminant.

# Returns
A `DefectResult` with a legacy ratio using the second-smallest absolute
Hermitian eigenvalue. This heuristic assumes a unique resolved zero mode;
it is not a kernel-aware gap diagnostic. Use `workspace_diagnostics` and
`spectral_gap_diagnostics` for stationary-manifold and reliability evidence.
"""
function compute_anti_hermitian_defect(L::Matrix{ComplexF64}, gibbs::Hermitian)
    # Math: $D = H + A$, where $H = (D + D^dagger)/2$ and $A = (D-D^dagger)/2$.
    D_matrix = materialize_discriminant(L, gibbs)
    H_part, A_part = hermitian_antihermitian_split(D_matrix)

    A_norm = opnorm(A_part)

    H_eigenvalues = eigvals(Hermitian(H_part))
    sorted_H_abs = sort(abs.(H_eigenvalues))
    H_gap = sorted_H_abs[2]

    defect_ratio = A_norm / max(H_gap, 1e-30)

    threshold = 0.1
    warning = defect_ratio > threshold
    if warning
        @warn "Anti-Hermitian defect ratio $(round(defect_ratio; digits=4)) > $(threshold) threshold. " *
              "Non-normality effects may cause oscillatory transients -- consider oscillatory fit model."
    end

    return DefectResult(A_norm, H_gap, defect_ratio, warning, threshold)
end

"""
    compute_overlap_coefficients(eigen_result, observables, observable_names, rho0, rho_beta;
                                  n_modes=20, initial_state_name="custom") -> OverlapResult

Compute observable coefficients in a biorthogonal Lindbladian expansion.

# Returns
An `OverlapResult` with one row per observable and one column per retained mode.
"""
function compute_overlap_coefficients(
    eigen_result::EigenDecompositionResult,
    observables::Vector{<:Matrix{<:Complex}},
    observable_names::Vector{String},
    rho0::Matrix{<:Complex},
    rho_beta::Hermitian;
    n_modes::Int=20,
    initial_state_name::String="custom",
)
    dim = size(rho0, 1)
    n_modes_actual = min(n_modes, length(eigen_result.eigenvalues))
    n_obs = length(observables)
    rho_diff = rho0 - Matrix(rho_beta)

    coeffs = zeros(ComplexF64, n_obs, n_modes_actual)

    for k in 1:n_modes_actual
        R_k = reshape(eigen_result.right_eigenvectors[:, k], dim, dim)
        L_k = reshape(eigen_result.left_eigenvectors[:, k], dim, dim)

        # Math: $c_k = tr(O R_k) tr(L_k^dagger (rho_0-rho_beta))$.
        lk_factor = dot(vec(L_k), vec(rho_diff))

        for (i, O) in enumerate(observables)
            ok_factor = tr(O * R_k)
            coeffs[i, k] = ok_factor * lk_factor
        end
    end

    gap_mode_overlap = Float64[abs(coeffs[i, 2]) for i in 1:n_obs]

    return OverlapResult(coeffs, observable_names, initial_state_name, gap_mode_overlap)
end

"""
    compute_sz_labels(eigen_result, eigvecs, n_qubits; n_modes=20) -> Vector{SzSectorLabel}

Assign a dominant `Delta_Sz` sector to each Lindbladian right mode.

# Arguments
- `eigen_result`: Captured right modes.
- `eigvecs`: Columns defining the working basis.
- `n_qubits`: System size.
- `n_modes`: Maximum number of modes to label.

# Returns
A vector of `SzSectorLabel` values.
"""
function compute_sz_labels(eigen_result::EigenDecompositionResult, eigvecs::Matrix{<:Complex},
                            n_qubits::Int; n_modes::Int=20)
    dim = size(eigvecs, 1)
    n_modes_actual = min(n_modes, length(eigen_result.eigenvalues))

    # Math: $S_z = 1/2 sum_i Z_i$.
    Sz_comp = zeros(ComplexF64, dim, dim)
    for site in 1:n_qubits
        Sz_comp .+= Matrix{ComplexF64}(pad_term([Z], n_qubits, site))
    end
    Sz_comp ./= 2

    V = eigvecs
    Sz_eigen = V' * Sz_comp * V
    sz_vals = real.(diag(Sz_eigen))

    labels = Vector{SzSectorLabel}(undef, n_modes_actual)

    for k in 1:n_modes_actual
        M_k = reshape(eigen_result.right_eigenvectors[:, k], dim, dim)
        weights = abs2.(M_k)

        delta_sz_map = Dict{Float64, Float64}()
        for j in 1:dim, i in 1:dim
            w = weights[i, j]
            w < 1e-14 && continue
            dsz = round(sz_vals[i] - sz_vals[j]; digits=6)
            delta_sz_map[dsz] = get(delta_sz_map, dsz, 0.0) + w
        end

        total_weight = sum(values(delta_sz_map))

        dominant_dsz = 0.0
        dominant_weight = 0.0
        for (dsz, wt) in delta_sz_map
            if wt > dominant_weight
                dominant_weight = wt
                dominant_dsz = dsz
            end
        end

        purity = dominant_weight / max(total_weight, 1e-30)
        is_pure = purity > 0.95

        labels[k] = SzSectorLabel(dominant_dsz, purity, is_pure, delta_sz_map)
    end

    return labels
end

"""
    compute_sz_labels(eigen_result, hamiltonian::HamHam; n_modes=20) -> Vector{SzSectorLabel}

Label modes in `hamiltonian.eigvecs`.
"""
function compute_sz_labels(eigen_result::EigenDecompositionResult, hamiltonian::HamHam;
                            n_modes::Int=20)
    n_qubits = Int(log2(size(hamiltonian.data, 1)))
    return compute_sz_labels(eigen_result, hamiltonian.eigvecs, n_qubits; n_modes=n_modes)
end

"""
    detect_multiplets(eigenvalues::Vector{ComplexF64}; rel_tol=0.01) -> Vector{MultipletGroup}

Group adjacent eigenvalues by relative complex-plane spacing.

# Keywords
- `rel_tol`: Maximum relative spacing within a group.

# Returns
Multiplets ordered by eigenvalue magnitude.
"""
function detect_multiplets(eigenvalues::Vector{ComplexF64}; rel_tol::Float64=0.01)
    n = length(eigenvalues)
    n == 0 && return MultipletGroup[]

    sorted_perm = sortperm(abs.(eigenvalues))
    sorted_vals = eigenvalues[sorted_perm]

    groups = Vector{MultipletGroup}()
    current_indices = [sorted_perm[1]]
    current_sum = sorted_vals[1]

    for i in 2:n
        idx = sorted_perm[i]
        val = sorted_vals[i]
        prev_val = sorted_vals[i-1]

        denom = max(abs(val), abs(prev_val), 1e-10)
        if abs(val - prev_val) / denom < rel_tol
            push!(current_indices, idx)
            current_sum += val
        else
            mean_val = current_sum / length(current_indices)
            push!(groups, MultipletGroup(copy(current_indices), mean_val, SzSectorLabel[]))
            current_indices = [idx]
            current_sum = val
        end
    end

    mean_val = current_sum / length(current_indices)
    push!(groups, MultipletGroup(copy(current_indices), mean_val, SzSectorLabel[]))

    return groups
end

"""
    run_exact_diagnostics(L, hamiltonian, gibbs; kwargs...) -> ExactDiagnosticsResult

Run the complete dense diagnostic bundle.

# Arguments
- `L`: Dense Lindbladian superoperator.
- `hamiltonian`: Hamiltonian and basis data.
- `gibbs`: Gibbs state in the selected working basis.

# Keywords
- `basis_eigvecs`: Working basis; defaults to the Hamiltonian eigenbasis.
- `observables`, `observable_names`: Optional observable set and labels.
- `initial_states`, `initial_state_names`: Optional initial states and labels.
- `n_modes`: Maximum number of modes.

# Returns
An `ExactDiagnosticsResult`.
"""
function run_exact_diagnostics(
    L::Matrix{ComplexF64},
    hamiltonian::HamHam,
    gibbs::Hermitian;
    basis_eigvecs::Union{Nothing, Matrix{<:Complex}}=nothing,
    observables::Union{Nothing, Vector{<:Matrix{<:Complex}}}=nothing,
    observable_names::Union{Nothing, Vector{String}}=nothing,
    initial_states::Union{Nothing, Vector{<:Matrix{<:Complex}}}=nothing,
    initial_state_names::Union{Nothing, Vector{String}}=nothing,
    n_modes::Int=20,
)
    dim = size(hamiltonian.data, 1)
    n = Int(log2(dim))

    V = basis_eigvecs === nothing ? hamiltonian.eigvecs : Matrix{ComplexF64}(basis_eigvecs)

    eigen_result = extract_leading_eigendata(L; n_modes=n_modes)

    fp_result = compute_fixed_point_distance(eigen_result, gibbs)

    defect_result = if basis_eigvecs === nothing
        compute_anti_hermitian_defect(L, gibbs)
    else
        # The discriminant implementation requires its Gibbs state to be
        # diagonal in the declared working basis. Rotate only this diagnostic
        # into the Hamiltonian eigenbasis; eigendata and overlaps remain in V.
        L_ham = _change_dense_superoperator_basis(
            L, V, Matrix{ComplexF64}(hamiltonian.eigvecs))
        gibbs_work_expected = Hermitian(
            V' * hamiltonian.eigvecs * Matrix(hamiltonian.gibbs) *
            hamiltonian.eigvecs' * V)
        isapprox(gibbs, gibbs_work_expected; atol = 1e-12, rtol = 0) ||
            throw(ArgumentError(
                "gibbs is inconsistent with hamiltonian.gibbs in basis_eigvecs."))
        compute_anti_hermitian_defect(L_ham, hamiltonian.gibbs)
    end

    # Build default observables if not provided
    if observables === nothing
        # Z1 in working basis
        Z1_comp = Matrix{ComplexF64}(pad_term([Z], n, 1))
        Z1_eigen = Matrix{ComplexF64}(V' * Z1_comp * V)
        # H in working basis (diagonal when V = hamiltonian.eigvecs, non-diagonal otherwise)
        H_eigen = Matrix{ComplexF64}(V' * hamiltonian.data * V)
        observables = Matrix{ComplexF64}[Z1_eigen, H_eigen]
        observable_names = String["Z1", "H"]
    end

    # Build default initial states if not provided
    if initial_states === nothing
        # |0>^n (all spins up) -- transform to working basis
        psi0_comp = zeros(ComplexF64, dim)
        psi0_comp[1] = 1.0
        psi0_eigen = V' * psi0_comp
        rho_up = psi0_eigen * psi0_eigen'

        # |+>^n (all X-plus) -- transform to working basis
        psi_plus_comp = fill(ComplexF64(1 / sqrt(2^n)), 2^n)
        psi_plus_eigen = V' * psi_plus_comp
        rho_plus = psi_plus_eigen * psi_plus_eigen'

        # I/dim (maximally mixed) -- same in any basis
        rho_mixed = Matrix{ComplexF64}(I(dim) / dim)

        initial_states = Matrix{ComplexF64}[rho_up, rho_plus, rho_mixed]
        initial_state_names = String["all_up", "all_plus", "maximally_mixed"]
    end

    overlaps_vec = OverlapResult[]
    for (rho0, name) in zip(initial_states, initial_state_names)
        overlap = compute_overlap_coefficients(
            eigen_result, observables, observable_names, rho0, gibbs;
            n_modes=n_modes, initial_state_name=name,
        )
        push!(overlaps_vec, overlap)
    end

    sz_labels = compute_sz_labels(eigen_result, V, n; n_modes=n_modes)

    # Multiplet detection
    multiplets = detect_multiplets(eigen_result.eigenvalues)

    # Fill multiplet sz_labels from computed labels
    for group in multiplets
        for idx in group.eigenvalue_indices
            if idx <= length(sz_labels)
                push!(group.sz_labels, sz_labels[idx])
            end
        end
    end

    return ExactDiagnosticsResult(eigen_result, fp_result, defect_result,
                                   overlaps_vec, sz_labels, multiplets)
end

"""One numerical check, with separate absolute/scaled quantities and evidence scope.
Unavailable quantities are `nothing`; a skipped check is never a pass.
"""
struct DiagnosticCheck{Q,T}
    status::Symbol
    quantity::Q
    tolerance::T
    method::Symbol
    scope::Symbol
    evidence::Symbol
    message::String
end

"""Immutable collection of checks on the implemented generator, in its raw clock."""
struct GibbsDiagnostics{C,P,R}
    checks::C
    parent_spectrum::P
    resources::R
    uniqueness::Symbol
end

_diagnostic(status, quantity, tolerance, method, scope, message) =
    DiagnosticCheck(status, quantity, tolerance, method, scope, :numerical, message)
_skipped(message) = _diagnostic(:not_run, nothing, nothing, :none, :none, message)

function _diagnostic_controls(rtol, dense_max_dim, max_dense_bytes)
    isfinite(rtol) && rtol >= 0 || throw(ArgumentError("rtol must be finite and nonnegative."))
    dense_max_dim >= 0 || throw(ArgumentError("dense_max_dim must be nonnegative."))
    max_dense_bytes >= 0 || throw(ArgumentError("max_dense_bytes must be nonnegative."))
end

# No additive absolute floor: changing the clock must not conceal a defect.
_scaled_residual(value, scale) = scale > 0 ? value / scale : (iszero(value) ? zero(value) : oftype(value, Inf))
function _residual_check(value, scale, rtol, method, scope, message)
    relative = _scaled_residual(value, scale)
    status = isfinite(relative) && relative <= rtol ? :pass : :fail
    return _diagnostic(status, (absolute=value, scaled=relative, scale=scale),
        rtol, method, scope, message)
end

"""
    state_diagnostics(rho; rtol=1e-9, dense_max_dim=64, max_dense_bytes=64*1024^2)

Inspect the raw state without repairing it. Positivity uses the Hermitian part
only after the Hermiticity check passes, within an explicit allocation budget.
"""
function state_diagnostics(rho::AbstractMatrix; rtol::Real=1e-9,
    dense_max_dim::Integer=64, max_dense_bytes::Integer=64*1024^2)
    _diagnostic_controls(rtol, dense_max_dim, max_dense_bytes)
    d = size(rho, 1)
    d > 0 && size(rho, 2) == d || throw(ArgumentError("State must be nonempty and square."))
    finite = all(isfinite, rho)
    fin = _diagnostic(finite ? :pass : :fail, finite, nothing, :entries, :complete,
        "State entries must be finite; no repair was applied.")
    if !finite
        skipped = _skipped("Nonfinite state; supply finite entries.")
        return (; finiteness=fin, trace=skipped, hermiticity=skipped, positivity=skipped)
    end
    trace_check = _residual_check(abs(tr(rho)-1), one(real(float(tr(rho)))), rtol,
        :trace, :complete, "Supply a unit-trace state; no normalisation was applied.")
    herm = _residual_check(norm(rho-rho'), norm(rho), rtol, :frobenius,
        :complete, "Supply a Hermitian state; no symmetrisation was applied.")
    bytes = big(8)*d^2*sizeof(float(eltype(rho)))
    positivity = if herm.status != :pass
        _skipped("Positivity unavailable for a non-Hermitian state.")
    elseif d > dense_max_dim || bytes > max_dense_bytes
        _skipped("Positivity budget exceeded; increase dense_max_dim/max_dense_bytes.")
    else
        minimum_value = eigmin(Hermitian((rho+rho')/2))
        _diagnostic(minimum_value >= -rtol ? :pass : :fail,
            (minimum_eigenvalue=minimum_value, negative_part=max(-minimum_value, zero(minimum_value))),
            rtol, :hermitian_eigenvalues, :complete, "State must be positive semidefinite; no clipping was applied.")
    end
    return (; finiteness=fin, trace=trace_check, hermiticity=herm, positivity)
end

"""
    workspace_diagnostics(ws, config::Config{Lindbladian}, ham; rho=nothing, ...)

Check the compiled full generator in the Hamiltonian eigenbasis. Stationarity
and trace preservation use exact matrix-free actions; the normalising operator
scale is a deterministic random-probe lower estimate, explicitly labelled.
Dense KMS/kernel checks default to Hilbert dimension <=16 and a 64 MiB working
allocation estimate (including eigensolver temporaries). Neither a small Gibbs
residual nor random probes establish uniqueness or a global gap. `rho`, when
supplied, is inspected in this same basis without repair. Unknown transform tails
remain unavailable; these checks do not certify a continuum implementation.
"""
function workspace_diagnostics(ws::Workspace{KrylovSpectrum}, config::Config{Lindbladian},
    ham::HamHam; rho::Union{Nothing,AbstractMatrix}=nothing, rtol::Real=1e-9,
    dense_max_dim::Integer=16, max_dense_bytes::Integer=64*1024^2,
    probes::Integer=3, seed::Integer=0x70606)
    _diagnostic_controls(rtol, dense_max_dim, max_dense_bytes)
    probes > 0 || throw(ArgumentError("probes must be positive."))
    config.domain isa TrotterDomain && throw(ArgumentError(
        "workspace_diagnostics currently requires the Hamiltonian eigenbasis; TrotterDomain is unsupported."))
    _validate_reused_krylov_workspace(ws, config, ham,
        config.domain isa TrotterDomain ? ws.ham_or_trott : nothing, ws.jumps)
    d = size(ham.data,1)
    rho === nothing || size(rho) == (d,d) || throw(ArgumentError(
        "rho must match the Hamiltonian dimension ($d, $d)."))
    CT = eltype(ws.G_left)
    gibbs = Matrix{CT}(ham.gibbs)
    rng = MersenneTwister(seed)
    scale = zero(real(zero(CT)))
    for _ in 1:probes
        v = randn(rng, CT, d, d)
        v ./= norm(v)
        scale = max(scale, norm(apply_lindbladian!(ws,v,config,ham)))
    end
    stationary = _residual_check(norm(apply_lindbladian!(ws,gibbs,config,ham)),
        scale*norm(gibbs), rtol, :matrix_free_action, :complete_action_probed_scale,
        "Gibbs stationarity only; refine the filter/grid if the scaled residual fails.")
    identity_d = Matrix{CT}(I,d,d)
    tp = _residual_check(norm(apply_adjoint_lindbladian!(ws,identity_d,config,ham)),
        scale*sqrt(d), rtol, :matrix_free_adjoint, :complete_action_probed_scale,
        "Trace preservation requires L†(I)=0; check gain/loss assembly on failure.")
    weights = real.(diag(gibbs))
    faithful = all(isfinite, weights) && minimum(weights) > 0
    ratio = faithful ? minimum(weights)/maximum(weights) : zero(eltype(weights))
    well_conditioned = faithful && ratio > eps(eltype(weights))
    conditioning = _diagnostic(!faithful ? :fail : well_conditioned ? :pass : :inconclusive,
        (minimum_weight=minimum(weights), reciprocal_condition=ratio, faithful=faithful),
        eps(eltype(weights)), :gibbs_weights, :complete,
        "Underflow or conditioning can invalidate inverse-Gibbs diagnostics; use higher precision or lower beta_phys.")
    state = rho === nothing ? _skipped("No raw state supplied.") :
        state_diagnostics(rho;rtol,dense_max_dim,max_dense_bytes)
    # Conservative working-set estimate: full operator, parent, SVD/eigen copies
    # and scratch. No dense superoperator is allocated before this gate.
    bytes = big(16)*d^4*sizeof(CT) + big(16)*d^2*sizeof(CT)
    permitted = d <= dense_max_dim && bytes <= max_dense_bytes
    parent_spectrum = nothing
    kms = _skipped(permitted ? "Gibbs faithfulness/conditioning gate failed." :
        "Dense KMS budget exceeded; increase dense_max_dim/max_dense_bytes.")
    kernel = _skipped("Complete kernel requires a passing KMS/positivity/stationarity check.")
    uniqueness = :not_established
    if permitted && well_conditioned
        L = Matrix{CT}(undef,d^2,d^2)
        basis = zeros(CT,d,d)
        for j in 1:d^2
            fill!(basis,0); basis[j] = 1
            L[:,j] .= vec(apply_lindbladian!(ws,basis,config,ham))
        end
        parent = materialize_kms_parent(L,gibbs)
        pn = opnorm(parent)
        tolerance = 10*d^2*eps(typeof(real(zero(CT))))*pn
        parent_spectrum = kms_parent_spectrum(parent,gibbs;
            kernel_tolerance=tolerance, hermiticity_tolerance=rtol)
        kms = _residual_check(opnorm(parent-parent')/2,pn,rtol,
            :dense_discriminant, :complete, "KMS self-adjointness of the implemented generator; refine Time grids on failure.")
        if kms.status == :pass && stationary.status == :pass && tp.status == :pass &&
            parent_spectrum.minimum_eigenvalue >= -tolerance
            count = parent_spectrum.kernel_count
            uniqueness = count > 1 ? :nonunique : count == 1 ? :established : :not_established
            kernel = _diagnostic(count >= 1 ? :pass : :inconclusive,
                (kernel_count=count, first_positive_rate=parent_spectrum.first_positive_eigenvalue),
                tolerance,:kms_parent_spectrum,:complete_small_system,
                count > 1 ? "Nonunique stationary space; add couplings that connect the conserved sectors." :
                "Complete finite-system numerical kernel at the reported resolution; no all-size theorem.")
        end
    end
    return GibbsDiagnostics((; stationarity=stationary, trace_preservation=tp,
        conditioning, state, kms, kernel, tails=_skipped("No integrated transform-tail evidence supplied.")),
        parent_spectrum, (; dense_permitted=permitted, estimated_dense_bytes=bytes,
            dense_max_dim,max_dense_bytes,probes,seed,operator_scale=scale,clock=:raw_generator), uniqueness)
end

"""
    spectral_gap_diagnostics(eigenvalues, residuals; operator_scale, complete=false,
                             channel_delta=nothing)

Classify raw generator eigenvalues (or raw channel multipliers when delta is
supplied). Zero-compatible modes satisfy `abs(lambda) <= 5*residual + roundoff`;
this is numerical resolution evidence, not proof of an exact zero. The first
resolved decay rate excludes the entire detected stationary space. Positive-real
generator modes / channel moduli above one are unstable, never absolute-valued
into decay. `complete=true` is reserved for a complete dense spectrum.
Ritz residuals of nonnormal operators do not certify eigenvalue errors; partial
spectra always have inconclusive global-gap reliability.
"""
function spectral_gap_diagnostics(values::AbstractVector, residuals::AbstractVector;
    operator_scale::Real, complete::Bool=false, channel_delta::Union{Nothing,Real}=nothing)
    length(values) == length(residuals) || throw(ArgumentError("One residual is required per eigenvalue."))
    isfinite(operator_scale) && operator_scale >= 0 || throw(ArgumentError("operator_scale must be finite and nonnegative."))
    channel_delta === nothing || (isfinite(channel_delta) && channel_delta > 0) ||
        throw(ArgumentError("channel_delta must be finite and positive."))
    all(isfinite,values) && all(x -> isfinite(x) && x >= 0,residuals) ||
        throw(ArgumentError("Eigenvalues and nonnegative residuals must be finite."))
    T = typeof(float(real(zero(eltype(values)))))
    scale = max(T(operator_scale), maximum(abs,values;init=zero(T)))
    roundoff = 10*max(length(values),1)*eps(T)*scale
    thresholds = 5 .* residuals .+ roundoff
    offsets = channel_delta === nothing ? values : values .- 1
    stationary = findall(i -> abs(offsets[i]) <= thresholds[i], eachindex(values))
    residual_ambiguous = filter(i -> abs(offsets[i]) > roundoff, stationary)
    decay = Int[]; unstable = Int[]; unresolved = Int[]
    rates = zeros(T,length(values))
    for i in eachindex(values)
        growth = channel_delta === nothing ? real(values[i]) : abs(values[i])-1
        rates[i] = channel_delta === nothing ? -real(values[i]) :
            (iszero(values[i]) ? T(Inf) : -log(abs(values[i]))/channel_delta)
        if growth > thresholds[i]
            push!(unstable,i)
        elseif i in stationary
            continue
        elseif -growth > thresholds[i]
            push!(decay,i)
        else
            push!(unresolved,i)
        end
    end
    gap_index = isempty(decay) ? nothing : decay[argmin(rates[decay])]
    candidate = gap_index === nothing ? nothing : rates[gap_index]
    status = !isempty(unstable) ? :fail :
        complete && !isempty(stationary) && isempty(unresolved) && isempty(residual_ambiguous) ? :pass : :inconclusive
    uniqueness = complete && status == :pass ?
        (length(stationary) == 1 ? :established : :nonunique) : :not_established
    # Even in complete spectra, all-zero or unresolved spectra have no resolved
    # relaxation rate. Retain detected kernel evidence without inventing a gap.
    gap_status = status == :pass && candidate === nothing ? :inconclusive : status
    return (; status=gap_status, uniqueness, detected_zero_modes=length(stationary),
        stationary_indices=stationary, unresolved_indices=unresolved, unstable_indices=unstable,
        residual_ambiguous_indices=residual_ambiguous,
        first_resolved_rate=candidate, gap_index, zero_thresholds=thresholds,
        operator_scale=scale, complete, evidence=:numerical,
        zero_classification=:compatible_with_zero,
        scope=complete ? :complete_small_system : :captured_ritz_pairs,
        spectral_gap=isempty(unstable) && isempty(unresolved) && isempty(residual_ambiguous) && !isempty(stationary) && candidate !== nothing ? candidate : T(NaN))
end

# Preserve raw eigensolver modes. Phase/trace normalisation is explicit and is
# followed by validity tests, with no Hermitian projection or positivity repair.
function _spectral_fixed_point(vecs, indices, dim; rtol=1e-9)
    for i in indices
        raw = reshape(copy(vecs[i]),dim,dim)
        trace_value = tr(raw)
        abs(trace_value) > sqrt(eps(typeof(real(trace_value))))*sqrt(dim)*norm(raw) || continue
        normalized = raw / trace_value
        checks = state_diagnostics(normalized;rtol)
        valid = all(getproperty(checks,k).status == :pass for k in (:finiteness,:trace,:hermiticity,:positivity))
        return (; state=normalized, checks, valid, index=i, raw_trace=trace_value,
            normalisation_multiplier=inv(trace_value), repair_norm=0.0)
    end
    return (; state=fill(ComplexF64(NaN),dim,dim),
        checks=_skipped("No zero-compatible mode with a stable nonzero trace; request more modes."),
        valid=false,index=nothing,raw_trace=nothing,normalisation_multiplier=nothing,repair_norm=0.0)
end
