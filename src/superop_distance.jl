# Matrix-free trace-norm comparisons of Lindbladian and channel propagators.
# Exact channel iteration avoids contaminating small implementation errors with a
# truncated spectral reconstruction.
# Math: $T(rho, sigma) = 1/2 norm(rho - sigma, 1)$.
# Each arm rotates through its own working basis before common-basis comparison.

"""
    PropagatorArm(apply!; kind, delta=NaN, basis=nothing, label="")

Matrix-free generator or channel action with basis and time-step metadata.

# Arguments
- `apply!`: in-place action `apply!(out, x)` in the arm's working basis.

# Keywords
- `kind`: `:lindbladian` for Krylov exponentiation or `:channel` for iteration.
- `delta`: positive channel step; ignored for Lindbladians.
- `basis`: working-basis vectors in the common basis, or `nothing` for identity.
- `label`: description copied into comparison results.
"""
struct PropagatorArm{F}
    apply!::F
    kind::Symbol
    delta::Float64
    basis::Union{Nothing, Matrix{ComplexF64}}
    label::String
end

function PropagatorArm(apply!::F; kind::Symbol, delta::Real = NaN,
                       basis::Union{Nothing, AbstractMatrix} = nothing,
                       label::AbstractString = "") where {F}
    kind in (:lindbladian, :channel) ||
        throw(ArgumentError("PropagatorArm kind must be :lindbladian or :channel (got :$kind)"))
    if kind === :channel
        (isfinite(delta) && delta > 0) ||
            throw(ArgumentError("a :channel PropagatorArm requires delta > 0 (got $delta)"))
    end
    b = basis === nothing ? nothing : Matrix{ComplexF64}(basis)
    return PropagatorArm{F}(apply!, kind, Float64(delta), b, String(label))
end

# Rotate a common-basis operator into the arm's working basis: V' X V.
_rotate_in(arm::PropagatorArm, X::AbstractMatrix) =
    arm.basis === nothing ? Matrix{ComplexF64}(X) : arm.basis' * X * arm.basis

# Rotate a working-basis operator back to the common basis: V X V'.
_rotate_out(arm::PropagatorArm, X::AbstractMatrix) =
    arm.basis === nothing ? X : arm.basis * X * arm.basis'

"""
    _iterate_channel_states(apply!, rho_0_work, k_grid; hermitize, renormalize)
        -> (states, ks_sorted, matvecs)

Iterate a channel once through `maximum(k_grid)` and collect requested states.
Returns states at sorted unique steps, those steps, and the matvec count.
"""
function _iterate_channel_states(apply!::F, rho_0_work::Matrix{ComplexF64},
                                 k_grid::AbstractVector{<:Integer};
                                 hermitize::Bool = true,
                                 renormalize::Bool = true) where {F}
    d = size(rho_0_work, 1)
    ks = sort(unique(Int.(k_grid)))
    ks[1] >= 0 || throw(ArgumentError("k_grid must be non-negative (got min $(ks[1]))"))
    pos = Dict(k => i for (i, k) in enumerate(ks))
    states = Vector{Matrix{ComplexF64}}(undef, length(ks))
    rho = copy(rho_0_work)
    out = Matrix{ComplexF64}(undef, d, d)
    haskey(pos, 0) && (states[pos[0]] = copy(rho))
    matvecs = 0
    @inbounds for k in 1:ks[end]
        apply!(out, rho)
        matvecs += 1
        copyto!(rho, out)
        if hermitize
            hermitianize!(rho)
        end
        if renormalize
            tr_now = real(tr(rho))
            tr_now != 0 && (rho ./= tr_now)
        end
        haskey(pos, k) && (states[pos[k]] = copy(rho))
    end
    return states, ks, matvecs
end

# Return one trajectory in the common basis, with work metadata.
# `t_grid` is absolute time from zero. Channel times must satisfy $t = k delta$;
# the Lindbladian integrator advances by consecutive differences.
function _propagate_arm(arm::PropagatorArm, rho_0::AbstractMatrix,
                        t_grid::AbstractVector{<:Real};
                        krylovdim::Integer, tol::Real,
                        hermitize::Bool, renormalize::Bool,
                        commensurate_atol::Real)
    rho_w = _rotate_in(arm, rho_0)
    if arm.kind === :lindbladian
        res = lindblad_action_integrate(
            arm.apply!, rho_w, zero(rho_w), t_grid;
            krylovdim = Int(krylovdim), tol = tol, save_states = true,
        )
        states_w = res.states
        return [_rotate_out(arm, s) for s in states_w], res.total_matvecs, res.all_converged
    else  # :channel
        k_of_t = Vector{Int}(undef, length(t_grid))
        @inbounds for (i, t) in enumerate(t_grid)
            kk = float(t) / arm.delta
            kr = round(Int, kk)
            abs(kk - kr) <= commensurate_atol * max(1.0, abs(kk)) || throw(ArgumentError(
                "channel arm '$(arm.label)': t=$t is not an integer multiple of δ=$(arm.delta) " *
                "(t/δ = $kk). Build t_grid as k_grid .* δ."))
            k_of_t[i] = kr
        end
        states_sorted, ks, mv = _iterate_channel_states(
            arm.apply!, rho_w, k_of_t; hermitize = hermitize, renormalize = renormalize)
        posmap = Dict(k => i for (i, k) in enumerate(ks))
        states_w = [states_sorted[posmap[k]] for k in k_of_t]
        return [_rotate_out(arm, s) for s in states_w], mv, true
    end
end

"""
    propagator_trace_distance(arm_A, arm_B, rho_0, t_grid; kwargs...) -> NamedTuple

Compare two matrix-free propagator trajectories in trace distance.

`rho_0` and returned states use the common basis. `t_grid` is sorted absolute
time starting at zero; channel times must be integer multiples of their `delta`.

# Keywords
- `krylovdim`, `tol`: Lindbladian exponentiation controls.
- `hermitize`, `renormalize`: clean channel states after each step.
- `save_states`: retain both common-basis trajectories.
- `commensurate_atol`: tolerance for zero time and channel-step checks.

# Returns
A named tuple containing the time-resolved and maximum distances, the first
positive-time distance, work metadata, labels, and optional trajectories.
"""
function propagator_trace_distance(
    arm_A::PropagatorArm, arm_B::PropagatorArm,
    rho_0::AbstractMatrix, t_grid::AbstractVector{<:Real};
    krylovdim::Integer = 30, tol::Real = 1e-12,
    hermitize::Bool = true, renormalize::Bool = true,
    save_states::Bool = false, commensurate_atol::Real = 1e-9,
)
    d = size(rho_0, 1)
    size(rho_0, 2) == d || throw(ArgumentError("rho_0 must be square"))
    isempty(t_grid) && throw(ArgumentError("t_grid must be non-empty"))
    issorted(t_grid) || throw(ArgumentError("t_grid must be sorted ascending"))
    # Both arms use absolute time, while the integrator advances relative steps.
    abs(float(t_grid[1])) <= commensurate_atol || throw(ArgumentError(
        "t_grid must start at 0 (got t_grid[1] = $(t_grid[1])); both arms measure " *
        "absolute time from t=0 with the t=0 snapshot equal to rho_0. Prepend 0.0."))

    statesA, mvA, cvA = _propagate_arm(arm_A, rho_0, t_grid;
        krylovdim = krylovdim, tol = tol, hermitize = hermitize,
        renormalize = renormalize, commensurate_atol = commensurate_atol)
    statesB, mvB, cvB = _propagate_arm(arm_B, rho_0, t_grid;
        krylovdim = krylovdim, tol = tol, hermitize = hermitize,
        renormalize = renormalize, commensurate_atol = commensurate_atol)

    T = [trace_distance_nh(statesA[i], statesB[i]) for i in eachindex(t_grid)]
    imax = argmax(T)

    # per-step T₁: trace distance at the smallest strictly-positive time.
    i_first_pos = findfirst(>(0), t_grid)
    per_step = i_first_pos === nothing ? T[1] : T[i_first_pos]

    return (
        t                 = collect(float.(t_grid)),
        trace_distances   = T,
        max_distance      = T[imax],
        argmax_t          = float(t_grid[imax]),
        argmax_index      = imax,
        per_step_distance = per_step,
        matvecs           = (mvA, mvB),
        converged         = (cvA, cvB),
        labels            = (arm_A.label, arm_B.label),
        states            = save_states ? (statesA, statesB) : nothing,
    )
end

# Convert an arm to a generator in its working basis.
# Math: $G = cal(L)$ for Lindbladians and $G = (Phi_delta - I) / delta$ for channels.
# The latter preserves channel eigenvectors while separating eigenvalues near one.
function _arm_generator(arm::PropagatorArm)
    arm.kind === :channel || return arm.apply!
    return let f = arm.apply!, δ = arm.delta
        (out::AbstractMatrix, x::AbstractMatrix) -> (f(out, x); @. out = (out - x) / δ; out)
    end
end

# Extract the leading generator eigenmode and rotate it to the common basis.
function _arm_fixed_point(arm::PropagatorArm, rho_seed::AbstractMatrix;
                          krylovdim::Integer, tol::Real)
    seed_w = _rotate_in(arm, rho_seed)
    dim = size(seed_w, 1)
    decomp = _krylov_spectral_decomposition(
        _arm_generator(arm), Matrix{ComplexF64}(seed_w), dim;
        krylovdim = Int(krylovdim), tol = tol, sort_mode = :lindbladian)
    return _rotate_out(arm, decomp.rho_inf), decomp.matvec_count
end

"""
    propagator_fixed_point_distance(arm_A, arm_B; kwargs...) -> NamedTuple

Compute the trace distance between two common-basis fixed points.

Channel fixed points are extracted from `(Phi_delta - I) / delta`; this avoids
the eigenvalue cluster near one but amplifies matvec noise as `delta` decreases.

# Keywords
- `rho_seed`: common-basis Arnoldi seed; defaults to `I/d`.
- `d`: dimension when neither a seed nor `arm_A.basis` supplies it.
- `krylovdim`, `tol`: eigensolver controls.

# Returns
A named tuple with `distance`, `sigma_A`, `sigma_B`, `matvecs`, and `labels`.
"""
function propagator_fixed_point_distance(
    arm_A::PropagatorArm, arm_B::PropagatorArm;
    rho_seed::Union{Nothing, AbstractMatrix} = nothing,
    d::Union{Nothing, Integer} = nothing,
    krylovdim::Integer = 40, tol::Real = 1e-10,
)
    dim = if rho_seed !== nothing
        size(rho_seed, 1)
    elseif arm_A.basis !== nothing
        size(arm_A.basis, 1)
    elseif d !== nothing
        Int(d)
    else
        throw(ArgumentError("provide rho_seed, or a basis on arm_A, or d to fix the dimension"))
    end
    seed = rho_seed === nothing ? Matrix{ComplexF64}(I(dim) / dim) : Matrix{ComplexF64}(rho_seed)

    σA, mvA = _arm_fixed_point(arm_A, seed; krylovdim = krylovdim, tol = tol)
    σB, mvB = _arm_fixed_point(arm_B, seed; krylovdim = krylovdim, tol = tol)
    return (
        distance = trace_distance_nh(σA, σB),
        sigma_A  = σA,
        sigma_B  = σB,
        matvecs  = (mvA, mvB),
        labels   = (arm_A.label, arm_B.label),
    )
end

# Fixed-point diagnostics complement slow-rate comparisons.
# A Gibbs seed improves Krylov overlap for low-temperature channel fixed points;
# stationarity and positivity checks certify the extracted state.

# Reference-free stationarity certificate in the common basis.
# Math: channel $r = 1/2 norm(Phi_delta(sigma) - sigma, 1)$;
# Lindbladian $r = 1/2 norm(cal(L)(sigma), 1)$.
function _arm_stationarity_residual(arm::PropagatorArm, sigma_common::AbstractMatrix)
    x = Matrix{ComplexF64}(_rotate_in(arm, sigma_common))
    out = Matrix{ComplexF64}(undef, size(x, 1), size(x, 2))
    arm.apply!(out, x)
    Y = Matrix{ComplexF64}(_rotate_out(arm, out))
    return arm.kind === :channel ?
        trace_distance_nh(Y, Matrix{ComplexF64}(sigma_common)) :
        trace_norm_nh(Y) / 2
end

# Dense fixed point from the eigenvalue nearest one (channel) or zero (generator).
# Returns the common-basis state, eigenvalue, anti-Hermitian fraction, and the
# separation from the second-nearest eigenvalue; a small separation signals
# a non-unique steady space.
function _dense_arm_fixed_point(arm::PropagatorArm, d::Integer)
    S = build_dense_superoperator(arm.apply!, Int(d))
    F = eigen(S)
    target = arm.kind === :channel ? one(ComplexF64) : zero(ComplexF64)
    order = sortperm(abs.(F.values .- target))
    i1 = order[1]
    eval1 = F.values[i1]
    steady_gap = length(order) >= 2 ? abs(F.values[order[2]] - target) : Inf
    R = reshape(Vector{ComplexF64}(F.vectors[:, i1]), Int(d), Int(d))
    _phase_align_trace!(R)
    nR = norm(R)
    antiherm = nR == 0 ? 0.0 : norm((R .- R') ./ 2) / nR
    hermitianize!(R)
    trR = real(tr(R))
    isfinite(trR) && trR > 0 || throw(ArgumentError(
        "phase-aligned dense fixed point must have positive finite trace."))
    R ./= trR
    return Matrix{ComplexF64}(_rotate_out(arm, R)), eval1, antiherm, steady_gap
end

# Return the smallest eigenvalue of the Hermitian part and warn if positivity fails.
function _fixed_point_min_eigval(sigma::AbstractMatrix, label::AbstractString;
                                 psd_tol::Real = 1e-10)
    λmin = minimum(real, eigvals(Hermitian(Matrix{ComplexF64}((sigma .+ sigma') ./ 2))))
    λmin < -psd_tol && @warn "arm_fixed_point: extracted fixed point ($label) is not " *
        "PSD — min eigenvalue $(round(λmin, sigdigits = 3)) < -psd_tol = " *
        "$(-float(psd_tol)). A steady state must be a valid density matrix; suspect a " *
        "non-CP parameter choice or a corrupted extraction, not float noise." maxlog = 1
    return λmin
end

"""
    arm_fixed_point(arm; seed=nothing, method=:auto, krylovdim=120, tol=1e-10,
                    residual_gate=1e-7, dense_max_dim=4096, psd_tol=1e-10) -> NamedTuple

Extract a common-basis fixed point with stationarity and positivity certificates.

`method=:krylov` uses the generator form; `:dense` materialises the exact
complex-linear superoperator; `:auto` falls back to dense when the Krylov
residual exceeds `residual_gate` and the dimension permits it. Use a Gibbs-like
seed for low-temperature channels, but not an exact Lindbladian fixed point.

# Keywords
- `seed`: common-basis Krylov seed; `nothing` uses `I/d`.
- `krylovdim`, `tol`: Krylov controls.
- `dense_max_dim`: largest superoperator dimension `d^2` allowed for dense
  fallback; the dense allocation has size `d^2 x d^2`.
- `degeneracy_tol`, `psd_tol`: warning thresholds for uniqueness and positivity.

# Returns
A named tuple with the fixed point, stationarity residual, spectral and
positivity certificates, method, matvec count, and convergence flag. A small
`steady_gap` means the returned state may be one member of a degenerate space.
"""
function arm_fixed_point(arm::PropagatorArm;
        seed::Union{Nothing, AbstractMatrix} = nothing,
        method::Symbol = :auto, krylovdim::Integer = 120, tol::Real = 1e-10,
        residual_gate::Real = 1e-7, dense_max_dim::Integer = 4096,
        degeneracy_tol::Real = 1e-9, psd_tol::Real = 1e-10)
    method in (:auto, :krylov, :dense) ||
        throw(ArgumentError("method must be :auto, :krylov or :dense (got :$method)"))
    dense_max_dim >= 1 || throw(ArgumentError("dense_max_dim must be >= 1."))
    d = arm.basis !== nothing ? size(arm.basis, 1) :
        (seed !== nothing ? size(seed, 1) :
         throw(ArgumentError("provide a seed, or an arm with a basis, to fix the dimension")))
    d > 0 || throw(ArgumentError("fixed-point dimension must be > 0."))
    superop_dim = try
        Base.checked_mul(d, d)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("fixed-point superoperator dimension d^2 overflows Int for d=$d."))
    end
    seed_m = seed === nothing ? Matrix{ComplexF64}(I(d) / d) : Matrix{ComplexF64}(seed)

    dense_result() = let (σ, μ, ah, sgap) = _dense_arm_fixed_point(arm, d)
        sgap < degeneracy_tol && @warn "arm_fixed_point: near-degenerate steady space " *
            "($(arm.label)): 2nd-closest eigenvalue is $(round(sgap, sigdigits = 3)) from the " *
            "fixed-point target (< degeneracy_tol = $degeneracy_tol). The returned σ is one " *
            "arbitrary member of a degenerate steady space — the fixed point is not unique." maxlog = 1
        (sigma = σ, residual = _arm_stationarity_residual(arm, σ), eigval = μ,
         antiherm_frac = ah, steady_gap = sgap,
         min_eigval = _fixed_point_min_eigval(σ, arm.label; psd_tol = psd_tol),
         method = :dense, matvecs = d * d, converged = true)
    end
    method === :dense && return dense_result()

    σk = nothing; mvk = 0; resk = NaN; broke = false
    try
        σk, mvk = _arm_fixed_point(arm, seed_m; krylovdim = Int(krylovdim), tol = tol)
        resk = _arm_stationarity_residual(arm, σk)
    catch err
        method === :krylov && rethrow(err)
        broke = true
        @warn "arm_fixed_point: Krylov broke down ($(arm.label)); escalating to dense" err maxlog = 1
    end

    # min_eigval of the (Krylov) σ — NaN when Krylov broke down (σk undefined).
    mineig_k() = broke ? NaN : _fixed_point_min_eigval(σk, arm.label; psd_tol = psd_tol)

    method === :krylov && return (sigma = σk, residual = resk, eigval = NaN,
        antiherm_frac = NaN, steady_gap = NaN, min_eigval = mineig_k(),
        method = :krylov, matvecs = mvk, converged = !broke)

    # :auto — accept a certified Krylov result, else escalate to dense when feasible.
    (!broke && isfinite(resk) && resk <= residual_gate) && return (sigma = σk,
        residual = resk, eigval = NaN, antiherm_frac = NaN, steady_gap = NaN,
        min_eigval = mineig_k(), method = :krylov, matvecs = mvk, converged = true)
    superop_dim <= dense_max_dim && return dense_result()
    @warn "arm_fixed_point: Krylov residual $(resk) > gate $(residual_gate), but " *
          "superoperator dimension d^2=$superop_dim exceeds dense_max_dim=$(dense_max_dim) — " *
          "returning UNGATED Krylov fixed point (approximate)." maxlog = 1
    return (sigma = σk, residual = resk, eigval = NaN, antiherm_frac = NaN,
            steady_gap = NaN, min_eigval = mineig_k(), method = :krylov_ungated,
            matvecs = mvk, converged = !broke)
end

"""
    fixed_point_gibbs_distance(arm, gibbs; seed=:auto, kwargs...) -> NamedTuple

Compute the trace distance between an arm's fixed point and an explicit Gibbs
state, both in the common basis.

`seed=:auto` uses `gibbs` for channels and `I/d` for Lindbladians. A matrix or
`nothing` overrides this choice; remaining keywords are passed to
[`arm_fixed_point`](@ref).

# Returns
A named tuple with `distance`, the fixed point and its certificates, work
metadata, and the arm label.
"""
function fixed_point_gibbs_distance(arm::PropagatorArm, gibbs::AbstractMatrix;
        seed::Union{Symbol, Nothing, AbstractMatrix} = :auto, kwargs...)
    size(gibbs, 1) == size(gibbs, 2) || throw(ArgumentError("gibbs must be square"))
    g = Matrix{ComplexF64}(gibbs)
    chosen = seed === :auto ? (arm.kind === :channel ? g : nothing) :
             (seed === nothing ? nothing : Matrix{ComplexF64}(seed))
    fp = arm_fixed_point(arm; seed = chosen, kwargs...)
    return (distance = trace_distance_nh(fp.sigma, g), sigma = fp.sigma,
            residual = fp.residual, eigval = fp.eigval, antiherm_frac = fp.antiherm_frac,
            steady_gap = fp.steady_gap, min_eigval = fp.min_eigval, method = fp.method,
            matvecs = fp.matvecs, converged = fp.converged, label = arm.label)
end

# Choose the unit phase that makes an eigenmode maximally Hermitian.
# Math: $alpha^2 = angle.l dot(R, R^dagger) / norm(R)^2$.
function _hermitizing_phase(R::AbstractMatrix)
    z = dot(vec(R), vec(R'))                 # ⟨R, R†⟩_HS
    n2 = real(dot(vec(R), vec(R)))           # ‖R‖²_HS
    (abs(z) < 1e-300 || n2 == 0) && return one(ComplexF64)
    α = sqrt(z / n2)
    α /= abs(α)                              # enforce |α| = 1
    best = α
    bestval = -1.0
    for c in (α, -α, im * α, -im * α)
        v = norm((c .* R .+ (c .* R)') ./ 2)
        if v > bestval
            bestval = v
            best = c
        end
    end
    return best
end

"""
    slow_subspace_generator_distance(ref_arm, test_arm, rho_seed; kwargs...) -> NamedTuple

Measure generator mismatch on a reference arm's slow spectral subspace.

The reference supplies biorthonormal modes `R_k, L_k`; each arm is applied in
its own working basis and rotated through the common basis. For channels,
`G = (Phi_delta - I) / delta`. The projected mismatch has entries
`M_jk = inner(L_j, (G_test - G_ref)(R_k))`, and `eps_slow = opnorm(M)`.

`num_slow_modes=1` is the validated first-order gap-shift diagnostic. Larger
blocks depend on the chosen biorthogonal basis and require a well-separated
retained subspace. `max_antiherm_frac` is descriptive because channel actions
are complex-linear; it is not a validity gate.

# Arguments
- `ref_arm`: source of the slow eigenbasis, normally a KMS-normal Lindbladian.
- `test_arm`: generator being compared; it may use a different working basis.
- `rho_seed`: common-basis Arnoldi seed that must overlap the slow modes. Avoid
  `I/d` when symmetry can hide traceless modes.

# Keywords
- `num_slow_modes`: number of non-stationary modes; `krylovdim > K + 1`.
- `include_stationary`: prepend the steady mode to the projected block.
- `krylovdim`, `tol`: reference Arnoldi controls.
- `antiherm_tol`: retained compatibility keyword; it does not gate results.
- `pt_spacing_frac`: warn for `K=1` when the perturbation is too large relative
  to the gap mode's nearest-neighbour spacing.

# Returns
A named tuple containing the absolute and relative mismatch norms, spectral
spacing diagnostics, projected block, reference modes and gap, convergence
certificates, matvec counts, and labels.
"""
function slow_subspace_generator_distance(
    ref_arm::PropagatorArm, test_arm::PropagatorArm,
    rho_seed::AbstractMatrix;
    num_slow_modes::Integer = 1,
    include_stationary::Bool = false,
    krylovdim::Integer = 60,
    tol::Real = 1e-10,
    antiherm_tol::Real = 1e-6,
    pt_spacing_frac::Real = 0.5,
)
    d = size(rho_seed, 1)
    size(rho_seed, 2) == d || throw(ArgumentError("rho_seed must be square"))
    K = Int(num_slow_modes)
    K >= 1 || throw(ArgumentError("num_slow_modes must be ≥ 1 (got $K)"))
    Int(krylovdim) > K + 1 || throw(ArgumentError(
        "krylovdim ($krylovdim) must exceed num_slow_modes + 1 ($(K + 1)) to resolve " *
        "the steady + $K slow modes"))

    # 1. Reference slow eigenpairs: a single forward Arnoldi on the reference
    #    GENERATOR, seeded by rho_seed rotated into the reference working basis.
    #    `_krylov_spectral_decomposition` returns biorthonormal R_modes / L_modes
    #    sorted with the steady state at index 1, the gap mode at index 2, … .
    gen_ref = _arm_generator(ref_arm)
    seed_ref_work = Matrix{ComplexF64}(_rotate_in(ref_arm, rho_seed))
    decomp = _krylov_spectral_decomposition(
        gen_ref, seed_ref_work, d;
        krylovdim = Int(krylovdim), tol = tol, sort_mode = :lindbladian)

    m = length(decomp.R_modes)
    m >= K + 1 || throw(ArgumentError(
        "reference Krylov decomposition returned only $m modes; need ≥ $(K + 1) " *
        "(steady + K=$K slow). Increase krylovdim (got $krylovdim)."))

    # Decomposition idx 1 = steady (design k=0), idx 2 = gap (design k=1). Retain
    # the K slowest NON-stationary modes (idx 2:K+1), optionally prepend the steady.
    idxs = include_stationary ? collect(1:(K + 1)) : collect(2:(K + 1))
    Ksel = length(idxs)

    # Rotate retained modes ONCE to the COMMON basis (HS biorthonormality is
    # unitarily invariant ⇒ ⟨L_j,R_k⟩ = δ_jk survives the rotation).
    R_common = [Matrix{ComplexF64}(_rotate_out(ref_arm, decomp.R_modes[i])) for i in idxs]
    L_common = [Matrix{ComplexF64}(_rotate_out(ref_arm, decomp.L_modes[i])) for i in idxs]
    Λ = decomp.eigenvalues[idxs]

    # Choose a modal phase that maximises each right mode's Hermitian part. Applying
    # the same unit phase to its left mode preserves biorthogonality and transforms
    # the projected mismatch only by a unitary diagonal gauge. Non-Hermitian modes
    # remain valid because both arms are complex-linear; the fraction is descriptive.
    antiherm_fracs = Vector{Float64}(undef, Ksel)
    for kk in 1:Ksel
        α = _hermitizing_phase(R_common[kk])
        R_common[kk] = α .* R_common[kk]
        L_common[kk] = α .* L_common[kk]
        antiherm_fracs[kk] = norm((R_common[kk] .- R_common[kk]') ./ 2) / norm(R_common[kk])
    end
    max_antiherm_frac = maximum(antiherm_fracs)
    _ = antiherm_tol

    # 2. M_jk = ⟨L_j | (G_test − G_ref) | R_k⟩_HS. Apply each generator to the
    #    common-basis R_k via its OWN arm (rotate in → generate → rotate out), then
    #    contract with L_j. ⟨L_j|G_ref|R_k⟩ = λ_k δ_jk exactly (Arnoldi relation),
    #    captured separately as a convergence diagnostic.
    gen_test = _arm_generator(test_arm)
    M = zeros(ComplexF64, Ksel, Ksel)
    Gref_proj = zeros(ComplexF64, Ksel, Ksel)
    out_t = Matrix{ComplexF64}(undef, d, d)
    out_r = Matrix{ComplexF64}(undef, d, d)
    mv_test = 0
    mv_ref_gen = 0
    for (kk, Rk) in enumerate(R_common)
        gen_test(out_t, Matrix{ComplexF64}(_rotate_in(test_arm, Rk)))
        mv_test += 1
        GtestRk = _rotate_out(test_arm, out_t)
        gen_ref(out_r, Matrix{ComplexF64}(_rotate_in(ref_arm, Rk)))
        mv_ref_gen += 1
        GrefRk = _rotate_out(ref_arm, out_r)
        MRk = GtestRk .- GrefRk
        for (jj, Lj) in enumerate(L_common)
            M[jj, kk] = dot(vec(Lj), vec(MRk))
            Gref_proj[jj, kk] = dot(vec(Lj), vec(GrefRk))
        end
    end

    # 3. Scalars. The gap mode is the slowest non-stationary retained mode: it sits
    #    at retained-index 1 (steady excluded) or 2 (steady prepended).
    eps_slow = opnorm(M, 2)
    gap_idx = include_stationary ? 2 : 1
    gap_ref = abs(real(Λ[gap_idx]))                 # gap RATE = |Re λ₂| (krylov_spectral_gap convention)
    block_norm = maximum(abs.(Λ))                   # ‖ΠLΠ‖₂ = max|λ| (diag in eigenbasis; |λ| not |Re λ|, valid for a non-normal reference too)
    ref_gen_residual = maximum(abs.(Gref_proj .- Diagonal(Λ)))
    # This projection is diagonal only when the reference modes are resolved.
    if ref_gen_residual > sqrt(tol)
        @warn "slow_subspace_generator_distance: reference generator is not diagonal " *
              "on its own slow modes (max|⟨L_j|G_ref|R_k⟩ − λ_k δ_jk| = " *
              "$(round(ref_gen_residual, sigdigits = 3)) > √tol). The reference arm's " *
              "slow modes are not well-resolved — use a KMS-normal reference (the ideal " *
              "Lindbladian), not a channel, and a seed overlapping its slow modes." maxlog = 1
    end

    # First-order perturbation theory requires the mismatch to be small relative
    # to the nearest non-stationary spacing, not merely the gap magnitude.
    λ_neighbor_spacing = length(decomp.eigenvalues) >= 3 ?
        abs(decomp.eigenvalues[2] - decomp.eigenvalues[3]) : NaN
    # Warn only for the K=1 gap-shift diagnostic: there λ₃ IS the gap mode's first
    # excluded neighbour and ε_slow IS the gap shift, so the check is exactly "is the
    # first-order gap shift ≪ the gap-to-neighbour spacing". For K>1 this scalar spacing
    # does not certify the whole retained block; interpretation requires checking the
    # retained/excluded spectral boundary separately.
    if K == 1 && isfinite(λ_neighbor_spacing) && λ_neighbor_spacing > 0 &&
       eps_slow > pt_spacing_frac * λ_neighbor_spacing
        @warn "slow_subspace_generator_distance: perturbation ε_slow = " *
              "$(round(eps_slow, sigdigits = 3)) is not ≪ the gap mode's spacing to its " *
              "nearest neighbor |λ₂−λ₃| = $(round(λ_neighbor_spacing, sigdigits = 3)) " *
              "(ratio $(round(eps_slow / λ_neighbor_spacing, sigdigits = 3)) > " *
              "pt_spacing_frac = $(pt_spacing_frac)). First-order PT for the gap " *
              "eigenvalue needs the perturbation ≪ the neighbor spacing, not just ≪ " *
              "|λ₂|; the gap shift may be unreliable (gap and neighbor can mix)." maxlog = 1
    end

    return (
        eps_slow            = eps_slow,
        eps_slow_rel_gap    = gap_ref > 0 ? eps_slow / gap_ref : NaN,
        eps_slow_rel_block  = block_norm > 0 ? eps_slow / block_norm : NaN,
        lambda_neighbor_spacing = λ_neighbor_spacing,
        eps_slow_rel_neighbor   = (isfinite(λ_neighbor_spacing) && λ_neighbor_spacing > 0) ?
                                  eps_slow / λ_neighbor_spacing : NaN,
        M                   = M,
        M_diagonal          = diag(M),
        eigenvalues_ref     = Λ,
        gap_ref             = gap_ref,
        block_norm_ref      = block_norm,
        num_slow_modes      = K,
        include_stationary  = include_stationary,
        max_antiherm_frac   = max_antiherm_frac,
        ref_gen_residual    = ref_gen_residual,
        matvecs             = (ref_decomp = decomp.matvec_count, test_gen = mv_test, ref_gen = mv_ref_gen),
        converged           = decomp.converged,
        labels              = (ref_arm.label, test_arm.label),
    )
end

"""
    lindbladian_arm(config::Config{Lindbladian}, hamiltonian, jumps;
                    basis=hamiltonian.eigvecs, workspace=nothing,
                    label="ideal L") -> PropagatorArm

Wrap `apply_lindbladian!` as a [`PropagatorArm`](@ref).

`jumps` use the matvec's working basis, while `basis` maps it to the common
basis. `workspace` may reuse precomputation. The coherent term remains enabled.
"""
function lindbladian_arm(config::Config{Lindbladian}, hamiltonian::HamHam,
                         jumps::Vector{JumpOp};
                         basis::Union{Nothing, AbstractMatrix} = hamiltonian.eigvecs,
                         workspace::Union{Nothing, Workspace{KrylovSpectrum}} = nothing,
                         label::AbstractString = "ideal L")
    validate_config!(config, hamiltonian)
    ws = _reuse_or_build_krylov_workspace(
        workspace, config, hamiltonian, jumps, nothing, size(hamiltonian.data, 1);
        caller="lindbladian_arm")
    apply! = let ws = ws, config = config, ham = hamiltonian
        (out::AbstractMatrix, x::AbstractMatrix) -> begin
            apply_lindbladian!(ws, Matrix{ComplexF64}(x), config, ham)
            copyto!(out, ws.scratch.rho_out)
            return out
        end
    end
    return PropagatorArm(apply!; kind = :lindbladian, basis = basis, label = label)
end

"""
    channel_arm(config::Config{Thermalize}, hamiltonian, jumps, trotter;
                basis=trotter.eigvecs, workspace=nothing,
                label="implemented Φ_δ") -> PropagatorArm

Wrap the faithful deterministic channel as a [`PropagatorArm`](@ref).

For `TrotterDomain`, `jumps` and `basis` use `trotter.eigvecs`. `delta` comes
from `config`; `hermitize=false` keeps the action complex-linear.
"""
function channel_arm(config::Config{Thermalize}, hamiltonian::HamHam,
                     jumps::Vector{JumpOp},
                     trotter::Union{Nothing, AbstractTrotter} = nothing;
                     basis::Union{Nothing, AbstractMatrix} =
                         trotter === nothing ? nothing : trotter.eigvecs,
                     workspace::Union{Nothing, Workspace{KrylovSpectrum}} = nothing,
                     label::AbstractString = "implemented Φ_δ")
    config.jump_selection === :sweep || throw(ArgumentError(
        "channel_arm requires config.jump_selection = :sweep (got :$(config.jump_selection))"))
    config.delta === nothing && throw(ArgumentError("channel_arm requires config.delta to be set"))
    validate_config!(config, hamiltonian)
    ws = _reuse_or_build_krylov_workspace(
        workspace, config, hamiltonian, jumps, trotter, size(hamiltonian.data, 1);
        caller="channel_arm")
    apply! = let ws = ws, config = config, ham = hamiltonian
        (out::AbstractMatrix, x::AbstractMatrix) -> begin
            apply_delta_channel!(
                ws, Matrix{ComplexF64}(x), config, ham; hermitize=false)
            copyto!(out, ws.scratch.rho_next)
            return out
        end
    end
    return PropagatorArm(apply!; kind = :channel, delta = float(config.delta),
                         basis = basis, label = label)
end

# Matrix-free detailed-balance defect in the Hamiltonian eigenbasis.
# Math: $D(G)(X) = sigma^(-1/4) G(sigma^(1/4) X sigma^(1/4)) sigma^(-1/4)$.
# KMS detailed balance is equivalent to $A = (D - D^dagger) / 2 = 0$.
# The operator norm of `A` is computed from forward and adjoint actions.

"""
    discriminant_antiherm_norm(gen!, gen_adj!, sigma_quarter, sigma_inv_quarter, d;
                               krylovdim=30, tol=1e-12, max_retries=3,
                               compute_discriminant_norm=false) -> NamedTuple

Compute the matrix-free operator norm of the anti-Hermitian part of a KMS
quantum discriminant.

`gen!` and `gen_adj!` are forward and Hilbert--Schmidt-adjoint actions on
`d`-dimensional operators. `sigma_quarter` and `sigma_inv_quarter` contain the
diagonal Gibbs factors in that working basis. Krylov keywords control the norm
estimate; `compute_discriminant_norm` also evaluates the relative defect.

# Returns
A named tuple with `antiherm_norm`, optional `discriminant_norm`, `relative`,
and the Gibbs similarity-transform `conditioning`.
"""
function discriminant_antiherm_norm(
    gen!::FG, gen_adj!::FA,
    sigma_quarter::AbstractVector{<:Real},
    sigma_inv_quarter::AbstractVector{<:Real},
    d::Integer;
    krylovdim::Integer = 30, tol::Real = 1e-12, max_retries::Integer = 3,
    compute_discriminant_norm::Bool = false,
) where {FG, FA}
    sq    = collect(Float64, sigma_quarter)
    sqinv = collect(Float64, sigma_inv_quarter)
    length(sq) == d || throw(ArgumentError("sigma_quarter length $(length(sq)) ≠ d=$d"))

    bufs_D  = DiscriminantBuffers(d)
    bufs_Dd = DiscriminantBuffers(d)
    bufD  = Matrix{ComplexF64}(undef, d, d)
    bufDd = Matrix{ComplexF64}(undef, d, d)

    # Math: $D(X) = sigma^(-1/4) G(sigma^(1/4) X sigma^(1/4)) sigma^(-1/4)$.
    D!(out, X) = apply_discriminant!(out, Matrix{ComplexF64}(X), gen!, sq, sqinv, bufs_D)
    # The adjoint discriminant swaps the two Gibbs scale vectors.
    Dadj!(out, X) = apply_discriminant!(out, Matrix{ComplexF64}(X), gen_adj!, sqinv, sq, bufs_Dd)

    # Math: $A(X) = (D(X) - D^dagger(X)) / 2$ and $A^dagger = -A$.
    A!(out, X) = begin
        D!(bufD, X); Dadj!(bufDd, X)
        @inbounds @. out = (bufD - bufDd) / 2
        out
    end
    negA!(out, X) = (A!(out, X); @inbounds @. out = -out; out)

    antiherm = hs_operator_norm_krylov(A!, negA!, d;
                                       tol = tol, krylovdim = Int(krylovdim),
                                       max_retries = Int(max_retries))

    discr_norm = NaN
    if compute_discriminant_norm
        discr_norm = hs_operator_norm_krylov(D!, Dadj!, d;
                                             tol = tol, krylovdim = Int(krylovdim),
                                             max_retries = Int(max_retries))
    end

    # Math: $kappa_sigma = (max(sigma^(1/4)) / min(sigma^(1/4)))^2$.
    κσ = (maximum(sq) / minimum(sq))^2
    return (
        antiherm_norm     = antiherm,
        discriminant_norm = discr_norm,
        relative          = compute_discriminant_norm ? antiherm / discr_norm : NaN,
        conditioning      = κσ,
    )
end

# σ^{±1/4} diagonal (Hamiltonian eigenbasis) from a Gibbs spectrum exp(-β_alg E)/Z.
function _gibbs_quarter_powers(hamiltonian::HamHam, beta_alg::Real)
    T = eltype(hamiltonian.eigvals)
    g = _gibbs_weights(hamiltonian.eigvals, beta_alg)
    powers = gibbs_fractional_powers(
        Hermitian(Matrix(Diagonal(Complex{T}.(g)))))
    return powers.sigma_quarter, powers.sigma_inv_quarter
end

"""
    channel_discriminant_antiherm_norm(config::Config{Thermalize}, hamiltonian, jumps,
                                       trotter=nothing; krylovdim=30, tol=1e-12,
                                       max_retries=3, workspace=nothing,
                                       compute_discriminant_norm=false) -> NamedTuple

Compute the detailed-balance defect of the effective channel generator.

The channel and its adjoint use `hermitize=false` to remain complex-linear.
The discriminant is evaluated in the Hamiltonian eigenbasis; Trotter-domain
actions are rotated accordingly. `jumps` use the channel working basis and
`workspace` may reuse precomputation. Returns the fields from
[`discriminant_antiherm_norm`](@ref).
"""
function channel_discriminant_antiherm_norm(
    config::Config{Thermalize}, hamiltonian::HamHam, jumps::Vector{JumpOp},
    trotter::Union{Nothing, AbstractTrotter} = nothing;
    krylovdim::Integer = 30, tol::Real = 1e-12, max_retries::Integer = 3,
    workspace::Union{Nothing, Workspace{KrylovSpectrum}} = nothing,
    compute_discriminant_norm::Bool = false,
)
    config.delta === nothing && throw(ArgumentError(
        "channel_discriminant_antiherm_norm requires config.delta to be set"))
    config.jump_selection === :sweep || throw(ArgumentError(
        "channel_discriminant_antiherm_norm requires config.jump_selection = :sweep " *
        "(got :$(config.jump_selection))"))
    validate_config!(config, hamiltonian)
    ws = _reuse_or_build_krylov_workspace(
        workspace, config, hamiltonian, jumps, trotter, size(hamiltonian.data, 1);
        caller="channel_discriminant_antiherm_norm")
    d = size(hamiltonian.data, 1)
    δ = float(config.delta)
    sq, sqinv = _gibbs_quarter_powers(hamiltonian, config.beta)

    # Rotation ham eigenbasis → channel working (trotter) basis: X_trott = W X_ham W'.
    Wrot = trotter === nothing ?
        Matrix{ComplexF64}(I, d, d) :
        Matrix{ComplexF64}(trotter.eigvecs)' * Matrix{ComplexF64}(hamiltonian.eigvecs)
    Xtr = Matrix{ComplexF64}(undef, d, d)
    rot = Matrix{ComplexF64}(undef, d, d)

    chan_gen!(out, Xham) = begin
        Xh = Matrix{ComplexF64}(Xham)
        mul!(rot, Wrot, Xh); mul!(Xtr, rot, Wrot')              # ham → trotter
        apply_delta_channel!(ws, Xtr, config, hamiltonian; hermitize = false)
        mul!(rot, Wrot', ws.scratch.rho_next); mul!(out, rot, Wrot)  # trotter → ham
        @inbounds @. out = (out - Xh) / δ                       # G_eff = (Φ−I)/δ
        out
    end
    chan_gen_adj!(out, Xham) = begin
        Xh = Matrix{ComplexF64}(Xham)
        mul!(rot, Wrot, Xh); mul!(Xtr, rot, Wrot')
        apply_adjoint_delta_channel!(ws, Xtr, config, hamiltonian; hermitize = false)
        mul!(rot, Wrot', ws.scratch.rho_next); mul!(out, rot, Wrot)
        @inbounds @. out = (out - Xh) / δ                       # G_eff† = (Φ†−I)/δ
        out
    end

    return discriminant_antiherm_norm(chan_gen!, chan_gen_adj!, sq, sqinv, d;
        krylovdim = krylovdim, tol = tol, max_retries = max_retries,
        compute_discriminant_norm = compute_discriminant_norm)
end

"""
    lindbladian_discriminant_antiherm_norm(config::Config{Lindbladian}, hamiltonian,
                                           jumps; krylovdim=30, tol=1e-12,
                                           max_retries=3, workspace=nothing,
                                           compute_discriminant_norm=false) -> NamedTuple

Compute the detailed-balance defect of a Lindbladian generator.

For a KMS detailed-balance generator this measures the numerical construction
floor. `jumps` use the Lindbladian working basis; `workspace` may reuse
precomputation. Returns the fields from [`discriminant_antiherm_norm`](@ref).
"""
function lindbladian_discriminant_antiherm_norm(
    config::Config{Lindbladian}, hamiltonian::HamHam, jumps::Vector{JumpOp};
    krylovdim::Integer = 30, tol::Real = 1e-12, max_retries::Integer = 3,
    workspace::Union{Nothing, Workspace{KrylovSpectrum}} = nothing,
    compute_discriminant_norm::Bool = false,
)
    validate_config!(config, hamiltonian)
    ws = _reuse_or_build_krylov_workspace(
        workspace, config, hamiltonian, jumps, nothing, size(hamiltonian.data, 1);
        caller="lindbladian_discriminant_antiherm_norm")
    d = size(hamiltonian.data, 1)
    sq, sqinv = _gibbs_quarter_powers(hamiltonian, config.beta)

    lind_gen!(out, X) = (apply_lindbladian!(ws, Matrix{ComplexF64}(X), config, hamiltonian);
                         copyto!(out, ws.scratch.rho_out); out)
    lind_gen_adj!(out, X) = (apply_adjoint_lindbladian!(ws, Matrix{ComplexF64}(X), config, hamiltonian);
                             copyto!(out, ws.scratch.rho_out); out)

    return discriminant_antiherm_norm(lind_gen!, lind_gen_adj!, sq, sqinv, d;
        krylovdim = krylovdim, tol = tol, max_retries = max_retries,
        compute_discriminant_norm = compute_discriminant_norm)
end
