"""
    oft_domain_prefactor(::EnergyDomain, w0, sigma)
    oft_domain_prefactor(::TimeDomain, w0, sigma, t0)
    oft_domain_prefactor(::TrotterDomain, w0, sigma, t0)

Return the domain-dependent scalar prefactor for OFT rates.

`BohrDomain` uses `gamma_norm_factor` directly and has no method here.
"""
oft_domain_prefactor(::EnergyDomain, w0::Real, sigma::Real) = w0 / (sigma * sqrt(2 * pi))
oft_domain_prefactor(::TimeDomain, w0::Real, sigma::Real, t0::Real) = w0 * t0^2 * (sigma * sqrt(2 / pi)) / (2 * pi)
oft_domain_prefactor(::TrotterDomain, w0::Real, sigma::Real, t0::Real) = w0 * t0^2 * (sigma * sqrt(2 / pi)) / (2 * pi)

function _precompute_labels(config::Config{<:Any, D}) where {D<:Union{BohrDomain, EnergyDomain}}
    # Energy-domain dissipation has no time grid.
    energy_labels = _create_energy_labels(register_r_D(config), register_w0_D(config))
    truncated_energy_labels = _truncate_energy_labels(energy_labels, config)
    return (truncated_energy_labels,)  # Energy labels
end

function _precompute_labels(config::Config{<:Any, D}) where {D<:Union{TimeDomain, TrotterDomain}}
    # Coherent labels are built per term from their own registers.
    # in `_precompute_data` from the `b_minus / b_plus` registers.
    r_D = register_r_D(config)
    w0_D = register_w0_D(config)
    t0_D = register_t0_D(config)
    energy_labels = _create_energy_labels(r_D, w0_D)
    truncated_energy_labels = _truncate_energy_labels(energy_labels, config)
    time_labels = energy_labels .* (t0_D / w0_D)
    return (truncated_energy_labels, time_labels) # Energy and time labels (dissipative grid)
end

function _precompute_data(
    config::Config{Lindbladian, BohrDomain},
    ham_or_trott::Union{HamHam, AbstractTrotter}
)

    alpha = _pick_alpha(config)
    # Grid-independent normalisation: `pick_gamma_sup(config)` is the
    # closed-form continuum sup of γ — 1.0 for every standard family.
    # Replaces the prior `1.0 / maximum(transition.(energy_labels))`
    # which sampled the sup on a discrete `(r_D, w0_D)`-dependent grid.
    gamma_norm_factor = 1.0 / pick_gamma_sup(config)
    return (
        alpha = alpha,
        gamma_norm_factor = gamma_norm_factor
    )
end

function _precompute_data(
    config::Config{Thermalize, BohrDomain},
    hamiltonian::HamHam
)
    alpha = _pick_alpha(config)

    # Use the continuum rate supremum, independent of the sampled grid.
    gamma_norm_factor = 1.0 / pick_gamma_sup(config)

    # Cache the Bohr buckets as plain Int index pairs to avoid CartesianIndex overhead
    # and avoid rebuilding any per-frequency index lists inside jump_contribution!.
    bohr_keys = collect(keys(hamiltonian.bohr_dict))
    bohr_is = Vector{Vector{Int}}(undef, length(bohr_keys))
    bohr_js = Vector{Vector{Int}}(undef, length(bohr_keys))
    @inbounds for (k, nu) in pairs(bohr_keys)
        idxs = hamiltonian.bohr_dict[nu]
        is = Vector{Int}(undef, length(idxs))
        js = Vector{Int}(undef, length(idxs))
        @inbounds for t in eachindex(idxs)
            is[t] = idxs[t][1]
            js[t] = idxs[t][2]
        end
        bohr_is[k] = is
        bohr_js[k] = js
    end

    return (
        alpha = alpha,
        gamma_norm_factor = gamma_norm_factor,
        bohr_keys = bohr_keys,
        bohr_is = bohr_is,
        bohr_js = bohr_js,
    )

end

function _precompute_data(
    config::Config{<:Any, EnergyDomain},
    ham_or_trott::Union{HamHam, AbstractTrotter}
)
    energy_labels, = _precompute_labels(config)
    transition = pick_transition(config)
    # Use the continuum rate supremum, independent of the sampled grid.
    gamma_norm_factor = 1.0 / pick_gamma_sup(config)
    # EnergyDomain dissipator only consults `w0_D` (no time grid).
    dp = oft_domain_prefactor(config.domain, register_w0_D(config), config.sigma)

    return (
        transition = transition,
        gamma_norm_factor = gamma_norm_factor,
        energy_labels = energy_labels,
        oft_domain_prefactor = dp,
    )
end

"""
    _precompute_data(config::Config{Lindbladian, BohrDomain, DLL}, hamiltonian::HamHam)

Resolve the DLL filter for an exact Bohr-domain construction.
"""
function _precompute_data(
    config::Config{Lindbladian, BohrDomain, DLL},
    hamiltonian::HamHam,
)
    filter = _prepare_dll_bohr_filter(_resolve_filter(config), hamiltonian.eigvals;
                                      beta=config.beta)
    return (; filter)
end

"""
    _precompute_data(config::Config{Lindbladian, TimeDomain, DLL}, hamiltonian::HamHam)

Precompute the DLL time grid and its zero-frequency NUFFT slice.

The filter supplies the KMS weighting, so this path has no transition rate or
gamma normalization. FINUFFT uses deterministic single-threaded precision
`1e-12`; the underlying trapezoidal quadrature is unchanged.
"""
function _precompute_data(
    config::Config{Lindbladian, TimeDomain, DLL},
    hamiltonian::HamHam{T},
) where {T<:AbstractFloat}
    filter = _resolve_filter(config)
    # DLL uses one time grid for dissipative and coherent terms; `w0_D` is unused.
    r_D = register_r_D(config)
    t0_D = register_t0_D(config)
    N = 2^r_D
    raw_time_labels = collect((-N÷2):(N÷2 - 1)) .* t0_D
    oft_time_labels = filter isa PreparedFilterTransform ? raw_time_labels :
        _truncate_time_labels_for_oft(raw_time_labels, config.sigma; filter=filter)
    if filter isa PreparedFilterTransform
        _prepare_dll_bohr_filter(filter,hamiltonian.eigvals;beta=config.beta)
    end

    # Single-slice NUFFT at ω = 0 per channel; replaces the per-jump explicit
    # `cis()` triple loop in `dll_lindblad_op_time` with a single FINUFFT eval.
    # For single-channel filters this is a length-1 list — the consumer loop
    # in `_jump_contribution!` (DLL TimeDomain) iterates uniformly.
    sub_filters = _filter_channels_for_dll_oft(filter)
    oft_nufft_at_zero_list = Matrix{Complex{T}}[]
    for sub in sub_filters
        if sub isa PreparedFilterTransform
            values=transform_values(sub,oft_time_labels;window_refinements=0)
            values.status==:unresolved && throw(ArgumentError("Dissipative Fourier quadrature budget exhausted."))
            prefactors=fourier_sum(oft_time_labels,values.values,vec(hamiltonian.bohr_freqs);
                backend=sub.controls.coherent.backend)
            push!(oft_nufft_at_zero_list,reshape(Complex{T}.(prefactors),size(hamiltonian.bohr_freqs)))
        else
            nufft = _prepare_oft_nufft_prefactors(
                hamiltonian.bohr_freqs, oft_time_labels, T[zero(T)], sub; eps=1e-12,
            )
            push!(oft_nufft_at_zero_list, Matrix(@view nufft.data[:, :, 1]))
        end
    end

    return (
        filter = filter,
        time_labels = oft_time_labels,
        t0 = t0_D,
        oft_nufft_at_zero_list = oft_nufft_at_zero_list,
    )
end

# Helper: enumerate the per-channel filters used to build the DLL TimeDomain
# OFT prefactor stack. For single-channel DLL filters the list has length 1
# (the filter itself); the `DLLMultiChannelFilter` overload lives in
# `src/dll_multichannel.jl`.
@inline _filter_channels_for_dll_oft(filter::AbstractFilter) = (filter,)

function _precompute_data(
    config::Config{<:Any, D},
    ham_or_trott::Union{HamHam, AbstractTrotter}
) where {D<:Union{TimeDomain, TrotterDomain}}
    energy_labels, time_labels = _precompute_labels(config)
    oft_time_labels = _truncate_time_labels_for_oft(time_labels, config.sigma; filter=_resolve_filter(config))

    transition = pick_transition(config)
    # Use the continuum rate supremum, independent of the sampled grid.
    gamma_norm_factor = 1.0 / pick_gamma_sup(config)

    # The two coherent kernels use independent outer and inner registers.
    b_minus, b_plus = if with_coherent(config.construction)
        time_labels_b_minus = _create_energy_labels(register_r_b_minus(config),
            register_w0_b_minus(config)) .* (register_t0_b_minus(config) / register_w0_b_minus(config))
        time_labels_b_plus  = _create_energy_labels(register_r_b_plus(config),
            register_w0_b_plus(config))  .* (register_t0_b_plus(config)  / register_w0_b_plus(config))
        _b_minus = _compute_truncated_func(_compute_b_minus, time_labels_b_minus, config.beta, config.sigma)
        chosen_b_plus, b_plus_args = _select_b_plus_calculator(config)
        _b_plus = _compute_truncated_func(chosen_b_plus, time_labels_b_plus, b_plus_args...)
        (_b_minus, _b_plus)
    else
        (nothing, nothing)
    end

    # Single-threaded FINUFFT preserves byte-deterministic construction.
    oft_nufft_prefactors = _prepare_oft_nufft_prefactors(
        ham_or_trott.bohr_freqs,
        oft_time_labels,
        energy_labels,
        _resolve_filter(config);
        eps=1e-12,
        nthreads=1,
    )

    dp = oft_domain_prefactor(config.domain, register_w0_D(config), config.sigma, register_t0_D(config))

    return (
        transition = transition,
        gamma_norm_factor = gamma_norm_factor,
        energy_labels = energy_labels,
        oft_nufft_prefactors = oft_nufft_prefactors,
        b_minus = b_minus,
        b_plus = b_plus,
        oft_domain_prefactor = dp,
    )
end

function _select_b_plus_calculator(config::Config{<:Any, <:Any, KMS})
    if !config.with_linear_combination
        # Gaussian
        return (_compute_b_plus, (config.beta, config.gaussian_parameters[1], config.gaussian_parameters[2]))
    else
        if config.a != 0.0
            # a-regularized smooth (covers Metro a>0,s=0 and Glauberish a>0,s>0)
            return (_compute_b_plus_smooth, (config.beta, config.sigma, config.a, config.s))
        else
            # eta-regularized smooth Metro (covers Chen plain s=0 and thesis-main s>0)
            s_val = something(config.s, 0.0)
            return (_compute_b_plus_metro, (config.beta, config.sigma, config.eta, s_val))
        end
    end
end

"""
    _build_cptp_channel(R, delta; atol=nothing, rtol=nothing) -> (; K0, U_residual, alpha)

Construct the no-event and residual Kraus matrices from a rate operator.

# Arguments
- `R`: Hermitian rate operator.
- `delta`: Channel step size.
- `atol`, `rtol`: Optional roundoff tolerances for the contraction check.

# Returns
`(; K0, U_residual, alpha)`. The weak-measurement block encoding requires
`0 <= R <= I`; violations beyond the stated roundoff tolerance are rejected.
"""
function _build_cptp_channel(
    R::Matrix{T},
    delta::Real;
    atol::Union{Nothing, Real} = nothing,
    rtol::Union{Nothing, Real} = nothing,
) where {T<:Complex}
    dim = LinearAlgebra.checksquare(R)
    dim > 0 || throw(ArgumentError("Channel rate operator R must be nonempty."))
    all(isfinite, R) || throw(ArgumentError(
        "Channel rate operator R must contain only finite values."))

    RT = typeof(real(zero(T)))
    delta_value = RT(delta)
    isfinite(delta_value) && zero(RT) < delta_value <= one(RT) ||
        throw(ArgumentError("delta must be finite and satisfy 0 < delta <= 1 (got $delta)."))

    atol_value, rtol_value = _density_tolerances(R; atol=atol, rtol=rtol)
    isapprox(R, adjoint(R); atol=atol_value, rtol=rtol_value) ||
        throw(ArgumentError("Channel rate operator R must be Hermitian."))

    # A unitary block encoding obeys R = sum_j L_j^dagger L_j <= I.  Check
    # both sides of the spectral interval before permitting bounded roundoff
    # cleanup; otherwise the residual alpha^2 R(I-R) is not positive.
    R_h = Hermitian((R + adjoint(R)) / 2)
    R_eigen = eigen(R_h)
    lower_bounded = _clamp_psd_eigenvalues(
        R_eigen.values, atol_value, rtol_value, "Channel rate operator R")
    upper_slack = _clamp_psd_eigenvalues(
        one(RT) .- lower_bounded, atol_value, rtol_value,
        "Channel rate complement I - R")
    bounded_values = one(RT) .- upper_slack
    R_bounded = Matrix{T}(
        R_eigen.vectors * Diagonal(bounded_values) * adjoint(R_eigen.vectors))

    # Stable form of alpha = 1 - sqrt(1-delta), avoiding cancellation at
    # small delta.  Since delta = 2alpha-alpha^2, the residual spectrum is
    # exactly alpha^2 lambda(1-lambda) for lambda in spectrum(R).
    alpha = delta_value / (one(RT) + sqrt(one(RT) - delta_value))
    K0 = Matrix{T}(I, dim, dim) .- alpha .* R_bounded
    residual_values = (alpha^2) .* bounded_values .* (one(RT) .- bounded_values)
    residual_values = _clamp_psd_eigenvalues(
        residual_values, atol_value, rtol_value, "Weak-measurement residual")
    U_residual = Matrix{T}(
        Diagonal(sqrt.(residual_values)) * adjoint(R_eigen.vectors))

    return (; K0, U_residual, alpha)
end

"""
    _apply_precomputed_channel!(evolving_dm, K0, U_residual, scratch)

Apply a precomputed weak-measurement channel using `scratch.rho_jump`.

Set `hermitize=false` for the genuine complex-linear map on arbitrary operators;
the default projection is appropriate for physical density matrices.
"""
function _apply_precomputed_channel!(
    evolving_dm::Matrix{<:Complex},
    K0::Matrix{<:Complex},
    U_residual::Matrix{<:Complex},
    scratch::ThermalizeScratch{<:Complex};
    hermitize::Bool = true,
)
    # Math: $rho' = K_0 rho K_0^dagger + rho_jump + U_r rho U_r^dagger$.
    mul!(scratch.sandwich_tmp, K0, evolving_dm)
    mul!(scratch.rho_next, scratch.sandwich_tmp, K0')

    # + rho_jump (pre-filled by _accumulate_rho_jump!)
    scratch.rho_next .+= scratch.rho_jump

    # + U_residual * rho * U_residual'
    mul!(scratch.sandwich_tmp, U_residual, evolving_dm)
    mul!(scratch.rho_next, scratch.sandwich_tmp, U_residual', 1.0, 1.0)

    # Keep it a density matrix numerically (ℝ-linear; skip for the ℂ-linear channel)
    hermitize && hermitianize!(scratch.rho_next)
    copyto!(evolving_dm, scratch.rho_next)
    return evolving_dm
end

"""
    _precompute_per_jump_channels(jumps, ham_or_trott, config, precomputed_data; rescale_by_inv_prob=true)

Precompute `K0` and residual Kraus matrices for every jump.

When `rescale_by_inv_prob=true`, each rate operator is divided by its jump
selection probability before channel construction.

# Returns
`(; K0s, U_residuals)` with one matrix pair per jump.
"""
function _precompute_per_jump_channels(
    jumps::AbstractVector{<:JumpOp},
    ham_or_trott::Union{HamHam, AbstractTrotter},
    config::Config{Thermalize},
    precomputed_data;
    rescale_by_inv_prob::Bool = true,
)
    CT = if ham_or_trott isa HamHam
        Complex{eltype(ham_or_trott.eigvals)}
    else
        eltype(ham_or_trott.eigvecs)  # already Complex{T}
    end
    dim = size(ham_or_trott isa HamHam ? ham_or_trott.data : ham_or_trott.eigvecs, 1)
    n_jumps = length(jumps)
    p_jump = 1.0 / n_jumps

    K0s = Vector{Matrix{CT}}(undef, n_jumps)
    U_residuals = Vector{Matrix{CT}}(undef, n_jumps)

    # Create temporary scratch for R construction (construction-time only)
    builder_scratch = ThermalizeScratch(CT, dim)

    @inbounds for a in 1:n_jumps
        _precompute_R([jumps[a]], ham_or_trott, config, precomputed_data, builder_scratch)
        R_a = copy(builder_scratch.R)
        if rescale_by_inv_prob
            R_a .*= (1.0 / p_jump)
        end
        (; K0, U_residual) = _build_cptp_channel(R_a, config.delta)
        K0s[a] = K0
        U_residuals[a] = U_residual
    end

    return (; K0s, U_residuals)
end

"""
    _apply_one_dm_substep!(evolving_dm, scratch, jump, U_coherent, K0, U_residual,
                           ham_or_trott, config, precomputed_data, jump_weight_scaling)

Apply one coherent-plus-dissipative per-jump channel substep.
"""
@inline function _apply_one_dm_substep!(
    evolving_dm::Matrix{<:Complex},
    scratch::ThermalizeScratch{<:Complex},
    jump::JumpOp,
    U_coherent::Union{Nothing, Matrix{<:Complex}},
    K0::Matrix{<:Complex},
    U_residual::Matrix{<:Complex},
    ham_or_trott::Union{HamHam, AbstractTrotter},
    config::Config{Thermalize},
    precomputed_data,
    jump_weight_scaling::Real;
    hermitize::Bool = true,
)
    _apply_coherent_unitary!(evolving_dm, U_coherent, scratch)
    _accumulate_rho_jump!(
        scratch, evolving_dm, jump, ham_or_trott, config, precomputed_data;
        jump_weight_scaling = jump_weight_scaling,
    )
    _apply_precomputed_channel!(evolving_dm, K0, U_residual, scratch; hermitize = hermitize)
    return nothing
end

"""
    _apply_one_adjoint_dm_substep!(evolving_dm, scratch, jump, U_coherent, K0, U_residual,
                                   ham_or_trott, config, precomputed_data_adj, jump_weight_scaling; hermitize)

Apply the Hilbert--Schmidt adjoint of one channel substep.

The caller supplies a frequency-negated transition function, allowing Hermitian
jumps to reuse the forward dissipator kernel before applying the adjoint
coherent unitary.
"""
@inline function _apply_one_adjoint_dm_substep!(
    evolving_dm::Matrix{<:Complex},
    scratch::ThermalizeScratch{<:Complex},
    jump::JumpOp,
    U_coherent::Union{Nothing, Matrix{<:Complex}},
    K0::Matrix{<:Complex},
    U_residual::Matrix{<:Complex},
    ham_or_trott::Union{HamHam, AbstractTrotter},
    config::Config{Thermalize},
    precomputed_data_adj,
    jump_weight_scaling::Real;
    hermitize::Bool = true,
)
    # WMₐ†: dissipator† on the INCOMING operator (rate-flipped forward path).
    _accumulate_rho_jump!(
        scratch, evolving_dm, jump, ham_or_trott, config, precomputed_data_adj;
        jump_weight_scaling = jump_weight_scaling,
    )
    # rho_next = K0† X K0 + rho_jump + U_res† X U_res
    mul!(scratch.sandwich_tmp, K0', evolving_dm)
    mul!(scratch.rho_next, scratch.sandwich_tmp, K0)
    scratch.rho_next .+= scratch.rho_jump
    mul!(scratch.sandwich_tmp, U_residual', evolving_dm)
    mul!(scratch.rho_next, scratch.sandwich_tmp, U_residual, 1.0, 1.0)
    hermitize && hermitianize!(scratch.rho_next)
    copyto!(evolving_dm, scratch.rho_next)
    # Cohₐ†: X ↦ U† X U  (applied AFTER WMₐ†, mirroring Sₐ† = Cohₐ† ∘ WMₐ†).
    _apply_adjoint_coherent_unitary!(evolving_dm, U_coherent, scratch)
    return nothing
end
