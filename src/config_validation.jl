"""
    _resolve_filter(config::Config) -> AbstractFilter

Returns `config.filter` if set, otherwise constructs a default
`GaussianFilter(config.sigma)`. Resolves the `Union{Nothing, AbstractFilter}`
field at the precompute boundary so all hot paths see a concrete filter
type and can specialise on it.
"""
@inline _resolve_filter(config::Config) =
    isnothing(config.filter) ? GaussianFilter(config.sigma) : config.filter

"""
    make_trotter_for_config(hamiltonian, config) -> AbstractTrotter

Build the Trotter cache required by a `TrotterDomain` configuration.

Coherent constructions return a [`TrotterTriple`](@ref) with independent
dissipative, outer-coherent, and inner-coherent Strang steps. Non-coherent
constructions return a single [`TrottTrott`](@ref). Missing register times or
substep counts raise `ArgumentError`.
"""
function make_trotter_for_config(hamiltonian::HamHam, config::Config)
    config.domain isa TrotterDomain ||
        throw(ArgumentError("make_trotter_for_config: config.domain must be TrotterDomain (got $(typeof(config.domain)))."))
    t0_D = register_t0_D(config)
    t0_D === nothing &&
        throw(ArgumentError("make_trotter_for_config: register_t0_D(config) must be set."))
    if with_coherent(config.construction)
        # Math: $t_(b-) = t0_(b-) / sigma$ and $t_(b+) = beta t0_(b+)$.
        t0_bm_evol = register_t0_b_minus(config) / config.sigma
        t0_bp_evol = config.beta * register_t0_b_plus(config)

        # Accessors resolve per-leg values through the shared compatibility field.
        M_D  = register_M_D(config)
        M_bm = register_M_b_minus(config)
        M_bp = register_M_b_plus(config)
        if M_D === nothing || M_bm === nothing || M_bp === nothing
            throw(ArgumentError(
                "make_trotter_for_config: per-leg M counts must be resolvable; got " *
                "M_D=$M_D, M_b_minus=$M_bm, M_b_plus=$M_bp (set " *
                "`num_trotter_steps_per_t0` for a common default or " *
                "`num_trotter_steps_per_t0_X` for per-leg control)."))
        end
        return TrotterTriple(hamiltonian, t0_D, t0_bm_evol, t0_bp_evol, M_D, M_bm, M_bp)
    else
        M_D = register_M_D(config)
        M_D === nothing &&
            throw(ArgumentError("make_trotter_for_config: register_M_D(config) must be set."))
        return TrottTrott(hamiltonian, t0_D, M_D)
    end
end

@inline function _require_trotter_value(actual, expected, label::AbstractString)
    isapprox(actual, expected; atol=0, rtol=100eps(float(expected))) ||
        throw(ArgumentError(
            "Trotter cache $label=$actual does not match the configuration value $expected. " *
            "Rebuild it with make_trotter_for_config(hamiltonian, config)."))
    return nothing
end

function _validate_trotter_cache!(
    config::Config{<:Any, TrotterDomain},
    hamiltonian::HamHam,
    trotter::TrottTrott,
)
    trotter.source_hamiltonian === hamiltonian || throw(ArgumentError(
        "The Trotter cache was built from a different Hamiltonian object. " *
        "Rebuild it for the supplied Hamiltonian."))
    size(trotter.eigvecs) == size(hamiltonian.data) || throw(ArgumentError(
        "Trotter cache dimension does not match the Hamiltonian."))
    _require_trotter_value(trotter.t0, register_t0_D(config), "t0_D")
    cached_M = trotter.num_trotter_steps_per_t0
    checks = with_coherent(config.construction) ?
        (("M_D", register_M_D(config)),
         ("M_b_minus", register_M_b_minus(config)),
         ("M_b_plus", register_M_b_plus(config))) :
        (("M_D", register_M_D(config)),)
    for (label, expected) in checks
        cached_M == expected || throw(ArgumentError(
            "Single Trotter cache $label=$cached_M does not match the configuration " *
            "value $expected. Use a TrotterTriple for independent coherent-leg counts."))
    end
    return nothing
end

function _validate_trotter_cache!(
    config::Config{<:Any, TrotterDomain},
    hamiltonian::HamHam,
    trotter::TrotterTriple,
)
    for (label, leg) in (("D", trotter.D),
                         ("b_minus", trotter.b_minus),
                         ("b_plus", trotter.b_plus))
        leg.source_hamiltonian === hamiltonian || throw(ArgumentError(
            "The Trotter $label cache was built from a different Hamiltonian object. " *
            "Rebuild it for the supplied Hamiltonian."))
        size(leg.eigvecs) == size(hamiltonian.data) || throw(ArgumentError(
            "Trotter $label cache dimension does not match the Hamiltonian."))
    end

    _require_trotter_value(trotter.D.t0, register_t0_D(config), "t0_D")
    checks = if with_coherent(config.construction)
        _require_trotter_value(
            trotter.b_minus.t0, register_t0_b_minus(config) / config.sigma,
            "t0_b_minus / sigma")
        _require_trotter_value(
            trotter.b_plus.t0, config.beta * register_t0_b_plus(config),
            "beta * t0_b_plus")
        (("M_D", trotter.D.num_trotter_steps_per_t0, register_M_D(config)),
         ("M_b_minus", trotter.b_minus.num_trotter_steps_per_t0,
          register_M_b_minus(config)),
         ("M_b_plus", trotter.b_plus.num_trotter_steps_per_t0,
          register_M_b_plus(config)))
    else
        (("M_D", trotter.D.num_trotter_steps_per_t0, register_M_D(config)),)
    end
    for (label, actual, expected) in checks
        actual == expected || throw(ArgumentError(
            "Trotter cache $label=$actual does not match the configuration value " *
            "$expected. Rebuild the cache."))
    end
    return nothing
end

function _collect_dll_filter_errors!(
    errors::Vector{String},
    filter::AbstractFilter,
    config_beta::T;
    label::String = string(nameof(typeof(filter))),
) where {T<:AbstractFloat}
    if filter isa DLLMultiChannelFilter
        beta_rtol = T(10) * eps(T)
        if !isfinite(filter.beta) ||
           !isapprox(filter.beta, config_beta;
                     atol = zero(T), rtol = beta_rtol)
            push!(errors, "$label.beta must match Config.beta.")
        end
        isempty(filter.channels) &&
            push!(errors, "$label must have at least one channel.")
        for (channel_index, channel) in enumerate(filter.channels)
            _collect_dll_filter_errors!(
                errors, channel, config_beta;
                label = "$label channel $channel_index ($(nameof(typeof(channel))))",
            )
        end
        return nothing
    end

    if !_is_admissible_dll_filter(filter)
        push!(errors,
              "$label is not an admissible DLL filter. Use DLLGaussianFilter, " *
              "DLLMetropolisFilter, a ShiftedSymmetricFilter of one of those " *
              "families, or DLLMultiChannelFilter with admissible channels.")
        return nothing
    end

    beta_rtol = T(10) * eps(T)
    if !hasproperty(filter, :beta) || !isfinite(filter.beta) ||
       !isapprox(filter.beta, config_beta;
                 atol = zero(T), rtol = beta_rtol)
        push!(errors, "$label.beta must match Config.beta.")
    end

    if filter isa DLLMetropolisFilter
        if !isfinite(filter.S) || filter.S <= 0
            push!(errors, "$label.S must be > 0 (finite; got $(filter.S)).")
        end
    elseif filter isa ShiftedSymmetricFilter
        !isfinite(filter.shift) &&
            push!(errors, "$label.shift must be finite.")
        (!isfinite(filter.weight) || filter.weight <= 0) &&
            push!(errors, "$label.weight must be finite and > 0.")
        _collect_dll_filter_errors!(
            errors, filter.base, config_beta;
            label = "$label.base ($(nameof(typeof(filter.base))))",
        )
    end
    return nothing
end

function validate_config!(
    config::Config;
    _allow_hypothetical_dll_trotter::Bool = false,
)
    errors = String[]

    # Common numerical domain. The implemented kernels divide by beta and
    # sigma, so the beta = 0 and sigma = 0 limits are not supported implicitly.
    config.num_qubits > 0 || push!(errors, "num_qubits must be > 0.")
    (!isfinite(config.beta) || config.beta <= 0) &&
        push!(errors, "beta (beta_alg) must be finite and > 0.")
    if config.beta_phys !== nothing &&
       (!isfinite(config.beta_phys) || config.beta_phys <= 0)
        push!(errors, "beta_phys must be finite and > 0 when provided.")
    end
    (!isfinite(config.sigma) || config.sigma <= 0) &&
        push!(errors, "sigma must be finite and > 0.")

    if !(config.construction isa DLL)
        for (name, value) in (("a", config.a), ("s", config.s))
            if value !== nothing && (!isfinite(value) || value < 0)
                push!(errors, "$name must be finite and >= 0 when provided.")
            end
        end
        if config.eta !== nothing && (!isfinite(config.eta) || config.eta <= 0)
            push!(errors, "eta must be finite and > 0 when provided.")
        end
    end

    if config.sim isa Thermalize
        if config.mixing_time === nothing || !isfinite(config.mixing_time) || config.mixing_time < 0
            push!(errors, "Thermalize requires finite mixing_time >= 0.")
        end
        if config.delta === nothing || !isfinite(config.delta) ||
           config.delta <= 0 || config.delta > 1
            push!(errors, "Thermalize requires 0 < delta <= 1.")
        end
    end

    # --- Domain-Specific Validation ---
    _collect_config_errors!(errors, config)
    _collect_simulation_errors!(errors, config)

    # --- Common Validation Logic ---
    # GNS coherent check removed: type system enforces with_coherent(::GNS) = false via trait.

    # DLL carries its thermal weighting in the filter, without a CKG rate.
    if !(config.construction isa DLL)
        if !(config.with_linear_combination) && config.gaussian_parameters == (nothing, nothing)
            push!(errors, "If with_linear_combination is false, gaussian_parameters must be set.")
        end

        if !(config.with_linear_combination)
            w_gamma, sigma_gamma = config.gaussian_parameters
            if w_gamma === nothing || sigma_gamma === nothing
                push!(errors, "For Gaussian transitions gaussian_parameters=(ω_γ, σ_γ) must be set.")
            else
                !isfinite(w_gamma) && push!(errors, "Gaussian transition ω_γ must be finite.")
                (!isfinite(sigma_gamma) || sigma_gamma <= 0) &&
                    push!(errors, "Gaussian transition σ_γ must be finite and > 0.")
                rhs = if config.construction isa GNS
                    2 * w_gamma / (sigma_gamma^2)
                else
                    2 * w_gamma / (config.sigma^2 + sigma_gamma^2)
                end
                parameter_relation_holds = isapprox(config.beta, rhs)
                if !(parameter_relation_holds)
                    if config.construction isa GNS
                        push!(errors, "For Gaussian transitions (GNS line) require beta ≈ 2*ω_γ/σ_γ^2")
                    else
                        push!(errors, "For Gaussian transitions (KMS line) require beta ≈ 2*ω_γ/(σ^2+σ_γ^2)")
                    end
                end
            end
        end

        if config.with_linear_combination
            if config.a === nothing || config.s === nothing
                push!(errors, "Linear-combination transitions require explicit finite a and s values.")
            end
            a_val = something(config.a, 0.0)
            s_val = something(config.s, 0.0)
            # (a, s) taxonomy: kinky Metropolis is exactly (s = 0, a = 0); smooth
            # Metropolis is (s > 0, any a ≥ 0). The (s = 0, a > 0) combination is
            # an a-regularised but unsmoothed rate that the thesis numerics never
            # use — reject it so we don't silently dispatch into an out-of-scope
            # rate function.
            if s_val == 0.0 && a_val != 0.0
                push!(errors, "For linear combinations require (s = 0, a = 0) for kinky Metropolis or (s > 0) for smooth Metropolis; got (s=0, a=$(a_val)).")
            end
            # Smooth Metropolis with `a == 0` requires positive `eta` in time domains.
            if a_val == 0.0 && config.domain isa Union{TimeDomain, TrotterDomain} && with_coherent(config.construction) && (isnothing(config.eta) || config.eta <= 0.0)
                push!(errors, "For linear combinations in the KMS DB case with a=0 in TIME or TROTTER domain, eta must be > 0.")
            end
        end
    end

    # --- GQSP coherent-step validation ---
    if config.with_gqsp
        if !with_coherent(config.construction)
            push!(errors, "with_gqsp requires a construction with coherent term (currently KMS only).")
        end
        # DLL+GQSP is not implemented: _precompute_data for (TimeDomain, DLL) does not
        # produce b_minus/b_plus/gamma_norm_factor that _gqsp_block_encoding_alpha needs.
        if config.construction isa DLL
            push!(errors, "with_gqsp is not supported with DLL construction (no DLL block-encoding norm yet).")
        end
        if !(config.domain isa Union{TimeDomain, TrotterDomain})
            push!(errors, "with_gqsp is only supported for TimeDomain or TrotterDomain.")
        end
        if config.gqsp_degree < 1
            push!(errors, "gqsp_degree must be ≥ 1.")
        end
        if config.gqsp_degree > 100
            push!(errors, "gqsp_degree must be ≤ 100.")
        end
    end

    # Jump-selection validation.
    if !(config.jump_selection in (:sweep, :random))
        push!(errors, "jump_selection must be :sweep or :random (got $(config.jump_selection)).")
    end

    # Filter families are part of the physical construction, not interchangeable
    # numerical windows. KMS/GNS use the CKG Gaussian at exactly `config.sigma`;
    # DLL uses a balance-weighted DLL filter at exactly `config.beta`.
    if config.filter !== nothing
        if config.construction isa DLL
            _collect_dll_filter_errors!(errors, config.filter, config.beta)
        elseif !(config.filter isa GaussianFilter)
            push!(errors,
                "$(nameof(typeof(config.construction))) construction requires " *
                "GaussianFilter(config.sigma); $(nameof(typeof(config.filter))) is " *
                "not supported for this construction.")
        else
            sigma_rtol = 10eps(typeof(config.sigma))
            if !isapprox(config.filter.sigma, config.sigma;
                         atol=zero(config.sigma), rtol=sigma_rtol)
                push!(errors, "GaussianFilter.sigma must match Config.sigma.")
            end
        end
    end

    # --- DLL construction validation (DLL-2) ---
    if config.construction isa DLL
        if config.sim isa Thermalize &&
           !(_allow_hypothetical_dll_trotter && config.domain isa TrotterDomain)
            push!(errors, "DLL Thermalize channels are not supported; use Lindbladian evolution.")
        end
        # DLL needs an explicit DLL filter at the OFT stage (Eq. 3.4 weighting).
        if config.filter === nothing
            push!(errors, "DLL construction requires an explicit AbstractFilter " *
                          "(e.g. DLLGaussianFilter(beta) or DLLMetropolisFilter(beta)).")
        end
        # EnergyDomain DLL is not in the DLL-2 scope; the paper's EnergyDomain
        # analogue would re-introduce an outer ω-grid and is deferred.
        if config.domain isa EnergyDomain
            push!(errors, "DLL construction is not supported in EnergyDomain (out of scope for DLL-2).")
        end
        # Trotter-domain DLL needs a quadrature defined on Trotter eigenvalues.
        if config.domain isa TrotterDomain && !_allow_hypothetical_dll_trotter
            push!(errors, "DLL construction in TrotterDomain is deferred — not yet supported.")
        end
    end

    # --- Error Throwing ---
    if !isempty(errors)
        error_message = "Invalid configuration found:\n" * join(["  - " * err for err in errors], "\n")
        throw(ArgumentError(error_message))
    end

    return nothing
end

_collect_simulation_errors!(::Vector{String}, ::Config) = nothing

function _collect_simulation_errors!(
    errors::Vector{String},
    config::Config{TensorNetworkSpectrum},
)
    config.domain isa BohrDomain || push!(errors,
        "TensorNetworkSpectrum requires BohrDomain as the exact mathematical target.")
    config.construction isa DLL || push!(errors,
        "TensorNetworkSpectrum currently supports only the DLL construction.")
    with_coherent(config.construction) || push!(errors,
        "TensorNetworkSpectrum requires the complete coherent correction.")
    config.beta_phys === nothing && push!(errors,
        "TensorNetworkSpectrum requires beta_phys so physical and algorithm frames are both explicit.")
    config.mixing_time === nothing || push!(errors,
        "TensorNetworkSpectrum does not accept mixing_time; it solves a parent spectrum, not a trajectory.")
    config.delta === nothing || push!(errors,
        "TensorNetworkSpectrum does not accept delta; it solves a parent spectrum, not a channel.")
    return nothing
end

"""
    validate_config!(config::Config, ham::HamHam; atol=1e-12, rtol=1e-10)

Validate a configuration and its physical/algorithmic temperature pair.

Require the configured system size and cached Gibbs state to match `ham`. When
`config.beta_phys` is set, also require `config.beta` to equal
`config.beta_phys * ham.rescaling_factor` within `atol` and `rtol`.
"""
function validate_config!(
    config::Config,
    ham::HamHam;
    atol::Real = 1e-12,
    rtol::Real = 1e-10,
    _allow_hypothetical_dll_trotter::Bool = false,
)
    validate_config!(config;
        _allow_hypothetical_dll_trotter = _allow_hypothetical_dll_trotter)
    dim = size(ham.data, 1)
    size(ham.data, 2) == dim || throw(ArgumentError("ham.data must be square."))
    expected_dim = 2^config.num_qubits
    dim == expected_dim || throw(ArgumentError(
        "config.num_qubits=$(config.num_qubits) implies dimension $expected_dim, " *
        "but ham.data has dimension $dim."))

    isfinite(ham.rescaling_factor) && ham.rescaling_factor > 0 ||
        throw(ArgumentError("ham.rescaling_factor must be finite and > 0."))
    if config.beta_phys !== nothing
        expected_beta_alg = config.beta_phys * ham.rescaling_factor
        if !isapprox(config.beta, expected_beta_alg; atol=atol, rtol=rtol)
            throw(ArgumentError(
                "Inconsistent (β_phys, β_alg) pair: config.beta_phys=$(config.beta_phys) and " *
                "ham.rescaling_factor=$(ham.rescaling_factor) imply β_alg=$(expected_beta_alg), " *
                "but config.beta=$(config.beta). Set them at construction so " *
                "`beta == beta_phys * ham.rescaling_factor`."))
        end
    end

    # HamHam caches the Gibbs state at construction. Check its diagonal weights
    # even on the legacy beta_alg-only path so dynamics and diagnostics cannot
    # silently use different temperatures.
    length(ham.eigvals) == dim || throw(ArgumentError(
        "ham.eigvals length $(length(ham.eigvals)) does not match dimension $dim."))
    emin = minimum(ham.eigvals)
    weights = exp.(-config.beta .* (ham.eigvals .- emin))
    weights ./= sum(weights)
    @inbounds for i in 1:dim
        isapprox(ham.gibbs[i, i], weights[i]; atol=atol, rtol=rtol) ||
            throw(ArgumentError(
                "ham.gibbs was cached at a beta_alg different from config.beta=$(config.beta). " *
                "Reconstruct HamHam with the same beta before building dynamics."))
    end
    return nothing
end

"""
    validate_config!(config::Config{TensorNetworkSpectrum}, hamiltonian::LocalHamiltonian1D)

Validate the tensor-network configuration against a backend-neutral local
Hamiltonian. Both inverse temperatures are mandatory. A physical-frame
`LocalHamiltonian1D` uses the physical Hamiltonian as the active calculation
coordinate and therefore has `rescaling_factor == 1` and
`beta_alg == beta_phys`. A nontrivial physical-to-algorithm scale is represented
by converting the local Hamiltonian to algorithm coordinates, where
`beta_alg == beta_phys * rescaling_factor`.
"""
function validate_config!(
    config::Config{TensorNetworkSpectrum},
    hamiltonian::LocalHamiltonian1D;
    atol::Real = 1e-12,
    rtol::Real = 1e-10,
)
    validate_config!(config)
    _validate_local_hamiltonian_integrity(hamiltonian)
    config.num_qubits == hamiltonian.num_sites || throw(ArgumentError(
        "config.num_qubits=$(config.num_qubits) does not match the local chain " *
        "length $(hamiltonian.num_sites)."))

    beta_phys_value = config.beta_phys
    beta_phys_value === nothing && throw(ArgumentError(
        "TensorNetworkSpectrum requires beta_phys."))
    expected_beta_alg = beta_phys_value * hamiltonian.rescaling_factor
    isapprox(config.beta, expected_beta_alg; atol=atol, rtol=rtol) ||
        throw(ArgumentError(
            "Inconsistent tensor-network temperature frames: beta_phys=$(beta_phys_value) " *
            "and rescaling_factor=$(hamiltonian.rescaling_factor) imply " *
            "beta_alg=$(expected_beta_alg), but Config.beta=$(config.beta)."))
    return nothing
end

function _collect_config_errors!(errors::Vector{String}, config::Config{<:Any, BohrDomain})
    return # No specific checks
end

# Validate each required register's Fourier relation independently.
# Math: $t0_X w0_X approx 2 pi / 2^(r_X)$.

function _check_register_fourier!(
    errors::Vector{String}, name::AbstractString, r, t0, w0;
    require_t0::Bool = true, require_w0::Bool = true,
)
    if isnothing(r) || r <= 0
        push!(errors, "register '$name': num_energy_bits_$name must be > 0.")
    end
    if require_t0 && (isnothing(t0) || t0 <= 0.0)
        push!(errors, "register '$name': t0_$name must be > 0.")
    end
    if require_w0 && (isnothing(w0) || w0 <= 0.0)
        push!(errors, "register '$name': w0_$name must be > 0.")
    end
    if require_t0 && require_w0 &&
       !isnothing(t0) && !isnothing(w0) && !isnothing(r) &&
       !isapprox(t0 * w0, 2pi / 2^r)
        push!(errors,
              "register '$name': Fourier relation t0_$name * w0_$name ≈ 2π / 2^r_$name must hold (got " *
              "t0=$t0, w0=$w0, r=$r).")
    end
    return errors
end

function _collect_config_errors!(errors::Vector{String}, config::Config{<:Any, EnergyDomain})
    # EnergyDomain dissipator uses analytical A(ω) — only (r_D, w0_D) needed,
    # no t0_D. Coherent term is built in BohrDomain so b_minus/b_plus registers
    # are not consulted here (validate_config! does not require them).
    _check_register_fourier!(
        errors, "D", register_r_D(config), register_t0_D(config), register_w0_D(config);
        require_t0 = false, require_w0 = true,
    )
end

function _collect_config_errors!(errors::Vector{String}, config::Config{<:Any, TimeDomain})
    # CKG/GNS TimeDomain dissipator: full (r_D, t0_D, w0_D) Fourier triple.
    _check_register_fourier!(
        errors, "D", register_r_D(config), register_t0_D(config), register_w0_D(config);
        require_t0 = true, require_w0 = true,
    )
    # Coherent term (KMS only; GNS short-circuited via with_coherent trait).
    if with_coherent(config.construction)
        _check_register_fourier!(
            errors, "b_minus",
            register_r_b_minus(config), register_t0_b_minus(config), register_w0_b_minus(config);
            require_t0 = true, require_w0 = true,
        )
        _check_register_fourier!(
            errors, "b_plus",
            register_r_b_plus(config), register_t0_b_plus(config), register_w0_b_plus(config);
            require_t0 = true, require_w0 = true,
        )
    end
end

# DLL TimeDomain has no ω-grid for the dissipator — `w0_D` is not part of
# the construction (Ding–Li–Lin 2024, Eq. 3.4). Only `r_D` and `t0_D` are
# required. DLL's coherent operator G is built directly on the same dissipative
# time grid via the DLL filter (Eq. 3.7 second equality) — there is no clean
# outer/inner split as in CKG B, so DLL never consumes `b_minus / b_plus`
# registers.
function _collect_config_errors!(errors::Vector{String}, config::Config{<:Any, TimeDomain, DLL})
    _check_register_fourier!(
        errors, "D", register_r_D(config), register_t0_D(config), register_w0_D(config);
        require_t0 = true, require_w0 = false,
    )
end

# DLL TrotterDomain is rejected later in `validate_config!` ("TrotterDomain DLL
# is deferred — not yet supported"). Specialise here only to skip the spurious
# `b_minus / b_plus` register checks that the generic TrotterDomain branch
# would otherwise emit for DLL constructions.
function _collect_config_errors!(errors::Vector{String}, config::Config{<:Any, TrotterDomain, DLL})
    _check_register_fourier!(
        errors, "D", register_r_D(config), register_t0_D(config), register_w0_D(config);
        require_t0 = true, require_w0 = false,
    )
    if isnothing(register_M_D(config)) || register_M_D(config) <= 0
        push!(errors, "For TrotterDomain, register_M_D(config) must be > 0.")
    end
end

function _collect_config_errors!(errors::Vector{String}, config::Config{<:Any, TrotterDomain})
    _check_register_fourier!(
        errors, "D", register_r_D(config), register_t0_D(config), register_w0_D(config);
        require_t0 = true, require_w0 = true,
    )
    if isnothing(register_M_D(config)) || register_M_D(config) <= 0
        push!(errors, "For TrotterDomain, register_M_D(config) must be > 0.")
    end
    if with_coherent(config.construction)
        if isnothing(register_M_b_minus(config)) || register_M_b_minus(config) <= 0
            push!(errors, "For coherent TrotterDomain, register_M_b_minus(config) must be > 0.")
        end
        if isnothing(register_M_b_plus(config)) || register_M_b_plus(config) <= 0
            push!(errors, "For coherent TrotterDomain, register_M_b_plus(config) must be > 0.")
        end
        _check_register_fourier!(
            errors, "b_minus",
            register_r_b_minus(config), register_t0_b_minus(config), register_w0_b_minus(config);
            require_t0 = true, require_w0 = true,
        )
        _check_register_fourier!(
            errors, "b_plus",
            register_r_b_plus(config), register_t0_b_plus(config), register_w0_b_plus(config);
            require_t0 = true, require_w0 = true,
        )
    end
end


function _print_press(config::Config{Lindbladian})
    params = [
        ("db", config.construction isa GNS ? :GNS : (config.construction isa DLL ? :DLL : :KMS)),
        ("domain", config.domain),
        ("num_qubits", config.num_qubits),
        ("r_D", register_r_D(config)),
        ("t0_D", register_t0_D(config)),
        ("w0_D", register_w0_D(config)),
        ("r_b_minus", register_r_b_minus(config)),
        ("t0_b_minus", register_t0_b_minus(config)),
        ("w0_b_minus", register_w0_b_minus(config)),
        ("r_b_plus", register_r_b_plus(config)),
        ("t0_b_plus", register_t0_b_plus(config)),
        ("w0_b_plus", register_w0_b_plus(config)),
        ("beta", config.beta),
        ("sigma", config.sigma),
        ("gaussian_parameters", config.gaussian_parameters),
        ("a", config.a),
        ("s", config.s),
        ("eta", config.eta),
        ("with_coherent", with_coherent(config.construction)),
        ("with_linear_combination", config.with_linear_combination),
        ("num_trotter_steps_per_t0", config.num_trotter_steps_per_t0)
    ]
    provided = filter(p -> p[2] !== nothing, params)
    if isempty(provided)
        return
    end

    println("--- The Press ---")
    for (name, value) in provided
        println("$name: $value")
    end
    println("-----------------")
end

function _print_press(config::Config{Thermalize})
    params = [
        ("db", config.construction isa GNS ? :GNS : (config.construction isa DLL ? :DLL : :KMS)),
        ("domain", config.domain),
        ("num_qubits", config.num_qubits),
        ("r_D", register_r_D(config)),
        ("t0_D", register_t0_D(config)),
        ("w0_D", register_w0_D(config)),
        ("r_b_minus", register_r_b_minus(config)),
        ("t0_b_minus", register_t0_b_minus(config)),
        ("w0_b_minus", register_w0_b_minus(config)),
        ("r_b_plus", register_r_b_plus(config)),
        ("t0_b_plus", register_t0_b_plus(config)),
        ("w0_b_plus", register_w0_b_plus(config)),
        ("beta", config.beta),
        ("sigma", config.sigma),
        ("gaussian_parameters", config.gaussian_parameters),
        ("a", config.a),
        ("s", config.s),
        ("eta", config.eta),
        ("with_coherent", with_coherent(config.construction)),
        ("with_linear_combination", config.with_linear_combination),
        ("num_trotter_steps_per_t0", config.num_trotter_steps_per_t0),
        ("mixing time", config.mixing_time),
        ("delta", config.delta),
    ]
    provided = filter(p -> p[2] !== nothing, params)
    if isempty(provided)
        return
    end

    println("--- The Press ---")
    for (name, value) in provided
        println("$name: $value")
    end
    println("-----------------")
end
