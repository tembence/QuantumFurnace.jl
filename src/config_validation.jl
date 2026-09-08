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

    if filter isa TimeFilter
        push!(errors, "Wrap TimeFilter in prepare_filter_transform with a declared support or numerical window.")
        return nothing
    end
    if !_is_admissible_dll_filter(filter) && !_is_dll_bohr_spec(filter)
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
            if config.domain isa TimeDomain && !_dll_time_supported(config.filter)
                push!(errors, "Custom DLL Time requires prepare_filter_transform with numerical controls.")
            end
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
    size(ham.gibbs) == (dim, dim) && all(isfinite, ham.gibbs) ||
        throw(ArgumentError("ham.gibbs must be a finite matrix matching ham dimensions."))
    emin = minimum(ham.eigvals)
    weights = exp.(-config.beta .* (ham.eigvals .- emin))
    weights ./= sum(weights)
    @inbounds for i in 1:dim
        isapprox(ham.gibbs[i, i], weights[i]; atol=atol, rtol=rtol) ||
            throw(ArgumentError(
                "ham.gibbs was cached at a beta_alg different from config.beta=$(config.beta). " *
                "Reconstruct HamHam with the same beta before building dynamics."))
    end
    @inbounds for j in 1:dim, i in 1:dim
        i == j && continue
        abs(ham.gibbs[i, j]) <= atol || throw(ArgumentError(
            "ham.gibbs must be diagonal in the stored Hamiltonian eigenbasis."))
    end
    return nothing
end

# These are coordinate conversions of existing full DLL Fourier pairs, with
# no generator-time rescaling: F_alg(nu) = F_phys(R*nu).
_physical_dll_filter(f::DLLGaussianFilter, R::T, beta::T) where {T} =
    DLLGaussianFilter(beta)
_physical_dll_filter(f::DLLMetropolisFilter, R::T, beta::T) where {T} =
    DLLMetropolisFilter(beta; S=T(f.S / R))
function _physical_dll_filter(f::ShiftedSymmetricFilter, R::T, beta::T) where {T}
    f.shift >= 0 || throw(ArgumentError(
        "Physical symmetric filter shifts must be nonnegative; use abs(shift) explicitly."))
    return ShiftedSymmetricFilter(_physical_dll_filter(f.base, R, beta),
        T(f.shift / R), T(f.weight))
end
function _physical_dll_filter(f::DLLMultiChannelFilter, R::T, beta::T) where {T}
    return DLLMultiChannelFilter([_physical_dll_filter(c, R, beta) for c in f.channels], beta)
end
_physical_dll_filter(f::AbstractFilter, R, beta) = throw(ArgumentError(
    "Physical conversion is implemented only for built-in DLL filters; got $(typeof(f))."))

"""
    prepare_gibbs_inputs(H; beta_phys=nothing, temperature=nothing, filter=DLLGaussianFilter,
        jumps=:onsite_paulis, rates=1, complete_adjoint=false,
        basis=:computational, domain=BohrDomain(), construction=DLL(),
        time_step=nothing, num_energy_bits=nothing, clock=nothing)

Prepare `(hamiltonian, config, jumps, provenance)` for the existing low-level
DLL constructors. `H` is a physical matrix, raw Hamiltonian tuple, or `HamHam`.
All filter parameters and `time_step` are physical-frame inputs. A filter type
or factory receives `beta_phys`; an instance must match that temperature.
Built-in Gaussian, Metropolis, symmetric translates and multiple channels are
supported. CKG configuration remains available through the legacy Config API.

Supply exactly one of `beta_phys` or `temperature` (energy units, `k_B=1`).
Finite `beta_phys > 0` is required: neither the infinite-temperature `beta=0`
limit nor the zero-temperature `beta=Inf` limit is implemented by these kernels.
A prebuilt Hamiltonian must already cache the requested Gibbs state; explicitly
use `HamHam(ham; beta_phys=new_beta)` to change temperature without diagonalising.

Bohr needs no registers. Time requires an explicit physical `time_step` and
`num_energy_bits`; legacy filters use their shared time grid for both terms. Prepared filters
accept independent coherent controls via prepare_filter_transform. No accuracy certificate
or default quadrature resolution is inferred from these controls.

The default clock is `:raw_generator`, multiplier one. An optional existing
`GeneratorClock` applies `sqrt(clock.multiplier)` to every prepared source,
thus multiplying the entire generator (including B). Its `rate` is a user
declaration, not an estimated spectral gap. For `L_new=m*L_raw`, decay rates
multiply by `m`, and an equal evolution uses `t_new=t_raw/m`.
"""
function prepare_gibbs_inputs(H; beta_phys::Union{Nothing,Real}=nothing,
    temperature::Union{Nothing,Real}=nothing, filter=DLLGaussianFilter,
    jumps=:onsite_paulis, rates=1, complete_adjoint::Bool=false,
    basis::Symbol=:computational, domain::AbstractDomain=BohrDomain(),
    construction::AbstractConstruction=DLL(), time_step::Union{Nothing,Real}=nothing,
    num_energy_bits::Union{Nothing,Int}=nothing, clock=nothing)
    construction isa DLL || throw(ArgumentError(
        "prepare_gibbs_inputs currently supports DLL; use the legacy Config API for CKG/GNS."))
    domain isa Union{BohrDomain,TimeDomain} || throw(ArgumentError(
        "DLL preparation supports BohrDomain and TimeDomain only."))
    (beta_phys === nothing) != (temperature === nothing) || throw(ArgumentError(
        "Provide exactly one of beta_phys or temperature (energy units, k_B=1)."))
    temperature_kind = temperature === nothing ? :beta_phys : :temperature
    if temperature !== nothing
        isfinite(temperature) && temperature > 0 || throw(ArgumentError(
            "temperature must be finite and positive (k_B=1); endpoint limits are unsupported."))
        beta_phys = inv(float(temperature))
    end
    isfinite(beta_phys) && beta_phys > 0 || throw(ArgumentError(
        "beta_phys must be finite and > 0; beta=0 (infinite temperature) and beta=Inf (zero temperature) are unsupported."))
    if domain isa BohrDomain
        time_step === nothing && num_energy_bits === nothing || throw(ArgumentError(
            "BohrDomain does not use time registers; omit time_step and num_energy_bits."))
    else
        time_step !== nothing && isfinite(time_step) && time_step > 0 &&
            num_energy_bits !== nothing && num_energy_bits > 0 || throw(ArgumentError(
                "TimeDomain requires positive physical time_step and num_energy_bits."))
    end
    clock === nothing || clock isa GeneratorClock || throw(ArgumentError(
        "clock must be nothing (raw generator) or an explicit GeneratorClock."))
    H isa Union{HamHam,AbstractMatrix,NamedTuple} || throw(ArgumentError(
        "H must be a physical matrix, raw Hamiltonian tuple, or HamHam."))
    ham = HamHam(H; beta_phys)
    T = eltype(ham.eigvals)
    physical_beta = T(beta_phys)
    algorithm_beta = beta_alg(ham, physical_beta)
    filter isa AbstractFilter || applicable(filter, physical_beta) || throw(ArgumentError(
        "filter must be a built-in DLL instance or a factory accepting physical beta."))
    physical_filter = filter isa AbstractFilter ? filter : filter(physical_beta)
    physical_filter isa AbstractFilter || throw(ArgumentError("Filter factory must return an AbstractFilter."))
    filter_errors = String[]
    _collect_dll_filter_errors!(filter_errors, physical_filter, physical_beta;
        label="physical filter")
    isempty(filter_errors) || throw(ArgumentError(
        "Filter must match beta_phys at Hamiltonian precision: " * join(filter_errors, "; ")))
    physical_filter = _physical_dll_filter(physical_filter, one(T), physical_beta)
    algorithm_filter = _physical_dll_filter(physical_filter, ham.rescaling_factor, algorithm_beta)
    cfg = Config(; sim=Lindbladian(), domain, construction,
        num_qubits=trailing_zeros(size(ham.data,1)), with_linear_combination=false,
        beta=algorithm_beta, beta_phys=physical_beta,
        sigma=one(T), # Unused DLL compatibility field, not a physical filter width.
        filter=algorithm_filter, num_energy_bits_D=num_energy_bits,
        t0_D=time_step === nothing ? nothing : T(time_step * ham.rescaling_factor))
    validate_config!(cfg, ham; atol=100eps(T), rtol=100eps(T))
    if H isa HamHam
        validate_config!(cfg, H; atol=100eps(T), rtol=100eps(T))
        _jump_matches(H.bohr_freqs, ham.bohr_freqs, 100eps(T)) &&
            H.bohr_dict == ham.bohr_dict || throw(ArgumentError(
                "Prebuilt Hamiltonian Bohr caches are stale; reconstruct HamHam explicitly."))
    end
    prepared = prepare_jumps(jumps, ham; basis, rates, complete_adjoint)
    multiplier = clock === nothing ? one(T) : T(clock.multiplier)
    isfinite(multiplier) && multiplier > 0 && isfinite(inv(multiplier)) || throw(ArgumentError(
        "Clock multiplier must be representable, finite and positive."))
    compiled_jumps = clock === nothing ? prepared.jumps :
        prepare_jumps(prepared.jumps, ham; rates=multiplier).jumps
    physical_channels = _flatten_local_dll_channels((physical_filter,))
    algorithm_channels = _flatten_local_dll_channels((algorithm_filter,))
    provenance = (; beta_phys=physical_beta, beta_alg=algorithm_beta,
        input_beta_phys=beta_phys, working_precision=T,
        spectral_preparation=H isa AbstractMatrix ? :diagonalised : :validated_cached,
        gibbs_preparation=H isa AbstractMatrix ? :matrix_spectral_preparation : :recomputed_from_validated_spectrum,
        cached_gibbs_check=H isa HamHam ? :passed : :not_applicable,
        cached_gibbs_atol=100eps(T), cached_gibbs_rtol=100eps(T),
        rescaling_factor=ham.rescaling_factor, energy_shift=ham.shift,
        input_frame=H isa AbstractMatrix ? :physical : :algorithm,
        filter_input_frame=:physical, resolved_frame=:algorithm,
        temperature_input=temperature_kind, temperature_unit=:energy_kB_one,
        physical_filters=Tuple(_filter_frame(c,:physical,T) for c in physical_channels),
        algorithm_filters=Tuple(_filter_frame(c,:algorithm,T) for c in algorithm_channels),
        physical_filter, algorithm_filter,
        filter_evidence=Tuple(filter_evidence(c) for c in physical_channels),
        sources=prepared.provenance,
        physical_time_step=time_step === nothing ? nothing : T(time_step),
        algorithm_time_step=cfg.t0_D, clock_label=clock === nothing ? :raw_generator : clock.label,
        generator_multiplier=multiplier, time_multiplier=inv(multiplier), derived_clock=clock,
        clock_rate_evidence=clock === nothing ? :not_provided : :user_declared,
        sigma_role=:unused_dll_compatibility,
        gibbs_underflow=any(iszero, diag(ham.gibbs)))
    return (; hamiltonian=ham, config=cfg, jumps=compiled_jumps, provenance)
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

# Coordinate adapter: preserve the entire callback, including widths/phases and
# thermal tilt. No extra amplitude or generator-clock factor is introduced.
struct _RescaledDLLFilter{T<:AbstractFloat,F<:_UserDLLFilter,M} <: _UserDLLFilter
    beta::T
    base::F
    energy_scale::T
    metadata::M
end
function _physical_dll_filter(f::_UserDLLFilter, R::T, beta::T) where {T<:AbstractFloat}
    isone(R) && typeof(f.beta) == T && f.beta == beta && return f
    metadata = _user_filter_metadata(beta;name=f.metadata.name,version=f.metadata.version,
        parameters=f.metadata.parameters,
        support=f.metadata.support === nothing ? nothing : f.metadata.support/R,
        tail_bound=f.metadata.tail_bound===nothing ? nothing : W->f.metadata.tail_bound(R*W)/R)
    return _RescaledDLLFilter(beta,f,R,metadata)
end
freq_kernel(f::_RescaledDLLFilter, nu::Real) =
    _finite_filter_value(freq_kernel(f.base,f.energy_scale*_user_filter_coordinate(f,nu)),f,nu)
q_weight(f::_RescaledDLLFilter, nu::Real) =
    _finite_filter_value(q_weight(f.base,f.energy_scale*_user_filter_coordinate(f,nu)),f,nu)
_is_admissible_dll_filter(f::_RescaledDLLFilter) = _is_admissible_dll_filter(f.base)
filter_evidence(f::_RescaledDLLFilter) = merge(filter_evidence(f.base),
    (;support=f.metadata.support,frequency_coordinate_scale=f.energy_scale))
function _filter_frame(f::_UserDLLFilter, frame::Symbol, ::Type{T}) where {T<:AbstractFloat}
    return DLLFilterFrame{T}(f.metadata.name,
        f.metadata.support === nothing ? nothing : T(f.metadata.support),zero(T),one(T),frame)
end

# Coordinate conversion includes the Fourier Jacobian: F_alg(v)=F_phys(R*v),
# f_alg(t)=f_phys(t/R)/R. Tail providers retain their declared input L1 meaning.
function _physical_dll_filter(f::TimeFilter,R::T,beta::T) where {T<:AbstractFloat}
    base=deepcopy(f)
    return TimeFilter(beta;kernel=t->time_kernel(base,t/R)/R,
        name=f.metadata.name,version=f.metadata.version,parameters=f.metadata.parameters,
        support=f.metadata.support===nothing ? nothing : T(f.metadata.support*R),
        tail_bound=f.metadata.tail_bound===nothing ? nothing : W->f.metadata.tail_bound(W/R))
end
function _physical_dll_filter(p::PreparedFilterTransform,R::T,beta::T) where {T<:AbstractFloat}
    c=p.controls
    scale=c.input==:time ? R : inv(R)
    coherent=merge(c.coherent,(;
        time_step=c.coherent.time_step===nothing ? nothing : T(c.coherent.time_step*R),
        time_window=c.coherent.time_window===nothing ? nothing : T(c.coherent.time_window*R),
        frequency_window=c.coherent.frequency_window===nothing ? nothing : T(c.coherent.frequency_window/R)))
    base=_physical_dll_filter(p.base,R,beta)
    controls=merge(c,(;coherent,window=c.window===nothing ? nothing : T(c.window*scale),
        support=c.support===nothing ? nothing : T(c.support*scale),
        breakpoints=Tuple(T(x*scale) for x in c.breakpoints),rtol=T(c.rtol),
        atol=T(c.atol*(c.input==:frequency ? inv(R) : one(T)))))
    return PreparedFilterTransform(beta,deepcopy(base),controls,
        Dict{Tuple{Symbol,T},Tuple{Complex{T},T,Symbol}}())
end
_filter_frame(p::PreparedFilterTransform,frame::Symbol,::Type{T}) where {T<:AbstractFloat} =
    _filter_frame(p.base,frame,T)
