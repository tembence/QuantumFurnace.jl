# Operator Fourier-transform filters.
# Math: $f(t) = 1/(2 pi) integral hat(f)(nu) exp(-i t nu) dif nu$.

"""
    AbstractFilter

Supertype for time/frequency kernels used by the operator Fourier transform.
"""
abstract type AbstractFilter end

# Only filters whose frequency kernel has the DLL form
# $\widehat f(\nu)=q(\nu)e^{-\beta\nu/4}$ with
# $q(-\nu)=\overline{q(\nu)}$ are admissible in a DLL construction.
@inline _is_admissible_dll_filter(::AbstractFilter) = false

"""
    GaussianFilter{T<:AbstractFloat}(sigma::T)

CKG Gaussian filter with energy width `sigma`.

Math: `\$f(t) = exp(-sigma^2 t^2)\$` and
`\$hat(f)(nu) prop exp(-nu^2/(4 sigma^2))\$`.
"""
struct GaussianFilter{T<:AbstractFloat} <: AbstractFilter
    sigma::T
    function GaussianFilter{T}(sigma::T) where {T<:AbstractFloat}
        isfinite(sigma) && sigma > zero(T) || throw(ArgumentError(
            "GaussianFilter.sigma must be finite and > 0."))
        return new{T}(sigma)
    end
end

GaussianFilter(sigma::T) where {T<:AbstractFloat} = GaussianFilter{T}(sigma)

"""
    DLLGaussianFilter{T<:AbstractFloat}(beta::T)

DLL Gaussian-type filter at inverse temperature `beta`.

The frequency kernel includes the KMS factor:
`\$hat(f)(nu) = exp(1/8) exp(-(beta nu + 1)^2/8)\$`. This numerical variant uses
`w = 1`, so it does not have the compact support required by the rigorous
Paley–Wiener bound.
"""
struct DLLGaussianFilter{T<:AbstractFloat} <: AbstractFilter
    beta::T
    function DLLGaussianFilter{T}(beta::T) where {T<:AbstractFloat}
        isfinite(beta) && beta > zero(T) || throw(ArgumentError(
            "DLLGaussianFilter.beta must be finite and > 0."))
        return new{T}(beta)
    end
end

DLLGaussianFilter(beta::T) where {T<:AbstractFloat} = DLLGaussianFilter{T}(beta)

@inline _is_admissible_dll_filter(::DLLGaussianFilter) = true

# NUFFT prefactors store every kernel as complex data.
Base.eltype(::GaussianFilter{T}) where {T} = Complex{T}
Base.eltype(::DLLGaussianFilter{T}) where {T} = Complex{T}

function time_kernel end
function freq_kernel end
function filter_time_cutoff end

# Owned finite-frequency evidence, not a continuum filter certificate. Sampling
# happens before threading and is shared by gain, loss and coherent construction.
struct _DLLBohrFilter{T<:AbstractFloat} <: AbstractFilter
    beta::T
    values::Dict{T,Complex{T}}
    function _DLLBohrFilter(beta::T, values::Dict{T,Complex{T}}) where {T<:AbstractFloat}
        isfinite(beta) && beta > 0 || throw(ArgumentError("DLL beta must be finite and positive."))
        for (nu, value) in values
            isfinite(nu) && isfinite(value) || throw(ArgumentError("Nonfinite DLL Bohr amplitude at $nu."))
            haskey(values, iszero(nu) ? nu : -nu) || throw(ArgumentError("DLL Bohr table must include both frequency signs."))
            nu < 0 && continue
            tilt = -(beta / T(2)) * nu
            expected = _scale_dll_amplitude(conj(values[iszero(nu) ? nu : -nu]), tilt)
            scale = max(abs(value), abs(expected))
            # Log-domain evaluation has a conditioning factor at steep tilt.
            rtol = T(128) * eps(T) * max(one(T), min(abs(tilt), -log(nextfloat(zero(T)))))
            abs(value - expected) <= max(rtol * scale, T(4)*nextfloat(zero(T))) || throw(ArgumentError(
                "DLL weighted reflection failed at frequency $nu; supply a balanced amplitude (thermal weight exactly once)."))
        end
        new{T}(beta, copy(values))
    end
end

function _scale_dll_amplitude(z::Complex{T}, logscale::Real) where {T<:AbstractFloat}
    iszero(z) && return zero(z)
    scale = max(abs(real(z)),abs(imag(z)))
    unit = z/scale
    magnitude = abs(unit)
    amplitude = exp((log(scale)+log(magnitude))+T(logscale))
    isfinite(amplitude) || throw(ArgumentError("DLL amplitude overflows working precision."))
    return amplitude*(unit/magnitude)
end
Base.eltype(::_DLLBohrFilter{T}) where {T} = Complex{T}
_is_admissible_dll_filter(::_DLLBohrFilter) = true
freq_kernel(f::_DLLBohrFilter{T}, nu::Real) where {T} = f.values[T(nu)]
function q_weight(f::_DLLBohrFilter{T}, nu::Real) where {T}
    x = abs(T(nu))
    value = _scale_dll_amplitude(freq_kernel(f, iszero(x) ? zero(T) : -x), -(f.beta / T(4)) * x)
    return nu < 0 ? value : conj(value)
end

function _prepare_dll_bohr_filter(filter::AbstractFilter, eigvals::AbstractVector{T};
                                  beta=filter.beta) where {T<:AbstractFloat}
    frequencies = sort!(unique!(T[a-b for a in eigvals for b in eigvals]))
    return _sample_dll_bohr_filter(filter, frequencies, T(beta))
end
function _sample_dll_bohr_filter(filter::AbstractFilter, frequencies::AbstractVector{T},
                                 beta::T) where {T<:AbstractFloat}
    isapprox(filter.beta, beta; atol=zero(T), rtol=10eps(T)) ||
        throw(ArgumentError("DLL filter beta must match construction beta."))
    values = Dict{T,Complex{T}}()
    for nu in frequencies
        value = freq_kernel(filter, nu)
        value isa Number && isfinite(value) || throw(ArgumentError(
            "DLL callback must return a finite amplitude at frequency $nu."))
        values[nu] = Complex{T}(value)
    end
    return _DLLBohrFilter(T(beta), values)
end

"""
    time_kernel(filter::GaussianFilter, t) -> Real

Evaluate `\$exp(-sigma^2 t^2)\$` with deterministic operation ordering.
"""
@inline time_kernel(f::GaussianFilter{T}, t::Real) where {T} =
    exp(-(f.sigma^2) * t^2)

"""
    freq_kernel(filter::GaussianFilter, ν)

Evaluate the unnormalised Gaussian `\$exp(-nu^2/(4 sigma^2))\$`.
"""
@inline freq_kernel(f::GaussianFilter{T}, nu::Real) where {T} =
    exp(-nu^2 / (4 * f.sigma^2))

"""
    _time_oft_prefactor_gaussian(filter::GaussianFilter)

Return the Gaussian OFT normalisation used for time-grid truncation.
"""
@inline _time_oft_prefactor_gaussian(f::GaussianFilter{T}) where {T} =
    sqrt(f.sigma * sqrt(T(2) / T(pi)) / (2 * T(pi)))

"""
    filter_time_cutoff(filter::GaussianFilter, tol)

Return a time cutoff whose Gaussian tail is below `tol`.
"""
@inline filter_time_cutoff(f::GaussianFilter{T}, tol::Real) where {T} =
    sqrt(log(_time_oft_prefactor_gaussian(f) / tol)) / f.sigma

"""
    q_weight(filter::DLLGaussianFilter, nu) -> Real

Evaluate the DLL balance weight `\$q(nu) = exp(-(beta nu)^2/8)\$`.
"""
@inline q_weight(f::DLLGaussianFilter{T}, nu::Real) where {T} =
    exp(-(f.beta * nu)^2 / 8)

"""
    freq_kernel(filter::DLLGaussianFilter, ν)

Evaluate the full DLL frequency kernel, including the KMS factor.
"""
@inline freq_kernel(f::DLLGaussianFilter{T}, nu::Real) where {T} =
    exp(T(1) / 8) * exp(-(f.beta * nu + 1)^2 / 8)

"""
    time_kernel(filter::DLLGaussianFilter, t)

Evaluate the complex closed-form inverse Fourier transform.

Math: `\$f(t) = exp(1/8) sqrt(2/pi)/beta exp(-2t^2/beta^2 + i t/beta)\$`.
"""
@inline function time_kernel(f::DLLGaussianFilter{T}, t::Real) where {T}
    pref = exp(T(1) / 8) * sqrt(T(2) / T(pi)) / f.beta
    decay = exp(-2 * t^2 / f.beta^2)
    phase = cis(t / f.beta)
    return pref * decay * phase
end

"""
    _time_oft_prefactor_dll(filter::DLLGaussianFilter)

Return `|f(0)|` for DLL time-grid truncation.
"""
@inline _time_oft_prefactor_dll(f::DLLGaussianFilter{T}) where {T} =
    exp(T(1) / 8) * sqrt(T(2) / T(pi)) / f.beta

"""
    filter_time_cutoff(filter::DLLGaussianFilter, tol)

Return a time cutoff whose DLL Gaussian tail is below `tol`.
"""
@inline filter_time_cutoff(f::DLLGaussianFilter{T}, tol::Real) where {T} =
    (f.beta / sqrt(T(2))) * sqrt(log(_time_oft_prefactor_dll(f) / tol))

# Hörmander bump used to compactly support the DLL Metropolis filter.
# Math: $eta(t) = exp(-1/t)$ for $t > 0$ and
# $w(x) = phi(2(1-abs(x)))$, with a flat top on $abs(x) <= 1/2$.

"""
    _hormander_eta(t)

Evaluate the one-sided smooth bump factor `eta`.
"""
@inline function _hormander_eta(t::T) where {T<:AbstractFloat}
    return t > zero(T) ? exp(-one(T) / t) : zero(T)
end

"""
    _hormander_phi(t)

Evaluate the smooth step from zero to one on `[0, 1]`.
"""
@inline function _hormander_phi(t::T) where {T<:AbstractFloat}
    if t <= zero(T)
        return zero(T)
    elseif t >= one(T)
        return one(T)
    end
    a = _hormander_eta(t)
    b = _hormander_eta(one(T) - t)
    return a / (a + b)
end

"""
    _hormander_bump(x)

Evaluate the even compact bump with unit plateau on `[-1/2, 1/2]`.
"""
@inline function _hormander_bump(x::T) where {T<:AbstractFloat}
    ax = abs(x)
    if ax >= one(T)
        return zero(T)
    elseif ax <= one(T) / 2
        return one(T)
    end
    return _hormander_phi(T(2) * (one(T) - ax))
end

# Promote non-floating inputs before evaluating the bump.
@inline _hormander_bump(x::Real) = _hormander_bump(float(x))

"""
    DLLMetropolisFilter{T<:AbstractFloat}(beta::T; S::Real = T(2))

Compactly supported DLL Metropolis-type filter.

# Arguments
- `beta`: Inverse temperature.

# Keywords
- `S`: Support radius; the bump is flat on `abs(nu) <= S/2` and zero for
  `abs(nu) >= S`.

The caller must ensure `\$S/2 >= max abs(nu_BH)\$` when the whole Bohr spectrum
should see the Metropolis plateau.
"""
struct DLLMetropolisFilter{T<:AbstractFloat} <: AbstractFilter
    beta::T
    S::T
    function DLLMetropolisFilter{T}(beta::T, S::T) where {T<:AbstractFloat}
        isfinite(beta) && beta > zero(T) || throw(ArgumentError(
            "DLLMetropolisFilter.beta must be finite and > 0."))
        isfinite(S) && S > zero(T) || throw(ArgumentError(
            "DLLMetropolisFilter.S must be finite and > 0."))
        return new{T}(beta, S)
    end
end

DLLMetropolisFilter(beta::T; S::Real = T(2)) where {T<:AbstractFloat} =
    DLLMetropolisFilter{T}(beta, T(S))

Base.eltype(::DLLMetropolisFilter{T}) where {T} = Complex{T}
@inline _is_admissible_dll_filter(::DLLMetropolisFilter) = true

"""Reject filters that do not implement an admissible DLL balance kernel."""
function _require_admissible_dll_filter(
    filter::AbstractFilter;
    beta::Union{Nothing, Real}=nothing,
)
    _is_admissible_dll_filter(filter) || throw(ArgumentError(
        "$(nameof(typeof(filter))) is not an admissible DLL filter. Use " *
        "DLLGaussianFilter, DLLMetropolisFilter, ShiftedSymmetricFilter, " *
        "or DLLMultiChannelFilter."))

    hasproperty(filter, :beta) || throw(ArgumentError(
        "$(nameof(typeof(filter))) must expose its DLL inverse temperature as `beta`."))
    filter_beta = getproperty(filter, :beta)
    isfinite(filter_beta) && filter_beta > 0 || throw(ArgumentError(
        "$(nameof(typeof(filter))).beta must be finite and > 0."))

    if beta !== nothing
        beta_value = float(beta)
        isfinite(beta_value) && beta_value > 0 || throw(ArgumentError(
            "DLL helper beta must be finite and > 0."))
        R = promote_type(typeof(float(filter_beta)), typeof(beta_value))
        beta_rtol = R(10) * eps(R)
        isapprox(R(filter_beta), R(beta_value); atol=zero(R), rtol=beta_rtol) ||
            throw(ArgumentError(
                "$(nameof(typeof(filter))).beta=$(filter_beta) must match beta=$(beta_value)."))
    end
    return filter
end

"""
    q_weight(filter::DLLMetropolisFilter, ν)

Evaluate the compactly supported DLL balance weight `q(nu)`.
"""
@inline q_weight(f::DLLMetropolisFilter{T}, nu::Real) where {T} =
    exp(-sqrt(one(T) + (f.beta * nu)^2) / 4) * _hormander_bump(nu / f.S)

"""
    freq_kernel(filter::DLLMetropolisFilter, ν)

Evaluate `\$hat(f)(nu) = q(nu) exp(-beta nu/4)\$`.

On the flat top this approaches one for downward transitions and
`\$exp(-beta nu/2)\$` for upward transitions; it is zero outside `[-S, S]`.
"""
@inline freq_kernel(f::DLLMetropolisFilter{T}, nu::Real) where {T} =
    q_weight(f, nu) * exp(-f.beta * nu / 4)

"""
    time_kernel(filter::DLLMetropolisFilter, t)

Numerically evaluate the compact-support inverse Fourier transform.

# Returns
A `Complex{T}` value. Quadrature uses relative tolerance `1e-12` and absolute
floor `64eps(T)` to control cancellation at large `abs(t)`.
"""
function time_kernel(f::DLLMetropolisFilter{T}, t::Real) where {T}
    # Math: $f(t) = 1/(2 pi) integral_(-S)^S hat(f)(nu) exp(-i t nu) dif nu$.
    integrand = nu -> freq_kernel(f, nu) * cis(-t * nu)
    val, _ = quadgk(integrand, -f.S, f.S;
                    rtol = T(1e-12), atol = T(64) * eps(T))
    return Complex{T}(val / T(2 * π))
end

"""
    _dll_metropolis_fourier_d4_l1_bound(filter) -> Real

Return a rigorous upper bound on the `L1` norm of the fourth frequency
derivative of the full Metropolis frequency kernel.

The bound uses the exact filter in Eq. (3.19) of Ding--Li--Lin and elementary
global derivative bounds for the implemented Hoermander step. Since the kernel
and its first three derivatives vanish at `+-S`, four integrations by parts
give `|f(t)| <= bound / (2pi |t|^4)` for every nonzero `t`.
"""
function _dll_metropolis_fourier_d4_l1_bound(
    f::DLLMetropolisFilter{T},
) where {T<:AbstractFloat}
    isfinite(f.beta) && f.beta > zero(T) ||
        throw(ArgumentError("DLLMetropolisFilter.beta must be finite and > 0."))
    isfinite(f.S) && f.S > zero(T) ||
        throw(ArgumentError("DLLMetropolisFilter.S must be finite and > 0."))

    # For eta(x)=exp(-1/x), x>0, every term exp(-1/x)x^(-m) is bounded by
    # (m/e)^m. These are bounds for eta^(k), k=0,...,4, obtained from the
    # explicit derivative polynomials.
    # Use the exact elementary inequality e > 5/2, rather than a rounded
    # evaluation of `exp(1)`, so these remain one-sided upper bounds.
    power_bound(m::Int) = (T(m) / (T(5) / T(2)))^m
    eta_bounds = Vector{T}(undef, 5)
    eta_bounds[1] = one(T)
    eta_bounds[2] = power_bound(2)
    eta_bounds[3] = power_bound(4) + T(2) * power_bound(3)
    eta_bounds[4] = power_bound(6) + T(6) * power_bound(5) +
                    T(6) * power_bound(4)
    eta_bounds[5] = power_bound(8) + T(12) * power_bound(7) +
                    T(36) * power_bound(6) + T(24) * power_bound(5)

    # phi=eta(x)/(eta(x)+eta(1-x)). At least one of x and 1-x is >=1/2,
    # hence the denominator is >=exp(-2). Differentiate phi*denominator=eta
    # recursively to bound phi^(k).
    phi_bounds = Vector{T}(undef, 5)
    phi_bounds[1] = one(T)
    # exp(2) < 8, hence 1/(eta(x)+eta(1-x)) <= exp(2) < 8.
    inv_denominator_bound = T(8)
    for k in 1:4
        rhs_bound = eta_bounds[k + 1]
        for j in 1:k
            denominator_derivative_bound = T(2) * eta_bounds[j + 1]
            rhs_bound += T(binomial(k, j)) * denominator_derivative_bound *
                         phi_bounds[k - j + 1]
        end
        phi_bounds[k + 1] = inv_denominator_bound * rhs_bound
    end

    # w(x)=phi(2(1-|x|)) on its transition region, so each kth derivative
    # gains at most 2^k. The constant regions have zero derivatives.
    bump_bounds = Vector{T}(undef, 5)
    for k in 0:4
        bump_bounds[k + 1] = T(2)^k * phi_bounds[k + 1]
    end

    # Let a(nu)=exp(-(sqrt(1+(beta*nu)^2)+beta*nu)/4). Bounds on the first
    # four derivatives of its exponent give the following derivative bounds
    # for a; also 0<a<=1 on the real line.
    amplitude_coefficients = T[
        one(T),
        one(T) / T(2),
        one(T) / T(2),
        T(5) / T(4),
        T(41) / T(8),
    ]

    # Leibniz rule for h(nu)=a(nu)w(nu/S), followed by
    # ||h''''||_1 <= 2S ||h''''||_infinity on supp(h)=[-S,S]. Evaluate each
    # final positive term in the log domain: forming beta^j and S^(4-j)
    # separately can underflow even when their ratio is representable.
    log_beta = log(f.beta)
    log_support = log(f.S)
    log_terms = Vector{T}(undef, 5)
    for j in 0:4
        log_terms[j + 1] = log(T(2) * T(binomial(4, j)) *
                               amplitude_coefficients[j + 1] *
                               bump_bounds[4 - j + 1]) +
                           T(j) * log_beta + T(j - 3) * log_support
    end
    max_log_term = maximum(log_terms)
    isfinite(max_log_term) || throw(ArgumentError(
        "Metropolis fourth-derivative bound is not representable for this beta and S."))
    scaled_sum = sum(exp(log_term - max_log_term) for log_term in log_terms)
    bound = exp(max_log_term + log(scaled_sum))

    # The analytic constants above have substantial one-sided slack. A factor
    # two additionally absorbs round-to-nearest error in the finite positive
    # arithmetic without changing the asymptotic cutoff scaling.
    bound *= T(2)
    isfinite(bound) && bound > zero(T) || throw(ArgumentError(
        "Metropolis fourth-derivative bound underflowed or overflowed; " *
        "use less extreme beta and S."))
    return bound
end

"""
    _dll_metropolis_time_envelope(filter, t) -> Real

Return the certified envelope `B4 / (2pi |t|^4)` for the Metropolis time
kernel, where `B4` bounds the fourth frequency-derivative `L1` norm.
"""
function _dll_metropolis_time_envelope(
    f::DLLMetropolisFilter{T},
    t::Real,
) where {T<:AbstractFloat}
    abs_t = abs(T(t))
    isnan(abs_t) && throw(ArgumentError("t must not be NaN."))
    iszero(abs_t) && return T(Inf)
    isinf(abs_t) && return zero(T)
    return _dll_metropolis_fourier_d4_l1_bound(f) /
           (T(2) * T(pi) * abs_t^4)
end

"""
    _dll_metropolis_time_tail_bound(filter, cutoff) -> Real

Bound the discarded, quadrature-weighted tail on every centred uniform time
lattice. If the spacing is `tau`, this bounds
`tau * sum_{|m*tau|>cutoff} |f(m*tau)|` independently of `tau`. It also bounds
the smaller continuum-tail estimate obtained from the same envelope.
"""
function _dll_metropolis_time_tail_bound(
    f::DLLMetropolisFilter{T},
    cutoff::Real,
) where {T<:AbstractFloat}
    cutoff_T = T(cutoff)
    isfinite(cutoff_T) && cutoff_T > zero(T) ||
        throw(ArgumentError("cutoff must be finite and > 0."))

    # For n=floor(cutoff/tau)+1 and y=cutoff/tau,
    #   tau * sum_{|m|>=n} B4/(2pi|m*tau|^4)
    #     = B4*y^3/(pi*cutoff^3) * sum_{m=n}^infinity m^-4
    #     <= B4*zeta(4)/(pi*cutoff^3).
    # Here zeta(4)=pi^4/90. The bound is valid for every tau>0 and therefore
    # matches the centred lattice consumed by `_truncate_time_labels_for_oft`.
    zeta_four = T(pi)^4 / T(90)
    roundoff_guard = one(T) + T(64) * eps(T)
    log_tail_bound = log(_dll_metropolis_fourier_d4_l1_bound(f)) +
                     log(zeta_four) - log(T(pi)) - T(3) * log(cutoff_T) +
                     log(roundoff_guard)
    # A positive real bound below the subnormal range must not be represented
    # by zero, which would be a false certificate.
    return max(exp(log_tail_bound), nextfloat(zero(T)))
end

"""
    filter_time_cutoff(filter::DLLMetropolisFilter, tol)

Return a cutoff whose complete quadrature-weighted tail is at most `tol` on
every centred uniform time lattice. The guarantee follows from the
fourth-order integration-by-parts envelope and `zeta(4)=pi^4/90`; it therefore
applies directly to the simulator's trapezoidal time grid without sampling the
oscillatory kernel.
"""
function filter_time_cutoff(f::DLLMetropolisFilter{T}, tol::Real) where {T}
    tol_T = T(tol)
    isfinite(tol_T) && tol_T > zero(T) ||
        throw(ArgumentError("tol must be finite and > 0."))
    derivative_bound = _dll_metropolis_fourier_d4_l1_bound(f)
    zeta_four = T(pi)^4 / T(90)
    roundoff_guard = one(T) + T(64) * eps(T)
    log_tail_coefficient = log(derivative_bound) + log(zeta_four) - log(T(pi)) +
                           log(roundoff_guard)
    cutoff = exp((log_tail_coefficient - log(tol_T)) / T(3))
    isfinite(cutoff) && cutoff > zero(T) || throw(ArgumentError(
        "filter_time_cutoff produced a non-positive or non-finite cutoff; " *
        "check beta, S, and tol."))

    # The cube root can round downward by enough that evaluating the certified
    # bound at `cutoff` lies a few ulps above `tol_T`. Advance to the first
    # representable cutoff that satisfies the inequality in the working type.
    tail_bound = max(
        exp(log_tail_coefficient - T(3) * log(cutoff)),
        nextfloat(zero(T)),
    )
    while tail_bound > tol_T
        cutoff = nextfloat(cutoff)
        isfinite(cutoff) || throw(ArgumentError(
            "filter_time_cutoff overflowed while enforcing its tail bound."))
        tail_bound = max(
            exp(log_tail_coefficient - T(3) * log(cutoff)),
            nextfloat(zero(T)),
        )
    end
    return cutoff
end

# Multi-channel filter types live in `dll_multichannel.jl`.

# User specifications retain typed callbacks and declared provenance. They do
# not claim transform existence or an efficient implementation theorem.
struct _UserFilterMetadata{T<:AbstractFloat,P<:NamedTuple,B}
    name::Symbol
    version::String
    parameters::P
    support::Union{Nothing,T}
    tail_bound::B
    _UserFilterMetadata{T,P,B}(name,version,parameters,support,tail_bound) where {T<:AbstractFloat,P<:NamedTuple,B} =
        new{T,P,B}(name,version,parameters,support,tail_bound)
end

function _user_filter_metadata(beta; name, version="1", parameters::NamedTuple=(;),
                               support=nothing, tail_bound=nothing)
    T = typeof(float(beta))
    isfinite(beta) && beta > 0 || throw(ArgumentError("Filter beta must be finite and positive."))
    name isa Union{Symbol,AbstractString} && !isempty(strip(string(name))) ||
        throw(ArgumentError("Filter name must be a nonempty stable symbol or string."))
    version isa AbstractString && !isempty(strip(version)) ||
        throw(ArgumentError("Filter version must be a nonempty string."))
    support === nothing || (support isa Real && isfinite(support) && support > 0 &&
                            isfinite(T(support)) && T(support) > 0) ||
        throw(ArgumentError("support must be nothing or a positive finite symmetric radius in the input frame."))
    tail_bound === nothing || applicable(tail_bound,one(T)) ||
        throw(ArgumentError("tail_bound must be a callable of a positive cutoff, or nothing."))
    return _UserFilterMetadata{T,typeof(parameters),typeof(tail_bound)}(Symbol(name), String(version), deepcopy(parameters),
        support === nothing ? nothing : T(support), tail_bound)
end

abstract type _UserDLLFilter <: AbstractFilter end

"""
    KMSFilter(beta; q_positive=nothing, logabs_q_positive=nothing,
              phase_positive=x->1, name, version="1", parameters=(;),
              support=nothing, tail_bound=nothing)

Construct `q(-x)=conj(q(x))` from a callback on `x>=0`, then apply the thermal
amplitude `F(nu)=q(nu)*exp(-beta*nu/4)` exactly once. `q(0)` must be real.
Alternatively provide log magnitude and a unit-modulus phase; `-Inf` represents
an exact zero. This avoids losing a small q before combining its thermal factor.
Callbacks use physical frequencies in the facade and algorithm frequencies in
legacy Config calls. `support` is an enforced symmetric frequency radius;
regularity, transform existence and supplied tail bounds remain unverified.
"""
struct KMSFilter{T<:AbstractFloat,F,L,P,M} <: _UserDLLFilter
    beta::T
    q_positive::F
    logabs_q_positive::L
    phase_positive::P
    metadata::M
end
function KMSFilter(beta::Real; q_positive=nothing, logabs_q_positive=nothing,
                   phase_positive=nothing, kwargs...)
    (q_positive === nothing) != (logabs_q_positive === nothing) || throw(ArgumentError(
        "Supply exactly one of q_positive or logabs_q_positive; neither is a thermally weighted amplitude."))
    q_positive === nothing || phase_positive === nothing || throw(ArgumentError(
        "phase_positive belongs to the logabs_q_positive route; q_positive already includes its phase."))
    metadata = _user_filter_metadata(beta; kwargs...)
    T = typeof(float(beta))
    callback = q_positive === nothing ? logabs_q_positive : q_positive
    applicable(callback,zero(T)) || throw(ArgumentError("The q callback must accept nonnegative frequencies."))
    phase = phase_positive === nothing ? (x -> one(x)) : phase_positive
    applicable(phase,zero(T)) || throw(ArgumentError("phase_positive must accept nonnegative frequencies."))
    f = KMSFilter(T(beta),q_positive,logabs_q_positive,phase,metadata)
    _kms_logphase(f,zero(T)) # enforce the origin condition immediately
    return f
end

"""
    FrequencyFilter(beta; amplitude, name, version="1", parameters=(;),
                    support=nothing, tail_bound=nothing)

Expert full frequency amplitude, already including the thermal factor. Bohr
compilation checks weighted conjugate reflection on the complete finite set.
These checks establish no functional identity away from the sampled spectrum.
"""
struct FrequencyFilter{T<:AbstractFloat,F,M} <: _UserDLLFilter
    beta::T
    amplitude::F
    metadata::M
end
function FrequencyFilter(beta::Real; amplitude, kwargs...)
    metadata = _user_filter_metadata(beta; kwargs...)
    applicable(amplitude,zero(float(beta))) || throw(ArgumentError("amplitude must be a frequency callback."))
    f = FrequencyFilter(float(beta),amplitude,metadata)
    freq_kernel(f,zero(float(beta)))
    return f
end

"""
    RateFilter(beta; downward_rate, name, version="1", parameters=(;),
               support=nothing, tail_bound=nothing)

`downward_rate(x)` takes energy release `x>=0` and returns finite nonnegative
`r(-x)`. Choose principal-root amplitudes: `F(-x)=sqrt(r(-x))` and
`F(x)=exp(-beta*x/2)*sqrt(r(-x))`. No phase, rate normalisation or extra
thermal weight is added. Zeros are allowed and do not imply ergodicity.
"""
struct RateFilter{T<:AbstractFloat,F,M} <: _UserDLLFilter
    beta::T
    downward_rate::F
    metadata::M
end
function RateFilter(beta::Real; downward_rate, kwargs...)
    metadata = _user_filter_metadata(beta; kwargs...)
    applicable(downward_rate,zero(float(beta))) || throw(ArgumentError("downward_rate must accept energy release x>=0."))
    f = RateFilter(float(beta),downward_rate,metadata)
    freq_kernel(f,zero(float(beta)))
    return f
end

"""
    TimeFilter(beta; kernel, name, version="1", parameters=(;),
               support=nothing, tail_bound=nothing)

Qualified time-input specification with `F(nu)=integral f(t)*exp(i*nu*t) dt`.
Support and tail declarations use time coordinates. Kernel evaluation is
available; simulation requires the future numerical transform/coherent compiler
(T12–T13) and currently rejects explicitly. No balance projection is performed.
"""
struct TimeFilter{T<:AbstractFloat,F,M} <: _UserDLLFilter
    beta::T
    kernel::F
    metadata::M
end
function TimeFilter(beta::Real; kernel, kwargs...)
    metadata = _user_filter_metadata(beta; kwargs...)
    applicable(kernel,zero(float(beta))) || throw(ArgumentError("kernel must be a time callback."))
    f = TimeFilter(float(beta),kernel,metadata)
    time_kernel(f,zero(float(beta)))
    return f
end

Base.eltype(f::_UserDLLFilter) = Complex{typeof(f.beta)}
_is_admissible_dll_filter(::Union{KMSFilter,RateFilter}) = true
_is_dll_bohr_spec(::AbstractFilter) = false
_is_dll_bohr_spec(::_UserDLLFilter) = true
_dll_time_supported(::AbstractFilter) = true
_dll_time_supported(::_DLLBohrFilter) = false
_dll_time_supported(::_UserDLLFilter) = false

function _user_filter_coordinate(f::_UserDLLFilter, nu::Real)
    x = typeof(f.beta)(nu)
    isfinite(x) || throw(ArgumentError("Filter coordinate must be finite at working precision."))
    return x
end
_filter_outside(f::_UserDLLFilter, x) = f.metadata.support !== nothing && abs(x) > f.metadata.support
function _finite_filter_value(value, f, x)
    value isa Number && isfinite(value) || throw(ArgumentError(
        "$(f.metadata.name) callback must return a finite number at $x."))
    z = eltype(f)(value)
    isfinite(z) || throw(ArgumentError("$(f.metadata.name) amplitude overflows working precision at $x."))
    return z
end

function _kms_logphase(f::KMSFilter{T}, x::T) where {T}
    if f.q_positive !== nothing
        z = _finite_filter_value(f.q_positive(x),f,x)
        iszero(x) && !isreal(z) && throw(ArgumentError("q_positive(0) must be real."))
        iszero(z) && return (T(-Inf),one(Complex{T}))
        scale = max(abs(real(z)),abs(imag(z)))
        unit = z/scale
        magnitude = abs(unit)
        return (log(scale)+log(magnitude), unit/magnitude)
    end
    ell = f.logabs_q_positive(x)
    ell isa Real && (isfinite(ell) || ell == -Inf) || throw(ArgumentError(
        "logabs_q_positive must return a finite real or -Inf for an exact zero."))
    converted = T(ell)
    (isfinite(converted) || ell == -Inf) ||
        throw(ArgumentError("Log magnitude overflows working precision."))
    ell = converted
    phase = _finite_filter_value(f.phase_positive(x),f,x)
    isapprox(abs(phase),one(T);atol=zero(T),rtol=32eps(T)) ||
        throw(ArgumentError("phase_positive must have unit modulus."))
    iszero(x) && ell != -Inf && !isreal(phase) && throw(ArgumentError("q(0) must be real."))
    return (ell,phase)
end
function freq_kernel(f::KMSFilter{T}, nu::Real) where {T}
    x = _user_filter_coordinate(f,nu)
    _filter_outside(f,x) && return zero(Complex{T})
    ell,phase = _kms_logphase(f,abs(x))
    ell == -Inf && return zero(Complex{T})
    logamp = ell - (f.beta/T(4))*x
    amplitude = exp(logamp)
    isfinite(amplitude) || throw(ArgumentError(
        "$(f.metadata.name) thermal amplitude overflows at $x; use logabs_q_positive or revise the filter/precision."))
    return amplitude * (x < 0 ? conj(phase) : phase)
end
function q_weight(f::KMSFilter{T}, nu::Real) where {T}
    x = _user_filter_coordinate(f,nu)
    _filter_outside(f,x) && return zero(Complex{T})
    ell,phase = _kms_logphase(f,abs(x))
    return _finite_filter_value(exp(ell)*(x < 0 ? conj(phase) : phase),f,x)
end
function freq_kernel(f::FrequencyFilter, nu::Real)
    x = _user_filter_coordinate(f,nu)
    _filter_outside(f,x) && return zero(eltype(f))
    return _finite_filter_value(f.amplitude(x),f,x)
end
function freq_kernel(f::RateFilter{T}, nu::Real) where {T}
    x = _user_filter_coordinate(f,nu)
    _filter_outside(f,x) && return zero(Complex{T})
    r = f.downward_rate(abs(x))
    r isa Real && isfinite(r) && r >= 0 || throw(ArgumentError(
        "downward_rate must return a finite nonnegative real rate at energy release $(abs(x))."))
    # Root before thermal multiplication: do not underflow the rate first.
    amplitude = T(sqrt(r))
    isfinite(amplitude) || throw(ArgumentError("Rate amplitude overflows working precision."))
    return x > 0 ? _scale_dll_amplitude(Complex{T}(amplitude),-(f.beta/T(2))*x) : Complex{T}(amplitude)
end
function q_weight(f::RateFilter{T}, nu::Real) where {T}
    x = abs(_user_filter_coordinate(f,nu))
    return _scale_dll_amplitude(freq_kernel(f,-x),-(f.beta/T(4))*x)
end
function time_kernel(f::TimeFilter, t::Real)
    x = _user_filter_coordinate(f,t)
    _filter_outside(f,x) && return zero(eltype(f))
    return _finite_filter_value(f.kernel(x),f,x)
end
freq_kernel(::TimeFilter, ::Real) = throw(ArgumentError(
    "TimeFilter simulation needs the numerical Fourier compiler (T12–T13); supply a frequency specification for BohrDomain."))
time_kernel(::Union{KMSFilter,FrequencyFilter,RateFilter}, ::Real) = throw(ArgumentError(
    "Custom DLL time transforms are not available yet (T12–T13); use BohrDomain."))
filter_time_cutoff(::_UserDLLFilter, ::Real) = throw(ArgumentError(
    "A custom time cutoff cannot be inferred from frequency samples; numerical transforms are T12–T13."))

"""
    filter_evidence(filter)

Separate algebraic balance provenance from transform existence, declared support
and tails, numerical checks, and implementation-theorem applicability. Custom
callbacks remain nonportable without their definitions; a name is not a registry.
"""
function filter_evidence(f::_UserDLLFilter)
    return (; name=f.metadata.name,version=f.metadata.version,
        parameters=deepcopy(f.metadata.parameters),
        algebraic_balance=f isa Union{KMSFilter,RateFilter} ? :conjugate_reflection : :unverified,
        continuum_transform=:unknown,numerical_checks=:not_run,
        implementation_theorem=:not_established,support=f.metadata.support,
        support_coordinate=f isa TimeFilter ? :time : :frequency,
        tail_bound=f.metadata.tail_bound === nothing ? :unknown : :user_supplied_unverified,
        phase_policy=f isa RateFilter ? :principal_root : :user_supplied,
        callback_portability=:requires_callable)
end

function filter_evidence(f::AbstractFilter)
    builtin = f isa Union{DLLGaussianFilter,DLLMetropolisFilter}
    return (;algebraic_balance=builtin ? :builtin_identity : :unverified,
        continuum_transform=builtin ? :builtin_methods : :unknown,numerical_checks=:not_run,
        implementation_theorem=:not_established)
end
filter_evidence(::_DLLBohrFilter) = (;algebraic_balance=:checked_finite_bohr_set,
    continuum_transform=:unknown,numerical_checks=:passed_at_working_precision,
    implementation_theorem=:not_established)
