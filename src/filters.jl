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
            expected = exp(-(beta / T(2)) * nu) * conj(values[iszero(nu) ? nu : -nu])
            scale = max(abs(value), abs(expected))
            abs(value - expected) <= T(128) * eps(T) * scale || throw(ArgumentError(
                "DLL weighted reflection failed at frequency $nu; supply a balanced amplitude (thermal weight exactly once)."))
        end
        new{T}(beta, copy(values))
    end
end
Base.eltype(::_DLLBohrFilter{T}) where {T} = Complex{T}
_is_admissible_dll_filter(::_DLLBohrFilter) = true
freq_kernel(f::_DLLBohrFilter{T}, nu::Real) where {T} = f.values[T(nu)]
function q_weight(f::_DLLBohrFilter{T}, nu::Real) where {T}
    x = abs(T(nu))
    value = freq_kernel(f, iszero(x) ? zero(T) : -x) * exp(-(f.beta / T(4)) * x)
    return nu < 0 ? value : conj(value)
end

function _prepare_dll_bohr_filter(filter::AbstractFilter, eigvals::AbstractVector{T};
                                  beta=filter.beta) where {T<:AbstractFloat}
    isapprox(filter.beta, beta; atol=zero(T), rtol=10eps(T)) ||
        throw(ArgumentError("DLL filter beta must match construction beta."))
    frequencies = sort!(unique!(T[a-b for a in eigvals for b in eigvals]))
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
