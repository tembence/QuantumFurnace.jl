const _BOHR_CHEBYSHEV_TARGETS = (:gaussian_q, :gaussian_f, :sech_quarter)
const _BOHR_CHEBYSHEV_PARITIES = (:even, :none)
const _BOHR_CHEBYSHEV_BOUND_METHOD = :bernstein_ellipse_interpolant
const _BOHR_BIGFLOAT_LOCK = ReentrantLock()

@inline _gaussian_q_dimensionless(x::Real) = exp(-(x * x) / 8)
@inline _gaussian_f_dimensionless(x::Real) = exp(-(x * x) / 8 - x / 4)

@inline function _sech_quarter_dimensionless(x::Real)
    decay = exp(-abs(x) / 4)
    return 2 * decay / (1 + decay * decay)
end

@inline function _bohr_target_value(target::Symbol, x::Real)
    target === :gaussian_q && return _gaussian_q_dimensionless(x)
    target === :gaussian_f && return _gaussian_f_dimensionless(x)
    target === :sech_quarter && return _sech_quarter_dimensionless(x)
    throw(ArgumentError("unsupported Bohr Chebyshev target $target."))
end

"""
    BohrChebyshevApproximation

A realized Chebyshev interpolant on the certified dimensionless interval
`[-interval_radius, interval_radius]`. Coefficients use the convention
`c[1] / 2 + sum(c[k + 1] * T_k(x / interval_radius), k=1:degree)`.
`uniform_error_bound` includes the Bernstein-ellipse interpolation remainder
and a directed-rounding enclosure of coefficient realization. It is a scalar,
exact-polynomial bound; floating recurrence and matrix-product errors are
separate diagnostics.
"""
struct BohrChebyshevApproximation{T<:AbstractFloat}
    target::Symbol
    interval_radius::T
    _coefficients::Vector{T}
    degree::Int
    bernstein_eta::T
    ellipse_log_supremum::T
    interpolation_error_bound::T
    coefficient_error_bound::T
    uniform_error_bound::T
    requested_tolerance::T
    working_precision::Int
    parity::Symbol
    bound_method::Symbol

    function BohrChebyshevApproximation{T}(
        target::Symbol,
        interval_radius::T,
        coefficients::Vector{T},
        bernstein_eta::T,
        ellipse_log_supremum::T,
        interpolation_error_bound::T,
        coefficient_error_bound::T,
        requested_tolerance::T,
        working_precision::Int,
        parity::Symbol,
    ) where {T<:AbstractFloat}
        target in _BOHR_CHEBYSHEV_TARGETS || throw(ArgumentError(
            "unsupported Bohr Chebyshev target $target."))
        isfinite(interval_radius) && interval_radius >= zero(T) ||
            throw(ArgumentError(
                "interval_radius must be finite and nonnegative."))
        isempty(coefficients) && throw(ArgumentError(
            "Chebyshev coefficients must be nonempty."))
        all(isfinite, coefficients) || throw(ArgumentError(
            "Chebyshev coefficients must be finite."))
        parity in _BOHR_CHEBYSHEV_PARITIES || throw(ArgumentError(
            "unsupported Chebyshev parity $parity."))
        parity === :even && any(!iszero(coefficients[index])
                                for index in 2:2:length(coefficients)) &&
            throw(ArgumentError(
                "an even Chebyshev approximation must have zero odd-order coefficients."))
        isfinite(bernstein_eta) && bernstein_eta >= zero(T) ||
            throw(ArgumentError("bernstein_eta must be finite and nonnegative."))
        isfinite(ellipse_log_supremum) || throw(ArgumentError(
            "ellipse_log_supremum must be finite."))
        for (name, value) in (
            (:interpolation_error_bound, interpolation_error_bound),
            (:coefficient_error_bound, coefficient_error_bound),
            (:requested_tolerance, requested_tolerance),
        )
            isfinite(value) && value >= zero(T) || throw(ArgumentError(
                "$name must be finite and nonnegative."))
        end
        requested_tolerance > zero(T) || throw(ArgumentError(
            "requested_tolerance must be positive."))
        uniform_error_bound = interpolation_error_bound +
                              coefficient_error_bound
        !iszero(uniform_error_bound) &&
            (uniform_error_bound = nextfloat(uniform_error_bound))
        isfinite(uniform_error_bound) || throw(ArgumentError(
            "the realized uniform error bound is not finite."))
        uniform_error_bound <= requested_tolerance || throw(ArgumentError(
            "uniform error bound $uniform_error_bound exceeds requested " *
            "tolerance $requested_tolerance."))
        working_precision >= 128 || throw(ArgumentError(
            "working_precision must be at least 128 bits."))
        if iszero(interval_radius)
            length(coefficients) == 1 || throw(ArgumentError(
                "a zero-width interval must use a degree-zero polynomial."))
            iszero(bernstein_eta) || throw(ArgumentError(
                "a zero-width interval must use bernstein_eta=0."))
        end
        return new{T}(
            target, interval_radius, copy(coefficients), length(coefficients) - 1,
            bernstein_eta, ellipse_log_supremum,
            interpolation_error_bound, coefficient_error_bound,
            uniform_error_bound, requested_tolerance, working_precision,
            parity, _BOHR_CHEBYSHEV_BOUND_METHOD)
    end
end

@inline _bohr_chebyshev_coefficients(data::BohrChebyshevApproximation) =
    getfield(data, :_coefficients)

function Base.getproperty(data::BohrChebyshevApproximation, name::Symbol)
    name === :coefficients && return copy(_bohr_chebyshev_coefficients(data))
    return getfield(data, name)
end

function Base.propertynames(::BohrChebyshevApproximation, private::Bool=false)
    public_names = (
        :target, :interval_radius, :coefficients, :degree, :bernstein_eta,
        :ellipse_log_supremum, :interpolation_error_bound,
        :coefficient_error_bound, :uniform_error_bound,
        :requested_tolerance, :working_precision, :parity, :bound_method,
    )
    return private ? (public_names..., :_coefficients) : public_names
end

"""
    DLLGaussianChebyshevData

Coupled scalar approximation data for the Gaussian DLL functions `q`,
`f_hat`, and `sech`. `modular_compatibility_bound` certifies
`sup_x |p_f(x) - exp(-x/4) p_q(x)|` on the stored interval.
"""
struct DLLGaussianChebyshevData{T<:AbstractFloat}
    interval_radius::T
    q::BohrChebyshevApproximation{T}
    f::BohrChebyshevApproximation{T}
    sech::BohrChebyshevApproximation{T}
    modular_compatibility_bound::T
    requested_tolerance::T
    interval_provenance::Symbol

    function DLLGaussianChebyshevData{T}(
        interval_radius::T,
        q::BohrChebyshevApproximation{T},
        f::BohrChebyshevApproximation{T},
        sech::BohrChebyshevApproximation{T},
        modular_compatibility_bound::T,
        requested_tolerance::T,
        interval_provenance::Symbol,
    ) where {T<:AbstractFloat}
        q.target === :gaussian_q || throw(ArgumentError(
            "q must target :gaussian_q."))
        f.target === :gaussian_f || throw(ArgumentError(
            "f must target :gaussian_f."))
        sech.target === :sech_quarter || throw(ArgumentError(
            "sech must target :sech_quarter."))
        all(item -> item.interval_radius == interval_radius, (q, f, sech)) ||
            throw(ArgumentError(
                "all Gaussian DLL polynomials must use the same interval."))
        isfinite(modular_compatibility_bound) &&
            modular_compatibility_bound >= zero(T) || throw(ArgumentError(
                "modular_compatibility_bound must be finite and nonnegative."))
        modular_compatibility_bound <= requested_tolerance ||
            throw(ArgumentError(
                "modular compatibility bound $modular_compatibility_bound " *
                "exceeds requested tolerance $requested_tolerance."))
        return new{T}(
            interval_radius, q, f, sech, modular_compatibility_bound,
            requested_tolerance, interval_provenance)
    end
end

@inline function _bohr_chebyshev_float_type(
    interval_radius::Real,
    tolerance::Real,
)
    T = promote_type(
        typeof(float(interval_radius)), typeof(float(tolerance)))
    T <: AbstractFloat || throw(ArgumentError(
        "Chebyshev data must promote to an AbstractFloat type, got $T."))
    return T
end

function _outward_upper(::Type{T}, value::BigFloat) where {T<:AbstractFloat}
    value >= 0 || throw(ArgumentError("an upper bound cannot be negative."))
    converted = T(value)
    isfinite(converted) || throw(ArgumentError(
        "a certified bound is not representable in $T."))
    if BigFloat(converted) < value
        converted = nextfloat(converted)
    end
    return converted
end

@inline function _bernstein_log_supremum(
    target::Symbol,
    radius::BigFloat,
    eta::BigFloat,
)
    imaginary_radius = sinh(eta)
    target === :gaussian_q &&
        return radius^2 * imaginary_radius^2 / 8
    target === :gaussian_f &&
        return (1 + radius^2 * imaginary_radius^2) / 8
    if target === :sech_quarter
        angle = radius * imaginary_radius / 4
        angle < BigFloat(pi) / 2 || return BigFloat(Inf)
        return -log(cos(angle))
    end
    throw(ArgumentError("unsupported Bohr Chebyshev target $target."))
end

@inline function _bernstein_interpolant_log_bound(
    target::Symbol,
    radius::BigFloat,
    degree::Int,
    eta::BigFloat,
)
    log_supremum = _bernstein_log_supremum(target, radius, eta)
    isfinite(log_supremum) || return BigFloat(Inf), log_supremum
    log_bound = log(BigFloat(4)) + log_supremum - degree * eta -
                log(expm1(eta))
    return log_bound, log_supremum
end

function _bernstein_candidates(
    target::Symbol,
    radius::BigFloat,
    degree::Int,
)
    candidates = BigFloat[]
    if target === :sech_quarter
        pole_radius = 2 * BigFloat(pi) / radius
        for index in 1:256
            fraction = BigFloat(index) / BigFloat(257)
            push!(candidates, asinh(pole_radius * fraction))
        end
        for exponent in 1:40
            edge = exp2(-BigFloat(exponent))
            push!(candidates, asinh(pole_radius * edge))
            push!(candidates, asinh(pole_radius * (1 - edge)))
        end
    else
        center = 2 * sqrt(BigFloat(degree + 1)) / radius
        for index in 0:256
            exponent = -8 + BigFloat(index) / 16
            push!(candidates, asinh(center * exp(exponent)))
        end
    end
    return candidates
end

function _best_bernstein_certificate(
    target::Symbol,
    radius::BigFloat,
    degree::Int,
)
    best_log_bound = BigFloat(Inf)
    best_eta = BigFloat(0)
    best_log_supremum = BigFloat(Inf)
    for eta in _bernstein_candidates(target, radius, degree)
        eta > 0 || continue
        log_bound, log_supremum = _bernstein_interpolant_log_bound(
            target, radius, degree, eta)
        if log_bound < best_log_bound
            best_log_bound = log_bound
            best_eta = eta
            best_log_supremum = log_supremum
        end
    end
    isfinite(best_log_bound) || throw(ArgumentError(
        "no admissible Bernstein ellipse was found for target=$target on " *
        "[-$radius, $radius]."))
    return best_log_bound, best_eta, best_log_supremum
end

function _bernstein_interpolant_bound_upper(
    target::Symbol,
    radius::BigFloat,
    degree::Int,
    eta::BigFloat,
)
    imaginary_radius = setrounding(BigFloat, RoundUp) do
        sinh(eta)
    end
    log_supremum = if target === :gaussian_q
        setrounding(BigFloat, RoundUp) do
            radius^2 * imaginary_radius^2 / 8
        end
    elseif target === :gaussian_f
        setrounding(BigFloat, RoundUp) do
            (1 + radius^2 * imaginary_radius^2) / 8
        end
    elseif target === :sech_quarter
        angle = setrounding(BigFloat, RoundUp) do
            radius * imaginary_radius / 4
        end
        pi_half_lower = setrounding(BigFloat, RoundDown) do
            BigFloat(pi) / 2
        end
        angle < pi_half_lower || return BigFloat(Inf), BigFloat(Inf)
        cosine_lower = setrounding(BigFloat, RoundDown) do
            cos(angle)
        end
        supremum = setrounding(BigFloat, RoundUp) do
            inv(cosine_lower)
        end
        setrounding(BigFloat, RoundUp) do
            log(supremum)
        end
    else
        throw(ArgumentError("unsupported Bohr Chebyshev target $target."))
    end
    supremum = setrounding(BigFloat, RoundUp) do
        exp(log_supremum)
    end
    degree_eta_lower = setrounding(BigFloat, RoundDown) do
        degree * eta
    end
    decay = setrounding(BigFloat, RoundUp) do
        exp(-degree_eta_lower)
    end
    denominator = setrounding(BigFloat, RoundDown) do
        expm1(eta)
    end
    bound = setrounding(BigFloat, RoundUp) do
        4 * supremum * decay / denominator
    end
    return bound, log_supremum
end

struct _BigFloatInterval
    lower::BigFloat
    upper::BigFloat

    function _BigFloatInterval(lower::BigFloat, upper::BigFloat)
        lower <= upper || throw(ArgumentError(
            "invalid BigFloat interval [$lower, $upper]."))
        return new(lower, upper)
    end
end

@inline _point_interval(value::BigFloat) = _BigFloatInterval(value, value)

function _add_interval(left::_BigFloatInterval, right::_BigFloatInterval)
    lower = setrounding(BigFloat, RoundDown) do
        left.lower + right.lower
    end
    upper = setrounding(BigFloat, RoundUp) do
        left.upper + right.upper
    end
    return _BigFloatInterval(lower, upper)
end

function _multiply_interval(left::_BigFloatInterval, right::_BigFloatInterval)
    lower_products = BigFloat[
        setrounding(BigFloat, RoundDown) do
            a * b
        end
        for a in (left.lower, left.upper), b in (right.lower, right.upper)
    ]
    upper_products = BigFloat[
        setrounding(BigFloat, RoundUp) do
            a * b
        end
        for a in (left.lower, left.upper), b in (right.lower, right.upper)
    ]
    return _BigFloatInterval(minimum(lower_products), maximum(upper_products))
end

function _square_interval(interval::_BigFloatInterval)
    maximum_absolute = max(abs(interval.lower), abs(interval.upper))
    upper = setrounding(BigFloat, RoundUp) do
        maximum_absolute^2
    end
    lower = if interval.lower <= 0 <= interval.upper
        BigFloat(0)
    else
        minimum(setrounding(BigFloat, RoundDown) do
            endpoint^2
        end for endpoint in (interval.lower, interval.upper))
    end
    return _BigFloatInterval(lower, upper)
end

function _cos_interval(
    angle::_BigFloatInterval,
    pi_interval::_BigFloatInterval,
)
    lower = min(
        setrounding(BigFloat, RoundDown) do
            cos(angle.lower)
        end,
        setrounding(BigFloat, RoundDown) do
            cos(angle.upper)
        end,
    )
    upper = max(
        setrounding(BigFloat, RoundUp) do
            cos(angle.lower)
        end,
        setrounding(BigFloat, RoundUp) do
            cos(angle.upper)
        end,
    )
    lower_ratio = setrounding(BigFloat, RoundDown) do
        angle.lower / pi_interval.upper
    end
    upper_ratio = setrounding(BigFloat, RoundUp) do
        angle.upper / pi_interval.lower
    end
    first_multiple = max(0, floor(Int, lower_ratio) - 1)
    last_multiple = ceil(Int, upper_ratio) + 1
    for multiple in first_multiple:last_multiple
        multiple_interval = _multiply_interval(
            _point_interval(BigFloat(multiple)), pi_interval)
        overlaps = multiple_interval.lower <= angle.upper &&
                   multiple_interval.upper >= angle.lower
        overlaps || continue
        if iseven(multiple)
            upper = one(BigFloat)
        else
            lower = -one(BigFloat)
        end
    end
    return _BigFloatInterval(lower, upper)
end

function _target_interval(target::Symbol, x::_BigFloatInterval)
    if target in (:gaussian_q, :gaussian_f)
        shifted = if target === :gaussian_f
            _add_interval(x, _point_interval(one(BigFloat)))
        else
            x
        end
        square = _square_interval(shifted)
        exponent = if target === :gaussian_f
            _BigFloatInterval(
                setrounding(BigFloat, RoundDown) do
                    BigFloat(1) / 8 - square.upper / 8
                end,
                setrounding(BigFloat, RoundUp) do
                    BigFloat(1) / 8 - square.lower / 8
                end,
            )
        else
            _BigFloatInterval(
                setrounding(BigFloat, RoundDown) do
                    -square.upper / 8
                end,
                setrounding(BigFloat, RoundUp) do
                    -square.lower / 8
                end,
            )
        end
        return _BigFloatInterval(
            setrounding(BigFloat, RoundDown) do
                exp(exponent.lower)
            end,
            setrounding(BigFloat, RoundUp) do
                exp(exponent.upper)
            end,
        )
    elseif target === :sech_quarter
        minimum_absolute = x.lower <= 0 <= x.upper ? BigFloat(0) :
            min(abs(x.lower), abs(x.upper))
        maximum_absolute = max(abs(x.lower), abs(x.upper))
        lower_cosh = setrounding(BigFloat, RoundUp) do
            cosh(maximum_absolute / 4)
        end
        upper_cosh = setrounding(BigFloat, RoundDown) do
            cosh(minimum_absolute / 4)
        end
        return _BigFloatInterval(
            setrounding(BigFloat, RoundDown) do
                inv(lower_cosh)
            end,
            setrounding(BigFloat, RoundUp) do
                inv(upper_cosh)
            end,
        )
    end
    throw(ArgumentError("unsupported Bohr Chebyshev target $target."))
end

function _chebyshev_interpolant_coefficient_intervals(
    target::Symbol,
    radius::BigFloat,
    degree::Int,
    parity::Symbol,
)
    sample_count = degree + 1
    pi_interval = _BigFloatInterval(
        setrounding(BigFloat, RoundDown) do
            BigFloat(pi)
        end,
        setrounding(BigFloat, RoundUp) do
            BigFloat(pi)
        end,
    )
    angles = Vector{_BigFloatInterval}(undef, sample_count)
    values = Vector{_BigFloatInterval}(undef, sample_count)
    denominator = BigFloat(2 * sample_count)
    for sample in 1:sample_count
        numerator = BigFloat(2 * sample - 1)
        angle = _BigFloatInterval(
            setrounding(BigFloat, RoundDown) do
                numerator * pi_interval.lower / denominator
            end,
            setrounding(BigFloat, RoundUp) do
                numerator * pi_interval.upper / denominator
            end,
        )
        angles[sample] = angle
        node = _BigFloatInterval(
            setrounding(BigFloat, RoundDown) do
                radius * cos(angle.upper)
            end,
            setrounding(BigFloat, RoundUp) do
                radius * cos(angle.lower)
            end,
        )
        values[sample] = _target_interval(target, node)
    end

    factor = _BigFloatInterval(
        setrounding(BigFloat, RoundDown) do
            BigFloat(2) / BigFloat(sample_count)
        end,
        setrounding(BigFloat, RoundUp) do
            BigFloat(2) / BigFloat(sample_count)
        end,
    )
    coefficients = Vector{_BigFloatInterval}(undef, degree + 1)
    for order in 0:degree
        if parity === :even && isodd(order)
            coefficients[order + 1] = _point_interval(BigFloat(0))
            continue
        end
        accumulator = _point_interval(BigFloat(0))
        for sample in 1:sample_count
            scaled_angle = _multiply_interval(
                _point_interval(BigFloat(order)), angles[sample])
            cosine = _cos_interval(scaled_angle, pi_interval)
            accumulator = _add_interval(
                accumulator, _multiply_interval(values[sample], cosine))
        end
        coefficients[order + 1] = _multiply_interval(factor, accumulator)
    end
    return coefficients
end

function _realize_coefficient_intervals(
    ::Type{T},
    intervals::Vector{_BigFloatInterval},
) where {T<:AbstractFloat}
    converted = Vector{T}(undef, length(intervals))
    for index in eachindex(intervals)
        midpoint = (intervals[index].lower + intervals[index].upper) / 2
        converted[index] = T(midpoint)
    end
    zero_value = converted[1] / 2
    for order in 2:2:(length(converted) - 1)
        zero_value += (isodd(order ÷ 2) ? -one(T) : one(T)) *
                      converted[order + 1]
    end
    converted[1] += 2 * (one(T) - zero_value)
    difference = BigFloat(0)
    for index in eachindex(converted)
        realized = BigFloat(converted[index])
        lower_distance = setrounding(BigFloat, RoundUp) do
            realized >= intervals[index].lower ?
                realized - intervals[index].lower :
                intervals[index].lower - realized
        end
        upper_distance = setrounding(BigFloat, RoundUp) do
            realized >= intervals[index].upper ?
                realized - intervals[index].upper :
                intervals[index].upper - realized
        end
        distance = max(lower_distance, upper_distance)
        weight = index == 1 ? BigFloat(1) / 2 : one(BigFloat)
        difference = setrounding(BigFloat, RoundUp) do
            difference + weight * distance
        end
    end
    return converted, difference
end

function _build_bohr_chebyshev_approximation(
    ::Type{T},
    target::Symbol,
    radius::BigFloat,
    tolerance::BigFloat,
    max_degree::Int,
    working_precision::Int,
) where {T<:AbstractFloat}
    parity = target in (:gaussian_q, :sech_quarter) ? :even : :none
    if iszero(radius)
        coefficient = T(2 * _bohr_target_value(target, BigFloat(0)))
        return BohrChebyshevApproximation{T}(
            target, zero(T), T[coefficient], zero(T), zero(T), zero(T),
            zero(T), _outward_upper(T, tolerance), working_precision, parity)
    end

    log_tolerance = log(tolerance)
    for degree in 0:max_degree
        log_bound, eta, log_supremum = _best_bernstein_certificate(
            target, radius, degree)
        log_bound <= log_tolerance || continue
        interpolation_bound, log_supremum_upper =
            _bernstein_interpolant_bound_upper(
                target, radius, degree, eta)
        interpolation_bound <= tolerance || continue
        coefficient_intervals = _chebyshev_interpolant_coefficient_intervals(
            target, radius, degree, parity)
        coefficients, coefficient_bound = _realize_coefficient_intervals(
            T, coefficient_intervals)
        total_bound = setrounding(BigFloat, RoundUp) do
            interpolation_bound + coefficient_bound
        end
        total_bound <= tolerance || continue
        return BohrChebyshevApproximation{T}(
            target, T(radius), coefficients,
            T(eta), _outward_upper(T, log_supremum_upper),
            _outward_upper(T, interpolation_bound),
            _outward_upper(T, coefficient_bound),
            _outward_upper(T, tolerance),
            working_precision, parity)
    end
    throw(ArgumentError(
        "no degree <= $max_degree certifies target=$target on " *
        "[-$radius, $radius] at tolerance $tolerance in $T."))
end

function _modular_compatibility_bound(
    ::Type{T},
    radius::BigFloat,
    q_error::T,
    f_error::T,
) where {T<:AbstractFloat}
    q_term = if iszero(q_error)
        BigFloat(0)
    else
        exponential = setrounding(BigFloat, RoundUp) do
            exp(radius / 4)
        end
        setrounding(BigFloat, RoundUp) do
            exponential * BigFloat(q_error)
        end
    end
    bound = setrounding(BigFloat, RoundUp) do
        BigFloat(f_error) + q_term
    end
    return _outward_upper(T, bound)
end

"""
    dll_gaussian_chebyshev_data(interval_radius; ...)

Construct independently certified Chebyshev interpolants for the dimensionless
Gaussian DLL functions on `[-interval_radius, interval_radius]`. Degree
selection uses Bernstein-ellipse uniform bounds, not sampled-grid errors. The
`q` tolerance is tightened so the returned modular-compatibility bound also
meets `tolerance`, or construction fails closed.
"""
function dll_gaussian_chebyshev_data(
    interval_radius::Real;
    tolerance::Real=1e-10,
    max_degree::Integer=4096,
    working_precision::Integer=256,
    interval_provenance::Symbol=:user_supplied,
)
    T = _bohr_chebyshev_float_type(interval_radius, tolerance)
    radius_T = T(interval_radius)
    tolerance_T = T(tolerance)
    isfinite(radius_T) && radius_T >= zero(T) || throw(ArgumentError(
        "interval_radius must be finite and nonnegative."))
    isfinite(tolerance_T) && tolerance_T > zero(T) || throw(ArgumentError(
        "tolerance must be finite and positive."))
    max_degree >= 0 || throw(ArgumentError("max_degree must be nonnegative."))
    working_precision >= 128 || throw(ArgumentError(
        "working_precision must be at least 128 bits."))

    return lock(_BOHR_BIGFLOAT_LOCK) do
        setprecision(BigFloat, Int(working_precision)) do
            radius = BigFloat(radius_T)
            requested = BigFloat(tolerance_T)
            # These allocations leave a factor-two margin in the final modular
            # triangle bound E_f + exp(a/4) E_q.
            f_tolerance = requested / 4
            q_tolerance = requested * exp(-radius / 4) / 4
            sech_tolerance = requested / 2
            minimum_positive = BigFloat(nextfloat(zero(T)))
            q_tolerance >= minimum_positive || throw(ArgumentError(
                "the requested modular tolerance is below the positive range " *
                "of $T on interval radius $radius."))
            q = _build_bohr_chebyshev_approximation(
                T, :gaussian_q, radius, q_tolerance, Int(max_degree),
                Int(working_precision))
            f = _build_bohr_chebyshev_approximation(
                T, :gaussian_f, radius, f_tolerance, Int(max_degree),
                Int(working_precision))
            sech = _build_bohr_chebyshev_approximation(
                T, :sech_quarter, radius, sech_tolerance, Int(max_degree),
                Int(working_precision))
            modular_bound = _modular_compatibility_bound(
                T, radius, q.uniform_error_bound, f.uniform_error_bound)
            return DLLGaussianChebyshevData{T}(
                radius_T, q, f, sech, modular_bound, tolerance_T,
                interval_provenance)
        end
    end
end

function dll_gaussian_chebyshev_data(
    hamiltonian::LocalHamiltonian1D,
    filter::DLLGaussianFilter;
    beta_phys::Union{Nothing, Real}=nothing,
    kwargs...,
)
    _validate_local_hamiltonian_integrity(hamiltonian)
    _require_admissible_dll_filter(filter)
    _frame_beta_metadata(hamiltonian, filter; beta_phys)
    T = promote_type(
        typeof(hamiltonian.spectral_width_bound), typeof(filter.beta))
    radius = T(filter.beta) * T(hamiltonian.spectral_width_bound)
    !iszero(radius) && (radius = nextfloat(radius))
    return dll_gaussian_chebyshev_data(
        radius;
        interval_provenance=hamiltonian.spectral_bound_provenance,
        kwargs...,
    )
end

function _frame_beta_metadata(
    hamiltonian::LocalHamiltonian1D,
    filter::DLLGaussianFilter;
    beta_phys::Union{Nothing, Real},
)
    T = promote_type(typeof(hamiltonian.rescaling_factor), typeof(filter.beta),
                     beta_phys === nothing ? typeof(filter.beta) :
                     typeof(float(beta_phys)))
    beta_frame = T(filter.beta)
    R = T(hamiltonian.rescaling_factor)
    if hamiltonian.coordinate_frame === :physical
        physical = beta_phys === nothing ? beta_frame : T(beta_phys)
        isfinite(physical) && physical > zero(T) || throw(ArgumentError(
            "beta_phys must be finite and positive."))
        isapprox(beta_frame, physical; atol=zero(T), rtol=10eps(T)) ||
            throw(ArgumentError(
                "physical-frame filter beta=$beta_frame must equal " *
                "beta_phys=$physical."))
        return (; beta_frame, beta_phys=physical, beta_alg=R * physical)
    end
    beta_phys === nothing && throw(ArgumentError(
        "algorithm-frame Bohr data require explicit beta_phys so " *
        "filter.beta=R*beta_phys can be validated."))
    physical = T(beta_phys)
    isfinite(physical) && physical > zero(T) || throw(ArgumentError(
        "beta_phys must be finite and positive."))
    algorithm = R * physical
    isapprox(beta_frame, algorithm; atol=zero(T), rtol=10eps(T)) ||
        throw(ArgumentError(
            "algorithm-frame filter beta=$beta_frame must equal " *
            "R*beta_phys=$algorithm."))
    return (; beta_frame, beta_phys=physical, beta_alg=algorithm)
end

"""Evaluate one realized Bohr Chebyshev interpolant inside its interval."""
function evaluate_bohr_chebyshev(
    data::BohrChebyshevApproximation,
    x::Real,
)
    T = promote_type(typeof(x), eltype(_bohr_chebyshev_coefficients(data)))
    x_T = T(x)
    radius = T(data.interval_radius)
    abs(x_T) <= radius || throw(DomainError(
        x, "Chebyshev evaluation lies outside the certified interval."))
    coefficients = _bohr_chebyshev_coefficients(data)
    iszero(radius) && return T(coefficients[1]) / 2
    scaled_x = x_T / radius
    b_next = zero(T)
    b_next_next = zero(T)
    for index in length(coefficients):-1:2
        b_current = 2 * scaled_x * b_next - b_next_next + T(coefficients[index])
        b_next_next = b_next
        b_next = b_current
    end
    return scaled_x * b_next - b_next_next + T(coefficients[1]) / 2
end

function _apply_chebyshev_family(
    action!::F,
    source::AbstractMatrix{S},
    approximations::Tuple{
        BohrChebyshevApproximation{T},
        Vararg{BohrChebyshevApproximation{T}},
    },
) where {F, S, T<:AbstractFloat}
    radius = first(approximations).interval_radius
    all(item -> item.interval_radius == radius, approximations) ||
        throw(ArgumentError(
            "all Chebyshev approximations must use the same interval."))
    CT = promote_type(S, T)
    base = Matrix{CT}(source)
    outputs = map(approximations) do approximation
        result = similar(base)
        coefficient = CT(_bohr_chebyshev_coefficients(approximation)[1]) / 2
        @. result = coefficient * base
        return result
    end
    maximum_degree = 0
    for approximation in approximations
        maximum_degree = max(maximum_degree, approximation.degree)
    end
    maximum_degree == 0 && return outputs

    previous = base
    current = similar(previous)
    next = similar(previous)
    action!(current, previous)
    for (result, approximation) in zip(outputs, approximations)
        approximation.degree >= 1 || continue
        coefficient = CT(_bohr_chebyshev_coefficients(approximation)[2])
        @. result += coefficient * current
    end
    for order in 2:maximum_degree
        action!(next, current)
        @. next = 2 * next - previous
        for (result, approximation) in zip(outputs, approximations)
            approximation.degree >= order || continue
            coefficient = CT(
                _bohr_chebyshev_coefficients(approximation)[order + 1])
            @. result += coefficient * next
        end
        previous, current, next = current, next, previous
    end
    return outputs
end

"""
    apply_bohr_chebyshev(action!, source, approximation)

Apply a Chebyshev polynomial through a caller-supplied linear action
`action!(destination, input)`. The action must represent the scaled operator
whose polynomial is intended. This routine only evaluates the stored
polynomial: a scalar uniform bound transfers directly to the action only when
the action is normal in the relevant norm (as the Hermitian-Hamiltonian
commutator is in Hilbert--Schmidt geometry).
"""
function apply_bohr_chebyshev(
    action!::F,
    source::AbstractMatrix,
    approximation::BohrChebyshevApproximation,
) where {F}
    return first(_apply_chebyshev_family(
        action!, source, (approximation,)))
end

function _dense_scaled_commutator(
    hamiltonian::AbstractMatrix,
    beta::Real,
    interval_radius::Real,
    ::Type{CT},
) where {CT<:Number}
    H = Matrix{CT}(hamiltonian)
    right = similar(H)
    scale = CT(beta / interval_radius)
    function action!(destination::AbstractMatrix, input::AbstractMatrix)
        mul!(destination, H, input)
        mul!(right, input, H)
        @. destination = scale * (destination - right)
        return destination
    end
    return action!
end

function _dense_gershgorin_bohr_radius_upper(
    hamiltonian::AbstractMatrix,
    beta::Real,
)::BigFloat
    return lock(_BOHR_BIGFLOAT_LOCK) do
        setprecision(BigFloat, 256) do
            lower = BigFloat(Inf)
            upper = BigFloat(-Inf)
            for row in axes(hamiltonian, 1)
                radius = BigFloat(0)
                for column in axes(hamiltonian, 2)
                    row == column && continue
                    entry = hamiltonian[row, column]
                    real_part = BigFloat(real(entry))
                    imaginary_part = BigFloat(imag(entry))
                    squared_magnitude = setrounding(BigFloat, RoundUp) do
                        real_part * real_part + imaginary_part * imaginary_part
                    end
                    magnitude = setrounding(BigFloat, RoundUp) do
                        sqrt(squared_magnitude)
                    end
                    radius = setrounding(BigFloat, RoundUp) do
                        radius + magnitude
                    end
                end
                center = BigFloat(real(hamiltonian[row, row]))
                row_lower = setrounding(BigFloat, RoundDown) do
                    center - radius
                end
                row_upper = setrounding(BigFloat, RoundUp) do
                    center + radius
                end
                lower = min(lower, row_lower)
                upper = max(upper, row_upper)
            end
            width = setrounding(BigFloat, RoundUp) do
                upper - lower
            end
            beta_upper = setrounding(BigFloat, RoundUp) do
                BigFloat(beta)
            end
            return setrounding(BigFloat, RoundUp) do
                beta_upper * width
            end
        end
    end
end

function _stored_bohr_interval_encloses(
    interval_radius::AbstractFloat,
    certified_radius::BigFloat,
)::Bool
    return lock(_BOHR_BIGFLOAT_LOCK) do
        setprecision(BigFloat, 256) do
            certified_radius <= BigFloat(interval_radius)
        end
    end
end

"""
    apply_dll_gaussian_chebyshev(hamiltonian, source, beta, data)

Apply the coupled Gaussian DLL polynomials to a dense source using
`(beta / interval_radius) * (H * X - X * H)`. `Q` and `L` share one rotating
Chebyshev basis. The returned `N` includes the canonical minus sign exactly
once after filtering `L' * L` with the `sech` polynomial. The supplied scalar
interval is validated against an outward-rounded Gershgorin enclosure of the
dense Hermitian spectrum; ordinary floating-point eigenvalues are not treated
as a certificate.
"""
function apply_dll_gaussian_chebyshev(
    hamiltonian::AbstractMatrix,
    source::AbstractMatrix,
    beta::Real,
    data::DLLGaussianChebyshevData,
)
    dimension = LinearAlgebra.checksquare(hamiltonian)
    dimension > 0 || throw(ArgumentError(
        "the dense Bohr commutator requires a nonempty Hamiltonian."))
    size(source) == (dimension, dimension) || throw(DimensionMismatch(
        "source and Hamiltonian dimensions must match."))
    ishermitian(hamiltonian) || throw(ArgumentError(
        "the dense Bohr commutator requires a Hermitian Hamiltonian."))
    isfinite(beta) && beta > 0 || throw(ArgumentError(
        "beta must be finite and positive."))
    CT = promote_type(eltype(hamiltonian), eltype(source),
                      typeof(complex(data.interval_radius)))
    RT = typeof(real(zero(CT)))
    RT in (Float32, Float64) || throw(ArgumentError(
        "apply_dll_gaussian_chebyshev supports Float32/Float64 outward-" *
        "certified dense interval validation; use apply_bohr_chebyshev with " *
        "an explicitly certified generic action for $RT data."))
    H = Matrix{CT}(hamiltonian)
    A = Matrix{CT}(source)
    all(isfinite, H) || throw(ArgumentError(
        "hamiltonian entries must be finite."))
    all(isfinite, A) || throw(ArgumentError(
        "source entries must be finite."))
    certified_radius = _dense_gershgorin_bohr_radius_upper(H, beta)
    _stored_bohr_interval_encloses(
        data.interval_radius, certified_radius) ||
        throw(ArgumentError(
            "the supplied interval radius $(data.interval_radius) does not " *
            "enclose the outward-certified dense Gershgorin Bohr radius " *
            "$certified_radius."))

    if iszero(data.interval_radius)
        Q = copy(A)
        L = copy(A)
        N = -(adjoint(L) * L)
        return (; Q, L, N)
    end
    action! = _dense_scaled_commutator(
        H, beta, data.interval_radius, CT)
    Q, L = _apply_chebyshev_family(action!, A, (data.q, data.f))
    rate = adjoint(L) * L
    N = -apply_bohr_chebyshev(action!, rate, data.sech)
    return (; Q, L, N)
end
