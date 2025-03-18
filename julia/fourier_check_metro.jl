using QuadGK
using SpecialFunctions: erfc
using FFTW
using Plots
using LinearAlgebra
include("coherent.jl")

function f_plus_t_metro(t, eta, beta)
    if abs(t) < 1e-12  # Handle t=0
        return complex(sqrt(1 / 2pi) / beta) 
    elseif abs(t) ≤ eta
        numerator = exp(-2 * t^2 / beta^2 - 1im * t / beta) + 1im * (2 * t / beta + 1im)
    else
        numerator = exp(-2 * t^2 / beta^2 - 1im * t / beta)
    end
    denominator = t * (2 * t / beta + 1im) / beta
    return (sqrt(1 / 2pi) / beta) * numerator / denominator
end

function b_t_metro(t, eta)
    if abs(t) < 1e-10  # Handle t=0
        return complex((1/(2*sqrt(2π))) )
    elseif abs(t) ≤ eta
        numerator = exp(-2*t^2 - 1im*t) + 1im*(2*t + 1im)
    else
        numerator = exp(-2*t^2 - 1im*t)
    end
    denominator = t * (2*t + 1im)
    return (1/(2*sqrt(2π))) * numerator / denominator
end

f_plus_nu_metro(nu, beta) = (erfc((1 + beta * nu) / sqrt(8)) + exp(-beta * nu / 2) * erfc((1 - beta * nu) / sqrt(8))) / 2

function fourier_of_f_plus_t_metro(nu, eta, beta)
    integrand(t) = exp(-1im * nu * t) * f_plus_t_metro(t, eta, beta) / sqrt(2pi)
    fourier, _ = quadgk(integrand, -Inf, Inf, rtol=1e-6)
    return fourier
end

function fourier_of_b_t_metro(nu, eta)
    integrand(t) = exp(-1im * nu * t) * b_t_metro(t, eta) / sqrt(2pi)
    fourier, _ = quadgk(integrand, -Inf, Inf, rtol=1e-6)
    return fourier
end

beta = 10.
eta = 0.0002

# nu_vals = [-2.0:0.01:2.0;]

# # Analytic
# f_nu_vals = f_plus_nu_metro.(nu_vals, beta)
# # Fourier
# fourier_nu_vals = fourier_of_f_plus_t_metro.(nu_vals, eta, beta)  # Imaginary parts are 1e-3-1e-4
# fourier_nu_vals_real =  real.(fourier_nu_vals)
# fourier_min = minimum(fourier_nu_vals_real)
# fourier_nu_vals_real = fourier_nu_vals_real .+ 0.5  # Dirac delta shift after Fourier

# # Difference
# display(norm(fourier_nu_vals_real - real.(f_nu_vals)))

# # Plots
# plot(nu_vals, fourier_nu_vals_real, label="Fourier")
# plot!(nu_vals, real.(f_nu_vals), label="Analytic")

time_labels = [-90.0:0.01:100.0;]
f_plus = @time compute_f_plus_metro.(time_labels, eta, beta)
f_plus_truncated = @time compute_truncated_f_plus_metro(time_labels, eta, beta)

plot(time_labels, real.(f_plus))
scatter!(collect(keys(f_plus_truncated)), real.(collect(values(f_plus_truncated))))
