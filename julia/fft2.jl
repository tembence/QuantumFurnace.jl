using FFTW
using Plots
using QuadGK


beta = 10.
sigma_E = 1 / beta
num_labels = Int(1e3)
energies = fftshift(LinRange(-100., 100., num_labels))
w0 = energies[2] - energies[1]

F(nu) = (tanh(-beta * nu / 4) / (4 * pi * im)) * exp(-nu^2 / (8 * sigma_E^2))
F_nu_vals = F.(energies)

# First, try the normalized one
@time f_t_vals = fftshift(ifft(F_nu_vals) * num_labels)
f_t_vals /= sqrt(sum(f_t_vals.^2))
times = collect(fftshift(fftfreq(num_labels, w0)))
#plot 
plot(times ./ (minimum(times)) .* 5, real(f_t_vals), label="Real part of FFT", title="Inverse Fourier Transform of f_", xlabel="t", ylabel="Amplitude")

#* Convolution via integral from
function convolute(f::Function, g::Function, t::Float64; atol=1e-12, rtol=1e-12)
    integrand(s) = f(s) * g(t - s)
    result, _ = quadgk(integrand, -Inf, Inf; atol=atol, rtol=rtol)
    return result
end

f_t_1(t) = 1 / cosh(2 * pi * t / beta)
f_t_2(t) = sin(- beta * sigma_E^2 * (t)) * exp(- 2 * sigma_E^2 * (t)^2)

t_values = range(-5, stop=5, length=1000)
# Compute the convolution for each time value
@time conv_results = (sigma_E / (pi * beta)) * exp(beta^2 * sigma_E^2 / 8) * convolute.(Ref(f_t_1), Ref(f_t_2), t_values)
conv_results /= sqrt(sum(conv_results.^2))
# Plot the results
plot!(t_values, real.(conv_results), label="Real part")