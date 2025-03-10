using Random
using LinearAlgebra
using Printf
using Plots
using QuadGK 
using FFTW
using SpecialFunctions: erfc

function convolute(f::Function, g::Function, t::Float64, args...; atol=1e-12, rtol=1e-12)
    integrand(s) = f(s, args...) * g(t - s, args...)
    result, _ = quadgk(integrand, -Inf, Inf; atol=atol, rtol=rtol)
    return result
end
# (f, g, t; f_args=[], g_args=[], lower=-Inf, upper=Inf) if the arguments were different.

function truncated(x::Vector{Number}, cut::Float64)
    return x[abs.(x) .<= cut]
end

# ---------------------------------------------------------------------------------------------------------------------------
#* Energy functions
f_plus_nu_gauss(nu, beta) = exp(-beta^2 * (nu + 2/beta)^2 / 16) / sqrt(2)
f_plus_nu_metro(nu, beta) = (erfc((1 + beta * nu) / sqrt(8)) + exp(-beta * nu / 2) * erfc((1 - beta * nu) / sqrt(8))) / 2

f_minus_nu(nu, beta) = tanh(-beta * nu / 4) * exp(-beta^2 * nu^2 / 8) / 2im / (2pi)

#* Time functions
f_plus_t_gauss(t, beta) = 2 * exp(-4 * t^2 / beta^2 - 2im * t / beta) / beta
f_plus_t_metro(t, eta, beta) = beta * sqrt(pi / 2) * (beta * exp(-2 * t^2 / beta^2 - 1im * t / beta) / (t * (2 * t / beta + 1im))
                                                + (abs(t) <= eta) * 1im * beta / t)

f1(t, beta) = 1 / cosh(2 * pi * t / beta)
f2(t, beta) = sin(-t / beta) * exp(-2 * t^2 / beta^2)
f_minus_t(t, beta) = exp(1/8) * convolute(f1, f2, t, beta) / (beta^2 * pi)

# ---------------------------------------------------------------------------------------------------------------------------
#* Fouriers f+
beta = 10.
eta = 0.02
num_estimating_bits = 10
N = 2^num_estimating_bits
t0 = 0.01
w0 = 2 * pi / (N * t0)
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
# N_labels_decimal_order = fftshift(N_labels)
time_labels = t0 * N_labels
energy_labels = w0 * N_labels

# Analytic results
# f_plus_gauss_nu_vals = f_plus_nu_gauss.(energy_labels, beta)
# f_plus_gauss_t_vals = f_plus_t_gauss.(time_labels, beta)

# f_minus_nu_vals = f_minus_nu.(energy_labels, beta)
# f_minus_t_vals = f_minus_t.(time_labels, beta)
time_labels_no_zero = time_labels[2:end]
f_plus_metro_nu_vals = f_plus_nu_metro.(energy_labels, beta)
f_plus_metro_t_vals = f_plus_t_metro.(time_labels_no_zero, eta, beta)

# Fouriers
# fourierd_f_plus_nu = ifft(f_plus_gauss_nu_vals) * w0 * N / sqrt(2pi)  #! The symmetric F convention, and IFFT factor
# fourierd_f_minus_nu = ifft(f_minus_nu_vals) * w0 * N / sqrt(2pi)
fourierd_f_plus_metro_nu = ifft(f_plus_metro_nu_vals) * w0 * N / sqrt(2pi) 
println(fourierd_f_plus_metro_nu)
# Norm checks
# norm(fourierd_f_plus_nu - f_plus_gauss_t_vals)
# norm(fourierd_f_minus_nu - f_minus_t_vals)
norm(fourierd_f_plus_metro_nu - f_plus_metro_t_vals)


# Plots
# plot(fftshift(time_labels), real.(fftshift(f_plus_gauss_t_vals)), label="Re Analytic")
# plot!(fftshift(time_labels), imag.(fftshift(f_plus_gauss_t_vals)), label="Im Analytic")
# plot!(fftshift(time_labels), real.(fftshift(fourierd_f_plus_nu)), label="Re FFT")
# plot!(fftshift(time_labels), imag.(fftshift(fourierd_f_plus_nu)), label="Im FFT")

plot(fftshift(time_labels), real.(fftshift(f_minus_t_vals)), label="Re Analytic")
plot!(fftshift(time_labels), real.(fftshift(fourierd_f_minus_nu)), label="Re FFT")

#* Basic working example ----------------------------------------------------------------------------------------------------
# Parameters
sigma = 1.0
N = 1024       # Number of points (preferrably even)
T = 20.0        # Total time window (should be large enough to capture Gaussian decay)
dt = T / N      # Time step
t = range(-T/2, stop=T/2 - dt, length=N)  # Centered time array
# Generate Gaussian function
g = exp.(-t.^2 ./ (2*sigma^2))

# Compute FFT
g_shifted = fftshift(g)         # Shift to FFT's expected order (t=0 at first index)
G_fft = fft(g_shifted)           # Compute FFT
G_fft_scaled = G_fft * dt        # Scale by dt to match continuous FT

# Frequency array (correctly shifted for plotting)
freqs = fftfreq(N, 1/dt)         # Frequencies in FFT order (not shifted)
freqs_shifted = fftshift(freqs)  # Shifted frequencies (zero-centered)

# Analytic Fourier transform (using angular frequency ω)
omega = 2π * freqs_shifted       # Convert to angular frequency
G_analytic = sigma * sqrt(2π) .* exp.(- (sigma^2 .* omega.^2) ./ 2)

# Shift FFT result to center for comparison
G_fft_scaled_shifted = fftshift(G_fft_scaled)

# Inverse FFT to recover original signal
G_inverse_unshifted = ifftshift(G_fft_scaled_shifted)  # Revert frequency shift
g_reconstructed_shifted = ifft(G_inverse_unshifted)    # Compute IFFT
g_reconstructed_shifted /= dt                          # Rescale by dt. 
                                                       #! Also: 1/dt = dw * N / 2pi which gets rid of 1/N base factor of ifft
g_reconstructed = fftshift(g_reconstructed_shifted)    # Shift back to centered time

# Plotting
p1 = plot(t, g, label="Original g(t)", title="Time Domain")
plot!(p1, t, real(g_reconstructed), linestyle=:dash, label="Reconstructed")
norm(real(g - g_reconstructed))

p2 = plot(freqs_shifted, abs.(G_fft_scaled_shifted), label="FFT", xlim=(-5,5), title="Frequency Domain")
plot!(p2, freqs_shifted, G_analytic, linestyle=:dash, label="Analytic")
norm(G_analytic - abs.(G_fft_scaled_shifted))

plot(p1, p2, layout=(2,1), size=(800,600))