using FFTW
using Plots

# Define the Gaussian function
# function gaussian(x)
#     return exp(-x^2)
# end

# # Analytical Fourier transform of the Gaussian function
# function analytical_ft(k)
#     return sqrt(pi) * exp(-k^2 / 4)
# end

# # Define the range and sample points
# N = 1024
# L = 10.0
# x = LinRange(-L, L, N)
# dx = x[2] - x[1]

# # Evaluate the Gaussian function at the sample points
# f_x = gaussian.(x)

# # Perform the numerical Fourier transform using FFTW
# F_k = fftshift(fft(f_x)) * dx / sqrt(2 * pi * N)
# k = fftshift(fftfreq(N, dx)) * 2 * pi

# # Analytical result for comparison
# analytical_result = analytical_ft.(k)

# # Plotting the results
# plot(k, abs.(F_k), label="Numerical FT", linewidth=2)
# plot!(k, analytical_result, label="Analytical FT", linestyle=:dash, linewidth=2)
# xlabel!("k")
# ylabel!("Magnitude")
# title!("Fourier Transform of Gaussian Function")

# Define the number of points
# N = 1024

# Define the range for the frequency domain
# N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
# #* 1) Start with labels, 2) one of the units, 3) other unit with 2 pi / N * k0
# x = LinRange(0, 20, N)
# x0 = x[2] - x[1]
# k0 = 2 * pi / (N * x0)
# k = collect(fftshift(fftfreq(N, N*k0)))
# freqs = collect(fftshift(fftfreq(N, 1)))

# # Define the Gaussian function in the frequency domain
# σ = 1.0
# f = exp.(-x.^2 / (σ^2))
# F_analytic = σ * exp.(-k.^2 * σ^2 / 4) / sqrt(2)
# plot(x, f, label="Gaussian", title="Gaussian Function", xlabel="x", ylabel="Amplitude")
# plot(k, F_analytic, label="Analytical FT", title="Analytical Fourier Transform of a Gaussian", xlabel="k", ylabel="Amplitude")

# # Perform the inverse Fourier transform
# F = fftshift(fft(f))

# # Plot the real part of the inverse Fourier transform
# plot(freqs, real(F), label="Real part", title="Fourier Transform of a Gaussian", xlabel="k", ylabel="Amplitude")

# Plot the imaginary part of the inverse Fourier transform
# plot!(x, imag(f), label="Imaginary part")


# Define the time vector
# t = 0:255
N = 2^8
times = LinRange(-10, 10, N)
times_binaryorder = fftshift(times)
deltat = abs(times[2] - times[1])

# Compute the Fourier Transform
# sp = abs.(fftshift(fft(sin.(2*pi*80*t))))
sigma_t = 2.0
gauss(t) = exp(-t^2 / sigma_t^2)
plot(times, gauss.(times), label="Gaussian", title="Gaussian Function", xlabel="t", ylabel="Amplitude")
sp = (fftshift(ifft(gauss.(times_binaryorder)))) * N

# Compute the corresponding frequencies
freq = fftshift(fftfreq(length(times_binaryorder), deltat))

# Plot the real and imaginary parts of the Fourier Transform
plot!(freq, real(sp), label="Real part")
# plot!(freq, imag(sp), label="Imaginary part")

fftfreq(5, 5) == fftfreq(5)*5