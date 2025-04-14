using FFTW
using Plots

#* Example from a tutorial
# N = 1000 # Number of samples, the more the better
# xj = (0:N-1)*2*π/N  # Here he uses 2pi / N units for the x variable so k will be integers
# f = 2*exp.(17*im*xj) + 3*exp.(6*im*xj) + rand(N)

# original_k = 1:N
# shifted_k = fftshift(fftfreq(N)*N)  # = fftfreq(N, N) probably just for simplicity
# # x0 * k0 here is 2pi/N as prescribed for FT
# # println("Is x0 * k0 = 2pi/N?  ", abs(xj[2] - xj[1]) * abs(original_k[2] - original_k[1]) == 2*π/N)

# original_fft = fft(f)  # fft() results in a vecotr 1:21
# shifted_fft = fftshift(fft(f))  # this centres it around 0

# p1 = plot(original_k,abs.(original_fft),title="Original FFT Coefficients", xticks=original_k[1:2:end], legend=false, ylims=(0,70));
# p2 = plot(shifted_k,abs.(shifted_fft),title="Shifted FFT Coefficients",xticks=shifted_k[1:2:end], legend=false, ylims=(0,70));
# plot(p1,p2,layout=(2,1))

#* Fourier transform of filter Gaussian

max_energy = 0.45
w0 = 0.001
N = Int(ceil(max_energy / w0)) * 4
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
energy_labels = w0 * N_labels
energy_labels_045 = energy_labels[abs.(energy_labels) .<= 0.45]
t0 = 2 * π / (N * w0)
time_labels = N_labels * t0
# fftfreq(N, N * w0) == energy_labels

sigma_t = 100.
# Gaussian in time
f_gauss(t) = exp(-t^2 / (4 * sigma_t^2))
ft_vals = f_gauss.(time_labels)
ft_vals /= sqrt(sum(ft_vals.^2))

# Gaussian in frequency
F_gauss(w) = exp(-w^2 * sigma_t^2) 
nu = 0.3
F_gauss_vals = F_gauss.(energy_labels .- nu)
F_gauss_vals /= sqrt(sum(F_gauss_vals.^2))

#* FFT
f_gauss_fft = real.(fft(ft_vals))
f_gauss_fft /= sqrt(sum(f_gauss_fft.^2))
# plot(fftshift(energy_labels), fftshift(f_gauss_fft), title="Gaussian in frequency", label="FFT")
# plot!(fftshift(energy_labels), fftshift(F_gauss_vals), label="F(w)")

#* IFFT
F_gauss_ifft = real.(ifft(F_gauss_vals))
F_gauss_ifft /= sqrt(sum(F_gauss_ifft.^2))
# plot(fftshift(time_labels), fftshift(F_gauss_ifft), title="Gaussian in time", label="IFFT")
# plot!(fftshift(time_labels), fftshift(ft_vals), label="f(t)")

#* IFFT on truncated F(w)
cutoff = 0.45
energy_labels_045 = energy_labels
# Turn all entires to zero above and below cutoff
F_gauss_trunc = F_gauss_vals
for i in eachindex(energy_labels)
    if abs(energy_labels[i]) > cutoff
        F_gauss_trunc[i] = 0
        energy_labels_045[i] = 0
    end
end

F_gauss_trunc /= sqrt(sum(F_gauss_trunc.^2))

F_gauss_ifft_trunc = real.(ifft(F_gauss_trunc))
F_gauss_ifft_trunc /= sqrt(sum(F_gauss_ifft_trunc.^2))
plot(fftshift(time_labels), fftshift(F_gauss_ifft_trunc), title="Gaussian in time (truncated)", label="IFFT")
# plot!(fftshift(time_labels), fftshift(ft_vals), label="f(t)")