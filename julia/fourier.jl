using LinearAlgebra
using FFTW
using Plots

w0 = 0.01284970
num_energy_bits = ceil(Int64, log2((0.45 * 2) / w0)) 
N = 2^num_energy_bits
t0 = 2 * pi / (w0 * N)
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
N_labels_decimal_order = [-Int(N/2):1:-1; 0:1:Int(N/2)-1]
time_labels = t0 * N_labels_decimal_order
energy_labels = w0 * N_labels

sigma = 10
f = exp.(- time_labels.^2 / (4 * sigma^2))
f = f / sqrt(sum(f.^2))

# plot f ove time labels
# plot(time_labels, f, label="f(t)", xlabel="t", ylabel="f(t)", title="Gaussian function over time labels")

#* FFT
F = fft(f)
F = F / sqrt(sum(F.^2))
println(fftfreq(N))
println

#* My Fourier form 
F_mine = exp.(-energy_labels.^2 * sigma^2)
F_mine = F_mine / sqrt(sum(F_mine.^2))

#! THEY OVERLAP.
#* Compare
plot(fftfreq(N), abs.(F), label="FFT", xlabel="Energy", ylabel="|F(ω)|", title="FFT of Gaussian function")
# plot!(fftfreq(N), abs.(F_mine), label="My Fourier form")
