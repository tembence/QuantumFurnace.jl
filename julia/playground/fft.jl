using FFTW
using Plots
using QuadGK

# Fourier transforming f(nu)_ -> f(t)_ in (A4)

beta = 1.
sigma_E = 1 / beta
num_labels = Int(1e1)
energies = LinRange(-40., 40., num_labels)

f(nu) = (tanh(-beta * nu / 4) / (4 * pi * im)) * exp(-nu^2 / (8 * sigma_E^2))
f_nu_vals = f.(energies)

plot(energies, imag.(f_nu_vals), label="Re(f(nu))")

f_t_vals = fft(f_nu_vals)
f_t_shifted = fftshift(f_t_vals)
times_from_fft = fftshift(fftfreq(length(energies), energies[2] - energies[1]))

plot!(times_from_fft, real.(f_t_shifted), label="Re(f(t))")

# Plots
# plot(times[1:50], real.(f_t_vals)[1:50], label="Re(f(t))")
# plot(times, imag.(f_t_vals), label="Im(f(t))")
# plot(energies, real.(f_nu_vals), label="Re(f(nu))")
# plot(energies, imag.(f_nu_vals), label="Im(f(nu))")


#* Integral written out
function inverse_fourier_by_hand(f_nu_vals::Vector{ComplexF64}, num_labels::Int64)
    f_t_vals = zeros(ComplexF64, num_labels)
    for j in 1:num_labels
        for k in 1:num_labels
            f_t_vals[j] += f_nu_vals[k] * exp(im * j * k * 2 * pi / num_labels)
        end
    end
    return f_t_vals / num_labels
end
# int_labels = [- num_labels / 2 : num_labels / 2 - 1;]
# energies = LinRange(-0.45, 0.45, num_labels)*100
# dE = energies[2] - energies[1]
# energies = dE * int_labels
# dt = 2 * pi / (num_labels * dE)
# times = dt * int_labels

f_t_vals_by_hand = inverse_fourier_by_hand(f_nu_vals, num_labels)

# Plots
# plot(times, real.(f_t_vals_by_hand), label="Re(f(t)) by hand")
# plot(times, imag.(f_t_vals_by_hand), label="Im(f(t)) by hand")


#* Convolution via integral from
f_t_1(s) = 1 / cosh(2 * pi * s / beta)
f_t_2(t, s) = sin(- beta * sigma_E^2 * (t - s)) * exp(- 2 * sigma_E^2 * (t - s)^2)

times = collect(LinRange(-5, 5, num_labels))
#TODO: FINISH THIS, NOT YET CORRECT
function convolution(f_t_1::Function, f_t_2::Function, times::Vector{Float64})
    f_t = zeros(Float64, length(times))
    for i in 1:length(times)
        t = times[i]
        for s in times
            f(s) = f_t_1(s) * f_t_2(t, s)
            f_t[i] = quadgk(f, -5., 5.)[1]
            println("Timestep done: ", i, " out of ", length(times))
        end
    end
    return f_t
end

convoluted_f_t = (sigma_E / (pi * beta)) * exp(beta^2 * sigma_E^2 / 8) * convolution(f_t_1, f_t_2, times)

plot(times, convoluted_f_t, label="f(t) convolution")

#Plots
# plot(times, f_t_1.(times), label="f(t)_1")
# plot!(times, f_t_2.(times), label="f(t)_2")

#TODO: Test the Fourier transforms of the separate functions from the paper!

F1(nu) = (1/sqrt(2pi)) * (sinh(-beta * nu / 4) / (2*im)) * exp(-nu^2 / (8 * sigma_E^2))
F1_vals = F1.(energies)
f1_t_vals = fft(F1_vals)
f1_t_shifted = fftshift(f1_t_vals)

# f1_times = fftfreq(num_labels, energies[2] - energies[1])
f1_times = fftshift(fftfreq(length(energies), energies[2] - energies[1]))

actual_f1_t(t) = sigma_E * exp(beta^2 * sigma_E^2 / 8) * sin(-beta * sigma_E^2 * t) * exp(-2 * sigma_E^2 * t^2)
actual_f1_t_vals = actual_f1_t.(f1_times)
plot(f1_times, real.(f1_t_vals), label="Re(f1(t))")
plot(LinRange(-0.1, 0.1, num_labels), real.(actual_f1_t_vals), label="Re(f1(t)) actual")

a = [0 1 2 3 4]
fftshift(a)