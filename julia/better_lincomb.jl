using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using BenchmarkTools
using Roots
using QuadGK
using Plots

include("hamiltonian.jl")
include("qi_tools.jl")
include("structs.jl")
include("bohr_picture.jl")
include("energy_picture.jl")
include("time_picture.jl")
include("ofts.jl")
include("coherent.jl")

atol = 1e-10
rtol = 1e-10
beta = 10.

t0 = 1e-2
@printf("Handmade t0: %f\n", t0)
handmade_times = [-20.0:t0:20.0;]
times = handmade_times

f_better(t, a, beta) = exp((-4 * t^2 + a * beta + 2im * t * beta) / (2 * beta^2)) / (4 * t^2 + a * beta + 2im * t * beta) / sqrt(2pi)
f_better_scaled(t, a, beta) = exp(a / (2 * beta)) * sqrt(4 * a / beta + 1) * f_better(t, a, beta)
scaling(a, beta) = exp(a / (2 * beta)) * sqrt(4 * a / beta + 1)

#* Plots
# plot(times, real.(f_better.(times, beta, beta)), label="Re, a=beta")
# plot!(times, imag.(f_better.(times, beta, beta)), label="Im, a=beta")
plot!(times, real.(f_better.(times, beta/10.0, beta)), label="Re, a=beta/10")
# plot!(times, imag.(f_better.(times, beta/10.0, beta)), label="Im, a=beta/10")

plot(times, real.(f_better_scaled.(times, beta/10.0, beta)), label="Re, a=beta/10")
plot!(times, imag.(f_better_scaled.(times, beta/10.0, beta)), label="Im, a=beta/10")

#* Riemann sums / integrals
f_better_integrated = quadgk(t -> f_better(t, beta / 100, beta), -Inf, Inf; atol=atol, rtol=rtol)[1]
f_better_riemann = riemann_sum(t -> f_better(t, beta / 100, beta), times)

norm(f_better_integrated - f_better_riemann)

# With exp(iwt)
w = 0.12
f_better_integrated_w = quadgk(t -> f_better(t, beta / 100, beta) * exp(1im * w * t), -Inf, Inf; atol=atol, rtol=rtol)[1]
f_better_riemann_w = riemann_sum(t -> f_better(t, beta / 100, beta) * exp(1im * w * t), times)

norm(f_better_integrated_w - f_better_riemann_w)

#* Gamma energy integral
w0 = 1e-2
energies = [-20.0:w0:20.0;]
gamma_beta(w, beta) = exp((-3 -2 * beta * w - sqrt(5) * abs(1 + 2 * beta * w)) / 4) / sqrt(5)
gamma_beta_over_10(w, beta) = sqrt(5/7) * exp((-3 - 5 * beta * w - sqrt(35/4) * abs(1 + 2 * beta * w)) / 10)
gamma_beta_over_100(w, beta) = sqrt(25/26) * exp((-51 - 100 * beta * w - sqrt(2600) * abs(1 + 2 * beta * w)) / 200)

#* Plots
# plot(energies, gamma_beta.(energies, beta))
# plot(energies, gamma_beta_over_10.(energies, beta))
# plot(energies, gamma_beta_over_100.(energies, beta))

#* Sums
gamma_beta_integrated = quadgk(w -> gamma_beta(w, beta), -Inf, Inf; atol=atol, rtol=rtol)[1]
gamma_beta_riemann = riemann_sum(w -> gamma_beta(w, beta), energies)

norm(gamma_beta_integrated - gamma_beta_riemann)

gamma_beta_over_10_integrated = quadgk(w -> gamma_beta_over_10(w, beta), -Inf, Inf; atol=atol, rtol=rtol)[1]
gamma_beta_over_10_riemann = riemann_sum(w -> gamma_beta_over_10(w, beta), energies)

norm(gamma_beta_over_10_integrated - gamma_beta_over_10_riemann)


w0 = 1e-3
energies = [-20.0:w0:20.0;]
gamma_beta_over_100_integrated = quadgk(w -> gamma_beta_over_100(w, beta), -Inf, Inf; atol=atol, rtol=rtol)[1]
gamma_beta_over_100_riemann = riemann_sum(w -> gamma_beta_over_100(w, beta), energies)

norm(gamma_beta_over_100_integrated - gamma_beta_over_100_riemann)


log2(2pi / (0.1 * 0.01))

w0 = 1e-3
energies = [-20.0:w0:20.0;]
transition_metro(w, beta) = exp(-beta * max(w + 1/(2 * beta), 0.0))
# plot(energies, transition_metro.(energies, beta))

metro_integrated = quadgk(w -> transition_metro(w, beta), energies[1], energies[end]; atol=atol, rtol=rtol)[1]
metro_riemann = riemann_sum(w -> transition_metro(w, beta), energies)

norm(metro_integrated - metro_riemann)