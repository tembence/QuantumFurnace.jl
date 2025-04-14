using LinearAlgebra
using Random
using Printf
using QuadGK
using Plots
using Distributions

include("hamiltonian.jl")
include("qi_tools.jl")

#* Functions
gauss_filter(w, nu, sigmaE) = exp(- (w - nu)^2 / (4 * sigmaE^2)) * sqrt(beta / sqrt(2 * pi))
gamma_gauss(w, x, sigmaG) = exp(-(w + x)^2 / (2 * sigmaG^2))

# Alpha with skew symmetry enforced
alpha_gauss(nu1, nu2, x, sigmaE, beta) = (sqrt(2 * x / beta - sigmaE^2) * exp(-(nu1 + nu2 + 2 * x)^2 / (16 * x / beta)) 
                                                    * exp(-(nu1 - nu2)^2 / (8 * sigmaE^2)) / sqrt(2 * x / beta))


#* Config
beta = 10.0
nu1 = 0.20
nu2 = 0.21
sigmaE = 1 / beta
sigmaG = 1 / beta   
x = 1/ beta
w0 = 1e-1
energies = [-2.0:w0:2.0;]
bohr_freqs = [-0.5:0.001:0.5;]
bohr_freqs_clustered = sort([-0.5:00.1:0.5; collect(rand(Uniform(0.2, 0.3), 900))])
gauss_integrand(w, nu1, nu2) = gamma_gauss(w, x, sigmaG) * gauss_filter(w, nu1, sigmaE) * gauss_filter(w, nu2, sigmaE)

#* Integral / Sum
# alpha_gauss_integral = quadgk(w -> gauss_integrand(w, nu1, nu2), -Inf, Inf)[1]

#* Compare
# alpha_gauss_analytic = alpha_gauss(nu1, nu2, x, sigmaE, beta)
# norm(alpha_gauss_integral - alpha_gauss_analytic)

average_error = 0.0
max_error = 0.0
for nu in bohr_freqs
    alpha_gauss_analytic = alpha_gauss(nu, nu2, x, sigmaE, beta)
    alpha_gauss_riemann = riemann_sum(w -> gauss_integrand(w, nu, nu2), energies)[1]
    error = norm(alpha_gauss_riemann - alpha_gauss_analytic)
    if error > max_error
        max_error = error
    end
    average_error += error
end
average_error /= length(bohr_freqs)
max_error

average_error = 0.0
max_error = 0.0
for nu in bohr_freqs_clustered
    alpha_gauss_analytic = alpha_gauss(nu, nu2, x, sigmaE, beta)
    alpha_gauss_riemann = riemann_sum(w -> gauss_integrand(w, nu, nu2), energies)[1]
    error = norm(alpha_gauss_riemann - alpha_gauss_analytic)
    if error > max_error
        max_error = error
    end
    average_error += error
end
average_error /= length(bohr_freqs)
max_error


