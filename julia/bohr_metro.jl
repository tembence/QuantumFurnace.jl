using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using BenchmarkTools
using QuadGK

include("hamiltonian.jl")
include("ofts.jl")
include("qi_tools.jl")
include("structs.jl")

function create_alpha_nu1_matrix_metro(bohr_freqs::Matrix{Float64}, nu_2::Float64, beta::Float64)
    """Gaussian parameters = 1/β, but w_gamma=x is a linear combination, i.e. integral."""
    alpha_fn(nu_1, x) = exp(-beta^2 * (nu_1 + nu_2 + 2*x)^2/16) * exp(-beta^2 * (nu_1 - nu_2)^2/8) / sqrt(2)
    alpha_nu_1_matrix(x) = alpha_fn.(bohr_freqs, x) / sqrt(2 * pi * (2*x/beta - 1/beta^2))
    alpha_metro_nu1_matrix = quadgk(alpha_nu_1_matrix, 1/(2*beta), Inf)[1]
    return alpha_metro_nu1_matrix
end

