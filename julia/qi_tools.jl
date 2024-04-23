using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter

function trace_distance(rho::Union{Matrix{ComplexF64}, Matrix{Float64}}, sigma::Union{Matrix{ComplexF64}, Matrix{Float64}})
    return 0.5 * norm(rho - sigma, 1)
end

function fidelity(rho::Union{Matrix{ComplexF64}, Matrix{Float64}}, sigma::Union{Matrix{ComplexF64}, Matrix{Float64}})
    return abs(tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))
end
 #TODO: FIX THESE
# Test
rho = [0.5 0.5; 0.5 0.5]
sigma = [1. 0.; 0. 0.]
println(trace_distance(rho, sigma))
println(fidelity(rho, sigma))