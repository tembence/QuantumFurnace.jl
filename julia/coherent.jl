using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using TensorOperations
using JLD
using Plots
using QuadGK

include("hamiltonian_tools.jl")
include("jump_op_tools.jl")
include("trotter.jl")
include("qi_tools.jl")
include("trotter.jl")


#TODO: Fix these, they dont match

# Eq. (2.5)
function coherent_bohr(hamiltonian::HamHam, jump::JumpOp, beta::Float64)
    #! Dont forget alpha_v1v2
    B = (tanh.(-beta * hamiltonian.bohr_freqs / 4) / (2*im)) .* (jump.in_eigenbasis' * jump.in_eigenbasis)
    return B
end

function coherent_bohr_explicit(hamiltonian::HamHam, jump::JumpOp, beta::Float64)

    dim = size(hamiltonian.bohr_freqs, 1)
    B::SparseMatrixCSC{ComplexF64, Int64} = spzeros(dim, dim)
    for j in 1:dim
        for i in 1:dim
            for k in 1:dim
                    v = hamiltonian.bohr_freqs[k, i]
                    sp_v2 = spzeros(dim, dim)
                    sp_v2[i, k] = jump.in_eigenbasis[i, k]
                    sp_v1 = spzeros(dim, dim)
                    sp_v1[i, j] = jump.in_eigenbasis[i, j]

                    B += (tanh(-beta * v / 4) / (2*im)) * sp_v2' * sp_v1
            end
        end
    end
    return B
end

# (3.1) and Proposition III.1
function coherent_gaussian_timedomain(jump::JumpOp, hamiltonian::HamHam, time_labels::Vector{Float64},
    sigma::Float64, beta::Float64)
    """Coherent term for the Gaussian case, written in timedomain and using ideal time evolution.
    Working in the energy eigenbasis for the time evolutions, where H is diagonal."""
    
    b1 = compute_b1(time_labels)
    b2 = exp.(-4*time_labels.^2 .- 2*im*time_labels) / sqrt(4*pi^3)

    diag_time_evolve(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))

    B = zeros(ComplexF64, size(hamiltonian.data, 1), size(hamiltonian.data, 1))

    jump_op_in_eigenbasis_dag = jump.in_eigenbasis'
    # Outer sum b1
    for (i, t) in enumerate(time_labels)
        time_evolution_outer = diag_time_evolve(beta * t)

        # Inner sum b2
        b2_sum = zeros(ComplexF64, size(hamiltonian.data, 1), size(hamiltonian.data, 1))
        for (j, s) in enumerate(time_labels)
            time_evolution_inner = diag_time_evolve(beta * s)
            b2_sum .+= b2[j] * time_evolution_inner^2 * (time_evolution_inner')^2 * (jump_op_in_eigenbasis_dag * jump.in_eigenbasis)
        end

        B .+= b1[i] * time_evolution_outer' * b2_sum * time_evolution_outer
    end
    return B
end

function coherent_metropolis_timedomain()
end

function convolute(f::Function, g::Function, t::Float64; atol=1e-12, rtol=1e-12)
    integrand(s) = f(s) * g(t - s)
    result, _ = quadgk(integrand, -Inf, Inf; atol=atol, rtol=rtol)
    return result
end

# Corollary III.1, every parameter = 1 / beta
function compute_b1(time_labels::Vector{Float64})
    f1(t) = 1 / cosh(2 * pi * t)
    f2(t) = sin(-t) * exp(-2 * t^2)
    return 2 * sqrt(pi) * exp(1/8) * convolute.(Ref(f1), Ref(f2), time_labels)
end

#* Testing
num_qubits = 4
sigma = 5.
beta = 1.
eig_index = 8
jump_site_index = 1

#* Hamiltonian
hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n4.jld")["ideal_ham"]
# hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3), batch_size=1)
initial_state = hamiltonian.eigvecs[:, eig_index]
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

#* Jump operators
sigmax::Matrix{ComplexF64} = [0 1; 1 0]
jump_op = Matrix(pad_term([sigmax], num_qubits, jump_site_index))
jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
jump = JumpOp(jump_op,
        jump_op_in_eigenbasis,
        Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
        zeros(0), 
        zeros(0, 0))

#* b1 
# N = 2^(12)
# N_labels = [-Int(N/2):1:-1; 0:1:Int(N/2)-1]

# w0 = 0.1
# t0 = 2 * pi / (N * w0)
# time_labels = t0 * N_labels

# @time b1 = compute_b1(time_labels)

#TODO: Check l1 norm 

# plot(time_labels, real.(b1), label="Real part")

#* Coherent dynamics
@time B = coherent_bohr(hamiltonian, jump, beta)
@time B_explicit = coherent_bohr_explicit(hamiltonian, jump, beta)
println(norm(B - B_explicit))
