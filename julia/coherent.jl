using LinearAlgebra
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
using SparseArrays
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
function coherent_term_from_timedomain(jump::JumpOp, hamiltonian::HamHam, 
    b1_vals::Vector{ComplexF64}, b1_times::Vector{Float64}, b2_vals::Vector{ComplexF64}, b2_times::Vector{Float64},
    beta::Float64)
    """Coherent term for the Gaussian AND Metropolis case IF jump op is [A Adag, H] = 0 (X, Y, Z)
    written in timedomain and using ideal time evolution.
    Working in the energy eigenbasis for the time evolutions, where H is diagonal.
        sigma_E = sigma_gamma = w_gamma = 1 / beta"""

    diag_time_evolve(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))

    #FIXME: Something takes awful lot here for even a tiny set of b1 b2 labels
    jump_op_in_eigenbasis_dag = jump.in_eigenbasis'
    B = zeros(ComplexF64, size(hamiltonian.data, 1), size(hamiltonian.data, 1))
    # Outer sum b1
    for (i, t) in enumerate(b1_times)
        # Inner sum b2
        b2_sum = zeros(ComplexF64, size(hamiltonian.data, 1), size(hamiltonian.data, 1))
        for (j, s) in enumerate(b2_times)
            time_evolution_inner = diag_time_evolve(beta * s)
            b2_sum .+= b2_vals[j] * time_evolution_inner * 
                                (jump_op_in_eigenbasis_dag * (time_evolution_inner')^2 * jump.in_eigenbasis) *
                                time_evolution_inner'
        end
        time_evolution_outer = diag_time_evolve(beta * t)
        B .+= b1_vals[i] * time_evolution_outer' * b2_sum * time_evolution_outer
    end
    return B
end

#TODO: FINISH TROTTER VERSION 
function coherent_term_timedomain_trotter()
end

function convolute(f::Function, g::Function, t::Float64; atol=1e-12, rtol=1e-12)
    integrand(s) = f(s) * g(t - s)
    result, _ = quadgk(integrand, -Inf, Inf; atol=atol, rtol=rtol)
    return result
end

function compute_truncated_b1(time_labels::Vector{Float64})

    b1 = Vector{ComplexF64}(compute_b1(time_labels))

    # Skip all elements where b1 b2 are smaller than 1e-14
    indices_b1 = get_truncated_indices_b(b1)
    b1_times = time_labels[indices_b1]
    b1_vals = b1[indices_b1]
    
    return (b1_vals, b1_times)
end

function compute_truncated_b2(time_labels::Vector{Float64})

    b2 = exp.(-4*time_labels.^2 .- 2*im*time_labels) / sqrt(4*pi^3)

    # Skip all elements where b1 b2 are smaller than 1e-14
    indices_b2 = get_truncated_indices_b(b2)
    b2_times = time_labels[indices_b2]
    b2_vals = b2[indices_b2]

    return (b2_vals, b2_times)
end

function compute_truncated_b2_metro(time_labels::Vector{Float64}, eta::Float64)
    """(3.6)"""
    b2_metro::Vector{ComplexF64} = (1/(4 * pi * sqrt(2))) * 
        exp.(-2 * time_labels.^2 .- 1im * time_labels) ./ (time_labels .* (2 * time_labels .+ 1im))

    # (1, 1, 1 (at eta), 0, 0, ..., 0)
    heaviside_eta = ifelse.(abs.(time_labels) .> eta, 0, ones(length(time_labels)))
    b2_metro .+= heaviside_eta .* (1im * (2 * time_labels .+ 1im) ./ (2 * time_labels .+ 1im))
    
    indices_b2_metro = get_truncated_indices_b(b2_metro)
    b2_metro_times = time_labels[indices_b2_metro]
    b2_metro_vals = b2_metro[indices_b2_metro]

    return (b2_metro_vals, b2_metro_times)
end

# Corollary III.1, every parameter = 1 / beta
function compute_b1(time_labels::Vector{Float64})
    f1(t) = 1 / cosh(2 * pi * t)
    f2(t) = sin(-t) * exp(-2 * t^2)
    return 2 * sqrt(pi) * exp(1/8) * convolute.(Ref(f1), Ref(f2), time_labels)
end

function get_truncated_indices_b(b::Vector{ComplexF64})
   
    # find elements in b1, b2 that are larger than 1e-14
    atol = 1e-14
    # indices_b1_real = findall(x -> abs(x) > atol, real.(b1))
    # indices_b1_imag = findall(x -> abs(x) > atol, imag.(b1))
    # indices_b2_real = findall(x -> abs(x) > atol, real.(b2))
    # indices_b2_imag = findall(x -> abs(x) > atol, imag.(b2))
    # indices_b1 = union(indices_b1_real, indices_b1_imag)

    indices_b = findall(x -> abs(real(x)) > atol || abs(imag(x)) > atol, b)

    # @printf("Number of nonzero elements in b1: %d\n", length(indices_b1))
    # @printf("Number of nonzero elements in b2: %d\n", length(indices_b2))

    return indices_b
end

#* Testing
# num_qubits = 8
# sigma = 5.
# beta = 0.1
# eig_index = 8
# jump_site_index = 1

# #* Hamiltonian
# hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n8.jld")["ideal_ham"]
# # hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3), batch_size=1)
# initial_state = hamiltonian.eigvecs[:, eig_index]
# hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

# #* Jump operators
# sigmax::Matrix{ComplexF64} = [0 1; 1 0]
# jump_op = Matrix(pad_term([sigmax], num_qubits, jump_site_index))
# jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
# jump = JumpOp(jump_op,
#         jump_op_in_eigenbasis,
#         Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
#         zeros(0), 
#         zeros(0, 0))

# #* Fourier labels
# num_energy_bits = ceil(Int64, log2((0.45 * 2) / hamiltonian.w0)) + 2 # paper (above 3.7.), later will be β dependent
# num_energy_bits = ceil(Int64, log2((0.45 * 4 + 2/beta) / hamiltonian.w0))  # Under Fig. 5. with secular approx.
# # num_energy_bits = 9
# N = 2^(num_energy_bits)
# N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

# t0 = 2 * pi / (N * hamiltonian.w0)
# time_labels = t0 * N_labels
# energy_labels = hamiltonian.w0 * N_labels

# @printf("Number of qubits: %d\n", num_qubits)
# @printf("Number of energy bits: %d\n", num_energy_bits)
# @printf("Energy unit: %e\n", hamiltonian.w0)
# @printf("Time unit: %e\n", t0)

#* b1 
# N = 2^(12)
# N_labels = [-Int(N/2):1:-1; 0:1:Int(N/2)-1]

# w0 = 0.1
# t0 = 2 * pi / (N * w0)
# time_labels = t0 * N_labels

# @time b1 = compute_b1(time_labels)

#TODO: Check l1 norm 

# plot(time_labels, real.(b1), label="Real part")

#* Coherent term
# @time B = coherent_bohr(hamiltonian, jump, beta)
# @time B_explicit = coherent_bohr_explicit(hamiltonian, jump, beta)
# println(norm(B - B_explicit))

# b1 = compute_b1(time_labels)
# display(abs(b1[2]))
# b1 = Vector{ComplexF64}(b1)
# all_ones = ones(ComplexF64, size(b1))
# b2 = exp.(-4*time_labels.^2 .- 2*im*time_labels) / sqrt(4*pi^3)

# truncated_b1_times, truncated_b2_times = truncate_time_labels_b1b2(time_labels, b1, b2)

# @printf("Truncated b1 times: %d\n", length(truncated_b1_times))
# display(truncated_b1_times)
# @printf("Truncated b2 times: %d\n", length(truncated_b2_times))
# display(truncated_b2_times)

# @time B = coherent_gaussian_timedomain(jump, hamiltonian, time_labels, beta)
# # norm(B - B')

# function round_complex(z::ComplexF64, digits::Int64)
#     rounded_real = round(real(z), digits=digits)
#     rounded_imag = round(imag(z), digits=digits)
#     return rounded_real + rounded_imag*im
# end

# # find nonzero elements in b
# nonzero_indices = findall(!iszero, round_complex.(B, 7))
# display(B[nonzero_indices])
