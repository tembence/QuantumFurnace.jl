using LinearAlgebra
using Random
using Printf
using ProgressMeter
using Distributed
using TensorOperations
using JLD
using Plots
using QuadGK
using SparseArrays

include("hamiltonian_tools.jl")
include("jump_op_tools.jl")
include("trotter.jl")
include("qi_tools.jl")
include("trotter.jl")

# const Number64 = Union{Float64, ComplexF64}
#TODO: Fix these, they are not close enough yet, check parameters for which maybe they could be closer to understand it
# Eq. (2.5)
function coherent_gaussian_bohr(hamiltonian::HamHam, jump::JumpOp, energy_labels::Vector{Float64}, beta::Float64)

    Fw = exp.(- beta^2 * (energy_labels).^2 / 4)
    Fw_norm = sqrt(sum(Fw.^2))

    gaussian_transition = exp.(-(energy_labels * beta .+ 1).^2 / 2)
    tanh_matrix = tanh.(-beta * hamiltonian.bohr_freqs / 4) / (2*im)
    B = zeros(ComplexF64, size(hamiltonian.data))
    for (i, w) in enumerate(energy_labels)
        jump_nu1 = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
        jump_nu2 = adjoint(entry_wise_oft_exact_db(jump, w, hamiltonian, beta)) 
        B .+= gaussian_transition[i] * tanh_matrix .* (jump_nu2 * jump_nu1)
    end
    return B / Fw_norm^2
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
# Has to be on a symmetric time domain, otherwise it can't be Hermitian.
function coherent_term_from_timedomain(jump::JumpOp, hamiltonian::HamHam, 
    b1::Dict{Float64, ComplexF64}, b2::Dict{Float64, ComplexF64}, t0::Float64, beta::Float64)
    """Coherent term for the Gaussian AND Metropolis case IF jump op is [A Adag, H] = 0 (X, Y, Z)
    written in timedomain and using ideal time evolution.
    Working in the energy eigenbasis for the time evolutions, where H is diagonal.
        sigma_E = sigma_gamma = w_gamma = 1 / beta"""
    
    diag_time_evolve(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))
    jump_op_in_eigenbasis_dag = jump.in_eigenbasis'

    # Inner b2 integral
    b2_integral = zeros(ComplexF64, size(hamiltonian.data))
    for s in keys(b2)
        time_evolution_inner = diag_time_evolve(beta * s)
        b2_integral .+= b2[s] * time_evolution_inner * 
        (jump_op_in_eigenbasis_dag * (time_evolution_inner')^2 * jump.in_eigenbasis) *
        time_evolution_inner
    end

    # Outer b1 integral
    B = zeros(ComplexF64, size(hamiltonian.data))
    for t in keys(b1)
        time_evolution_outer = diag_time_evolve(beta * t)
        B .+= b1[t] * time_evolution_outer' * b2_integral * time_evolution_outer
    end
    return B * t0^2
end

function coherent_term_trotter(jump::JumpOp, hamiltonian::HamHam, trotter::TrottTrott, 
    b1::Dict{Float64, ComplexF64}, b2::Dict{Float64, ComplexF64}, beta::Float64)

    diag_time_evolve(t) = Diagonal(trotter.eigvals_t0.^Int(ceil(t / trotter.t0))) # Trotter steps
    jump_op_in_eigenbasis_dag = jump.in_eigenbasis'

    # Inner b2 integral
    b2_integral = zeros(ComplexF64, size(hamiltonian.data))
    for s in keys(b2)
        time_evolution_inner = diag_time_evolve(beta * s)
        b2_integral .+= b2[s] * time_evolution_inner * 
        (jump_op_in_eigenbasis_dag * (time_evolution_inner')^2 * jump.in_eigenbasis) *
        time_evolution_inner
    end

    # Outer b1 integral
    B = zeros(ComplexF64, size(hamiltonian.data))
    for t in keys(b1)
        time_evolution_outer = diag_time_evolve(beta * t)
        B .+= b1[t] * time_evolution_outer' * b2_integral * time_evolution_outer
    end
    return B * trotter.t0^2

end

function coherent_term_timedomain_integrated_gauss(jump::JumpOp, hamiltonian::HamHam, beta::Float64, 
    time_domain::Tuple{Float64, Float64} = (-Inf, Inf))

    f1(t) = 1 / cosh(2 * pi * t)
    f2(t) = sin(-t) * exp(-2 * t^2)
    b1_fn(t) = 2 * sqrt(pi) * exp(1/8) * convolute.(Ref(f1), Ref(f2), t)
    b2_fn(t) = exp(-4*t^2 - 2im*t) / sqrt(4*pi^3)
    diag_time_evolve(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))
    jump_op_in_eigenbasis_dag = jump.in_eigenbasis'

    # Inner b2 integral
    b2_integrand(s) = b2_fn(s) * diag_time_evolve(beta * s) * 
                           (jump_op_in_eigenbasis_dag * diag_time_evolve(- 2 * beta * s) * jump.in_eigenbasis) *
                           diag_time_evolve(beta * s)
    b2_integral, _ = quadgk(b2_integrand, time_domain[1], time_domain[2]; atol=1e-12, rtol=1e-12)

    # Outer b1 integral
    b1_integrand(t) = b1_fn(t) * diag_time_evolve(- beta * t) * b2_integral * diag_time_evolve(beta * t)
    B, _ = quadgk(b1_integrand, time_domain[1], time_domain[2]; atol=1e-12, rtol=1e-12)
    return B
end

function heaviside(x::Float64)
    return x <= 0 ? 1 : 0 
end

function coherent_term_timedomain_integrated_metro(jump::JumpOp, hamiltonian::HamHam, eta::Float64, beta::Float64, 
    time_domain::Tuple{Float64, Float64} = (-Inf, Inf))

    f1(t) = 1 / cosh(2 * pi * t)
    f2(t) = sin(-t) * exp(-2 * t^2)
    b1_fn(t) = 2 * sqrt(pi) * exp(1/8) * convolute.(Ref(f1), Ref(f2), t)
    b2_metro_fn(t) = (exp(-2 * t^2 - im * t) + (abs(t) <= eta ? 1 : 0) * im * (2 * t + im)) / (sqrt(32) * pi * t * (2 * t + im))
    diag_time_evolve(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))
    jump_op_in_eigenbasis_dag = jump.in_eigenbasis'

    # Inner b2 integral
    b2_integrand(s) = b2_metro_fn(s) * diag_time_evolve(beta * s) * 
                           (jump_op_in_eigenbasis_dag * diag_time_evolve(- 2 * beta * s) * jump.in_eigenbasis) *
                           diag_time_evolve(beta * s)
    # b2 function has singularity in t = 0
    eps = 1e-12
    b2_integral_1, _ = quadgk(b2_integrand, time_domain[1], -eps; atol=1e-12, rtol=1e-12)
    b2_integral_2, _ = quadgk(b2_integrand, eps, time_domain[2]; atol=1e-12, rtol=1e-12)
    b2_integral = b2_integral_1 + b2_integral_2

    # Outer b1 integral
    b1_integrand(t) = b1_fn(t) * diag_time_evolve(- beta * t) * b2_integral * diag_time_evolve(beta * t)
    B, _ = quadgk(b1_integrand, time_domain[1], time_domain[2]; atol=1e-12, rtol=1e-12)
    return B
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

function compute_truncated_b1(time_labels::Vector{Float64}, atol::Float64 = 1e-14)

    b1 = Vector{ComplexF64}(compute_b1(time_labels))

    # Skip all elements where b1 b2 are smaller than 1e-14
    indices_b1 = get_truncated_indices_b(b1, atol)
    b1_times = time_labels[indices_b1]
    b1_vals = b1[indices_b1]
    
    return Dict(zip(b1_times, b1_vals))
end

function compute_b2(time_labels::Vector{Float64})
    return exp.(- 4 * time_labels.^2 .- 2 * im * time_labels) / sqrt(4 * pi^3)
end

function compute_truncated_b2(time_labels::Vector{Float64}, atol::Float64 = 1e-14)

    b2 = Vector{ComplexF64}(compute_b2(time_labels))

    # Skip all elements where b1 b2 are smaller than 1e-14
    indices_b2 = get_truncated_indices_b(b2, atol)
    b2_times = time_labels[indices_b2]
    b2_vals = b2[indices_b2]

    return Dict(zip(b2_times, b2_vals))
end

function compute_truncated_b2_metro(time_labels::Vector{Float64}, eta::Float64, atol::Float64 = 1e-14)
    """(3.6)"""
    b2_metro::Vector{ComplexF64} = (1/(4 * pi * sqrt(2))) * 
        exp.(-2 * time_labels.^2 .- 1im * time_labels) ./ (time_labels .* (2 * time_labels .+ 1im))

    # (1, 1, 1 (at eta), 0, 0, ..., 0)
    heaviside_eta = ifelse.(abs.(time_labels) .> eta, 0, ones(length(time_labels)))
    b2_metro .+= heaviside_eta .* (1im * (2 * time_labels .+ 1im) ./ (2 * time_labels .+ 1im))
    
    indices_b2_metro = get_truncated_indices_b(b2_metro, atol)
    b2_metro_times = time_labels[indices_b2_metro]
    b2_metro_vals = b2_metro[indices_b2_metro]

    return Dict(zip(b2_metro_times, b2_metro_vals))
end

function get_truncated_indices_b(b::Vector{ComplexF64}, atol::Float64 = 1e-14)
   """Find elements in b1, b2 that are larger than `atol`"""

    indices_b = findall(x -> abs(real(x)) >= atol || abs(imag(x)) >= atol, b)
    # @printf("Number of nonzero elements in b1: %d\n", length(indices_b1))
    # @printf("Number of nonzero elements in b2: %d\n", length(indices_b2))

    return indices_b
end

function check_B_gauss(coherent_term::Matrix{ComplexF64}, jump::JumpOp, hamiltonian::HamHam, beta::Float64)
    b = SpinBasis(1//2)^num_qubits
    trnorm = tracenorm_nh(Operator(b, coherent_term))

    B_integrated = coherent_term_timedomain_integrated_gauss(jump, hamiltonian, beta)
    deviation = norm(B_integrated - coherent_term)

    @printf("\nTracenorm of coherent term: %e\n", trnorm)
    @printf("Deviation from integral: %e\n", deviation)
end

function check_B_metro(coherent_term::Matrix{ComplexF64}, jump::JumpOp, hamiltonian::HamHam, eta::Float64, beta::Float64)
    b = SpinBasis(1//2)^num_qubits
    trnorm = tracenorm_nh(Operator(b, coherent_term))

    B_integrated = coherent_term_timedomain_integrated_metro(jump, hamiltonian, eta, beta)
    deviation = norm(B_integrated - coherent_term)

    @printf("\nTracenorm of coherent term: %e\n", trnorm)
    @printf("Deviation from integral: %e\n", deviation)
end

#* Testing
# num_qubits = 5
# beta = 10.
# eig_index = 3
# jump_site_index = 1

# # #* Hamiltonian
# hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n5.jld")["ideal_ham"]
# # hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3), batch_size=1)
# initial_state = hamiltonian.eigvecs[:, eig_index]
# hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

# # #* Jump operators
# sigmax::Matrix{ComplexF64} = [0 1; 1 0]
# jump_op = Matrix(pad_term([sigmax], num_qubits, jump_site_index))
# jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
# jump = JumpOp(jump_op,
#         jump_op_in_eigenbasis,
#         Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
#         zeros(0), 
#         zeros(0, 0))

# # #* Fourier labels
# # # num_energy_bits = ceil(Int64, log2((0.45 * 2) / hamiltonian.w0)) + 2 # paper (above 3.7.), later will be β dependent
# num_energy_bits = ceil(Int64, log2((0.45 * 4 + 2/beta) / hamiltonian.w0)) + 1  # For good integral approx we might need more r than expected
# #! r scales even worse for a good integral in system size... Maybe it is just not efficient to approximate the integral for B.
# #! This could be big, since they don't mention how well can we approximate the integral within a quantum computer!
# # # num_energy_bits = 9
# N = 2^(num_energy_bits)
# N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

# t0 = 2 * pi / (N * hamiltonian.w0)
# time_labels = t0 * N_labels
# ((maximum(time_labels) - minimum(time_labels)) / N)^2

# energy_labels = hamiltonian.w0 * N_labels

# @printf("Number of qubits: %d\n", num_qubits)
# @printf("Number of energy bits: %d\n", num_energy_bits)
# @printf("Energy unit: %e\n", hamiltonian.w0)
# @printf("Time unit: %e\n", t0)

# atol = 1e-12
# # atol = 0.0
# # time_labels = collect(-maximum(time_labels):t0:maximum(time_labels))  # symmetric but not necessary I think
# @time b1 = compute_truncated_b1(time_labels, atol)
# @time b2 = compute_truncated_b2(time_labels, atol)
# #
# @printf("Number of b1 terms kept: %d\n", length(b1))
# @printf("Number of b2 terms kept: %d\n", length(b2))
# b2 = Dict(zip(collect(keys(b1)), compute_b2(collect(keys(b1)))))

# show(stdout, "text/plain", sort(collect(keys(b1))))
# show(stdout, "text/plain", sort(collect(keys(b2))))

# @time B_explicit = coherent_term_from_timedomain(jump, hamiltonian, b1, b2, beta)
# @time B_integrated = coherent_term_timedomain_integrated(jump, hamiltonian, beta)

# norm(B_integrated - B_integrated')
# norm(B_explicit - B_explicit')

# norm(B_explicit) / norm(B_integrated)
# norm(B_explicit - B_integrated)

#* l1 norms of b1, b2
# f1(t) = 1 / cosh(2 * pi * t)
# f2(t) = sin(-t) * exp(-2 * t^2)
# abs_b1_fn(t) = abs(2 * sqrt(pi) * exp(1/8) * convolute.(Ref(f1), Ref(f2), t))
# b1_fn(t) = 2 * sqrt(pi) * exp(1/8) * convolute.(Ref(f1), Ref(f2), t)
# abs_b2_fn(t) = abs(exp(-4*t^2 - 2im*t) / sqrt(4*pi^3))
# b2_fn(t) = exp(-4*t^2 - 2im*t) / sqrt(4*pi^3)

# b2_fn(-2.) == conj(b2_fn(2.))

#* Comlex conjugates of b1, b2
# Dictionary is not keeping order, but it's correct
# For truncated case
# b1_vals = collect(values(b1))
# b2_vals = collect(values(b2))
# b2_neg = compute_truncated_b2(-time_labels, atol)
# collect(keys(b2))
# display(keys(b2_neg))
# display(values(b2_neg))
# b2_vals_negative_time = sort(collect(values(b2_neg)))
# b1_vals_conj = conj.(b1_vals)
# b2_vals_conj = conj.(b2_vals)
# for k in keys(b2_neg)
#     if k == -0.0
#         display(b2_neg[k] == conj(b2[-k]))
#     else
#         display(b2_neg[-k] == conj(b2[k]))
#     end
# end

# b1_vals == b1_vals_conj
# b2_vals_negative_time == b2_vals_conj
# norm(b2_vals_negative_time - b2_vals_conj)

# # True for the whole time labels
# b1_vals = b1_fn.(time_labels)
# b2_vals = b2_fn.(time_labels)
# b2_vals_negative_time = b2_fn.(-time_labels)
# b1_vals_conj = conj.(b1_vals)
# b2_vals_conj = conj.(b2_vals)

# b1_vals == b1_vals_conj
# b2_vals_negative_time == b2_vals_conj

# Print b1 b2 Plots
# times = collect(-5:t0:5)
# plot(times, real.(b1_fn.(times)), label="Real part b1")
# plot!(times, imag.(b1_fn.(times)), label="Imag part b1")
# plot(times, real.(b2_fn.(times)), label="Real part b2")
# plot!(times, imag.(b2_fn.(times)), label="Imag part b2")

# truncated_times_b1 = collect(keys(b1))
# truncated_times_b2 = collect(keys(b2))
# plot!(truncated_times_b1, real.(values(b1)), label="Real part b1 trunc")

# b1_integrated, _ = quadgk(abs_b1_fn, -Inf, Inf; atol=1e-12, rtol=1e-12)
# b2_integrated, _ = quadgk(abs_b2_fn, -Inf, Inf; atol=1e-12, rtol=1e-12, order=10)

# @printf("l1 norm of b1: %e\n", b1_integrated)
# @printf("l1 norm of b2: %e\n", b2_integrated)

# @time B_integrated = coherent_term_timedomain_integrated(jump, hamiltonian, beta)
# @printf("Deviation Hermiticity: %e\n", norm(B_integrated - B_integrated'))
# B_deviation_sum_integral = norm(B_integrated - B_explicit)
# @printf("Deviation sum integral: %e\n", B_deviation_sum_integral)

# # plot(time_labels, real.(b1), label="Real part")

# # * Coherent term in energy domain
# @time B_bohr = coherent_gaussian_bohr(hamiltonian, jump, energy_labels, beta)
# # @time B_explicit = coherent_bohr_explicit(hamiltonian, jump, beta)

# #* Coherent term in time domain
# b1_vals, b1_times = compute_truncated_b1(time_labels)
# b2_vals, b2_times = compute_truncated_b2(time_labels)

# @time B_t::Vector{Matrix{ComplexF64}} = coherent_term_from_timedomain.([jump], 
# Ref(hamiltonian), Ref(b1_vals), Ref(b1_times), Ref(b2_vals), Ref(b2_times), Ref(beta))

# #* Compare
# deviation = norm(B_bohr - B_t[1])
# b = SpinBasis(1//2)^num_qubits
# trdist = tracedistance(Operator(b, B_bohr), Operator(b, B_t[1]))
# @printf("Deviation: %e\n", deviation)

# function round_complex(z::ComplexF64, digits::Int64)
#     rounded_real = round(real(z), digits=digits)
#     rounded_imag = round(imag(z), digits=digits)
#     return rounded_real + rounded_imag*im
# end

# # find nonzero elements in b
# nonzero_indices = findall(!iszero, round_complex.(B_bohr, 6))
# length(nonzero_indices)
# minimum(imag.(B_bohr[nonzero_indices]))
