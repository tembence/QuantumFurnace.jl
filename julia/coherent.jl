using Random
using LinearAlgebra
using Printf
using ProgressMeter
using Distributed
using JLD
using Plots
using QuadGK
using SparseArrays

include("hamiltonian.jl")
include("ofts.jl")
include("qi_tools.jl")

#* COHERENT TERMS -----------------------------------------------------------------------------------------------------------
# (3.1) and Proposition III.1
# Has to be on a symmetric time domain, otherwise it can't be Hermitian.
function coherent_term_time(jump::JumpOp, hamiltonian::HamHam, f_minus::Dict{Float64, ComplexF64}, 
    f_plus::Dict{Float64, ComplexF64}, t0::Float64)
    
    dim = size(hamiltonian.data)
    diag_time_evolve(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))

    # Inner summand f_plus
    f_plus_summand = zeros(ComplexF64, dim)
    for s in keys(f_plus)
        U_s = diag_time_evolve(s)
        U_minus_2s = diag_time_evolve(-2.0 * s)
        f_plus_summand .+= f_plus[s] * U_s * jump.in_eigenbasis' * U_minus_2s * jump.in_eigenbasis * U_s
    end

    # Outer summand f_minus
    # A is Hermitian
    B = zeros(ComplexF64, dim)
    for t in keys(f_minus)
        U_t = diag_time_evolve(t)
        B .+= f_minus[t] * U_t' * f_plus_summand * U_t
    end

    # If A is non-Hermitian
    # for t in keys(f_minus)
    #     U_t = diag_time_evolve(t)
    #     B .+= f_minus[t] * U_t' * (f_plus_summand * t0 + jump.in_eigenbasis' * jump.in_eigenbasis / (2pi * sqrt(2))) * U_t
    # end
    return B * t0^2
end

function coherent_term_trotter(jump::JumpOp, trotter::TrottTrott, 
    f_minus::Dict{Float64, ComplexF64}, f_plus::Dict{Float64, ComplexF64})

    dim = size(trotter.eigvecs)
    trotter_time_evolution(n::Int64) = Diagonal(trotter.eigvals_t0 .^ n)  # n - number of t0 time chunks

    # Inner summand f_plus
    f_plus_summand = zeros(ComplexF64, dim)
    for s in keys(f_plus)
        num_t0_steps = Int(round(s / trotter.t0))

        trott_U_s = trotter_time_evolution(num_t0_steps)
        trott_U_2s = trotter_time_evolution(2 * num_t0_steps)

        f_plus_summand .+= (f_plus[s] *
            trott_U_s * jump.in_trotter_basis' * trott_U_2s' * jump.in_trotter_basis * trott_U_s)
    end
    B = zeros(ComplexF64, dim)
    for t in keys(f_minus)
        num_t0_steps = Int(round(t / trotter.t0))
        trott_U_t = trotter_time_evolution(num_t0_steps)

        B .+= f_minus[t] * trott_U_t' * f_plus_summand * trott_U_t
    end

    # Outer summand f_minus
    # A is Hermitian (if A is non-Hermitian, see coherent_term_times
    return B * t0^2  # B in Trotter basis
end

function coherent_term_time_b(jump::JumpOp, hamiltonian::HamHam, 
    b1::Dict{Float64, ComplexF64}, b2::Dict{Float64, ComplexF64}, t0::Float64, beta::Float64)
    """Coherent term for the Gaussian AND Metropolis case IF jump op is [A Adag, H] = 0 (X, Y, Z)
    written in timedomain and using ideal time evolution.
    Working in the energy eigenbasis for the time evolutions, where H is diagonal.
        sigma_E = sigma_gamma = w_gamma = 1 / beta"""
    
    diag_time_evolve(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))

    # Inner b2 sum
    b2_integral = zeros(ComplexF64, size(hamiltonian.data))
    for s in keys(b2)
        time_evolution_inner = diag_time_evolve(beta * s)
        b2_integral .+= b2[s] * (time_evolution_inner * jump.in_eigenbasis' 
                            * (time_evolution_inner')^2 * jump.in_eigenbasis * time_evolution_inner)
    end

    # Outer b1 sum
    B = zeros(ComplexF64, size(hamiltonian.data))
    for t in keys(b1)
        time_evolution_outer = diag_time_evolve(beta * t)
        B .+= b1[t] * time_evolution_outer' * b2_integral * time_evolution_outer
    end
    return B * t0^2  # Correction in b2 already
end


function coherent_term_time_integrated_metro(jump::JumpOp, hamiltonian::HamHam, eta::Float64, beta::Float64; 
    time_domain::Tuple{Float64, Float64} = (-Inf, Inf), atol=1e-12, rtol=1e-12)

    diag_time_evolve(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))
    f_plus_inegrand(s) = (compute_f_plus_metro(s, eta, beta) * 
            diag_time_evolve(s) * jump.in_eigenbasis' * diag_time_evolve(-2 * s) * jump.in_eigenbasis * diag_time_evolve(s))

    f_plus_integral, _ = quadgk(f_plus_inegrand, time_domain[1], time_domain[2]; atol=atol, rtol=rtol)

    f_minus_integrand(t) = (compute_f_minus(t, beta) * diag_time_evolve(-t) 
                            * (f_plus_integral + jump.in_eigenbasis' * jump.in_eigenbasis / (2pi * sqrt(2))) 
                            * diag_time_evolve(t))
    B, _ = quadgk(f_minus_integrand, time_domain[1], time_domain[2]; atol=atol, rtol=rtol)

    return B
end

function coherent_term_time_metro_exact(jump::JumpOp, hamiltonian::HamHam, time_labels::Vector{Float64}, beta::Float64)

    dim = size(hamiltonian.data)
    t0 = time_labels[2] - time_labels[1]
    time_labels_no_zero = time_labels[2:end]
    diag_time_evolve(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))

    f_minus = compute_truncated_f_minus(time_labels_no_zero, beta)
    # Oh = construct_metro_oh(jump, hamiltonian, time_labels_no_zero, beta) #!
    Oh = construct_metro_oh_integrated(jump, hamiltonian, beta)

    B = zeros(ComplexF64, dim)
    for t in keys(f_minus)
        U_t = diag_time_evolve(t)
        B .+= f_minus[t] * U_t' * Oh * U_t
    end

    return t0 * B
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
    b2_integrand(s) = b2_fn(s) * (diag_time_evolve(beta * s) * 
                           (jump_op_in_eigenbasis_dag * diag_time_evolve(- 2 * beta * s) * jump.in_eigenbasis) *
                           diag_time_evolve(beta * s))
    b2_integral, _ = quadgk(b2_integrand, time_domain[1], time_domain[2]; atol=1e-12, rtol=1e-12)

    # Outer b1 integral
    b1_integrand(t) = b1_fn(t) * diag_time_evolve(- beta * t) * b2_integral * diag_time_evolve(beta * t)
    B, _ = quadgk(b1_integrand, time_domain[1], time_domain[2]; atol=1e-12, rtol=1e-12)
    return B
end

function coherent_term_timedomain_integrated_metro(jump::JumpOp, hamiltonian::HamHam, eta::Float64, beta::Float64, 
    time_domain::Tuple{Float64, Float64} = (-Inf, Inf))

    f1(t) = 1 / cosh(2 * pi * t)
    f2(t) = sin(-t) * exp(-2 * t^2)
    b1_fn(t) = 2 * sqrt(pi) * exp(1/8) * convolute(Ref(f1), Ref(f2), t)
    b2_metro_fn(t) = (exp(-2 * t^2 - im * t) + (abs(t) <= eta ? 1 : 0) * im * (2 * t + im)) / (sqrt(32) * pi * t * (2 * t + im))
    diag_time_evolve(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))
    jump_op_in_eigenbasis_dag = jump.in_eigenbasis'

    # Inner b2 integral
    b2_integrand(s) = (b2_metro_fn(s) * diag_time_evolve(beta * s) * 
                           (jump_op_in_eigenbasis_dag * diag_time_evolve(- 2 * beta * s) * jump.in_eigenbasis) *
                           diag_time_evolve(beta * s))
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

#* B1 AND B2 ----------------------------------------------------------------------------------------------------------------
# Corollary III.1, every parameter = 1 / beta
function compute_f_minus(t::Float64, beta::Float64)
    """For all cases the same"""
    f1(t) = 1 / cosh(2 * pi * t / beta)
    f2(t) = sin(-t / beta) * exp(-2 * t^2 / beta^2)
    return (1 / (pi * beta^2)) * exp(1/8) * convolute(f1, f2, t)
end

function compute_f_plus(t::Float64, beta::Float64)
    """Gaussian case"""
    return 2 * exp(-4 * t^2 / beta^2 - 2im * t / beta) / beta
end

function compute_f_plus_eh(t::Float64, beta::Float64, a::Float64, b::Float64)
    """b = 0: Metro, b = 2: Glauber"""
    f = exp(-t * (2t + 1im * beta) * (1 + b) / beta^2 - a * b / (2 * beta)) / (4 * t^2 + a * beta + 2im * t * beta)
    return 2 * beta * sqrt(4 * a / beta + 1) * f / sqrt(2pi)
end

# Actually faster broadcasting the whole function than taking in a vector argument
function compute_f_plus_metro(t::Float64, beta::Float64, eta::Float64)

    if abs(t) < 1e-12  # Handle t=0
        return complex(sqrt(1 / 2pi) / beta) 
    elseif abs(t) ≤ eta
        numerator = exp(-2 * t^2 / beta^2 - 1im * t / beta) + 1im * (2 * t / beta + 1im)
    else
        numerator = exp(-2 * t^2 / beta^2 - 1im * t / beta)
    end
    denominator = t * (2 * t / beta + 1im) / beta
    return (sqrt(1 / 2pi) / beta) * numerator / denominator
end

function compute_truncated_f_minus(time_labels::Vector{Float64}, beta::Float64; atol::Float64 = 1e-12)
    f_minus = Vector{ComplexF64}(compute_f_minus.(time_labels, beta))
    # Skip all elements where b1 b2 are smaller than 1e-12
    indices_f_minus = get_truncated_indices(f_minus; atol=atol)
    return Dict(zip(time_labels[indices_f_minus], f_minus[indices_f_minus]))
end

function compute_truncated_f_plus(time_labels::Vector{Float64}, beta::Float64; atol::Float64 = 1e-12)
    f_plus = Vector{ComplexF64}(compute_f_plus.(time_labels, beta))
    indices_f_plus = get_truncated_indices(f_plus; atol=atol)
    return Dict(zip(time_labels[indices_f_plus], f_plus[indices_f_plus]))
end

function compute_truncated_f_plus_eh(time_labels::Vector{Float64}, beta::Float64, a::Float64, b::Float64; atol::Float64 = 1e-12)
    f_plus = Vector{ComplexF64}(compute_f_plus_eh.(time_labels, beta, a, b))
    good_indices = get_truncated_indices(f_plus; atol=atol)
    return Dict(zip(time_labels[good_indices], f_plus[good_indices]))
end

function compute_truncated_f_plus_metro(time_labels::Vector{Float64}, beta::Float64, eta::Float64; atol::Float64 = 1e-12)
    f_plus = Vector{ComplexF64}(compute_f_plus_metro.(time_labels, beta, eta))
    good_indices = get_truncated_indices(f_plus; atol=atol)
    return Dict(zip(time_labels[good_indices], f_plus[good_indices]))
end

function compute_b1(time_labels::Vector{Float64})
    f1(t) = 1 / cosh(2 * pi * t)
    f2(t) = sin(-t) * exp(-2 * t^2)
    return 2 * sqrt(pi) * exp(1/8) * convolute.(Ref(f1), Ref(f2), time_labels)
end

function compute_truncated_b1(time_labels::Vector{Float64}; atol::Float64 = 1e-14)

    b1 = Vector{ComplexF64}(compute_b1(time_labels))

    # Skip all elements where b1 b2 are smaller than 1e-14
    indices_b1 = get_truncated_indices(b1; atol)
    b1_times = time_labels[indices_b1]
    b1_vals = b1[indices_b1]
    
    return Dict(zip(b1_times, b1_vals))
end

function compute_b2(time_labels::Vector{Float64})
    return 2 * exp.(- 4 * time_labels.^2 .- 2 * im * time_labels) / sqrt(4 * pi^3)  # Corrected
end

function compute_truncated_b2(time_labels::Vector{Float64}; atol::Float64 = 1e-14)

    b2 = Vector{ComplexF64}(compute_b2(time_labels))

    # Skip all elements where b1 b2 are smaller than 1e-14
    indices_b2 = get_truncated_indices(b2; atol)
    b2_times = time_labels[indices_b2]
    b2_vals = b2[indices_b2]

    return Dict(zip(b2_times, b2_vals))
end

#* TOOLS --------------------------------------------------------------------------------------------------------------------
function get_truncated_indices(fvals::Vector{Float64}; atol::Float64 = 1e-12)
   """Find elements in `fvals` that are larger than `atol`"""
    return findall(abs.(fvals) .>= atol)
end

function get_truncated_indices(fvals::Vector{ComplexF64}; atol::Float64 = 1e-12)
    """Find elements in `fvals` that are larger than `atol`"""
     return findall(abs.(fvals) .>= atol)
 end

function convolute(f::Function, g::Function, t::Float64; atol=1e-12, rtol=1e-12)
    integrand(s) = f(s) * g(t - s)
    result, _ = quadgk(integrand, -Inf, Inf; atol=atol, rtol=rtol)
    return result
end

function check_B_gauss(coherent_term::Matrix{ComplexF64}, jump::JumpOp, hamiltonian::HamHam, beta::Float64)
    trnorm = trace_norm_nh(coherent_term)
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
#* --------------------------------------------------------------------------------------------------------------------------
#* --------------------------------------------------------------------------------------------------------------------------

#* Testing
# num_qubits = 6
# beta = 10.
# eig_index = 3
# jump_site_index = 1

# # #* Hamiltonian
# hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n6.jld")["ideal_ham"]
# # hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3), batch_size=1)
# initial_state = hamiltonian.eigvecs[:, eig_index]
# hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

# # # #* Fourier labels
# num_energy_bits = ceil(Int64, log2((0.45 * 4 + 2/beta) / hamiltonian.nu_min)) + 3  # For good integral approx we might need more r than expected
# # num_energy_bits = 11
# N = 2^(num_energy_bits)
# N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
# t0 = 2 * pi / (N * hamiltonian.nu_min)
# time_labels = t0 * N_labels
# energy_labels = hamiltonian.nu_min * N_labels

# #* Trotter
# #TODO: Where is the Trotter scaling? Why does it get worse if r is higher = smaller t0, but more B terms.
# num_trotter_steps_per_t0 = 100
# trotter = create_trotter(hamiltonian, t0, num_trotter_steps_per_t0)
# trotter_error_T = compute_trotter_error(hamiltonian, trotter, 2^num_energy_bits * t0)
# @printf("Num trotter steps / t0: %d\n", num_trotter_steps_per_t0)
# @printf("Max order Trotter error on an OFT: %s\n", trotter_error_T)

# #* Jump operators
# X::Matrix{ComplexF64} = [0 1; 1 0]
# Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
# Z::Matrix{ComplexF64} = [1 0; 0 -1]
# jump_paulis = [[X], [Y], [Z]]

# all_jumps_generated::Vector{JumpOp} = []
# for pauli in jump_paulis
#     for site in 1:num_qubits
#     jump_op = Matrix(pad_term(pauli, num_qubits, site))
#     jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
#     jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
#     orthogonal = (jump_op == adjoint(jump_op))
#     jump = JumpOp(jump_op,
#             jump_op_in_eigenbasis,
#             Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
#             zeros(0),
#             jump_in_trotter_basis,
#             orthogonal) 
#     push!(all_jumps_generated, jump)
#     end
# end

# @printf("Number of qubits: %d\n", num_qubits)
# @printf("Number of energy bits: %d\n", num_energy_bits)
# @printf("Energy unit: %e\n", hamiltonian.nu_min)
# @printf("Time unit: %e\n", t0)

# atol = 1e-12
# # atol = 0.0
# # time_labels = collect(-maximum(time_labels):t0:maximum(time_labels))  # symmetric but not necessary I think
# @time b1 = compute_truncated_b1(time_labels, atol)
# @time b2 = compute_truncated_b2(time_labels, atol)
# @printf("Number of b1 terms kept: %d\n", length(b1))
# @printf("Number of b2 terms kept: %d\n", length(b2))

# jump = all_jumps_generated[10]
# @time B_explicit = coherent_term_from_timedomain(jump, hamiltonian, b1, b2, t0, beta)
# @time B_bohr = coherent_gaussian_bohr(hamiltonian, jump, beta)
# @time B_bohr_2 = coherent_gaussian_bohr_slow(hamiltonian, jump, beta)
# @time B_integrated = coherent_term_timedomain_integrated_gauss(jump, hamiltonian, beta)
# @time B_trotter = coherent_term_trotter(jump, hamiltonian, trotter, b1, b2, beta)
# B_trotter_in_eigenbasis = hamiltonian.eigvecs' * trotter.eigvecs * B_trotter * trotter.eigvecs' * hamiltonian.eigvecs

# @printf("HS dist of bohr and bohr 2: %s\n", norm(B_bohr - B_bohr_2))
# @printf("HS distance of bohr and explicit: %s\n", norm(B_bohr - B_explicit))
# @printf("HS distance of bohr and trotter: %s\n", norm(B_bohr - B_trotter_in_eigenbasis))
# @printf("HS distance of trotter and explicit: %s\n", norm(B_trotter_in_eigenbasis - B_explicit))

# #* l1 norms of b1, b2
# # f1(t) = 1 / cosh(2 * pi * t)
# # f2(t) = sin(-t) * exp(-2 * t^2)
# # abs_b1_fn(t) = abs(2 * sqrt(pi) * exp(1/8) * convolute.(Ref(f1), Ref(f2), t))
# # b1_fn(t) = 2 * sqrt(pi) * exp(1/8) * convolute.(Ref(f1), Ref(f2), t)
# # abs_b2_fn(t) = abs(exp(-4*t^2 - 2im*t) / sqrt(4*pi^3))
# # b2_fn(t) = exp(-4*t^2 - 2im*t) / sqrt(4*pi^3)

# # b2_fn(-2.) == conj(b2_fn(2.))

# #* Comlex conjugates of b1, b2
# # Dictionary is not keeping order, but it's correct
# # For truncated case
# # b1_vals = collect(values(b1))
# # b2_vals = collect(values(b2))
# # b2_neg = compute_truncated_b2(-time_labels, atol)
# # collect(keys(b2))
# # display(keys(b2_neg))
# # display(values(b2_neg))
# # b2_vals_negative_time = sort(collect(values(b2_neg)))
# # b1_vals_conj = conj.(b1_vals)
# # b2_vals_conj = conj.(b2_vals)
# # for k in keys(b2_neg)
# #     if k == -0.0
# #         display(b2_neg[k] == conj(b2[-k]))
# #     else
# #         display(b2_neg[-k] == conj(b2[k]))
# #     end
# # end

# # b1_vals == b1_vals_conj
# # b2_vals_negative_time == b2_vals_conj
# # norm(b2_vals_negative_time - b2_vals_conj)

# # # True for the whole time labels
# # b1_vals = b1_fn.(time_labels)
# # b2_vals = b2_fn.(time_labels)
# # b2_vals_negative_time = b2_fn.(-time_labels)
# # b1_vals_conj = conj.(b1_vals)
# # b2_vals_conj = conj.(b2_vals)

# # b1_vals == b1_vals_conj
# # b2_vals_negative_time == b2_vals_conj

# # Print b1 b2 Plots
# # times = collect(-5:t0:5)
# # plot(times, real.(b1_fn.(times)), label="Real part b1")
# # plot!(times, imag.(b1_fn.(times)), label="Imag part b1")
# # plot(times, real.(b2_fn.(times)), label="Real part b2")
# # plot!(times, imag.(b2_fn.(times)), label="Imag part b2")

# # truncated_times_b1 = collect(keys(b1))
# # truncated_times_b2 = collect(keys(b2))
# # plot!(truncated_times_b1, real.(values(b1)), label="Real part b1 trunc")

# # b1_integrated, _ = quadgk(abs_b1_fn, -Inf, Inf; atol=1e-12, rtol=1e-12)
# # b2_integrated, _ = quadgk(abs_b2_fn, -Inf, Inf; atol=1e-12, rtol=1e-12, order=10)

# # @printf("l1 norm of b1: %e\n", b1_integrated)
# # @printf("l1 norm of b2: %e\n", b2_integrated)

# # @time B_integrated = coherent_term_timedomain_integrated(jump, hamiltonian, beta)
# # @printf("Deviation Hermiticity: %e\n", norm(B_integrated - B_integrated'))
# # B_deviation_sum_integral = norm(B_integrated - B_explicit)
# # @printf("Deviation sum integral: %e\n", B_deviation_sum_integral)

# # # plot(time_labels, real.(b1), label="Real part")

# # # * Coherent term in energy domain
# # @time B_bohr = coherent_gaussian_bohr(hamiltonian, jump, energy_labels, beta)
# # # @time B_explicit = coherent_bohr_explicit(hamiltonian, jump, beta)

# # #* Coherent term in time domain
# # b1_vals, b1_times = compute_truncated_b1(time_labels)
# # b2_vals, b2_times = compute_truncated_b2(time_labels)

# # @time B_t::Vector{Matrix{ComplexF64}} = coherent_term_from_timedomain.([jump], 
# # Ref(hamiltonian), Ref(b1_vals), Ref(b1_times), Ref(b2_vals), Ref(b2_times), Ref(beta))

# # #* Compare
# # deviation = norm(B_bohr - B_t[1])
# # b = SpinBasis(1//2)^num_qubits
# # trdist = tracedistance(Operator(b, B_bohr), Operator(b, B_t[1]))
# # @printf("Deviation: %e\n", deviation)

# # function round_complex(z::ComplexF64, digits::Int64)
# #     rounded_real = round(real(z), digits=digits)
# #     rounded_imag = round(imag(z), digits=digits)
# #     return rounded_real + rounded_imag*im
# # end

# # # find nonzero elements in b
# # nonzero_indices = findall(!iszero, round_complex.(B_bohr, 6))
# # length(nonzero_indices)
# # minimum(imag.(B_bohr[nonzero_indices]))
