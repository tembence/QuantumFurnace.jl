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

num_qubits = 2
dim = 2^num_qubits
beta = 10.
eta = 1e-10
atol = 1e-12
rtol = 1e-12
Random.seed!(666)

hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=100)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
initial_dm = Matrix{ComplexF64}(I(dim) / dim)
@assert norm(real(tr(initial_dm)) - 1.) < 1e-15 "Trace is not 1.0"
@assert norm(initial_dm - initial_dm') < 1e-15 "Not Hermitian"

#* Jumps
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
H::Matrix{ComplexF64} = [1 1; 1 -1] / sqrt(2)
id::Matrix{ComplexF64} = I(2)
jump_paulis = [[X]]#, [Y], [Z]]

# All jumps once
all_jumps_generated::Vector{JumpOp} = []
for pauli in jump_paulis
    for site in 1:num_qubits
    jump_op = Matrix(pad_term(pauli, num_qubits, site))
    jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
    jump_in_trotter_basis = zeros(0, 0)
    orthogonal = (jump_op == transpose(jump_op))
    jump = JumpOp(jump_op,
            jump_op_in_eigenbasis,
            Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
            zeros(0),
            jump_in_trotter_basis,
            orthogonal) 
    push!(all_jumps_generated, jump)
    end
end

jump = all_jumps_generated[1]

#* Labels
num_energy_bits = 21
N = 2^(num_energy_bits)
w0 = 0.001
t0 = 2 * pi / (N * w0)
@printf("t0: %s\n", t0)
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
energy_labels = w0 * N_labels
@assert maximum(energy_labels) >= 2.0
time_labels = t0 * N_labels

#* Handmade times
t0 = 1e-1
@printf("Handmade t0: %f\n", t0)
handmade_times = [-50.0:t0:50.0;]
times = handmade_times
# times = time_labels

#* Functions
function riemann_sum(f::Function, time_labels::Vector{Float64}, args...)
    t0 = time_labels[2] - time_labels[1]
    return t0 * sum(t -> f(t, args...), time_labels)
end

function riemann_sum_stretched(f::Function, time_labels::Vector{Float64}, args...)
    t0 = time_labels[2] - time_labels[1]
    times = [(-1.0 + t0):t0:(1.0 - t0);]
    return t0 * sum(t -> f(t / (1-t^2), args...), times)
end

function riemann_sum(fvals::Vector{<:Number}, t0::Float64)
    return t0 * sum(fvals)
end

f_minus = compute_f_minus.(times, beta)
f_plus_metro = compute_f_plus_metro.(times, eta, beta)

riemann_summed_f_plus_metro = riemann_sum(compute_f_plus_metro, times, eta, beta)
integrated_f_plus_metro, errp = quadgk(t -> compute_f_plus_metro(t, eta, beta), -Inf, Inf; atol=atol, rtol=rtol)

@printf("F plus, Riemann - Integrated\n")
display(norm(riemann_summed_f_plus_metro - integrated_f_plus_metro))

# diag_time_evolve(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))
# f_plus_summand(s) = (compute_f_plus_metro(s, eta, beta) * 
#     diag_time_evolve(s) * jump.in_eigenbasis' * diag_time_evolve(-2.0 * s) * jump.in_eigenbasis * diag_time_evolve(s))
w = 0.12
test_t = 0.1
diag_time_evolve_w(t) = exp(1im * w * t)
f_plus_summand_w(s) = compute_f_plus_metro(s, eta, beta) * diag_time_evolve_w(s)
f_plus_summand(s) = compute_f_plus_metro(s, eta, beta)

f_plus_summand_w(test_t)
f_plus_summand(test_t)
f_plus_summand(test_t) * diag_time_evolve_w(test_t)
diag_time_evolve_w(test_t)
plot(times, real.(f_plus_summand_w.(times)))
plot!(times, real.(f_plus_summand.(times)))
plot(times, imag.(f_plus_summand_w.(times)))
plot(times, imag.(f_plus_summand.(times)))

riemann_summed_inner = riemann_sum(f_plus_summand_w, times)
riemann_summed_inner_stretch = riemann_sum_stretched(f_plus_summand_w, times)
integrated_inner, errpp = quadgk(f_plus_summand_w, -Inf, Inf; atol=atol, rtol=rtol)
@printf("Full F plus integrand fn, Riemann - Integrated\n")
display(norm(riemann_summed_inner - integrated_inner))
display(norm(riemann_summed_inner_stretch - integrated_inner))


# riemann_summed_f_minus = riemann_sum(compute_f_minus, times, beta)
# integrated_f_minus, errm = quadgk(t -> compute_f_minus(t, beta), -Inf, Inf;
#     atol=atol, rtol=rtol)
# norm(riemann_summed_f_minus - integrated_f_minus)

# diag_time_evolve(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))
# f_plus_inegrand(s) = (compute_f_plus_metro(s, eta, beta)
#            * diag_time_evolve(s) * jump.in_eigenbasis' * diag_time_evolve(-2 * s) * jump.in_eigenbasis * diag_time_evolve(s))

# f_plus_integral_inf, _ = quadgk(f_plus_inegrand, -Inf, Inf; atol=atol, rtol=rtol)
# f_plus_integral, _ = quadgk(f_plus_inegrand, time_domain[1], time_domain[2]; atol=atol, rtol=rtol)
# norm(f_plus_integral_inf - f_plus_integral)


#* B --- 
f_minus_truncated = compute_truncated_f_minus(times, beta)
f_plus_metro_truncated = compute_truncated_f_plus_metro(times, eta, beta)

B_bohr_metro = coherent_metro_bohr(hamiltonian, bohr_dict, jump, beta) 
B_time_metro_f = coherent_term_time_metro_f(jump, hamiltonian, f_minus_truncated, f_plus_metro_truncated, t0)
B_time_metro_f_integrated = coherent_term_time_integrated_metro_f(jump, hamiltonian, eta, beta; time_domain=(-50., 50.))

# norm(B_bohr_metro)
@printf("Bohr - Time\n")
display(norm(B_bohr_metro - B_time_metro_f))
@printf("Bohr - Integrated time\n")
display(norm(B_bohr_metro - B_time_metro_f_integrated))

# Quadrature error (most basic)
# t0 * (maximum(time_labels) - minimum(time_labels)) / 2

# f derivatives checks for quadrature error bound
# f_minus_deriv = zeros(ComplexF64, N)
# f_plus_deriv = zeros(ComplexF64, N)

# for i in 2:(N - 1)
#     f_minus_deriv[i] = (compute_f_minus(time_labels_decimal[i + 1], beta) 
#                                 -  compute_f_minus(time_labels_decimal[i - 1], beta)) / (2 * t0)
#     f_plus_deriv[i] = (compute_f_plus_metro(time_labels_decimal[i + 1], eta, beta) 
#                                 -  compute_f_plus_metro(time_labels_decimal[i - 1], eta, beta)) / (2 * t0)
# end

# f_minus_deriv[1] = (compute_f_minus(time_labels_decimal[2], beta) 
#                                 - compute_f_minus(time_labels_decimal[1], beta)) / t0
# f_minus_deriv[N] = (compute_f_minus(time_labels_decimal[N], beta) 
#                                 - compute_f_minus(time_labels_decimal[N - 1], beta)) / t0
# f_plus_deriv[1] = (compute_f_plus_metro(time_labels_decimal[2], eta, beta) 
#                                 - compute_f_plus_metro(time_labels_decimal[1], eta, beta)) / t0
# f_plus_deriv[N] = (compute_f_plus_metro(time_labels_decimal[N], eta, beta) 
#                                 - compute_f_plus_metro(time_labels_decimal[N - 1], eta, beta)) / t0

# @printf("Maximum derivative f plus abs: %s\n", maximum(abs.(f_plus_deriv)))
# @printf("Maximum derivative f minus abs: %s", maximum(abs.(f_minus_deriv)))

# my_t = [-5:0.01:5.0;]
# plot(my_t, imag.(compute_f_plus_metro.(my_t, eta, beta)))
# # plot(time_labels_decimal, real.(f_minus_deriv), label="f min RE")
# # plot!(time_labels_decimal, imag.(f_minus_deriv), label="f min IM")
# plot!(time_labels_decimal, real.(f_plus_deriv), label="f plus RE")
# plot(time_labels_decimal, imag.(f_plus_deriv), label="f plus IM")