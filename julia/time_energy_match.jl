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

ENV["COLUMNS"] = "128"
ENV["ROWS"] = "128"

# (n, r, w0, tmix) = (3, 10, 32/N, 15.0) -> 1e-13
#* Configs
num_qubits = 2
dim = 2^num_qubits
num_energy_bits = 14
beta = 10.
Random.seed!(666)
with_coherent = true

# Config for algorithmic thermalization
mixing_time = 15.0
delta = 0.01

#* Hamiltonian
# hamiltonian_terms = [["X", "X"], ["Z"]]
# hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
# hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits)
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=100)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
initial_dm = Matrix{ComplexF64}(I(dim) / dim)
@assert norm(real(tr(initial_dm)) - 1.) < 1e-15 "Trace is not 1.0"
@assert norm(initial_dm - initial_dm') < 1e-15 "Not Hermitian"

N = 2^(num_energy_bits)
# w0 = 32 / N  #! Increasing this not always makes OFT better, why?
w0 = 0.001
t0 = 2 * pi / (N * w0)
@printf("Smallest Bohr frequency: %s\n", hamiltonian.nu_min)
@printf("Chosen w0: %s\n", w0)
# w0 = hamiltonian.nu_min
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
energy_labels = w0 * N_labels
@assert maximum(energy_labels) >= 2.0
time_labels = t0 * N_labels

#* Energy labels truncation

gauss_energy_integrand(w, nu_1, nu_2, beta) = ((beta / sqrt(2*pi)) * exp(-beta^2 * (w + 1/beta)^2 /2)
                                                    * exp(-beta^2 * (w - nu_1)^2 / 4) * exp(-beta^2 * (w - nu_2)^2 / 4))
metro_energy_integrand(w, nu_1, nu_2, beta) = ((beta / sqrt(2*pi)) *  exp(-beta * max(w + 1/(2*beta), 0.0))
                                                    * exp(-beta^2 * (w - nu_1)^2 / 4) * exp(-beta^2 * (w - nu_2)^2 / 4))

truncated_energies = truncate_energy_labels(energy_labels, metro_energy_integrand, beta)

#* Jumps
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
H::Matrix{ComplexF64} = [1 1; 1 -1] / sqrt(2)
id::Matrix{ComplexF64} = I(2)
jump_paulis = [[X]]#, [Y], [Z]] #! Identity

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

#* Thermalization
# results = @time thermalize_gauss(all_jumps_generated, hamiltonian, initial_dm, energy_labels, with_coherent, 
# delta, mixing_time, beta)
# results_time = @time thermalize_gauss_time(all_jumps_generated, hamiltonian, initial_dm, energy_labels, time_labels,
# with_coherent, delta, mixing_time, beta)

# evolved_dm = results.evolved_dm
# evolved_dm_time = results_time.evolved_dm
# @printf("Difference between evolved dms: %s\n", norm(evolved_dm - evolved_dm_time))

# @printf("Last distance to Gibbs in TIME: %s\n", results_time.distances_to_gibbs[end])
# @printf("Last distance to Gibbs: %s\n", results.distances_to_gibbs[end])

#* Time vs Bohr Liouvillians
# with_coherent = false
# liouv_bohr = @time construct_liouvillian_bohr_gauss(all_jumps_generated, hamiltonian, with_coherent, beta)
# liouv_time = @time construct_liouvillian_time_gauss(all_jumps_generated, hamiltonian, time_labels, truncated_energies, 
# with_coherent, beta)
# @printf("Difference between Liouvillians (GAUSS): %s\n", norm(liouv_bohr - liouv_time))

# eta = 0.001
# liouv_bohr_metro = @time construct_liouvillian_bohr_metro(all_jumps_generated, hamiltonian, with_coherent, beta)
# liouv_metro = @time construct_liouvillian_metro(all_jumps_generated, hamiltonian, truncated_energies, with_coherent, beta)
# liouv_metro_truncated = @time construct_liouvillian_metro(all_jumps_generated, hamiltonian, truncated_energies, 
#     with_coherent, beta)
# liouv_time_metro = @time construct_liouvillian_time_metro(all_jumps_generated, hamiltonian, time_labels, truncated_energies, 
#     with_coherent, beta, eta)
# @printf("Difference between Bohr - Time (METRO): %s\n", norm(liouv_bohr_metro - liouv_time_metro))
# @printf("Difference between Energy - Time (METRO): %s\n", norm(liouv_metro - liouv_time_metro))
# @printf("Difference between Energy - Bohr (METRO): %s\n", norm(liouv_metro - liouv_bohr_metro))
# @printf("Difference between Energy (TRUNCATED) - Bohr (METRO): %s\n", norm(liouv_metro_truncated - liouv_bohr_metro))
# @printf("Difference between Energy - Energy (TRUNCATED): %s\n", norm(liouv_metro - liouv_metro_truncated))

#* Transition part
# T_energy_metro = @time transition_metro(all_jumps_generated, hamiltonian, truncated_energies, beta)
# T_bohr_metro = @time transition_bohr_metro(all_jumps_generated, hamiltonian, beta)
# T_time_metro = @time transition_time_metro(all_jumps_generated, hamiltonian, time_labels, truncated_energies, beta)
# @printf("Distance between Bohr and Energy pictures (T METRO): %s\n", norm(T_bohr_metro - T_energy_metro))
# @printf("Distance between Bohr and Time pictures (T METRO): %s\n", norm(T_bohr_metro - T_time_metro))
# @printf("Distance between Time and Energy pictures (T METRO): %s\n", norm(T_time_metro - T_energy_metro))

#* Coherent term
# B1 integral
num_energy_bits = 20
N = 2^(num_energy_bits)
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
sorted_N_labels = [-Int(N/2):1:-1; 0:1:Int(N/2)-1]
jump = all_jumps_generated[2]
bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
eta = 0.02
t0 = 0.00006
time_labels = t0 * N_labels
time_labels_no_zero = time_labels[2:end]
sorted_time_labels_no_zero = sort(time_labels_no_zero)
maximum(time_labels)
b1 = compute_truncated_b1(time_labels)
b2 = compute_truncated_b2(time_labels)
b2_metro = compute_truncated_b2_metro(time_labels, eta)  #! Needs max(t) ~ 30

# Gauss
# B_bohr = coherent_gaussian_bohr(hamiltonian, bohr_dict, jump, beta)
# B_time = coherent_term_time(jump, hamiltonian, b1, b2, t0, beta)
# @printf("Difference between coherent terms: %s\n", norm(B_bohr - B_time))

# Oh = construct_metro_oh(jump, hamiltonian, time_labels[2:end], beta)
# diag_time_evolve(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))
# U_minus2t = diag_time_evolve(-2.0 * t) 
# norm((0.0 + 1.0im) * U_minus2t / (sqrt(8 * pi) * t))
# norm(exp(-2 * t^2 / beta^2 - 1im * t / beta) * U_minus2t / (sqrt(8 * pi) * t * (2 * t / beta + 1im)))
# theta = 0.1
# time_labels_no_zero_in_theta = sorted_time_labels_no_zero[abs.(sorted_time_labels_no_zero) .<= theta]
# snippet(t) = exp(-2 * t^2 / beta^2 - 1im * t / beta) / (sqrt(8 * pi) * t * (2 * t / beta + 1im))
# snippet_values = snippet.(sorted_time_labels_no_zero)
# relevant_time_labels = sorted_time_labels_no_zero[norm.(snippet.(sorted_time_labels_no_zero)) .>= 1e-10]

# snippet_values_theta = snippet.(time_labels_no_zero_in_theta)
# norm(imag.(snippet_values_theta))

# Oh_integrated = construct_metro_oh_integrated(jump, hamiltonian, beta)
# Oh = construct_metro_oh(jump, hamiltonian, time_labels_no_zero, beta)
# norm(Oh - Oh_integrated)

# Metro
B_bohr_metro = @time coherent_metro_bohr(hamiltonian, bohr_dict, jump, beta)
# norm(B_bohr_metro)
B_time_metro = @time coherent_term_time_metro(jump, hamiltonian, b1, b2_metro, t0, beta)
# B_time_metro_exact = @time coherent_term_time_metro_exact(jump, hamiltonian, time_labels, beta)
B_time_metro_integrated = @time coherent_time_metro_integrated(jump, hamiltonian, time_labels, beta, eta)
# norm(B_time_metro)
# norm(B_time_metro_exact)
@printf("Difference between Metro coherent terms (Bohr vs Approx): %s\n", norm(B_bohr_metro - B_time_metro))
# @printf("Difference between coherent terms (EXKT METRO): %s\n", norm(B_bohr_metro - B_time_metro_exact))
@printf("Difference between Metro coherent terms (Bohr vs Integrated approx): %s\n", norm(B_bohr_metro - B_time_metro_integrated))
norm(B_time_metro_integrated - B_time_metro)
norm(B_time_metro - B_time_metro_exact)
# norm(B_time_metro_exact)

# show(IOContext(stdout, :limit=>false), MIME"text/plain"(), round.(imag.(B_bohr_metro ./ B_time_metro), digits=4))
# show(IOContext(stdout, :limit=>false), MIME"text/plain"(), round.(imag.(B_bohr_metro), digits=4))
# show(IOContext(stdout, :limit=>false), MIME"text/plain"(), round.(imag.(B_time_metro), digits=4))

#* B2 Metro
# eta = 0.02
# t0_test = 0.01
# time_labels_test = t0_test * [-5:1:5;]

# Compute b2 metro
# b2_metro::Vector{ComplexF64} = ((1/(4 * pi * sqrt(2))) * 
#         exp.(-2 * time_labels_test.^2 .- 1im * time_labels_test) ./ (time_labels_test .* (2 * time_labels_test .+ 1im)))

# (1, 1, 1 (at eta), 0, 0, ..., 0)
# heaviside_eta = ifelse.(abs.(time_labels_test) .> eta, 0, ones(length(time_labels_test)))
# b2_metro .+= heaviside_eta .* (1im * (2 * time_labels_test .+ 1im) ./ (time_labels_test .* (2 * time_labels_test .+ 1im)))

#* Time gaussian normalization
# Fw = exp.(- beta^2 * (energy_labels).^2 / 4)
# Fw_norm = sqrt(sum(Fw.^2))
# ft = exp.(- time_labels.^2 / beta^2)
# ft_norm = sqrt(sum(ft.^2))

# oft_w_norm = sqrt(beta / sqrt(2 * pi))  #! sqrt(8 * pi) -> sqrt(2 * pi)

#* Discrete Fourier transform
# Random.seed!(666)
# rand_w = energy_labels[rand(1:end)]
# gaussian_t = exp.(- time_labels.^2 / beta^2) * sqrt(sqrt(2/pi)/beta)
# gaussian_t_dft(w) = t0 * sum(gaussian_t .* exp.(-1im * w * time_labels)) / sqrt(2 * pi)
# gaussian_w(w) = exp(- beta^2 * w^2 / 4) * sqrt(beta / sqrt(2 * pi))  #! sqrt(8 * pi) -> sqrt(2 * pi)

# norm(gaussian_t_dft.(energy_labels) - gaussian_w.(energy_labels))

#* Fourier integral
# jump_oft_time_integrated = time_oft_integrated(jump, rand_w, hamiltonian, beta)
# jump_oft_energy = oft(jump, rand_w, hamiltonian, beta) * oft_w_norm
# norm(jump_oft_time_integrated - jump_oft_energy)

# Integrating just the gaussian 
# total_fourier_error = 0.0
# for w in energy_labels
#     gaussian_t_integrand(t) = exp(-t^2 / beta^2) * exp(-1im * w * t)
#     gaussian_t_integral = quadgk(gaussian_t_integrand, -Inf, Inf)[1] * sqrt(sqrt(2/pi)/beta) / sqrt(2 * pi)
#     err = norm(gaussian_t_integral - gaussian_w(w))
#     display(err)
#     total_fourier_error += err
# end
# total_fourier_error

#* A(w) from time sum
# rand_w = energy_labels[rand(1:end)]
# oft_w_norm = sqrt(beta / sqrt(2 * pi))  #! THIS IS CORRECTED
# norm(sqrt(beta / sqrt(8 * pi)) * sqrt(2) - sqrt(beta / sqrt(2 * pi)))
# oft_t_norm = t0 * sqrt(sqrt(2 / pi)/beta) / sqrt(2 * pi)  #! THIS IS THE FT NORMALIZATION WITH FOURTIER (SO UNCHANGED)
# total_oft_error = 0.0
# for w in energy_labels
#     # @printf("Energy: %s\n", w)
#     jump_oft_energy = oft(jump, w, hamiltonian, beta) * oft_w_norm  # With this normalization we had DB!
#     jump_oft_time = time_oft(jump, w, hamiltonian, time_labels, beta) * oft_t_norm
#     # jump_oft_energy = oft(jump, w, hamiltonian, beta) / Fw_norm
#     # jump_oft_time = time_oft(jump, w, hamiltonian, time_labels, beta) / (ft_norm * sqrt(length(time_labels)))
#     err = norm(jump_oft_energy - jump_oft_time)
#     total_oft_error += err
#     # @printf("OFT energy vs time deviation: %s\n", err)
# end
# @printf("Total error: %s\n", total_oft_error)

#! HALLELUJA BOTH ARE NORMALIZED TO 1, GOOD ASS AMPLITUDES
# ft_vals_squared_sum = t0 * sum(abs.(exp.(- time_labels.^2 / beta^2) * sqrt(sqrt(2 / pi)/beta)).^2)
# Fw_vals_squared_sum = w0 * sum(abs.(exp.(- beta^2 * energy_labels.^2 / 4) * sqrt(beta / sqrt(2 * pi))).^2)