using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
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
include("misc_tools.jl")

Random.seed!(666)

#* Configs
num_qubits = 4
dim = 2^num_qubits
beta = 10.
a = beta / 50.  # exp(-ax) factor in the weight g(x)
b = 0.5         # Integration LB shift to 1/2beta * (1 + b); b = 2 is the Glauber
with_coherent = true

#* Hamiltonian
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
sigmam::Matrix{ComplexF64} = (X - 1im * Y) / 2
sigmap::Matrix{ComplexF64} = (X + 1im * Y) / 2
id::Matrix{ComplexF64} = I(2)
jump_paulis = [[X], [Y], [Z], [H]]

# All jumps once
jumps::Vector{JumpOp} = []
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
    push!(jumps, jump)
    end
end

jump = jumps[2]

#* Bohr Liouvillians
liouv_bohr_eh = @time construct_liouvillian_bohr_eh(jumps, hamiltonian, with_coherent, beta, a, b)
# liouv_bohr_gauss = @time construct_liouvillian_bohr_gauss(jumps, hamiltonian, with_coherent, beta)
liouv_bohr_metro = @time construct_liouvillian_bohr_metro(jumps, hamiltonian, with_coherent, beta)

# liouv_eigvals, liouv_eigvecs = eigen(liouv_bohr_eh) 
# steady_state_vec = liouv_eigvecs[:, end]
# steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
# steady_state_dm /= tr(steady_state_dm)

# lambda2 = liouv_eigvals[end] - liouv_eigvals[end-1]
# @printf("Lambda2: %s\n", lambda2)

# @printf("Steady state closeness to Gibbs for Liouvillian (Energy): %s\n", norm(steady_state_dm - gibbs))

#* Labels
w0 = 1e-2
t0 = 0.1
num_energy_bits = ceil(log2(2pi / (w0 * t0)))

N = 2^(num_energy_bits)
t0 = 2pi / (N * w0)  # Actual t0, for the found num estimating qubits

N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
energy_labels = w0 * N_labels
@assert maximum(energy_labels) >= 2.0
time_labels = t0 * N_labels

print_press(num_qubits=num_qubits, beta=beta, a=a, b=b, t0=t0, w0=w0, num_energy_bits=num_energy_bits)

# Helping functions
sqrtA(a, beta) = sqrt((4 * a / beta + 1) / 8)
sqrtB(w, beta) = beta * abs(w + 1 / (2 * beta)) / sqrt(2)
btilde(b, beta) = (b / (2 * beta))^(1/4)  # integration LB shifted by b
gaussfilter(w, nu, beta) = exp(-beta^2 * (w - nu)^2 / 4) * sqrt(beta / sqrt(2 * pi))

# Transitions
transition_eh(w, beta, a) = exp((- 2 * sqrtA(a, beta) * sqrtB(w, beta) - beta * w / 2 - 1 / 4))
transition_eh_smooth(w, beta, a, b) = transition_eh(w, beta, a) * (
    erfc(sqrtA(a, beta) * btilde(b, beta) - sqrtB(w, beta) / btilde(b, beta)) +
    exp(4 * sqrtA(a, beta) * sqrtB(w, beta)) * erfc(sqrtA(a, beta) * btilde(b, beta) + sqrtB(w, beta) / btilde(b, beta))) / 2

# Energy integrands
eh_energy_integrand(w, nu_1, nu_2, beta, a) = transition_eh(w, beta, a) * gaussfilter(w, nu_1, beta) * gaussfilter(w, nu_2, beta)
eh_energy_integrand_smooth(w, nu_1, nu_2, beta, a, b) = (
    transition_eh_smooth(w, beta, a, b) * gaussfilter(w, nu_1, beta) * gaussfilter(w, nu_2, beta))

# w0 = 1e-1
# energies = [-1.0:w0:1.0;]
# b = 2.0
# proper_int = quadgk(w -> eh_energy_integrand(w, nu_1, nu_2, beta, a), -1.0, 1.0)[1]
# proper_int_smooth = quadgk(w -> eh_energy_integrand_smooth(w, nu_1, nu_2, beta, a, b), -1.0, 1.0)[1]
# summed = riemann_sum(w -> eh_energy_integrand(w, nu_1, nu_2, beta, a), energies)
# summed_smooth = riemann_sum(w -> eh_energy_integrand_smooth(w, nu_1, nu_2, beta, a, b), energies)
# norm(proper_int - summed)
# norm(proper_int_smooth - summed_smooth)

# plot(energies, gaussfilter.(energies, 0.12, beta))
# plot(energies, transition_eh.(energies, beta, a))
# plot!(energies, transition_eh_smooth.(energies, beta, a, b))
# nu_1 = -0.1
# nu_2 = 0.0
# plot(energies, eh_energy_integrand.(energies, nu_1, nu_2, beta, a))
# plot!(energies, eh_energy_integrand_smooth.(energies, nu_1, nu_2, beta, a, b))
truncated_energies = truncate_energy_labels(energy_labels, eh_energy_integrand_smooth, (beta, a, b))
# Checked truncation, Liouv is insanely close to nontruncated.

#* Energy Liouvillian
liouv_energy_eh = @time construct_liouvillian_eh(jumps, hamiltonian, truncated_energies, with_coherent, beta, a, b)
liouv_energy_metro = @time construct_liouvillian_metro(jumps, hamiltonian, truncated_energies, with_coherent, beta)
norm(liouv_energy_eh - liouv_bohr_eh)
norm(liouv_energy_metro - liouv_bohr_metro)

#* Coherent part
# eta = 1e-2
# f_minus = compute_truncated_f_minus(time_labels, beta)
# f_plus_eh = compute_truncated_f_plus_eh(time_labels, beta, a, b)
# f_plus_metro = compute_truncated_f_plus_metro(time_labels, eta, beta)

# B_bohr_eh = coherent_bohr_eh(hamiltonian, bohr_dict, jump, beta, a, b)
# B_time_eh = coherent_term_time_f(jump, hamiltonian, f_minus, f_plus_eh, t0)
# @printf("Deviation Bohr-Time EH: %e\n", norm(B_bohr_eh - B_time_eh))

# B_bohr_metro = coherent_bohr_metro(hamiltonian, bohr_dict, jump, beta)
# B_time_metro = coherent_term_time_f(jump, hamiltonian, f_minus, f_plus_metro, t0)

# @printf("Deviation Bohr-Time METRO: %e\n", norm(B_bohr_metro - B_time_metro))

#* Time Liouvillian
eta = 1e-2
liouv_time_eh = construct_liouvillian_time_eh(jumps, hamiltonian, time_labels, truncated_energies, with_coherent, beta, a, b)
liouv_time_metro = construct_liouvillian_time_metro(jumps, hamiltonian, time_labels, truncated_energies, with_coherent, beta, eta)
norm(liouv_time_eh - liouv_bohr_eh)

norm(liouv_time_metro - liouv_bohr_metro)
liouv_eigvals, liouv_eigvecs = eigen(liouv_time_metro) 
steady_state_vec = liouv_eigvecs[:, end]
steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
steady_state_dm /= tr(steady_state_dm)

@printf("Steady state closeness to Gibbs for Liouvillian (Energy): %s\n", norm(steady_state_dm - gibbs))

#* OFT time analysis
# average_deviation = 0.0
# max_deviation = 0.0
# for w in truncated_energies
#     oft_w = oft(jump, w, hamiltonian, beta) * sqrt(beta / sqrt(2 * pi))
#     oft_time = time_oft(w, jump, hamiltonian, time_labels, beta) * t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
#     d = norm(oft_w - oft_time)
#     if d > max_deviation
#         max_deviation = d
#     end
#     average_deviation += d
# end
# average_deviation /= length(truncated_energies)
# max_deviation