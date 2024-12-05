using LinearAlgebra
using Random
using Printf
using JLD

include("hamiltonian_tools.jl")
include("jump_op_tools.jl")
include("trotter.jl")
include("qi_tools.jl")
include("thermalizing_tools.jl")
include("coherent.jl")
include("spectral_analysis_tools.jl")
include("structs.jl")
include("coherent.jl")

#* Parameters
num_qubits = 1
beta = 2.
mixing_time = 5.
delta = 0.1
with_coherent = true

#* System
hamiltonian_terms = [["X"]]
hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
@printf("Bohr freqs:\n")
display(hamiltonian.bohr_freqs)

ham_in_eigenbasis = hamiltonian.eigvecs' * hamiltonian.data * hamiltonian.eigvecs

num_energy_bits = ceil(Int64, log2((0.45 * 4 + 2/beta) / hamiltonian.w0)) + 3

# initial_dm = ones(ComplexF64, size(hamiltonian.data)) # In eigenbasis
# initial_dm_OG = zeros(ComplexF64, size(hamiltonian.data))
# initial_dm_OG[2, 2] = 1.0
# initial_dm_OG[1, 2] = 1.0
# initial_dm_OG[2, 1] = 1.0

initial_dm_OG = diagm([0.25, 0.75])

# initial_dm_OG = initial_dm_OG * initial_dm_OG'
# initial_dm_OG /= tr(initial_dm_OG)
@assert initial_dm_OG == adjoint(initial_dm_OG) "Initial DM is not Hermitian"
# eigvals(initial_dm_OG)
# tr(initial_dm_OG)
@printf("Initial DM:\n")
display(initial_dm_OG)

#* Gibbs
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
gibbs_vec = vec(gibbs)
gibbs_largest_eigval = real(eigen(gibbs).values[1])

#* Fourier labels, Gaussians
# N = 2^(num_energy_bits)
# N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
# energy_labels = hamiltonian.w0 * N_labels
transition(energy) = exp(-(beta * energy + 1)^2 / 2)
filter_gauss_w(energy) = exp.(- beta^2 * (energy).^2 / 4)

# Plot transition 
# ordered_energy_labels = sort(energy_labels[abs.(energy_labels) .< 2.])
# plot(ordered_energy_labels, transition.(ordered_energy_labels), label="Transition", xlabel="Energy", ylabel="Transition probability", title="Transition probability vs energy")

# Plot filter
# nu = hamiltonian.bohr_freqs[2, 1]
# plot(ordered_energy_labels, filter_gauss_w.(ordered_energy_labels .- nu), label="Filter", xlabel="Energy", ylabel="Filter", title="Filter vs energy")

#* Jump operators
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
H::Matrix{ComplexF64} = [1 1; 1 -1] / sqrt(2)
T::Matrix{ComplexF64} = [1 0; 0 exp(im * pi / 4)]
Tdag::Matrix{ComplexF64} = [1 0; 0 exp(-im * pi / 4)]
SM::Matrix{ComplexF64} = [0 1; 0 0]
SP::Matrix{ComplexF64} = [0 0; 1 0]
XplusY = X + Y
XplusT = X + T
XplusTdag = X + Tdag

# Have all adjoints in there too:
jump_set = [[X], [Y], [Z], [H], [T], [Tdag], [SM], [SP]]
# jump_set = [[H]]

# Do we have all adjoints in the set too?
for jump in jump_set
    if !(in(adjoint(jump[1]), vcat(jump_set...)))
        @printf("Adjoint is not in the jump set\n")
    end
end

# Normalize jumps
# jump_sum = zeros(2^num_qubits, 2^num_qubits)
# for j in jump_set
#     jump_sum .+= j[1] * j[1]'
# end
# jump_normalization = maximum(svdvals(jump_sum))
jump_normalization = 1.

all_jumps_generated::Vector{JumpOp} = []
for jump_type in jump_set
    for site in 1:num_qubits
    jump_op = Matrix(pad_term(jump_type, num_qubits, site)) / jump_normalization
    jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
    jump_in_trotter_basis = zeros(0, 0)
    orthogonal = (jump_op == adjoint(jump_op))
    jump = JumpOp(jump_op,
            jump_op_in_eigenbasis,
            Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
            zeros(0),
            jump_in_trotter_basis,
            orthogonal) 
    push!(all_jumps_generated, jump)
    end
end

# The Press
@printf("Number of qubits: %d\n", num_qubits)
@printf("Number of energy bits: %d\n", num_energy_bits)
@printf("Energy unit: %e\n", hamiltonian.w0)
@printf("Mixing time: %s\n", mixing_time)
@printf("Delta: %s\n", delta)

#* Testing DB of Lindbladian parts
#* Is A_nu an eigenvector of \sigma (.) \sigma^{-1}
# hamiltonian.bohr_freqs
# jump_from_the_set = 1
# (i, j) = (1, 2)
# nu = hamiltonian.bohr_freqs[i, j]
# jump_nu = zeros(ComplexF64, 2^num_qubits, 2^num_qubits)
# jump_nu[i, j] = all_jumps_generated[jump_from_the_set].in_eigenbasis[i, j]
# energy = tr(ham_in_eigenbasis * initial_dm_OG)
# jumped_DM = jump_nu * initial_dm_OG * jump_nu'
# jumped_DM /= tr(jumped_DM)
# jumped_energy = tr(ham_in_eigenbasis * jumped_DM)

# # Checked!
# for jump_from_the_set in eachindex(all_jumps_generated)
#     for i in 1:2
#         for j in 1:2
#             jump_nu = zeros(ComplexF64, 2, 2)
#             nu = hamiltonian.bohr_freqs[i, j]
#             jump_nu[i, j] = all_jumps_generated[jump_from_the_set].in_eigenbasis[i, j]
#             # @printf("Is it in eigenvector of Delta? \n")
#             # display(gibbs * jump_nu * gibbs^(-1) == exp(-beta * nu) * jump_nu)
#             if norm(gibbs * jump_nu * gibbs^(-1) - exp(-beta * nu) * jump_nu) > 1e-15
#                 @printf("Nope\n")
#                 @printf("Jump from the set: %d\n", jump_from_the_set)
#                 @printf("i: %d, j: %d\n", i, j)
#             end
#         end
#     end
# end

#* Does T have DB? = Its adjoint has the given form as in DBC? != T(gibbs) = 0 (which it isn't)
# input_dm = gibbs
# id = Matrix{ComplexF64}(I(2^num_qubits)) / 2^num_qubits
# T_adjoint_on_id = transition_bohr(all_jumps_generated, hamiltonian, id, beta; adjoint=true)
# @printf("T ADJOINT ON ID\n")
# display(T_adjoint_on_id)
# println()
# id = Matrix{ComplexF64}(I(2^num_qubits)) / 2^num_qubits
# T_gibbsed_on_id = transition_bohr_gibbsed(all_jumps_generated, hamiltonian, id, beta)
# @printf("T GIBBSED ON ID\n")
# display(T_gibbsed_on_id)

# norm(T_gibbsed_on_id - T_adjoint_on_id)
# T_on_dm = transition_bohr(all_jumps_generated, hamiltonian, input_dm, beta; adjoint=false)
# tr(T_on_dm)

# gibbsed(op) = gibbs^(-0.5) * op * gibbs^(0.5)
# jump = all_jumps_generated[1]
# gibbsed_jump = gibbsed(jump.data)
# gibbsed_jump_in_eigenbasis = hamiltonian.eigvecs' * gibbsed_jump * hamiltonian.eigvecs
# eigenbasis_jump_gibbsed = gibbsed(jump.in_eigenbasis)

# T = transition_bohr_vec(all_jumps_generated, hamiltonian, beta; adjoint=true)

# T_gibbsed = transition_bohr_gibbsed_vec(all_jumps_generated, hamiltonian, beta)
# norm(T - T_gibbsed)
#! YES

#* Thermalize
# for jump in all_jumps_generated[1:1]
#     @printf("--- Jump in eigenbasis:\n")
#     display(jump.in_eigenbasis)
#     @printf("Jump effect on initial DM\n")
#     display(jump.in_eigenbasis * initial_dm_OG)
#     @printf("Energy before jump: %s\n", tr(ham_in_eigenbasis * initial_dm_OG))
#     @printf("Energy after jump: %s\n", tr(ham_in_eigenbasis * jump.in_eigenbasis * initial_dm_OG * jump.in_eigenbasis'))

    # initial_dm = copy(initial_dm_OG)  # Deepcopy in julia
    # @time results = thermalize_gaussian([jump], hamiltonian, with_coherent, initial_dm, num_energy_bits,
    #     filter_gauss_w, transition, delta, mixing_time, beta)

    # Reinitalize
    # initial_dm = copy(initial_dm_OG)
    # coherent_term = coherent_gaussian_bohr(hamiltonian, jump, beta)
    # initial_dm .+= - im * delta * (coherent_term * initial_dm - initial_dm * coherent_term)
    # @printf("DM after coherent part:\n")
    # display(initial_dm)
    # @time evolved_dm = evolve_bohr_explicit!(jump, hamiltonian, with_coherent, initial_dm, delta, beta)
    
    # println(trace_distance_nh(evolved_dm, results.evolved_dm), "\n")
# end

# liouv_matrix = construct_liouvillian_gauss(all_jumps_generated, hamiltonian, with_coherent, num_energy_bits, 
#     filter_gauss_w, transition, beta)

# liouv = LiouvLiouv(liouv_matrix, zeros(0, 0), 0.0, 0.0)
# trdist_eps = 1e-3
# mixing_time_bound(liouv, trdist_eps)
# @printf("spectral gap Liouv: %s\n", liouv.spectral_gap)
# @printf("Mixing time bound: %s\n", liouv.mixing_time_bound)

# trdist_fix_gibbs = trace_distance_nh(liouv.steady_state, gibbs)
# @printf("Trace distance Gibbs - steady state: %s\n", trdist_fix_gibbs)

#* Construct Liouvillian
liouvillian = construct_liouvillian_gauss_bohr(all_jumps_generated, hamiltonian, with_coherent, beta)
liouv_eigvals, liouv_eigvecs = eigen(liouvillian) 
steady_state_vec = liouv_eigvecs[:, end]
steady_state_dm = reshape(steady_state_vec, size(initial_dm_OG))
steady_state_dm /= tr(steady_state_dm)
steady_state_vec = vec(steady_state_dm)

#* Steady state, Gibbs?
norm(gibbs_vec - steady_state_vec)
# trace_distance_h(Hermitian(gibbs), Hermitian(steady_state_dm))

#* Liouvillian checks
# liouv_time_evolution(t) = exp(t * liouvillian)
# t = 1.0
# initial_vec = vec(initial_dm_OG)
# time_evolved_vec = liouv_time_evolution(t) * initial_vec

# Trace preserving
# id = I(2^num_qubits)
# vec_id = vec(id)
# tr_of_evolved_dm = vec_id' * time_evolved_vec

# Hermiticity preserving
# time_evolved_dm = reshape(time_evolved_vec, size(initial_dm_OG))
# norm(time_evolved_dm' - time_evolved_dm)

# Positivity and time evolution
# for i in 1:1000
#     t = rand()
#     time_evolved_vec = liouv_time_evolution(t) * initial_vec
#     time_evolved_vec_with_derivative = initial_vec + t * liouv_time_evolution(t) * initial_vec
#     hs_dist = norm(time_evolved_vec - time_evolved_vec_with_derivative)
#     if hs_dist < 1e-10
#         @printf("HS dist is %s\n", hs_dist)
#     end

#     time_evolved_dm = reshape(time_evolved_vec, size(initial_dm_OG))
#     eigvals_dm = eigvals(time_evolved_dm)
#     if any(real.(eigvals_dm) .<= 0)
#         display(eigvals_dm)
#     end
# end

# Steady state check
# is_zero = norm(liouvillian * steady_state_vec)

# Convergence to steady state
# fidelity(Hermitian(initial_dm_OG), Hermitian(steady_state_dm)) # There is overlap
# t = 1000.
# long_time_evolved_vec = liouv_time_evolution(t) * initial_vec
# norm(long_time_evolved_vec - steady_state_vec)


