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
hamiltonian_terms = [["Z"]]
hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
@printf("Bohr freqs:\n")
display(hamiltonian.bohr_freqs)

ham_in_eigenbasis = hamiltonian.eigvecs' * hamiltonian.data * hamiltonian.eigvecs

num_energy_bits = ceil(Int64, log2((0.45 * 4 + 2/beta) / hamiltonian.w0)) + 3

# initial_dm = ones(ComplexF64, size(hamiltonian.data)) # In eigenbasis
initial_dm_OG = zeros(ComplexF64, size(hamiltonian.data))
initial_dm_OG[2, 2] = 1.0
initial_dm_OG[1, 2] = 1.0
initial_dm_OG /= tr(initial_dm_OG)

@printf("Initial DM:\n")
display(initial_dm_OG)

#* Gibbs
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
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
jump_paulis = [[X], [Y], [Z], [H]]
jump_paulis = [[H]]

all_jumps_generated::Vector{JumpOp} = []
for pauli in jump_paulis
    for site in 1:num_qubits
    jump_op = Matrix(pad_term(pauli, num_qubits, site))
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

for jump in all_jumps_generated[1:1]
    @printf("--- Jump in eigenbasis:\n")
    display(jump.in_eigenbasis)
    @printf("Jump effect on initial DM\n")
    display(jump.in_eigenbasis * initial_dm_OG)
    @printf("Energy before jump: %s\n", tr(ham_in_eigenbasis * initial_dm_OG))
    @printf("Energy after jump: %s\n", tr(ham_in_eigenbasis * jump.in_eigenbasis * initial_dm_OG * jump.in_eigenbasis'))

    # initial_dm = copy(initial_dm_OG)  # Deepcopy in julia
    # @time results = thermalize_gaussian([jump], hamiltonian, with_coherent, initial_dm, num_energy_bits,
    #     filter_gauss_w, transition, delta, mixing_time, beta)

    # Reinitalize
    initial_dm = copy(initial_dm_OG)
    coherent_term = coherent_gaussian_bohr(hamiltonian, jump, beta)
    initial_dm .+= - im * delta * (coherent_term * initial_dm - initial_dm * coherent_term)
    @printf("DM after coherent part:\n")
    display(initial_dm)
    # @time evolved_dm = evolve_bohr_explicit!(jump, hamiltonian, with_coherent, initial_dm, delta, beta)
    
    # println(trace_distance_nh(evolved_dm, results.evolved_dm), "\n")
end

# liouv_matrix = construct_liouvillian_gauss(all_jumps_generated, hamiltonian, with_coherent, num_energy_bits, 
#     filter_gauss_w, transition, beta)

# liouv = LiouvLiouv(liouv_matrix, zeros(0, 0), 0.0, 0.0)
# trdist_eps = 1e-3
# mixing_time_bound(liouv, trdist_eps)
# @printf("spectral gap Liouv: %s\n", liouv.spectral_gap)
# @printf("Mixing time bound: %s\n", liouv.mixing_time_bound)

# trdist_fix_gibbs = trace_distance_nh(liouv.steady_state, gibbs)
# @printf("Trace distance Gibbs - steady state: %s\n", trdist_fix_gibbs)

