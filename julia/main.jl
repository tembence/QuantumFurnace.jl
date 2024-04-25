using LinearAlgebra
using StatsBase
using SparseArrays
using Random
using Printf
using ProgressMeter
using JLD

include("jump_op_tools.jl")
include("hamiltonian_tools.jl")
include("liouvillian_tools.jl")
include("qi_tools.jl")

# NOTE: The whole code works in the eigenbasis of the Hamiltonian.

#* Parameters
num_qubits = 4
eig_index = Int(2^num_qubits / 2)
beta = 1.

liouvillian_time = 0.2
delta = 0.1
num_liouvillian_steps = ceil(Int, liouvillian_time / delta)
sigma = 5.

sigmax::Matrix{ComplexF64} = [0 1; 1 0]
sigmay::Matrix{ComplexF64} = [0 -im; im 0]
sigmaz::Matrix{ComplexF64} = [1 0; 0 -1]

#* Hamiltonian
# coeffs = fill(1.0, 3)
# - Find Hamiltonian
# hamiltonian = find_ideal_heisenberg(num_qubits, coeffs, batch_size=1)
# - Or Load Hamiltonian 
hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n4.jld")["ideal_ham"]

# initial_state = hamiltonian.eigvecs[:, eig_index]
# initial_dm = initial_state * initial_state'
# initial_dm in the eigenbasis is an all zeros function with a 1 at the (eig_index, eig_index)
initial_dm = zeros(ComplexF64, size(hamiltonian.data))
initial_dm[eig_index, eig_index] = 1.0  #* In eigenbasis

# Gibbs in eigenbasis
gibbs = gibbs_state(hamiltonian, beta)
distance_to_gibbs = trace_distance(Hermitian(initial_dm), Hermitian(gibbs))
@printf("Initial distance to Gibbs: %e\n", distance_to_gibbs)

#* Labels
# Ideal - takes way more time evolution since t0 = 2pi instead of 1
num_energy_bits = ceil(Int64, log2(1 / hamiltonian.w0))
N = 2^num_energy_bits
@printf("Number of energy bits: %d, to w0: %f\n", num_energy_bits, hamiltonian.w0)
energy_labels = get_energy_labels(hamiltonian.w0)

# - Or Not
# num_energy_bits = 8
# N = 2^num_energy_bits
# w0_by_hand = 2 * pi / N #! This actually might not be correct really, think it through
# energy_labels = get_energy_labels(N, w0_by_hand)

oft_precision = ceil(Int64, abs(log10(N^(-1))))
hamiltonian.bohr_freqs = round.(hamiltonian.eigvals .- transpose(hamiltonian.eigvals), digits=oft_precision+3)

#* Jump operators
# Generate the set of random jumps on the sites
Random.seed!(666)
dict_of_jumps = Dict("x" => sigmax, "y" => sigmay, "z" => sigmaz)

#* /// RUN ///
# Up to δ^2 order of the algorithm, we only need to calculate Liouvillians for unique jumps acting on the initial initial_dm
distances_to_gibbs = Vector{Float64}[]
evolved_dm = deepcopy(initial_dm)

# We know which Liouvillians appear more than once, and we can reuse them without recomputing anything.
# Generate random jump and index tupled
jump_ops_n_indices = [(rand(keys(dict_of_jumps)), rand(1:num_qubits)) for i in 1:num_liouvillian_steps]
println(jump_ops_n_indices)
# Find non-unique jumps
counts_of_jumps = countmap(jump_ops_n_indices)
repeating_jumps = [k for (k, v) in counts_of_jumps if v > 1]

saved_liouvillian_actions = Dict{Tuple{String, Matrix{ComplexF64}}, Matrix{ComplexF64}}() #? Are these evolved DMs sparse?

p = Progress(num_liouvillian_steps)
@showprogress dt=1 desc="Evolving Liouvillian trajectories..." for i in 1:num_liouvillian_steps
        
    # Generate a random jump
    jump_op_n_index = jump_ops_n_indices[i]

    # Check if this jump appeared before
    if haskey(saved_liouvillian_actions, jump_op_n_index)
        evolved_branch_dm = saved_liouvillian_actions[jump_op_n_index]
    else
        jump_op_matrix = dict_of_jumps[jump_op_n_index[1]]  # Currently written for 1q jumps
        jump_op_index = jump_op_n_index[2]
        jump_op = Matrix(pad_term([jump_op_matrix], num_qubits, jump_op_index))
        jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
        jump = JumpOp(jump_op,
                jump_op_in_eigenbasis,
                Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
                zeros(0))
        # find_unique_jump_freqs(jump, hamiltonian)

        evolved_branch_dm = liouvillian_delta_trajectory(jump, hamiltonian, energy_labels, initial_dm, delta, sigma, beta)

        # Save the Liouvillian action if it will come up later again
        if jump_op_n_index in repeating_jumps
            saved_liouvillian_actions[jump_op_n_index] = evolved_branch_dm
        end
    end
        evolved_dm += evolved_branch_dm

        min_eig_val = minimum(round.(eigvals(Hermitian(evolved_dm)), digits=12))
        println(min_eig_val)
        # Compute distance to Gibbs
        distance_to_gibbs = trace_distance(Hermitian(evolved_dm), gibbs)
        push!(distances_to_gibbs, distance_to_gibbs)
        next!(p, showvalues = [(:trdist, distance_to_gibbs)])
end


