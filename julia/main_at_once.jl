using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using JLD

include("ofts.jl")
include("hamiltonian.jl")
include("liouvillian_tools.jl")
include("qi_tools.jl")

#* Parameters
num_qubits = 4
eig_index = 2
beta = 1.

liouvillian_time = 0.1
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

initial_state = hamiltonian.eigvecs[:, eig_index]
initial_dm = initial_state * initial_state'

# Gibbs
gibbs = gibbs_state(hamiltonian, beta)
distance_to_gibbs = trace_distance(Hermitian(initial_dm), Hermitian(gibbs))
@printf("Initial distance to Gibbs: %e\n", distance_to_gibbs)

#* Labels
# Ideal - takes way more time evolution since t0 = 2pi/ (N w0)
num_energy_bits = ceil(Int64, log2(1 / hamiltonian.nu_min))
N = 2^num_energy_bits
energy_labels = get_energy_labels(hamiltonian.nu_min)

# - Or Not
# num_energy_bits = 8
# N = 2^num_energy_bits
# w0_by_hand = 2 * pi / N #! This actually might not be correct really, think it through
# energy_labels = get_energy_labels(N, w0_by_hand)

oft_precision = ceil(Int, abs(log10(N^(-1))))
hamiltonian.bohr_freqs = round.(hamiltonian.eigvals .- transpose(hamiltonian.eigvals), digits=oft_precision+3)

#* Jump operators
# Generate the set of random jumps on the sites
Random.seed!(666)
# Random sites
jump_site_indices = rand(1:num_qubits, num_liouvillian_steps)
# Random jump operators from set

set_of_jumps = [sigmax, sigmay, sigmaz]
string_set_of_jumps = ["x", "y", "z"]
jump_ops = [set_of_jumps[rand(1:length(set_of_jumps))] for i in 1:num_liouvillian_steps]   # Not padded yet
string_jump_ops = [string_set_of_jumps[rand(1:length(string_set_of_jumps))] for i in 1:num_liouvillian_steps]
jump_ops_n_indices = [(string_jump_ops[i], jump_site_indices[i]) for i in 1:num_liouvillian_steps]
jumps_ops_n_amount = Dict{Tuple{String, Int64}, Int64}()
for jump_n_index in jump_ops_n_indices
        if haskey(jumps_ops_n_amount, jump_n_index)
                jumps_ops_n_amount[jump_n_index] += 1
        else
                jumps_ops_n_amount[jump_n_index] = 1
        end
end

println("\nGenerated the following jumps:")
for i in 1:num_liouvillian_steps
        @printf("Jump %d: %s on site %d\n", i, string_jump_ops[i], jump_site_indices[i])
end

#* /// RUN ///
# Up to δ^2 order of the algorithm, we only need to calculate Liouvillians for unique jumps acting on the initial initial_dm
num_unique_jumps = length(keys(jumps_ops_n_amount))

distances_to_gibbs = Vector{Float64}(undef, num_unique_jumps)
evolved_dm = deepcopy(initial_dm)
p = Progress(length(seeds))
@showprogress dt=1 desc="Evolving Liouvillian trajectories..." for i in 1:num_unique_jumps
        jump_op = Matrix(pad_term([jump_ops[i]], num_qubits, jump_site_indices[i]))
        jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
        jump = JumpOp(jump_op,
                jump_op_in_eigenbasis,
                Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
                zeros(0))
        # find_unique_jump_freqs(jump, hamiltonian)
        
        evolved_branch_dm = liouvillian_delta_trajectory(jump, hamiltonian, initial_dm, delta, sigma, beta)
        evolved_dm += jumps_ops_n_amount[jump_ops_n_indices[i]] * evolved_branch_dm
        #! Difficult to record the ditances to gibbs state since here we simplify and just multiply the amount
        #! But in reality they jumps don't combine but come one after the other. But I also don't wanna save all evolutions
end


