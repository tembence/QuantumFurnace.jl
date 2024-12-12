using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using QuantumOptics
using BenchmarkTools

include("hamiltonian.jl")
include("ofts.jl")
include("trotter.jl")
include("qi_tools.jl")
include("liouvillian_tools.jl")
include("coherent.jl")
include("db_tools.jl")

#* Parameters
num_qubits = 4
mixing_time = 60.0
delta = 0.1
num_liouv_steps = Int(mixing_time / delta)
num_liouv_steps = 3 * num_qubits
sigma = 5.
beta = 10.
eig_index = 8

#* Hamiltonian
hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n4.jld")["ideal_ham"]
initial_dm = zeros(ComplexF64, size(hamiltonian.data))
initial_dm[eig_index, eig_index] = 1.0  # In eigenbasis

hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

#* Jump operators
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
jump_paulis = [X, Y, Z]

#* Fourier labels
# num_energy_bits = ceil(Int64, log2((0.45 * 2) / hamiltonian.nu_min)) + 1 # paper (above 3.7.), later will be β dependent
num_energy_bits = ceil(Int64, log2((0.45 * 4 + 2/beta) / hamiltonian.nu_min))# Under Fig. 5. with secular approx.
# num_energy_bits = 1
N = 2^(num_energy_bits)
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

t0 = 2 * pi / (N * hamiltonian.nu_min)
time_labels = t0 * N_labels
energy_labels = hamiltonian.nu_min * N_labels

energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]
    
Fw = exp.(- beta^2 * (energy_labels).^2 / 4)
Fw_norm = sqrt(sum(Fw.^2))

@printf("Number of qubits: %d\n", num_qubits)
@printf("Number of energy bits: %d\n", num_energy_bits)
@printf("Energy unit: %e\n", hamiltonian.nu_min)
@printf("Time unit: %e\n", t0)

#* Gibbs
b = SpinBasis(1//2)^num_qubits
gibbs_in_eigen = gibbs_state_in_eigen(hamiltonian, beta)
evolved_dm = copy(initial_dm)

#* Pregenerate all random jumps
all_jumps_generated = []
Random.seed!(666)
for pauli in jump_paulis
    for site in 1:num_qubits
    jump_op = Matrix(pad_term([pauli], num_qubits, site))
    jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
    # jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
    jump = JumpOp(jump_op,
            jump_op_in_eigenbasis,
            Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
            zeros(0),
            zeros(0, 0)) #jump_in_trotter_basis
    push!(all_jumps_generated, jump)
    end
end

#* Coherent term Gaussian
b1_vals, b1_times = compute_truncated_b1(time_labels)
b2_vals, b2_times = compute_truncated_b2(time_labels)

coherent_terms::Vector{Matrix{ComplexF64}} = coherent_term_time.(all_jumps_generated, 
Ref(hamiltonian), Ref(b1_vals), Ref(b1_times), Ref(b2_vals), Ref(b2_times), Ref(beta))

tspan =[0.0:delta:mixing_time;]

#* Constructing Liouvillian
transition_gaussian(energy) = exp(-(beta * energy + 1)^2 / 2)
transition_weights = transition_gaussian.(energy_labels)

liouv = zeros(ComplexF64, num_qubits^4, num_qubits^4)
p = Progress(length(num_liouv_steps))
@showprogress dt=1 desc="Constructing Liouvillian" for j in 1:num_liouv_steps

    # Random jump
    jump = all_jumps_generated[j]
    # Corresponding coherent term
    coherent_term = coherent_terms[j]

    for (i, w) in enumerate(energy_labels)
        jump_oft = sqrt(transition_weights[i]) * entry_wise_oft_exact_db(jump, w, hamiltonian, beta) / Fw_norm
        # jump_oft = sqrt(transition_weights[i]) * explicit_oft_exact_db(jump, hamiltonian, w, time_labels, beta)
        coherent_term = hamiltonian.eigvecs * coherent_term * hamiltonian.eigvecs'
        jump_oft = hamiltonian.eigvecs * jump_oft * hamiltonian.eigvecs'
        liouv .+= construct_liouvillian(coherent_term, [jump_oft])
    end
end

#* Eigenanalysis of Liouvillian
max_mixed_dm = Matrix(I, 2^num_qubits, 2^num_qubits) / 2^num_qubits
eigvals_liouv, eigvecs_liouv = eigen(liouv)
stationary_vec = eigvecs_liouv[:, end]
stationary_dm = reshape(stationary_vec, num_qubits^2, num_qubits^2)

# trdist = tracedistance_nh(Operator(b, stationary_dm), Operator(b, max_mixed_dm))
gibbs_vec = vec(gibbs_in_eigen)
random_dm = random_density_matrix(num_qubits)
random_dm_vec = vec(random_dm)
# trace_of_timederiv_of_dm = transpose(vec(I(2^num_qubits))) * vec(liouv * random_dm_vec)  # Check if correct DM output

println("Eigvals of transition matrix:")
println(round.(real.(exp.((mixing_time .* eigvals_liouv[end - 10:end]))), digits=10))

dist = tracedistance_nh(Operator(b, stationary_dm), Operator(b, gibbs_in_eigen))
rand_dist = tracedistance_nh(Operator(b, random_dm), Operator(b, gibbs))
dist_to_maxmixed = tracedistance_nh(Operator(b, stationary_dm), Operator(b, max_mixed_dm))
println("Distance to Gibbs: $dist")
println("Random distance: $rand_dist")
println("Distance to max mixed: $dist_to_maxmixed")
