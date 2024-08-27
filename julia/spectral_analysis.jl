using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using QuantumOptics
using BenchmarkTools

include("hamiltonian_tools.jl")
include("jump_op_tools.jl")
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
# num_liouv_steps = 3 * num_qubits
beta = 10.
eig_index = 3
Random.seed!(666)

#* Hamiltonian
hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n4.jld")["ideal_ham"]
initial_dm = zeros(ComplexF64, size(hamiltonian.data))
initial_dm[eig_index, eig_index] = 1.0  # In eigenbasis

hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

#* Gibbs 
b = SpinBasis(1//2)^num_qubits
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
evolved_dm = copy(initial_dm)

#* Fourier labels
num_energy_bits = ceil(Int64, log2((0.45 * 4 + 2/beta) / hamiltonian.w0)) - 2 # Under Fig. 5. with secular approx.
N = 2^(num_energy_bits)
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

t0 = 2 * pi / (N * hamiltonian.w0)
time_labels = t0 * N_labels
energy_labels = hamiltonian.w0 * N_labels
energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]  # Energies outside are impossible in a QC

# OFT normalizations for energy basis
Fw = exp.(- beta^2 * (energy_labels).^2 / 4)
Fw_norm = sqrt(sum(Fw.^2))

# Transition weights in the Liouvillian
transition_gaussian(energy) = exp(-(beta * energy + 1)^2 / 2)
transition_weights = transition_gaussian.(energy_labels)

#* Jump operators
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
jump_paulis = [X, Y, Z]

all_jumps_generated = []
# for pauli in jump_paulis
#     for site in 1:num_qubits
#     jump_op = Matrix(pad_term([pauli], num_qubits, site))
#     jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
#     # jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
#     jump = JumpOp(jump_op,
#             jump_op_in_eigenbasis,
#             Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
#             zeros(0),
#             zeros(0, 0)) #jump_in_trotter_basis
#     push!(all_jumps_generated, jump)
#     end
# end

# Random jumps
for _ in 1:num_liouv_steps
    random_site = rand(1:num_qubits)
    random_pauli = rand(jump_paulis)
    jump_op = Matrix(pad_term([random_pauli], num_qubits, random_site))
    jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
    # jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
    jump = JumpOp(jump_op,
            jump_op_in_eigenbasis,
            Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
            zeros(0),
            zeros(0, 0)) #jump_in_trotter_basis)
    push!(all_jumps_generated, jump)
end

#* Coherent terms
# --- 

#* The Press
@printf("Number of qubits: %d\n", num_qubits)
@printf("Number of energy bits: %d\n", num_energy_bits)
@printf("Energy unit: %e\n", hamiltonian.w0)
@printf("Time unit: %e\n", t0)

#* Constructing Liouvillian / Algorithm
tspan =[0.0:delta:mixing_time;]
p = Progress(length(num_liouv_steps))

liouv = zeros(ComplexF64, num_qubits^4, num_qubits^4)
evolved_dm = copy(initial_dm)
distances_to_gibbs = [tracedistance_nh(Operator(b, evolved_dm), Operator(b, gibbs))]
#TODO: Collect all jumps applied and construct the Liouvillian from the SAME for loop!
@showprogress dt=1 desc="Run..." for step in 1:num_liouv_steps

    # Jump
    jump = all_jumps_generated[step]

    # Coherent term
    # coherent_term = coherent_terms[j]
    # evolved_dm .+= - im * delta * (coherent_term * initial_dm - initial_dm * coherent_term)

    evolved_part = zeros(ComplexF64, size(initial_dm))
    for (i, w) in enumerate(energy_labels)
        # oft_matrix = explicit_oft_exact_db(jump, hamiltonian, w, time_labels, beta)
        oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)  #!
        oft_matrix_dag = oft_matrix'
        
        evolved_part .+= delta * transition_weights[i] *
                    (oft_matrix * evolved_dm * oft_matrix_dag
                    - 0.5 * oft_matrix_dag * oft_matrix * evolved_dm
                    - 0.5 * evolved_dm * oft_matrix_dag * oft_matrix)
    end

    evolved_dm .+= evolved_part / Fw_norm^2  #!
    evolved_dm /= tr(evolved_dm)
    
    dist = tracedistance_nh(Operator(b, evolved_dm), Operator(b, gibbs))
    @printf("\nDistance to Gibbs: %f\n", dist)

    push!(distances_to_gibbs, dist)
end

min_dist = minimum(distances_to_gibbs)
max_mixed_dm = Matrix(I, 2^num_qubits, 2^num_qubits) / 2^num_qubits
dist_to_maxmixed = tracedistance_nh(Operator(b, evolved_dm), Operator(b, max_mixed_dm))
@printf("Minimum distance to Gibbs: %f\n", min_dist)
@printf("Distance to max mixed: %f\n", dist_to_maxmixed)

