using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using TensorOperations
using JLD
using Plots
using QuadGK
using BenchmarkTools

include("hamiltonian_tools.jl")
include("jump_op_tools.jl")
include("trotter.jl")
include("qi_tools.jl")
include("trotter.jl")
include("liouvillian_tools.jl")
include("coherent.jl")

function compute_spectral_gap()
end

#TODO: Continue after extending Liouvillian step with B and apply it here onto Gibbs state.
function gibbs_is_fix(jump::JumpOp, hamiltonian::HamHam, coherent_term::Matrix{ComplexF64}, beta::Float64)

    # Gibbs
    gibbs = gibbs_state(hamiltonian, beta)

    # Jump
    Random.seed!(666)
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

    # Coherent term
end

function construct_liouvillian(coherent_term::Union{Matrix{ComplexF64}, Nothing}, jump_ops::Vector{Matrix{ComplexF64}})

    dim = size(jump_ops[1])[1]
    spI = sparse(I, dim, dim)

    vectorized_liouv = zeros(ComplexF64, dim^2, dim^2)
    if hamiltonian !== nothing
        # -i ((I ⊗ B) - (B^T ⊗ I))
        vectorized_liouv .+= -1im * (kron(spI, coherent_term) - kron(transpose(coherent_term), spI))
    end

    for jump in jump_ops
        jump_dag_jump = jump' * jump
        # 
        vectorized_liouv .+= kron(conj(jump), jump) - 0.5 * kron(spI, jump_dag_jump) - 0.5 * kron(transpose(jump_dag_jump), spI)
    end

    return vectorized_liouv
end

function liouvillian_by_one(jump::Matrix{ComplexF64})

    dim = size(jump)[1]
    spI = sparse(I, dim, dim)

    vectorized_liouv = zeros(ComplexF64, dim^2, dim^2)
    jump_dag = jump'
    # L^\dag * L
    jump_dag_jump = jump_dag * jump
    vectorized_liouv += kron(jump_dag, jump) -0.5 * kron(spI, jump_dag_jump) - 0.5 * kron(transpose(jump_dag_jump), spI)

    return vectorized_liouv
end

function spectral_gap_liouv(liouvillian::Matrix{ComplexF64})
    eig_vals = eigvals(liouvillian)
    return abs(eig_vals[2] - eig_vals[1])
end

###* Tests

#* Parameters
num_qubits = 4
mixing_time = 10.0
delta = 0.1
num_liouv_steps = Int(mixing_time / delta)
sigma = 5.
beta = 1.
eig_index = 8

#* Hamiltonian
hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n4.jld")["ideal_ham"]
# hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3), batch_size=1)

initial_dm = zeros(ComplexF64, size(hamiltonian.data))
initial_dm[eig_index, eig_index] = 1.0  # In eigenbasis
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

#* Gibbs
b = SpinBasis(1//2)^num_qubits
gibbs = gibbs_state(hamiltonian, beta)

#* Jump operators
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
jump_paulis = [X, Y, Z]

#* Fourier labels
# num_energy_bits = ceil(Int64, log2((0.45 * 2) / hamiltonian.w0)) + 1 # paper (above 3.7.), later will be β dependent
num_energy_bits = 1
N = 2^(num_energy_bits)
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

t0 = 2 * pi / (N * hamiltonian.w0)
time_labels = t0 * N_labels
energy_labels = hamiltonian.w0 * N_labels
# Truncated!
energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]

@printf("Number of qubits: %d\n", num_qubits)
@printf("Number of energy bits: %d\n", num_energy_bits)
@printf("Energy unit: %e\n", hamiltonian.w0)
@printf("Time unit: %e\n", t0)

#* Pregenerate all jumps
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
            zeros(0, 0)) #jump_in_trotter_basis)
    push!(all_jumps_generated, jump)
    end
end

#* Coherent term Gaussian
b1_vals, b1_times = compute_truncated_b1(time_labels)
b2_vals, b2_times = compute_truncated_b2(time_labels)

coherent_terms::Vector{Matrix{ComplexF64}} = coherent_term_from_timedomain.(all_jumps_generated, 
Ref(hamiltonian), Ref(b1_vals), Ref(b1_times), Ref(b2_vals), Ref(b2_times), Ref(beta))

tspan =[0.0:delta:mixing_time;]

# Random.seed!(666)
# for _ in 1:num_liouv_steps
#     random_site = rand(1:num_qubits)
#     random_pauli = rand(jump_paulis)
#     jump_op = Matrix(pad_term([random_pauli], num_qubits, random_site))
#     jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
#     # jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
#     jump_in_trotter_basis = zeros(0, 0)
#     jump = JumpOp(jump_op,
#             jump_op_in_eigenbasis,
#             Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
#             zeros(0),
#             jump_in_trotter_basis)
#     push!(all_random_jumps_generated, jump)
# end


#* Liouvillian
# transition_gaussian(energy) = sqrt(exp(-(beta * energy + 1)^2 / 2))
# transition_weights = transition_gaussian.(energy_labels)

# Fw = exp.(- beta^2 * (energy_labels).^2 / 4)
# Fw_norm = sqrt(sum(Fw.^2))

# liouv = zeros(ComplexF64, num_qubits^4, num_qubits^4)
# @time for j in eachindex(all_jumps_generated)
#     jump = all_jumps_generated[j]
#     coherent_term = all_coherent_terms_generated[j]

#     jump_oft_at_all_energies = 
#         transition_weights .* entry_wise_oft_exact_db.(Ref(jump), energy_labels, Ref(hamiltonian), Ref(beta)) / Fw_norm
#     liouv .+= construct_liouvillian(coherent_term, jump_oft_at_all_energies) 
# end

# #! Aha!
# #! /N is not correct though, but it made the trace 0.
# println("Tr(L) = ", tr(liouv))

# eigvals_liouv, eigvecs_liouv = eigen(liouv)
# stationary_vec = eigvecs_liouv[:, end]
# stationary_dm = reshape(stationary_vec, num_qubits^2, num_qubits^2)
# close_to_stationary_dms = [reshape(eigvecs_liouv[:, end - i], num_qubits^2, num_qubits^2) for i in 0:15]

# gibbs_vec = vec(gibbs)
# random_dm = random_density_matrix(num_qubits)

# println(round.(real.(exp.((mixing_time .* eigvals_liouv[end - 10:end]))), digits=10))

# # gibbs_in_comp = hamiltonian.eigvecs * gibbs * hamiltonian.eigvecs'
# # vectorized_gibbs_in_eigen = kron(transpose(hamiltonian.eigvecs), hamiltonian.eigvecs') * vec(gibbs_in_comp)

# distances = []
# for dm in close_to_stationary_dms
#     dist = tracedistance_nh(Operator(b, dm), Operator(b, gibbs))
#     push!(distances, dist)
# end
# display(minimum(distances))

# dist = tracedistance_nh(Operator(b, stationary_dm), Operator(b, gibbs))
# rand_dist = tracedistance_nh(Operator(b, random_dm), Operator(b, gibbs))
# println("Distance to Gibbs: $dist")
# println("Random distance: $rand_dist")

# #TODO: Compare steady state for different r, i.e. see how much they differ from each other and also from the Gibbs state.
# #! WHY IS THE STEADY STATE SO FAR FROM GIBBS

# #* DB up to δ^2
# should_be_zeros = liouv * gibbs_vec
# println("Should be zeros: ", norm(should_be_zeros))

#* Checking Liouvillian construction

jumps = all_jumps_generated[1:2]
coherent_terms = coherent_terms[1:2]
@time liouv_mine = construct_liouvillian(coherent_terms[1], [jumps[1].data, jumps[2].data])
@time liouv_qo = liouvillian(Operator(b, coherent_terms[1]), [Operator(b, jumps[1].data), Operator(b, jumps[2].data)])

println("Norm of difference: ", norm(liouv_mine - liouv_qo.data))