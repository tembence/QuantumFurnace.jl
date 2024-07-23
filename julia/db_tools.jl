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

function liouvillian(hamiltonian::Union{Matrix{ComplexF64}, Nothing}, jump_ops::Vector{Matrix{ComplexF64}})

    dim = size(jump_ops[1])[1]
    spI = sparse(I, dim, dim)

    vectorized_liouv = zeros(ComplexF64, dim^2, dim^2)
    if hamiltonian !== nothing
        vectorized_liouv .+= -1im * (kron(spI, hamiltonian) - kron(transpose(hamiltonian), spI))
    end

    for jump in jump_ops
        jump_dag = jump'
        # L^\dag * L
        jump_dag_jump = jump_dag * jump
        vectorized_liouv .+= kron(conj(jump), jump) -0.5 * kron(spI, jump_dag_jump) - 0.5 * kron(transpose(jump_dag_jump), spI)
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
num_energy_bits = ceil(Int64, log2((0.45 * 2) / hamiltonian.w0)) + 1 # paper (above 3.7.), later will be β dependent
num_energy_bits = 8
N = 2^(num_energy_bits)
N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

t0 = 2 * pi / (N * hamiltonian.w0)
time_labels = t0 * N_labels
energy_labels = hamiltonian.w0 * N_labels

@printf("Number of qubits: %d\n", num_qubits)
@printf("Number of energy bits: %d\n", num_energy_bits)
@printf("Energy unit: %e\n", hamiltonian.w0)
@printf("Time unit: %e\n", t0)

#* Coherent term Gaussian
b1_vals, b1_times = compute_truncated_b1(time_labels)
b2_vals, b2_times = compute_truncated_b2(time_labels)

#* Pregenerate all random jumps
all_jumps_generated = []
all_coherent_terms_generated = []
for pauli in jump_paulis
    for site in 1:num_qubits
        jump_op = Matrix(pad_term([pauli], num_qubits, site))
        jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
        # jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
        jump_in_trotter_basis = zeros(0, 0)
        jump = JumpOp(jump_op,
                jump_op_in_eigenbasis,
                Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
                zeros(0),
                jump_in_trotter_basis)
        push!(all_jumps_generated, jump)

        #* Coherent term for DB
        coherent_term::Matrix{ComplexF64} = coherent_term_from_timedomain(jump, 
            hamiltonian, b1_vals, b1_times, b2_vals, b2_times, beta)
        push!(all_coherent_terms_generated, coherent_term)
    end
end

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
liouv = zeros(ComplexF64, num_qubits^4, num_qubits^4)
# @time for w in energy_labels
#     jumps_at_w = entry_wise_oft_exact_db.(all_jumps_generated, Ref(w), Ref(hamiltonian), Ref(beta))
#     liouv .+= liouvillian(nothing, jumps_at_w)
# end

@time for j in eachindex(all_jumps_generated)
    jump = all_jumps_generated[j]
    coherent_term = all_coherent_terms_generated[j]

    jump_oft_at_all_energies = entry_wise_oft_exact_db.(Ref(jump), energy_labels, Ref(hamiltonian), Ref(beta))
    liouv .+= liouvillian(coherent_term, jump_oft_at_all_energies)
end

eigvals_liouv, eigvecs_liouv = eigen(liouv)
gibbs_vec = vec(gibbs)
random_dm_vec = vec(random_density_matrix(num_qubits))

@printf("Steady state distance to Gibbs: %f\n", norm(gibbs_vec - eigvecs_liouv[:, end], 2))
@printf("Random state distance to Gibbs: %f\n", norm(gibbs_vec - random_dm_vec))

println(round.(real.(exp.((delta .* eigvals_liouv[end - 10:end]))), digits=3))


#TODO: Compare steady state for different r, i.e. see how much they differ from each other and also from the Gibbs state.
#! WHY IS THE STEADY STATE SO FAR FROM GIBBS