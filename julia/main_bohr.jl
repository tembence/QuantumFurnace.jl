using LinearAlgebra
using Random
using Printf
using ProgressMeter
using JLD

include("hamiltonian_tools.jl")
include("jump_op_tools.jl")
include("trotter.jl")
include("qi_tools.jl")
include("thermalizing_tools.jl")
include("coherent.jl")
include("spectral_analysis_tools.jl")

#* Parameters
num_qubits = 3
delta = 0.01
mixing_time = 1 * delta
num_liouv_steps = Int(round(mixing_time / delta, digits=0))
beta = 10.
eta = 0.02  # Just don't make it smaller than 0.018
eig_index = 3
Random.seed!(666)

save_it = false
with_coherent = true

#* Hamiltonian
ham_filename(n) = @sprintf("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n%d.jld", n)
hamiltonian = load(ham_filename(num_qubits))["ideal_ham"]
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
bohr_dict = create_bohr_dict(hamiltonian)

initial_dm_OG = diagm([0.25, 0.75])

# maxmixed = hamiltonian.eigvecs' * I(2^num_qubits) / 2^num_qubits * hamiltonian.eigvecs
# maxmixed /= tr(maxmixed)
# ones_dm = ones(ComplexF64, 2^num_qubits, 2^num_qubits)
# ones /= tr(ones)
# initial_dm = ones

#* Gibbs
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
gibbs_vec = vec(gibbs)
gibbs_largest_eigval = real(eigen(gibbs).values[1])

#* Jump operators
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
H::Matrix{ComplexF64} = [1 1; 1 -1] / sqrt(2)
jump_paulis = [[X], [Y], [Z], [H]]

# All jumps once
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

#* The Press
@printf("Number of qubits: %d\n", num_qubits)

#* Construct Bohr Liouvillian
liouv_matrix = construct_liouvillian_gauss_bohr(all_jumps_generated, hamiltonian, bohr_dict, with_coherent, beta)
liouv_eigvals, liouv_eigvecs = eigen(liouv_matrix) 
steady_state_vec = liouv_eigvecs[:, end]
steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
steady_state_dm /= tr(steady_state_dm)
steady_state_vec = vec(steady_state_dm)

#* Steady state, Gibbs?
norm(gibbs_vec - steady_state_vec)