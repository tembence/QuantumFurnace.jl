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

Random.seed!(666)

#* Configs
num_qubits = 5
dim = 2^num_qubits
beta = 10.
a = beta / 10.
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
id::Matrix{ComplexF64} = I(2)
jump_paulis = [[X]]#, [Y], [Z]] #! Identity

# All jumps once
all_jumps_generated::Vector{JumpOp} = []
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
    push!(all_jumps_generated, jump)
    end
end

liouv_bohr_eh = @time construct_liouvillian_bohr_eh(all_jumps_generated, hamiltonian, with_coherent, beta, a)
liouv_bohr_gauss = @time construct_liouvillian_bohr_gauss(all_jumps_generated, hamiltonian, with_coherent, beta)
liouv_bohr_metro = @time construct_liouvillian_bohr_metro(all_jumps_generated, hamiltonian, with_coherent, beta)

liouv_eigvals, liouv_eigvecs = eigen(liouv_bohr_eh) 
steady_state_vec = liouv_eigvecs[:, end]
steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
steady_state_dm /= tr(steady_state_dm)

lambda2 = liouv_eigvals[end] - liouv_eigvals[end-1]
@printf("Lambda2: %s\n", lambda2)

@printf("Steady state closeness to Gibbs for Liouvillian (Energy): %s\n", norm(steady_state_dm - gibbs))