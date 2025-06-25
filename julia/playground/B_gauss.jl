using LinearAlgebra
using Random
using Printf
using ProgressMeter
using JLD2

include("../src/hamiltonian.jl")
include("../src/qi_tools.jl")
include("../src/structs.jl")
include("../src/bohr_picture.jl")
include("../src/energy_picture.jl")
include("../src/time_picture.jl")
include("../src/ofts.jl")
include("../src/coherent.jl")
include("../src/misc_tools.jl")

ENV["COLUMNS"] = "30"
ENV["ROWS"] = "30"

function construct_liouvillian_bohr_gauss_explicit(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, beta::Float64)

    dim = size(hamiltonian.data, 1)

    # Bohr dictionary
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
    unique_freqs = Set(keys(bohr_dict))

    liouv = zeros(ComplexF64, dim^2, dim^2)
    @showprogress desc="Liouvillian (Bohr)..." for jump in jumps
        # Coherent part
        if with_coherent
            coherent_term = coherent_gauss_bohr_explicit(hamiltonian, bohr_dict, jump, beta)
            liouv .+= vectorize_liouvillian_coherent(coherent_term)
        end

        # Dissipative part
        for nu_2 in unique_freqs
            alpha_nu1_matrix = create_alpha_nu1_matrix(hamiltonian.bohr_freqs, nu_2, beta)

            A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
            A_nu_2_dagger::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)

            A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
            A_nu_2_dagger .= A_nu_2'

            liouv .+= vectorize_liouvillian_diss(alpha_nu1_matrix .* jump.in_eigenbasis, A_nu_2_dagger)
        end
    end
    return liouv
end

function coherent_gauss_bohr_explicit(hamiltonian::HamHam, 
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}}, jump::JumpOp, beta::Float64)

    dim = size(hamiltonian.data, 1)
    unique_freqs = Set(keys(bohr_dict))

    B = zeros(ComplexF64, dim, dim)

    for nu_1 in unique_freqs
        for nu_2 in unique_freqs
            nu = nu_1 - nu_2
            if !(nu in unique_freqs)
                continue
            else
                # @printf("Contribution to B when nu1 - nu2 is: %s\n", nu)
                A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                A_nu_1[bohr_dict[nu_1]] .= jump.in_eigenbasis[bohr_dict[nu_1]]
                A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
                f_nu1_nu2 = create_f_gauss(nu_1, nu_2, beta)

                B .+= f_nu1_nu2 * A_nu_2' * A_nu_1
            end
        end
    end
    return B
end

#* Parameters
num_qubits = 4
dim = 2^num_qubits
beta = 10.
Random.seed!(666)

with_coherent = true

#* Hamiltonian
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=10)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

#* Gibbs
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
gibbs_vec = vec(gibbs)
gibbs_largest_eigval = real(eigen(gibbs).values[1])

#* Jump operators
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
H::Matrix{ComplexF64} = [1 1; 1 -1] / sqrt(2)
jump_paulis = [[X]]#, [Y], [Z], [H]]

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
# --- 

#* Construct Bohr Liouvillian
@time liouv_matrix = construct_liouvillian_bohr_gauss_explicit(all_jumps_generated, hamiltonian, with_coherent, beta)
liouv_eigvals, liouv_eigvecs = eigen(liouv_matrix) 
steady_state_vec = liouv_eigvecs[:, end]
steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
steady_state_dm /= tr(steady_state_dm)
steady_state_vec = vec(steady_state_dm)

norm(gibbs_vec - steady_state_vec)

#* B
Random.seed!(667)
bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
unique_freqs = Set(keys(bohr_dict))
jump = all_jumps_generated[2]

B_explicit = @time coherent_gauss_bohr_explicit(hamiltonian, bohr_dict, jump, beta)
norm(B_explicit)

