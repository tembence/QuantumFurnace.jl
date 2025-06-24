using LinearAlgebra
using Random
using Printf
using ProgressMeter
using JLD2

include("hamiltonian.jl")
include("ofts.jl")
include("qi_tools.jl")
include("bohr_picture.jl")

ENV["COLUMNS"] = "30"
ENV["ROWS"] = "30"

function construct_liouvillian_bohr_metro_explicit(jumps::Vector{JumpOp}, hamiltonian::HamHam, 
    with_coherent::Bool, beta::Float64)

    dim = size(hamiltonian.data, 1)

    # Bohr dictionary
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
    unique_freqs = Set(keys(bohr_dict))

    liouv = zeros(ComplexF64, dim^2, dim^2)
    @showprogress desc="Liouvillian (Bohr)..." for jump in jumps
        # Coherent part
        if with_coherent
            coherent_term = coherent_metro_bohr_explicit(hamiltonian, bohr_dict, jump, beta)
            liouv .+= vectorize_liouvillian_coherent(coherent_term)
        end

        # Dissipative part
        for nu_2 in unique_freqs
            alpha_nu1_matrix = create_alpha_nu1_matrix_metro(hamiltonian.bohr_freqs, nu_2, beta)

            A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
            A_nu_2_dagger::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)

            A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
            A_nu_2_dagger .= A_nu_2'

            liouv .+= vectorize_liouvillian_diss(alpha_nu1_matrix .* jump.in_eigenbasis, A_nu_2_dagger)
        end
    end
    return liouv
end

# slightly faster than the other version
function coherent_metro_bohr_explicit(hamiltonian::HamHam, 
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
                f_nu1_nu2 = create_f_metro2(nu_1, nu_2, beta)

                B .+= f_nu1_nu2 * A_nu_2' * A_nu_1
            end
        end
    end
    return B
end

function coherent_metro_bohr_explicit2(hamiltonian::HamHam, 
    bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}}, jump::JumpOp, beta::Float64)
    B = zeros(ComplexF64, dim, dim)
    for nu in unique_freqs
        # @printf("/////////////// nu: %s\n", nu)
        B_nu = B_nu_metro(nu, hamiltonian, bohr_dict, jump, beta)
        # @printf("TOTAL B_NU ///////////////\n")
        # display(B_nu)
        B .+= B_nu
    end
    return B
end

function B_nu_metro(nu::Float64, hamiltonian::HamHam, bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}}, 
    jump::JumpOp, beta::Float64)

    dim = size(hamiltonian.data, 1)
    unique_freqs = Set(keys(bohr_dict))
    B_nu = zeros(ComplexF64, dim, dim)
    for nu_2 in unique_freqs
        good_nu1s = find_all_nu1s_to_nu2(nu_2, nu, unique_freqs)
        # @printf("nu_2: %s\n", nu_2)
        # display(good_nu1s)
        for nu_1 in collect(good_nu1s)
            # @printf("Contribution to B:\n")
            # @printf("nu_1: %s\n", nu_1)
            A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
            A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
            A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
            A_nu_1[bohr_dict[nu_1]] .= jump.in_eigenbasis[bohr_dict[nu_1]]
            f_nu1_nu2 = create_f_metro(nu_1, nu_2, beta)
            
            B_nu_temp = f_nu1_nu2 * A_nu_2' * (A_nu_1)
            # display(B_nu_temp)
            B_nu .+= B_nu_temp
        end
    end
    return B_nu
end

function R_nu_metro(nu::Float64, hamiltonian::HamHam, bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}}, 
    jump::JumpOp, beta::Float64)

    dim = size(hamiltonian.data, 1)
    unique_freqs = Set(keys(bohr_dict))
    R_nu = zeros(ComplexF64, dim, dim)
    for nu_2 in unique_freqs
        good_nu1s = find_all_nu1s_to_nu2(nu_2, nu, unique_freqs)
        # @printf("nu_2: %s\n", nu_2)
        # display(good_nu1s)
        for nu_1 in collect(good_nu1s)
            # @printf("Contribution to B:\n")
            # @printf("nu_1: %s\n", nu_1)
            A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
            A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
            A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]
            A_nu_1[bohr_dict[nu_1]] .= jump.in_eigenbasis[bohr_dict[nu_1]]
            alpha_nu1_nu2 = create_alpha_metro(nu_1, nu_2, beta)
            
            R_nu_temp = alpha_nu1_nu2 * A_nu_2' * (A_nu_1)
            # display(B_nu_temp)
            R_nu .+= R_nu_temp
        end
    end
    return R_nu
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

#* Freqs
bohr_dict::Dict{Float64, Vector{CartesianIndex{2}}} = create_bohr_dict(hamiltonian)
unique_freqs = Set(keys(bohr_dict))

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

#* f and alpha
# for nu_1 in unique_freqs
#     for nu_2 in unique_freqs
#         f = create_f_metro2(nu_1, nu_2, beta)
#         f_from_alpha = tanh(-beta * (nu_1 - nu_2) / 4) * create_alpha_metro(nu_1, nu_2, beta) / (2*im)
#         diff = norm(f - f_from_alpha)
#         display(diff)
#     end
# end

#* Construct Bohr Liouvillian
@time liouv_matrix = construct_liouvillian_bohr_metro_explicit(all_jumps_generated, hamiltonian, with_coherent, beta)
# @time liouv_matrix = construct_liouvillian_bohr_gauss(all_jumps_generated, hamiltonian, with_coherent, beta)
liouv_eigvals, liouv_eigvecs = eigen(liouv_matrix) 
steady_state_vec = liouv_eigvecs[:, end]
steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
steady_state_dm /= tr(steady_state_dm)
steady_state_vec = vec(steady_state_dm)

norm(gibbs_vec - steady_state_vec)

#* B
# jump = all_jumps_generated[2]

# # nu_2 = collect(unique_freqs)[3]
# nu = collect(unique_freqs)[8]
# @printf("nu: %s\n", nu)
# # @printf("nu2: %s\n", nu_2)
# println("Jump in eigen:")
# display(jump.in_eigenbasis)
# # good_nu1s = find_all_nu1s_to_nu2(nu_2, nu, unique_freqs)
# B = zeros(ComplexF64, dim, dim)
# for nu in unique_freqs
#     @printf("/////////////// nu: %s\n", nu)
#     B_nu = B_nu_metro(nu, hamiltonian, bohr_dict, jump, beta)
#     @printf("TOTAL B_NU ///////////////\n")
#     display(B_nu)
#     B .+= B_nu
# end
# display(B)
# norm(B' - B)

# B_explicit = @time coherent_metro_bohr_explicit(hamiltonian, bohr_dict, jump, beta)
# B_explicit2 = @time coherent_metro_bohr_explicit2(hamiltonian, bohr_dict, jump, beta)
# norm(B_explicit - B_explicit')
# norm(B_explicit2 - B_explicit2')

# norm(B_explicit - B_explicit2)




