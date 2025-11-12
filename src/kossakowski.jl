using QuantumFurnace
using LinearAlgebra
using SparseArrays
using Pkg
using BSON 
using Printf

const num_qubits = 4
const dim = 2^num_qubits
const beta = 10.0
const a = 0.0
const b = 0.0

# const N_nu = 6000 # Total number of Bohr frequencies.
# const nus = collect(LinRange(-0.5, 0.5, N_nu))
# const W = 4.0 / beta # 4 sigmas

# kossakowski = zeros(Float64, N_nu, N_nu)

# for i in 1:N_nu
#     for j in i:N_nu # Loop over upper triangle
#         val = create_alpha(nus[i], nus[j], beta, a, b)
#         kossakowski[i, j] = val
#         kossakowski[j, i] = val # Enforce symmetry
#     end
# end

# eigvecs, eigvals = @time eigen(Hermitian(kossakowski))

project_root = Pkg.project().path |> dirname
data_dir = joinpath(project_root, "hamiltonians")
output_filename = join(["heis", "disordered", "periodic", "n$num_qubits"], "_") * ".bson"
ham_path = joinpath(data_dir, output_filename)

# Load Hamiltonian
bson_ham_data = BSON.load(ham_path)
hamiltonian = bson_ham_data[:hamiltonian]
@printf("Hamiltonian is loaded.\n")
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
hamiltonian.bohr_dict = create_bohr_dict(hamiltonian)
hamiltonian.nu_min
unique_freqs = keys(hamiltonian.bohr_dict)
hamiltonian.gibbs = Hermitian(gibbs_state_in_eigen(hamiltonian, beta))
initial_dm = Matrix{ComplexF64}(I(dim) / dim)

X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
H::Matrix{ComplexF64} = [1 1; 1 -1] / sqrt(2)
id::Matrix{ComplexF64} = I(2)
jump_paulis = [[X], [Y], [Z]]

num_of_jumps = length(jump_paulis) * num_qubits
jump_normalization = sqrt(num_of_jumps)
jumps = []
for pauli in jump_paulis
    for site in 1:num_qubits
        jump_op = Matrix(pad_term(pauli, num_qubits, site)) / jump_normalization
        jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
        orthogonal = (jump_op == transpose(jump_op))
        jump = JumpOp(jump_op,
                jump_op_in_eigenbasis,
                orthogonal)
        push!(jumps, jump)
    end
end

jump_sum = zeros(dim, dim)
for jump in jumps
    jump_sum += jump.in_eigenbasis
end

threshold = 1e-10
jumps[1].in_eigenbasis[abs.(jump_sum) .< threshold] .= 0.0

sparse(jumps[1].in_eigenbasis)


