using LinearAlgebra
using Plots

num_qubits = 4
num_liouv_steps = 10

#* Hamiltonian
ham_filename(n) = @sprintf("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n%d.jld", n)
hamiltonian = load(ham_filename(num_qubits))["ideal_ham"]
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)

#* Jump operators
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
jump_paulis = [[X], [Y], [Z]]

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

#* Heatmap of jumps
jump_products = zeros(2^num_qubits, 2^num_qubits)
for jump in all_jumps_generated
    jump_products += jump.in_eigenbasis
end
jump_products /= sqrt(norm(jump_products))

for _ in 1:(num_liouv_steps - 1)
    one_step_jump_sum = zeros(2^num_qubits, 2^num_qubits)
    for jump in all_jumps_generated
        one_step_jump_sum += jump.in_eigenbasis
    end
    jump_products *= one_step_jump_sum
    jump_products /= sqrt(norm(jump_products))
end


xticks = 1:2^num_qubits
yticks = 1:2^num_qubits
@printf("Number of zero elements in jump products: %d\n", sum(jump_products .== 0))
heatmap(abs.(jump_products), color=:thermal, xticks=xticks, yticks=(yticks, string.(reverse(yticks))), 
xlabel="i", ylabel="j", xmirror=true, title="Entries of all jumps after $num_liouv_steps Liouvillian steps")
