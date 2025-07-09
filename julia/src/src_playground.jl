using LinearAlgebra, Random, Printf, SparseArrays, JLD2, BSON, Arpack
using Pkg

include("hamiltonian.jl")
include("qi_tools.jl")
include("misc_tools.jl")
include("structs.jl")
include("oven.jl")
include("errors.jl")
include("oven_utensils.jl")
include("coherent.jl")
include("jump_workers.jl")
include("structs.jl")
include("energy_picture.jl")
include("bohr_picture.jl")
include("time_picture.jl")
include("trotter_picture.jl")
include("timelike_tools.jl")
include("ofts.jl")

num_qubits = 4
dim = 2^num_qubits
beta = 10.  # 5, 10, 30

# Smooth Metro
a = beta / 50. # a = beta / 50.
b = 0.5  # b = 0.5
eta = 0.0  # eta = 0.2

with_coherent = true
with_linear_combination = true
# energy_picture = EnergyPicture()
picture = TrotterPicture()
num_energy_bits = 11
w0 = 0.05
max_E = w0 * 2^num_energy_bits / 2
t0 = 2pi / (2^num_energy_bits * w0)  # Max time evolution pi / w0
num_trotter_steps_per_t0 = 10

config = LiouvConfig(
        num_qubits = num_qubits, 
        with_coherent = with_coherent,
        with_linear_combination = with_linear_combination, 
        picture = picture,
        beta = beta,
        a = a,
        b = b,
        num_energy_bits = num_energy_bits,
        w0 = w0,
        t0 = t0,
        eta = eta,
        num_trotter_steps_per_t0 = num_trotter_steps_per_t0
)

#* Hamiltonian
# Hamiltonian path
project_root = Pkg.project().path |> dirname
project_root = joinpath(project_root, "julia")  #! Omit for cluster
data_dir = joinpath(project_root, "hamiltonians")
output_filename = join(["heis", "disordered", "periodic", "n=$num_qubits"], "_") * ".jld2"
ham_path = joinpath(data_dir, output_filename)

# Load Hamiltonian
jld_ham_data = JLD2.load(ham_path)
hamiltonian = jld_ham_data["hamiltonian"]
@printf("Hamiltonian is loaded.\n")
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
hamiltonian.bohr_dict = create_bohr_dict(hamiltonian)
hamiltonian.gibbs = Hermitian(gibbs_state_in_eigen(hamiltonian, beta))
initial_dm = Matrix{ComplexF64}(I(dim) / dim)
@assert norm(real(tr(initial_dm)) - 1.) < 1e-15 "Trace is not 1.0"
@assert norm(initial_dm - initial_dm') < 1e-15 "Not Hermitian"

#* Trotter
trotter = create_trotter(hamiltonian, t0, num_trotter_steps_per_t0)
trotter_error_T = compute_trotter_error(hamiltonian, trotter, 2^num_energy_bits * t0 / 2)
gibbs_in_trotter = Hermitian(trotter.eigvecs' * gibbs_state(hamiltonian, beta) * trotter.eigvecs)
@printf("Trotter is created.\n")

#* Jumps
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
H::Matrix{ComplexF64} = [1 1; 1 -1] / sqrt(2)
id::Matrix{ComplexF64} = I(2)
jump_paulis = [[X], [Y], [Z]]

jumps::Vector{JumpOp} = []
for pauli in jump_paulis
        for site in 1:num_qubits
        jump_op = Matrix(pad_term(pauli, num_qubits, site))
        jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
        jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
        orthogonal = (jump_op == transpose(jump_op))
        jump = JumpOp(jump_op,
                jump_op_in_eigenbasis,
                Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
                zeros(0),
                jump_in_trotter_basis,
                orthogonal)
        push!(jumps, jump)
        end
end
@printf("Jumps are created.\n")

energy_labels, time_labels = precompute_labels(config.picture, config)
w = -0.12
oft_time_labels = truncate_time_labels_for_oft(time_labels, beta)
jump = jumps[9]
# oft_trott = trotter_oft(jump, w, trotter, oft_time_labels, beta) * t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
# oft_trott = trotter.trafo_from_eigen_to_trotter' * oft_trott * trotter.trafo_from_eigen_to_trotter
# oft_time = time_oft(jump, w, hamiltonian, oft_time_labels, beta) * t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
# norm(oft_w - oft_trott)
# norm(oft_w - oft_time)



oft_w = zeros(ComplexF64, dim, dim)
oft_w_old = oft(jump, w, hamiltonian, beta) * sqrt(beta / sqrt(2 * pi))
oft_fast!(oft_w, jump, w, hamiltonian, beta)
norm(oft_w * sqrt(beta / sqrt(2 * pi)) - oft_w_old)


oft_t = similar(oft_w)
oft_trott = similar(oft_w)
oft_caches = OFTCaches(dim)
time_oft_fast!(oft_t, oft_caches, jump, w, hamiltonian, oft_time_labels, beta)

oft_caches = OFTCaches(dim)
trotter_oft_fast!(oft_trott, oft_caches, jump, w, trotter, oft_time_labels, beta)
oft_trott = trotter.trafo_from_eigen_to_trotter' * oft_trott * trotter.trafo_from_eigen_to_trotter  * t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))

norm(oft_w * sqrt(beta / sqrt(2 * pi)) - oft_trott)
norm(oft_w * sqrt(beta / sqrt(2 * pi)) - oft_t * t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi)))