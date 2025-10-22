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
a = beta / 20. # a = beta / 50 or beta / 20 for b
b = 0.5  # b = 0.5
eta = 0.0  # eta = 0.2

# a = 0
# b = 0
# eta = 0.2

with_coherent = true
with_linear_combination = true
# energy_picture = EnergyPicture()
picture = TimePicture()
num_energy_bits = 14
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

#* Coherent part norm check with f+-
energy_labels, time_labels = precompute_labels(config.picture, config)
if config.with_coherent
        f_minus = compute_truncated_func(compute_f_minus, time_labels, config.beta)
        if config.with_linear_combination  
                if config.a != 0.0  # Improved Metro / Glauber
                        f_plus = compute_truncated_func(compute_f_plus_eh, time_labels, config.beta, config.a, config.b)
                else  # Metro
                        f_plus = compute_truncated_func(compute_f_plus_metro, time_labels, config.beta, config.eta)
                end
        else  # Gaussian
                f_plus = compute_truncated_func(compute_f_plus, time_labels, config.beta)
        end
end

# ! Testing for b.
if config.with_coherent
        b_minus = compute_truncated_func(compute_b_minus, time_labels)
        if config.with_linear_combination  
                if config.a != 0.0  # Improved Metro / Glauber
                        b_plus = compute_truncated_func(compute_b_plus_eh, time_labels, config.a / config.beta, config.b)
                else  # Metro
                        b_plus = compute_truncated_func(compute_b_plus_metro, time_labels, config.eta)
                end
        else  # Gaussian
                b_plus = compute_truncated_func(compute_b_plus, time_labels)
        end
end

# btimes = sort(collect(keys(b_minus)))
# bfvals = [b_minus[ti] for ti in btimes]
# plot(btimes, real.(bfvals))

# ftimes = sort(collect(keys(f_minus)))
# ffvals = [f_minus[ti] for ti in ftimes]
# plot!(ftimes, 2pi * sqrt(pi) * real.(ffvals) * beta^2)

# for t in keys(f_minus)
#         b_minus_t = b_minus[t]
#         f_minus_t = f_minus[t]
#         diff = b_minus[t] - 2pi * sqrt(pi) * f_minus[t] * beta^2
#         display(norm(diff))
# end

 #* B check
jump = jumps[1]
B_bohr = coherent_bohr(hamiltonian, jump, config)
B_f = coherent_term_time(jump, hamiltonian, f_minus, f_plus, t0)
B_b = B_time(jump, hamiltonian, b_minus, b_plus, t0, beta)

@printf("Bohr - f: %s\n", norm(B_bohr - B_f))
@printf("Bohr - b: %s\n", norm(B_bohr - B_b))


# t0 = abs(time_labels[2] - time_labels[1])

# function l1_norm(func_dict, t0)
#         l1_norm = 0.0
#         for t in keys(func_dict)
#                 l1_norm += abs(func_dict[t]) * t0
#         end
#         return l1_norm
# end

# f_plus_norm = l1_norm(f_plus, t0)
# f_minus_norm = l1_norm(f_minus, t0)
# b_plus_norm = l1_norm(b_plus, t0)

# f_plus_norm / b_plus_norm * beta #* this matches with the maths on L1 norms, careful with the same t0 labels after rescaling.

# pi*sqrt(pi)*beta / 8


# coherent_term = coherent_term_time(jump, hamiltonian, f_minus, f_plus, t0) 
# norm(coherent_term)


#* OFT checks
# energy_labels, time_labels = precompute_labels(config.picture, config)

# w = -0.12
# oft_time_labels = truncate_time_labels_for_oft(time_labels, beta)
# jump = jumps[9]
# oft_trott = trotter_oft(jump, w, trotter, oft_time_labels, beta) * t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
# oft_trott = trotter.trafo_from_eigen_to_trotter' * oft_trott * trotter.trafo_from_eigen_to_trotter
# oft_time = time_oft(jump, w, hamiltonian, oft_time_labels, beta) * t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
# norm(oft_w - oft_trott)
# norm(oft_w - oft_time)

# oft_w = zeros(ComplexF64, dim, dim)
# oft_w_old = oft(jump, w, hamiltonian, beta) * sqrt(beta / sqrt(2 * pi))
# oft_fast!(oft_w, jump, w, hamiltonian, beta)
# norm(oft_w * sqrt(beta / sqrt(2 * pi)) - oft_w_old)

# beta = 10.
# a = beta / 50
# b = 0.5
# sqrt(beta * (4*a + beta)) / (a * sqrt(1 + b))exp(-a*b/(2*beta)) /(2*pi*sqrt(pi)/beta)


# oft_t = similar(oft_w)
# oft_trott = similar(oft_w)
# oft_caches = OFTCaches(dim)
# time_oft_fast!(oft_t, oft_caches, jump, w, hamiltonian, oft_time_labels, beta)

# oft_caches = OFTCaches(dim)
# trotter_oft_fast!(oft_trott, oft_caches, jump, w, trotter, oft_time_labels, beta)
# oft_trott = trotter.trafo_from_eigen_to_trotter' * oft_trott * trotter.trafo_from_eigen_to_trotter  * t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))

# norm(oft_w * sqrt(beta / sqrt(2 * pi)) - oft_trott)
# norm(oft_w * sqrt(beta / sqrt(2 * pi)) - oft_t * t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi)))