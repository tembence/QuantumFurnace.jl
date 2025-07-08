using LinearAlgebra
using Random
using Printf
using JLD2
using Base
using Random
using SpecialFunctions: erfc
using QuadGK
using BenchmarkTools
using Pkg

include("../src/hamiltonian.jl")
include("../src/qi_tools.jl")
include("../src/structs.jl")
include("../src/bohr_picture.jl")
include("../src/energy_picture.jl")
include("../src/time_picture.jl")
include("../src/ofts.jl")
include("../src/coherent.jl")
include("../src/misc_tools.jl")
include("../src/jump_workers.jl")
include("../src/oven.jl")
include("../src/oven_utensils.jl")

function showall(io, x, limit = false) 
    println(io, summary(x), ":")
    Base.print_matrix(IOContext(io, :limit => limit), x)
end

#* Config
num_qubits = 5
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
num_trotter_steps_per_t0 = 1

mixing_time = 10.0
delta = 0.1
unravel = false

config_therm = ThermalizeConfig(
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
    num_trotter_steps_per_t0 = num_trotter_steps_per_t0, 
    mixing_time = mixing_time,
    delta = delta,
    unravel = unravel
)

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
# jump_paulis = [[X]]

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

#* Liouv jump contributions test
labels = precompute_labels(config.picture, config)
liouv = zeros(ComplexF64, dim^2, dim^2)
liouv_le_ultimate = zeros(ComplexF64, dim^2, dim^2)
@time liouv .+= jump_contribution(config.picture, jumps[5], trotter, config, labels...)
@time liouv_le_ultimate .+= jump_contribution_le_ultimate(config.picture, jumps[5], trotter, config, labels...)
norm(liouv - liouv_le_ultimate)

#* Thermalization jump contributions test
# energy_labels = precompute_labels(config_therm.picture, config_therm)[1]

# transition = pick_transition(config_therm.beta, config_therm.a, config_therm.b, config_therm.with_linear_combination)
#     # Functions for B
# f_minus, f_plus = if config_therm.with_coherent && config_therm.picture isa Union{TimePicture, TrotterPicture}
#         _f_minus = compute_truncated_f(compute_f_minus, time_labels, config_therm.beta)

#         f_plus_calculator, f_plus_args = select_f_plus_calculator(config_therm)
#         _f_plus = compute_truncated_f(f_plus_calculator, time_labels, config_therm.beta, f_plus_args...)

#         (_f_minus, _f_plus)
# else
#         (nothing, nothing)
# end

# dm_contribution = zeros(ComplexF64, dim, dim)
# dm_contribution_fast = zeros(ComplexF64, dim, dim)
# @time dm_contribution .+= jump_contribution(config_therm.picture, initial_dm, jumps[5], hamiltonian, config_therm)
# @time dm_contribution_fast .+= jump_contribution_fast(config_therm.picture, initial_dm, jumps[5], hamiltonian, config_therm)
# norm(dm_contribution - dm_contribution_fast)






#kron!
# dim = 2^7
# A = rand(ComplexF64, dim, dim)
# B = rand(ComplexF64, dim, dim)
# E = rand(ComplexF64, dim, dim)
# C1 = zeros(ComplexF64, dim^2, dim^2)
# kron!(C1, A, B)
# norm(C1 - C2)

# C2 = kron(A, B)
# D = kron(A, E)
# @time res = kron(A, B) + 2 * kron(A, E)


# C3 = zeros(ComplexF64, dim^2, dim^2)
# @time begin
#     kron_axpy!(C3, A, B, 1.0)
#     kron_axpy!(C3, A, E, 2.0)
# end
# norm(res - C3)

# better vectorization
# n = 4
# dim = 2^n
# J1 = rand(ComplexF64, dim, dim)
# J2 = rand(ComplexF64, dim, dim)
# liouv_ultimate = zeros(ComplexF64, dim^2, dim^2)
# liouv = zeros(ComplexF64, dim^2, dim^2)
# @time begin
#     liouv .= vectorize_liouvillian_diss(J1)
#     liouv .+= 2im * vectorize_liouvillian_diss(J2)
# end
# @time begin
#     vectorize_liouv_diss_and_add!(liouv_ultimate, J1, 1.0)
#     vectorize_liouv_diss_and_add!(liouv_ultimate, J2, 2im)
# end

# norm(liouv - liouv_ultimate) / norm(liouv_ultimate)


# println("done")
# norm(liouv - liouv_better)

# H = find_ideal_heisenberg(n, fill(1.0, 3); batch_size=10)
# B_better = vectorize_liouvillian_coherent_better(H.data)
# B = vectorize_liouvillian_coherent(H.data)
# norm(B - B_better)

