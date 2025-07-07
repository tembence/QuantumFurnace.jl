using LinearAlgebra
using Random
using Printf
using JLD2
using Distributed
using ClusterManagers
using BenchmarkTools
using Profile

include("hamiltonian.jl")
include("qi_tools.jl")
include("misc_tools.jl")
include("structs.jl")
include("oven.jl")

#* Config
num_qubits = 3
dim = 2^num_qubits
beta = 10.  # 5, 10, 30

# Kinky Metro
a = 0.0
b = 0.0
eta = 0.2
# Smooth Metro
a = beta / 50. # a = beta / 50.
b = 0.5  # b = 0.5
eta = 0.0  # eta = 0.2

with_coherent = true
with_linear_combination = true
picture = TimePicture()
num_energy_bits = 10
w0 = 0.1
max_E = w0 * 2^num_energy_bits / 2
t0 = 2pi / (2^num_energy_bits * w0)
num_trotter_steps_per_t0 = 10

# Thermalizing configs:
mixing_time = 10.0
delta = 0.1
unravel = false
if unravel
    # mixing_time /= (4 * num_qubits)
end

config = ThermalizeConfig(
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

#* Hamiltonian
# hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=100)
# hamiltonian_terms = [["X", "X"], ["Y", "Y"], ["Z", "Z"]]
# hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
# hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits)

# Hamiltonian path
project_root = Pkg.project().path |> dirname
project_root = joinpath(project_root, "julia")  #! Omit for cluster
data_dir = joinpath(project_root, "data")
output_filename = join(["heis", "disordered", "periodic", "n=$num_qubits"], "_") * ".jld2"
ham_path = joinpath(data_dir, output_filename)
jld_ham_data = JLD2.load(ham_path)  # Load Hamiltonian

hamiltonian = jld_ham_data["hamiltonian"]
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
hamiltonian.bohr_dict = create_bohr_dict(hamiltonian)
hamiltonian.gibbs = Hermitian(gibbs_state_in_eigen(hamiltonian, beta))
initial_dm = Matrix{ComplexF64}(I(dim) / dim)
@assert norm(real(tr(initial_dm)) - 1.) < 1e-15 "Trace is not 1.0"
@assert norm(initial_dm - initial_dm') < 1e-15 "Not Hermitian"

#* Trotter
trotter = create_trotter(hamiltonian, t0, num_trotter_steps_per_t0)
trotter_error_T = compute_trotter_error(hamiltonian, trotter, 2^num_energy_bits * t0 / 2)

#* Jumps
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
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

#* Thermalization
alg_results = @time run_thermalization_fast(jumps, config, initial_dm, hamiltonian; trotter=trotter)
@printf("\n Last distance to Gibbs: %s\n", alg_results.distances_to_gibbs[end])
# plot(alg_results.time_steps, alg_results.distances_to_gibbs, label="Distance to Gibbs", xlabel="Time", ylabel="Distance", title="Distance to Gibbs over time")

# Save
# project_root = Pkg.project().path |> dirname
# project_root = joinpath(project_root, "julia")  #! Omit this on cluster
# results_dir = joinpath(project_root, "results")
# output_filename = generate_filename(config)
# full_path = joinpath(results_dir, output_filename)

# println("Saving results to: ", full_path)
# JLD2.jldsave(full_path; results=alg_results)

# Load
# jld_data = JLD2.load(full_path)
# loaded_results = jld_data["results"]