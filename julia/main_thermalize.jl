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
beta = 10.
a = beta / 50. # a = beta / 50.
b = 0.5  # b = 0.5
eta = 0.2
with_coherent = true
with_linear_combination = true
picture = TrotterPicture()
num_energy_bits = 12
w0 = 0.02
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
validate_config!(config)

#* Hamiltonian
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=100)
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
results = @time run_thermalization_fast(jumps, config, initial_dm, hamiltonian; trotter=trotter)
@printf("\n Last distance to Gibbs: %s\n", results.distances_to_gibbs[end])

# results = @time run_thermalization(jumps, config, initial_dm, hamiltonian; trotter=trotter)
# @printf("\n Last distance to Gibbs: %s\n", results.distances_to_gibbs[end])

# Profile.print()

# open("profile_output.txt", "w") do f
#     # Redirect the output of Profile.print() to the file
#     Profile.print(f)
# end

# Save
results_dir = joinpath(@__DIR__, "results")
output_filename = generate_filename(config)
full_path = joinpath(results_dir, output_filename)

println("Saving results to: ", full_path)
JLD2.jldsave(full_path; results=results)

# Load
# jld_data = JLD2.load(full_path)
# loaded_results = jld_data["results"]










# #* Hamiltonian
# ham_filename(n) = @sprintf("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n%d.jld", n)
# hamiltonian = load(ham_filename(num_qubits))["ideal_ham"]

# if save_it == true
#     if with_coherent == false
#         filename(furnace, n, r) = @sprintf("/Users/bence/code/liouvillian_metro/julia/data/%s_n%d_r%d.jld", furnace, n, r)
#     else
#         filename(furnace, n, r) = @sprintf("/Users/bence/code/liouvillian_metro/julia/data/%s_n%d_r%d_B_100.jld", furnace, n, r)
#     end
#     # Save objects with jld
#     save(filename(furnace, num_qubits, num_energy_bits), "alg_results", therm_results, "liouv", liouv)
# end
# lll = load(filename(furnace, num_qubits, num_energy_bits))["liouv"]
# rrr = load(filename(furnace, num_qubits, num_energy_bits))["alg_results"]

# plot(therm_results.time_steps, therm_results.distances_to_gibbs, label="Distance to Gibbs", xlabel="Time", ylabel="Distance", title="Distance to Gibbs over time")
