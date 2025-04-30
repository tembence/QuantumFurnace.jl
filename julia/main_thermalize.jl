using LinearAlgebra
using Random
using Printf
using JLD

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
with_linear_combination = false
pictures = [TROTTER]
num_energy_bits = 13
w0 = 0.01
t0 = 2pi / (2^num_energy_bits * w0)
num_trotter_steps_per_t0 = 10
# Thermalizing configs:
mixing_time = 10.0
delta = 0.1
unravel = false
if unravel
    # mixing_time /= (4 * num_qubits)
end

configs::Vector{ThermalizeConfig} = []
for picture in pictures
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
    is_config_valid(config)
    push!(configs, config)
end

#* Hamiltonian
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=100)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
initial_dm = Matrix{ComplexF64}(I(dim) / dim)
@assert norm(real(tr(initial_dm)) - 1.) < 1e-15 "Trace is not 1.0"
@assert norm(initial_dm - initial_dm') < 1e-15 "Not Hermitian"

#* Trotter
trotter = create_trotter(hamiltonian, t0, num_trotter_steps_per_t0)
trotter_error_T = compute_trotter_error(hamiltonian, trotter, 2^num_energy_bits * t0)
gibbs_in_trotter = trotter.eigvecs' * gibbs_state(hamiltonian, beta) * trotter.eigvecs

#* Jumps
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
H::Matrix{ComplexF64} = [1 1; 1 -1] / sqrt(2)
id::Matrix{ComplexF64} = I(2)
jump_paulis = [[X], [Y], [Z], [H]]

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
# results = thermalize(jumps, configs[1], initial_dm; hamiltonian=hamiltonian)
results = thermalize(jumps, configs[1], initial_dm; trotter=trotter)
plot(results.time_steps, results.distances_to_gibbs)
println(results.distances_to_gibbs[end])













#TODO: memory map parallelization
# #* Hamiltonian
# ham_filename(n) = @sprintf("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n%d.jld", n)
# hamiltonian = load(ham_filename(num_qubits))["ideal_ham"]

# #* ////////////////////////// Algorithm //////////////////////////
# if furnace == GAUSS
#     therm_results = thermalize_gaussian(all_jumps_generated, hamiltonian, with_coherent, initial_dm, num_energy_bits,
#     filter_gauss_w, transition, delta, mixing_time, beta)
# elseif furnace == METRO
#     therm_results = thermalize_metro(all_jumps_generated, hamiltonian, with_coherent, initial_dm,
#     num_energy_bits, filter_gauss_w, transition, eta, delta, mixing_time, beta)
# elseif furnace == TROTT_GAUSS
#     therm_results = thermalize_gaussian_trotter(all_jumps_generated, hamiltonian, trotter, with_coherent, initial_dm,
#     num_energy_bits, filter_gauss_t, transition, delta, mixing_time, beta)
# elseif furnace == TROTT_METRO
#     therm_results = thermalize_metro_trotter(all_jumps_generated, hamiltonian, trotter, with_coherent, initial_dm,
#     num_energy_bits, filter_gauss_t, transition, eta, delta, mixing_time, beta)
# elseif furnace == TIME_GAUSS
#     therm_results = thermalize_gaussian_ideal_time(all_jumps_generated, hamiltonian, with_coherent, initial_dm, num_energy_bits,
#     filter_gauss_t, transition, delta, mixing_time, beta)
# end

# evolved_dm = therm_results.evolved_dm
# distances_to_gibbs = therm_results.distances_to_gibbs

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
