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
num_qubits = 2
dim = 2^num_qubits
beta = 10.
a = beta / 50 # a = beta / 50.
b = 0.5  # b = 0.5
eta = 0.0
with_coherent = true
with_linear_combination = false
pictures = [TROTTER]
num_energy_bits = 12
w0 = 0.01
t0 = 2pi / (2^num_energy_bits * w0)
num_trotter_steps_per_t0 = 1

configs::Vector{LiouvConfig} = []
for picture in pictures
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
    is_config_valid(config)
    push!(configs, config)
end

#* Hamiltonian
# hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=100)
hamiltonian_terms = [["X", "X"], ["Y", "Y"], ["Z", "Z"]]
hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
initial_dm = Matrix{ComplexF64}(I(dim) / dim)
@assert norm(real(tr(initial_dm)) - 1.) < 1e-15 "Trace is not 1.0"
@assert norm(initial_dm - initial_dm') < 1e-15 "Not Hermitian"

plot!(hamiltonian.eigvals, exp.(-beta * hamiltonian.eigvals) / sum(exp.(-beta * hamiltonian.eigvals)))

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
jump_paulis = [[X], [Y], [Z]]  # Omitting H for faster runtimes

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

# energy_labels = create_energy_labels(num_energy_bits, w0)
# truncated_energy_labels = truncate_energy_labels(energy_labels, beta,
# a, b, with_linear_combination)
# time_labels = energy_labels .* (t0 / w0)
# oft_time_labels = truncate_time_labels_for_oft(time_labels, beta; cutoff=1e-5)
# maximum(oft_time_labels)

#* Liouvillian
liouv_result = run_liouvillian(jumps, configs[1], hamiltonian; trotter = trotter)
norm(liouv_result.fixed_point - gibbs)
# liouv_energy = construct_liouvillian(jumps, configs[2]; hamiltonian=hamiltonian)
# liouv_time = construct_liouvillian(jumps, configs[3]; hamiltonian=hamiltonian)
# liouv_trotter = construct_liouvillian(jumps, configs[1]; trotter=trotter)

# norm(liouv_bohr - liouv_time)
# norm(liouv_trotter - liouv_time)
# norm(liouv_time - liouv_energy)

#* Spectral analysis
# norm(real.(liouv_eigvals) - liouv_eigvals)
# opt_delta =  2 / (abs(liouv_eigvals[1]) + abs(liouv_eigvals[end-1]))
# stability_barrier = 2 / (abs(liouv_eigvals[end-1]))








#TODO: memory map parallelization
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
