using LinearAlgebra
using Random
using Printf
using JLD2
include("../src/hamiltonian.jl")
include("../src/qi_tools.jl")
include("../src/structs.jl")
include("../src/bohr_picture.jl")
include("../src/energy_picture.jl")
include("../src/time_picture.jl")
include("../src/ofts.jl")
include("../src/coherent.jl")
include("../src/misc_tools.jl")
#* Unraveling in the canonical sense would not make sense for the first paper.
#* Unpacking in the way I wanted, maybe.

#* Config
num_qubits = 4
dim = 2^num_qubits
beta = 10.  # 5, 10, 30
a = beta / 50. # a = beta / 50.
b = 0.5  # b = 0.5
eta = 0.2
with_coherent = true
with_linear_combination = true
pictures = [ENERGY]
num_energy_bits = 13
w0 = 0.05
max_E = w0 * 2^num_energy_bits / 2
t0 = 2pi / (2^num_energy_bits * w0)
num_trotter_steps_per_t0 = 100
delta = 0.1

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
hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3); batch_size=100)
# hamiltonian_terms = [["X", "X"], ["Y", "Y"], ["Z", "Z"]]
# hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
# hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits)
hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
gibbs = gibbs_state_in_eigen(hamiltonian, beta)
initial_dm = Matrix{ComplexF64}(I(dim) / dim)
@assert norm(real(tr(initial_dm)) - 1.) < 1e-15 "Trace is not 1.0"
@assert norm(initial_dm - initial_dm') < 1e-15 "Not Hermitian"

#* Trotter
trotter = create_trotter(hamiltonian, t0, num_trotter_steps_per_t0)
trotter_error_T = compute_trotter_error(hamiltonian, trotter, 2^num_energy_bits * t0 / 2)
gibbs_in_trotter = trotter.eigvecs' * gibbs_state(hamiltonian, beta) * trotter.eigvecs

# compute_errors(hamiltonian, configs[1]; trotter = trotter)

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

#* Checking unraveling
delta = 0.1
T = 1.0
num_steps = Int(round(T / delta))
liouv_result_jumpsum = run_liouvillian(jumps, configs[1], hamiltonian)
exact_map_jumpsum = exp(T * liouv_result_jumpsum.data)
alg_map_jumpsum = (I(4^num_qubits) + delta * liouv_result_jumpsum.data)^num_steps
norm(exact_map_jumpsum - alg_map_jumpsum)

# Fixed point checks
# exact_map_jumpsum_eigvals, exact_map_jumpsum_eigvecs = eigen(exact_map_jumpsum) 
# steady_state_vec = exact_map_jumpsum_eigvecs[:, end]
# exact_map_jumpsum_ss = reshape(steady_state_vec, size(hamiltonian.data))
# exact_map_jumpsum_ss /= tr(exact_map_jumpsum_ss)
# norm(exact_map_jumpsum_ss - gibbs)

# alg_map_jumpsum_eigvals, alg_map_jumpsum_eigvecs = eigen(alg_map_jumpsum) 
# steady_state_vec = alg_map_jumpsum_eigvecs[:, end]
# alg_map_jumpsum_ss = reshape(steady_state_vec, size(hamiltonian.data))
# alg_map_jumpsum_ss /= tr(alg_map_jumpsum_ss)
# norm(alg_map_jumpsum_ss - gibbs)

# Gap from exact map
jumpsum_spectral_gap = abs(real(liouv_result_jumpsum.lambda_2))
jumpsum_spectral_gap_from_exact_map = 1 - exact_map_jumpsum_eigvals[end - 1]
jumpsum_spectral_gap_from_exact_map_generator = -log(1 - jumpsum_spectral_gap_from_exact_map) / T / 10 #! why is there a 10 factor
norm(jumpsum_spectral_gap - jumpsum_spectral_gap_from_exact_map_generator)

# Gap from alg map
jumpsum_spectral_gap_from_alg_map = 1 - alg_map_jumpsum_eigvals[end - 1]
jumpsum_spectral_gap_from_alg_map_generator = - log(1 - jumpsum_spectral_gap_from_alg_map) / T / 10 #! why is there a 10 factor and why is the gap bigger here
norm(jumpsum_spectral_gap - jumpsum_spectral_gap_from_alg_map_generator)

# Unraveling
total_map_delta::Matrix{ComplexF64} = I(4^num_qubits)
spectral_gaps = []
for jump in jumps
    liouv_result = run_liouvillian([jump], configs[1], hamiltonian; trotter = trotter)
    total_map_delta *= I(4^num_qubits) + delta * liouv_result.data / length(jumps)
    push!(spectral_gaps, abs(real(liouv_result.lambda_2)))
end

total_map = total_map_delta^(Int(round(T / delta)))

average_spectral_gap = sum(spectral_gaps) / length(spectral_gaps)
liouv_eigvals, liouv_eigvecs = eigen(total_map)  # exp(T L)
total_map_spectral_gap = 1 - abs(real(liouv_eigvals[end-1]))
unraveled_spectral_gap = - log(1 - total_map_spectral_gap) / T
norm(unraveled_spectral_gap - jumpsum_spectral_gap)
