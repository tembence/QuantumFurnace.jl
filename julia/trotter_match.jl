using LinearAlgebra
using Random
using Printf

include("hamiltonian.jl")
include("qi_tools.jl")
include("misc_tools.jl")
include("structs.jl")
include("oven.jl")
include("timelike_tools.jl")
include("coherent.jl")
include("ofts.jl")
include("energy_picture.jl")
include("time_picture.jl")

num_qubits = 4
dim = 2^num_qubits
beta = 10.
a = beta / 50.
b = 0.5
eta = 0.0
with_coherent = true
with_linear_combination = true
num_energy_bits = 13
w0 = 0.01
t0 = 2pi / (2^num_energy_bits * w0)
num_trotter_steps_per_t0 = 1000

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

energy_labels = create_energy_labels(num_energy_bits, w0)
truncated_energy_labels = truncate_energy_labels(energy_labels, beta, a, b, with_linear_combination)
time_labels = energy_labels .* (t0 / w0)

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

#* Trotter OFT checks
# jump = jumps[2]
# w = 0.12
# energy_labels = create_energy_labels(num_energy_bits, w0)
# truncated_energy_labels = truncate_energy_labels(energy_labels, beta, a, b, with_linear_combination)
# time_labels = energy_labels .* (t0 / w0)

# time_labels_for_oft = truncate_time_labels_for_oft(time_labels, beta)
# oft_trott = @time trotter_oft(jump, w, trotter, time_labels_for_oft, beta) * t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
# oft_trott_ineigen  = hamiltonian.eigvecs' * trotter.eigvecs * oft_trott * trotter.eigvecs' * hamiltonian.eigvecs
# oft_w = oft(jump, w, hamiltonian, beta) * sqrt(beta / sqrt(2 * pi))
# oft_t = @time time_oft(jump, w, hamiltonian, time_labels, beta) * t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))

# norm(oft_trott_ineigen - oft_t)
# norm(oft_t - oft_w)

liouv_bohr = nothing
liouv_time = nothing
liouv_trotter = nothing
pictures = [BOHR, TIME]
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

    print_press(config)

    if picture == BOHR
        liouv_bohr = construct_liouvillian(jumps, config; hamiltonian=hamiltonian)
    elseif picture == TIME
        liouv_time = construct_liouvillian(jumps, config; hamiltonian=hamiltonian)
    elseif picture == TROTTER
        liouv_trotter = construct_liouvillian(jumps, config; trotter=trotter)
    end
end

@printf("Distance time-bohr: %s\n", norm(liouv_bohr - liouv_time))
@printf("Distance time-trotter: %s\n", norm(liouv_trotter - liouv_time))
@printf("Distance trotter-bohr: %s\n", norm(liouv_trotter - liouv_bohr))

