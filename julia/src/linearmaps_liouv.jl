using LinearAlgebra
using LinearMaps
using Distributed
using Arpack
using Pkg

include("qi_tools.jl")
include("misc_tools.jl")
include("structs.jl")
include("oven_utensils.jl")
include("energy_picture.jl")
include("jump_workers.jl")
include("coherent.jl")

#TODO: Finish me
function create_liouvillian_map(jumps::Vector{JumpOp}, config::LiouvConfig,
    hamiltonian::HamHam;
    trotter::Union{TrottTrott, Nothing}=nothing)

    dim = size(hamiltonian.data, 1)
    validate_config!(config)
    print_press(config)
    picture_name = replace(string(typeof(config.picture)), "Picture" => "")
    println("Creating Linear Map for ($(picture_name))")

    if config.picture isa TrotterPicture
        @assert trotter !== nothing "A Trotter object must be provided for the TrotterPicture"
        ham_or_trott = trotter
    else
        ham_or_trott = hamiltonian
    end

    # Labels
    energy_labels, time_labels = precompute_labels(config.picture, config)
    # Transition rate gamma
    transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)
    # Functions for B
    f_minus, f_plus = if config.with_coherent && config.picture isa Union{TimePicture, TrotterPicture}
        _f_minus = compute_truncated_f(compute_f_minus, time_labels, config.beta)

        f_plus_calculator, f_plus_args = select_f_plus_calculator(config)
        _f_plus = compute_truncated_f(f_plus_calculator, time_labels, config.beta, f_plus_args...)
        
        (_f_minus, _f_plus)
    else
        (nothing, nothing)
    end

    if !isa(config.picture, BohrPicture)
        w0 = abs(energy_labels[2] - energy_labels[1])
    end

    if isa(config.picture, TimePicture) || isa(config.picture, TrotterPicture)
        t0 = abs(time_labels[2] - time_labels[1])
        oft_time_labels = truncate_time_labels_for_oft(time_labels, config.beta)
    end

    precomputed_data = (
        w0 = w0,
        t0 = t0,
        transition = transition,
        f_minus = f_minus,
        f_plus = f_plus,
        energy_labels = energy_labels,
        time_labels = time_labels,
        oft_time_labels = oft_time_labels
    )

    caches = (
        jump_caches = JumpCaches(dim),
        oft_caches = OFTCaches(dim)
    )

    d_rho = zeros(ComplexF64, dim, dim)
    function liouv_linearmap!(dv_vec::AbstractVector, v_vec::AbstractVector)
        rho = reshape(v_vec, size(hamiltonian.data))
        # rho ./= tr(rho) #! This might ruins it, have to comment this out for linearity

        fill!(d_rho, 0.0)
        for jump in jumps
            jump_contribution!(d_rho, config.picture, rho, jump, ham_or_trott, config, precomputed_data, caches)
        end

        copyto!(dv_vec, vec(d_rho))
        
        return dv_vec
    end
    return LinearMap{ComplexF64}(liouv_linearmap!, dim^2, ismutating=true)
end

#* TEST
#* Config
num_qubits = 6
dim = 2^num_qubits
beta = 10.  # 5, 10, 30

# Smooth Metro
a = beta / 50. # a = beta / 50.
b = 0.5  # b = 0.5
eta = 0.0  # eta = 0.2

with_coherent = true
with_linear_combination = true
picture = TimePicture()
num_energy_bits = 11
w0 = 0.05
max_E = w0 * 2^num_energy_bits / 2
t0 = 2pi / (2^num_energy_bits * w0)  # Max time evolution pi / w0
num_trotter_steps_per_t0 = 1

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
jump_paulis = [[Y]]

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


linmap = create_liouvillian_map([jumps[1]], config, hamiltonian; trotter=trotter)

# Arpack eigs
@time eigvals_near_zero, eigvecs_near_zero = eigs(linmap, nev=2, which=:SM, tol=1e-9)
ss_index = findmin(abs.(eigvals_near_zero))[2]
gap_index = (ss_index == 1) ? 2 : 1
steady_state_eigval = eigvals_near_zero[ss_index]
lambda_2 = eigvals_near_zero[gap_index] # This is the spectral gap eigenvalue
@printf("lambda 2: %s\n", lambda_2)

steady_state_vec = eigvecs_near_zero[:, ss_index]
steady_state_dm = reshape(steady_state_vec, size(hamiltonian.data))
steady_state_dm ./= tr(steady_state_dm) # Normalize
@printf("Dist to Gibbs state: %s\n", norm(steady_state_dm - hamiltonian.gibbs))

# eigvals_end, _ = eigs(linmap, nev=1, which=:LM)
# lambda_end = eigvals_end[1]

# Nevermind:
# G = randn(ComplexF64, dim, dim)
# M = G * G'                 # G * G†
# M /= tr(M)

# Mvec = vec(M)
# Magain = reshape(Mvec, size(hamiltonian.data))
# tr(Magain)
