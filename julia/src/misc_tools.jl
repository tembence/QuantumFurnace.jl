function generate_filename(config::LiouvConfig)
    pic_str = string(typeof(config.picture))
    
    beta_str = "beta=$(config.beta)"
    a_str = "a=$(config.a)"
    b_str = "b=$(config.b)"
    nqb_str = "n=$(config.num_qubits)"
    if config.with_coherent
        B = "B"
    else
        B = "noB"
    end

    return join(["liouv", pic_str, nqb_str, beta_str, B, a_str, b_str], "_") * ".bson"
end

function generate_filename(config::ThermalizeConfig)
    pic_str = string(typeof(config.picture))
    
    beta_str = "beta=$(config.beta)"
    a_str = "a=$(config.a)"
    b_str = "b=$(config.b)"
    nqb_str = "n=$(config.num_qubits)"
    if config.with_coherent
        B = "B"
    else
        B = "noB"
    end
    mix = "mix=$(config.mixing_time)"
    return join(["alg", pic_str, nqb_str, beta_str, B, a_str, b_str, mix], "_") * ".bson"  #! BSON now
end

function riemann_sum(f::Function, grid::Vector{Float64})
    """Uniform grid, rectangle method"""
    d0 = grid[2] - grid[1]
    return d0 * sum(f, grid)
end

function riemann_sum(fvals::Vector{Float64}, d0::Float64)
    return d0 * sum(fvals)
end

function riemann_sum(fvals::Vector{ComplexF64}, d0::Float64)
    return d0 * sum(fvals)
end

# function pick_transition(beta::Float64, a::Float64, b::Float64)

#     sqrtA = sqrt((4 * a / beta + 1) / 8)
#     if (b == 0 && a != 0)  # No time singularity but kinky Metro in energy
#         return w -> begin
#             sqrtB = beta * abs(w + 1 / (2 * beta)) / sqrt(2)
#             return exp((- 2 * sqrtA * sqrtB - beta * w / 2 - 1 / 4))
#         end
#     elseif (b != 0 && a != 0)  # No time singularity and no kinky Metro (Glauberish)
#         return w -> begin
#             sqrtB = beta * abs(w + 1 / (2 * beta)) / sqrt(2)
#             transition_eh = exp((- 2 * sqrtA * sqrtB - beta * w / 2 - 1 / 4))

#             return (transition_eh * (erfc(sqrtA * sqrt(b) - sqrtB / sqrt(b)) 
#                 + exp(4 * sqrtA * sqrtB) * erfc(sqrtA * sqrt(b) + sqrtB / sqrt(b))) / 2)
#         end
#     elseif a == 0  # Time singularity and kinky Metro
#         return w -> begin
#             return exp(-beta * max(w + 1/(2 * beta), 0.0))
#         end
#     end
# end

# function is_config_valid(config::Union{LiouvConfig, ThermalizeConfig})::Bool
#     errors = String[]

#     # Check based on the picture type.
#     if config.picture == BOHR
#         nothing
#     elseif config.picture == ENERGY
#         if config.num_energy_bits <= 0
#             push!(errors, "For picture ENERGY, num_energy_bits must be > 0.")
#         end
#         if config.w0 <= 0.
#             push!(errors, "For picture ENERGY, w0 must be > 0.")
#         end
#     elseif config.picture == TIME
#         if config.num_energy_bits <= 0
#             push!(errors, "For picture TIME, num_energy_bits must be > 0.")
#         end
#         if config.t0 <= 0.
#             push!(errors, "For picture TIME, t0 must be > 0.")
#         end
#         if config.w0 <= 0.
#             push!(errors, "For picture TIME, w0 must be > 0.")
#         end
#         if (config.t0 * config.w0 != 2pi/2^config.num_energy_bits)
#             push!(errors, "t0 * w0 != 2pi / N")
#         end
#         if (config.a == 0. && config.eta <= 0. && config.with_linear_combination)
#             push!(errors, "For linear combinations and picture TIME, a = 0 needs an eta > 0")
#         end 
#     elseif config.picture == TROTTER
#         if config.num_energy_bits <= 0
#             push!(errors, "For picture TROTTER, num_energy_bits must be > 0.")
#         end
#         if config.t0 <= 0.
#             push!(errors, "For picture TROTTER, t0 must be > 0.")
#         end
#         if config.w0 <= 0.
#             push!(errors, "For picture TROTTER, w0 must be > 0.")
#         end
#         if config.num_trotter_steps_per_t0 <= 0
#             push!(errors, "For picture TROTTER, num_trotter_steps_per_t0 must be > 0.")
#         end
#         if (norm(config.t0 * config.w0 - 2pi/2^config.num_energy_bits) > 1e-15)
#             push!(errors, "t0 * w0 != 2pi / N")
#         end
#         if (config.a == 0. && config.eta <= 0. && config.with_linear_combination)
#             push!(errors, "For linear combinations and picture TROTTER, a = 0 needs an eta > 0")
#         end 
#     else
#         push!(errors, "Unknown picture type.")
#     end

#     if (config.b != 0. && config.a == 0. && config.with_linear_combination)
#         push!(errors, "For linear combinations when b > 0, then we need, a > 0.")
#     end

#     if !isempty(errors)
#         for err in errors
#             println(err)
#         end
#         return false
#     end

#     return true
# end

function validate_config!(config::Union{LiouvConfig, ThermalizeConfig})
    errors = String[]

    # --- Picture-Specific Validation ---
    _collect_config_errors!(errors, config.picture, config)

    # --- Common Validation Logic ---
    if config.with_linear_combination && config.a == 0.0
        if config.b != 0.0
            push!(errors, "For linear combinations with b != 0, a must also be non-zero.")
        end
        if config.picture isa Union{TimePicture, TrotterPicture} && config.eta <= 0.0
            push!(errors, "For linear combinations with a=0 in TIME or TROTTER picture, eta must be > 0.")
        end
    end

    # --- Error Throwing ---
    if !isempty(errors)
        error_message = "Invalid configuration found:\n" * join(["  - " * err for err in errors], "\n")
        throw(ArgumentError(error_message))
    end

    return nothing
end

function _collect_config_errors!(errors::Vector{String}, ::BohrPicture, config)
    return # No specific checks
end

function _collect_config_errors!(errors::Vector{String}, ::EnergyPicture, config)
    if config.num_energy_bits <= 0
        push!(errors, "For EnergyPicture, num_energy_bits must be > 0.")
    end
    if config.w0 <= 0.0
        push!(errors, "For EnergyPicture, w0 must be > 0.")
    end
end

function _collect_config_errors!(errors::Vector{String}, ::TimePicture, config)
    if config.num_energy_bits <= 0
        push!(errors, "For TimePicture, num_energy_bits must be > 0.")
    end
    if config.t0 <= 0.0
        push!(errors, "For TimePicture, t0 must be > 0.")
    end
    if config.w0 <= 0.0
        push!(errors, "For TimePicture, w0 must be > 0.")
    end
    if !isapprox(config.t0 * config.w0, 2pi / 2^config.num_energy_bits)
        push!(errors, "For TimePicture, the relation t0 * w0 ≈ 2π / 2^N must hold.")
    end
end

function _collect_config_errors!(errors::Vector{String}, ::TrotterPicture, config)
    if config.num_energy_bits <= 0
        push!(errors, "For TrotterPicture, num_energy_bits must be > 0.")
    end
    if config.t0 <= 0.0
        push!(errors, "For TrotterPicture, t0 must be > 0.")
    end
    if config.w0 <= 0.0
        push!(errors, "For TrotterPicture, w0 must be > 0.")
    end
    if config.num_trotter_steps_per_t0 <= 0
        push!(errors, "For TrotterPicture, num_trotter_steps_per_t0 must be > 0.")
    end
    if !isapprox(config.t0 * config.w0, 2pi / 2^config.num_energy_bits)
        push!(errors, "For TrotterPicture, the relation t0 * w0 ≈ 2π / 2^N must hold.")
    end
end


function print_press(config::LiouvConfig)
    params = [
        ("picture", config.picture),
        ("num_qubits", config.num_qubits),
        ("num_energy_bits", config.num_energy_bits),
        ("beta", config.beta),
        ("a", config.a),
        ("b", config.b),
        ("eta", config.eta),
        ("t0", config.t0),
        ("w0", config.w0),
        ("with_coherent", config.with_coherent),
        ("with_linear_combination", config.with_linear_combination),
        ("num_trotter_steps_per_t0", config.num_trotter_steps_per_t0)
    ]
    provided = filter(p -> p[2] != -1.0, params)
    if isempty(provided)
        return
    end

    println("--- The Press ---")
    for (name, value) in provided
        println("$name: $value")
    end
    println("-----------------")
end

function print_press(config::ThermalizeConfig)
    params = [
        ("picture", config.picture),
        ("num_qubits", config.num_qubits),
        ("num_energy_bits", config.num_energy_bits),
        ("beta", config.beta),
        ("a", config.a),
        ("b", config.b),
        ("eta", config.eta),
        ("t0", config.t0),
        ("w0", config.w0),
        ("with_coherent", config.with_coherent),
        ("with_linear_combination", config.with_linear_combination),
        ("num_trotter_steps_per_t0", config.num_trotter_steps_per_t0),
        ("mixing time", config.mixing_time),
        ("delta", config.delta),
        ("unravel", config.unravel)
    ]
    provided = filter(p -> p[2] != -1.0, params)
    if isempty(provided)
        return
    end

    println("--- The Press ---")
    for (name, value) in provided
        println("$name: $value")
    end
    println("-----------------")
end

function create_jumpop(pauli::Vector{String}, num_qubits::Int64, site::Int64, hamiltonian::HamHam; 
    trotter::Union{TrottTrott, Nothing} = nothing)

    pauli = pauli_string_to_matrix(pauli)
    jump_op = Matrix(pad_term(pauli, num_qubits, site))
    jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
    if trotter !== nothing
        jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
    else
        jump_in_trotter_basis = zeros(2^num_qubits, 2^num_qubits)
    end
    orthogonal = (jump_op == transpose(jump_op))
    jump = JumpOp(jump_op,
            jump_op_in_eigenbasis,
            Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
            zeros(0),
            jump_in_trotter_basis,
            orthogonal)
    return jump
end

function pauli_string_to_matrix(paulistring::Vector{String})
    sigmax::Matrix{ComplexF64} = [0 1; 1 0]
    sigmay::Matrix{ComplexF64} = [0.0 -im; im 0.0]
    sigmaz::Matrix{ComplexF64} = [1 0; 0 -1]

    pauli_matrices::Vector{Matrix{ComplexF64}} = []
    pauli_dict = Dict("X" => sigmax, "Y" => sigmay, "Z" => sigmaz, "I" => Matrix{ComplexF64}(I(2)))
    for pauli_str in paulistring
        push!(pauli_matrices, pauli_dict[pauli_str])
    end
    return pauli_matrices
end