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

function pick_transition(beta::Float64, a::Float64, b::Float64, with_linear_combination::Bool)

    if with_linear_combination  # Gaussian case
        return w -> begin
            return exp(-beta^2 * (w + 1/beta)^2 /2)
        end
    end

    sqrtA = sqrt((4 * a / beta + 1) / 8)
    if (b == 0 && a != 0)  # No time singularity but kinky Metro in energy
        return w -> begin
            sqrtB = beta * abs(w + 1 / (2 * beta)) / sqrt(2)
            return exp((- 2 * sqrtA * sqrtB - beta * w / 2 - 1 / 4))
        end
    elseif (b != 0 && a != 0)  # No time singularity and no kinky Metro (Glauberish)
        return w -> begin
            sqrtB = beta * abs(w + 1 / (2 * beta)) / sqrt(2)
            transition_eh = exp((- 2 * sqrtA * sqrtB - beta * w / 2 - 1 / 4))

            return (transition_eh * (erfc(sqrtA * sqrt(b) - sqrtB / sqrt(b)) 
                + exp(4 * sqrtA * sqrtB) * erfc(sqrtA * sqrt(b) + sqrtB / sqrt(b))) / 2)
        end
    elseif a == 0  # Time singularity and kinky Metro
        return w -> begin
            return exp(-beta * max(w + 1/(2 * beta), 0.0))
        end
    end
end

function pick_transition(beta::Float64, a::Float64, b::Float64)

    sqrtA = sqrt((4 * a / beta + 1) / 8)
    if (b == 0 && a != 0)  # No time singularity but kinky Metro in energy
        return w -> begin
            sqrtB = beta * abs(w + 1 / (2 * beta)) / sqrt(2)
            return exp((- 2 * sqrtA * sqrtB - beta * w / 2 - 1 / 4))
        end
    elseif (b != 0 && a != 0)  # No time singularity and no kinky Metro (Glauberish)
        return w -> begin
            sqrtB = beta * abs(w + 1 / (2 * beta)) / sqrt(2)
            transition_eh = exp((- 2 * sqrtA * sqrtB - beta * w / 2 - 1 / 4))

            return (transition_eh * (erfc(sqrtA * sqrt(b) - sqrtB / sqrt(b)) 
                + exp(4 * sqrtA * sqrtB) * erfc(sqrtA * sqrt(b) + sqrtB / sqrt(b))) / 2)
        end
    elseif a == 0  # Time singularity and kinky Metro
        return w -> begin
            return exp(-beta * max(w + 1/(2 * beta), 0.0))
        end
    end
end

function is_config_valid(config::Union{LiouvConfig, ThermalizeConfig})::Bool
    errors = String[]

    # Check based on the picture type.
    if config.picture == BOHR
        nothing
    elseif config.picture == ENERGY
        if config.num_energy_bits <= 0
            push!(errors, "For picture ENERGY, num_energy_bits must be > 0.")
        end
        if config.w0 <= 0.
            push!(errors, "For picture ENERGY, w0 must be > 0.")
        end
    elseif config.picture == TIME
        if config.num_energy_bits <= 0
            push!(errors, "For picture TIME, num_energy_bits must be > 0.")
        end
        if config.t0 <= 0.
            push!(errors, "For picture TIME, t0 must be > 0.")
        end
        if config.w0 <= 0.
            push!(errors, "For picture TIME, w0 must be > 0.")
        end
        if (config.t0 * config.w0 != 2pi/2^config.num_energy_bits)
            push!(errors, "t0 * w0 != 2pi / N")
        end
        if (config.a == 0. && config.eta <= 0. && config.with_linear_combination)
            push!(errors, "For linear combinations and picture TIME, a = 0 needs an eta > 0")
        end 
    elseif config.picture == TROTTER
        if config.num_energy_bits <= 0
            push!(errors, "For picture TROTTER, num_energy_bits must be > 0.")
        end
        if config.t0 <= 0.
            push!(errors, "For picture TROTTER, t0 must be > 0.")
        end
        if config.w0 <= 0.
            push!(errors, "For picture TROTTER, w0 must be > 0.")
        end
        if config.num_trotter_steps_per_t0 <= 0
            push!(errors, "For picture TROTTER, num_trotter_steps_per_t0 must be > 0.")
        end
        if (norm(config.t0 * config.w0 - 2pi/2^config.num_energy_bits) > 1e-15)
            push!(errors, "t0 * w0 != 2pi / N")
        end
        if (config.a == 0. && config.eta <= 0. && config.with_linear_combination)
            push!(errors, "For linear combinations and picture TROTTER, a = 0 needs an eta > 0")
        end 
    else
        push!(errors, "Unknown picture type.")
    end

    if (config.b != 0. && config.a == 0. && config.with_linear_combination)
        push!(errors, "For linear combinations when b > 0, then we need, a > 0.")
    end

    if !isempty(errors)
        for err in errors
            println(err)
        end
        return false
    end

    return true
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
