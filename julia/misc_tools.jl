#TODO: Finish this
# function quadrature_error(fvals::Vector{Float64}, grid::Vector{Float64})
# end

# function quadrature_error(fvals::Vector{Float64}, analytic_vals)
# end

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

function print_press(config::LiouvConfig)
    params = [
        ("num_qubits", config.num_qubits),
        ("num_energy_bits", config.num_energy_bits),
        ("beta", config.beta),
        ("a", config.a),
        ("b", config.b),
        ("eta", config.eta),
        ("t0", config.t0),
        ("w0", config.w0),
        ("with_coherent", config.with_coherent),
        ("with_linear_combination", config.with_linear_combination)
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

#TODO: Rewrite this whole thing such that, only the actual problems are showing up as errors
function is_config_valid(config::Union{LiouvConfig})::Bool
    errors = String[]

    # (a,b) are only for linear combinations
    if !(config.with_linear_combination)
        if config.a != 0.
            push!(errors, "For the Gaussian (no linear combination) case, a must be 0.")
        end
        if config.b != 0.
            push!(errors, "For the Gaussian (no linear combination) case, b must be 0.")
        end
        if config.eta > 0
            push!(errors, "For the Gaussian (no linear combination) case, eta is not needed")
        end
    end

    # Check based on the picture type.
    if config.picture == BOHR
        if config.num_energy_bits != -1
            push!(errors, "For picture BOHR, num_energy_bits should be -1.")
        end
        if config.t0 != -1.
            push!(errors, "For picture BOHR, t0 should be -1.")
        end
        if config.w0 != -1.
            push!(errors, "For picture BOHR, w0 should be -1.")
        end
        if config.eta != -1.
            push!(errors, "For picture BOHR, eta should be -1.")
        end
        if config.num_trotter_steps_per_t0 != -1.
            push!(errors, "For picture BOHR, num_trotter_steps_per_t0 should be -1.")
        end
    elseif config.picture == ENERGY
        if config.num_energy_bits == -1
            push!(errors, "For picture ENERGY, num_energy_bits must be > 0.")
        end
        if config.w0 == -1.
            push!(errors, "For picture ENERGY, w0 must be > 0.")
        end
        if config.t0 != -1.
            push!(errors, "For picture ENERGY, t0 should be -1.")
        end
        if config.num_trotter_steps_per_t0 != -1.
            push!(errors, "For picture ENERGY, num_trotter_steps_per_t0 should be -1.")
        end
    elseif config.picture == TIME
        if config.num_energy_bits == -1
            push!(errors, "For picture TIME, num_energy_bits must be > 0.")
        end
        if config.t0 == -1.
            push!(errors, "For picture TIME, t0 must be > 0.")
        end
        if config.w0 == -1.
            push!(errors, "For picture TIME, w0 must be > 0.")
        end
        if config.num_trotter_steps_per_t0 != -1.
            push!(errors, "For picture TIME, num_trotter_steps_per_t0 should be -1.")
        end
        if (config.t0 * config.w0 != 2pi/2^config.num_energy_bits)
            push!(errors, "t0 * w0 != 2pi / N")
        end
        if (config.a == 0 && config.eta == -1.)
            push!(errors, "For picture TIME, a = 0 needs an eta > 0")
        end 
    elseif config.picture == TROTTER
        if config.num_energy_bits == -1.
            push!(errors, "For picture TROTTER, num_energy_bits must be > 0.")
        end
        if config.t0 == -1.
            push!(errors, "For picture TROTTER, t0 must be > 0.")
        end
        if config.w0 == -1.
            push!(errors, "For picture TROTTER, w0 must be > 0.")
        end
        if config.num_trotter_steps_per_t0 == -1
            push!(errors, "For picture TROTTER, num_trotter_steps_per_t0 must be > 0.")
        end
        if (norm(config.t0 * config.w0 - 2pi/2^config.num_energy_bits) > 1e-15)
            push!(errors, "t0 * w0 != 2pi / N")
        end
        if (config.a == 0 && config.eta == -1.)
            push!(errors, "For picture TROTTER, a = 0 needs an eta > 0")
        end 
    else
        push!(errors, "Unknown picture type.")
    end

    if config.b != 0.
        if config.a == 0.
            push!(errors, "When b > 0, then a must be positve too, a > 0.")
        end
    end

    if !isempty(errors)
        for err in errors
            println(err)
        end
        return false
    end

    return true
end