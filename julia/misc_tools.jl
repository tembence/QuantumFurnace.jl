using LinearAlgebra
using QuadGK

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

function print_press(; num_qubits=nothing, num_energy_bits=nothing, beta=nothing, a=nothing, b=nothing, eta=nothing, t0=nothing, w0=nothing)
    params = [
        ("num_qubits", num_qubits),
        ("num_energy_bits", num_energy_bits),
        ("beta", beta),
        ("a", a),
        ("b", b),
        ("eta", eta),
        ("t0", t0),
        ("w0", w0)
    ]
    provided = filter(p -> p[2] !== nothing, params)
    if isempty(provided)
        return
    end

    println("--- The Press ---")
    for (name, value) in provided
        println("$name: $value")
    end
end