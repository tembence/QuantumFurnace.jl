using LinearAlgebra

include("coherent.jl")
include("misc_tools.jl")

function compute_errors(hamiltonian::HamHam, config::LiouvConfig; trotter::Union{TrottTrott, Nothing} = nothing)

    energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
    truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta,
    config.a, config.b, config.with_linear_combination)
    transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)

    enegy_error = compute_energy_quadrature_error(transition, truncated_energy_labels)

    time_labels = energy_labels .* (config.t0 / config.w0)
    if trotter === nothing
        oft_error = compute_time_oft_quadrature_error()
        B_error = compute_time_B_quadrature_error()
    else
        oft_error = compute_trotter_oft_quadrature_error()
        B_error =  compute_trotter_B_quadrature_error()
    end
    return (energy_error = energy_error, oft_error = oft_error, B_error = B_error)
end

function compute_quadrature_error()  #TODO: We probably don't need a separate function like all the ones below, just a scalar and matrix version of this
end

function compute_energy_quadrature_error(transition::Function, energy_labels::Vector{Float64})
end

function compute_time_oft_quadrature_error()
end

function compute_trotter_oft_quadrature_error()
end

function compute_time_B_quadrature_error()
end

function compute_trotter_B_quadrature_error()
end