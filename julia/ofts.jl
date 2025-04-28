using LinearAlgebra
using SparseArrays
using Random
using Printf
using Plots
using QuadGK

include("hamiltonian.jl")
include("qi_tools.jl")
include("bohr_picture.jl")
include("timelike_tools.jl")

function oft(jump::JumpOp, energy::Float64, hamiltonian::HamHam, beta::Float64)
    """sigma_E = 1 / beta. Subnormalized, multiply by sqrt(beta / sqrt(2 * pi))"""
    return jump.in_eigenbasis .* exp.(-beta^2 * (energy .- hamiltonian.bohr_freqs).^2 / 4) 
end

function trotter_oft(jump::JumpOp, energy::Float64, trotter::TrottTrott, time_labels::Vector{Float64}, beta::Float64)

    prefactors = @fastmath exp.(- time_labels.^2 / beta^2 .- 1im * energy * time_labels) # Gauss and Fourier factors
    mid_point = findlast(t -> t >= 0, time_labels) # Up to positive times
    trotter_time_evolution(n::Int64) = Diagonal(trotter.eigvals_t0 .^ n)  # n - number of t0 time chunks

    jump_oft = zeros(ComplexF64, size(jump.data))
    if jump.orthogonal # Orthogonal symmetry that halves time labels
        trott_U_plus = trotter_time_evolution(0)  # t = 0.0
        jump_oft .+= @fastmath (prefactors[1] * trott_U_plus * jump.in_trotter_basis * trott_U_plus')

        for i in range(2, mid_point)  # t > 0.0 ∼ t < 0.0
            trott_U_plus = trotter_time_evolution(i - 1)
            temp_op = trott_U_plus * jump.in_trotter_basis * trott_U_plus'
            jump_oft .+=  @fastmath (prefactors[i] * temp_op)
            jump_oft .+=  @fastmath (conj(prefactors[i]) * transpose(temp_op))
        end
    else
        for i in range(1, length(time_labels))  # 0 and positive times
            if i <= mid_point
                num_t0_steps = i - 1
            else
                num_t0_steps = i - 2 * mid_point # Pain
            end
            trott_U_plus = trotter_time_evolution(num_t0_steps)
            jump_oft .+=  @fastmath (prefactors[i] * trott_U_plus * jump.in_trotter_basis * trott_U_plus')
        end
    end
    # Return in Trotter basis
    return jump_oft
end

function time_oft(jump::JumpOp, energy::Float64, hamiltonian::HamHam, time_labels::Vector{Float64}, beta::Float64)
    """sigma_E = 1 / beta, subnormalized OFT: multiply by: t0 * sqrt(sqrt(2 / pi)/beta) / sqrt(2 * pi)"""

    prefactors = @fastmath exp.(- time_labels.^2 / beta^2 .- 1im * energy * time_labels) # Gauss and Fourier factors
    mid_point = findlast(t -> t >= 0, time_labels) # Up to positive times

    time_evolve(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))

    jump_oft = zeros(ComplexF64, size(jump.data))
    if jump.orthogonal # Orthogonal symmetry that halves time labels
        U_plus = time_evolve(time_labels[1])  # t = 0.0
        jump_oft .+= @fastmath (prefactors[1] * U_plus * jump.in_eigenbasis * adjoint(U_plus))

        for i in range(2, mid_point)  # t > 0.0
            U_plus = time_evolve(time_labels[i])

            temp_op = U_plus * jump.in_eigenbasis * adjoint(U_plus)
            jump_oft .+=  @fastmath (prefactors[i] * temp_op)
            jump_oft .+=  @fastmath (conj(prefactors[i]) * transpose(temp_op))
        end
    else  # For non-orthogonal jumps
        for i in eachindex(time_labels)
            U = time_evolve(time_labels[i])
            jump_oft .+= @fastmath (prefactors[i] * U * jump.in_eigenbasis * adjoint(U))
        end
    end
    return jump_oft
end

function time_oft_integrated(energy::Float64, jump::JumpOp, hamiltonian::HamHam, beta::Float64)

    diag_exponentiate(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))
    integrand(t) = exp(-t^2 / beta^2 - 1im * energy * t) * diag_exponentiate(t) * jump.in_eigenbasis * diag_exponentiate(-t)
    jump_oft = quadgk(integrand, -Inf, Inf)[1] / sqrt(2 * pi) * sqrt(sqrt(2 / pi) / beta)
    return jump_oft
end

#* Trotter OFT check
# energy_labels = create_energy_labels(num_energy_bits, w0)
# truncated_energy_labels = truncate_energy_labels(energy_labels, beta,
# a, b, with_linear_combination)
# time_labels = energy_labels .* (t0 / w0)
# w = -0.12
# oft_time_labels = truncate_time_labels_for_oft(time_labels, beta)
# jump = jumps[6]
# oft_trott = trotter_oft(jump, w, trotter, oft_time_labels, beta) * t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
# oft_trott = trotter.trafo_from_eigen_to_trotter' * oft_trott * trotter.trafo_from_eigen_to_trotter
# oft_w = oft(jump, w, hamiltonian, beta) * sqrt(beta / sqrt(2 * pi))
# norm(oft_w - oft_trott)