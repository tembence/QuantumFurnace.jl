using LinearAlgebra
using SparseArrays
using Random
using Printf
using Plots
using QuadGK

include("hamiltonian.jl")
include("qi_tools.jl")
include("bohr_picture.jl")
include("trotter_tools.jl")

function oft(jump::JumpOp, energy::Float64, hamiltonian::HamHam, beta::Float64)
    """sigma_E = 1 / beta. Subnormalized, multiply by sqrt(beta / sqrt(2 * pi))"""
    return jump.in_eigenbasis .* exp.(-beta^2 * (energy .- hamiltonian.bohr_freqs).^2 / 4) 
end

#FIXME:
function trotter_oft(jump::JumpOp, energy::Float64, trotter::TrottTrott, time_labels::Vector{Float64}, beta::Float64)

    prefactors = @fastmath exp.(- time_labels.^2 / beta^2 .- 1im * energy * time_labels) # Gauss and Fourier factors

    mid_point = findlast(t -> t >= 0, time_labels) # Up to positive times
    eigvals_t0_diag = Diagonal(trotter.eigvals_t0)

    jump_oft = zeros(ComplexF64, size(jump.data))
    if jump.orthogonal # Orthogonal symmetry that halves time labels
        # t = 0.0
        trott_U_plus = I(length(trotter.eigvals_t0))
        jump_oft .+= @fastmath (prefactors[1] * trott_U_plus * jump.in_trotter_basis * trott_U_plus')

        # t > 0.0
        #TODO: DEBUG how many trotter steps are being applied
        for i in range(2, mid_point)
            trott_U_plus = eigvals_t0_diag^(i-1)

            temp_op = trott_U_plus * jump.in_trotter_basis * trott_U_plus'
            jump_oft .+=  @fastmath (prefactors[i] * temp_op)
            jump_oft .+=  @fastmath (conj(prefactors[i]) * transpose(temp_op))
        end
    else  #TODO: haven't done the non orthogonal yet
        for i in eachindex(time_labels)
            # trott_U = eigvals_t0_diag^num_t0_steps[i]
            # jump_oft .+= @fastmath (prefactors[i] * trott_U * jump.in_eigenbasis * adjoint(trott_U))

            trott_U_plus = eigvals_t0_diag^num_t0_steps[i]
            trott_U_minus = adjoint(trott_U_plus)

            # # less allocs this way:
            temp_op = (i <= mid_point) ? trott_U_plus * jump.in_trotter_basis * trott_U_minus : 
                                        trott_U_minus * jump.in_trotter_basis * trott_U_plus
            jump_oft .+= @fastmath (prefactors[i] * temp_op)
        end
    end

    # Return in Trotter basis
    return jump_oft
end

function time_oft(jump::JumpOp, energy::Float64, hamiltonian::HamHam, time_labels::Vector{Float64}, beta::Float64)
    """sigma_E = 1 / beta, subnormalized OFT: multiply by: t0 * sqrt(sqrt(2 / pi)/beta) / sqrt(2 * pi)"""

    prefactors = @fastmath exp.(- time_labels.^2 / beta^2 .- 1im * energy * time_labels) # Gauss and Fourier factors
    mid_point = findlast(t -> t >= 0, time_labels) # Up to positive times

    diag_exponentiate(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))

    jump_oft = zeros(ComplexF64, size(jump.data))
    if jump.orthogonal # Orthogonal symmetry that halves time labels
        # t = 0.0
        U_plus = diag_exponentiate(time_labels[1])
        jump_oft .+= @fastmath (prefactors[1] * U_plus * jump.in_eigenbasis * adjoint(U_plus))

        # t > 0.0
        for i in range(2, mid_point)
            U_plus = diag_exponentiate(time_labels[i])

            temp_op = U_plus * jump.in_eigenbasis * adjoint(U_plus)
            jump_oft .+=  @fastmath (prefactors[i] * temp_op)
            jump_oft .+=  @fastmath (conj(prefactors[i]) * transpose(temp_op))
        end
    else  # For non-orthogonal jumps
        for i in eachindex(time_labels)
            U = diag_exponentiate(time_labels[i])
            jump_oft .+= @fastmath (prefactors[i] * U * jump.in_eigenbasis * adjoint(U))

            # U_plus = diag_exponentiate(time_labels[i])
            # U_minus = adjoint(U_plus)

            # # less allocs this way:
            # temp_op = (i <= mid_point) ? U_plus * jump.in_eigenbasis * U_minus : 
            #                             U_minus * jump.in_eigenbasis * U_plus
            # jump_oft .+= @fastmath (prefactors[i] * temp_op)
        end
    end
    
    # jump_oft = zeros(ComplexF64, size(jump.data))
    # @showprogress "Explicit OFT..." for t in eachindex(time_labels)
    #     jump_oft .+= fourier_phase_factors[t] * normalized_time_gaussian_factors[t] * 
    #     diag_exponentiate(time_labels[t]) * jump.in_eigenbasis * diag_exponentiate(-time_labels[t])
    # end

    return jump_oft
end

#TODO: Check if this matches with time_oft
function time_oft_integrated(energy::Float64, jump::JumpOp, hamiltonian::HamHam, beta::Float64)

    diag_exponentiate(t) = Diagonal(exp.(1im * hamiltonian.eigvals * t))
    integrand(t) = exp(-t^2 / beta^2 - 1im * energy * t) * diag_exponentiate(t) * jump.in_eigenbasis * diag_exponentiate(-t)
    jump_oft = quadgk(integrand, -Inf, Inf)[1] / sqrt(2 * pi) * sqrt(sqrt(2 / pi) / beta)
    return jump_oft
end