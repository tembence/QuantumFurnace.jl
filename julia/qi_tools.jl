using LinearAlgebra
using Random
using Printf
using QuantumOptics

include("hamiltonian_tools.jl")

function trace_distance(rho::Hermitian{ComplexF64, Matrix{ComplexF64}}, 
    sigma::Hermitian{ComplexF64, Matrix{ComplexF64}}; validate::Bool = true)
    """Qutip apparently uses some sparse eigval solver, but let's go with the dense one for now."""
    if validate && (!is_density_matrix(rho) || !is_density_matrix(sigma))
        throw(ArgumentError("Input matrices are not density matrices"))
    end
    eig_vals = eigvals(rho - sigma)
    return sum(abs.(eig_vals)) / 2
end

function fidelity(rho::Hermitian{ComplexF64, Matrix{ComplexF64}}, 
    sigma::Hermitian{ComplexF64, Matrix{ComplexF64}}; validate::Bool = true)

    if validate && (!is_density_matrix(rho) || !is_density_matrix(sigma))
        throw(ArgumentError("Input matrices are not density matrices"))
    end

    eig_vals = real(eigvals(rho * sigma))
    return real(sum(sqrt.(eig_vals[eig_vals.>0])))^2
end

function frobenius_norm(A::Matrix{ComplexF64})
    eig_vals = eigvals(A)
    return sqrt(sum(abs.(eig_vals).^2))
end

function is_density_matrix(rho::Matrix{ComplexF64})
    if !isapprox(rho, rho')
        throw(ArgumentError("Input matrix is not Hermitian"))
    end

    eig_vals = real(round.(eigvals(rho), digits=15))
    # check if eigenvalues are approximately nonnegative
    if any(eig_vals .< 0)
        throw(ArgumentError("Input matrix has negative eigenvalues"))
    end

    if !isapprox(sum(eig_vals), 1.0)
        throw(ArgumentError("Input matrix has got trace different from 1"))
    end

    return true
end

function is_density_matrix(rho::Hermitian{ComplexF64, Matrix{ComplexF64}})
    if !isapprox(rho, rho')
        throw(ArgumentError("Input matrix is not Hermitian"))
    end

    eig_vals = real(round.(eigvals(rho), digits=13))
    # check if eigenvalues are approximately nonnegative
    if any(eig_vals .< 0)
        throw(ArgumentError("Input matrix has negative eigenvalues"))
    end

    if !isapprox(sum(eig_vals), 1.0)
        throw(ArgumentError("Input matrix has got trace different from 1"))
    end

    return true
end

function gibbs_state(hamiltonian::HamHam, beta::Float64)
    """Computes Gibbs state in the eigenbasis of the Hamiltonian"""
    Z = sum(exp.(-beta * hamiltonian.eigvals))
    rho = sum([exp(-beta * hamiltonian.eigvals[i]) * hamiltonian.eigvecs[:, i] * hamiltonian.eigvecs[:, i]' 
                                                                                    for i in 1:length(hamiltonian.eigvals)])
    return Matrix{ComplexF64}(rho / Z)
end

#* Compared with QuantumOptics implementations, my code is the same or faster and gives the same results
# Test
#! 12 qubits needs 8s for trdist (10^-15 precise), 32s for fidelity (10^-9 precise)
# n = 10
# # generate random n qubit density Matrix
# Random.seed!(666)
# rho = rand(2^n, 2^n) + 1im * rand(2^n, 2^n)
# rho = rho * rho'
# rho = rho / tr(rho)
# rho = Hermitian(rho)

# sigma = rand(2^n, 2^n) + 1im * rand(2^n, 2^n)
# sigma = sigma * sigma'
# sigma = sigma / tr(sigma)
# sigma = Hermitian(sigma)

# @time println(trace_distance(rho, sigma, validate=false))
# @time println(fidelity(rho, sigma, validate=false))

# # println(norm(trace_distance(rho, sigma; validate=false) - 0.5*tr(sqrt((rho-sigma)' * (rho-sigma)))))
# # println(norm(fidelity(rho, sigma; validate=false) - tr(sqrt(sqrt(rho)*sigma*sqrt(rho)))^2))

# # QuantumOptics
# b = GenericBasis(2^n)
# rho = DenseOperator(Operator(b, b, rho))
# sigma = DenseOperator(Operator(b, b, sigma))
# #! Okay first left basis, right basis (the same here) and then data.
# @time println(tracedistance_h(rho, sigma))
# @time println(fidelity(rho, sigma)^2)