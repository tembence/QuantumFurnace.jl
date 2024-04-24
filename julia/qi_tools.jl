using LinearAlgebra
using Random
using Printf

function trace_distance(rho::Hermitian{ComplexF64, Matrix{ComplexF64}}, 
    sigma::Hermitian{ComplexF64, Matrix{ComplexF64}}; validate::Bool = true)
    """Qutip apparently uses some sparse eigval solver, but let's go with the dense one for now."""
    if validate && (!is_density_matrix(rho) || !is_density_matrix(sigma))
        throw(ArgumentError("Input matrices are not density matrices"))
    end
    # return 0.5 * sum(svdvals(rho - sigma)[1])
    # return 0.5 * norm(rho - sigma, 1)
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

function gibbs_state(hamiltonian::HamHam, beta::Float64)
    Z = sum(exp.(-beta * hamiltonian.eigvals))
    rho = sum([exp(-beta * hamiltonian.eigvals[i]) * hamiltonian.eigvecs[:, i] * hamiltonian.eigvecs[:, i]' 
                                                                                    for i in 1:length(hamiltonian.eigvals)])
    return Matrix{ComplexF64}(rho / Z)
end

# Test
#! 12 qubits needs 8s for trdist (10^-15 precise), 32s for fidelity (10^-9 precise)
# n = 12
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

# println(norm(trace_distance(rho, sigma; validate=false) - 0.5*tr(sqrt((rho-sigma)' * (rho-sigma)))))
# println(norm(fidelity(rho, sigma; validate=false) - tr(sqrt(sqrt(rho)*sigma*sqrt(rho)))^2))