using LinearAlgebra
using Random
using Printf
# using QuantumOptics # Add it back if needed
using JLD

include("hamiltonian_tools.jl")

function trace_distance_h(rho::Union{Hermitian{<:Real}, Hermitian{<:Complex}}, 
    sigma::Union{Hermitian{<:Real}, Hermitian{<:Complex}})
    """Qutip apparently uses some sparse eigval solver, but let's go with the dense one for now."""
    return sum(abs.(eigvals(rho - sigma))) / 2
end

function trace_distance_nh(rho::Union{Matrix{<:Real}, Matrix{<:Complex}}, 
    sigma::Union{Matrix{<:Real}, Matrix{<:Complex}})
    return sum(svdvals(rho - sigma)) / 2
end

function trace_norm_h(rho::Union{Hermitian{<:Real}, Hermitian{<:Complex}})
    return sum(abs.(eigvals(rho)))
end

function trace_norm_nh(rho::Union{Matrix{<:Real}, Matrix{<:Complex}})
    return sum(svdvals(rho))
end

function fidelity(rho::Union{Hermitian{<:Real}, Hermitian{<:Complex}}, 
    sigma::Union{Hermitian{<:Real}, Hermitian{<:Complex}}; validate::Bool = true)

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

function is_density_matrix(rho::Union{Hermitian{<:Real}, Hermitian{<:Complex}})
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
    """Computes Gibbs state in computational basis!"""
    Z = sum(exp.(-beta * hamiltonian.eigvals))
    rho = sum([exp(-beta * hamiltonian.eigvals[i]) * hamiltonian.eigvecs[:, i] * hamiltonian.eigvecs[:, i]' 
                                                                                    for i in 1:length(hamiltonian.eigvals)])
    return Matrix{ComplexF64}(rho / Z)
end

function gibbs_state_in_eigen(hamiltonian::HamHam, beta::Float64)
    """Computes Gibbs state in eigenbasis"""

    eigvecs_in_eigen = I(size(hamiltonian.data)[1])
    Z = sum(exp.(-beta * hamiltonian.eigvals))
    rho = sum([exp(-beta * hamiltonian.eigvals[i]) * eigvecs_in_eigen[:, i] * eigvecs_in_eigen[:, i]' 
                                                                                    for i in 1:length(hamiltonian.eigvals)])
    return Matrix{ComplexF64}(rho / Z)
end

function random_density_matrix(num_qubits::Int)
    # Generate a random complex matrix
    A = randn(ComplexF64, 2^num_qubits, 2^num_qubits)
    
    # Compute A * A^†
    ρ = A * A'
    
    # Normalize the matrix to make the trace equal to 1
    ρ /= tr(ρ)
    
    return Hermitian(ρ)
end

function are_we_tp(liouv::Matrix{ComplexF64})
    initial_dm_OG = zeros(ComplexF64, size(hamiltonian.data))
    initial_dm_OG[2, 2] = 1.0
    initial_dm_OG[1, 2] = 1.0
    initial_dm_OG /= tr(initial_dm_OG)
    initial_vec = vec(initial_dm_OG)

    liouv_time_evolution(t) = exp(t * liouv)
    t = 1.0
    time_evolved_vec = liouv_time_evolution(t) * initial_vec
    id = I(2^num_qubits)
    vec_id = vec(id)
    @printf("Are we TP?: %s\n", vec_id' * time_evolved_vec)
end

# num_qubits = 4
# beta = 10.
# hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n4.jld")["ideal_ham"]

# hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
# gibbs = gibbs_state(hamiltonian, beta)
# gibbs_ineigen = hamiltonian.eigvecs * gibbs_state_in_eigen(hamiltonian, beta) * hamiltonian.eigvecs'
# norm(gibbs - gibbs_ineigen)

#* Compared with QuantumOptics implementations, my code is the same or faster and gives the same results
# Test
#! 12 qubits needs 8s for trdist (10^-15 precise), 32s for fidelity (10^-9 precise)
# n = 12
# # generate random n qubit density Matrix
# Random.seed!(666)
# rho = random_density_matrix(n)
# sigma = random_density_matrix(n)
# trdist = @time trace_distance_h(rho, sigma)
# trdist_with_nh = @time trace_distance_nh(Matrix(rho), Matrix(sigma))

# random_matrix_1 = rand(ComplexF64, 2^n, 2^n)
# random_matrix_2 = rand(ComplexF64, 2^n, 2^n)
# trdist_nh = @time trace_distance_nh(random_matrix_1, random_matrix_2)

# b = SpinBasis(1//2)^n
# trdist_qo_h = @time tracedistance_h(Operator(b, Matrix(rho)), Operator(b, Matrix(sigma)))
# trdist_qo_nh = @time tracedistance_nh(Operator(b, random_matrix_1), Operator(b, random_matrix_2))

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