using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using QuantumOptics
using BenchmarkTools
using Plots


include("hamiltonian.jl")
include("ofts.jl")
include("trotter.jl")
include("qi_tools.jl")
include("spectral_analysis.jl")
include("coherent.jl")

num_qubits = 10
b = SpinBasis(1//2)^num_qubits

hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n10.jld")["ideal_ham"]
display(hamiltonian.eigvals)

# Boltzmann probabilities
boltzmann_vals_10 = exp.( - 10. * hamiltonian.eigvals) ./ sum(exp.( - 10. * hamiltonian.eigvals))
boltzmann_vals_5 = exp.( - 5. * hamiltonian.eigvals) ./ sum(exp.( - 5. * hamiltonian.eigvals))
boltzmann_vals_1 = exp.( - 1. * hamiltonian.eigvals) ./ sum(exp.( - 1. * hamiltonian.eigvals))
boltzmann_vals_01 = exp.( - 0.1 * hamiltonian.eigvals) ./ sum(exp.( - 0.1 * hamiltonian.eigvals))
boltzmann_vals_inf = ones(2^num_qubits) / 2^num_qubits

# Plot the boltzmann vals over the eigenvalues 
plot(hamiltonian.eigvals, boltzmann_vals_10, seriestype = :scatter, label = "beta = 10", xlabel = "Eigenvalues", ylabel = "Boltzmann vals")
plot!(hamiltonian.eigvals, boltzmann_vals_5, seriestype = :scatter, label = "beta = 5")
plot!(hamiltonian.eigvals, boltzmann_vals_1, seriestype = :scatter, label = "beta = 1")
plot!(hamiltonian.eigvals, boltzmann_vals_01, seriestype = :scatter, label = "beta = 0.1")
plot!(hamiltonian.eigvals, boltzmann_vals_inf, seriestype = :scatter, label = "beta = inf")


# Mean energies
mean_energy_10 = sum(hamiltonian.eigvals .* boltzmann_vals_10)
mean_energy_1 = sum(hamiltonian.eigvals .* boltzmann_vals_1)
mean_energy_01 = sum(hamiltonian.eigvals .* boltzmann_vals_01)
mean_energs_inf = sum(hamiltonian.eigvals .* boltzmann_vals_inf)

println("Mean energies: ")
println("beta = 10: E = ", mean_energy_10)
println("beta = 1: E = ", mean_energy_1)
println("beta = 0.1: E = ", mean_energy_01)
println("beta = inf: E = ", mean_energs_inf)

max_mixed_dm = Matrix(I, 2^num_qubits, 2^num_qubits) / 2^num_qubits
mean_energy_max_mixed = real(tr(max_mixed_dm * hamiltonian.data))

beta = 10.
# Gibbs computational
gibbs = gibbs_state(hamiltonian, beta)
gibbs_vec = vec(gibbs)
# Gibbs eigen
gibbs_eigen = hamiltonian.eigvecs' * gibbs * hamiltonian.eigvecs
gibbs_eigen_vec = vec(gibbs_eigen)

trdist = tracedistance_nh(Operator(b, max_mixed_dm), Operator(b, gibbs))
println("Tracedistance between max mixed and gibbs state: ", trdist)

# random_dm = random_density_matrix(num_qubits)
# trdist_rand = tracedistance_nh(Operator(b, random_dm), Operator(b, gibbs))
# # Vectorized
# gibbs_vec = vec(gibbs)
# max_mixed_vec = vec(max_mixed_dm)
# overlap = abs(dot(gibbs_vec, max_mixed_vec))
