# # Creating a Hamiltonian
# Each Pauli term occurs once at its explicit sites. Site 1 is the leftmost
# tensor factor; nonadjacent interactions and site-dependent fields are allowed.
using QuantumFurnace, LinearAlgebra

H = pauli_hamiltonian(3, [
    -1.0 => (1 => :Z, 3 => :Z),
    -0.4 => (2 => :X,),
    0.2 => (),  # scalar identity shift
])
@assert H ≈ -kron(Z, I(2), Z) - 0.4kron(I(2), X, I(2)) + 0.2I
println("Number of qubits: 3")
println("Hamiltonian size: ", size(H))
println("Hermitian: ", ishermitian(H))

# Three qubits have eight basis states, so their Hamiltonian is an 8-by-8
# matrix. Hermitian means it represents an observable with real energies.

# Construction is sparse and does not diagonalise H. Preparing `HamHam` does
# perform dense spectral work and caches the Gibbs state in its eigenbasis.
ham = HamHam(H; beta_phys=0.8)
@assert beta_alg(ham, 0.8) ≈ 0.8ham.rescaling_factor
@assert tr(ham.gibbs) ≈ 1
println("Physical inverse temperature: 0.8")
println("Total Gibbs probability: ", round(real(tr(ham.gibbs)); sigdigits=3))

# A total probability of one checks that the prepared target state is
# normalised. This constructs the Gibbs target; no evolution has run yet.

# A finite Hermitian qubit matrix is also accepted. Trotter synthesis requires
# a separate local decomposition. Temperature uses energy units, k_B=1.
matrix_H = Hermitian(0.3X + 0.4Y + 0.7Z)
preflight = simulate_gibbs(matrix_H; temperature=1.25, times=[0.0, 0.1], dry_run=true)
@assert preflight.spectral_preparation == :not_run
@assert preflight.beta_phys ≈ 0.8
println("Estimated construction memory (bytes): ", preflight.estimated_construction_bytes)
println("Spectral preparation: ", preflight.spectral_preparation)

# The dry run reports a memory estimate before diagonalisation. `not_run` is
# expected here: we requested a resource check, not a simulation.
