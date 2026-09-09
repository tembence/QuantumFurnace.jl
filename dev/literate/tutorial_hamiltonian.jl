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

# Construction is sparse and does not diagonalise H. Preparing `HamHam` does
# perform dense spectral work and caches the Gibbs state in its eigenbasis.
ham = HamHam(H; beta_phys=0.8)
@assert beta_alg(ham, 0.8) ≈ 0.8ham.rescaling_factor
@assert tr(ham.gibbs) ≈ 1

# A finite Hermitian qubit matrix is also accepted. It retains no invented local
# decomposition for Trotter synthesis. Temperature uses energy units, k_B=1.
matrix_H = Hermitian(0.3X + 0.4Y + 0.7Z)
preflight = simulate_gibbs(matrix_H; temperature=1.25, times=[0.0, 0.1], dry_run=true)
@assert preflight.spectral_preparation == :not_run
@assert preflight.beta_phys ≈ 0.8
preflight.estimated_construction_bytes
