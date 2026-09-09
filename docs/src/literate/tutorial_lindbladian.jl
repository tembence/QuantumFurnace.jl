# # Constructing a Gibbs-sampling Lindbladian
# DLL is the physical-input default. A Workspace retains filtered source
# operators and mutable Krylov scratch; sequential simulations can reuse it.
using QuantumFurnace, LinearAlgebra
H = Hermitian(0.3X + 0.7Z)
ws = Workspace(H; beta_phys=0.8)
dll = simulate_gibbs(ws; times=[0.0, 20.0], epsilon=1e-3, diagnostics=:strict)
@assert dll.provenance.coherent
@assert dll.trajectory.all_converged
@assert last(dll.trajectory.distances) < dll.convergence.epsilon
println("DLL evolution completed: ", dll.trajectory.all_converged)
println("Coherent correction included: ", dll.provenance.coherent)
println("Final distance to Gibbs: ", round(last(dll.trajectory.distances); sigdigits=3))
println("Reached distance < 0.001: ", last(dll.trajectory.distances) < dll.convergence.epsilon)

# The first two lines confirm that the numerical evolution finished and the
# construction included its coherent correction. The distance reports remaining
# disagreement with the thermal target; zero would mean an exact match.

# The full sampler contains -im[B,rho], with the canonical B correction. The
# interface does not silently add -im[H,rho]. Supplied sources keep their amplitudes;
# the default onsite Pauli family has amplitudes 1/sqrt(3n).
# CKG KMS is selected through the same physical-input interface:
ckg = simulate_gibbs(H; beta_phys=0.8, construction=KMS(),
    times=[0.0, 20.0], epsilon=1e-3, diagnostics=:strict)
@assert ckg.trajectory.all_converged
@assert last(ckg.trajectory.distances) < ckg.convergence.epsilon
println("CKG evolution completed: ", ckg.trajectory.all_converged)
println("Final distance to Gibbs: ", round(last(ckg.trajectory.distances); sigdigits=3))
println("Reached distance < 0.001: ", last(ckg.trajectory.distances) < ckg.convergence.epsilon)

# For an independent small-system reference, reuse the low-level constructor.
# The dense superoperator below is suitable only for this one-qubit check.
p = prepare_gibbs_inputs(H; beta_phys=0.8)
L = construct_lindbladian(p.jumps, p.config, p.hamiltonian)
U = p.hamiltonian.eigvecs
rho0 = fill(ComplexF64(0.5), 2, 2)
reference = U * reshape(exp(last(dll.trajectory.t) * L) * vec(U' * rho0 * U), 2, 2) * U'
@assert isapprox(reference, dll.trajectory.rho_final; atol=1e-10)
println("Difference from direct matrix exponential: ",
    round(norm(reference - dll.trajectory.rho_final); sigdigits=3))
println("Agreement within absolute tolerance 1e-10: ",
    isapprox(reference, dll.trajectory.rho_final; atol=1e-10))

# This compares two ways of evolving the same initial state for the same time.
# A small matrix difference checks the numerical evolution on this example;
# it does not say that the evolved state has already reached Gibbs.

# `Config` keeps algorithm-frame beta. DLL Energy/Trotter/GQSP and custom
# CKG Trotter are unavailable. See the capability contract before changing a
# domain; a finite grid requires its own accuracy checks.
