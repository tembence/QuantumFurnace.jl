# # Constructing a Gibbs-sampling Lindbladian
# DLL is the physical-input default. A Workspace retains filtered source
# operators and mutable Krylov scratch; sequential simulations can reuse it.
using QuantumFurnace, LinearAlgebra
H = Hermitian(0.3X + 0.7Z)
ws = Workspace(H; beta_phys=0.8)
dll = simulate_gibbs(ws; times=[0.0, 0.1], diagnostics=:quick)
@assert dll.provenance.coherent

# The full sampler contains -im[B,rho], with the canonical B correction. The
# facade does not silently add -im[H,rho]. Supplied sources keep their amplitudes;
# the default onsite Pauli family has amplitudes 1/sqrt(3n).
# CKG KMS is selected through the same physical-input interface:
ckg = simulate_gibbs(H; beta_phys=0.8, construction=KMS(),
    times=[0.0, 0.1], diagnostics=:quick)
@assert ckg.trajectory.all_converged

# For an independent small-system reference, reuse the low-level constructor.
# The dense superoperator below is suitable only for this one-qubit check.
p = prepare_gibbs_inputs(H; beta_phys=0.8)
L = construct_lindbladian(p.jumps, p.config, p.hamiltonian)
U = p.hamiltonian.eigvecs
rho0 = fill(ComplexF64(0.5), 2, 2)
reference = U * reshape(exp(0.1L) * vec(U' * rho0 * U), 2, 2) * U'
@assert isapprox(reference, dll.trajectory.rho_final; atol=1e-10)

# Legacy Config keeps algorithm-frame beta. DLL Energy/Trotter/GQSP and custom
# CKG Trotter are unavailable. See the capability contract before changing a
# domain; a finite grid requires its own accuracy checks.
