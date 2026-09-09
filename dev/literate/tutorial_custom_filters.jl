# # Custom DLL filters and CKG rates
# These examples run a one-qubit system with three different prescriptions.
# Each output is computed during the documentation build. Distance to Gibbs
# is zero at the target thermal state; smaller values mean closer agreement.
using QuantumFurnace, LinearAlgebra
beta_phys = 0.8
H = Hermitian(0.3X + 0.7Z);

# Supply q on nonnegative physical frequencies. The constructor reflects it as
# q(-nu)=conj(q(nu)) and adds exp(-beta_phys*nu/4) once. A complex phase is allowed.
# A width of 2.0 covers this system's energy gap without strongly suppressing
# its transitions; 40 units of generator time let the state reach our target.
f = KMSFilter(beta_phys; q_positive=x -> exp(-(x/2.0)^2)*cis(0.4x),
    name=:phase_gaussian, version="1", parameters=(width=2.0, shift=0.4))
custom = simulate_gibbs(H; beta_phys, filter=f,
    times=range(0, 40; length=21), epsilon=1e-3, diagnostics=:strict)
@assert custom.trajectory.all_converged
@assert last(custom.trajectory.distances) < custom.convergence.epsilon
println("Custom filter: evolution completed = ", custom.trajectory.all_converged)
println("Distance to Gibbs: ", round(first(custom.trajectory.distances); sigdigits=3),
    " -> ", round(last(custom.trajectory.distances); sigdigits=3))
println("Reached distance < 0.001: ", last(custom.trajectory.distances) < custom.convergence.epsilon)

# This confirms both that the supplied callback produced a usable simulation
# and that its final state is within the requested distance of the Gibbs target.

# Alternatively prescribe the downward rate at positive energy release x.
# RateFilter supplies its Boltzmann-reflected upward rate and principal-root
# amplitude. The callback returns a rate, not an amplitude.
rate_filter = RateFilter(beta_phys; downward_rate=x -> 2exp(-x*x), name=:downward)
rate_result = simulate_gibbs(H; beta_phys, filter=rate_filter,
    times=range(0, 40; length=21), epsilon=1e-3, diagnostics=:strict)
@assert rate_result.trajectory.all_converged
@assert last(rate_result.trajectory.distances) < rate_result.convergence.epsilon
println("Downward rate: evolution completed = ", rate_result.trajectory.all_converged)
println("Distance to Gibbs: ", round(first(rate_result.trajectory.distances); sigdigits=3),
    " -> ", round(last(rate_result.trajectory.distances); sigdigits=3))
println("Reached distance < 0.001: ", last(rate_result.trajectory.distances) < rate_result.convergence.epsilon)

# CKG Gaussian mixtures require a matching normalised Gaussian OFT. Component
# centres and widths share physical beta; weights need not sum to one.
mixture = GaussianMixtureTransition(beta_phys; sigma=0.35,
    centers=(0.2, 0.6), weights=(0.3, 0.8))
ckg = simulate_gibbs(H; beta_phys, construction=KMS(), transition_weight=mixture,
    times=range(0, 40; length=21), epsilon=1e-3, diagnostics=:strict)
@assert ckg.trajectory.all_converged
@assert last(ckg.trajectory.distances) < ckg.convergence.epsilon
println("CKG mixture: evolution completed = ", ckg.trajectory.all_converged)
println("Distance to Gibbs: ", round(first(ckg.trajectory.distances); sigdigits=3),
    " -> ", round(last(ckg.trajectory.distances); sigdigits=3))
println("Reached distance < 0.001: ", last(ckg.trajectory.distances) < ckg.convergence.epsilon)

# All three examples use the same Hamiltonian and temperature. Their rates and
# normalisations differ, so these outputs are not a matched-cost
# comparison of sampler efficiency.

# These examples use exact Bohr-frequency filtering. For custom Time evolution
# and its separate Fourier/coherent controls, see the
# [interface contract](../api_contract.md#Independent-coherent-controls-and-custom-DLL-Time).

# Names and parameter tuples alone do not define a closure. Portable
# reconstruction needs the matching registered definition or explicit resupply;
# consult the persistence section of the interface contract.
