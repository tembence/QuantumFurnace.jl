# # Interpreting diagnostics
using QuantumFurnace, LinearAlgebra

# Dephasing preserves every diagonal state. Gibbs is stationary, yet the
# stationary state is nonunique. A complete tiny-system numerical kernel check
# distinguishes this from a trajectory that merely ran for too short a time.
nonergodic = simulate_gibbs(Hermitian(Z); beta_phys=0.8, jumps=[Z],
    times=[0.0, 0.1], diagnostics=:strict)
@assert nonergodic.diagnostics.uniqueness == :nonunique
@assert nonergodic.diagnostics.checks.stationarity.status == :pass
println("Gibbs is stationary: ", nonergodic.diagnostics.checks.stationarity.status)
println("Uniqueness of stationary state: ", nonergodic.diagnostics.uniqueness)
println("Final distance to Gibbs: ", round(last(nonergodic.trajectory.distances); sigdigits=3))

# `pass` means the computed generator leaves Gibbs unchanged within the check's
# tolerance. `nonunique` means other stationary states exist too: this sampler
# cannot drive every initial state to the same thermal target.

# A deliberately coarse Time grid preserves trace while missing KMS balance.
# Decrease time_step to refine spacing and increase num_energy_bits to extend
# the window, checking their effects separately. Custom prepared Time filters
# additionally expose independent coherent frequency/time controls and policy.
p = prepare_gibbs_inputs(Hermitian(0.4X + 0.7Z); beta_phys=0.8,
    domain=TimeDomain(), time_step=0.8, num_energy_bits=3)
checks = workspace_diagnostics(Workspace(p.config, p.hamiltonian, p.jumps),
    p.config, p.hamiltonian)
@assert checks.checks.trace_preservation.status == :pass
@assert checks.checks.kms.status == :fail
println("Trace preservation: ", checks.checks.trace_preservation.status)
println("KMS balance: ", checks.checks.kms.status)
println("Gibbs stationarity: ", checks.checks.stationarity.status)

# These are deliberately mixed results. Preserving total probability (`pass`)
# does not ensure the thermal balance condition (`fail`). The coarse grid needs
# refinement before this calculation can be trusted as a Gibbs sampler.
# The integration warning above comes from this deliberately coarse setup.

# A work cap returns usable partial evidence and the last valid state. It
# does not relax tolerances or relabel skipped diagnostics as successful.
partial = simulate_gibbs(Hermitian(0.3X + 0.7Z); beta_phys=0.8,
    times=[0.0, 1.0], max_matvecs=0, diagnostics=:quick,
    gap_options=(max_matvecs=0,))
@assert partial.convergence.status == :inconclusive
@assert !partial.trajectory.all_converged
println("Numerical evolution completed: ", partial.trajectory.all_converged)
println("Convergence assessment: ", partial.convergence.status)
println("Spectral reliability: ", partial.spectrum.reliability)

# Here `false` and `inconclusive` are the expected outputs: the work budget was
# deliberately set to zero. They describe an unfinished calculation, not a
# physical failure to thermalise. Missing evidence must not appear as success.

# Trace distance is half the full trace norm. Finite trajectories constrain
# their chosen initial states; multistart agreement is not all-mode coverage or
# a certified worst-case mixing bound. Gibbs underflow can invalidate inverse
# Gibbs-based checks even when direct-generator diagnostics remain available.
