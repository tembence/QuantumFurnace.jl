# # Interpreting diagnostics
using QuantumFurnace, LinearAlgebra

# Dephasing preserves every diagonal state. Gibbs is stationary, yet the
# stationary state is nonunique. A complete tiny-system numerical kernel check
# distinguishes this from a trajectory that merely ran for too short a time.
nonergodic = simulate_gibbs(Hermitian(Z); beta_phys=0.8, jumps=[Z],
    times=[0.0, 0.1], diagnostics=:strict)
@assert nonergodic.diagnostics.uniqueness == :nonunique
@assert nonergodic.diagnostics.checks.stationarity.status == :pass
nonergodic.convergence

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
checks.checks.stationarity

# A work cap returns usable partial evidence and the last valid state. It
# does not relax tolerances or relabel skipped diagnostics as successful.
partial = simulate_gibbs(Hermitian(0.3X + 0.7Z); beta_phys=0.8,
    times=[0.0, 1.0], max_matvecs=0, diagnostics=:quick,
    gap_options=(max_matvecs=0,))
@assert partial.convergence.status == :inconclusive
@assert !partial.trajectory.all_converged
partial.spectrum.reliability

# Trace distance is half the full trace norm. Finite trajectories constrain
# their chosen initial states; multistart agreement is not all-mode coverage or
# a certified worst-case mixing bound. Gibbs underflow can invalidate inverse
# Gibbs-based checks even when direct-generator diagnostics remain available.
