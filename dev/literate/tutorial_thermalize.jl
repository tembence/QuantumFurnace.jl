# # Simulating a Gibbs sampler
# This is full-density-matrix Lindbladian evolution, with continuous generator
# time. It is not a sampled quantum-jump trajectory or a channel step count.
using QuantumFurnace

H = pauli_hamiltonian(2, [
    -1.0 => (1 => :Z, 2 => :Z),
    -0.7 => (1 => :X,),
    -0.7 => (2 => :X,),
])
result = simulate_gibbs(H; beta_phys=0.8,
    times=range(0, 4; length=21), diagnostics=:standard)
result.trajectory.distances
result.diagnostics
result.spectrum.reliability
@assert result.trajectory.all_converged
@assert result.trajectory.trace_norms ≈ 2result.trajectory.distances

# Input/output states use the computational basis by default. The initial state
# is |+><+| tensor power. Saved intermediate states are off by default; request
# save_states=true with a suitable max_saved_bytes cap if needed.
# A finite-time failure to reach Gibbs does not establish nonergodicity.
result.convergence

# The independent spectral policy uses its own operator starts. A small Ritz
# residual does not rule out missed modes; inspect reliability and coverage.
result.spectrum.coverage

# Save the combined result as versioned data, then continue from its final state.
# A fresh workspace is rebuilt; no mutable backend plans are loaded. The new
# segment starts at zero additional time, and records the preceding time origin.
continuation = mktempdir() do directory
    path = save_result(result, joinpath(directory, "gibbs.bson"))
    restored = load_result(path)
    @assert restored.trajectory.rho_final ≈ result.trajectory.rho_final
    simulate_gibbs(restored; times=[0.0, 0.1], diagnostics=:quick)
end
@assert continuation.provenance.resume_time_origin ≈ last(result.trajectory.t)

# ## Finite channel steps
# Select the existing run_thermalize backend explicitly. This applies the
# implemented weak-measurement channel five times, including its coherent step.
# CKG is selected explicitly because DLL finite channels are not implemented.
channel = simulate_gibbs(H; sim=Thermalize(), construction=KMS(),
    beta_phys=0.8, delta=0.01, steps=5, save_states=true,
    channel_options=(save_every=2,))
@assert channel.trajectory.channel_steps == [0, 2, 4, 5]
@assert channel.trajectory.states[end] ≈ channel.trajectory.rho_final
@assert channel.spectrum.reliability == :not_run
channel.provenance.channel_representation

# Lindblad evolution computes exp(t*L)rho0: method=:krylov uses successive
# Arnoldi exponential actions, while method=:predictor reuses captured modes.
# Finite channel evolution instead applies E_delta repeatedly. These targets
# need not agree at finite delta. Times for channel runs must lie on the delta
# grid; integer steps avoid rounding ambiguity. One sweep applies every source.
# The clock includes the backend rate normalisation recorded in provenance.
# Hamiltonian rescaling does not independently rescale delta.

# channel_options=(jump_selection=:random, seed=42) selects one source per
# outer step with the backend's default probability compensation. Each returned
# density matrix is conditioned on those source choices; it is not the average
# over source histories or a sampled measurement-outcome trajectory.
# Channel diagnostics check sampled states; spectrum, stationarity and KMS
# checks are explicitly not_run. Crossing epsilon is only an observed crossing.
# The finite channel can have a bias from Gibbs even if its ideal generator
# fixes Gibbs. Source rates must obey the channel's operator-normalisation bound.

# Channel results support save_result/load_result as evidence. Automatic
# Workspace(result) and simulate_gibbs(result; times=...) reconstruction reject.
# Continue using the same original inputs and the returned final density matrix:
next_channel = simulate_gibbs(H; sim=Thermalize(), construction=KMS(),
    beta_phys=0.8, delta=0.01, steps=2, rho0=channel.trajectory.rho_final)
@assert next_channel.trajectory.completed_steps == 2
