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
