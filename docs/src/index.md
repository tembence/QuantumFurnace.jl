# QuantumFurnace.jl

Construct and simulate Gibbs samplers based on KMS detailed-balance
Lindbladians. The physical-input interface defaults to Ding–Li–Lin (DLL)
with the coherent correction enabled.

```@example home
using QuantumFurnace

H = pauli_hamiltonian(2, [
    -1.0 => (1 => :Z, 2 => :Z),
    -0.7 => (1 => :X,),
    -0.7 => (2 => :X,),
])
result = simulate_gibbs(H; beta_phys=0.8,
    times=range(0, 20; length=21), epsilon=1e-3, diagnostics=:strict)
@assert result.trajectory.all_converged # hide
@assert last(result.trajectory.distances) < result.convergence.epsilon # hide
println("Numerical evolution completed: ", result.trajectory.all_converged)
println("Initial distance to Gibbs: ", round(first(result.trajectory.distances); sigdigits=3))
println("Final distance to Gibbs:   ", round(last(result.trajectory.distances); sigdigits=3))
println("Reached the requested distance < 0.001: ",
    last(result.trajectory.distances) < result.convergence.epsilon)
```

The output above is calculated when this site is built. **Distance to Gibbs**
measures how distinguishable the simulated state is from the target thermal
state: zero means they match, and one is the largest possible distance for
normalised states. Here the final distance is below the requested `0.001`, so
this run reached its target accuracy. “Numerical evolution completed” separately
confirms that the solver finished the requested time interval within its tolerances.

The Hamiltonian entries use physical energy units, `beta_phys` their inverse,
and times the generator clock set by the jump amplitudes. `distances` stores
half the trace norm. A trajectory crossing is specific to its initial state;
converged eigenpairs alone do not certify a complete gap calculation.

Start with [Creating a Hamiltonian](generated/tutorial_hamiltonian.md),
[Simulating a Gibbs sampler](generated/tutorial_thermalize.md), then
[Custom filters](generated/tutorial_custom_filters.md) and
[Interpreting diagnostics](generated/tutorial_diagnostics.md).
The [interface contract](api_contract.md) records supported domains, controls,
physical conversions and reconstruction limits; [filter theory](theory_filters.md)
explains the finite-spectrum CKG–DLL comparison. For the evolution equation
and its relation to finite channel steps, see [Lindblad dynamics](theory_dynamics.md).

Dense Hamiltonian diagonalisation and density-matrix storage remain exponential
in qubit number. `dry_run=true` checks resource estimates before spectral
preparation; this software makes no arbitrary-Hamiltonian scalability promise.

## Installation

The package is under active development and is not registered in Julia's
General registry:

```julia
import Pkg
Pkg.add(url="https://github.com/benzabonanza/QuantumFurnace.jl")
```

## Sources

- Ding, Li and Lin, [Efficient quantum Gibbs samplers with Kubo–Martin–Schwinger detailed balance condition](https://arxiv.org/abs/2404.05998).
- Chen, Kastoryano and Gilyén, [An efficient and exact noncommutative quantum Gibbs sampler](https://arxiv.org/abs/2311.09207).
- Chen, Kastoryano, Brandão and Gilyén, [Quantum thermal state preparation](https://arxiv.org/abs/2303.18224).
