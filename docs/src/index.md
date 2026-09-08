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
    times=range(0, 4; length=21), diagnostics=:standard)
result.trajectory.distances
result.diagnostics
result.spectrum.reliability
```

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
explains the finite-spectrum CKG–DLL comparison.

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
