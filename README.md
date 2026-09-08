# QuantumFurnace.jl

QuantumFurnace.jl is a Julia package for constructing and simulating quantum
Gibbs samplers based on detailed-balance Lindbladians. It supports dense
reference calculations, matrix-free Krylov methods, and full-density-matrix
channel simulations across Bohr, energy, time, and Trotter domains.

The package is under active development and should currently be treated as
pre-alpha research software.

## Features

- Construct KMS, GNS, and Ding--Li--Lin (DLL) Gibbs-sampling Lindbladians.
- Compare exact Bohr-domain constructions with energy-, time-, and
  Trotter-domain approximations.
- Compute fixed points and spectral gaps with dense or matrix-free methods.
- Simulate retained weak-measurement channels and track convergence to the
  Gibbs state.
- Build reproducible Heisenberg and transverse-field Ising Hamiltonians.

## Installation

QuantumFurnace.jl is not yet registered in Julia's General registry. Install
the development version directly from GitHub:

```julia
import Pkg
Pkg.add(url="https://github.com/benzabonanza/QuantumFurnace.jl")
```

Then load it with:

```julia
using QuantumFurnace
```

## Quick start

Build a two-qubit Hamiltonian in physical energy units and evolve the default
DLL sampler, including its coherent correction:

```julia
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

`beta_phys` is inverse temperature in reciprocal energy units (`k_B=1`). The
default initial state is the computational-basis product state `|+⟩⊗|+⟩`.
`distances` contains trace distance, half the trace norm. A threshold crossing
concerns this initial state; independent spectral diagnostics can remain
inconclusive. The times use the generator clock set by the source amplitudes.
Hamiltonian rescaling alone does not multiply that clock.

Select `construction=KMS()` for CKG KMS. DLL supports Bohr and Time domains;
custom Time filters require explicit transform controls. DLL Energy, Trotter
and GQSP reject. General CKG joint kernels support Bohr, Energy and controlled
Time conversion; custom Trotter rejects. Legacy CKG/GNS APIs remain available.
See the [capability contract](docs/src/api_contract.md) for precise limits.

The matrix-free evolution still requires dense Hamiltonian spectral preparation
and density matrices. It does not make arbitrary many-body Hamiltonians
scalable. Inspect resource estimates with `dry_run=true` before a larger run.

## Documentation

Tutorials, background material, and the API reference are available at
[benzabonanza.github.io/QuantumFurnace.jl](https://benzabonanza.github.io/QuantumFurnace.jl/).

## Local documentation build

```bash
QF_DOCS_DEPLOY=false JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=docs --startup-file=no --heap-size-hint=1500M docs/make.jl
```

This develops the local package in a disposable docs environment, executes the
tutorials, and writes `docs/build/`. Deployment is off by default.

## References

- Z. Ding, B. Li, and L. Lin, "Efficient quantum Gibbs samplers with
  Kubo–Martin–Schwinger detailed balance condition," arXiv:2404.05998 (2024).

- C.-F. Chen, M. J. Kastoryano, F. G. S. L. Brandao, and A. Gilyen,
  "Quantum thermal state preparation," arXiv:2303.18224 (2023).
- C.-F. Chen, M. J. Kastoryano, and A. Gilyen, "An efficient and exact
  noncommutative quantum Gibbs sampler," arXiv:2311.09207 (2023).

## Citing

A formal software citation will accompany the first release. Until then,
please cite the repository URL together with the exact commit used in your
work.

## License

QuantumFurnace.jl is available under the MIT License.
