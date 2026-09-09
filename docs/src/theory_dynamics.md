# Lindblad dynamics and Gibbs sampling

A density matrix ``\rho`` describes the state of a quantum system: it is
positive semidefinite and has trace one. A time-independent Lindblad generator
evolves it according to

```math
\frac{d\rho}{dt}=\mathcal L(\rho)
=-i[B,\rho]+\sum_a\left(
L_a\rho L_a^\dagger-\frac12\{L_a^\dagger L_a,\rho\}\right),
\qquad \rho(t)=e^{t\mathcal L}(\rho_0).
```

Here ``B`` is Hermitian and ``\{A,C\}=AC+CA``. Positive rates can be absorbed
into the jump operators as ``L_a=\sqrt{\gamma_a}\,A_a``. The evolution preserves
positivity and trace. The deterministic density-matrix trajectory represents
the ensemble state; it does not sample individual quantum jumps.

For Hamiltonian ``H`` and inverse temperature ``\beta``, the Gibbs target is

```math
\rho_\beta=\frac{e^{-\beta H}}{Z},\qquad Z=\operatorname{Tr}(e^{-\beta H}).
```

A Gibbs sampler chooses filtered sources and a coherent correction ``B`` so
that ``\mathcal L(\rho_\beta)=0``. In the DLL and CKG constructions used here,
``B`` is the sampler's correction; it is generally different from ``H``.
The package does not additionally insert ``-i[H,\rho]``. For the filter
identities and the correction formula, see [Filter theory](theory_filters.md).

Stationarity alone does not imply convergence to a unique Gibbs state. For
example, pure dephasing leaves every diagonal state stationary. Use
[diagnostics](generated/tutorial_diagnostics.md) to examine stationarity,
uniqueness and the numerical accuracy of a chosen discretisation separately.

## Continuous evolution and finite channel steps

`simulate_gibbs` defaults to continuous Lindblad evolution. Its Krylov method
approximates the action of ``e^{t\mathcal L}`` without storing a dense
superoperator. Hamiltonian spectral preparation and density-matrix storage
are still dense.

With `sim=Thermalize()`, the solver repeatedly applies a finite
weak-measurement channel ``\mathcal E_\delta``. After ``k`` steps the state is
``\mathcal E_\delta^k(\rho_0)``. At finite step size this channel generally
differs from ``e^{\delta\mathcal L}`` and may have a bias from Gibbs. Generator
time, channel-step count and elapsed wall time are separate quantities.
The [simulation tutorial](generated/tutorial_thermalize.md) demonstrates both
evolution targets and how to interpret their output.

## Sources

- G. Lindblad, [On the generators of quantum dynamical semigroups](https://doi.org/10.1007/BF01608499) (1976).
- Z. Ding, B. Li and L. Lin, [Efficient quantum Gibbs samplers with Kubo–Martin–Schwinger detailed balance condition](https://arxiv.org/abs/2404.05998).
- C.-F. Chen, M. J. Kastoryano and A. Gilyén, [An efficient and exact noncommutative quantum Gibbs sampler](https://arxiv.org/abs/2311.09207).
