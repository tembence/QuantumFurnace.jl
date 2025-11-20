```@meta
EditURL = "../literate/theory_oqs_dynamics.jl"
```

# Open Quantum System Dynamics
We will be working with open quantum systems, i.e. where a system we are interested in $S$ is interacting with an
environment #E#. In the weak coupling limit, one can make some more or less reasonable assumptions (Born - lets us
treat the subsystem separately from the environment; Markov - the interactions in the bath leave no mark, thus there won't
be any back-action on the system; Secular - dropping fast oscillating terms to restore positivity) for which we can find the following
Lindbladian generator, that fully describes the time evolution of state $\rho$ in our system $S$:
```math
\mathcal{L} = -i[H, \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right),
```
which generates all the possible temporal maps from some time to a later one, $\rho(t=0) \mapsto \rho(t), as$
```math
\rho(t) = e^{\mathcal{L}t}(\rho_0).
```
Or we can understand it as the infinitesimal change of the state $\rho$ in time:
```math
\frac{\mathrm{d}}{\mathrm{d}t}\rho = \mathcal{L}(\rho).
```
We can distinguish two parts in the Lindbladian $\mathcal{L}$, one corresponds to a coherent evolution with respect
to the Hermitian operator $H$. This describes the dynamics of the system where there are no dissipation to the
environment due to the interactions. While the sum part, describes the dissipation with respect to the jump
operators $L_k$ and with jump rates $\sqrt{\gamma_k}$ that are connected to the correlation functions in the environment.

Such an open quantum system evolution brings the system $S$ from some initial state $\rho_0$ to a final state $\rho(t)$
by randomly (given by the jump rates) exploring the state space. If the coherent evolution and the jump operators are such
that all possible states in the space is reachable, then such an evolution is called, irreducible and has a unique
stationary state $\sigma$ to which the evolution is converging to. In principle if there are no inescapable traps into which our
wandering state can fall into then it will eventually reach its destination.

If the environment $E$ is in a thermal equilibrium, then we can derive that for a sufficiently well-chosen set of jump
operators, the system $S$ itself will be driven to its thermal equilibrium as well. This state we call the Gibbs state
and it is defined as
```math
\sigma_\beta = \frac{e^{-\beta H_S}}{Z_S},
```
with the inverse temperature $\beta$, system Hamiltonian $H_S$ and the partition function $Z_S = \mathrm{tr}[e^{-\beta H_S}]$.
But the environment being in a thermal equilibrium is not a necessary condition, just a sufficient one.
In a bottom-top approach, we can also construct jump operators and their rates, such that we don't require the environment to be in
thermal equilibrium, yet the system $S$ would still evolve towards its Gibbs state $\sigma_\beta$. How this is done,
we refer to another entry in the Theory section, called [Detailed Balance](theory_detailed_balance.md).

For now, let’s assume that the Gibbs state is indeed what the environment is driving the system toward. By this we
also assume that the evolution is irreducible, i.e. that all parts of the state space is reachable over the evolution,
in which case the Gibbs state is the unique stationary state of the generator:
```math
\mathcal{L}(\sigma_\beta) = 0,
```
or
```math
e^{\mathcal{Lt}(\sigma_\beta)} = \sigma_\beta.
```
The quantum algorithms used and analysed in `QuantumFurnace` (see [Weak Measurement](theory_weak_measurement.md))
evolve the system not by the above full form, but rather by a sequence of discrete time steps of duration $\delta$.
A single algorithmic step is a quantum channel (or a quantum circuit construction to better imagine it),
let's call it $\mathcal{C}_\delta$, that *approximates* the ideal evolution $e^{\delta\mathcal{L}}$ via the form
(1 + \delta \mathcal{L}) up to errors of $\mathcal{O}(\delta^2)$ [CKGB23 Theorem III.1].

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

