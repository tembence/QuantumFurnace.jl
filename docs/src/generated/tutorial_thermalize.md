```@meta
EditURL = "../literate/tutorial_thermalize.jl"
```

# Finding a Thermal State
In this tutorial we will give more details on one of the core functions of the software package:
on how to drive a given quantum system to the thermal state via a simulated open quantum system evolution.
We will see how to set up everything for the function `run_thermalization` and also what it actually does behind
the curtains.

In summary, `run_thermalization` takes in a basic set of configurations (number of qubits, temperature, etc.),
the Hamiltonian $H$ that describes the system, an initial state $\rho_0$ where the evolution starts from and the set of jumps
$\{A^a\}$ that prescribe that evolution. It will simulate the [open quantum system evolution](theory_oqs_dynamics.md) or Lindbladian evolution
with a [weak-measurement based algorithm](theory_weak_measurement.md). You can read more about them in the Theory section if you follow
the links. Here, it is enough to just think of it as the following process:
- Start from an initial density matrix $\rho_0$.
- Evolve it for a short $\delta$ amount of time, $\rho_\delta = (\mathds{1} + \delta \mathcal{L})\rho_0 + \mathcal{O}(\delta^2) \simeq e^{\delta \mathcal{L}}(\rho_0)$.
- Repeat until $\rho_t$ gets close enough to the target state $\sigma_\beta = e^{-\beta H} / Z$, i.e. the Gibbs state.
Here the magic is still hidden in the generator $\mathcal{L}$, but the crucial property it has is that the dynamics it generates,
given by $e^{t \mathcal{L}}$, has the Gibbs state as its unique fixed point. Thus if we evolve the system for a long enough time
all other contribution will decay and the only state that survives is the Gibbs state.

Let us go through the main steps of the code:

## 1. Configure the algorithm parameters

````@example tutorial_thermalize
using QuantumFurnace
using LinearAlgebra

num_qubits = 3
dim = 2^num_qubits
num_energy_bits = 10
beta = 10.0
w0 = 0.05                            # Energy estimating precision
t0 = 2pi / (2^num_energy_bits * w0)  # Time estimating precision

domain = TimeDomain()

with_coherent = true                  # For exact detailed balance
with_linear_combination = false       # Gaussian transitions
a = 0.0
b = 0.0
eta = 0.0

mixing_time_bound = 10.0
delta = 0.1

config = ThermalizeConfig(
    num_qubits = num_qubits,
    with_coherent = with_coherent,
    with_linear_combination = with_linear_combination,
    domain = domain,
    beta = beta,
    a = a,
    b = b,
    num_energy_bits = num_energy_bits,
    w0 = w0,
    t0 = t0,
    mixing_time = mixing_time_bound,
    delta = delta,
) ;
nothing #hide
````

**Domains** $\quad$ Algorithms are translated to the quantum computer in the form of quantum circuits, a set of unitary
quantum gates, that inherently work in the time domain, which is also what we mean by choosing `domain` to be `TimeDomain`.
More realistically, time evolutions are decomposed into a Trotter product, which is also possible to see in our code via
`TrotterDomain`. Though the original mathematical problem of finding a Lindbladian that evolves the system to the thermal
state is formulated in the Bohr and Energy domains, that should pose no problem for us. We can always move from the time
domain to the energy domain with the Fourier transform.

**Coherent term** $\quad$ It has been proven (see [Theory](theory_detailed_balance.md)) that by adding a specific coherent term
to the Lindbladian, we can have the Gibbs state as the unique fixed point of the generator. But even if we omit it
by setting `with_coherent = false`, we would find an approximately good (or bad) result.

**Linear combinations** $\quad$ A simpler version of the theory is when we don't take a convex combination of
Lindbladians. We can think of this as singling out a thin Gaussian region for which we allow transitions for certain
energy differences induced by the jumps. We can also turn on the linear combination option, and thus have more borad
transition while keeping the Gibbs state as the target state (see [Theory](theory_convex_combination.md)).

## 2. Define the system Hamiltonian

Next, we will define and construct the Hamiltonian of the subsystem we want to thermalize. We will use the function called
`create_hamham`. If we know how the terms in our Hamiltonian decompose into tensor products of single qubit terms, then
this function will generate a Hamiltonian whose spectrum is within [0.0, 0.45] and creates a HamHam object with some other
necessary fields in it like its eigenvectors, or its Gibbs state.

````@example tutorial_thermalize
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]

hamiltonian_terms = [[X, X], [Y, Y], [Z, Z]]
hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
disordering_term = [Z]
disordering_coeffs = rand(num_qubits) ;
nothing #hide
````

Generate a 4-qubit chain antiferromagnetic Heisenberg Hamiltonian with a disordering field

````@example tutorial_thermalize
hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, disordering_term, disordering_coeffs, num_qubits)
hamiltonian.gibbs = Hermitian(gibbs_state_in_eigen(hamiltonian, beta)) ;
nothing #hide
````

Note that we added here a disordering field to the Hamiltonian in order to make its spectrum unique. A priori the algorithm
should also work with degenerate spectra, but a unique one definitely makes things easier to converge. Nevertheless
exploring what effects a degenerate spectrum have on the algorithm would be quite interesting too.

## 3. Define the jump operators for the evolution

````@example tutorial_thermalize
jump_set = [[X], [Y], [Z]] ;
nothing #hide
````

1-site Pauli jumps are generated over each system site and save their form in the eigenbasis
we work in for effficiency:

````@example tutorial_thermalize
jumps::Vector{JumpOp} = []
jump_normalization = sqrt(length(jump_set) * num_qubits)
for jump_a in jump_set
    for site in 1:num_qubits
        jump_op = Matrix(pad_term(jump_a, num_qubits, site)) / jump_normalization
        jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
        orthogonal = (jump_op == transpose(jump_op))
        jump = JumpOp(jump_op, jump_op_in_eigenbasis, orthogonal)
        push!(jumps, jump)
    end
end ;
nothing #hide
````

Even though it seems unassuming, that we use single-site Pauli jump operators, the actual jumps that are
applied to the system are spread out time evolved operators of the form $A(t) = f(t) exp(iHt) A exp(-iHt)$,
with some Gaussian filter function f(t). Since the time evolutions can be quite large, while the simulable systems
quite small, the actual jumps are effectively full system sized in most cases.
The jump normalization is required for the block encoding in the algorithm, as the operators should have a norm
less than or equal to 1.

## 4. Find the thermal state
Finally we can run the core function, that we will evolve the initial state by approximated Lindbladian evolution,
always a $\delta$ step at a time. The result then will be deviating by $\mathcal{O}(\delta^2)$ errors from the
target Gibbs state.

````@example tutorial_thermalize
initial_dm = Matrix{ComplexF64}(I(dim) / dim) ;
nothing #hide
````

Evolve the system:

````@example tutorial_thermalize
results = run_thermalization(jumps, config, initial_dm, hamiltonian)

println("\n Last distance to Gibbs: %s\n", results.distances_to_gibbs[end])
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

