```@meta
EditURL = "../literate/tutorial.jl"
```

````@example tutorial
#' # Tutorial: Finding a Thermal State
#'
#' In this tutorial, we will walk through the process of finding the steady-state
#' thermal density matrix $\rho_{ss}$ for a system coupled to a thermal bath.
#' The evolution of such a system is governed by the Lindblad master equation:
#'
#' $$ \frac{d\rho}{dt} = -i[H, \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right) $$
#'
#' Our goal is to find the state $\rho_{ss}$ where $\frac{d\rho}{dt} = 0$.

using QuantumFurnace
using LinearAlgebra

#' ## Step 1: Define the System
#' First, we define the system's Hamiltonian `H`. We'll use a simple
#' two-qubit Heisenberg model.
num_qubits = 2
````

H = generate_heisenberg_hamiltonian(num_qubits, 1.0)

````@example tutorial
#' ## Step 2: Define the Environment
#' Next, we define the jump operators `L` and the temperature `T`.
````

jump_operators = [get_lowering_operator(num_qubits, i) for i in 1:2]

````@example tutorial
temperature = 0.5

#' ## Step 3: Solve for the Thermal State
#' Now, we can call our main solver function.
````

rho_thermal = thermalize(H, jump_operators, temperature)

````@example tutorial
#' ## Step 4: Analyze the Result
#' The resulting density matrix should have a trace of 1.
````

tr(rho_thermal)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

