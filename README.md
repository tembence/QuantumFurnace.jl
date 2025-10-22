# QuantumFurnace.jl

## Overview
A high-performance Julia package for simulating open quantum systems that prepare quantum Gibbs states. `QuantumFurnace` is a framework in which the user can both learn about the details of quantum Gibbs sampling, and also research the topic further with the efficient implementations provided here. This project serves as a complementary resource to the recent papers [[CKBG23](#references), [CKG23](#references)] that achieved a breakthrough in the theory of quantum Gibbs sampling. 

### References
    [CKBG23]   Chen, C.F., Kastoryano, M.J., Brandão, F.G. and Gilyén, A., 2023. Quantum thermal state preparation. arXiv:2303.18224.

    [CKG23]    Chen, C.F., Kastoryano, M.J. and Gilyén, A., 2023. An efficient and exact noncommutative quantum Gibbs sampler. arXiv:2311.09207.

## Features
- Construct approximate and exact detailed balanced Liouvillians.
- Efficiently simulate the algorithm step-by-step to reach the quantum Gibbs state, either on a single node or  distributed over multiple nodes.
- Analyse the errors due to approximations with the separately provided `BOHR`, `ENERGY`, `TIME` and `TROTTER` pictures (see exmaples).
- Input Hamitlonians and jump operators of your choice, and choose or come up with the required functions that can make or break the thermalization process.

Upcoming:
- Generate the corresponding quantum circuits to see the incurring costs, e.g. number of qubits, gates, runtime etc.

## Installation

The package can be installed using the Julia package manager. From the Julia REPL, type `]` to enter `Pkg` mode and run:

```julia
pkg> add QuantumFurnace
```
## Quick Start: $\,$ Finding the Thermal State
This example demonstrates one of the core workflows of `QuantumFurnace.jl`. We will prepare the quantum Gibbs state 
$$\sigma_\beta = \frac{e^{\,-\beta H}}{\text{tr}\;e^{\,-\beta H}}$$
at some inverse temperature $\beta$, for a system that is defined by the Hamiltonian $H$ of the 1D Heisenberg model with an external field. The algorithm then drives the system to the thermal state by applying carefully constructed jump operators.

The process involves four main steps:

1.  **Configure the algorithm parameters:** Set the number of system qubits $n$, inverse temperature $\beta$, timestep size $\delta$, etc.
2. **Define the system:** Provide the Hamiltonian
3. **Define the environment:** Provide the set of jump operators $\{A^a\}$.
4. **Solve:** Use the `run_thermalization` function to find the resulting thermal state up to $\mathcal{O}(\delta^2)$ errors.


```julia
using QuantumFurnace

# --- 1. Configure the algorithm parameters ---
num_qubits = 4
dim = 2^num_qubits
num_energy_bits = 11
beta = 10.0
w0 = 0.05                            # energy estimating precision
t0 = 2pi / (2^num_energy_bits * w0)  # time estimating precision

# Choose the picture to work in:
picture = TimePicture()

# Add coherent term to the evolution for exact detailed balance
# or omit for approx. detailed balance:
with_coherent = true

# Linear combination parameters
with_linear_combination = false
a = 0.0
b = 0.0
eta = 0.0

mixing_time_bound = 10.0
delta = 0.1

config = ThermalizeConfig(
    num_qubits = num_qubits, 
    with_coherent = with_coherent,
    with_linear_combination = with_linear_combination, 
    picture = picture,
    beta = beta,
    a = a,
    b = b,
    num_energy_bits = num_energy_bits,
    w0 = w0,
    t0 = t0,
    mixing_time = mixing_time_bound,
    delta = delta,
)

# --- 2. Define the system Hamiltonian ---

# We generate a 4-qubit chain Heisenberg Hamiltonian
hamiltonian_terms = [["X", "X"], ["Y", "Y"], ["Z", "Z"]]
hamiltonian_coeffs = fill(1.0, length(hamiltonian_terms))
hamiltonian = create_hamham(hamiltonian_terms, hamiltonian_coeffs, num_qubits)

# --- 3. Define the jump operators for the evolution ---
X::Matrix{ComplexF64} = [0 1; 1 0]
Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
Z::Matrix{ComplexF64} = [1 0; 0 -1]
jump_set = [[X], [Y], [Z]]

# We choose 1-site Pauli jumps over each system site
jumps::Vector{JumpOp} = []
jump_normalization = sqrt(length(jump_paulis) * num_qubits)
for jump_A in jump_set
    for site in 1:num_qubits
        jump_op = Matrix(pad_term(jump_A, num_qubits, site)) / jump_normalization
        jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
        orthogonal = (jump_op == transpose(jump_op))
        jump = JumpOp(jump_op, jump_op_in_eigenbasis, orthogonal) 
        push!(jumps, jump)
    end
end

# --- 4. Find the thermal state ---

# Start from some initial state, here, the maximally mixed state:
initial_dm = Matrix{ComplexF64}(I(dim) / dim)

# Evolve the system:
results = run_thermalization(jumps, config, initial_dm, hamiltonian)

@printf("\n Last distance to Gibbs: %s\n", results.distances_to_gibbs[end])
```

## Documentation
For detailed tutorials, background theory, and the full API reference, please see our **[documentation website](https/tembence.github.io/QuantumFurnace.jl/dev/)**.

## Citing
Paper on this work is still in progress but if you use `QuantumFurnace.jl` in your research, please cite the Zenodo DOI of the software package for now. You can find it in the badge at the top the page.

## License

This project is licensed under the MIT License.