# # Tutorial: Finding a Thermal State
#
# In this tutorial, we will walk through the process of finding the steady-state
# thermal density matrix $\rho_{ss}$ for a system coupled to a thermal bath.
# The evolution of such a system is governed by the Lindblad master equation:
#
# $$ \frac{d\rho}{dt} = -i[H, \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right) $$
#
# Our goal is to find the state $\rho_{ss}$ where $\frac{d\rho}{dt} = 0$.

using QuantumFurnace
using LinearAlgebra

# ## Step 1: Define the System
# First, we define the system's Hamiltonian `H`. We'll use a simple
# two-qubit Heisenberg model.
num_qubits = 2

# ## Step 2: Define the Environment
# Next, we define the jump operators `L` and the temperature `T`.
temperature = 0.5
picture = TimePicture()

# ## Step 3: Solve for the Thermal State
# Now, we can call our main solver function.

# ## Step 4: Analyze the Result
# The resulting density matrix should have a trace of 1.