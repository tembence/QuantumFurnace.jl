# # Open Quantum System Dynamics
# We will be working with open quantum systems, i.e. where a system we are interested in $S$ is interacting with an 
# environment #E#. In the weak coupling limit, one can make some more or less reasonable assumptions (Born - lets us
# treat the subsystem separately from the environment; Markov - the interactions in the bath leave no mark, and there won'the
# be any back-action; Secular - dropping fast oscillating terms to restore positivity) for which we can find the following 
# Lindbladian generator, that fully describes the time evolution of state $\rho$ in our system $S$:
# ```math
# \mathcal{L} = -i[H, \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right),
# ```
# which generates the time evolutions for a given $t$ as:
# ```math
# \rho(t) = e^{\mathcal{L}t}(\rho_0)
# ```
# or
# ```math
# \frac{\mathrm{d}}{\mathrm{d}t}\rho = \mathcal{L}(\rho).
# ```
#