import numpy as np
import qutip as qt
from scipy.linalg import logm, expm
from qiskit.quantum_info import Operator, state_fidelity
from qiskit import QuantumCircuit
from qiskit_aer import StatevectorSimulator
from qiskit import Aer
from qiskit.circuit.library import QFT
import pickle
from time import time

from op_fourier_trafo_unitary import *
from boltzmann import *
from liouv_step_unitary import *
from tools.classical import *
from tools.quantum import *


#TODO: Truncated Gaussian?
#TODO: figure out delta value, and also could be adaptive later
#TODO: How can we get some information out for the energy in mid run?

np.random.seed(667)
num_qubits = 3
num_energy_bits = 6
bohr_bound = 2 ** (-num_energy_bits + 1) #!
eps = 0.1
sigma = 5
eig_index = 2
T = 1
shots = 1
delta = 0.1
liouv_time = 1
liouv_steps = int(liouv_time / delta)

# --- Hamiltonian
hamiltonian = find_ideal_heisenberg(num_qubits, bohr_bound, eps, signed=False, for_oft=True)
rescaled_coeff = hamiltonian.rescaled_coeffs
# Corresponding Trotter step circuit
trotter_step_circ = trotter_step_heisenberg(num_qubits, coeffs=rescaled_coeff, symbreak=True)
inverse_trotter_step_circ = inverse_trotter_step_heisenberg(num_qubits, coeffs=rescaled_coeff, symbreak=True)
hamiltonian.trotter_step_circ = trotter_step_circ
hamiltonian.inverse_trotter_step_circ = inverse_trotter_step_circ

# Initial state = eigenstate
initial_state = hamiltonian.eigenstates[:, eig_index]
initial_state = Statevector(initial_state)
print(f'Initial energy: {hamiltonian.spectrum[eig_index]}')

t0 = time()
#* --- Circuit
qr_delta = QuantumRegister(1, name='delta')
qr_boltzmann = QuantumRegister(1, name='boltz')
qr_energy = QuantumRegister(num_energy_bits, name="w")
qr_sys  = QuantumRegister(num_qubits, name="sys")

cr_delta = ClassicalRegister(1, name='cr_delta')
cr_boltzmann = ClassicalRegister(1, name='cr_boltz')
cr_energy = ClassicalRegister(num_energy_bits, name="cr_w")
bithandler = BitHandler([cr_delta, cr_boltzmann, cr_energy])

circ = QuantumCircuit(qr_delta, qr_boltzmann, qr_energy, qr_sys, cr_delta, cr_boltzmann, cr_energy)

# --- Initialize qregs
# Gaussian prep on energy register
if sigma != 0.:
    prep_circ = brute_prepare_gaussian_state(num_energy_bits, sigma)
    circ.compose(prep_circ, qr_energy, inplace=True)
else:  # Conventional QPE
    circ.h(qr_energy)
    
# System prep
circ.initialize(initial_state, qr_sys)

#* --- Liouvillian steps
#TODO: See if pretranspiled subcircuits can be easily added to circuits, meaning no long retranspilation
liouv_circ = liouv_step_circ(num_qubits, num_energy_bits, delta, initial_state)
tr_liouv_circ = transpile(liouv_circ, basis_gates=['u', 'cx'], optimization_level=0)  #! Test it
for _ in range(liouv_steps):
    circ.compose(tr_liouv_circ, [qr_delta[0], qr_boltzmann[0], *list(qr_energy), *list(qr_sys)], inplace=True)

    # --- Measure ancillas
    circ.measure(qr_delta, cr_delta)
    circ.measure(qr_energy, cr_energy)
    circ.measure(qr_boltzmann, cr_boltzmann)
    
# print(circ)
t1 = time()

# Pickle circuit
with open(f'data/one_liouv_step_n{num_qubits}k{num_energy_bits}.pkl', 'wb') as f:
    pickle.dump(circ, f)

print(f'Circuit constructed in {t1 - t0} s.')
#* --- Results
t2 = time()
tr_circ = transpile(circ, basis_gates=['u', 'cx'], optimization_level=0)
t3 = time()
print(f'Circuit transpiled in {t3 - t2} s.')

# Pickle transpiled circuit
with open(f'data/tr_one_liouv_step_n{num_qubits}k{num_energy_bits}.pkl', 'wb') as f:
    pickle.dump(tr_circ, f)

simulator = Aer.get_backend('statevector_simulator')
job = simulator.run(tr_circ, shots=shots)
counts = job.result().get_counts()
counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
t4 = time()
print(f'Circuit run in {t4 - t3} s for {shots} shots.')
print(counts)
bithandler.measured_counts = counts
energy_counts = bithandler.get_counts_for_creg(cr_energy)
phase_bits = list(energy_counts.keys())[0] # take the most often obtaned result
phase_bits_shots = energy_counts[phase_bits]
# Main bitstring result
# signed binary to decimal:
if phase_bits[0] == '1':
    phase = (int(phase_bits[1:], 2) - 2**(num_energy_bits - 1)) / 2**num_energy_bits  # exp(i 2pi phase)
else:
    phase = int(phase_bits[1:], 2) / 2**num_energy_bits

# Combine phases
combined_phase = 0.
for i in range(len(energy_counts.keys())):
    if list(energy_counts.keys())[i][0] == '1':
        phase_part = (int(list(energy_counts.keys())[i][1:], 2) - 2**(num_energy_bits - 1)) / 2**num_energy_bits
    else:
        phase_part = int(list(energy_counts.keys())[i][1:], 2) / 2**num_energy_bits
        
    combined_phase += phase_part * list(energy_counts.values())[i] / shots
    
estimated_energy = phase / T  # exp(i 2pi phase) = exp(i 2pi E T)
estimated_combined_energy = combined_phase / T

print(f'Estimated energy: {estimated_energy}')  # I guess it peaks at the two most probable eigenstates 
                                                # and it will give either one of them and
                                                # not the energy in between them.
print(f'Combined estimated energy: {estimated_combined_energy}')  

#* Boltzmann result
boltzmann_counts = bithandler.get_counts_for_creg(cr_boltzmann)
print(f'Boltzmann counts: {boltzmann_counts}')

#* Delta result
delta_counts = bithandler.get_counts_for_creg(cr_delta)
print(f'Delta counts: {delta_counts}')

