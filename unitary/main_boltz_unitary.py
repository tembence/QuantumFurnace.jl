import numpy as np
import qutip as qt
from scipy.linalg import logm, expm
from qiskit.quantum_info import Operator, state_fidelity
from qiskit import QuantumCircuit
from qiskit_aer import StatevectorSimulator
from qiskit import Aer
from qiskit.circuit.library import QFT
import pickle

import sys
sys.path.append('/Users/bence/code/liouvillian_metro/')

from op_fourier_trafo_unitary import operator_fourier_circuit, brute_prepare_gaussian_state
from boltzmann import lookup_table_boltzmann
from tools.classical import *
from tools.quantum import *

we_pickle_questionmark = True  #!

np.random.seed(667)
num_qubits = 3
num_energy_bits = 6
bohr_bound = 2 ** (-num_energy_bits + 1)
eps = 0.1
sigma = 5
eig_index = 1
T = 1
shots = 1

hamiltonian = find_ideal_heisenberg(num_qubits, bohr_bound, eps, signed=False, for_oft=True)
rescaled_coeff = hamiltonian.rescaled_coeffs
# Corresponding Trotter step circuit
trotter_step_circ = trotter_step_heisenberg(num_qubits, coeffs=rescaled_coeff, symbreak=True)
hamiltonian.trotter_step_circ = trotter_step_circ

#* --- Circuit
qr_boltzmann = QuantumRegister(1, name='boltz')
qr_energy = QuantumRegister(num_energy_bits, name="w")
qr_sys  = QuantumRegister(num_qubits, name="sys")

cr_boltzmann = ClassicalRegister(1, name='cr_boltz')
cr_energy = ClassicalRegister(num_energy_bits, name="cr_w")
bithandler = BitHandler([cr_boltzmann, cr_energy])

circ = QuantumCircuit(qr_boltzmann, qr_energy, qr_sys, cr_boltzmann, cr_energy)
U_circ = QuantumCircuit(qr_boltzmann, qr_energy, qr_sys, name='U')

# --- Initialize qregs
# Gaussian prep on energy register
if sigma != 0.:
    prep_circ = brute_prepare_gaussian_state(num_energy_bits, sigma)
    circ.compose(prep_circ, [*list(qr_energy)], inplace=True)
else:  # Conventional QPE
    circ.h(qr_energy)
    
# System prep, initial state = eigenstate
initial_state = hamiltonian.eigenstates[:, eig_index]
initial_state = Statevector(initial_state)

if we_pickle_questionmark:
    with open(f'data/block_initial_state_n{num_qubits}k{num_energy_bits}.pkl', 'wb') as f:
        pickle.dump(initial_state, f)
        
print(f'Initial energy: {hamiltonian.spectrum[eig_index]}')

circ.initialize(initial_state, qr_sys)
    
# --- Operator Fourier Transform of jump operator
jump_op = Operator(Pauli('X'))
oft_circ = operator_fourier_circuit(jump_op, num_qubits, num_energy_bits, hamiltonian, 
                                    initial_state=initial_state)
U_circ.compose(oft_circ, [*list(qr_energy), *list(qr_sys)], inplace=True)
print('OFT')

# --- Act on Boltzmann coin
boltzmann_circ = lookup_table_boltzmann(num_energy_bits)
U_circ.compose(boltzmann_circ, [qr_boltzmann[0], *list(qr_energy)], inplace=True)
print('Boltzmann')

circ.compose(U_circ, [qr_boltzmann[0], *list(qr_energy), *list(qr_sys)], inplace=True)

state_before_measure = Statevector(circ)

if we_pickle_questionmark:
    with open(f'data/block_state_before_measure_n{num_qubits}k{num_energy_bits}.pkl', 'wb') as f:
        pickle.dump(state_before_measure, f)

# --- Measure
circ.measure(qr_energy, cr_energy)
# Measure this after energy register, then the boltzmann sees the projected states energy
circ.measure(qr_boltzmann, cr_boltzmann)  
print(circ)

print('Circuit constructed.')
#* --- Results
tr_circ = transpile(circ, basis_gates=['u', 'cx'], optimization_level=1)

if we_pickle_questionmark:
    with open(f'data/block_tr_n{num_qubits}k{num_energy_bits}.pkl', 'wb') as f:
        pickle.dump(tr_circ, f)
        
print('Circuit transpiled.')
simulator = Aer.get_backend('statevector_simulator')
job = simulator.run(tr_circ, shots=shots)
counts = job.result().get_counts()
counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
print(counts)
bithandler.measured_counts = counts
energy_counts = bithandler.get_counts_for_creg(cr_energy)
phase_bits = list(energy_counts.keys())[0] # take the most often obtaned result
phase_bits_shots = energy_counts[phase_bits]
# Main bitstring result
# signed binary to decimal:
if phase_bits[0] == '1':
    phase = (int(phase_bits[1:], 2) - 2**(num_energy_bits - 1))  # exp(i 2pi phase)
else:
    phase = int(phase_bits[1:], 2)

# Combine phases
combined_phase = 0.
for i in range(len(energy_counts.keys())):
    if list(energy_counts.keys())[i][0] == '1':
        phase_part = (int(list(energy_counts.keys())[i][1:], 2) - 2**(num_energy_bits - 1)) 
    else:
        phase_part = int(list(energy_counts.keys())[i][1:], 2)
        
    combined_phase += phase_part * list(energy_counts.values())[i] / shots
    
estimated_energy = 2 * np.pi * phase / (T * 2**num_energy_bits)  # exp(i 2pi phase) = exp(i E T)
estimated_combined_energy = 2 * np.pi * combined_phase / (T * 2**num_energy_bits)


print(f'Estimated energy: {estimated_energy}')  # I guess it peaks at the two most probable eigenstates 
                                                # and it will give either one of them and
                                                # not the energy in between them.
print(f'Combined estimated energy: {estimated_combined_energy}')  

#* Boltzmann result
print('Boltzmann result:')
boltzmann_counts = bithandler.get_counts_for_creg(cr_boltzmann)
print(boltzmann_counts)

