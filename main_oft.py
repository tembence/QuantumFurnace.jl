import numpy as np
import qutip as qt
from scipy.linalg import logm, expm
from qiskit.quantum_info import Operator, state_fidelity
from qiskit import QuantumCircuit
from qiskit_aer import StatevectorSimulator
from qiskit import Aer
from qiskit.circuit.library import QFT

from op_fourier_trafo import *
from tools.classical import *
from tools.quantum import *
from jump_ops import sigmam_LCU

#TODO: incorporate the smallest bohr freq -> time unit, instead of T = 1
#TODO: Understand Gaussian and sigma - N, bohr more.

np.random.seed(667)
num_qubits = 3
num_energy_bits = 6
bohr_bound = 2 ** (-num_energy_bits + 1) #!
eps = 0.05
sigma = 10
eig_index = 2
T = 1
shots = 100

hamiltonian = find_ideal_heisenberg(num_qubits, bohr_bound, eps, signed=False, for_oft=True)
rescaled_coeff = hamiltonian.rescaled_coeffs
# Corresponding Trotter step circuit
trotter_step_circ = trotter_step_heisenberg(num_qubits, coeffs=rescaled_coeff, symbreak=True)
inverse_trotter_step_circ = inverse_trotter_step_heisenberg(num_qubits, coeffs=rescaled_coeff, symbreak=True)
hamiltonian.trotter_step_circ = trotter_step_circ
hamiltonian.inverse_trotter_step_circ = inverse_trotter_step_circ

#* Initial state = eigenstate
initial_state = hamiltonian.eigenstates[:, eig_index]
initial_state = Statevector(initial_state)
print(f'Initial energy: {hamiltonian.spectrum[eig_index]}')

t0 = time()
#* --- Circuit
qr_energy = QuantumRegister(num_energy_bits, name="w")
qr_b = QuantumRegister(1, name='b')  # Block-encoding ancilla for the 1-qubit nonunitary jump
qr_sys  = QuantumRegister(num_qubits, name="sys")

cr_energy = ClassicalRegister(num_energy_bits, name="cr_w")
cr_b = ClassicalRegister(1, name='cr_b')
bithandler = BitHandler([cr_energy, cr_b])

circ = QuantumCircuit(qr_energy, qr_b, qr_sys, cr_energy, cr_b)

# --- Initialize qregs
# Gaussian prep on energy register
if sigma != 0.:
    prep_circ = brute_prepare_gaussian_state(num_energy_bits, sigma)
    circ.compose(prep_circ, qr_energy, inplace=True)
else:  # Conventional QPE
    circ.h(qr_energy)
    
# System prep
circ.initialize(initial_state, qr_sys)
    
# --- Operator Fourier Transform of jump operator
# jump_op = Operator(Pauli('X'))
jump_op = sigmam_LCU()
oft_circ = operator_fourier_circuit(jump_op, num_qubits, num_energy_bits, hamiltonian, 
                                    initial_state=initial_state)

circ.compose(oft_circ, [*list(qr_energy), qr_b[0], *list(qr_sys)], inplace=True)

circ.measure(qr_energy, cr_energy)
circ.measure(qr_b, cr_b)

#* --- Results
tr_circ = transpile(circ, basis_gates=['u', 'cx'], optimization_level=1)
simulator = Aer.get_backend('statevector_simulator')
job = simulator.run(tr_circ, shots=shots)
counts = job.result().get_counts()
counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

print(counts)


bithandler.measured_counts = counts
block_ancilla_counts = bithandler.get_counts_for_creg(cr_b)
successful_shots = block_ancilla_counts['0']
print('Successful shots:', successful_shots)
print(f'Block encoding ancilla counts: {block_ancilla_counts}')

counts_with_successful_jump = bithandler.get_counts_with_condition(cr_b, '0')
print('Successful jump counts:')
print(counts_with_successful_jump)
bithandler.measured_counts = counts_with_successful_jump
energy_counts = bithandler.get_counts_for_creg(cr_energy)

phase_bits = list(energy_counts.keys())[0] # take the most often obtaned result
phase_bits_shots = energy_counts[phase_bits]
# Main bitstring result
# signed binary to decimal:
if phase_bits[0] == '1':
    phase = (int(phase_bits[1:], 2) - 2**(num_energy_bits - 1)) / 2**num_energy_bits  # exp(i 2pi phase)
else:
    phase = int(phase_bits[1:], 2) / 2**num_energy_bits

#TODO: Only look at results with the block ancilla in 0 state:
# Combine phases
combined_phase = 0.
for i in range(len(energy_counts.keys())):
    if list(energy_counts.keys())[i][0] == '1':
        phase_part = (int(list(energy_counts.keys())[i][1:], 2) - 2**(num_energy_bits - 1)) / 2**num_energy_bits
    else:
        phase_part = int(list(energy_counts.keys())[i][1:], 2) / 2**num_energy_bits
        
    combined_phase += phase_part * list(energy_counts.values())[i] / successful_shots
T = 1
estimated_energy = phase / T  # exp(i 2pi phase) = exp(i 2pi E T)
estimated_combined_energy = combined_phase / T


print(f'Estimated energy: {estimated_energy}')  # I guess it peaks at the two most probable eigenstates and it will give either one of them and
                                                # not the energy in between them.
print(f'Combined estimated energy: {estimated_combined_energy}')  



