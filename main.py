import numpy as np
import qutip as qt
from scipy.linalg import logm, expm
from qiskit.quantum_info import Operator, state_fidelity
from qiskit import QuantumCircuit
from qiskit_aer import StatevectorSimulator
from qiskit import Aer
from qiskit.circuit.library import QFT

from op_fourier_trafo import *
from boltzmann import *
from tools.classical import *
from tools.quantum import *

np.random.seed(666)
num_qubits = 3
num_energy_bits = 6
bohr_bound = 2 ** (-num_energy_bits + 1) #!
eps = 0.1
sigma = 10
eig_index = 7

hamiltonian = find_ideal_heisenberg(num_qubits, bohr_bound, eps, signed=True, for_oft=True)
rescaled_coeff = hamiltonian.rescaled_coeffs
# Corresponding Trotter step circuit
trotter_step_circ = trotter_step_heisenberg(num_qubits, coeffs=rescaled_coeff, symbreak=True)
hamiltonian.trotter_step_circ = trotter_step_circ

#* Initial state = eigenstate
initial_state = hamiltonian.eigenstates[:, eig_index]
print(f'Initial energy: {hamiltonian.spectrum[eig_index]}')

#* --- Circuit
initial_state = Statevector(initial_state)
qr_boltzmann = QuantumRegister(1, name='boltz')  #TODO: Careful, tensorproducts might not be correct now
qr_energy = QuantumRegister(num_energy_bits, name="w")
cr_energy = ClassicalRegister(num_energy_bits, name="cr_w")
cr_boltzmann = ClassicalRegister(1, name='cr_boltz')
qr_sys  = QuantumRegister(num_qubits, name="sys")
circ = QuantumCircuit(qr_boltzmann, qr_energy, qr_sys, cr_boltzmann, cr_energy)

# Operator Fourier Transform of jump operator
jump_op = Operator(Pauli('X'))
oft_circ = operator_fourier_circuit(jump_op, num_qubits, num_energy_bits, hamiltonian, 
                                    initial_state=initial_state, sigma=sigma)
circ.compose(oft_circ, inplace=True)
circ.measure(qr_energy, cr_energy)

#* Act on Boltzmann coin
boltzmann_circ = look_up_table_boltzmann(num_energy_bits)

print(circ)

#* --- Results
tr_circ = transpile(circ, basis_gates=['u', 'cx'], optimization_level=3)
simulator = Aer.get_backend('statevector_simulator')
shots = 1000
job = simulator.run(tr_circ, shots=shots)
counts = job.result().get_counts()
counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

print(counts)

phase_bits = list(counts.keys())[0] # take the most often obtaned result
phase_bits_shots = counts[phase_bits]
# Main bitstring result
# signed binary to decimal:
if phase_bits[0] == '1':
    phase = (int(phase_bits[1:], 2) - 2**(num_energy_bits - 1)) / 2**num_energy_bits  # exp(i 2pi phase)
else:
    phase = int(phase_bits[1:], 2) / 2**num_energy_bits

# Combine phases
combined_phase = 0.
for i in range(len(counts.keys())):
    if list(counts.keys())[i][0] == '1':
        phase_part = (int(list(counts.keys())[i][1:], 2) - 2**(num_energy_bits - 1)) / 2**num_energy_bits
    else:
        phase_part = int(list(counts.keys())[i][1:], 2) / 2**num_energy_bits
        
    combined_phase += phase_part * list(counts.values())[i] / shots
T = 1
estimated_energy = phase / T  # exp(i 2pi phase) = exp(i 2pi E T)
estimated_combined_energy = combined_phase / T


print(f'Estimated energy: {estimated_energy}')  # I guess it peaks at the two most probable eigenstates and it will give either one of them and
                                                # not the energy in between them.
print(f'Combined estimated energy: {estimated_combined_energy}')  

