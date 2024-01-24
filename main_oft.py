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

#TODO: incorporate the smallest bohr freq -> time unit, instead of T = 1
#TODO: make Gaussian work 

# (q, w) = (3, 4) or (4, 6) are simpler but all of them could work.
np.random.seed(666)
num_qubits = 3
num_energy_bits = 6
qpe_precision = 2 ** (-num_energy_bits + 1) #!
eps = 0.15
sigma = 10
eig_index = 7

X = qt.sigmax()
Y = qt.sigmay()
Z = qt.sigmaz()

#TODO: Write a function that gives ideal HamHam, i.e. has the lines below
#* Find ideal spectrum
coeff_lower_bound = 0
coeff_xx = np.arange(1, coeff_lower_bound, -0.1)
coeff_yy = np.arange(1, coeff_lower_bound, -0.1)
coeff_zz = np.arange(1, coeff_lower_bound, -0.1)
coeff_z = np.arange(1, coeff_lower_bound, -0.1)

coeffs = np.array(np.meshgrid(coeff_xx, coeff_yy, coeff_zz, coeff_z)).T.reshape(-1, 4)
hamiltonian_ideal_qt = None
coeffs_ideal_spec = []
found_ideal_spectrum = False
for coeff in coeffs:
    hamiltonian_qt = hamiltonian_matrix([X, X], [Y, Y], [Z, Z], coeffs=coeff, num_qubits=num_qubits, symbreak_term=[Z])
    rescaling_factor, shift = rescaling_and_shift_factors(hamiltonian_qt, signed=True)
    rescaling_factor /= (1 - eps)  # [0, 1 - eps]
    rescaling_factor *= 2 #! [-0.25, 0.25]
    shift *= (1 - eps)
    shift /= 2 #!
    
    rescaled_hamiltonian_qt = hamiltonian_qt / rescaling_factor + shift * qt.qeye(hamiltonian_qt.shape[0])
    
    rescaled_exact_spec = np.linalg.eigvalsh(rescaled_hamiltonian_qt)
    
    # Accept coeff only if all spectrum elements are not closer than qpe_precision
    for eigval_i in rescaled_exact_spec:
        spec_without_eigval_i = np.delete(rescaled_exact_spec, np.where(rescaled_exact_spec == eigval_i))
        if np.any(np.abs(spec_without_eigval_i - eigval_i) < qpe_precision):
            break
    else:
        found_ideal_spectrum = True
        hamiltonian_ideal_qt = hamiltonian_qt
        coeffs_ideal_spec = coeff
        exact_spec = np.linalg.eigvalsh(hamiltonian_qt)
        print('Original spectrum: ', np.round(exact_spec, 4))
        print("Ideal spectrum found: ", np.round(rescaled_exact_spec, 4))
        print("Nonrescaled coefficients: ", coeffs_ideal_spec)
        break

if not found_ideal_spectrum:
    print("No ideal spectrum found")
    
rescaling_factor, shift = rescaling_and_shift_factors(hamiltonian_ideal_qt, signed=True)
rescaling_factor /= (1 - eps)  # [min * (1 - eps) / gamma , max * (1 - eps) / gamma] ~ [-4.7, 4.7]
rescaling_factor *= 2 #! [-0.25, 0.25]
shift *= (1 - eps)
shift /= 2 #!
print(f'Rescaling factor {rescaling_factor}, shift {shift}')

rescaled_hamiltonian_qt = hamiltonian_qt / rescaling_factor + shift * qt.qeye(hamiltonian_qt.shape[0])  #* Shift / No Shift
rescaled_coeff = coeffs_ideal_spec / rescaling_factor
print('Rescaled coefficients: ', rescaled_coeff)


trotter_step_circ = trotter_step_heisenberg(num_qubits, coeffs=rescaled_coeff, symbreak=True)

#* Hamiltonian
hamiltonian = HamHam(rescaled_hamiltonian_qt, shift=shift, rescaling_factor=rescaling_factor, 
                     trotter_step_circ=trotter_step_circ)

#* Initial state = eigenstate
initial_state = hamiltonian.eigenstates[:, eig_index]
print(f'Initial energy: {hamiltonian.spectrum[eig_index]}')

#* --- Circuit
initial_state = Statevector(initial_state)
qr_energy = QuantumRegister(num_energy_bits, name="w")
cr_energy = ClassicalRegister(num_energy_bits, name="cr_w")
qr_sys  = QuantumRegister(num_qubits, name="sys")
circ = QuantumCircuit(qr_energy, qr_sys, cr_energy)

# Operator Fourier Transform of jump operator
jump_op = Operator(Pauli('X'))
oft_circ = operator_fourier_circuit(jump_op, num_qubits, num_energy_bits, hamiltonian, 
                                    initial_state=initial_state, sigma=sigma)
circ.compose(oft_circ, inplace=True)
circ.measure(qr_energy, cr_energy)
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


      
#FIXME: With gaussian it doesnt work yet
#TODO: Do it with and without gauss and look at the distributions!

