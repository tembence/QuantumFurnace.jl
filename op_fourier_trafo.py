import numpy as np
import qutip as qt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer
from qiskit_aer import StatevectorSimulator
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit import Parameter
from qiskit.circuit.library import QFT
from random import randint
from time import time

from tools.quantum import ham_evol, trotter_step_heisenberg
from tools.classical import get_smallest_bohr_freq, get_hamiltonian_matrix, pad_term


def operator_fourier_circuit(op: Operator, num_sys_qubits: int, num_energy_bits: int, sigma: float,
                             trott_ham_step: QuantumCircuit) -> QuantumCircuit:

    qr_op = QuantumRegister(num_sys_qubits, name='op')
    qr_energy = QuantumRegister(num_energy_bits, name='w')
    circ = QuantumCircuit(qr_energy, qr_op, name="OFT")
    
    ham_time_evol = lambda T: ham_evol(num_qubits, total_time=T, trotter_step=trott_ham_step)

    # Gaussian prep
    prep_circ = brute_prepare_gaussian_state(num_energy_bits, sigma)
    circ.compose(prep_circ, qr_energy, inplace=True)
    
    # exp(-iTH)
    for w in range(num_energy_bits):
        if w == 0:  # Sign bit
            T = 2 ** (num_energy_bits - 1)
        else:
            T = - 2 ** (w - 1)
        
        controlled_time_evol = ham_time_evol(T).control(1, label=f'{T}t0')
        circ.compose(controlled_time_evol, qubits=[qr_energy[w], *qr_op], inplace=True)
      
    # A
    random_sys_qubit = randint(0, num_qubits - 1)
    op_circ = QuantumCircuit(1, name="A")
    op_circ.append(op, [0])
    circ.compose(op_circ, qr_op[random_sys_qubit], inplace=True)
    print(f'Jump applied to {random_sys_qubit}th qubit')
    
    # exp(iTH)
    for w in reversed(range(num_energy_bits)):
        if w == 0:
            T = -2 ** (num_energy_bits - 1)
        else:
            T = 2 ** (w - 1)
        
        controlled_time_evol = ham_time_evol(T).control(1, label=f'{T}t0')
        circ.compose(controlled_time_evol, qubits=[qr_energy[w], *qr_op], inplace=True)
        
    # QFT #FIXME: QFT 2pi factor probably wrong somewhere!
    # Maybe we can just rescale the energy values that come out of the qiskit QFT and not write a new QFT
    circ.compose(QFT(num_energy_bits), qubits=qr_energy, inplace=True)
    
    return circ
    
def brute_prepare_gaussian_state(num_energy_bits: int, sigma: float) -> QuantumCircuit:
    """An early implementation for low number of energy register qubits. For more qubits we would use QSVT maybe."""
    
    # Time labels in computational basis
    decimal_time_labels = list(range(2**(num_energy_bits - 1)))
    decimal_time_labels.extend(list(range(- 2**(num_energy_bits - 1), 0)))
    
    gauss_amplitude = lambda decimal_time: np.exp(-(decimal_time ** 2) / (4 * sigma ** 2))
    amplitudes = [gauss_amplitude(decimal_time) for decimal_time in decimal_time_labels]
    # Normalize 
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    print(amplitudes)
    
    prep_circ = QuantumCircuit(num_energy_bits, name="gauss")
    prep_circ.initialize(amplitudes, range(num_energy_bits))
    
    return prep_circ
    

if __name__ == "__main__":
    num_qubits = 4
    num_energy_bits = 4
    sigma = 0.5
    trottereized_ham_step = trotter_step_heisenberg(num_qubits)
    pauliX = Operator(Pauli('X'))
    
    # print(hamiltonian)
    # Bohr frequency
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()
    hamiltonian_matrix = get_hamiltonian_matrix([X, X], [Y, Y], [Z, Z], [Z], num_qubits=num_qubits)
    smallest_bohr_freq = get_smallest_bohr_freq(hamiltonian_matrix)
    print(smallest_bohr_freq)

    prep_circ = brute_prepare_gaussian_state(num_energy_bits, sigma)
    
    op_fourier_circ = operator_fourier_circuit(pauliX, num_qubits, num_energy_bits, sigma, trottereized_ham_step)
    
    cr_energy = ClassicalRegister(num_energy_bits, name='cw')
    qr_oft = QuantumRegister(num_qubits + num_energy_bits, name='oft')
    circ = QuantumCircuit(*op_fourier_circ.qregs, cr_energy)
    circ.compose(op_fourier_circ, range(num_energy_bits + num_qubits), inplace=True)
    circ.measure(range(num_energy_bits), cr_energy)
    print(circ)
    tr_circ = transpile(circ, basis_gates=['cx', 'u3'], optimization_level=3)
    
    simulator = Aer.get_backend('statevector_simulator')
    job = simulator.run(tr_circ, shots=1000)
    counts = job.result().get_counts()
    print(counts)