import numpy as np
import qutip as qt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer
from qiskit_aer import StatevectorSimulator
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import Statevector, random_statevector, partial_trace
from qiskit.circuit import Parameter
from qiskit.circuit.library import QFT
import random
from time import time
from typing import Optional

from tools.quantum import ham_evol, trotter_step_heisenberg
from tools.classical import *


#TODO: With a fixed step size we are not doing the exact amount of time evolution needed... but sometimes more
#TODO: Make spectrum nondegenerate
#TODO: Do it for 1-2 qubits, and ES as initial state with a jump that brings it to an ES exactly

def operator_fourier_circuit(op: Operator, num_sys_qubits: int, num_energy_bits: int,
                             trott_ham_step: QuantumCircuit, with_gauss: bool=True, sigma: float = None) -> QuantumCircuit:

    qr_energy = QuantumRegister(num_energy_bits, name='w')
    qr_sys = QuantumRegister(num_sys_qubits, name='sys')
    circ = QuantumCircuit(qr_energy, qr_sys, name="OFT")
    
    ham_time_evol = lambda T: ham_evol(num_qubits, total_time=T, trotter_step=trott_ham_step, 
                                       shift=shift, rescaling_factor=rescaling_factor)  

    # System initialization
    rand_initial_state = random_statevector(2**num_sys_qubits, seed=667)
    circ.initialize(rand_initial_state, qr_sys)
    
    energy_prejump = energy_from_full_state(circ, hamiltonian, [num_energy_bits, num_sys_qubits], qr_index=1)
    
    # Gaussian prep
    if with_gauss:
        if sigma is None:
            raise ValueError("If with_gauss=True, sigma must be provided.")
        prep_circ = brute_prepare_gaussian_state(num_energy_bits, sigma)
        circ.compose(prep_circ, qr_energy, inplace=True)
    
    # exp(-iTH)
    for w in range(num_energy_bits):
        if w == 0:  # Sign bit
            T = 2 ** (num_energy_bits - 1)
        else:
            T = - 2 ** (w - 1)
            
        # Total time evolution is 2pi * T for correct QPE kickback:
        controlled_time_evol = ham_time_evol(2*np.pi*T).control(1, label=f'{T}t0') 
        circ.compose(controlled_time_evol, qubits=[qr_energy[w], *qr_sys], inplace=True)
    
    # Jump A
    random.seed(667)
    random_sys_qubit = random.randint(0, num_qubits - 1)
    op_circ = QuantumCircuit(1, name="A")
    op_circ.append(op, [0])
    circ.compose(op_circ, qr_sys[random_sys_qubit], inplace=True)
    print(f'Jump applied to {random_sys_qubit}th qubit')
    
    energy_postjump = energy_from_full_state(circ, hamiltonian, [num_energy_bits, num_sys_qubits], qr_index=1)
    omega = energy_postjump - energy_prejump
    print(f'Energy jump: {omega}')
    
    # exp(iTH)
    for w in reversed(range(num_energy_bits)):
        if w == 0:
            T = -2 ** (num_energy_bits - 1)
        else:
            T = 2 ** (w - 1)
        
        controlled_time_evol = ham_time_evol(T).control(1, label=f'{T}t0')
        circ.compose(controlled_time_evol, qubits=[qr_energy[w], *qr_sys], inplace=True)
        
    circ.compose(QFT(num_energy_bits, inverse=True), qubits=qr_energy, inplace=True)
    
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
    
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    num_qubits = 4
    num_energy_bits = 4
    sigma = 0.5
    trottereized_ham_step = trotter_step_heisenberg(num_qubits)
    pauliX = Operator(Pauli('X'))
    
    # Bohr frequency
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()
    hamiltonian = hamiltonian_matrix([X, X], [Y, Y], [Z, Z], [Z], num_qubits=num_qubits)
    hamiltonian = shift_spectrum(hamiltonian)
    # bohr_unit = smallest_bohr_freq(hamiltonian)
    # print(bohr_unit)
    rescaling_factor, shift = rescaling_and_shift_factors(hamiltonian)
    
    op_fourier_circ = operator_fourier_circuit(pauliX, num_qubits, num_energy_bits, trottereized_ham_step,
                                               with_gauss=False, sigma=sigma)
    
    # full_state = Statevector(op_fourier_circ)
    # traced_over_qubits = [op_fourier_circ.qubits.index(qubit) for qubit in op_fourier_circ.qregs[1]]
    # omega_dm = partial_trace(full_state, traced_over_qubits)
    # print(omega_dm.probabilities([0, 1, 2, 3]))
    
    #* Run on simulator
    cr_energy = ClassicalRegister(num_energy_bits, name='cw')
    qr_oft = QuantumRegister(num_qubits + num_energy_bits, name='oft')
    circ = QuantumCircuit(*op_fourier_circ.qregs, cr_energy)
    circ.compose(op_fourier_circ, range(num_energy_bits + num_qubits), inplace=True)
    circ.measure(range(num_energy_bits), cr_energy)
    # print(circ)
    tr_circ = transpile(circ, basis_gates=['cx', 'u3'], optimization_level=3)
    
    simulator = Aer.get_backend('statevector_simulator')
    shots = 1000
    job = simulator.run(tr_circ, shots=shots)
    counts = job.result().get_counts()
    print(counts)
    
    energy = stich_energy_bits_to_value(counts, shots=shots)
    print(energy)