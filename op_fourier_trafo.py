import numpy as np
import qutip as qt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer
from qiskit_aer import StatevectorSimulator
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import Statevector, random_statevector, partial_trace, DensityMatrix
from qiskit.circuit import Parameter
from qiskit.circuit.library import QFT
import random
from time import time
from typing import Optional

from tools.quantum import *
from tools.classical import *

#TODO: Check if qiskit transpiles many T = 1 controlled evolutions the same as just having T = 1, 2, 4, 8 one controlled ev.
#! Careful with different random seeds in main and here.

def operator_fourier_circuit(jump_op: QuantumCircuit, num_qubits: int, num_energy_bits: int, hamiltonian: HamHam,
                             initial_state: np.ndarray = None) -> QuantumCircuit:

    trotter_step_circ = hamiltonian.trotter_step_circ
    qr_energy = QuantumRegister(num_energy_bits, name='w')
    qr_b = QuantumRegister(1, name='b')  # Block-encoding ancilla for the 1-qubit nonunitary jump
    qr_sys = QuantumRegister(num_qubits, name='sys')
    circ = QuantumCircuit(qr_energy, qr_b, qr_sys, name="OFT")
    
    circ_to_analyze = QuantumCircuit(qr_energy, qr_b, qr_sys, name="OFT/")
    circ_to_analyze.initialize(initial_state, qr_sys)
    
    # Energy before jump
    initial_statevector = Statevector(circ_to_analyze).data
    padded_rescaled_hamiltonian = np.kron(hamiltonian.qt.full(), np.eye(2**(num_energy_bits + 1)))  # top-bottom = right-left
    energy_before_jump = initial_statevector.conj().T @ padded_rescaled_hamiltonian @ initial_statevector
    print(f'Energy before jump: {energy_before_jump.real}')
    
    # Time evolutions for unit time, T = 1 (in H's units and scale)
    num_trotter_steps = 10
    T = 1
    total_time = 2 * np.pi * T
    U_pos = ham_evol(num_qubits, trotter_step=trotter_step_circ, num_trotter_steps=num_trotter_steps, time=total_time)
    U_neg = ham_evol(num_qubits, trotter_step=trotter_step_circ, num_trotter_steps=num_trotter_steps, time=(-1)*total_time)
    cU_pos = U_pos.control(1, label='+')
    cU_neg = U_neg.control(1, label='-')
    
    # exp(-i 2pi H T) E_old
    for w in range(num_energy_bits):
        # circ.p(- total_time * hamiltonian.shift * 2**w, qr_energy[w])  #!
        if w != num_energy_bits - 1:
            for _ in range(2**w):
                circ.compose(cU_neg, [w, *list(qr_sys)], inplace=True)
        else:  # q = last qubit (MSB) has opposite sign
            for _ in range(2**w):
                circ.compose(cU_pos, [w, *list(qr_sys)], inplace=True)
    
    # Jump A
    random_sys_qubit = np.random.randint(0, num_qubits - 1)
    circ.compose(jump_op, [qr_sys[random_sys_qubit], qr_b[0]], inplace=True)
    print(f'Jump applied to {random_sys_qubit}th qubit')
    
    # For analysis
    circ_to_analyze.compose(circ, [*list(qr_energy), qr_b[0], *list(qr_sys)], inplace=True)
    statevector = Statevector(circ_to_analyze).data
    dm = DensityMatrix(circ_to_analyze)
    sys_dm = dm.partial_trace(list(range(num_energy_bits + 1)))
    
    actual_energy_on_sys = sys_dm.expectation_value(hamiltonian.qt.full())
    print(f'Energy with DM: {actual_energy_on_sys}')
    zerozero = np.array([[1, 0], [0, 0]])
    padded_zerozero = np.kron(np.eye(2**num_qubits), zerozero)
    padded_zerozero = np.kron(padded_zerozero, np.eye(2**num_energy_bits))
    statevector_with_block0 = padded_zerozero @ statevector  # sv with successful jump block encoding
    #TODO: Compute this energy after jump outside of the algorithm, isolated, maybe subspace order is messed up
    energy_after_jump = statevector_with_block0.conj().T @ padded_rescaled_hamiltonian @ statevector_with_block0
    print(f'Energy after jump: {energy_after_jump.real}')
    omega = energy_after_jump - energy_before_jump
    print(f'Energy jump: {omega.real}')
    
    # # exp(i 2pi H T)
    for w in range(num_energy_bits):
        # circ.p(total_time * hamiltonian.shift * 2**w, qr_energy[w])  #! Shift cancel
        if w != num_energy_bits - 1:
            for _ in range(2**w):
                circ.compose(cU_pos, [w, *list(qr_sys)], inplace=True)
        else:  # q = last qubit (MSB) has opposite sign
            for _ in range(2**w):
                circ.compose(cU_neg, [w, *list(qr_sys)], inplace=True)
        
    circ.compose(QFT(num_energy_bits, inverse=True), qubits=qr_energy, inplace=True)

    return circ
    
def brute_prepare_gaussian_state(num_energy_bits: int, sigma: float) -> QuantumCircuit:
    """An early implementation for low number of energy register qubits. 
    For more qubits we would use QSVT maybe."""
    
    # Time labels in computational basis
    decimal_time_labels = list(range(2**(num_energy_bits - 1)))
    decimal_time_labels.extend(list(range(- 2**(num_energy_bits - 1), 0)))
    
    gauss_amplitude = lambda decimal_time: np.exp(-(decimal_time ** 2) / (4 * sigma ** 2))
    amplitudes = [gauss_amplitude(decimal_time) for decimal_time in decimal_time_labels]
    # Normalize 
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    # print(amplitudes)
    
    prep_circ = QuantumCircuit(num_energy_bits, name="gauss")
    prep_circ.initialize(amplitudes, range(num_energy_bits))
    
    return prep_circ

def inverse_operator_fourier_transform(jump_op: QuantumCircuit, num_qubits: int, 
                                       num_energy_bits: int, hamiltonian: HamHam) -> QuantumCircuit:
    
    trotter_step_circ = hamiltonian.inverse_trotter_step_circ  # Inverse!
    qr_energy = QuantumRegister(num_energy_bits, name='w')
    qr_b = QuantumRegister(1, name='b')  # Block-encoding ancilla for the 1-qubit nonunitary jump
    qr_sys = QuantumRegister(num_qubits, name='sys')
    circ = QuantumCircuit(qr_energy, qr_b, qr_sys, name="OFT")
    
    circ.compose(QFT(num_energy_bits, inverse=False), qubits=qr_energy, inplace=True)  # Normal QFT
    
    num_trotter_steps = 10
    T = 1
    total_time = - 2 * np.pi * T  #! Minus for inverse
    U_pos = ham_evol(num_qubits, trotter_step=trotter_step_circ, num_trotter_steps=num_trotter_steps, time=total_time)
    U_neg = ham_evol(num_qubits, trotter_step=trotter_step_circ, num_trotter_steps=num_trotter_steps, time=(-1)*total_time)
    cU_pos = U_pos.control(1, label='+')
    cU_neg = U_neg.control(1, label='-')
    
    # # exp(i 2pi H T)
    for w in reversed(range(num_energy_bits)):
        # circ.p(total_time * hamiltonian.shift * 2**w, qr_energy[w])  #! Shift cancel
        if w != num_energy_bits - 1:
            for _ in range(2**w):
                circ.compose(cU_pos, [w, *list(qr_sys)], inplace=True)
        else:  # q = last qubit (MSB) has opposite sign
            for _ in range(2**w):
                circ.compose(cU_neg, [w, *list(qr_sys)], inplace=True)
                
    # Jump A
    random_sys_qubit = np.random.randint(0, num_qubits - 1)
    circ.compose(jump_op, [qr_sys[random_sys_qubit], qr_b[0]], inplace=True)
    print(f'Inverse jump applied to {random_sys_qubit}th qubit')       
    
    # exp(-i 2pi H T) E_old
    for w in reversed(range(num_energy_bits)):
        # circ.p(- total_time * hamiltonian.shift * 2**w, qr_energy[w])  #!
        if w != num_energy_bits - 1:
            for _ in range(2**w):
                circ.compose(cU_neg, [w, *list(qr_sys)], inplace=True)
        else:  # q = last qubit (MSB) has opposite sign
            for _ in range(2**w):
                circ.compose(cU_pos, [w, *list(qr_sys)], inplace=True)
                
    return circ