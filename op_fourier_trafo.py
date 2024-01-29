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

from tools.quantum import *
from tools.classical import *

#TODO: Check if qiskit transpiles many T = 1 controlled evolutions the same as just having T = 1, 2, 4, 8 one controlled ev.
#! Careful with different random seeds in main and here.

def operator_fourier_circuit(op: Operator, num_qubits: int, num_energy_bits: int, hamiltonian: HamHam,
                             initial_state: np.ndarray = None, sigma: float = 0.) -> QuantumCircuit:

    trotter_step_circ = hamiltonian.trotter_step_circ
    qr_energy = QuantumRegister(num_energy_bits, name='w')
    qr_sys = QuantumRegister(num_qubits, name='sys')
    circ = QuantumCircuit(qr_energy, qr_sys, name="OFT")
    
    circ_to_analyze = QuantumCircuit(qr_energy, qr_sys, name="OFT/")
    circ_to_analyze.initialize(initial_state, qr_sys)
    
    # Energy before jump
    initial_statevector = Statevector(circ_to_analyze).data
    padded_rescaled_hamiltonian = np.kron(hamiltonian.qt.full(), np.eye(2**num_energy_bits))  # top-bottom = right-left
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
    op_circ = QuantumCircuit(1, name="A")
    op_circ.append(op, [0])
    circ.compose(op_circ, qr_sys[random_sys_qubit], inplace=True)
    print(f'Jump applied to {random_sys_qubit}th qubit')
    
    # For analysis
    circ_to_analyze.compose(circ, [*list(qr_energy), *list(qr_sys)], inplace=True)
    statevector = Statevector(circ_to_analyze).data
    energy_after_jump = statevector.conj().T @ padded_rescaled_hamiltonian @ statevector
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
    
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    num_qubits = 4
    num_energy_bits = 4
    sigma = 0.5
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()
    hamiltonian = HamHam(hamiltonian_matrix([X, X], [Y, Y], [Z, Z], coeffs=[1, 1, 1], num_qubits=num_qubits))
    hamiltonian.trotter_step_circ = trotter_step_heisenberg(num_qubits)
    spectrum = np.linalg.eigvalsh(hamiltonian.qt)
    print(f'Spectrum: {spectrum}')
    
    pauliX = Operator(Pauli('X'))
    
    # Bohr frequency
    
    # bohr_unit = smallest_bohr_freq(hamiltonian)
    # print(bohr_unit)
    hamiltonian.rescaling_factor, hamiltonian.shift = rescaling_and_shift_factors(hamiltonian.qt)
    hamiltonian.rescaled_qt = hamiltonian.qt / hamiltonian.rescaling_factor + \
        hamiltonian.shift * qt.qeye(hamiltonian.qt.shape[0])
    
    print(f'normalized spectrum {np.linalg.eigvalsh(hamiltonian.rescaled_qt)}')
    
    op_fourier_circ = operator_fourier_circuit(pauliX, num_qubits, num_energy_bits, hamiltonian,
                                               with_gauss=False, sigma=sigma)
    
    # full_state = Statevector(op_fourier_circ)
    # traced_over_qubits = [op_fourier_circ.qubits.index(qubit) for qubit in op_fourier_circ.qregs[1]]
    # omega_dm = partial_trace(full_state, traced_over_qubits)
    # print(omega_dm.probabilities([0, 1, 2, 3]))
    
    #* Run on simulator
    # cr_energy = ClassicalRegister(num_energy_bits, name='cw')
    # qr_oft = QuantumRegister(num_qubits + num_energy_bits, name='oft')
    # circ = QuantumCircuit(*op_fourier_circ.qregs, cr_energy)
    # circ.compose(op_fourier_circ, range(num_energy_bits + num_qubits), inplace=True)
    # circ.measure(range(num_energy_bits), cr_energy)
    # # print(circ)
    # tr_circ = transpile(circ, basis_gates=['cx', 'u3'], optimization_level=3)
    
    # simulator = Aer.get_backend('statevector_simulator')
    # shots = 1000
    # job = simulator.run(tr_circ, shots=shots)
    # counts = job.result().get_counts()
    # print(counts)
    
    # energy = stich_energy_bits_to_value(counts, shots=shots)
    # print(energy)