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

from op_fourier_trafo import *
from boltzmann import *
from tools.classical import *
from tools.quantum import *


def liouv_step_circ(num_qubits: int, num_energy_bits: int, delta: float,
                    initial_state: Statevector) -> QuantumCircuit:
        
    #* --- Circuit
    qr_delta = QuantumRegister(1, name='delta')
    qr_boltzmann = QuantumRegister(1, name='boltz')
    qr_energy = QuantumRegister(num_energy_bits, name="w")
    qr_sys  = QuantumRegister(num_qubits, name="sys")

    circ = QuantumCircuit(qr_delta, qr_boltzmann, qr_energy, qr_sys)
    U_circ = QuantumCircuit(qr_boltzmann, qr_energy, qr_sys, name='U')
    U_dag_circ = QuantumCircuit(qr_boltzmann, qr_energy, qr_sys, name='U_dag')
        
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

    # --- Delta rotation (pg. 17 New QTSP)
    Y_angle = lambda theta: 2 * np.arcsin(np.sqrt(theta))
    delta_circ = QuantumCircuit(1, name='delta')
    delta_circ.ry(Y_angle(delta), 0)
    C_delta_circ = delta_circ.control(1)
    print('Delta')

    # --- U dagger
    inverse_boltzmann_circ = inverse_lookup_table_boltzmann(num_energy_bits)
    U_dag_circ.compose(inverse_boltzmann_circ, [qr_boltzmann[0], *list(qr_energy)], inplace=True)

    inverse_oft_circ = inverse_operator_fourier_transform(jump_op, num_qubits, num_energy_bits, hamiltonian)
    U_dag_circ.compose(inverse_oft_circ, [*list(qr_energy), *list(qr_sys)], inplace=True)

    CU_dag_circ = U_dag_circ.control(1)
    print('CU dag')

    # --- Create Liouvillian step
    circ.compose(U_circ, [qr_boltzmann[0], *list(qr_energy), *list(qr_sys)], inplace=True)
    state_after_U = Statevector(circ)

    circ.x(qr_boltzmann[0])
    circ.compose(C_delta_circ, [qr_boltzmann[0], qr_delta[0]], inplace=True)
    circ.x(qr_boltzmann[0])

    circ.x(qr_delta[0])
    circ.compose(CU_dag_circ, [qr_delta[0], qr_boltzmann[0], *list(qr_energy), *list(qr_sys)], inplace=True)
    circ.x(qr_delta[0])

    state_before_measure = Statevector(circ)

    with open(f'data/state_after_U_n{num_qubits}k{num_energy_bits}.pkl', 'wb') as f:
        pickle.dump(state_after_U, f)

    with open(f'data/state_before_measure_n{num_qubits}k{num_energy_bits}.pkl', 'wb') as f:
        pickle.dump(state_before_measure, f)
        
    return circ