import numpy as np
import qutip as qt
from scipy.linalg import logm, expm
from qiskit.quantum_info import Operator, state_fidelity, partial_trace, DensityMatrix
from qiskit import QuantumCircuit
from qiskit_aer import StatevectorSimulator
from qiskit import Aer
from qiskit.circuit.library import QFT

import pickle
from time import time

from op_fourier_trafo import *
from boltzmann import *
from liouv_step import *
from tools.classical import *
from tools.quantum import *


def sigmam_LCU() -> QuantumCircuit:
    """sigmam = 0.5 * (X - iY)
    When you add it to the main circuit, compose it with [sys, b] order.
    """
    
    term_coeffs = np.array([0.5, -1j*0.5])
    sqrt_term_coeffs = np.sqrt(term_coeffs / np.sum(np.abs(term_coeffs)))
    
    B = np.array([[sqrt_term_coeffs[0], sqrt_term_coeffs[1].conj()], [sqrt_term_coeffs[1], -sqrt_term_coeffs[0]]])
    B_op = Operator(B)
    num_qubits = 1
    num_block_encoding_qubits = 1
    qr_sys = QuantumRegister(num_qubits, 'sys')
    qr_b = QuantumRegister(num_block_encoding_qubits, 'b')

    #* LCU
    circ = QuantumCircuit(qr_sys, qr_b)  #! Order matters in a confusing way
    # Prep
    circ.append(B_op, [qr_b[0]])

    # Select
    circ.x(qr_b)
    circ.cx(qr_b, qr_sys)
    circ.x(qr_b)

    circ.cy(qr_b, qr_sys)

    # Prep dagger
    circ.append(B_op.transpose(), [qr_b[0]])
    
    return circ