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

def look_up_table_boltzmann(num_energy_bits: int, beta: float = 1) -> QuantumCircuit:
    """2^(k - 1) many (k - 1)-Toffolis"""
    # If sign bit is negative then we accept the step by an X gate
    # else: we need the value in the rest of the qr and based on that apply the appropriate rotation
    qr_energy = QuantumRegister(num_energy_bits, name='w')
    qr_boltzmann = QuantumRegister(1, name='boltz')
    circ = QuantumCircuit(qr_energy, qr_boltzmann, name="boltz")
    circ.cx(qr_energy[-1], qr_boltzmann[0])  # If 1, omega < 0, accept
    
    bitstrings = [bin(i)[2:].zfill(num_energy_bits - 1) for i in range(2**(num_energy_bits - 1))]
    for bitstring in bitstrings:
        omega = int(bitstring, 2) / 2**(num_energy_bits - 1) #! *N or 2pi, write it up! #TODO:
        boltzmann_angle = lambda omega: np.arccos(np.exp(-beta * omega))
        
        W = QuantumCircuit(qr_boltzmann)
        W.x(qr_boltzmann[0])  #! W has an X in it, but we have a cx on here too due to sign qubit which we need to cancel at each W, check! #TODO:
        W.ry(boltzmann_angle(omega), qr_boltzmann[0])
        cW = W.control(num_energy_bits - 1)

        for i, bit in enumerate(bitstring):  #! This is backwards, remember qubit order in Qiskit #TODO:
            if bit == '1':
                circ.x(qr_energy[i])
                
        circ.compose(cW, *list(qr_energy)[:num_energy_bits - 1], qr_boltzmann[0], inplace=True)

        for i, bit in enumerate(bitstring):
            if bit == '1':
                circ.x(qr_energy[i])
                
    return circ
    

    