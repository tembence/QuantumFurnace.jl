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

#TODO: Change boltzmann weight to 1-boltzmann weight to make it compatible with QTSP paper
def look_up_table_boltzmann(num_energy_bits: int, beta: float = 1) -> QuantumCircuit:
    """2^(k - 1) many (k - 1)-Toffolis"""
    # If sign bit is negative then we accept the step by an X gate
    # else: we need the value in the rest of the qr and based on that apply the appropriate rotation
    qr_energy = QuantumRegister(num_energy_bits, name='w')
    qr_boltzmann = QuantumRegister(1, name='boltz')
    circ = QuantumCircuit(qr_boltzmann, qr_energy, name="boltz")
    circ.x(qr_boltzmann[0])  # If 1, omega <=0, accept otherwise reject
    circ.x(qr_energy[-1])
    
    boltzmann_weight = lambda omega: np.exp(-beta * omega / 2)  #! /2 made it work.
    # Without MSB (sign):
    bitstrings = [bin(i)[2:].zfill(num_energy_bits - 1) for i in range(2**(num_energy_bits - 1))]
    bitstrings = bitstrings[1:]  # All 0 state is already default accepting
    for bitstring in bitstrings:
        omega = int(bitstring, 2) / 2**(num_energy_bits - 1)
        boltzmann_angle = - 2 * np.arccos(np.sqrt(boltzmann_weight(omega))) #* Angle!

        # Create W_{bitstring}
        W = QuantumCircuit(qr_boltzmann, name="W")
        # W.x(qr_boltzmann[0])  # Normally W = X RY, but we need to undo the default X: X X RY = RY     
        # W.ry(boltzmann_angle, qr_boltzmann[0])
        # W.x(qr_boltzmann[0]) 
        W.ry(boltzmann_angle, qr_boltzmann[0])  # After commuting X thorugh, angle changes to negative, XX = I
        cW = W.control(num_energy_bits, label='0'+bitstring)

        bitstring_qiskit_ordered = bitstring[::-1]  # Reverse bitstring to match with qubit order in Qiskit
        for i, bit in enumerate(bitstring_qiskit_ordered):
            if bit == '0':
                circ.x(qr_energy[i])
                
        circ.compose(cW, [*list(qr_energy), qr_boltzmann[0]], inplace=True)

        for i, bit in enumerate(bitstring_qiskit_ordered):  # Undo it before next Toffoli
            if bit == '0':
                circ.x(qr_energy[i])
        #?Do we need RY dagger like in old Metro? And why?
        
    circ.x(qr_energy[-1])
    
                
    return circ
    

    