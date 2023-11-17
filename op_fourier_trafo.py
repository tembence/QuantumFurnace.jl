import numpy as np
import qutip as qt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit import Parameter

from tools.quantum import heisenber_hamiltonian
from tools.classical import get_smallest_bohr_freq, get_hamiltonian_matrix

def operator_fourier_circuit(op: Operator, num_energy_bits: int, hamiltonian: QuantumCircuit) -> QuantumCircuit:

    qr_op = QuantumRegister(op.num_qubits, name='op')
    qr_energy = QuantumRegister(num_energy_bits, name='w')
    
    




def prepare_gaussian_state(num_energy_bits: int) -> QuantumCircuit:
    """An early implementation for low number of energy register qubits. For more qubits we would use QSVT maybe."""
    ...
    

if __name__ == "__main__":
    num_qubits = 4
    num_trotter_steps = 4
    hamiltonian = heisenber_hamiltonian(num_qubits, num_trotter_steps)
    # print(hamiltonian)
    # Bohr frequency
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()
    hamiltonian_matrix = get_hamiltonian_matrix([X, X], [Y, Y], [Z, Z], [Z], num_qubits=num_qubits)
    smallest_bohr_freq = get_smallest_bohr_freq(hamiltonian_matrix)
    print(smallest_bohr_freq)
    
    
    pauliX = Operator(Pauli('X'))
    # op_fourier_circ = operator_fourier_circuit(pauliX, num_qubits, hamiltonian)