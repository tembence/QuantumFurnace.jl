import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit import Parameter

def trotter_step_heisenberg(num_qubits: int, step_size: float = 0.25, nondegenerate = True) -> QuantumCircuit:
    """Parametrized trotter step circuit for 1D Heisenberg chain: H = XX + YY + ZZ
       It has a fixed step size, for which Trotter errors should be fine, thus for longer time evolution
       we just need to adjust the number of steps 
    """
    
    trott_hamiltonian = QuantumCircuit(num_qubits, name="H")
    
    theta = 2 * step_size
    # Periodic boundary conditions
    for i in range(num_qubits):
        if i != num_qubits - 1:
            trott_hamiltonian.rxx(theta, i, i + 1)
            trott_hamiltonian.ryy(theta, i, i + 1)
            trott_hamiltonian.rzz(theta, i, i + 1)
        if (i == num_qubits - 1) and (num_qubits > 2):
            trott_hamiltonian.rxx(theta, i, 0)
            trott_hamiltonian.ryy(theta, i, 0)
            trott_hamiltonian.rzz(theta, i, 0)
        if nondegenerate:
            interaction_strength = 3.5
            trott_hamiltonian.rz(interaction_strength * theta, 0)
            
    return trott_hamiltonian

def ham_evol(num_qubits: int, total_time: float, trotter_step: QuantumCircuit,
                       step_size: float = 0.25, nondegenerate = True) -> QuantumCircuit:
    """Time parametrized Hamiltonian evolution"""
    
    circ = QuantumCircuit(num_qubits, name="H")
    
    for i in range(int(np.ceil(total_time / step_size))):
        circ.compose(trotter_step, inplace=True)
    
    return circ

def qft_circ(num_qubits: int) -> QuantumCircuit:
    """QFT with QTSP convention: |t> -> sum exp(-iwt) |w>"""
    
    circ = QuantumCircuit(num_qubits, name="QFT")
    for j in reversed(range(num_qubits)):
        circ.h(j)
        num_entanglements = max(0, j)
        for k in reversed(range(j - num_entanglements, j)):
            lam = np.pi * (2.0 ** (k - j))
            circ.cp(lam, j, k)
                
    for i in range(num_qubits // 2):
        circ.swap(i, num_qubits - i - 1)
