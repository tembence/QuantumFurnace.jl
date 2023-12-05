import numpy as np
from scipy.linalg import logm
import qutip as qt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit import Parameter
from classical import *

class Hamiltonian:  #TODO: Write a trotter circuit generator for an input qt.Qobj Hamiltonian
    def __init__(self, hamiltonian_qt: qt.Qobj):
        self.hamiltonian_qt = hamiltonian_qt
        self.shift: float = 0.
        self.rescaling_factor: float = 1.
        self.rescaled_hamiltonian_qt = None
        self.trotter_step_circ: QuantumCircuit = None
        self.trotter_circ: QuantumCircuit = None

def trotter_step_heisenberg(num_qubits: int, step_size: float = 0.25, nondegenerate = True) -> QuantumCircuit:
    """Parametrized trotter step circuit for 1D Heisenberg chain: H = XX + YY + ZZ
       It has a fixed step size, for which Trotter errors should be fine, thus for longer time evolution
       we just need to adjust the number of steps 
    """
    
    trott_hamiltonian = QuantumCircuit(num_qubits, name="H")
    
    theta = 2 * step_size  # Qiskit convention, rxx(2x) = exp(...x...)
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
        # if nondegenerate:
        #     interaction_strength = 3.5
        #     trott_hamiltonian.rz(interaction_strength * theta, 0)
            
    return trott_hamiltonian

def ham_evol(num_qubits: int, total_time: float, trotter_step: QuantumCircuit,
                       step_size: float = 0.25, shift: float = None, rescaling_factor: float = None) -> QuantumCircuit:
    """Time parametrized Hamiltonian evolution"""  #TODO: Make it actually parametrized, so doesnt have to be regenerated
    
    circ = QuantumCircuit(num_qubits, name="H")
    # Shift and rescale circuit Hamiltonian, spec(H) in [0, 1]
    circ.global_phase = - shift
    total_time /= rescaling_factor
    
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
    
def trott_ham_spectrum_from_circ(trott_hamiltonian_circ: QuantumCircuit, total_time: float) -> tuple[list, list]:
    """U = exp(-i T H)
    Matrix logarithm is bijective only if the spectrum of H is at least in (-pi, pi)
    """
    
    trotterized_evolution_unitary = Operator(trott_hamiltonian_circ).data
    H = logm(trotterized_evolution_unitary)  # log U = i H T
    trotterized_ham_from_circuit = H / (1j * total_time)
    
    return np.linalg.eigvalsh(trotterized_ham_from_circuit)


if __name__ == "__main__":
    
    num_qubits = 4
    total_time = 2 * np.pi
    step_size = 0.25
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()
    hamiltonian = hamiltonian_matrix([X, X], [Y, Y], [Z, Z], num_qubits=num_qubits)
    print(np.linalg.eigvalsh(hamiltonian))
    rescaling_factor, shift = rescaling_and_shift_factors(hamiltonian)
    trotter_step = trotter_step_heisenberg(num_qubits)
    
    circ = QuantumCircuit(num_qubits, name="H")
    # Shift and rescale circuit Hamiltonian, spec(H) in [0, 1]
    circ.global_phase = - shift
    total_time /= rescaling_factor

    for i in range(int(np.ceil(total_time / step_size))):
        circ.compose(trotter_step, inplace=True)

    spectrum_from_circ = trott_ham_spectrum_from_circ(circ, total_time)
    print(spectrum_from_circ)
    
    # print(circ)