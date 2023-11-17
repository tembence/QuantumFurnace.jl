import numpy as np
import qutip as qt
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from itertools import combinations

def get_hamiltonian_matrix(*terms: list[qt.Qobj], num_qubits: int) -> np.ndarray:  #TODO: Add coefficients for terms
    """Gets the Hamiltonian as a Qobj. Assumes periodic boundaries.
    Args:
        terms: list of Qobjs, each Qobj is a single body term in the list
                building a many-body term together, e.g. [X, Y, Z] -> X^Y^Z"""
    
    hamiltonian = np.zeros((2**num_qubits, 2**num_qubits))
    for term in terms:
        for q in range(num_qubits):
            hamiltonian += pad_term(term, num_qubits, q).data
            
    return hamiltonian

def pad_term(terms: list[qt.Qobj], num_qubits: int, position: int) -> qt.Qobj:
    """ Pads a many-body term with identity operators for the rest of the system.
    Assumes periodic boundary conditions."""
    
    term_size = len(terms)
    end_position = (position + term_size - 1)
    I = qt.qeye(2)
    padded_tensor_list = [I for _ in range(num_qubits)]
    term_indices = [i % num_qubits for i in range(position, end_position + 1)]
    
    for i, term_index in enumerate(term_indices):
        padded_tensor_list[term_index] = terms[i]
    
    return qt.tensor(padded_tensor_list)

def get_smallest_bohr_freq(hamiltonian_matrix) -> float:
    """Get the smallest Bohr frequency \omega_0 for a given Hamiltonian.
    """

    eigvals = np.linalg.eigvalsh(hamiltonian_matrix)

    return np.min([eigvals[j] - eigvals[i] for i, j in combinations(range(len(eigvals)), 2)])
