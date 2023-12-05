import numpy as np
import qutip as qt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import Statevector, partial_trace, random_statevector
from itertools import combinations
from time import time

# ----------------------------------------------- Matrix related functions ----------------------------------------------- #
def hamiltonian_matrix(*terms: list[qt.Qobj], num_qubits: int) -> np.ndarray:  #TODO: Add coefficients for terms
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


def shift_spectrum(hamiltonian: qt.Qobj) -> qt.Qobj:

    eigenenergies, _ = np.linalg.eigh(hamiltonian)
    smallest_eigval = np.round(eigenenergies[0], 5)
    largest_eigval = np.round(eigenenergies[-1], 5)
    
    # Shift spectrum to nonnegatives
    if smallest_eigval < 0:
        shift = abs(smallest_eigval)
    else:
        shift = 0
        
    shifted_hamiltonian = hamiltonian + shift * qt.qeye(hamiltonian.shape[0])
        
    return shifted_hamiltonian

def rescale_and_shift_spectrum(hamiltonian: qt.Qobj) -> qt.Qobj:
    """Rescale and shift to get spectrum in [0, 1]
    Note, it's only in this interval later after we set shift = shift / rescaling_factor
    """
    eigenenergies, _ = np.linalg.eigh(hamiltonian)
    smallest_eigval = np.round(eigenenergies[0], 5)
    largest_eigval = np.round(eigenenergies[-1], 5)
    
    # Rescale and shift spectrum [0, 1]
    if smallest_eigval < 0:
        rescaling_factor = (abs(smallest_eigval) + abs(largest_eigval))
        shift = abs(smallest_eigval)
    else:
        rescaling_factor = abs(largest_eigval)
        shift = 0
        
    shifted_rescaled_hamiltonian = (hamiltonian + shift * qt.qeye(hamiltonian.shape[0])) / rescaling_factor
        
    return shifted_rescaled_hamiltonian

def rescaling_and_shift_factors(hamiltonian: qt.Qobj) -> tuple[float, float]:
    """Rescale and shift to get spectrum in [0, 1]
    Note, it's only in this interval later after we set shift = shift / rescaling_factor
    """
    eigenenergies, _ = np.linalg.eigh(hamiltonian)
    smallest_eigval = np.round(eigenenergies[0], 5)
    largest_eigval = np.round(eigenenergies[-1], 5)
    
    # Rescale and shift spectrum [0, 1]
    if smallest_eigval < 0:
        rescaling_factor = (abs(smallest_eigval) + abs(largest_eigval))
        shift = abs(smallest_eigval)
    else:
        rescaling_factor = abs(largest_eigval)
        shift = 0
        
    return rescaling_factor, shift


# ----------------------------------------------- Energy related functions ----------------------------------------------- #
def smallest_bohr_freq(hamiltonian_matrix) -> float:
    """Get the smallest Bohr frequency \omega_0 for a given Hamiltonian.
    """
    eigvals = np.linalg.eigvalsh(hamiltonian_matrix)
    return np.min([eigvals[j] - eigvals[i] for i, j in combinations(range(len(eigvals)), 2)])


def energy_from_full_state(circ: QuantumCircuit, hamiltonian: qt.Qobj, subspace_qubits: list[int], qr_index: int):
    """Compute energy of a subsystem from full statevector of the circuit. 
    Subspace of interest is the one with `qr_index` in the `subspace_qubits` list (in Qiskit order)
    (Order between qutip tensor() and qiskit QRs are reversed) (*)
    """
    
    total_num_qubits = circ.num_qubits
    observable_list = []
    for q in range(len(subspace_qubits)):
        if q != qr_index:
            num_qubits_in_subspace = subspace_qubits[q]
            observable_list.append(qt.qeye(2**num_qubits_in_subspace))
        else:
            observable_list.append(qt.Qobj(hamiltonian))
    
    observable_list.reverse()
    full_observable = qt.tensor(observable_list)  # (*)
    full_observable.dims = [[2**total_num_qubits], [2**total_num_qubits]]
    full_circ_statevector = qt.Qobj(Statevector(circ).data)
    energy = qt.expect(full_observable, full_circ_statevector)
    
    return energy

def reduced_density_matrix(circ: QuantumCircuit, subspace_qubits: list[int], qr_indices: list[int]):
    """Compute partial trace of a statevector over not `qr_indices` 
    to get the reduced density matrix of subsystems `qr_indidces`.
    `subspace_qubits` list is in Qiskit circuit order, so is `qr_index`.
    """
    statevector = Statevector(circ).data  # 'abcd'
    full_density_matrix = np.einsum('i, j -> ij', statevector, statevector.conj())
    subspace_dims = [2**q for q in subspace_qubits]
    full_density_matrix = full_density_matrix.reshape(subspace_dims * 2)  # dims: '2, 8, 4, 2, 8, 4' - indices: 'abcdABCD'
    print(full_density_matrix.shape)

    lowercase_letters = 'abcdefghijklmnopqrstuvwxyz'
    uppercase_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    subsystem_indices = lowercase_letters[:len(subspace_qubits)]*2  # 'abcdabcd'
    indices_of_reduced_density_matrix = ''

    for qr_index in qr_indices:  # 'aBcDabcd'
        index_of_reduced_density_matrix = uppercase_letters[qr_index]
        indices_of_reduced_density_matrix += index_of_reduced_density_matrix
        subsystem_indices = subsystem_indices.replace(subsystem_indices[qr_index], index_of_reduced_density_matrix, 1)
        
    for qr_index in qr_indices:
        indices_of_reduced_density_matrix += lowercase_letters[qr_index]
    
    # 'aBcDabcd' -> 'BDbd' (summed over a and c)
    reduced_density_matrix = np.einsum(subsystem_indices + '->' + indices_of_reduced_density_matrix, full_density_matrix)
    reduced_density_matrix_dim = np.prod([subspace_dims[i] for i in qr_indices])
    
    return reduced_density_matrix.reshape((reduced_density_matrix_dim, reduced_density_matrix_dim))

def stich_energy_bits_to_value(counts: dict[str, int], shots: int) -> int:
    bitstrings = list(counts.keys())
    num_estimating_qubits = len(bitstrings[0])
    energy_values = [- int(bitstring[0])*2**(num_estimating_qubits - 1) + int(bitstring[1:], 2) for bitstring in bitstrings]

    omega_prime = np.sum([energy_values[i] * counts[bitstrings[i]] for i in range(len(bitstrings))]) / shots
    omega = - omega_prime * 2**(num_estimating_qubits) / 2*np.pi
    return omega_prime
    

if __name__ == '__main__':
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()

    num_qubits = 3
    H = hamiltonian_matrix([X, X], [Y, Y], [Z, Z], [Z], num_qubits=num_qubits)
    H_prime = rescale_and_shift_spectrum(H)
    print(np.linalg.eigvalsh(H_prime))
    
    qr0 = QuantumRegister(1, 'qr0')
    qr1 = QuantumRegister(2, 'qr1')
    qrsys = QuantumRegister(num_qubits, 'sys')
    qr3 = QuantumRegister(4, 'q3')

    circ = QuantumCircuit(qr0, qr1, qrsys, qr3)
    qr_index = circ.qregs.index(qrsys)

    # energy = energy_from_full_state(circ, H, subspace_qubits=[qr0.size, qr1.size, qrsys.size, qr3.size], qr_index=qr_index)

    # rand_state = random_statevector(2**np.sum([qr.size for qr in circ.qregs]))
    # circ.initialize(rand_state, range(np.sum([qr.size for qr in circ.qregs])))
    # rdm = reduced_density_matrix(circ, [1, 2, 3, 4], qr_indices=[1, 3])
    # rdm_qiskit = partial_trace(Statevector(circ), [0, 3, 4, 5])  #FIXME: QIskit and my reduced density matrix doesnt match up 
    # print(rdm)
    # rdm_qiskit = rdm_qiskit.data.reshape(rdm.shape)
    # print(np.isclose(rdm, rdm_qiskit))
    
    measurement_result = {'1010': 79, '0111': 63, '0110': 44, '1000': 45, '0000': 58, '1100': 59, '0100': 65, '0001': 69, '0011': 58, '1111': 69, '1101': 75, '0010': 57, '1110': 59, '1011': 65, '1001': 65, '0101': 70}
