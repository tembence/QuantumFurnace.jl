import numpy as np
import qutip as qt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import Statevector, partial_trace, random_statevector
from itertools import combinations
from time import time
from scipy.linalg import expm

# ----------------------------------------------- Matrix related functions ----------------------------------------------- #
class HamHam:  #TODO: Write a qiskit trotter circuit generator for an input qt.Qobj Hamiltonian
    def __init__(self, hamiltonian_qt: qt.Qobj, shift: float, rescaling_factor: float, 
                 rescaled_coeffs: list[float] = None):
        self.qt = hamiltonian_qt
        self.shift: float = shift
        self.rescaling_factor: float = rescaling_factor
        self.spectrum, self.eigenstates = np.linalg.eigh(self.qt.full())
        self.rescaled_coeffs = rescaled_coeffs
        self.trotter_step_circ: QuantumCircuit = None
        self.inverse_trotter_step_circ: QuantumCircuit = None

#FIXME: A bit off sometimes, the centering of the spectrum
def find_ideal_heisenberg(num_qubits: int, bohr_bound: float, eps: float,
                          signed: bool = True, for_oft: bool = True) -> HamHam:
    """Find a Heisenberg Hamiltonian with ideal spectrum for QPE, by which we mean
    that the spectrum is not degenerate and the Bohr frequencies are not shorter than the `bohr_bound`.
    And also that the spec(H) is in [0, 0.5 - eps/2] for the QPE in OFT. 
    For normal signed QPE it gives [-0.5 + eps, 0.5 - eps]
    
    Args:
        num_qubits (int): Number of system qubits.
        bohr_bound (float): Smallest Bohr frequency allowed for the spectrum.
        eps (float): Distance from the spectrum bounds, to avoid overlap of the boundary eigenstates in measurements.
    """
    
    if for_oft == True and signed == True:
        raise ValueError("For OFT we want unsigned spectrum")
    
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()

    # Find ideal spectrum
    coeff_lower_bound = 0
    coeff_xx = np.arange(1, coeff_lower_bound, -0.1)
    coeff_yy = np.arange(1, coeff_lower_bound, -0.1)
    coeff_zz = np.arange(1, coeff_lower_bound, -0.1)
    coeff_z = np.arange(1, coeff_lower_bound, -0.1)
    coeff_mesh = np.array(np.meshgrid(coeff_xx, coeff_yy, coeff_zz, coeff_z)).T.reshape(-1, 4)
    
    found_ideal_spectrum = False
    for coeffs in coeff_mesh:
        hamiltonian_qt = hamiltonian_matrix([X, X], [Y, Y], [Z, Z], coeffs=coeffs, num_qubits=num_qubits, symbreak_term=[Z])
        rescaling_factor, shift = rescaling_and_shift_factors(hamiltonian_qt, eps=eps, signed=signed)
        if for_oft:
            rescaling_factor *= 2 # [0.5 - eps / 2] 
            shift /= 2
        
        rescaled_hamiltonian_qt = hamiltonian_qt / rescaling_factor + shift * qt.qeye(hamiltonian_qt.shape[0])
        rescaled_spectrum = np.linalg.eigvalsh(rescaled_hamiltonian_qt)
        
        # Accept coeff only if all bohr freuquencies are not shorter than the bohr_bound
        for eigval_i in rescaled_spectrum:
            spec_without_eigval_i = np.delete(rescaled_spectrum, np.where(rescaled_spectrum == eigval_i))
            if np.any(np.abs(spec_without_eigval_i - eigval_i) < bohr_bound):
                break
        else:
            found_ideal_spectrum = True
            coeffs_ideal_spec = coeffs
            break

    if not found_ideal_spectrum:
        print("No ideal spectrum found")
        
    exact_spec = np.linalg.eigvalsh(hamiltonian_qt)
    print('Original spectrum: ', np.round(exact_spec, 4))
    print("Ideal spectrum: ", np.round(rescaled_spectrum, 4))
    print("Nonrescaled coefficients: ", coeffs_ideal_spec)
    rescaled_coeffs = coeffs_ideal_spec / rescaling_factor
    print('Rescaled coefficients: ', rescaled_coeffs)
    # print(f'Rescaling factor {rescaling_factor}, shift {shift}')

    #* Hamiltonian
    hamiltonian = HamHam(rescaled_hamiltonian_qt, shift=shift, rescaling_factor=rescaling_factor, 
                        rescaled_coeffs=rescaled_coeffs)
    
    return hamiltonian

        
def hamiltonian_matrix(*terms: list[qt.Qobj], coeffs: list[float], num_qubits: int, 
                       symbreak_term: list[qt.Qobj] = []) -> np.ndarray:
    """Gets the Hamiltonian as a Qobj. Assumes periodic boundaries.
    Args:
        terms: list of Qobjs, each Qobj is a single body term in the list
                building a many-body term together, e.g. [X, Y, Z] -> X^Y^Z
        coeffs: list of coefficients for each term in `terms` and the last one is for the `symbreak_term` if
                we want to break the symmetry.
    """
    
    if (len(terms) + len(symbreak_term)) != len(coeffs):
        raise ValueError("Number of terms and coefficients must match.")
    
    hamiltonian = np.zeros((2**num_qubits, 2**num_qubits))
    for coeff_i, term in enumerate(terms):
        for q in range(num_qubits):
            hamiltonian += coeffs[coeff_i] * pad_term(term, num_qubits, q).data
    
    # Add symmetry breaking term (breaks translation symmetry but also spin flip sym if chosen well) 
    # -> makes spectrum unique
    if (len(symbreak_term) != 0) and (num_qubits != 2):
        for q in range(1, num_qubits - 1):
            # print(f'Applied sym breaking term onto qubit {q}')
            hamiltonian += coeffs[-1] * pad_term(symbreak_term, num_qubits, q).data
            
    return hamiltonian


def pad_term(terms: list[qt.Qobj], num_qubits: int, position: int) -> qt.Qobj:
    """ Pads a many-body term with identity operators for the rest of the system.
    Assumes periodic boundary conditions.
    Position: is the index of the first qubit of the term. (Qiskit order)
    QISKIT AND QUTIP ORDER ARE REVERSED ! QISKIT 0 = LSB (TOP TO BOTTOM), QUTIP 0 = MSB (LEFT TO RIGHT)
    """
    
    term_size = len(terms)
    end_position = (position + term_size - 1)
    I = qt.qeye(2)
    padded_tensor_list = [I for _ in range(num_qubits)]
    term_indices = [i % num_qubits for i in range(position, end_position + 1)]
    
    for i, term_index in enumerate(term_indices):
        padded_tensor_list[term_index] = terms[i]
    
    padded_tensor_list.reverse()  #!
    
    return qt.tensor(padded_tensor_list)

def trotter_heisenberg_qutip(num_qubits: int, step_size: float, num_trotter_steps: int, coeffs: list[float],
                             shift: float = 0, symbreak: bool = True) -> qt.Qobj:
    
    if symbreak == True and len(coeffs) != 4:
        raise ValueError("If symbreak is True, then 3+1 coefficients are needed.")
        
    trott_hamiltonian_qt = qt.qeye(2**num_qubits)
    trott_step_qt = qt.qeye(2**num_qubits)
    
    if shift != 0:
        trott_hamiltonian_qt = trott_hamiltonian_qt * qt.Qobj(expm(1j * shift * np.eye(2**num_qubits)))
    
    # End of a product is the beginning of the operator chain applied.
    for i in range(num_qubits):
        print(f'eXX, eYY, eZZ for qubit {i, (i+1)%num_qubits}')
        XX = pad_term([qt.sigmax(), qt.sigmax()], num_qubits, i)
        YY = pad_term([qt.sigmay(), qt.sigmay()], num_qubits, i)
        ZZ = pad_term([qt.sigmaz(), qt.sigmaz()], num_qubits, i)
        eXX = expm(1j * coeffs[0] * step_size * XX.full())
        eYY = expm(1j * coeffs[1] * step_size * YY.full())
        eZZ = expm(1j * coeffs[2] * step_size * ZZ.full())
        if (i in range(1, num_qubits - 1)) and (symbreak == True):
            # print(f'eZ for qubit {i}')
            Z = pad_term([qt.sigmaz()], num_qubits, i)
            eZ = expm(1j * coeffs[3] * step_size * Z.full())
            trott_step_qt = qt.Qobj(eZ) * qt.Qobj(eZZ) * qt.Qobj(eYY) * qt.Qobj(eXX) * trott_step_qt
        else:
            trott_step_qt = qt.Qobj(eZZ) * qt.Qobj(eYY) * qt.Qobj(eXX) * trott_step_qt
        
    for _ in range(num_trotter_steps):
        # print(f'Applied Trotter step {_}')
        trott_hamiltonian_qt =  trott_step_qt * trott_hamiltonian_qt
    
    return trott_hamiltonian_qt

def shift_spectrum(hamiltonian: qt.Qobj) -> qt.Qobj:

    eigenenergies, _ = np.linalg.eigh(hamiltonian)
    smallest_eigval = np.round(eigenenergies[0], 5)
    
    # Shift spectrum to nonnegatives
    if smallest_eigval < 0:
        shift = abs(smallest_eigval)
    else:
        shift = 0
        
    shifted_hamiltonian = hamiltonian + shift * qt.qeye(hamiltonian.shape[0])
        
    return shifted_hamiltonian

def rescaling_and_shift_factors(hamiltonian: qt.Qobj, eps: float = 0, signed: bool=False) -> tuple[float, float]:
    """Rescale and shift to get spectrum in [0, 1]
    """
    eigenenergies = np.linalg.eigvalsh(hamiltonian)
    smallest_eigval = np.round(eigenenergies[0])
    largest_eigval = np.round(eigenenergies[-1])
    
# Rescaling factor and shift for [-0.5, 0.5]
    rescaling_factor = largest_eigval - smallest_eigval
    if eps != 0:
        rescaling_factor /= (1 - 2 * eps) # [-0.5 + eps, 0.5 - eps]
    # Centre spectrum around 0:
    shift = -(largest_eigval + smallest_eigval) / (2 * rescaling_factor)
    
    if signed == False:  # shift to [0, 1]
        shift += 0.5
        
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

#* BitHandler
class BitHandler:
    """Handles qiskit classical bitstrings for multiple measured registers in.
       Order matters! Top register is the last bits in the bitstring."""

    def __init__(self, classical_registers: list[ClassicalRegister]):
        self.classical_registers = classical_registers
        self.measured_counts = {}
        self.measured_bitstring = ''
        self.bitstring_of_creg = {register: '' for register in classical_registers}
    
    def get_counts_for_creg(self, cr: ClassicalRegister) -> dict[str, int]:
        counts_for_cr = {}
        for bits in self.measured_counts.keys():
            self.measured_bitstring = bits
            val = self.measured_counts[bits]
            bitstring_for_cr = self.get_bitstring_for_creg(cr)
            if bitstring_for_cr in counts_for_cr.keys():
                counts_for_cr[bitstring_for_cr] += val
            else:
                counts_for_cr[bitstring_for_cr] = val
            
        return counts_for_cr
        
    def get_bitstring_for_creg(self, cr: ClassicalRegister) -> str:
        """Slices into the whole bitstring to get the wanted classical register bits"""
        
        # Remove spaces in qiskit bitstring
        self.measured_bitstring = self.measured_bitstring.replace(' ', '')
        
        cr_index = self.classical_registers.index(cr)
        if cr_index == 0:
            wanted_bits = self.measured_bitstring[::-1][:cr.size][::-1]  # Ugly as hell but for a few bits it's fine
            return wanted_bits
        
        num_of_previous_bits = sum([cr.size for cr in self.classical_registers[:cr_index]])
        wanted_bits = self.measured_bitstring[::-1][num_of_previous_bits:num_of_previous_bits + cr.size][::-1]
        
        self.bitstring_of_creg[cr] = wanted_bits
        
        return wanted_bits
    
    def get_counts_with_condition(self, cond_creg: ClassicalRegister, cond_val: str) -> dict[str, int]:
        """E.g. successful jump means b ancilla is zero, and we wanna find the bitstrings in this case"""
        
        if cond_creg.size != len(cond_val):
            raise ValueError("Condition register size and condition value length must match.")
        
        counts_with_condition = {}
        for bitstring in self.measured_counts.keys():
            self.measured_bitstring = bitstring
            if self.get_bitstring_for_creg(cond_creg) == cond_val:
                counts_with_condition[bitstring] = self.measured_counts[bitstring]
        
        return counts_with_condition
        




if __name__ == '__main__':
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()

    num_qubits = 3
    H = hamiltonian_matrix([X, X], [Y, Y], [Z, Z], [Z], num_qubits=num_qubits)
    
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
    # rdm_qiskit = partial_trace(Statevector(circ), [0, 3, 4, 5])  
    # #FIXME: QIskit and my reduced density matrix doesnt match up 
    # print(rdm)
    # rdm_qiskit = rdm_qiskit.data.reshape(rdm.shape)
    # print(np.isclose(rdm, rdm_qiskit))
    
    measurement_result = {'1010': 79, '0111': 63, '0110': 44, '1000': 45, 
                          '0000': 58, '1100': 59, '0100': 65, '0001': 69, 
                          '0011': 58, '1111': 69, '1101': 75, '0010': 57, 
                          '1110': 59, '1011': 65, '1001': 65, '0101': 70}
