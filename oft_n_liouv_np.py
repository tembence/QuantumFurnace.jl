
import numpy as np
import qutip as qt
from scipy.linalg import logm, expm
import time
from copy import deepcopy
from scipy.sparse import issparse

import sys
sys.path.append('/Users/bence/code/liouvillian_metro/')

from tools.classical import HamHam
from tools.classical import find_ideal_heisenberg

def oft(jump_op: np.ndarray, phase: float, num_labels: int, sigma: float,
        hamiltonian: HamHam = None, trotter: np.ndarray = None) -> np.ndarray:
    """Input is PHASE not ENERGY, energy = 2 pi phase / N"""
    
    # In binary order
    N_labels = np.arange(num_labels / 2, dtype=int)
    N_labels_neg = np.arange(- num_labels / 2, 0, dtype=int)
    N_labels = np.concatenate((N_labels, N_labels_neg))
    time_labels = N_labels

    gauss = lambda t: np.exp(-t**2 / (4 * sigma**2))
    gauss_values = gauss(time_labels)
    normalized_gauss_values_bin_order = gauss_values / np.sqrt(np.sum(gauss_values**2))
    
    phase_factors = np.exp(- 1j * 2 * np.pi * phase * time_labels / num_labels)
    
    if hamiltonian is None and trotter is not None:  #* Trotterized
        time_evolution = lambda n: np.linalg.matrix_power(trotter, n) #TODO: add numpy einsum option
    elif hamiltonian is not None and trotter is None:  #* Exact
        # Prediagonalized, preexponentiated Hamiltonian for speed
        diag_elements_for_all_times = np.exp(1j * time_labels[:, np.newaxis] * hamiltonian.spectrum)
        
        oft_op = np.einsum('i, i, ja, ia, ak, km, mb, ib, bn  -> jn', 
                        normalized_gauss_values_bin_order, phase_factors, 
                        hamiltonian.eigvecs, diag_elements_for_all_times, hamiltonian. eigvecs.conj().T, 
                        jump_op, 
                        hamiltonian.eigvecs, diag_elements_for_all_times.conj(), hamiltonian.eigvecs.conj().T, 
                        optimize=True) / np.sqrt(num_labels)
    return oft_op

def oft_the_bohr_way(jump_op: np.ndarray, phase: float, N: int, sigma: float, 
                  hamiltonian: HamHam = None) -> np.ndarray:
    """Input is PHASE not ENERGY, energy = 2 pi phase / N
    No Trotter for now"""
    
    # In binary order
    # N_labels = np.arange(N / 2, dtype=int)
    # N_labels_neg = np.arange(- N / 2, 0, dtype=int)
    # N_labels = np.concatenate((N_labels, N_labels_neg))
    # time_labels = N_labels
    oft_precision = int(np.ceil(np.abs(np.log10(N**(-1)))))
    
    bohr_freqs = hamiltonian.spectrum[:, np.newaxis] - hamiltonian.spectrum
    
    jump_in_eigenbasis = hamiltonian.eigvecs.conj().T @ jump_op @ hamiltonian.eigvecs
    jump_nonzero_indices = np.nonzero(jump_in_eigenbasis)
    nonzero_index_pairs = list(zip(*jump_nonzero_indices))
    
    # For now we take energy diffs close to each other as the same energy diff (cuts off a ton of sum terms)
    bohr_freqs_of_jump = np.round(bohr_freqs[jump_nonzero_indices], oft_precision+2)
    
    
    
    
    uniqe_freqs = np.unique(bohr_freqs_of_jump, return_index=True)
    same_freq_indices_in_energy_list = [np.nonzero(bohr_freqs_of_jump == bohr_freqs_of_jump[uniqe_freqs[1][i]]) 
                                    for i in range(len(uniqe_freqs[0]))]

def get_truncated_energy_axis(jump_op: np.ndarray, initial_state: np.ndarray, hamiltonian: HamHam, 
                              N: int, sigma: float) -> np.ndarray:
    
    phase_labels = np.arange(N / 2, dtype=int)
    phase_labels_neg = np.arange(- N / 2, 0, dtype=int)
    phase_labels = np.concatenate((phase_labels, phase_labels_neg))
    energy_labels = 2 * np.pi * phase_labels / N
    
    oft_precision = int(np.ceil(np.abs(np.log10(N**(-1)))))

    jump_op_eigen = hamiltonian.eigvecs.conj().T @ jump_op @ hamiltonian.eigvecs
    jump_nonzero_indices = np.nonzero(jump_op_eigen)
    bohr_freqs = hamiltonian.spectrum[:, np.newaxis] - hamiltonian.spectrum
    bohr_freqs_of_jump = np.round(bohr_freqs[jump_nonzero_indices], oft_precision+3)
    uniqe_freqs = np.unique(bohr_freqs_of_jump)
    
    
    
    unique_closest_energy_labels = int(uniqe_freqs * N / (2 * np.pi))[:, np.newaxis]
    
    # distance from 0 of gaussian that has amplitudes above epsilon 
    ft_gauss = lambda energy: np.exp(-sigma**2 * energy**2)
    ft_gauss_amplitudes = np.array([ft_gauss(energy) for energy in energy_labels])  # Binary ordered
    ft_gauss_normalization = np.linalg.norm(ft_gauss_amplitudes)
    eps = 1e-4
    cut_off_dist = np.sqrt(-np.log(ft_gauss_normalization * eps) / sigma**2)  # in energy units
    cut_off_num_labels = int(np.ceil(cut_off_dist * N / (2 * np.pi)))  # to left and right
    
    # Truncated labels around unique energy labels
    truncated_regions = np.concatenate((unique_closest_energy_labels - cut_off_num_labels, 
                                        unique_closest_energy_labels + cut_off_num_labels), axis=1)
    
    

def liouvillian_step(initial_dm: np.ndarray, beta: float, delta: float, jump_op: np.ndarray,
                     N: int, sigma: float, hamiltonian: HamHam) -> np.ndarray:
    
    
    N_labels = np.arange(N / 2, dtype=int)
    N_labels_neg = np.arange(- N / 2, 0, dtype=int)
    N_labels = np.concatenate((N_labels, N_labels_neg))
    energy_labels = 2 * np.pi * N_labels / N
    energy_diff_bounds = [-0.45, 0.45] #! Kinda confused about this 
    truncated_phase_indices = np.where((energy_labels >= energy_diff_bounds[0]) & (energy_labels <= energy_diff_bounds[1]))[0]
    
    boltzmann = lambda beta, energy: np.min([1, np.exp(-beta * energy)]) #! change it to new paper's way
    boltzmann_vectorized = np.vectorize(boltzmann)
    #* Exact
    evolved_dm = deepcopy(initial_dm)
    
    energy_factors = 2 * np.pi * N_labels / N
    boltzmann_factors = boltzmann_vectorized(beta, energy_factors)
    
    for phase in truncated_phase_indices:
        energy = 2 * np.pi * phase / N
        oft_op = oft(jump_op, phase, N, sigma, hamiltonian=hamiltonian)
        oft_op_dag = oft_op.conj().T
        evolved_dm += delta * boltzmann(beta, energy)*(-0.5 * oft_op_dag @ oft_op @ initial_dm
                                        -0.5 * initial_dm @ oft_op_dag @ oft_op
                                        + oft_op @ initial_dm @ oft_op_dag)
    
    return evolved_dm / np.trace(evolved_dm)

def liouvillian_evol():
    num_liouv_steps = int(np.ceil(time / delta))
    
    
if __name__ == '__main__':
    
    np.random.seed(667)
    num_qubits = 8
    num_energy_bits = 9
    sigma = 5
    bohr_bound = 0
    eps = 0.1
    
    hamiltonian = find_ideal_heisenberg(num_qubits, bohr_bound, eps, signed=False, for_oft=True)

    N = 2**num_energy_bits
    N_labels = np.arange(N / 2, dtype=int)
    N_labels_neg = np.arange(- N / 2, 0, dtype=int)
    N_labels = np.concatenate((N_labels, N_labels_neg))
    site_list = [qt.qeye(2) for _ in range(num_qubits)]
    x_jump_ops = []
    for q in range(num_qubits):
        site_list[q] = qt.sigmax()
        x_jump_ops.append(qt.tensor(site_list).full())
    
    boltzmann = lambda beta, energy: np.min([1, np.exp(-beta * energy)])

    rand_jump_index = np.random.randint(0, len(x_jump_ops))
    jump_op = x_jump_ops[rand_jump_index]
    print(f'Jump on site {rand_jump_index}')
    
    phase = 0
    t0 = time.time()
    oft_op = oft(jump_op, phase, N, sigma, hamiltonian=hamiltonian)
    t1 = time.time()
    print(f'oft time: {t1 - t0}')
    
    from scipy.sparse import csr_matrix
    sp_oft_op = csr_matrix(oft_op)
    print(isinstance(sp_oft_op, csr_matrix))