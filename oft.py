
import numpy as np
import qutip as qt
from scipy.linalg import logm, expm

import sys
sys.path.append('/Users/bence/code/liouvillian_metro/')

def oft(jump_op: np.ndarray, phase: float, num_labels: int, sigma: float,
        hamiltonian: np.ndarray = None, trotter: np.ndarray = None) -> np.ndarray:
    """Input is PHASE not ENERGY, energy = 2 pi phase / N"""
    
    #! In binary order
    N_labels = np.arange(num_labels / 2, dtype=int)
    N_labels_neg = np.arange(- num_labels / 2, 0, dtype=int)
    N_labels = np.concatenate((N_labels, N_labels_neg))
    time_labels = N_labels

    gauss = lambda t: np.exp(-t**2 / (4 * sigma**2))
    gauss_values = gauss(time_labels)
    normalized_gauss_values_bin_order = gauss_values / np.sqrt(np.sum(gauss_values**2))
    
    if hamiltonian is None and trotter is not None:  #* Trotterized
        time_evolution = lambda n: np.linalg.matrix_power(trotter, n)
    elif hamiltonian is not None and trotter is None:  #* Exact
        time_evolution = lambda t: expm(1j * t * hamiltonian)
        
    oft_op = np.zeros_like(jump_op, dtype=np.complex128)
    for i, t in enumerate(time_labels):
        oft_op += (np.exp(-1j * 2 * np.pi * phase * t / num_labels)  #!
                    * normalized_gauss_values_bin_order[i] 
                    * time_evolution(t) @ jump_op @ time_evolution(-t)) / np.sqrt(num_labels)
    return oft_op