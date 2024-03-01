
import numpy as np
import qutip as qt
from scipy.linalg import logm, expm

import sys
sys.path.append('/Users/bence/code/liouvillian_metro/')

def oft(jump_op: np.ndarray, energy: float, num_labels: int, sigma: float,
        hamiltonian: np.ndarray = None, trotter: np.ndarray = None) -> np.ndarray:
    
    #! In math order
    # time_labels = np.arange(-0.5, 0.5, 1/num_labels)
    # N_labels = np.arange(-num_labels/2, num_labels/2, 1, dtype=int)
    
    #! In binary order
    N_labels = np.arange(num_labels / 2, dtype=int)
    N_labels_neg = np.arange(- num_labels / 2, 0, dtype=int)
    N_labels = np.concatenate((N_labels, N_labels_neg))
    # time_labels = np.array(N_labels) / num_labels

    gauss = lambda t: np.exp((-t ** 2) / (4 * sigma ** 2))
    gauss_normalization = np.sqrt(np.sum([np.sqrt(np.abs(gauss(n))**2) for n in N_labels]))
    # gauss_normalization = np.sqrt(np.sum([np.sqrt(np.abs(gauss(t))**2) for t in time_labels]))
    
    oft_op = np.zeros_like(jump_op, dtype=np.complex128)
    if hamiltonian is None and trotter is not None:  #* Trotterized
        time_evolution = lambda n: np.linalg.matrix_power(trotter, n)
        for n in N_labels:
            oft_op += (np.exp(-1j * 2 * np.pi * energy * n)  #!
                       * gauss(n) * time_evolution(n) @ jump_op @ time_evolution(-n))
            
    elif hamiltonian is not None and trotter is None:  #* Exact
        time_evolution = lambda t: expm(1j * t * hamiltonian)
        for n in N_labels:
            oft_op += np.exp(-1j * energy * 2 * np.pi * n) * gauss(n) * time_evolution(2 * np.pi * n) @ jump_op @ time_evolution(-2 * np.pi * n)
        
    return oft_op / (np.sqrt(num_labels) * gauss_normalization)  #? Maybe energy unit prefactor too