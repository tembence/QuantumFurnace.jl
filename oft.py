
import numpy as np
import qutip as qt
from scipy.linalg import logm, expm

import sys
sys.path.append('/Users/bence/code/liouvillian_metro/')

def oft(jump_op: np.ndarray, energy: float, num_labels: int, sigma: float,
        hamiltonian: np.ndarray = None, U: np.ndarray = None) -> np.ndarray:
    
    time_labels = np.arange(-0.5, 0.5, 1/num_labels)
    
    gauss = lambda t: np.exp(-(t ** 2) / (4 * sigma ** 2))
    gauss_normalization = np.sum([np.sqrt(np.abs(gauss(t))**2) for t in time_labels])
    
    if hamiltonian is None:
        time_evolution = lambda t: np.linalg.matrix_power(U, t)  #TODO: NOT CORRECT, How many Trotter steps do we need, for each time label?
        #TODO: How long do we evolve the system for, T or 2pi T in total? It is 2 pi T I think.
    else:
        time_evolution = lambda t: expm(1j * t * hamiltonian)
    
    oft_op = np.zeros_like(jump_op, dtype=np.complex128)
    for t in time_labels:
        oft_op += np.exp(1j * 2 * np.pi * energy * t / num_labels) * gauss(t) * time_evolution(t) @ jump_op @ time_evolution(-t)
        
    return oft_op / (np.sqrt(num_labels) * gauss_normalization)  #? Maybe energy unit prefactor too