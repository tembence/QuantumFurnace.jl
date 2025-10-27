import numpy as np
from scipy.linalg import logm
import qutip as qt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit import Parameter
from tools.classical import hamiltonian_matrix, rescaling_and_shift_factors

def trotter_step_heisenberg(num_qubits: int, coeffs: list[float], disordering: bool = False) -> QuantumCircuit:
    """Parametrized trotter step circuit for 1D Heisenberg chain: H = XX + YY + ZZ
       It has a fixed step size, for which Trotter errors should be fine, thus for longer time evolution
       we just need to adjust the number of steps 
       disordering done with Z gates, with strength = 1 and at default the non-bdr qubits.
    """
    
    if disordering == True and len(coeffs) != 4:
        raise ValueError("disorderinging Heisenberg requires 4 coefficients")
    
    trotter_step_circ = QuantumCircuit(num_qubits, name="H")
    disordering_positions = list(range(1, num_qubits - 1))  # Bulk qubits
      
    step_size = Parameter('theta')
    for i in range(num_qubits):
        if i != num_qubits - 1:
            trotter_step_circ.rxx(-2 * coeffs[0] * step_size, i, i + 1)   # Qiskit convention, rxx(-2x) = exp(x)
            trotter_step_circ.ryy(-2 * coeffs[1] * step_size, i, i + 1)
            trotter_step_circ.rzz(-2 * coeffs[2] * step_size, i, i + 1)
        if (i == num_qubits - 1):  # Periodic boundary conditions
            trotter_step_circ.rxx(-2 * coeffs[0] * step_size, i, 0)
            trotter_step_circ.ryy(-2 * coeffs[1] * step_size, i, 0)
            trotter_step_circ.rzz(-2 * coeffs[2] * step_size, i, 0)
        if disordering == True:
            if i in disordering_positions:
                trotter_step_circ.rz(-2 * coeffs[3] * step_size, i)
            
    return trotter_step_circ

def inverse_trotter_step_heisenberg(num_qubits: int, coeffs: list[float], disordering: bool = False) -> QuantumCircuit:
    if disordering == True and len(coeffs) != 4:
        raise ValueError("disorderinging Heisenberg requires 4 coefficients")
    
    trotter_step_circ = QuantumCircuit(num_qubits, name="H")
    disordering_positions = list(range(1, num_qubits - 1))  # Bulk qubits
      
    step_size = Parameter('theta')
    for i in reversed(range(num_qubits)):
        if disordering == True:
            if i in disordering_positions:
                trotter_step_circ.rz(-2 * coeffs[3] * step_size, i)
        if i != num_qubits - 1:
            trotter_step_circ.rzz(-2 * coeffs[2] * step_size, i, i + 1)
            trotter_step_circ.ryy(-2 * coeffs[1] * step_size, i, i + 1)
            trotter_step_circ.rxx(-2 * coeffs[0] * step_size, i, i + 1)   #! Qiskit convention, rxx(-2x) = exp(x)
        if (i == num_qubits - 1):  # Periodic boundary conditions
            trotter_step_circ.rzz(-2 * coeffs[2] * step_size, i, 0)
            trotter_step_circ.ryy(-2 * coeffs[1] * step_size, i, 0)
            trotter_step_circ.rxx(-2 * coeffs[0] * step_size, i, 0)
            
    return trotter_step_circ

def second_order_trotter_step_circ(num_qubits: int, coeffs = [1, 1, 1]):
    qr = QuantumRegister(num_qubits, name="q")
    first_stage_circ = QuantumCircuit(qr)
    step_size = Parameter('theta')
    
    for i in range(qr.size):
        if i != num_qubits - 1:
            first_stage_circ.rxx(-2 * coeffs[0] * step_size / 2, i, i + 1)
            first_stage_circ.ryy(-2 * coeffs[1] * step_size / 2, i, i + 1)
            first_stage_circ.rzz(-2 * coeffs[2] * step_size / 2, i, i + 1)
        if (i == num_qubits - 1):
            first_stage_circ.rxx(-2 * coeffs[0] * step_size / 2, i, 0)
            first_stage_circ.ryy(-2 * coeffs[1] * step_size / 2, i, 0)
            first_stage_circ.rzz(-2 * coeffs[2] * step_size / 2, i, 0)
    
    second_stage_circ = first_stage_circ.reverse_ops()
    trotter_step_circ = first_stage_circ.compose(second_stage_circ, qr)
    
    return trotter_step_circ

def ham_evol(num_qubits: int, trotter_step: QuantumCircuit, num_trotter_steps: int, time: float) -> QuantumCircuit:
    """Time parametrized Hamiltonian evolution
    """
    
    circ = QuantumCircuit(num_qubits, name="H")
    
    for _ in range(num_trotter_steps):
        circ.compose(trotter_step, inplace=True)
    
    step_size = time / num_trotter_steps
    circ.assign_parameters([step_size], inplace=True)
    
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
    hamiltonian = hamiltonian_matrix([X, X], [Y, Y], [Z, Z], coeffs=[1, 1, 1], num_qubits=num_qubits)
    print(np.linalg.eigvalsh(hamiltonian))
    rescaling_factor, shift = rescaling_and_shift_factors(hamiltonian)
    trotter_step = trotter_step_heisenberg(num_qubits)
    
    circ = QuantumCircuit(num_qubits, name="H")
    # Shift and rescale circuit Hamiltonian, spec(H) in [0, 1]
    total_time /= rescaling_factor

    for i in range(int(np.ceil(total_time / step_size))):
        circ.compose(trotter_step, inplace=True)

    spectrum_from_circ = trott_ham_spectrum_from_circ(circ, total_time)
    print(spectrum_from_circ)
    
    # print(circ)