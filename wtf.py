import numpy as np
import qutip as qt
from scipy.linalg import logm, expm
from qiskit.quantum_info import Operator, state_fidelity
from qiskit import QuantumCircuit
from copy import deepcopy
import random

from tools.classical import *
from tools.quantum import *


# np.random.seed(138)

num_qubits = 10
dist_qutip_qiskit = 0.
while dist_qutip_qiskit < 1:
    step_size = random.uniform(0, 2*np.pi)
    # step_size = 0.9844728807633085
    print(f'RXX angle , {step_size}')
    i = np.random.randint(0, num_qubits)
    j = (i + 1) % num_qubits
    k = (i + 2) % num_qubits
    wherever = np.random.randint(0, num_qubits)
    # i = 2
    # j = 3
    print(f'Sites {i, j, k, wherever}')

    #* Qutip
    XX_0 = pad_term([qt.sigmax(), qt.sigmax()], num_qubits, i)
    YY_0 = pad_term([qt.sigmay(), qt.sigmay()], num_qubits, i)
    ZZ_0 = pad_term([qt.sigmaz(), qt.sigmaz()], num_qubits, i)
    XX_1 = pad_term([qt.sigmax(), qt.sigmax()], num_qubits, j)
    YY_1 = pad_term([qt.sigmay(), qt.sigmay()], num_qubits, j)
    ZZ_1 = pad_term([qt.sigmaz(), qt.sigmaz()], num_qubits, j)
    Z_wherever = pad_term([qt.sigmaz()], num_qubits, wherever)
    
    eXX_0 = expm(1j*step_size*XX_0.full())
    eYY_0 = expm(1j*step_size*YY_0.full())
    eZZ_0 = expm(1j*step_size*ZZ_0.full())
    eXX_1 = expm(1j*step_size*XX_1.full())
    eYY_1 = expm(1j*step_size*YY_1.full())
    eZZ_1 = expm(1j*step_size*ZZ_1.full())
    eZ = expm(1j*step_size*Z_wherever.full())
    
    qutip_op = eZZ_1 @ eYY_1 @ eXX_1 @ eZ @ eZZ_0 @ eYY_0 @ eXX_0

    # XX_byhand_reversed = qt.tensor([qt.qeye(2), qt.sigmax(), qt.sigmax()])
    # eXX_byhand_reversed = expm(1j*step_size*XX_byhand_reversed.full())
    # print('Qutip')
    # print(eXX_0)
    #* Qiskit
    rxx_circ = QuantumCircuit(num_qubits)
    rxx_circ.rxx(-2 * step_size, i, j)
    rxx_circ.ryy(-2 * step_size, i, j)
    rxx_circ.rzz(-2 * step_size, i, j)
    rxx_circ.rz(-2 * step_size, wherever)
    rxx_circ.rxx(-2 * step_size, j, k)
    rxx_circ.ryy(-2 * step_size, j, k)
    rxx_circ.rzz(-2 * step_size, j, k)
    qiskit_op = Operator(rxx_circ)
    # print('Qiskit')
    # print(qiskit_op)

    dist_qutip_qiskit = np.linalg.norm(qutip_op - qiskit_op.data)
    # dist_qutip_qiskit_reversed = np.linalg.norm(eXX_byhand_reversed - rxx_op.data)
    print(f'Distance between Qutip and Qiskit Ops {dist_qutip_qiskit}')
    # print(f'Distance between Qutip and Qiskit RXX reversed {dist_qutip_qiskit_reversed}')

"""So apparently something is unstable here, because time to time we get a large distance
between these ops. Doesn't seem to be dependent on number of qubits or the positions. More
likely that qutip or qiskit or the norm is a bit unstable and generates something

Otherwise qiskit and qutip order seems to match up now. Question if the right state will be initialized in the implementation still
"""