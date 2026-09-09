"""Qiskit proof-of-concept circuit for GQSP.

End-to-end:
    1. Random Hermitian H on n_sys qubits with ||H|| <= 1.
    2. 1-ancilla qubitization U_H, walk W = (Z_anc) * U_H.
    3. Jacobi-Anger c_m, BS complementary Q, MW angles.
    4. Build Qiskit GQSP circuit; statevector simulate.
    5. Compare top-left QSP=|0>, anc=|0> block of the resulting unitary
       to exp(-i delta H). Verify slope-2 in delta.

Run:
    .venv-uv/bin/python -m src.python.gqsp.poc
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Operator

# Allow `python -m src.python.gqsp.poc` and direct `python src/python/gqsp/poc.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from python.gqsp.jacobi_anger import jacobi_anger_coeffs, evaluate_polynomial_on_circle
    from python.gqsp.berntson_sunderhauf import complementary_polynomial_bs
    from python.gqsp.motlagh_wiebe import gqsp_extract_angles, reconstruct_PQ
    from python.gqsp.circuit import build_block_encoding_one_anc, build_walk, gqsp_circuit
else:
    from .jacobi_anger import jacobi_anger_coeffs, evaluate_polynomial_on_circle
    from .berntson_sunderhauf import complementary_polynomial_bs
    from .motlagh_wiebe import gqsp_extract_angles, reconstruct_PQ
    from .circuit import build_block_encoding_one_anc, build_walk, gqsp_circuit


def random_hermitian(n_sys: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = 2 ** n_sys
    G = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    H = (G + G.conj().T) / 2
    H = H / np.linalg.norm(H, ord=2)
    return 0.9 * H


def gqsp_top_left_block_qiskit(qc, n_sys_qubits: int) -> np.ndarray:
    """Statevector-simulate the GQSP circuit and return the top-left
    (QSP=|0>, anc=|0>) block of the resulting unitary, of size 2^n_sys x 2^n_sys.
    """
    U = Operator(qc).data  # shape (2^(n_sys+2), 2^(n_sys+2))
    # Qiskit qubit order in our circuit: q0 = qsp, q1 = anc, q2.. = sys (little-endian).
    # The state-space basis is |sys>|anc>|qsp> reading bit positions 0..n_sys+1
    # from least significant. Operator.data is in this little-endian basis.
    # We want <0_qsp 0_anc| U |0_qsp 0_anc> on the sys subspace:
    # i.e. project onto qsp = anc = 0, which is bit positions 0 and 1 = 00 in the
    # composite index. So row/col indices with bits 0,1 == 00 — equivalently
    # indices 0, 4, 8, ... for n_sys=1, etc.
    n_sys_dim = 2 ** n_sys_qubits
    full_dim = 4 * n_sys_dim
    block = np.zeros((n_sys_dim, n_sys_dim), dtype=complex)
    for i_sys in range(n_sys_dim):
        for j_sys in range(n_sys_dim):
            i_full = i_sys * 4 + 0  # bit pattern: sys = i_sys, anc = 0, qsp = 0
            j_full = j_sys * 4 + 0
            block[i_sys, j_sys] = U[i_full, j_full]
    return block


def laurent_apply_direct(W: np.ndarray, delta_alpha: float, d: int) -> np.ndarray:
    """Compute L_d(W) = sum_{k=-d}^d (-i)^k J_k(delta_alpha) W^k directly. Used as ground truth."""
    from scipy.special import jv

    n2 = W.shape[0]
    L = np.zeros_like(W)
    for k in range(-d, d + 1):
        coef = np.exp(-1j * np.pi * k / 2) * jv(k, delta_alpha)
        if k == 0:
            L = L + coef * np.eye(n2, dtype=complex)
        elif k > 0:
            L = L + coef * np.linalg.matrix_power(W, k)
        else:
            L = L + coef * np.linalg.matrix_power(W.conj().T, -k)
    return L


def angles_from_jacobi_anger(d: int, delta_alpha: float, *, N_bs: int | None = None,
                              margin: float = 0.99):
    """Return (lam, thetas, phis, rescale) for L_d via Jacobi-Anger -> BS -> MW."""
    p = jacobi_anger_coeffs(d, delta_alpha)
    theta_grid = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    Pmax = float(np.max(np.abs(evaluate_polynomial_on_circle(p, theta_grid))))
    rescale = margin / max(Pmax, margin)
    p_rs = rescale * p
    if N_bs is None:
        N_bs = max(256, 32 * (d + 1))
    while N_bs & (N_bs - 1) != 0:
        N_bs += 1
    q = complementary_polynomial_bs(p_rs, N_bs)
    lam, thetas, phis = gqsp_extract_angles(p_rs, q)
    return lam, thetas, phis, rescale


def main():
    print("== Qiskit proof of concept for GQSP ==\n")
    np.random.seed(42)

    for n_sys in (1, 2):
        H = random_hermitian(n_sys, seed=42 + n_sys)
        UH = build_block_encoding_one_anc(H)
        W = build_walk(UH)
        print(f"\n[n_sys={n_sys}]  ||H|| = {np.linalg.norm(H, ord=2):.4f}")
        print(f"  delta            ||tl_qiskit - exp(-i delta H)||      slope         ||tl_qiskit - L_d(W)_top||")
        prev_err = None
        prev_delta = None
        for delta in (1e-1, 1e-2, 1e-3):
            d = 1
            delta_alpha = delta * 1.0  # alpha = 1
            lam, thetas, phis, rescale = angles_from_jacobi_anger(d, delta_alpha)

            qc = gqsp_circuit(UH, lam, thetas, phis, d, realise="L_d")

            tl_qiskit = gqsp_top_left_block_qiskit(qc, n_sys) / rescale  # undo BS rescaling

            # exp(-i delta H) via eigen
            eigvals, eigvecs = np.linalg.eigh(H)
            target = eigvecs @ np.diag(np.exp(-1j * delta * eigvals)) @ eigvecs.conj().T

            err = float(np.linalg.norm(tl_qiskit - target, ord=2))

            # Compare to the direct L_d(W) ground truth (top-left of L_d(W)):
            Ld_W = laurent_apply_direct(W, delta_alpha, d)
            n_sys_dim = 2 ** n_sys
            Ld_tl = Ld_W[:n_sys_dim, :n_sys_dim]
            err_match = float(np.linalg.norm(tl_qiskit - Ld_tl, ord=2))

            slope_str = "—"
            if prev_err is not None and prev_delta is not None:
                ratio = err / prev_err
                slope = np.log(ratio) / np.log(delta / prev_delta)
                slope_str = f"{slope:.3f}"
            print(f"  {delta:.2e}        {err:.6e}                     {slope_str}        {err_match:.6e}")
            prev_err = err
            prev_delta = delta

    print("\n== Round-trip sanity (no Qiskit): MW reconstructs (P, Q) ==\n")
    for delta_alpha in (0.1, 0.5, 1.0):
        for d in (1, 2):
            lam, thetas, phis, rescale = angles_from_jacobi_anger(d, delta_alpha)
            p = jacobi_anger_coeffs(d, delta_alpha)
            p_rs = rescale * p
            theta_grid = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
            q = complementary_polynomial_bs(p_rs, max(256, 32 * (d + 1)))
            P_rec, Q_rec = reconstruct_PQ(lam, thetas, phis)
            P_pad = np.concatenate([p_rs, np.zeros(max(0, len(P_rec) - len(p_rs)), dtype=complex)])
            Prec_pad = np.concatenate([P_rec, np.zeros(max(0, len(p_rs) - len(P_rec)), dtype=complex)])
            Q_pad = np.concatenate([q, np.zeros(max(0, len(Q_rec) - len(q)), dtype=complex)])
            Qrec_pad = np.concatenate([Q_rec, np.zeros(max(0, len(q) - len(Q_rec)), dtype=complex)])
            n_p = max(len(P_pad), len(Prec_pad))
            n_q = max(len(Q_pad), len(Qrec_pad))
            P_pad = np.concatenate([P_pad, np.zeros(n_p - len(P_pad), dtype=complex)])
            Prec_pad = np.concatenate([Prec_pad, np.zeros(n_p - len(Prec_pad), dtype=complex)])
            Q_pad = np.concatenate([Q_pad, np.zeros(n_q - len(Q_pad), dtype=complex)])
            Qrec_pad = np.concatenate([Qrec_pad, np.zeros(n_q - len(Qrec_pad), dtype=complex)])
            err_p = float(np.max(np.abs(P_pad - Prec_pad)))
            err_q = float(np.max(np.abs(Q_pad - Qrec_pad)))
            print(f"  delta_alpha={delta_alpha:.2f}  d={d}   ||P - P_rec||_inf = {err_p:.3e}    ||Q - Q_rec||_inf = {err_q:.3e}")


if __name__ == "__main__":
    main()
