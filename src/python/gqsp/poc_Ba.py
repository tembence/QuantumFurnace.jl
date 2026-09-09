"""GQSP applied to B_a loaded from a Julia export.

Loads the canonical B_a matrix produced by `scripts/scratch_gqsp_B_n3.jl`
and applies the GQSP pipeline (Jacobi-Anger + BS Q + MW angles + Qiskit circuit)
from Steps 1-3, with B_a in place of the random Hermitian H.

This Python POC uses the simple 1-ancilla qubitization block encoding for
GQSP, not the thesis nested-LCU U_{B_a} (which is verified separately in
Julia). The qualitative test — that GQSP achieves O(δ^2) error in
realising e^{-i δ B_a} from a block encoding of B_a/α — is the same.

Run:
    .venv-uv/bin/python src/python/gqsp/poc_Ba.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Operator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from python.gqsp.berntson_sunderhauf import complementary_polynomial_bs
    from python.gqsp.circuit import build_block_encoding_one_anc, build_walk, gqsp_circuit
    from python.gqsp.jacobi_anger import (
        evaluate_polynomial_on_circle,
        jacobi_anger_coeffs,
    )
    from python.gqsp.motlagh_wiebe import gqsp_extract_angles
else:
    from .berntson_sunderhauf import complementary_polynomial_bs
    from .circuit import build_block_encoding_one_anc, build_walk, gqsp_circuit
    from .jacobi_anger import evaluate_polynomial_on_circle, jacobi_anger_coeffs
    from .motlagh_wiebe import gqsp_extract_angles


def load_julia_export(data_dir: Path):
    """Load the Julia-exported B_a, H, A_a binaries plus metadata."""
    meta = {}
    with (data_dir / "Ba_n3.meta").open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split(maxsplit=1)
            try:
                meta[key] = float(value)
            except ValueError:
                meta[key] = value
    n_sys = int(meta["n_sys"])
    dim = 2 ** n_sys
    # Julia writes column-major; numpy reads row-major. Reshape with order='F'.
    B_a = np.fromfile(data_dir / "Ba_n3.bin", dtype=np.complex128).reshape(dim, dim, order="F")
    H = np.fromfile(data_dir / "H_n3.bin", dtype=np.complex128).reshape(dim, dim, order="F")
    A_a = np.fromfile(data_dir / "Aa_n3.bin", dtype=np.complex128).reshape(dim, dim, order="F")
    return B_a, H, A_a, meta


def gqsp_top_left_block_qiskit(qc, n_sys_qubits: int) -> np.ndarray:
    """Extract ⟨0_qsp 0_anc| U |0_qsp 0_anc⟩ on the system register from a GQSP circuit."""
    U = Operator(qc).data
    n_sys_dim = 2 ** n_sys_qubits
    block = np.zeros((n_sys_dim, n_sys_dim), dtype=complex)
    for i in range(n_sys_dim):
        for j in range(n_sys_dim):
            block[i, j] = U[i * 4, j * 4]
    return block


def main():
    data_dir = Path(__file__).resolve().parent.parent / "tests" / "data"
    B_a, H, A_a, meta = load_julia_export(data_dir)
    n_sys = int(meta["n_sys"])
    alpha = float(meta["alpha"])
    Ba_norm = float(meta["Ba_norm"])

    print("== GQSP on B_a ==\n")
    print(f"  n_sys = {n_sys}, dim = {B_a.shape[0]}")
    print(f"  ‖H‖   = {np.linalg.norm(H, ord=2):.4f}")
    print(f"  ‖B_a‖ = {Ba_norm:.4f}, α = {alpha:.4f}, ‖B_a/α‖ = {Ba_norm/alpha:.4f}")
    if Ba_norm / alpha > 1:
        print(f"  WARNING: ‖B_a/α‖ > 1 — block-encoding precondition violated")
        return

    # Use one-ancilla qubitization with H' := B_a/α
    # (rescaled so ‖H'‖ ≤ 1 for the simple block-encoding construction).
    H_eff = B_a / alpha
    UH = build_block_encoding_one_anc(H_eff)
    W = build_walk(UH)

    print("\n  delta            ‖tl_qiskit − exp(-iδ B_a)‖     slope         ‖tl_qiskit − L_d(W)_top‖")
    prev_err, prev_delta = None, None
    for delta in (1e-1, 1e-2, 1e-3):
        d = 1
        # IMPORTANT: in this 1-anc qubitization, B_a/α plays the role of "H" with α' = 1.
        # The GQSP polynomial L_d realises e^{-i delta_alpha cos theta} where cos theta = λ
        # eigenvalue of B_a/α. To get e^{-i δ B_a}, we use δ_α = δ × α (effective subnorm).
        # That is: the Jacobi-Anger argument is δ × α (not δ).
        delta_alpha = delta * alpha

        p = jacobi_anger_coeffs(d, delta_alpha)
        theta = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
        Pmax = float(np.max(np.abs(evaluate_polynomial_on_circle(p, theta))))
        rescale = 0.99 / max(Pmax, 0.99)
        p_rs = rescale * p
        N_bs = max(256, 32 * (d + 1))
        while N_bs & (N_bs - 1) != 0:
            N_bs += 1
        q = complementary_polynomial_bs(p_rs, N_bs)
        lam, thetas, phis = gqsp_extract_angles(p_rs, q)
        qc = gqsp_circuit(UH, lam, thetas, phis, d, realise="L_d")
        block = gqsp_top_left_block_qiskit(qc, n_sys) / rescale

        eigvals, eigvecs = np.linalg.eigh(B_a)
        target = eigvecs @ np.diag(np.exp(-1j * delta * eigvals)) @ eigvecs.conj().T
        err = float(np.linalg.norm(block - target, ord=2))

        # Direct L_d(W) ground truth
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
        Ld_tl = L[:2 ** n_sys, :2 ** n_sys]
        err_match = float(np.linalg.norm(block - Ld_tl, ord=2))

        slope_str = "—"
        if prev_err is not None:
            slope = np.log(err / prev_err) / np.log(delta / prev_delta)
            slope_str = f"{slope:.3f}"
        print(f"  {delta:.2e}        {err:.6e}                 {slope_str}        {err_match:.3e}")
        prev_err, prev_delta = err, delta


if __name__ == "__main__":
    main()
