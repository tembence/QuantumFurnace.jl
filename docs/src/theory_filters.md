# Finite-spectrum CKG and DLL equivalence

`ckg_to_dll` is a bounded reference conversion for a small Hamiltonian's Bohr
frequencies. It represents a positive semidefinite CKG coefficient matrix by
separate DLL channels. It does not alter the production CKG route.

Let `nu = E_i - E_j` and write the source components as ``A_\nu``. For a
Hermitian source, or an equally weighted adjoint-closed source family, the CKG
gain is

```math
\Phi(X)=\sum_{a,\nu,\nu'}\alpha_{\nu\nu'}
A^a_\nu X(A^a_{\nu'})^\dagger.
```

The required coefficient identities on the complete finite Bohr set are
``\alpha\succeq0`` and

```math
\alpha_{\nu\nu'}=e^{-\beta(\nu+\nu')/2}\alpha_{-\nu',-\nu}.
```

Define the tilted matrix
``C_{\nu\nu'}=e^{\beta(\nu+\nu')/4}\alpha_{\nu\nu'}`` and the frequency
reflection permutation ``P``. Hermiticity and the second identity give
``C=P\overline C P``. The antiunitary involution ``Kz=P\overline z`` thus
commutes with ``C``. For each frequency pair choose the two columns
``(e_\nu+e_{-\nu})/\sqrt2`` and
``i(e_\nu-e_{-\nu})/\sqrt2``, plus the real zero-frequency column. These
columns form a unitary matrix ``U`` satisfying ``P\overline U=U``.
Consequently ``U^\dagger C U`` is real symmetric and positive semidefinite.
Real orthogonal diagonalisation gives

```math
U^\dagger C U=O D O^T,\qquad Q=UO\sqrt D,\qquad
F_\ell(\nu)=e^{-\beta\nu/4}Q_{\nu\ell}.
```

Every column obeys ``Q_{-\nu,\ell}=\overline{Q_{\nu,\ell}}`` and
``\alpha_{\nu\nu'}=\sum_\ell F_\ell(\nu)\overline{F_\ell(\nu')}``.
This argument includes complex kernels, imaginary odd channels with zero
amplitude at the origin, and degenerate eigenspaces. Ding–Li–Lin §4,
Eqs. (4.8)–(4.12), establishes the real tilted-matrix construction. The
antiunitary argument above supplies the complex finite-matrix extension.

The DLL operators are ``L_{a\ell}=\sum_\nu F_\ell(\nu)A^a_\nu``. Sum their
dissipators separately: ``\mathcal D[\sum_\ell L_{a\ell}]`` introduces cross
terms and is a different generator. Since both representations have the same
gain, they have the same loss ``R=\Phi^\dagger(I)`` and coherent correction

```math
B_{ij}=\frac{i}{2}\tanh\!\left(\frac{\beta(E_i-E_j)}4\right)R_{ij}.
```

They therefore agree in the full generator
``\mathcal L(X)=\Phi(X)-\{R,X\}/2-i[B,X]``. This uses the package's sampler
frame. Any separately prescribed Hamiltonian commutator must be identical in
both representations.

## Reference API and numerical evidence

```julia
using QuantumFurnace, LinearAlgebra

nus = [-0.5, 0.0, 0.5]  # the complete Bohr set of a two-level Hamiltonian
beta = 0.8
rate = GaussianTransition(beta; sigma=0.4, sigma_gamma=0.6)
alpha = [transition_alpha(rate, u, v) for u in nus, v in nus]
reference = ckg_to_dll(alpha, nus, beta)
reference.evidence.channel_count
reference.evidence.coefficient_total_error_norm
```

Rows follow the supplied frequency order. The result's `filter` is a
`DLLMultiChannelFilter` containing owned finite-frequency tables. These tables
currently use the low-level `Config` API in the same frequency/temperature frame;
the physical-input `Workspace(H; ...)` facade rejects them. Use the same Bohr
spectrum, inverse temperature, source amplitudes and generator clock. A prepared `CKGJointKernel` can be supplied directly after compilation;
it must have passing validation evidence. Conversion checks its coefficients
again with the tighter roundoff tolerance. A passing quadrature check alone
need not meet this tighter requirement; refine the original quadrature if
necessary. No rate normalisation is introduced.

`max_bohr_frequencies` (default 65) and `max_bytes` (default 64 MiB) cap the
reference working-set estimate before quadratic/dense allocations. The
coefficient diagonalisation supports `Float32` and `Float64`; excessive thermal
tilts reject. Hermiticity, KMS reflection and PSD defects are checked before
repair. The default relative roundoff tolerance is
`min(128*m*eps(T), sqrt(eps(T)))`; a custom tolerance cannot exceed
`sqrt(eps(T))`. Within this tolerance, symmetry defects and eigenvalues whose
magnitude is at the roundoff scale are removed. Evidence records the raw
defects, minimum eigenvalue and the resulting coefficient repair norm. These
are numerical measurements, not certified error bounds.

For optional rank compression, pass `max_rank` or `rank_rtol`, together with
`hamiltonian=ham` and compiled `jumps`. Truncation acts on the tilted matrix's
eigenvalues. Evidence distinguishes repair, requested compression and total
errors. Coefficient errors use the spectral 2-norm of the **untilted** alpha
matrix. Generator errors use the induced Hilbert–Schmidt norm of the complete
dense generator difference for the supplied Hamiltonian and sources, including
the coherent correction. This norm is not a diamond norm. Without compression,
Hamiltonian/source inputs remain optional and generator evidence is `nothing`
when they are omitted. A zero kernel has one zero placeholder channel and
`retained_rank=0`.

Finite tables define no globally smooth filter, inverse Fourier transform,
Time/Trotter implementation, mixing bound or quantum speedup. The channel count
can grow with the Bohr-set size. Extending or changing the Hamiltonian spectrum
requires a new conversion. The reference test suite independently compares
alpha, loss, coherent correction and full generators for Gaussian, Metropolis
and complex kernels at small sizes to `1e-9` or tighter.

## Sources

- Z. Ding, B. Li and L. Lin, *Efficient quantum Gibbs samplers with
  Kubo–Martin–Schwinger detailed balance condition* (2024), Theorem 10 and §4.
- C.-F. Chen, M. J. Kastoryano and A. Gilyén, *An efficient and exact
  noncommutative quantum Gibbs sampler* (2025), §II, Propositions II.2–II.3.
