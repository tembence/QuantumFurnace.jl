# Symbolic filter feasibility

The T21 experiment recommends **keeping numerical transforms as the supported
path**. No Symbolics extension or new package dependency is added. This is a
bounded assessment of Symbolics 7.39.0 and SymbolicIntegration 3.7.0 on Julia
1.12.5, not an impossibility result for symbolic Fourier integration.

Use `prepare_filter_transform` for user callbacks and the existing analytic
Gaussian methods when available. Symbolic expressions can suggest formulas,
but do not establish convergence, KMS balance, regularity or error bounds.

## What was tested

The convention is

```math
F(\omega)=\int_{\mathbb R}f(t)e^{i\omega t}\,dt,\qquad
f(t)=\frac{1}{2\pi}\int_{\mathbb R}F(\omega)e^{-i\omega t}\,d\omega.
```

All parameters below are fixed real numbers; time is the remaining real
symbol. Complex integrands are explicitly split into real and imaginary parts,
and the piecewise amplitude is split into its two half-lines. These algebraic
adapters belong to the probe. They are not symbolic-backend discoveries.

| Input amplitude | Convergence assumptions and independent reference |
|:--|:--|
| `exp(-ω^2)` | Positive quadratic coefficient; inverse `exp(-t^2/4)/(2sqrt(π))`. |
| `exp(-ω^2-0.2ω+0.3im*ω)` | Same Gaussian decay; inverse `exp((-0.2+im*(0.3-t))^2/4)/(2sqrt(π))`. |
| `exp(-abs(ω)/2-0.4ω)` | Principal square-root amplitude of the piecewise smooth rate `exp(-abs(ω)-0.8ω)`. Both half-line decay rates, `0.9` and `0.1`, must be positive. Inverse `[1/(0.9+im*t)+1/(0.1-im*t)]/(2π)`. |
| `exp(-1/(1-ω^2)-0.4ω)` on `abs(ω)<1`, zero elsewhere | Compact smooth bump, explicitly zero at both endpoints. Inverse checked by independent integration on `[-1,1]`; no analytic formula supplied. |

The Gaussian quadratic coefficient is positive and real, so its square root
has no branch ambiguity. The piecewise inverse has no real-time poles.
The complex Gaussian obeys the DLL amplitude reflection at scalar `beta=0.8` in the frequency units used here;
the piecewise and bump amplitudes use `beta=1.6`. The symmetric Gaussian
is only a Fourier reference here: its positive API metadata beta is inert and
does **not** certify KMS balance at that beta. No Hamiltonian or generator
clock is introduced by this scalar experiment.

## Backend result and cost

`Symbolics.Integral` successfully represents the requested domains. A represented
integral is not an evaluated transform. The installed standard
`SymbolicIntegration.integrate` methods compute antiderivatives and have no
four-argument definite-integral method. The probe therefore tries the default
antiderivative engine, checks each returned candidate by differentiation and
finite-interval quadrature, and separately attempts endpoint limits.

The real/imaginary Fourier integrands for the Gaussian, shifted complex Gaussian
and compact bump returned unevaluated integrals. The piecewise integrands
returned elementary antiderivatives: all four passed derivative and finite-
interval checks. After unwrapping the expression and variable to the limit
API's required symbolic type, endpoint evaluation reports `Not implemented: sin`.
No complete evaluated improper Fourier family was obtained in the measured route.

The first disposable installation took 12.4 seconds. First loading, including
precompiling previously uncached dependencies, took 220.4 seconds in a separate
cold-cache discovery run. Subsequent family workers loaded the backends in
roughly two seconds, with additional process/probe preparation overhead.
Per-family runtime and startup measurements are saved separately in TOML.
These are single-machine measurements that include JIT compilation; they are
not warmed throughput benchmarks.

Maintaining an extension would add two optional packages and their transitive
algebra/special-function dependencies, expression adapters, branch and endpoint
handling, timeout supervision and validation. The tested route has not shown
an advantage over the analytic Gaussian fast path or numerical callback
preparation that warrants that cost. Other representations, future releases
or an independently evaluated external backend may merit a new experiment.
The optional Maxima integration backend was not installed or tested.

## Numerical verification and its limits

Inverse transforms used the existing provider with `analytic=false`, frequency
windows 8 for the Gaussians and 320 for the piecewise amplitude, and the exact
support radius 1 for the bump. Targets included negative time, zero and
positive time. Gaussian inverse errors were below `1e-15`; the piecewise error
was below `2e-14`; the bump agreed with independent QuadGK to below `1e-15`.
Gaussian forward round trips at time window 16 agreed below `1e-15`.

For the piecewise inverse,

```math
f(t)=\frac{1}{2\pi(0.9+it)(0.1-it)},\qquad
\int_{|t|>T}|f(t)|\,dt\leq\frac{1}{\pi T}.
```

At `T=400`, the observed forward error was about `7.95774e-4`, consistent with
this tail allowance even though quadrature estimates were small. This round
trip does **not** achieve `1e-9`. The probe asserts the explicit tail allowance
and never substitutes a quadrature estimate for it.

For the bump, numerical inversion of the numerical inverse improved from
about `3.28e-7` at time window 128 to about `1.07e-12` at window 512 on three
interior frequency targets. This is finite-window numerical evidence; the
provider correctly leaves the omitted time tail unknown. Neither this check
nor any symbolic return proves a Gevrey class or a global error bound.

## Reproduce in a disposable environment

From the public package checkout:

```bash
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  julia --project --startup-file=no --heap-size-hint=1500M \
  tools/symbolic_filter_probe.jl --output=/tmp/qf-symbolic-report
```

The driver creates a separate temporary Julia environment and installs only
there, with the two directly tested versions pinned. It does not modify the package's `Project.toml` or `Manifest.toml`.
The shared Julia depot may acquire downloaded packages and caches. Results,
expressions, versions, environment snapshots, child logs and timings remain in the printed report
directory; the environment and reports are disposable.

To reuse that installation, pass `--env=/path/to/disposable/environment
--skip-install`. Installation has a 240-second cap; each worker has a
180-second startup cap and a 60-second work cap by default. Override the latter
two with `--startup-timeout=SECONDS` and `--timeout=SECONDS`. Workers use
existing compilation caches only, so termination does not orphan precompile
workers. Startup cost is separate from integration/validation time.

An unsupported or unevaluated expression, error, or timeout selects the same
numerical fallback. The probe also kills an actual nonterminating control
worker after 0.2 seconds of work, verifies reproducible numerical evaluation
without expressions, and checks package loading with the symbolic extension
absent. No symbolic calculation runs inside a sampler or a matrix-vector action.
Only the standalone experiment installs these packages; the Julia simulation
requires no Python-backed symbolic dependency.

## Sources and scope

- [Symbolics 7.39.0 integration documentation](https://github.com/JuliaSymbolics/Symbolics.jl/blob/v7.39.0/docs/src/manual/integration.md): representation versus integration backends.
- [SymbolicIntegration 3.7.0 methods](https://github.com/JuliaSymbolics/SymbolicIntegration.jl/blob/v3.7.0/src/methods.jl): the installed default tries rules, then Risch; returns an unevaluated integral on failure.
- [Symbolics 7.39.0 limit implementation](https://github.com/JuliaSymbolics/Symbolics.jl/blob/v7.39.0/src/limits.jl): supported expression classes and heuristic limitations.

The result concerns these versions, representations and fixed parameters.
It does not establish that no other symbolic route can evaluate the integrals,
and does not validate arbitrary callback integrability or a sampler theorem.
