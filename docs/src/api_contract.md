# DLL research interface contract

This page freezes the interface planned by tasks T00–T21. The new facade and
filter constructors below are **target API, not yet available**. Existing
`Config`, `HamHam`, `JumpOp`, `Workspace` and result APIs keep their meanings.
`test/test_research_contract.jl` verifies the repaired T00–T02 regressions
and is registered in the default test runner.

## Entry points and units

`pauli_hamiltonian` and `HamHam(H; beta_phys)` are available after T03.
The names `simulate_gibbs`, `KMSFilter`, `FrequencyFilter`, `RateFilter`,
`TimeFilter`, and `GibbsSimulationResult` remain reserved target API. `Workspace` gains
a convenience constructor; it remains the existing compiled workspace type.

Target example (Hamiltonian input: T03; configuration: T05; facade: T09):

```julia
H = pauli_hamiltonian(3, [
    -1.0 => (1 => :Z, 2 => :Z),
    -1.0 => (2 => :Z, 3 => :Z),
    -0.7 => (1 => :X,),
    -0.7 => (2 => :X,),
    -0.7 => (3 => :X,),
])
result = simulate_gibbs(H; beta_phys=0.8, construction=DLL(),
    filter=DLLGaussianFilter, jumps=:onsite_paulis,
    times=range(0, 20; length=101), diagnostics=:standard)
```

Each term occurs once at its listed sites; site 1 is the leftmost tensor
factor. Identity terms use an empty site tuple. Labels are `:I`, `:X`, `:Y`,
`:Z`; repeated sites in a term reject, repeated terms add. Coefficients are
finite real numbers. Construction returns a sparse physical Hermitian matrix
without diagonalisation. `HamHam(H; beta_phys=0.8)` accepts finite, exactly
Hermitian qubit matrices and performs dense spectral preparation in Float32 or
Float64 (integer matrices use Float64; other floating precisions reject).
It stores the Gibbs state in the eigenbasis. No local/Trotter decomposition is
invented for matrix input; Trotter synthesis rejects even scalar matrix input.

Zero and scalar Hamiltonians `H=c*I` use an explicit reference scale of one
physical energy unit: `rescaling_factor=1`, `shift=-c`, `H_alg=0`, `nu_min=0`.
Their Gibbs state is maximally mixed. A zero `nu_min` also occurs for ordinary
degenerate spectra; it denotes the minimum adjacent spacing, not a positive
frequency cutoff. Physical temperature must still be finite and positive.

| Keyword/input | Frozen meaning | Task |
|---|---|---|
| `beta_phys` or `temperature` | Exactly one, finite and positive; temperature uses energy units with `k_B=1`. Both zero and infinite endpoint limits reject initially. No implicit unit conversion. | T05 |
| `construction=DLL()` | New facade default. `KMS()` selects CKG KMS; GNS remains a legacy API. | T05, T09 |
| `domain=BohrDomain()` | New default. Matrix-free Krylov evolution with dense Hamiltonian spectral preparation. | T09 |
| `filter=DLLGaussianFilter` | Factory called with physical beta; a supplied filter instance also has physical-frame parameters and must match beta. | T05 |
| `jumps=:onsite_paulis` | All `3n` onsite Pauli sources with amplitude `1/sqrt(3n)`. Supplied matrices retain their amplitudes. | T04 |
| `complete_adjoint=false` | Missing adjoints or unequal multiplicities reject. Explicit `true` adds only missing multiplicities and reports additions; Hermitian sources are not doubled. | T04 |
| `basis=:computational` | Input/output states and supplied jump matrices use the computational basis. Explicit `:eigen` means the Hamiltonian eigenbasis. H itself always uses computational coordinates. | T04, T09 |
| `rho0` | Default `|+><+|` tensor power, recorded in provenance. Internally rotate once. | T09 |
| `times` | Nonempty finite strictly increasing nonnegative generator times, starting at zero; time is continuous semigroup time, not steps or wall time. | T09 |
| `transition_weight=nothing` | DLL thermal tilt is already in the amplitude. An arbitrary CKG rate cannot be multiplied into it. For CKG this input is the joint-validated rate `gamma`. | T05, T15–T17 |
| `diagnostics=:standard` | Bounded independent spectral starts and subspace refinement; `:quick` can be tentative, `:strict` adds budgeted reference checks. | T06–T09 |
| `dry_run=true` | Return resolved settings, basis/frame/clock, capabilities, source/channel counts, resource estimates and planned checks before expensive spectral preparation. | T09 |

`Workspace(H; beta_phys, jumps, filter, construction, domain, ...)` compiles once;
`simulate_gibbs(ws; times, rho0, diagnostics, ...)` reuses that workspace (T09).
An existing raw builder tuple or `HamHam` retains its explicit frame metadata;
cached Gibbs data and requested temperature must agree or be recomputed with
recorded provenance. No magnitude-based inference of energy frame is allowed.
State storage is off by default; requested saved states use the declared basis
and a memory cap. The facade never falls back silently to a dense Liouvillian.

For `H_alg = H_phys/R + sI`, `beta_alg = R*beta_phys`. Legacy `Config.beta`
and positional `HamHam(raw, beta)` retain algorithm-beta semantics. Physical
DLL amplitudes convert as `F_alg(nu)=F_phys(R*nu)` and
`f_alg(t)=f_phys(t/R)/R`, including support/width/phase conversions. This
preserves filtered matrices for unchanged source amplitudes. It does **not**
imply multiplication of the generator by R. The default facade clock is the
raw generator defined by those amplitudes; a derived clock must state its
generator multiplier, and rates/times transform reciprocally (T05).

`GeneratorClock` and `DLLFilterFrame` already live in the core module's
`tensor_networks.jl`, loaded without ITensors. **T00 decision: no relocation is
needed.** T05 should reuse them without editing the unrelated TN work.
`GeneratorClock` describes positive derived clocks and intentionally rejects
`:raw_generator`; store that raw label separately with multiplier one and no
derived-clock object. A zero generator has no positive derived rate. The
existing frame record covers built-in support/shift/weight, not arbitrary
callback identity or phase; retain additional filter provenance separately.

The T04 source preparation API is executable:

```julia
ham = HamHam(0.4X + 0.7Z; beta_phys=0.8)
jump = JumpOp(Y, ham) # owns computational and eigenbasis matrices
prepared = prepare_jumps(:onsite_paulis, ham)
paired = prepare_jumps([ComplexF64[0 1; 0 0]], ham;
    complete_adjoint=true, rates=2.0)
# Pass prepared.jumps to the existing low-level constructors.
```

`prepare_jumps` returns `jumps` and `provenance`. Matrices retain their input
amplitudes; each positive rate contributes its square root to the source.
Adjoint completion preserves multiplicity, appends only missing partners, and
rejects unequal rates on existing partners. Re-preparing the returned jumps
with default rates is idempotent. Stored `JumpOp` inputs are checked against
both bases and flags, then copied. `orthogonal` retains the legacy meaning of
transpose symmetry. Zero, identity and dephasing sources are accepted without
asserting ergodicity. The onsite preset records its `1/sqrt(3n)` amplitude;
user matrices receive no proposal normalisation. Nonuniform rates change the
generator and are recorded per source, without inventing a single clock factor.

## Filter and rate authoring

These routes are reserved for T11–T17; names do not assert current support.

| Route | Meaning and required evidence | Task |
|---|---|---|
| `KMSFilter(beta_phys; q_positive, name, support, tail_bound=nothing)` | Callback receives physical `nu >= 0`; enforce `q(-nu)=conj(q(nu))`, real `q(0)`, then `F=exp(-beta*nu/4)*q`. Balance by construction; regularity and tails separately classified. | T11 |
| `FrequencyFilter(beta_phys; amplitude, name, support, tail_bound=nothing)` | Full physical-frequency amplitude on both signs. Test weighted reflection on actual Bohr frequencies; sampled tests are not a global proof. | T10–T12 |
| `RateFilter(beta_phys; downward_rate, name, support, tail_bound=nothing)` | Callback takes nonnegative energy release `x`, gives `r(-x)`; set `r(x)=exp(-beta*x)*r(-x)` and use the principal square-root amplitude with zero phase. Nonnegative finite rates required. | T11 |
| `TimeFilter(beta_phys; kernel, name, support, tail_bound=nothing)` | Full physical-time kernel with the DLL Fourier convention. Numerical forward transform, actual-Bohr checks; no silent balance projection. Here support and tails are in time. | T11–T12 |
| Source/channel assignments | Every adjoint partner has the same filter/rate multiplicity. Channels contribute separate dissipators; no dissipator of their summed amplitude. | T14 |
| CKG Gaussian-mixture rate | Nonnegative mixture compatible with the same beta and explicitly normalised Gaussian OFT. | T15 |
| General CKG filter plus rate | Validate joint-kernel PSD and two-frequency KMS symmetry; a classical rate ratio alone is insufficient. | T16–T17 |

`support=nothing` means unknown support, not a proven infinite-tail bound;
a finite interval is a user-declared exact support in the input variable.
Tail callbacks bound the integrated omitted absolute kernel outside a supplied
window, with assumptions/provenance retained; scalar quadrature estimates are
separate. An even nonnegative DLL rate envelope, if explicitly selected,
multiplies the amplitude by its square root. Filter values are amplitudes, not
probabilities. Source-rate factors likewise enter sources as square roots.

DLL uses `f(t)=(2pi)^(-1) integral F(nu) exp(-im*nu*t) dnu`.
Legacy CKG Gaussian shapes have external normalisation factors; adapt rather
than reinterpret them. Keep four evidence fields separate: algebraic balance,
continuum transform existence, numerical error evidence, and applicability of
an efficient implementation theorem. There is no universal `certified=true`.
FINUFFT currently evaluates Float64/ComplexF64 internally. Unknown tails stay
unknown; T12–T13 provide direct references and independent refinement controls.

## Construction and domain capabilities

This table describes **low-level Lindbladian** paths after T02.
The facade remains pending until T09. Existence of a domain type does not
establish support. “Available” does not certify a chosen quadrature tolerance.

T01 repaired the Time source adjoint. T02 supports DLL Time workspaces with
retained per-channel matrices and the same coherent quadrature as dense Time.
DLL validation no longer requires CKG transition-rate parameters. DLL
`Thermalize` channels reject explicitly; use `Lindbladian()` evolution.

| Construction/filter | Dense Bohr | Bohr workspace | Dense Time | Time workspace | Energy/Trotter | Implementing task |
|---|---|---|---|---|---|---|
| DLL built-ins, Hermitian sources | Available | Available | Available | Available | Rejected | T02 complete; T09 facade |
| DLL built-ins, adjoint-paired sources | Available | Available | Available | Available | Rejected | T01–T02 complete |
| DLL existing global multichannel filters | Available, separate channels | Available | Available, separate channels | Available | Rejected | T01–T02 complete; heterogeneous per-source expansion T14 |
| DLL custom complex filters | Real-only/trait limitations | Same limitations | Unavailable as general interface | Unavailable | Rejected | T10–T11 Bohr; T12–T13 Time; T14 channels |
| CKG built-in Gaussian OFT/rates | Available | Available | Available | Available | Available with valid registers/local Trotter cache | Preserve; T15 typed rates; T20 release checks |
| CKG general joint filter/rate | Unavailable | Unavailable | Unavailable | Unavailable | General Energy pending; custom Trotter gated | T16 Bohr/Energy; T17 Time and explicit Trotter gate |

DLL GQSP remains rejected (T02 keeps the rejection); DLL Energy/Trotter and a
new DLL circuit implementation are outside scope. Existing CKG thermalisation,
GQSP, GNS and optional TN paths retain their own documented capabilities.
CKG-to-DLL finite-Bohr factorisation is a bounded reference task (T18), not a
default scalable replacement. Symbolic transforms are optional feasibility
work (T21); numerical execution cannot depend on their success.

The strict sampler includes `-im[B,rho]`, with
`B_ij=(im/2)*tanh(beta*(E_i-E_j)/4)*sum(L' * L)_ij`.
It does not silently add a physical Hamiltonian commutator. The Time correction
contracts `A(t')' * A(t)`, reducing to `A(t') * A(t)` for Hermitian sources.
Separate gains/losses built from identical implemented operators preserve GKLS;
finite quadrature may still break exact KMS. Rebuilding B from approximate
jumps is a distinct hybrid reference, not proof of the full two-time method.

## Results and evidence

`GibbsSimulationResult <: AbstractResults` (T09) owns `trajectory`, `spectrum`,
`diagnostics` and provenance, retaining existing payloads without unnecessary
array copies. Missing spectrum after a solver failure does not erase a valid
trajectory. Portable persistence and callback reconstruction belong to T19.

Every check has `status` in `:pass`, `:fail`, `:inconclusive`, `:not_run`, plus
quantity, tolerance, method and evidence scope. A separate evidence field
distinguishes structural facts from numerical tests. T06 implements this
schema; T07–T08 implement the spectral reliability policies.

| Result question | Required distinction |
|---|---|
| Stationarity/KMS | Report absolute and scaled residuals, complete dense checks versus matrix-free probes, and numerical Gibbs faithfulness. |
| Uniqueness | `:established`, `:nonunique`, `:not_established`, with scope; a small-system complete numerical kernel is not a general theorem. |
| Spectrum | Raw complex eigenvalues, residuals, detected zero-mode count, target operator/clock, starts/subspace agreement and reliability. `:inconclusive` if tiny decay rates cannot be resolved from zero. |
| Gap | Relaxation rate above the whole detected stationary manifold; no absolute-value repair of unstable eigenvalues, and no global certificate from converged Ritz pairs. |
| Trajectory | `:reached`, `:not_reached_by_horizon`, `:inconclusive`; crossing is specific to rho0, with horizon, threshold, numerical floor and error evidence. |
| Budget | Report partial evidence and `:budget_exhausted`; skipped checks remain `:not_run`. No silent tolerance relaxation or long sweep. |
| State validity | Raw trace, Hermiticity and positivity defects before any repair; record repair magnitude. |

Expose both full `trace_norm` and `trace_distance=trace_norm/2`. Existing
`distances` arrays retain the half-norm convention. Report simulated time,
channel steps, wall time, memory and matvec counts separately. Resource
estimates include dense H/filtered matrices, Krylov basis and per-thread/channel
scratch; a dense Liouvillian requires its own explicit small-system cap.

Numerical budgets and default tolerances are measured in T07–T08. T20 gates
public tutorials and supported capability claims on executable examples and
tests; this contract page does not advertise the target snippets as runnable.
