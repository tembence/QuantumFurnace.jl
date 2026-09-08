# DLL research interface contract

This page describes the implemented research interface, numerical controls and
capability limits. DLL custom filters, CKG joint compilation and finite-spectrum
CKG-to-DLL conversion are available. Existing
`Config`, `HamHam`, `JumpOp`, `Workspace` and result APIs keep their meanings.
`test/test_research_contract.jl` verifies the repaired T00–T02 regressions
and is registered in the default test runner.

## Entry points and units

`pauli_hamiltonian` and `HamHam(H; beta_phys)` are available.
`simulate_gibbs`, `GibbsSimulationResult`, and the physical-input `Workspace`
constructor are available. `KMSFilter`, `FrequencyFilter`, `RateFilter`, and
`TimeFilter` are available; `TimeFilter` must be prepared before numerical simulation. `Workspace` remains the existing type.

Executable built-in DLL example:

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
needed.** The physical-input interface reuses these types.
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

T05 adds an executable preparation helper for the existing DLL backends:

```julia
p = prepare_gibbs_inputs(0.4X + 0.7Z; temperature=1.25,
    filter=DLLGaussianFilter, jumps=:onsite_paulis)
ws = Workspace(p.config, p.hamiltonian, p.jumps)
p.provenance
```

Supply exactly one of `beta_phys` or `temperature`, with `k_B=1` and finite,
positive values. Prebuilt filters must match the requested physical beta at
Hamiltonian working precision. Gaussian widths set by beta, Metropolis support
and symmetric frequency shifts convert together. Channel weights are unchanged.
The helper supports the existing DLL Gaussian, Metropolis, symmetric-translate
and multichannel families, plus custom frequency, q and rate specifications in BohrDomain. It rejects negative symmetric shifts explicitly (supply `abs(shift)` if
intended). CKG also accepts typed transitions in Bohr/Energy through `construction=KMS()`, as described below. GNS retains its low-level `Config` API. The simulation facade uses these same preparation rules.

For DLL Time, supply `domain=TimeDomain()`, physical `time_step` and
`num_energy_bits`; the algorithm time step is `R*time_step`. Prepared filters accept independent coherent controls as described below. Legacy
built-ins retain their shared-grid defaults; accuracy must be checked.
Bohr preparation needs no register or CKG rate parameters; the returned
`Config.sigma=1` is an unused DLL compatibility value recorded as such.

`HamHam(ham; beta_phys=new_beta)` validates and reuses its eigensystem without
another diagonalisation, copies the spectral arrays, and recomputes Gibbs/Bohr
data. Passing an old-temperature Hamiltonian directly to the helper rejects.
The helper records whether H arrived in physical matrix or algorithm cache
coordinates, the physical/resolved filters, beta, energy scale and shift,
source rates, working precision, and Gibbs underflow. It also rejects stored
Gibbs offdiagonals and stale Bohr caches. Scalar H has the same Gibbs state at
every beta, so a prior temperature cannot be inferred from that state alone.
Cache checks establish numerical Gibbs agreement within the recorded precision
tolerances, including when cold states become numerically indistinguishable.

An optional `clock=GeneratorClock{Float64}(:my_clock, multiplier, declared_rate)`
scales every source amplitude by `sqrt(multiplier)`. The complete generator and
decay rates scale by `multiplier`; equal evolutions use times divided by it.
`declared_rate` is user-provided reference metadata, never a computed gap or a
positive relaxation-rate claim for a zero generator. With no clock the raw
multiplier is one, independently of Hamiltonian rescaling. Prepared provenance
is returned separately from legacy `Config`; portable combined result storage
is described in the persistence section below.

## Filter and rate authoring

These routes are implemented within the domain and evidence limits below.

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

This table describes the available **Lindbladian** paths.
The DLL facade supports Bohr and Time execution. Existence of a domain type does not
establish support. “Available” does not certify a chosen quadrature tolerance.

T01 repaired the Time source adjoint. T02 supports DLL Time workspaces with
retained per-channel matrices and the same coherent quadrature as dense Time.
DLL validation no longer requires CKG transition-rate parameters. DLL
`Thermalize` channels reject explicitly; use `Lindbladian()` evolution.

| Construction/filter | Dense Bohr | Bohr workspace | Dense Time | Time workspace | Energy/Trotter | Implementing task |
|---|---|---|---|---|---|---|
| DLL built-ins, Hermitian sources | Available | Available | Available | Available | Rejected | T02 complete; T09 facade |
| DLL built-ins, adjoint-paired sources | Available | Available | Available | Available | Rejected | T01–T02 complete |
| DLL existing global multichannel filters | Available, separate channels | Available | Available, separate channels | Available | Rejected | T01–T02 and T14 complete; source assignments preserve channel multiplicity |
| DLL custom complex filters | Available, finite Bohr checks | Available, retained samples | Available with prepared transforms | Available with prepared transforms | Rejected | T10–T14 complete |
| CKG built-in Gaussian OFT/rates | Available | Available | Available | Available | Available with valid registers/local Trotter cache | T15 typed rates available; T20 release checks |
| CKG general joint filter/rate | Available, bounded joint compilation | Available, retained samples | Available, explicit transform controls | Available, owned samples | Energy available; custom Trotter rejected | T16–T17 complete |

DLL GQSP remains rejected (T02 keeps the rejection); DLL Energy/Trotter and a
new DLL circuit implementation are outside scope. Existing CKG thermalisation,
GQSP, GNS and optional TN paths retain their own documented capabilities.
CKG-to-DLL finite-Bohr factorisation is a bounded reference task (T18), not a
default scalable replacement. The [T21 symbolic feasibility experiment](symbolic_filters.md)
retains the numerical provider; no symbolic extension or dependency is added.

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
trajectory. Portable persistence and callback reconstruction are described below.

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
| Trajectory | `:reached_threshold`, `:already_within_threshold`, `:not_reached_by_horizon`, `:inconclusive`; crossing is specific to rho0, with horizon, threshold, numerical floor and error evidence. |
| Budget | Report partial evidence and `:budget_exhausted`; skipped checks remain `:not_run`. No silent tolerance relaxation or long sweep. |
| State validity | Raw trace, Hermiticity and positivity defects before any repair; record repair magnitude. |

Expose both full `trace_norm` and `trace_distance=trace_norm/2`. Existing
`distances` arrays retain the half-norm convention. Report simulated time,
channel steps, wall time, memory and matvec counts separately. Resource
estimates include dense H/filtered matrices, Krylov basis and per-thread/channel
scratch; a dense Liouvillian requires its own explicit small-system cap.

Public tutorials execute during the local documentation build and in default
integration tests. The capability tests also cover rejected combinations.
Finite reference checks do not certify continuum tails or scalability.

### Implemented diagnostic checks (T06)

`workspace_diagnostics(ws, config, ham; rho=nothing, dense_max_dim=16,
max_dense_bytes=64*1024^2)` inspects the compiled Lindbladian in the Hamiltonian
eigenbasis. It retains absolute and probe-normalised residuals for stationarity
and trace preservation, raw state defects, Gibbs conditioning, and budgeted
complete KMS-parent/kernel evidence. A failed probe-normalised tolerance is
conservative: its scale is a lower estimate of the operator norm. Dense budgets
cover an estimated temporary working set, additional to the existing workspace;
they are not operating-system memory limits. Trotter-basis diagnostics currently
reject explicitly. Unknown integrated transform tails stay `:not_run`.
`state_diagnostics(rho)` also works independently. No state repair is applied.
A complete numerical kernel result is scoped to this finite system and the
reported tolerance; it does not establish uniform mixing.

### Implemented spectral extraction (T07)

`krylov_spectral_gap` now selects the first resolved decay rate beyond the
entire detected stationary space. It retains raw eigenvalues/vectors and adds
`spectrum_diagnostics`, `fixed_point_diagnostics`, and `gap_mode_index`.
`NaN` in the legacy scalar gap means no resolved relaxation rate (including
unstable or unresolved peripheral modes). A positive candidate from a partial
Krylov spectrum still has `:inconclusive` global reliability and never establishes
uniqueness. Single-vector Arnoldi can miss multiplicities and slow sectors;
independent-start refinement is T08. Residual-compatible zeros are numerical
resolution statements, not exact kernel certificates. Residuals do not bound
eigenvalue errors for a general nonnormal operator.

`fixed_point` is phase/trace-normalised without Hermitian or positivity repair;
inspect its validity report before using it. If no zero-compatible mode has a
stable nonzero trace, this legacy matrix field contains NaNs. The raw modes
remain unmodified. `run_krylov_spectrum` retains diagnostic reports in metadata.
Channel residuals remain in raw channel units and the legacy converted
`eigenvalues=(mu-1)/delta` convention is unchanged.

`extract_leading_eigendata` now uses the complete dense spectrum for its gap.
Older `compute_fixed_point_distance`/`run_exact_diagnostics` retain their legacy
state-repair path, and overlap/defect/discriminant convenience fields retain
second-mode semantics as documented in their docstrings. Use the new checks
for nonunique or unresolved systems. `kms_parent_spectrum` uses a homogeneous
operator-scale tolerance; `dense_dll_irreducibility` also accepts validated
adjoint pairs through an equivalent Hermitian source family. Its finite-system
commutant witness still requires a faithful invariant Gibbs state.

## Independent gap checks (T08)

`robust_spectral_gap(ws; diagnostics=:standard)` runs three deterministic,
independent O(1) operator starts and a larger-subspace repeat on the same compiled
workspace, sequentially. `operator_starts=[A,B,...]` explicitly supplies eigenbasis
operators; these need not be density matrices. The low-level
`krylov_spectral_gap(...; operator_start=A)` supports a single explicit start.

`runs` retains raw spectra/residuals and `agreement` compares captured rates and
slow-cluster spans without averaging incompatible gaps. Consensus does not prove
full coverage or uniqueness. Degenerate modes can give different captured
subspaces even with identical rates. `reliability=:pass` is reserved for agreement
with a passing complete, finite-system numerical reference in `:strict` mode;
`:quick` and `:standard` remain globally inconclusive.

`max_bytes`, `max_matvecs`, `max_seconds` bound the diagnostic work; memory is
checked before each solve/refinement and includes retained results, compiled
scratch and projected algebra. Time is checked at operation boundaries, so a
native eigensolve/matvec can overrun the deadline. These are allocation estimates
and cooperative time limits, not operating-system caps. Exhaustion preserves
completed runs and records a reason. No dense fallback occurs on solver failure.
Strict mode separately budgets complete KMS/kernel checks through
`dense_max_dim`; `parent_action=true` also measures a transformed Ritz residual
using the existing KMS-parent action after conditioning and Hermiticity pass.
This optional probe adds no continuum or exhaustive-spectrum guarantee.

## Built-in simulation facade (T09)

```julia
ws = Workspace(0.3X + 0.4Y + 0.7Z; beta_phys=0.8)
r = simulate_gibbs(ws; times=range(0, 12; length=61))
r.convergence          # sampled, initial-state-specific threshold evidence
r.trajectory.distances # trace distance, half the trace norm
r.trajectory.trace_norms
r.spectrum.agreement   # independent captured operator modes
r.diagnostics          # stationarity and available KMS/kernel checks
```

`simulate_gibbs(H; beta_phys, times, ...)` combines these two steps. Built-in DLL
Bohr and explicitly discretised Time paths are supported. Typed CKG rates support
Bohr/Energy through `construction=KMS()`; other legacy CKG/GNS paths remain available. `temperature` is an alternative to `beta_phys`, with `k_B=1`.
`transition_weight` must be `nothing` for DLL. The canonical coherent correction
is included. The onsite preset is always assembled in computational coordinates;
`basis=:eigen` changes supplied state/output coordinates and supplied matrix-source
coordinates, not the definition of the preset.

`dry_run=true` checks the dimension, physical settings, source/channel upper
bounds and memory estimates before Hamiltonian diagonalisation. Quantities that
require spectral preparation, such as `beta_alg` for a matrix input, remain
`nothing` and appear in `unresolved`. Source pairing/basis correspondence and
cached spectral data receive their full checks during compilation. A permitted
preflight is not evidence of KMS, mixing or accuracy. `max_bytes` gates construction
and trajectory working-set estimates; `max_saved_bytes` separately caps retained
states. Estimates include per-thread scratch and temporary output rotations.

Times must start at zero and be finite, nonnegative and strictly increasing.
The default initial state is computational `|+><+|` tensor power. State storage
is off by default; final/saved states and predictor modes use the declared basis.
Independent `spectrum` modes explicitly use `spectrum.basis=:eigen`; its clock is
the declared compiled generator clock. Continuous semigroup time, wall seconds
and channel-step counts are separate; this facade's `channel_steps` is `nothing`.

The default `method=:krylov` uses existing Krylov exponentiation. Raw validity
checks precede any optional `repair_states=true` Hermitian/trace correction;
`repair_norms` records its size. No positivity clipping occurs. `max_matvecs` and
`max_seconds` bound trajectory work cooperatively. Diagnostic work has a separate
bounded budget, adjustable with `gap_options=(;max_matvecs=..., max_seconds=...)`.
Completed trajectory samples survive diagnostic exhaustion. Physical checks
skipped by that exhausted budget remain `not_run`.

The threshold `epsilon` refers to trace distance. `threshold_time` is the first
sample at or below it, without interpolation. `:not_reached_by_horizon` does not
assert nonergodicity. `:already_within_threshold` and `:reached_threshold` require
successful propagation and raw validity checks at their numerical tolerances;
they are not worst-case mixing certificates. The numerical floor remains unknown.
`max_extensions=0` disables automatic extension; a positive count allows bounded
horizon doubling, capped by `max_time`, only while the target remains unreached.

Optional `method=:predictor` retains all captured modes and performs independent
full-state spot checks at the initial, middle and final samples. Those checks are
sampled evidence, not a uniform-time certificate. `modal_crossing` uses
`eigenmode_mixing_time` on this predictor's own retained decay modes and stationary
projection, and is explicitly labelled unpropagated. It may lie beyond the actual
horizon and never changes the sampled convergence status. No biexponential fit is
used. Raw all-mode low-level predictor payloads without a stationary projection
reject the mixing-time helper rather than silently dropping their stationary
contribution. Predictor Time-domain use rejects; direct Krylov Time remains
available. Combined-result serialization is described below.

T10 compiles each DLL Bohr channel into an owned complex frequency table before
source construction or threading. Every sampled value is finite and the full
sampled set passes weighted conjugate reflection at working precision. This is
finite-spectrum numerical evidence; no continuum or implementation theorem is
inferred. Gain, loss and coherent construction share these samples, and the
workspace forms its canonical correction from the retained loss matrix.
Standalone coherent construction retains an independent source-product path.

## Custom frequency filters (T11)

```julia
beta_phys = 0.8
filter = KMSFilter(beta_phys;
    q_positive=x -> exp(-(x/0.7)^2) * cis(0.4x),
    name=:shifted_gaussian, version="1", parameters=(width=0.7, shift=0.4))
H = pauli_hamiltonian(2, [-0.7 => (1=>:Z, 2=>:Z), -0.4 => (1=>:X,)])
result = simulate_gibbs(H; beta_phys, filter, times=[0.0, 0.1], diagnostics=:quick)
result.provenance.filter_evidence
result.provenance.filter_compilation
```

Callbacks are concretely typed. Supply a stable nonempty `name`, an optional
`version` (default `"1"`) and a `parameters` named tuple. Parameters are copied;
callbacks remain user-owned definitions. Compilation owns the numerical samples,
so subsequent callback mutation cannot change an existing workspace. A new
workspace resamples the callback. Names and parameter records do not make a
closure portable; reconstruction needs registration or explicit resupply, as described below.

`KMSFilter` takes unweighted q on nonnegative physical frequencies. Negative
frequencies are conjugate-reflected and the thermal factor is added once. For
steep factors, use `logabs_q_positive=x -> ...` instead of `q_positive`, optionally
with `phase_positive=x -> ...` of unit modulus. `-Inf` means an exact zero.
The phase at a nonzero origin value must be real. A callback that already rounded
q to zero cannot recover its lost magnitude; the log route avoids that loss.
Overflowing final amplitudes and nonfinite callback values reject explicitly.

`RateFilter(beta_phys; downward_rate=x -> 2exp(-x*x), name=:rate)` takes a finite
nonnegative downward rate at energy release x. The upward amplitude is computed
in the log domain as `exp(log(r_down)/2-beta_phys*x/2)` with principal-root phase.
Rates fix populations but do not uniquely determine nonsecular coherences;
this constructor explicitly chooses zero phase. No rate maximum is normalised.

`FrequencyFilter` accepts the complete amplitude including its thermal factor.
Its reflected identity is checked on the complete finite Bohr set at working
precision. This is numerical finite-system evidence, without a functional KMS
claim elsewhere. Explicit DLL `transition_weight` is rejected; sampled callbacks
cannot reveal whether the author intended a different unweighted q.

For all four specifications, `support=nothing` means unknown support; a positive
finite radius enforces zero outside `[-support,support]`. Frequency specifications
use physical energy coordinates, whereas `TimeFilter` uses physical time. The
facade evaluates frequency callbacks at `R*nu_alg`, preserving widths, phases and
amplitudes without an additional clock multiplier. `tail_bound` is an optional
callable of a cutoff, retained as **user-supplied, unverified** information.
Compact support alone proves neither smoothness nor a transform theorem.

`filter_evidence` separates structural balance, transform existence, tail
information and implementation-theorem applicability. The compiled report records
finite-Bohr balance checks and flags active transition zeros as possibly reducing
connectivity. Such zeros are allowed and are not a balance failure or a proof of
nonergodicity. One global filter/channel family is applied to both adjoint partners;
per-source assignments are available through `DLLSourceFilters`. Wrap `TimeFilter` or a custom frequency filter in `prepare_filter_transform`
for numerical Time execution. Unprepared callbacks still reject in TimeDomain
because their numerical windows and controls have not been specified.

### Prepared numerical Fourier transforms (T12)

`prepare_filter_transform(f; window, breakpoints, rtol, atol, maxevals,
max_panels, analytic=true)` owns a deep copy of the filter and its captured data.
Callbacks must be deterministic and must not read mutable global or external
state. Rebuild the preparation after changing a callback; compiled workspaces
retain their own sampled operators.

`transform_values(p, targets; direction=:inverse, window_refinements=1)` returns
values, QuadGK error estimates, declared or unknown input L1 tails, and bounded
window-expansion differences (at most three doublings). These differences are
numerical evidence, not tail bounds. The returned values use the original window.
Compact support is integrated over its exact declared extent; breakpoints split
that extent. Oscillation-aware subdivision rejects targets exceeding the panel
budget. Neither a small endpoint value nor a quadrature estimate certifies an
omitted tail. The numerical path reports `:unresolved` when its quadrature budget
cannot meet the requested tolerance.

The convention is `F(ν)=∫f(t)exp(iνt)dt` with inverse `1/(2π)`. Analytic Gaussian
pairs are preferred. A prepared legacy `GaussianFilter` adapts its frequency
shape by `sqrt(π)/sigma`, leaving the existing CKG kernel methods unchanged.
`fourier_sum(nodes, weights, targets; sign=1, backend=:direct)` provides a generic
precision reference for finite sums; `backend=:finufft` uses deterministic
single-threaded Float64/ComplexF64 and explicitly rejects higher precision.
Quadrature weights and Fourier normalisation belong in the supplied weights.

```julia
f = KMSFilter(1.0; q_positive=x -> exp(-x^2/8)*cis(0.4x), name=:phase)
p = prepare_filter_transform(f; window=16.0)
evidence = transform_values(p, [-1.0, 0.0, 1.0])
# evidence.tail_status == :unknown; this is not a continuum certificate.
```

### Independent coherent controls and custom DLL Time (T13)

Prepared filters accept `coherent=(; ...)` with independent `time_step`,
`time_window`, `frequency_window` and `frequency_grid_size`. Omitted coherent
time controls inherit the supplied dissipative grid. Frequency-input filters
inherit their transform window; time-input filters must declare a coherent
frequency window. All supplied coordinates are physical in the facade and
algorithm coordinates in legacy Config. Conversion includes the Jacobian
`f_alg(t)=f_phys(t/R)/R`; it changes no generator clock.

```julia
H = ComplexF64[0.2 0.3im; -0.3im -0.2]
f = KMSFilter(0.8; q_positive=x -> exp(-(0.8x)^2/8)*cis(0.2x), name=:phase_time)
p = prepare_filter_transform(f; window=20.0,
    coherent=(;time_step=0.12, time_window=6.0,
        frequency_grid_size=257, policy=:error))
ws = Workspace(H; beta_phys=0.8, filter=p, domain=TimeDomain(),
    time_step=0.12, num_energy_bits=7)
result = simulate_gibbs(ws; times=[0.0, 0.05], diagnostics=:quick)
result.provenance.time_compilation
```

The default `method=:time` transforms the complex two-frequency kernel with
factor `(2π)^(-2)`, then contracts `A(t′)'*A(t)`. `backend=:direct` supplies a
precision-preserving finite-sum reference; `:finufft` uses Float64 internally.
No quadrature or callback runs in a source matvec. The facade includes the
largest refinement grids in its construction-memory estimate; `max_points`
bounds each coherent axis (default 2049). Larger grids require explicit budgets.

With `refine=true` (default), the report measures four independent changes:
doubling the frequency window at fixed spacing, halving frequency spacing,
doubling the time window, and halving time spacing. It also compares implemented
dissipative amplitudes with the complete finite-Bohr input and compares B with
the implemented-loss reference. `tolerance=1e-9` controls these **numerical
checks**, not a rigorous continuum bound. `policy=:warn` returns an explicitly
`:unresolved` result when a check fails; `:error` rejects it. `refine=false`
retains the reference checks and records refinement status as `:not_checked`.
Omitted tails remain uncertified even when measured differences are small.

Explicit `method=:hybrid` uses time-integrated jumps with the canonical correction
from their implemented loss R. Provenance calls this
`:time_jumps_implemented_loss_correction`, distinct from
`:full_two_time_quadrature`. It validates neither the two-time representation nor
its implementation cost and does not restore KMS to inaccurate jumps.
Hermiticity is checked before optional `repair=true` roundoff symmetrisation;
a material defect rejects. The repair size is reported separately and is never
called a KMS correction. Prepared custom channels can be grouped with `DLLMultiChannelFilter`;
the existing built-in multichannel paths retain their behaviour.


### Per-source DLL channels (T14)

`DLLMultiChannelFilter((f, g, f), beta_phys)` stores three separate channels.
Nested families flatten in order; the repeated `f` doubles its contribution to
both dissipator and coherent term. Tuple and vector inputs are accepted and
stored as a concrete tuple. Scalar kernel sums remain diagnostic quantities.

Use `filter=DLLSourceFilters((family_for_A, family_for_B), beta_phys)` with
`jumps=[A, B]` to assign one prescription to each source. Integer-index pairs
are also accepted, for example `(2 => g, 1 => f)`; missing or duplicate indices
reject. Adjoint partners must have the same channel multiset, including
multiplicity and source rates. Reuse the same custom callback or prepared
filter object; a matching name or matching finite samples does not establish
matching prescriptions. Channel order may differ between partners.
`complete_adjoint=true` copies the originating prescription to newly appended
partners; it does not repair conflicting assignments on existing partners.

```julia
beta_phys = 0.8
H = ComplexF64[0 0; 0 1]
X = ComplexF64[0 1; 1 0]
Z = ComplexF64[1 0; 0 -1]
f = DLLGaussianFilter(beta_phys)
g = KMSFilter(beta_phys; q_positive=x -> exp(-x^2)*cis(0.2x), name=:phase)
channels = DLLMultiChannelFilter((f, g), beta_phys)
ws = Workspace(H; beta_phys, jumps=[X, Z],
    filter=DLLSourceFilters((channels, g), beta_phys))
```

For TimeDomain, prepare each custom channel separately, with its own Fourier
and coherent controls, before grouping. Each hybrid correction uses that
channel's implemented loss operator. Result provenance records source partner
indices, channel counts, physical/algorithm filters, and per-channel Time
reports. Filter callbacks run during serial preparation, and matrix-free actions
use only retained matrices. No continuum tail or ergodicity theorem follows
from these finite-system checks.


### Typed CKG rates and Gaussian mixtures (T15)

Select `construction=KMS()` and a typed `transition_weight`. `GaussianTransition`,
`MetropolisTransition` and `SmoothMetropolisTransition` replace the legacy rate
parameter combinations while retaining their analytic coefficients, coherent
term and clock. Each prescription takes physical beta and physical OFT width
`sigma` (default `1/beta_phys`). The default CKG prescription is Gaussian.

`GaussianMixtureTransition` accepts finite nonnegative weights and positive
centres `x`, where the rate Gaussian is centred at `-x` and has variance
`2x/beta_phys - sigma^2`. Thus every component obeys
`beta_phys = 2x/(sigma^2 + sigma_gamma^2)`. The variance must be strictly positive;
the singular zero-variance endpoint rejects. These rates require the normalised
Gaussian OFT, `C(w)=(2π sigma^2)^(-1/4) exp(-w^2/(4sigma^2))`, implemented using
`GaussianFilter` and the existing external normalisation. An arbitrary OFT
substitution is rejected on this analytic route; use the joint compiler below for a different OFT.

```julia
beta_phys = 0.8
rate = GaussianMixtureTransition(beta_phys; sigma=0.35,
    centers=(0.2, 0.6), weights=(0.3, 0.8))
H = ComplexF64[0 0; 0 1]
ws = Workspace(H; beta_phys, construction=KMS(), transition_weight=rate)
result = simulate_gibbs(ws; times=[0.0, 0.1], diagnostics=:quick)
```

Mixtures default to `normalization=:none`, preserving their weights. Explicit
`normalization=:bound, supremum_bound=M` divides the **entire generator**,
including B, by the fixed positive bound `M >= sum(weights)`. This bound need not
be the exact supremum. Changing an outer-frequency grid never changes it.
A separate `GeneratorClock` remains an additional explicit multiplier.

`prepare_gaussian_mixture(beta_phys; sigma, density, interval=(lo, hi),
regularity=:continuous_nonnegative, panels=64)` uses positive midpoint weights
and retains the doubled-panel approximation. Continuity and nonnegativity are
caller assumptions; callback samples cannot certify them. The report includes
mass and alpha-probe refinement differences. An infinite upper interval also
requires a finite `cutoff` and a caller-supplied integrated `tail_mass_bound`
below `tail_atol`; this assumption bounds omitted rate and alpha entries.
Refinement differences are numerical evidence, not rigorous quadrature bounds.
For bound normalisation use the same declared `supremum_bound` across refinements.

BohrDomain retains analytic alpha and B. EnergyDomain additionally requires
physical `energy_step` and `num_energy_bits`, controlling a separate outer
frequency grid; the full requested grid is retained for mixtures. Physical
widths, centres and energy steps are divided by the Hamiltonian rescaling factor;
the normalised OFT and integration measure cancel their Jacobians in alpha.
Mixture Time requires the explicit joint compiler and transform controls below; custom Trotter and GQSP reject.
Existing built-in Time/Trotter configurations remain available through the legacy
API. Retained finite-mixture parameters can be saved with Config; the original
continuum density or arbitrary callbacks are not reconstructed by that snapshot.

### General CKG joint kernels (T16)

`CKGJointKernel(beta_phys; oft=C, rate=gamma, frequency_window=(lo,hi))`
accepts the full complex, normalised transform, with `integral(abs2(C))=1`,
and a nonnegative rate **together**. Supply it as `transition_weight` with
`construction=KMS()`. Bohr, Energy and the explicit Time conversion below support Float64 Hamiltonians.
Custom Trotter, thermalisation channels and GQSP remain rejected.

```julia
using QuantumFurnace, QuadGK
beta_phys = 0.8
normalization = inv(sqrt(first(quadgk(
    x -> exp(beta_phys*x/2 - 2x^4), -Inf, Inf))))
pair = CKGJointKernel(beta_phys;
    oft=x -> normalization * exp(beta_phys*x/4 - x^4) * cis(0.3x),
    rate=w -> exp(-w^2 - beta_phys*w/2),
    frequency_window=(-8.0, 8.0), panels=32)
ws = Workspace(ComplexF64[0 0; 0 0.7]; beta_phys,
    construction=KMS(), transition_weight=pair)
result = simulate_gibbs(ws; times=[0.0, 0.1], diagnostics=:quick)
ws.research_provenance.filter_evidence[1]
```

This example has the structural form `C(x)=exp(beta_phys*x/4)q(x)` and
`gamma(w)=exp(-beta_phys*w/2)g(w)`, where `q(-x)=conj(q(x))` and `g` is even
and nonnegative. Substitution in the integral proves the joint reflection
identity. Normalising C preserves it. A classical rate ratio alone is
insufficient: retaining this gamma but replacing C by an ordinary untilted
Gaussian fails the joint test.

The compiler forms
`alpha(u,v)=integral(gamma(w)*C(w-u)*conj(C(w-v)), dw)` with positive quadrature
weights, preserving Gram positivity. It checks **all pairs** in the complete
Hamiltonian Bohr set against independent adaptive real-line integrals and
`alpha(u,v)=exp(-beta_alg*(u+v)/2)*alpha(-v,-u)` in algorithm coordinates.
The coherent correction uses the implemented loss, including its quadrature.
Both domains retain complex samples; callbacks and quadrature do not run in
matrix-vector actions. No sampled supremum or implicit clock rescaling is used.

Bohr uses an eight-point composite Gauss rule: `frequency_window` and `panels`
control its outer window and spacing. Energy uses the explicit physical
`energy_step` and `num_energy_bits` register instead; its actual retained window
is reported separately. Both must meet the default `balance_rtol=1e-8` against
the adaptive reference (`rtol=1e-10`). Tighten these tolerances and discretisations
when a smaller numerical floor is needed. `:not_KMS` means the resolved reference
fails balance; `:quadrature_unresolved` means the reference passes but the
implemented grid does not agree; `:reference_unresolved` means the adaptive
error estimate exceeds tolerance. Failed kernels reject from the standard
facade. `compile_ckg_kernel(pair, frequencies; strict=false)` exposes these
statuses for inspection without accepting a failed KMS simulation.

The balance metric is a bounded reflection residual divided by the largest
reference alpha entry; it is neither a relative error for every tiny transition
nor a Gibbs-weighted generator bound. QuadGK error estimates, callback
nonnegativity and finite-spectrum balance checks are numerical evidence, not
functional proofs or certified tails. Optional `structural_provenance` records
a caller-supplied source; it does not bypass validation. Neither this compiler
nor a small defect establishes mixing or an efficient implementation theorem.

Compilation defaults to at most 65 distinct Bohr frequencies and a 64 MiB
working-set estimate, checked before quadratic kernel allocation. The facade
also checks its total construction budget. `maxevals` is a **per integral** cap;
there are `1+m*(m+1)/2` adaptive integrals for `m` frequencies. These are bounded
research references, with O(m²) kernel storage, not an arbitrary-size production
algorithm. A changed Hamiltonian, beta or Energy grid requires recompilation.
Prepared data own samples; callback reconstruction requires the original
definition and the checks described below.


### General CKG Time conversion (T17)

Pass physical windows and steps explicitly. This executable example uses the
normalised `pair.oft` and `pair.rate` from the preceding example:

```julia
time_pair = CKGJointKernel(beta_phys; oft=pair.oft, rate=pair.rate,
    frequency_window=(-8.0,8.0), panels=32,
    time_transform=(frequency_window=8.0, frequency_grid_size=257,
        coherent_frequency_window=8.0, coherent_frequency_grid_size=129,
        coherent_time_window=19.2, coherent_time_step=0.15, backend=:direct))
time_ws = Workspace(ComplexF64[0 0; 0 0.7]; beta_phys,
    construction=KMS(), transition_weight=time_pair, domain=TimeDomain(),
    time_step=0.15, num_energy_bits=8)
```

The dissipative time grid has `2^num_energy_bits` points separated by
`time_step`. Its inverse-transform window and spacing are controlled separately
by `time_transform.frequency_window` and `frequency_grid_size`. The outer
frequency integral still uses the joint kernel's `frequency_window` and
`panels`, with positive weights and no sampled rate normalisation.

The coherent term uses the full two-dimensional inverse transform of
`tanh(beta*(v-u)/4)*alpha(u,v)/(2im)`, with its own frequency window/grid and
time window/step. It contracts the ordered product `A(s)'*A(t)`, including
non-Hermitian paired sources. It does not use the built-in `b_minus/b_plus`
formulas or replace the two-time calculation with a correction from the
implemented loss. Both transforms are prepared once; actions use owned samples.
`:finufft` may be selected for the dissipative Fourier sums and coherent
contraction; the bounded two-dimensional inverse itself uses matrix products.

Every Bohr pair is compared with the retained frequency reference. Independent
dissipative and coherent differences, the actual Time coefficient balance
defect, and nested frequency-reference evidence are recorded. Exceeding
`balance_rtol` rejects with the controls to refine. These are small-system
numerical comparisons, not integrated-tail certificates. Valid KMS kernels,
particularly rates with nondecaying tails such as Metropolis, need not admit
absolutely integrable two-frequency kernels. Explicit finite windows may fail
to resolve them within the resource budget. No universal Time convergence or
quantum implementation claim follows. `max_bytes` includes time/frequency
matrices, prefactors and dimension-dependent contraction scratch.

Custom CKG Trotter remains unsupported: even a retained local Hamiltonian
needs a separately validated evolution and coherent-kernel algorithm. An
opaque dense Hamiltonian additionally lacks a local decomposition. The error
states this gate; legacy built-in `Config` plus `make_trotter_for_config`
continues to support its established Time/Trotter paths. A future custom
Trotter task must verify basis, local evolution and independent coherent error
before this gate can be opened.

## Portable results and continuation

`save_result(result, path)` and `load_result(path)` support
`GibbsSimulationResult` through a versioned tagged-data schema, alongside the
legacy BSON result formats. Saved trajectories retain the trace-distance
convention, basis, diagnostic statuses, and partial results. Loading evidence
does not require a custom callback definition.

```julia
using QuantumFurnace
checkpoint = simulate_gibbs(0.3X + 0.7Z; beta_phys=0.8,
    times=[0.0, 0.1], diagnostics=:quick)
continued = mktempdir() do directory
    saved = save_result(checkpoint, joinpath(directory, "checkpoint.bson"))
    loaded = load_result(saved)
    simulate_gibbs(loaded; times=[0.0, 0.2], diagnostics=:quick)
end
@assert continued.provenance.resume_time_origin ≈ 0.1
```

For results produced through the physical-input facade, `Workspace(loaded)`
rebuilds fresh mutable buffers from owned model and source
matrices, the retained spectral basis, tagged filter/rate parameters and
transform controls. It preserves already-weighted source amplitudes and the
original generator clock. A legacy low-level workspace result without a replay
snapshot retains evidence but requires the original inputs for reconstruction.
Temperature, domain and transform-control changes
require preparation from new physical inputs; replay does not reuse an old
cache for a changed problem.

`simulate_gibbs(loaded; times, ...)` continues the saved final state. Its times
start at zero **additional** generator time; the returned result contains this
segment and a cumulative `provenance.resume_time_origin`. It does not restore
an interrupted internal Arnoldi iteration. Saved solver settings are defaults
and may be explicitly overridden for the new segment. The original result
retains its own trajectory and diagnostic evidence.

Built-in filters store tagged parameters. Custom definitions retain their
name, version, parameters, sampled grids and change-detection digests; a
closure's printed representation is never treated as executable source. Use
`register_filter!(name, version, factory)` in each process, where
`factory(beta, parameters)` returns the matching custom DLL specification, or
resupply that specification with `filters=Dict((name, version) => filter)`.
Custom CKG joint callbacks require `filters=Dict(:ckg_joint => kernel)`.
Missing definitions, changed versions, sample mismatches or modified replay
data reject reconstruction. Rebuilt compiled generator data are also compared
with the saved preparation; transform caches are rebuilt from controls. A saved closure remains labelled as requiring its
definition; a name and version alone provide no standalone reproducibility.

Provenance records the initial density matrix and requested times, physical and
algorithm beta, energy frame, source basis and normalisation, generator clock,
solver/transform controls, spectral RNG seed,
Julia/package revision and working-tree status. The algorithm uses deterministic
Krylov propagation and seeded spectral starts; no global random state is needed
to continue the final density matrix. Timestamps and timing/thread metadata
are runtime evidence, not deterministic physics outputs. Sample digests are
change checks, not authentication or proof of global function identity. Unknown
continuum tails and unavailable tests remain explicitly unknown or not run.
