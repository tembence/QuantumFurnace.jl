# # Custom DLL filters and CKG rates
using QuantumFurnace, LinearAlgebra
beta_phys = 0.8
H = Hermitian(0.3X + 0.7Z)

# Supply q on nonnegative physical frequencies. The constructor reflects it as
# q(-nu)=conj(q(nu)) and adds exp(-beta_phys*nu/4) once. A complex phase is allowed.
f = KMSFilter(beta_phys; q_positive=x -> exp(-(x/0.7)^2)*cis(0.4x),
    name=:phase_gaussian, version="1", parameters=(width=0.7, shift=0.4))
custom = simulate_gibbs(H; beta_phys, filter=f,
    times=[0.0, 0.1], diagnostics=:quick)
@assert custom.trajectory.all_converged

# Alternatively prescribe the downward rate at positive energy release x.
# RateFilter supplies its Boltzmann-reflected upward rate and principal-root
# amplitude. The callback returns a rate, not an amplitude.
rate_filter = RateFilter(beta_phys; downward_rate=x -> 2exp(-x*x), name=:downward)
rate_result = simulate_gibbs(H; beta_phys, filter=rate_filter,
    times=[0.0, 0.1], diagnostics=:quick)
@assert rate_result.trajectory.all_converged

# CKG Gaussian mixtures require a matching normalised Gaussian OFT. Component
# centres and widths share physical beta; weights need not sum to one.
mixture = GaussianMixtureTransition(beta_phys; sigma=0.35,
    centers=(0.2, 0.6), weights=(0.3, 0.8))
ckg = simulate_gibbs(H; beta_phys, construction=KMS(), transition_weight=mixture,
    times=[0.0, 0.1], diagnostics=:quick)
@assert ckg.trajectory.all_converged

# For custom Time evolution first prepare explicit Fourier/coherent controls.
# Unknown integrated tails stay unknown even if the finite quadrature resolves.
p = prepare_filter_transform(f; window=16.0)
evidence = transform_values(p, [-1.0, 0.0, 1.0])
@assert evidence.tail_status == :unknown

# Names and parameter tuples alone do not define a closure. Portable
# reconstruction needs the matching registered definition or explicit resupply;
# consult the persistence section of the interface contract.
