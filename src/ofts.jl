"""
    oft!(out, eigenbasis, bohr_freqs, energy, inv_4sigma2) -> nothing

Compute one energy-domain operator Fourier component in place.

# Arguments
- `out`: Destination matrix.
- `eigenbasis`: Jump operator in the Hamiltonian eigenbasis.
- `bohr_freqs`: Matrix of Bohr frequencies.
- `energy`: Target energy label.
- `inv_4sigma2`: Gaussian exponent factor `1 / (4 sigma^2)`.

# Returns
`nothing`.
"""
@inline function oft!(
    out::Matrix{T},
    eigenbasis::Matrix{T},
    bohr_freqs::Matrix{<:Real},
    energy::Real,
    inv_4sigma2::Real,
) where {T<:Complex}
    # Math: $A(omega)_(i j) = A_(i j) exp(-(omega - Delta_(i j))^2 / (4 sigma^2))$.
    @. out = eigenbasis * exp(-(energy - bohr_freqs)^2 * inv_4sigma2)
    return nothing
end

# The numerical argument retains the legacy Gaussian fast path. A compiled
# full transform already includes its amplitude normalisation.
_energy_oft_kernel(config) = _is_joint_ckg(config) ? config.transition_weight.oft : 1.0 / (4 * config.sigma^2)
@inline function oft!(out::Matrix{CT}, eigenbasis::Matrix{CT}, bohr_freqs::Matrix{<:Real},
    energy::Real, filter::TabulatedCKGOFT) where {CT<:Complex}
    row=filter.labels[energy]
    @inbounds for i in eachindex(out)
        nu=bohr_freqs[i]
        col=filter.frequencies[iszero(nu) ? zero(nu) : nu]
        out[i]=eigenbasis[i]*filter.values[row,col]
    end
    nothing
end

# A frequency loop loads either an analytic energy component or a retained
# time-transform sample. Non-Hermitian sources retain signed label indices.
@inline _frequency_oft!(out, A, data::Tuple, w, label_index, folded) =
    oft!(out, A, data[1], w, data[2])

@inline function _frequency_oft!(out, A, data::NUFFTPrefactors, w, label_index, folded)
    index = folded ? data.energy_to_index[w] : label_index
    sample = @view data.data[:, :, index]
    @. out = A * sample
    return nothing
end

_frequency_oft_data(config::Config{<:Any,EnergyDomain}, ham, precomputed) =
    (ham.bohr_freqs, _energy_oft_kernel(config))
_frequency_oft_data(config::Config{<:Any,D}, ham, precomputed) where {D<:Union{TimeDomain,TrotterDomain}} =
    precomputed.oft_nufft_prefactors
