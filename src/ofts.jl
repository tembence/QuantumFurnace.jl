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
