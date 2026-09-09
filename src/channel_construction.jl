"""
    _precompute_R(jumps, ham, config, precomputed, scratch)

Accumulate the channel rate operator into `scratch.R`, excluding the step size.
Hermitian sources use the folded grid with an explicit adjoint partner.
"""
function _precompute_R(jumps::AbstractVector{<:JumpOp}, ham::Union{HamHam,AbstractTrotter},
    config::Config{Thermalize,D}, precomputed, scratch::ThermalizeScratch{T},
) where {T<:Complex,D<:Union{EnergyDomain,TimeDomain,TrotterDomain}}
    fill!(scratch.R, 0)
    data = _frequency_oft_data(config, ham, precomputed)
    labels = precomputed.energy_labels
    prefactor = precomputed.oft_domain_prefactor * precomputed.gamma_norm_factor
    threaded = Threads.nthreads() > 1 && length(labels) >= OMEGA_THREAD_THRESHOLD
    # Reduce each source separately, preserving channel construction's source order.
    bases = Matrix{T}[scratch.jump_oft]
    hermitian = Bool[false]
    for jump in jumps
        bases[1] = jump.in_eigenbasis
        hermitian[1] = jump.hermitian
        _accumulate_frequency_loss!(scratch.R, bases, hermitian, labels,
            precomputed.transition, prefactor, data; threaded, buffers=(scratch.jump_oft,scratch.LdagL))
    end
    hermitianize!(scratch.R)
    return scratch.R
end

"""
    _precompute_R(jumps, hamiltonian, config::Config{Thermalize, BohrDomain}, precomputed_data, scratch)

Accumulate the Bohr-domain rate operator.

Math: \$R = sum_(nu_2) A_(nu_2)^dagger sum_(nu_1) alpha_(nu_1,nu_2) A_(nu_1)\$.
Per-jump probability rescaling is left to the caller.

# Returns
Hermitianized `scratch.R`.
"""
function _precompute_R(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam,
    config::Config{Thermalize, BohrDomain},
    precomputed_data,
    scratch::ThermalizeScratch{<:Complex},
)
    (; gamma_norm_factor) = precomputed_data

    bohr_keys = hasproperty(precomputed_data, :bohr_keys) ? precomputed_data.bohr_keys : collect(keys(hamiltonian.bohr_dict))
    bohr_is   = hasproperty(precomputed_data, :bohr_is)   ? precomputed_data.bohr_is   : nothing
    bohr_js   = hasproperty(precomputed_data, :bohr_js)   ? precomputed_data.bohr_js   : nothing

    n_keys = length(bohr_keys)
    if Threads.nthreads() > 1 && n_keys >= OMEGA_THREAD_THRESHOLD
        return _precompute_R_threaded_bohr!(jumps, hamiltonian, config, precomputed_data, scratch, bohr_keys, bohr_is, bohr_js)
    end

    fill!(scratch.R, 0)

    for jump in jumps
        _precompute_R_chunk_bohr!(scratch.R, scratch.jump_oft, jump, hamiltonian,
            precomputed_data, bohr_keys, bohr_is, bohr_js, eachindex(bohr_keys);
            gamma_norm_factor)
    end

    hermitianize!(scratch.R)
    return scratch.R
end

# --- BohrDomain threaded _precompute_R ---

function _precompute_R_threaded_bohr!(
    jumps::AbstractVector{<:JumpOp},
    hamiltonian::HamHam,
    config::Config{Thermalize, BohrDomain},
    precomputed_data,
    scratch::ThermalizeScratch{CT},
    bohr_keys::AbstractVector,
    bohr_is::Union{Nothing, Vector{Vector{Int}}},
    bohr_js::Union{Nothing, Vector{Vector{Int}}},
) where {CT<:Complex}
    dim = size(hamiltonian.data, 1)
    (; gamma_norm_factor) = precomputed_data

    fill!(scratch.R, 0)

    @inbounds for jump in jumps
        n_keys = length(bohr_keys)
        nt = min(Threads.nthreads(), n_keys)
        chunks = _partition_range(1:n_keys, nt)

        partials = [zeros(CT, dim, dim) for _ in chunks]
        operators = [similar(scratch.jump_oft) for _ in chunks]

        @sync for (idx, chunk) in enumerate(chunks)
            Threads.@spawn _precompute_R_chunk_bohr!(
                partials[idx], operators[idx], jump, hamiltonian, precomputed_data,
                bohr_keys, bohr_is, bohr_js, chunk;
                gamma_norm_factor=gamma_norm_factor)
        end

        for partial in partials
            scratch.R .+= partial
        end
    end

    hermitianize!(scratch.R)
    return scratch.R
end

function _precompute_R_chunk_bohr!(
    R::Matrix{CT},
    jump_oft::Matrix{CT},
    jump::JumpOp,
    hamiltonian::HamHam,
    precomputed_data,
    bohr_keys::AbstractVector,
    bohr_is::Union{Nothing, Vector{Vector{Int}}},
    bohr_js::Union{Nothing, Vector{Vector{Int}}},
    key_indices::AbstractUnitRange{Int};
    gamma_norm_factor::Real,
) where {CT<:Complex}
    dim = size(hamiltonian.data, 1)
    (; alpha) = precomputed_data

    # Keep the hot-loop matrix view concretely typed.
    in_eb = jump.in_eigenbasis::Matrix{CT}

    @inbounds for k in key_indices
        nu_2 = bohr_keys[k]
        @. jump_oft = alpha(hamiltonian.bohr_freqs, nu_2) * in_eb

        if bohr_is !== nothing
            is = bohr_is[k]
            js = bohr_js[k]
            @inbounds for t in eachindex(is)
                i = is[t]
                j = js[t]
                v = conj(in_eb[i, j]) * gamma_norm_factor
                @inbounds for q in 1:dim
                    R[j, q] += v * jump_oft[i, q]
                end
            end
        else
            indices = hamiltonian.bohr_dict[nu_2]
            @inbounds for idx in indices
                i = idx[1]
                j = idx[2]
                v = conj(in_eb[i, j]) * gamma_norm_factor
                @inbounds for q in 1:dim
                    R[j, q] += v * jump_oft[i, q]
                end
            end
        end
    end

    return nothing
end
