struct NUFFTPrefactors{T<:AbstractFloat, A<:AbstractArray{Complex{T}, 3}}
    data::A
    energy_labels::Vector{T}
    energy_to_index::Dict{T, Int}
end

"""Return exact unique values and an inverse map to the original array."""
function _unique_with_invmap(v::AbstractVector{<:AbstractFloat})
    uniq = unique(v)
    T = eltype(v)
    idx = Dict{T,Int}(u => i for (i,u) in enumerate(uniq))
    invmap = Vector{Int32}(undef, length(v))
    @inbounds for k in eachindex(v)
        invmap[k] = Int32(idx[v[k]])
    end
    return uniq, invmap
end

function _prepare_oft_nufft_prefactors(
    bohr_freqs::AbstractMatrix{<:AbstractFloat},
    time_labels::Vector{<:AbstractFloat},
    energy_labels::Vector{<:AbstractFloat},
    filter::AbstractFilter;
    eps::Float64 = 1e-12,
    nthreads::Int = 1,
)
    dim1, dim2 = size(bohr_freqs)
    @assert dim1 == dim2
    dim = dim1

    # Hermitian-source kernels evaluate nonpositive labels through their positive
    # partners. An untruncated even grid contains -N*w0/2 but not +N*w0/2.
    # Cache any missing partners explicitly, retaining original slice indices;
    # the caller's quadrature grid and its weights remain unchanged.
    energy_labels = unique(vcat(energy_labels, abs.(energy_labels)))

    # Store prefactors using the energy-grid precision.
    T = eltype(energy_labels)

    # FINUFFT requires Float64 coordinates.
    bohr_freqs_f64 = Float64.(bohr_freqs)
    time_labels_f64 = Float64.(time_labels)
    energy_labels_f64 = Float64.(energy_labels)

    # Transform each distinct Bohr frequency once.
    bohr_flat = vec(bohr_freqs_f64)
    unique_bohr_flat, invmap = _unique_with_invmap(bohr_flat)

    # Kernels may be real or complex; FINUFFT consumes ComplexF64 weights.
    base_weights = ComplexF64.(time_kernel.(Ref(filter), time_labels_f64))
    input_weights = Matrix{ComplexF64}(undef, length(time_labels_f64), 1)
    out_nufft = Matrix{ComplexF64}(undef, length(unique_bohr_flat), 1)

    CT = Complex{T}
    prefactors = Array{CT}(undef, dim, dim, length(energy_labels))

    # Math: type-3 NUFFT uses $sum_j c_j exp(i s_k x_j)$.
    plan = FINUFFT.finufft_makeplan(3, 1, +1, 1, eps; dtype=Float64, nthreads=nthreads)
    # In one dimension, `xj` are sources and `s` are targets.
    empty = Float64[]
    FINUFFT.finufft_setpts!(plan,
        time_labels_f64,    # xj
        empty,              # yj (unused)
        empty,              # zj (unused)
        unique_bohr_flat,   # s (targets)
        empty,              # t (unused)
        empty               # u (unused)
    )

    @inbounds for (k, omega) in enumerate(energy_labels_f64)
        # Math: $c_j = f(t_j) exp(-i omega t_j)$.
        @fastmath @. input_weights[:, 1] = base_weights * cis(-omega * time_labels_f64)

        FINUFFT.finufft_exec!(plan, input_weights, out_nufft)

        # Restore the full matrix ordering.
        @views full_bohr_prefac_omega = prefactors[:, :, k]
        @inbounds for p in eachindex(invmap)
            full_bohr_prefac_omega[p] = CT(out_nufft[Int(invmap[p]), 1])
        end
    end

    FINUFFT.finufft_destroy!(plan)

    energy_to_index = Dict{T,Int}(omega => i for (i, omega) in enumerate(energy_labels))
    return NUFFTPrefactors(prefactors, energy_labels, energy_to_index)
end

"""Return a non-allocating view of the prefactor matrix at `omega`."""
@inline function _prefactor_view(nufft_prefactors::NUFFTPrefactors, omega)
    k = nufft_prefactors.energy_to_index[omega]
    return @view nufft_prefactors.data[:, :, k]
end

"""
    fourier_sum(nodes, weights, targets; sign=1, backend=:direct, tolerance=1e-12)

Evaluate `sum(weights[j]*exp(sign*im*nodes[j]*targets[k]))`. Weights must
already contain the quadrature weights and any Fourier normalisation.
`:direct` retains promoted input precision; deterministic single-threaded
`:finufft` explicitly uses Float64/ComplexF64 and rejects higher precision.
This evaluates a finite sum, not a continuum error certificate.
"""
function fourier_sum(nodes::AbstractVector{<:Real},weights::AbstractVector{<:Number},
    targets::AbstractVector{<:Real};sign::Int=1,backend::Symbol=:direct,tolerance::Real=1e-12)
    length(nodes)==length(weights) || throw(DimensionMismatch("nodes and weights must match."))
    sign in (-1,1) || throw(ArgumentError("Fourier sign must be ±1."))
    backend in (:direct,:finufft) || throw(ArgumentError("backend must be :direct or :finufft."))
    all(isfinite,nodes) && all(isfinite,weights) && all(isfinite,targets) ||
        throw(ArgumentError("Fourier samples must be finite."))
    isfinite(tolerance) && tolerance>0 || throw(ArgumentError("tolerance must be finite and positive."))
    T = promote_type(typeof(float(zero(eltype(nodes)))),typeof(real(float(zero(eltype(weights))))),typeof(float(zero(eltype(targets)))))
    if backend == :direct
        values = zeros(Complex{T},length(targets))
        for k in eachindex(targets),j in eachindex(nodes,weights)
            values[k] += Complex{T}(weights[j])*cis(T(sign)*T(nodes[j])*T(targets[k]))
        end
        return values
    end
    T in (Float32,Float64) || throw(ArgumentError("FINUFFT supports Float64 working precision; use backend=:direct for $T."))
    (isempty(nodes) || isempty(targets)) && return zeros(ComplexF64,length(targets))
    plan = FINUFFT.finufft_makeplan(3,1,sign,1,Float64(tolerance);dtype=Float64,nthreads=1)
    try
        FINUFFT.finufft_setpts!(plan,Float64.(nodes),Float64[],Float64[],Float64.(targets),Float64[],Float64[])
        output = Vector{ComplexF64}(undef,length(targets))
        FINUFFT.finufft_exec!(plan,ComplexF64.(weights),output)
        return output
    finally
        FINUFFT.finufft_destroy!(plan)
    end
end
