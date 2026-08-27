# These bridges are exact small-system references. They deliberately reject
# chains beyond the dense overlap regime so production code cannot silently
# replace a local MPO construction by dense factorisation.

const DENSE_BRIDGE_MAX_SITES = 4
const DENSE_BRIDGE_MAX_CHAIN_DIM = 4096

function _validate_fused_sites(sites)
    isempty(sites) && throw(ArgumentError("fused site collection must be nonempty."))
    n = length(sites)
    n <= DENSE_BRIDGE_MAX_SITES || throw(ArgumentError(
        "dense ITensor bridges are reference-only and support at most " *
        "$DENSE_BRIDGE_MAX_SITES sites, got $n."))

    fused_dim = ITensors.dim(first(sites))
    physical_dim = isqrt(fused_dim)
    physical_dim^2 == fused_dim || throw(ArgumentError(
        "each fused site dimension must be a perfect square, got $fused_dim."))
    all(site -> ITensors.dim(site) == fused_dim, sites) || throw(ArgumentError(
        "all fused sites must have the same dimension $fused_dim."))

    chain_dim = _checked_power(fused_dim, n)
    chain_dim <= DENSE_BRIDGE_MAX_CHAIN_DIM || throw(ArgumentError(
        "dense ITensor bridges support chain dimension at most " *
        "$DENSE_BRIDGE_MAX_CHAIN_DIM, got $chain_dim."))
    return (; n, physical_dim, fused_dim, chain_dim)
end

"""
    dense_to_mpo(matrix, sites) -> MPO

Factor a dense fused-order matrix into an effectively exact MPO. This is a
reference-only bridge limited to at most four sites and chain dimension 4096;
it is not a scalable parent construction.
"""
function dense_to_mpo(matrix::AbstractMatrix, sites)
    dimensions = _validate_fused_sites(sites)
    expected = dimensions.chain_dim
    size(matrix) == (expected, expected) || throw(DimensionMismatch(
        "matrix has size $(size(matrix)), expected ($expected, $expected)."))
    operator_indices = vcat(ITensors.prime.(sites), ITensors.dag.(sites))
    tensor = ITensors.ITensor(
        reshape(matrix, ntuple(_ -> dimensions.fused_dim, 2 * dimensions.n)),
        operator_indices...,
    )
    return ITensorMPS.MPO(tensor, sites; cutoff=0.0, maxdim=expected)
end

"""Reconstruct a dense fused-order matrix from a reference MPO."""
function mpo_to_dense(mpo::ITensorMPS.MPO, sites)
    dimensions = _validate_fused_sites(sites)
    length(mpo) == dimensions.n || throw(DimensionMismatch(
        "MPO has length $(length(mpo)), expected $(dimensions.n)."))
    full_tensor = reduce(*, mpo)
    ordered = Array(
        full_tensor,
        ITensors.prime.(sites)...,
        ITensors.dag.(sites)...,
    )
    return reshape(ordered, dimensions.chain_dim, dimensions.chain_dim)
end

"""
    dense_to_mps(vector, sites) -> MPS

Factor a dense fused-order vector into an effectively exact MPS. This is a
reference-only bridge with the same size limits as [`dense_to_mpo`](@ref).
"""
function dense_to_mps(vector::AbstractVector, sites)
    dimensions = _validate_fused_sites(sites)
    length(vector) == dimensions.chain_dim || throw(DimensionMismatch(
        "vector has length $(length(vector)), expected $(dimensions.chain_dim)."))
    tensor = ITensors.ITensor(
        reshape(vector, ntuple(_ -> dimensions.fused_dim, dimensions.n)),
        sites...,
    )
    return ITensorMPS.MPS(
        tensor, sites; cutoff=0.0, maxdim=dimensions.chain_dim)
end

"""Contract a reference MPS to a dense vector in fused array order."""
function mps_to_dense(state::ITensorMPS.MPS, sites)
    dimensions = _validate_fused_sites(sites)
    length(state) == dimensions.n || throw(DimensionMismatch(
        "MPS has length $(length(state)), expected $(dimensions.n)."))
    return vec(Array(reduce(*, state), sites...))
end

"""
    physical_partial_trace(fused_state, n; physical_dim=2)

Trace the auxiliary copy out of a fused purification. The input norm is
preserved: the returned density matrix has trace `norm(fused_state)^2`.
"""
function physical_partial_trace(
    fused_state::AbstractVector,
    n::Integer;
    physical_dim::Integer=2,
)
    amplitude = fused_to_matrix(fused_state, n; physical_dim=physical_dim)
    return amplitude * adjoint(amplitude)
end

"""Trace the auxiliary copy out of a small reference MPS purification."""
function physical_partial_trace(state::ITensorMPS.MPS, sites)
    dimensions = _validate_fused_sites(sites)
    fused_state = mps_to_dense(state, sites)
    return physical_partial_trace(
        fused_state,
        dimensions.n;
        physical_dim=dimensions.physical_dim,
    )
end
