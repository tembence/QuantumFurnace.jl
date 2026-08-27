# QF uses column stacking. For a global matrix X, the row or ket label is
# therefore faster than the column or bra label in vec(X). QF's Kronecker
# convention makes physical site N fastest inside each global ket/bra label.
#
# ITensor Array conversion makes the first listed site index fastest. We retain
# physical site labels 1:N and perfect-shuffle the dense data so site 1 is the
# fastest fused-chain index. Within each local fused index the ket label is
# fastest: local = ket + d * bra in zero-based notation.

function _checked_power(base::Int, exponent::Int)
    exponent >= 0 || throw(ArgumentError("exponent must be nonnegative."))
    value = 1
    for _ in 1:exponent
        value = Base.checked_mul(value, base)
    end
    return value
end

function _validate_chain_dimensions(n::Integer, physical_dim::Integer)
    n > 0 || throw(ArgumentError("n must be positive, got $n."))
    physical_dim > 0 || throw(ArgumentError(
        "physical_dim must be positive, got $physical_dim."))
    return Int(n), Int(physical_dim)
end

"""
    fused_siteinds(n; physical_dim=2)

Create one ITensor `Qudit` index of dimension `physical_dim^2` per physical
site. Array order follows the extension convention: site 1 is fastest, and
the ket label is faster than the bra label within each fused site.
"""
function fused_siteinds(n::Integer; physical_dim::Integer=2)
    n_int, d = _validate_chain_dimensions(n, physical_dim)
    fused_dim = Base.checked_mul(d, d)
    return ITensorMPS.siteinds("Qudit", n_int; dim=fused_dim)
end

"""
    qf_to_fused_permutation(n; physical_dim=2)

Return `perm` such that `fused = qf_vec[perm]`. `qf_vec` is QF column-stacked
data with the full ket index fastest. `fused` uses ITensor dense-array order,
where fused site 1 is fastest and local zero-based state
`ket + physical_dim * bra` has the ket index fastest.
"""
function qf_to_fused_permutation(n::Integer; physical_dim::Integer=2)
    n_int, d = _validate_chain_dimensions(n, physical_dim)
    global_dim = _checked_power(d, n_int)
    fused_local_dim = Base.checked_mul(d, d)
    total_dim = Base.checked_mul(global_dim, global_dim)

    qf_site_strides = Vector{Int}(undef, n_int)
    fused_site_strides = Vector{Int}(undef, n_int)
    for site in 1:n_int
        qf_site_strides[site] = _checked_power(d, n_int - site)
        fused_site_strides[site] = _checked_power(fused_local_dim, site - 1)
    end

    permutation = Vector{Int}(undef, total_dim)
    for bra_global in 0:(global_dim - 1), ket_global in 0:(global_dim - 1)
        qf_index = ket_global + global_dim * bra_global + 1
        fused_offset = 0
        for site in 1:n_int
            ket = (ket_global ÷ qf_site_strides[site]) % d
            bra = (bra_global ÷ qf_site_strides[site]) % d
            local_state = ket + d * bra
            fused_offset += local_state * fused_site_strides[site]
        end
        permutation[fused_offset + 1] = qf_index
    end
    return permutation
end

function _validate_doubled_length(
    length_value::Integer,
    n::Integer,
    physical_dim::Integer,
)
    n_int, d = _validate_chain_dimensions(n, physical_dim)
    global_dim = _checked_power(d, n_int)
    expected = Base.checked_mul(global_dim, global_dim)
    length_value == expected || throw(DimensionMismatch(
        "doubled vector has length $length_value, expected $expected for " *
        "n=$n_int and physical_dim=$d."))
    return n_int, d, global_dim
end

"""Convert a QF column-stacked vector to fused doubled-site array order."""
function vec_to_fused(
    vector::AbstractVector,
    n::Integer;
    physical_dim::Integer=2,
)
    n_int, d, _ = _validate_doubled_length(length(vector), n, physical_dim)
    permutation = qf_to_fused_permutation(n_int; physical_dim=d)
    return collect(vector[permutation])
end

"""Convert a fused doubled-site vector to QF column-stacked order."""
function fused_to_vec(
    vector::AbstractVector,
    n::Integer;
    physical_dim::Integer=2,
)
    n_int, d, _ = _validate_doubled_length(length(vector), n, physical_dim)
    permutation = qf_to_fused_permutation(n_int; physical_dim=d)
    return collect(vector[invperm(permutation)])
end

"""Column-stack a global physical matrix and convert it to fused order."""
function matrix_to_fused(
    matrix::AbstractMatrix,
    n::Integer;
    physical_dim::Integer=2,
)
    n_int, d = _validate_chain_dimensions(n, physical_dim)
    global_dim = _checked_power(d, n_int)
    size(matrix) == (global_dim, global_dim) || throw(DimensionMismatch(
        "matrix has size $(size(matrix)), expected ($global_dim, $global_dim)."))
    return vec_to_fused(vec(matrix), n_int; physical_dim=d)
end

"""Convert a fused doubled-site vector to its global physical matrix."""
function fused_to_matrix(
    vector::AbstractVector,
    n::Integer;
    physical_dim::Integer=2,
)
    n_int, d, global_dim = _validate_doubled_length(
        length(vector), n, physical_dim)
    return reshape(fused_to_vec(vector, n_int; physical_dim=d), global_dim, global_dim)
end

"""Perfect-shuffle a QF column-stacked superoperator into fused order."""
function superoperator_to_fused(
    matrix::AbstractMatrix,
    n::Integer;
    physical_dim::Integer=2,
)
    n_int, d = _validate_chain_dimensions(n, physical_dim)
    permutation = qf_to_fused_permutation(n_int; physical_dim=d)
    expected = length(permutation)
    size(matrix) == (expected, expected) || throw(DimensionMismatch(
        "superoperator has size $(size(matrix)), expected ($expected, $expected)."))
    return Matrix(matrix[permutation, permutation])
end

"""Undo [`superoperator_to_fused`](@ref) and recover QF ordering."""
function fused_to_superoperator(
    matrix::AbstractMatrix,
    n::Integer;
    physical_dim::Integer=2,
)
    n_int, d = _validate_chain_dimensions(n, physical_dim)
    permutation = qf_to_fused_permutation(n_int; physical_dim=d)
    expected = length(permutation)
    size(matrix) == (expected, expected) || throw(DimensionMismatch(
        "fused superoperator has size $(size(matrix)), expected " *
        "($expected, $expected)."))
    inverse = invperm(permutation)
    return Matrix(matrix[inverse, inverse])
end
