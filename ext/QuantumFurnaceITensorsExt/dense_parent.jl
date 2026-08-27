# Exact dense-to-MPO overlap path for the DLL parent. This deliberately stops
# at four sites; the scalable Bohr functional calculus is implemented later.

"""
    ExactDLLParentReference

Small-system reference joining the exact dense DLL parent to the fused-site
ITensor representation. `dense` remains the authoritative eigenbasis object;
`parent_fused` and `block_parents_fused` are computational-basis operators in
the extension's doubled-site order. The MPOs in `bundle` are reference-only
dense factorisations.
"""
struct ExactDLLParentReference{
    T<:AbstractFloat,
    F<:QuantumFurnace.AbstractFilter,
    P<:QuantumFurnace.DLLParentBundle,
    S,
}
    dense::QuantumFurnace.DenseDLLParent{T, F}
    bundle::P
    sites::Vector{ITensors.Index{Int}}
    parent_fused::Matrix{Complex{T}}
    block_parents_fused::Vector{Matrix{Complex{T}}}
    eigenvalues::Vector{T}
    kernel_vectors_fused::Matrix{Complex{T}}
    gibbs_vector_fused::Vector{Complex{T}}
    gibbs_state::S
    diagnostics::QuantumFurnace.DLLParentDiagnostics{T}
end

function _validate_exact_dll_sites(sites, n::Int, physical_dim::Int)
    dimensions = _validate_fused_sites(sites)
    dimensions.n == n || throw(DimensionMismatch(
        "received $(dimensions.n) fused sites, expected $n."))
    dimensions.physical_dim == physical_dim || throw(DimensionMismatch(
        "fused sites imply physical dimension $(dimensions.physical_dim), " *
        "expected $physical_dim."))
    return dimensions
end

function _to_computational_superoperator(
    operator::AbstractMatrix,
    eigenvectors::AbstractMatrix,
)
    basis_rotation = kron(conj(eigenvectors), eigenvectors)
    return basis_rotation * operator * adjoint(basis_rotation)
end

function _to_fused_superoperator(
    operator::AbstractMatrix,
    eigenvectors::AbstractMatrix,
    n::Int,
    physical_dim::Int,
)
    computational = _to_computational_superoperator(operator, eigenvectors)
    return superoperator_to_fused(
        computational, n; physical_dim=physical_dim)
end

function _validate_exact_dll_inputs(
    config::QuantumFurnace.Config{
        QuantumFurnace.TensorNetworkSpectrum,
        QuantumFurnace.BohrDomain,
        QuantumFurnace.DLL,
        T,
    },
    local_hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    local_blocks::AbstractVector{<:QuantumFurnace.LocalDLLBlock1D},
    hamiltonian::QuantumFurnace.HamHam{T},
    jumps::AbstractVector{<:QuantumFurnace.JumpOp},
) where {T<:AbstractFloat}
    controls = QuantumFurnace.BohrMPOControls(
        target_label=:exact_bohr_dense)
    QuantumFurnace.validate_dll_tensor_network(
        config, local_hamiltonian, local_blocks, controls)
    QuantumFurnace.validate_config!(config, hamiltonian)
    hamiltonian.periodic && throw(ArgumentError(
        "the one-dimensional DLL tensor-network overlap path supports only OBC."))

    local_matrix = Matrix(QuantumFurnace.materialize_local_hamiltonian(
        local_hamiltonian))
    scale = max(T(norm(local_matrix)), T(norm(hamiltonian.data)), one(T))
    tolerance = T(200) * T(size(local_matrix, 1)) * eps(T) * scale
    isapprox(
        local_matrix,
        hamiltonian.data;
        atol=tolerance,
        rtol=tolerance,
    ) || throw(ArgumentError(
        "LocalHamiltonian1D does not materialise to the supplied HamHam in " *
        "the declared coordinate frame."))

    length(local_blocks) == length(jumps) || throw(ArgumentError(
        "the exact overlap path requires one dense JumpOp per LocalDLLBlock1D " *
        "source; got $(length(jumps)) jumps and $(length(local_blocks)) blocks."))
    for index in eachindex(local_blocks, jumps)
        source = getfield(local_blocks[index], :source)
        expected = Matrix{Complex{T}}(
            QuantumFurnace.materialize_local_jump(source))
        actual = jumps[index]
        isapprox(actual.data, expected; atol=tolerance, rtol=tolerance) ||
            throw(ArgumentError(
                "JumpOp $index does not match its LocalDLLBlock1D source."))
        expected_eigen = adjoint(hamiltonian.eigvecs) * expected *
                         hamiltonian.eigvecs
        isapprox(
            actual.in_eigenbasis,
            expected_eigen;
            atol=tolerance,
            rtol=tolerance,
        ) || throw(ArgumentError(
            "JumpOp $index eigenbasis data does not match its local source."))
    end
    return nothing
end

function _exact_dense_parent_ledger(
    ::Type{T},
    total_error::Real,
    block_errors::AbstractVector{<:Real},
    block_keys::AbstractVector{Tuple{Int, Int}},
) where {T<:AbstractFloat}
    assembly = Tuple(
        QuantumFurnace.ParentErrorBound(
            T,
            Symbol("dense_mpo_block_", source, "_", channel);
            magnitude=block_errors[index],
            evidence=:floating_point_norm,
            note="dense reference MPO reconstruction Frobenius norm",
        )
        for (index, (source, channel)) in pairs(block_keys)
    )
    return QuantumFurnace.ParentErrorLedger(
        T;
        block_assembly=assembly,
        total_sum_compression=QuantumFurnace.ParentErrorBound(
            T,
            :total_sum_compression;
            magnitude=total_error,
            evidence=:floating_point_norm,
            note="dense reference total-MPO reconstruction Frobenius norm",
        ),
    )
end

"""
    build_dll_parent(config, local_hamiltonian, local_blocks,
                     hamiltonian, jumps; sites=nothing,
                     physical_dim=2) -> ExactDLLParentReference

Construct the exact `N <= 4` Bohr-domain DLL parent with the canonical coherent
correction, rotate it from QF's Hamiltonian eigenbasis to the computational
basis, perfect-shuffle it into fused doubled-site order, and factor the total
and every source/channel block into effectively exact reference MPOs.

This method is an overlap validator. It rejects periodic chains and sizes
beyond the guarded dense bridge; it is not the scalable tensor-network parent
builder.
"""
function QuantumFurnace.build_dll_parent(
    config::QuantumFurnace.Config{
        QuantumFurnace.TensorNetworkSpectrum,
        QuantumFurnace.BohrDomain,
        QuantumFurnace.DLL,
        T,
    },
    local_hamiltonian::QuantumFurnace.LocalHamiltonian1D{T},
    local_blocks::AbstractVector{<:QuantumFurnace.LocalDLLBlock1D},
    hamiltonian::QuantumFurnace.HamHam{T},
    jumps::AbstractVector{<:QuantumFurnace.JumpOp};
    sites=nothing,
    physical_dim::Integer=2,
) where {T<:AbstractFloat}
    _validate_exact_dll_inputs(
        config, local_hamiltonian, local_blocks, hamiltonian, jumps)
    physical_dim_int = Int(physical_dim)
    physical_dim_int > 0 || throw(ArgumentError("physical_dim must be positive."))
    n = config.num_qubits
    physical_dim_int^n == size(hamiltonian.data, 1) || throw(DimensionMismatch(
        "physical_dim^n does not match the Hamiltonian dimension."))
    n <= DENSE_BRIDGE_MAX_SITES || throw(ArgumentError(
        "the exact dense DLL parent bridge supports at most " *
        "$DENSE_BRIDGE_MAX_SITES sites, got $n."))

    filter = config.filter
    filter isa QuantumFurnace.AbstractFilter || throw(ArgumentError(
        "the exact DLL parent requires an explicit admissible DLL filter."))
    dense = QuantumFurnace.dense_dll_parent(jumps, hamiltonian, filter)
    expected_block_keys = Tuple{Int, Int}[
        (source_index, channel_index)
        for (source_index, block) in pairs(local_blocks)
        for channel_index in eachindex(getfield(block, :channels))
    ]
    actual_block_keys = Tuple{Int, Int}[
        (block.source_index, block.channel_index) for block in dense.blocks
    ]
    actual_block_keys == expected_block_keys || error(
        "dense DLL parent block accounting disagrees with LocalDLLBlock1D channels.")
    fused_sites = sites === nothing ?
        fused_siteinds(n; physical_dim=physical_dim_int) : collect(sites)
    _validate_exact_dll_sites(fused_sites, n, physical_dim_int)

    parent_fused = Matrix{Complex{T}}(_to_fused_superoperator(
        dense.parent, hamiltonian.eigvecs, n, physical_dim_int))
    block_parents_fused = Matrix{Complex{T}}[
        _to_fused_superoperator(
            block.parent, hamiltonian.eigvecs, n, physical_dim_int)
        for block in dense.blocks
    ]

    total_mpo = dense_to_mpo(parent_fused, fused_sites)
    block_mpos = [dense_to_mpo(block, fused_sites) for block in block_parents_fused]
    block_keys = actual_block_keys
    total_error = T(norm(mpo_to_dense(total_mpo, fused_sites) - parent_fused))
    block_errors = T[
        norm(mpo_to_dense(block_mpos[index], fused_sites) -
             block_parents_fused[index])
        for index in eachindex(block_mpos)
    ]
    ledger = _exact_dense_parent_ledger(
        T, total_error, block_errors, block_keys)
    bundle = QuantumFurnace.DLLParentBundle(
        :exact_bohr_dense,
        total_mpo,
        block_mpos,
        block_keys,
        true,
        ledger,
    )

    hermitian_parent = Hermitian((dense.parent + adjoint(dense.parent)) / T(2))
    eigendecomposition = eigen(hermitian_parent)
    kernel_indices = findall(
        value -> abs(value) <= dense.spectrum.kernel_tolerance,
        eigendecomposition.values,
    )
    length(kernel_indices) == dense.spectrum.kernel_count || error(
        "exact parent kernel eigenspace disagrees with its stored spectrum.")
    basis_rotation = kron(conj(hamiltonian.eigvecs), hamiltonian.eigvecs)
    permutation = qf_to_fused_permutation(n; physical_dim=physical_dim_int)
    kernel_vectors_fused = Matrix{Complex{T}}(
        (basis_rotation * eigendecomposition.vectors[:, kernel_indices])[permutation, :])

    powers = QuantumFurnace.gibbs_fractional_powers(hamiltonian.gibbs)
    gibbs_eigen = zeros(Complex{T}, size(dense.parent, 1))
    d = length(hamiltonian.eigvals)
    @inbounds for index in 1:d
        gibbs_eigen[(index - 1) * d + index] = powers.sigma_half[index]
    end
    gibbs_vector_fused = Vector{Complex{T}}(
        (basis_rotation * gibbs_eigen)[permutation])
    gibbs_state = dense_to_mps(gibbs_vector_fused, fused_sites)

    block_gibbs_residuals = T[
        norm(block.parent * gibbs_eigen) for block in dense.blocks
    ]
    gibbs_residual = T(norm(dense.parent * gibbs_eigen))
    gibbs_energy = T(real(dot(gibbs_eigen, dense.parent * gibbs_eigen)))
    diagnostics = QuantumFurnace.DLLParentDiagnostics(
        observed_kernel_dimension=dense.spectrum.kernel_count,
        kernel_complete=true,
        kernel_tolerance=dense.spectrum.kernel_tolerance,
        kernel_evidence=:exact_dense_complete,
        primitivity_established=dense.spectrum.primitivity_established,
        primitivity_provenance=:exact_filtered_commutant,
        hermiticity_defect=dense.spectrum.hermiticity_defect,
        minimum_energy=dense.spectrum.minimum_eigenvalue,
        gibbs_energy=gibbs_energy,
        gibbs_residual=gibbs_residual,
        block_gibbs_residuals=block_gibbs_residuals,
    )

    return ExactDLLParentReference{
        T,
        typeof(filter),
        typeof(bundle),
        typeof(gibbs_state),
    }(
        dense,
        bundle,
        fused_sites,
        parent_fused,
        block_parents_fused,
        Vector{T}(eigendecomposition.values),
        kernel_vectors_fused,
        gibbs_vector_fused,
        gibbs_state,
        diagnostics,
    )
end

"""Return a copy of the exact dense Gibbs-purification reference MPS."""
function QuantumFurnace.prepare_gibbs_purification(
    reference::ExactDLLParentReference,
)
    return copy(reference.gibbs_state)
end

"""Return the exact dense kernel, positivity, and Gibbs diagnostics."""
function QuantumFurnace.verify_dll_parent(
    reference::ExactDLLParentReference,
)
    return reference.diagnostics
end
