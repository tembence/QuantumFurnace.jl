using Test
using LinearAlgebra
using Random

using QuantumFurnace

# Loading core alone must not activate the optional extension. ITensors alone
# is also insufficient because the extension contract requires both packages.
@test Base.get_extension(QuantumFurnace, :QuantumFurnaceITensorsExt) === nothing
using ITensors
@test Base.get_extension(QuantumFurnace, :QuantumFurnaceITensorsExt) === nothing
using ITensorMPS

const QFITensors = Base.get_extension(
    QuantumFurnace, :QuantumFurnaceITensorsExt)
@test QFITensors !== nothing

function exact_dll_config(
    simulation::AbstractSimulation,
    n::Int,
    beta_algorithm::T,
    beta_physical::T,
    filter::AbstractFilter,
) where {T<:AbstractFloat}
    return Config(;
        sim=simulation,
        domain=BohrDomain(),
        construction=DLL(),
        num_qubits=n,
        with_linear_combination=true,
        beta=beta_algorithm,
        beta_phys=beta_physical,
        sigma=inv(beta_algorithm),
        a=zero(T),
        s=T(0.25),
        filter,
    )
end

function exact_dll_jumps(
    hamiltonian::HamHam{T},
    n::Int;
    normalized::Bool=true,
) where {T<:AbstractFloat}
    local_sources = local_pauli_jumps_1d(n; boundary=:open)
    jumps = JumpOp[]
    sizehint!(jumps, length(local_sources))
    for source in local_sources
        coefficient = normalized ? T(source.coefficient) : one(T)
        local_source = LocalJump1D(
            source.sites,
            Matrix{Complex{T}}(source.matrix),
            coefficient,
            n;
            local_dim=source.local_dim,
            boundary=:open,
        )
        operator = Matrix{Complex{T}}(materialize_local_jump(local_source))
        eigen_operator = adjoint(hamiltonian.eigvecs) * operator *
                         hamiltonian.eigvecs
        push!(jumps, JumpOp(
            operator,
            eigen_operator,
            operator == transpose(operator),
            ishermitian(operator),
        ))
    end
    return jumps
end

function exact_dll_blocks(
    n::Int,
    filter::AbstractFilter;
    normalized::Bool=true,
)
    return [
        LocalDLLBlock1D(
            LocalJump1D(
                source.sites,
                source.matrix,
                normalized ? source.coefficient : 1.0,
                n;
                local_dim=source.local_dim,
                boundary=:open,
            ),
            filter,
        )
        for source in local_pauli_jumps_1d(n; boundary=:open)
    ]
end

function exact_dll_fixture(
    n::Int,
    filter_builder::F;
    beta_phys::Float64=0.5,
    seed::Int=46,
) where {F}
    raw = build_heis_1d(
        n,
        [1.0, 1.0, 1.0];
        seed,
        periodic=false,
        disorder_strength=0.1,
    )
    hamiltonian = HamHam(raw; beta_phys)
    beta_algorithm = beta_alg(hamiltonian, beta_phys)
    filter = filter_builder(beta_algorithm)
    local_hamiltonian = build_local_heis_1d(
        n,
        [1.0, 1.0, 1.0];
        seed,
        periodic=false,
        disorder_strength=0.1,
        coordinate_frame=:algorithm,
        rescaling_factor=hamiltonian.rescaling_factor,
        shift=hamiltonian.shift,
        scale_provenance=:exact_dense,
    )
    config = exact_dll_config(
        TensorNetworkSpectrum(), n, beta_algorithm, beta_phys, filter)
    jumps = exact_dll_jumps(hamiltonian, n)
    blocks = exact_dll_blocks(n, filter)
    return (;
        raw,
        hamiltonian,
        local_hamiltonian,
        beta_algorithm,
        beta_phys,
        filter,
        config,
        jumps,
        blocks,
    )
end

include("test_conventions.jl")
include("test_exact_dll_parent.jl")
include("test_exact_dll_dmrg.jl")
include("test_bohr_mpo.jl")
