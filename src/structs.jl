# Pictures
abstract type AbstractPicture end

struct BohrPicture <: AbstractPicture end
struct EnergyPicture <: AbstractPicture end
struct TimePicture <: AbstractPicture end
struct TrotterPicture <: AbstractPicture end

struct OFTCaches
    prefactors::Vector{ComplexF64}
    U::Diagonal{ComplexF64, Vector{ComplexF64}}
    temp_op::Matrix{ComplexF64}
    
    function OFTCaches(dim::Int)
        prefactors = zeros(ComplexF64, 0) # Will be resized later
        U = Diagonal(zeros(ComplexF64, dim))
        temp_op = zeros(ComplexF64, dim, dim)
        new(prefactors, U, temp_op)
    end
end

# For Linear Maps (for single node)
struct JumpCaches
    jump_1::AbstractMatrix{ComplexF64}
    jump_2_dag_jump_1::AbstractMatrix{ComplexF64}
    temp1::AbstractMatrix{ComplexF64}

    function JumpCaches(dim::Int)
        jump_1 = zeros(ComplexF64, dim, dim)
        jump_2_dag_jump_1 = zeros(ComplexF64, dim, dim)
        temp1 = zeros(ComplexF64, dim, dim)
        new(jump_1, jump_2_dag_jump_1, temp1)
    end

end

# Let's keep this structure, and have the "give w0 for desired energy integral error" type of config optimization
# before the construct_liouvillian function
@kwdef struct LiouvConfig
    num_qubits::Int64 
    with_coherent::Bool
    with_linear_combination::Bool
    picture::AbstractPicture  #!
    beta::Float64
    a::Float64 = nothing
    b::Float64 = nothing
    num_energy_bits::Int64 = -1
    t0::Float64 = -1.
    w0::Float64 = -1.
    eta::Float64 = -1.
    num_trotter_steps_per_t0::Int64 = -1
end

@kwdef struct ThermalizeConfig
    num_qubits::Int64 
    with_coherent::Bool
    with_linear_combination::Bool
    picture::AbstractPicture
    beta::Float64
    a::Float64
    b::Float64
    num_energy_bits::Int64 = -1
    t0::Float64 = -1.
    w0::Float64 = -1.
    eta::Float64 = -1.
    num_trotter_steps_per_t0::Int64 = -1

    # For thermalization the configs:
    mixing_time::Float64
    delta::Float64
    unravel::Bool = false
end

mutable struct HamHam
    data::Matrix{ComplexF64}
    bohr_freqs::Union{Matrix{Float64}, Nothing}
    bohr_dict::Union{Dict{Float64, Vector{CartesianIndex{2}}}, Nothing}
    base_terms::Vector{Vector{String}}
    base_coeffs::Vector{Float64}
    symbreak_terms::Union{Vector{String}, Nothing}
    symbreak_coeffs::Union{Vector{Float64}, Nothing}
    eigvals::Vector{Float64}
    eigvecs::Matrix{ComplexF64}
    nu_min::Float64  # Smallest bohr frequency
    shift::Float64
    rescaling_factor::Float64
    periodic::Bool
    gibbs::Hermitian{ComplexF64, Matrix{ComplexF64}}
end

mutable struct JumpOp  #TODO: deleted 2 fields, didnt adjust definitions in mains
    data::Matrix{ComplexF64}
    in_eigenbasis::Matrix{ComplexF64}  #! Now either energy or trotter eigenbasis
    orthogonal::Bool
end

mutable struct LiouvLiouv
    data::Matrix{ComplexF64}
    steady_state::Matrix{ComplexF64}
    spectral_gap::Float64
    mixing_time_bound::Float64
end

mutable struct TrottTrott
    t0::Float64
    num_trotter_steps_per_t0::Float64
    eigvals_t0::Vector{ComplexF64}
    eigvecs::Matrix{ComplexF64}
    trafo_from_eigen_to_trotter::Matrix{ComplexF64}
end

@kwdef struct HotAlgorithmResults
    evolved_dm::Matrix{ComplexF64}
    distances_to_gibbs::Vector{Float64}
    time_steps::Vector{Float64}
    hamiltonian::HamHam
    trotter::Union{TrottTrott,Nothing} = nothing
    config::ThermalizeConfig
end

@kwdef struct HotSpectralResults
    data::Matrix{ComplexF64}  #! Remove when space will matter
    fixed_point::Matrix{ComplexF64}
    lambda_2::ComplexF64    # For spectral gap
    hamiltonian::HamHam
    trotter::Union{TrottTrott,Nothing} = nothing
    config::LiouvConfig
end