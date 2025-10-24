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
"""
    LiouvConfig

A configuration object that holds all the parameters for the core function: `run_liouvillian`, which constructs the Lindbladian of the thermalizing system.

# Fields
- `num_qubits::Int64`: The number of system qubits.
- `with_coherent::Bool`: The option to add (=true) or omit (=false) the coherent term in the Lindbladian.\nIf added, the target state of the evolution will be the exactly the Gibbs state, otherwise only approximately.
- `with_linear_combination::Bool`: The option to choose if we want to apply a convex combination of Lindbladians for a faster mixing. Could add extra complexities if the resulting transition function is not smooth. (See more in `Theory`).
- `a::Float64` and `b::Float64`: The parameters that specify the type of linear combination.
- `eta::Float64`: in the case of the Metropolis linear combination, η is an additional coefficient that determines the accuracy of the time picture approximation.
- `picture::AbstractPicture`: The picture the simulation runs in (`BOHR`, `ENERGY`, `TIME`, `TROTTER`). The choice of the picture represents the levels of approximations we need to get form theory down to quantum circuitry.
- `num_energy_bits::Int64`: Determines the how coarse the energy and time grid is and thus how accurate the approximations between each picture are.
- `t0::Float64` and `w0::Float64`: are the time and energy units we are working with in the Riemann summed integrals. Of course, the smaller the better but also the costlier, and the two are intertwined due to Fourier: ω₀t₀ = 2π / N.
- `num_trotter_steps_per_t0::Int64`: The number of Trotter steps used for a unit of time t₀.

## Currently possible linear combinations
(a, b) = {(0, 0) - Gaussian; (>0, 0) - Metropolis; (>0, >0) - Glauber}

## Available pictures:
The `picture` field can be set to one of the following options:
- **`BohrPicture()`**: The highest level picture where the jump operators and thus the Lindbladian are written in a decomposition of Bohr frequencies.
- **`EnergyPicture()`**: A level lower, in which the operators are approximated by energy integrals.
- **`TimePicture()`**: Another level lower, in which the energy approximates are written up as Fourier's of the temporal equals.
- **`TrotterPicture()`**: The lowest level, thus also the only one implementable on a quantum computer, in which all time evolutions are replaced via their Trotter series.
"""
@kwdef struct LiouvConfig
    num_qubits::Int64 
    with_coherent::Bool
    with_linear_combination::Bool
    picture::AbstractPicture
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