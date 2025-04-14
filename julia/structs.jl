
using Base

@enum Picture BOHR ENERGY TIME TROTTER

# Let's keep this structure, and have the "give w0 for desired energy integral error" type of config optimization
# before the construct_liouvillian function
@kwdef struct LiouvConfig
    num_qubits::Int64 
    with_coherent::Bool
    with_linear_combination::Bool
    picture::Picture
    beta::Float64
    a::Float64
    b::Float64
    num_energy_bits::Int64 = -1
    t0::Float64 = -1.
    w0::Float64 = -1.
    eta::Float64 = -1.
    num_trotter_steps_per_t0::Int64 = -1
end

# struct mutable ThermConfig
# end

mutable struct HamHam
    data::Matrix{ComplexF64}
    bohr_freqs::Union{Matrix{Float64}, Nothing}
    base_terms::Vector{Vector{String}}
    base_coeffs::Vector{Float64}
    symbreak_terms::Union{Vector{String}, Nothing}
    symbreak_coeffs::Union{Vector{Float64}, Nothing}
    eigvals::Vector{Float64}
    eigvecs::Matrix{ComplexF64}
    nu_min::Float64  # Smallest bohr frequency
    shift::Float64
    rescaling_factor::Float64
end

mutable struct JumpOp
    data::Matrix{ComplexF64}
    in_eigenbasis::Matrix{ComplexF64}
    bohr_decomp::Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}
    unique_freqs::Vector{Float64}
    in_trotter_basis::Matrix{ComplexF64}
    orthogonal::Bool
end

mutable struct LiouvLiouv
    data::Matrix{ComplexF64}
    steady_state::Matrix{ComplexF64}
    spectral_gap::Float64
    mixing_time_bound::Float64
end

#TODO: add unitary that transforms from energy to trotter basis
mutable struct TrottTrott
    t0::Float64
    num_trotter_steps_per_t0::Rational{Int64}
    eigvals_t0::Vector{ComplexF64}
    eigvecs::Matrix{ComplexF64}
    trafo_from_eigen_to_trotter::Matrix{ComplexF64}
end

mutable struct HotAlgorithmResults
    evolved_dm::Matrix{ComplexF64}
    distances_to_gibbs::Vector{Float64}
    time_steps::Vector{Float64}
end

mutable struct HotSpectralResults
    fixed_point::Matrix{ComplexF64}
    spectral_gap::Float64
    gibbs_ineigen::Matrix{ComplexF64}
end