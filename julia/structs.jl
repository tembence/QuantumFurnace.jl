
#TODO: Think this Config struct over, not sure about it.
struct Config
    num_qubits::Int64
    num_energy_bits::Int64
    w0::Float64
    mixing_time::Float64
    delta::Float64
    beta::Float64
    initial_dm::Matrix{ComplexF64}
    with_coherent::Bool
    jumps_are_random::Bool
    furnace::Symbol
    transition_cutoff::Float64
    coherent_cutoff::Float64
end

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

#TODO: Add orthogonality
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

mutable struct TrottTrott
    t0::Float64
    num_trotter_steps_per_t0::Rational{Int64}
    eigvals_t0::Vector{ComplexF64}
    eigvecs::Matrix{ComplexF64}
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