module QuantumFurnace

using BSON
using Arpack
using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using ClusterManagers
using Roots
using DataStructures
using SpecialFunctions: erfc
using QuadGK
using Base

# --- Public API ---
export LiouvConfig, ThermalizeConfig, HamHam, TrottTrott, HotAlgorithmResults, HotSpectralResults, JumpOp,
       BohrDomain, EnergyDomain, TimeDomain, TrotterDomain
export run_liouvillian, run_thermalization
export generate_filename, validate_config!, create_trotter, compute_trotter_error, gibbs_state, gibbs_state_in_eigen,
       create_bohr_dict, pad_term, pick_transition, create_hamham, find_ideal_heisenberg, create_alpha

# --- Internal Implementation ---
include("structs.jl")
include("hamiltonian.jl")
include("qi_tools.jl")
include("misc_tools.jl")
include("ofts.jl")
include("errors.jl")
include("jump_workers.jl")
include("coherent.jl")
include("bohr_domain.jl")
include("energy_domain.jl")
include("time_domain.jl")
include("timelike_tools.jl")
include("trotter_domain.jl")
include("oven_utensils.jl")
include("oven.jl")

end