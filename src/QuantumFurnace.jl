module QuantumFurnace

using JLD2, BSON
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
       BohrPicture, EnergyPicture, TimePicture, TrotterPicture
export run_liouvillian, run_thermalization
export generate_filename, validate_config!, create_trotter, compute_trotter_error, gibbs_state, gibbs_state_in_eigen,
       create_bohr_dict, pad_term, pick_transition, create_hamham

# --- Internal Implementation ---
include("structs.jl")
include("hamiltonian.jl")
include("qi_tools.jl")
include("misc_tools.jl")
include("ofts.jl")
include("errors.jl")
include("jump_workers.jl")
include("coherent.jl")
include("bohr_picture.jl")
include("energy_picture.jl")
include("time_picture.jl")
include("timelike_tools.jl")
include("trotter_picture.jl")
include("oven_utensils.jl")
include("oven.jl")

end