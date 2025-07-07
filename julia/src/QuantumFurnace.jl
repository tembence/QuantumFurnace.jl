module QuantumFurnace

# --- Public API ---
export LiouvConfig, ThermalizeConfig, HamHam, TrottTrott, HotAlgorithmResults, HotSpectralResults, JumpOp,
       BohrPicture, EnergyPicture, TimePicture, TrotterPicture,
       run_liouvillian,
       run_thermalization_fast,
       generate_filename,
       validate_config!,
       create_trotter, compute_trotter_error,
       gibbs_state, gibbs_state_in_eigen,
       create_bohr_dict, pad_term

# --- Internal Implementation ---
include("hamiltonian.jl")
include("qi_tools.jl")
include("misc_tools.jl")
include("structs.jl")
include("oven.jl")

include("errors.jl")
include("oven_utensils.jl")
include("coherent.jl")
end