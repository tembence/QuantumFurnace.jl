module QuantumFurnace

# --- Public API ---
export LiouvConfig, ThermalizeConfig, HamHam, TrottTrott, HotAlgorithmResults, HotSpectralResults, JumpOp,
       BohrPicture, EnergyPicture, TimePicture, TrotterPicture,
       run_liouvillian,
       run_thermalization,
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
include("jump_workers.jl")
include("structs.jl")
include("energy_picture.jl")
include("bohr_picture.jl")
include("time_picture.jl")
include("trotter_picture.jl")
include("timelike_tools.jl")
include("ofts.jl")
end