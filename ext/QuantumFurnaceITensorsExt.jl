module QuantumFurnaceITensorsExt

using LinearAlgebra
using Random
using QuantumFurnace
using ITensors
using ITensorMPS

export fused_siteinds, qf_to_fused_permutation
export vec_to_fused, fused_to_vec, matrix_to_fused, fused_to_matrix
export superoperator_to_fused, fused_to_superoperator
export dense_to_mpo, mpo_to_dense, dense_to_mps, mps_to_dense
export physical_partial_trace
export ExactDLLParentReference
export local_operator_siteinds, local_hamiltonian_mpo, local_jump_mpo
export MPOCompressionRecord, DLLGaussianBohrMPO

include("QuantumFurnaceITensorsExt/conventions.jl")
include("QuantumFurnaceITensorsExt/dense_bridge.jl")
include("QuantumFurnaceITensorsExt/dense_parent.jl")
include("QuantumFurnaceITensorsExt/gap_solver.jl")
include("QuantumFurnaceITensorsExt/hamiltonian_mpo.jl")
include("QuantumFurnaceITensorsExt/bohr_mpo.jl")

end
