module QuantumFurnace

using Pkg
using Base
using Printf
using BSON
using Arpack
using LinearAlgebra
using SparseArrays
using Random
using Statistics: median
using ProgressMeter
using Roots
using DataStructures
using SpecialFunctions: erfc, besselj
using QuadGK
using Base.Threads
using FINUFFT
using KrylovKit
using LibGit2
using LsqFit
using Dates

# --- Public API ---

# --- Lindbladian ---
export run_lindblad, construct_lindbladian
export LindbladResults
export apply_lindbladian!, apply_adjoint_lindbladian!
export krylov_spectral_gap, apply_delta_channel!, apply_adjoint_delta_channel!

# --- Thermalize ---
export run_thermalize
export ThermalizeResults

# --- Krylov ---
export run_krylov_spectrum
export KrylovSpectrumResults

# --- Diagnostics ---
export EigenDecompositionResult, FixedPointResult, DefectResult, OverlapResult,
       SzSectorLabel, MultipletGroup, ExactDiagnosticsResult,
       extract_leading_eigendata, compute_fixed_point_distance,
       compute_anti_hermitian_defect, compute_overlap_coefficients,
       compute_sz_labels, detect_multiplets, run_exact_diagnostics
export SpectralModeDiagnostics, spectral_mode_diagnostics
export DiagnosticCheck, GibbsDiagnostics, state_diagnostics, workspace_diagnostics

# --- Discriminant ---
export DiscriminantBuffers, gibbs_fractional_powers,
       apply_discriminant!, apply_kms_parent!
export materialize_discriminant, materialize_discriminant!, materialize_kms_parent,
       hermitian_antihermitian_split, hermitian_antihermitian_split!
export DiscriminantSpectrum, discriminant_spectrum
export KMSParentSpectrum, kms_parent_spectrum
export DBVerificationResult, verify_detailed_balance

# --- KMS geometry diagnostics ---
export kms_inner_product, kms_norm, kms_variance, kms_dirichlet_form
export build_dense_superoperator
export spectral_gap_kms, max_dirichlet_rate_kms, intrinsic_mixing_ratio
export dissipator_one_to_one_norm_bound, dissipator_trace_alpha, hs_operator_norm
export hs_operator_norm_krylov

# --- Common ---
export Config, AbstractSimulation, Lindbladian, Thermalize, KrylovSpectrum,
       TensorNetworkSpectrum
export AbstractConstruction, KMS, GNS, DLL, with_coherent
export Workspace, LiouvillianScratch, ThermalizeScratch, KrylovScratch
export AbstractResults, save_result, load_result
export BohrDomain, EnergyDomain, TimeDomain, TrotterDomain
export HamHam, AbstractTrotter, TrottTrott, TrotterTriple, JumpOp
export LocalTerm1D, LocalHamiltonian1D, LocalJump1D, LocalDLLBlock1D
export BohrMPOControls, GibbsMPSControls, ParentGapControls
export BohrChebyshevApproximation, DLLGaussianChebyshevData,
       ExactDLLBohrPatch
export ParentErrorBound, ParentErrorLedger
export DLLFilterFrame, GeneratorClock, DLLParentProvenance
export DLLParentBundle, DLLParentLowEnergyState, DLLParentDiagnostics,
       DLLTensorNetworkResult
export DLL_PARENT_TARGET_LABELS, DLL_PARENT_GAP_LABELS,
       DLL_KERNEL_EVIDENCE_LABELS, PARENT_ERROR_EVIDENCE_LABELS
export build_dll_parent, build_dll_bohr_mpo, prepare_gibbs_purification,
       solve_dll_parent_gap, verify_dll_parent, validate_dll_tensor_network,
       dll_parent_provenance
export dll_gaussian_chebyshev_data, evaluate_bohr_chebyshev,
       apply_bohr_chebyshev, apply_dll_gaussian_chebyshev,
       exact_dll_bohr_patch, approximate_dll_bohr_patch
export trace_distance_h, trace_distance_nh, trace_norm_h, trace_norm_nh,
       fidelity, is_density_matrix, random_density_matrix,
       hermitianize!, validate_jump_pairing, prepare_jumps
export gibbs_state, gibbs_state_in_eigen,
       build_heis_1d, build_local_heis_1d, build_tfim_2d, load_hamiltonian,
       materialize_local_hamiltonian, materialize_local_jump,
       local_pauli_jumps_1d, validate_local_dll_tensor_network,
       create_bohr_dict, compute_trotter_error, make_trotter_for_config
# Physical and algorithm-side inverse-temperature conversion.
export beta_alg, beta_phys
export pick_transition, pick_gamma_sup, create_alpha, create_alpha_gns, create_alpha_gauss,
       create_f, create_f_gauss, check_alpha_skew_symmetry
# `default_smooth_s` remains internal; scripts may qualify it explicitly.
export B_time, B_trotter, B_bohr
export X, Y, Z, Had,
       pad_term, expm_pauli_padded, pauli_string_to_matrix, pauli_hamiltonian,
       trotterize, group_hamiltonian_terms
export validate_config!, prepare_gibbs_inputs
export register_t0_D, register_w0_D, register_r_D,
       register_t0_b_minus, register_w0_b_minus, register_r_b_minus,
       register_t0_b_plus, register_w0_b_plus, register_r_b_plus,
       register_M_D, register_M_b_minus, register_M_b_plus
export oft!

# --- DLL filters ---
export AbstractFilter, GaussianFilter, DLLGaussianFilter, DLLMetropolisFilter
export time_kernel, freq_kernel, filter_time_cutoff

# --- Multi-channel DLL ---
export DLLMultiChannelFilter, ShiftedSymmetricFilter, dll_multichannel_translates

# --- DLL dissipator helpers ---
export dll_lindblad_op_bohr, dll_lindblad_op_time

# --- DLL coherent helpers ---
export dll_coherent_op_bohr, dll_coherent_op_time
# `dll_coherent_kernel_bohr` is an internal reference kernel.

# --- DLL Kossakowski representation ---
export dll_kossakowski_bohr

# --- Exact dense DLL parent reference ---
export DenseDLLParentBlock, DLLIrreducibilityResult, DenseDLLParent
export dense_dll_parent_block, dense_dll_irreducibility, dense_dll_parent

# --- Lindbladian integration ---
export lindblad_action_integrate, discriminant_action_integrate, integrate_to_gibbs, sweep_mixing_times

# --- Krylov spectral dynamics ---
export predict_lindbladian_trajectory, predict_channel_trajectory

# --- Matrix-free superoperator distances ---
export PropagatorArm, propagator_trace_distance, propagator_fixed_point_distance
export lindbladian_arm, channel_arm
export slow_subspace_generator_distance
export arm_fixed_point, fixed_point_gibbs_distance
export discriminant_antiherm_norm, channel_discriminant_antiherm_norm,
       lindbladian_discriminant_antiherm_norm

# --- Channel sweeps ---
export sweep_channel_mixing

# --- Fitting and resource estimates ---
export fit_exponential_decay, FitResult
export fit_biexponential_decay, BiexpFitResult
export estimate_mixing_time, MixingTimeEstimate
export eigenmode_mixing_time
export SimulationTimeBudget, compute_simulation_time
export TrotterStepBudget, count_trotter_steps
export RxxBudget, estimate_rxx_count, load_rxx_table

# --- Empirical scaling laws ---
export ScalingFit, fit_scaling, predict_scaling, aicc_weights, compare_models,
       formula_string, scaling_fit_grid
# --- Internal Implementation ---
include("constants.jl")
include("jump_threading.jl")
include("pauli_tools.jl")
include("hamiltonian.jl")
include("hamiltonian_io.jl")
include("trotter_domain.jl")
include("filters.jl")
include("nufft.jl")
include("structs.jl")
include("dense_lindbladian_workspace.jl")
include("qi_tools.jl")
include("time_domain.jl")
include("ofts.jl")
include("energy_domain.jl")
include("bohr_domain.jl")
include("coherent.jl")
include("dll.jl")
include("dll_multichannel.jl")
include("local_models_1d.jl")
include("bohr_functional_calculus.jl")
include("config_validation.jl")
include("tensor_networks.jl")
include("dll_parent_patches.jl")
include("jump_workers.jl")
include("channel_construction.jl")
include("furnace_utensils.jl")
include("furnace.jl")
include("krylov_workspace.jl")
include("krylov_matvec.jl")
include("krylov_eigsolve.jl")
include("diagnostics.jl")
include("discriminant.jl")
include("dll_parent.jl")
include("kms_geometry.jl")
include("lindblad_dynamics.jl")
include("krylov_dynamics.jl")
include("mixing_sweeps.jl")
include("superop_distance.jl")
include("results.jl")
include("fitting.jl")
include("mixing.jl")
include("scaling_fit.jl")
include("simulation_time.jl")

# Sign-free stochastic-series-expansion baseline.
include("classical_qmc.jl")
using .ClassicalQMC
export SSEResult, run_sse, build_sse_heis_model, build_sse_tfim_model,
    sse_exact_reference, sse_reconstruction_error

end
