# API reference

## Physical-input interface

```@docs
QuantumFurnace.simulate_gibbs
QuantumFurnace.GibbsSimulationResult
QuantumFurnace.pauli_hamiltonian
QuantumFurnace.prepare_gibbs_inputs
QuantumFurnace.prepare_jumps
```

See the [contract](api_contract.md) for physical units, construction/domain
limits, diagnostic evidence, and portable reconstruction.

## Core types

```@docs
QuantumFurnace.Config
QuantumFurnace.HamHam
QuantumFurnace.JumpOp
QuantumFurnace.LindbladResults
QuantumFurnace.ThermalizeResults
```

## Hamiltonians

```@docs
QuantumFurnace.load_hamiltonian
QuantumFurnace.build_heis_1d
QuantumFurnace.build_tfim_2d
QuantumFurnace.beta_alg
QuantumFurnace.beta_phys
```

## Dynamics

```@docs
QuantumFurnace.construct_lindbladian
QuantumFurnace.run_lindblad
QuantumFurnace.run_thermalize
QuantumFurnace.predict_lindbladian_trajectory
QuantumFurnace.krylov_spectral_gap
QuantumFurnace.eigenmode_mixing_time
```

## Filters and validation

```@docs
QuantumFurnace.GaussianFilter
QuantumFurnace.DLLGaussianFilter
QuantumFurnace.DLLMetropolisFilter
QuantumFurnace.validate_config!
```

## Custom filters and rates

```@docs
QuantumFurnace.KMSFilter
QuantumFurnace.FrequencyFilter
QuantumFurnace.RateFilter
QuantumFurnace.TimeFilter
QuantumFurnace.prepare_filter_transform
QuantumFurnace.GaussianMixtureTransition
QuantumFurnace.CKGJointKernel
QuantumFurnace.ckg_to_dll
```

## Results

```@docs
QuantumFurnace.save_result
QuantumFurnace.load_result
QuantumFurnace.register_filter!
```
