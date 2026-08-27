@testset "Dependency-free DLL tensor-network contract" begin
    function tn_local_hamiltonian(n::Int=2)
        terms = LocalTerm1D{Float64}[
            LocalTerm1D([site], Z, 0.2 + 0.1site, n; boundary=:open)
            for site in 1:n
        ]
        if n > 1
            push!(terms, LocalTerm1D([1, 2], kron(X, X), 0.4, n;
                                     boundary=:open))
        end
        return LocalHamiltonian1D(
            terms;
            num_sites=n,
            boundary=:open,
            coordinate_frame=:physical,
            global_shift=0.0,
            rescaling_factor=1.0,
            scale_provenance=:identity_physical_frame,
        )
    end

    function tn_config(
        n::Int,
        beta_alg_value::T,
        beta_phys_value::Union{Nothing, T},
        filter::AbstractFilter;
        sim::AbstractSimulation=TensorNetworkSpectrum(),
        domain::QuantumFurnace.AbstractDomain=BohrDomain(),
        construction::AbstractConstruction=DLL(),
    ) where {T<:AbstractFloat}
        return Config(;
            sim,
            domain,
            construction,
            num_qubits=n,
            with_linear_combination=true,
            beta=beta_alg_value,
            beta_phys=beta_phys_value,
            sigma=inv(beta_alg_value),
            a=zero(T),
            s=T(0.25),
            filter,
        )
    end

    function tn_blocks(n::Int, filter::AbstractFilter;
                       coefficient::Union{Nothing, Real}=nothing,
                       boundary::Symbol=:open)
        sources = local_pauli_jumps_1d(n; boundary)
        if coefficient !== nothing
            sources = [
                LocalJump1D(
                    source.sites, source.matrix, coefficient, n;
                    local_dim=source.local_dim, boundary,
                )
                for source in sources
            ]
        end
        return [LocalDLLBlock1D(source, filter) for source in sources]
    end

    @testset "plain load and generic backend hooks" begin
        loaded_packages = Set(package_id.name for package_id in keys(Base.loaded_modules))
        @test !("ITensors" in loaded_packages)
        @test !("ITensorMPS" in loaded_packages)
        @test length(methods(build_dll_parent)) == 0
        @test length(methods(build_dll_bohr_mpo)) == 0
        @test length(methods(prepare_gibbs_purification)) == 0
        @test length(methods(solve_dll_parent_gap)) == 0
        @test length(methods(verify_dll_parent)) == 0
        @test TensorNetworkSpectrum() isa AbstractSimulation
    end

    @testset "orthogonal typed controls" begin
        mpo = BohrMPOControls(
            scalar_tolerance=Float32(1e-5),
            recurrence_cutoff=Float32(1e-6),
            product_cutoff=Float32(1e-6),
            sum_cutoff=Float32(1e-6),
        )
        @test mpo isa BohrMPOControls{Float32}
        @test mpo.target_label == :bohr_polynomial_full_chain
        patch = BohrMPOControls(
            target_label=:finite_patch_bohr_surrogate, patch_radius=3)
        @test patch.patch_radius == 3
        @test_throws ArgumentError BohrMPOControls(
            target_label=:finite_patch_bohr_surrogate)
        @test_throws ArgumentError BohrMPOControls(patch_radius=2)
        @test_throws ArgumentError BohrMPOControls(target_label=:not_a_target)
        @test_throws ArgumentError BohrMPOControls(recurrence_maxdim=0)

        gibbs = GibbsMPSControls(
            time_step=Float32(0.1), cutoff=Float32(1e-6),
            maxdim_schedule=(8, 16))
        @test gibbs isa GibbsMPSControls{Float32, Tuple{Int, Int}}
        @test gibbs.maxdim_schedule == (8, 16)
        @test_throws ArgumentError GibbsMPSControls(time_step=0.0)
        @test_throws ArgumentError GibbsMPSControls(maxdim_schedule=())

        gap = ParentGapControls(
            cutoff=Float32(1e-6), residual_tolerance=Float32(1e-5),
            overlap_tolerance=Float32(1e-5),
            energy_tolerance=Float32(1e-6),
            penalty_factors=(Float32(0.5), Float32(2)),
            sectors=(:even, :odd))
        @test gap isa ParentGapControls{Float32}
        @test gap.penalty_factors == (Float32(0.5), Float32(2))
        @test gap.sectors == (:even, :odd)
        @test_throws ArgumentError ParentGapControls(num_starts=0)
        @test_throws ArgumentError ParentGapControls(overlap_tolerance=1.1)
        @test_throws ArgumentError ParentGapControls(penalty_factors=())

        for object in (mpo, patch, gibbs, gap)
            @test all(field_type -> field_type !== Any, fieldtypes(typeof(object)))
        end
    end

    @testset "typed error ledger preserves evidence boundaries" begin
        interval = ParentErrorBound(
            Float64, :spectral_interval;
            magnitude=0.0, evidence=:rigorous_bound,
            note="certified local enclosure")
        recurrence = ParentErrorBound(
            Float64, :q_recurrence;
            magnitude=2e-12, evidence=:floating_point_norm)
        sampled = ParentErrorBound(
            Float64, :probe_action;
            magnitude=3e-10, evidence=:sampled_evidence)
        ledger = ParentErrorLedger(
            Float64;
            spectral_interval=interval,
            mpo_recurrences=(recurrence,),
            block_assembly=(sampled,),
        )
        @test ledger.spectral_interval.evidence == :rigorous_bound
        @test ledger.mpo_recurrences[1].magnitude ≈ 2e-12 rtol=1e-14
        @test ledger.scalar_polynomial.evidence == :unmeasured
        @test ledger.scalar_polynomial.magnitude === nothing
        @test ledger.locality_radius.evidence == :unmeasured
        @test all(field_type -> field_type !== Any, fieldtypes(typeof(ledger)))
        @test_throws ArgumentError ParentErrorBound(
            Float64, :bad; magnitude=1e-3, evidence=:unmeasured)
        @test_throws ArgumentError ParentErrorBound(
            Float64, :bad; evidence=:floating_point_norm)
        @test_throws ArgumentError ParentErrorBound(
            Float64, :bad; magnitude=1e-3, evidence=:claimed)
    end

    @testset "configuration, frame, filter, and coherent gates" begin
        hamiltonian = tn_local_hamiltonian()
        beta_value = 0.7
        gaussian = DLLGaussianFilter(beta_value)
        config = tn_config(2, beta_value, beta_value, gaussian)
        blocks = tn_blocks(2, gaussian)
        full_controls = BohrMPOControls()
        @test validate_config!(config, hamiltonian) === nothing
        @test validate_dll_tensor_network(
            config, hamiltonian, blocks, full_controls) === nothing

        @test_throws ArgumentError validate_dll_tensor_network(
            config, hamiltonian, blocks, full_controls;
            include_coherent=false)

        mismatched = copy(blocks)
        mismatched[1] = LocalDLLBlock1D(
            mismatched[1].source, DLLGaussianFilter(beta_value + 0.1))
        @test_throws ArgumentError validate_dll_tensor_network(
            config, hamiltonian, mismatched, full_controls)

        missing_beta_phys = tn_config(2, beta_value, nothing, gaussian)
        @test_throws ArgumentError validate_config!(
            missing_beta_phys, hamiltonian)
        wrong_beta = tn_config(2, beta_value, beta_value / 2, gaussian)
        @test_throws ArgumentError validate_config!(wrong_beta, hamiltonian)

        wrong_domain = tn_config(
            2, beta_value, beta_value, gaussian; domain=TimeDomain())
        @test_throws ArgumentError validate_config!(wrong_domain)
        wrong_construction = tn_config(
            2, beta_value, beta_value, gaussian; construction=KMS())
        @test_throws ArgumentError validate_config!(wrong_construction)

        lindblad_config = tn_config(
            2, beta_value, beta_value, gaussian; sim=Lindbladian())
        @test_throws ArgumentError validate_dll_tensor_network(
            lindblad_config, hamiltonian, blocks, full_controls)

        metro = DLLMetropolisFilter(beta_value; S=2.0)
        metro_config = tn_config(2, beta_value, beta_value, metro)
        metro_blocks = tn_blocks(2, metro)
        @test_throws ArgumentError validate_dll_tensor_network(
            metro_config, hamiltonian, metro_blocks, full_controls)
        exact_controls = BohrMPOControls(target_label=:exact_bohr_dense)
        @test validate_dll_tensor_network(
            metro_config, hamiltonian, metro_blocks, exact_controls) === nothing

        negative_shift = ShiftedSymmetricFilter(metro, -0.2, 1.0)
        negative_shift_config = tn_config(
            2, beta_value, beta_value, negative_shift)
        negative_shift_blocks = tn_blocks(2, negative_shift)
        @test_throws ArgumentError validate_dll_tensor_network(
            negative_shift_config, hamiltonian, negative_shift_blocks,
            exact_controls)

        periodic_hamiltonian = build_local_heis_1d(
            3, [1.0, 1.0, 1.0]; seed=2, periodic=true)
        periodic_config = tn_config(3, beta_value, beta_value, gaussian)
        periodic_blocks = tn_blocks(3, gaussian; boundary=:periodic)
        @test_throws ArgumentError validate_dll_tensor_network(
            periodic_config, periodic_hamiltonian, periodic_blocks,
            full_controls)
    end

    @testset "unit and clock provenance has no implicit R conversion" begin
        physical_hamiltonian = tn_local_hamiltonian()
        R = 4.0
        shift = 0.125
        algorithm_hamiltonian = QuantumFurnace._to_algorithm_frame(
            physical_hamiltonian;
            rescaling_factor=R,
            shift,
            scale_provenance=:exact_dense,
        )
        beta_phys_value = 0.5
        beta_alg_value = R * beta_phys_value
        gaussian = DLLGaussianFilter(beta_alg_value)
        config = tn_config(
            2, beta_alg_value, beta_phys_value, gaussian)
        blocks = tn_blocks(2, gaussian)
        provenance = dll_parent_provenance(
            config, algorithm_hamiltonian, blocks;
            include_sweep_clock=true)
        @test provenance.beta_phys ≈ beta_phys_value rtol=1e-14
        @test provenance.beta_alg ≈ beta_alg_value rtol=1e-14
        @test provenance.hamiltonian_frame == :algorithm
        @test provenance.rescaling_factor ≈ R rtol=1e-14
        @test provenance.energy_shift ≈ shift rtol=1e-14
        @test provenance.filters == [
            DLLFilterFrame{Float64}(
                :dll_gaussian, nothing, 0.0, 1.0, :algorithm)]
        @test provenance.raw_generator_rate == 1.0
        @test provenance.proposal_normalisation == :qf_averaged
        @test provenance.proposal_amplitude ≈ inv(sqrt(6)) rtol=1e-14
        @test provenance.sweep_clock.label == :sweep_normalised
        @test provenance.sweep_clock.multiplier == 6.0
        @test provenance.sweep_clock.rate == 6.0
        @test provenance.legacy_clock === nothing

        legacy = dll_parent_provenance(
            config, algorithm_hamiltonian, blocks;
            legacy_clock_label=:legacy_global_R_clock,
            legacy_clock_multiplier=R)
        @test legacy.legacy_clock.label == :legacy_global_R_clock
        @test legacy.legacy_clock.multiplier == R
        @test legacy.legacy_clock.rate == R
        @test_throws ArgumentError dll_parent_provenance(
            config, algorithm_hamiltonian, blocks;
            legacy_clock_multiplier=R)

        extensive_blocks = tn_blocks(2, gaussian; coefficient=1.0)
        extensive = dll_parent_provenance(
            config, algorithm_hamiltonian, extensive_blocks;
            proposal_normalisation=:extensive)
        @test extensive.proposal_amplitude == 1.0
        @test extensive.sweep_clock === nothing
        @test_throws ArgumentError dll_parent_provenance(
            config, algorithm_hamiltonian, extensive_blocks;
            proposal_normalisation=:extensive,
            include_sweep_clock=true)

        metro = DLLMetropolisFilter(beta_alg_value; S=0.8)
        shifted = ShiftedSymmetricFilter(metro, 0.2, 0.75)
        shifted_config = tn_config(
            2, beta_alg_value, beta_phys_value, shifted)
        shifted_blocks = tn_blocks(2, shifted)
        shifted_provenance = dll_parent_provenance(
            shifted_config, algorithm_hamiltonian, shifted_blocks;
            controls=BohrMPOControls(target_label=:exact_bohr_dense))
        @test shifted_provenance.filters[1].support_radius ≈ 0.8 rtol=1e-14
        @test shifted_provenance.filters[1].shift ≈ 0.2 rtol=1e-14
        @test shifted_provenance.filters[1].weight ≈ 0.75 rtol=1e-14
        @test shifted_provenance.filters[1].coordinate_frame == :algorithm
        @test_throws ArgumentError DLLParentProvenance{Float64}(
            beta_phys_value,
            beta_alg_value,
            :physical,
            R,
            0.0,
            [DLLFilterFrame{Float64}(
                :dll_gaussian, nothing, 0.0, 1.0, :physical)],
            1.0,
            :qf_averaged,
            inv(sqrt(6)),
            nothing,
            nothing,
        )

        beta32 = Float32(0.7)
        gaussian32 = DLLGaussianFilter(beta32)
        config32 = tn_config(2, beta32, beta32, gaussian32)
        coefficient32 = Float32(inv(sqrt(6)))
        blocks32 = [
            LocalDLLBlock1D(
                LocalJump1D(
                    source.sites, Matrix{ComplexF32}(source.matrix),
                    coefficient32, 2; boundary=:open),
                gaussian32,
            )
            for source in local_pauli_jumps_1d(2)
        ]
        provenance32 = dll_parent_provenance(
            config32, physical_hamiltonian, blocks32)
        @test provenance32 isa DLLParentProvenance{Float32}
        @test provenance32.beta_phys == beta32
        @test provenance32.beta_alg == beta32
        @test provenance32.hamiltonian_frame == :physical
        @test provenance32.rescaling_factor == Float32(1)
        @test only(provenance32.filters).coordinate_frame == :physical
        @test provenance32.proposal_amplitude ≈ coefficient32 rtol=1e-6
    end

    @testset "parametric parent, state, diagnostics, and result records" begin
        hamiltonian = tn_local_hamiltonian()
        beta_value = 0.7
        gaussian = DLLGaussianFilter(beta_value)
        config = tn_config(2, beta_value, beta_value, gaussian)
        blocks = tn_blocks(2, gaussian)
        exact_controls = BohrMPOControls(target_label=:exact_bohr_dense)
        provenance = dll_parent_provenance(
            config, hamiltonian, blocks; controls=exact_controls)
        ledger = ParentErrorLedger(Float64)
        block_matrix = Matrix{ComplexF64}(I, 4, 4)
        parent = DLLParentBundle(
            :exact_bohr_dense,
            6 .* block_matrix,
            [copy(block_matrix) for _ in 1:6],
            [(source_index, 1) for source_index in 1:6],
            true,
            ledger,
        )
        low_state = @inferred DLLParentLowEnergyState(
            ComplexF64[1, 0, 0, 0];
            energy=0.2,
            residual=1e-12,
            variance=1e-14,
            kernel_overlaps=[0.0],
            bond_dimensions=[1, 2],
            converged=true,
        )
        diagnostics = @inferred DLLParentDiagnostics(
            observed_kernel_dimension=1,
            kernel_complete=true,
            kernel_tolerance=1e-12,
            kernel_evidence=:exact_dense_complete,
            primitivity_established=true,
            primitivity_provenance=:exact_filtered_commutant,
            hermiticity_defect=1e-15,
            minimum_energy=-1e-15,
            gibbs_energy=2e-15,
            gibbs_residual=3e-15,
            block_gibbs_residuals=fill(2e-15, 6),
            tighter_parent_residual=1e-12,
        )
        gibbs_controls = GibbsMPSControls()
        gap_controls = ParentGapControls()
        result = DLLTensorNetworkResult(
            config, parent, [low_state], diagnostics, provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls,
            gap_label=:exact_gap,
            gap_value=0.2,
            metadata=(backend=:dense_reference,),
        )
        @test result.gap_value ≈ 0.2 rtol=1e-14
        @test result.gap_lower === nothing
        @test result.gap_upper === nothing
        @test result.target_label == :exact_bohr_dense
        @test result.parent.total_parent isa Matrix{ComplexF64}
        @test eltype(result.low_energy_states).parameters[1] == Vector{ComplexF64}
        @test_throws ArgumentError typeof(result)(
            Val(:validated), config, parent, [low_state], diagnostics,
            provenance, exact_controls, gibbs_controls, gap_controls,
            :exact_bohr_dense, :certified_bracket, nothing, 0.1, 0.2,
            (backend=:dense_reference,),
        )
        for object in (parent, low_state, diagnostics, result)
            @test all(field_type -> field_type !== Any, fieldtypes(typeof(object)))
        end

        incomplete = DLLParentDiagnostics(
            observed_kernel_dimension=2,
            kernel_complete=false,
            kernel_tolerance=1e-5,
            kernel_evidence=:observed_only,
            primitivity_established=false,
            primitivity_provenance=:observed_only,
            hermiticity_defect=1e-6,
            minimum_energy=-1e-6,
            gibbs_energy=1e-5,
            gibbs_residual=1e-4,
            block_gibbs_residuals=fill(1e-4, 6),
        )
        observed_state = DLLParentLowEnergyState(
            ComplexF64[0, 1, 0, 0];
            energy=0.2,
            residual=1e-12,
            variance=1e-14,
            bond_dimensions=[2],
            converged=true,
        )
        observed_ground = DLLParentLowEnergyState(
            ComplexF64[1, 0, 0, 0];
            energy=0.05,
            residual=1e-12,
            variance=1e-14,
            bond_dimensions=[2],
            converged=true,
        )
        observed_edge = DLLParentLowEnergyState(
            ComplexF64[0, 0, 1, 0];
            energy=0.1,
            residual=1e-12,
            variance=1e-14,
            bond_dimensions=[2],
            converged=true,
        )
        observed = DLLTensorNetworkResult(
            config, parent,
            [observed_state, observed_ground, observed_edge],
            incomplete, provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls,
            gap_label=:observed_manifold_spacing,
            gap_value=0.1,
        )
        @test observed.gap_label == :observed_manifold_spacing
        observed_higher = DLLParentLowEnergyState(
            ComplexF64[0, 0, 1, 0];
            energy=0.3,
            residual=1e-12,
            variance=1e-14,
            bond_dimensions=[2],
            converged=true,
        )
        @test_throws ArgumentError DLLTensorNetworkResult(
            config, parent,
            [observed_higher, observed_ground, observed_state, observed_edge],
            incomplete, provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls,
            gap_label=:observed_manifold_spacing,
            gap_value=0.25,
        )
        unresolved_controls = ParentGapControls(energy_tolerance=1e-3)
        unresolved_state = DLLParentLowEnergyState(
            ComplexF64[0, 0, 0, 1];
            energy=0.0505,
            residual=1e-12,
            variance=1e-14,
            bond_dimensions=[2],
            converged=true,
        )
        unresolved_edge = DLLParentLowEnergyState(
            ComplexF64[0, 0, 1, 0];
            energy=0.0502,
            residual=1e-12,
            variance=1e-14,
            bond_dimensions=[2],
            converged=true,
        )
        @test_throws ArgumentError DLLTensorNetworkResult(
            config, parent,
            [observed_ground, unresolved_edge, unresolved_state],
            incomplete, provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls=unresolved_controls,
            gap_label=:observed_manifold_spacing,
            gap_value=3e-4,
        )
        @test_throws ArgumentError DLLTensorNetworkResult(
            config, parent, [low_state], incomplete, provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls,
            gap_label=:variational_upper_estimate,
            gap_value=0.15,
        )
        @test_throws ArgumentError DLLTensorNetworkResult(
            config, parent, [low_state], diagnostics, provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls,
            gap_label=:certified_bracket,
            gap_lower=0.1,
            gap_upper=0.2,
        )
        @test_throws ArgumentError DLLParentBundle(
            :exact_bohr_dense, block_matrix, [block_matrix], [(1, 1)],
            false, ledger)

        measured = name -> ParentErrorBound(
            Float64, name; magnitude=1e-12,
            evidence=:floating_point_norm)
        variational_ledger = ParentErrorLedger(
            Float64;
            total_sum_compression=measured(:total_sum_compression),
            gibbs_mps=measured(:gibbs_mps),
            eigensolver=measured(:eigensolver),
        )
        variational_parent = DLLParentBundle(
            :exact_bohr_dense,
            parent.total_parent,
            parent.block_parents,
            parent.block_keys,
            true,
            variational_ledger,
        )
        variational = DLLTensorNetworkResult(
            config, variational_parent, [low_state], diagnostics, provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls,
            gap_label=:variational_upper_estimate,
            gap_value=0.2,
        )
        @test variational.gap_lower === nothing
        @test variational.gap_upper ≈ 0.2 rtol=1e-14
        @test_throws ArgumentError DLLTensorNetworkResult(
            config, variational_parent, [low_state], diagnostics, provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls,
            gap_label=:variational_upper_estimate,
            gap_value=0.2,
            gap_lower=0.1,
        )
        unresolved_variational_state = DLLParentLowEnergyState(
            low_state.state;
            energy=5e-13,
            residual=1e-12,
            variance=1e-14,
            kernel_overlaps=[0.0],
            converged=true,
        )
        @test_throws ArgumentError DLLTensorNetworkResult(
            config, variational_parent, [unresolved_variational_state],
            diagnostics, provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls,
            gap_label=:variational_upper_estimate,
            gap_value=5e-13,
        )
        shifted_baseline_diagnostics = DLLParentDiagnostics(
            observed_kernel_dimension=1,
            kernel_complete=true,
            kernel_tolerance=1e-12,
            kernel_evidence=:verified_complete,
            primitivity_established=true,
            primitivity_provenance=:verified_backend,
            hermiticity_defect=1e-15,
            minimum_energy=0.05,
            gibbs_energy=0.05,
            gibbs_residual=3e-15,
            block_gibbs_residuals=fill(2e-15, 6),
            tighter_parent_residual=1e-12,
        )
        @test_throws ArgumentError DLLTensorNetworkResult(
            config, variational_parent, [low_state],
            shifted_baseline_diagnostics, provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls,
            gap_label=:variational_upper_estimate,
            gap_value=0.2,
        )

        unconverged = DLLParentLowEnergyState(
            low_state.state;
            energy=0.2,
            residual=10.0,
            variance=1.0,
            kernel_overlaps=[0.0],
            converged=false,
        )
        @test_throws ArgumentError DLLTensorNetworkResult(
            config, parent, [unconverged], diagnostics, provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls,
            gap_label=:exact_gap,
            gap_value=0.2,
        )
        @test_throws ArgumentError DLLTensorNetworkResult(
            config, parent, Any[low_state], diagnostics, provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls,
            gap_label=:exact_gap,
            gap_value=0.2,
        )

        truncated_parent = DLLParentBundle(
            :exact_bohr_dense, block_matrix, [block_matrix], [(1, 1)], true,
            ledger)
        @test_throws ArgumentError DLLTensorNetworkResult(
            config, truncated_parent, [low_state], diagnostics, provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls,
            gap_label=:exact_gap,
            gap_value=0.2,
        )
        short_diagnostics = DLLParentDiagnostics(
            observed_kernel_dimension=1,
            kernel_complete=true,
            kernel_tolerance=1e-12,
            kernel_evidence=:exact_dense_complete,
            primitivity_established=true,
            primitivity_provenance=:exact_filtered_commutant,
            hermiticity_defect=1e-15,
            minimum_energy=-1e-15,
            gibbs_energy=2e-15,
            gibbs_residual=3e-15,
            block_gibbs_residuals=[2e-15],
        )
        @test_throws ArgumentError DLLTensorNetworkResult(
            config, parent, [low_state], short_diagnostics, provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls,
            gap_label=:exact_gap,
            gap_value=0.2,
        )

        metro = DLLMetropolisFilter(beta_value; S=2.0)
        metro_config = tn_config(2, beta_value, beta_value, metro)
        metro_provenance = dll_parent_provenance(
            metro_config, hamiltonian, tn_blocks(2, metro);
            controls=exact_controls)
        @test_throws ArgumentError DLLTensorNetworkResult(
            config, parent, [low_state], diagnostics, metro_provenance;
            bohr_controls=exact_controls,
            gibbs_controls,
            gap_controls,
            gap_label=:exact_gap,
            gap_value=0.2,
        )

        wrong_slot = ParentErrorBound(
            Float64, :gibbs_mps; magnitude=0.0,
            evidence=:rigorous_bound)
        @test_throws ArgumentError ParentErrorLedger(
            Float64; spectral_interval=wrong_slot)
        @test_throws ArgumentError DLLParentBundle(
            :bohr_polynomial_full_chain,
            parent.total_parent,
            parent.block_parents,
            parent.block_keys,
            true,
            ledger,
        )
        @test_throws ArgumentError DLLParentBundle(
            :exact_bohr_dense,
            block_matrix,
            Any[block_matrix],
            [(1, 1)],
            true,
            ledger,
        )
        @test_throws ArgumentError DLLParentLowEnergyState{Any, Float64}(
            low_state.state, 0.2, -1.0, 0.0, Float64[], Int[], 1,
            :unrestricted, false)
        @test_throws ArgumentError DLLParentDiagnostics{Float64}(
            1, true, 1e-12, :observed_only, false, :none,
            0.0, 0.0, 0.0, 0.0, Float64[], nothing)
    end
end
