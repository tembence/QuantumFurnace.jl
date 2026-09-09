using Test
using LinearAlgebra

@testset "Serial Bohr gain and loss agree with spectral projectors" begin
    for T in (Float32, Float64)
        CT = Complex{T}
        ham = HamHam(CT.(0.25X + 0.35Z); beta_phys=T(0.8))
        source = CT[0.2+0.1im 0.8-0.3im; -0.2+0.7im 0.4-0.1im]
        operators = [CT.(X), source, Matrix(source')]
        jumps = [JumpOp(A, Matrix(ham.eigvecs' * A * ham.eigvecs),
            issymmetric(A), ishermitian(A)) for A in operators]
        cfg = Config(; sim=Thermalize(), domain=BohrDomain(), construction=KMS(),
            num_qubits=1, beta=beta_alg(ham, T(0.8)), sigma=inv(beta_alg(ham, T(0.8))),
            with_linear_combination=true, a=zero(T), s=T(0.25), delta=T(0.01))
        data = QuantumFurnace._precompute_data(cfg, ham)
        @test length(data.bohr_keys) < QuantumFurnace.OMEGA_THREAD_THRESHOLD
        reference = zeros(CT, 2, 2)
        state = CT[0.7 0.12+0.08im; 0.12-0.08im 0.3]
        weight = T(1.7) * data.gamma_norm_factor
        gain_references = Matrix{CT}[]
        for jump in jumps
            components = map(data.bohr_keys) do frequency
                component = zeros(CT, 2, 2)
                for index in ham.bohr_dict[frequency]
                    component[index] = jump.in_eigenbasis[index]
                end
                component
            end
            gain = zeros(CT, 2, 2)
            for i in eachindex(components), j in eachindex(components)
                reference .+= data.gamma_norm_factor * data.alpha(data.bohr_keys[i], data.bohr_keys[j]) *
                    (components[j]' * components[i])
                gain .+= cfg.delta * weight * data.alpha(data.bohr_keys[i], data.bohr_keys[j]) *
                    (components[i] * state * components[j]')
            end
            push!(gain_references, gain)
        end
        # Exercise cached index lists and the dictionary fallback independently.
        for precomputed in (data, (; alpha=data.alpha, gamma_norm_factor=data.gamma_norm_factor))
            scratch = QuantumFurnace.ThermalizeScratch(CT, 2)
            actual = QuantumFurnace._precompute_R(jumps, ham, cfg, precomputed, scratch)
            @test actual isa Matrix{CT}
            @test actual ≈ reference rtol=20eps(T) atol=20eps(T)
            @test actual ≈ actual' atol=20eps(T)
            for (jump, gain) in zip(jumps, gain_references)
                fill!(scratch.rho_jump, 2)
                QuantumFurnace._accumulate_rho_jump!(scratch, state, jump, ham, cfg, precomputed;
                    jump_weight_scaling=weight)
                @test scratch.rho_jump ≈ gain rtol=20eps(T) atol=20eps(T)*cfg.delta*weight
            end
        end
    end
end

# The retained density-matrix and Krylov channel paths share these construction
# helpers. Check every supported domain and both source-selection rate scalings.
@testset "Retained per-jump CPTP construction" begin
    for (domain, label, trotter) in [
        (BohrDomain(), "Bohr", nothing),
        (EnergyDomain(), "Energy", nothing),
        (TimeDomain(), "Time", nothing),
        (TrotterDomain(), "Trotter", TEST_TROTTER),
    ]
        config = make_config(Thermalize(), domain; delta=TEST_DELTA)
        jumps = domain isa TrotterDomain ? TEST_TROTTER_JUMPS : TEST_JUMPS
        ham_or_trott = trotter === nothing ? TEST_HAM : trotter
        precomputed_data = QuantumFurnace._precompute_data(config, ham_or_trott)
        n_jumps = length(jumps)
        p_jump = 1.0 / n_jumps
        identity = Matrix{ComplexF64}(I, DIM, DIM)

        for (selection, rescale) in ((:sweep, false), (:random, true))
            (; K0s, U_residuals) = QuantumFurnace._precompute_per_jump_channels(
                jumps, ham_or_trott, config, precomputed_data;
                rescale_by_inv_prob=rescale,
            )
            U_coherents = QuantumFurnace._precompute_coherent_unitary(
                jumps, TEST_HAM, config, precomputed_data;
                trotter=trotter,
                delta_scale=rescale ? 1.0 / p_jump : 1.0,
            )

            @test length(K0s) == n_jumps
            @test length(U_residuals) == n_jumps
            @test length(U_coherents) == n_jumps

            builder_scratch = QuantumFurnace.ThermalizeScratch(ComplexF64, DIM)
            max_completeness_err = 0.0
            for a in eachindex(jumps)
                R_bare = copy(@inferred QuantumFurnace._precompute_R(
                    [jumps[a]], ham_or_trott, config, precomputed_data,
                    builder_scratch,
                ))
                @test R_bare isa Matrix{ComplexF64}
                @test isapprox(R_bare, R_bare'; atol=1e-12)

                R = rescale ? R_bare ./ p_jump : R_bare
                built = @inferred QuantumFurnace._build_cptp_channel(R, TEST_DELTA)
                @test isapprox(K0s[a], built.K0; atol=1e-15)
                # The eigendecomposition may choose different signs/phases for
                # degenerate eigenvectors. The channel depends only on U†U.
                @test isapprox(
                    U_residuals[a]' * U_residuals[a],
                    built.U_residual' * built.U_residual;
                    atol=1e-14,
                )
                @test U_coherents[a] !== nothing

                completeness = K0s[a]' * K0s[a] + TEST_DELTA * R +
                    U_residuals[a]' * U_residuals[a]
                err = norm(completeness - identity)
                max_completeness_err = max(max_completeness_err, err)
                @test isapprox(completeness, identity; atol=1e-10)
            end

            # The retained density-matrix substep must stay concretely inferred.
            step_return = Core.Compiler.return_type(
                QuantumFurnace._apply_one_dm_substep!,
                Tuple{
                    Matrix{ComplexF64},
                    QuantumFurnace.ThermalizeScratch{ComplexF64},
                    typeof(jumps[1]),
                    Matrix{ComplexF64},
                    Matrix{ComplexF64},
                    Matrix{ComplexF64},
                    typeof(ham_or_trott),
                    typeof(config),
                    typeof(precomputed_data),
                    Float64,
                },
            )
            @test step_return === Nothing
            @info "Retained CPTP construction ($label, $selection)" n_jumps max_completeness_err threshold_atol=1e-10
        end
    end
end

@testset "Weak-measurement rate contraction" begin
    delta = 0.01
    identity2 = Matrix{ComplexF64}(I, 2, 2)

    # The residual is alpha^2 R(I-R), so a material violation of either
    # spectral bound is not a CPTP channel and must not be clamped away.
    @test_throws ArgumentError QuantumFurnace._build_cptp_channel(
        Matrix{ComplexF64}(2I, 2, 2), delta)
    @test_throws ArgumentError QuantumFurnace._build_cptp_channel(
        ComplexF64[-1e-6 0; 0 0.5], delta)
    @test_throws ArgumentError QuantumFurnace._build_cptp_channel(
        ComplexF64[0.5 1e-4im; 0 0.5], delta)
    @test_throws ArgumentError QuantumFurnace._build_cptp_channel(
        ComplexF64[NaN 0; 0 0.5], delta)

    # Explicitly bounded endpoint roundoff is cleaned up. Completeness stays
    # at machine precision rather than hiding a material residual defect.
    endpoint_roundoff = 2eps(Float64)
    R_roundoff = Matrix(Diagonal(ComplexF64[
        -endpoint_roundoff, 1 + endpoint_roundoff]))
    built = QuantumFurnace._build_cptp_channel(R_roundoff, delta)
    completeness = built.K0' * built.K0 + delta * R_roundoff +
        built.U_residual' * built.U_residual
    @test isapprox(completeness, identity2; atol=1e-14, rtol=0)

    # Stable alpha evaluation retains the delta/2 small-step limit.
    tiny_delta = 1e-14
    tiny = QuantumFurnace._build_cptp_channel(0.5identity2, tiny_delta)
    @test isapprox(tiny.alpha, tiny_delta / 2; rtol=1e-14, atol=0)
end

@testset "Density-matrix helper invariants" begin
    @test length(methods(is_density_matrix)) == 1

    rho = Hermitian(ComplexF64[0.75 0.0; 0.0 0.25])
    sigma = Hermitian(ComplexF64[0.25 0.0; 0.0 0.75])
    delta = rho - sigma

    @test is_density_matrix(rho)
    @test is_density_matrix(sigma)
    @test trace_norm_h(Hermitian(delta)) ≈ trace_norm_nh(Matrix(delta)) atol=1e-15
    @test trace_distance_h(rho, sigma) ≈ trace_norm_h(Hermitian(delta)) / 2 atol=1e-15
    @test trace_distance_nh(Matrix(rho), Matrix(sigma)) ≈
          trace_norm_nh(Matrix(delta)) / 2 atol=1e-15
    @test fidelity(rho, rho) ≈ 1.0 atol=1e-15
    @test fidelity(rho, sigma) ≈ 0.75 atol=1e-15

    @test_throws ArgumentError is_density_matrix(Hermitian(ComplexF64[1.1 0.0; 0.0 -0.1]))
    @test_throws ArgumentError fidelity(rho, Hermitian(ComplexF64[0.8 0.0; 0.0 0.8]))

    rho_roundoff = ComplexF64[1.0 + 1e-14 0.0; 0.0 -1e-14]
    pure = ComplexF64[1.0 0.0; 0.0 0.0]
    pure_orthogonal = ComplexF64[0.0 0.0; 0.0 1.0]
    @test is_density_matrix(rho_roundoff; atol=2e-14, rtol=0)
    @test is_density_matrix(Hermitian(rho_roundoff); atol=2e-14, rtol=0)
    @test fidelity(rho_roundoff, pure; atol=2e-14, rtol=0) ≈ 1.0 atol=1e-15
    @test fidelity(pure, pure) ≈ 1.0 atol=1e-15
    @test fidelity(pure, pure_orthogonal) ≈ 0.0 atol=1e-15

    dim_roundoff = 128
    negative_roundoff = 2e-15
    spectrum_roundoff = fill(-negative_roundoff, dim_roundoff)
    spectrum_roundoff[1] = 1 + (dim_roundoff - 1) * negative_roundoff
    rho_high_dim_roundoff = Matrix(Diagonal(ComplexF64.(spectrum_roundoff)))
    @test is_density_matrix(rho_high_dim_roundoff)
    @test fidelity(rho_high_dim_roundoff, rho_high_dim_roundoff) ≈ 1.0 atol=1e-14

    cumulative_negative = 5e-13
    invalid_spectrum = fill(-cumulative_negative, dim_roundoff)
    invalid_spectrum[1] = 1 + (dim_roundoff - 1) * cumulative_negative
    rho_cumulative_negative = Matrix(Diagonal(ComplexF64.(invalid_spectrum)))
    @test_throws ArgumentError is_density_matrix(rho_cumulative_negative)
    @test_throws ArgumentError fidelity(rho_cumulative_negative, rho_cumulative_negative)

    theta = 0.4
    rotation = ComplexF64[cos(theta) -sin(theta); sin(theta) cos(theta)]
    rho_near_singular = ComplexF64[1.0 - 1e-16 0.0; 0.0 1e-16]
    sigma_near_singular = rotation * rho_near_singular * rotation'
    fidelity_forward = fidelity(rho_near_singular, sigma_near_singular)
    fidelity_reverse = fidelity(sigma_near_singular, rho_near_singular)
    fidelity_analytic = real(tr(rho_near_singular * sigma_near_singular)) +
        2sqrt(real(det(rho_near_singular) * det(sigma_near_singular)))
    @test isapprox(fidelity_forward, fidelity_reverse; atol=1e-14, rtol=0)
    @test isapprox(fidelity_forward, fidelity_analytic; atol=1e-12, rtol=0)
    @test isapprox(fidelity_reverse, fidelity_analytic; atol=1e-12, rtol=0)
    @test 0.0 <= fidelity_forward <= 1.0

    @test_throws ArgumentError is_density_matrix(ones(ComplexF64, 2, 3))
    @test_throws ArgumentError is_density_matrix(ComplexF64[0.5 0.1; 0.0 0.5])
    @test_throws ArgumentError is_density_matrix(ComplexF64[0.6 0.0; 0.0 0.6])
    @test_throws ArgumentError is_density_matrix(ComplexF64[NaN 0.0; 0.0 NaN])
    @test_throws ArgumentError is_density_matrix(
        ComplexF64[1.0 + 1e-6 0.0; 0.0 -1e-6]; atol=2e-14, rtol=0)
    @test_throws ArgumentError is_density_matrix(Hermitian(
        ComplexF64[1.0 + 1e-6 0.0; 0.0 -1e-6]); atol=2e-14, rtol=0)
    @test_throws ArgumentError fidelity(rho, ones(ComplexF64, 3, 3) / 3)
    @test_throws ArgumentError fidelity(rho, ComplexF64[0.5 0.1; 0.0 0.5])
    @test_throws ArgumentError fidelity(
        ComplexF64[NaN 0.0; 0.0 NaN], pure; validate=false)
end
