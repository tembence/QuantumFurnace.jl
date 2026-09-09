"""
DM reference tests: detailed balance and domain error hierarchy.

These tests verify the Gibbs fixed point independently in every construction
domain. No ordering between independent quadrature and Trotter errors is
assumed; those errors can cancel.
"""

@testset "Dense Gibbs fixed point by domain (4-qubit)" begin
    distances = Dict{Symbol, Float64}()
    thresholds = Dict(:bohr => 1e-12, :energy => 1e-12, :time => 1e-12,
        :trotter => 1e-7)

    for (name, domain) in [(:bohr, BohrDomain()), (:energy, EnergyDomain()),
                            (:time, TimeDomain()), (:trotter, TrotterDomain())]
        config = make_config(Lindbladian(),domain)
        trotter_obj = (domain isa TrotterDomain) ? TEST_TROTTER : nothing
        domain_jumps = (domain isa TrotterDomain) ? TEST_TROTTER_JUMPS : TEST_JUMPS
        liouv = construct_lindbladian(domain_jumps, config, TEST_HAM; trotter=trotter_obj)

        # Full eigendecomposition (256x256 dense matrix -- fast enough)
        eig = eigen(liouv)
        ss_idx = argmin(abs.(eig.values))
        ss_vec = eig.vectors[:, ss_idx]
        ss_dm = reshape(ss_vec, DIM, DIM)
        ss_dm = (ss_dm + ss_dm') / 2
        ss_dm ./= tr(ss_dm)
        @test abs(eig.values[ss_idx]) < 1e-10
        @test norm(liouv * vec(ss_dm)) < 1e-10

        # TrotterDomain Liouvillian operates in Trotter eigenbasis, so transform
        # Gibbs state: eigenbasis -> computational -> Trotter eigenbasis
        gibbs_ref = if domain isa TrotterDomain
            gibbs_comp = TEST_HAM.eigvecs * TEST_GIBBS * TEST_HAM.eigvecs'
            Hermitian(TEST_TROTTER.eigvecs' * gibbs_comp * TEST_TROTTER.eigvecs)
        else
            TEST_GIBBS
        end
        distances[name] = trace_distance_h(Hermitian(ss_dm), gibbs_ref)
        @test distances[name] < thresholds[name]
    end

    @info "Dense fixed-point distances" distances thresholds
end

@testset "Complex coherent convention" begin
    # A complex Hermitian coupling is required here: real fixtures cannot
    # distinguish B from transpose(B); complex fixtures resolve this ambiguity.
    eigvals = [0.0, 0.4]
    raw = (
        matrix = ComplexF64[0 0; 0 0.4],
        terms = Vector{Vector{Matrix{ComplexF64}}}(),
        base_coeffs = Float64[],
        disordering_terms = nothing,
        disordering_coeffs = nothing,
        eigvals = eigvals,
        eigvecs = Matrix{ComplexF64}(I, 2, 2),
        nu_min = 0.4,
        shift = 0.0,
        rescaling_factor = 1.0,
        periodic = false,
    )
    ham = HamHam(raw, 2.0)
    A = ComplexF64[0.3 0.4 + 0.2im; 0.4 - 0.2im -0.1]
    jumps = JumpOp[JumpOp(A, A, false, true)]
    config = Config(;
        sim = Lindbladian(), domain = BohrDomain(), construction = KMS(),
        num_qubits = 1, with_linear_combination = true,
        beta = 2.0, sigma = 0.5, a = 0.1, s = 0.4,
    )

    precomputed = QuantumFurnace._precompute_data(config, ham)
    B = QuantumFurnace._precompute_coherent_B(jumps, ham, config, precomputed)
    @test norm(B - transpose(B)) > 1e-3

    L = construct_lindbladian(jumps, config, ham)
    L_diss = construct_lindbladian(jumps, config, ham; include_coherent = false)
    rho = ComplexF64[0.6 0.1 + 0.2im; 0.1 - 0.2im 0.4]
    coherent_direct = -1im .* (B * rho - rho * B)
    @test isapprox(reshape((L - L_diss) * vec(rho), 2, 2), coherent_direct;
        atol = 1e-14, rtol = 0)

    ws = Workspace(config, ham, jumps)
    @test isapprox(vec(apply_lindbladian!(ws, rho, config, ham)), L * vec(rho);
        atol = 1e-14, rtol = 0)
    @test norm(L * vec(Matrix(ham.gibbs))) < 1e-14

    D = materialize_discriminant(L, ham.gibbs)
    @test opnorm(D - D') < 1e-13
end
