@testset "Non-Hermitian jumps end-to-end (qf-bm1)" begin

    # =====================================================================
    # End-to-end correctness checks for non-Hermitian jump operators in
    # the CKG-KMS construction.
    #
    # Background (qf-bm1, see proofs/non-hermitian-coherent/proof-v1.md
    # and .claude-memory/feedback_non_hermitian_jumps.md):
    #
    # Q1 — KMS-DB requires (A, A†) pairs.
    #      Single non-Hermitian jumps make the Lindbladian violate
    #      α(ω₁,ω₂) = α(-ω₂,-ω₁) e^{-β(ω₁+ω₂)/2} skew-symmetry.
    # Q2 — Coherent term has a per-jump contact contribution
    #      κ_NH ∫ b_-(t) e^{-iHt/σ} A^a† A^a e^{iHt/σ} dt.
    #      For physically-valid (paired) jump sets, the contact terms
    #      cancel (Σ A^a† A^a is symmetric and the cancellation makes the
    #      summed B match B_bohr). The production code does not need to
    #      add the contact term separately.
    # Q3 — All `JumpOp.hermitian` dispatch branches in the codebase have
    #      been audited clean (see .claude-memory/q3_branch_audit.md).
    #
    # The tests below verify these claims against the existing production
    # code paths.
    # =====================================================================

    # ---- Fixtures ----------------------------------------------------------
    # σ⁺ = (X + iY)/2 = [0 1; 0 0], σ⁻ = (X - iY)/2 = [0 0; 1 0].
    _SIGMA_PLUS  = ComplexF64[0 1; 0 0]
    _SIGMA_MINUS = ComplexF64[0 0; 1 0]

    # n=3 disordered Heisenberg fixture, parametrised by β so the smooth-
    # Metropolis filter and Gibbs reference both track β. Reuses
    # `make_dll_n3_system`-style loader.
    function _load_n3_ham(beta::Real)
        ham_path = test_hamiltonian_path(3)
        return QuantumFurnace._load_hamiltonian_bson(ham_path, Float64(beta))
    end

    """
        nh_pair_jumps(ham; site=1) -> Vector{JumpOp}

    Paired (σ⁺, σ⁻) on the requested site, normalised by 1/√2 so the
    overall magnitude matches a single Hermitian Pauli. Both jumps marked
    `hermitian=false`. Built in the supplied basis (defaults to
    `ham.eigvecs`; pass `trotter.eigvecs` for TrotterDomain).
    """
    function nh_pair_jumps(ham::HamHam; site::Int=1, basis::AbstractMatrix=ham.eigvecs)
        n = Int(log2(size(ham.data, 1)))
        norm_fac = 1.0 / sqrt(2)
        jumps = Vector{JumpOp}(undef, 2)
        for (k, op2x2) in enumerate((_SIGMA_PLUS, _SIGMA_MINUS))
            op = Matrix(pad_term([op2x2], n, site)) .* norm_fac
            op_eb = basis' * op * basis
            jumps[k] = JumpOp(op, op_eb, false, false)
        end
        return jumps
    end

    """
        herm_x_jump(ham; site=1) -> Vector{JumpOp}

    Hermitian X jump on the same site, normalised identically. Used as a
    same-magnitude reference to confirm paired NH behaves identically to
    Hermitian within the discretisation floor.
    """
    function herm_x_jump(ham::HamHam; site::Int=1, basis::AbstractMatrix=ham.eigvecs)
        n = Int(log2(size(ham.data, 1)))
        norm_fac = 1.0 / sqrt(2)
        op = Matrix(pad_term([X], n, site)) .* norm_fac
        op_eb = basis' * op * basis
        return Vector{JumpOp}([JumpOp(op, op_eb, false, true)])
    end

    """
        unpaired_nh_jump(ham; site=1) -> Vector{JumpOp}

    Single σ⁺ alone — the Q1 violator fixture. Used to confirm validation
    catches it and that allow_unpaired_nonhermitian=true is required.
    """
    function unpaired_nh_jump(ham::HamHam; site::Int=1, basis::AbstractMatrix=ham.eigvecs)
        n = Int(log2(size(ham.data, 1)))
        norm_fac = 1.0 / sqrt(2)
        op = Matrix(pad_term([_SIGMA_PLUS], n, site)) .* norm_fac
        op_eb = basis' * op * basis
        return Vector{JumpOp}([JumpOp(op, op_eb, false, false)])
    end

    # Wider grid than 4-qubit defaults — at low β the smooth-Metropolis
    # filter spreads in ω, and a 12-bit register at w0=0.05 cuts off the
    # tail. 14 bits at w0=0.1 covers β=1 cleanly; β≥5 saturates earlier.
    _NUM_ENERGY_BITS = 14
    _W0 = 0.1
    _NUM_TROTTER_STEPS_PER_T0 = 10

    function _nh_config(
        domain;
        sim = Lindbladian(),
        beta::Real,
        num_energy_bits::Int = _NUM_ENERGY_BITS,
        w0::Real = _W0,
    )
        Config(;
            sim = sim,
            domain = domain,
            construction = KMS(),
            num_qubits = 3,
            with_linear_combination = true,
            beta = Float64(beta),
            sigma = 1.0 / Float64(beta),
            a = beta / 30.0,
            s = 0.4,
            num_energy_bits = num_energy_bits,
            w0 = Float64(w0),
            t0 = 2pi / (2^num_energy_bits * Float64(w0)),
            num_trotter_steps_per_t0 = _NUM_TROTTER_STEPS_PER_T0,
        )
    end

    function _signed_serial_R(
        jump::JumpOp,
        hamiltonian::HamHam,
        config::Config{Thermalize, EnergyDomain},
        precomputed_data,
    )
        dim = size(hamiltonian.data, 1)
        R = zeros(ComplexF64, dim, dim)
        jump_oft = similar(R)
        LdagL = similar(R)
        prefactor = precomputed_data.oft_domain_prefactor *
            precomputed_data.gamma_norm_factor
        inv_4sigma2 = 1.0 / (4 * config.sigma^2)

        for w in precomputed_data.energy_labels
            QuantumFurnace.oft!(
                jump_oft, jump.in_eigenbasis, hamiltonian.bohr_freqs,
                w, inv_4sigma2,
            )
            mul!(LdagL, jump_oft', jump_oft)
            R .+= (prefactor * precomputed_data.transition(w)) .* LdagL
        end
        return (R + R') / 2
    end

    function _signed_serial_R(
        jump::JumpOp,
        ham_or_trott::Union{HamHam, AbstractTrotter},
        config::Config{Thermalize, D},
        precomputed_data,
    ) where {D<:Union{TimeDomain, TrotterDomain}}
        dim = size(jump.in_eigenbasis, 1)
        R = zeros(ComplexF64, dim, dim)
        jump_oft = similar(R)
        LdagL = similar(R)
        prefactor = precomputed_data.oft_domain_prefactor *
            precomputed_data.gamma_norm_factor

        for w in precomputed_data.energy_labels
            nufft_prefactor = QuantumFurnace._prefactor_view(
                precomputed_data.oft_nufft_prefactors, w,
            )
            @. jump_oft = jump.in_eigenbasis * nufft_prefactor
            mul!(LdagL, jump_oft', jump_oft)
            R .+= (prefactor * precomputed_data.transition(w)) .* LdagL
        end
        return (R + R') / 2
    end

    @testset "Threaded per-jump R preserves signed frequencies" begin
        if Threads.nthreads() > 1
            beta = 5.0
            ham = _load_n3_ham(beta)

            for domain in (EnergyDomain(), TimeDomain(), TrotterDomain())
                config = _nh_config(
                    domain; sim=Thermalize(), beta=beta,
                    num_energy_bits=8, w0=0.1,
                )
                trotter = domain isa TrotterDomain ?
                    make_trotter_for_config(ham, config) : nothing
                ham_or_trott = trotter === nothing ? ham : trotter
                jump = first(nh_pair_jumps(
                    ham; basis=trotter === nothing ? ham.eigvecs : trotter.eigvecs,
                ))
                precomputed_data = QuantumFurnace._precompute_data(config, ham_or_trott)
                @test length(precomputed_data.energy_labels) >=
                    QuantumFurnace.OMEGA_THREAD_THRESHOLD

                scratch = QuantumFurnace.ThermalizeScratch(ComplexF64, size(ham.data, 1))
                R_threaded = copy(QuantumFurnace._precompute_R(
                    [jump], ham_or_trott, config, precomputed_data, scratch,
                ))
                R_serial = _signed_serial_R(
                    jump, ham_or_trott, config, precomputed_data,
                )
                relative_error = norm(R_threaded - R_serial) / norm(R_serial)
                @test relative_error < 1e-12
                @info "non-Hermitian per-jump R threaded parity" domain relative_error
            end
        else
            @test_skip Threads.nthreads() > 1
        end
    end

    @testset "EnergyDomain signed-frequency refinement approaches Bohr" begin
        β = 5.0
        ham = _load_n3_ham(β)
        jumps = nh_pair_jumps(ham)
        L_bohr = Matrix{ComplexF64}(construct_lindbladian(
            jumps, _nh_config(BohrDomain(); beta=β), ham))
        errors = Float64[]
        for w0 in (0.4, 0.2, 0.1)
            cfg = _nh_config(EnergyDomain(); beta=β, num_energy_bits=8, w0=w0)
            L_energy = Matrix{ComplexF64}(construct_lindbladian(jumps, cfg, ham))
            push!(errors, opnorm(L_energy - L_bohr))
        end
        @test all(diff(errors) .< 0)
        @test errors[end] <= 1e-9
        @info "non-Hermitian Energy→Bohr refinement" w0=(0.4, 0.2, 0.1) errors
    end

    # =====================================================================
    # Test 1: KMS-DB at quadrature precision across all four domains for
    # paired (σ⁺, σ⁻).
    # =====================================================================
    @testset "KMS-DB rel_norm — paired (σ⁺, σ⁻) all domains" begin
        # β=5: discretisation in Energy/Time saturates near machine precision,
        # so we can demand ≤ 1e-10. β=10 saturates at the quadrature floor
        # (~5e-6) — we relax accordingly.
        # tol_trotter is separated from tol_quad because the TrotterDomain
        # construction carries its own Strang error in addition to the
        # Energy/Time quadrature error.  At β=5 the EnergyDomain / TimeDomain
        # discretisation saturates near machine precision, but Trotter sits
        # just above 1e-10 at the default per-leg substep counts (qf-e4z.20).
        for (β, tol_machine, tol_quad, tol_trotter) in [
            (5.0,  1e-10, 1e-10, 5e-10),
            (10.0, 1e-10, 5e-5,  5e-5),
        ]
            @testset "β=$β" begin
                ham = _load_n3_ham(β)

                # BohrDomain — exact analytic, machine precision.
                cfg = _nh_config(BohrDomain(); beta=β)
                jumps = nh_pair_jumps(ham)
                L = Matrix{ComplexF64}(construct_lindbladian(jumps, cfg, ham))
                res = verify_detailed_balance(L, ham.gibbs)
                @test res.relative_norm < tol_machine
                @test res.fixed_point_residual < tol_machine

                # EnergyDomain.
                cfg = _nh_config(EnergyDomain(); beta=β)
                jumps = nh_pair_jumps(ham)
                L = Matrix{ComplexF64}(construct_lindbladian(jumps, cfg, ham))
                res = verify_detailed_balance(L, ham.gibbs)
                @test res.relative_norm < tol_quad

                # TimeDomain.
                cfg = _nh_config(TimeDomain(); beta=β)
                jumps = nh_pair_jumps(ham)
                L = Matrix{ComplexF64}(construct_lindbladian(jumps, cfg, ham))
                res = verify_detailed_balance(L, ham.gibbs)
                @test res.relative_norm < tol_quad

                # TrotterDomain. Rotate the generator back to the Hamiltonian
                # eigenbasis before applying the diagonal-Gibbs discriminant.
                cfg = _nh_config(TrotterDomain(); beta=β)
                trotter = make_trotter_for_config(ham, cfg)
                jumps = nh_pair_jumps(ham; basis=trotter.eigvecs)
                L_trotter = Matrix{ComplexF64}(
                    construct_lindbladian(jumps, cfg, ham; trotter=trotter))
                L_ham = QuantumFurnace._change_dense_superoperator_basis(
                    L_trotter, trotter.eigvecs, ham.eigvecs)
                res = verify_detailed_balance(L_ham, ham.gibbs)
                @test res.relative_norm < tol_trotter
            end
        end
    end

    # =====================================================================
    # Test 2a: Hermitian-baseline KMS-DB equivalence.
    # Both paired NH and Hermitian X should hit quadrature precision. The
    # two are NOT the same Lindbladian (paired-NH per-jump construction
    # lacks the σ⁺†σ⁻ cross-term that X² = I produces in B_X), so we
    # compare only their KMS-DB residuals — both clean — not the matrices.
    # =====================================================================
    @testset "Hermitian-baseline equivalence (paired NH ≡ X within quadrature)" begin
        β = 5.0
        ham = _load_n3_ham(β)
        cfg = _nh_config(EnergyDomain(); beta=β)

        L_nh = Matrix{ComplexF64}(construct_lindbladian(nh_pair_jumps(ham), cfg, ham))
        L_h  = Matrix{ComplexF64}(construct_lindbladian(herm_x_jump(ham), cfg, ham))

        res_nh = verify_detailed_balance(L_nh, ham.gibbs)
        res_h  = verify_detailed_balance(L_h,  ham.gibbs)
        @test res_nh.relative_norm < 1e-10
        @test res_h.relative_norm  < 1e-10
    end

    # =====================================================================
    # Test 2b: Hermitian-limit reduction (acceptance criterion #2).
    # Take a Hermitian operator X, mark `hermitian=false`, and construct
    # the Lindbladian. Compare to the same X with `hermitian=true`. The
    # two paths exercise different code branches (full-grid vs half-grid
    # fold) but should produce identical Lindbladians. Validation is
    # bypassed via `allow_unpaired_nonhermitian=true` (X is its own
    # adjoint, so it counts as "unpaired" by the structural check).
    # =====================================================================
    @testset "Hermitian-limit reduction: jumps with hermitian=true and =false agree" begin
        for dom in (EnergyDomain(), TimeDomain())
            β = 5.0
            ham = _load_n3_ham(β)
            cfg = _nh_config(dom; beta=β)

            n = 3
            norm_fac = 1.0 / sqrt(2)
            op = Matrix(pad_term([X], n, 1)) .* norm_fac
            op_eb = ham.eigvecs' * op * ham.eigvecs
            herm_jump = Vector{JumpOp}([JumpOp(op, op_eb, false, true)])
            nh_jump   = Vector{JumpOp}([JumpOp(op, op_eb, false, false)])

            L_h  = Matrix{ComplexF64}(construct_lindbladian(herm_jump, cfg, ham))
            L_nh = Matrix{ComplexF64}(construct_lindbladian(nh_jump, cfg, ham;
                allow_unpaired_nonhermitian=true))

            # Byte-equivalence (FP roundoff only): half-grid Hermitian fold
            # ≡ full-grid non-Hermitian iteration when A = A†.
            @test norm(L_h - L_nh) < 1e-12
        end
    end

    # =====================================================================
    # Test 3: Validation rejects unpaired non-Hermitian jumps.
    # =====================================================================
    @testset "validate_jump_pairing rejects unpaired non-Hermitian" begin
        β = 5.0
        ham = _load_n3_ham(β)
        cfg = _nh_config(EnergyDomain(); beta=β)
        unpaired = unpaired_nh_jump(ham)

        # Default behaviour — error.
        @test_throws ArgumentError construct_lindbladian(unpaired, cfg, ham)

        # Helper directly.
        @test_throws ArgumentError validate_jump_pairing(unpaired)

        # Opt-out kwarg — passes (and the resulting L is non-KMS-DB, but
        # validation no longer prevents construction).
        L = construct_lindbladian(unpaired, cfg, ham; allow_unpaired_nonhermitian=true)
        @test size(L) == (8 * 8, 8 * 8)
    end

    @testset "validate_jump_pairing accepts paired sets and Hermitian-only sets" begin
        β = 5.0
        ham = _load_n3_ham(β)

        @test validate_jump_pairing(nh_pair_jumps(ham)) === nothing
        @test validate_jump_pairing(herm_x_jump(ham))    === nothing

        # Mixed Hermitian + paired non-Hermitian is fine.
        mixed = vcat(herm_x_jump(ham), nh_pair_jumps(ham))
        @test validate_jump_pairing(mixed) === nothing
    end

    # =====================================================================
    # Test 3.5: Multi-site non-Hermitian fixture.
    # Per the proof at proofs/non-hermitian-coherent/proof-v1.md, the
    # contact-term cancellation requires [H, Σ A^a†A^a] = 0 (summed over
    # jumps). For 2-site jumps `A = σ⁺_1 σ⁻_2` paired with `A† = σ⁻_1 σ⁺_2`,
    # the per-pair sum is (I - Z_1 Z_2)/4 which does NOT commute with
    # H_XXX boundary terms (`[Z_1 Z_2, X_2 X_3] ≠ 0`). Empirically the
    # existing B_time formula already produces B_bohr at machine precision
    # for this fixture too — confirming that the production code's
    # implementation captures the contact-term-like contribution implicitly
    # (likely via the t'=0 grid-point L'Hôpital sample).
    # =====================================================================
    @testset "Multi-site non-Hermitian fixture: B_bohr ≡ B_time" begin
        β = 5.0
        ham = _load_n3_ham(β)
        cfg_bohr = _nh_config(BohrDomain(); beta=β)
        cfg_time = _nh_config(TimeDomain(); beta=β)

        # 2-site jump on sites (1, 2) — per-pair A†A + AA† = (I - Z₁Z₂)/4.
        norm_fac = 1.0 / sqrt(2)
        op_pm = Matrix(pad_term([_SIGMA_PLUS], 3, 1)) *
                Matrix(pad_term([_SIGMA_MINUS], 3, 2)) .* norm_fac
        op_mp = Matrix(pad_term([_SIGMA_MINUS], 3, 1)) *
                Matrix(pad_term([_SIGMA_PLUS], 3, 2)) .* norm_fac
        op_pm_eb = ham.eigvecs' * op_pm * ham.eigvecs
        op_mp_eb = ham.eigvecs' * op_mp * ham.eigvecs
        jumps = Vector{JumpOp}([
            JumpOp(op_pm, op_pm_eb, false, false),
            JumpOp(op_mp, op_mp_eb, false, false),
        ])

        pd_bohr = QuantumFurnace._precompute_data(cfg_bohr, ham)
        pd_time = QuantumFurnace._precompute_data(cfg_time, ham)
        B_bohr = QuantumFurnace._precompute_coherent_B(jumps, ham, cfg_bohr, pd_bohr)
        B_time = QuantumFurnace._precompute_coherent_B(jumps, ham, cfg_time, pd_time)

        # B_time should match B_bohr at machine precision even though the
        # vanishing condition [H, Σ A†A] = 0 is *violated* for this fixture.
        @test opnorm(B_bohr - B_time) / opnorm(B_bohr) < 1e-10

        # Sanity: both are KMS-DB at quadrature precision.
        L_bohr = Matrix{ComplexF64}(construct_lindbladian(jumps, cfg_bohr, ham))
        L_time = Matrix{ComplexF64}(construct_lindbladian(jumps, cfg_time, ham))
        @test verify_detailed_balance(L_bohr, ham.gibbs).relative_norm < 1e-10
        @test verify_detailed_balance(L_time, ham.gibbs).relative_norm < 1e-10
    end

    # =====================================================================
    # Test 4: Krylov ↔ dense matvec equivalence for paired (σ⁺, σ⁻).
    # Confirms apply_lindbladian! and apply_adjoint_lindbladian! produce
    # the same matvec as the dense `construct_lindbladian` for paired NH.
    # Downstream Krylov consumers (krylov_spectral_gap,
    # lindblad_action_integrate, predict_lindbladian_trajectory) inherit
    # this equivalence.
    # =====================================================================
    @testset "Krylov matvec ≡ dense for paired (σ⁺, σ⁻)" begin
        β = 5.0
        ham = _load_n3_ham(β)
        dim = size(ham.data, 1)
        rho = Matrix(random_density_matrix(Int(log2(dim))))

        for dom in (EnergyDomain(), TimeDomain())
            cfg = _nh_config(dom; beta=β)
            jumps = nh_pair_jumps(ham)

            L_dense = Matrix{ComplexF64}(construct_lindbladian(jumps, cfg, ham))

            # Forward
            ws = QuantumFurnace.Workspace(cfg, ham, jumps)
            r_kry = copy(apply_lindbladian!(ws, rho, cfg, ham))
            r_dense = reshape(L_dense * vec(rho), dim, dim)
            @test norm(r_dense - r_kry) < 1e-12

            # Adjoint
            ws_adj = QuantumFurnace.Workspace(cfg, ham, jumps)
            r_kry_adj = copy(apply_adjoint_lindbladian!(ws_adj, rho, cfg, ham))
            r_dense_adj = reshape(L_dense' * vec(rho), dim, dim)
            @test norm(r_dense_adj - r_kry_adj) < 1e-12
        end
    end
end

@testset "DLL Time adjoint pairs and controlled full-generator references" begin
    ham = HamHam(build_heis_1d(2, [0.8, 0.7, 1.2]; seed=14,
        periodic=false, disorder_strength=0.2); beta_phys=0.5)
    beta = beta_alg(ham, 0.5)
    rho_beta = Matrix(ham.gibbs)
    makejump(A) = JumpOp(Matrix(A), ham.eigvecs' * A * ham.eigvecs,
                        false, ishermitian(A))
    cfg(domain, filter) = Config(; sim=Lindbladian(), domain,
        construction=DLL(), num_qubits=2, with_linear_combination=true,
        beta, beta_phys=0.5, sigma=1/beta, a=beta/30, s=0.25,
        num_energy_bits_D=8, t0_D=0.5, filter)
    random_source = randn(QuantumFurnace.Random.MersenneTwister(42), ComplexF64, 4, 4) / 4
    raising = Matrix(pad_term([(X + im*Y)/2], 2, 1)) / 2
    xy = Matrix(pad_term([X + Y], 2, 1)) / 4

    @testset "Pair equivalence: $label, $(typeof(f))" for (label, A) in
        ((:complex, random_source), (:raising_lowering, raising), (:complex_hermitian, xy)),
        f in (DLLGaussianFilter(beta), DLLMetropolisFilter(beta; S=2.0))
        paired = JumpOp[makejump(A), makejump(A')]
        hermitian = JumpOp[makejump((A+A')/sqrt(2)), makejump((A-A')/(im*sqrt(2)))]
        Lb = construct_lindbladian(paired, cfg(BohrDomain(), f), ham)
        Lbh = construct_lindbladian(hermitian, cfg(BohrDomain(), f), ham)
        Lt = construct_lindbladian(paired, cfg(TimeDomain(), f), ham)
        Lth = construct_lindbladian(hermitian, cfg(TimeDomain(), f), ham)
        @test isapprox(Lb, Lbh; atol=1e-12, rtol=0)
        @test isapprox(Lt, Lth; atol=1e-12, rtol=0)
        pd = QuantumFurnace._precompute_data(cfg(TimeDomain(), f), ham)
        Bt = dll_coherent_op_time(paired, ham, pd.time_labels, f, beta, pd.t0)
        Bth = dll_coherent_op_time(hermitian, ham, pd.time_labels, f, beta, pd.t0)
        Bb = dll_coherent_op_bohr(paired, ham, f, beta)
        @test isapprox(Bt, Bth; atol=1e-12, rtol=0)
        # Production truncation still couples the coherent and dissipative
        # cutoffs (T13). Verify the NH defects equal the Hermitian floor;
        # the explicit-window checks below separately reach 1e-9.
        Dt = materialize_discriminant(Lt, rho_beta)
        Dth = materialize_discriminant(Lth, rho_beta)
        @test isapprox(norm(Lt*vec(rho_beta)), norm(Lth*vec(rho_beta)); atol=1e-12, rtol=0)
        @test isapprox(opnorm(Dt-Dt'), opnorm(Dth-Dth'); atol=1e-12, rtol=0)
        @info "DLL pair production grid" label filter=typeof(f) coherent_error=opnorm(Bt-Bb) full_error=opnorm(Lt-Lb) gibbs_residual=norm(Lt*vec(rho_beta)) kms_defect=opnorm(Dt-Dt')/opnorm(Dt)
        if !ishermitian(A)
            for domain in (BohrDomain(), TimeDomain())
                @test_throws ArgumentError construct_lindbladian(paired[1:1], cfg(domain, f), ham)
                @test_throws ArgumentError construct_lindbladian(vcat(paired, paired[1:1]), cfg(domain, f), ham)
            end
        end
    end

    @testset "Explicit-window refinement: $(typeof(f))" for (f, windows) in (
        (DLLGaussianFilter(beta), (10.0, 20.0)),
        (DLLMetropolisFilter(beta; S=2.0), (40.0, 180.0)),
    )
        paired = JumpOp[makejump(random_source), makejump(random_source')]
        Lb = construct_lindbladian(paired, cfg(BohrDomain(), f), ham)
        Bb = dll_coherent_op_bohr(paired, ham, f, beta)
        Id = Matrix{ComplexF64}(I, 4, 4)
        full_errors, coherent_errors = Float64[], Float64[]
        for tmax in windows
            ts = collect(-tmax:0.5:tmax)
            Bt = dll_coherent_op_time(paired, ham, ts, f, beta, 0.5)
            # Independent column-vectorised GKLS assembly from direct time
            # jumps and the full two-time B. No Bohr correction substitution.
            Lt = -im .* (kron(Id, Bt) - kron(transpose(Bt), Id))
            for jump in paired
                M = dll_lindblad_op_time(jump, ham, ts, f, 0.5)
                R = M' * M
                Lt .+= kron(conj(M), M) - (kron(Id, R) + kron(transpose(R), Id))/2
            end
            push!(full_errors, opnorm(Lt-Lb))
            push!(coherent_errors, opnorm(Bt-Bb))
            if tmax == last(windows)
                D = materialize_discriminant(Lt, rho_beta)
                @test norm(Lt*vec(rho_beta)) < 1e-9
                @test opnorm(D-D')/opnorm(D) < 1e-9
                @test norm(Lt'*vec(Id)) < 1e-9
                @test norm(Bt-Bt') < 1e-9
            end
        end
        @test full_errors[end] < full_errors[1]
        @test coherent_errors[end] < coherent_errors[1]
        @test full_errors[end] < 1e-9
        @test coherent_errors[end] < 1e-9
        @info "DLL explicit-window refinement" filter=typeof(f) windows full_errors=Tuple(full_errors) coherent_errors=Tuple(coherent_errors)
    end
end
