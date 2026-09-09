@testset "DLL Kossakowski" begin

    # =====================================================================
    # Direct unit tests for `dll_kossakowski_bohr`. The DLL Kossakowski
    # is rank-1 by construction (Ding–Li–Lin 2024, Sec. 4): a single
    # weighting function `q^a` yields one Lindblad operator per coupling,
    # hence the per-coupling Kossakowski matrix is `v · v†` with
    # v_k = freq_kernel(filter, ν_k). This contrasts with CKG (full-rank
    # Kossakowski via Eq. 4.6 from Ramkumar–Soleimanifar Lemma 7.1).
    # =====================================================================

    _BETAS = (1.0, 5.0, 10.0)

    # ν grid differs by filter: Gaussian's tails decay smoothly so a wide
    # symmetric grid is fine; Metropolis lives inside its flat top |ν| ≤ S/2.
    _filter_for(label, beta) = label === :gaussian ?
        (filter = DLLGaussianFilter(beta),         ν_full = collect(-3.0:0.2:3.0), ν_outer = collect(-2.0:0.3:2.0)) :
        (filter = DLLMetropolisFilter(beta; S=2.0), ν_full = collect(-0.9:0.1:0.9), ν_outer = collect(-0.9:0.15:0.9))

    # ---------------------------------------------------------------------
    # (a/h2) Shape, Hermitian, PSD, rank-1 — sweep across both DLL filters.
    # ---------------------------------------------------------------------
    @testset "(a/h2) shape / Hermitian / PSD / rank-1 — $label" for label in (:gaussian, :metropolis)
        for beta in _BETAS
            v = _filter_for(label, beta)
            α = dll_kossakowski_bohr(v.filter, v.ν_full)
            K = length(v.ν_full)
            @test size(α) == (K, K)
            @test norm(α - α') <= 1e-12
            eigs = real.(eigvals(Hermitian(α)))
            @test minimum(eigs) >= -1e-12
            sv = svdvals(α)
            @test sv[1] > 1e-3
            @test sv[2] / sv[1] < 1e-12
        end
    end

    # ---------------------------------------------------------------------
    # (c/h3) Outer-product identity α[p,q] = v[p] · conj(v[q]) — both filters.
    # ---------------------------------------------------------------------
    @testset "(c/h3) outer-product identity — $label" for label in (:gaussian, :metropolis)
        beta = 5.0
        v = _filter_for(label, beta)
        α = dll_kossakowski_bohr(v.filter, v.ν_outer)
        vec_v = [freq_kernel(v.filter, ν) for ν in v.ν_outer]
        @test norm(α - vec_v * vec_v') <= 1e-14
        @test dissipator_trace_alpha(α) ≈ sum(abs2, vec_v) rtol=1e-14
    end

    # ---------------------------------------------------------------------
    # (d) HamHam overload returns sorted Bohr frequencies.
    # ---------------------------------------------------------------------
    @testset "(d) HamHam overload" begin
        Xm  = ComplexF64[0 1; 1 0]
        Ym  = ComplexF64[0 -im; im 0]
        Zm  = ComplexF64[1 0; 0 -1]
        Id2 = ComplexF64[1 0; 0 1]
        H = kron(Xm, Xm) + kron(Ym, Ym) + 1.5 * kron(Zm, Zm) +
            0.7 * kron(Zm, Id2) + 0.3 * kron(Id2, Zm)
        F = eigen(Hermitian(Matrix(Hermitian(H))))
        eigvals_vec = F.values
        eigvecs_mat = Matrix{ComplexF64}(F.vectors)
        bohr_freqs_mat = eigvals_vec .- transpose(eigvals_vec)
        bohr_dict = create_bohr_dict(bohr_freqs_mat)
        nu_min = minimum(filter(x -> x > 1e-12, abs.(bohr_freqs_mat)))
        gibbs = Hermitian(Matrix(I, 4, 4) * ComplexF64(0.25))

        ham = HamHam{Float64}(
            Matrix(Hermitian(H)), bohr_freqs_mat, bohr_dict,
            Vector{Vector{Matrix{ComplexF64}}}(), Float64[],
            nothing, nothing,
            Float64.(eigvals_vec), eigvecs_mat,
            nu_min, 0.0, 1.0, true, gibbs,
        )

        beta = 5.0
        filt = DLLGaussianFilter(beta)
        α, νs = dll_kossakowski_bohr(filt, ham)
        @test length(νs) == length(keys(bohr_dict))
        @test issorted(νs)
        @test size(α) == (length(νs), length(νs))
        # Match the explicit-grid version.
        α_explicit = dll_kossakowski_bohr(filt, νs)
        @test norm(α - α_explicit) <= 1e-14
    end

    # ---------------------------------------------------------------------
    # (e) CKG reference: full-rank Kossakowski (uses existing `create_alpha`)
    # ---------------------------------------------------------------------
    @testset "(e) CKG vs DLL: rank gap" begin
        beta = 5.0
        sigma = 1.0 / beta
        ν_grid = collect(-2.0:0.3:2.0)
        K = length(ν_grid)

        # CKG smooth-Metropolis Kossakowski (a, s = 0 plain Metropolis)
        α_ckg = Matrix{Float64}(undef, K, K)
        for q in 1:K, p in 1:K
            α_ckg[p, q] = create_alpha(ν_grid[p], ν_grid[q], beta, sigma, 0.1, 0.4)
        end

        filt = DLLGaussianFilter(beta)
        α_dll = dll_kossakowski_bohr(filt, ν_grid)

        sv_ckg = svdvals(α_ckg)
        sv_dll = svdvals(α_dll)

        # DLL is rank-1 (one dominant SV).
        @test sv_dll[2] / sv_dll[1] < 1e-12

        # CKG has more than one significant SV (full-rank in general).
        @test sv_ckg[2] / sv_ckg[1] > 1e-3
        # Number of CKG SVs above 1% of leading: at least 2 (often many).
        n_significant_ckg = count(s -> s / sv_ckg[1] > 1e-2, sv_ckg)
        @test n_significant_ckg >= 2
    end

    # ---------------------------------------------------------------------
    # KMS-DBC skew-symmetry α(ν,ν') = α(-ν',-ν) e^{-β(ν+ν')/2} (Eq. 4.7).
    # The rescaled matrix `α·e^{β(ν+ν')/4}` is centrosymmetric (Eq. 4.8) —
    # mathematically equivalent to skew-symmetry, so we keep only the
    # primary witness. Sweep parameterised over both DLL filters in
    # the (f/h4) testset below.

    # ---------------------------------------------------------------------
    # (h) β-scaling: DLL α norm decreases with β (filter narrows ∝ 1/β)
    # ---------------------------------------------------------------------
    @testset "(h) β-scaling" begin
        ν_grid = collect(-2.0:0.2:2.0)
        norms = Float64[]
        for beta in _BETAS
            filt = DLLGaussianFilter(beta)
            α = dll_kossakowski_bohr(filt, ν_grid)
            push!(norms, norm(α))
        end
        # All bounded; β = 10 strictly less than β = 1 (filter narrows).
        @test all(<=(20.0), norms)
        @test norms[3] < norms[1]
    end

    # =====================================================================
    # DLL Metropolis-type Kossakowski — α is again rank-1, but
    # |α| stays O(1) at low T because f̂ has compact O(1) support and the
    # Metropolis weight saturates at 1 for ν < 0 (vs Gaussian's β-shrinking
    # weight). Structural shape/PSD/rank-1/outer-product/KMS-skew tests for
    # Metropolis are merged into (a/h2), (c/h3), (f/h4) above. (h5) below
    # is the only Metropolis-only motivating contrast (different intent).
    # =====================================================================

    # ---------------------------------------------------------------------
    # (f/h4) KMS-DBC skew-symmetry — sweep both DLL filters.
    # ---------------------------------------------------------------------
    @testset "(f/h4) KMS-DBC α(ν,ν') = α(-ν',-ν) e^{-β(ν+ν')/2} — $label" for label in (:gaussian, :metropolis)
        for beta in _BETAS
            v = _filter_for(label, beta)
            ν_grid = label === :gaussian ? collect(-3.0:0.5:3.0) : collect(-0.9:0.3:0.9)
            α = dll_kossakowski_bohr(v.filter, ν_grid)
            assert_kms_skew_symmetric(α, ν_grid, beta; atol=1e-12)
        end
    end

    # ---------------------------------------------------------------------
    # (h5) Metropolis vs Gaussian qualitative contrast at low T (β = 10):
    # the Metropolis filter saturates at f̂(ν) → 1 for ν ≪ 0 (Eq. 3.20),
    # while the Gaussian filter is concentrated at -1/β with width 2/β
    # — anything farther than a few widths from -1/β shrinks exponentially.
    # We probe the diagonal Kossakowski entry |α(ν,ν)| = |f̂(ν)|² for
    # negative ν outside the Gaussian's main lobe (|ν| ≥ 0.5 ≫ 1/β = 0.1).
    # This is THE motivating contrast for the Metropolis-type filter.
    # ---------------------------------------------------------------------
    @testset "(h5) Metropolis stays O(1) at low T; Gaussian shrinks" begin
        beta = 10.0
        # Gaussian filter centre = -1/β = -0.1, width 2/β = 0.2, so |ν| ≥ 0.5
        # puts us > 2σ outside the lobe. Restrict the probe to that region.
        ν_grid = (-0.9, -0.7, -0.5)
        f_metro = DLLMetropolisFilter(beta; S = 2.0)
        f_gauss = DLLGaussianFilter(beta)
        for ν in ν_grid
            metro_diag = abs2(freq_kernel(f_metro, ν))
            gauss_diag = abs2(freq_kernel(f_gauss, ν))
            # Metropolis: at least 0.9 (saturates at f̂² → 1 for ν ≪ 0)
            @test metro_diag >= 0.9
            # Gaussian: < 0.1 (decayed exponentially outside its narrow lobe)
            @test gauss_diag <= 0.1
            # Ratio: at least 10× larger for Metropolis at ν = -0.5,
            # exponentially larger as |ν| grows.
            @test metro_diag / gauss_diag >= 10
        end
    end
end
