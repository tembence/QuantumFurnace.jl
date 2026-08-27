# Boundary-condition invariants for 1D/2D Hamiltonian construction, disorder,
# Pauli exponentials, Trotterisation, and a small OBC Krylov integration case.

using Test
using LinearAlgebra
using SparseArrays
using Random
using QuantumFurnace
using QuantumFurnace: X, Y, Z, pad_term, expm_pauli_padded,
    _construct_base_ham, _construct_disordering_terms,
    _construct_disordering_terms_2d, _construct_2d_heisenberg_base,
    _pad_two_site_op, _trotterize2, trotterize

# Heisenberg base term list and uniform couplings used throughout
const HEIS_TERMS = Vector{Matrix{ComplexF64}}[[X, X], [Y, Y], [Z, Z]]
const HEIS_COEFFS = [1.0, 1.0, 1.0]

# Helper: Pauli-bond count via Hilbert-Schmidt trace of a sum of disjoint-support
# Pauli operators with positive multiplicities. For a sum `H = Σ_b m_b · P_b`
# (each P_b a Pauli string with P_b² = I, distinct bonds having different
# Pauli supports), `tr(H²) = (Σ_b m_b²) · dim`. The double-counted bonds at
# Lx==2 / Ly==2 PBC therefore show up squared.
function _pauli_bond_msq(H::AbstractMatrix, num_qubits::Int)
    real(tr(H * H)) / 2^num_qubits
end

@testset "qf-91g — boundary-condition audit (1D + 2D × disorder × Trotter)" begin

    # ------------------------------------------------------------------
    # (a) `pad_term` BC behaviour — the single source of truth for wrap.
    # ------------------------------------------------------------------
    @testset "(a) pad_term BC base case" begin
        n = 4
        # Two-site Z⊗Z at position n: PBC wraps to Z_n Z_1; OBC returns zeros.
        P_pbc = pad_term([Z, Z], n, n; periodic=true)
        P_obc = pad_term([Z, Z], n, n; periodic=false)
        @test opnorm(Matrix(P_pbc)) ≈ 1.0   # nontrivial Pauli string
        @test iszero(Matrix(P_obc))         # explicit zero — OBC has no wrap
        @test P_obc isa SparseMatrixCSC{ComplexF64}
        @test size(P_obc) == (2^n, 2^n)
        @test nnz(P_obc) == 0

        # Single-site at any position: BC is irrelevant
        S_pbc = pad_term([Z], n, n; periodic=true)
        S_obc = pad_term([Z], n, n; periodic=false)
        @test isapprox(Matrix(S_pbc), Matrix(S_obc))
    end

    # ------------------------------------------------------------------
    # (b) 1D _construct_disordering_terms — propagates `periodic`.
    # ------------------------------------------------------------------
    @testset "(b) 1D disorder term BC propagation" begin
        for n in (3, 4, 5)
            term_coeffs = Float64.(1:n)  # nonzero per-site distinct values

            # Single-site disorder: PBC == OBC (no wrap)
            S_pbc = Matrix(_construct_disordering_terms(
                Vector{Matrix{ComplexF64}}[[Z]], [term_coeffs], n; periodic=true))
            S_obc = Matrix(_construct_disordering_terms(
                Vector{Matrix{ComplexF64}}[[Z]], [term_coeffs], n; periodic=false))
            @test isapprox(S_pbc, S_obc)

            # Two-site disorder: PBC has the wrap term (Z_n ⊗ Z_1 with coeff term_coeffs[n]),
            # OBC does not. Difference should equal exactly that wrap term.
            T_pbc = Matrix(_construct_disordering_terms(
                Vector{Matrix{ComplexF64}}[[Z, Z]], [term_coeffs], n; periodic=true))
            T_obc = Matrix(_construct_disordering_terms(
                Vector{Matrix{ComplexF64}}[[Z, Z]], [term_coeffs], n; periodic=false))
            wrap = term_coeffs[n] * Matrix(pad_term([Z, Z], n, n; periodic=true))
            @test isapprox(T_pbc - T_obc, wrap; atol=1e-14)

            # Default kwarg matches periodic=true
            T_default = Matrix(_construct_disordering_terms(
                Vector{Matrix{ComplexF64}}[[Z, Z]], [term_coeffs], n))
            @test isapprox(T_default, T_pbc; atol=1e-14)
        end
    end

    # ------------------------------------------------------------------
    # (c) 1D HamHam ctor (multi-term disorder) honours `periodic`.
    # ------------------------------------------------------------------
    @testset "(c) 1D HamHam multi-term disorder ctor BC propagation" begin
        for n in (3, 4, 5)
            dis_terms = Vector{Matrix{ComplexF64}}[[Z], [Z, Z]]
            dis_coeffs = [Float64.(1:n), Float64.(0.1:0.1:0.1n)]
            H_pbc = HamHam(HEIS_TERMS, HEIS_COEFFS, dis_terms, dis_coeffs, n, 1.0; periodic=true)
            H_obc = HamHam(HEIS_TERMS, HEIS_COEFFS, dis_terms, dis_coeffs, n, 1.0; periodic=false)
            @test H_pbc.periodic === true
            @test H_obc.periodic === false
            # Physical Hamiltonians (before rescale) differ on every wrap site we touched.
            # After the rescale + shift to [0, 0.45] the matrices remain distinct.
            @test !isapprox(H_pbc.data, H_obc.data; atol=1e-8)
            # Eigenvalues live in [0, 0.45] up to a few-ulp floor (well-defined Gibbs).
            @test all(-1e-12 .<= H_pbc.eigvals .<= 0.45 + 1e-12)
            @test all(-1e-12 .<= H_obc.eigvals .<= 0.45 + 1e-12)
        end
    end

    # ------------------------------------------------------------------
    # (d) 1D build_heis_1d fixture honours `periodic`.
    # ------------------------------------------------------------------
    @testset "(d) 1D build_heis_1d fixture BC propagation" begin
        for n in (3, 4, 5)
            dis_terms = Vector{Matrix{ComplexF64}}[[Z], [Z, Z]]
            Random.seed!(91)
            raw_pbc = build_heis_1d(n, HEIS_COEFFS; seed=91, 
                disordering_terms=dis_terms, disorder_strength=1.0, periodic=true)
            Random.seed!(91)
            raw_obc = build_heis_1d(n, HEIS_COEFFS; seed=91, 
                disordering_terms=dis_terms, disorder_strength=1.0, periodic=false)
            local_pbc = build_local_heis_1d(n, HEIS_COEFFS; seed=91,
                disordering_terms=dis_terms, disorder_strength=1.0, periodic=true)
            local_obc = build_local_heis_1d(n, HEIS_COEFFS; seed=91,
                disordering_terms=dis_terms, disorder_strength=1.0, periodic=false)
            @test raw_pbc.periodic === true
            @test raw_obc.periodic === false
            @test local_pbc.boundary == :periodic
            @test local_obc.boundary == :open
            @test any(term -> term.wraps_boundary, local_pbc.terms)
            @test all(term -> !term.wraps_boundary, local_obc.terms)
            @test !isapprox(raw_pbc.matrix, raw_obc.matrix; atol=1e-8)
            # Both must be Hermitian rescaled Hamiltonians
            @test isapprox(raw_pbc.matrix, raw_pbc.matrix'; atol=1e-12)
            @test isapprox(raw_obc.matrix, raw_obc.matrix'; atol=1e-12)
        end
    end

    # ------------------------------------------------------------------
    # (e) PBC fixture is bit-identical after the fix (no regression).
    # ------------------------------------------------------------------
    @testset "(e) PBC reconstruction is bit-identical to stored fixture" begin
        for n in (3, 4, 5)
            ham_path = test_hamiltonian_path(n)
            isfile(ham_path) || error("missing required Hamiltonian fixture: $ham_path")
            raw = BSON.load(ham_path)[:hamiltonian]
            @test raw.periodic === true
            # Rebuild from the stored disorder coefficients
            H_base = _construct_base_ham(HEIS_TERMS,
                raw.base_coeffs .* raw.rescaling_factor, n; periodic=true)
            H_dis = _construct_disordering_terms(
                Vector{Vector{Matrix{ComplexF64}}}(raw.disordering_terms),
                [Vector{Float64}(c) .* raw.rescaling_factor for c in raw.disordering_coeffs],
                n; periodic=true)
            H_phys = Matrix(H_base) + Matrix(H_dis)
            H_recon = H_phys ./ raw.rescaling_factor .+ raw.shift * I(2^n)
            @test isapprox(Matrix(raw.matrix), H_recon; atol=1e-12)
        end
    end

    # ------------------------------------------------------------------
    # (f) 2D _construct_2d_heisenberg_base — bond counts per BC.
    # ------------------------------------------------------------------
    @testset "(f) 2D base Hamiltonian bond counts vs BC" begin
        # Expected bond counts for a single ZZ term, uniform coeff 1.0.
        # Lx==2 or Ly==2 PBC double-counts the wrap (matches 1D n=2 convention) →
        # the wrap-bond is added on top of the regular bond, doubling its
        # multiplicity. tr(H²) counts m_b² for each bond, hence the squared mult.
        cases = [
            # (Lx, Ly, periodic_x, periodic_y, expected sum_b m_b²)
            (2, 3, true,  true,  18.0),  # x bonds (Lx=2): 3 bonds × m=2 → 12;  y bonds: 6 × m=1 → 6
            (2, 3, true,  false, 16.0),  # x: 12; y OBC: 4
            (2, 3, false, true,  9.0),   # x OBC: 3; y PBC: 6
            (2, 3, false, false, 7.0),   # x OBC: 3; y OBC: 4
            (3, 3, true,  true,  18.0),  # x bonds (Lx=3): 9; y bonds: 9 → 18
            (3, 3, true,  false, 15.0),  # x: 9; y OBC: 6
            (3, 3, false, false, 12.0),  # x OBC: 6; y OBC: 6
            (1, 4, true,  true,  4.0),   # Lx=1: no x bonds; y PBC: 4
            (1, 4, true,  false, 3.0),   # y OBC: 3
        ]
        terms = Vector{Matrix{ComplexF64}}[[Z, Z]]
        coeffs = [1.0]
        for (Lx, Ly, px, py, expected) in cases
            H = Matrix(_construct_2d_heisenberg_base(Lx, Ly, terms, coeffs;
                periodic_x=px, periodic_y=py))
            n = Lx * Ly
            @test _pauli_bond_msq(H, n) ≈ expected
        end

        # Lx=1, Ly=4 PBC matches 1D _construct_base_ham n=4 PBC
        H_2d_chain = Matrix(_construct_2d_heisenberg_base(1, 4, terms, coeffs;
            periodic_x=true, periodic_y=true))
        H_1d = Matrix(_construct_base_ham(terms, coeffs, 4; periodic=true))
        @test isapprox(H_2d_chain, H_1d; atol=1e-14)
    end

    # ------------------------------------------------------------------
    # (g) 2D _construct_disordering_terms_2d matches base when uniform coeffs.
    # ------------------------------------------------------------------
    @testset "(g) 2D disorder builder matches base bond pattern (uniform coeff)" begin
        for (Lx, Ly) in ((2, 3), (3, 3))
            n = Lx * Ly
            base_terms = Vector{Matrix{ComplexF64}}[[Z, Z]]
            base_coeffs = [1.0]
            for (px, py) in ((true, true), (true, false), (false, true), (false, false))
                H_base = Matrix(_construct_2d_heisenberg_base(Lx, Ly, base_terms, base_coeffs;
                    periodic_x=px, periodic_y=py))
                H_dis = Matrix(_construct_disordering_terms_2d(Lx, Ly,
                    Vector{Matrix{ComplexF64}}[[Z, Z]], [ones(n)];
                    periodic_x=px, periodic_y=py))
                @test isapprox(H_base, H_dis; atol=1e-12)
            end
        end
    end

    # ------------------------------------------------------------------
    # (h) 2D single-site disorder is BC-independent for any (px, py).
    # ------------------------------------------------------------------
    @testset "(h) 2D single-site disorder is BC-independent" begin
        Lx, Ly = 2, 3
        n = Lx * Ly
        dis_terms = Vector{Matrix{ComplexF64}}[[Z]]
        dis_coeffs = [Float64.(1:n)]
        H_pp = Matrix(_construct_disordering_terms_2d(Lx, Ly, dis_terms, dis_coeffs;
            periodic_x=true, periodic_y=true))
        H_ff = Matrix(_construct_disordering_terms_2d(Lx, Ly, dis_terms, dis_coeffs;
            periodic_x=false, periodic_y=false))
        @test isapprox(H_pp, H_ff; atol=1e-14)
    end

    # ------------------------------------------------------------------
    # (i) 2D build_tfim_2d fixture honours periodic_x / periodic_y.
    # ------------------------------------------------------------------
    @testset "(i) 2D build_tfim_2d mixed-BC propagation" begin
        Lx, Ly = 2, 3
        for (px, py) in ((true, true), (true, false), (false, true), (false, false))
            raw = build_tfim_2d(Lx, Ly; J=1.0, h=1.0, seed=91,
                disordering_terms=Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
                disorder_strength=1e-2,
                periodic_x=px, periodic_y=py)
            @test raw.periodic === (px && py)
            @test raw.Lx == Lx
            @test raw.Ly == Ly
            @test raw.periodic_x === px
            @test raw.periodic_y === py
            @test isapprox(raw.matrix, raw.matrix'; atol=1e-12)

            # The SSE builder uses the same per-axis geometry contract.
            sse = build_sse_tfim_model(
                Lx, Ly; seed=91, J=1.0, h=1.0, disorder_strength=1e-2,
                periodic_x=px, periodic_y=py)
            @test sse.Lx == Lx
            @test sse.Ly == Ly
            @test sse.periodic_x === px
            @test sse.periodic_y === py
            @test sse_reconstruction_error(sse, raw) <= 1e-10
        end

        cylinder = build_tfim_2d(Lx, Ly; J=1.0, h=1.0, seed=91,
            disorder_strength=1e-2, periodic_x=true, periodic_y=false)
        @test HamHam(cylinder, 1.0; spectral_validation=:probed).periodic === false
        @test_throws ArgumentError HamHam(
            merge(cylinder, (periodic=true,)), 1.0; spectral_validation=:probed)
        @test_throws ArgumentError HamHam(
            merge(cylinder, (periodic_y=nothing,)), 1.0; spectral_validation=:probed)

        # The four BC variants give four distinct matrices (different bond sets)
        raw_pp = build_tfim_2d(Lx, Ly; J=1.0, h=1.0, seed=91,
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
            disorder_strength=1e-2, periodic_x=true, periodic_y=true)
        raw_pf = build_tfim_2d(Lx, Ly; J=1.0, h=1.0, seed=91,
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
            disorder_strength=1e-2, periodic_x=true, periodic_y=false)
        raw_fp = build_tfim_2d(Lx, Ly; J=1.0, h=1.0, seed=91,
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
            disorder_strength=1e-2, periodic_x=false, periodic_y=true)
        raw_ff = build_tfim_2d(Lx, Ly; J=1.0, h=1.0, seed=91,
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
            disorder_strength=1e-2, periodic_x=false, periodic_y=false)
        @test !isapprox(raw_pp.matrix, raw_pf.matrix; atol=1e-8)
        @test !isapprox(raw_pp.matrix, raw_fp.matrix; atol=1e-8)
        @test !isapprox(raw_pp.matrix, raw_ff.matrix; atol=1e-8)
        @test !isapprox(raw_pf.matrix, raw_fp.matrix; atol=1e-8)
    end

    # ------------------------------------------------------------------
    # (j) expm_pauli_padded BC behaviour.
    # ------------------------------------------------------------------
    @testset "(j) expm_pauli_padded BC behaviour" begin
        n = 4
        c = 0.1

        # PBC wrap: closed form cos(c) I + i sin(c) P_wrap
        U_pbc = expm_pauli_padded([Z, Z], c, n, n; periodic=true)
        ZZ_wrap = Matrix(pad_term([Z, Z], n, n; periodic=true))
        @test isapprox(U_pbc, cos(c) * I(16) + 1im * sin(c) * ZZ_wrap; atol=1e-14)

        # OBC wrap: identity (NOT cos(c) I — see the docstring rationale).
        U_obc = expm_pauli_padded([Z, Z], c, n, n; periodic=false)
        @test isapprox(U_obc, Matrix{ComplexF64}(I, 16, 16); atol=1e-14)

        # Default kwarg matches periodic=true (backward compat)
        U_default = expm_pauli_padded([Z, Z], c, n, n)
        @test isapprox(U_default, U_pbc; atol=1e-14)

        # Single-site at any position: BC is irrelevant
        U_s_pbc = expm_pauli_padded([Z], c, n, n; periodic=true)
        U_s_obc = expm_pauli_padded([Z], c, n, n; periodic=false)
        @test isapprox(U_s_pbc, U_s_obc; atol=1e-14)
    end

    # ------------------------------------------------------------------
    # (k) _trotterize2 reproduces exp(-iδH) to Strang order for OBC.
    # ------------------------------------------------------------------
    @testset "(k) 1D Trotter convergence for OBC and PBC" begin
        # Build fixtures at n ∈ {3, 4, 5} for both BCs; sign convention in
        # _trotterize2 is U ≈ exp(+i δ H), see pauli_tools docstrings.
        for n in (3, 4, 5)
            Random.seed!(91)
            raw_pbc = build_heis_1d(n, HEIS_COEFFS; seed=91, 
                disordering_terms=Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
                disorder_strength=1.0, periodic=true)
            Random.seed!(91)
            raw_obc = build_heis_1d(n, HEIS_COEFFS; seed=91, 
                disordering_terms=Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
                disorder_strength=1.0, periodic=false)
            ham_pbc = HamHam(raw_pbc, 1.0)
            ham_obc = HamHam(raw_obc, 1.0)
            for ham in (ham_pbc, ham_obc)
                δ = 1e-3
                U_trot = _trotterize2(ham, δ, 1)
                U_exact = exp(1im * δ * Matrix(ham.data))
                # Strang 2nd-order: O(δ^3) ≈ 1e-9, with O(1) prefactor — pad
                # comfortably and assert well below the 1e-9 invariant target.
                @test opnorm(U_trot - U_exact) < 1e-9
            end

            # OBC Trotter must differ from PBC Trotter (no phantom wrap gate)
            U_pbc = _trotterize2(ham_pbc, 1e-3, 1)
            U_obc = _trotterize2(ham_obc, 1e-3, 1)
            @test !isapprox(U_pbc, U_obc; atol=1e-8)
        end
    end

    # ------------------------------------------------------------------
    # (l) Strang 2nd-order error scaling at OBC (sanity check).
    # ------------------------------------------------------------------
    @testset "(l) Strang Trotter 2nd-order error at OBC" begin
        n = 3
        Random.seed!(91)
        raw = build_heis_1d(n, HEIS_COEFFS; seed=91, 
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z], [Z, Z]],
            disorder_strength=1.0, periodic=false)
        ham = HamHam(raw, 1.0)
        H = Matrix(ham.data)
        # Pick δ values comfortably above the floating-point floor for the n=3
        # operator norm of [Z,Z]+[Y,Y]+[X,X] (norm ≲ 1 in rescaled units).
        δ1, δ2 = 1e-2, 5e-3
        err1 = opnorm(_trotterize2(ham, δ1, 1) - exp(1im * δ1 * H))
        err2 = opnorm(_trotterize2(ham, δ2, 1) - exp(1im * δ2 * H))
        # 2nd-order Strang: err ~ δ^3. err1/err2 ≈ (δ1/δ2)^3 = 8.
        # Allow a factor of 2 tolerance.
        ratio = err1 / err2
        @test 4 < ratio < 16
    end

    # ------------------------------------------------------------------
    # (m) `trotterize` (1st-order) honours `periodic`.
    # ------------------------------------------------------------------
    @testset "(m) 1st-order trotterize honours periodic" begin
        n = 3
        Random.seed!(91)
        raw_pbc = build_heis_1d(n, HEIS_COEFFS; seed=91, 
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z]],
            disorder_strength=1.0, periodic=true)
        Random.seed!(91)
        raw_obc = build_heis_1d(n, HEIS_COEFFS; seed=91, 
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z]],
            disorder_strength=1.0, periodic=false)
        ham_pbc = HamHam(raw_pbc, 1.0)
        ham_obc = HamHam(raw_obc, 1.0)
        T, M = 0.05, 10
        U_pbc = trotterize(ham_pbc, T, M)
        U_obc = trotterize(ham_obc, T, M)
        # Both produce unitaries (Pauli-string product of unitaries)
        @test isapprox(U_pbc * U_pbc', I(2^n); atol=1e-10)
        @test isapprox(U_obc * U_obc', I(2^n); atol=1e-10)
        @test !isapprox(U_pbc, U_obc; atol=1e-8)
    end

    # ------------------------------------------------------------------
    # (o) 2D HamHam fed to TrottTrott raises an explicit error.
    # ------------------------------------------------------------------
    @testset "(o) TrottTrott rejects 2D HamHam (qf-91g.3 guard)" begin
        raw2d = build_tfim_2d(2, 3; J=1.0, h=1.0, seed=91,
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z]],
            disorder_strength=1e-2, periodic_x=true, periodic_y=true)
        ham2d = HamHam(raw2d, 1.0)
        @test_throws ArgumentError TrottTrott(ham2d, 0.5, 4)

        # 1D HamHam (PBC + OBC) still constructs fine
        raw1d_pbc = build_heis_1d(4, HEIS_COEFFS; seed=91,
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z]],
            disorder_strength=1.0, periodic=true)
        raw1d_obc = build_heis_1d(4, HEIS_COEFFS; seed=91, 
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z]],
            disorder_strength=1.0, periodic=false)
        @test TrottTrott(HamHam(raw1d_pbc, 1.0), 0.5, 4) isa TrottTrott
        @test TrottTrott(HamHam(raw1d_obc, 1.0), 0.5, 4) isa TrottTrott
    end

    # ------------------------------------------------------------------
    # (n) Cross-domain Krylov spectral gap on OBC fixture.
    # ------------------------------------------------------------------
    # qf-91g: an OBC fixture must produce a well-defined Lindbladian with a
    # finite spectral gap, distinct from the PBC version. Keeps n small so the
    # test stays SANDBOX-tier.
    @testset "(n) OBC Lindbladian via Krylov has a finite gap and differs from PBC" begin
        using QuantumFurnace: Config, KMS, Lindbladian, EnergyDomain,
            krylov_spectral_gap

        n = 3
        Random.seed!(91)
        raw_pbc = build_heis_1d(n, HEIS_COEFFS; seed=91, 
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z]],
            disorder_strength=1.0, periodic=true)
        Random.seed!(91)
        raw_obc = build_heis_1d(n, HEIS_COEFFS; seed=91, 
            disordering_terms=Vector{Matrix{ComplexF64}}[[Z]],
            disorder_strength=1.0, periodic=false)
        ham_pbc = HamHam(raw_pbc, 5.0)
        ham_obc = HamHam(raw_obc, 5.0)

        cfg_pbc = Config(;
            sim=Lindbladian(), domain=EnergyDomain(), construction=KMS(),
            num_qubits=n,
            with_linear_combination=true,
            beta=5.0,
            sigma=1.0/5.0,
            a=5.0 / 30.0,
            s=0.25,
            num_energy_bits=6,
            w0=1.0,
            t0=0.5,
            num_trotter_steps_per_t0=1,
        )
        cfg_obc = cfg_pbc  # same config: only the HamHam differs

        jump_paulis = [[X], [Y], [Z]]
        num_jumps = length(jump_paulis) * n
        jump_norm = sqrt(num_jumps)
        function build_jumps(ham)
            jumps = JumpOp[]
            for pauli in jump_paulis
                for site in 1:n
                    op = Matrix(pad_term(pauli, n, site)) ./ jump_norm
                    op_eb = ham.eigvecs' * op * ham.eigvecs
                    push!(jumps, JumpOp(op, op_eb, op == transpose(op), op == op'))
                end
            end
            return jumps
        end

        jumps_pbc = build_jumps(ham_pbc)
        jumps_obc = build_jumps(ham_obc)

        gap_pbc = krylov_spectral_gap(cfg_pbc, ham_pbc, jumps_pbc).spectral_gap
        gap_obc = krylov_spectral_gap(cfg_obc, ham_obc, jumps_obc).spectral_gap

        @test isfinite(gap_pbc) && gap_pbc > 1e-6
        @test isfinite(gap_obc) && gap_obc > 1e-6
        # PBC and OBC Hamiltonians differ ⇒ Lindbladian spectra differ.
        @test !isapprox(gap_pbc, gap_obc; rtol=1e-3)
    end
end
