"""
Targeted tests for `validate_jump_pairing`.

Exercises edge cases beyond what `test_non_hermitian_jumps.jl` covers:
empty input, atol behaviour, multi-pair sets, mixed Hermitian flags,
size mismatches.

These run on synthetic 2x2 / 4x4 matrices — no Hamiltonian / OFT machinery
required, so this file is fast.
"""

@testset "validate_jump_pairing edge cases" begin
    # 2x2 fixtures
    sigma_plus  = ComplexF64[0 1; 0 0]
    sigma_minus = ComplexF64[0 0; 1 0]
    pauli_x     = ComplexF64[0 1; 1 0]
    pauli_z     = ComplexF64[1 0; 0 -1]

    # Helper: minimal JumpOp with arbitrary data (basis copy is irrelevant for the validator).
    _mkjump(d::AbstractMatrix; hermitian::Bool=false) =
        JumpOp(Matrix{ComplexF64}(d), Matrix{ComplexF64}(d), false, hermitian)

    # 4x4 fixtures (two-site)
    I2 = Matrix{ComplexF64}(I, 2, 2)
    sigma_plus_site1  = kron(sigma_plus,  I2)
    sigma_minus_site1 = kron(sigma_minus, I2)
    sigma_plus_site2  = kron(I2, sigma_plus)
    sigma_minus_site2 = kron(I2, sigma_minus)

    @testset "empty jump set returns nothing" begin
        @test validate_jump_pairing(JumpOp[]) === nothing
    end

    @testset "single Hermitian jump (hermitian=true) returns nothing" begin
        @test validate_jump_pairing(JumpOp[_mkjump(pauli_x; hermitian=true)]) === nothing
    end

    @testset "all Hermitian (multiple) returns nothing" begin
        jumps = JumpOp[
            _mkjump(pauli_x; hermitian=true),
            _mkjump(pauli_z; hermitian=true),
        ]
        @test validate_jump_pairing(jumps) === nothing
    end

    @testset "single paired (sigma+, sigma-) returns nothing" begin
        jumps = JumpOp[_mkjump(sigma_plus), _mkjump(sigma_minus)]
        @test validate_jump_pairing(jumps) === nothing
    end

    @testset "two paired sets on different sites returns nothing" begin
        jumps = JumpOp[
            _mkjump(sigma_plus_site1),
            _mkjump(sigma_minus_site1),
            _mkjump(sigma_plus_site2),
            _mkjump(sigma_minus_site2),
        ]
        @test validate_jump_pairing(jumps) === nothing
    end

    @testset "mismatched-site pair (s+_1, s-_2) raises" begin
        jumps = JumpOp[
            _mkjump(sigma_plus_site1),
            _mkjump(sigma_minus_site2),
        ]
        @test_throws ArgumentError validate_jump_pairing(jumps)
    end

    @testset "single non-Hermitian raises" begin
        jumps = JumpOp[_mkjump(sigma_plus)]
        @test_throws ArgumentError validate_jump_pairing(jumps)
    end

    @testset "self-paired single A=A† marked hermitian=false raises" begin
        # A = A† but no other jump in the set: validation excludes self-match
        # (j == k), so the only candidate is missing -> reject.
        jumps = JumpOp[_mkjump(pauli_x; hermitian=false)]
        @test_throws ArgumentError validate_jump_pairing(jumps)
    end

    @testset "two copies of A=A† non-Hermitian: each pairs with the other" begin
        jumps = JumpOp[
            _mkjump(pauli_x; hermitian=false),
            _mkjump(pauli_x; hermitian=false),
        ]
        @test validate_jump_pairing(jumps) === nothing
    end

    @testset "adjoint multiplicities must balance" begin
        A = sigma_plus_site1
        Adag = A'
        unbalanced = JumpOp[_mkjump(A), _mkjump(A), _mkjump(Adag)]
        @test_throws ArgumentError validate_jump_pairing(unbalanced)

        balanced = JumpOp[_mkjump(A), _mkjump(A), _mkjump(Adag), _mkjump(Adag)]
        @test validate_jump_pairing(balanced) === nothing
    end

    @testset "atol behaviour" begin
        # Within default atol=1e-12: noise ~1e-13 -> pass.
        Adag_close = (1.0 + 1e-13) * sigma_plus'
        jumps_close = JumpOp[_mkjump(sigma_plus), _mkjump(Adag_close)]
        @test validate_jump_pairing(jumps_close) === nothing

        # Outside default atol: noise ~1e-10 -> fail.
        Adag_far = (1.0 + 1e-10) * sigma_plus'
        jumps_far = JumpOp[_mkjump(sigma_plus), _mkjump(Adag_far)]
        @test_throws ArgumentError validate_jump_pairing(jumps_far)

        # But with relaxed atol=1e-9 -> pass.
        @test validate_jump_pairing(jumps_far; atol=1e-9) === nothing

        # Tighter atol=1e-15 catches even normally-equal pairs (no noise -> still pass).
        jumps_clean = JumpOp[_mkjump(sigma_plus), _mkjump(sigma_minus)]
        @test validate_jump_pairing(jumps_clean; atol=1e-15) === nothing
    end

    @testset "allow_unpaired_nonhermitian=true bypasses the check" begin
        # Single unpaired NH would normally fail.
        jumps_unpaired = JumpOp[_mkjump(sigma_plus)]
        @test validate_jump_pairing(jumps_unpaired;
                                    allow_unpaired_nonhermitian=true) === nothing
    end

    @testset "size mismatch is rejected (4x4 vs 2x2)" begin
        # Even though sigma_plus_site1[1:2,1:2] resembles sigma_minus, the
        # implementation requires matching shapes — mismatched-size pairs
        # are not partners.
        jumps = JumpOp[
            _mkjump(sigma_plus_site1),  # 4x4
            _mkjump(sigma_minus),        # 2x2
        ]
        @test_throws ArgumentError validate_jump_pairing(jumps)
    end

    @testset "mismarked non-Hermitian partner is rejected" begin
        A = sigma_plus_site1
        jumps = JumpOp[
            _mkjump(A; hermitian=false),
            _mkjump(A'; hermitian=true),
        ]
        @test_throws ArgumentError validate_jump_pairing(jumps)
    end

    @testset "Hermitian flag is validated in both stored bases" begin
        bogus_herm = ComplexF64[0 1; 0 0]  # not actually Hermitian
        @test_throws ArgumentError validate_jump_pairing(JumpOp[
            _mkjump(bogus_herm; hermitian=true),
        ])

        stale_basis = JumpOp(pauli_x, bogus_herm, false, true)
        @test_throws ArgumentError validate_jump_pairing(JumpOp[stale_basis])
    end

    @testset "paired eigenbasis data must also be adjoints" begin
        A = _mkjump(sigma_plus)
        stale_partner = JumpOp(sigma_minus, sigma_plus, false, false)
        @test_throws ArgumentError validate_jump_pairing(JumpOp[A, stale_partner])
    end

    @testset "informative error message includes index of unpaired jump" begin
        jumps = JumpOp[_mkjump(sigma_plus)]
        try
            validate_jump_pairing(jumps)
            @test false  # should not reach
        catch e
            @test isa(e, ArgumentError)
            msg = sprint(showerror, e)
            @test occursin("[1]", msg)
            @test occursin("KMS detailed balance", msg)
            @test occursin("allow_unpaired_nonhermitian", msg)
        end
    end

    @testset "informative error message reports multiple indices" begin
        # Two distinct unpaired NH jumps -> indices [1, 2].
        jumps = JumpOp[_mkjump(sigma_plus_site1), _mkjump(sigma_plus_site2)]
        try
            validate_jump_pairing(jumps)
            @test false
        catch e
            @test isa(e, ArgumentError)
            msg = sprint(showerror, e)
            @test occursin("[1, 2]", msg)
        end
    end
end
