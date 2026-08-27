@testset "ITensor fused doubled-site conventions" begin
    @testset "explicit QF-to-fused permutation" begin
        for (n, physical_dim) in ((1, 2), (2, 2), (2, 3))
            global_dim = physical_dim^n
            fused_local_dim = physical_dim^2
            permutation = QFITensors.qf_to_fused_permutation(
                n; physical_dim=physical_dim)
            @test sort(permutation) == collect(1:global_dim^2)

            qf_vector = ComplexF64.(1:global_dim^2)
            fused = QFITensors.vec_to_fused(
                qf_vector, n; physical_dim=physical_dim)
            @test QFITensors.fused_to_vec(
                fused, n; physical_dim=physical_dim) ≈ qf_vector

            for bra_global in 0:(global_dim - 1), ket_global in 0:(global_dim - 1)
                qf_index = ket_global + global_dim * bra_global + 1
                fused_offset = 0
                for site in 1:n
                    qf_stride = physical_dim^(n - site)
                    ket = (ket_global ÷ qf_stride) % physical_dim
                    bra = (bra_global ÷ qf_stride) % physical_dim
                    local_state = ket + physical_dim * bra
                    fused_offset += local_state * fused_local_dim^(site - 1)
                end
                @test fused[fused_offset + 1] ≈ qf_vector[qf_index]
            end
        end

        @test_throws ArgumentError QFITensors.qf_to_fused_permutation(0)
        @test_throws ArgumentError QFITensors.qf_to_fused_permutation(
            2; physical_dim=0)
        @test_throws DimensionMismatch QFITensors.vec_to_fused(zeros(15), 2)
    end

    @testset "complex matrix and superoperator actions" begin
        rng = MersenneTwister(0x51f17e)
        n = 2
        physical_dim = 2
        global_dim = physical_dim^n
        X = randn(rng, ComplexF64, global_dim, global_dim)
        A = randn(rng, ComplexF64, global_dim, global_dim)
        B = randn(rng, ComplexF64, global_dim, global_dim)

        fused_X = QFITensors.matrix_to_fused(
            X, n; physical_dim=physical_dim)
        @test QFITensors.fused_to_matrix(
            fused_X, n; physical_dim=physical_dim) ≈ X

        for transformed in (transpose(X), conj(X), adjoint(X))
            transformed_fused = QFITensors.matrix_to_fused(
                transformed, n; physical_dim=physical_dim)
            @test QFITensors.fused_to_matrix(
                transformed_fused, n; physical_dim=physical_dim) ≈ transformed
        end

        qf_action = kron(transpose(B), A)
        fused_action = QFITensors.superoperator_to_fused(
            qf_action, n; physical_dim=physical_dim)
        @test QFITensors.fused_to_superoperator(
            fused_action, n; physical_dim=physical_dim) ≈ qf_action
        @test fused_action * fused_X ≈ QFITensors.matrix_to_fused(
            A * X * B, n; physical_dim=physical_dim)

        # Pauli Y makes transpose, conjugation, and adjoint genuinely distinct.
        Y = ComplexF64[0 -im; im 0]
        probe = randn(rng, ComplexF64, 2, 2)
        left = QFITensors.superoperator_to_fused(kron(I(2), Y), 1)
        right = QFITensors.superoperator_to_fused(kron(transpose(Y), I(2)), 1)
        fused_probe = QFITensors.matrix_to_fused(probe, 1)
        @test left * fused_probe ≈ QFITensors.matrix_to_fused(Y * probe, 1)
        @test right * fused_probe ≈ QFITensors.matrix_to_fused(probe * Y, 1)
        @test vec(Y * probe * Y) ≈ kron(transpose(Y), Y) * vec(probe)

        site_1_operator = kron(Y, I(2))
        site_2_operator = kron(I(2), Y)
        for site_operator in (site_1_operator, site_2_operator)
            fused_left_action = QFITensors.superoperator_to_fused(
                kron(I(global_dim), site_operator), n)
            @test fused_left_action * fused_X ≈
                  QFITensors.matrix_to_fused(site_operator * X, n)
        end
    end

    @testset "small dense MPS and MPO bridges" begin
        rng = MersenneTwister(0xd35e)
        sites = QFITensors.fused_siteinds(3; physical_dim=2)
        @test length(sites) == 3
        @test all(ITensors.dim(site) == 4 for site in sites)

        chain_dim = 4^3
        vector = randn(rng, ComplexF64, chain_dim)
        state = QFITensors.dense_to_mps(vector, sites)
        @test QFITensors.mps_to_dense(state, sites) ≈ vector atol=2e-12 rtol=2e-12

        matrix = randn(rng, ComplexF64, chain_dim, chain_dim)
        operator = QFITensors.dense_to_mpo(matrix, sites)
        @test QFITensors.mpo_to_dense(operator, sites) ≈ matrix atol=2e-11 rtol=2e-11
        image = ITensorMPS.apply(
            operator, state; cutoff=0.0, maxdim=chain_dim)
        @test QFITensors.mps_to_dense(image, sites) ≈
              matrix * vector atol=2e-11 rtol=2e-11

        matrix_unit = zeros(ComplexF64, chain_dim, chain_dim)
        matrix_unit[37, 11] = 1 - 2im
        unit_mpo = QFITensors.dense_to_mpo(matrix_unit, sites)
        @test QFITensors.mpo_to_dense(unit_mpo, sites) ≈ matrix_unit atol=2e-12 rtol=2e-12

        @test_throws DimensionMismatch QFITensors.dense_to_mps(
            zeros(ComplexF64, chain_dim - 1), sites)
        @test_throws DimensionMismatch QFITensors.dense_to_mpo(
            zeros(ComplexF64, chain_dim, chain_dim - 1), sites)
        @test_throws ArgumentError QFITensors.dense_to_mps(
            zeros(ComplexF64, 1), QFITensors.fused_siteinds(5))
        @test_throws ArgumentError QFITensors.dense_to_mps(
            zeros(ComplexF64, 1),
            QFITensors.fused_siteinds(4; physical_dim=3),
        )
        nonsquare_sites = ITensorMPS.siteinds("Qudit", 2; dim=6)
        @test_throws ArgumentError QFITensors.dense_to_mps(
            zeros(ComplexF64, 1), nonsquare_sites)
        mixed_sites = [ITensors.Index(4, "Site"), ITensors.Index(9, "Site")]
        @test_throws ArgumentError QFITensors.dense_to_mps(
            zeros(ComplexF64, 1), mixed_sites)
        @test_throws ArgumentError QFITensors.dense_to_mps(
            ComplexF64[], ITensors.Index[])
    end

    @testset "physical partial trace" begin
        rng = MersenneTwister(0x7fd)
        n = 2
        physical_dim = 2
        global_dim = physical_dim^n

        bell_amplitude = Matrix{ComplexF64}(I, global_dim, global_dim) /
                         sqrt(global_dim)
        bell_fused = QFITensors.matrix_to_fused(
            bell_amplitude, n; physical_dim=physical_dim)
        bell_density = QFITensors.physical_partial_trace(
            bell_fused, n; physical_dim=physical_dim)
        @test bell_density ≈ Matrix{ComplexF64}(I, global_dim, global_dim) /
                             global_dim atol=2e-14 rtol=2e-14

        raw = randn(rng, ComplexF64, global_dim, global_dim)
        rho = raw * adjoint(raw)
        rho ./= real(tr(rho))
        sqrt_rho = Matrix(sqrt(Hermitian(rho)))
        purification = QFITensors.matrix_to_fused(
            sqrt_rho, n; physical_dim=physical_dim)
        @test QFITensors.physical_partial_trace(
            purification, n; physical_dim=physical_dim) ≈ rho atol=2e-13 rtol=2e-13

        sites = QFITensors.fused_siteinds(n; physical_dim=physical_dim)
        purification_mps = QFITensors.dense_to_mps(purification, sites)
        @test QFITensors.physical_partial_trace(
            purification_mps, sites) ≈ rho atol=2e-12 rtol=2e-12

        unnormalised = randn(rng, ComplexF64, global_dim, global_dim)
        unnormalised_fused = QFITensors.matrix_to_fused(unnormalised, n)
        unnormalised_density = QFITensors.physical_partial_trace(
            unnormalised_fused, n)
        @test real(tr(unnormalised_density)) ≈ norm(unnormalised_fused)^2
    end
end
