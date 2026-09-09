using Test
using QuantumFurnace
using LinearAlgebra
using BSON

# Test the sweep harness β_phys-first mode against the legacy
# β_alg-first mode. We exercise `sweep_mixing_times` (KMS BohrDomain, dense)
# at the smallest cell (n=3) so the test finishes in a few seconds.

const _BPS_HAM_PATH = test_hamiltonian_path(3)

# Read the rescaling_factor once so we can predict the derived β_alg.
const _BPS_HAM = QuantumFurnace._load_hamiltonian_bson(_BPS_HAM_PATH, 10.0)
const _BPS_RESCALE = _BPS_HAM.rescaling_factor

@testset "sweep_mixing_times / sweep_channel_mixing β_phys mode" begin

    @testset "(a) error: both beta_values and beta_phys_values" begin
        @test_throws ArgumentError sweep_mixing_times(
            [3], [10.0]; beta_phys_values = [1.0],
            mode = :L, method = :krylov,
            domain = BohrDomain(),
            target_epsilon = 1e-3,
            seeds = [42], use_threads = false,
        )
    end

    @testset "(b) error: neither beta_values nor beta_phys_values" begin
        @test_throws ArgumentError sweep_mixing_times(
            [3]; mode = :L, method = :krylov,
            domain = BohrDomain(),
            target_epsilon = 1e-3,
            seeds = [42], use_threads = false,
        )
    end

    @testset "(c) legacy β_alg mode: result rows carry β_phys / β_alg / rescaling_factor" begin
        results = sweep_mixing_times(
            [3], [10.0];
            mode = :L, method = :krylov,
            domain = BohrDomain(),
            target_epsilon = 1e-3,
            seeds = [42], use_threads = false,
        )
        @test length(results) == 1
        r = results[1]
        @test r.n == 3
        @test r.beta == 10.0                          # legacy alias = β_alg
        @test r.beta_alg == 10.0
        @test isapprox(r.beta_phys, 10.0 / _BPS_RESCALE; rtol=1e-12)
        @test r.rescaling_factor ≈ _BPS_RESCALE
        @test isfinite(r.mixing_time)
    end

    @testset "(d) β_phys mode: β_alg derived per cell; sidecar uses 'betaphys' tag" begin
        β_phys = 1.0
        mktempdir() do tmp
            results = sweep_mixing_times(
                [3];
                beta_phys_values = [β_phys],
                mode = :L, method = :krylov,
                domain = BohrDomain(),
                target_epsilon = 1e-3,
                seeds = [42], use_threads = false,
                output_dir = tmp,
            )
            @test length(results) == 1
            r = results[1]
            β_alg_expected = β_phys * _BPS_RESCALE
            @test r.n == 3
            @test r.beta_phys == β_phys
            @test r.beta_alg ≈ β_alg_expected rtol=1e-12
            @test r.beta == r.beta_alg                # legacy alias mirrors β_alg
            @test r.rescaling_factor ≈ _BPS_RESCALE
            @test isfinite(r.mixing_time)

            # Sidecar filename must carry the betaphys<β_phys> tag, not beta<β_alg>.
            files = filter(endswith(".bson"), readdir(tmp))
            expected_file = "sweep_n3_betaphys1_seed42_L_KMS_Bohr.bson"
            @test files == [expected_file]

            # Loaded sidecar dict carries the new keys.
            loaded = BSON.load(joinpath(tmp, expected_file), QuantumFurnace)[:result]
            @test loaded[:beta_phys] == β_phys
            @test loaded[:beta_alg] ≈ β_alg_expected
            @test haskey(loaded, :rescaling_factor)
        end
    end

    @testset "(e) Gibbs-state byte-identity: β_phys path matches β_alg path" begin
        # In β_phys-first mode with β_phys=1 at n=3, the constructed HamHam
        # has β_alg = rescaling_factor. The legacy path with β_alg = rescale
        # should produce a byte-identical Gibbs state.
        ham_phys = HamHam(QuantumFurnace._parse_hamiltonian_bson(_BPS_HAM_PATH);
                          beta_phys = 1.0)
        ham_alg  = QuantumFurnace._load_hamiltonian_bson(_BPS_HAM_PATH, _BPS_RESCALE)
        @test Matrix(ham_phys.gibbs) ≈ Matrix(ham_alg.gibbs) atol=1e-15
    end
end
