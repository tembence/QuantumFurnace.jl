# test/test_dll_kms_db_sandbox.jl
#
# Sandbox shadow of test_dll_kms_db.jl::(j). The heavy (j)
# testset is gated NO_SANDBOX and runs the same multi-t0 convergence sweep
# at Nt = 4096 (r_D = 12). This shadow keeps the complete t0-factor ladder
# on the same β = 10 fixture at r_D = 10 (Nt = 1024).
#
# Why a separate file: inlined inside test_dll_kms_db.jl the (a)..(i)
# testsets accumulate Nt = 1024 NUFFT workspaces; layering a fresh
# Nt = 2048 pair on top crosses the 1.5 GB heap-size-hint, so we let
# the inter-file `GC.gc(true)` in runtests.jl clear the prior state.
#
# Multiplying t0 by 1, 2, and 4 expands t_max at fixed Nt. The errors must
# decrease strictly and the factor-4 endpoint must reach 1e-9.

using LinearAlgebra: opnorm
using Test
using QuantumFurnace


function _sandbox_dll_meta_cfg_t(beta::Real, num_energy_bits::Int, t0_factor::Real)
    Config(;
        sim = Lindbladian(), domain = TimeDomain(), construction = DLL(),
        num_qubits = 3, with_linear_combination = true,
        beta = beta, sigma = 1.0 / beta, a = beta / 30, s = 0.4,
        num_energy_bits = num_energy_bits,
        t0 = t0_factor * 2π / (2^num_energy_bits * 0.05),
        num_trotter_steps_per_t0 = 10,
        filter = DLLMetropolisFilter(beta; S = 2.0),
    )
end

function _sandbox_dll_meta_cfg_b(beta::Real, num_energy_bits::Int)
    Config(;
        sim = Lindbladian(), domain = BohrDomain(), construction = DLL(),
        num_qubits = 3, with_linear_combination = true,
        beta = beta, sigma = 1.0 / beta, a = beta / 30, s = 0.4,
        num_energy_bits = num_energy_bits,
        t0 = 2π / (2^num_energy_bits * 0.05),
        num_trotter_steps_per_t0 = 10,
        filter = DLLMetropolisFilter(beta; S = 2.0),
    )
end


@testset "DLL KMS-DB (j-sb) sandbox shadow" begin
    @testset "(j-sb) DLL Metropolis Bohr ↔ Time t0 ladder" begin
        beta = 10.0
        r_D = 10
        sys = make_dll_n3_system(beta)
        L_b = Matrix(construct_lindbladian(sys.jumps,
                                            _sandbox_dll_meta_cfg_b(beta, r_D),
                                            sys.ham))
        errors = Float64[]
        for factor in (1.0, 2.0, 4.0)
            L_t = Matrix(construct_lindbladian(sys.jumps,
                _sandbox_dll_meta_cfg_t(beta, r_D, factor), sys.ham))
            push!(errors, opnorm(L_b - L_t))
            L_t = nothing
            GC.gc()
        end
        L_b = nothing; sys = nothing
        GC.gc()

        @test all(diff(errors) .< 0)
        @test errors[end] <= 1e-9
        @info "(j-sb) DLL Metropolis Bohr ↔ Time" β=beta r_D errors
    end
end
