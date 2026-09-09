using Test
using LinearAlgebra

@testset "Thermalize recording and cooperative budgets" begin
    config = make_config(Thermalize(), EnergyDomain(); num_qubits=3, mixing_time=0.1)
    run = (; kwargs...) -> run_thermalize(N3_JUMPS, config, N3_HAM;
        convergence_cutoff=0, kwargs...)

    full = run(num_steps=7, save_states=true, hermitize=false)
    seen = Int[]
    sparse = run(num_steps=7, record_steps=[0, 2, 7], save_states=true,
        hermitize=false, observation_callback=(step, rho) -> push!(seen, step))
    @test sparse.metadata[:recorded_steps] == seen == [0, 2, 7]
    @test sparse.time_steps ≈ [0, 2, 7] .* config.delta atol=1e-15
    @test sparse.final_dm ≈ full.final_dm atol=1e-13
    @test sparse.metadata[:states][2] ≈ full.metadata[:states][3] atol=1e-13
    @test sparse.metadata[:states][end] ≈ sparse.final_dm atol=1e-15
    @test sparse.metadata[:completed_steps] == 7
    @test sparse.metadata[:failure] === nothing
    @test !sparse.metadata[:hermitized]
    for (rho, distance) in zip(sparse.metadata[:states], sparse.trace_distances)
        @test distance ≈ sum(svdvals(rho - N3_HAM.gibbs)) / 2 atol=1e-14
    end

    endpoint = run(num_steps=7, save_every=3)
    @test endpoint.metadata[:recorded_steps] == [0, 3, 6, 7]
    @test endpoint.metadata[:states] === nothing
    @test endpoint.final_dm ≈ full.final_dm atol=1e-13
    zero_steps = run(num_steps=0, record_steps=[0], save_states=true)
    @test zero_steps.metadata[:completed_steps] == 0
    @test zero_steps.metadata[:recorded_steps] == [0]
    @test zero_steps.final_dm ≈ Matrix{ComplexF64}(I, N3_DIM, N3_DIM) / N3_DIM atol=1e-15

    calls = Ref(0)
    budget = () -> begin
        calls[] += 1
        calls[] <= 3 || throw(QuantumFurnace._WorkLimit(:steps))
    end
    partial = run(num_steps=7, record_steps=[0, 7], work_callback=budget,
        save_states=true, hermitize=false)
    @test partial.metadata[:completed_steps] == 3
    @test partial.metadata[:recorded_steps] == [0, 3]
    @test partial.metadata[:failure].reason == :steps
    @test partial.final_dm ≈ full.metadata[:states][4] atol=1e-13
    @test partial.metadata[:states][end] ≈ partial.final_dm atol=1e-15
    stopped = run(num_steps=7, work_callback=() -> throw(QuantumFurnace._WorkLimit(:time)))
    @test stopped.metadata[:completed_steps] == 0
    @test stopped.metadata[:recorded_steps] == [0]
    @test_throws ErrorException run(num_steps=1, work_callback=() -> error("callback failure"))
    @test_throws ErrorException run(num_steps=1, observation_callback=(step, rho) -> error("observation failure"))
    @test_throws ArgumentError run(num_steps=-1)
    for grid in (Int[], [1, 7], [0, 6], [0, 2, 2, 7], [0.0, 7.0], [0, 8, 7])
        @test_throws ArgumentError run(num_steps=7, record_steps=grid)
    end
    for cutoff in (-1.0, Inf, NaN)
        @test_throws ArgumentError run_thermalize(N3_JUMPS, config, N3_HAM; convergence_cutoff=cutoff)
    end
    converged = run_thermalize(N3_JUMPS, config, N3_HAM;
        num_steps=7, record_steps=[0, 2, 7], convergence_cutoff=2)
    @test converged.metadata[:completed_steps] == 2
    @test converged.metadata[:recorded_steps] == [0, 2]
end

@testset "NUFFT cache covers reflected even-grid endpoint" begin
    energies = collect(-16:15) .* 0.05
    times = collect(-16:15) .* (2pi / (32 * 0.05))
    bohr = [0.0 0.13; -0.13 0.0]
    filter = GaussianFilter(0.4)
    prefactors = QuantumFurnace._prepare_oft_nufft_prefactors(bohr, times, energies, filter)
    @test prefactors.energy_labels[1:length(energies)] ≈ energies atol=1e-15
    @test length(prefactors.energy_labels) == length(energies) + 1
    omega = -first(energies)
    cached = QuantumFurnace._prefactor_view(prefactors, omega)
    reference = [sum(time_kernel(filter, t) * cis((nu - omega) * t) for t in times) for nu in bohr]
    @test cached ≈ reference atol=1e-10 rtol=1e-10
    @test_throws KeyError QuantumFurnace._prefactor_view(prefactors, 0.825)
end
