using Test, QuantumFurnace, LinearAlgebra

@testset "Executable public tutorials" begin
    # Execute the actual published source in isolated namespaces. Each tutorial
    # checks its physical invariant/reference, including deliberately failed KMS
    # diagnostics and partial-budget output, without private Hamiltonian assets.
    for name in ("hamiltonian", "lindbladian", "thermalize", "custom_filters", "diagnostics")
        @testset "$name" begin
            tutorial = Module(gensym(:Tutorial))
            @test begin
                Base.include(tutorial,
                    joinpath(@__DIR__, "..", "docs", "src", "literate", "tutorial_" * name * ".jl"))
                true
            end
        end
    end
end

@testset "Public Markdown examples" begin
    # Contract examples deliberately build on earlier snippets, just as on the
    # page. Isolation by page avoids namespace leakage into legacy test files.
    for page in ("api_contract.md", "theory_filters.md")
        scope = Module(gensym(:PublicExamples))
        Core.eval(scope, :(using QuantumFurnace, LinearAlgebra))
        contents = read(joinpath(@__DIR__, "..", "docs", "src", page), String)
        for (i, block) in enumerate(eachmatch(r"```julia\n(.*?)```"s, contents))
            @testset "$page example $i" begin
                @test begin
                    Base.include_string(scope, block.captures[1], page * ":example_$i")
                    true
                end
            end
        end
    end
    readme = read(joinpath(@__DIR__, "..", "README.md"), String)
    quick_start = split(split(readme, "## Quick start"; limit=2)[2], "## Documentation"; limit=2)[1]
    scope = Module(gensym(:Readme))
    block = match(r"```julia\n(.*?)```"s, quick_start)
    @test block !== nothing
    for (i, example) in enumerate(eachmatch(r"```julia\n(.*?)```"s, quick_start))
        Base.include_string(scope, example.captures[1], "README.md:quick_start_$i")
    end
    result = getproperty(scope, :result)
    @test result.trajectory.all_converged
    @test result.trajectory.trace_norms ≈ 2result.trajectory.distances
end
