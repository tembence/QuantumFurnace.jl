# Build from the local checkout without changing either tracked environment.
# Deployment is an explicit opt-in, including in CI.
import Pkg
const deploy_setting = get(ENV, "QF_DOCS_DEPLOY", "false")
deploy_setting in ("false", "true") || error("QF_DOCS_DEPLOY must be true or false.")
const docs_env = mktempdir()
cp(joinpath(@__DIR__, "Project.toml"), joinpath(docs_env, "Project.toml"))
Pkg.activate(docs_env)
Pkg.develop(Pkg.PackageSpec(path=dirname(@__DIR__)))
Pkg.instantiate()

using Documenter, Literate, QuantumFurnace
using LinearAlgebra
BLAS.set_num_threads(1)
realpath(dirname(dirname(pathof(QuantumFurnace)))) == realpath(dirname(@__DIR__)) ||
    error("Documentation must execute against the local QuantumFurnace checkout.")

# Stage sources so generating executed examples never overwrites tracked files.
const staged = mktempdir()
const sources = joinpath(staged, "src")
cp(joinpath(@__DIR__, "src"), sources)
const generated = joinpath(sources, "generated")
rm(generated; recursive=true, force=true)
mkpath(generated)
for filename in sort(readdir(joinpath(sources, "literate")))
    endswith(filename, ".jl") || continue
    input = joinpath(sources, "literate", filename)
    Literate.markdown(input, generated; documenter=true)
    Literate.notebook(input, generated; execute=true)
    GC.gc(true)
end

makedocs(
    root=@__DIR__, source=sources, build=joinpath(@__DIR__, "build"),
    sitename="QuantumFurnace.jl", checkdocs=:none, remotes=nothing,
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://benzabonanza.github.io/QuantumFurnace.jl/dev/",
        repolink="https://github.com/benzabonanza/QuantumFurnace.jl",
        edit_link=nothing,
    ),
    modules=[QuantumFurnace],
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Creating a Hamiltonian" => "generated/tutorial_hamiltonian.md",
            "Simulating a Gibbs sampler" => "generated/tutorial_thermalize.md",
            "Constructing the Lindbladian" => "generated/tutorial_lindbladian.md",
            "Custom filters and rates" => "generated/tutorial_custom_filters.md",
            "Interpreting diagnostics" => "generated/tutorial_diagnostics.md",
        ],
        "Interface and capabilities" => "api_contract.md",
        "Filter theory" => "theory_filters.md",
        "Symbolic filter feasibility" => "symbolic_filters.md",
        "API reference" => "api.md",
    ],
)

if deploy_setting == "true"
    deploydocs(repo="github.com/benzabonanza/QuantumFurnace.jl.git", devbranch="main")
end
