using Documenter
using Literate
using QuantumFurnace


# --- 1. Generate tutorials and theory pages from Literate.jl scripts ---
const literate_dir = joinpath(@__DIR__, "src/literate")
const generated_dir = joinpath(@__DIR__, "src/generated")

# Clean up old generated files
if isdir(generated_dir)
    rm(generated_dir, recursive=true)
end
mkdir(generated_dir)

# Process each .jl file in the literate directory
for filename in readdir(literate_dir)
    if endswith(filename, ".jl")
        input_file = joinpath(literate_dir, filename)
        output_file_stem = first(splitext(filename))
        
        # Generate Markdown file
        Literate.markdown(input_file, generated_dir, name=output_file_stem, documenter=true)
        
        # Generate Jupyter Notebook
        Literate.notebook(input_file, generated_dir, name=output_file_stem)
    end
end


# --- 2. Configure Documenter.jl to build the site ---
makedocs(
    sitename = "QuantumFurnace.jl",
    format = Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tembence.github.io/QuantumFurnace.jl/stable/",
        mathengine = MathJax2(
            Dict(
                :TeX => Dict(
                    :packages => ["base", "ams", "autoload", "dsfont"],
                ),
            )
        ),
        assets=String[],
    ),
    modules = [QuantumFurnace],
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Finding a Thermal State" => "generated/tutorial_thermalize.md",
            "Constructing the Detailed Balanced Lindbladian" => "generated/tutorial_lindbladian.md",
            "Create a Hamiltonian" => "generated/tutorial_hamiltonian.md"
        ],
        "Theory" => [
            "Open Quantum System Dynamics" => "generated/theory_oqs_dynamics.md",
            "Detailed Balanced Lindbladian" => "generated/theory_detailed_balance.md",
            "Weak-measurement based Lindbladian Evolution" => "generated/theory_weak_measurement.md",
            "Convex Combination of Lindbladians" => "generated/theory_convex_combination.md"
        ],
        "API Reference" => "api.md",
    ]
)

# --- 3. Deploy the documentation to GitHub Pages ---
deploydocs(
    repo = "github.com/tembence/QuantumFurnace.jl.git",
    devbranch = "main",
)