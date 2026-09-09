"""
One-time script to generate frozen BSON reference data for regression tests.

Run via: julia --project test/reference/generate_references.jl

Generates 2 reference files:
  - energy_dm_reference.bson             (EnergyDomain, DM evolution)
  - trotter_coherent_dm_reference.bson   (TrotterDomain+coherent, DM evolution)

Each BSON file stores a plain Matrix{ComplexF64} density matrix plus metadata.

The archived stochastic-trajectory regressions are intentionally not generated
here; this script maintains only the retained dense-DM references.
"""

using QuantumFurnace, LinearAlgebra, BSON

# Load shared 3-qubit test fixtures
include(joinpath(@__DIR__, "..", "test_helpers.jl"))

# ---------------------------------------------------------------------------
# Constants for reference generation
# ---------------------------------------------------------------------------
const REF_DIR = @__DIR__
const REF_DELTA = 0.1

# ---------------------------------------------------------------------------
# DM reference generator
# ---------------------------------------------------------------------------
"""
    generate_dm_reference(domain; with_coherent, filename)

Compute a single-step density matrix evolution via exp(delta*L) and save as BSON.
"""
function generate_dm_reference(domain; with_coherent::Bool, filename::String)
    liouv_config = make_config(Lindbladian(), domain; num_qubits=3, construction=with_coherent ? KMS() : GNS())
    trotter_kw = domain isa TrotterDomain ? (; trotter=N3_TROTTER) : (;)
    jumps = domain isa TrotterDomain ? N3_TROTTER_JUMPS : N3_JUMPS
    L = construct_lindbladian(jumps, liouv_config, N3_HAM; trotter_kw...)

    psi0 = fill(ComplexF64(1.0), N3_DIM) / sqrt(N3_DIM)
    rho0 = psi0 * psi0'
    rho_dm = reshape(exp(REF_DELTA * L) * vec(rho0), N3_DIM, N3_DIM)
    rho_dm = (rho_dm + rho_dm') / 2  # Hermitianize

    BSON.bson(joinpath(REF_DIR, filename), Dict(
        :rho => Matrix(rho_dm),
        :delta => REF_DELTA,
        :domain => string(typeof(domain)),
        :with_coherent => with_coherent,
        :num_qubits => 3,
    ))
    println("Saved: $filename")
end

# ---------------------------------------------------------------------------
# Generate 2 DM reference files
# ---------------------------------------------------------------------------
println("Generating frozen BSON reference data...")
println()

generate_dm_reference(EnergyDomain(); with_coherent=false, filename="energy_dm_reference.bson")
generate_dm_reference(TrotterDomain(); with_coherent=true, filename="trotter_coherent_dm_reference.bson")

println()
println("2 reference files generated successfully.")
