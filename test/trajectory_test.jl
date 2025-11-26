using Revise

if !isdefined(Main, :QuantumFurnace)
    includet("../src/QuantumFurnace.jl")
end

using .QuantumFurnace
using LinearAlgebra, Random

num_qubits = 1


hamiltonian = create_hamham()