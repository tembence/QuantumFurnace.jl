using Test
using LinearAlgebra
using Random

using QuantumFurnace

# Loading core alone must not activate the optional extension. ITensors alone
# is also insufficient because the extension contract requires both packages.
@test Base.get_extension(QuantumFurnace, :QuantumFurnaceITensorsExt) === nothing
using ITensors
@test Base.get_extension(QuantumFurnace, :QuantumFurnaceITensorsExt) === nothing
using ITensorMPS

const QFITensors = Base.get_extension(
    QuantumFurnace, :QuantumFurnaceITensorsExt)
@test QFITensors !== nothing

include("test_conventions.jl")
