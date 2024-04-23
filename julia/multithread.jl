using LinearAlgebra
using Distributed
using SharedArrays
using Printf
using Random

@printf("Number of threads: %d\n", Threads.nthreads())
seed = 666
Random.seed!(seed)

# random complex matrix
N = 1000
mat = SharedArray{ComplexF64}(rand(ComplexF64, N, N))
display(mat[1, 1])

for j in 1:N
    @distributed for i in 1:N
        mat[i, j] = mat[i, j] * 10
    end
end

display(fetch(mat[1, 1]))

# this syntax accesses multiple threads even if julia was started with a single thread
@time begin
    nheads = @distributed (+) for i = 1:2000000000
        Int(rand(Bool))
    end
end

@time begin
    nheads = 0
    for i in 1:2000000000
        nheads += Int(rand(Bool))
    end
end
