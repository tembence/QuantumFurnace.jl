using LinearAlgebra
using Random
using ITensors
include("jl_classical_tools.jl")

a = rescale_shift_factors()

i = Index(2,"i")
j = Index(2,"j")
m = Index(2,"m")
n = Index(2,"n")

o = Index(2,"o")
p = Index(2,"p")

A = [1.0 2.0; 3.0 4.0]
B = [5.0 6.0; 7.0 8.0]
D = [9.0 10.0; 11.0 12.0]
A = ITensor(A, i, m)
display(array(A))
B = ITensor(B, j, n)
display(array(B))
D = ITensor(D, o, p)
C = contract(A, B, C) # outer product of A and B, no contraction
println("ITensors outer product")
display(array(C))
println("ITensors reshape")
# display(reshape(array(C), 16, 1))

Eh = combiner(n, m)
Meh = combiner(j, i)
MehEhC = Meh * (Eh * C)

display(array(MehEhC))

#change back to normal matrix from itensor object
A = array(A)
B = array(B)
ckron = kron(A, B) # kronecker product of A and B
println("Normal matrix kron")
display(ckron)

