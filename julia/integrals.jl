using QuadGK

f(x) = exp(x)
result, error = quadgk(f, 0, 1)
exp(1) - exp(0) ≈ result