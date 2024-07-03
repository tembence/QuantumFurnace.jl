using QuadGK
using Plots

#* Testing
# f_t_1(t) = 1 / cosh(2 * pi * t / beta)
# f_t_2(t) = sin(- beta * sigma_E^2 * (t)) * exp(- 2 * sigma_E^2 * (t)^2)

# t_values = range(-5, stop=5, length=1000)
# # Compute the convolution for each time value
# @time conv_results = (sigma_E / (pi * beta)) * exp(beta^2 * sigma_E^2 / 8) * convolute.(Ref(f_t_1), Ref(f_t_2), t_values)
# conv_results /= sqrt(sum(conv_results.^2))
# # Plot the results
# plot!(t_values, real.(conv_results), label="Real part")