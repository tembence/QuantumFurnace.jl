using Printf # For formatted printing

# Define the function to integrate
f(x) = x^2

# Define the integration limits
a = 0.0
b = 2.0

# Analytical exact value
exact_value = b^3 / 3.0

# --- Function to implement the Left-Point Rule ---
function left_point_rule(func, a, b, N)
    h = (b - a) / N  # Step size
    integral_sum = 0.0
    for i = 0:(N-1) # Loop over N intervals, using points 0 to N-1
        xi = a + i * h # Left endpoint of the i-th interval
        integral_sum += func(xi)
    end
    return h * integral_sum
end

# --- Function to implement the Midpoint Rule ---
function midpoint_rule(func, a, b, N)
    h = (b - a) / N # Step size
    integral_sum = 0.0
    for i = 0:(N-1) # Loop over N intervals
        mi = a + (i + 0.5) * h # Midpoint of the i-th interval
        integral_sum += func(mi)
    end
    return h * integral_sum
end

# --- Perform calculations and display results ---

println("Integrating f(x) = x^2 from $a to $b")
@printf("Exact Value = %.10f\n", exact_value)
println("-"^40)

for N in [4, 10, 100, 1000] # Try different numbers of intervals
    println("Using N = $N intervals:")

    # Calculate Left-Point approximation and error
    left_approx = left_point_rule(f, a, b, N)
    left_error = abs(left_approx - exact_value)
    @printf("  Left-Point Approx = %.10f, Error = %.10f\n", left_approx, left_error)

    # Calculate Midpoint approximation and error
    midpoint_approx = midpoint_rule(f, a, b, N)
    midpoint_error = abs(midpoint_approx - exact_value)
    @printf("  Midpoint Approx   = %.10f, Error = %.10f\n", midpoint_approx, midpoint_error)

    # Compare errors
    if left_error > 1e-15 # Avoid division by zero if error is tiny
        ratio = left_error / midpoint_error
        @printf("  Error Ratio (Left/Midpoint) = %.2f\n", ratio)
    end
    println("-"^40)
end