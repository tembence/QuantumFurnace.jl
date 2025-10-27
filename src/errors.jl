function compute_errors(hamiltonian::HamHam, config::LiouvConfig; trotter::Union{TrottTrott, Nothing} = nothing)

    energy_labels = create_energy_labels(config.num_energy_bits, config.w0)
    truncated_energy_labels = truncate_energy_labels(energy_labels, config.beta,
    config.a, config.b, config.with_linear_combination)

    energy_error = compute_energy_quadrature_error(config, hamiltonian, truncated_energy_labels; trotter = trotter)
    @printf("Worst quadrature error for the energy integral: %s\n", energy_error)
    # time_labels = energy_labels .* (config.t0 / config.w0)
    # if trotter === nothing
    #     oft_error = compute_time_oft_quadrature_error()
    #     B_error = compute_time_B_quadrature_error()
    # else
    #     oft_error = compute_trotter_oft_quadrature_error()
    #     B_error =  compute_trotter_B_quadrature_error()
    # end
    # return (energy_error = energy_error, oft_error = oft_error, B_error = B_error)
end

function compute_quadrature_error(integrand::Function, labels::Vector{Float64}, args...)
    integral = quadgk(t->integrand(t, args...), minimum(labels), maximum(labels); atol=1e-10, rtol=1e-10)[1]
    sum = riemann_sum(t->integrand(t, args...), labels)
    return norm(integral - sum)
end

function compute_energy_quadrature_error(config::LiouvConfig, hamiltonian::HamHam, energy_labels::Vector{Float64};
    trotter::Union{TrottTrott, Nothing} = nothing)
    transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)
    gaussian_filter(w) =  exp(-config.beta^2 * (w + 1 / (2 * config.beta))^2 / 4) * beta / sqrt(2 * pi) # Worst point at -1/2β
    jump = create_jumpop(["X"], config.num_qubits, Int(round(config.num_qubits / 2)), hamiltonian; trotter = trotter)
    jump_oft(w) = oft(jump, w, hamiltonian, config.beta)
    dm = Matrix{ComplexF64}(I(2^num_qubits) / 2^num_qubits)
    # integrand(w) = transition(w) * gaussian_filter(w)^2
    integrand(w) = transition(w) * tr(jump_oft(w) * dm * jump_oft(w)')
    # energies = [-2:0.01:2.0;]
    # display(plot(energies, integrand.(energies)))

    return compute_quadrature_error(integrand, energy_labels)
end

function compute_time_oft_quadrature_error()
end

function compute_trotter_oft_quadrature_error()
end

function compute_time_B_quadrature_error()
end

function compute_trotter_B_quadrature_error()
end

#* Quadrature errors
# energy_labels, time_labels = precompute_labels(config.domain, config)
# f_minus = compute_truncated_f(compute_f_minus, time_labels, config.beta)
# # f_plus_metro = compute_truncated_f(compute_f_plus_metro, time_labels, config.beta, config.eta)
# f_plus_eh = compute_truncated_f(compute_f_plus_eh, time_labels, config.beta, config.a, config.b)
# # truncated_time_labels_metro = sort(collect(keys(f_plus_metro)))
# truncated_time_labels_eh = sort(collect(keys(f_plus_eh)))


# B_time_eh = coherent_term_time(jumps[1], hamiltonian, f_minus, f_plus_eh, t0)
# B_time_eh_integrated = coherent_term_time_integrated_eh(
#         jumps[1], hamiltonian, beta, a, b, (minimum(time_labels), maximum(time_labels))
# )
# B_bohr = coherent_bohr(hamiltonian, jumps[1], config)
# norm(B_time_eh_integrated - B_bohr)
# err_B_eh = norm(B_time_eh - B_time_eh_integrated)
# @printf("Quadrature error for B: %s\n", err_B_eh)
# err_plus_eh = compute_quadrature_error(compute_f_plus_eh, truncated_time_labels_eh, config.beta, config.a, config.b)
# @printf("Quadrature error for f plus: %s\n", err_plus_eh)


# B_time_metro_integrated = coherent_term_time_integrated_metro(
#         jumps[1], hamiltonian, eta, beta
# )
# B_time_metro = coherent_term_time(jumps[1], hamiltonian, f_minus, f_plus_metro, t0)
# norm(B_time_metro - B_bohr)

# #!
# norm(B_time_metro - B_time_metro_integrated)
# norm(B_time_metro_integrated - B_bohr)

# norm(B_bohr - B_time_eh)
# norm(B_bohr - B_time_eh_integrated)

# eta_error = min(eta * beta * 0.5 / (sqrt(2) * pi), (eta * beta * 0.5)^3)

# err_minus = compute_quadrature_error(compute_f_minus, collect(keys(f_minus)), config.beta)

# err_plus = compute_quadrature_error(compute_f_plus_metro, truncated_time_labels_metro, config.beta, config.eta)
# err_plus2 = compute_quadrature_error(compute_f_plus_metro, time_labels, config.beta, config.eta)
