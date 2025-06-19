include("ofts.jl")
include("qi_tools.jl")
include("coherent.jl")
include("timelike_tools.jl")

#* Liouvillian jump contributions
function jump_contribution_bohr(jump::JumpOp, hamiltonian::HamHam, config::LiouvConfig)
    dim = size(hamiltonian.data, 1)
    unique_freqs = keys(hamiltonian.bohr_dict)

    alpha = pick_alpha(config)

    liouv_for_jump = zeros(ComplexF64, dim^2, dim^2)  
    # Coherent part
    if config.with_coherent
        coherent_term = coherent_bohr(hamiltonian, jump, config)
        liouv_for_jump .+= vectorize_liouvillian_coherent(coherent_term)
    end

    # Dissipative part
    for nu_2 in unique_freqs
        alpha_nu1_matrix = alpha.(hamiltonian.bohr_freqs, nu_2, config.beta, config.a, config.b)

        A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
        A_nu_2[hamiltonian.bohr_dict[nu_2]] .= jump.in_eigenbasis[hamiltonian.bohr_dict[nu_2]]

        liouv_for_jump .+= vectorize_liouvillian_diss(alpha_nu1_matrix .* jump.in_eigenbasis, A_nu_2')
    end
    return liouv_for_jump
end

function jump_contribution_energy(jump::JumpOp, hamiltonian::HamHam, config::LiouvConfig)
    dim = size(hamiltonian.data, 1)
    w0 = energy_labels[2] - energy_labels[1]

    transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)

    liouv_coherent_part_for_jump = zeros(ComplexF64, dim^2, dim^2)
    liouv_diss_part_for_jump = zeros(ComplexF64, dim^2, dim^2)
    if with_coherent
        coherent_term = coherent_bohr(hamiltonian, jump, config)
        liouv_coherent_part_for_jump .+= vectorize_liouvillian_coherent(coherent_term)
    end

    for w in energy_labels
        jump_oft = oft(jump, w, hamiltonian, config.beta)
        liouv_diss_part_for_jump .+= transition(w) * vectorize_liouvillian_diss(jump_oft)
    end
    oft_norm_squared = config.beta / sqrt(2 * pi)
    return liouv_coherent_part_for_jump .+ w0 * oft_norm_squared * liouv_diss_part_for_jump
end

function jump_contribution_time(jump::JumpOp, hamiltonian::HamHam, config::LiouvConfig)
    dim = size(hamiltonian.data, 1)
    w0 = energy_labels[2] - energy_labels[1]
    t0 = time_labels[2] - time_labels[1]
    oft_time_labels = truncate_time_labels_for_oft(time_labels, config.beta)

    transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)

    if with_coherent
        f_minus = compute_truncated_f(compute_f_minus, time_labels, config.beta)
        if config.with_linear_combination  
            if config.a != 0.0  # Improved Metro / Glauber
                f_plus = compute_truncated_f(compute_f_plus_eh, time_labels, config.beta, config.a, config.b)
            else  # Metro
                f_plus = compute_truncated_f(compute_f_plus_metro, time_labels, config.beta, config.eta)
            end
        else  # Gaussian
            f_plus = compute_truncated_f(compute_f_plus, time_labels, config.beta)
        end
    end

    liouv_coherent_part_for_jump = zeros(ComplexF64, dim^2, dim^2)
    total_liouv_diss_part = zeros(ComplexF64, dim^2, dim^2)
    if config.with_coherent 
        coherent_term = coherent_term_time(jump, hamiltonian, f_minus, f_plus, t0)  
        liouv_coherent_part_for_jump .+= vectorize_liouvillian_coherent(coherent_term)
    end

    for w in energy_labels
        jump_oft = time_oft(jump, w, hamiltonian, oft_time_labels, config.beta) # subnorm = t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
        liouv_diss_part_for_jump .+= transition(w) * vectorize_liouvillian_diss(jump_oft)
        next!(p)
    end
    prefactor = w0 * t0^2 * (sqrt(2 / pi) / config.beta) / (2 * pi)  # time ints t0^2, energy int w0, OFT time norm^2, Fourier
    return liouv_coherent_part_for_jump .+ prefactor * liouv_diss_part_for_jump
end

function jump_contribution_trotter(jump::JumpOp, trotter::TrottTrott, config::LiouvConfig)

    dim = size(trotter.eigvecs, 1)
    w0 = energy_labels[2] - energy_labels[1]
    oft_time_labels = truncate_time_labels_for_oft(time_labels, config.beta)

    transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)

    if config.with_coherent
        f_minus = compute_truncated_f(compute_f_minus, time_labels, config.beta)
        if config.with_linear_combination  
            if config.a != 0.0  # Improved Metro / Glauber
                f_plus = compute_truncated_f(compute_f_plus_eh, time_labels, config.beta, config.a, config.b)
            else  # Metro
                f_plus = compute_truncated_f(compute_f_plus_metro, time_labels, config.beta, config.eta)
            end
        else  # Gaussian
            f_plus = compute_truncated_f(compute_f_plus, time_labels, config.beta)
        end
    end

    liouv_coherent_part_for_jump = zeros(ComplexF64, dim^2, dim^2)
    liouv_diss_part_for_jump = zeros(ComplexF64, dim^2, dim^2)
    if config.with_coherent  # There is no energy formulation of the coherent term, only Bohr and time.
        coherent_term = coherent_term_trotter(jump, trotter, f_minus, f_plus)
        coherent_term = trotter.trafo_from_eigen_to_trotter' * coherent_term * trotter.trafo_from_eigen_to_trotter
        liouv_coherent_part_for_jump .+= vectorize_liouvillian_coherent(coherent_term)
    end

    for w in energy_labels
        jump_oft = trotter_oft(jump, w, trotter, oft_time_labels, config.beta) # t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
        jump_oft = trotter.trafo_from_eigen_to_trotter' * jump_oft * trotter.trafo_from_eigen_to_trotter

        liouv_diss_part_for_jump .+= transition(w) * vectorize_liouvillian_diss(jump_oft)
    end
    
    prefactor = w0 * trotter.t0^2 * (sqrt(2 / pi) / config.beta) / (2 * pi)
    return liouv_coherent_part_for_jump .+ prefactor * liouv_diss_part_for_jump  # L in energy basis
end

#* Algorithmic jump contributions
