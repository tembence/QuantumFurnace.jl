

function jump_contribution_bohr()
     # Coherent part
        if with_coherent
            coherent_term = coherent_bohr(hamiltonian, bohr_dict, jump, beta, a, b)
            liouv .+= vectorize_liouvillian_coherent(coherent_term)
        end

        # Dissipative part
        for nu_2 in unique_freqs
            alpha_nu1_matrix = create_alpha.(hamiltonian.bohr_freqs, nu_2, beta, a, b)

            A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
            A_nu_2[bohr_dict[nu_2]] .= jump.in_eigenbasis[bohr_dict[nu_2]]

            liouv .+= vectorize_liouvillian_diss(alpha_nu1_matrix .* jump.in_eigenbasis, A_nu_2')
        end
end

function jump_contribution_energy(jump::JumpOp, hamiltonian::HamHam)

    if with_coherent
            coherent_term = coherent_bohr(hamiltonian, bohr_dict, jump, beta, a, b)
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = oft(jump, w, hamiltonian, beta)
            total_liouv_diss_part .+= transition(w) * vectorize_liouvillian_diss(jump_oft)
        end
end

function jump_contribution_time()
    if with_coherent 
            coherent_term = coherent_term_time(jump, hamiltonian, f_minus, f_plus, t0)  
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = time_oft(jump, w, hamiltonian, oft_time_labels, beta) # subnorm = t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
            total_liouv_diss_part .+= transition(w) * vectorize_liouvillian_diss(jump_oft)
        end
end

function jump_contribution_trotter()

    dim = size(trotter.eigvecs, 1)
    w0 = energy_labels[2] - energy_labels[1]
    oft_time_labels = truncate_time_labels_for_oft(time_labels, beta)

    transition_gauss(w) = exp(-beta^2 * (w + 1/beta)^2 /2)

    if with_coherent  # Steup for coherent term in time domain
        f_minus = compute_truncated_f_minus(time_labels, beta)
        f_plus = compute_truncated_f_plus(time_labels, beta)
    end

    total_liouv_coherent_part = zeros(ComplexF64, dim^2, dim^2)
    total_liouv_diss_part = zeros(ComplexF64, dim^2, dim^2)

     if with_coherent
            coherent_term = coherent_term_trotter(jump, trotter, f_minus, f_plus)
            coherent_term = trotter.trafo_from_eigen_to_trotter' * coherent_term * trotter.trafo_from_eigen_to_trotter
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = trotter_oft(jump, w, trotter, oft_time_labels, beta) # t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
            jump_oft = trotter.trafo_from_eigen_to_trotter' * jump_oft * trotter.trafo_from_eigen_to_trotter  #! Get rid of this

            total_liouv_diss_part .+= transition_gauss(w) * vectorize_liouvillian_diss(jump_oft)
        end
end