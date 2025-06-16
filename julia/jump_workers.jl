

function jump_contribution_bohr()
end

function jump_contribution_energy(jump::JumpOp, hamiltonian::HamHam)

    if with_coherent  # There is no energy formulation of the coherent term, only Bohr and time.
            coherent_term = coherent_bohr(hamiltonian, bohr_dict, jump, beta, a, b)
            total_liouv_coherent_part .+= vectorize_liouvillian_coherent(coherent_term)
        end

        for w in energy_labels
            jump_oft = oft(jump, w, hamiltonian, beta)
            total_liouv_diss_part .+= transition(w) * vectorize_liouvillian_diss(jump_oft)
        end

end

function jump_contribution_time()
end

function jump_contribution_trotter()
end