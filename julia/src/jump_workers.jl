include("ofts.jl")
include("qi_tools.jl")
include("coherent.jl")
include("timelike_tools.jl")
include("structs.jl")

#* Liouvillian jump contributions
function jump_contribution(::BohrPicture, jump::JumpOp, hamiltonian::HamHam, config::LiouvConfig)

    dim = size(hamiltonian.data, 1)
    unique_freqs = keys(hamiltonian.bohr_dict)
    alpha = pick_alpha(config)

    liouv_for_jump = zeros(ComplexF64, dim^2, dim^2)
    if config.with_coherent 
        coherent_term = coherent_bohr(hamiltonian, jump, config) 
        vectorize_liouvillian_coherent!(liouv_for_jump, coherent_term)
    end

    alpha_A_nu1 = zeros(ComplexF64, dim, dim)
    for nu_2 in unique_freqs
        @. alpha_A_nu1 = alpha(hamiltonian.bohr_freqs, nu_2, config.beta, config.a, config.b) * jump.in_eigenbasis

        A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
        indices = hamiltonian.bohr_dict[nu_2]
        A_nu_2[indices] .= jump.in_eigenbasis[indices]

        vectorize_liouv_diss_and_add!(liouv_for_jump, alpha_A_nu1, A_nu_2', 1.0)
    end

    return liouv_for_jump
end

function jump_contribution(::EnergyPicture, jump::JumpOp, hamiltonian::HamHam, config::LiouvConfig, 
    energy_labels::Vector{Float64})

    dim = size(hamiltonian.data, 1)
    w0 = abs(energy_labels[2] - energy_labels[1])

    transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)

    liouv_for_jump = zeros(ComplexF64, dim^2, dim^2)
    if config.with_coherent 
        coherent_term = coherent_bohr(hamiltonian, jump, config) 
        vectorize_liouvillian_coherent!(liouv_for_jump, coherent_term)
    end

    jump_oft = zeros(ComplexF64, dim, dim)
    prefactor = w0 * config.beta / sqrt(2 * pi)  # time ints t0^2, energy int w0, OFT time norm^2, Fourier
    for w in energy_labels
        oft_fast!(jump_oft, jump, w, hamiltonian, config.beta) # subnorm = t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
        loop_scalar = prefactor * transition(w)

        vectorize_liouv_diss_and_add!(liouv_for_jump, jump_oft, loop_scalar)
    end
    return liouv_for_jump
end

function jump_contribution(::TimePicture, jump::JumpOp, hamiltonian::HamHam, config::LiouvConfig, 
    energy_labels::Vector{Float64}, time_labels::Vector{Float64})

    dim = size(hamiltonian.data, 1)
    w0 = abs(energy_labels[2] - energy_labels[1])
    t0 = time_labels[2] - time_labels[1]
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

    liouv_for_jump = zeros(ComplexF64, dim^2, dim^2)
    if config.with_coherent 
        coherent_term = coherent_term_time(jump, hamiltonian, f_minus, f_plus, t0)  
        vectorize_liouvillian_coherent!(liouv_for_jump, coherent_term)
    end

    jump_oft = zeros(ComplexF64, dim, dim)
    time_oft_caches = OFTCaches(dim)
    prefactor = w0 * t0^2 * (sqrt(2 / pi) / config.beta) / (2 * pi)  # time ints t0^2, energy int w0, OFT time norm^2, Fourier
    for w in energy_labels
        time_oft_fast!(jump_oft, time_oft_caches, jump, w, hamiltonian, oft_time_labels, config.beta) # subnorm = t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
        loop_scalar = prefactor * transition(w)

        vectorize_liouv_diss_and_add!(liouv_for_jump, jump_oft, loop_scalar)
    end
    return liouv_for_jump
end

function jump_contribution(::TrotterPicture, jump::JumpOp, trotter::TrottTrott, config::LiouvConfig, 
    energy_labels::Vector{Float64}, time_labels::Vector{Float64})

    dim = size(trotter.eigvecs, 1)
    w0 = abs(energy_labels[2] - energy_labels[1])
    t0 = time_labels[2] - time_labels[1]
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

    liouv_for_jump = zeros(ComplexF64, dim^2, dim^2)
    if config.with_coherent 
        coherent_term = coherent_term_trotter(jump, trotter, f_minus, f_plus)
        vectorize_liouvillian_coherent!(liouv_for_jump, coherent_term)
    end

    jump_oft = zeros(ComplexF64, dim, dim)
    time_oft_caches = OFTCaches(dim)
    prefactor = w0 * t0^2 * (sqrt(2 / pi) / config.beta) / (2 * pi)  # time ints t0^2, energy int w0, OFT time norm^2, Fourier
    for w in energy_labels
        trotter_oft_fast!(jump_oft, time_oft_caches, jump, w, trotter, oft_time_labels, config.beta) # subnorm = t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
        loop_scalar = prefactor * transition(w)

        vectorize_liouv_diss_and_add!(liouv_for_jump, jump_oft, loop_scalar)
    end
    return liouv_for_jump
end

#* Algorithmic jump contributions -----
function jump_contribution(::BohrPicture, evolving_dm::Matrix{ComplexF64}, jump::JumpOp, hamiltonian::HamHam, 
    config::ThermalizeConfig)

    dim = size(evolving_dm, 1)
    alpha = pick_alpha(config)
    unique_freqs = keys(hamiltonian.bohr_dict)

    jump_dm_contribution = zeros(ComplexF64, dim, dim)
    # Coherent part
    if config.with_coherent
        coherent_term = coherent_bohr(hamiltonian, jump, config)
        mul!(jump_dm_contribution, coherent_term, evolving_dm, -1im * config.delta, 1.0)
        mul!(jump_dm_contribution, evolving_dm, coherent_term, 1im * config.delta, 1.0)
    end

    # Dissipative part
    alpha_A_nu1 = zeros(ComplexF64, dim, dim)
    temp1 = similar(alpha_A_nu1)
    A_nu_2_dag_alpha_A_nu1 = similar(alpha_A_nu1)
    # mul!(C, A, B, α, β) computes C = α*A*B + β*C
    for nu_2 in unique_freqs
        @. alpha_A_nu1 = alpha(hamiltonian.bohr_freqs, nu_2, config.beta, config.a, config.b) * jump.in_eigenbasis

        A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
        indices = hamiltonian.bohr_dict[nu_2]
        A_nu_2[indices] .= jump.in_eigenbasis[indices]

        mul!(A_nu_2_dag_alpha_A_nu1, A_nu_2', alpha_A_nu1)

        # Term 1
        mul!(temp1, evolving_dm, A_nu_2')
        mul!(jump_dm_contribution, alpha_A_nu1, temp1, config.delta, 1.0)

        # Term 2
        mul!(jump_dm_contribution, A_nu_2_dag_alpha_A_nu1, evolving_dm, -0.5 * config.delta, 1.0)

        # Term 3
        mul!(jump_dm_contribution, evolving_dm, A_nu_2_dag_alpha_A_nu1, -0.5 * config.delta, 1.0)
    end
        
    return jump_dm_contribution
end

function jump_contribution(::EnergyPicture, evolving_dm::Matrix{ComplexF64}, jump::JumpOp, hamiltonian::HamHam, 
    config::ThermalizeConfig)

    dim = size(evolving_dm, 1)
    w0 = abs(energy_labels[2] - energy_labels[1])

    jump_dm_contribution = zeros(ComplexF64, dim, dim)
    # Coherent part
    if config.with_coherent
        coherent_term = coherent_bohr(hamiltonian, jump, config)
        mul!(jump_dm_contribution, coherent_term, evolving_dm, -1im * config.delta, 1.0)
        mul!(jump_dm_contribution, evolving_dm, coherent_term, 1im * config.delta, 1.0)
    end

    # Dissipative part
    prefactor = config.delta * w0 * config.beta / sqrt(2 * pi)
    jump_oft = zeros(ComplexF64, dim, dim)
    jump_dag_jump = similar(jump_oft)
    temp1 = similar(jump_oft)
    # mul!(C, A, B, α, β) computes C = α*A*B + β*C
    for w in energy_labels
        oft_fast!(jump_oft, jump, w, hamiltonian, config.beta)
        mul!(jump_dag_jump, jump_oft', jump_oft)

        loop_factor = transition(w) * prefactor
        # Term 1
        mul!(temp1, evolving_dm, jump_oft')  # rho * A'
        mul!(jump_dm_contribution, jump_oft, temp1, loop_factor, 1.0)  # L += prefactor * A * rho * A'

        # Term 2
        mul!(jump_dm_contribution, jump_dag_jump, evolving_dm, -0.5 * loop_factor, 1.0)
        
        # Term 3
        mul!(jump_dm_contribution, evolving_dm, jump_dag_jump, -0.5 * loop_factor, 1.0)
    end
    
    return jump_dm_contribution
end

function jump_contribution(::TimePicture, evolving_dm::Matrix{ComplexF64}, jump::JumpOp, hamiltonian::HamHam, 
    config::ThermalizeConfig)

    dim = size(evolving_dm, 1)
    w0 = abs(energy_labels[2] - energy_labels[1])
    t0 = time_labels[2] - time_labels[1]
    oft_time_labels = truncate_time_labels_for_oft(time_labels, config.beta)

    jump_dm_contribution = zeros(ComplexF64, dim, dim)

    # Coherent part
    if config.with_coherent
        coherent_term = coherent_term_time(jump, hamiltonian, f_minus, f_plus, t0)
        mul!(jump_dm_contribution, coherent_term, evolving_dm, -1im * config.delta, 1.0)
        mul!(jump_dm_contribution, evolving_dm, coherent_term, 1im * config.delta, 1.0)
    end

    jump_oft = zeros(ComplexF64, dim, dim)
    jump_dag_jump = similar(jump_oft)
    temp1 = similar(jump_oft)

    # Pre-allocate caches for the time_oft function as well
    oft_caches = OFTCaches(dim)
    prefactor = config.delta * w0 * t0^2 * (sqrt(2 / pi) / config.beta) / (2 * pi)
    for w in energy_labels
        time_oft_fast!(jump_oft, oft_caches, jump, w, hamiltonian, oft_time_labels, config.beta)
        
        # jump_dag_jump = jump_oft' * jump_oft
        mul!(jump_dag_jump, jump_oft', jump_oft)

        loop_factor = transition(w) * prefactor
        # Term 1
        mul!(temp1, evolving_dm, jump_oft')  # rho * A'
        mul!(jump_dm_contribution, jump_oft, temp1, loop_factor, 1.0)  # L += prefactor * A * rho * A'

        # Term 2
        mul!(jump_dm_contribution, jump_dag_jump, evolving_dm, -0.5 * loop_factor, 1.0)

        # Term 3
        mul!(jump_dm_contribution, evolving_dm, jump_dag_jump, -0.5 * loop_factor, 1.0)
    end
    return jump_dm_contribution
end

function jump_contribution(::TrotterPicture, evolving_dm::Matrix{ComplexF64}, jump::JumpOp, trotter::TrottTrott, 
    config::ThermalizeConfig)

    dim = size(evolving_dm, 1)
    w0 = abs(energy_labels[2] - energy_labels[1])
    t0 = time_labels[2] - time_labels[1]
    oft_time_labels = truncate_time_labels_for_oft(time_labels, config.beta)

    jump_dm_contribution = zeros(ComplexF64, dim, dim)

    # Coherent part
    if config.with_coherent
        coherent_term = coherent_term_trotter(jump, trotter, f_minus, f_plus)
        mul!(jump_dm_contribution, coherent_term, evolving_dm, -1im * config.delta, 1.0)
        mul!(jump_dm_contribution, evolving_dm, coherent_term, 1im * config.delta, 1.0)
    end

    # Pre-allocate caches
    jump_oft = zeros(ComplexF64, dim, dim)
    jump_dag_jump = similar(jump_oft)
    temp1 = similar(jump_oft)

    oft_caches = OFTCaches(dim)
    prefactor = config.delta * w0 * t0^2 * (sqrt(2 / pi) / config.beta) / (2 * pi)
    for w in energy_labels
        trotter_oft_fast!(jump_oft, oft_caches, jump, w, trotter, oft_time_labels, config.beta)
        
        # jump_dag_jump = jump_oft' * jump_oft
        mul!(jump_dag_jump, jump_oft', jump_oft)

        loop_factor = transition(w) * prefactor
        # Term 1
        mul!(temp1, evolving_dm, jump_oft')  # rho * A'
        mul!(jump_dm_contribution, jump_oft, temp1, loop_factor, 1.0)  # L += prefactor * A * rho * A'

        # Term 2
        mul!(jump_dm_contribution, jump_dag_jump, evolving_dm, -0.5 * loop_factor, 1.0)

        # Term 3
        mul!(jump_dm_contribution, evolving_dm, jump_dag_jump, -0.5 * loop_factor, 1.0)
    end

    return jump_dm_contribution
end

#* Linear Map jump contributions -----
function jump_contribution!(
    target_d_rho::AbstractMatrix{ComplexF64}, 
    ::BohrPicture, 
    rho::AbstractMatrix{ComplexF64}, 
    jump::JumpOp, 
    hamiltonian::HamHam,
    config::LiouvConfig,
    precomputed_data,
    caches
    )

    (; w0, t0, transition, f_minus, f_plus, energy_labels, oft_time_labels) = precomputed_data
    (; jump_caches, oft_caches) = caches

    alpha = pick_alpha(config)
    unique_freqs = keys(hamiltonian.bohr_dict)

    # Coherent part
    if config.with_coherent
        coherent_term = coherent_bohr(hamiltonian, jump, config)
        mul!(target_d_rho, coherent_term, rho, -1im, 1.0)
        mul!(target_d_rho, rho, coherent_term, 1im, 1.0)
    end

    # Dissipative part
    # mul!(C, A, B, α, β) computes C = α*A*B + β*C
    for nu_2 in unique_freqs
        @. jump_caches.jump_1 = alpha(hamiltonian.bohr_freqs, nu_2, config.beta, config.a, config.b) * jump.in_eigenbasis

        A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
        indices = hamiltonian.bohr_dict[nu_2]
        A_nu_2[indices] .= jump.in_eigenbasis[indices]

        mul!(jump_caches.jump_2_dag_jump_1, A_nu_2', jump_caches.jump_1)

        # Term 1
        mul!(jump_caches.temp1, rho, A_nu_2')
        mul!(target_d_rho, jump_caches.jump_1, jump_caches.temp1, 1.0, 1.0)

        # Term 2
        mul!(target_d_rho, jump_caches.jump_2_dag_jump_1, rho, -0.5, 1.0)

        # Term 3
        mul!(target_d_rho, rho, jump_caches.jump_2_dag_jump_1, -0.5, 1.0)
    end
        
    return target_d_rho
end

function jump_contribution!(
    target_d_rho::AbstractMatrix{ComplexF64}, 
    ::EnergyPicture, 
    rho::AbstractMatrix{ComplexF64}, 
    jump::JumpOp, 
    hamiltonian::HamHam,
    config::LiouvConfig,
    precomputed_data,
    caches
    )

    (; w0, t0, transition, f_minus, f_plus, energy_labels, oft_time_labels) = precomputed_data
    (; jump_caches, oft_caches) = caches

    # Coherent part
    if config.with_coherent
        coherent_term = coherent_bohr(hamiltonian, jump, config)
        mul!(target_d_rho, coherent_term, rho, -1im, 1.0)
        mul!(target_d_rho, rho, coherent_term, 1im, 1.0)
    end

    # Dissipative part
    prefactor = w0 * config.beta / sqrt(2 * pi)
    # mul!(C, A, B, α, β) computes C = α*A*B + β*C
    for w in energy_labels
        oft_fast!(jump_caches.jump_1, jump, w, hamiltonian, config.beta)
        mul!(jump_caches.jump_2_dag_jump_1, jump_caches.jump_1', jump_caches.jump_1)

        loop_factor = transition(w) * prefactor
        # Term 1
        mul!(jump.temp1, rho, jump_caches.jump_1')  # rho * A'
        mul!(target_d_rho, jump_caches.jump_1, jump.temp1, loop_factor, 1.0)  # L += prefactor * A * rho * A'

        # Term 2
        mul!(target_d_rho, jump_caches.jump_2_dag_jump_1, rho, -0.5 * loop_factor, 1.0)
        
        # Term 3
        mul!(target_d_rho, rho, jump_caches.jump_2_dag_jump_1, -0.5 * loop_factor, 1.0)
    end
    
    return target_d_rho
end

function jump_contribution!(
    target_d_rho::AbstractMatrix{ComplexF64}, 
    ::TimePicture, 
    rho::AbstractMatrix{ComplexF64}, 
    jump::JumpOp, 
    hamiltonian::HamHam,
    config::LiouvConfig,
    precomputed_data,
    caches
    )

    (; w0, t0, transition, f_minus, f_plus, energy_labels, oft_time_labels) = precomputed_data
    (; jump_caches, oft_caches) = caches

    # Coherent part
    if config.with_coherent
        coherent_term = coherent_term_time(jump, hamiltonian, f_minus, f_plus, t0)
        mul!(target_d_rho, coherent_term, rho, -1im, 1.0)
        mul!(target_d_rho, rho, coherent_term, 1im, 1.0)
    end

    prefactor = w0 * t0^2 * (sqrt(2 / pi) / config.beta) / (2 * pi)
    for w in energy_labels
        time_oft_fast!(jump_caches.jump_1, oft_caches, jump, w, hamiltonian, oft_time_labels, config.beta)
        
        # jump_dag_jump = jump_oft' * jump_oft
        mul!(jump_caches.jump_2_dag_jump_1, jump_caches.jump_1', jump_caches.jump_1)

        loop_factor = transition(w) * prefactor
        # Term 1
        mul!(jump_caches.temp1, rho, jump_caches.jump_1')  # rho * A'
        mul!(target_d_rho, jump_caches.jump_1, jump_caches.temp1, loop_factor, 1.0)  # L += prefactor * A * rho * A'

        # Term 2
        mul!(target_d_rho, jump_caches.jump_2_dag_jump_1, rho, -0.5 * loop_factor, 1.0)

        # Term 3
        mul!(target_d_rho, rho, jump_caches.jump_2_dag_jump_1, -0.5 * loop_factor, 1.0)
    end
    return target_d_rho
end

function jump_contribution!(
    target_d_rho::AbstractMatrix{ComplexF64}, 
    ::TrotterPicture, 
    rho::AbstractMatrix{ComplexF64}, 
    jump::JumpOp, 
    trotter::TrottTrott,
    config::LiouvConfig,
    precomputed_data,
    caches
    )

    (; w0, t0, transition, f_minus, f_plus, energy_labels, oft_time_labels) = precomputed_data
    (; jump_caches, oft_caches) = caches

    # Coherent part
    if config.with_coherent
        coherent_term = coherent_term_trotter(jump, trotter, f_minus, f_plus)
        mul!(target_d_rho, coherent_term, rho, -1im, 1.0)
        mul!(target_d_rho, rho, coherent_term, 1im, 1.0)
    end

    prefactor = w0 * t0^2 * (sqrt(2 / pi) / config.beta) / (2 * pi)
    for w in energy_labels
        trotter_oft_fast!(jump_caches.jump_1, oft_caches, jump, w, trotter, oft_time_labels, config.beta)
        
        # jump_dag_jump = jump_oft' * jump_oft
        mul!(jump_caches.jump_2_dag_jump_1, jump_caches.jump_1', jump_caches.jump_1)

        loop_factor = transition(w) * prefactor
        # Term 1
        mul!(jump_caches.temp1, rho, jump_caches.jump_1')  # rho * A'
        mul!(target_d_rho, jump_caches.jump_1, jump_caches.temp1, loop_factor, 1.0)  # L += prefactor * A * rho * A'

        # Term 2
        mul!(target_d_rho, jump_caches.jump_2_dag_jump_1, rho, -0.5 * loop_factor, 1.0)

        # Term 3
        mul!(target_d_rho, rho, jump_caches.jump_2_dag_jump_1, -0.5 * loop_factor, 1.0)
    end

    return target_d_rho
end



#* Slow and old
# function jump_contribution_slow(::TrotterPicture, jump::JumpOp, trotter::TrottTrott, config::LiouvConfig, 
#     energy_labels::Vector{Float64}, time_labels::Vector{Float64})

#     dim = size(trotter.eigvecs, 1)
#     w0 = abs(energy_labels[2] - energy_labels[1])
#     oft_time_labels = truncate_time_labels_for_oft(time_labels, config.beta)

#     transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)

#     if config.with_coherent
#         f_minus = compute_truncated_f(compute_f_minus, time_labels, config.beta)
#         if config.with_linear_combination  
#             if config.a != 0.0  # Improved Metro / Glauber
#                 f_plus = compute_truncated_f(compute_f_plus_eh, time_labels, config.beta, config.a, config.b)
#             else  # Metro
#                 f_plus = compute_truncated_f(compute_f_plus_metro, time_labels, config.beta, config.eta)
#             end
#         else  # Gaussian
#             f_plus = compute_truncated_f(compute_f_plus, time_labels, config.beta)
#         end
#     end

#     liouv_coherent_part_for_jump = zeros(ComplexF64, dim^2, dim^2)
#     liouv_diss_part_for_jump = zeros(ComplexF64, dim^2, dim^2)
#     if config.with_coherent  # There is no energy formulation of the coherent term, only Bohr and time.
#         coherent_term = coherent_term_trotter(jump, trotter, f_minus, f_plus)
#         # coherent_term = trotter.trafo_from_eigen_to_trotter' * coherent_term * trotter.trafo_from_eigen_to_trotter
#         liouv_coherent_part_for_jump .+= vectorize_liouvillian_coherent(coherent_term)
#     end

#     for w in energy_labels
#         jump_oft = trotter_oft(jump, w, trotter, oft_time_labels, config.beta) # t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
#         # jump_oft = trotter.trafo_from_eigen_to_trotter' * jump_oft * trotter.trafo_from_eigen_to_trotter
#         liouv_diss_part_for_jump .+= transition(w) * vectorize_liouvillian_diss(jump_oft)
#     end
    
#     prefactor = w0 * trotter.t0^2 * (sqrt(2 / pi) / config.beta) / (2 * pi)
#     return liouv_coherent_part_for_jump .+ prefactor * liouv_diss_part_for_jump  #! L in trotter basis
# end

# function jump_contribution_slow(::TimePicture, jump::JumpOp, hamiltonian::HamHam, config::LiouvConfig, 
#     energy_labels::Vector{Float64}, time_labels::Vector{Float64})

#     dim = size(hamiltonian.data, 1)
#     w0 = abs(energy_labels[2] - energy_labels[1])
#     t0 = time_labels[2] - time_labels[1]
#     oft_time_labels = truncate_time_labels_for_oft(time_labels, config.beta)

#     transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)

#     if config.with_coherent
#         f_minus = compute_truncated_f(compute_f_minus, time_labels, config.beta)
#         if config.with_linear_combination  
#             if config.a != 0.0  # Improved Metro / Glauber
#                 f_plus = compute_truncated_f(compute_f_plus_eh, time_labels, config.beta, config.a, config.b)
#             else  # Metro
#                 f_plus = compute_truncated_f(compute_f_plus_metro, time_labels, config.beta, config.eta)
#             end
#         else  # Gaussian
#             f_plus = compute_truncated_f(compute_f_plus, time_labels, config.beta)
#         end
#     end

#     liouv_coherent_part_for_jump = zeros(ComplexF64, dim^2, dim^2)
#     liouv_diss_part_for_jump = zeros(ComplexF64, dim^2, dim^2)
#     if config.with_coherent 
#         coherent_term = coherent_term_time(jump, hamiltonian, f_minus, f_plus, t0)  
#         liouv_coherent_part_for_jump .+= vectorize_liouvillian_coherent(coherent_term)
#     end

#     for w in energy_labels
#         jump_oft = time_oft(jump, w, hamiltonian, oft_time_labels, config.beta) # subnorm = t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
#         liouv_diss_part_for_jump .+= transition(w) * vectorize_liouvillian_diss(jump_oft)
#     end
#     prefactor = w0 * t0^2 * (sqrt(2 / pi) / config.beta) / (2 * pi)  # time ints t0^2, energy int w0, OFT time norm^2, Fourier
#     return liouv_coherent_part_for_jump .+ prefactor * liouv_diss_part_for_jump
# end

# function jump_contribution_slow(::EnergyPicture, jump::JumpOp, hamiltonian::HamHam, config::LiouvConfig, 
#     energy_labels::Vector{Float64})

#     dim = size(hamiltonian.data, 1)
#     w0 = abs(energy_labels[2] - energy_labels[1])

#     transition = pick_transition(config.beta, config.a, config.b, config.with_linear_combination)

#     liouv_coherent_part_for_jump = zeros(ComplexF64, dim^2, dim^2)
#     liouv_diss_part_for_jump = zeros(ComplexF64, dim^2, dim^2)
#     if config.with_coherent
#         coherent_term = coherent_bohr(hamiltonian, jump, config)
#         liouv_coherent_part_for_jump .+= vectorize_liouvillian_coherent(coherent_term)
#     end

#     for w in energy_labels
#         jump_oft = oft(jump, w, hamiltonian, config.beta)
#         liouv_diss_part_for_jump .+= transition(w) * vectorize_liouvillian_diss(jump_oft)
#     end
#     oft_norm_squared = config.beta / sqrt(2 * pi)
#     return liouv_coherent_part_for_jump .+ w0 * oft_norm_squared * liouv_diss_part_for_jump
# end

# function jump_contribution_slow(::BohrPicture, jump::JumpOp, hamiltonian::HamHam, config::LiouvConfig)
#     dim = size(hamiltonian.data, 1)
#     unique_freqs = keys(hamiltonian.bohr_dict)

#     alpha = pick_alpha(config)

#     liouv_for_jump = zeros(ComplexF64, dim^2, dim^2)  
#     # Coherent part
#     if config.with_coherent
#         coherent_term = coherent_bohr(hamiltonian, jump, config)
#         liouv_for_jump .+= vectorize_liouvillian_coherent(coherent_term)
#     end

#     # Dissipative part
#     for nu_2 in unique_freqs
#         alpha_nu1_matrix = alpha.(hamiltonian.bohr_freqs, nu_2, config.beta, config.a, config.b)

#         A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
#         A_nu_2[hamiltonian.bohr_dict[nu_2]] .= jump.in_eigenbasis[hamiltonian.bohr_dict[nu_2]]

#         liouv_for_jump .+= vectorize_liouvillian_diss(alpha_nu1_matrix .* jump.in_eigenbasis, A_nu_2')
#     end
#     return liouv_for_jump
# end

# function jump_contribution_slow(::BohrPicture, evolving_dm::Matrix{ComplexF64}, jump::JumpOp, hamiltonian::HamHam, 
#     config::ThermalizeConfig)

#     dim = size(evolving_dm, 1)
#     alpha = pick_alpha(config)
#     unique_freqs = keys(hamiltonian.bohr_dict)

#     jump_coherent = zeros(ComplexF64, dim, dim)
#     jump_dissipative = zeros(ComplexF64, dim, dim)
#     # Coherent part
#     if config.with_coherent
#         coherent_term = coherent_bohr(hamiltonian, jump, config)
#         jump_coherent .+= - 1im * (coherent_term * evolving_dm - evolving_dm * coherent_term)
#     end

#     # Dissipative part
#     for nu_2 in unique_freqs
#         A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
#         indices_nu_2 = hamiltonian.bohr_dict[nu_2]
#         A_nu_2[indices_nu_2] .= jump.in_eigenbasis[indices_nu_2]

#         alpha_A_nu1 = alpha.(hamiltonian.bohr_freqs, nu_2, config.beta, config.a, config.b) .* jump.in_eigenbasis

#         jump_dissipative .+= (alpha_A_nu1 * evolving_dm * A_nu_2' - 0.5 * (A_nu_2' * alpha_A_nu1 * evolving_dm 
#                                         + evolving_dm * A_nu_2' * alpha_A_nu1)
#                                         )
#     end
#     return config.delta * (jump_coherent + jump_dissipative)
# end

# function jump_contribution_slow(::EnergyPicture, evolving_dm::Matrix{ComplexF64}, jump::JumpOp, hamiltonian::HamHam, 
#     config::ThermalizeConfig)

#     dim = size(evolving_dm, 1)
#     w0 = abs(energy_labels[2] - energy_labels[1])

#     jump_coherent = zeros(ComplexF64, dim, dim)
#     jump_dissipative = zeros(ComplexF64, dim, dim)
#     # Coherent part
#     if config.with_coherent
#         coherent_term = coherent_bohr(hamiltonian, jump, config)
#         jump_coherent .+= - 1im * (coherent_term * evolving_dm - evolving_dm * coherent_term)
#     end

#     # Dissipative part
#     for w in energy_labels
#         jump_oft = oft(jump, w, hamiltonian, config.beta)
#         jump_dag_jump = jump_oft' * jump_oft
#         jump_dissipative .+= transition(w) * (
#             jump_oft * evolving_dm * jump_oft' - 0.5 * (jump_dag_jump * evolving_dm + evolving_dm * jump_dag_jump)
#             )
#     end

#     oft_prefactor = config.beta / sqrt(2 * pi)
#     return config.delta * (jump_coherent + w0 * oft_prefactor * jump_dissipative)
# end

# function jump_contribution_slow(::TimePicture, evolving_dm::Matrix{ComplexF64}, jump::JumpOp, hamiltonian::HamHam, 
#     config::ThermalizeConfig)

#     dim = size(evolving_dm, 1)
#     w0 = abs(energy_labels[2] - energy_labels[1])
#     t0 = time_labels[2] - time_labels[1]
#     oft_time_labels = truncate_time_labels_for_oft(time_labels, config.beta)

#     jump_coherent = zeros(ComplexF64, dim, dim)
#     jump_dissipative = zeros(ComplexF64, dim, dim)

#     # Coherent part
#     if config.with_coherent
#         coherent_term = coherent_term_time(jump, hamiltonian, f_minus, f_plus, t0)
#         jump_coherent .+= - 1im * (coherent_term * evolving_dm - evolving_dm * coherent_term)
#     end

#     # Dissipative part
#     for w in energy_labels
#         jump_oft = time_oft(jump, w, hamiltonian, oft_time_labels, config.beta)
#         jump_dag_jump = jump_oft' * jump_oft
#         jump_dissipative .+= transition(w) * (
#             jump_oft * evolving_dm * jump_oft' - 0.5 * (jump_dag_jump * evolving_dm + evolving_dm * jump_dag_jump)
#             )
#     end

#     oft_prefactor = (sqrt(2 / pi) / config.beta) / (2 * pi)
#     return config.delta * (jump_coherent + w0 * t0^2 * oft_prefactor * jump_dissipative)
# end

# function jump_contribution_slow(::TrotterPicture, evolving_dm::Matrix{ComplexF64}, jump::JumpOp, trotter::TrottTrott, 
#     config::ThermalizeConfig)

#     dim = size(evolving_dm, 1)
#     w0 = abs(energy_labels[2] - energy_labels[1])
#     oft_time_labels = truncate_time_labels_for_oft(time_labels, config.beta)

#     jump_coherent = zeros(ComplexF64, dim, dim)
#     jump_dissipative = zeros(ComplexF64, dim, dim)

#     # Coherent part
#     if config.with_coherent
#         coherent_term = coherent_term_trotter(jump, trotter, f_minus, f_plus)
#         jump_coherent .+= - 1im * (coherent_term * evolving_dm - evolving_dm * coherent_term)
#     end

#     # Dissipative part
#     for w in energy_labels
#         jump_oft = trotter_oft(jump, w, trotter, oft_time_labels, config.beta) # t0 * sqrt((sqrt(2 / pi)/beta) / (2 * pi))
#         jump_dag_jump = jump_oft' * jump_oft
#         jump_dissipative .+= transition(w) * (
#             jump_oft * evolving_dm * jump_oft' - 0.5 * (jump_dag_jump * evolving_dm + evolving_dm * jump_dag_jump)
#             )
#     end

#     oft_prefactor = (sqrt(2 / pi) / config.beta) / (2 * pi)
#     return config.delta * (jump_coherent + w0 * trotter.t0^2 * oft_prefactor * jump_dissipative)
# end

