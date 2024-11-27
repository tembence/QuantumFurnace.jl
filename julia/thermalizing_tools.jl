using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using LaTeXStrings

include("jump_op_tools.jl")
include("hamiltonian_tools.jl")
include("qi_tools.jl")
include("trotter.jl")
include("structs.jl")

#TODO: Struct for Configs / Initialization of the thermalizations
#TODO: Compare this to other thermalizations

function thermalize_bohr(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, evolved_dm::Matrix{ComplexF64},
    delta::Float64, mixing_time::Float64, beta::Float64)

    # num_liouv_steps = Int(round(mixing_time / delta, digits=0))
    # gibbs = Hermitian(gibbs_state_in_eigen(hamiltonian, beta))
    # distances_to_gibbs = [trace_distance_h(Hermitian(evolved_dm), gibbs)]

    # time_steps = [0.0:delta:(num_liouv_steps * delta);]
    # @showprogress dt=1 desc="Algorithm (Bohr)..." for step in 1:num_liouv_steps
    #     for jump in jumps
    #     end

    # end

end

function dissipative_bohr(jump::JumpOp, hamiltonian::HamHam, evolved_dm::Matrix{ComplexF64}, delta::Float64, beta::Float64)

    dim = size(hamiltonian.data, 1)
    # Transition Gaussian, 2 filter Gaussians
    alpha(nu_1, nu_2) = exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2 / 16) * exp(-beta^2 * (nu_1 - nu_2)^2 / 8) / sqrt(8)
    # Setup coherent part
    # if with_coherent
    #     coherent_term = coherent_gaussian_bohr(hamiltonian, jump, beta)
    #     @printf("Coherent term\n")
    #     display(coherent_term)
    #     temp_coh = - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)
    #     @printf("Coherently evolved part\n")
    #     display(temp_coh)
    #     evolved_dm .+= temp_coh
    # else
    #     @printf("Not adding coherent terms! \n")
    # end

    jump_dissipated_dm = zeros(ComplexF64, dim, dim)
    for j in 1:dim
        for k in 1:dim
            for i in 1:dim
                A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                A_nu_2_dagger::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                nu_1 = hamiltonian.bohr_freqs[i, j]
                nu_2 = hamiltonian.bohr_freqs[i, k]  #! Could be (k, i)
                @printf("---For j, k, i: %d, %d, %d\n", j, k, i)
                @printf("nu_1: %f, nu_2: %f\n", nu_1, nu_2)
                check_alpha_skew_symmetry(alpha, nu_1, nu_2, beta)
                A_nu_1[i, j] = jump.in_eigenbasis[i, j]
                A_nu_2_dagger[k, i] = adjoint(jump.in_eigenbasis[i, k])
                @printf("A_nu_1\n")
                display(A_nu_1)
                @printf("A_nu_2_dagger\n")
                display(A_nu_2_dagger)
                temp_diss = alpha(nu_1, nu_2) * (A_nu_1 * evolved_dm * A_nu_2_dagger
                    - 0.5 * (A_nu_2_dagger * A_nu_1 * evolved_dm + evolved_dm * A_nu_2_dagger * A_nu_1)) 
                @printf("///// Dissipated part\n")
                display(temp_diss)
                println()
                jump_dissipated_dm .+= temp_diss
            end
        end
    end
    @printf("Resulting total dissipation part\n")
    display(jump_dissipated_dm)
    evolved_dm .+= delta * jump_dissipated_dm
    @printf("Resulting delta evolved DM\n")
    display(evolved_dm)
    return evolved_dm
end

function check_alpha_skew_symmetry(alpha::Function, nu_1::Float64, nu_2::Float64, beta::Float64)
    @assert norm(alpha(nu_1, nu_2) - alpha(-nu_2, -nu_1) * exp(-beta * (nu_1 + nu_2) / 2)) < 1e-14
end

function thermalize_gaussian(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, evolved_dm::Matrix{ComplexF64},
    num_energy_bits::Int64, filter_gauss_w::Function, transition_gauss::Function,
    delta::Float64, mixing_time::Float64, beta::Float64)

    # Energy labels
    num_qubits = Int(log2(size(hamiltonian.data)[1]))
    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = hamiltonian.w0 * N_labels

    num_liouv_steps = Int(round(mixing_time / delta, digits=0))
    gibbs = Hermitian(gibbs_state_in_eigen(hamiltonian, beta))

    # Filter Gaussian normalization for the jumps
    filter_gauss_values = filter_gauss_w.(energy_labels)
    filter_gauss_norm_sq = sum(filter_gauss_values.^2)
    
    # Truncate energy -> 0.45 -> transition Gaussian truncation
    energy_labels_045 = energy_labels[abs.(energy_labels) .<= 0.45]
    transition_cutoff = 1e-4  #!
    energy_cutoff_of_transition_gauss = find_zero(x -> transition_gauss(x) - transition_cutoff, 0)
    energy_labels_sym = filter(x -> x <= energy_cutoff_of_transition_gauss && x > 0., energy_labels_045)
    energy_labels_rest = - filter(x -> x > energy_cutoff_of_transition_gauss, energy_labels_045)
    push!(energy_labels_rest, 0.0)

    # Setup coherent part
    # if with_coherent
        # Time labels for coherent
        # t0 = 2 * pi / (N * hamiltonian.w0)
        # time_labels = t0 * N_labels

        # atol = 1e-12
        # b1 = compute_truncated_b1(time_labels, atol)
        # b2 = compute_truncated_b2(time_labels, atol)
        # @printf("t0: %e\n", t0)
        # @printf("Number of b1 terms: %d\n", length(keys(b1)))
        # @printf("Number of b2 terms: %d\n", length(keys(b2)))
    # else
    #     @printf("Not adding coherent terms! \n")
    # end

    distances_to_gibbs = [trace_distance_h(Hermitian(evolved_dm), gibbs)]
    time_steps = [0.0:delta:(num_liouv_steps * delta);]
    @showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps

        # Sum of all jumps at once
        for jump in jumps
            # Coherent term
            if with_coherent
                # coherent_term = coherent_term_from_timedomain(jump, hamiltonian, b1, b2, t0, beta)
                coherent_term = coherent_gaussian_bohr(hamiltonian, jump, beta)
                evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)
    
                if step == 1 && jump == jumps[1]
                    check_B_gauss(coherent_term, jump, hamiltonian, beta)
                end
            end
    
            dissipative_dm_part = zeros(ComplexF64, size(initial_dm))

            # w <= cutoff, A(-w) = A(w)^\dagger
            for w in energy_labels_sym
                oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
                oft_matrix_dag = oft_matrix'  # = A(-w)
                oft_dag_oft = oft_matrix_dag * oft_matrix
                oft_oft_dag = oft_matrix * oft_matrix_dag
                
                # Boy, the brackets in this multiline expression are important.
                dissipative_dm_part .+= transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
                                    - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
                dissipative_dm_part .+=  transition(-w) * (oft_matrix_dag * evolved_dm * oft_matrix 
                                    - 0.5 * (oft_oft_dag * evolved_dm + evolved_dm * oft_oft_dag))
            end

            # w < -cutoff && w = 0.0
            for w in energy_labels_rest
                oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
                oft_matrix_dag = oft_matrix'
                oft_dag_oft = oft_matrix_dag * oft_matrix
                
                dissipative_dm_part .+=  transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
                                    - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
            end
            evolved_dm .+= delta * dissipative_dm_part / filter_gauss_norm_sq
        end

        dist = trace_distance_h(Hermitian(evolved_dm), gibbs)
        push!(distances_to_gibbs, dist)
        # @printf("\nDistance to Gibbs: %f\n", dist)
    end
    return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
end

function thermalize_gaussian_nh(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, 
    num_energy_bits::Int64, evolved_dm::Matrix{ComplexF64}, filter_gauss_w::Function, transition_gauss::Function,
    delta::Float64, mixing_time::Float64, beta::Float64)

    # Energy labels
    num_qubits = Int(log2(size(hamiltonian.data)[1]))
    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = hamiltonian.w0 * N_labels

    num_liouv_steps = Int(round(mixing_time / delta, digits=0))
    gibbs = gibbs_state_in_eigen(hamiltonian, beta)

    # Filter Gaussian normalization for the jumps
    filter_gauss_values = filter_gauss_w.(energy_labels)
    filter_gauss_norm_sq = sum(filter_gauss_values.^2)
    
    # Truncate energy -> 0.45 -> transition Gaussian truncation
    energy_labels_045 = energy_labels[abs.(energy_labels) .<= 0.45]
    transition_cutoff = 1e-4 #!
    energy_cutoff_of_transition_gauss = find_zero(x -> transition_gauss(x) - transition_cutoff, 0)
    energy_labels = filter(x -> x <= energy_cutoff_of_transition_gauss, energy_labels_045)
    @printf("Max energy label: %f\n", maximum(energy_labels))

    # Setup coherent part
    if with_coherent
        # Time labels for coherent
        t0 = 2 * pi / (N * hamiltonian.w0)
        time_labels = t0 * N_labels

        atol = 1e-12
        b1 = compute_truncated_b1(time_labels, atol)
        b2 = compute_truncated_b2(time_labels, atol)
        @printf("t0: %e\n", t0)
        @printf("Number of b1 terms: %d\n", length(keys(b1)))
        @printf("Number of b2 terms: %d\n", length(keys(b2)))
    else
        @printf("Not adding coherent terms! \n")
    end

    distances_to_gibbs = [trace_distance_nh(evolved_dm, gibbs)]
    @showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps
        # Sum of all jumps at once
        for jump in jumps
            # Coherent part
            if with_coherent
                coherent_term = coherent_term_from_timedomain(jump, hamiltonian, b1, b2, t0, beta)
                evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)
    
                if step == 1 && jump == jumps[1]
                    check_B_gauss(coherent_term, jump, hamiltonian, beta)
                end
            end

            # Dissipative part
            dissipative_dm_part = zeros(ComplexF64, size(initial_dm))
            for w in energy_labels
                oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
                oft_matrix_dag = oft_matrix'
                oft_dag_oft = oft_matrix_dag * oft_matrix
                
                dissipative_dm_part .+=  transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
                                    - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
            end

            evolved_dm .+= delta * dissipative_dm_part / filter_gauss_norm_sq
            
            dist = trace_distance_nh(evolved_dm, gibbs)
            push!(distances_to_gibbs, dist)
            # @printf("\nDistance to Gibbs: %f\n", dist)
        end
    end
    return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
end

function thermalize_gaussian_random(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, 
    num_energy_bits::Int64, evolved_dm::Matrix{ComplexF64}, filter_gauss_w::Function, transition_gauss::Function,
    delta::Float64, mixing_time::Float64, beta::Float64)

    # Energy labels
    num_qubits = Int(log2(size(hamiltonian.data)[1]))
    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = hamiltonian.w0 * N_labels

    num_liouv_steps = Int(round(mixing_time / delta, digits=0))
    gibbs = gibbs_state_in_eigen(hamiltonian, beta)

    # Filter Gaussian normalization for the jumps
    filter_gauss_values = filter_gauss_w.(energy_labels)
    filter_gauss_norm_sq = sum(filter_gauss_values.^2)
    
    # Truncate energy -> 0.45 -> transition Gaussian truncation
    energy_labels_045 = energy_labels[abs.(energy_labels) .<= 0.45]
    transition_cutoff = 1e-4  #!
    energy_cutoff_of_transition_gauss = find_zero(x -> transition_gauss(x) - transition_cutoff, 0)
    energy_labels_sym = filter(x -> x <= energy_cutoff_of_transition_gauss && x > 0., energy_labels_045)
    energy_labels_rest = - filter(x -> x > energy_cutoff_of_transition_gauss, energy_labels_045)
    push!(energy_labels_rest, 0.0)

    # Setup coherent part
    if with_coherent
        # Time labels for coherent
        t0 = 2 * pi / (N * hamiltonian.w0)
        time_labels = t0 * N_labels

        atol = 1e-12
        b1 = compute_truncated_b1(time_labels, atol)
        b2 = compute_truncated_b2(time_labels, atol)
        @printf("t0: %e\n", t0)
        @printf("Number of b1 terms: %d\n", length(keys(b1)))
        @printf("Number of b2 terms: %d\n", length(keys(b2)))
    else
        @printf("Not adding coherent terms! \n")
    end

    distances_to_gibbs = [trace_distance_nh(evolved_dm, gibbs)]
    @showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps

        # Pick a random jump for this step
        jump = jumps[step]

        # Coherent term
        if with_coherent
            coherent_term = coherent_term_from_timedomain(jump, hamiltonian, b1, b2, t0, beta)
            evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)

            if step == 1 && jump == jumps[1]
                check_B_gauss(coherent_term, jump, hamiltonian, beta)
            end
        end

        dissipative_dm_part = zeros(ComplexF64, size(initial_dm))
        # w <= cutoff, A(-w) = A(w)^\dagger
        for w in energy_labels_sym
            oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
            oft_matrix_dag = oft_matrix'  # = A(-w)
            oft_dag_oft = oft_matrix_dag * oft_matrix
            oft_oft_dag = oft_matrix * oft_matrix_dag
            
            # Boy, the brackets in this multiline expression are important.
            dissipative_dm_part .+= transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
                                - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
            dissipative_dm_part .+=  transition(-w) * (oft_matrix_dag * evolved_dm * oft_matrix 
                                - 0.5 * (oft_oft_dag * evolved_dm + evolved_dm * oft_oft_dag))
        end

        # w < -cutoff && w = 0.0
        for w in energy_labels_rest
            oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
            oft_matrix_dag = oft_matrix'
            oft_dag_oft = oft_matrix_dag * oft_matrix
            
            dissipative_dm_part .+=  transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
                                - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
        end

        evolved_dm .+= delta * dissipative_dm_part / filter_gauss_norm_sq
        
        dist = trace_distance_nh(evolved_dm, gibbs)
        push!(distances_to_gibbs, dist)
        # @printf("\nDistance to Gibbs: %f\n", dist)
    end
    return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
end

function thermalize_gaussian_ideal_time(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, 
    evolved_dm::Matrix{ComplexF64}, num_energy_bits::Int64, filter_gauss_t::Function, transition_gauss::Function, 
    delta::Float64, mixing_time::Float64, beta::Float64)

    num_qubits = Int(log2(size(hamiltonian.data)[1]))
    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = hamiltonian.w0 * N_labels
    t0 = 2 * pi / (N * hamiltonian.w0)
    time_labels = t0 * N_labels

    filter_gauss_t_values = filter_gauss_t.(time_labels)  # exp.(- time_labels.^2 / beta^2)
    filter_gauss_t_norm_sq = sum(filter_gauss_t_values.^2)

    num_liouv_steps = Int(round(mixing_time / delta, digits=0))
    gibbs = gibbs_state_in_eigen(hamiltonian, beta)
    
    # Truncate energy -> 0.45 -> transition Gaussian truncation
    energy_labels_045 = energy_labels[abs.(energy_labels) .<= 0.45]
    transition_cutoff = 1e-4  #!
    energy_cutoff_of_transition_gauss = find_zero(x -> transition_gauss(x) - transition_cutoff, 0)
    energy_labels_sym = filter(x -> x <= energy_cutoff_of_transition_gauss && x > 0., energy_labels_045)
    energy_labels_rest = - filter(x -> x > energy_cutoff_of_transition_gauss, energy_labels_045)
    push!(energy_labels_rest, 0.0)

    # Setup coherent part
    if with_coherent
        atol = 1e-12
        b1 = compute_truncated_b1(time_labels, atol)
        b2 = compute_truncated_b2(time_labels, atol)
        @printf("t0: %e\n", t0)
        @printf("Number of b1 terms: %d\n", length(keys(b1)))
        @printf("Number of b2 terms: %d\n", length(keys(b2)))
    else
        @printf("Not adding coherent terms! \n")
    end

    distances_to_gibbs = [trace_distance_nh(evolved_dm, gibbs)]
    time_steps = [0.0:delta:(num_liouv_steps * delta);]
    @showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps

        # Sum of all jumps at once
        for jump in jumps
            # Coherent term 
            if with_coherent
                coherent_term = coherent_term_from_timedomain(jump, hamiltonian, b1, b2, t0, beta)
                evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)
    
                if step == 1 && jump == jumps[1]
                    check_B_gauss(coherent_term, jump, hamiltonian, beta)
                end
            end
    
            # w <= cutoff, A(-w) = A(w)^\dagger
            dissipative_dm_part = zeros(ComplexF64, size(initial_dm))
            for w in energy_labels_sym
                oft_matrix = explicit_oft_exact_db(jump, hamiltonian, w, time_labels, beta)
                oft_matrix_dag = oft_matrix'  # = A(-w)
                oft_dag_oft = oft_matrix_dag * oft_matrix
                oft_oft_dag = oft_matrix * oft_matrix_dag
                
                # Boy, the brackets in this multiline expression are important.
                dissipative_dm_part .+= transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
                                    - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
                dissipative_dm_part .+=  transition(-w) * (oft_matrix_dag * evolved_dm * oft_matrix 
                                    - 0.5 * (oft_oft_dag * evolved_dm + evolved_dm * oft_oft_dag))
            end

            # w < -cutoff && w = 0.0
            for w in energy_labels_rest
                oft_matrix = explicit_oft_exact_db(jump, hamiltonian, w, time_labels, beta)
                oft_matrix_dag = oft_matrix'
                oft_dag_oft = oft_matrix_dag * oft_matrix
                
                dissipative_dm_part .+=  transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
                                    - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
            end

            evolved_dm .+= delta * dissipative_dm_part / (filter_gauss_t_norm_sq * length(time_labels))
        end  
        dist = trace_distance_nh(evolved_dm, gibbs)
        push!(distances_to_gibbs, dist)
        # @printf("\nDistance to Gibbs: %f\n", dist)
    end
    return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
end

#TODO: Write it up s.t. the error is like for the algorithm, i.e. twos complement Trotter.
function thermalize_gaussian_trotter(jumps::Vector{JumpOp}, hamiltonian::HamHam, trotter::TrottTrott, with_coherent::Bool, 
    evolved_dm::Matrix{ComplexF64}, num_energy_bits::Int64, filter_gauss_t::Function, transition_gauss::Function, 
    delta::Float64, mixing_time::Float64, beta::Float64)
    """In Trotter basis"""

    num_qubits = Int(log2(size(hamiltonian.data)[1]))
    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = hamiltonian.w0 * N_labels
    time_labels = trotter.t0 * N_labels

    filter_gauss_t_values = filter_gauss_t.(time_labels)  # exp.(- time_labels.^2 / beta^2)
    filter_gauss_t_norm_sq = sum(filter_gauss_t_values.^2)

    num_liouv_steps = Int(round(mixing_time / delta, digits=0))
    gibbs_in_trotter = trotter.eigvecs' * gibbs_state(hamiltonian, beta) * trotter.eigvecs
    evolved_dm = trotter.eigvecs' * initial_dm * trotter.eigvecs
    
    # Truncate energy -> 0.45 -> transition Gaussian truncation
    energy_labels_045 = energy_labels[abs.(energy_labels) .<= 0.45]
    transition_cutoff = 1e-4  #!
    energy_cutoff_of_transition_gauss = find_zero(x -> transition_gauss(x) - transition_cutoff, 0)
    energy_labels_sym = filter(x -> x <= energy_cutoff_of_transition_gauss && x > 0., energy_labels_045)
    energy_labels_rest = - filter(x -> x > energy_cutoff_of_transition_gauss, energy_labels_045)
    push!(energy_labels_rest, 0.0)

    # Setup coherent part
    if with_coherent
        atol = 1e-12
        b1 = compute_truncated_b1(time_labels, atol)
        b2 = compute_truncated_b2(time_labels, atol)
        @printf("t0: %e\n", trotter.t0)
        @printf("Number of b1 terms: %d\n", length(keys(b1)))
        @printf("Number of b2 terms: %d\n", length(keys(b2)))
    else
        @printf("Not adding coherent terms! \n")
    end

    distances_to_gibbs = [trace_distance_nh(evolved_dm, gibbs_in_trotter)]
    time_steps = [0.0:delta:(num_liouv_steps * delta);]
    @showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps

        # Sum of all jumps at once
        for jump in jumps
            # Coherent term 
            if with_coherent
                coherent_term = coherent_term_trotter(jump, hamiltonian, trotter, b1, b2, beta)
                evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)
    
                if step == 1 && jump == jumps[1]
                    check_B_gauss(coherent_term, jump, hamiltonian, beta)
                end
            end
    
            # w <= cutoff, A(-w) = A(w)^\dagger
            dissipative_dm_part = zeros(ComplexF64, size(initial_dm))
            for w in energy_labels_sym
                oft_matrix = trotter_oft(jump, trotter, w, time_labels, beta)
                oft_matrix_dag = oft_matrix'  # = A(-w)
                oft_dag_oft = oft_matrix_dag * oft_matrix
                oft_oft_dag = oft_matrix * oft_matrix_dag
                
                # Boy, the brackets in this multiline expression are important.
                dissipative_dm_part .+= transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
                                    - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
                dissipative_dm_part .+=  transition(-w) * (oft_matrix_dag * evolved_dm * oft_matrix 
                                    - 0.5 * (oft_oft_dag * evolved_dm + evolved_dm * oft_oft_dag))
            end

            # w < -cutoff && w = 0.0
            for w in energy_labels_rest
                oft_matrix = trotter_oft(jump, trotter, w, time_labels, beta)
                oft_matrix_dag = oft_matrix'
                oft_dag_oft = oft_matrix_dag * oft_matrix
                
                dissipative_dm_part .+=  transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
                                    - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
            end

            evolved_dm .+= delta * dissipative_dm_part / (filter_gauss_t_norm_sq * length(time_labels))
        end  
        dist = trace_distance_nh(evolved_dm, gibbs_in_trotter)
        push!(distances_to_gibbs, dist)
        # @printf("\nDistance to Gibbs: %f\n", dist)
    end
    return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
end

function thermalize_metro(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, evolved_dm::Matrix{ComplexF64},
    num_energy_bits::Int64, filter_gauss_w::Function, transition_metro::Function,
    eta::Float64, delta::Float64, mixing_time::Float64, beta::Float64)

    num_qubits = Int(log2(size(hamiltonian.data)[1]))
    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = hamiltonian.w0 * N_labels

    num_liouv_steps = Int(round(mixing_time / delta, digits=0))
    gibbs = gibbs_state_in_eigen(hamiltonian, beta)

    # Filter Gaussian normalization for the jumps
    filter_gauss_values = filter_gauss_w.(energy_labels)
    filter_gauss_norm_sq = sum(filter_gauss_values.^2)
    
    # Truncate energy -> 0.45 -/-> no other truncation possible really
    energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]
    energy_labels_no_zero = energy_labels[energy_labels .!= 0.0]

    # Setup coherent part
    if with_coherent
        # Time labels for coherent
        t0 = 2 * pi / (N * hamiltonian.w0)
        time_labels = t0 * N_labels

        coherent_terms_atol = 1e-12
        b1 = compute_truncated_b1(time_labels, coherent_terms_atol)
        b2 = compute_truncated_b2_metro(time_labels, eta, coherent_terms_atol)
        @printf("t0: %e\n", t0)
        @printf("Number of b1 terms: %d\n", length(keys(b1)))
        @printf("Number of b2 terms: %d\n", length(keys(b2)))
    else
        @printf("Not adding coherent terms! \n")
    end

    distances_to_gibbs = [trace_distance_nh(evolved_dm, gibbs)]
    time_steps = [0.0:delta:(num_liouv_steps * delta);]
    @showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps
        # Sum of all jumps at once
        for jump in jumps
            # Coherent term
            if with_coherent
                coherent_term = coherent_term_from_timedomain(jump, hamiltonian, b1, b2, t0, beta)
                evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)
    
                if step == 1 && jump == jumps[1]
                    check_B_metro(coherent_term, jump, hamiltonian, eta, beta)
                end
            end
    
            dissipative_dm_part = zeros(ComplexF64, size(initial_dm))

            # w != 0, A(-w) = A(w)^\dagger
            for w in energy_labels_no_zero
                oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
                oft_matrix_dag = oft_matrix'  # = A(-w)
                oft_dag_oft = oft_matrix_dag * oft_matrix
                oft_oft_dag = oft_matrix * oft_matrix_dag
                
                dissipative_dm_part .+= transition_metro(w) * (oft_matrix * evolved_dm * oft_matrix_dag
                                    - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
                dissipative_dm_part .+=  transition_metro(-w) * (oft_matrix_dag * evolved_dm * oft_matrix 
                                    - 0.5 * (oft_oft_dag * evolved_dm + evolved_dm * oft_oft_dag))
            end

            w = 0.0
            oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
            oft_matrix_dag = oft_matrix'
            oft_dag_oft = oft_matrix_dag * oft_matrix

            dissipative_dm_part .+=  transition_metro(w) * (oft_matrix * evolved_dm * oft_matrix_dag
                                - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))

            evolved_dm .+= delta * dissipative_dm_part / filter_gauss_norm_sq
        end   
        dist = trace_distance_nh(evolved_dm, gibbs)
        push!(distances_to_gibbs, dist)
        # @printf("\nDistance to Gibbs: %f\n", dist)
    end
    return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
end

function thermalize_metro_trotter(jumps::Vector{JumpOp}, hamiltonian::HamHam, trotter::TrottTrott, with_coherent::Bool,
    evolved_dm::Matrix{ComplexF64},num_energy_bits::Int64, filter_gauss_t::Function, transition_metro::Function,
    eta::Float64, delta::Float64, mixing_time::Float64, beta::Float64)

    num_qubits = Int(log2(size(hamiltonian.data)[1]))
    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = hamiltonian.w0 * N_labels
    time_labels = trotter.t0 * N_labels

    filter_gauss_t_values = filter_gauss_t.(time_labels)  # exp.(- time_labels.^2 / beta^2)
    filter_gauss_t_norm_sq = sum(filter_gauss_t_values.^2)

    num_liouv_steps = Int(round(mixing_time / delta, digits=0))
    gibbs_in_trotter = trotter.eigvecs' * gibbs_state(hamiltonian, beta) * trotter.eigvecs
    evolved_dm = trotter.eigvecs' * initial_dm * trotter.eigvecs
    
    # Truncate energy -> 0.45 -/-> no other truncation possible really
    energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]
    energy_labels_no_zero = energy_labels[energy_labels .!= 0.0]

    # Setup coherent part
    if with_coherent
        coherent_terms_atol = 1e-12
        b1 = compute_truncated_b1(time_labels, coherent_terms_atol)
        b2 = compute_truncated_b2_metro(time_labels, eta, coherent_terms_atol)
        @printf("t0: %e\n", trotter.t0)
        @printf("Number of b1 terms: %d\n", length(keys(b1)))
        @printf("Number of b2 terms: %d\n", length(keys(b2)))
    else
        @printf("Not adding coherent terms! \n")
    end

    distances_to_gibbs = [trace_distance_nh(evolved_dm, gibbs_in_trotter)]
    time_steps = [0.0:delta:(num_liouv_steps * delta);]
    @showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps
        # Sum of all jumps at once
        for jump in jumps
            # Coherent term
            if with_coherent
                coherent_term = coherent_term_trotter(jump, hamiltonian, trotter, b1, b2, beta)
                evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)
    
                if step == 1 && jump == jumps[1]
                    check_B_metro(coherent_term, jump, hamiltonian, eta, beta)
                end
            end
    
            dissipative_dm_part = zeros(ComplexF64, size(initial_dm))

            # w != 0, A(-w) = A(w)^\dagger
            for w in energy_labels_no_zero
                oft_matrix = trotter_oft(jump, trotter, w, time_labels, beta)
                oft_matrix_dag = oft_matrix'  # = A(-w)
                oft_dag_oft = oft_matrix_dag * oft_matrix
                oft_oft_dag = oft_matrix * oft_matrix_dag
                
                dissipative_dm_part .+= transition_metro(w) * (oft_matrix * evolved_dm * oft_matrix_dag
                                    - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
                dissipative_dm_part .+=  transition_metro(-w) * (oft_matrix_dag * evolved_dm * oft_matrix 
                                    - 0.5 * (oft_oft_dag * evolved_dm + evolved_dm * oft_oft_dag))
            end

            w = 0.0
            oft_matrix = trotter_oft(jump, trotter, w, time_labels, beta)
            oft_matrix_dag = oft_matrix'
            oft_dag_oft = oft_matrix_dag * oft_matrix

            dissipative_dm_part .+=  transition_metro(w) * (oft_matrix * evolved_dm * oft_matrix_dag
                                - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))

            evolved_dm .+= delta * dissipative_dm_part / (filter_gauss_t_norm_sq * length(time_labels))
        end
        dist = trace_distance_nh(evolved_dm, gibbs_in_trotter)
        push!(distances_to_gibbs, dist)
        # @printf("\nDistance to Gibbs: %f\n", dist)
    end
    return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
end

function liouvillian_delta_trajectory_trotter_gaussian_exact_db(jump::JumpOp, trotter::TrottTrott, 
    coherent_term::Matrix{ComplexF64}, energy_labels::Vector{Float64}, initial_dm::Matrix{ComplexF64}, 
    delta::Float64, sigma::Float64, beta::Float64)

    # keep only energies in between -0.45 and 0.45 (physically impossible to get values outside of this range)
    energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]

    transition_gaussian(energy) = exp(-(beta * energy + 1)^2 / 2)
    transition_weights = transition_gaussian.(energy_labels)

    evolved_dm = - im * delta * (coherent_term * initial_dm - initial_dm * coherent_term)
    for (i, w) in enumerate(energy_labels)
        oft_matrix = trotter_oft(jump, trotter, w, time_labels, beta)
        oft_matrix_dag = oft_matrix'
        
        evolved_dm .+= delta * transition_weights[i] *
                    (oft_matrix * initial_dm * oft_matrix_dag
                    - 0.5 * oft_matrix_dag * oft_matrix * initial_dm
                    - 0.5 * initial_dm * oft_matrix_dag * oft_matrix)
    end

    # Return in energy basis
    return evolved_dm
end

# Trotter and alg match too well, 1 step per t0 is too good.
function liouvillian_delta_trajectory_trotter(jump::JumpOp, trotter::TrottTrott,
    energy_labels::Vector{Float64}, initial_dm::Matrix{ComplexF64}, delta::Float64, sigma::Float64, beta::Float64)
    """Everything in Trotter basis, initial_dm too as input."""

    boltzmann_factor(energy) = min(1, exp(-beta * energy))
    
    # keep only energies in between -0.45 and 0.45
    energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]
    boltzmann_values = boltzmann_factor.(energy_labels)

    #! Added enumerate, not debugged yet
    evolved_dm = zeros(ComplexF64, size(initial_dm))
    for (i, w) in enumerate(energy_labels)
        oft_matrix = explicit_trotter_oft(jump, trotter, w, time_labels, sigma, beta)
        oft_matrix_dag = oft_matrix'
        
        evolved_dm .+= delta * boltzmann_values[i] *
                    (oft_matrix * initial_dm * oft_matrix_dag
                    - 0.5 * oft_matrix_dag * oft_matrix * initial_dm
                    - 0.5 * initial_dm * oft_matrix_dag * oft_matrix)
    end

    # Return in Trotter basis
    return evolved_dm
end

function liouvillian_delta_trajectory_metro_exact_db(jump::JumpOp, hamiltonian::HamHam, coherent_term::Matrix{ComplexF64},
    energy_labels::Vector{Float64}, initial_dm::Matrix{ComplexF64}, delta::Float64, beta::Float64)

    # keep only energies in between -0.45 and 0.45 (physically impossible to get values outside of this range)
    energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]

    transition_metropolis(energy) = exp(-beta * maximum([energy + 1/(2*beta), 0]))
    transition_weights = transition_metropolis.(energy_labels)

    # Is there any transiton weight that is smaller than 1e-14?
    if any(transition_weights .< 1e-14)
        @printf("Transition weights smaller than 1e-14: %s\n", transition_weights[transition_weights .< 1e-14])
    end

    evolved_dm = - im * delta * (coherent_term * initial_dm - initial_dm * coherent_term)
    for (i, w) in enumerate(energy_labels)
        oft_matrix = explicit_oft_exact_db(jump, hamiltonian, w, time_labels, beta)
        oft_matrix_dag = oft_matrix'
        
        evolved_dm .+= delta * transition_weights[i] *
                    (oft_matrix * initial_dm * oft_matrix_dag
                    - 0.5 * oft_matrix_dag * oft_matrix * initial_dm
                    - 0.5 * initial_dm * oft_matrix_dag * oft_matrix)
    end

    # Return in energy basis
    return evolved_dm
end

function liouvillian_delta_trajectory(jump::JumpOp, hamiltonian::HamHam,
    energy_labels::Vector{Float64}, initial_dm::Matrix{ComplexF64}, delta::Float64, sigma::Float64, beta::Float64)
    """Computes δ * L[rho_0]"""

    Fw = exp.(- sigma^2 * (energy_labels).^2)
    Fw_norm = sqrt(sum(Fw.^2))
    boltzmann_factor(energy) = min(1, exp(-beta * energy))

    # keep only energies in between -0.45 and 0.45
    energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]

    evolved_dm = zeros(ComplexF64, size(initial_dm))
    for w in energy_labels
        oft_matrix = entry_wise_oft(jump, w, hamiltonian, sigma, beta) / Fw_norm
        oft_matrix_dag = oft_matrix'
        
        evolved_dm .+= delta * boltzmann_factor(w) *
                    (oft_matrix * initial_dm * oft_matrix_dag
                    - 0.5 * oft_matrix_dag * oft_matrix * initial_dm
                    - 0.5 * initial_dm * oft_matrix_dag * oft_matrix)
    end
    return evolved_dm
end

function full_liouvillian_step(jump::JumpOp, hamiltonian::HamHam, energy_labels::Vector{Float64},
    initial_dm::Matrix{ComplexF64}, delta::Float64, sigma::Float64, beta::Float64)

    num_qubits = Int(log2(size(hamiltonian.data)[1]))

    initial_dm = Operator(b, initial_dm)
    evolution_hamiltonian = Operator(b, spzeros(ComplexF64, 2^num_qubits, 2^num_qubits))
    Fw = exp.(- sigma^2 * (energy_labels).^2)
    Fw_norm = sqrt(sum(Fw.^2))
    boltzmann_factor(energy) = min(1, exp(-beta * energy))
    # All jumps
    all_jumps = [Operator(b, sqrt(boltzmann_factor(w)) * 
                                entry_wise_oft(jump, w, hamiltonian, sigma, beta) / Fw_norm) for w in energy_labels]

    tout, evolved_dms = timeevolution.master([0.0, delta], initial_dm, evolution_hamiltonian, all_jumps) 
    # Trace is already = 1
    return evolved_dms[2].data
end

#* ---------------------- TESTING ---------------------- *#
#! 68s (q, r) = (8, 16)

#* Parameters
# num_qubits = 4
# mixing_time = 10.0
# delta = 0.1
# num_liouv_steps = Int(mixing_time / delta)
# sigma = 5.
# beta = 1.
# eig_index = 8

# #* Hamiltonian
# hamiltonian = load("/Users/bence/code/liouvillian_metro/julia/data/hamiltonian_n4.jld")["ideal_ham"]
# # hamiltonian = find_ideal_heisenberg(num_qubits, fill(1.0, 3), batch_size=1)
# # initial_state = hamiltonian.eigvecs[:, eig_index]

# initial_dm = zeros(ComplexF64, size(hamiltonian.data))
# initial_dm[eig_index, eig_index] = 1.0  # In eigenbasis
# hamiltonian.bohr_freqs = hamiltonian.eigvals .- transpose(hamiltonian.eigvals)
# # display(hamiltonian.bohr_freqs)

# #* Jump operators
# X::Matrix{ComplexF64} = [0 1; 1 0]
# Y::Matrix{ComplexF64} = [0.0 -im; im 0.0]
# Z::Matrix{ComplexF64} = [1 0; 0 -1]
# jump_paulis = [X, Y, Z]

# # jump_op = Matrix(pad_term([X], num_qubits, jump_site_index))
# # jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
# # jump = JumpOp(jump_op,
# #         jump_op_in_eigenbasis,
# #         Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
# #         zeros(0))

# #* Fourier labels
# num_energy_bits = ceil(Int64, log2((0.45 * 2) / hamiltonian.w0)) + 1 # paper (above 3.7.), later will be β dependent
# num_energy_bits = 9
# N = 2^(num_energy_bits)
# N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]

# t0 = 2 * pi / (N * hamiltonian.w0)
# time_labels = t0 * N_labels
# energy_labels = hamiltonian.w0 * N_labels

# @printf("Number of qubits: %d\n", num_qubits)
# @printf("Number of energy bits: %d\n", num_energy_bits)
# @printf("Energy unit: %e\n", hamiltonian.w0)
# @printf("Time unit: %e\n", t0)

# # Bohr freqs rounded
# # oft_precision = ceil(Int, abs(log10(N^(-1))))
# # hamiltonian.bohr_freqs = round.(hamiltonian.eigvals .- transpose(hamiltonian.eigvals), digits=oft_precision)

# #* Trotter
# num_trotter_steps_per_t0 = Int(1)
# trotter = create_trotter(hamiltonian, t0, num_trotter_steps_per_t0)
# @time trotter_error_T = compute_trotter_error(hamiltonian, trotter, N*t0 / 2)
# trotter_error_t0 = compute_trotter_error(hamiltonian, trotter, t0)

# @printf("t0: %e\n", trotter.t0)
# @printf("Steps per t0: %d\n", trotter.num_trotter_steps_per_t0)
# @printf("Max time: %e\n", N*t0)
# @printf("Trotter error T: %e\n", trotter_error_T)
# @printf("Trotter error t0: %e\n", trotter_error_t0)

# #* Many steps convergence:
# gibbs = gibbs_state(hamiltonian, beta)
# round.(abs.(gibbs), digits=4)
# evolved_dm = copy(initial_dm)
# distances_to_gibbs = [trace_distance_nh(evolved_dm, gibbs)]

# # Pregenerate all random jumps
# all_random_jumps_generated = []
# Random.seed!(666)
# for _ in 1:num_liouv_steps
#     random_site = rand(1:num_qubits)
#     random_pauli = rand(jump_paulis)
#     jump_op = Matrix(pad_term([random_pauli], num_qubits, random_site))
#     jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
#     jump_in_trotter_basis = trotter.eigvecs' * jump_op * trotter.eigvecs
#     jump = JumpOp(jump_op,
#             jump_op_in_eigenbasis,
#             Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
#             zeros(0),
#             jump_in_trotter_basis)
#     push!(all_random_jumps_generated, jump)
# end

# tspan =[0.0:delta:mixing_time;]
#* Trotter Alg
# evolved_dm_trott = trotter.eigvecs' * hamiltonian.eigvecs * initial_dm * hamiltonian.eigvecs' * trotter.eigvecs
# gibbs_in_trotter_basis = trotter.eigvecs' * hamiltonian.eigvecs * gibbs * hamiltonian.eigvecs' * trotter.eigvecs
# trott_distances_to_gibbs = [distances_to_gibbs[1]]
# for delta_step in 1:num_liouv_steps
#     # Random jump
#     jump_delta = all_random_jumps_generated[delta_step]

#     # Evolve by delta time steps in Trotter basis
#     evolved_dm_trott += liouvillian_delta_trajectory_trotter(jump_delta, trotter, energy_labels, evolved_dm_trott, 
#                     delta, sigma, beta)
#     evolved_dm_trott /= tr(evolved_dm_trott)
#     dist = trace_distance_nh(evolved_dm, gibbs_in_trotter_basis)
#     @printf("Distance to Gibbs: %f\n", dist)
#     push!(trott_distances_to_gibbs, dist)
# end

# plot(tspan, trott_distances_to_gibbs, ylims=(0, 1), color=:purple,
#     label="time domain", xlabel="Time", ylabel="Trace distance", 
#     title="Convergence to Gibbs state")

# println(round.(abs.(diag(hamiltonian.eigvecs' * trotter.eigvecs * evolved_dm_trott * trotter.eigvecs' * hamiltonian.eigvecs)), digits=4))
# println(round.(abs.(diag(gibbs)), digits=4))
# trace_distance_nh(hamiltonian.eigvecs' * trotter.eigvecs * evolved_dm_trott * trotter.eigvecs' * hamiltonian.eigvecs, gibbs)


# plot!(tspan, trott_distances_to_gibbs, ylims=(0, 1),
#         label="0")

# @printf("Final distance to Gibbs (Trotter): %f\n", trott_distances_to_gibbs[end])

#* Alg

# evolved_dm = copy(initial_dm)
# p = Progress(length(num_liouv_steps))
# @showprogress dt=1 desc="Algorithmic converging to Gibbs..." for delta_step in 1:num_liouv_steps
#     # Random jump
#     jump_delta = all_random_jumps_generated[delta_step]

#     # Evolve by delta time steps
#     evolved_dm += liouvillian_delta_trajectory(jump_delta, hamiltonian, energy_labels, evolved_dm, delta, sigma, beta)
#     evolved_dm /= tr(evolved_dm)
#     dist = tracedistance_nh(Operator(b, evolved_dm), Operator(b, gibbs))
#     @printf("Distance to Gibbs: %f\n", dist)
#     # next!(p, showvalues = [(:dist, dist)])
#     push!(distances_to_gibbs, dist)
# end

# plot(tspan, distances_to_gibbs, ylims=(0, 1), color=:purple,
#     label="r=1", xlabel="Time", ylabel="Trace distance", 
#     title="Convergence to Gibbs state, n4-Heisenberg-Z")

# plot!(tspan, distances_to_gibbs, ylims=(0, 1),
#     label="r=9")
    
# savefig("n4-r1-r9_gibbs_conv.pdf")

# plot!(tspan, distances_to_gibbs, ylims=(0, 1), color=:maroon,
#     label="Algorithm")

# @printf("Final distance to Gibbs: %f\n", distances_to_gibbs[end])

# #* Deviations between Trotter and Exact time evolution
# deviations = norm(distances_to_gibbs - trott_distances_to_gibbs)
# @printf("Trotter error T: %e\n", trotter_error_T)
# @printf("Trotter error t0: %e\n", trotter_error_t0)
# @printf("Deviation between Trotter and Alg: %s\n", deviations)

# #* Exact
# evolved_dm = copy(initial_dm)
# fids = [fidelity(Hermitian(evolved_dm), Hermitian(gibbs))]
# exact_distances_to_gibbs = [tracedistance_nh(Operator(b, evolved_dm), Operator(b, gibbs))]
# for delta_step in 1:num_liouv_steps

#     # Random jump
#     jump_delta = all_random_jumps_generated[delta_step]

#     # Evolve by delta time steps
#     evolved_dm = exact_liouvillian_step(jump_delta, hamiltonian, energy_labels, evolved_dm, delta, sigma, beta)
#     dist = tracedistance_h(Operator(b, evolved_dm), Operator(b, gibbs))
#     fid = fidelity(Hermitian(evolved_dm), Hermitian(gibbs))
#     @printf("Distance to Gibbs: %f\n", dist)
#     @printf("Fidelity to Gibbs: %f\n", fid)
#     push!(exact_distances_to_gibbs, dist)
#     push!(fids, fid)
# end

# @printf("Final distance to Gibbs: %f\n", distances_to_gibbs[end])
# @printf("Final fidelity to Gibbs: %f\n", fids[end])

# #* Plot
# # Ribbon with 2 * delta^2 width around exact trace distance curve
# plot!(tspan, exact_distances_to_gibbs, ribbon=2*delta^2, fillalpha=0.2, fillcolor=:orange, label="Exact", color=:orange)
# # savefig("4-8_gibbs_conv")
# # annotate!(1, 0.5, text("Final distance to Gibbs: $round((distances_to_gibbs[end]), digits=3)", 10, :left))

# # plot!(tspan, fids, ylims=(0, 1),
# #     label="Exact Fidelity to Gibbs", xlabel="Time", ylabel="Fidelity", 
# #     title="Liouvillian dynamics")

# #* Exact Liouvillian evolution at once
# # gibbs = gibbs_state(hamiltonian, beta)

# # evolved_dm = copy(initial_dm)
# # Random.seed!(667)
# # random_site = rand(1:num_qubits)
# # random_pauli = rand(jump_paulis)

# # @printf("Random site: %d\n", random_site)
# # @printf("Random Pauli: %s\n", random_pauli)
# # jump_op = Matrix(pad_term([random_pauli], num_qubits, random_site))
# # jump_op_in_eigenbasis = hamiltonian.eigvecs' * jump_op * hamiltonian.eigvecs
# # jump = JumpOp(jump_op,
# #         jump_op_in_eigenbasis,
# #         Dict{Float64, SparseMatrixCSC{ComplexF64, Int64}}(), 
# #         zeros(0))

# # evolved_dm_exact = exact_liouvillian_step(jump, hamiltonian, initial_dm, energy_labels, delta, sigma, beta)
# # evolved_dm_alg = initial_dm + liouvillian_delta_trajectory(jump, hamiltonian, energy_labels, initial_dm, delta, sigma, beta)
# # evolved_dm_alg /= tr(evolved_dm_alg)

# # println("Eigvals of evolved dm alg")
# # println(eigvals(evolved_dm_alg))
# # # trace_distance(Hermitian(evolved_dm_alg), Hermitian(gibbs))
# # # Compare
# # @printf("Distance to each other: %s\n", tracedistance_nh(Operator(b, evolved_dm_exact), Operator(b, evolved_dm_alg)))
# # @printf("While delta^2 is: %s\n", delta^2)

# # @printf("Alg dist to gibbs: %s\n", tracedistance_nh(Operator(b, gibbs), Operator(b, evolved_dm_alg)))
# # @printf("Exact dist to gibbs: %s\n", tracedistance_nh(Operator(b, gibbs), Operator(b, evolved_dm_exact)))
# # printf("Trace distance: %f\n", trace_distance(Hermitian(evolved_dm_exact), Hermitian(evolved_dm_alg)))
