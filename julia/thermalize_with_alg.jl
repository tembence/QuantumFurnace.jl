# using LinearAlgebra
# using SparseArrays
# using Random
# using Printf
# using ProgressMeter
# using Distributed
# using LaTeXStrings

# include("ofts.jl")
# include("hamiltonian.jl")
# include("qi_tools.jl")
# include("structs.jl")

# #TODO: Struct for Configs / Initialization of the thermalizations
# #TODO: Compare this to other thermalizations
# #TODO: Figure out how to best structure the code


# function thermalize_gaussian(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, evolved_dm::Matrix{ComplexF64},
#     num_energy_bits::Int64, filter_gauss_w::Function, transition_gauss::Function,
#     delta::Float64, mixing_time::Float64, beta::Float64)

#     # Energy labels
#     num_qubits = Int(log2(size(hamiltonian.data)[1]))
#     N = 2^(num_energy_bits)
#     N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
#     energy_labels = hamiltonian.nu_min * N_labels

#     num_liouv_steps = Int(round(mixing_time / delta, digits=0))
#     gibbs = Hermitian(gibbs_state_in_eigen(hamiltonian, beta))

#     # Filter Gaussian normalization for the jumps
#     filter_gauss_values = filter_gauss_w.(energy_labels)
#     filter_gauss_norm_sq = sum(filter_gauss_values.^2)
    
#     # Truncate energy -> 0.45 -> transition Gaussian truncation
#     energy_labels_045 = energy_labels[abs.(energy_labels) .<= 0.45]
#     transition_cutoff = 1e-4  #!
#     energy_cutoff_of_transition_gauss = find_zero(x -> transition_gauss(x) - transition_cutoff, 0)
#     energy_labels_sym = filter(x -> x <= energy_cutoff_of_transition_gauss && x > 0., energy_labels_045)
#     energy_labels_rest = - filter(x -> x > energy_cutoff_of_transition_gauss, energy_labels_045)
#     push!(energy_labels_rest, 0.0)

#     # Setup coherent part
#     # if with_coherent
#         # Time labels for coherent
#         # t0 = 2 * pi / (N * hamiltonian.nu_min)
#         # time_labels = t0 * N_labels

#         # atol = 1e-12
#         # b1 = compute_truncated_b1(time_labels, atol)
#         # b2 = compute_truncated_b2(time_labels, atol)
#         # @printf("t0: %e\n", t0)
#         # @printf("Number of b1 terms: %d\n", length(keys(b1)))
#         # @printf("Number of b2 terms: %d\n", length(keys(b2)))
#     # else
#     #     @printf("Not adding coherent terms! \n")
#     # end

#     distances_to_gibbs = [trace_distance_h(Hermitian(evolved_dm), gibbs)]
#     time_steps = [0.0:delta:(num_liouv_steps * delta);]
#     @showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps

#         # Sum of all jumps at once
#         for jump in jumps
#             # Coherent term
#             if with_coherent
#                 # coherent_term = coherent_term_from_timedomain(jump, hamiltonian, b1, b2, t0, beta)
#                 coherent_term = coherent_bohr_gauss(hamiltonian, jump, beta)
#                 evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)
    
#                 if step == 1 && jump == jumps[1]
#                     check_B_gauss(coherent_term, jump, hamiltonian, beta)
#                 end
#             end
    
#             dissipative_dm_part = zeros(ComplexF64, size(initial_dm))

#             # w <= cutoff, A(-w) = A(w)^\dagger
#             for w in energy_labels_sym
#                 oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
#                 oft_matrix_dag = oft_matrix'  # = A(-w)
#                 oft_dag_oft = oft_matrix_dag * oft_matrix
#                 oft_oft_dag = oft_matrix * oft_matrix_dag
                
#                 # Boy, the brackets in this multiline expression are important.
#                 dissipative_dm_part .+= transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
#                                     - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
#                 dissipative_dm_part .+=  transition(-w) * (oft_matrix_dag * evolved_dm * oft_matrix 
#                                     - 0.5 * (oft_oft_dag * evolved_dm + evolved_dm * oft_oft_dag))
#             end

#             # w < -cutoff && w = 0.0
#             for w in energy_labels_rest
#                 oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
#                 oft_matrix_dag = oft_matrix'
#                 oft_dag_oft = oft_matrix_dag * oft_matrix
                
#                 dissipative_dm_part .+=  transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
#                                     - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
#             end
#             evolved_dm .+= delta * dissipative_dm_part / filter_gauss_norm_sq
#         end

#         dist = trace_distance_h(Hermitian(evolved_dm), gibbs)
#         push!(distances_to_gibbs, dist)
#         # @printf("\nDistance to Gibbs: %f\n", dist)
#     end
#     return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
# end

# function thermalize_gaussian_nh(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, 
#     num_energy_bits::Int64, evolved_dm::Matrix{ComplexF64}, filter_gauss_w::Function, transition_gauss::Function,
#     delta::Float64, mixing_time::Float64, beta::Float64)

#     # Energy labels
#     num_qubits = Int(log2(size(hamiltonian.data)[1]))
#     N = 2^(num_energy_bits)
#     N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
#     energy_labels = hamiltonian.nu_min * N_labels

#     num_liouv_steps = Int(round(mixing_time / delta, digits=0))
#     gibbs = gibbs_state_in_eigen(hamiltonian, beta)

#     # Filter Gaussian normalization for the jumps
#     filter_gauss_values = filter_gauss_w.(energy_labels)
#     filter_gauss_norm_sq = sum(filter_gauss_values.^2)
    
#     # Truncate energy -> 0.45 -> transition Gaussian truncation
#     energy_labels_045 = energy_labels[abs.(energy_labels) .<= 0.45]
#     transition_cutoff = 1e-4 #!
#     energy_cutoff_of_transition_gauss = find_zero(x -> transition_gauss(x) - transition_cutoff, 0)
#     energy_labels = filter(x -> x <= energy_cutoff_of_transition_gauss, energy_labels_045)
#     @printf("Max energy label: %f\n", maximum(energy_labels))

#     # Setup coherent part
#     if with_coherent
#         # Time labels for coherent
#         t0 = 2 * pi / (N * hamiltonian.nu_min)
#         time_labels = t0 * N_labels

#         atol = 1e-12
#         b1 = compute_truncated_b1(time_labels, atol)
#         b2 = compute_truncated_b2(time_labels, atol)
#         @printf("t0: %e\n", t0)
#         @printf("Number of b1 terms: %d\n", length(keys(b1)))
#         @printf("Number of b2 terms: %d\n", length(keys(b2)))
#     else
#         @printf("Not adding coherent terms! \n")
#     end

#     distances_to_gibbs = [trace_distance_nh(evolved_dm, gibbs)]
#     @showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps
#         # Sum of all jumps at once
#         for jump in jumps
#             # Coherent part
#             if with_coherent
#                 coherent_term = coherent_term_time(jump, hamiltonian, b1, b2, t0, beta)
#                 evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)
    
#                 if step == 1 && jump == jumps[1]
#                     check_B_gauss(coherent_term, jump, hamiltonian, beta)
#                 end
#             end

#             # Dissipative part
#             dissipative_dm_part = zeros(ComplexF64, size(initial_dm))
#             for w in energy_labels
#                 oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
#                 oft_matrix_dag = oft_matrix'
#                 oft_dag_oft = oft_matrix_dag * oft_matrix
                
#                 dissipative_dm_part .+=  transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
#                                     - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
#             end

#             evolved_dm .+= delta * dissipative_dm_part / filter_gauss_norm_sq
            
#             dist = trace_distance_nh(evolved_dm, gibbs)
#             push!(distances_to_gibbs, dist)
#             # @printf("\nDistance to Gibbs: %f\n", dist)
#         end
#     end
#     return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
# end

# function thermalize_gaussian_random(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, 
#     num_energy_bits::Int64, evolved_dm::Matrix{ComplexF64}, filter_gauss_w::Function, transition_gauss::Function,
#     delta::Float64, mixing_time::Float64, beta::Float64)

#     # Energy labels
#     num_qubits = Int(log2(size(hamiltonian.data)[1]))
#     N = 2^(num_energy_bits)
#     N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
#     energy_labels = hamiltonian.nu_min * N_labels

#     num_liouv_steps = Int(round(mixing_time / delta, digits=0))
#     gibbs = gibbs_state_in_eigen(hamiltonian, beta)

#     # Filter Gaussian normalization for the jumps
#     filter_gauss_values = filter_gauss_w.(energy_labels)
#     filter_gauss_norm_sq = sum(filter_gauss_values.^2)
    
#     # Truncate energy -> 0.45 -> transition Gaussian truncation
#     energy_labels_045 = energy_labels[abs.(energy_labels) .<= 0.45]
#     transition_cutoff = 1e-4  #!
#     energy_cutoff_of_transition_gauss = find_zero(x -> transition_gauss(x) - transition_cutoff, 0)
#     energy_labels_sym = filter(x -> x <= energy_cutoff_of_transition_gauss && x > 0., energy_labels_045)
#     energy_labels_rest = - filter(x -> x > energy_cutoff_of_transition_gauss, energy_labels_045)
#     push!(energy_labels_rest, 0.0)

#     # Setup coherent part
#     if with_coherent
#         # Time labels for coherent
#         t0 = 2 * pi / (N * hamiltonian.nu_min)
#         time_labels = t0 * N_labels

#         atol = 1e-12
#         b1 = compute_truncated_b1(time_labels, atol)
#         b2 = compute_truncated_b2(time_labels, atol)
#         @printf("t0: %e\n", t0)
#         @printf("Number of b1 terms: %d\n", length(keys(b1)))
#         @printf("Number of b2 terms: %d\n", length(keys(b2)))
#     else
#         @printf("Not adding coherent terms! \n")
#     end

#     distances_to_gibbs = [trace_distance_nh(evolved_dm, gibbs)]
#     @showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps

#         # Pick a random jump for this step
#         jump = jumps[step]

#         # Coherent term
#         if with_coherent
#             coherent_term = coherent_term_time(jump, hamiltonian, b1, b2, t0, beta)
#             evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)

#             if step == 1 && jump == jumps[1]
#                 check_B_gauss(coherent_term, jump, hamiltonian, beta)
#             end
#         end

#         dissipative_dm_part = zeros(ComplexF64, size(initial_dm))
#         # w <= cutoff, A(-w) = A(w)^\dagger
#         for w in energy_labels_sym
#             oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
#             oft_matrix_dag = oft_matrix'  # = A(-w)
#             oft_dag_oft = oft_matrix_dag * oft_matrix
#             oft_oft_dag = oft_matrix * oft_matrix_dag
            
#             # Boy, the brackets in this multiline expression are important.
#             dissipative_dm_part .+= transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
#                                 - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
#             dissipative_dm_part .+=  transition(-w) * (oft_matrix_dag * evolved_dm * oft_matrix 
#                                 - 0.5 * (oft_oft_dag * evolved_dm + evolved_dm * oft_oft_dag))
#         end

#         # w < -cutoff && w = 0.0
#         for w in energy_labels_rest
#             oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
#             oft_matrix_dag = oft_matrix'
#             oft_dag_oft = oft_matrix_dag * oft_matrix
            
#             dissipative_dm_part .+=  transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
#                                 - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
#         end

#         evolved_dm .+= delta * dissipative_dm_part / filter_gauss_norm_sq
        
#         dist = trace_distance_nh(evolved_dm, gibbs)
#         push!(distances_to_gibbs, dist)
#         # @printf("\nDistance to Gibbs: %f\n", dist)
#     end
#     return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
# end

# function thermalize_gaussian_ideal_time(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, 
#     evolved_dm::Matrix{ComplexF64}, num_energy_bits::Int64, filter_gauss_t::Function, transition_gauss::Function, 
#     delta::Float64, mixing_time::Float64, beta::Float64)

#     num_qubits = Int(log2(size(hamiltonian.data)[1]))
#     N = 2^(num_energy_bits)
#     N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
#     energy_labels = hamiltonian.nu_min * N_labels
#     t0 = 2 * pi / (N * hamiltonian.nu_min)
#     time_labels = t0 * N_labels

#     filter_gauss_t_values = filter_gauss_t.(time_labels)  # exp.(- time_labels.^2 / beta^2)
#     filter_gauss_t_norm_sq = sum(filter_gauss_t_values.^2)

#     num_liouv_steps = Int(round(mixing_time / delta, digits=0))
#     gibbs = gibbs_state_in_eigen(hamiltonian, beta)
    
#     # Truncate energy -> 0.45 -> transition Gaussian truncation
#     energy_labels_045 = energy_labels[abs.(energy_labels) .<= 0.45]
#     transition_cutoff = 1e-4  #!
#     energy_cutoff_of_transition_gauss = find_zero(x -> transition_gauss(x) - transition_cutoff, 0)
#     energy_labels_sym = filter(x -> x <= energy_cutoff_of_transition_gauss && x > 0., energy_labels_045)
#     energy_labels_rest = - filter(x -> x > energy_cutoff_of_transition_gauss, energy_labels_045)
#     push!(energy_labels_rest, 0.0)

#     # Setup coherent part
#     if with_coherent
#         atol = 1e-12
#         b1 = compute_truncated_b1(time_labels, atol)
#         b2 = compute_truncated_b2(time_labels, atol)
#         @printf("t0: %e\n", t0)
#         @printf("Number of b1 terms: %d\n", length(keys(b1)))
#         @printf("Number of b2 terms: %d\n", length(keys(b2)))
#     else
#         @printf("Not adding coherent terms! \n")
#     end

#     distances_to_gibbs = [trace_distance_nh(evolved_dm, gibbs)]
#     time_steps = [0.0:delta:(num_liouv_steps * delta);]
#     @showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps

#         # Sum of all jumps at once
#         for jump in jumps
#             # Coherent term 
#             if with_coherent
#                 coherent_term = coherent_term_time(jump, hamiltonian, b1, b2, t0, beta)
#                 evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)
    
#                 if step == 1 && jump == jumps[1]
#                     check_B_gauss(coherent_term, jump, hamiltonian, beta)
#                 end
#             end
    
#             # w <= cutoff, A(-w) = A(w)^\dagger
#             dissipative_dm_part = zeros(ComplexF64, size(initial_dm))
#             for w in energy_labels_sym
#                 oft_matrix = explicit_oft_exact_db(jump, hamiltonian, w, time_labels, beta)
#                 oft_matrix_dag = oft_matrix'  # = A(-w)
#                 oft_dag_oft = oft_matrix_dag * oft_matrix
#                 oft_oft_dag = oft_matrix * oft_matrix_dag
                
#                 # Boy, the brackets in this multiline expression are important.
#                 dissipative_dm_part .+= transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
#                                     - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
#                 dissipative_dm_part .+=  transition(-w) * (oft_matrix_dag * evolved_dm * oft_matrix 
#                                     - 0.5 * (oft_oft_dag * evolved_dm + evolved_dm * oft_oft_dag))
#             end

#             # w < -cutoff && w = 0.0
#             for w in energy_labels_rest
#                 oft_matrix = explicit_oft_exact_db(jump, hamiltonian, w, time_labels, beta)
#                 oft_matrix_dag = oft_matrix'
#                 oft_dag_oft = oft_matrix_dag * oft_matrix
                
#                 dissipative_dm_part .+=  transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
#                                     - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
#             end

#             evolved_dm .+= delta * dissipative_dm_part / (filter_gauss_t_norm_sq * length(time_labels))
#         end  
#         dist = trace_distance_nh(evolved_dm, gibbs)
#         push!(distances_to_gibbs, dist)
#         # @printf("\nDistance to Gibbs: %f\n", dist)
#     end
#     return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
# end

# #TODO: Write it up s.t. the error is like for the algorithm, i.e. twos complement Trotter.
# function thermalize_gaussian_trotter(jumps::Vector{JumpOp}, hamiltonian::HamHam, trotter::TrottTrott, with_coherent::Bool, 
#     evolved_dm::Matrix{ComplexF64}, num_energy_bits::Int64, filter_gauss_t::Function, transition_gauss::Function, 
#     delta::Float64, mixing_time::Float64, beta::Float64)
#     """In Trotter basis"""

#     num_qubits = Int(log2(size(hamiltonian.data)[1]))
#     N = 2^(num_energy_bits)
#     N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
#     energy_labels = hamiltonian.nu_min * N_labels
#     time_labels = trotter.t0 * N_labels

#     filter_gauss_t_values = filter_gauss_t.(time_labels)  # exp.(- time_labels.^2 / beta^2)
#     filter_gauss_t_norm_sq = sum(filter_gauss_t_values.^2)

#     num_liouv_steps = Int(round(mixing_time / delta, digits=0))
#     gibbs_in_trotter = trotter.eigvecs' * gibbs_state(hamiltonian, beta) * trotter.eigvecs
#     evolved_dm = trotter.eigvecs' * initial_dm * trotter.eigvecs
    
#     # Truncate energy -> 0.45 -> transition Gaussian truncation
#     energy_labels_045 = energy_labels[abs.(energy_labels) .<= 0.45]
#     transition_cutoff = 1e-4  #!
#     energy_cutoff_of_transition_gauss = find_zero(x -> transition_gauss(x) - transition_cutoff, 0)
#     energy_labels_sym = filter(x -> x <= energy_cutoff_of_transition_gauss && x > 0., energy_labels_045)
#     energy_labels_rest = - filter(x -> x > energy_cutoff_of_transition_gauss, energy_labels_045)
#     push!(energy_labels_rest, 0.0)

#     # Setup coherent part
#     if with_coherent
#         atol = 1e-12
#         b1 = compute_truncated_b1(time_labels, atol)
#         b2 = compute_truncated_b2(time_labels, atol)
#         @printf("t0: %e\n", trotter.t0)
#         @printf("Number of b1 terms: %d\n", length(keys(b1)))
#         @printf("Number of b2 terms: %d\n", length(keys(b2)))
#     else
#         @printf("Not adding coherent terms! \n")
#     end

#     distances_to_gibbs = [trace_distance_nh(evolved_dm, gibbs_in_trotter)]
#     time_steps = [0.0:delta:(num_liouv_steps * delta);]
#     @showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps

#         # Sum of all jumps at once
#         for jump in jumps
#             # Coherent term 
#             if with_coherent
#                 coherent_term = coherent_term_trotter(jump, hamiltonian, trotter, b1, b2, beta)
#                 evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)
    
#                 if step == 1 && jump == jumps[1]
#                     check_B_gauss(coherent_term, jump, hamiltonian, beta)
#                 end
#             end
    
#             # w <= cutoff, A(-w) = A(w)^\dagger
#             dissipative_dm_part = zeros(ComplexF64, size(initial_dm))
#             for w in energy_labels_sym
#                 oft_matrix = trotter_oft(jump, trotter, w, time_labels, beta)
#                 oft_matrix_dag = oft_matrix'  # = A(-w)
#                 oft_dag_oft = oft_matrix_dag * oft_matrix
#                 oft_oft_dag = oft_matrix * oft_matrix_dag
                
#                 # Boy, the brackets in this multiline expression are important.
#                 dissipative_dm_part .+= transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
#                                     - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
#                 dissipative_dm_part .+=  transition(-w) * (oft_matrix_dag * evolved_dm * oft_matrix 
#                                     - 0.5 * (oft_oft_dag * evolved_dm + evolved_dm * oft_oft_dag))
#             end

#             # w < -cutoff && w = 0.0
#             for w in energy_labels_rest
#                 oft_matrix = trotter_oft(jump, trotter, w, time_labels, beta)
#                 oft_matrix_dag = oft_matrix'
#                 oft_dag_oft = oft_matrix_dag * oft_matrix
                
#                 dissipative_dm_part .+=  transition(w) * (oft_matrix * evolved_dm * oft_matrix_dag
#                                     - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
#             end

#             evolved_dm .+= delta * dissipative_dm_part / (filter_gauss_t_norm_sq * length(time_labels))
#         end  
#         dist = trace_distance_nh(evolved_dm, gibbs_in_trotter)
#         push!(distances_to_gibbs, dist)
#         # @printf("\nDistance to Gibbs: %f\n", dist)
#     end
#     return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
# end

# function thermalize_metro(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, evolved_dm::Matrix{ComplexF64},
#     num_energy_bits::Int64, filter_gauss_w::Function, transition_metro::Function,
#     eta::Float64, delta::Float64, mixing_time::Float64, beta::Float64)

#     num_qubits = Int(log2(size(hamiltonian.data)[1]))
#     N = 2^(num_energy_bits)
#     N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
#     energy_labels = hamiltonian.nu_min * N_labels

#     num_liouv_steps = Int(round(mixing_time / delta, digits=0))
#     gibbs = gibbs_state_in_eigen(hamiltonian, beta)

#     # Filter Gaussian normalization for the jumps
#     filter_gauss_values = filter_gauss_w.(energy_labels)
#     filter_gauss_norm_sq = sum(filter_gauss_values.^2)
    
#     # Truncate energy -> 0.45 -/-> no other truncation possible really
#     energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]
#     energy_labels_no_zero = energy_labels[energy_labels .!= 0.0]

#     # Setup coherent part
#     if with_coherent
#         # Time labels for coherent
#         t0 = 2 * pi / (N * hamiltonian.nu_min)
#         time_labels = t0 * N_labels

#         coherent_terms_atol = 1e-12
#         b1 = compute_truncated_b1(time_labels, coherent_terms_atol)
#         b2 = compute_truncated_b2_metro(time_labels, eta, coherent_terms_atol)
#         @printf("t0: %e\n", t0)
#         @printf("Number of b1 terms: %d\n", length(keys(b1)))
#         @printf("Number of b2 terms: %d\n", length(keys(b2)))
#     else
#         @printf("Not adding coherent terms! \n")
#     end

#     distances_to_gibbs = [trace_distance_nh(evolved_dm, gibbs)]
#     time_steps = [0.0:delta:(num_liouv_steps * delta);]
#     @showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps
#         # Sum of all jumps at once
#         for jump in jumps
#             # Coherent term
#             if with_coherent
#                 coherent_term = coherent_term_time(jump, hamiltonian, b1, b2, t0, beta)
#                 evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)
    
#                 if step == 1 && jump == jumps[1]
#                     check_B_metro(coherent_term, jump, hamiltonian, eta, beta)
#                 end
#             end
    
#             dissipative_dm_part = zeros(ComplexF64, size(initial_dm))

#             # w != 0, A(-w) = A(w)^\dagger
#             for w in energy_labels_no_zero
#                 oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
#                 oft_matrix_dag = oft_matrix'  # = A(-w)
#                 oft_dag_oft = oft_matrix_dag * oft_matrix
#                 oft_oft_dag = oft_matrix * oft_matrix_dag
                
#                 dissipative_dm_part .+= transition_metro(w) * (oft_matrix * evolved_dm * oft_matrix_dag
#                                     - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
#                 dissipative_dm_part .+=  transition_metro(-w) * (oft_matrix_dag * evolved_dm * oft_matrix 
#                                     - 0.5 * (oft_oft_dag * evolved_dm + evolved_dm * oft_oft_dag))
#             end

#             w = 0.0
#             oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
#             oft_matrix_dag = oft_matrix'
#             oft_dag_oft = oft_matrix_dag * oft_matrix

#             dissipative_dm_part .+=  transition_metro(w) * (oft_matrix * evolved_dm * oft_matrix_dag
#                                 - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))

#             evolved_dm .+= delta * dissipative_dm_part / filter_gauss_norm_sq
#         end   
#         dist = trace_distance_nh(evolved_dm, gibbs)
#         push!(distances_to_gibbs, dist)
#         # @printf("\nDistance to Gibbs: %f\n", dist)
#     end
#     return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
# end

# function thermalize_metro_trotter(jumps::Vector{JumpOp}, hamiltonian::HamHam, trotter::TrottTrott, with_coherent::Bool,
#     evolved_dm::Matrix{ComplexF64},num_energy_bits::Int64, filter_gauss_t::Function, transition_metro::Function,
#     eta::Float64, delta::Float64, mixing_time::Float64, beta::Float64)

#     num_qubits = Int(log2(size(hamiltonian.data)[1]))
#     N = 2^(num_energy_bits)
#     N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
#     energy_labels = hamiltonian.nu_min * N_labels
#     time_labels = trotter.t0 * N_labels

#     filter_gauss_t_values = filter_gauss_t.(time_labels)  # exp.(- time_labels.^2 / beta^2)
#     filter_gauss_t_norm_sq = sum(filter_gauss_t_values.^2)

#     num_liouv_steps = Int(round(mixing_time / delta, digits=0))
#     gibbs_in_trotter = trotter.eigvecs' * gibbs_state(hamiltonian, beta) * trotter.eigvecs
#     evolved_dm = trotter.eigvecs' * initial_dm * trotter.eigvecs
    
#     # Truncate energy -> 0.45 -/-> no other truncation possible really
#     energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]
#     energy_labels_no_zero = energy_labels[energy_labels .!= 0.0]

#     # Setup coherent part
#     if with_coherent
#         coherent_terms_atol = 1e-12
#         b1 = compute_truncated_b1(time_labels, coherent_terms_atol)
#         b2 = compute_truncated_b2_metro(time_labels, eta, coherent_terms_atol)
#         @printf("t0: %e\n", trotter.t0)
#         @printf("Number of b1 terms: %d\n", length(keys(b1)))
#         @printf("Number of b2 terms: %d\n", length(keys(b2)))
#     else
#         @printf("Not adding coherent terms! \n")
#     end

#     distances_to_gibbs = [trace_distance_nh(evolved_dm, gibbs_in_trotter)]
#     time_steps = [0.0:delta:(num_liouv_steps * delta);]
#     @showprogress dt=1 desc="Algorithm..." for step in 1:num_liouv_steps
#         # Sum of all jumps at once
#         for jump in jumps
#             # Coherent term
#             if with_coherent
#                 coherent_term = coherent_term_trotter(jump, hamiltonian, trotter, b1, b2, beta)
#                 evolved_dm .+= - im * delta * (coherent_term * evolved_dm - evolved_dm * coherent_term)
    
#                 if step == 1 && jump == jumps[1]
#                     check_B_metro(coherent_term, jump, hamiltonian, eta, beta)
#                 end
#             end
    
#             dissipative_dm_part = zeros(ComplexF64, size(initial_dm))

#             # w != 0, A(-w) = A(w)^\dagger
#             for w in energy_labels_no_zero
#                 oft_matrix = trotter_oft(jump, trotter, w, time_labels, beta)
#                 oft_matrix_dag = oft_matrix'  # = A(-w)
#                 oft_dag_oft = oft_matrix_dag * oft_matrix
#                 oft_oft_dag = oft_matrix * oft_matrix_dag
                
#                 dissipative_dm_part .+= transition_metro(w) * (oft_matrix * evolved_dm * oft_matrix_dag
#                                     - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))
#                 dissipative_dm_part .+=  transition_metro(-w) * (oft_matrix_dag * evolved_dm * oft_matrix 
#                                     - 0.5 * (oft_oft_dag * evolved_dm + evolved_dm * oft_oft_dag))
#             end

#             w = 0.0
#             oft_matrix = trotter_oft(jump, trotter, w, time_labels, beta)
#             oft_matrix_dag = oft_matrix'
#             oft_dag_oft = oft_matrix_dag * oft_matrix

#             dissipative_dm_part .+=  transition_metro(w) * (oft_matrix * evolved_dm * oft_matrix_dag
#                                 - 0.5 * (oft_dag_oft * evolved_dm + evolved_dm * oft_dag_oft))

#             evolved_dm .+= delta * dissipative_dm_part / (filter_gauss_t_norm_sq * length(time_labels))
#         end
#         dist = trace_distance_nh(evolved_dm, gibbs_in_trotter)
#         push!(distances_to_gibbs, dist)
#         # @printf("\nDistance to Gibbs: %f\n", dist)
#     end
#     return HotAlgorithmResults(evolved_dm, distances_to_gibbs, time_steps)
# end

# function liouvillian_delta_trajectory_trotter_gaussian_exact_db(jump::JumpOp, trotter::TrottTrott, 
#     coherent_term::Matrix{ComplexF64}, energy_labels::Vector{Float64}, initial_dm::Matrix{ComplexF64}, 
#     delta::Float64, sigma::Float64, beta::Float64)

#     # keep only energies in between -0.45 and 0.45 (physically impossible to get values outside of this range)
#     energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]

#     transition_gaussian(energy) = exp(-(beta * energy + 1)^2 / 2)
#     transition_weights = transition_gaussian.(energy_labels)

#     evolved_dm = - im * delta * (coherent_term * initial_dm - initial_dm * coherent_term)
#     for (i, w) in enumerate(energy_labels)
#         oft_matrix = trotter_oft(jump, trotter, w, time_labels, beta)
#         oft_matrix_dag = oft_matrix'
        
#         evolved_dm .+= delta * transition_weights[i] *
#                     (oft_matrix * initial_dm * oft_matrix_dag
#                     - 0.5 * oft_matrix_dag * oft_matrix * initial_dm
#                     - 0.5 * initial_dm * oft_matrix_dag * oft_matrix)
#     end

#     # Return in energy basis
#     return evolved_dm
# end

# # Trotter and alg match too well, 1 step per t0 is too good.
# function liouvillian_delta_trajectory_trotter(jump::JumpOp, trotter::TrottTrott,
#     energy_labels::Vector{Float64}, initial_dm::Matrix{ComplexF64}, delta::Float64, sigma::Float64, beta::Float64)
#     """Everything in Trotter basis, initial_dm too as input."""

#     boltzmann_factor(energy) = min(1, exp(-beta * energy))
    
#     # keep only energies in between -0.45 and 0.45
#     energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]
#     boltzmann_values = boltzmann_factor.(energy_labels)

#     #! Added enumerate, not debugged yet
#     evolved_dm = zeros(ComplexF64, size(initial_dm))
#     for (i, w) in enumerate(energy_labels)
#         oft_matrix = explicit_trotter_oft(jump, trotter, w, time_labels, sigma, beta)
#         oft_matrix_dag = oft_matrix'
        
#         evolved_dm .+= delta * boltzmann_values[i] *
#                     (oft_matrix * initial_dm * oft_matrix_dag
#                     - 0.5 * oft_matrix_dag * oft_matrix * initial_dm
#                     - 0.5 * initial_dm * oft_matrix_dag * oft_matrix)
#     end

#     # Return in Trotter basis
#     return evolved_dm
# end

# function liouvillian_delta_trajectory_metro_exact_db(jump::JumpOp, hamiltonian::HamHam, coherent_term::Matrix{ComplexF64},
#     energy_labels::Vector{Float64}, initial_dm::Matrix{ComplexF64}, delta::Float64, beta::Float64)

#     # keep only energies in between -0.45 and 0.45 (physically impossible to get values outside of this range)
#     energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]

#     transition_metropolis(energy) = exp(-beta * maximum([energy + 1/(2*beta), 0]))
#     transition_weights = transition_metropolis.(energy_labels)

#     # Is there any transiton weight that is smaller than 1e-14?
#     if any(transition_weights .< 1e-14)
#         @printf("Transition weights smaller than 1e-14: %s\n", transition_weights[transition_weights .< 1e-14])
#     end

#     evolved_dm = - im * delta * (coherent_term * initial_dm - initial_dm * coherent_term)
#     for (i, w) in enumerate(energy_labels)
#         oft_matrix = explicit_oft_exact_db(jump, hamiltonian, w, time_labels, beta)
#         oft_matrix_dag = oft_matrix'
        
#         evolved_dm .+= delta * transition_weights[i] *
#                     (oft_matrix * initial_dm * oft_matrix_dag
#                     - 0.5 * oft_matrix_dag * oft_matrix * initial_dm
#                     - 0.5 * initial_dm * oft_matrix_dag * oft_matrix)
#     end

#     # Return in energy basis
#     return evolved_dm
# end

# function liouvillian_delta_trajectory(jump::JumpOp, hamiltonian::HamHam,
#     energy_labels::Vector{Float64}, initial_dm::Matrix{ComplexF64}, delta::Float64, sigma::Float64, beta::Float64)
#     """Computes δ * L[rho_0]"""

#     Fw = exp.(- sigma^2 * (energy_labels).^2)
#     Fw_norm = sqrt(sum(Fw.^2))
#     boltzmann_factor(energy) = min(1, exp(-beta * energy))

#     # keep only energies in between -0.45 and 0.45
#     energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]

#     evolved_dm = zeros(ComplexF64, size(initial_dm))
#     for w in energy_labels
#         oft_matrix = entry_wise_oft(jump, w, hamiltonian, sigma, beta) / Fw_norm
#         oft_matrix_dag = oft_matrix'
        
#         evolved_dm .+= delta * boltzmann_factor(w) *
#                     (oft_matrix * initial_dm * oft_matrix_dag
#                     - 0.5 * oft_matrix_dag * oft_matrix * initial_dm
#                     - 0.5 * initial_dm * oft_matrix_dag * oft_matrix)
#     end
#     return evolved_dm
# end

# function full_liouvillian_step(jump::JumpOp, hamiltonian::HamHam, energy_labels::Vector{Float64},
#     initial_dm::Matrix{ComplexF64}, delta::Float64, sigma::Float64, beta::Float64)

#     num_qubits = Int(log2(size(hamiltonian.data)[1]))

#     initial_dm = Operator(b, initial_dm)
#     evolution_hamiltonian = Operator(b, spzeros(ComplexF64, 2^num_qubits, 2^num_qubits))
#     Fw = exp.(- sigma^2 * (energy_labels).^2)
#     Fw_norm = sqrt(sum(Fw.^2))
#     boltzmann_factor(energy) = min(1, exp(-beta * energy))
#     # All jumps
#     all_jumps = [Operator(b, sqrt(boltzmann_factor(w)) * 
#                                 entry_wise_oft(jump, w, hamiltonian, sigma, beta) / Fw_norm) for w in energy_labels]

#     tout, evolved_dms = timeevolution.master([0.0, delta], initial_dm, evolution_hamiltonian, all_jumps) 
#     # Trace is already = 1
#     return evolved_dms[2].data
# end
