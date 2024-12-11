using LinearAlgebra
using SparseArrays
using Random
using Printf
using ProgressMeter
using Distributed
using BenchmarkTools
using Roots

include("hamiltonian_tools.jl")
include("jump_op_tools.jl")
include("trotter.jl")
include("qi_tools.jl")
include("thermalizing_tools.jl")
include("coherent.jl")
include("structs.jl")

#TODO: Write transition_bohr_gibbsed_vec, transition_bohr_vec with new bohr way, and delete the rest

function construct_liouvillian_gauss(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, 
    num_energy_bits::Int64, filter_gauss_w::Function, transition_gauss::Function, beta::Float64)
    """Constructs the vectorized Davies Liouvillian that thermalizes the system. This function works for jumps that are
    Hermitian, because we use a symmetry of them for the energy labels the reduce computation. 
    Energy labels are also truncated for faster computation.
    """

    # Energy labels
    num_qubits = Int(log2(size(hamiltonian.data)[1]))
    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = hamiltonian.nu_min * N_labels

    # Square root of transition 
    sqrt_transition_gauss(w) = sqrt(transition_gauss(w))

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
        t0 = 2 * pi / (N * hamiltonian.nu_min)
        time_labels = t0 * N_labels

        coherent_terms_atol = 1e-12
        b1 = compute_truncated_b1(time_labels, coherent_terms_atol)
        b2 = compute_truncated_b2(time_labels, coherent_terms_atol)
        @printf("t0: %e\n", t0)
        @printf("Number of b1 terms: %d\n", length(keys(b1)))
        @printf("Number of b2 terms: %d\n", length(keys(b2)))
    else
        @printf("Not adding coherent terms! \n")
    end

    liouv = zeros(ComplexF64, 4^num_qubits, 4^num_qubits)
    @showprogress dt=1 desc="Liouvillian..." for jump in jumps
        # Coherent part
        if with_coherent
            # coherent_term = coherent_term_from_timedomain(jump, hamiltonian, b1, b2, t0, beta)
            coherent_term = coherent_gaussian_bohr(hamiltonian, jump, beta)
            # coherent_term = coherent_term_timedomain_integrated(jump, hamiltonian, beta)
            liouv .+= vectorize_liouvillian_coherent(coherent_term)
        end

        # Dissipative part
        # w <= cutoff, A(-w) = A(w)^\dagger
        liouv_diss = zeros(ComplexF64, 4^num_qubits, 4^num_qubits)
        for w in energy_labels_sym
            oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
            liouv_diss .+= vectorize_liouvillian_diss([sqrt_transition_gauss(w) * oft_matrix, 
                                                    sqrt_transition_gauss(-w) * oft_matrix'])
        end
    
        # w < -cutoff && w = 0.0
        for w in energy_labels_rest
            oft_matrix = sqrt_transition_gauss(w) * entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
            liouv_diss .+= vectorize_liouvillian_diss([oft_matrix])
        end
        # Very important normalization
        liouv .+= liouv_diss / filter_gauss_norm_sq
    end
    
    return liouv
end

function construct_liouvillian_gauss_ideal_time(jumps::Vector{JumpOp}, hamiltonian::HamHam, 
    with_coherent::Bool, num_energy_bits::Int64, filter_gauss_t::Function, transition_gauss::Function, beta::Float64)
    """In Trotter basis"""

    # Energy labels
    num_qubits = Int(log2(size(hamiltonian.data)[1]))
    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = hamiltonian.nu_min * N_labels
    t0 = 2 * pi / (N * hamiltonian.nu_min)
    time_labels = t0 * N_labels

    # Square root of transition 
    sqrt_transition_gauss(w) = sqrt(transition_gauss(w))

    # Filter Gaussian normalization for the jumps
    filter_gauss_t_values = filter_gauss_t.(time_labels)  # exp.(- time_labels.^2 / beta^2)
    filter_gauss_t_norm_sq = sum(filter_gauss_t_values.^2)

    # Truncate energy -> 0.45 -> transition Gaussian truncation
    energy_labels_045 = energy_labels[abs.(energy_labels) .<= 0.45]
    transition_cutoff = 1e-4  #!
    energy_cutoff_of_transition_gauss = find_zero(x -> transition_gauss(x) - transition_cutoff, 0)
    energy_labels_sym = filter(x -> x <= energy_cutoff_of_transition_gauss && x > 0., energy_labels_045)
    energy_labels_rest = - filter(x -> x > energy_cutoff_of_transition_gauss, energy_labels_045)
    push!(energy_labels_rest, 0.0)

    # Setup coherent part
    if with_coherent
        coherent_terms_atol = 1e-12
        b1 = compute_truncated_b1(time_labels, coherent_terms_atol)
        b2 = compute_truncated_b2(time_labels, coherent_terms_atol)
        @printf("t0: %e\n", t0)
        @printf("Number of b1 terms: %d\n", length(keys(b1)))
        @printf("Number of b2 terms: %d\n", length(keys(b2)))
    else
        @printf("Not adding coherent terms! \n")
    end

    liouv = zeros(ComplexF64, 4^num_qubits, 4^num_qubits)
    @showprogress dt=1 desc="Liouvillian..." for jump in jumps
        # Coherent part
        if with_coherent
            coherent_term = coherent_term_from_timedomain(jump, hamiltonian, b1, b2, t0, beta)
            # coherent_term = coherent_term_timedomain_integrated(jump, hamiltonian, beta)
            liouv .+= vectorize_liouvillian_coherent(coherent_term)
        end

        # Dissipative part
        # w <= cutoff, A(-w) = A(w)^\dagger
        liouv_diss = zeros(ComplexF64, 4^num_qubits, 4^num_qubits)
        for w in energy_labels_sym
            oft_matrix = explicit_oft_exact_db(jump, hamiltonian, w, time_labels, beta)
            liouv_diss .+= vectorize_liouvillian_diss([sqrt_transition_gauss(w) * oft_matrix, 
                                                    sqrt_transition_gauss(-w) * oft_matrix'])
        end
    
        # w < -cutoff && w = 0.0
        for w in energy_labels_rest
            oft_matrix = sqrt_transition_gauss(w) * explicit_oft_exact_db(jump, hamiltonian, w, time_labels, beta)
            liouv_diss .+= vectorize_liouvillian_diss([oft_matrix])
        end
        # Very important normalization
        liouv .+= liouv_diss / (filter_gauss_t_norm_sq * length(time_labels))
    end
    
    return liouv
end

function construct_liouvillian_gauss_trotter(jumps::Vector{JumpOp}, hamiltonian::HamHam, trotter::TrottTrott, 
    with_coherent::Bool, num_energy_bits::Int64, filter_gauss_t::Function, transition_gauss::Function, beta::Float64)
    """In Trotter basis"""

    # Energy labels
    num_qubits = Int(log2(size(hamiltonian.data)[1]))
    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = hamiltonian.nu_min * N_labels
    time_labels = trotter.t0 * N_labels

    # Square root of transition 
    sqrt_transition_gauss(w) = sqrt(transition_gauss(w))

    # Filter Gaussian normalization for the jumps
    filter_gauss_t_values = filter_gauss_t.(time_labels)  # exp.(- time_labels.^2 / beta^2)
    filter_gauss_t_norm_sq = sum(filter_gauss_t_values.^2)

    # Truncate energy -> 0.45 -> transition Gaussian truncation
    energy_labels_045 = energy_labels[abs.(energy_labels) .<= 0.45]
    transition_cutoff = 1e-4  #!
    energy_cutoff_of_transition_gauss = find_zero(x -> transition_gauss(x) - transition_cutoff, 0)
    energy_labels_sym = filter(x -> x <= energy_cutoff_of_transition_gauss && x > 0., energy_labels_045)
    energy_labels_rest = - filter(x -> x > energy_cutoff_of_transition_gauss, energy_labels_045)
    push!(energy_labels_rest, 0.0)

    # Setup coherent part
    if with_coherent
        coherent_terms_atol = 1e-12
        b1 = compute_truncated_b1(time_labels, coherent_terms_atol)
        b2 = compute_truncated_b2(time_labels, coherent_terms_atol)
        @printf("t0: %e\n", trotter.t0)
        @printf("Number of b1 terms: %d\n", length(keys(b1)))
        @printf("Number of b2 terms: %d\n", length(keys(b2)))
    else
        @printf("Not adding coherent terms! \n")
    end

    liouv = zeros(ComplexF64, 4^num_qubits, 4^num_qubits)
    @showprogress dt=1 desc="Liouvillian..." for jump in jumps
        # Coherent part
        if with_coherent
            coherent_term = coherent_term_trotter(jump, hamiltonian, trotter, b1, b2, beta)
            # coherent_term = coherent_term_timedomain_integrated(jump, hamiltonian, beta)
            liouv .+= vectorize_liouvillian_coherent(coherent_term)
        end

        # Dissipative part
        # w <= cutoff, A(-w) = A(w)^\dagger
        liouv_diss = zeros(ComplexF64, 4^num_qubits, 4^num_qubits)
        for w in energy_labels_sym
            oft_matrix = trotter_oft(jump, trotter, w, time_labels, beta)
            liouv_diss .+= vectorize_liouvillian_diss([sqrt_transition_gauss(w) * oft_matrix, 
                                                    sqrt_transition_gauss(-w) * oft_matrix'])
        end
    
        # w < -cutoff && w = 0.0
        for w in energy_labels_rest
            oft_matrix = sqrt_transition_gauss(w) * trotter_oft(jump, trotter, w, time_labels, beta)
            liouv_diss .+= vectorize_liouvillian_diss([oft_matrix])
        end
        liouv .+= liouv_diss / (filter_gauss_t_norm_sq * length(time_labels))
    end
    
    # Very important normalization
    return liouv
end

function construct_liouvillian_metro(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, 
    num_energy_bits::Int64, filter_gauss_w::Function, transition_metro::Function, eta::Float64, beta::Float64)

    # Energy labels
    num_qubits = Int(log2(size(hamiltonian.data)[1]))
    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = hamiltonian.nu_min * N_labels

    # Square root of transition 
    sqrt_transition_metro(w) = sqrt(transition_metro(w))

    # Filter Gaussian normalization for the jumps
    filter_gauss_values = filter_gauss_w.(energy_labels)
    filter_gauss_norm_sq = sum(filter_gauss_values.^2)

    # Truncate energy -> 0.45 -> transition Gaussian truncation
    energy_labels = energy_labels[abs.(energy_labels) .<= 0.45]
    energy_labels_no_zero = energy_labels[energy_labels .!= 0.0]

    # Setup coherent part
    if with_coherent
        # Time labels for coherent
        t0 = 2 * pi / (N * hamiltonian.nu_min)
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

    liouv = zeros(ComplexF64, 4^num_qubits, 4^num_qubits)
    @showprogress dt=1 desc="Liouvillian..." for jump in jumps
        # Coherent part
        if with_coherent
            coherent_term = coherent_term_from_timedomain(jump, hamiltonian, b1, b2, t0, beta)
            # coherent_term = coherent_term_timedomain_integrated(jump, hamiltonian, beta)
            liouv .+= vectorize_liouvillian_coherent(coherent_term)
        end

        # Disiipative part
        # w != 0, A(-w) = A(w)^\dagger
        liouv_diss = zeros(ComplexF64, 4^num_qubits, 4^num_qubits)
        for w in energy_labels_no_zero
            oft_matrix = entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
            liouv_diss .+= vectorize_liouvillian_diss([sqrt_transition_metro(w) * oft_matrix, 
                                                    sqrt_transition_metro(-w) * oft_matrix'])
        end
    
        w = 0.0
        oft_matrix = sqrt_transition_metro(w) * entry_wise_oft_exact_db(jump, w, hamiltonian, beta)
        liouv_diss .+= vectorize_liouvillian_diss([oft_matrix])

        liouv .+= liouv_diss / filter_gauss_norm_sq
    end
    # Very important normalization
    return liouv
end

function construct_liouvillian_metro_trotter(jumps::Vector{JumpOp}, hamiltonian::HamHam, trotter::TrottTrott, 
    with_coherent::Bool, num_energy_bits::Int64, filter_gauss_t::Function, transition_metro::Function, eta::Float64, beta::Float64)
    """In Trotter basis"""

    # Energy labels
    num_qubits = Int(log2(size(hamiltonian.data)[1]))
    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = hamiltonian.nu_min * N_labels
    time_labels = trotter.t0 * N_labels

    # Square root of transition 
    sqrt_transition_metro(w) = sqrt(transition_metro(w))

    # Filter Gaussian normalization for the jumps
    filter_gauss_t_values = filter_gauss_t.(time_labels)  # exp.(- time_labels.^2 / beta^2)
    filter_gauss_t_norm_sq = sum(filter_gauss_t_values.^2)

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

    liouv = zeros(ComplexF64, 4^num_qubits, 4^num_qubits)
    @showprogress dt=1 desc="Liouvillian..." for jump in jumps
        # Coherent part
        if with_coherent
            coherent_term = coherent_term_trotter(jump, hamiltonian, trotter, b1, b2, beta)
            # coherent_term = coherent_term_timedomain_integrated(jump, hamiltonian, beta)
            liouv .+= vectorize_liouvillian_coherent(coherent_term)
        end

        # Dissipative part
        # w != 0, A(-w) = A(w)^\dagger
        liouv_diss = zeros(ComplexF64, 4^num_qubits, 4^num_qubits)
        for w in energy_labels_no_zero
            oft_matrix = trotter_oft(jump, trotter, w, time_labels, beta)
            liouv_diss .+= vectorize_liouvillian_diss([sqrt_transition_metro(w) * oft_matrix, 
                                                    sqrt_transition_metro(-w) * oft_matrix'])
        end
    
        w = 0.0
        oft_matrix = sqrt_transition_metro(w) * trotter_oft(jump, trotter, w, time_labels, beta)
        liouv_diss .+= vectorize_liouvillian_diss([oft_matrix])

        liouv .+= liouv_diss / (filter_gauss_t_norm_sq * length(time_labels))
    end
    
    # Very important normalization
    return liouv
end

function construct_liouvillian_nh(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, 
    num_energy_bits::Int64, filter_gauss_w::Function, transition_gauss::Function, beta::Float64)

    # Energy labels
    num_qubits = Int(log2(size(hamiltonian.data)[1]))
    N = 2^(num_energy_bits)
    N_labels = [0:1:Int(N/2)-1; -Int(N/2):1:-1]
    energy_labels = hamiltonian.nu_min * N_labels

    # Square root of transition
    sqrt_transition_gauss(w) = sqrt(transition_gauss(w))

    # Filter Gaussian normalization for the jumps
    filter_gauss_values = filter_gauss_w.(energy_labels)
    filter_gauss_norm_sq = sum(filter_gauss_values.^2)

    # Truncate energy -> 0.45 -> transition Gaussian truncation
    energy_labels_045 = energy_labels[abs.(energy_labels) .<= 0.45]
    transition_cutoff = 1e-4  #!
    energy_cutoff_of_transition_gauss = find_zero(x -> transition_gauss(x) - transition_cutoff, 0)
    energy_labels = filter(x -> x <= energy_cutoff_of_transition_gauss, energy_labels_045)

    # Setup coherent part
    if with_coherent
        # Time labels for coherent
        t0 = 2 * pi / (N * hamiltonian.nu_min)
        time_labels = t0 * N_labels

        atol = 1e-12
        b1 = compute_truncated_b1(time_labels, atol)
        b2 = compute_truncated_b2(time_labels, atol)
        @printf("Number of b1 terms: %d\n", length(keys(b1)))
        @printf("Number of b2 terms: %d\n", length(keys(b2)))
    else
        @printf("Not adding coherent terms! \n")
    end

    liouv = zeros(ComplexF64, 4^num_qubits, 4^num_qubits)
    @showprogress dt=1 desc="Liouvillian..." for jump in jumps
    
        # Coherent part
        if with_coherent
            coherent_term = coherent_term_from_timedomain(jump, hamiltonian, b1, b2, t0, beta)
            # coherent_term = coherent_term_timedomain_integrated(jump, hamiltonian, beta)
            liouv .+= vectorize_liouvillian_coherent(coherent_term)
        end

        # Disiipative part
        for w in energy_labels
            oft_matrix = sqrt_transition_gauss(w) * entry_wise_oft_exact_db(jump, w, hamiltonian, beta)

            #* Liouv
            liouv .+= vectorize_liouvillian_diss([oft_matrix])
        end
    end
    # Very important normalization
    return liouv / filter_gauss_norm_sq
end

function construct_liouvillian_time_h(jumps::Vector{Matrix{ComplexF64}}, with_coherent::Bool, energy_labels::Vector{Float64},
    filter_gauss_t::Function, transition_gauss::Function, beta::Float64)
end

function mixing_time_bound(liouvillian::LiouvLiouv, trdist_epsilon::Float64)
    """Involves eigen(), so it can be slow for large Liouvillians. But the Krylov methods were too unstable."""

    eigvals_liouv = eigvals(liouvillian.data)
    liouvillian.spectral_gap = real(-eigvals_liouv[end - 1])

    fixed_point_vec = nullspace(liouvillian.data, atol=1e-12)[:, 1]
    dm_dim = Int(sqrt(length(fixed_point_vec)))
    liouvillian.steady_state = reshape(fixed_point_vec, (dm_dim, dm_dim))
    liouvillian.steady_state /= tr(liouvillian.steady_state)
    min_fixed_point_eigval = real(eigvals(liouvillian.steady_state)[1])

    # Loose upperbound for mixing time
    liouvillian.mixing_time_bound = log(1 / (trdist_epsilon * min_fixed_point_eigval)) / liouvillian.spectral_gap
end