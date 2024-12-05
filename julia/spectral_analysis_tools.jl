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

function transition_bohr_gibbsed(jumps::Vector{JumpOp}, hamiltonian::HamHam, dm::Matrix{ComplexF64}, 
    beta::Float64)
    """This is the generator with an input density matrix, but not the evolution."""

    dim = size(hamiltonian.data, 1)
    alpha(nu_1, nu_2) = exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2 / 16) * exp(-beta^2 * (nu_1 - nu_2)^2 / 8) / sqrt(8)
    # alpha^* = alpha
    
    T_on_dm = zeros(ComplexF64, dim, dim)  # Vectorized transition part of the Liouvillian
    for jump in jumps
        jump_diag = diag(jump.in_eigenbasis)
        for j in 1:dim
            for k in 1:dim
                for i in 1:dim
                    A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                    A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                    nu_1 = hamiltonian.bohr_freqs[i, j]
                    nu_2 = hamiltonian.bohr_freqs[i, k]
                    @printf("---For j, k, i: %d, %d, %d\n", j, k, i)
                    @printf("nu_1: %f, nu_2: %f\n", nu_1, nu_2)
                    check_alpha_skew_symmetry(alpha, nu_1, nu_2, beta)

                    # alpha A_nu_1 (.) A_nu_2^\dagger
                    if nu_1 != 0.0
                        A_nu_1[i, j] = jump.in_eigenbasis[i, j]         # A_nu_1
                    else
                        A_nu_1 .= spdiagm(0 => jump_diag)
                    end
                    if nu_2 != 0.0
                        A_nu_2[k, i] = conj(jump.in_eigenbasis[i, k])   # A_nu_2^\dagger
                    else  #! Want a check where the conj matters to see how fucking correct this is:
                        A_nu_2 .= conj.(spdiagm(0 => jump_diag))
                    end

                    @printf("A_nu_1\n")
                    display(A_nu_1)
                    @printf("A_nu_2_dagger\n")
                    display(A_nu_2)

                    #! Gibbsing it
                    A_nu_1 = gibbs^(-0.5) * A_nu_1 * gibbs^(0.5)
                    A_nu_2 = gibbs^(0.5) * A_nu_2 * gibbs^(-0.5)
                    @printf("GIBBSED\n")
                    @printf("A_nu_1\n")
                    display(A_nu_1)
                    @printf("A_nu_2_dagger\n")
                    display(A_nu_2)

                    T_on_dm_temp = alpha(nu_1, nu_2) * A_nu_1 * dm * A_nu_2
                    @printf("T_on_dm_temp\n")
                    display(T_on_dm_temp)
                    T_on_dm .+= T_on_dm_temp
                end
            end
        end
    end
    return T_on_dm
end

function transition_bohr(jumps::Vector{JumpOp}, hamiltonian::HamHam, dm::Matrix{ComplexF64}, 
    beta::Float64; adjoint::Bool=false)
    """This is the generator with an input density matrix, but not the evolution."""

    if adjoint
        @printf("Adjoint transition part\n")
    else
        @printf("Transition part\n")
    end

    dim = size(hamiltonian.data, 1)
    alpha(nu_1, nu_2) = exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2 / 16) * exp(-beta^2 * (nu_1 - nu_2)^2 / 8) / sqrt(8)
    # alpha^* = alpha
    
    T_on_dm = zeros(ComplexF64, dim, dim)  # Vectorized transition part of the Liouvillian
    for jump in jumps
        jump_diag = diag(jump.in_eigenbasis)
        for j in 1:dim
            for k in 1:dim
                for i in 1:dim
                    A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                    A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                    nu_1 = hamiltonian.bohr_freqs[i, j]
                    nu_2 = hamiltonian.bohr_freqs[i, k]
                    @printf("---For j, k, i: %d, %d, %d\n", j, k, i)
                    @printf("nu_1: %f, nu_2: %f\n", nu_1, nu_2)
                    check_alpha_skew_symmetry(alpha, nu_1, nu_2, beta)

                    if adjoint  # alpha^* A_nu_1^\dagger (.) A_nu_2
                        if nu_1 != 0.0
                            A_nu_1[j, i] = conj(jump.in_eigenbasis[i, j])  # A_nu_1^\dagger
                        else
                            A_nu_1 .= conj.(spdiagm(0 => jump_diag))
                        end

                        if nu_2 != 0.0
                            A_nu_2[i, k] = jump.in_eigenbasis[i, k]        # A_nu_2
                        else
                            A_nu_2 .= spdiagm(0 => jump_diag)
                        end
                        
                        # Their adjoint
                        # A_nu_1[i, j] = jump.in_eigenbasis[i, j]         # A_nu_1
                        # A_nu_2[k, i] = conj(jump.in_eigenbasis[i, k])   # A_nu_2^\dagger

                        @printf("A_nu_1_dagger\n")
                        display(A_nu_1)
                        @printf("A_nu_2\n")
                        display(A_nu_2)
                    else  # alpha A_nu_1 (.) A_nu_2^\dagger
                        if nu_1 != 0.0
                            A_nu_1[i, j] = jump.in_eigenbasis[i, j]         # A_nu_1
                        else
                            A_nu_1 .= spdiagm(0 => jump_diag)
                        end

                        if nu_2 != 0.0
                            A_nu_2[k, i] = conj(jump.in_eigenbasis[i, k])   # A_nu_2^\dagger
                        else
                            A_nu_2 .= conj.(spdiagm(0 => jump_diag))
                        end

                        @printf("A_nu_1\n")
                        display(A_nu_1)
                        @printf("A_nu_2_dagger\n")
                        display(A_nu_2)
                    end
                    T_on_dm_temp = alpha(nu_1, nu_2) * A_nu_1 * dm * A_nu_2
                    @printf("T_on_dm_temp\n")
                    display(T_on_dm_temp)
                    T_on_dm .+= T_on_dm_temp
                end
            end
        end
    end
    return T_on_dm
end


function transition_bohr_gibbsed_vec(jumps::Vector{JumpOp}, hamiltonian::HamHam, beta::Float64)
    """adjoint=true, means it prepares the adjoint transition part wrt to the HS inner product."""

    dim = size(hamiltonian.data, 1)

    alpha(nu_1, nu_2) = exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2 / 16) * exp(-beta^2 * (nu_1 - nu_2)^2 / 8) / sqrt(8)

    gibbs = gibbs_state_in_eigen(hamiltonian, beta)
    
    T = zeros(ComplexF64, dim^2, dim^2)  # Vectorized transition part of the Liouvillian
    all_the_nu1s = []
    all_the_nu2s = []
    for jump in jumps
        jump_diag = diag(jump.in_eigenbasis)
        for j in 1:dim
            for k in 1:dim
                for i in 1:dim
                    A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                    A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                    nu_1 = hamiltonian.bohr_freqs[i, j]
                    nu_2 = hamiltonian.bohr_freqs[i, k]
                    check_alpha_skew_symmetry(alpha, nu_1, nu_2, beta)

                    if nu_1 != 0.0
                        A_nu_1[i, j] = jump.in_eigenbasis[i, j]         # A_nu_1
                    else
                        A_nu_1 .= spdiagm(0 => jump_diag)
                    end

                    if nu_2 != 0.0
                        A_nu_2[k, i] = conj(jump.in_eigenbasis[i, k])   # A_nu_2^\dagger
                    else
                        A_nu_2 .= conj.(spdiagm(0 => jump_diag))
                    end
                    
                    # Testing Gibbsing 
                    # @printf("Is Gibbsing A_nu1 what we expect?: %s\n", norm(A_nu_1*exp(beta * nu_1 / 2) - gibbs^(-0.5) * A_nu_1 * gibbs^(0.5))< 1e-15)
                    # @printf("Is Gibbsing A_nu2 what we expect?: %s\n", norm(A_nu_2*exp(beta * nu_2 / 2) - gibbs^(0.5) * A_nu_2 * gibbs^(-0.5))< 1e-15)
                    # Testin skew symmetry
                    # @printf("Is alpha skew symmetric?: %s\n", norm(alpha(nu_1, nu_2) * exp(beta*(nu_1 + nu_2)/2) - alpha(-nu_2, -nu_1)) < 1e-15)
                    # Testing A dagger symmetry
                    # A_nu_1_dag::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                    # A_nu_1_dag[j, i] = conj(jump.in_eigenbasis[i, j])
                    # A_dag_minus_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                    # A_dag_minus_nu_1[j, i] = adjoint(jump.in_eigenbasis)[j, i]
                    # @printf("Is A_nu_1 dagger what we expect?: %s\n", norm(A_nu_1_dag - A_dag_minus_nu_1) < 1e-15)
                    # all jumps had hermitian adjoint pairs.
                    # Checkking all the nus present
                    # push!(all_the_nu1s, nu_1)
                    # push!(all_the_nu2s, nu_2)
                    # alpha^* symmetry
                    # @printf("Is alpha^* what we expect?: %s\n", norm(conj(alpha(nu_1, nu_2)) - alpha(nu_2, nu_1)) < 1e-15)

                    #! Gibbsing it
                    A_nu_1 = gibbs^(-0.5) * A_nu_1 * gibbs^(0.5)
                    A_nu_2 = gibbs^(0.5) * A_nu_2 * gibbs^(-0.5)

                    # Vectorized
                    # T .+= alpha(nu_1, nu_2) * kron(transpose(A_nu_2), A_nu_1)
                    #! Vectorized Watrous
                    T .+= alpha(nu_1, nu_2) * kron(A_nu_1, transpose(A_nu_2))
                end
            end
        end
    end
    # Did each nu have a negative pair in the set?
    return T
end


function transition_bohr_vec(jumps::Vector{JumpOp}, hamiltonian::HamHam, beta::Float64; adjoint::Bool=false)
    """adjoint=true, means it prepares the adjoint transition part wrt to the HS inner product."""

    if adjoint
        @printf("Adjoint transition part\n")
    else
        @printf("Transition part\n")
    end

    dim = size(hamiltonian.data, 1)

    alpha(nu_1, nu_2) = exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2 / 16) * exp(-beta^2 * (nu_1 - nu_2)^2 / 8) / sqrt(8)
    # alpha^* = alpha
    
    T = zeros(ComplexF64, dim^2, dim^2)  # Vectorized transition part of the Liouvillian
    for jump in jumps
        jump_diag = diag(jump.in_eigenbasis)
        for j in 1:dim
            for k in 1:dim
                for i in 1:dim
                    A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                    A_nu_2::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                    nu_1 = hamiltonian.bohr_freqs[i, j]
                    nu_2 = hamiltonian.bohr_freqs[i, k]

                    if adjoint  # alpha^* A_nu_1^\dagger (.) A_nu_2
                        if nu_1 != 0.0
                            A_nu_1[j, i] = conj(jump.in_eigenbasis[i, j])  # A_nu_1^\dagger
                        else
                            A_nu_1 .= conj.(spdiagm(0 => jump_diag))
                        end

                        if nu_2 != 0.0
                            A_nu_2[i, k] = jump.in_eigenbasis[i, k]        # A_nu_2
                        else
                            A_nu_2 .= spdiagm(0 => jump_diag)
                        end
                        # Their adjoint
                        # A_nu_1[i, j] = jump.in_eigenbasis[i, j]         # A_nu_1
                        # A_nu_2[k, i] = conj(jump.in_eigenbasis[i, k])   # A_nu_2^\dagger
                    else  # alpha A_nu_1 (.) A_nu_2^\dagger
                        if nu_1 != 0.0
                            A_nu_1[i, j] = jump.in_eigenbasis[i, j]         # A_nu_1
                        else
                            A_nu_1 .= spdiagm(0 => jump_diag)
                        end

                        if nu_2 != 0.0
                            A_nu_2[k, i] = conj(jump.in_eigenbasis[i, k])   # A_nu_2^\dagger
                        else
                            A_nu_2 .= conj.(spdiagm(0 => jump_diag))
                        end
                    end

                    #! Vectorized Watrous
                    T .+= alpha(nu_1, nu_2) * kron(A_nu_1, transpose(A_nu_2))
                    # T .+= alpha(nu_1, nu_2) * kron(A_nu_2, transpose(A_nu_1))  # Theirs
                end
            end
        end
    end
    return T
end

function construct_liouvillian_gauss_bohr(jumps::Vector{JumpOp}, hamiltonian::HamHam, with_coherent::Bool, beta::Float64)

    dim = size(hamiltonian.data, 1)
    alpha(nu_1, nu_2) = exp(-beta^2 * (nu_1 + nu_2 + 2/beta)^2 / 16) * exp(-beta^2 * (nu_1 - nu_2)^2 / 8) / sqrt(8)

    liouv = zeros(ComplexF64, dim^2, dim^2)
    @showprogress dt=1 desc="Liouvillian (Bohr)..." for jump in jumps

        # Coherent part
        if with_coherent
            coherent_term = coherent_gaussian_bohr_slow(hamiltonian, jump, beta)
            liouv .+= vectorize_liouvillian_coherent(coherent_term)
        end

        # Dissipative part
        #TODO: if slow, rearrange for loops, put j low and sum up all A_nu_1s while keeping nu2 fix, and vec the sum.
        jump_diag = diag(jump.in_eigenbasis)
        for j in 1:dim
            for k in 1:dim
                for i in 1:dim
                    A_nu_1::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                    A_nu_2_dagger::SparseMatrixCSC{ComplexF64} = spzeros(dim, dim)
                    nu_1 = hamiltonian.bohr_freqs[i, j]
                    nu_2 = hamiltonian.bohr_freqs[i, k]
                    check_alpha_skew_symmetry(alpha, nu_1, nu_2, beta)

                    #!!!
                    if nu_1 != 0.0
                        A_nu_1[i, j] = jump.in_eigenbasis[i, j]         # A_nu_1
                    else
                        A_nu_1 .= spdiagm(0 => jump_diag)
                    end

                    if nu_2 != 0.0
                        A_nu_2_dagger[k, i] = conj(jump.in_eigenbasis[i, k])   # A_nu_2^\dagger
                    else
                        A_nu_2_dagger .= conj.(spdiagm(0 => jump_diag))
                    end

                    liouv .+= vectorize_liouvillian_diss(alpha(nu_1, nu_2) * A_nu_1, A_nu_2_dagger)
                end
            end
        end

    end
    return liouv
end

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
    energy_labels = hamiltonian.w0 * N_labels

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
        t0 = 2 * pi / (N * hamiltonian.w0)
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
    energy_labels = hamiltonian.w0 * N_labels
    t0 = 2 * pi / (N * hamiltonian.w0)
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
    energy_labels = hamiltonian.w0 * N_labels
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
    energy_labels = hamiltonian.w0 * N_labels

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
    energy_labels = hamiltonian.w0 * N_labels
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
    energy_labels = hamiltonian.w0 * N_labels

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
        t0 = 2 * pi / (N * hamiltonian.w0)
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

function vectorize_liouvillian_coherent(coherent_term::Matrix{ComplexF64})

    dim = size(coherent_term)[1]
    spI = sparse(I, dim, dim)

    # -i ((I ⊗ B) - (B^T ⊗ I))
    # vecotrized_coherent_term = -1im * (kron(spI, coherent_term) - kron(transpose(coherent_term), spI))

    # Watrous
    vecotrized_coherent_part = -1im *(kron(coherent_term, spI) - kron(spI, transpose(coherent_term)))
    return vecotrized_coherent_part
end

function vectorize_liouvillian_diss(jump_1::SparseMatrixCSC{ComplexF64}, jump_2::SparseMatrixCSC{ComplexF64})
    """L = J1 * X * J2 - 0.5 * (J2 * J1 * X + X * J2 * J1)"""


    dim = size(jump_1)[1]
    spI = sparse(I, dim, dim)

    jump_2_jump_1 = jump_2 * jump_1
    # vectorized_diss_part = kron(transpose(jump_2), jump_1) - 0.5 * (kron(spI, jump_2_jump_1) + kron(transpose(jump_2_jump_1), spI))
    
    # Watrous
    vectorized_diss_part = kron(jump_1, transpose(jump_2)) - 0.5 * (kron(jump_2_jump_1, spI) + kron(spI, transpose(jump_2_jump_1))) 
    
    return vectorized_diss_part
end

#! Rewrote it from jumps to jump, no for loop inside here
function vectorize_liouvillian_diss(jump_op::Matrix{ComplexF64})

    dim = size(jump_ops[1])[1]
    spI = sparse(I, dim, dim)

    vectorized_liouv = zeros(ComplexF64, dim^2, dim^2)
    jump_dag_jump = jump' * jump

    # Jump^\dag^T = Jump^*
    # Watrous
    vectorized_liouv .+= kron(jump, conj(jump)) - 0.5 * (kron(jump_dag_jump, spI) + kron(spI, transpose(jump_dag_jump))) 

    return vectorized_liouv
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