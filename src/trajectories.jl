#TODO: Test this code!
struct KrausFramework{T}
    M0::Matrix{T}           # non-Hermitian no jump evolution
    M_jumps::Matrix{T}      # jump Kraus
    sum_MdagM::Matrix{T}    # sum(M^\dagger M) for speed
    psi_temp::Vector{T}     # buffer for less allocations
    delta::Float64          # delta steps for trajectory
end

function build_krausframework(H::Matrix{ComplexF64}, L_jumps::Vector{Matrix{ComplexF64}}, delta::Float64)::KrausFramework
    dim = size(H, 1)
    T = ComplexF64
    
    M_jumps = [sqrt(delta) .* L for L in L_jumps]

    # Construct M0 = I - delta * H_eff = I - detla * (iH + 0.5 * sum(L^\dagger L))
    sum_LdagL = zeros(T, dim, dim)
    for L in L_jumps
        mul!(sum_LdagL, L', L, 1.0, 1.0)
    end

    H_eff = 1im * H  + 0.5 * sum_LdagL
    M0 = Matrix{T}(I, dim, dim) - delta * H_eff

    sum_MdagM = delta .* sum_LdagL

    # Buffer
    psi_temp = zeros(ComplexF64, dim)

    return KrausFramework(M0, M_jumps, sum_MdagM, psi_temp, delta)
end

function step_along_the_trajectory!(psi::Vector{ComplexF64}, fw::KrausFramework)

    # No jump
    mul!(fw.psi_temp, fw.M0, psi)
    prob_no_jump = norm(fw.psi_temp)^2

    # Total jump probability
    p_jump_total = real(dot(psi, fw.sum_MdagM, psi))

    total_weight = prob_no_jump + p_jump_total

    r = rand() * total_weight

    if r < prob_no_jump
        # No jump
        copyto!(psi, fw.psi_temp)

        # Force normalize
        rmul!(psi, 1.0 / sqrt(prob_no_jump))
    else
        # Jump, but which jump?
        # Iterate through jumps and their probabilites till we find the winner
        target_cummulative = r - prob_no_jump
        current_cummulative = 0.0
        
        for k in 1:length(fw.M_jumps)
            mul!(fw.psi_temp, fw.M_jumps[k], psi)
            prob_jump_k = norm(fw.psi_temp)^2

            current_cummulative += prob_jump_k

            if current_cummulative >= target_cummulative
                copyto!(psi,  fw.psi_temp)

                # Force normalize
                rmul!(psi, 1.0 / sqrt(prob_jump_k))
                return
            end
        end

        # If somehow we haven't picked any jumps, then use the last one.
        copyto!(psi, fw.psi_temp)
        normalize!(psi)
    end
end

function evolve_along_trajectory(psi0::Vector{ComplexF64}, fw::KrausFramework, T::Float64)

    num_steps = round(Int, T / fw.delta)  # Number of trajectory steps

    psi = copy(psi0)
    for _ in 1:num_steps
        step_along_the_trajectory!(psi, fw)
    end

    return psi
end

function evolve_and_measure_along_trajectory(
    psi0::Vector{ComplexF64}, 
    fw::KrausFramework, 
    T::Float64, 
    observables::Vector{Matrix{ComplexF64}};
    save_every::Int = 1
    )

    num_steps = round(Int, T / fw.delta)  # Number of trajectory steps
    num_saves = div(num_steps, save_every) + 1
    num_obs = length(observables)
    data = zeros(Float64, num_obs, num_saves)
    times = zeros(Float64, num_saves)

    psi = copy(psi0)
    # Measure initial state
    save_index = 1
    times[save_index] = 0.0
    measure!(view(data, :, save_index), psi, observables)

    for step in 1:num_steps
        step_along_the_trajectory!(psi, fw)

        if step % save_every == 0
            save_index += 1
            times[save_index] = step * fw.delta
            measure!(view(data, :, save_index), psi, observables)
        end
    end

    return (psi = psi, times = times, measurements = data)
end

function measure!(measured_values, state, observable_list)
    for i in eachindex(observable_list)
        measured_values[i] = real(dot(state, observable_list[i], state))
    end
end

function construct_gksl_lindbladian(
    H::AbstractMatrix{ComplexF64}, 
    L_jumps::Vector{Matrix{ComplexF64}}
    )
    dim = size(H, 1)
    dim2 = dim^2

    lindblad = zeros(ComplexF64, dim2, dim2)

    # Add coherent part
    vectorize_liouvillian_coherent!(lindblad, H)

    for L_k in L_jumps
        vectorize_liouv_diss_and_add!(lindblad, L_k, 1.0)
    end

    return lindblad
end