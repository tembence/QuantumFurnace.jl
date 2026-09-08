"""Supertype for construction domains."""
abstract type AbstractDomain end

"""Exact Bohr-frequency construction domain."""
struct BohrDomain <: AbstractDomain end
"""Energy-quadrature construction domain."""
struct EnergyDomain <: AbstractDomain end
"""Time-quadrature construction domain."""
struct TimeDomain <: AbstractDomain end
"""Trotterised time-domain construction."""
struct TrotterDomain <: AbstractDomain end

"""Supertype for simulation modes."""
abstract type AbstractSimulation end
"""Dense Lindbladian simulation mode."""
struct Lindbladian    <: AbstractSimulation end
"""Full-density-matrix channel simulation mode."""
struct Thermalize     <: AbstractSimulation end
"""Matrix-free Krylov spectral mode."""
struct KrylovSpectrum <: AbstractSimulation end
"""Tensor-network parent-spectrum simulation mode."""
struct TensorNetworkSpectrum <: AbstractSimulation end

"""Supertype for detailed-balance constructions."""
abstract type AbstractConstruction end
"""KMS detailed-balance construction with a coherent correction."""
struct KMS <: AbstractConstruction end
"""GNS detailed-balance construction without a coherent correction."""
struct GNS <: AbstractConstruction end
"""Ding–Li–Lin construction with a coherent correction."""
struct DLL <: AbstractConstruction end

# Concrete cached callable used by Bohr-domain KMS/GNS workspaces.
struct BohrAlphaKernel{C<:AbstractConstruction, T<:AbstractFloat} <: Function
    with_linear_combination::Bool
    beta::T
    sigma::T
    a::T
    s::T
    gaussian_parameters::Tuple{T, T}
end

"""Return whether a detailed-balance construction includes the coherent term."""
with_coherent(::KMS) = true
with_coherent(::GNS) = false
with_coherent(::DLL) = true

"""
    Config{S, D, C, T}

Typed configuration for Gibbs-sampler construction and simulation.

The parameters `S`, `D`, `C`, and `T` encode the simulation mode, construction
domain, detailed-balance family, and floating-point precision. Coherent-term
presence is determined by `with_coherent(C())`.

# Fields
- `sim`, `domain`, `construction`: Dispatch singletons.
- `num_qubits`: System size.
- `beta`: Algorithm-side inverse temperature for the rescaled Hamiltonian.
- `beta_phys`: Optional physical inverse temperature.
- `sigma`, `gaussian_parameters`, `a`, `s`: Filter and rate parameters.
- `num_energy_bits_*`, `t0_*`, `w0_*`: Independent dissipative and coherent
  register triples.
- `mixing_time`, `delta`: Channel duration and step size.
- `with_gqsp`, `gqsp_degree`: Coherent polynomial approximation controls.
- `jump_selection`: `:sweep` or `:random` for full-DM channel evolution.
- `filter`: Construction-specific filter. KMS/GNS accept `nothing` or a
  `GaussianFilter` matching `sigma`; DLL requires an explicit DLL filter.

Each register obeys `\$w0_X t0_X = 2 pi / 2^r_X\$`. Unsuffixed register fields
are compatibility fallbacks promoted by `validate_config!`.
"""
@kwdef struct Config{S <: AbstractSimulation, D <: AbstractDomain, C <: AbstractConstruction, T <: AbstractFloat}
    # Type-encoding singletons
    sim::S
    domain::D
    construction::C

    # System parameters
    num_qubits::Int
    with_linear_combination::Bool

    # `beta` is beta_alg; `beta_phys`, when present, must satisfy
    # Math: $beta_alg = beta_phys dot ham.rescaling_factor$.
    beta::T
    beta_phys::Union{T, Nothing} = nothing
    sigma::T
    gaussian_parameters::Union{Tuple{T, T}, Tuple{Nothing, Nothing}} = (nothing, nothing)
    a::Union{T, Nothing} = nothing
    s::Union{T, Nothing} = nothing

    # Independent dissipative and coherent register triples.
    num_energy_bits_D::Union{Int, Nothing} = nothing
    t0_D::Union{T, Nothing} = nothing
    w0_D::Union{T, Nothing} = nothing
    num_energy_bits_b_minus::Union{Int, Nothing} = nothing
    t0_b_minus::Union{T, Nothing} = nothing
    w0_b_minus::Union{T, Nothing} = nothing
    num_energy_bits_b_plus::Union{Int, Nothing} = nothing
    t0_b_plus::Union{T, Nothing} = nothing
    w0_b_plus::Union{T, Nothing} = nothing
    # Compatibility fields promoted to all triples by `validate_config!`.
    num_energy_bits::Union{Int, Nothing} = nothing
    t0::Union{T, Nothing} = nothing
    w0::Union{T, Nothing} = nothing
    eta::Union{T, Nothing} = nothing
    num_trotter_steps_per_t0::Union{Int, Nothing} = nothing

    # Per-leg Strang counts; accessors fall back to the common count.
    num_trotter_steps_per_t0_D::Union{Int, Nothing} = nothing
    num_trotter_steps_per_t0_b_minus::Union{Int, Nothing} = nothing
    num_trotter_steps_per_t0_b_plus::Union{Int, Nothing} = nothing

    # Thermalize-specific
    mixing_time::Union{T, Nothing} = nothing
    delta::Union{T, Nothing} = nothing

    # GQSP-specific (Thermalize coherent step)
    with_gqsp::Bool = false
    gqsp_degree::Int = 1

    # Full-DM jump rule: deterministic sweep or rate-rescaled random choice.
    jump_selection::Symbol = :sweep

    # `nothing` selects the CKG Gaussian with width `sigma` for KMS/GNS.
    # DLL constructions require an explicit DLL filter.
    filter::Union{Nothing, AbstractFilter} = nothing
end

"""Return the dissipative time cutoff, falling back to the common register."""
@inline register_t0_D(cfg::Config) = cfg.t0_D !== nothing ? cfg.t0_D : cfg.t0

"""Return the dissipative frequency spacing, falling back to the common register."""
@inline register_w0_D(cfg::Config) = cfg.w0_D !== nothing ? cfg.w0_D : cfg.w0

"""Return the dissipative register size, falling back to the common register."""
@inline register_r_D(cfg::Config)  = cfg.num_energy_bits_D !== nothing ? cfg.num_energy_bits_D : cfg.num_energy_bits

"""Return the outer-coherent time cutoff, falling back to the common register."""
@inline register_t0_b_minus(cfg::Config) = cfg.t0_b_minus !== nothing ? cfg.t0_b_minus : cfg.t0

"""Return the outer-coherent frequency spacing, falling back to the common register."""
@inline register_w0_b_minus(cfg::Config) = cfg.w0_b_minus !== nothing ? cfg.w0_b_minus : cfg.w0

"""Return the outer-coherent register size, falling back to the common register."""
@inline register_r_b_minus(cfg::Config)  = cfg.num_energy_bits_b_minus !== nothing ? cfg.num_energy_bits_b_minus : cfg.num_energy_bits

"""Return the inner-coherent time cutoff, falling back to the common register."""
@inline register_t0_b_plus(cfg::Config) = cfg.t0_b_plus !== nothing ? cfg.t0_b_plus : cfg.t0

"""Return the inner-coherent frequency spacing, falling back to the common register."""
@inline register_w0_b_plus(cfg::Config) = cfg.w0_b_plus !== nothing ? cfg.w0_b_plus : cfg.w0

"""Return the inner-coherent register size, falling back to the common register."""
@inline register_r_b_plus(cfg::Config)  = cfg.num_energy_bits_b_plus !== nothing ? cfg.num_energy_bits_b_plus : cfg.num_energy_bits

"""
    beta_alg(cfg::Config) -> T

Return the required algorithm-side inverse temperature.
"""
@inline beta_alg(cfg::Config) = cfg.beta

"""
    beta_phys(cfg::Config) -> Union{T, Nothing}

Return the optional physical inverse temperature.
"""
@inline beta_phys(cfg::Config) = cfg.beta_phys

"""Return the dissipative Strang count, falling back to the common count."""
@inline register_M_D(cfg::Config) =
    cfg.num_trotter_steps_per_t0_D       !== nothing ? cfg.num_trotter_steps_per_t0_D       : cfg.num_trotter_steps_per_t0

"""Return the outer-coherent Strang count, falling back to the common count."""
@inline register_M_b_minus(cfg::Config) =
    cfg.num_trotter_steps_per_t0_b_minus !== nothing ? cfg.num_trotter_steps_per_t0_b_minus : cfg.num_trotter_steps_per_t0

"""Return the inner-coherent Strang count, falling back to the common count."""
@inline register_M_b_plus(cfg::Config) =
    cfg.num_trotter_steps_per_t0_b_plus  !== nothing ? cfg.num_trotter_steps_per_t0_b_plus  : cfg.num_trotter_steps_per_t0

"""
    JumpOp

Jump operator stored in computational and Hamiltonian eigenbases.

# Fields
- `data`: Computational-basis matrix.
- `in_eigenbasis`: Matrix in the construction basis.
- `orthogonal`: Legacy flag for transpose symmetry in the computational basis.
- `hermitian`: Whether both stored matrices are Hermitian.
"""
struct JumpOp{T <: AbstractMatrix{<:Complex}}
    data::T
    in_eigenbasis::Matrix{<:Complex}
    orthogonal::Bool
    hermitian::Bool
end

"""Supertype for serialisable simulation results."""
abstract type AbstractResults end

"""
    LindbladResults{T<:AbstractFloat} <: AbstractResults

Dense Liouvillian spectrum, fixed point, gap mode, and metadata.
"""
struct LindbladResults{T<:AbstractFloat} <: AbstractResults
    config::Config
    eigenvalues::Vector{Complex{T}}
    fixed_point::Matrix{Complex{T}}
    gap_mode::Matrix{Complex{T}}
    spectral_gap::Complex{T}
    metadata::Dict{Symbol, Any}
end

"""
    ThermalizeResults{T<:AbstractFloat} <: AbstractResults

Full-density-matrix channel result and its distance history.
"""
struct ThermalizeResults{T<:AbstractFloat} <: AbstractResults
    config::Config
    final_dm::Matrix{Complex{T}}
    trace_distances::Vector{T}
    time_steps::Vector{T}
    metadata::Dict{Symbol, Any}
end

"""
    KrylovSpectrumResults{T<:AbstractFloat} <: AbstractResults

Matrix-free spectrum, stationary mode, gap mode, and convergence metadata.
"""
struct KrylovSpectrumResults{T<:AbstractFloat} <: AbstractResults
    config::Config
    eigenvalues::Vector{Complex{T}}
    spectral_gap::T
    fixed_point::Matrix{Complex{T}}
    gap_mode::Matrix{Complex{T}}
    converged::Int
    matvec_count::Int
    num_restarts::Int
    normres::Vector{T}
    channel_eigenvalues::Union{Nothing, Vector{Complex{T}}}
    delta_used::Union{Nothing, T}
    metadata::Dict{Symbol, Any}
end

"""
    ThermalizeScratch{T<:Complex}

Reusable matrices for full-density-matrix channel evolution.

`task_scratches` optionally owns one independent scratch set per Julia thread.
"""
struct ThermalizeScratch{T<:Complex}
    jump_oft::Matrix{T}
    LdagL::Matrix{T}
    R::Matrix{T}
    rho_jump::Matrix{T}
    sandwich_tmp::Matrix{T}    # was tmp1
    rho_work::Matrix{T}        # was tmp2
    rho_next::Matrix{T}
    task_scratches::Vector{ThermalizeScratch{T}}
end

function ThermalizeScratch(::Type{CT}, dim::Int;
                           with_task_pool::Bool=false,
                           num_threads::Int=Threads.nthreads()) where {CT<:Complex}
    Zm() = zeros(CT, dim, dim)

    # Child scratches own no further pool, terminating recursive construction.
    task_pool = if with_task_pool && num_threads > 1
        [ThermalizeScratch{CT}(Zm(), Zm(), Zm(), Zm(), Zm(), Zm(), Zm(),
                               ThermalizeScratch{CT}[]) for _ in 1:num_threads]
    else
        ThermalizeScratch{CT}[]
    end

    return ThermalizeScratch{CT}(Zm(), Zm(), Zm(), Zm(), Zm(), Zm(), Zm(), task_pool)
end

"""
    KrylovScratch{T<:Complex}

Reusable matrices and thread-local state for Krylov operator actions.

`work_list` stores the precomputed `(jump_index, label_index)` schedule.
"""
struct KrylovScratch{T<:Complex}
    jump_oft::Matrix{T}
    bohr_component_dag::Matrix{T}
    sandwich_tmp::Matrix{T}
    sandwich_out::Matrix{T}
    rho_out::Matrix{T}
    channel_rho_jump::Union{Nothing, Matrix{T}}
    task_scratches::Vector{KrylovScratch{T}}
    work_list::Vector{Tuple{Int, Int}}
end

function KrylovScratch(::Type{CT}, dim::Int;
                       with_channel_rho_jump::Bool=false,
                       num_threads::Int=Threads.nthreads()) where {CT<:Complex}
    Zm() = zeros(CT, dim, dim)
    crj = with_channel_rho_jump ? Zm() : nothing

    # Child scratches own empty pools to prevent aliasing and recursion.
    task_pool = if num_threads > 1
        [KrylovScratch{CT}(Zm(), Zm(), Zm(), Zm(), Zm(),
                           with_channel_rho_jump ? Zm() : nothing,
                           KrylovScratch{CT}[],
                           Tuple{Int, Int}[]) for _ in 1:num_threads]
    else
        KrylovScratch{CT}[]
    end

    return KrylovScratch{CT}(Zm(), Zm(), Zm(), Zm(), Zm(), crj,
                             task_pool, Tuple{Int, Int}[])
end

"""
    Workspace{S, D, C, T}

Precomputed state and scratch storage for matrix-free and full-DM paths.

Type parameters encode simulation mode, domain, detailed-balance construction,
and precision. Dense Liouvillian construction uses `DenseLindbladianWorkspace`.
"""
struct Workspace{S<:AbstractSimulation, D<:AbstractDomain, C<:AbstractConstruction, T<:AbstractFloat}
    # Physics data (Krylov/Thermalize)
    jump_eigenbases::Union{Nothing, Vector{Matrix{Complex{T}}}}
    jump_hermitian::Union{Nothing, Vector{Bool}}
    jumps::Union{Nothing, Vector{JumpOp}}

    # DLL per-jump Bohr-domain Lindblad operators.
    dll_lindblads::Union{Nothing, Vector{Matrix{Complex{T}}}}

    # Krylov effective Hamiltonian (Lindbladian mode)
    G_left::Union{Nothing, Matrix{Complex{T}}}
    G_right::Union{Nothing, Matrix{Complex{T}}}

    # Domain-specific precomputed data.
    transition::Union{Nothing, Function}
    gamma_norm_factor::Union{Nothing, Float64}
    energy_labels::Union{Nothing, Vector{Float64}}
    oft_domain_prefactor::Union{Nothing, Float64}
    oft_nufft_prefactors::Union{Nothing, NUFFTPrefactors{T}}
    bohr_alpha::Union{Nothing, BohrAlphaKernel{C, T}}
    bohr_keys::Union{Nothing, Vector{T}}
    bohr_is::Union{Nothing, Vector{Vector{Int}}}
    bohr_js::Union{Nothing, Vector{Vector{Int}}}
    b_minus::Union{Nothing, Dict{T, ComplexF64}}
    b_plus::Union{Nothing, Dict{T, ComplexF64}}

    # Per-jump channel state used by the retained full-DM and Krylov paths.
    ham_or_trott::Union{Nothing, HamHam{T}, AbstractTrotter{T}}
    K0s::Union{Nothing, Vector{Matrix{Complex{T}}}}
    U_residuals::Union{Nothing, Vector{Matrix{Complex{T}}}}
    # `nothing` elements skip coherent evolution without another construction
    # dispatch inside the channel matvec.
    U_coherents::Union{Nothing, Vector{Union{Nothing, Matrix{Complex{T}}}}}

    # Scratch buffers (nested, simulation-path-specific)
    scratch::Union{KrylovScratch{Complex{T}}, ThermalizeScratch{Complex{T}}}

    # Cached construction config for workspace-reuse validation.
    cached_cfg::Union{Nothing, Config}
end
