# Result serialization infrastructure.

const DOMAIN_LOOKUP = Dict(
    "TrotterDomain" => TrotterDomain(),
    "TimeDomain"    => TimeDomain(),
    "EnergyDomain"  => EnergyDomain(),
    "BohrDomain"    => BohrDomain(),
)

_string_to_domain(s::AbstractString) = DOMAIN_LOOKUP[s]

"""
    _config_to_dict(config::Config) -> Dict{Symbol, Any}

Serialize a config struct to a Dict with string-tagged type info.
"""
function _config_to_dict(config::Config)
    d = Dict{Symbol, Any}()

    # Type tags
    d[:config_type] = config.construction isa GNS ? "GNS" : config.construction isa KMS ? "KMS" : "DLL"
    d[:config_kind] = config.sim isa Thermalize ? "thermalize" :
        config.sim isa KrylovSpectrum ? "krylov_spectrum" : "lindbladian"
    d[:domain] = string(typeof(config.domain))

    # Shared fields (all config types have these)
    d[:num_qubits]              = config.num_qubits
    d[:with_coherent]           = with_coherent(config.construction)
    d[:with_linear_combination] = config.with_linear_combination
    d[:beta]                    = config.beta
    d[:beta_phys]               = config.beta_phys
    d[:sigma]                   = config.sigma
    d[:gaussian_parameters]     = config.gaussian_parameters
    d[:a]                       = config.a
    d[:s]                       = config.s
    # Preserve shared-register fields so older caches remain readable.
    d[:num_energy_bits]         = config.num_energy_bits
    d[:t0]                      = config.t0
    d[:w0]                      = config.w0
    d[:num_energy_bits_D]       = config.num_energy_bits_D
    d[:t0_D]                    = config.t0_D
    d[:w0_D]                    = config.w0_D
    d[:num_energy_bits_b_minus] = config.num_energy_bits_b_minus
    d[:t0_b_minus]              = config.t0_b_minus
    d[:w0_b_minus]              = config.w0_b_minus
    d[:num_energy_bits_b_plus]  = config.num_energy_bits_b_plus
    d[:t0_b_plus]               = config.t0_b_plus
    d[:w0_b_plus]               = config.w0_b_plus
    d[:eta]                     = config.eta
    d[:num_trotter_steps_per_t0] = config.num_trotter_steps_per_t0
    d[:num_trotter_steps_per_t0_D] = config.num_trotter_steps_per_t0_D
    d[:num_trotter_steps_per_t0_b_minus] = config.num_trotter_steps_per_t0_b_minus
    d[:num_trotter_steps_per_t0_b_plus] = config.num_trotter_steps_per_t0_b_plus
    d[:with_gqsp]               = config.with_gqsp
    d[:gqsp_degree]             = config.gqsp_degree
    d[:jump_selection]          = config.jump_selection
    d[:filter]                  = _portable_pack(config.filter)
    d[:transition_weight]       = _portable_pack(config.transition_weight)

    # Thermalize-specific fields
    if config.sim isa Thermalize
        d[:mixing_time] = config.mixing_time
        d[:delta]       = config.delta
    end

    return d
end


"""
    _reconstruct_config(d::Dict) -> Config

Reconstruct the correct Config struct from a serialized Dict.
Uses config_type ("KMS"/"GNS"/"DLL") and config_kind ("liouv"/"thermalize") to pick the singletons.
"""
function _reconstruct_config(d::Dict)
    domain = _string_to_domain(d[:domain])
    config_type = d[:config_type]   # "KMS", "GNS", or "DLL"

    # Determine sim type: prefer config_kind tag, fall back to presence of mixing_time
    # Both "liouv" and "lindbladian" deserialize as Lindbladian().
    config_kind = get(d, :config_kind, nothing)

    kwargs = _dict_to_config_kwargs(d)

    construction = if config_type == "KMS"
        KMS()
    elseif config_type == "GNS"
        GNS()
    elseif config_type == "DLL"
        DLL()
    else
        throw(ArgumentError("unknown serialized construction tag: $(repr(config_type))"))
    end
    sim = if config_kind == "thermalize"
        Thermalize()
    elseif config_kind == "krylov_spectrum"
        KrylovSpectrum()
    elseif config_kind == "lindbladian" || config_kind == "liouv"
        Lindbladian()
    elseif config_kind === nothing
        # Untagged compatibility schema: infer from `mixing_time`.
        (haskey(d, :mixing_time) && d[:mixing_time] !== nothing) ? Thermalize() : Lindbladian()
    else
        throw(ArgumentError("unknown serialized simulation tag: $(repr(config_kind))"))
    end

    Config(; sim=sim, domain=domain, construction=construction, kwargs...)
end

"""
    _dict_to_config_kwargs(d::Dict) -> Dict{Symbol, Any}

Build a kwargs Dict from serialized config fields, suitable for @kwdef constructors.
Filters out nothing values for optional fields to let defaults apply.
"""
function _dict_to_config_kwargs(d::Dict)
    kwargs = Dict{Symbol, Any}()

    # Required fields (with_coherent derived from construction type, not stored)
    kwargs[:num_qubits]              = d[:num_qubits]
    kwargs[:with_linear_combination] = d[:with_linear_combination]
    kwargs[:beta]                    = d[:beta]
    kwargs[:sigma]                   = d[:sigma]

    # Optional per-leg and shared-register fields support both cache schemas.
    for key in (
        :beta_phys, :a, :s, :eta, :num_trotter_steps_per_t0,
        :num_trotter_steps_per_t0_D,
        :num_trotter_steps_per_t0_b_minus,
        :num_trotter_steps_per_t0_b_plus,
        :num_energy_bits, :t0, :w0,
        :num_energy_bits_D, :t0_D, :w0_D,
        :num_energy_bits_b_minus, :t0_b_minus, :w0_b_minus,
        :num_energy_bits_b_plus, :t0_b_plus, :w0_b_plus,
        :filter, :transition_weight,
    )
        val = get(d, key, nothing)
        if val !== nothing
            kwargs[key] = key in (:filter, :transition_weight) ? _portable_unpack(val;restore=true) : val
        end
    end

    for (key, default) in (
        (:with_gqsp, false),
        (:gqsp_degree, 1),
        (:jump_selection, :sweep),
    )
        kwargs[key] = get(d, key, default)
    end

    # gaussian_parameters: BSON stores tuples as arrays, so convert back
    gp = get(d, :gaussian_parameters, nothing)
    if gp !== nothing
        if gp isa AbstractVector
            kwargs[:gaussian_parameters] = (gp[1], gp[2])
        else
            kwargs[:gaussian_parameters] = gp
        end
    end

    # Thermalize-specific
    for key in (:mixing_time, :delta)
        val = get(d, key, nothing)
        if val !== nothing
            kwargs[key] = val
        end
    end

    return kwargs
end

# ============================================================================
# Metadata auto-capture
# ============================================================================

"""
    _capture_metadata(; n_threads, wall_time_seconds, extra) -> Dict{Symbol, Any}

Auto-capture run metadata: timestamp, git hash, thread count, wall time.
Merges any extra key-value pairs from the `extra` Dict.
"""
function _capture_metadata(;
    n_threads::Int = Threads.nthreads(),
    wall_time_seconds::Union{Float64, Nothing} = nothing,
    extra::Dict{Symbol, Any} = Dict{Symbol, Any}(),
)
    meta = Dict{Symbol, Any}(
        :timestamp         => Dates.format(Dates.now(), dateformat"yyyy-mm-dd_HH:MM:SS"),
        :git_hash          => _capture_git_hash(),
        :n_threads         => n_threads,
        :wall_time_seconds => wall_time_seconds,
    )
    merge!(meta, extra)
    return meta
end

"""
    _capture_git_hash() -> String

Capture the current HEAD commit hash via LibGit2. Returns "unknown" on failure.
"""
function _capture_git_hash()
    try
        project_root = dirname(@__DIR__)
        repo = LibGit2.GitRepo(project_root)
        hash = string(LibGit2.head_oid(repo))
        close(repo)
        return hash
    catch
        return "unknown"
    end
end

# Result type tags.

_result_type_tag(::LindbladResults) = "lindblad"
_result_type_tag(::ThermalizeResults) = "thermalize"
_result_type_tag(::KrylovSpectrumResults) = "krylov_spectrum"

"""
    _result_to_dict(r::AbstractResults) -> Dict{Symbol, Any}

Convert a typed result to a dictionary for BSON serialization.
"""

function _result_to_dict(r::LindbladResults)
    return Dict{Symbol, Any}(
        :result_type  => "lindblad",
        :config       => _config_to_dict(r.config),
        :eigenvalues  => r.eigenvalues,
        :fixed_point  => Matrix(r.fixed_point),
        :gap_mode     => Matrix(r.gap_mode),
        :spectral_gap => r.spectral_gap,
        :metadata     => r.metadata,
    )
end

function _result_to_dict(r::ThermalizeResults)
    return Dict{Symbol, Any}(
        :result_type      => "thermalize",
        :config           => _config_to_dict(r.config),
        :final_dm         => Matrix(r.final_dm),
        :trace_distances  => r.trace_distances,
        :time_steps       => r.time_steps,
        :metadata         => r.metadata,
    )
end

function _result_to_dict(r::KrylovSpectrumResults)
    return Dict{Symbol, Any}(
        :result_type          => "krylov_spectrum",
        :config               => _config_to_dict(r.config),
        :eigenvalues          => r.eigenvalues,
        :spectral_gap         => r.spectral_gap,
        :fixed_point          => Matrix(r.fixed_point),
        :gap_mode             => Matrix(r.gap_mode),
        :converged            => r.converged,
        :matvec_count         => r.matvec_count,
        :num_restarts         => r.num_restarts,
        :normres              => r.normres,
        :channel_eigenvalues  => r.channel_eigenvalues,
        :delta_used           => r.delta_used,
        :metadata             => r.metadata,
    )
end

# ---------------------------------------------------------------------------
# Dict -> Result reconstruction
# ---------------------------------------------------------------------------

function _dict_to_lindblad_results(d::Dict)
    config = _reconstruct_config(d[:config])
    T = real(eltype(d[:eigenvalues]))
    return LindbladResults{T}(
        config,
        d[:eigenvalues],
        d[:fixed_point],
        d[:gap_mode],
        d[:spectral_gap],
        d[:metadata],
    )
end

function _dict_to_thermalize_results(d::Dict)
    config = _reconstruct_config(d[:config])
    T = eltype(d[:trace_distances])
    return ThermalizeResults{T}(
        config,
        d[:final_dm],
        d[:trace_distances],
        d[:time_steps],
        d[:metadata],
    )
end

function _dict_to_krylov_spectrum_results(d::Dict)
    config = _reconstruct_config(d[:config])
    T = eltype(d[:normres])
    return KrylovSpectrumResults{T}(
        config,
        d[:eigenvalues],
        d[:spectral_gap],
        d[:fixed_point],
        d[:gap_mode],
        d[:converged],
        d[:matvec_count],
        d[:num_restarts],
        d[:normres],
        get(d, :channel_eigenvalues, nothing),
        get(d, :delta_used, nothing),
        d[:metadata],
    )
end

# ---------------------------------------------------------------------------
# save_result / load_result
# ---------------------------------------------------------------------------

"""
    save_result(result::AbstractResults, path::String) -> String

Save a typed Result to a canonical `.bson` target derived from `path`, plus a
companion `.txt` file with the same stem. Creates parent directories as needed
and returns the canonical BSON path.
"""
function save_result(result::AbstractResults, path::String)
    stem, _ = splitext(path)
    bson_path = stem * ".bson"
    txt_path = stem * ".txt"
    d = _result_to_dict(result)
    mkpath(dirname(bson_path))
    BSON.bson(bson_path, d)
    _write_result_companion_txt(result, txt_path)
    return bson_path
end

"""
    load_result(path::String) -> AbstractResults

Load a typed Result from a BSON file. Auto-detects the result type via the
`:result_type` tag stored in the BSON Dict.
"""
function load_result(path::String)
    d = BSON.load(path)
    tag = d[:result_type]
    if tag == "gibbs_simulation"
        return _dict_to_gibbs_results(d)
    elseif tag == "lindblad"
        return _dict_to_lindblad_results(d)
    elseif tag == "thermalize"
        return _dict_to_thermalize_results(d)
    elseif tag == "krylov_spectrum"
        return _dict_to_krylov_spectrum_results(d)
    else
        error("Unknown result type: $tag")
    end
end

# ---------------------------------------------------------------------------
# Companion .txt file (per result type)
# ---------------------------------------------------------------------------

"""
    _write_result_companion_txt(result::AbstractResults, path::String)

Write a human-readable summary alongside the BSON file.
Format varies by result type, showing key metrics for quick browsing.
"""
function _write_result_companion_txt(result::AbstractResults, path::String)
    open(path, "w") do io
        cfg  = result.config
        meta = result.metadata
        type_name = typeof(result).name.name  # e.g. :LindbladResults

        println(io, "=== QuantumFurnace [$type_name] ===")
        println(io)
        println(io, "Date:       ", get(meta, :timestamp, "unknown"))
        println(io, "Git:        ", get(meta, :git_hash, "unknown"))
        println(io, "Threads:    ", get(meta, :n_threads, "unknown"))
        wt = get(meta, :wall_time_seconds, nothing)
        println(io, "Wall time:  ", wt === nothing ? "unknown" : "$wt s")
        println(io)
        println(io, "--- Config ---")
        println(io, "Construction: ", cfg.construction isa GNS ? "GNS" : cfg.construction isa KMS ? "KMS" : "DLL")
        println(io, "Domain:     ", typeof(cfg.domain))
        println(io, "n_qubits:   ", cfg.num_qubits)
        println(io, "beta:       ", cfg.beta)
        println(io)
        println(io, "--- Results ---")

        if result isa LindbladResults
            println(io, "Spectral gap (real): ", real(result.spectral_gap))
            println(io, "Spectral gap (imag): ", imag(result.spectral_gap))
            println(io, "Fixed point dim:     ", size(result.fixed_point, 1), "x", size(result.fixed_point, 2))
            println(io, "N eigenvalues:       ", length(result.eigenvalues))

        elseif result isa ThermalizeResults
            td = result.trace_distances
            println(io, "Final trace dist:    ", isempty(td) ? "N/A" : last(td))
            println(io, "N time steps:        ", length(result.time_steps))
            println(io, "Final DM dim:        ", size(result.final_dm, 1), "x", size(result.final_dm, 2))

        elseif result isa KrylovSpectrumResults
            println(io, "Spectral gap:        ", result.spectral_gap)
            println(io, "Matvec count:        ", result.matvec_count)
            println(io, "Converged:           ", result.converged)
            println(io, "Num restarts:        ", result.num_restarts)
            if result.delta_used !== nothing
                println(io, "Delta used:          ", result.delta_used)
            end
            if result.channel_eigenvalues !== nothing
                println(io, "Channel eigenvalues: ", length(result.channel_eigenvalues))
            end

        end
    end
end

# ---------------------------------------------------------------------------
# Auto-filename generation for new Results
# ---------------------------------------------------------------------------

"""
    _generate_result_filename(result::AbstractResults) -> String

Generate a descriptive filename: `{type}_{construction}_{n}_{beta}_{domain}_{date}.bson`.
"""
function _generate_result_filename(result::AbstractResults)
    cfg = result.config
    type_str = _result_type_tag(result)
    db_str = cfg.construction isa GNS ? "gns" : cfg.construction isa KMS ? "kms" : "dll"
    domain_str = lowercase(replace(string(typeof(cfg.domain)), "Domain" => ""))
    n_str = "n$(cfg.num_qubits)"
    beta_str = "beta$(repr(cfg.beta))"
    date_str = Dates.format(Dates.now(), dateformat"yyyymmdd")
    return "$(type_str)_$(db_str)_$(n_str)_$(beta_str)_$(domain_str)_$(date_str).bson"
end
