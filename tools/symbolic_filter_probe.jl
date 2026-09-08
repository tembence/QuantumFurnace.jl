#!/usr/bin/env julia
# Optional T21 experiment; no symbolic dependency enters the package environment.
# Run: julia --project tools/symbolic_filter_probe.jl
# Reuse a disposable installation: --env=/tmp/qf-symbolic --skip-install
# Outputs default to a fresh temporary directory. See docs/src/symbolic_filters.md.
using TOML, Test
const BACKEND_LOAD_START = time()
if "--worker" in ARGS
    using Symbolics, SymbolicIntegration
else
    using QuantumFurnace, QuadGK
end
const BACKEND_LOAD_SECONDS = time()-BACKEND_LOAD_START

const PROBE_ROOT = dirname(@__DIR__)
const PROBE_CASES = ("gaussian", "shifted_complex_gaussian", "piecewise_rate", "compact_bump")
option(name, default) = let prefix = "--$name=", hits = filter(a -> startswith(a,prefix), ARGS)
    isempty(hits) ? default : only(hits)[length(prefix)+1:end]
end
function write_report(path, data)
    open(path,"w") do io
        TOML.print(io,data;sorted=true)
    end
end

# Separate processes allow interruption even inside non-yielding symbolic code.
# Workers use existing caches only: killing one cannot orphan precompile workers.
function capped_process(cmd, logfile; startup_cap=120., work_cap=60., ready=nothing)
    started=time(); work_started=nothing; status="running"
    open(logfile,"w") do io
        proc=run(pipeline(cmd;stdout=io,stderr=io);wait=false)
        while process_running(proc)
            if ready !== nothing && work_started === nothing && isfile(ready)
                work_started=time()
            end
            if (work_started === nothing && time()-started > startup_cap) ||
               (work_started !== nothing && time()-work_started > work_cap)
                status=work_started === nothing ? "startup_timeout" : "timeout"
                kill(proc,Base.SIGKILL)
                break
            end
            sleep(0.05)
        end
        wait(proc)
        status == "running" && (status=success(proc) ? "completed" : "process_error")
    end
    return Dict("status"=>status,"wall_seconds"=>time()-started,
        "startup_seconds"=>work_started === nothing ? time()-started : work_started-started)
end

function symbolic_body(name,outfile,ready)
    x,t=Core.eval(Main, :(Symbolics.@variables x t))
    report=Dict{String,Any}("case"=>name,"julia"=>string(VERSION),
        "symbolics"=>string(Base.pkgversion(Symbolics)),
        "symbolic_integration"=>string(Base.pkgversion(SymbolicIntegration)),
        "status"=>"started", "expressions"=>Any[])
    write_report(outfile,report)
    write(ready,"ready")
    start=time()
    # Split into real/imaginary integrands and actual smooth support branches.
    # This is an explicit algebraic adapter, not an integration-engine result.
    parts = if name == "gaussian"
        [(exp(-x^2)*cos(t*x),-8.,8.,"real")]
    elseif name == "shifted_complex_gaussian"
        [(exp(-x^2-(1//5)*x)*cos((3//10-t)*x),-8.,8.,"real"),
         (exp(-x^2-(1//5)*x)*sin((3//10-t)*x),-8.,8.,"imag")]
    elseif name == "piecewise_rate"
        [(exp(-(9//10)*x)*cos(t*x),0.,320.,"real"),
         (-exp(-(9//10)*x)*sin(t*x),0.,320.,"imag"),
         (exp((1//10)*x)*cos(t*x),-320.,0.,"real"),
         (-exp((1//10)*x)*sin(t*x),-320.,0.,"imag")]
    else
        [(exp(-1/(1-x^2)-(2//5)*x)*cos(t*x),-1.,1.,"real"),
         (-exp(-1/(1-x^2)-(2//5)*x)*sin(t*x),-1.,1.,"imag")]
    end
    # Represent the actual support of every branch; a positive half-line
    # exponential would diverge if it were incorrectly represented on all R.
    domains=[name == "compact_bump" ? (-1.,1.) :
        (iszero(left) ? 0. : -Inf,iszero(right) ? 0. : Inf)
        for (_,left,right,_) in parts]
    report["integral_representations"]=[string(Symbolics.Integral(
        x in Symbolics.DomainSets.ClosedInterval(lo,hi))(part[1]))
        for (part,(lo,hi)) in zip(parts,domains)]
    # The standard API is indefinite. Do not invent a definite integration API.
    report["definite_api_available"]=applicable(SymbolicIntegration.integrate,parts[1][1],x,domains[1]...)
    write_report(outfile,report)
    for (integrand,left,right,component) in parts
        row=Dict{String,Any}("integrand"=>string(integrand),"component"=>component,
            "left"=>left,"right"=>right)
        push!(report["expressions"],row)
        write_report(outfile,report)
        tic=time()
        try
            primitive=SymbolicIntegration.integrate(integrand,x)
            row["runtime_seconds"]=time()-tic
            row["primitive"]=string(primitive)
            # This backend's own recursive detector, not a display-string heuristic.
            unevaluated=SymbolicIntegration.contains_int(Symbolics.unwrap(primitive))
            row["status"]=unevaluated ? "unevaluated" : "candidate_primitive"
            write_report(outfile,report)
            if !unevaluated
                # Substitution can leave constant symbolic sin/cos/log unevaluated.
                # Compile each returned expression once through the supported API.
                functions=Dict()
                function value(expr,a,b)
                    callable=get!(functions,expr) do
                        Symbolics.build_function(expr,x,t;expression=Val{false})
                    end
                    Float64(Base.invokelatest(callable,a,b))
                end
                derivative=Symbolics.expand_derivatives(Symbolics.Differential(x)(primitive))
                xs=left == 0 ? [0.2,0.7] : right == 0 ? [-0.7,-0.2] : [-0.7,0.,0.7]
                ts=[-0.7,0.,0.8]
                row["derivative_error"]=maximum(abs(value(derivative,a,b)-value(integrand,a,b)) for a in xs for b in ts)
                # Finite endpoints test candidate constants/branches; they are not infinity limits.
                row["finite_targets"]=ts
                row["finite_values"]=[(value(primitive,right,b)-value(primitive,left,b))/(2pi) for b in ts]
                write_report(outfile,report)
                try
                    endpoints = name == "compact_bump" ?
                        (Symbolics.limit(Symbolics.unwrap(primitive),Symbolics.unwrap(x),1.,:left),Symbolics.limit(Symbolics.unwrap(primitive),Symbolics.unwrap(x),-1.,:right)) :
                        (Symbolics.limit(Symbolics.unwrap(primitive),Symbolics.unwrap(x),iszero(right) ? 0. : Inf),
                         Symbolics.limit(Symbolics.unwrap(primitive),Symbolics.unwrap(x),iszero(left) ? 0. : -Inf))
                    expr=(endpoints[1]-endpoints[2])/(2pi)
                    row["endpoint_expression"]=string(expr)
                    row["endpoint_values"]=[value(expr,0.,b) for b in ts]
                    row["endpoint_status"]="candidate_evaluated"
                catch err
                    row["endpoint_status"]="unsupported"
                    row["endpoint_error"]=sprint(showerror,err)
                end
            end
        catch err
            row["status"]="error"
            row["error"]=sprint(showerror,err)
            row["runtime_seconds"]=time()-tic
        end
        write_report(outfile,report)
    end
    report["runtime_seconds"]=time()-start
    report["status"]="completed"
    write_report(outfile,report)
end

function numerical_body(symbolic_reports)
    QF=QuantumFurnace
    targets=[-0.7,0.,0.8]
    amplitudes=(x->exp(-x^2), x->exp(-x^2-.2x)*cis(.3x),
        x->exp(-abs(x)/2-.4x), x->abs(x)<1 ? exp(-1/(1-x^2)-.4x) : 0.)
    inverses=(t->exp(-t^2/4)/(2sqrt(pi)),
        t->exp((-.2+im*(.3-t))^2/4)/(2sqrt(pi)),
        t->(inv(.9+im*t)+inv(.1-im*t))/(2pi))
    result=Dict{String,Any}()
    @testset "T21 numerical provider and returned symbolic candidates" begin
        for (i,name) in enumerate(PROBE_CASES)
            # beta=1 is inert positive metadata for the symmetric Fourier-only anchor.
            beta=i == 1 ? 1. : i == 2 ? .8 : 1.6
            kwargs=i == 4 ? (;support=1.) : (;)
            filter=QF.FrequencyFilter(beta;amplitude=amplitudes[i],name=Symbol(name),kwargs...)
            W=i == 3 ? 320. : 8.
            prepared=QF.prepare_filter_transform(filter;window=W,analytic=false)
            tic=time()
            measured=QF.transform_values(prepared,targets;window_refinements=0)
            elapsed=time()-tic
            @test measured.status == :estimated
            @test maximum(measured.quadrature_estimates)<1e-10
            refined=QF.prepare_filter_transform(filter;window=W,analytic=false,rtol=1e-13,atol=1e-14)
            @test QF.transform_values(refined,targets;window_refinements=0).values ≈ measured.values atol=1e-10 rtol=1e-10
            # Repeat fallback from an independent preparation, without a symbolic expression.
            @test QF.transform_values(QF.prepare_filter_transform(filter;window=W,analytic=false),targets;
                window_refinements=0).values ≈ measured.values atol=1e-13 rtol=1e-13
            row=Dict{String,Any}("inverse_seconds"=>elapsed,"window"=>measured.window,
                "quadrature_estimate"=>maximum(measured.quadrature_estimates),
                "tail_status"=>string(measured.tail_status))
            if i <= 3
                err=maximum(abs.(measured.values-inverses[i].(targets)))
                @test err<1e-9
                row["analytic_inverse_error"]=err
                tw=i == 3 ? 400. : 16.
                tail=i == 3 ? (T->1/(pi*T)) : (T->exp(.01-max(T-.3,0)^2/4))
                tf=QF.TimeFilter(beta;kernel=inverses[i],name=Symbol(name*"_time"),tail_bound=tail)
                forward=QF.transform_values(QF.prepare_filter_transform(tf;window=tw),[-1.,0.,1.];
                    direction=:forward,window_refinements=0)
                ferr=maximum(abs.(forward.values-amplitudes[i].([-1.,0.,1.])))
                @test ferr <= forward.tail_bound+1e-9
                i <= 2 && @test ferr<1e-9
                row["forward_window"]=tw; row["forward_error"]=ferr
                row["forward_tail_bound"]=forward.tail_bound
            else
                # Independent nonoscillatory/oscillatory QuadGK reference on exact support.
                direct=[QuadGK.quadgk(x->amplitudes[i](x)*cis(-t*x)/(2pi),-1.,0.,1.;rtol=1e-13,atol=1e-14)[1] for t in targets]
                @test direct ≈ measured.values atol=1e-11 rtol=1e-11
                row["direct_inverse_error"]=maximum(abs.(direct-measured.values))
                # Numerical inversion of the numerical inverse; independently expand
                # the time window. No analytic bump transform or tail certificate.
                tf=QF.TimeFilter(beta;kernel=t->QuantumFurnace.time_kernel(prepared,t),name=:bump_time)
                errors=Float64[]
                for tw in (128.,512.)
                    forward=QF.transform_values(QF.prepare_filter_transform(tf;window=tw,
                        atol=1e-12,rtol=1e-11),[-.5,0.,.5];direction=:forward,window_refinements=0)
                    @test forward.status == :estimated
                    @test forward.tail_status == :unknown
                    push!(errors,maximum(abs.(forward.values-amplitudes[i].([-.5,0.,.5]))))
                end
                @test errors[2]<errors[1]
                @test errors[2]<1e-9
                row["forward_windows"]=[128.,512.]
                row["forward_errors"]=errors
                row["forward_tail_status"]="unknown"
            end
            report=get(symbolic_reports,name,Dict())
            exprs=get(report,"expressions",[])
            evaluated=0
            for e in exprs
                if haskey(e,"derivative_error")
                    @test isfinite(e["derivative_error"]) && e["derivative_error"] < 1e-9
                end
                if haskey(e,"finite_values")
                    # Independently integrate exactly the same support branch and component.
                    part=e["component"] == "real" ? real : imag
                    reference=[QuadGK.quadgk(x->part(amplitudes[i](x)*cis(-t*x))/(2pi),
                        e["left"],0. in (e["left"],e["right"]) ? e["right"] : 0.,e["right"];
                        rtol=1e-12,atol=1e-13)[1] for t in targets]
                    @test e["finite_values"] ≈ reference atol=1e-9 rtol=1e-9
                    if get(e,"endpoint_status","") == "candidate_evaluated"
                        @test e["endpoint_values"] ≈ reference atol=1e-9 rtol=1e-9
                        evaluated+=1
                    end
                end
            end
            row["evaluated_endpoint_parts"]=evaluated
            row["required_parts"]=i == 1 ? 1 : i == 3 ? 4 : 2
            result[name]=row
            println(name,": ",row)
        end
        # Unknown continuum tails stay unknown; a cap must not silently relax accuracy.
        f=QF.FrequencyFilter(1.;amplitude=x->exp(-x^2),name=:budget)
        p=QF.prepare_filter_transform(f;window=8.,max_panels=2,analytic=false)
        @test_throws ArgumentError QF.time_kernel(p,100.)
        @test QF.transform_values(QF.prepare_filter_transform(f;window=8.),[0.]).tail_status == :unknown
    end
    result
end

function main()
    if "--worker" in ARGS
        println("backend_load_seconds=",BACKEND_LOAD_SECONDS)
        Base.invokelatest(symbolic_body,option("case","gaussian"),option("report","result.toml"),option("ready","ready"))
        return
    end
    out=option("output",""); env=option("env","")
    out=abspath(isempty(out) ? mktempdir(;cleanup=false) : out)
    env=abspath(isempty(env) ? mktempdir(;cleanup=false) : env)
    mkpath(out); mkpath(env)
    realpath(env) in (realpath(PROBE_ROOT),realpath(joinpath(PROBE_ROOT,"docs"))) &&
        error("Use a disposable environment, never the package or docs environment.")
    cap=parse(Float64,option("timeout","60")); startup=parse(Float64,option("startup-timeout","180"))
    all(x->isfinite(x)&&x>0,(cap,startup)) || error("Timeouts must be positive finite seconds.")
    julia=Base.julia_cmd()
    println("Reports: ",out,"\nDisposable symbolic environment: ",env)
    summary=Dict{String,Any}("julia"=>string(VERSION),"work_cap_seconds"=>cap,"startup_cap_seconds"=>startup)
    if !("--skip-install" in ARGS)
        println("Installing optional probe-only packages in ",env)
        install=`$julia --startup-file=no --heap-size-hint=1500M --project=$env -e 'using Pkg; Pkg.add([Pkg.PackageSpec(name="Symbolics", version="7.39.0"), Pkg.PackageSpec(name="SymbolicIntegration", version="3.7.0")]; allow_autoprecomp=false); Pkg.status()'`
        summary["installation"]=capped_process(install,joinpath(out,"install.log");startup_cap=240.)
    else
        isfile(joinpath(env,"Project.toml")) || error("--skip-install requires an existing disposable environment.")
    end
    for filename in ("Project.toml","Manifest.toml")
        source=joinpath(env,filename)
        isfile(source) && cp(source,joinpath(out,"symbolic_"*filename);force=true)
    end
    # Test the watchdog on real nonterminating code, not a mocked timeout return.
    ready=joinpath(out,"watchdog.ready"); rm(ready;force=true)
    control=joinpath(out,"watchdog_control.jl")
    write(control,"write(ARGS[1], \"ready\"); while true; sleep(0.1); end\n")
    watchdog=capped_process(`$julia --startup-file=no $control $ready`,joinpath(out,"watchdog.log");
        startup_cap=startup,work_cap=.2,ready)
    @test watchdog["status"] == "timeout"
    summary["watchdog"]=watchdog
    reports=Dict{String,Any}()
    for name in PROBE_CASES
        ready=joinpath(out,name*".ready"); rm(ready;force=true)
        file=joinpath(out,name*".toml"); rm(file;force=true)
        cmd=`$julia --startup-file=no --compiled-modules=existing --heap-size-hint=1500M --project=$env $(@__FILE__) --worker --case=$name --report=$file --ready=$ready`
        process=capped_process(cmd,joinpath(out,name*".log");startup_cap=startup,work_cap=cap,ready)
        report=isfile(file) ? TOML.parsefile(file) : Dict{String,Any}()
        report["process"]=process
        reports[name]=report
        println(name," symbolic: ",process)
    end
    summary["symbolic"]=reports
    write_report(joinpath(out,"summary.toml"),summary)
    @test !isdefined(Main,:Symbolics)
    @test Base.get_extension(QuantumFurnace,:QuantumFurnaceSymbolicsExt) === nothing
    summary["numerical"]=Base.invokelatest(numerical_body,reports)
    summary["fully_evaluated_families"]=count(v->v["evaluated_endpoint_parts"]==v["required_parts"],values(summary["numerical"]))
    summary["decision"]="no_extension_without_demonstrated_benefit"
    write_report(joinpath(out,"summary.toml"),summary)
    println("Verified numerical fallback; evaluated symbolic families: ",summary["fully_evaluated_families"],"/4")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
