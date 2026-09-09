using Test, QuantumFurnace, LinearAlgebra, QuadGK

# Independent component-index reference, without production alpha/generator
# contraction helpers. Matrices act on column-stacked density operators.
function ckg_equivalence_reference(alpha,nus,beta,ham,jumps)
    d=length(ham.eigvals); lookup=Dict(u=>i for (i,u) in enumerate(nus))
    index(i,j)=lookup[ham.eigvals[i]-ham.eigvals[j]]
    gain=zeros(ComplexF64,d^2,d^2); R=zeros(ComplexF64,d,d)
    for source in jumps
        A=source.in_eigenbasis
        for i in 1:d,j in 1:d,k in 1:d,l in 1:d
            gain[i+d*(k-1),j+d*(l-1)]+=alpha[index(i,j),index(k,l)]*A[i,j]*conj(A[k,l])
        end
        for i in 1:d,j in 1:d,k in 1:d
            R[i,j]+=alpha[index(k,j),index(k,i)]*conj(A[k,i])*A[k,j]
        end
    end
    B=[im/2*tanh(beta*(ham.eigvals[i]-ham.eigvals[j])/4)*R[i,j] for i in 1:d,j in 1:d]
    Id=Matrix{ComplexF64}(I,d,d)
    L=gain-(kron(Id,R)+kron(transpose(R),Id))/2-im*kron(Id,B)+im*kron(transpose(B),Id)
    (;R,B,L)
end

@testset "finite-Bohr CKG to DLL reference" begin
    H=Matrix(Diagonal([0.,0.25,0.75,1.25]))
    A=ComplexF64[0.2 0.3im 0.1 0.7; 0.1 -0.2 0.4im 0.2;
        0.3 0.0 0.2 0.1im; -0.2im 0.1 0.5 0.3]
    p=prepare_gibbs_inputs(H;beta_phys=0.8,jumps=[A,Matrix(A')])
    ham=p.hamiltonian; jumps=p.jumps; beta=p.config.beta
    nus=sort!(collect(keys(ham.bohr_dict))); m=length(nus)
    options=(;hamiltonian=ham,jumps)
    tilt=exp.(-beta.*nus./4)
    ranks=Int[]
    for rate in (GaussianTransition(beta;sigma=0.4,sigma_gamma=0.6),
                 MetropolisTransition(beta;sigma=0.4))
        # Primary alpha reference is an independent normalized-OFT integral.
        C(x)=(2pi*rate.sigma^2)^(-1/4)*exp(-x^2/(4rate.sigma^2))
        a=ComplexF64[first(quadgk(w->transition_value(rate,w)*C(w-u)*conj(C(w-v)),
            -Inf,-beta*rate.sigma^2/2,Inf;atol=1e-13,rtol=1e-12)) for u in nus,v in nus]
        result=QuantumFurnace.ckg_to_dll(a,nus,beta;options...)
        push!(ranks,result.evidence.retained_rank)
        @test result.evidence.retained_rank>1
        @test result.alpha≈a atol=1e-11 rtol=1e-11
        @test QuantumFurnace.dll_kossakowski_bohr(result.filter,nus)≈a atol=1e-11 rtol=1e-11
        ref=ckg_equivalence_reference(a,nus,beta,ham,jumps)
        operators=[L for jump in jumps for L in dll_lindblad_op_bohr(jump,ham,result.filter)]
        loss=sum(L'*L for L in operators)
        @test loss≈ref.R atol=1e-11 rtol=1e-11
        @test dll_coherent_op_bohr(jumps,ham,result.filter,beta)≈ref.B atol=1e-11 rtol=1e-11
        cfg=Config(;sim=Lindbladian(),domain=BohrDomain(),construction=DLL(),
            num_qubits=2,beta,sigma=inv(beta),with_linear_combination=false,filter=result.filter)
        actual=construct_lindbladian(jumps,cfg,ham)
        @test actual≈ref.L atol=1e-10 rtol=1e-10
        @test norm(actual*vec(ham.gibbs))<1e-10
        @test result.evidence.generator_error_norms.total<1e-10
        # Analytic production CKG gives a third independent construction.
        ckg=Config(;sim=Lindbladian(),domain=BohrDomain(),construction=KMS(),
            num_qubits=2,beta,sigma=rate.sigma,QuantumFurnace._ckg_legacy_fields(rate)...,transition_weight=rate)
        @test construct_lindbladian(jumps,ckg,ham)≈actual atol=1e-10 rtol=1e-10
        truncated=QuantumFurnace.ckg_to_dll(a,nus,beta;options...,max_rank=1)
        partial=ckg_equivalence_reference(truncated.alpha,nus,beta,ham,jumps)
        @test truncated.evidence.retained_rank==1
        @test truncated.evidence.coefficient_truncation_norm>1e-4
        @test truncated.evidence.generator_error_norms.total≈opnorm(partial.L-ref.L) atol=1e-12 rtol=1e-11
        @test truncated.evidence.coefficient_total_error_norm≈opnorm(truncated.alpha-a) atol=1e-13
        @test truncated.evidence.generator_error_norms.total<=
            truncated.evidence.generator_error_norms.repair+truncated.evidence.generator_error_norms.truncation+1e-12
    end
    @testset "Rank-one, imaginary odd, complex and degenerate channels" begin
        q=ComplexF64[(1+0.2im*u)*exp(-u^2) for u in nus]
        anchor=(tilt.*q)*(tilt.*q)'
        result=QuantumFurnace.ckg_to_dll(anchor,nus,beta)
        @test result.evidence.retained_rank==1
        @test result.alpha≈anchor atol=1e-12 rtol=1e-12
        odd=im.*nus.*exp.(-nus.^2)
        result=QuantumFurnace.ckg_to_dll((tilt.*odd)*(tilt.*odd)',nus,beta)
        @test result.evidence.retained_rank==1
        @test abs(freq_kernel(only(result.filter.channels),0.))<1e-13
        @test maximum(abs(imag(freq_kernel(only(result.filter.channels),u))) for u in nus)>0.01
        # The identity tilted matrix has a completely degenerate eigenvalue.
        degenerate=Matrix(Diagonal(tilt.^2))
        result=QuantumFurnace.ckg_to_dll(degenerate,nus,beta)
        @test result.evidence.retained_rank==m
        @test result.alpha≈degenerate atol=1e-12
        @test any(maximum(abs(imag(freq_kernel(c,u))) for u in nus)>0.1 for c in result.filter.channels)
        # Use exactly representable frequencies: casting a fine Float64 Bohr
        # set can merge distinct rows and correctly requires rebuilding alpha.
        nus32=Float32[-0.5,0,0.5]
        alpha32=Matrix(Diagonal(exp.(-Float32(beta).*nus32./2)))
        result32=QuantumFurnace.ckg_to_dll(alpha32,nus32,Float32(beta))
        @test eltype(result32.alpha)===ComplexF32
        @test result32.alpha≈alpha32 atol=1e-6 rtol=1e-6
        complex_alpha=anchor+degenerate
        @test norm(imag(complex_alpha))>0.1
        permutation=reverse(collect(1:m))
        result=QuantumFurnace.ckg_to_dll(complex_alpha[permutation,permutation],nus[permutation],beta;options...)
        @test result.alpha≈complex_alpha[permutation,permutation] atol=1e-11
        ref=ckg_equivalence_reference(complex_alpha,nus,beta,ham,jumps)
        cfg=Config(;sim=Lindbladian(),domain=BohrDomain(),construction=DLL(),
            num_qubits=2,beta,sigma=inv(beta),with_linear_combination=false,filter=result.filter)
        @test construct_lindbladian(jumps,cfg,ham)≈ref.L atol=1e-10 rtol=1e-10
        @test dll_coherent_op_bohr(jumps,ham,result.filter,beta)≈ref.B atol=1e-11 rtol=1e-11
        @test sum(L'*L for source in jumps for L in dll_lindblad_op_bohr(source,ham,result.filter))≈ref.R atol=1e-11 rtol=1e-11
        for channel in result.filter.channels,u in nus
            @test freq_kernel(channel,u)≈exp(-beta*u/2)*conj(freq_kernel(channel,iszero(u) ? 0. : -u)) atol=1e-12
        end
        @test !QuantumFurnace._dll_time_supported(result.filter)
        @test_throws KeyError freq_kernel(first(result.filter.channels),42.)
        zero_result=QuantumFurnace.ckg_to_dll(zeros(m,m),nus,beta)
        @test zero_result.evidence.retained_rank==0
        @test zero_result.evidence.channel_count==1
        @test opnorm(zero_result.alpha)<1e-15
    end
    @testset "Numerical repairs are bounded and reported" begin
        # Exact reflection-invariant diagonal with tiny negative origin mode.
        weights=ones(m); origin=findfirst(iszero,nus); weights[origin]=-1e-14
        a=Matrix(Diagonal((tilt.^2).*weights))
        result=QuantumFurnace.ckg_to_dll(a,nus,beta;options...)
        @test result.evidence.minimum_tilted_eigenvalue<0
        @test 0<result.evidence.coefficient_repair_norm<1e-12
        @test result.evidence.coefficient_truncation_norm<1e-14
        perturbed=complex.(Matrix(Diagonal(tilt.^2))); perturbed[1,1]+=1e-14
        perturbed[1,2]+=1e-14im
        symmetry_repair=QuantumFurnace.ckg_to_dll(perturbed,nus,beta)
        @test symmetry_repair.evidence.tilted_hermiticity_defect>0
        @test symmetry_repair.evidence.tilted_reflection_defect>0
        @test 0<symmetry_repair.evidence.coefficient_repair_norm<1e-12
        weights[origin]=1e-4
        compressed=QuantumFurnace.ckg_to_dll(Matrix(Diagonal((tilt.^2).*weights)),nus,beta;
            options...,rank_rtol=0.1)
        @test compressed.evidence.retained_rank==m-1
        @test compressed.evidence.coefficient_truncation_norm≈1e-4 atol=1e-12
        weights[origin]=-1e-4
        @test_throws ArgumentError QuantumFurnace.ckg_to_dll(Matrix(Diagonal((tilt.^2).*weights)),nus,beta)
        @test_throws ArgumentError QuantumFurnace.ckg_to_dll(a,nus,beta;roundoff_rtol=1e-2)
        bad=Matrix(Diagonal(tilt.^2)); bad[1,1]+=0.01
        @test_throws ArgumentError QuantumFurnace.ckg_to_dll(bad,nus,beta)
        bad=complex.(Matrix(Diagonal(tilt.^2))); bad[1,2]=0.01im
        @test_throws ArgumentError QuantumFurnace.ckg_to_dll(bad,nus,beta)
        for extra in ((;max_bytes=100),(;max_bohr_frequencies=2),(;max_rank=1),
            (;rank_rtol=0.1),(;hamiltonian=ham),(;beta=-1.))
            if haskey(extra,:beta)
                @test_throws ArgumentError QuantumFurnace.ckg_to_dll(a,nus,extra.beta)
            else
                @test_throws ArgumentError QuantumFurnace.ckg_to_dll(a,nus,beta;extra...)
            end
        end
        @test_throws ArgumentError QuantumFurnace.ckg_to_dll(a[2:end,2:end],nus[2:end],beta)
        @test_throws ArgumentError QuantumFurnace.ckg_to_dll(fill(NaN,m,m),nus,beta)
        @test_throws ArgumentError QuantumFurnace.ckg_to_dll(a,nus,1e8)
    end
    @testset "Prepared complex CKG kernel and finite scope" begin
        normalization=inv(sqrt(first(quadgk(x->exp(beta*x/2-2x^4),-Inf,Inf;rtol=1e-12))))
        C=x->normalization*exp(beta*x/4-x^4)*cis(0.3x)
        rate=w->exp(-w^2-beta*w/2)
        joint=CKGJointKernel(beta;oft=C,rate,frequency_window=(-8.,8.),panels=48)
        prepared=compile_ckg_kernel(joint,nus)
        result=QuantumFurnace.ckg_to_dll(prepared;options...)
        @test result.alpha≈prepared.alpha atol=1e-10 rtol=1e-10
        @test result.evidence.generator_error_norms.total<1e-10
        ref=ckg_equivalence_reference(prepared.alpha,nus,beta,ham,jumps)
        cfg=Config(;sim=Lindbladian(),domain=BohrDomain(),construction=DLL(),
            num_qubits=2,beta,sigma=inv(beta),with_linear_combination=false,filter=result.filter)
        @test construct_lindbladian(jumps,cfg,ham)≈ref.L atol=1e-10 rtol=1e-10
        invalid=CKGJointKernel(beta;oft=x->(2pi)^(-1/4)*exp(-x^2/4),rate,frequency_window=(-8.,8.))
        failed=compile_ckg_kernel(invalid,nus;strict=false)
        @test_throws ArgumentError QuantumFurnace.ckg_to_dll(failed)
    end
end
