module PCE

using   PolyChaos
import  PolyChaos: showbasis, showpoly, rec2coeff, convert2affinePCE, assign2multi
using   SparseArrayKit: SparseArray

using ..typesMeasureParametric: AbstractCanonicalMeasureParametric,
                                DiracMeasureParametric, 
                                GaussMeasureParametric, 
                                UniformMeasureParametric,
                                BetaMeasureParametric,
                                GammaMeasureParametric

using   ..typesOrthonoPoly: AbstractOrthonoPoly, 
                            AbstractCanonicalOrthonoPoly,  
                            ConstantOrthonoPoly, 
                            HermiteOrthonoPoly, 
                            LegendreOrthonoPoly, 
                            JacobiOrthonoPoly, 
                            LaguerreOrthonoPoly, 
                            OrthonoPoly,
                            MultiOrthonoPoly


struct InconsistencyError <: Exception
    var::String
end

struct OrthonoPCE{P<:AbstractOrthonoPoly, VM<:Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}}}
    basis::P
    coeff::VM # vector or matrix
end
                            
# show orthonormal basis functions
function showbasis(onp::AbstractOrthonoPoly; sym::String = "x", digits::Integer = 2)
    showbasis(onp.α, onp.β, onp.sp; sym = sym, digits = digits)
end

function showbasis(onpPCE::OrthonoPCE; sym::String = "x", digits::Integer = 2)
    showbasis(onpPCE.basis; sym = sym, digits = 2)
end

function showbasis(monp::MultiOrthonoPoly; sym::String = "x", digits::Integer = 2)
    ind = monp.ind
    mα  = getfield.(monp.uni, :α)
    mβ  = getfield.(monp.uni, :β)
    msp = getfield.(monp.uni, :sp)

    showbasis(ind, mα, mβ, msp; sym=sym, digits=digits)
end

function showbasis(ind::AbstractMatrix{Int}, mα::Vector{<:AbstractVector}, mβ::Vector{<:AbstractVector}, msp::Vector{<:AbstractVector};
                    sym::String = "x", digits::Integer = 2)
    io = Base.stdout
    for j = 1:size(ind,1)
        print(io, "ϕ^", j-1, ": ")
        showmultipoly(ind[j,:], mα, mβ, msp; sym = sym, digits = digits)
    end
    print()
end

function showbasis(α::Vector{<:Real}, β::Vector{<:Real}, sp::Vector{<:Real}; sym::String = "x", digits::Integer = 2)
    showpoly(0:length(sp)-1, α, β, sp; sym = sym, digits = digits)
end

function showbasis_normalized(α::Vector{<:Real}, β::Vector{<:Real}; sym::String = "x", digits::Integer = 2)
    length(α) == length(β) ||
        throw(InconsistencyError("incorrect number of recurrence coefficients"))
    sp = computeSP2(length(β-1), β)
    showpoly(0:length(sp)-1, α, β, sp; sym = sym, digits = digits)
end

function showpoly(d::Union{Integer, Range}, op::AbstractOrthonoPoly; sym::String = "x", digits::Integer = 2) where {Range <: OrdinalRange}
    showpoly(d, op.α, op.β, op.sp; sym = sym, digits = digits)
end

function showpoly(d::Range, α::Vector{<:Real}, β::Vector{<:Real}, sp::Vector{<:Real}; sym::String = "x", digits::Integer = 2) where {Range <: OrdinalRange}
    minimum(d) >= 0 ||
        throw(DomainError("degree must be positive"))
    length(α) == length(β) ||
        throw(InconsistencyError("incorrect number of recurrence coefficients"))
    length(α) >= maximum(d)+1 ||
        throw(InconsistencyError("maximum degree is larger than the recurrence coefficients"))

    map(c -> showpoly(c, α, β, sp; sym = sym, digits = digits), d)
    print()
end

function showpoly(d::Integer, α::Vector{<:Real}, β::Vector{<:Real}, sp::Vector{<:Real}; sym::String = "x", digits::Integer = 2)
    @assert d>=0 "degree has to be non-negative."
    d == 0 && return print("1\n")
    showpoly_normalized(rec2coeff(d, α, β, sp)[end], sym = sym, digits = digits, newline = true)
end

function showmultipoly(indj::AbstractVector{Int}, mα::Vector{<:AbstractVector}, mβ::Vector{<:AbstractVector}, msp::Vector{<:AbstractVector};
                    sym::String = "x", digits::Integer = 2)
        io  = Base.stdout
        ind_active = findall(!=(0), indj)
        n = length(ind_active)
        isempty(ind_active) && return print(io, "1\n")
        
        if n == 1
            i = ind_active[1]
            showpoly(indj[i], mα[i], mβ[i], msp[i]; sym = sym * "_$i", digits = digits)
        else
            for i = 1:n
                print(io, '(')
                showpoly_normalized(rec2coeff(indj[i], mα[i], mβ[i], msp[i])[end]; sym = sym * "_$i", digits = digits, newline=false)
                print(io, i < n ? ") * " : ")")
            end
            print(io, '\n')
        end
    print()
end

function showpoly_normalized(d::Range, α::Vector{<:Real}, β::Vector{<:Real}; sym::String = "x", digits::Integer = 2) where {Range <: OrdinalRange}
    sp = computeSP2(length(β-1), β)
    map(c -> showpoly(c, α, β, sp; sym = sym, digits = digits), d)
    print()
end

function showpoly_normalized(coeffs::Vector{<:Real}; sym::String = "x", digits::Integer = 2, newline::Bool = true)
    io = Base.stdout
    length(coeffs) > 2 ? print(io, round(coeffs[end], sigdigits=digits), sym * "^", length(coeffs)-1) : print(io, round(coeffs[end], sigdigits=digits), sym)
    for (i, c) in enumerate(reverse(coeffs[1:end-1]))
        abs(round(c, sigdigits = digits)) == 0.0 && continue
        ex = length(coeffs) - i - 1
        print(io, ' ', c > 0 ? '+' : '-', ' ')
        print(io, abs(round(c, sigdigits = digits)))
        ex > 0 && print(io, sym)
        ex > 1 && print(io, '^', ex)
    end
    newline == true && print(io, '\n')
end

function rec2coeff(deg::Int, a::Vector{<:Real}, b::Vector{<:Real}, sp::Vector{<:Real})
    deg <= 0 &&
        throw(DomainError(deg, "degree must be positive (you asked for degree = $deg"))
    !(length(a) == length(b) && length(a) >= deg+1) &&
        throw(InconsistencyError("incorrect number of recurrence coefficients"))
    !(length(a) >= length(sp) && length(sp) >= deg+1) &&
        throw(InconsistencyError("incorrect number of scalar products of polynomials"))
    c = Vector{Vector{<:Real}}(undef, deg)
    norm2 = sqrt.(sp[2:deg+1])

    @inbounds c[1] = [-a[1]]
    deg == 1 && return push!.(c, 1) ./ norm2[deg] 
    @inbounds c[2] = [a[1] * a[2] - b[2], -a[1] - a[2]]
    deg == 2 && return push!.(c, 1) ./ norm2[1:deg]
    for k in 2:(deg - 1)
        c[k + 1] = [-a[k + 1] * c[k][1] - b[k + 1] * c[k - 1][1]]
        for i in 1:(k - 2)
            push!(c[k + 1], c[k][i] - a[k + 1] * c[k][i + 1] - b[k + 1] * c[k - 1][i + 1])
        end
        push!(c[k + 1], c[k][k - 1] - a[k + 1] * c[k][k] - b[k + 1])
        push!(c[k + 1], c[k][k] - a[k + 1])
    end

    push!.(c, 1)
    c = [ci ./ ni for (ci, ni) in zip(c, norm2)]
    return c
end
rec2coeff(α, β, sp) = rec2coeff_normalized(length(sp-1), α, β, sp)

# this function can be extended to measures defined in PolyChaos.jl
function  measure2poly(m::AbstractCanonicalMeasure) # m -> measure
    if isa(m, DiracMeasureParametric)
            return ConstantOrthonoPoly()
        elseif isa(m, GaussMeasureParametric)
            return HermiteOrthonoPoly(1)
        elseif isa(m, UniformMeasureParametric)
            return LegendreOrthonoPoly(1)
        elseif isa(m, BetaMeasureParametric)
            return JacobiOrthonoPoly(1, m.pars...)
        elseif isa(m, GammaMeasureParametric)
            return LaguerreOrthonoPoly(1, m.pars...)
        else
            throw(ArgumentError("Unsupported measure type: $(typeof(m))"))
        end
end

"""
Generate PCE of a vectorized random variable
The joint PCE basis depends on the dependence of elements in the random variable 
"""

function genPCE(m::AbstractCanonicalMeasureParametric)
    onp     = measure2poly(m)
    coeff   = convert2affinePCE(m.pars..., onp) 

    OrthonoPCE{AbstractOrthonoPoly, AbstractVector{<:Real}}(
        onp,
        coeff)
end

function genPCE(monp::MultiOrthonoPoly, coeffs::Vector{<:AbstractVector})
    all(c -> c isa Vector{<:Real}, coeffs) ||
        throw(ArgumentError("All coefficients must be real-valued vectors"))
    n = length(monp.name)
    n == length(coeffs) || 
        throw(ArgumentError("Number of variants $n must match number of coefficient vectors $(length(coeffs))."))

    coeff_sparse = [assign2multi(coeffs[i], i, monp) for i = 1:n]
    # coeff = vcat(collect.(coeff_sparse)'...) # convert to dense matrix
    coeff = vcat(coeff_sparse'...)

    OrthonoPCE{AbstractOrthonoPoly, AbstractMatrix{<:Real}}(
        monp,
        coeff)
end

function genPCE(vecMeasure::AbstractVector{<:AbstractCanonicalMeasureParametric})
    vecPoly = measure2poly.(vecMeasure)
    monp    = MultiOrthonoPoly(vecPoly)
    coeffs  = [convert2affinePCE(m.pars..., monp.uni[i]) for (i,m) in enumerate(vecMeasure)]
    
    genPCE(monp, coeffs)
end

function genPCE(uniPCEs::AbstractVector{<:OrthonoPCE})
    monp    = MultiOrthonoPoly(getfield.(uniPCEs, :basis))
    coeffs  = getfield.(uniPCEs, :coeff)
    
    genPCE(monp, coeffs)
end

"""
Extend the function convert2affinePCE from PolyChaos.jl to orthonormal basis functions
Computes the affine PCE coefficients ``x_0`` and ``x_1`` from

```math
X = a_1 + a_2 \\Xi = x_0 + x_1 \\phi_1(\\Xi),
```

where ``\\phi_1(t) = 1/sqrt(sp_1) * (t-\\alpha_0)`` is the first-order monic basis polynomial.

Works for subtypes of AbstractCanonicalOrthoPoly. The keyword `kind in ["lbub", "μσ"]`
specifies whether `p1` and `p2` have the meaning of lower/upper bounds or mean/standard deviation.
"""
# should the following function be renamed to convert2affinePCE_normalized?
function convert2affinePCE(a1::Real, a2::Real, α0::Real, sp1::Real)
    [a1 + α0 * a2; a2/sqrt(sp1)]
end

function convert2affinePCE(c::Real, onp::ConstantOrthonoPoly)
    [c]
end

function convert2affinePCE(mu::Real, sigma::Real, onp::HermiteOrthonoPoly)
    _checkStandardDeviation(sigma)
    convert2affinePCE(mu, sigma, first(onp.α), last(onp.sp))
end

function convert2affinePCE(par1::Real, par2::Real, onp::LegendreOrthonoPoly; kind::String = "lbub")
    kind = _checkKind(kind)
    a1, a2 = if kind == "lbub"
        _checkBounds(par1, par2)
        par1, par2 - par1
    elseif kind == "μσ"
        _checkStandardDeviation(par2)
        par1 - sqrt(3) * par2, 2 * sqrt(3) * par2
    end
    convert2affinePCE(a1, a2, first(onp.α), last(onp.sp))
end

function convert2affinePCE(p1::Real, p2::Real, onp::JacobiOrthonoPoly; kind::String = "lbub")
    kind = _checkKind(kind)
    α, β = onp.measure.pars
    a1, a2 = if kind == "lbub"
        _checkBounds(p1, p2)
        a1, a2 = p1, p2 - p1
    elseif kind == "μσ"
        _checkStandardDeviation(p2)
        a1,
        a2 = p1 - sqrt(α / β) * sqrt(1 + α + β) * p2,
        (α + β) * sqrt((α + β + 1) / (α * β)) * p2
    end
    convert2affinePCE(a1, a2, first(onp.α))
end

function convert2affinePCE(p1::Real, p2::Real, onp::LaguerreOrthonoPoly)
    throw(error("convert2affine not yet implemented for $(typeof(onp))"))
end

# extend the function assign2multi to the case of ConstantOrthonoPoly/0-degree polynomials
function assign2multi(x::AbstractVector{<:Real}, i::Int, monp::MultiOrthonoPoly)
    assign2multi(x, i, monp.ind, monp.dep)
end

function assign2multi(x::AbstractVector{<:Real}, i::Int, ind::AbstractMatrix{<:Int}, dep::Bool)
    if dep == false
        _assign2multi_indep(x, i, ind)
    else
        assign2multi(x, i, ind)
    end
end

function _assign2multi_indep(x::AbstractVector{<:Real}, i::Int, ind::AbstractMatrix{<:Int})
    l, p = size(ind)
    nx   = length(x)
    col  = ind[:, i]
    deg  = maximum(col)
    nx > deg + 1 &&
        throw(DomainError(nx, "inconsistent number of coefficients ($nx vs $(deg+1))"))
    i > p && throw(DomainError((i, p), "basis is $p-variate, you requested $i-variate"))
    if deg > 0
        ind1 = findfirst(==(1), col)
        myind = [1; ind1:ind1+deg-1]
    else
        myind = [1]
    end
    y = SparseArray(zeros(l))
    # y = spzeros(Float64, l)
    y[myind] = x
    
    return y
end

## Generate joint PCE basis and coefficient for the whole horizon
## First consider i.i.d. disturbances
function jointPCE(x0coeff::AbstractMatrix{<:Real}, wcoeff::AbstractMatrix{<:Real}, N)

    nx, Lx  = size(x0coeff)
    nw, Lw  = size(wcoeff)
    L       = Lx + N*(Lw-1)

    x0coeff_joint         = [x0coeff SparseArray(zeros(nx,L-Lx))]
    wcoeff_joint          = SparseArray(zeros(nw, N, L))
    # x0coeff_joint         = [x0coeff SparseArray(zeros(nx,L-Lx))]
    # wcoeff_joint          = SparseArray(zeros(nw, N, L))
    wcoeff_joint[:,:,1]   = wcoeff[:,1]*ones(1,N)
    for k = 1:N
        i                   = (k-1)*(Lw-1)+Lx+1:k*(Lw-1)+Lx
        wcoeff_joint[:,k,i] = wcoeff[:,2:end]
    end

    return x0coeff_joint, wcoeff_joint
end

function jointPCE(x0basis::MultiOrthonoPoly, wbasis::MultiOrthonoPoly, N)
    MultiOrthonoPoly([x0basis.uni[:]; repeat(wbasis.uni, N)])
end

function jointPCE(x0PCE::OrthonoPCE, wPCE::OrthonoPCE, N)

    x0coeff, x0basis    = x0PCE.coeff, x0PCE.basis
    wcoeff, wbasis      = wPCE.coeff, wPCE.basis

    basis_joint = jointPCE(x0basis, wbasis, N)
    x0coeff_joint, wcoeff_joint = jointPCE(x0coeff, wcoeff, N)
    return basis_joint, x0coeff_joint, wcoeff_joint
end

# Then consider non-i.i.d. disturbances
# In the current version, the input can only be PCE coefficients
function jointPCE(x0coeff::AbstractMatrix{<:Real},
                    wcoeff::AbstractVector{T}) where T<:AbstractMatrix{<:Real}
    nx, Lx  = size(x0coeff)
    N       = length(wcoeff)
    nw, _   = size(wcoeff[1])
    Lwk     = [size(wk,2) for wk in wcoeff]
    L       = Lx + sum(Lwk) - N # length of joint basis

    x0coeff_joint   = [x0coeff SparseArray(zeros(nx, L-Lx))]
    wcoeff_joint    = SparseArray(zeros(nw, N, L))

    ## This part can be conviently implemented via blockdiag in SparseArrays and reshape
    ## However, the sparse array of wcoeff is created in SparseArrayKit,
    ## which uses a different type of sparse data and cannot be used in SparseArrays
    Lk = Lx
    for k in 1:N
        wk = wcoeff[k]
        wcoeff_joint[:,k,1] = wk[:,1]

        for (idx,v) in pairs(wk[:,2:end])
            i,j = idx.I
            wcoeff_joint[i, k, Lk + j] = v
        end

        Lk += (Lwk[k]-1)
    end
    return x0coeff_joint, wcoeff_joint
end

function jointPCE(x0coeff::AbstractMatrix{<:Real},
                    wcoeff::AbstractVector{T},
                    N::Int) where T<:AbstractMatrix{<:Real}
    N == length(wcoeff) || 
        throw(ArugmentError("The wcoeff must have the same length as prediction horizon $N (got $(length(wcoeff))"))
    jointPCE(x0coeff, wcoeff)
end

## auxiliary functions from PolyChaos.jl
function _checkKind(kind::String)
    lowercase(kind) ∉ ["lbub", "μσ"] &&
        throw(DomainError(kind, "this kind is not supported"))
    lowercase(kind)
end

function _checkBounds(lb::Real, ub::Real)
    lb >= ub && throw(DomainError((lb, ub), "inconsistent bounds"))
end

# function _createMethodVector(n::Int, word::String = "adaptiverejection")
#     [word for i in 1:n]
# end

# function _createMethodVector(m::ProductMeasure, word::String = "adaptiverejection")
#     _createMethodVector(length(m.measures), word)
# end

# function _createMethodVector(mop::MultiOrthoPoly, word::String = "adaptiverejection")
#     _createMethodVector(mop.measure, word)
# end

function _checkStandardDeviation(σ::Real)
    σ < 0 && throw(DomainError(σ, "σ has to be non-negative"))
    σ < 1e-4 && @warn "σ is close to zero (σ = $σ)"
end


end