"""
This module extends orthogonal polynomials in PolyChaos.jl to orthonormal polynomials
"""
module typesOrthonoPoly

# using PolyChaos: AbstractOrthoPoly, Quad, EmptyQuad
using SpecialFunctions: beta, gamma
using PolyChaos
import PolyChaos: showbasis, showpoly, dim


using ..typesMeasureParametric: DiracMeasureParametric,
                                GaussMeasureParametric,
                                UniformMeasureParametric,
                                BetaMeasureParametric, 
                                GammaMeasureParametric


abstract type   AbstractOrthonoPoly{M <: AbstractMeasure, Q <: AbstractQuad} <:
                AbstractOrthoPoly{M, Q} end
abstract type   AbstractCanonicalOrthonoPoly{V <: AbstractVector{<:Real}, M, Q} <:
                AbstractOrthonoPoly{M, Q} end

struct ConstantOrthonoPoly{V, M, Q} <: AbstractCanonicalOrthonoPoly{V, M, Q}
    deg::Int          # maximum degree (should be 0)
    α::V              # recurrence coefficients (α₀ = 0)
    β::V              # recurrence coefficients (β₀ = 1)
    sp::V
    measure::M        # DiracMeasure
    quad::Q           # quadrature (usually EmptyQuad)
end

function ConstantOrthonoPoly()
    deg = typemax(Int)
    
    α  = [0.0]
    β  = [1.0]
    sp = [1.0]
    quadrature = EmptyQuad()
    
    ConstantOrthonoPoly{promote_type(typeof(α), typeof(β)), DiracMeasureParametric, typeof(quadrature)}(
        deg,
        α,
        β,
        sp,
        DiracMeasureParametric(),
        quadrature)
end

function OrthonoPoly(::DiracMeasureParametric; deg::Int = 0, Nrec::Int = deg + 1, addQuadrature::Bool = true)
    ConstantOrthonoPoly()
end


struct HermiteOrthonoPoly{V, M, Q} <: AbstractCanonicalOrthonoPoly{V, M, Q}
    deg::Int          # maximum degree
    α::V  # recurrence coefficients
    β::V  # recurrence coefficients
    sp::V # scalar product of basis functions
    measure::M
    quad::Q
end

function HermiteOrthonoPoly(deg::Int; Nrec::Int = deg + 1, addQuadrature::Bool = true)
    _checkConsistency(deg, Nrec)
    α, β = r_scale(1 / sqrt(2pi), rm_hermite_prob(Nrec)...)
    sp = computeSP2(deg, β)
    quadrature = addQuadrature ? Quad(length(α) - 1, α, β) : EmptyQuad()

    HermiteOrthonoPoly{promote_type(typeof(α), typeof(β)), GaussMeasureParametric, typeof(quadrature)}(
        deg,
        α,
        β,
        sp,
        GaussMeasureParametric(),
        quadrature)
end

function OrthonoPoly(::GaussMeasureParametric, deg::Int; Nrec::Int = deg + 1, addQuadrature::Bool = true)
    HermiteOrthonoPoly(deg; Nrec = Nrec, addQuadrature = addQuadrature)
end

struct LegendreOrthonoPoly{V, M, Q} <: AbstractCanonicalOrthonoPoly{V, M, Q}
    deg::Int          # maximum degree
    α::V  # recurrence coefficients
    β::V  # recurrence coefficients
    sp::V # scalar product of basis functions
    measure::M
    quad::Q
end

function LegendreOrthonoPoly(deg::Int; Nrec::Int = deg + 1, addQuadrature::Bool = true)
    _checkConsistency(deg, Nrec)
    α, β = r_scale(1.0, rm_legendre01(Nrec)...)
    sp = computeSP2(deg, β)
    quadrature = addQuadrature ? Quad(length(α) - 1, α, β) : EmptyQuad()

    LegendreOrthonoPoly{promote_type(typeof(α), typeof(β)), UniformMeasureParametric,typeof(quadrature)}(
        deg,
        α,
        β,
        sp,
        UniformMeasureParametric(),
        quadrature)
end

function OrthonoPoly(::UniformMeasureParametric, deg::Int; Nrec::Int = deg + 1, addQuadrature::Bool = true)
    LegendreOrthonoPoly(deg; Nrec = Nrec, addQuadrature = addQuadrature)
end

struct JacobiOrthonoPoly{V, M, Q} <: AbstractCanonicalOrthonoPoly{V, M, Q}
    deg::Int          # maximum degree
    α::V # recurrence coefficients
    β::V # recurrence coefficients
    sp:: V # scalar product of basis functions
    measure::M
    quad::Q
end

function JacobiOrthonoPoly(deg::Int, shape_a::Real, shape_b::Real; Nrec::Int = deg + 1,
        addQuadrature::Bool = true)
    _checkConsistency(deg, Nrec)
    α, β = r_scale(1 / beta(shape_a, shape_b), rm_jacobi01(Nrec, shape_b - 1.0, shape_a - 1.0)...)
    sp = computeSP2(deg, β)
    quadrature = addQuadrature ? Quad(length(α) - 1, α, β) : EmptyQuad()

    JacobiOrthonoPoly{promote_type(typeof(α), typeof(β)), BetaMeasureParametric, typeof(quadrature)}(
        deg,
        α,
        β,
        sp,
        BetaMeasureParametric(shape_a, shape_b),
        quadrature)
end

function OrthonoPoly(μ::BetaMeasureParametric, deg::Int; Nrec::Int = deg + 1, addQuadrature::Bool = true)
    JacobiOrthonoPoly(deg, u.pars...; Nrec = Nrec, addQuadrature = addQuadrature)
end

struct LaguerreOrthonoPoly{V, M, Q} <: AbstractCanonicalOrthonoPoly{V, M, Q}
    deg::Int          # maximum degree
    α::V # recurrence coefficients
    β::V # recurrence coefficients
    sp::V # scalar product of basis functions
    measure::M
    quad::Q
end

function LaguerreOrthonoPoly(deg::Int, shape::Real, rate::Real; Nrec::Int = deg + 1,
        addQuadrature::Bool = true)
    _checkConsistency(deg, Nrec)
    α, β = r_scale((rate^shape) / gamma(shape), rm_laguerre(Nrec, shape - 1.0)...)
    sp = computeSP2(deg, β)
    quadrature = addQuadrature ? Quad(length(α) - 1, α, β) : EmptyQuad()

    LaguerreOrthonoPoly{promote_type(typeof(α), typeof(β)), GammaMeasureParametric, typeof(quadrature)}(
        deg,
        α,
        β,
        sp,
        GammaMeasureParametric(shape, rate),
        quadrature)
end

function OrthonoPoly(μ::GammaMeasureParametric, deg::Int; Nrec::Int = deg + 1, addQuadrature::Bool = true)
    LaguerreOrthonoPoly(deg, μ.pars...; Nrec = Nrec, addQuadrature = addQuadrature)
end


"""
For LTI systems, the variants in polynomials are in general assumed to be independent for multivariant polynomial
"""
struct MultiOrthonoPoly{M, Q, V <: AbstractVector} <: AbstractOrthonoPoly{M, Q}
    name::Vector{String}
    deg::Int
    dim::Int
    ind::Matrix{Int} # multi-index
    sp::AbstractVector{<:Real}
    measure::ProductMeasure
    uni::V
    dep::Bool
end

function MultiOrthonoPoly(uniOrthonoPolys::AbstractVector{<:AbstractCanonicalOrthonoPoly};
                          dep::Bool = false, deg::Union{Int, Nothing} = nothing)
    dep ? _multiorthono_dep(uniOrthonoPolys, deg) : _multiorthono_indep(uniOrthonoPolys)
end

function _multiorthono_indep(uniOrthonoPolys)
    w(t) = prod([onp.measure.w(t) for op in uniOrthonoPolys])
    measures = [onp.measure for onp in uniOrthonoPolys]
    measure  = ProductMeasure(w, measures)

    names = [hasfield(typeof(onp), :name) ? onp.name : string(typeof(onp))
                for onp in uniOrthonoPolys]
    n    = length(uniOrthonoPolys)
    degs = [length(onp.α)-1 for onp in uniOrthonoPolys]
    deg  = maximum(degs)
    dim  = sum(degs) + 1 
    ind  = zeros(Int, dim, n)
    L    = 1
    for i = 1:n
        d = degs[i]
        if d > 0
            ind[L+1:L+d,i] = 1:d
            L += d
        end
    end
    sp = _computeMultiSP(ind, uniOrthonoPolys)

    MultiOrthonoPoly{typeof(measure), typeof(first(uniOrthonoPolys).quad), typeof(uniOrthonoPolys)}(
        names,
        deg,
        dim,
        ind,
        sp,
        measure,
        uniOrthonoPolys,
        false)
end

function _multiorthono_dep(uniOrthonoPolys, deg)
        degs = [onp.deg for onp in uniOrthonoPolys]
        isnothing(deg) && throw(ArgumentError("degree must be specified when `dep=true`."))
        deg > minimum(degs) && throw(DomainError(deg,
            "Requested degree $deg is greater than smallest univariate degree $(minimum(degs))."))

        w(t) = prod([onp.measure.w(t) for onp in uniOrthonoPolys])
        measures = [onp.measure for onp in uniOrthonoPolys]
        measure = ProductMeasure(w, measures)

        names = [hasfield(typeof(onp), :name) ? onp.name : string(typeof(onp))
                 for onp in uniOrthonoPolys]
        ind = calculateMultiIndices(length(uniOrthonoPolys), deg)
        sp  = sp = _computeMultiSP(ind, uniOrthonoPolys) 
        dim = size(ind, 1)

        MultiOrthonoPoly{typeof(measure), typeof(first(uniOrthonoPolys).quad), typeof(uniOrthonoPolys)}(
            names,
            deg,
            dim,
            ind,
            sp,
            measure,
            uniOrthonoPolys,
            true)
end

dim(monp::MultiOrthonoPoly) = monp.dim

function _computeMultiSP(ind, uniOrthonoPolys)
    sp_monp = [onp.sp for onp in uniOrthonoPolys]
    dim, n  = size(ind)
    sp      = Vector{Float64}(undef, dim)
    sp[1]   = 1

    @inbounds for i in 2:dim
        sp[i] = prod(@inbounds sp_monp[j][ind[i,j]+1] for j in 1:n)
    end
    return sp
end

function _checkConsistency(deg::Int, Nrec::Int)
    deg < 0 && throw(DomainError(deg, "degree has to be non-negative"))
    Nrec < deg + 1 && throw(DomainError(Nrec,
        "not enough recurrence coefficients specified (need >= $(deg + 1))"))
end

function _hasfield(op::AbstractOrthoPoly, name::Symbol)
    name in fieldnames(op)
end

end