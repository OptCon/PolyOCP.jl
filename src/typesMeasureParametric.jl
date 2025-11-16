module typesMeasureParametric

using PolyChaos: AbstractMeasure, AbstractCanonicalMeasure
using SpecialFunctions: beta, gamma

abstract type AbstractCanonicalMeasureParametric <: AbstractCanonicalMeasure end

struct DiracMeasureParametric <: AbstractCanonicalMeasureParametric
    pars::Real
    w::Function
    dom::Real
    symmetric::Bool

    function DiracMeasureParametric(c::Real=0.0)
         w = t -> w_dirac_parametric(t, c)
        new(c, w, c, true)
    end
end

struct GaussMeasureParametric <: AbstractCanonicalMeasureParametric
    pars::Tuple{Real,Real}
    w::Function
    dom::Tuple{Real, Real}
    symmetric::Bool

    function GaussMeasureParametric(μ::Real=0.0, σ::Real=1.0)
        w = t -> w_gaussian_parametric(t, μ, σ)
        new((μ,σ), w, (-Inf, Inf), true)
    end
end

struct UniformMeasureParametric <: AbstractCanonicalMeasureParametric
    pars::Tuple{Real,Real}
    w::Function
    dom::Tuple{Real, Real}
    symmetric::Bool

    function UniformMeasureParametric(a::Real=0.0, b::Real=1.0)
        w = t -> w_uniform_parametric(t, a, b)
        new((a,b), w, (a, b), true)
    end
end

struct BetaMeasureParametric <: AbstractCanonicalMeasureParametric
    pars::Tuple{Real,Real}
    w::Function
    dom::Tuple{Real, Real}
    symmetric::Bool

    function BetaMeasureParametric(α::Real=2.0, β::Real=2.0)
        α <= 0 && throw(DomainError(α, "shape parameter a must be positive"))
        β <= 0 && throw(DomainError(β, "shape parameter b must be positive"))

        w = t -> w_beta_parametric(t, α, β)
        new((α,β), w, (0, 1), true)
    end
end

struct GammaMeasureParametric <: AbstractCanonicalMeasureParametric
    pars::Tuple{Real,Real}
    w::Function
    dom::Tuple{Real, Real}
    symmetric::Bool

    function GammaMeasureParametric(α::Real=5.0, β::Real=1.0)
        α <= 0 && throw(DomainError(α, "shape parameter needs to be positive"))
        β != 1 && throw(DomainError(β, "rate must be unity (currently!)"))

        w = t -> w_gamma_parametric(t, α, β)
        new((α,β), w, (0, Inf), true)
    end
end

##
struct ProductMeasureParametric <: AbstractMeasure
    w::Function
    measures::AbstractVector{<:AbstractCanonicalMeasureParametric}
end

function w_dirac_parametric(t, c)
    t == c ? 1.0 : 0.0
end

function w_uniform_parametric(t, a, b)
    (t ≥ a && t ≤ b) ? 1.0/(b - a) : 0.0
end

function w_gaussian_parametric(t, μ, σ)
    1 / (sqrt(2*pi)*σ) * exp(-0.5*((x - μ)/σ)^2)
end

function w_beta_parametric(t, α, β)
    α > 0 || throw(ArgumentError("α must be > 0"))
    β > 0 || throw(ArgumentError("β must be > 0"))

    0 <= t <= 1 ? t^(α-1)*(1-t)^(β-1)/beta(α, β) : _throwError(t)
end

function w_gamma_parametric(t, α, β)
    α > 0 || throw(ArgumentError("α must be > 0"))
    β > 0 || throw(ArgumentError("β must be > 0"))

    t >= 0.0 ? (β^α/gamma(α)*t^(α-1)*exp(-β*t)) : _throwError(t)
end

_throwError(t) = throw(DomainError(t, "not in support"))
end