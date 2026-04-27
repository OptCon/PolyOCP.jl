module Sampling

using Random, Distributions

using ..typesMeasureParametric: DiracMeasureParametric,
                                GaussMeasureParametric,
                                UniformMeasureParametric,
                                BetaMeasureParametric,
                                GammaMeasureParametric

using ..typesOrthonoPoly: AbstractOrthonoPoly,
                          AbstractCanonicalOrthonoPoly,
                          ConstantOrthonoPoly,
                          HermiteOrthonoPoly,
                          LegendreOrthonoPoly,
                          JacobiOrthonoPoly,
                          LaguerreOrthonoPoly,
                          MultiOrthonoPoly

using ..PCE: OrthonoPCE, rec2coeff

export samplePCE, samplePCE_traj


"""
    samplePCE(onpPCE::OrthonoPCE, nsamples::Int; rng=Random.default_rng(), return_germ=false)

Sample from an `OrthonoPCE`.

Returns:
- vector of length `nsamples` for scalar coefficients
- matrix of size `(n, nsamples)` for vector-valued coefficients
"""
function samplePCE(onpPCE::OrthonoPCE, nsamples::Int; rng=Random.default_rng(), return_germ::Bool=false)
    samplePCE(onpPCE.basis, onpPCE.coeff, nsamples; rng=rng, return_germ=return_germ)
end

function samplePCE(basis::AbstractOrthonoPoly,
                   coeff::AbstractVector{<:Real},
                   nsamples::Int;
                   rng=Random.default_rng(), return_germ::Bool=false)

    nsamples > 0 || throw(ArgumentError("`nsamples` must be positive."))

    Ψ, ξ = _basis_matrix(rng, basis, nsamples)
    length(coeff) == size(Ψ, 2) ||
        throw(ArgumentError("Coefficient length $(length(coeff)) does not match basis dimension $(size(Ψ,2))."))

    y = Ψ * coeff
    return return_germ ? (y, ξ) : y
end

function samplePCE(basis::AbstractOrthonoPoly,
                   coeff::AbstractMatrix{<:Real},
                   nsamples::Int;
                   rng=Random.default_rng(), return_germ::Bool=false)

    nsamples > 0 || throw(ArgumentError("`nsamples` must be positive."))

    Ψ, ξ = _basis_matrix(rng, basis, nsamples)
    size(coeff, 2) == size(Ψ, 2) ||
        throw(ArgumentError("Coefficient width $(size(coeff,2)) does not match basis dimension $(size(Ψ,2))."))

    Y = coeff * transpose(Ψ)   # (n × L) * (L × ns) = (n × ns)
    return return_germ ? (Y, ξ) : Y
end


"""
    samplePCE_traj(basis, coeff_traj, nsamples; rng=Random.default_rng(), return_germ=false)

Sample trajectories from PCE coefficients `coeff_traj` of size `(n, Nt, L)`.
Returns an array of size `(n, Nt, nsamples)`.
"""
function samplePCE_traj(onpPCE::OrthonoPCE,
                       coeff_traj::AbstractArray{<:Real,3},
                       nsamples::Int; rng=Random.default_rng(), return_germ::Bool=false)
    samplePCE_traj(onpPCE.basis, coeff_traj, nsamples; rng=rng, return_germ=return_germ)
end

function samplePCE_traj(basis::AbstractOrthonoPoly,
                       coeff_traj::AbstractArray{<:Real,3},
                       nsamples::Int; rng=Random.default_rng(), return_germ::Bool=false)

    nsamples > 0 || throw(ArgumentError("`nsamples` must be positive."))

    n, Nt, L = size(coeff_traj)
    Ψ, ξ = _basis_matrix(rng, basis, nsamples)

    L == size(Ψ, 2) ||
        throw(ArgumentError("Third dimension of `coeff_traj` ($L) does not match basis dimension $(size(Ψ,2))."))

    Y = Array{Float64}(undef, n, Nt, nsamples)

    @inbounds for k in 1:Nt
        Y[:, k, :] .= coeff_traj[:, k, :] * transpose(Ψ)
    end

    return return_germ ? (Y, ξ) : Y
end

_rand_measure(rng::AbstractRNG, m::DiracMeasureParametric, ns::Int) =
    fill(Float64(m.pars), ns)

_rand_measure(rng::AbstractRNG, ::GaussMeasureParametric, ns::Int) =
    rand(rng, Normal(0.0, 1.0), ns)

_rand_measure(rng::AbstractRNG, ::UniformMeasureParametric, ns::Int) =
    rand(rng, Uniform(0.0, 1.0), ns)

_rand_measure(rng::AbstractRNG, m::BetaMeasureParametric, ns::Int) = begin
    α, β = m.pars
    rand(rng, Beta(α, β), ns)
end

_rand_measure(rng::AbstractRNG, m::GammaMeasureParametric, ns::Int) = begin
    α, β = m.pars   # rate β
    rand(rng, Gamma(α, 1 / β), ns)
end

function _eval_uni_basis(onp::AbstractCanonicalOrthonoPoly, x::AbstractVector{<:Real})
    deg = onp.deg
    ns  = length(x)

    polys = Vector{Vector{Float64}}(undef, deg + 1)
    polys[1] = [1.0]
    deg >= 1 && (polys[2:end] = rec2coeff(deg, onp.α, onp.β, onp.sp))

    Ψ = Matrix{Float64}(undef, ns, deg + 1)
    for j in 0:deg
        Ψ[:, j+1] .= evalpoly.(x, Ref(polys[j+1]))
    end

    return Ψ
end

function _eval_multi_basis(monp::MultiOrthonoPoly, ξ::AbstractMatrix{<:Real})
    ns, nξ = size(ξ)
    size(monp.ind, 2) == nξ ||
        throw(ArgumentError("Number of sampled variables ($nξ) does not match basis dimension $(size(monp.ind, 2))."))

    Ψuni = [_eval_uni_basis(monp.uni[i], vec(ξ[:, i])) for i in 1:nξ]

    L = monp.dim
    Ψ = ones(Float64, ns, L)

    @inbounds for j in 2:L
        for i in 1:nξ
            d = monp.ind[j, i]
            d == 0 && continue
            Ψ[:, j] .*= Ψuni[i][:, d + 1]
        end
    end

    return Ψ
end

function _sample_germ(rng::AbstractRNG, onp::AbstractCanonicalOrthonoPoly, ns::Int)
    ξ = _rand_measure(rng, onp.measure, ns)
    reshape(ξ, :, 1)
end

function _sample_germ(rng::AbstractRNG, monp::MultiOrthonoPoly, ns::Int)
    nξ = length(monp.uni)
    ξ = Matrix{Float64}(undef, ns, nξ)

    for i in 1:nξ
        ξ[:, i] .= _rand_measure(rng, monp.uni[i].measure, ns)
    end

    return ξ
end

function _basis_matrix(rng::AbstractRNG, basis::AbstractCanonicalOrthonoPoly, ns::Int)
    ξ = _sample_germ(rng, basis, ns)
    Ψ = _eval_uni_basis(basis, vec(ξ))

    return Ψ, ξ
end

function _basis_matrix(rng::AbstractRNG, basis::MultiOrthonoPoly, ns::Int)
    ξ = _sample_germ(rng, basis, ns)
    Ψ = _eval_multi_basis(basis, ξ)
    
    return Ψ, ξ
end

end