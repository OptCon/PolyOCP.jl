module Density

using FFTW
using HypergeometricFunctions: pFq

using   ..typesOrthonoPoly: AbstractOrthonoPoly, 
                            AbstractCanonicalOrthonoPoly,  
                            ConstantOrthonoPoly, 
                            HermiteOrthonoPoly, 
                            LegendreOrthonoPoly, 
                            JacobiOrthonoPoly, 
                            LaguerreOrthonoPoly, 
                            OrthonoPoly,
                            MultiOrthonoPoly

using ..PCE: OrthonoPCE

struct PCEgrouped
    μ::Float64
    groups::Dict{AbstractCanonicalOrthonoPoly, Vector{Float64}}
end


_basis_list(basis::MultiOrthonoPoly) = basis.uni
_basis_list(basis::AbstractCanonicalOrthonoPoly) = [basis]

function PCEgrouped(onpPCE::OrthonoPCE; tol=1e-12)
    basis = onpPCE.basis
    coeff = onpPCE.coeff

    ubasis = _basis_list(basis)
    @assert length(coeff) == length(ubasis) + 1

    if basis.deg > 1
        error("Only PCEs (degree ≤ 1) are supported, but got degree = $(basis.deg).")
    end

    μ = coeff[1]
    groups = Dict{AbstractCanonicalOrthonoPoly, Vector{Float64}}()

    @inbounds for j in eachindex(ubasis)
        c = coeff[j + 1]

        abs(c) ≤ tol && continue

        bj = ubasis[j]

        if bj isa ConstantOrthonoPoly
            μ += c
        else
            push!(get!(groups, bj, Float64[]), c)
        end
    end

    return PCEgrouped(μ, groups)
end

function char_fun(ω::AbstractVector{<:Real}, grouped::PCEgrouped)
    φ = exp.(im .* ω .* grouped.μ)

    for (onp, coeffs) in grouped.groups
        φ .*= char_fun_dist(ω, coeffs, onp)
    end

    return φ
end

function char_fun_dist(ω, coeffs, onp::HermiteOrthonoPoly)
    σ2 = sum(c^2 for c in coeffs)
    return exp.(-0.5.* σ2 .* ω.^2)
end

function char_fun_dist(ω, coeffs, onp::LegendreOrthonoPoly)
    # char_fun of normalized basis phi = 2*sqrt{3}*(\xi-0.5)
    φ = ones(ComplexF64, length(ω))

    for c in coeffs
        a = sqrt(3) * c
        for i in eachindex(ω)
            z = ω[i] * a
            φ[i] *= abs(z) < 1e-12 ? 1.0 : sin(z) / z
        end
    end

    return φ
end

function char_fun_dist(ω, coeffs, onp::JacobiOrthonoPoly)
    α, β = onp.measure.pars
    μ = α / (α + β)
    σ = sqrt(α * β / ((α + β)^2 * (α + β + 1)))

    φ = ones(ComplexF64, length(ω))
    for c in coeffs
        z = im .* ω .* (c / σ)
        φ .*= exp.(-im .* ω .* c .* μ / σ) .* pFq.(Ref((α,)), Ref((α + β,)), z)
    end
    return φ
end

function char_fun_dist(ω, coeffs, onp::LaguerreOrthonoPoly)
    α, β = onp.measure.pars
    sα = sqrt(α)   # since μ/σ = sqrt(α), and σ*β = sqrt(α)

    φ = ones(ComplexF64, length(ω))
    for c in coeffs
        z = ω .* c
        φ .*= exp.(-im .* z .* sα) .* (1 .- im .* z ./ sα) .^ (-α)
    end
    return φ
end

"""
    pdfPCE computes the probability density function (PDF) of a scalar PCE from charateristic function (Fourier transformation) and its inverse.

The method
`pdfPCE(interval_x::Tuple{<:Real,<:Real}, onpPCE::OrthonoPCE; N::Int=2048, tol::Real=1e-12)`
computes the PDF on the prescribed interval `interval_x` from the given PCE
coefficients `coeff`.

The method
`pdfPCE(onpPCE::OrthonoPCE; nsigma::Real=5.0, N::Int=2048, tol::Real=1e-12)`
computes the PDF directly from an `OrthonoPCE` if `interval_x` is not provided.
The interval_x is chosen automatically as `(μ - 5σ, μ + 5σ)`,
where `μ` and `σ` are the mean and standard deviation of the PCE.

# Arguments
- `interval_x`: interval on which the PDF is evaluated
- `onpPCE`: scalar orthonormal PCE
- `N`: numerical resolution parameter
- `tol`: threshold for neglecting small coefficients / numerical contributions

# Returns
- `x`: grid points in the evaluation interval
- `pdf`: approximated PDF values on `x`
"""
function pdfPCE(interval_x::Tuple{<:Real,<:Real},
                onpPCE::OrthonoPCE;
                N::Int=2048,
                tol::Real=1e-12)

    isempty(onpPCE.coeff) &&
        throw(ArgumentError("`onpPCE.coeff` must be nonempty"))

    xmin, xmax = interval_x
    (isfinite(xmin) && isfinite(xmax) && xmin < xmax) ||
        throw(ArgumentError("Invalid PDF interval: $interval_x"))

     N >= 2 || throw(ArgumentError("N must be at least 2, but got N = $N"))

    if isodd(N)
        N_old = N
        N -= 1
        @info "N must be even for the symmetric FFT grid. Using N = $N instead of N = $N_old."
    end

    xabs_max    = max(abs(xmin), abs(xmax))
    Δω          = π / xabs_max
    ω           = collect(-N÷2:N÷2-1) .* Δω

    groupedPCE  = PCEgrouped(onpPCE; tol=tol)
    φ           = char_fun(ω, groupedPCE)

    # Δx * Δω = 2π / N in angular-frequency convention
    Δx  = 2*xabs_max / N
    x   = collect(-N÷2:N÷2-1) .* Δx

    # shift frequency to have 0 at index 1
    φ_fft = ifftshift(φ)

    # p(x) ≈ (Δω / 2π) * Σ φ(ω_k) exp(-i ω_k x)
    pdf_vals = fft(φ_fft) .* (Δω / (2π))

    # Reorder back so that x is centered at 0
    pdf_vals = real(fftshift(pdf_vals))

    # Return requested x-interval
    keep = (x .>= xmin) .& (x .<= xmax)
    return x[keep], pdf_vals[keep]
end

function pdfPCE(onpPCE::OrthonoPCE;
                nsigma::Real=5.0,
                N::Int=2048,
                tol::Real=1e-12)

    coeff = onpPCE.coeff

    nsigma > 0 ||
        throw(ArgumentError("`nsigma` must be positive"))

    μ = coeff[1]
    σ = sqrt(sum(coeff[2:end].^2))

    interval_x = σ > 0 ? (μ - nsigma*σ, μ + nsigma*σ) : (μ - 1, μ + 1)

    @info "No interval_x provided. Using default interval μ ± $(nsigma)σ." μ=μ σ=σ interval_x=interval_x

    return pdfPCE(interval_x, onpPCE; N=N, tol=tol)
end

end
 