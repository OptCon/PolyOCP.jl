module Problem

using ..Auxfuns: _to_matrix, _maybe_to_matrix, _check_wcoeff

const RealMat = AbstractMatrix{<:Real}
const RealVec = AbstractVector{<:Real}
const RealMatVec = AbstractVector{<:RealMat}
const RealVecVec = AbstractVector{<:RealVec}

const Weighting = Union{RealMat, Real, Nothing}
const Bound = Union{Tuple{RealVec, RealVec}, Tuple{Real, Real}, Nothing}


mutable struct StochProb
    N::Int

    x0coeff::Union{RealMat, Nothing}
    wcoeff::Union{RealMat, RealMatVec, Nothing}

    A::RealMat
    B::RealMat
    E::RealMat

    Q::Weighting
    R::Weighting
    QN::Weighting

    lbx::Bound
    ubx::Bound
    lbu::Bound
    ubu::Bound

    gauss::Bool

    nx::Int
    nu::Int
    nw::Int
end

"""
    StochProb(N, A, B, E; kwargs...)

Construct a `StochProb` instance given the prediction horizon and system matrices.
All optional data (initial conditions, disturbances, costs, and bounds) default to
`nothing` so that they can be supplied or modified after construction. Dimension
checks are carried out for any optional arguments that are provided.
"""
function StochProb(
    N::Int,
    A::Union{RealMat, Real},
    B::Union{RealMat, RealVec, Real},
    E::Union{RealMat, RealVec, Real};
    x0coeff::Union{RealMat, RealVec, Real, Nothing} = nothing,
    wcoeff::Union{RealMat, RealVec, RealMatVec, RealVecVec, Real, Nothing} = nothing,
    Q::Weighting   = nothing,
    R::Weighting   = nothing,
    QN::Weighting = nothing,
    lbx::Bound = nothing,
    ubx::Bound = nothing,
    lbu::Bound = nothing,
    ubu::Bound = nothing,
    gauss::Bool = false
)

    A = _to_matrix(A)
    B = _to_matrix(B)
    E = _to_matrix(E)

    if size(A, 1) != size(A, 2)
        throw(ArgumentError("A must be a square matrix (got $(size(A, 1))x$(size(A, 2)))."))
    end

    (nx, nu) = size(B)
    nw = size(E, 2)

    x0coeff = _maybe_to_matrix(x0coeff)
    wcoeff  = _maybe_to_matrix(wcoeff)
    Q       = _maybe_to_matrix(Q)
    R       = _maybe_to_matrix(R)
    QN      = _maybe_to_matrix(QN)

    checks  = [
        (size(A, 1), nx, "B must have the same number of rows as A"),
        (size(E, 1), nx, "E must have the same number of rows as A"),
    ]

    for (actual, expected, msg) in checks
        if actual != expected
            throw(ArgumentError("$msg ($expected expected, got $actual)"))
        end
    end

    ( x0coeff !== nothing && size(x0coeff, 1) != nx ) &&
        throw(ArgumentError("PCE coefficients of X0 must have $nx rows (got $(size(x0coeff, 1)))."))

    _check_wcoeff(wcoeff, N, nw)

    _check_bound(lbx, nx, "lbx")
    _check_bound(ubx, nx, "ubx")
    _check_bound(lbu, nu, "lbu")
    _check_bound(ubu, nu, "ubu")

    _check_bound_consistency(lbx, ubx, "state bounds")
    _check_bound_consistency(lbu, ubu, "input bounds")


    return StochProb(
        N,
        x0coeff, wcoeff,
        A, B, E,
        Q, R, QN,
        lbx, ubx,
        lbu, ubu,
        gauss,
        nx, nu, nw
    )
end

function StochProb(
    N::Int,
    A::Union{RealMat, Real},
    B::Union{RealMat, RealVec, Real},
    E::Union{RealMat, RealVec, Real},
    x0coeff::Union{RealMat, RealVec, Real},
    wcoeff::Union{RealMat, RealVec, RealMatVec, RealVecVec, Real};
    kwargs...
)
    return StochProb(
        N,
        A, B, E;
        kwargs...,
        x0coeff = x0coeff,
        wcoeff = wcoeff
    )
end

function _check_bound(bound_pars::Bound, dim::Int, name::String)
    bound_pars === nothing && return nothing

    bound, risk = bound_pars
    (length(bound) != dim || length(risk) != dim) &&
        throw(ArgumentError("$name must have length $dim (got $(length(bound)) and $(length(risk)))"))

    # only check finite bounds
    finite_idx = findall(!isinf, bound)
    if !isempty(finite_idx)
        r = risk[finite_idx]
        (any(r .== 0) || any(abs.(r) .>= 1)) &&
            throw(ArgumentError("The risk tolerance of $name must be in (0,1) or (-1,0)"))
    end

    return nothing
end

function _check_bound_consistency(lb_pars::Bound, ub_pars::Bound, name::String)
   ( (lb_pars === nothing) || (ub_pars === nothing) ) && return nothing
 
    lb_bound, _ = lb_pars
    ub_bound, _ = ub_pars

    any(lb_bound .> ub_bound) &&
        throw(ArgumentError("Lower bound exceeds upper bound for $name"))

    return nothing
end

end
