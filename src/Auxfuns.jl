module Auxfuns

function _to_matrix(x::Real)
    reshape([x], 1, 1)
end

function _to_matrix(x::AbstractVector{<:Real})
    reshape(x, :, 1)
end

function _to_matrix(x::AbstractMatrix{<:Real})
    x
end

_maybe_to_matrix(::Nothing) = nothing

_maybe_to_matrix(x::Real) = _to_matrix(x)
_maybe_to_matrix(x::AbstractVector{<:Real}) = _to_matrix(x)
_maybe_to_matrix(x::AbstractMatrix{<:Real}) = x


function _maybe_to_matrix(x::AbstractVector{T}) where T <:AbstractVector{<:Real}
    [_to_matrix(xi) for xi in x]
end

function _maybe_to_matrix(x::AbstractVector{T}) where T <:AbstractMatrix{<:Real}
    x
end

function _maybe_to_matrix(x)
    throw(ArgumentError("Unsupported type for _maybe_to_matrix: $(typeof(x))"))
end

function _check_wcoeff(wcoeff, N::Int, nw::Int)
    if wcoeff !== nothing
        if isa(wcoeff, AbstractMatrix{<:Real})
            size(wcoeff, 1) == nw ||
                throw(ArgumentError("PCE coefficients of W must have $nw rows (got $(size(wcoeff,1)))."))

        else
            length(wcoeff) == N ||
                throw(ArgumentError("Vector of W coefficients must have length N = $N (got $(length(wcoeff)))."))

            for (k, Wk) in enumerate(wcoeff)
                size(Wk, 1) == nw ||
                    throw(ArgumentError("Matrix wcoeff[$k] must have $nw rows (got $(size(Wk,1)))."))
            end
        end
    end
end

end
