module Constraints

using JuMP, ParameterJuMP
using SpecialFunctions: erfinv
using SparseArrayKit: SparseArray

using ..Problem:StochProb

# export add_initial_conditions, apply_chance_constraints, add_dynamics
# export  _constraint_dynamics, constraint_chance_constraints, constraint_causality
"""
Apply chance constraints in form of P(x_i < bound) >= 1-risk.
The default option with ind > 0 is for the upper bound constraint.
For the case where a lower bound constraint, i.e. P(x_i > bound) >= 1-risk, is needed,
the constraint is reformulated as P(-x_i < -bound) >= 1-risk.
We refer to the lower bound constraint via a negative ind in function "con_chance".
For Gaussian random variables, the reformulation is exact if γ is chosen via inverse Gaussian distribution
"""
function con_chance(model::Model, var_name::Symbol, ind::Int, bound::Real, risk::Real;
                    gauss::Bool = false, name::Union{Nothing, String} = nothing)
    nx = length(model[var_name][:,1,1]) 
    ind in 1: nx ||
        throw(ArgumentError("Invalid index $ind: must be between 1 and $nx for variable \"$(String(var_name))\""))
    
    if !isinf(bound)
        if 0 < risk && risk < 1
            var     = model[var_name][ind,:,:]
            name === nothing && (name = "con_chance_ub_$(String(var_name))$(ind)")
        elseif -1 < risk && risk < 0
            risk    = -risk
            var     = -model[var_name][ind,:,:]
            bound   = -bound
            name === nothing && (name = "con_chance_lb_$(String(var_name))$(ind)")
        else
            throw(ArgumentError("The risk tolerance of $(String(var_name)) must be in (0,1) or (-1,0)"))
        end
        
        h, L   = size(var) # h -> horizon, h=N for input and h=N+1 for state

        gauss ? γ = sqrt(2)*erfinv(1-2*risk) : γ = sqrt((1-risk)/risk)
        # for k in 2:N1  # Skip initial state
        var_Var = [@inbounds sum(var[k,2:end].^2) for k = 1:h] # variance of variable
        # @constraint(model, [k = 1:h], var[k,1] <= bound) # constraint w.r.t. the mean 
        # @constraint(model, [k = 1:h], γ^2*var_Var[k] - (bound-var[k,1])^2 <=0 ) # second order cone
        @constraint(model, [k = 1:h], var[k,1] <= bound, base_name = name*"_mean")
        @constraint(model, [k = 1:h], γ^2 * var_Var[k] - (bound - var[k,1])^2 <= 0, base_name = name*"_soc")
    end

    return nothing
end

function con_chance(model::Model, var_name::Symbol, inds::AbstractVector{Int},
                    bound::AbstractVector{<:Real}, risk::AbstractVector{<:Real};
                    gauss::Bool = false, name::Union{Nothing, String} = nothing)
    length(risk) == length(bound) ||
        throw(ArgumentError("risk and bound must have the same length"))
    length(risk) == length(inds) ||
        throw(ArgumentError("`risk` and `bound` must match number of constrained states ($(length(inds)))"))
    map(i -> con_chance(model, var_name, inds[i], bound[i], risk[i]; gauss = gauss, name = name), 1:length(inds))
end

function con_chance(model::Model, var_name::Symbol, bound::AbstractVector{<:Real}, risk::AbstractVector{<:Real};
                    gauss::Bool = false, name::Union{Nothing, String} = nothing)
    nx = length(model[var_name][:,1,1]) 
    length(risk) == length(bound) ||
        throw(ArgumentError("risk and bound must have the same length"))
    length(risk) == nx ||
        throw(ArgumentError("risk and bound must match number of constrained variable $(String(var_name))"))
    map(i -> con_chance(model, var_name, i, bound[i], risk[i]; gauss = gauss, name = name), 1:nx)
end

function con_chance(model::Model, var_name::Symbol,
                    cc_pars::Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}};
                    gauss::Bool = false, name::Union{Nothing,String} = nothing)
    bound, risk = cc_pars
    con_chance(model, var_name, bound, risk; gauss = gauss, name = name)
end

function con_chance(model::Model,
                    lbx::Union{Nothing, Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}},
                    ubx::Union{Nothing, Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}},
                    lbu::Union{Nothing, Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}},
                    ubu::Union{Nothing, Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}};
                    gauss::Bool = false, state::Symbol = :x, input::Symbol = :u)

    # state
    if lbx !== nothing
        lbx[2] .= -lbx[2]
        con_chance(model, state, lbx; gauss = gauss)  # lower bounds for state
    end
    if ubx !== nothing
        con_chance(model, state, ubx; gauss = gauss)  # upper bounds for state
    end

    # Inputs
    if lbu !== nothing
        lbu[2] .= -lbu[2]
        con_chance(model, input, lbu; gauss = gauss)  # lower bounds for inptu
    end
    if ubu !== nothing
        con_chance(model, input, ubu; gauss = gauss)  # upper bounds for input
    end
end

# add constraint for causality
function con_causality(model::Model, problem::StochProb; input::Symbol = :u)
    con_causality(model, problem.x0coeff, problem.wcoeff, problem.N; input=input)
end

function con_causality(model::Model, x0coeff::AbstractMatrix{<:Real}, wcoeff::AbstractMatrix{<:Real}, N::Int; input::Symbol = :u)

    N   = N
    # L   = size(u, 3)
    Lx  = size(x0coeff, 2)
    Lw  = size(wcoeff, 2)
    u   = model[input]
    L   = size(u, 3)
    L == Lx + N*(Lw-1) ||
        throw(DomainError("The dimension of joint PCE basis does not match PCE dimensions of initial condition and disturbances"))

    @constraint(model, causality[k in 1:N], u[:,k,(k-1)*(Lw-1)+Lx+1:end] .== 0)

end

function con_causality(model::Model,
                    x0coeff::AbstractMatrix{<:Real}, wcoeff::AbstractVector{T};
                    input::Symbol = :u) where T<:AbstractMatrix{<:Real}
    N   = length(wcoeff)
    Lx  = size(x0coeff, 2)
    Lwk = [size(wk,2) for wk in wcoeff]
    Lk = Lx .+ [0; cumsum(Lwk[2:end].-1)]
    L   = Lx + sum(Lwk) - N

    u = model[input]
    L == size(u, 3) ||
        throw(DomainError("The dimension of joint PCE basis does not match PCE dimensions of initial condition and disturbances"))
    
    @constraint(model, causality[k in 1:N], u[:,k,Lk[k]+1:end] .== 0)
end

function con_causality(model::Model,
                    x0coeff::AbstractMatrix{<:Real}, wcoeff::AbstractVector{T},
                    N::Int; input::Symbol = :u) where T<:AbstractMatrix{<:Real}
    N == length(wcoeff) || 
        throw(ArugmentError("The wcoeff must have the same length as prediction horizon $N (got $(length(wcoeff))"))
    con_causality(model, x0coeff, wcoeff; input=input)
end


"Add constraints w.r.t. dynamics for all PCE terms."
function con_dynamics(model::Model, problem::StochProb, x0coeff_joint::AbstractMatrix{<:Real}, wcoeff_joint::AbstractArray{<:Real,3};
                        state::Symbol = :x, input::Symbol = :u)

    L = size(wcoeff_joint, 3)
    x = model[state]
    u = model[input]

    @constraint(model, initial_condition, x[:, 1, :] .== x0coeff_joint)
    @constraint(model, dynamics[j in 1:L], x[:,2:end,j] .== problem.A*x[:,1:end-1,j] + problem.B*u[:,:,j] + problem.E*wcoeff_joint[:,:,j])
end

function con_initial_param(model::Model; x0::Union{AbstractVector{<:Real},Nothing} = nothing, state::Symbol = :x)

    x0coeff_joint = model[state][:,1,:]
    nx, L =  size(x0coeff_joint)

    delete.(model, model[:initial_condition])
    unregister(model, :initial_condition)
    if x0 === nothing
        x0 = zeros(nx)
    else 
        length(x0) == nx || throw(ArgumentError("x0 must have legnth $(nx) (got $(length(x0))"))
    end

    @variable(model, x0Param[i in 1:nx]==x0[i], Param())
    @constraint(model, initial_condition, x0coeff_joint .== [x0Param SparseArray(zeros(nx,L-1))])
end

function update_initial_param(model::Model, x0::AbstractVector{<:Real}; state::Symbol = :x0Param)

    x0Param = model[:x0Param]
    length(x0) == length(x0Param) || throw(ArgumentError("x0 must have legnth $(length(x0Param)) (got $(length(x0))"))

    set_value.(x0Param, x0);
end

end