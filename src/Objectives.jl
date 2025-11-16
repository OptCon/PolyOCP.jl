module Objectives

using JuMP

using ..Problem: StochProb

"Define a quadratic objective function."
function quadobj(model::JuMP.Model, problem::StochProb;
                state::Symbol = :x, input::Symbol = :u)
    
    Q   = problem.Q
    R   = problem.R
    QN  = problem.QN

    x   = model[state]
    u   = model[input]
    N   = problem.N
    L   = size(x, 3)

    obj = 0
    for j = 1:L
        for k = 1:N
            obj = obj + transpose(x[:,k,j])*Q*x[:,k,j] + transpose(u[:,k,j])*R*u[:,k,j]
        end
        obj = obj + transpose(x[:,N+1,j])*QN*x[:,N+1,j]
    end
    @objective(model, Min, obj)
end

end
