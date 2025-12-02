module Solver

using JuMP, Ipopt, LinearAlgebra

using ..Auxfuns:        _to_matrix, _maybe_to_matrix, _check_wcoeff
using ..PCE:            jointPCE
using ..Problem:        StochProb
using ..Constraints:    con_chance, con_causality, con_dynamics
using ..Objectives:     quadobj

function buildOCP(problem::StochProb;
                optimizer::Union{DataType,Nothing} = Ipopt.Optimizer,
                print_level::Union{Int, Nothing} = nothing,
                max_cpu_time::Union{Real, Nothing} = nothing)
    
    println("\n" * "*"^60)
    println("Starting to build JuMP model for stochastic OCP...")
    println("*"^60 * "\n")

    _ensure_problem_ready(problem)

    N       = problem.N
    nx,nu   = size(problem.B)
    lbx     = problem.lbx
    ubx     = problem.ubx
    lbu     = problem.lbu
    ubu     = problem.ubu
    x0coeff = problem.x0coeff
    wcoeff  = problem.wcoeff

    x0coeff_joint, wcoeff_joint = jointPCE(x0coeff, wcoeff, N)
    L = size(x0coeff_joint, 2)

    
    model = JuMP.Model(optimizer)
    @variable(model, x[1:nx, 1:N+1, 1:L])
    @variable(model, u[1:nu, 1:N, 1:L])

    # Constraints._constraint_chance_constraints(model, problem.risk_tolerance, problem.bound)
    if problem.R !== nothing
        quadobj(model, problem; state = :x, input = :u)
        println("Quadratic ojective added")
    else
        println("Quadratic objective is skipped (R not provided).")
    end

    println("Building constraints:")
    con_dynamics(model, problem, x0coeff_joint, wcoeff_joint; state=:x, input=:u)
    println("  • Dynamics and initial condition constraints added")
    con_causality(model, problem; input = :u)
    println("  • Causality constraint added")
    active_cc = con_chance(model, lbx, ubx, lbu, ubu; gauss = problem.gauss)
    if isempty(active_cc)
        println("  • No chance constraint provided.")
    else
        println("  • Following chance constraints added:")
        for cc in active_cc
            println("      - ", cc)
        end
    end


    !isnothing(print_level) && set_optimizer_attribute(model, "print_level", print_level);
    !isnothing(max_cpu_time) && set_optimizer_attribute(model, "max_cpu_time", max_cpu_time);

    println("\n" * "*"^60)
    println("JuMP model building completed.")
    println("*"^60 * "\n")
    return model
end



function solveOCP(model::JuMP.Model;
                    print_level::Union{Int, Nothing} = nothing,
                    max_cpu_time::Union{Real, Nothing} = nothing
                    )
    !isnothing(print_level) && set_optimizer_attribute(model, "print_level", print_level);
    !isnothing(max_cpu_time) && set_optimizer_attribute(model, "max_cpu_time", max_cpu_time);
    try
        if JuMP.objective_sense(model) == JuMP.MOI.FEASIBILITY_SENSE
            @warn "No objective defined — solver will run a feasibility problem."
        end
        JuMP.optimize!(model)
        status = JuMP.termination_status(model)

        if status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
            # println("Optimal solution found.")
            return JuMP.value.(model[:x]), JuMP.value.(model[:u]), JuMP.objective_value(model)
        else
            println("Optimization did not converge to a feasible solution.")
            println("Termination status: ", status)
            println("Primal status: ", JuMP.primal_status(model))
            println("Dual status: ", JuMP.dual_status(model))
            println("Solver message: ", JuMP.raw_status(model))
            return nothing, nothing, nothing
        end
    catch err
        println("Exception during optimization: ", err)
        return nothing, nothing, nothing
    end
end

function _ensure_problem_ready(problem::StochProb)

    x0coeff = problem.x0coeff
    wcoeff  = problem.wcoeff
    missing = String[]

    isnothing(x0coeff) && push!(missing, "x0coeff")
    isnothing(wcoeff) && push!(missing, "wcoeff")

    if !isempty(missing)
        missing_fields = join(missing, ", ")
        throw(ArgumentError("Cannot build JuMP model: missing required field(s): $missing_fields"))
    end

    x0coeff = _maybe_to_matrix(x0coeff)
    wcoeff  = _maybe_to_matrix(wcoeff)
    size(x0coeff, 1) != problem.nx && throw(ArgumentError("PCE coefficients of X0 must have $nx rows (got $(size(x0coeff, 1)))."))
    _check_wcoeff(wcoeff, problem.N, problem.nw)
    _ensure_weights!(problem)

    return nothing
end

function _ensure_weights!(problem::StochProb)
    nx, nu = problem.nx, problem.nu

    if problem.R === nothing
        @warn "Weighting matrix R not specified; user must define objective manually."
        return nothing
    else
        R = _to_matrix(problem.R)
        _isposdef(R, nu, "R")
        problem.R = R
    end

    if problem.Q === nothing
        @warn "Q not specified; using zeros($nx, $nx)."
        problem.Q = zeros(nx, nx)
    else
        Q = _to_matrix(problem.Q)
        _ispossemidef(Q, nx, "Q")
        problem.Q = Q
    end

    if problem.QN === nothing
        @warn "QN not specified; using zeros($nx, $nx)."
        problem.QN = zeros(nx, nx)
    else
        QN = _to_matrix(problem.QN)
        _ispossemidef(QN, nx, "QN")
        problem.QN = QN
    end

    return nothing
end

function _isposdef(R, nu, name::String)
    if (size(R, 1) != nu) || (size(R, 1) != size(R, 2))
        throw(ArgumentError("$name must be a square matrix with dimension $nu."))
    end
    if !isposdef(R)
        throw(ArgumentError("$name must be a positive definite matrix."))
    end
end

function _ispossemidef(Q, nx, name::String)
    if (size(Q, 1) != nx) || (size(Q, 1) != size(Q, 2))
        throw(ArgumentError("$name must be a square matrix with dimension $nx."))
    end
    if minimum(eigvals(Symmetric(Q))) < -1e-10
        throw(ArgumentError("$name must be positive semidefinite."))
    end
end

end