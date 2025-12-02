__precompile__()
module PolyOCP

include("Auxfuns.jl")

## used packages: PolyChaos, SpecialFunctions
include("typesMeasureParametric.jl")
using .typesMeasureParametric:  AbstractCanonicalMeasureParametric,
                                DiracMeasureParametric, 
                                GaussMeasureParametric, 
                                UniformMeasureParametric, 
                                BetaMeasureParametric, 
                                GammaMeasureParametric
export  AbstractCanonicalMeasureParametric,
        DiracMeasureParametric,
        GaussMeasureParametric, 
        UniformMeasureParametric, 
        BetaMeasureParametric, 
        GammaMeasureParametric

include("typesOrthonoPoly.jl")
using   .typesOrthonoPoly:  AbstractOrthonoPoly, 
                            AbstractCanonicalOrthonoPoly,  
                            ConstantOrthonoPoly, 
                            HermiteOrthonoPoly, 
                            LegendreOrthonoPoly, 
                            JacobiOrthonoPoly, 
                            LaguerreOrthonoPoly, 
                            OrthonoPoly,
                            MultiOrthonoPoly                          
export  AbstractOrthonoPoly, 
        AbstractCanonicalOrthonoPoly,
        ConstantOrthonoPoly, 
        HermiteOrthonoPoly, 
        LegendreOrthonoPoly, 
        JacobiOrthonoPoly, 
        LaguerreOrthonoPoly, 
        OrthonoPoly,
        MultiOrthonoPoly

## used packages: PolyChaos, SparseArrayKit
include("PCE.jl")
using   .PCE:   OrthonoPCE,
                showbasis_normalized, 
                showpoly_normalized,
                showmultipoly,
                measure2poly,
                genPCE,
                jointPCE

export  OrthonoPCE,
        showbasis_normalized, 
        showpoly_normalized,
        showmultipoly,
        measure2poly,
        genPCE,
        jointPCE

include("Problem.jl")
using   .Problem: StochProb, defineOCP
export  StochProb, defineOCP

## used packages: JuMP, ParameterJuMP, SpecialFunctions, SparseArrayKit
include("Constraints.jl")
using   .Constraints: con_dynamics, con_causality, con_chance, con_initial_param, update_initial_param
export  con_dynamics, con_causality, con_chance, con_initial_param, update_initial_param

## used packages: JuMP
include("Objectives.jl")
using   .Objectives:quadobj
export  quadobj

## used packages: JuMP, Ipopt, LinearAlgebra
include("Solver.jl")
using .Solver: buildOCP, solveOCP
export buildOCP, solveOCP

end  # module StochasticToolkit
