#=
This example revisits the chemical reactor example, but with non-i.i.d. disturbances
We consider the periodic disturbances with period = 2, i.e.
W(0) = U(-0.1, 0.1), W(1) = U(-0.2, 0.2),
W(2) ~ W(0), W(3) ~ W(1),...
where U denotes the uniform distribution
=#
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))   # activate package environment

using Revise
using PolyOCP
using LinearAlgebra, JuMP
using FFTW, PyPlot, LaTeXStrings
using Distributions, Random

## Define problem parameters
N    = 10  # Number of time steps

A    = [0.95123 0;              # 2×2 system
        0.08833 0.81873]
B    =[-0.0048771; -0.0020429]
E    = [1.0; 1.0]

Q    = Matrix(1.0*I, 2, 2)
R    = 1.
QN   = Q

ubx = ([Inf; 0.24], [1; 0.1])

# Define uncertainties
X0  = [GaussMeasureParametric(0.5,0.05); GaussMeasureParametric(0.1,0.01)]
W0  = [UniformMeasureParametric(-0.1, 0.1)]
W1  = [UniformMeasureParametric(-0.2, 0.2)]

X0PCE   = genPCE(X0)
W0PCE   = genPCE(W0)
W1PCE   = genPCE(W1)
wcoeff  = repeat([W0PCE.coeff,W1PCE.coeff], div(N,2))

##
problem = StochProb(
    N, 
    A, B, E,
    X0PCE.coeff, wcoeff;
    Q=Q, R=R, QN=QN,
    ubx = ubx
)

model= build(problem)
solution_x, solution_u, obj = solveOCP(model)
x1sol = solution_x[1,:,:]
x2sol = solution_x[2,:,:]
usol  = solution_u[1,:,:]
