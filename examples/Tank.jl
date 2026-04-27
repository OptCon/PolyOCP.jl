"""
This example is descripted in the following arXiv preprint in detail:
PolyOCP.jl -- A Julia Package for Stochastic OCPs and MPC
https://arxiv.org/abs/2511.19084

The realization samples in MPC loop are carried out in parallel computing in Julia.
In the arXiv preprint, 1000 samplings are run in 10 cores, which may take up to a few hours.
Here the number of cores and number of samples are adjusted to a reasonable size.
To get the proper plots, please adjust the numbers of cores and samplings
"""

using Distributed

ncores = 2 # number of cores for parallel computing
addprocs(ncores, topology = :master_worker, exeflags = "--project=$(Base.active_project())")

@everywhere using PolyOCP
@everywhere using LinearAlgebra, Random, SharedArrays

## --------------------------------------------------------------------------
N       = 10    # prediction horizon of OCP
Nmpc    = 50    # closed-loop steps
Ns      = 10  # number of samplings

A = [0.921  0.0    0.041  0.0;
    0.0    0.918  0.0    0.033;
    0.0    0.0    0.924  0.0;
    0.0    0.0    0.0    0.937]
B = [0.017  0.001;
    0.001  0.023;
    0.0    0.061;
    0.072  0.0]
E = Matrix(1.0*I, 4, 4)

nx, nu = size(B)
nw = size(E, 2)

Q    = Matrix(3.0*I, nx, nx)
R    = Matrix(10^-4*I, nu, nu)
QN   = Q

#= Frist create PCE for each component of Wi -> WiPCE
Then construct a random vector of W -> [WiPCE for i=1:nw], where each component is WiPCE
Function genPCE(...) directly genrate PCE of random vector W
In MPC loop, the initial condition is measured and thus deterministic =#
Wi_OrthonoPoly  = HermiteOrthonoPoly(2) # PCE basis of Wi
Wi_coeff        = 0.05*[1; 1; sqrt(2)]  # PCE coefficients of Wi
WiPCE           = OrthonoPCE(Wi_OrthonoPoly, Wi_coeff)
WPCE            = genPCE([WiPCE for i=1:nw])
x0              = ones(nx)

lbx = ([-2;-2;-Inf;-Inf], [0.1;0.1;1;1])
ubx = ([2;2;Inf;Inf], [0.1;0.1;1;1])

problem = defineOCP(
    N, 
    A, B, E,
    x0, WPCE.coeff;
    Q=Q, R=R, QN=QN,
    lbx=lbx, ubx=ubx
)

Random.seed!(1);
xTrajs = SharedArray{Float64}((nx, Nmpc+1, Ns))
uTrajs = SharedArray{Float64}((nu, Nmpc, Ns))
wTrajs = SharedArray{Float64}((nw, Nmpc, Ns))
tMPC   = SharedArray{Float64}((Ns,1))
tAll   = 0

xTrajs[:,1,:] = 2*rand(nx,Ns).-1
# Sample i.i.d. disturbances and reshape
rng = MersenneTwister(1)
W = samplePCE(WPCE, Nmpc*Ns; rng=rng) 
wTrajs = reshape(W, nw, Nmpc, Ns)

#= print_level decides the printed information of the solver
print_level = 0 hides the information in the repeated MPC loop
max_cpu_time set the maximal computation time of each OCP =#
model = buildOCP(problem; print_level=0, max_cpu_time=10.)

# Here starts the MPC loop
con_initial_param(model) # parametric initial condition
tAllParallel = @elapsed begin
@sync @distributed for s = 1:Ns
    tMPCs = @elapsed begin
        for k = 1:Nmpc
            update_initial_param(model, xTrajs[:,k,s]); # update the initial condition
            ~, usol, ~ = solveOCP(model)
            uTrajs[:, k, s] = usol[:,1,1]
            xTrajs[:,k+1, s] = A*xTrajs[:,k, s] + B*uTrajs[:, k, s] + E*wTrajs[:, k, s] 
        end
    end
    tMPC[s] = tMPCs
    # the following line prints the finished samplings
    # println(lpad(i, 3))
end
end

## --------------------------------------------------------------------------
# loading saved data if there are
xTrajs = Array(xTrajs)
uTrajs = Array(uTrajs)
tMPC   = Array(tMPC)
nx, Nmpc, Ns  = size(xTrajs)
Nmpc = Nmpc - 1

## --------------------------------------------------------------------------
# plot the sampled trajectories
FS = 14
fig, ax = plot_traj(xTrajs[:, :, 1:10];
    ylabel = "x",
    labelsize=FS,
    ticksize=FS,
    figsize = (8.4, 5),
    linewidth = 1.0,
    showfig=false)

fig.text(0.5, -0.05,
    "10 different closed-loop realizations of state trajectories",
    fontsize=FS, ha = "center")
display(fig)

## --------------------------------------------------------------------------
# plot the histograms of samples to approximate probability density functions
steps   = 0:10:Nmpc
x1Trajs = [xTrajs[1,k+1,:] for k in steps]
x2Trajs = [xTrajs[2,k+1,:] for k in steps]

fig_x1, ax_x1 = plot3d_hist(x1Trajs, steps; nbins=20, xlabel=L"X_1", showfig=false)
fig_x1.text(0.5, 0.08,
    "Histograms of $(Ns) samples of " * string(L"X_1"),
    fontsize = 12, ha = "center")
display(fig_x1)
# savefig("figures/EX2_TankDistributionX1.pdf")

fig_x2, ax_x2 = plot3d_hist(x2Trajs, steps; nbins=20, xlabel=L"X_2", showfig=false)
fig_x2.text(0.5, 0.08,
    "Histograms of $(Ns) samples of " * string(L"X_2"),
    fontsize = 12, ha = "center")
display(fig_x2)
# savefig("figures/EX2_TankDistributionX2.pdf")
## ===========================================================================
#=
The following part is for the Serial sampling, which takes a long max_cpu_time
Adjust the number of samples before start
This Serial sampling part is commented out
=#
# ============================================================================

# N       = 10
# Nmpc    = 50
# Ns      = 10;
# A = [0.921  0.0    0.041  0.0;
#     0.0    0.918  0.0    0.033;
#     0.0    0.0    0.924  0.0;
#     0.0    0.0    0.0    0.937]

# B = [0.017  0.001;
#     0.001  0.023;
#     0.0    0.061;
#     0.072  0.0]

# E = Matrix(1.0*I, 4, 4)

# nx, nu = size(B)
# nw = size(E, 2)

# Q    = Matrix(3.0*I, 4, 4)
# R    = Matrix(10^-4*I, 2, 2)
# QN   = Q

# Wi_OrthonoPoly  = HermiteOrthonoPoly(2) # PCE basis of Wi
# Wi_coeff        = 0.05*[1; 1; sqrt(2)]  # PCE coefficients of Wi
# WiPCE           = OrthonoPCE(Wi_OrthonoPoly, Wi_coeff)
# WPCE            = genPCE([WiPCE for i=1:nw])
# x0              = ones(nx, 1)

# lbx = ([-2;-2;-Inf;-Inf], [0.1;0.1;1;1])
# ubx = ([2;2;Inf;Inf], [0.1;0.1;1;1])
# problem = defineOCP(
#     N, 
#     A, B, E,
#     x0, WPCE.coeff;
#     Q=Q, R=R, QN=QN,
#     lbx=lbx, ubx=ubx
# )

# Random.seed!(1);
# xTrajs = Array{Float64}(undef, nx, Nmpc+1, Ns)
# xTrajs[:,1,:] = 2*rand(nx,Ns).-1
# uTrajs = Array{Float64}(undef, nu, Nmpc, Ns)
# rng = MersenneTwister(1)
# W = samplePCE(WPCE, Nmpc*Ns; rng=rng) 
# wTrajs = reshape(W, nw, Nmpc, Ns)

# model = buildOCP(problem; print_level=0, max_cpu_time=10.)

# # Here starts the MPC loop
# con_initial_param(model) # parametric initial condition
# tAllSerial = @elapsed begin
# for s = 1:Ns
#     tMPCs = @elapsed begin
#         for k = 1:Nmpc
#             update_initial_param(model, xTrajs[:,k,s]);
#             ~, usol, ~ = solveOCP(model)
#             uTrajs[:, k, s] = usol[:,1,1]
#             xTrajs[:,k+1, s] = A*xTrajs[:,k, s] + B*uTrajs[:, k, s] + E*wTrajs[:, k, s] 
#         end
#     end
# end
# end

## --------------------------------------------------------------------------
# Compare the computation time of parallel and serial sampling
# tAverageSerial      = tAllSerial/Ns
# tAverageParallel    = tAllParallel/Ns
# tOCPSerial          = tAverageSerial/Nmpc
# tOCPParallel        = tAverageParallel/Nmpc
