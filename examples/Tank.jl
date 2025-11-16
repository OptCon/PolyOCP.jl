using Distributed

ncores = 10
addprocs(ncores, topology = :master_worker, exeflags = "--project=$(Base.active_project())")

@everywhere using PolyOCP
@everywhere using LinearAlgebra, Random, SharedArrays, Printf

##
N       = 10
Nmpc    = 50
Ns      = 1000;
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

Q    = Matrix(3.0*I, 4, 4)
R    = Matrix(10^-4*I, 2, 2)
QN   = Q

Wk_OrthonoPoly = HermiteOrthonoPoly(2)
Wk_coeff = 0.05*[1; 1; sqrt(2)]
W_OrthonoPoly = MultiOrthonoPoly([Wk_OrthonoPoly for k=1:4])
WPCE = genPCE(W_OrthonoPoly, [Wk_coeff for k=1:4])
x0 = ones(4, 1)
x0joint,wkjoint = jointPCE(x0, WPCE.coeff, N)
L = size(x0joint, 2) 

lbx = ([-2;-2;-Inf;-Inf], [0.1;0.1;1;1])
ubx = ([2;2;Inf;Inf], [0.1;0.1;1;1])
problem = StochProb(
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
ξTrajs        = randn(nw, Nmpc, Ns)
wTrajs[:,:,:] = 0.05*(ξTrajs .+ ξTrajs.^2)

model = build(problem; print_level=0, max_cpu_time=10.)
## MPC
con_initial_param(model)
tAll = @elapsed begin
@sync @distributed for s = 1:Ns
    tMPCs = @elapsed begin
        for k = 1:Nmpc
            update_initial_param(model, xTrajs[:,k,s]);
            ~, usol, ~ = solveOCP(model)
            uTrajs[:, k, s] = usol[:,1,1]
            xTrajs[:,k+1, s] = A*xTrajs[:,k, s] + B*uTrajs[:, k, s] + E*wTrajs[:, k, s] 
        end
    end
    tMPC[s] = tMPCs
    ## the following line could be used to check the status of sampling
    # @printf "%3i\n" s
end
end

## loading data
xTrajs = Array(xTrajs)
uTrajs = Array(uTrajs)
tMPC   = Array(tMPC)
tAllParallel   = tAll
nx, Nmpc, Ns  = size(xTrajs)
Nmpc = Nmpc - 1

 ## plot the sampled trajectories
using PyPlot, LaTeXStrings
const plt = PyPlot

FS  = 18; LW = 1;

rc("font",style="italic")
fig, ax = plt.subplots(4, sharex=true,sharey=false, figsize=(8.4, 5))
Nplot = 10
for k = 1:nx
    ax[k].plot(0:Nmpc, xTrajs[k,:,1:Nplot], linewidth=LW)
    ax[k].set_xlim(0, Nmpc)
    ax[k].set_ylabel(latexstring("x_{", k, "}"), fontsize=FS)
    ax[k].tick_params(labelsize=FS)
    ax[k].grid("both")
end
ax[4].set_xlabel("k", fontsize=FS)
fig.align_ylabels(ax)
tight_layout()
display(gcf())
# savefig("figures/EX2_TankSampleTraj.pdf")

## plot the figure
steps = [1; 11; 21; 31; 41; 51]
nbins = 20 .* ones(Int, length(steps))

function get_hist_data(xTrajs, steps, nbins)
    map(zip(steps, nbins)) do (k, nb)
        xk = xTrajs[k, :]

        n, bins = hist(xk; bins=nb, density=true, edgecolor="black")
        width = bins[2] - bins[1]
        xrange = bins[1:end-1] .+ width/2

        (n=n, xrange=xrange, width=width)
    end
end

function plot_hist3d(hist_data, steps, info)
    FS, LW = 10, 2
    fig = figure(figsize=(8.4, 4))
    ax = fig.add_subplot(111, projection="3d")

    for (y, h) in Iterators.reverse(zip(steps, hist_data))
        bar(h.xrange, h.n, 0.8 * h.width; zs=y, zdir="y", alpha=0.7)
    end

    xlabel(info, size=FS)
    ylabel(L"k", size=FS)
    zlabel(L"PDF", size=FS)
    ax.set_box_aspect([1.0, 1, 0.6])
    ax.view_init(elev=25, azim=-140)
    display(fig)
end

hist_data_1 = get_hist_data(xTrajs[1,:,:], steps, nbins)
hist_data_2 = get_hist_data(xTrajs[2,:,:], steps, nbins)

plot_hist3d(hist_data_1, steps, L"X_1")
# savefig("figures/EX2_TankDistributionX1.pdf")
plot_hist3d(hist_data_2, steps, L"X_2")
# savefig("figures/EX2_TankDistributionX2.pdf")

#=
The following part is for the Serial sampling, which takes a long max_cpu_time
Adjust the number of samples before start
This Serial sampling part is commented out
=#
# include("../src/StochasticToolkit.jl")
# using .StochasticToolkit, ParameterJuMP, JuMP
# using LinearAlgebra, Random

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

# Wk_OrthonoPoly = HermiteOrthonoPoly(2)
# Wk_coeff = 0.05*[1; 1; sqrt(2)]
# W_OrthonoPoly = MultiOrthonoPoly([Wk_OrthonoPoly for k=1:4])
# WPCE = genPCE(W_OrthonoPoly, [Wk_coeff for k=1:4])
# x0 = ones(4, 1)
# x0joint,wkjoint = jointPCE(x0, WPCE.coeff, N)
# L = size(x0joint, 2) 

# lbx = ([-2;-2;-Inf;-Inf], [0.1;0.1;1;1])
# ubx = ([2;2;Inf;Inf], [0.1;0.1;1;1])
# problem = StochProb(
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
# ξTrajs = randn(nw, Nmpc, Ns)
# wTrajs = 0.05*(ξTrajs .+ ξTrajs.^2)
# model = build(problem)
# ## MPC loop
# # chnage from set initial condition to parametric
# delete.(model, model[:initial_condition])
# unregister(model, :initial_condition)
# @variable(model, x0Param[i in 1:nx]==xTrajs[i,1,1], Param())
# set_optimizer_attribute(model, "print_level", 0);
# set_optimizer_attribute(model, "max_cpu_time", 10.0);
# @constraint(model, initial_condition, model[:x][:,1,:] .== [x0Param zeros(nx,L-1)])
# tAll = @elapsed begin
# for s = 1:Ns
#     for k = 1:Nmpc
#         set_value.(x0Param, xTrajs[:,k,s]);
#         ~, usol, ~ = solveOCP(model)
#         uTrajs[:, k, s] = usol[:,1,1]
#         xTrajs[:,k+1, s] = A*xTrajs[:,k, s] + B*uTrajs[:, k, s] + E*wTrajs[:, k, s] 
#     end
# end
# end

## Compare the computation time of parallel and serial sampling
tAllSerial = tAll
tAllParallel        = Trajs["tAll"]
tAverageSerial      = tAllSerial/Ns
tAverageParallel    = tAllParallel/Ns
tOCPSerial          = tAverageSerial/Nmpc
tOCPParallel        = tAverageParallel/Nmpc 