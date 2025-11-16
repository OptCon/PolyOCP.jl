using Distributed

ncores = 2
addprocs(ncores, topology = :master_worker, exeflags = "--project=$(Base.active_project())")

@everywhere using PolyOCP
@everywhere using LinearAlgebra, Random, SharedArrays, Printf, JLD2

##
N       = 10
Nmpc    = 10
Ns      = 2;
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
# tmodel = SharedArray{Float64}((Ns,1));
tMPC   = SharedArray{Float64}((Ns,1))
tAll   = 0

xTrajs[:,1,:] = 2*rand(nx,Ns).-1
ξTrajs        = randn(nw, Nmpc, Ns)
wTrajs[:,:,:] = 0.05*(ξTrajs .+ ξTrajs.^2)

model = build(problem; print_level=0, max_cpu_time=10.)
# set_optimizer_attribute(model, "print_level", 0);
# set_optimizer_attribute(model, "max_cpu_time", 10.0);
# delete.(model, model[:initial_condition])
# unregister(model, :initial_condition)
# @variable(model, x0Param[i in 1:nx]==xTrajs[i,1,1], Param())
# @constraint(model, initial_condition, model[:x][:,1,:] .== [x0Param zeros(nx,L-1)])

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