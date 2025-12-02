"""
This example is descripted in the following arXiv preprint in detail:
PolyOCP.jl -- A Julia Package for Stochastic OCPs and MPC
https://arxiv.org/abs/2511.19084
"""

using PolyOCP
using LinearAlgebra
using FFTW, PyPlot, LaTeXStrings
using Distributions, Random

## Define problem parameters
N    = 50 

A    = [0.95123 0;
        0.08833 0.81873]
B    =[-0.0048771; -0.0020429]
E    = [1.0; 1.0]

Q    = Matrix(1.0*I, 2, 2)
R    = 1.
QN   = Q

ubx = ([Inf; 0.24], [1; 0.1])

# Define X0 and W and generate their PCEs
# X0 is Gaussian distributed
# W is uniformly distributed
X0  = [GaussMeasureParametric(0.5,0.05); GaussMeasureParametric(0.1,0.01)]
W   = [UniformMeasureParametric(-0.0173, 0.0173)]
X0PCE   = genPCE(X0)
WPCE    = genPCE(W)

# define and build the stochastic OCP
problem = defineOCP(
    N, 
    A, B, E,
    X0PCE.coeff, WPCE.coeff;
    Q=Q, R=R, QN=QN,
    ubx = ubx
)
model= buildOCP(problem)

## solve stochastic OCP
# The solutions are given in PCE coefficients
xsolPCE, usolPCE, obj = solveOCP(model)
x1solPCE = xsolPCE[1,:,:]
x2solPCE = xsolPCE[2,:,:]
usolPCE  = usolPCE[1,:,:]

## ===========================================================================
#=
The next part solve it 1000 times to get average computation time.
This block is commented out as it takes long time.
=#
# ============================================================================

# Ns = 1000
# set_optimizer_attribute(model, "print_level", 0);
# tAll = @elapsed begin 
#     for _ = 1:Ns
#         solveOCP(model)
#     end
# end


## ===========================================================================
#=
The folloing part visualizes the results in a 3D figure
This part first define several functions to computes probability density functions from PCE expressions.
The computation of PDF and visualization will be integrated as compact functions in the next version.
=#
# ============================================================================
## compute the charateristic function from PCE
function char_fun(f, coeff)
    Nf = length(f);
    Lx = 3;
    L  = length(coeff);

    y  = zeros(Nf, 1)+im*zeros(Nf,1);
    quad_gauss  = sum([coeff[j]^2 for j = 2:Lx]);
    sum_uniform = sum(coeff[j] for j = Lx+1:L)

    for i = 1:Nf
        y0 = exp((coeff[1]-0.5*sum_uniform)*im*2pi*f[i]);
        yg = exp(-2*quad_gauss*pi^2*f[i]^2);
        y[i] = y0*yg;
        if f[i] != 0
            for j = Lx+1:L
                if coeff[j] > 1e-3
                        yuniform = (exp(im*2*pi*f[i]*coeff[j])-1)/(im*2pi*coeff[j]*f[i]);
                        y[i] = y[i]*yuniform;
                end
            end
    else
                y[i] = y0*(1.0 + 0*im)*yg;
        end
    end

    return y;
end

# compute probability density function of given interval from PCE
function cal_distribution(interval::Tuple{<:Real,<:Real}, coeff)
    xmin, xmax = interval
    Fs = 2 * maximum(abs.((xmin, xmax)))

    T = 1 / Fs
    f = -100:T:100
    L = length(f)

    coeff[4:end] = 2*sqrt(3)*coeff[4:end];
    ϕ = char_fun(f, coeff)
    sk = fft(ϕ) .* T

    freqs = fftfreq(L, Fs)
    mags  = abs.(sk)

    i0 = argmin(freqs)                        # reorder to ascending
    freqs = vcat(freqs[i0:end], freqs[1:i0-1])
    mags  = vcat(mags[i0:end],  mags[1:i0-1])

    keep = (freqs .>= xmin) .& (freqs .<= xmax)  # trim to interval
    return freqs[keep], mags[keep]
end

# sample the realizations of PCE basis
function sample_vector(n_samples::Int, dim::Int)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    dim ≥ 3 || throw(ArgumentError("dim must be at least 3"))

    X = zeros(n_samples, dim)

    X[:, 1:2]   .= randn(n_samples, 2)
    X[:, 3:end] .= rand(n_samples, dim - 2)

    # Compute PCE basis values
    psi = Matrix{Float64}(undef, n_samples, dim + 1)
    psi[:, 1] .= 1.0                       
    psi[:, 2:3] .= X[:, 1:2]
    psi[:, 4:end] .= 2 * sqrt(3) .* (X[:, 3:end] .- 0.5)

    return X, psi
end
## definitions of functions end here

L = size(x1solPCE, 2)
FS=10; LW=2;
jmax = 30 # plot the first 30 PCE coefficients
fig = figure(figsize=(8.4, 4));
ax = fig.add_subplot(111,projection="3d")
for j = 1:jmax
        t1 = max(0,j-3);
        tx = t1:N;
        plot3D(j*ones(length(tx)), tx, x1solPCE[tx.+1,j]; linewidth=LW);
end

xlabel(L"j", size=FS); ylabel(L"k", size=FS);
zlabel(L"x_1^{j,*}", fontsize=FS)  
ax.set_box_aspect([1.0, 1, 0.6])
ax.view_init(elev=20, azim=-65)
ax.set_yticks(0:10:N)
ax.set_xlim(0, jmax)
ax.set_ylim(N, 0)

fig.text(0.53, 0.01,
        "Trajectories of the first 30 PCE coefficients of "*L"X_1",
        fontsize=FS, ha="center")

display(gcf())
# savefig("figures/EX1_ReactorPCEX1.pdf")

## compute PDFs and Histogarms
intervals = [(0, 1),
            (-1.5, 0),
            (-1.5, 0), 
            (-1.5, 0), 
            (-1.5, 0), 
            (-1.5, 0)];
steps = 1:10:51;

pdfs = map(zip(steps, intervals)) do (r, intv)
    # cal_distribution(4, x1solPCE[r, :])
    cal_distribution(intv, x1solPCE[r, :])
end

ξ, psi = sample_vector(10^4, L - 1) 
nbins = [8, 8, 8, 8, 8, 8]         # corresponding bin counts

hist_data = map(zip(steps, nbins)) do (k, nb)
    xN = x1solPCE[k, :]' * psi'

    n, bins = hist(xN; bins=nb, density=true, edgecolor="black")
    width = bins[2] - bins[1]
    xrange = bins[1:end-1] .+ width/2

    (n=n, xrange=xrange, width=width)
end

##
fig = figure(figsize=(8.4, 4));
ax = fig.add_subplot(111,projection="3d")

for (y, h) in Iterators.reverse(zip(steps, hist_data))
    width = h.xrange[2] - h.xrange[1]
    bar(h.xrange, h.n, 0.8 * width; zs=y, zdir="y", alpha=0.7)
end

for (y, (freqs, mags)) in Iterators.reverse(zip(steps, pdfs))
    plot3D(freqs, fill(y + 0.2, length(freqs)), mags; c="black", lw=LW)
end

xlabel(L"X_1(k)", size=FS); ylabel(L"k", size=FS); zlabel(L"PDF", size=FS)
ax.set_box_aspect([1.0, 1, 0.6])
ax.view_init(elev=25, azim=-140)
ax.set_zticks(0:1:6)
ax.set_zlim(0,6)

fig.text(0.53, 0.01,
        "Comparison of PDFs and histograms of "*L"10^4"*" samples",
        fontsize=FS, ha="center")

display(gcf())
# savefig("figures/EX1_ReactorDistributionX1.pdf")
