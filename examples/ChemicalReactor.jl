"""
This example is described in the following arXiv preprint:
PolyOCP.jl -- A Julia Package for Stochastic OCPs and MPC
https://arxiv.org/abs/2511.19084
"""

using PolyOCP
using LinearAlgebra
using PyPlot, LaTeXStrings, Random

## --------------------------------------------------------------------------
# Define problem parameters
N = 50

A = [0.95123 0.0;
     0.08833 0.81873]
B = [-0.0048771;
     -0.0020429]
E = [1.0;
     1.0]

Q  = Matrix(1.0 * I, 2, 2)
R  = 1.0
QN = Q

ubx = ([Inf; 0.24], [1.0; 0.1])

# Define X0 and W and generate their PCEs
X0 = [GaussMeasureParametric(0.5, 0.05);
      GaussMeasureParametric(0.1, 0.01)]
W  = [UniformMeasureParametric(-0.0173 * 12, 0.0173 * 12)]

X0PCE = genPCE(X0)
WPCE  = genPCE(W)

# Build the joint basis over the horizon
basis_joint, _, _ = jointPCE(X0PCE, WPCE, N)

## --------------------------------------------------------------------------
# Define and build the stochastic OCP
problem = defineOCP(
    N,
    A, B, E,
    X0PCE.coeff, WPCE.coeff;
    Q = Q, R = R, QN = QN,
    ubx = ubx
)

model = buildOCP(problem)

# Solve stochastic OCP
xsol_coeff, usol_coeff, obj = solveOCP(model)

## --------------------------------------------------------------------------
# The next part solve it 1000 times to get average computation time.
# This block is commented out as it takes long time.

# Ns = 1000
# set_optimizer_attribute(model, "print_level", 0);
# tAll = @elapsed begin 
#     for _ = 1:Ns
#         solveOCP(model)
#     end
# end

## --------------------------------------------------------------------------
# Compare theoretical PDFs with histograms of sampled realizations
steps     = 1:10:N+1
x1sol_coeff = xsol_coeff[1, :, :]
x1PCEs = [OrthonoPCE(basis_joint, vec(x1sol_coeff[k, :])) for k in steps]

# PDFs
intervals = [(0.0, 1.0),
             (-1.5, 0.0),
             (-1.5, 0.0),
             (-1.5, 0.0),
             (-1.5, 0.0),
             (-1.5, 0.0)]
pdfs = [pdfPCE(intervals[i], x1PCEs[i]; N = 4096)
    for i in eachindex(steps)]

# Samples for histograms
Ns  = 10^4
rng = MersenneTwister(1)
samples = [samplePCE(x1PCEs[i], Ns; rng=rng) for i in eachindex(steps)]

fig, ax = plot3d_pdf(
    pdfs, steps;
    samples=samples,
    nbins=8,
    xlabel=L"X_1(k)",
    # ylabel=L"k",
    # zlabel="PDF",
    labelsize=10,
    view=(25, -140),
    zlim=(0, 6),
    pdf_color="black",
    showfig=false

)

fig.text(0.5, 0.08,
    "Comparison of PDFs and histograms of "*L"10^4"*" samples",
    fontsize = 12, ha="center")
display(fig)
# savefig("figures/EX1_ReactorDistributionX1.pdf")

## --------------------------------------------------------------------------
# 3D plot of the first jmax PCE coefficients of X1
x1sol_coeff = xsol_coeff[1, :, :]
jmax = min(30, size(x1sol_coeff, 2))
FS, LW = 10, 2

fig = figure()
ax = fig.add_subplot(111, projection = "3d")

for j in 1:jmax
    t1 = max(0, j - 3)
    tx = t1:N
    plot3D(j * ones(length(tx)), tx, x1sol_coeff[tx .+ 1, j]; linewidth = LW)
end

xlabel(L"j", size = FS)
ylabel(L"k", size = FS)
zlabel(L"x_1^{j,*}", fontsize = FS)
ax.set_box_aspect([1.0, 1, 0.6])
ax.view_init(elev = 20, azim = -65)
ax.set_yticks(0:10:N)
ax.set_xlim(0, jmax)
ax.set_ylim(N, 0)

fig.text(0.55, 0.1,
    "Trajectories of the first $(jmax) PCE coefficients of " * string(L"X_1"),
    fontsize = 12, ha = "center")

display(gcf())
# savefig("figures/EX1_ReactorPCEX1.pdf")