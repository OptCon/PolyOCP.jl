module Visualization
using PyPlot, LaTeXStrings

"""
    plot_quick(x::AbstractVector, y::AbstractVector;
        fig=nothing, ax=nothing,
        figsize=nothing,
        label::Union{Nothing,String}=nothing,
        linestyle::AbstractString="-",
        linewidth::Real=2.0,
        color=nothing,
        xlabel=nothing,
        ylabel=nothing,
        title=nothing,
        grid::Bool=true,
        legend::Bool=true,
        xlim=nothing,
        ylim=nothing,
        showfig::Bool=true)

Generic 2D line plot.
"""
function plot_quick(x::AbstractVector, y::AbstractVecOrMat;
    fig=nothing,
    ax=nothing,
    figsize::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    label::Union{Nothing,AbstractString}=nothing,
    xlabel::Union{Nothing,AbstractString}=nothing,
    ylabel::Union{Nothing,AbstractString}=nothing,
    labelsize::Real=10,
    title=nothing,
    linestyle::AbstractString="-",
    linewidth::Real=2.0,
    color=nothing,
    grid::Bool=true,
    legend::Bool=true,
    xlim::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    ylim::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    xticks::Union{Nothing,AbstractVector{<:Real}}=nothing,
    yticks::Union{Nothing,AbstractVector{<:Real}}=nothing,
    ticksize::Real=10,
    showfig::Bool=true
)
    length(x) == size(y,1) ||
        throw(ArgumentError("`x` and `y` must have the same length"))

    if ax === nothing
        if figsize === nothing
            fig, ax = PyPlot.subplots()
        else
            fig, ax = PyPlot.subplots(figsize=figsize)
        end
    elseif fig === nothing
        fig = ax.figure
    end

    if color === nothing
        ax.plot(x, y; linestyle=linestyle, linewidth=linewidth, label=label)
    else
        ax.plot(x, y; linestyle=linestyle, linewidth=linewidth,
                color=color, label=label)
    end

    xlabel !== nothing && ax.set_xlabel(xlabel, fontsize=labelsize)
    ylabel !== nothing && ax.set_ylabel(ylabel, fontsize=labelsize)
    title  !== nothing && ax.set_title(title, fontsize=labelsize)

    grid && ax.grid(true)

    xlim !== nothing && ax.set_xlim(xlim...)
    ylim !== nothing && ax.set_ylim(ylim...)

    if legend && label !== nothing
        ax.legend()
    end

    ax.tick_params(axis="both", labelsize=ticksize)

    showfig && display(fig)

    return fig, ax
end

"""
    plot_pdf(x, pdf; truncate=false, threshold=1e-6, kwargs...)

Plot a PDF with optional truncation of near-zero regions.
"""
function plot_pdf(x::AbstractVector, pdf::AbstractVector;
    truncate::Bool=false,
    threshold::Real=1e-6,
    kwargs...
)
    length(x) == length(pdf) ||
        throw(ArgumentError("`x` and `pdf` must have same length"))

    all(isfinite, pdf) ||
        throw(ArgumentError("`pdf` must be finite"))

    if !truncate
        return plot_quick(x, pdf; kwargs...)
    end

    inds = findall(pdf .> threshold)
    isempty(inds) && throw(ArgumentError("No values above threshold"))

    i1 = first(inds)
    i2 = last(inds)

    @info "Interval of PDF truncated" interval=(x[i1], x[i2]) threshold=threshold

    return plot_quick(x[i1:i2], pdf[i1:i2]; kwargs...)
end

"""
    plot_traj_PCE(var::AbstractArray{<:Real,3}, j::Int; kwargs...)

Plot the trajectory of the `j`-th PCE coefficient for all components.
Input size must be `(ncomp, Nt, L)`.
"""
function plot_traj_PCE(var::AbstractArray{<:Real,3}, j::Int;
    ylabel::Union{Nothing,AbstractString}="x",
    xlabel::Union{Nothing,AbstractString}=L"k",
    labelsize::Real=10,
    figsize::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    xlim::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    showfig::Bool=true,
    kwargs...)

    ncomp, Nt, L = size(var)

    0 ≤ j ≤ L-1 ||
        throw(ArgumentError("`j` must satisfy 0 ≤ j ≤ size(var, 3)-1"))

    k = 0:Nt-1
    xlim === nothing && (xlim = (0, Nt-1))

    if figsize === nothing
        fig, axes = PyPlot.subplots(ncomp, 1, sharex=true)
    else
        fig, axes = PyPlot.subplots(ncomp, 1, sharex=true, figsize=figsize)
    end
    axes = ncomp == 1 ? [axes] : axes

    for i in 1:ncomp
        ylabel_i = latexstring("$(ylabel)_$(i)^{$(j),*}")

        plot_quick(k, vec(var[i, :, j+1]);
            fig=fig,
            ax=axes[i],
            ylabel=ylabel_i,
            xlabel=nothing,
            labelsize=labelsize,
            xlim=xlim,
            kwargs...,
            showfig=false,
            legend=false)
    end
    xlabel !== nothing && axes[ncomp].set_xlabel(xlabel, fontsize=labelsize)

    showfig && display(fig)
    return fig, axes
end

"""
    plot_traj(var::AbstractArray{<:Real,3}; kwargs...)

Plot sampled trajectories.
Input size must be `(ncomp, Nt, Ns)`, where `Ns` is the number of trajectories.
"""
function plot_traj(var::AbstractArray{<:Real,3};
    ylabel::Union{Nothing,AbstractString}=L"x",
    xlabel::Union{Nothing,AbstractString}=L"k",
    labelsize::Real=10,
    figsize::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    xlim::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    showfig::Bool=true,
    kwargs...)

    ncomp, Nt, Ns = size(var)

    k = 0:Nt-1
    xlim === nothing && (xlim = (0, Nt-1))

    if figsize === nothing
        fig, axes = PyPlot.subplots(ncomp, 1, sharex=true)
    else
        fig, axes = PyPlot.subplots(ncomp, 1, sharex=true, figsize=figsize)
    end
    axes = ncomp == 1 ? [axes] : axes

    for i in 1:ncomp
        ylabel_i = latexstring("$(ylabel)_$(i)")

        plot_quick(k, var[i, :, :];
            fig=fig,
            ax=axes[i],
            ylabel=ylabel_i,
            xlabel=nothing,
            labelsize=labelsize,
            xlim=xlim,
            kwargs...,
            showfig=false,
            legend=false)
    end
    xlabel !== nothing && axes[ncomp].set_xlabel(xlabel, fontsize=labelsize)

    showfig && display(fig)
    return fig, axes
end

"""
    plot3d_pdf(...; hist_color=nothing, hist_alpha=0.7, ...)

Plot PDFs in 3D with optional histogram overlays.

# Keyword Arguments
- `hist_color`: Controls the color of histogram bars.
    - If `nothing` (default), Matplotlib's default color cycle is used.
    - If a single color (e.g., `"tab:blue"` or RGB tuple), the same color is applied to all histograms.
    - If a vector of colors, each entry is used for the corresponding time step in `steps`.

- `hist_alpha`: Transparency of histogram bars (default: `0.7`).

# Notes
- Histogram bars are plotted using `bar(...; zs=y, zdir="y")`, i.e., stacked along the time axis.
- The color setting affects only the histogram bars, not the PDF curves.
"""
function plot3d_pdf(pdfs::AbstractVector, steps::AbstractVector;
    samples=nothing,
    nbins=10,
    kwargs...)

    hist_data = samples === nothing ? nothing : _hist_data(samples, nbins)

    return _plot3d_distribution(
        steps;
        pdfs=pdfs,
        hist_data=hist_data,
        kwargs...)
end

function plot3d_hist(samples::AbstractVector, steps::AbstractVector;
    nbins=10,
    kwargs...)

    hist_data = _hist_data(samples, nbins)

    return _plot3d_distribution(
        steps;
        hist_data=hist_data,
        kwargs...)
end

function _plot3d_distribution(
    steps::AbstractVector{<:Real};
    pdfs::Union{Nothing,AbstractVector}=nothing,
    hist_data::Union{Nothing,AbstractVector}=nothing,
    xlabel::Union{Nothing,AbstractString}=L"X",
    ylabel::Union{Nothing,AbstractString}=L"k",
    zlabel::Union{Nothing,AbstractString}=L"PDF",
    labelsize::Real=10,
    figsize::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    view::Union{Nothing,Tuple{<:Real,<:Real}}=(25, -140),
    box_aspect::Union{Nothing,NTuple{3,<:Real},AbstractVector{<:Real}}=(1.0, 1.0, 0.6),
    pdf_color=nothing,
    pdf_linewidth::Real=2.0,
    pdf_offset::Real=0.0,
    hist_color=nothing,
    hist_alpha::Real=0.7,
    hist_edgecolor::Union{Nothing,AbstractString}= "none",
    hist_linewidth::Real=0.5,
    xlim::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    ylim::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    zlim::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    xticks::Union{Nothing,AbstractVector{<:Real}}=nothing,
    yticks::Union{Nothing,AbstractVector{<:Real}}=nothing,
    zticks::Union{Nothing,AbstractVector{<:Real}}=nothing,
    ticksize::Real=10,
    showfig::Bool=true
)
    (pdfs !== nothing || hist_data !== nothing) ||
        throw(ArgumentError("At least one of `pdfs` or `hist_data` must be provided."))

    pdfs !== nothing && length(pdfs) == length(steps) ||
        pdfs === nothing ||
        throw(ArgumentError("`pdfs` and `steps` must have the same length"))

    hist_data !== nothing && length(hist_data) == length(steps) ||
        hist_data === nothing ||
        throw(ArgumentError("`hist_data` and `steps` must have the same length"))

    fig = figsize === nothing ? figure() : figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if hist_data !== nothing
        for (y, h) in Iterators.reverse(zip(steps, hist_data))
            if hist_color === nothing
                bar(h.xrange, h.n, 0.8*h.width;
                    zs=y, zdir="y",
                    alpha=hist_alpha,
                    edgecolor=hist_edgecolor,
                    linewidth=hist_linewidth)
            else
                bar(h.xrange, h.n, 0.8*h.width;
                    zs=y, zdir="y",
                    alpha=hist_alpha,
                    color=hist_color,
                    edgecolor=hist_edgecolor,
                    linewidth=hist_linewidth)
            end
        end
    end

    if pdfs !== nothing
        for (y, (xpdf, ypdf)) in Iterators.reverse(zip(steps, pdfs))
            yvec = fill(y + pdf_offset, length(xpdf))

            if pdf_color === nothing
                plot3D(xpdf, yvec, ypdf; lw=pdf_linewidth)
            else
                plot3D(xpdf, yvec, ypdf; c=pdf_color, lw=pdf_linewidth)
            end
        end
    end

    xlabel !== nothing && ax.set_xlabel(xlabel, fontsize=labelsize)
    ylabel !== nothing && ax.set_ylabel(ylabel, fontsize=labelsize)
    zlabel !== nothing && ax.set_zlabel(zlabel, fontsize=labelsize,
                                        rotation=90, labelpad=-2.5)

    ax.set_box_aspect(box_aspect)
    ax.view_init(elev=view[1], azim=view[2])

    xlim !== nothing && ax.set_xlim(xlim...)
    ylim !== nothing && ax.set_ylim(ylim...)
    zlim !== nothing && ax.set_zlim(zlim...)

    xticks !== nothing && ax.set_xticks(xticks)
    yticks !== nothing && ax.set_yticks(yticks)
    zticks !== nothing && ax.set_zticks(zticks)

    ax.tick_params(axis="both", labelsize=ticksize)
    ax.tick_params(axis="z", labelsize=ticksize)

    showfig && display(fig)
    
    return fig, ax
end

function _hist_data(samples::AbstractVector, nbins)
    nbins_vec = nbins isa Integer ? fill(nbins, length(samples)) : collect(nbins)

    length(nbins_vec) == length(samples) ||
        throw(ArgumentError("`nbins` must be scalar or match `samples`."))

    map(zip(samples, nbins_vec)) do (xk, nb)
        n, bins = hist(xk; bins=nb, density=true)
        width = bins[2] - bins[1]
        xrange = bins[1:end-1] .+ width / 2
        (n=n, xrange=xrange, width=width)
    end
end

end