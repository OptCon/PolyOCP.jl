# PolyOCP -- A package for stochastic OCP and MPC in the PCE framework

[![Paper@arXiv](https://img.shields.io/badge/arXiv-2511.19084-green.svg)](https://arxiv.org/abs/2511.19084)

PolyOCP is an easy-to-use julia package for modeling and solving stochastic Optimal Control Problems (OCPs) and Model Predictive Control (MPC).

## Installation
The package requires `Julia 1.11` or newer.
In `Julia` switch to the package manager

```julia
using Pkg
Pkg.add("PolyOCP")
```

This will install `PolyOCP` and its dependencies.
Once `PolyOCP` is installed, load the package:

```julia
using PolyOCP
```
For quick guidance on how to use the toolbox, see the examples provided in the package.


## Overview and Motivation

Stochastic OCPs and MPCs are subject to uncertainties, which affect constraints, objectives, or dynamics. PolyOCP simplifies the modeling and solving of such problems by providing:

- A modular framework for defining stochastic OCPs.
- Polynomial Chaos Expansion for uncertainty propagation.
- Solver interfaces for JuMP-compatible solvers like Ipopt and Mosek.

This is the first stable version of the package. More features, e.g. visualization of computation results, are on the way.

## Key Features

- **Polynomial Chaos Expansion:**
  - The PCE module is developed base on PolyChaos.jl.
  - Tools for constructing orthonormal polynomial bases and coefficients for uncertainty quantification.
  - Supports canonical probability distributions (e.g., Gaussian, Uniform) and multi-dimensional PCE.


- **JuMP Integration:**

  - Simplifies the conversion of stochastic OCPs into JuMP models.
  - Supports nonlinear and conic solvers like Ipopt and Mosek.
  - Offers flexibility in solver settings and optimization strategies.

- **Example Problems:**

  - Includes practical examples such as chemical reactor and a four tank system.
  - Demonstrates the application of PCE and stochastic OCP techniques.
  - Provides a starting point for users to adapt the toolkit to their own problems.

- **Extensibility:**
  - Modular design allows users to add custom solvers, constraints, objective functions, etc.
  - Easily extendable to new problem formulations and stochastic modeling approaches.


## Documentation

Comprehensive documentation is still in developing. For detailed examples, see the `examples/` directory.

## Citing

If you use `PolyOCP.jl` in your research, please cite [this paper](https://arxiv.org/abs/2511.19084):
```
@Article{ou2025polyocp,
     title = {PolyOCP.jl -- A Julia Package for Stochastic OCPs and MPC}, 
    author = {Ruchuan Ou and Learta Januzi and Jonas Schie{\ss}l and Michael Heinrich Baumann and Lars Gr{\"u}ne and Timm Faulwasser},
      year = {2025},
   journal = {arXiv: 2511.19084}
}
```


## Acknowledgements

This project was developed at the Technical University of Hamburg, Institue of Control Systems.
The authors acknowledge funding by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - project number 499435839.
