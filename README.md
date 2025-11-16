# PolyOCP -- A package for stochastic OCP and MPC in the PCE framework

## Description

PolyOCP is a toolkit written in Julia for modeling and solving stochastic Optimal Control Problems (OCPs) and Model Predictive Control (MPC). This toolbox provides mechanisms for Polynomial Chaos Expansion (PCE), parametric measures, and solver interfaces to address uncertainties in OCPs.

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
  - Supports common probability distributions (e.g., Gaussian, Uniform) and multi-dimensional PCE.
  - Enables efficient propagation of uncertainty through system dynamics.

- **Parametric Measures:**

  - Provides canonical measures (i.e., Gaussian, Beta) and allows user-defined probability measures.
  - Facilitates modeling of complex stochastic inputs, such as uncertain demand or supply.
  - Ensures compatibility with PCE for seamless integration.

- **JuMP Integration:**

  - Simplifies the conversion of stochastic OCPs into JuMP models.
  - Supports nonlinear and conic solvers like Ipopt and Mosek.
  - Offers flexibility in solver settings and optimization strategies.

- **Example Problems:**

  - Includes practical examples such a chemical reactor and a four tank system.
  - Demonstrates the application of PCE and stochastic OCP techniques.
  - Provides a starting point for users to adapt the toolkit to their own problems.

- **Extensibility:**
  - Modular design allows users to add custom solvers, constraints, objective functions, etc.
  - Easily extendable to new problem formulations and stochastic modeling approaches.
  - Encourages experimentation and adaptation for research and practical applications.


## Documentation

Comprehensive documentation is still in developing. For detailed examples, see the `examples/` directory.

## Citing

A paper on this toolbox will appear on arXiv soon.


## Acknowledgements

This project was developed at the Technical University of Hamburg, Institue of Control Systems.
The authors acknowledge funding by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - project number 499435839.
