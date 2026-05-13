# Active Matter Simulation

This repository contains numerical implementations and simulations of several **active matter and non-equilibrium stochastic systems**, including nonlinear friction models, energy depot dynamics, and coupled molecular motor systems.

## Repository Structure

The repository is organised into separate folders, where each folder corresponds to a different class of physical models:

- **Active Particle Models**
  - Rayleigh–Helmholtz friction model
  - Schienbein–Gruler friction model
  - Energy depot model
- **Coupled Molecular Motor systems**

Each folder is self-contained and focuses on a specific dynamical framework.

## Code Structure

- `.py` files contain the **core numerical integrators and simulation engines**.
- Jupyter notebooks (`.ipynb`) provide:
  - Model explanations
  - Physical intuition
  - Example simulations
  - Visualisation of trajectories and statistical properties

The notebooks are intended as both **documentation and usage guides** for the corresponding simulation codes.

## Numerical Method

All stochastic differential equations in this repository are solved using the **Euler–Maruyama integration scheme**, which is well-suited to overdamped Langevin dynamics and multiplicative-noise systems.

## Theoretical Background

The models implemented here are strongly inspired by:

> C. Bechinger et al., *Active particles in complex and crowded environments*,  
> arXiv:1202.2442 (2012)

This review provides an excellent introduction to **active matter physics, stochastic propulsion mechanisms, and non-equilibrium statistical behaviour**. It is highly recommended as further reading.

---

## Citation / Reference

If you use this code, please cite the above review paper and acknowledge this repository.
