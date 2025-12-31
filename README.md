# Solving the 1D Schrödinger Equation Using Physics-Informed Neural Networks (PINNs)

This repository contains a Physics-Informed Neural Network (PINN) implementation to solve the 1D time-independent Schrödinger eigenvalue problem for the **infinite square well**.  
The model learns both the **wave function** and the **energy eigenvalue** by minimizing the PDE residual together with normalization and (for excited states) orthogonality constraints.

## Problem Setup: Infinite Square Well (L = 1)

Inside the well (dimensionless form used here):
\[
-\psi''(x) = E\,\psi(x), \quad x\in(0,L)
\]
Boundary conditions:
\[
\psi(0)=0,\quad \psi(L)=0
\]
Analytical reference solutions:
\[
\psi_n(x)=\sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right), \qquad
E_n=\left(\frac{n\pi}{L}\right)^2
\]

## PINN Method (Summary)

### Ansatz (enforces boundary conditions)
We embed boundary conditions directly into the trial function:
\[
\psi_\theta(x)=x(L-x)\,[1+\mathcal{N}_\theta(x)]
\]
so that \(\psi_\theta(0)=\psi_\theta(L)=0\) automatically.

### Residual (physics loss)
\[
R(x)=-\psi''(x)-E\psi(x)
\]
Second derivatives are computed via **PyTorch autograd**.

### Training objectives
- **PDE loss:** mean squared residual at collocation points
- **Normalization loss:** enforces \(\int_0^L \psi^2(x)\,dx = 1\)
- **Orthogonality loss (n=2):** enforces \(\int_0^L \psi_2(x)\psi_1(x)\,dx = 0\)

The total loss is:
\[
\mathcal{L}=\mathcal{L}_{PDE}+w_{norm}\mathcal{L}_{norm}+w_{ortho}\mathcal{L}_{ortho}
\]

## Results (example run)

The script prints training logs such as:

- `device: cpu`
- learned energies (PINN vs analytic)
- residual statistics (max/mean |R|)

Example accuracy from an example run:
- Ground state (n=1): relative error ~ \(10^{-4}\)
- First excited state (n=2): relative error ~ \(10^{-3}\)

## How to Run

### 1) Install dependencies
```bash
pip install numpy torch matplotlib
