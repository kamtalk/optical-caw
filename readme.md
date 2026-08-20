# Optical-CAW: Exact Dual-Space Spectral Transport for Non-Paraxial Optics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Physics: Optics](https://img.shields.io/badge/Physics-Physical%20Optics-orange.svg)]()

Official Python implementation and reproducible benchmark suite for the paper:  
**"Unifying Helmholtz Optics and Wigner–Moyal Mechanics"**  
*Author:* Kenneth A. Menard (*CREOL, The College of Optics and Photonics / Independent Researcher*)  
*Email:* `ken.menard@uwaterloo.ca`

---

<p align="center">
  <!-- 🛑 Note: Upload your 3-panel image to your repo (e.g. as docs/phase_space_catastrophe.png) so it renders here -->
  <img src="docs/phase_space_catastrophe.png" alt="Wigner Phase-Space Vortex Topology" width="900"/>
  <br>
  <em><b>Exact Non-Paraxial Focal Catastrophe:</b> (a) Phase-space nodal lines and streamlines resolving exact Wigner vortex cores. (b) Core zoom showing closed 2$\pi$ phase-space circulation. (c) Corresponding real-space highly oscillatory focal intensity.</em>
</p>

---

## 🔬 Overview

Standard Beam Propagation Methods (BPM) rely on paraxial approximations that break down at wide angles ($\text{NA} > 0.7$), while finite-order differential Wigner–Moyal expansions generate infinite derivative hierarchies that cause numerical stiffness and non-physical caustic tearing.

**Optical-CAW** provides an exact, truncation-free dual-space spectral algorithm for non-paraxial scalar Helmholtz beam propagation:
* **Closed-Form Dual-Space Resummation:** Maps phase space $(x, k_x) \to (\xi, \eta)$ where the infinite Helmholtz kinetic and refractive Moyal hierarchies collapse into exact, closed-form unitary phase factors $\Phi_T(\xi, k_x)$ and $\Phi_V(x, \eta)$.
* **Machine-Precision Unitarity:** Strictly conserves total optical power and state purity ($|\Phi| = 1$, norm drift $< 10^{-14}$).
* **Unconditional Geometric Stability:** Governed by an optical Shannon–Nyquist step bound $\Delta z < \frac{k_{x,\max}}{k_0 \max|\partial_x n|}$ that is **independent of spatial grid resolution $\Delta x$**, eliminating Courant–Friedrichs–Lewy (CFL) time-step stiffness.
* **Topological Wigner Vortices:** Captures the topological nucleation of classical phase-space vortex–antivortex pairs in aberrated caustic catastrophes.

---

## ⚡ Quick Start & Installation

```bash
git clone https://github.com/kamtalk/optical-caw.git
cd optical-caw
pip install -r requirements.txt
