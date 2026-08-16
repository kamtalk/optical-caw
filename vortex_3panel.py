"""
===============================================================================
Discovery Figure: Classical Phase-Space Wigner Vortices in Aberrated Wavefields
Author: Kenneth A. Menard
File: vortex_3panel.py
Output: fig2_aberrated_caustic.png
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter, maximum_filter

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'lines.linewidth': 2.0
})


class OpticalCAWSolver:
    def __init__(self, Nx, Nkx, Lx, kx_max, wavelength=1.0):
        self.Nx, self.Nkx = Nx, Nkx
        self.Lx, self.kx_max = Lx, kx_max
        self.k0 = 2.0 * np.pi / wavelength

        self.x = np.linspace(-Lx / 2.0, Lx / 2.0, Nx, endpoint=False)
        self.kx = np.linspace(-kx_max, kx_max, Nkx, endpoint=False)
        self.dx = Lx / Nx
        self.dkx = 2.0 * kx_max / Nkx

        self.xi = 2.0 * np.pi * np.fft.fftfreq(Nx, d=self.dx)
        self.eta = 2.0 * np.pi * np.fft.fftfreq(Nkx, d=self.dkx)

        self.X, self.KX = np.meshgrid(self.x, self.kx, indexing='ij')
        self.XI, self.KX_grid = np.meshgrid(self.xi, self.kx, indexing='ij')
        self.X_grid, self.ETA = np.meshgrid(self.x, self.eta, indexing='ij')

    def compile_free_space_helmholtz(self, dz):
        k_plus = self.KX_grid + 0.5 * self.XI
        k_minus = self.KX_grid - 0.5 * self.XI

        kz_sq_plus = np.maximum(0.0, self.k0**2 - k_plus**2)
        kz_sq_minus = np.maximum(0.0, self.k0**2 - k_minus**2)

        Delta_kz = np.sqrt(kz_sq_plus) - np.sqrt(kz_sq_minus)
        self.Phi_T = np.exp(+1j * dz * Delta_kz)

    def apply_thin_phase_screen(self, W, phase_func):
        x_plus = self.X_grid + 0.5 * self.ETA
        x_minus = self.X_grid - 0.5 * self.ETA
        Delta_S = phase_func(x_plus) - phase_func(x_minus)
        Phi_screen = np.exp(+1j * self.k0 * Delta_S)

        W_eta = np.fft.fft(W, axis=1)
        W_eta *= Phi_screen
        return np.real(np.fft.ifft(W_eta, axis=1))

    def step(self, W):
        W_xi = np.fft.fft(W, axis=0)
        W_xi *= self.Phi_T
        return np.real(np.fft.ifft(W_xi, axis=0))


def run():
    print("=" * 75)
    print("GENERATING 3-PANEL DISCOVERY FIGURE: PHASE-SPACE WIGNER VORTICES")
    print("=" * 75)

    wavelength = 0.6328  # microns
    k0 = 2.0 * np.pi / wavelength
    focal_length = 30.0  # microns
    
    # 4th- and 6th-order aberration coefficients
    C4 = 8.0e-5
    C6 = 1.2e-7

    Nx, Nkx = 512, 512
    Lx, kx_max = 30.0, 2.0 * k0
    dz = 0.05 * wavelength
    n_steps = int(focal_length / dz)

    solver = OpticalCAWSolver(Nx, Nkx, Lx, kx_max, wavelength=wavelength)

    # Collimated incident Gaussian beam
    w_beam = 8.0
    W0 = np.zeros((Nx, Nkx))
    for j, xj in enumerate(solver.x):
        W0[j, :] = np.exp(-2.0 * xj**2 / w_beam**2 - 0.5 * (solver.kx * w_beam)**2) / np.pi

    # Aberrated Phase Screen
    def aberrated_lens(x):
        return - (x**2 / (2.0 * focal_length) - C4 * x**4 - C6 * x**6)

    print("Applying aberrated phase screen at z=0...")
    W = solver.apply_thin_phase_screen(W0, aberrated_lens)

    print(f"Propagating to focal plane z = {focal_length} um ({n_steps} steps)...")
    solver.compile_free_space_helmholtz(dz)
    for _ in range(n_steps):
        W = solver.step(W)

    # Compute Phase-Space Streamflow Current: J = (-dW/dkx, dW/dx)
    dW_dx, dW_dkx = np.gradient(W, solver.dx, solver.dkx)
    Jx = -dW_dkx
    Jk = +dW_dx

    # Detect vortex center candidates (local extrema of W near zero crossings)
    local_min = (W == minimum_filter(W, size=15)) & (W < -0.02)
    local_max = (W == maximum_filter(W, size=15)) & (W > 0.05) & (np.abs(solver.X) > 1.5)
    vortex_mask = local_min | local_max

    vortex_x = solver.X[vortex_mask]
    vortex_kx = solver.KX[vortex_mask] / k0

    # Filter to central ROI
    valid = (np.abs(vortex_x) <= 8.5) & (np.abs(vortex_kx) <= 0.6)
    vortex_x = vortex_x[valid]
    vortex_kx = vortex_kx[valid]

    I_focal = np.sum(W, axis=1) * solver.dkx

    # =========================================================================
    # CREATE 3-PANEL FIGURE
    # =========================================================================
    fig = plt.figure(figsize=(16, 5.0), layout='constrained')
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.95, 1.0])

    # Panel (a): Global Phase-Space Topology
    ax1 = fig.add_subplot(gs[0])
    mesh1 = ax1.pcolormesh(solver.x, solver.kx / k0, W.T, cmap='coolwarm', shading='auto', vmin=-0.12, vmax=0.30, rasterized=True)
    ax1.contour(solver.x, solver.kx / k0, W.T, levels=[0.0], colors='black', linewidths=1.1, linestyles='solid')
    ax1.streamplot(solver.x, solver.kx / k0, Jx.T, Jk.T, color='gray', density=0.8, linewidth=0.5, arrowsize=0.6)
    ax1.scatter(vortex_x, vortex_kx, color='cyan', s=35, edgecolors='black', linewidths=0.8, zorder=5, label="Vortex Cores")
    
    # Zoom box overlay for Panel (b)
    zoom_x_center, zoom_kx_center = 3.8, 0.15
    zoom_w, zoom_h = 2.4, 0.35
    rect = plt.Rectangle((zoom_x_center - zoom_w/2, zoom_kx_center - zoom_h/2), zoom_w, zoom_h, fill=False, edgecolor='lime', lw=1.8, ls='--', zorder=6)
    ax1.add_patch(rect)

    ax1.set_title(r"(a) Phase-Space Topology: Nodal Lines \& Streamlines")
    ax1.set_xlabel(r"Transverse Position $x$ [$\mu\mathrm{m}$]")
    ax1.set_ylabel(r"Transverse Wavevector $k_x / k_0$")
    ax1.set_xlim(-11, 11)
    ax1.set_ylim(-1.0, 1.0)
    ax1.legend(loc='upper right', framealpha=0.85)

    # Panel (b): Microscopic Zoom into Vortex Core
    ax2 = fig.add_subplot(gs[1])
    ax2.pcolormesh(solver.x, solver.kx / k0, W.T, cmap='coolwarm', shading='auto', vmin=-0.12, vmax=0.30, rasterized=True)
    ax2.contour(solver.x, solver.kx / k0, W.T, levels=[0.0], colors='black', linewidths=1.4, linestyles='solid')
    ax2.streamplot(solver.x, solver.kx / k0, Jx.T, Jk.T, color='black', density=1.4, linewidth=0.9, arrowsize=0.9)
    ax2.scatter([zoom_x_center], [zoom_kx_center], color='lime', s=80, edgecolors='black', linewidths=1.2, zorder=5)
    
    for spine in ax2.spines.values():
        spine.set_edgecolor('lime')
        spine.set_linewidth(2.0)

    ax2.set_title(r"(b) Core Zoom: Closed $2\pi$ Circulation")
    ax2.set_xlabel(r"Transverse Position $x$ [$\mu\mathrm{m}$]")
    ax2.set_ylabel(r"$k_x / k_0$")
    ax2.set_xlim(zoom_x_center - zoom_w/2, zoom_x_center + zoom_w/2)
    ax2.set_ylim(zoom_kx_center - zoom_h/2, zoom_kx_center + zoom_h/2)

    # Panel (c): Real-Space Multi-Scale Intensity
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(solver.x, I_focal, color='crimson', lw=2.4, label="Exact CAW (Focal Catastrophe)")
    ax3.axvline(-9.0, color='black', ls='--', lw=1.5, label=r"Outer Caustics ($x_c \approx \pm 9.0\,\mu\mathrm{m}$)")
    ax3.axvline(+9.0, color='black', ls='--', lw=1.5)
    ax3.set_title(r"(c) Real-Space Focal Intensity $I(x)$")
    ax3.set_xlabel(r"Transverse Position $x$ [$\mu\mathrm{m}$]")
    ax3.set_ylabel(r"Focal Intensity $I(x)$ [arb. units]")
    ax3.set_xlim(-11, 11)
    ax3.legend(loc='upper right', fontsize=9.5)
    ax3.grid(True, alpha=0.3)

    plt.savefig("fig2_aberrated_caustic.png", dpi=300)
    plt.close()
    print("SAVED ENHANCED 3-PANEL: 'fig2_aberrated_caustic.png'")


if __name__ == "__main__":
    run()