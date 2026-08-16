"""
===============================================================================
Paper #4: Unifying Helmholtz Optics and Wigner–Moyal Mechanics
Author: Kenneth A. Menard (Waterloo, Ontario, Canada)
File: 4optics.py
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Global plotting aesthetics
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'lines.linewidth': 2.0
})


# =============================================================================
# 1. OPTICAL CAW SPECTRAL SOLVER ENGINE
# =============================================================================
class OpticalCAWSolver:
    def __init__(self, Nx, Nkx, Lx, kx_max, wavelength=1.0, n0=1.0):
        self.Nx = Nx
        self.Nkx = Nkx
        self.Lx = Lx
        self.kx_max = kx_max
        self.wavelength = wavelength
        self.n0 = n0
        self.k0 = 2.0 * np.pi / wavelength

        # Real-space (x) and Transverse Momentum (kx) grids
        self.x = np.linspace(-Lx / 2.0, Lx / 2.0, Nx, endpoint=False)
        self.kx = np.linspace(-kx_max, kx_max, Nkx, endpoint=False)
        self.dx = Lx / Nx
        self.dkx = 2.0 * kx_max / Nkx

        # Dual-space conjugate frequency grids (unshifted FFT ordering)
        self.xi = 2.0 * np.pi * np.fft.fftfreq(Nx, d=self.dx)    # Conjugate to x
        self.eta = 2.0 * np.pi * np.fft.fftfreq(Nkx, d=self.dkx)  # Conjugate to kx

        # 2D Meshgrids
        self.X, self.KX = np.meshgrid(self.x, self.kx, indexing='ij')
        self.XI, self.KX_grid = np.meshgrid(self.xi, self.kx, indexing='ij')
        self.X_grid, self.ETA = np.meshgrid(self.x, self.eta, indexing='ij')

        self.Phi_T = None
        self.Phi_V_half = None

    def compile_free_space_helmholtz(self, dz):
        """Exact Non-Paraxial Helmholtz Kinetic Phase Factor in dual xi-space"""
        k_plus = self.KX_grid + 0.5 * self.XI
        k_minus = self.KX_grid - 0.5 * self.XI

        kz_sq_plus = np.maximum(0.0, (self.k0 * self.n0)**2 - k_plus**2)
        kz_sq_minus = np.maximum(0.0, (self.k0 * self.n0)**2 - k_minus**2)

        Delta_kz = np.sqrt(kz_sq_plus) - np.sqrt(kz_sq_minus)
        # +1j sign ensures exact forward ray and wave advection
        self.Phi_T = np.exp(+1j * dz * Delta_kz)

    def compile_paraxial_kinetic(self, dz):
        """Paraxial Fresnel Kinetic Phase"""
        Delta_kz_parax = - (self.KX_grid * self.XI) / self.k0
        self.Phi_T = np.exp(+1j * dz * Delta_kz_parax)

    def compile_refractive_medium(self, dz, n_profile_func):
        """Exact Inhomogeneous Refractive Phase in dual eta-space"""
        x_plus = self.X_grid + 0.5 * self.ETA
        x_minus = self.X_grid - 0.5 * self.ETA
        Delta_n = n_profile_func(x_plus) - n_profile_func(x_minus)
        self.Phi_V_half = np.exp(+1j * (self.k0 * dz / 2.0) * Delta_n)

    def compile_photonic_lattice(self, dz, coupling_C, pitch_d):
        """Exact Tight-Binding Photonic Lattice Phase"""
        Delta_E = 4.0 * coupling_C * np.sin(self.KX_grid * pitch_d) * np.sin(0.5 * self.XI * pitch_d)
        self.Phi_T = np.exp(+1j * dz * Delta_E)

    def apply_thin_phase_screen(self, W, phase_func):
        """Applies a thin lens/phase screen S(x) directly in eta-space once at z=0"""
        x_plus = self.X_grid + 0.5 * self.ETA
        x_minus = self.X_grid - 0.5 * self.ETA
        Delta_S = phase_func(x_plus) - phase_func(x_minus)
        Phi_screen = np.exp(+1j * self.k0 * Delta_S)
        
        W_eta = np.fft.fft(W, axis=1)
        W_eta *= Phi_screen
        return np.real(np.fft.ifft(W_eta, axis=1))

    def step(self, W):
        """2nd-order Symplectic Strang Splitting Step"""
        if self.Phi_V_half is not None:
            W_eta = np.fft.fft(W, axis=1)
            W_eta *= self.Phi_V_half
            W = np.real(np.fft.ifft(W_eta, axis=1))

        W_xi = np.fft.fft(W, axis=0)
        W_xi *= self.Phi_T
        W = np.real(np.fft.ifft(W_xi, axis=0))

        if self.Phi_V_half is not None:
            W_eta = np.fft.fft(W, axis=1)
            W_eta *= self.Phi_V_half
            W = np.real(np.fft.ifft(W_eta, axis=1))

        return W


# =============================================================================
# 2. RUN BENCHMARKS AND GENERATE PUBLICATION FIGURES
# =============================================================================
def run():
    print("=" * 75)
    print("RUNNING PAPER #4: EXACT DUAL-SPACE OPTICAL WIGNER SIMULATIONS")
    print("=" * 75)

    # =========================================================================
    # FIG 1: High-NA Wide-Angle Diffraction (NA = 0.85)
    # =========================================================================
    print("\n[1/3] Generating Figure 1: High-NA Diffraction...")
    wavelength = 0.6328  # He-Ne wavelength in microns
    k0 = 2.0 * np.pi / wavelength
    w0 = 0.50 * wavelength
    z_prop = 12.0 * wavelength

    Nx, Nkx = 512, 512
    Lx, kx_max = 30.0, 2.0 * k0
    dz = 0.05 * wavelength
    n_steps = int(z_prop / dz)

    solver = OpticalCAWSolver(Nx, Nkx, Lx, kx_max, wavelength=wavelength)

    # Initial Gaussian Wigner State
    W0 = np.zeros((Nx, Nkx))
    for j, xj in enumerate(solver.x):
        W0[j, :] = np.exp(-2.0 * xj**2 / w0**2 - 0.5 * (solver.kx * w0)**2) / np.pi

    # 1. Paraxial Fresnel BPM
    solver.compile_paraxial_kinetic(dz)
    W_parax = np.copy(W0)
    for _ in range(n_steps):
        W_parax = solver.step(W_parax)

    # 2. Exact Non-Paraxial CAW
    solver.compile_free_space_helmholtz(dz)
    W_caw = np.copy(W0)
    for _ in range(n_steps):
        W_caw = solver.step(W_caw)

    x_display_lim = 12.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout='constrained')

    # Panel (a) Main
    axes[0].pcolormesh(solver.x, solver.kx / k0, W0.T, cmap='inferno', shading='auto', rasterized=True)
    axes[0].set_title(r"(a) Initial Beam Waist ($w_0 = 0.5\lambda$)")
    axes[0].set_xlabel(r"Transverse Position $x$ [$\mu\mathrm{m}$]")
    axes[0].set_ylabel(r"Transverse Wavevector $k_x / k_0$")
    axes[0].set_xlim(-x_display_lim, x_display_lim)
    axes[0].set_ylim(-1.8, 1.8)

    # Panel (a) Zoomed Inset Axis
    axins = inset_axes(axes[0], width="44%", height="44%", loc="upper right", borderpad=1.0)
    axins.pcolormesh(solver.x, solver.kx / k0, W0.T, cmap='inferno', shading='auto', rasterized=True)
    axins.set_xlim(-1.2, 1.2)
    axins.set_ylim(-1.4, 1.4)
    axins.set_title(r"Zoom ($|x| \leq 1.2\,\mu\mathrm{m}$)", fontsize=9, color='cyan', pad=3)
    axins.tick_params(axis='both', colors='white', labelsize=8)
    for spine in axins.spines.values():
        spine.set_edgecolor('cyan')
        spine.set_linewidth(1.2)
    mark_inset(axes[0], axins, loc1=2, loc2=4, fc="none", ec="cyan", ls="--", lw=0.9)

    # Panel (b) Paraxial BPM
    axes[1].pcolormesh(solver.x, solver.kx / k0, W_parax.T, cmap='inferno', shading='auto', rasterized=True)
    axes[1].set_title(r"(b) Paraxial BPM (Underestimates Spread)")
    axes[1].set_xlabel(r"Transverse Position $x$ [$\mu\mathrm{m}$]")
    axes[1].set_xlim(-x_display_lim, x_display_lim)
    axes[1].set_ylim(-1.8, 1.8)

    # Panel (c) Exact CAW
    im2 = axes[2].pcolormesh(solver.x, solver.kx / k0, W_caw.T, cmap='inferno', shading='auto', rasterized=True)
    axes[2].set_title(r"(c) Exact Non-Paraxial CAW ($z = 12\lambda$)")
    axes[2].set_xlabel(r"Transverse Position $x$ [$\mu\mathrm{m}$]")
    axes[2].set_xlim(-x_display_lim, x_display_lim)
    axes[2].set_ylim(-1.8, 1.8)

    fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.85, label=r"Wigner Density $W(x, k_x)$")
    plt.savefig("fig1_high_na_diffraction.png", dpi=300)
    plt.close()
    print("  -> Saved enhanced 'fig1_high_na_diffraction.png'")

    # =========================================================================
    # FIG 2: Aberrated Lens Focusing & Pearcey Caustic Catastrophe
    # =========================================================================
    print("\n[2/3] Generating Figure 2: Aberrated Lens & Pearcey Caustic...")
    focal_length = 30.0
    C4 = 8.0e-5

    w_beam = 8.0
    W_incident = np.zeros((Nx, Nkx))
    for j, xj in enumerate(solver.x):
        W_incident[j, :] = np.exp(-2.0 * xj**2 / w_beam**2 - 0.5 * (solver.kx * w_beam)**2) / np.pi

    # Lens phase screen applied ONCE at z=0
    def lens_phase(x):
        return - (x**2 / (2.0 * focal_length) - C4 * x**4)

    W_focused = solver.apply_thin_phase_screen(W_incident, lens_phase)

    # Propagate through free space from z=0 to focal plane z=f
    solver.compile_free_space_helmholtz(dz)
    solver.Phi_V_half = None
    focal_steps = int(focal_length / dz)
    for _ in range(focal_steps):
        W_focused = solver.step(W_focused)

    intensity_caustic = np.sum(W_focused, axis=1) * solver.dkx
    x_caustic = np.sqrt(1.0 / (12.0 * focal_length * C4))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    mesh = ax1.pcolormesh(solver.x, solver.kx / k0, W_focused.T, cmap='coolwarm', shading='auto', vmin=-0.1, vmax=0.3)
    ax1.set_title(r"(a) Phase-Space Fold & Wigner Interference")
    ax1.set_xlabel(r"Transverse Position $x$ [$\mu\mathrm{m}$]")
    ax1.set_ylabel(r"$k_x / k_0$")
    ax1.set_xlim(-15, 15)
    ax1.set_ylim(-1.5, 1.5)
    fig.colorbar(mesh, ax=ax1, label="Wigner Density")

    ax2.plot(solver.x, intensity_caustic, color='crimson', lw=2.2, label="Exact CAW (Pearcey Wave)")
    ax2.axvline(-x_caustic, color='black', ls='--', lw=1.8, label=r"Ray Caustics ($x_c = \pm 5.89\,\mu\mathrm{m}$)")
    ax2.axvline(+x_caustic, color='black', ls='--', lw=1.8)
    ax2.set_title(r"(b) Focal Intensity $I(x)$ (Caustic Regularized)")
    ax2.set_xlabel(r"Transverse Position $x$ [$\mu\mathrm{m}$]")
    ax2.set_ylabel(r"Intensity $I(x)$ [arb. units]")
    ax2.set_xlim(-15, 15)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.savefig("fig2_aberrated_caustic.png", dpi=300)
    plt.close()
    print("  -> Saved 'fig2_aberrated_caustic.png'")

    # =========================================================================
    # FIG 3: Photonic Lattice Bloch Oscillations
    # =========================================================================
    print("\n[3/3] Generating Figure 3: Photonic Bloch Oscillations...")
    pitch_d = 4.0
    coupling_C = 0.08
    tilt_alpha = 0.0025

    def linear_index(x):
        return tilt_alpha * x

    solver_b = OpticalCAWSolver(Nx, Nkx, Lx=60.0, kx_max=np.pi / pitch_d * 3.0, wavelength=wavelength)
    solver_b.compile_photonic_lattice(dz, coupling_C, pitch_d)
    solver_b.compile_refractive_medium(dz, linear_index)

    W_b = np.zeros((Nx, Nkx))
    w_wg = 5.0
    for j, xj in enumerate(solver_b.x):
        W_b[j, :] = np.exp(-2.0 * xj**2 / w_wg**2 - 0.5 * (solver_b.kx * w_wg)**2) / np.pi

    z_period = (2.0 * np.pi) / (solver_b.k0 * tilt_alpha * pitch_d)
    bloch_steps = int(2.0 * z_period / dz)
    traj_x = []
    z_axis = np.linspace(0, 2.0 * z_period, bloch_steps)

    for _ in range(bloch_steps):
        W_b = solver_b.step(W_b)
        tot = np.sum(W_b)
        traj_x.append(np.sum(solver_b.X * W_b) / tot if tot > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
    ax.plot(z_axis, traj_x, color='teal', lw=2.4, label=r"Optical Bloch Orbit $\langle x(z) \rangle$")
    ax.set_title(r"Photonic Lattice Bloch Oscillations (Clean Periodic Return)")
    ax.set_xlabel(r"Propagation Distance $z$ [$\mu\mathrm{m}$]")
    ax.set_ylabel(r"Transverse Beam Center $\langle x \rangle$ [$\mu\mathrm{m}$]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.savefig("fig3_photonic_bloch.png", dpi=300)
    plt.close()
    print("  -> Saved 'fig3_photonic_bloch.png'")

    print("\n" + "=" * 75)
    print("ALL 3 PUBLICATION FIGURES GENERATED WITH ZERO ERRORS!")
    print("=" * 75)


if __name__ == "__main__":
    run()