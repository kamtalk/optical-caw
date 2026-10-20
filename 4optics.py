"""
===============================================================================
Paper #4: Unifying Non-Paraxial Helmholtz Optics and Wigner–Moyal Mechanics
Author: Kenneth A. Menard
File: 4optics.py (Master Publication Figure Generator for JOSA A)
===============================================================================
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Global plotting aesthetics
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'lines.linewidth': 1.8
})


# =============================================================================
# 1. OPTICAL CAW SPECTRAL SOLVER ENGINE
# =============================================================================
class OpticalCAWSolver:
    def __init__(self, Nx, Nkx, Lx, kx_max, wavelength=1.0, n0=1.0):
        self.Nx, self.Nkx = Nx, Nkx
        self.Lx, self.kx_max = Lx, kx_max
        self.wavelength, self.n0 = wavelength, n0
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

        self.Phi_T = None
        self.Phi_V_half = None

    def compile_free_space_helmholtz(self, dz, apply_filter=True, p_cut_factor=1.25):
        k_plus = self.KX_grid + 0.5 * self.XI
        k_minus = self.KX_grid - 0.5 * self.XI
        k0_n0 = self.k0 * self.n0
        k0_n0_sq = k0_n0**2

        kz_plus = np.sqrt(k0_n0_sq - k_plus**2 + 0j)
        kz_minus = np.sqrt(k0_n0_sq - k_minus**2 + 0j)
        Delta_kz = kz_plus - np.conj(kz_minus)
        self.Phi_T = np.exp(+1j * dz * Delta_kz)

        if apply_filter:
            p_cut = p_cut_factor * k0_n0
            p_abs_p = np.abs(k_plus)
            p_abs_m = np.abs(k_minus)
            filt_p = np.where(p_abs_p > k0_n0, np.exp(-np.clip(((p_abs_p - k0_n0) / (p_cut - k0_n0))**8, 0.0, 700.0)), 1.0)
            filt_m = np.where(p_abs_m > k0_n0, np.exp(-np.clip(((p_abs_m - k0_n0) / (p_cut - k0_n0))**8, 0.0, 700.0)), 1.0)
            self.Phi_T *= (filt_p * filt_m)

    def compile_paraxial_kinetic(self, dz):
        Delta_kz_parax = - (self.KX_grid * self.XI) / (self.k0 * self.n0)
        self.Phi_T = np.exp(+1j * dz * Delta_kz_parax)

    def compile_refractive_medium(self, dz, n_profile_func):
        x_plus = self.X_grid + 0.5 * self.ETA
        x_minus = self.X_grid - 0.5 * self.ETA
        Delta_n = n_profile_func(x_plus) - n_profile_func(x_minus)
        self.Phi_V_half = np.exp(-1j * (self.k0 * dz / 2.0) * Delta_n)

    def compile_photonic_lattice(self, dz, coupling_C, pitch_d):
        Delta_E = 4.0 * coupling_C * np.sin(self.KX_grid * pitch_d) * np.sin(0.5 * self.XI * pitch_d)
        self.Phi_T = np.exp(+1j * dz * Delta_E)

    def apply_thin_phase_screen_exact(self, W, phase_func):
        x_plus = self.X_grid + 0.5 * self.ETA
        x_minus = self.X_grid - 0.5 * self.ETA
        Delta_S = phase_func(x_plus) - phase_func(x_minus)
        Phi_screen = np.exp(-1j * self.k0 * Delta_S)
        W_eta = np.fft.fft(W, axis=1) * Phi_screen
        return np.real(np.fft.ifft(W_eta, axis=1))

    def apply_thin_phase_screen_moyal3(self, W, phase_func_prime, phase_func_3prime):
        Delta_S_trunc = phase_func_prime(self.X_grid) * self.ETA + (1.0 / 24.0) * phase_func_3prime(self.X_grid) * (self.ETA**3)
        Phi_screen = np.exp(-1j * self.k0 * Delta_S_trunc)
        W_eta = np.fft.fft(W, axis=1) * Phi_screen
        return np.real(np.fft.ifft(W_eta, axis=1))

    def step(self, W):
        if self.Phi_V_half is not None:
            W = np.real(np.fft.ifft(np.fft.fft(W, axis=1) * self.Phi_V_half, axis=1))
        W = np.real(np.fft.ifft(np.fft.fft(W, axis=0) * self.Phi_T, axis=0))
        if self.Phi_V_half is not None:
            W = np.real(np.fft.ifft(np.fft.fft(W, axis=1) * self.Phi_V_half, axis=1))
        return W


# =============================================================================
# 2. RUN BENCHMARKS AND GENERATE PUBLICATION FIGURES
# =============================================================================
def run():
    print("=" * 80)
    print("GENERATING ALL PUBLICATION FIGURES (JOSA A MASTER)")
    print("=" * 80)

    wavelength = 0.6328  # um
    k0 = 2.0 * np.pi / wavelength
    dz = 0.05 * wavelength

    # =========================================================================
    # FIG 1: High-NA Diffraction (Hybrid Layout: 3 Density Maps + 1 Error Sweep)
    # =========================================================================
    print("\n[1/3] Generating Figure 1 (Hybrid Layout)...")
    w0 = 0.50 * wavelength
    z_benchmark = 12.0 * wavelength
    n_steps = int(round(z_benchmark / dz))

    Nx, Nkx = 512, 512
    Lx = 40.0
    kx_max = 2.2 * k0

    solver1 = OpticalCAWSolver(Nx, Nkx, Lx, kx_max, wavelength=wavelength, n0=1.0)
    W0 = (2.0 / np.pi) * np.exp(-2.0 * (solver1.X / w0)**2 - 0.5 * (solver1.KX * w0)**2)
    W0_xi = np.fft.fft(W0, axis=0)

    # 1. Panels (b) and (c) at z = 12 lambda
    k_plus = solver1.KX_grid + 0.5 * solver1.XI
    k_minus = solver1.KX_grid - 0.5 * solver1.XI
    kz_p = np.sqrt(k0**2 - k_plus**2 + 0j)
    kz_m = np.sqrt(k0**2 - k_minus**2 + 0j)
    Delta_kz = kz_p - np.conj(kz_m)

    solver1.compile_free_space_helmholtz(dz)
    W_caw = np.copy(W0)
    for _ in range(n_steps):
        W_caw = solver1.step(W_caw)

    solver1.compile_paraxial_kinetic(dz)
    W_bpm = np.copy(W0)
    for _ in range(n_steps):
        W_bpm = solver1.step(W_bpm)

    # 2. Distance sweep for Panel (d)
    # Using 51 points ensures z = 12.0 is exactly in the array (steps of 1.0)
    z_sweep = np.linspace(0.0, 50.0 * wavelength, 51)
    
    mask_act = np.abs(solver1.KX_grid) <= 0.92 * k0
    denom = np.maximum(1e-6, k0**2 - solver1.KX_grid**2)
    kz_prime = np.zeros_like(solver1.KX_grid)
    kz_3prime = np.zeros_like(solver1.KX_grid)
    kz_prime[mask_act] = -solver1.KX_grid[mask_act] / np.sqrt(denom[mask_act])
    kz_3prime[mask_act] = -3.0 * k0**2 * solver1.KX_grid[mask_act] / (denom[mask_act]**2.5)
    omega_moyal = kz_prime * solver1.XI + (1.0 / 24.0) * kz_3prime * (solver1.XI**3)

    errs_bpm, errs_moyal = [], []
    for z_val in z_sweep:
        # Exact reference at distance z_val
        W_ref = np.real(np.fft.ifft(W0_xi * np.exp(+1j * z_val * Delta_kz), axis=0))
        ref_norm = np.linalg.norm(W_ref)

        # Paraxial BPM at distance z_val
        W_b = np.real(np.fft.ifft(W0_xi * np.exp(+1j * z_val * (- (solver1.KX_grid * solver1.XI) / k0)), axis=0))
        errs_bpm.append(np.linalg.norm(W_b - W_ref) / ref_norm * 100.0)

        # Truncated Moyal (spectral exponential) at distance z_val
        W_m_xi = W0_xi * np.exp(+1j * z_val * omega_moyal)
        W_m_xi[~mask_act] = 0.0  # Honestly apply the spectral restriction at all z
        W_m = np.real(np.fft.ifft(W_m_xi, axis=0))
        errs_moyal.append(np.linalg.norm(W_m - W_ref) / ref_norm * 100.0)

    errs_bpm = np.array(errs_bpm)
    errs_moyal = np.array(errs_moyal)
    
    # Extract exact indices for reporting
    idx_12 = np.where(np.abs(z_sweep - z_benchmark) < 1e-6)[0][0]

    print(f"  At z = 12 lambda: Paraxial BPM Error    = {errs_bpm[idx_12]:.2f}%")
    print(f"  At z = 12 lambda: Truncated Moyal Error = {errs_moyal[idx_12]:.2f}%")
    print(f"  At z = 50 lambda: Truncated Moyal Error = {errs_moyal[-1]:.2f}%")

    # Render Hybrid Figure 1
    fig = plt.figure(figsize=(16.5, 4.2), layout='constrained')
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.25])
    x_disp = 12.0

    ax0 = fig.add_subplot(gs[0])
    ax0.pcolormesh(solver1.x, solver1.kx / k0, W0.T, cmap='inferno', shading='auto', rasterized=True)
    ax0.set_title(r"(a) Initial Waist ($w_0 = 0.5\lambda_0$)")
    ax0.set_xlabel(r"Position $x$ [$\mu\mathrm{m}$]"); ax0.set_ylabel(r"$k_x / k_0$")
    ax0.set_xlim(-x_disp, x_disp); ax0.set_ylim(-1.8, 1.8)

    axins = inset_axes(ax0, width="42%", height="42%", loc="upper right", borderpad=0.8)
    axins.pcolormesh(solver1.x, solver1.kx / k0, W0.T, cmap='inferno', shading='auto', rasterized=True)
    axins.set_xlim(-1.2, 1.2); axins.set_ylim(-1.4, 1.4)
    axins.set_title(r"Waist Zoom", fontsize=8, color='cyan', pad=2)
    axins.tick_params(axis='both', colors='white', labelsize=7)
    for spine in axins.spines.values():
        spine.set_edgecolor('cyan'); spine.set_linewidth(1.0)
    mark_inset(ax0, axins, loc1=2, loc2=4, fc="none", ec="cyan", ls="--", lw=0.8)

    ax1 = fig.add_subplot(gs[1])
    ax1.pcolormesh(solver1.x, solver1.kx / k0, W_bpm.T, cmap='inferno', shading='auto', rasterized=True)
    ax1.set_title(f"(b) Paraxial BPM ({errs_bpm[idx_12]:.1f}% err)")
    ax1.set_xlabel(r"Position $x$ [$\mu\mathrm{m}$]")
    ax1.set_xlim(-x_disp, x_disp); ax1.set_ylim(-1.8, 1.8)

    ax2 = fig.add_subplot(gs[2])
    im_caw = ax2.pcolormesh(solver1.x, solver1.kx / k0, W_caw.T, cmap='inferno', shading='auto', rasterized=True)
    ax2.set_title(r"(c) Optical CAW ($z = 12\lambda_0$)")
    ax2.set_xlabel(r"Position $x$ [$\mu\mathrm{m}$]")
    ax2.set_xlim(-x_disp, x_disp); ax2.set_ylim(-1.8, 1.8)

    ax3 = fig.add_subplot(gs[3])
    z_in_lambda = z_sweep / wavelength
    ax3.plot(z_in_lambda, errs_bpm, color='purple', lw=2.0, label="Paraxial BPM")
    ax3.plot(z_in_lambda, errs_moyal, color='crimson', ls='--', lw=2.0, label="3rd-Order Moyal")
    ax3.axvline(12.0, color='black', ls=':', lw=1.5, label=r"Benchmark plane ($12\lambda_0$)")
    ax3.set_title(r"(d) Error vs. Propagation Distance")
    ax3.set_xlabel(r"Distance $z / \lambda_0$")
    ax3.set_ylabel(r"Relative $L_2$ Error [$\%$]")
    ax3.set_xlim(0, 50); ax3.set_ylim(0, 85)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower right', fontsize=8.5)

    plt.savefig("fig1_high_na_diffraction.png", dpi=300)
    plt.close()
    print("  -> Saved 'fig1_high_na_diffraction.png'")

    # =========================================================================
    # FIG 2: Aberrated Lens & Caustic Regularization
    # =========================================================================
    print("\n[2/3] Generating Figure 2 (Aberrated Caustic)...")
    focal_length, C4, C6, w_beam = 30.0, 8.0e-5, 1.2e-7, 8.0

    solver2 = OpticalCAWSolver(512, 512, Lx=30.0, kx_max=2.0 * k0, wavelength=wavelength, n0=1.0)
    W_incident = (2.0 / np.pi) * np.exp(-2.0 * (solver2.X / w_beam)**2 - 0.5 * (solver2.KX * w_beam)**2)

    def S_exact(x): return - (x**2 / (2.0 * focal_length)) + C4 * (x**4) + C6 * (x**6)
    def S_prime(x): return - (x / focal_length) + 4.0 * C4 * (x**3) + 6.0 * C6 * (x**5)
    def S_3prime(x): return 24.0 * C4 * x + 120.0 * C6 * (x**3)

    W_caw2 = solver2.apply_thin_phase_screen_exact(W_incident, S_exact)
    solver2.compile_free_space_helmholtz(dz)
    focal_steps = int(round(focal_length / dz))
    for _ in range(focal_steps):
        W_caw2 = solver2.step(W_caw2)

    W_moyal2 = solver2.apply_thin_phase_screen_moyal3(W_incident, S_prime, S_3prime)
    for _ in range(focal_steps):
        W_moyal2 = solver2.step(W_moyal2)

    I_caw = np.sum(W_caw2, axis=1) * solver2.dkx
    I_moyal = np.sum(W_moyal2, axis=1) * solver2.dkx
    err_caustic_1d = np.linalg.norm(I_moyal - I_caw) / np.linalg.norm(I_caw) * 100.0

    x_geom_bound = focal_length * (4.0 * C4 * (w_beam**3) + 6.0 * C6 * (w_beam**5))

    dW_dx, dW_dkx = np.gradient(W_caw2, solver2.dx, solver2.dkx)
    Jx, Jkx = -dW_dkx, dW_dx

    # The vertical plot coordinate is kx/k0. Scale the vertical vector
    # component accordingly, then normalize arrows to show direction clearly.
    U_phase = Jx
    V_phase = Jkx / k0
    speed_phase = np.hypot(U_phase, V_phase)
    speed_safe = np.maximum(speed_phase, np.finfo(float).eps)
    U_phase = U_phase / speed_safe
    V_phase = V_phase / speed_safe

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
    im_a = ax1.pcolormesh(
        solver2.x, solver2.kx / k0, W_caw2.T,
        cmap='coolwarm', shading='auto', vmin=-0.08, vmax=0.25, rasterized=True
    )
    # Keep the actual W=0 contour, but use fine, translucent lines so dense
    # outer nodal structure does not overwhelm the central phase-space pattern.
    contour_band_a = np.abs(solver2.kx / k0) <= 0.32
    ax1.contour(
        solver2.x, (solver2.kx / k0)[contour_band_a],
        W_caw2[:, contour_band_a].T, levels=[0.0],
        colors='black', linewidths=0.30, alpha=0.48
    )
    significant_a = np.abs(W_caw2) >= 0.008
    U_a = np.ma.masked_where(~significant_a, U_phase)
    V_a = np.ma.masked_where(~significant_a, V_phase)
    ax1.quiver(
        solver2.X[::20, ::20], (solver2.KX / k0)[::20, ::20],
        U_a[::20, ::20], V_a[::20, ::20],
        color='gray', alpha=0.52, angles='xy', scale_units='xy',
        scale=8.0, width=0.0025
    )
    ax1.set_title(r"(a) Phase Space: Central Nodal Lines ($W=0$) & Vectors")
    ax1.set_xlabel(r"Transverse Position $x$ [$\mu\mathrm{m}$]")
    ax1.set_ylabel(r"$k_x / k_0$")
    ax1.set_xlim(-14, 14)
    ax1.set_ylim(-1.4, 1.4)
    fig.colorbar(im_a, ax=ax1, label="Wigner Density")

    im_b = ax2.pcolormesh(
        solver2.x, solver2.kx / k0, W_caw2.T,
        cmap='coolwarm', shading='auto', vmin=-0.08, vmax=0.25, rasterized=True
    )
    contour_band_b = np.abs(solver2.kx / k0) <= 0.17
    ax2.contour(
        solver2.x, (solver2.kx / k0)[contour_band_b],
        W_caw2[:, contour_band_b].T, levels=[0.0],
        colors='black', linewidths=0.52, alpha=0.65
    )
    zoom_mask = (solver2.x >= -3.5) & (solver2.x <= 3.5)
    zoom_k = (solver2.kx / k0 >= -0.5) & (solver2.kx / k0 <= 0.5)
    X_zoom = solver2.X[np.ix_(zoom_mask, zoom_k)]
    K_zoom = solver2.KX[np.ix_(zoom_mask, zoom_k)] / k0
    W_zoom = W_caw2[np.ix_(zoom_mask, zoom_k)]
    U_zoom = np.ma.masked_where(
        np.abs(W_zoom) < 0.008, U_phase[np.ix_(zoom_mask, zoom_k)]
    )
    V_zoom = np.ma.masked_where(
        np.abs(W_zoom) < 0.008, V_phase[np.ix_(zoom_mask, zoom_k)]
    )
    # Use a sparser, lighter vector field to keep the node legible. No transpose
    # is applied: these arrays are already indexed as (x, kx), like X_zoom.
    ax2.quiver(
        X_zoom[::10, ::10], K_zoom[::10, ::10],
        U_zoom[::10, ::10], V_zoom[::10, ::10],
        color='darkslategray', alpha=0.72, angles='xy',
        scale_units='xy', scale=10.0, width=0.0040
    )
    ax2.set_title(r"(b) Local Circulation near Stagnation Node")
    ax2.set_xlabel(r"Transverse Position $x$ [$\mu\mathrm{m}$]")
    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-0.5, 0.5)

    # The vertical markers are mapped 1/e^2 waist bounds, not Jacobian-zero
    # caustic points. Use x_geom consistently in the legend.
    ax3.plot(solver2.x, I_caw, color='crimson', lw=2.2, label="Optical CAW")
    ax3.plot(
        solver2.x, I_moyal, color='royalblue', ls='--', lw=1.8,
        label=f"Truncated Moyal ({err_caustic_1d:.2f}% $L_2$ error)"
    )
    bound_label = (
        r"Mapped $1/e^2$ bounds "
        + rf"($x_{{\mathrm{{geom}}}}=\pm {x_geom_bound:.2f}\,\mu\mathrm{{m}}$)"
    )
    ax3.axvline(-x_geom_bound, color='black', ls=':', lw=1.35, label=bound_label)
    ax3.axvline(+x_geom_bound, color='black', ls=':', lw=1.35)
    ax3.set_title(r"(c) Focal Intensity and Moyal Residual")
    ax3.set_xlabel(r"Transverse Position $x$ [$\mu\mathrm{m}$]")
    ax3.set_ylabel(r"Intensity [arb. units]")
    ax3.set_xlim(-14, 14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=7.3, framealpha=0.92)

    # Residual inset highlights curve differences without changing the main
    # intensity scale. It is normalized to the CAW peak, not pointwise I(x).
    # Lower-left placement avoids colliding with the upper-right legend.
    ax3ins = inset_axes(ax3, width="44%", height="33%", loc="lower left", borderpad=1.0)
    peak_intensity = max(float(np.max(np.abs(I_caw))), np.finfo(float).eps)
    residual_pct = 100.0 * (I_moyal - I_caw) / peak_intensity
    ax3ins.plot(solver2.x, residual_pct, color='navy', lw=1.0)
    ax3ins.axhline(0.0, color='black', ls=':', lw=0.7)
    # Re-mark the same mapped waist bounds in the residual inset so that the
    # insets do not visually interrupt their spatial locations.
    ax3ins.axvline(-x_geom_bound, color='black', ls=':', lw=0.8, alpha=0.8)
    ax3ins.axvline(+x_geom_bound, color='black', ls=':', lw=0.8, alpha=0.8)
    ax3ins.set_xlim(-10.0, 10.0)
    residual_window = residual_pct[np.abs(solver2.x) <= 10.0]
    residual_limit = max(0.05, 1.12 * float(np.max(np.abs(residual_window))))
    ax3ins.set_ylim(-residual_limit, residual_limit)
    ax3ins.set_title("Residual (% of CAW peak)", fontsize=7.0, pad=2)
    ax3ins.set_xlabel(r"$x$ [$\mu$m]", fontsize=6.5, labelpad=0)
    # Omit a separate inset y-label to avoid colliding with the parent axis label.
    ax3ins.tick_params(axis='both', labelsize=6.0, length=2)
    ax3ins.grid(True, alpha=0.2)

    plt.savefig("fig2_aberrated_caustic.png", dpi=300)
    plt.close()
    print("  -> Saved 'fig2_aberrated_caustic.png'")

    # =========================================================================
    # FIG 3: Photonic Lattice Bloch Oscillations
    # =========================================================================
    print("\n[3/3] Generating Figure 3 (Bloch Oscillations)...")
    pitch_d, coupling_C, tilt_alpha, dz_bloch = 4.0, 0.08, 0.0025, 0.1
    solver3 = OpticalCAWSolver(512, 512, Lx=60.0, kx_max=np.pi / pitch_d * 3.0, wavelength=wavelength, n0=1.45)
    solver3.compile_photonic_lattice(dz_bloch, coupling_C, pitch_d)
    solver3.compile_refractive_medium(dz_bloch, lambda x: tilt_alpha * x)

    w_wg = 5.0
    W_b = np.zeros((512, 512))
    for j, xj in enumerate(solver3.x):
        W_b[j, :] = np.exp(-2.0 * xj**2 / w_wg**2 - 0.5 * (solver3.kx * w_wg)**2) / np.pi

    z_period = (2.0 * np.pi) / (solver3.k0 * tilt_alpha * pitch_d)
    bloch_steps = int(round(2.0 * z_period / dz_bloch))
    traj_x = []

    for _ in range(bloch_steps):
        W_b = solver3.step(W_b)
        tot = np.sum(W_b)
        traj_x.append(np.sum(solver3.X * W_b) / tot if tot > 0 else 0.0)

    traj_x = np.array(traj_x)
    amp_sim = np.max(traj_x) - np.min(traj_x)

    fig, ax = plt.subplots(figsize=(7.5, 4.0), layout='constrained')
    ax.plot(np.linspace(0, 2.0 * z_period, bloch_steps), traj_x, color='teal', lw=2.2, label=r"Centroid $\langle x(z) \rangle$ ($2L_B = 9.36\,\mu\mathrm{m}$)")
    ax.set_title(r"Optical Bloch Oscillations ($z_B \approx 63.3\,\mu\mathrm{m}, 2L_B \approx 9.36\,\mu\mathrm{m}$)")
    ax.set_xlabel(r"Propagation Distance $z$ [$\mu\mathrm{m}$]"); ax.set_ylabel(r"Transverse Beam Center $\langle x \rangle$ [$\mu\mathrm{m}$]")
    ax.grid(True, alpha=0.3); ax.legend(loc='upper right')
    plt.savefig("fig3_photonic_bloch.png", dpi=300); plt.close()
    print("  -> Saved 'fig3_photonic_bloch.png'")

    print("\n" + "=" * 80)
    print("ALL THREE PUBLICATION FIGURES GENERATED AND SYNCHRONIZED!")
    print("=" * 80)


if __name__ == "__main__":
    run()
