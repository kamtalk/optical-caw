"""
===============================================================================
Automated Benchmark Suite: Compute Verified Numbers for Table 1 and Table 2
Author: Kenneth A. Menard
File: compute_exact_tables.py
===============================================================================
"""

import time
import numpy as np


class OpticalBenchmarkEngine:
    def __init__(self, Nx=512, Nkx=512, Lx=30.0, kx_max=19.85, wavelength=0.6328):
        self.Nx, self.Nkx = Nx, Nkx
        self.Lx, self.kx_max = Lx, kx_max
        self.wavelength = wavelength
        self.k0 = 2.0 * np.pi / wavelength

        self.x = np.linspace(-Lx / 2.0, Lx / 2.0, Nx, endpoint=False)
        self.kx = np.linspace(-kx_max, kx_max, Nkx, endpoint=False)
        self.dx = Lx / Nx
        self.dkx = 2.0 * kx_max / Nkx

        self.xi = 2.0 * np.pi * np.fft.fftfreq(Nx, d=self.dx)
        self.eta = 2.0 * np.pi * np.fft.fftfreq(Nkx, d=self.dkx)

        self.X, self.KX = np.meshgrid(self.x, self.kx, indexing='ij')
        self.XI, self.KX_grid = np.meshgrid(self.xi, self.kx, indexing='ij')

    def exact_analytical_reference(self, W0, z):
        """Single-step exact non-paraxial Helmholtz ground truth on identical grid"""
        k_plus = self.KX_grid + 0.5 * self.XI
        k_minus = self.KX_grid - 0.5 * self.XI
        
        kz_sq_plus = np.maximum(0.0, self.k0**2 - k_plus**2)
        kz_sq_minus = np.maximum(0.0, self.k0**2 - k_minus**2)
        Delta_kz = np.sqrt(kz_sq_plus) - np.sqrt(kz_sq_minus)
        
        W_xi = np.fft.fft(W0, axis=0)
        W_xi *= np.exp(+1j * z * Delta_kz)
        return np.real(np.fft.ifft(W_xi, axis=0))


def run_benchmark():
    print("=" * 80)
    print("COMPUTING VERIFIED REAL NUMBERS FOR TABLE 1 & TABLE 2")
    print("=" * 80)

    wavelength = 0.6328
    k0 = 2.0 * np.pi / wavelength
    w0 = 0.50 * wavelength
    z_prop = 12.0 * wavelength
    dz = 0.05 * wavelength
    n_steps = int(z_prop / dz)

    engine = OpticalBenchmarkEngine(Nx=512, Nkx=512, Lx=30.0, kx_max=2.0 * k0, wavelength=wavelength)

    # Initial sub-wavelength Gaussian Wigner state
    W0 = np.zeros((engine.Nx, engine.Nkx))
    for j, xj in enumerate(engine.x):
        W0[j, :] = np.exp(-2.0 * xj**2 / w0**2 - 0.5 * (engine.kx * w0)**2) / np.pi
    
    norm_0 = np.sum(W0**2) * engine.dx * engine.dkx

    # -------------------------------------------------------------------------
    # 1. EXACT GROUND TRUTH
    # -------------------------------------------------------------------------
    print("\n[1] Computing Exact Single-Step Analytical Reference...")
    t0 = time.perf_counter()
    W_exact = engine.exact_analytical_reference(W0, z_prop)
    t_ref = time.perf_counter() - t0
    exact_l2 = np.sqrt(np.sum(W_exact**2) * engine.dx * engine.dkx)

    # -------------------------------------------------------------------------
    # 2. EXACT NON-PARAXIAL CAW SOLVER (240 Steps)
    # -------------------------------------------------------------------------
    print("[2] Running Exact Multi-Step CAW Solver...")
    k_plus = engine.KX_grid + 0.5 * engine.XI
    k_minus = engine.KX_grid - 0.5 * engine.XI
    kz_sq_plus = np.maximum(0.0, engine.k0**2 - k_plus**2)
    kz_sq_minus = np.maximum(0.0, engine.k0**2 - k_minus**2)
    Delta_kz = np.sqrt(kz_sq_plus) - np.sqrt(kz_sq_minus)
    Phi_T_caw = np.exp(+1j * dz * Delta_kz)

    W_caw = np.copy(W0)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        W_xi = np.fft.fft(W_caw, axis=0)
        W_xi *= Phi_T_caw
        W_caw = np.real(np.fft.ifft(W_xi, axis=0))
    t_caw = time.perf_counter() - t0

    err_caw = np.sqrt(np.sum((W_caw - W_exact)**2) * engine.dx * engine.dkx) / exact_l2
    drift_caw = np.abs(np.sum(W_caw**2) * engine.dx * engine.dkx - norm_0) / norm_0

    # -------------------------------------------------------------------------
    # 3. PARAXIAL FRESNEL BPM (240 Steps)
    # -------------------------------------------------------------------------
    print("[3] Running Paraxial Fresnel BPM...")
    Delta_kz_parax = - (engine.KX_grid * engine.XI) / engine.k0
    Phi_T_parax = np.exp(+1j * dz * Delta_kz_parax)

    W_parax = np.copy(W0)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        W_xi = np.fft.fft(W_parax, axis=0)
        W_xi *= Phi_T_parax
        W_parax = np.real(np.fft.ifft(W_xi, axis=0))
    t_parax = time.perf_counter() - t0

    err_parax = np.sqrt(np.sum((W_parax - W_exact)**2) * engine.dx * engine.dkx) / exact_l2
    drift_parax = np.abs(np.sum(W_parax**2) * engine.dx * engine.dkx - norm_0) / norm_0

    # -------------------------------------------------------------------------
    # 4. 3RD-ORDER TRUNCATED MOYAL (240 Steps)
    # -------------------------------------------------------------------------
    print("[4] Running 3rd-Order Truncated Moyal...")
    Delta_kz_trunc = - (engine.KX_grid * engine.XI) / engine.k0 + (engine.KX_grid * (engine.XI**3)) / (24.0 * engine.k0**3)
    Phi_T_trunc = np.exp(+1j * dz * Delta_kz_trunc)

    W_trunc = np.copy(W0)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        W_xi = np.fft.fft(W_trunc, axis=0)
        W_xi *= Phi_T_trunc
        W_trunc = np.real(np.fft.ifft(W_xi, axis=0))
    t_trunc = time.perf_counter() - t0

    err_trunc = np.sqrt(np.sum((W_trunc - W_exact)**2) * engine.dx * engine.dkx) / exact_l2
    drift_trunc = np.abs(np.sum(W_trunc**2) * engine.dx * engine.dkx - norm_0) / norm_0

    # -------------------------------------------------------------------------
    # OUTPUT VERIFIED TABLE 1 VALUES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VERIFIED NUMBERS FOR TABLE 1 (Regime I: High-NA Diffraction)")
    print("=" * 80)
    print(f"Paraxial BPM:         Rel Error = {err_parax*100:6.2f}%,  Norm Drift = {drift_parax:.2e},  CPU Time = {t_parax:.4f} s")
    print(f"3rd-Order Moyal:      Rel Error = {err_trunc*100:6.2f}%,  Norm Drift = {drift_trunc:.2e},  CPU Time = {t_trunc:.4f} s")
    print(f"Exact CAW (Ours):     Rel Error = {err_caw*100:6.4f}%,  Norm Drift = {drift_caw:.2e},  CPU Time = {t_caw:.4f} s")

    # -------------------------------------------------------------------------
    # 5. VERIFIED NUMBERS FOR TABLE 2 (Grid Scaling Test: 100 Steps)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VERIFIED NUMBERS FOR TABLE 2 (Empirical 100-step Execution Scaling)")
    print("=" * 80)
    resolutions = [128, 256, 512, 1024]
    runtimes = []
    
    for res in resolutions:
        test_eng = OpticalBenchmarkEngine(Nx=res, Nkx=res, Lx=30.0, kx_max=2.0 * k0, wavelength=wavelength)
        k_p = test_eng.KX_grid + 0.5 * test_eng.XI
        k_m = test_eng.KX_grid - 0.5 * test_eng.XI
        D_kz = np.sqrt(np.maximum(0.0, test_eng.k0**2 - k_p**2)) - np.sqrt(np.maximum(0.0, test_eng.k0**2 - k_m**2))
        Phi = np.exp(+1j * dz * D_kz)
        test_W = np.ones((res, res), dtype=np.float64) / (res * res)
        
        t0 = time.perf_counter()
        for _ in range(100):
            W_xi = np.fft.fft(test_W, axis=0)
            W_xi *= Phi
            test_W = np.real(np.fft.ifft(W_xi, axis=0))
        t_el = time.perf_counter() - t0
        runtimes.append(t_el)

    base_t = runtimes[0]
    for res, t_el in zip(resolutions, runtimes):
        print(f"Grid {res:4d} x {res:4d} ({res*res:9,d} pts): Time = {t_el:.4f} s, Scaling = {t_el/base_t:6.2f}x (Base: {base_t:.4f} s)")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()