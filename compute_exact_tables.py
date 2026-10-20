"""
===============================================================================
Theory-consistent benchmark suite for Optical CAW (JOSA A)
Author: Kenneth A. Menard
File: compute_exact_tables_theory.py

Purpose
-------
1. Compare the exact dual-space free-space CAW update against its same-grid
   angular-spectrum reference, plus paraxial and third-order Moyal models.
2. Report the correct axial-step consistency test for homogeneous free space.
   Since the CAW kinetic multiplier is exact, this error should be roundoff-level
   for all tested dz; it is NOT expected to show O(dz^2) convergence.
3. Compare marginal intensities with an independent continuous RS-I quadrature.
4. Measure CPU scaling separately from accuracy.

Important interpretation notes
------------------------------
- The third-order Moyal result below uses an exponential spectral propagator for
  the truncated generator. It is NOT an explicit RK4 run and cannot be used to
  claim RK4 CFL instability or an RK4 runtime.
- The subcritical |kx| <= 0.92 k0 band is applied to the Wigner-array columns for
  the truncated-Moyal comparison. The comparison reference uses the same
  projected initial array, so the printed error is conditional on that band.
- The RS-I comparison is made on each engine's own x grid and normalized over
  that finite computational window. Do not compare raw values across different
  windows without considering this normalization convention.
===============================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.special import hankel1


@dataclass
class EngineConfig:
    Nx: int = 512
    Nkx: int = 512
    Lx: float = 40.0
    kx_max: float = 21.84
    wavelength: float = 0.6328


class OpticalBenchmarkEngine:
    def __init__(
        self,
        Nx: int = 512,
        Nkx: int = 512,
        Lx: float = 40.0,
        kx_max: float = 21.84,
        wavelength: float = 0.6328,
    ) -> None:
        if Nx < 8 or Nkx < 8:
            raise ValueError("Nx and Nkx must both be at least 8.")
        if Lx <= 0 or kx_max <= 0 or wavelength <= 0:
            raise ValueError("Lx, kx_max, and wavelength must be positive.")

        self.Nx, self.Nkx = int(Nx), int(Nkx)
        self.Lx, self.kx_max = float(Lx), float(kx_max)
        self.wavelength = float(wavelength)
        self.k0 = 2.0 * np.pi / self.wavelength

        self.dx = self.Lx / self.Nx
        self.kx = -self.kx_max + np.arange(self.Nkx) * (2.0 * self.kx_max / self.Nkx)
        self.dkx = 2.0 * self.kx_max / self.Nkx
        self.x = -self.Lx / 2.0 + np.arange(self.Nx) * self.dx

        # np.fft.fftfreq order is exactly the order expected by np.fft.fft.
        self.xi = 2.0 * np.pi * np.fft.fftfreq(self.Nx, d=self.dx)
        self.eta = 2.0 * np.pi * np.fft.fftfreq(self.Nkx, d=self.dkx)
        self.X, self.KX = np.meshgrid(self.x, self.kx, indexing="ij")
        self.XI, self.KX_grid = np.meshgrid(self.xi, self.kx, indexing="ij")

    def delta_kz(self) -> np.ndarray:
        """Bilinear Helmholtz difference kz(kx+xi/2)-kz*(kx-xi/2)."""
        k_plus = self.KX_grid + 0.5 * self.XI
        k_minus = self.KX_grid - 0.5 * self.XI
        kz_plus = np.sqrt(self.k0**2 - k_plus**2 + 0j)
        kz_minus = np.sqrt(self.k0**2 - k_minus**2 + 0j)
        return kz_plus - np.conj(kz_minus)

    def compile_propagator(self, dz: float) -> np.ndarray:
        """Exact free-space Wigner multiplier for one axial step."""
        return np.exp(1j * float(dz) * self.delta_kz())

    def exact_analytical_reference_wigner(self, W0: np.ndarray, z: float) -> np.ndarray:
        """Single-step same-grid exact angular-spectrum reference at distance z."""
        W_xi = np.fft.fft(np.asarray(W0, dtype=np.float64), axis=0)
        W_xi *= np.exp(1j * float(z) * self.delta_kz())
        return np.fft.ifft(W_xi, axis=0).real

    def propagate_exact(self, W0: np.ndarray, z: float, n_steps: int) -> tuple[np.ndarray, float]:
        """Apply the exact kinetic multiplier repeatedly; return field and runtime."""
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        dz = float(z) / int(n_steps)
        phi = self.compile_propagator(dz)
        W = np.array(W0, dtype=np.float64, copy=True)
        t0 = time.perf_counter()
        for _ in range(int(n_steps)):
            W_xi = np.fft.fft(W, axis=0)
            W_xi *= phi
            W = np.fft.ifft(W_xi, axis=0).real
        return W, time.perf_counter() - t0

    def propagate_paraxial(self, W0: np.ndarray, z: float, n_steps: int) -> tuple[np.ndarray, float]:
        """Exact split-step propagation for paraxial Fresnel dispersion."""
        dz = float(z) / int(n_steps)
        # kz(kx+xi/2)-kz(kx-xi/2) for kz ~= k0-kx^2/(2k0).
        delta_parax = -(self.KX_grid * self.XI) / self.k0
        phi = np.exp(1j * dz * delta_parax)
        W = np.array(W0, dtype=np.float64, copy=True)
        t0 = time.perf_counter()
        for _ in range(int(n_steps)):
            W_xi = np.fft.fft(W, axis=0)
            W_xi *= phi
            W = np.fft.ifft(W_xi, axis=0).real
        return W, time.perf_counter() - t0

    def propagate_truncated_moyal_exponential(
        self, W0: np.ndarray, z: float, n_steps: int, kx_fraction: float = 0.92
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Exponentiate the third-order truncated Moyal generator spectrally.

        Returns (propagated projected Wigner array, projected exact reference,
        runtime). This is an exponential spectral propagator, not explicit RK4.
        """
        active_k = np.abs(self.kx) <= kx_fraction * self.k0
        W0_proj = np.asarray(W0, dtype=np.float64) * active_k[np.newaxis, :]
        denom = self.k0**2 - self.KX_grid**2
        kz_prime = np.zeros_like(self.KX_grid)
        kz_3prime = np.zeros_like(self.KX_grid)
        active_2d = np.abs(self.KX_grid) <= kx_fraction * self.k0
        denom_safe = np.maximum(denom, np.finfo(float).tiny)
        kz_prime[active_2d] = -self.KX_grid[active_2d] / np.sqrt(denom_safe[active_2d])
        kz_3prime[active_2d] = (
            -3.0 * self.k0**2 * self.KX_grid[active_2d]
            / (denom_safe[active_2d] ** 2.5)
        )

        # Fourier generator: i * (kz' xi + kz''' xi^3 / 24).
        omega = kz_prime * self.XI + kz_3prime * (self.XI**3) / 24.0
        dz = float(z) / int(n_steps)
        phi = np.exp(1j * dz * omega)
        W = np.array(W0_proj, copy=True)
        t0 = time.perf_counter()
        for _ in range(int(n_steps)):
            W_xi = np.fft.fft(W, axis=0)
            W_xi *= phi
            W = np.fft.ifft(W_xi, axis=0).real
        elapsed = time.perf_counter() - t0
        W_ref_proj = self.exact_analytical_reference_wigner(W0_proj, z)
        return W, W_ref_proj, elapsed

    def normalized_marginal_intensity(self, W: np.ndarray) -> np.ndarray:
        """Return I(x)=integral W dkx, normalized to unit sampled-window power."""
        intensity = np.sum(W, axis=1) * self.dkx
        power = np.sum(intensity) * self.dx
        if not np.isfinite(power) or abs(power) < 1e-300:
            raise FloatingPointError("Cannot normalize intensity: non-finite/zero power.")
        return intensity / power

    def independent_rayleigh_sommerfeld_intensity(
        self, w0: float, z: float, n_quad: int = 1000
    ) -> np.ndarray:
        """Continuous RS-I quadrature, sampled only at this engine's x grid."""
        x_limit = 10.0 * w0
        nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
        x_prime = x_limit * nodes
        w_prime = x_limit * weights
        E0 = np.exp(-((x_prime / w0) ** 2))
        dx_mat = self.x[:, np.newaxis] - x_prime[np.newaxis, :]
        R = np.sqrt(dx_mat**2 + float(z) ** 2)
        integrand = E0[np.newaxis, :] * hankel1(1, self.k0 * R) / R
        E_exact = (self.k0 * float(z) / (2.0j)) * np.sum(
            integrand * w_prime[np.newaxis, :], axis=1
        )
        intensity = np.abs(E_exact) ** 2
        power = np.sum(intensity) * self.dx
        if not np.isfinite(power) or power <= 0:
            raise FloatingPointError("RS-I quadrature returned non-finite/zero power.")
        return intensity / power


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(b.ravel())
    if denom == 0 or not np.isfinite(denom):
        raise ValueError("Reference norm is zero or non-finite.")
    return float(np.linalg.norm((a - b).ravel()) / denom * 100.0)


def norm_value(W: np.ndarray, engine: OpticalBenchmarkEngine) -> float:
    return float(np.sum(W) * engine.dx * engine.dkx)


def gaussian_wigner(engine: OpticalBenchmarkEngine, w0: float) -> np.ndarray:
    return (2.0 / np.pi) * np.exp(
        -2.0 * (engine.X / w0) ** 2 - 0.5 * (engine.KX * w0) ** 2
    )


def run_benchmark() -> None:
    np.seterr(over="raise", invalid="raise", divide="raise")
    wavelength = 0.6328
    k0 = 2.0 * np.pi / wavelength
    w0 = 0.50 * wavelength
    z_prop = 12.0 * wavelength
    dz_nominal = 0.05 * wavelength
    n_steps_nominal = round(z_prop / dz_nominal)
    kx_max = 2.2 * k0

    engine = OpticalBenchmarkEngine(
        Nx=512, Nkx=512, Lx=40.0, kx_max=kx_max, wavelength=wavelength
    )
    W0 = gaussian_wigner(engine, w0)
    norm0 = norm_value(W0, engine)

    print("=" * 88)
    print("THEORY-CONSISTENT OPTICAL CAW BENCHMARKS")
    print("=" * 88)
    print(f"Grid: {engine.Nx} x {engine.Nkx}; Lx={engine.Lx:g} um; "
          f"kx_max={engine.kx_max:.6g} um^-1; z={z_prop:.6g} um")
    print(f"Nominal dz={dz_nominal:.6g} um; nominal steps={n_steps_nominal}")

    # -- Table 1: baseline method comparisons on a common discrete grid --------
    print("\nTABLE 1A — Baseline high-divergence benchmark")
    W_ref = engine.exact_analytical_reference_wigner(W0, z_prop)
    W_caw, t_caw = engine.propagate_exact(W0, z_prop, n_steps_nominal)
    W_parax, t_parax = engine.propagate_paraxial(W0, z_prop, n_steps_nominal)
    W_moyal, W_moyal_ref, t_moyal = engine.propagate_truncated_moyal_exponential(
        W0, z_prop, n_steps_nominal, kx_fraction=0.92
    )

    drift_ref = abs(norm_value(W_ref, engine) - norm0) / abs(norm0)
    drift_caw = abs(norm_value(W_caw, engine) - norm0) / abs(norm0)
    drift_parax = abs(norm_value(W_parax, engine) - norm0) / abs(norm0)
    drift_moyal = abs(norm_value(W_moyal, engine) - norm_value(W_moyal_ref, engine)) / max(
        abs(norm_value(W_moyal_ref, engine)), 1e-300
    )

    print("Metric: relative L2 error of full phase-space Wigner array vs SAME-GRID reference")
    print(f"  Paraxial BPM (exact paraxial multiplier): {relative_l2(W_parax, W_ref):9.5f}%  "
          f"norm drift={drift_parax:.3e}  runtime={t_parax:.4f} s")
    print(f"  Third-order Moyal (exponential spectral, |kx|<=0.92 k0): "
          f"{relative_l2(W_moyal, W_moyal_ref):9.5f}%  "
          f"projected norm drift={drift_moyal:.3e}  runtime={t_moyal:.4f} s")
    print(f"  Optical CAW:                                {relative_l2(W_caw, W_ref):9.5e}%  "
          f"norm change={drift_caw:.3e}  runtime={t_caw:.4f} s")
    print(f"  Single-step analytic reference norm change: {drift_ref:.3e}")

    # -- RS-I marginal intensity comparison ------------------------------------
    print("\nTABLE 1B — Marginal intensity vs independent continuous RS-I")
    t0 = time.perf_counter()
    I_rs = engine.independent_rayleigh_sommerfeld_intensity(w0, z_prop, n_quad=1000)
    t_rs = time.perf_counter() - t0
    I_caw = engine.normalized_marginal_intensity(W_caw)
    I_parax = engine.normalized_marginal_intensity(W_parax)
    I_moyal = engine.normalized_marginal_intensity(W_moyal)
    print(f"  Paraxial BPM vs RS-I:                 {relative_l2(I_parax, I_rs):9.5f}%")
    print(f"  Third-order Moyal projection vs RS-I:  {relative_l2(I_moyal, I_rs):9.5f}%  "
          "(contains the stated kx-band projection)")
    print(f"  Optical CAW vs RS-I:                   {relative_l2(I_caw, I_rs):9.5f}%")
    print(f"  RS-I quadrature runtime (1000 nodes):  {t_rs:.4f} s")

    # -- Correct axial-step study ----------------------------------------------
    print("\nTABLE 2 — AXIAL-STEP CONSISTENCY IN HOMOGENEOUS FREE SPACE")
    print("Theory: CAW's kinetic multiplier is exact; error vs its same-grid analytic "
          "reference should be roundoff-level, not O(dz^2).")
    print(f"{'target dz/lambda':>15} {'steps':>8} {'actual dz (um)':>16} "
          f"{'phase-space rel L2 error (%)':>29} {'norm change':>15}")
    for dz_over_lambda in (0.20, 0.10, 0.05, 0.025):
        target_dz = dz_over_lambda * wavelength
        steps = int(round(z_prop / target_dz))
        actual_dz = z_prop / steps
        W_step, elapsed = engine.propagate_exact(W0, z_prop, steps)
        err = relative_l2(W_step, W_ref)
        drift = abs(norm_value(W_step, engine) - norm0) / abs(norm0)
        print(f"{dz_over_lambda:15.3f} {steps:8d} {actual_dz:16.8g} {err:29.8e} {drift:15.3e}")

    # -- Grid resolution convergence vs continuous reference -------------------
    print("\nTABLE 3A — GRID-RESOLUTION CONVERGENCE VS CONTINUOUS RS-I MARGINAL INTENSITY")
    print("Each resolution is evaluated against RS-I at the same x coordinates; "
          "these are intensity errors, not phase-space errors.")
    print(f"{'grid':>12} {'dx (um)':>12} {'RS-I L2 intensity error (%)':>29} {'runtime (s)':>14}")
    for res in (128, 256, 512, 1024):
        e = OpticalBenchmarkEngine(
            Nx=res, Nkx=res, Lx=40.0, kx_max=kx_max, wavelength=wavelength
        )
        W = gaussian_wigner(e, w0)
        W_out, elapsed = e.propagate_exact(W, z_prop, round(z_prop / dz_nominal))
        I_out = e.normalized_marginal_intensity(W_out)
        I_rs_grid = e.independent_rayleigh_sommerfeld_intensity(w0, z_prop, n_quad=1000)
        err = relative_l2(I_out, I_rs_grid)
        print(f"{res:6d} x {res:<5d} {e.dx:12.7f} {err:29.6f} {elapsed:14.4f}")

    # -- Computational window study, keep dx roughly constant ------------------
    print("\nTABLE 3B — COMPUTATIONAL-WINDOW STUDY VS CONTINUOUS RS-I INTENSITY")
    print("dx is held near 0.078125 um; reported intensity is normalized over each window.")
    print(f"{'Lx (um)':>10} {'Nx':>8} {'dx (um)':>12} {'RS-I L2 intensity error (%)':>29} {'runtime (s)':>14}")
    dx_target = 40.0 / 512.0
    for Lx in (20.0, 30.0, 40.0, 60.0):
        Nx = int(round(Lx / dx_target))
        e = OpticalBenchmarkEngine(
            Nx=Nx, Nkx=512, Lx=Lx, kx_max=kx_max, wavelength=wavelength
        )
        W = gaussian_wigner(e, w0)
        W_out, elapsed = e.propagate_exact(W, z_prop, round(z_prop / dz_nominal))
        I_out = e.normalized_marginal_intensity(W_out)
        I_rs_grid = e.independent_rayleigh_sommerfeld_intensity(w0, z_prop, n_quad=1000)
        err = relative_l2(I_out, I_rs_grid)
        print(f"{Lx:10.1f} {Nx:8d} {e.dx:12.7f} {err:29.6f} {elapsed:14.4f}")

    # -- Runtime scaling only: no accuracy inference from timing ---------------
    print("\nTABLE 4 — EMPIRICAL 100-STEP FFT EXECUTION SCALING")
    print("Timings are hardware-dependent. This is a cost benchmark, not an accuracy test.")
    resolutions = (128, 256, 512, 1024)
    times: list[float] = []
    for res in resolutions:
        e = OpticalBenchmarkEngine(
            Nx=res, Nkx=res, Lx=40.0, kx_max=kx_max, wavelength=wavelength
        )
        phi = e.compile_propagator(dz_nominal)
        test_W = np.ones((res, res), dtype=np.float64) / (res * res)
        # Warm-up reduces first-call FFT planning / cache effects.
        warm = np.fft.fft(test_W, axis=0)
        _ = np.fft.ifft(warm * phi, axis=0).real
        t0 = time.perf_counter()
        for _ in range(100):
            W_xi = np.fft.fft(test_W, axis=0)
            W_xi *= phi
            test_W = np.fft.ifft(W_xi, axis=0).real
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    base = times[0]
    print(f"{'grid':>12} {'points':>12} {'100-step time (s)':>19} {'relative scaling':>18} {'N log2 N ratio':>17}")
    for res, elapsed in zip(resolutions, times):
        points = res * res
        theoretical = (points * np.log2(points)) / ((resolutions[0] ** 2) * np.log2(resolutions[0] ** 2))
        print(f"{res:6d} x {res:<5d} {points:12,d} {elapsed:19.5f} {elapsed/base:18.3f} {theoretical:17.3f}")

    print("\nInterpretation guardrails:")
    print("  • Do not put a second-order dz convergence rate in the homogeneous CAW table:")
    print("    this kinetic step is analytic and should agree with the same-grid exact reference")
    print("    to roundoff at every dz tested.")
    print("  • If the manuscript wants an explicit-RK4 CFL/instability result, benchmark RK4")
    print("    separately; the exponential Moyal propagator printed above is not RK4.")
    print("  • Do not label same-grid exact-reference error as continuous RS-I error; the two")
    print("    metrics answer different questions.")
    print("=" * 88)


if __name__ == "__main__":
    run_benchmark()
