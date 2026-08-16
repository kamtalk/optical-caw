"""
===============================================================================
Complex-Action Wigner (CAW) Spectral Transport Engine for Physical Optics
Author: Kenneth A. Menard (Waterloo, Ontario, Canada)
===============================================================================
"""

import numpy as np


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
        """Exact Non-Paraxial Helmholtz Kinetic Phase (Corrected Fourier Sign)"""
        k_plus = self.KX_grid + 0.5 * self.XI
        k_minus = self.KX_grid - 0.5 * self.XI

        kz_sq_plus = np.maximum(0.0, (self.k0 * self.n0)**2 - k_plus**2)
        kz_sq_minus = np.maximum(0.0, (self.k0 * self.n0)**2 - k_minus**2)

        Delta_kz = np.sqrt(kz_sq_plus) - np.sqrt(kz_sq_minus)
        # +1j sign ensures positive kx moves to positive x
        self.Phi_T = np.exp(+1j * dz * Delta_kz)

    def compile_paraxial_kinetic(self, dz):
        """Paraxial Fresnel Kinetic Phase (Corrected Fourier Sign)"""
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
        """2nd-order Strang Splitting Step"""
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