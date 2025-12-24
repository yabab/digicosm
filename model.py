import numpy as np
from collections import deque


# ============================================================
# Phase dynamics core (reversible, local, linear)
#
# Desired microscopic law (proper time τ):
#   i dψ/dτ = ω ψ + κ L ψ + drive
# where L is a local isotropic coupling operator (discrete Laplacian-like).
#
# Implemented with an explicit, time-reversible leapfrog on (Re ψ, Im ψ).
# ============================================================


def laplacian_iso(f):
    """Isotropic 2D stencil using axial + diagonal neighbors (periodic)."""
    axial = (
        np.roll(f, 1, 0) + np.roll(f, -1, 0)
        + np.roll(f, 1, 1) + np.roll(f, -1, 1)
    )
    diag = (
        np.roll(np.roll(f, 1, 0), 1, 1)
        + np.roll(np.roll(f, 1, 0), -1, 1)
        + np.roll(np.roll(f, -1, 0), 1, 1)
        + np.roll(np.roll(f, -1, 0), -1, 1)
    )
    return (4 / 6) * (axial - 4 * f) + (1 / 6) * (diag - 4 * f)


def apply_hamiltonian(psi, omega, kappa, *, laplacian=laplacian_iso):
    """Apply the linear Hermitian operator Hψ = ωψ + κ Lψ.

    `omega` may be a scalar or an array (per-site frequency, future curvature hook).
    `kappa` may be a scalar (uniform coupling) or an array (future anisotropy/metric).
    """
    return omega * psi + kappa * laplacian(psi)


def init_phase_leapfrog(psi0, dt, omega, kappa, *, drive0=None, clock_rate=None, laplacian=laplacian_iso):
    """Initialize leapfrog half-step storage for phase dynamics.

    Stores v_{1/2} (imag part at half step) given ψ(0) = u0 + i v0.
    If provided, `clock_rate` rescales the local proper-time step: dτ = dt * clock_rate.
    """
    u0 = np.real(psi0)
    v0 = np.imag(psi0)
    dt_eff = dt if clock_rate is None else dt * clock_rate

    drive_r0 = 0.0 if drive0 is None else np.real(drive0)
    Au0 = omega * u0 + kappa * laplacian(u0)
    dvdt0 = -(Au0 + drive_r0)
    v_half0 = v0 + 0.5 * dt_eff * dvdt0
    return v_half0


def step_phase_leapfrog(
    psi,
    v_half,
    dt,
    omega,
    kappa,
    *,
    drive=None,
    clock_rate=None,
    laplacian=laplacian_iso,
):
    """One time-reversible leapfrog step for i ψdot = Hψ + drive.

    State:
      ψ^n is stored as a complex field at integer steps.
      v_half stores Im(ψ) at half steps.

    Returns (psi_next, v_half_next).
    """
    u = np.real(psi)
    dt_eff = dt if clock_rate is None else dt * clock_rate

    drive_r = 0.0 if drive is None else np.real(drive)
    drive_i = 0.0 if drive is None else np.imag(drive)

    Av_half = omega * v_half + kappa * laplacian(v_half)
    u_next = u + dt_eff * (Av_half + drive_i)

    Au_next = omega * u_next + kappa * laplacian(u_next)
    dvdt_next = -(Au_next + drive_r)

    v_next = v_half + 0.5 * dt_eff * dvdt_next
    v_half_next = v_half + dt_eff * dvdt_next

    psi_next = u_next + 1j * v_next
    return psi_next, v_half_next


def run_driven_phase_wave(
    psi0,
    steps,
    dt,
    *,
    kappa=1.0,
    omega0=0.0,
    drive_omega=0.0,
    sources=(),
    avg_last=40,
    warmup_frac=0.5,
):
    """Run monochromatically-driven phase dynamics; return (avg_field, buffer)."""
    buf = deque(maxlen=avg_last)
    warmup_step = int(steps * warmup_frac)

    psi = np.array(psi0, dtype=np.complex128, copy=True)

    t0 = 0.0
    drive0 = None
    if sources:
        drive0 = np.zeros_like(psi)
        for (y, x, amp) in sources:
            drive0[y, x] += amp * np.exp(-1j * drive_omega * t0)

    v_half = init_phase_leapfrog(psi, dt, omega0, kappa, drive0=drive0)

    for n in range(steps):
        t = n * dt
        drive = None
        if sources:
            drive = np.zeros_like(psi)
            ph = np.exp(-1j * drive_omega * t)
            for (y, x, amp) in sources:
                drive[y, x] += amp * ph

        psi, v_half = step_phase_leapfrog(psi, v_half, dt, omega0, kappa, drive=drive)

        if n >= warmup_step:
            buf.append(psi.copy())

    if len(buf) == 0:
        return psi, [psi.copy()]

    avg = sum(buf) / len(buf)
    return avg, list(buf)


def run_pulsed_drive_samples(
    shape,
    steps,
    dt,
    *,
    kappa=1.0,
    omega0=0.0,
    source_pos=(0, 0),
    pulse_amp=1.0,
    pulse_t0=0.0,
    pulse_sigma=1.0,
    sample_every=5,
):
    """Run a Gaussian pulse drive and return sampled frames: [(t, psi), ...]."""
    psi = np.zeros(shape, dtype=np.complex128)

    # Initialize with drive at t=0.
    t0 = 0.0
    drive0 = np.zeros_like(psi)
    sy, sx = source_pos
    drive0[sy, sx] = pulse_amp * np.exp(-0.5 * ((t0 - pulse_t0) / pulse_sigma) ** 2)
    v_half = init_phase_leapfrog(psi, dt, omega0, kappa, drive0=drive0)

    samples = []
    for n in range(steps):
        t = n * dt
        drive = np.zeros_like(psi)
        drive[sy, sx] = pulse_amp * np.exp(-0.5 * ((t - pulse_t0) / pulse_sigma) ** 2)
        psi, v_half = step_phase_leapfrog(psi, v_half, dt, omega0, kappa, drive=drive)
        if n % sample_every == 0:
            samples.append((t, psi.copy()))
    return samples


def energy_density(psi):
    """Microscopic energy proxy used for coarse-graining: E_i = |ψ_i|^2."""
    return np.abs(psi) ** 2


def curvature_proxy(psi):
    """Local curvature proxy C_i ∝ Σ_j |ψ_i - ψ_j|^2 (axial + diagonal weighted)."""
    axial = (
        np.abs(psi - np.roll(psi, -1, 0)) ** 2
        + np.abs(psi - np.roll(psi, 1, 0)) ** 2
        + np.abs(psi - np.roll(psi, -1, 1)) ** 2
        + np.abs(psi - np.roll(psi, 1, 1)) ** 2
    )
    diag = (
        np.abs(psi - np.roll(np.roll(psi, -1, 0), -1, 1)) ** 2
        + np.abs(psi - np.roll(np.roll(psi, -1, 0), 1, 1)) ** 2
        + np.abs(psi - np.roll(np.roll(psi, 1, 0), -1, 1)) ** 2
        + np.abs(psi - np.roll(np.roll(psi, 1, 0), 1, 1)) ** 2
    )
    return (4 / 6) * axial + (1 / 6) * diag


def hamiltonian_total(psi, *, omega0=0.0, kappa=1.0):
    """Total Hamiltonian proxy matching H = Σ (ω|ψ|^2 + κ Σ |ψ_i-ψ_j|^2)."""
    return float(np.sum(omega0 * energy_density(psi) + kappa * curvature_proxy(psi)))
