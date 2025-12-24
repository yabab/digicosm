import numpy as np
from collections import deque


def laplacian_iso(f):
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


def step_relativistic_2nd_order(psi_nm1, psi_n, dt, c, m=0.0, drive=None):
    """Second-order-in-time relativistic update for a complex scalar field.

    Implements:
        psi^{n+1} = 2 psi^n - psi^{n-1} + dt^2 ( c^2 ∇^2_iso psi^n - m^2 psi^n + drive )

    `drive` may be None or an array broadcastable to psi.
    Returns (psi_n, psi_np1) suitable for `psi_nm1, psi_n = psi_n, psi_np1`.
    """
    lap = laplacian_iso(psi_n)
    drive_term = 0.0 if drive is None else drive
    psi_np1 = (
        2 * psi_n
        - psi_nm1
        + (dt**2) * (c**2 * lap - (m**2) * psi_n + drive_term)
    )
    return psi_n, psi_np1


def energy_relativistic_2nd_order(psi_nm1, psi_n, dt, c, m=0.0):
    """Discrete total energy proxy for the relativistic complex field.

    Uses:
      - time-derivative proxy: psi_dot ≈ (psi^n - psi^{n-1}) / dt
      - gradient energy proxy: -Re(conj(psi) * ∇² psi)
      - mass term: m^2 |psi|^2
    """
    psi_dot = (psi_n - psi_nm1) / dt
    lap = laplacian_iso(psi_n)
    grad_density = -np.real(np.conj(psi_n) * lap)
    return float(
        0.5 * np.sum(np.abs(psi_dot) ** 2)
        + 0.5 * (c**2) * np.sum(grad_density)
        + 0.5 * (m**2) * np.sum(np.abs(psi_n) ** 2)
    )


def run_driven_relativistic_wave(
    psi_nm1,
    psi_n,
    steps,
    dt,
    c,
    m,
    omega,
    sources,
    *,
    avg_last=40,
    warmup_frac=0.5,
):
    """Run driven relativistic wave and return averaged complex field and buffer."""
    buf = deque(maxlen=avg_last)
    warmup_step = int(steps * warmup_frac)

    for n in range(steps):
        lap = laplacian_iso(psi_n)

        t = n * dt
        drive = np.zeros_like(psi_n)
        for (y, x, amp) in sources:
            drive[y, x] += amp * np.exp(-1j * omega * t)

        psi_np1 = 2 * psi_n - psi_nm1 + dt**2 * (c**2 * lap - m**2 * psi_n + drive)
        psi_nm1, psi_n = psi_n, psi_np1

        if n >= warmup_step:
            buf.append(psi_n.copy())

    if len(buf) == 0:
        return psi_n, [psi_n.copy()]

    avg = sum(buf) / len(buf)
    return avg, list(buf)


def run_driven_pulse(
    N,
    steps,
    dt,
    c,
    m,
    source_pos,
    pulse_amp,
    pulse_t0,
    pulse_sigma,
):
    """Run a short Gaussian pulse drive then return (times, radii) detections.

    This is used by Test 3 to measure the light-cone / front speed.
    """
    sy, sx = source_pos

    psi_nm1 = np.zeros((N, N), dtype=np.complex128)
    psi_n = np.zeros_like(psi_nm1)

    times = []
    radii = []

    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    max_r = min(sy, sx, N - 1 - sy, N - 1 - sx) - 2

    for n in range(steps):
        t = n * dt
        lap = laplacian_iso(psi_n)

        drive = np.zeros_like(psi_n)
        amp = pulse_amp * np.exp(-0.5 * ((t - pulse_t0) / pulse_sigma) ** 2)
        drive[sy, sx] = amp

        psi_np1 = 2 * psi_n - psi_nm1 + dt**2 * (c**2 * lap - m**2 * psi_n + drive)
        psi_nm1, psi_n = psi_n, psi_np1

        if n % 5 != 0:
            continue

        intensity = np.abs(psi_n) ** 2

        radial_bins = [[] for _ in range(max_r)]
        for ang in angles:
            for r in range(1, max_r):
                y = int(round(sy + r * np.sin(ang)))
                x = int(round(sx + r * np.cos(ang)))
                radial_bins[r].append(intensity[y, x])

        radial_profile = np.array([np.mean(b) if b else 0.0 for b in radial_bins])

        r_min = int(0.2 * max_r)
        r_peak = r_min + np.argmax(radial_profile[r_min:])

        if radial_profile[r_peak] > 1e-8:
            radii.append(r_peak)
            times.append(t)

    return np.array(times), np.array(radii)
