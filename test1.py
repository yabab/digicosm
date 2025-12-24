import numpy as np
from collections import deque
from model import laplacian_iso, step_complex

# ============================================================
# TEST 1 — Single Source Isotropy (Driven, Complex, Relativistic)
# ============================================================

def test_single_source_isotropy_relativistic():
    print("\nTEST 1: Relativistic Complex Single-Source Isotropy (Driven)")

    N = 400
    steps = 1200
    dt = 0.2
    c = 1.0
    m = 0.0
    omega = 4.0
    center = N // 2

    psi_nm1 = np.zeros((N, N), dtype=np.complex128)
    psi_n   = np.zeros_like(psi_nm1)

    # Single point source for isotropy test
    amp = 1.0
    sources = [(center, center - 120, amp)]

    # --- run driven evolution (local arrays to avoid closure issues) ---
    buf = deque(maxlen=600)
    warmup_step = steps // 2

    psi0 = psi_nm1.copy()
    psi1 = psi_n.copy()

    for n in range(steps):
        lap = laplacian_iso(psi1)
        t = n * dt

        drive = np.zeros_like(psi1)
        for (y, x, a) in sources:
            drive[y, x] += a * np.exp(-1j * omega * t)

        psi2 = (
            2*psi1 - psi0
            + dt**2 * (c**2 * lap - m**2 * psi1 + drive)
        )

        psi0, psi1 = psi1, psi2

        if n >= warmup_step:
            buf.append(psi1.copy())

    if len(buf) == 0:
        raise RuntimeError('No frames collected for averaging')

    psi_avg = sum(buf) / len(buf)
    intensity = np.abs(psi_avg)**2

    # --- isotropy measurement around the source center ---
    # source center (middle of the vertical line)
    sy = center
    sx = center - 120

    yy, xx = np.indices((N, N))
    r = np.sqrt((xx - sx)**2 + (yy - sy)**2)

    # compute radial mean profile and find the peak radius
    r_int = r.astype(int)
    max_r = r_int.max()
    flat_int = intensity.ravel()
    flat_r = r_int.ravel()
    radial_sum = np.bincount(flat_r, weights=flat_int, minlength=max_r+1)
    radial_count = np.bincount(flat_r, minlength=max_r+1)
    radial_mean = radial_sum / np.maximum(radial_count, 1)

    # ignore very small radii (near source) and find global peak
    search_start = 5
    r_peak = int(np.argmax(radial_mean[search_start:]) + search_start)

    dr = 3
    mask = (r >= r_peak - dr) & (r <= r_peak + dr)
    ring_vals = intensity[mask]

    mean_I = ring_vals.mean()
    std_I = ring_vals.std()
    iso_error = std_I / mean_I

    print(f"peak radius = {r_peak}, σ/I = {iso_error:.4f}")

    if iso_error < 0.05:
        print("✅ PASS: Isotropic steady-state wavefront")
        return True
    else:
        print("❌ FAIL: Anisotropy detected")
        return False

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    ok = test_single_source_isotropy_relativistic()
    if not ok:
        print("Test failed.")
    else:
        print("Test passed.")
