import numpy as np
from collections import deque


# ============================================================
# Shared model helpers for wave tests (renamed from wave_helpers)
# ============================================================


def laplacian_iso(f):
    axial = (
        np.roll(f, 1, 0) + np.roll(f, -1, 0) +
        np.roll(f, 1, 1) + np.roll(f, -1, 1)
    )
    diag = (
        np.roll(np.roll(f, 1, 0), 1, 1) +
        np.roll(np.roll(f, 1, 0), -1, 1) +
        np.roll(np.roll(f, -1, 0), 1, 1) +
        np.roll(np.roll(f, -1, 0), -1, 1)
    )
    return (4/6) * (axial - 4 * f) + (1/6) * (diag - 4 * f)


def run_driven_relativistic_wave(
    psi_nm1, psi_n, steps, dt, c, m, omega, sources, *, avg_last=40, warmup_frac=0.5
):
    """Run driven relativistic wave and return averaged complex field and buffer.

    This copies the implementation previously embedded in `test2.py` so tests
    can reuse a canonical routine.
    """
    buf = deque(maxlen=avg_last)
    warmup_step = int(steps * warmup_frac)

    for n in range(steps):
        lap = laplacian_iso(psi_n)

        # monochromatic driving term
        t = n * dt
        drive = np.zeros_like(psi_n)
        for (y, x, amp) in sources:
            drive[y, x] += amp * np.exp(-1j * omega * t)

        psi_np1 = (
            2 * psi_n - psi_nm1 + dt**2 * (c**2 * lap - m**2 * psi_n + drive)
        )

        psi_nm1, psi_n = psi_n, psi_np1

        if n >= warmup_step:
            buf.append(psi_n.copy())

    if len(buf) == 0:
        return psi_n, [psi_n.copy()]

    avg = sum(buf) / len(buf)
    return avg, list(buf)


def step_relativistic_2nd_order(psi_nm1, psi_n, dt, c, m=0.0, drive=None):
    """Second-order-in-time relativistic update for a complex scalar field.

    Implements:
        psi^{n+1} = 2 psi^n - psi^{n-1} + dt^2 ( c^2 ∇^2_iso psi^n - m^2 psi^n + drive )

    `drive` may be None or an array broadcastable to psi.
    Returns (psi_n, psi_np1) suitable for `psi_nm1, psi_n = psi_n, psi_np1`.
    """
    lap = laplacian_iso(psi_n)
    if drive is None:
        drive_term = 0.0
    else:
        drive_term = drive
    psi_np1 = 2 * psi_n - psi_nm1 + (dt ** 2) * (c ** 2 * lap - (m ** 2) * psi_n + drive_term)
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
        + 0.5 * (c ** 2) * np.sum(grad_density)
        + 0.5 * (m ** 2) * np.sum(np.abs(psi_n) ** 2)
    )


def run_driven_pulse(
    N, steps, dt, c, m,
    source_pos,
    pulse_amp, pulse_t0, pulse_sigma,
):
    """Run a short Gaussian pulse drive then return (times, radii) detections.

    This helper is used by Test 3 to measure the light-cone / front speed.
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

        psi_np1 = (
            2 * psi_n - psi_nm1
            + dt ** 2 * (c ** 2 * lap - m ** 2 * psi_n + drive)
        )

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


def precompute_radial_reduction(shape, center):
    """Precompute helpers to compute radial max profiles efficiently.

    Used by Test 5's robust front detector.
    """
    h, w = shape
    cy, cx = center
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_int = rr.astype(np.int32).ravel()

    order = np.argsort(r_int)
    r_sorted = r_int[order]
    starts = np.flatnonzero(np.r_[True, r_sorted[1:] != r_sorted[:-1]])
    r_vals = r_sorted[starts]
    max_r = int(r_vals.max())

    return {
        "order": order,
        "starts": starts,
        "r_vals": r_vals,
        "max_r": max_r,
    }


def detect_front_outermost(intensity, radial_cache, *, rmin=5, threshold=1e-8):
    """Return the *outermost* radius where intensity exceeds threshold.

    Using the outermost radius avoids false negatives/oscillations near the
    source and produces a monotonic front radius until periodic wrap-around.
    """
    flat_sorted = intensity.ravel()[radial_cache["order"]]
    max_per_seg = np.maximum.reduceat(flat_sorted, radial_cache["starts"])

    radial_max = np.zeros(radial_cache["max_r"] + 1, dtype=max_per_seg.dtype)
    radial_max[radial_cache["r_vals"]] = max_per_seg

    for r in range(
        min(radial_cache["max_r"], intensity.shape[0] // 2 - 1),
        rmin - 1,
        -1,
    ):
        if radial_max[r] > threshold:
            return r
    return None


def detect_from_buffer(buf, center=None, center_y=None):
    """Detect fringe counts from a buffer of recent complex frames.

    It returns a dict with keys: 'count', 'params', 'peaks'.
    """
    best_local = {"count": 0, "params": None, "peaks": []}
    avail = len(buf)
    avg_lengths = sorted(set([1, 3, 5, 10, 20, 40, 80, 120]))
    avg_lengths = [a for a in avg_lengths if a <= max(1, avail)]

    for a in avg_lengths:
        frames_to_avg = buf[-a:]
        avg_field = sum(frames_to_avg) / len(frames_to_avg)
        intensity = np.abs(avg_field) ** 2
        h, w = intensity.shape
        if center is None:
            cy = h // 2
            cx = w // 2
        else:
            cy, cx = center

        for x_offset in range(-20, 161, 20):
            x0 = max(0, cx + x_offset - 40)
            x1 = min(w, cx + x_offset + 160)
            y0 = max(0, cy - 3)
            y1 = min(h, cy + 4)
            band = intensity[y0:y1, x0:x1]
            if band.size == 0:
                continue

            line_h = band.mean(axis=0)
            line_v = band.mean(axis=1)
            for line, orientation in ((line_h, "h"), (line_v, "v")):
                for win in (3, 5, 7, 11):
                    window = np.ones(win) / win
                    smooth = np.convolve(line, window, mode="same")
                    baseline = np.median(smooth)
                    for thresh_factor in (0.02, 0.04, 0.06, 0.1):
                        peak_thresh = baseline + thresh_factor * (smooth.max() - baseline)
                        raw_peaks = [
                            i
                            for i in range(1, len(smooth) - 1)
                            if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] > peak_thresh
                        ]
                        min_dist = max(3, win // 2)
                        peaks = []
                        last = -min_dist * 2
                        for p in raw_peaks:
                            if p - last >= min_dist:
                                peaks.append(p)
                                last = p
                        if len(peaks) > best_local["count"]:
                            best_local.update({"count": len(peaks), "params": dict(avg=a, x_offset=x_offset, win=win, thresh=thresh_factor, orient=orientation), "peaks": peaks})

                        arr = smooth - smooth.mean()
                        spec = np.abs(np.fft.rfft(arr))
                        if spec.size > 1:
                            spec[0] = 0
                            dominant_bin = int(np.argmax(spec))
                            cycles = dominant_bin
                            if cycles > best_local["count"]:
                                best_local.update({"count": cycles, "params": dict(avg=a, x_offset=x_offset, win=win, thresh=thresh_factor, orient=orientation, method='fft'), "peaks": []})

    return best_local
