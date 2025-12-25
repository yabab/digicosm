import numpy as np

def precompute_radial_reduction(shape, center):
    """Precompute helpers to compute radial max profiles efficiently."""
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
    """Return the *outermost* radius where intensity exceeds threshold."""
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

def isotropy_ring_error(intensity, source_pos, *, dr=3, search_start=5):
    """Compute isotropy error σ/μ on the brightest ring around a source."""
    sy, sx = source_pos
    h, w = intensity.shape
    yy, xx = np.indices((h, w))
    r = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)

    r_int = r.astype(np.int32)
    max_r = int(r_int.max())
    flat_int = intensity.ravel()
    flat_r = r_int.ravel()

    radial_sum = np.bincount(flat_r, weights=flat_int, minlength=max_r + 1)
    radial_count = np.bincount(flat_r, minlength=max_r + 1)
    radial_mean = radial_sum / np.maximum(radial_count, 1)

    r_peak = int(np.argmax(radial_mean[search_start:]) + search_start)

    mask = (r >= r_peak - dr) & (r <= r_peak + dr)
    ring_vals = intensity[mask]
    mean_I = float(ring_vals.mean())
    std_I = float(ring_vals.std())
    iso_error = std_I / mean_I if mean_I != 0.0 else float("inf")

    return {
        "r_peak": r_peak,
        "iso_error": iso_error,
        "mean_I": mean_I,
        "std_I": std_I,
    }

def cardinal_diagonal_peak_delta(intensity, center):
    """Compare peak offsets along +x (cardinal) vs down-right diagonal."""
    cy, cx = center
    h, w = intensity.shape

    row = intensity[cy, cx:]
    card = int(np.argmax(row))

    max_len = min(h - cy, w - cx)
    diag_line = np.array([intensity[cy + i, cx + i] for i in range(max_len)])
    diag = int(np.argmax(diag_line))

    return {"card": card, "diag": diag, "delta": abs(card - diag)}

def detect_from_buffer(buf, center=None, center_y=None):
    """Detect fringe counts from a buffer of recent complex frames."""
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
                            if smooth[i] > smooth[i - 1]
                            and smooth[i] > smooth[i + 1]
                            and smooth[i] > peak_thresh
                        ]
                        min_dist = max(3, win // 2)
                        peaks = []
                        last = -min_dist * 2
                        for p in raw_peaks:
                            if p - last >= min_dist:
                                peaks.append(p)
                                last = p
                        if len(peaks) > best_local["count"]:
                            best_local.update(
                                {
                                    "count": len(peaks),
                                    "params": dict(
                                        avg=a,
                                        x_offset=x_offset,
                                        win=win,
                                        thresh=thresh_factor,
                                        orient=orientation,
                                    ),
                                    "peaks": peaks,
                                }
                            )

                        arr = smooth - smooth.mean()
                        spec = np.abs(np.fft.rfft(arr))
                        if spec.size > 1:
                            spec[0] = 0
                            dominant_bin = int(np.argmax(spec))
                            cycles = dominant_bin
                            if cycles > best_local["count"]:
                                best_local.update(
                                    {
                                        "count": cycles,
                                        "params": dict(
                                            avg=a,
                                            x_offset=x_offset,
                                            win=win,
                                            thresh=thresh_factor,
                                            orient=orientation,
                                            method="fft",
                                        ),
                                        "peaks": [],
                                    }
                                )

    return best_local

def front_radii_by_angle(intensity, center, *, threshold, angles=64, rmin=3, rmax=None):
    """Estimate wavefront radius as a function of angle.

    For each angle, samples points along the ray and returns the *outermost*
    radius with intensity > threshold. Radii with no detections are NaN.

    This is intended for isotropy tests (σ/r ≪ 1).
    """
    h, w = intensity.shape
    cy, cx = center
    if rmax is None:
        rmax = min(cy, cx, h - 1 - cy, w - 1 - cx) - 2
    rmax = int(rmax)
    rmin = int(rmin)

    radii = np.full(int(angles), np.nan, dtype=float)
    angs = np.linspace(0.0, 2 * np.pi, int(angles), endpoint=False)
    rs = np.arange(rmin, rmax + 1, dtype=int)

    for i, ang in enumerate(angs):
        ys = np.rint(cy + rs * np.sin(ang)).astype(int)
        xs = np.rint(cx + rs * np.cos(ang)).astype(int)
        valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        ys = ys[valid]
        xs = xs[valid]
        if ys.size == 0:
            continue

        vals = intensity[ys, xs]
        above = np.flatnonzero(vals > threshold)
        if above.size == 0:
            continue
        radii[i] = float(rs[valid][above[-1]])

    return radii

def estimate_angular_frequency(z_values, times):
    """Estimate angular frequency ω from complex samples z(t) by phase-unwrapping.

    Returns ω (radians per unit time) from a least-squares fit of unwrap(angle(z))
    versus time.
    """
    z = np.asarray(z_values)
    t = np.asarray(times, dtype=float)
    if z.shape[0] != t.shape[0]:
        raise ValueError("z_values and times must have the same length")
    if z.shape[0] < 3:
        raise ValueError("Need at least 3 samples to estimate frequency")

    phases = np.unwrap(np.angle(z))
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _intercept = np.linalg.lstsq(A, phases, rcond=None)[0]
    return float(slope)

def weighted_norm_change_residual(psi0, psi1, clock_rate0, clock_rate1, dt):
    """Estimate the residual in the weighted-norm balance over one step.

    Define W = Σ |ψ|^2 / N with N = clock_rate.

    For externally-prescribed N(x,t) (independent of ψ) and Hermitian H in
        i dψ/dt = N H ψ
    one has the exact identity:
        dW/dt = - Σ |ψ|^2 * (Ndot / N^2)

    This function returns (dWdt_actual, dWdt_predicted_from_Ndot, residual)
    using midpoint discretization.

    If N depends on ψ (backreaction), residual captures additional coupling terms.
    """
    psi0 = np.asarray(psi0)
    psi1 = np.asarray(psi1)
    N0 = np.asarray(clock_rate0, dtype=float)
    N1 = np.asarray(clock_rate1, dtype=float)
    dt = float(dt)
    if dt <= 0:
            raise ValueError("dt must be positive")
    if psi0.shape != psi1.shape or psi0.shape != N0.shape or psi0.shape != N1.shape:
            raise ValueError("psi0, psi1, clock_rate0, clock_rate1 must have same shape")

    if np.any(N0 <= 0) or np.any(N1 <= 0):
            raise ValueError("clock_rate must be strictly positive")

    rho0 = np.abs(psi0) ** 2
    rho1 = np.abs(psi1) ** 2

    W0 = float(np.sum(rho0 / N0))
    W1 = float(np.sum(rho1 / N1))
    dWdt_actual = (W1 - W0) / dt

    Nmid = 0.5 * (N0 + N1)
    rhomid = 0.5 * (rho0 + rho1)
    Ndot_mid = (N1 - N0) / dt
    dWdt_pred = -float(np.sum(rhomid * Ndot_mid / (Nmid ** 2)))

    residual = dWdt_actual - dWdt_pred
    return dWdt_actual, dWdt_pred, residual