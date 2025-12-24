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
