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
    """Initialize second-order leapfrog for a Klein–Gordon–like complex field.

    Returns `psi_prev` representing ψ at t - dt given ψ(0)=psi0 and assuming
    initial velocity ψ_dot(0)=0 (unless `drive0` provides an initial accel).
    If `clock_rate` is provided it rescales the proper-time step: dτ = dt * clock_rate.
    """
    psi0 = np.array(psi0, dtype=np.complex128, copy=True)
    dt_eff = dt if clock_rate is None else dt * clock_rate

    drive0_arr = 0.0 if drive0 is None else np.asarray(drive0, dtype=np.complex128)

    # psi_tt(0) = -H psi0 + drive0
    Hpsi0 = apply_hamiltonian(psi0, omega, kappa, laplacian=laplacian)
    psi_tt0 = -(Hpsi0) + drive0_arr

    psi_prev = psi0 - 0.5 * (dt_eff ** 2) * psi_tt0
    return psi_prev

def init_phase_leapfrog_backreacting(
    psi0,
    dt,
    omega,
    kappa,
    *,
    drive0=None,
    base_clock_rate=1.0,
    curvature_beta=0.0,
    curvature_mode="exp",
    curvature_clip=(1e-3, 1e3),
    laplacian=laplacian_iso,
):
    """Initialize leapfrog when clock_rate depends on `psi` (backreaction).

    Computes an initial local `clock_rate` from `psi0` and delegates to
    `init_phase_leapfrog` to produce `psi_prev`.
    """
    if curvature_beta == 0.0:
        clock_rate0 = float(base_clock_rate)
    else:
        clock_rate0 = clock_rate_from_psi(
            psi0,
            base=float(base_clock_rate),
            beta=float(curvature_beta),
            mode=curvature_mode,
            clip=curvature_clip,
        )
    return init_phase_leapfrog(
        psi0,
        dt,
        omega,
        kappa,
        drive0=drive0,
        clock_rate=clock_rate0,
        laplacian=laplacian,
    )

def step_phase_leapfrog(
    psi,
    psi_prev,
    dt,
    omega,
    kappa,
    *,
    drive=None,
    clock_rate=None,
    laplacian=laplacian_iso,
):
    """Second-order leapfrog step for ψ_tt = -H ψ + drive.

    `psi` is ψ^n and `psi_prev` is ψ^{n-1}. Returns (psi_next, psi) where
    the second element is the new "prev" for the next step (keeps old API).
    """
    psi = np.array(psi, dtype=np.complex128, copy=False)
    psi_prev = np.array(psi_prev, dtype=np.complex128, copy=False)

    dt_eff = dt if clock_rate is None else dt * clock_rate

    drive_arr = 0.0 if drive is None else np.asarray(drive, dtype=np.complex128)

    Hpsi = apply_hamiltonian(psi, omega, kappa, laplacian=laplacian)
    psi_tt = -(Hpsi) + drive_arr

    psi_next = 2.0 * psi - psi_prev + (dt_eff ** 2) * psi_tt

    return psi_next, psi

def step_phase_leapfrog_backreacting(
    psi,
    psi_prev,
    dt,
    omega,
    kappa,
    *,
    drive=None,
    base_clock_rate=1.0,
    curvature_beta=0.0,
    curvature_mode="exp",
    curvature_clip=(1e-3, 1e3),
    laplacian=laplacian_iso,
):
    """Backreaction-aware second-order step using midpoint clock-rate estimate.

    Predictor/corrector: evaluate rate from `psi`, step, re-evaluate, average rates,
    and perform final step using midpoint rate. This mirrors the previous
    first-order behavior but for the second-order integrator.
    """
    if curvature_beta == 0.0:
        rate_n = float(base_clock_rate)
        return step_phase_leapfrog(
            psi,
            psi_prev,
            dt,
            omega,
            kappa,
            drive=drive,
            clock_rate=rate_n,
            laplacian=laplacian,
        )

    rate_n = clock_rate_from_psi(
        psi,
        base=float(base_clock_rate),
        beta=float(curvature_beta),
        mode=curvature_mode,
        clip=curvature_clip,
    )

    psi_pred, psi_prev_pred = step_phase_leapfrog(
        psi,
        psi_prev,
        dt,
        omega,
        kappa,
        drive=drive,
        clock_rate=rate_n,
        laplacian=laplacian,
    )

    rate_pred = clock_rate_from_psi(
        psi_pred,
        base=float(base_clock_rate),
        beta=float(curvature_beta),
        mode=curvature_mode,
        clip=curvature_clip,
    )
    rate_mid = 0.5 * (rate_n + rate_pred)

    return step_phase_leapfrog(
        psi,
        psi_prev,
        dt,
        omega,
        kappa,
        drive=drive,
        clock_rate=rate_mid,
        laplacian=laplacian,
    )

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
    base_clock_rate=1.0,
    curvature_beta=0.0,
    curvature_mode="exp",
    curvature_clip=(1e-3, 1e3),
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

    if curvature_beta == 0.0:
        psi_prev = init_phase_leapfrog(psi, dt, omega0, kappa, drive0=drive0, clock_rate=base_clock_rate)
    else:
        psi_prev = init_phase_leapfrog_backreacting(
            psi,
            dt,
            omega0,
            kappa,
            drive0=drive0,
            base_clock_rate=base_clock_rate,
            curvature_beta=curvature_beta,
            curvature_mode=curvature_mode,
            curvature_clip=curvature_clip,
        )

    for n in range(steps):
        t = n * dt
        drive = None
        if sources:
            drive = np.zeros_like(psi)
            ph = np.exp(-1j * drive_omega * t)
            for (y, x, amp) in sources:
                drive[y, x] += amp * ph

        psi, psi_prev = step_phase_leapfrog_backreacting(
            psi,
            psi_prev,
            dt,
            omega0,
            kappa,
            drive=drive,
            base_clock_rate=base_clock_rate,
            curvature_beta=curvature_beta,
            curvature_mode=curvature_mode,
            curvature_clip=curvature_clip,
        )

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
    base_clock_rate=1.0,
    curvature_beta=0.0,
    curvature_mode="exp",
    curvature_clip=(1e-3, 1e3),
):
    """Run a Gaussian pulse drive and return sampled frames: [(t, psi), ...]."""
    psi = np.zeros(shape, dtype=np.complex128)

    # Initialize with drive at t=0.
    t0 = 0.0
    drive0 = np.zeros_like(psi)
    sy, sx = source_pos
    drive0[sy, sx] = pulse_amp * np.exp(-0.5 * ((t0 - pulse_t0) / pulse_sigma) ** 2)
    if curvature_beta == 0.0:
        psi_prev = init_phase_leapfrog(psi, dt, omega0, kappa, drive0=drive0, clock_rate=base_clock_rate)
    else:
        psi_prev = init_phase_leapfrog_backreacting(
            psi,
            dt,
            omega0,
            kappa,
            drive0=drive0,
            base_clock_rate=base_clock_rate,
            curvature_beta=curvature_beta,
            curvature_mode=curvature_mode,
            curvature_clip=curvature_clip,
        )

    samples = []
    for n in range(steps):
        t = n * dt
        drive = np.zeros_like(psi)
        drive[sy, sx] = pulse_amp * np.exp(-0.5 * ((t - pulse_t0) / pulse_sigma) ** 2)
        psi, psi_prev = step_phase_leapfrog_backreacting(
            psi,
            psi_prev,
            dt,
            omega0,
            kappa,
            drive=drive,
            base_clock_rate=base_clock_rate,
            curvature_beta=curvature_beta,
            curvature_mode=curvature_mode,
            curvature_clip=curvature_clip,
        )
        if n % sample_every == 0:
            samples.append((t, psi.copy()))
    return samples

def energy_density(psi):
    """Microscopic energy proxy used for coarse-graining: E_i = |ψ_i|^2."""
    return np.abs(psi) ** 2


def kinetic_energy_density(psi_now, psi_prev, dt):
    """Per-site kinetic energy density ~ |(ψ_now - ψ_prev)/dt|^2."""
    psi_now = np.asarray(psi_now)
    psi_prev = np.asarray(psi_prev)
    vel = (psi_now - psi_prev) / float(dt)
    return np.abs(vel) ** 2


def kinetic_energy(psi_now, psi_prev, dt):
    """Total kinetic energy (scalar)."""
    return float(np.sum(kinetic_energy_density(psi_now, psi_prev, dt)))


def potential_energy_density(psi, *, omega0=0.0, kappa=1.0):
    """Per-site potential energy density: ω|ψ|^2 + κ C_i where C_i is curvature_proxy."""
    return float(omega0) * energy_density(psi) + float(kappa) * curvature_proxy(psi)


def potential_energy(psi, *, omega0=0.0, kappa=1.0):
    """Total potential energy (scalar)."""
    return float(np.sum(potential_energy_density(psi, omega0=omega0, kappa=kappa)))


def total_energy_density(psi_now, psi_prev, dt, *, omega0=0.0, kappa=1.0):
    """Total per-site energy density = KE_density + PE_density."""
    ke = kinetic_energy_density(psi_now, psi_prev, dt)
    pe = potential_energy_density(psi_now, omega0=omega0, kappa=kappa)
    return ke + pe


def total_energy(psi_now, psi_prev, dt, *, omega0=0.0, kappa=1.0):
    """Total scalar energy: sum over sites of KE+PE."""
    return float(np.sum(total_energy_density(psi_now, psi_prev, dt, omega0=omega0, kappa=kappa)))

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

def clock_rate_from_curvature(
    curvature,
    *,
    base=1.0,
    beta=0.0,
    mode="exp",
    clip=(1e-3, 1e3),
):
    """Map curvature proxy to a local clock-rate field (time dilation).

    This returns a dimensionless scalar field `clock_rate` used by the integrator
    via dτ = dt * clock_rate.

    - `beta` controls coupling strength; `beta=0` yields constant `base`.
    - `mode` controls the monotone map (higher curvature => slower clock):
        - "exp":     base * exp(-beta * curvature)
        - "rational": base / (1 + beta * curvature)
    - `clip` avoids non-physical or numerically extreme rates.
    """
    curv = np.asarray(curvature, dtype=float)

    # Allow `base` and `beta` to be scalars or arrays broadcastable with `curv`.
    base_arr = np.asarray(base, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)

    try:
        if mode == "exp":
            rate = base_arr * np.exp(-beta_arr * curv)
        elif mode == "rational":
            rate = base_arr / (1.0 + beta_arr * curv)
        else:
            raise ValueError(f"Unknown mode={mode!r}; expected 'exp' or 'rational'.")
    except ValueError as e:
        # Provide a clearer error when shapes are incompatible for broadcasting.
        raise ValueError(
            "`base` and `beta` must be broadcastable to the shape of `curvature`"
        ) from e

    if clip is not None:
        lo, hi = clip
        rate = np.clip(rate, float(lo), float(hi))
    return rate

def clock_rate_from_psi(
    psi,
    *,
    base=1.0,
    beta=0.0,
    mode="exp",
    clip=(1e-3, 1e3),
):
    """Convenience: compute curvature proxy then map to clock_rate."""
    return clock_rate_from_curvature(
        curvature_proxy(psi),
        base=base,
        beta=beta,
        mode=mode,
        clip=clip,
    )


def compute_clock_rate(*, psi=None, curvature=None, base=1.0, beta=0.0, mode="exp", clip=(1e-3, 1e3)):
    """Centralized helper to compute a clock_rate field.

    Exactly one of `psi` or `curvature` must be provided. If `psi` is given,
    the function computes the curvature proxy and maps it to a clock_rate.
    Otherwise it maps the provided `curvature` array.

    This helper centralizes argument validation and clipping semantics so callers
    can rely on a single API for clock-rate computation.
    """
    if (psi is None) == (curvature is None):
        raise ValueError("Provide exactly one of 'psi' or 'curvature'")

    if psi is not None:
        return clock_rate_from_psi(psi, base=base, beta=beta, mode=mode, clip=clip)
    else:
        return clock_rate_from_curvature(curvature, base=base, beta=beta, mode=mode, clip=clip)

def gravity_source_from_psi(
    psi,
    *,
    strength=1.0,
    kind="curvature",
    subtract_mean=True,
):
    """Compute a scalar source field for a dynamical clock_rate (gravity) field.

    This is intentionally simple and local.
    - kind="curvature": uses curvature_proxy(psi)
    - kind="density":   uses |psi|^2

    `subtract_mean=True` keeps the spatial mean of the source near zero so a
    base clock rate (e.g. 1.0) remains the natural background.
    """
    if kind == "curvature":
        s = curvature_proxy(psi).astype(float)
    elif kind == "density":
        s = energy_density(psi).astype(float)
    else:
        raise ValueError(f"Unknown kind={kind!r}; expected 'curvature' or 'density'.")

    if subtract_mean:
        s = s - float(np.mean(s))
    return float(strength) * s


def gravity_source_from_fields(fields, *, strength=1.0, kind=None, subtract_mean=True):
    """Adapter: build a gravity source from a flexible `fields` input.

    `fields` can be:
    - a complex array: interpreted as `psi` and forwarded to `gravity_source_from_psi`.
    - a real array: interpreted as a precomputed `density` (|psi|^2) unless `kind` is
      explicitly set to 'curvature', in which case it's treated as curvature.
    - a mapping containing one of the keys: 'psi', 'curvature', 'density'.

    The `kind` argument can force interpretation ('curvature'|'density'|'psi').
    If ambiguous or no usable data is found, raises `ValueError`.
    """
    # If a mapping-like object, prefer explicit keys
    keys = getattr(fields, "keys", None)
    if callable(keys):
        # dict-like
        fkeys = set(fields.keys())
        if "psi" in fkeys and (kind is None or kind == "psi"):
            return gravity_source_from_psi(fields["psi"], strength=strength, kind="curvature", subtract_mean=subtract_mean)
        if "curvature" in fkeys and (kind is None or kind == "curvature"):
            s = np.asarray(fields["curvature"], dtype=float)
            if subtract_mean:
                s = s - float(np.mean(s))
            return float(strength) * s
        if "density" in fkeys and (kind is None or kind == "density"):
            s = np.asarray(fields["density"], dtype=float)
            if subtract_mean:
                s = s - float(np.mean(s))
            return float(strength) * s
        # If mapping-like but contains none of the expected keys, fail fast.
        raise ValueError("Mapping input must contain one of: 'psi', 'curvature', 'density'")

    arr = np.asarray(fields)
    if np.iscomplexobj(arr) or (kind == "psi"):
        return gravity_source_from_psi(arr, strength=strength, kind="curvature", subtract_mean=subtract_mean)

    # At this point arr is real-valued. Interpret based on `kind` if provided.
    if kind == "curvature":
        s = arr.astype(float)
    elif kind == "density" or kind is None:
        # by default treat a real array as density
        s = arr.astype(float)
    else:
        raise ValueError("Unable to interpret `fields` for gravity source; provide 'psi', 'curvature' or 'density'.")

    if subtract_mean:
        s = s - float(np.mean(s))
    return float(strength) * s

def init_clock_rate_wave(
    clock_rate0,
    dt,
    *,
    c_g=1.0,
    gamma=0.0,
    mu=0.0,
    base=1.0,
    source0=None,
    clip=(1e-3, 1e3),
    laplacian=laplacian_iso,
):
    """Initialize leapfrog state for a dynamical clock_rate field.

    Evolves a scalar field N(t,x) with:
      N_tt = c_g^2 ∇^2 N + source - mu^2 (N-base) - gamma N_t

    Returns N_prev for use in `step_clock_rate_wave`.
    Assumes N_t(t=0)=0.
    """
    N0 = np.array(clock_rate0, dtype=float, copy=True)
    src0 = 0.0 if source0 is None else np.asarray(source0, dtype=float)

    accel0 = (c_g ** 2) * laplacian(N0) + src0 - (mu ** 2) * (N0 - float(base))
    # With N_t(0)=0, leapfrog consistent initialization:
    #   N(-dt) = N(0) - 0.5 dt^2 * N_tt(0)
    N_prev = N0 - 0.5 * (dt ** 2) * accel0

    if clip is not None:
        lo, hi = clip
        N0 = np.clip(N0, float(lo), float(hi))
        N_prev = np.clip(N_prev, float(lo), float(hi))
    return N_prev

def step_clock_rate_wave(
    clock_rate,
    clock_rate_prev,
    dt,
    *,
    c_g=1.0,
    gamma=0.0,
    mu=0.0,
    base=1.0,
    source=None,
    clip=(1e-3, 1e3),
    laplacian=laplacian_iso,
):
    """Advance the dynamical clock_rate (gravity) field by one leapfrog step."""
    N = np.asarray(clock_rate, dtype=float)
    N_prev = np.asarray(clock_rate_prev, dtype=float)
    src = 0.0 if source is None else np.asarray(source, dtype=float)

    # N_t ≈ (N - N_prev)/dt
    N_t = (N - N_prev) / float(dt)
    accel = (c_g ** 2) * laplacian(N) + src - (mu ** 2) * (N - float(base)) - float(gamma) * N_t

    N_next = 2.0 * N - N_prev + (dt ** 2) * accel

    if clip is not None:
        lo, hi = clip
        N_next = np.clip(N_next, float(lo), float(hi))
    return N_next

def init_coupled_gravity_matter(
    psi0,
    dt,
    omega,
    kappa,
    *,
    clock_rate0=1.0,
    gravity_c_g=1.0,
    gravity_gamma=0.0,
    gravity_mu=0.0,
    gravity_base=1.0,
    gravity_source_strength=1.0,
    gravity_source_kind="density",
    gravity_subtract_mean=False,
    gravity_source=None,
    gravity_clip=(1e-3, 1e3),
):
    """Initialize coupled evolution for (psi, clock_rate).

    - clock_rate evolves with a local wave equation (finite-speed gravity field)
    - psi evolves with local proper time: dτ = dt * clock_rate

    Returns (psi, psi_prev, clock_rate, clock_rate_prev).
    """
    psi = np.array(psi0, dtype=np.complex128, copy=True)

    clock_rate = np.array(clock_rate0, dtype=float, copy=True)
    if gravity_source is None:
        src0 = gravity_source_from_psi(
            psi,
            strength=gravity_source_strength,
            kind=gravity_source_kind,
            subtract_mean=gravity_subtract_mean,
        )
    else:
        src0 = np.asarray(gravity_source, dtype=float)
    clock_rate_prev = init_clock_rate_wave(
        clock_rate,
        dt,
        c_g=gravity_c_g,
        gamma=gravity_gamma,
        mu=gravity_mu,
        base=gravity_base,
        source0=src0,
        clip=gravity_clip,
    )

    psi_prev = init_phase_leapfrog(psi, dt, omega, kappa, clock_rate=clock_rate)
    return psi, psi_prev, clock_rate, clock_rate_prev

def step_coupled_gravity_matter(
    psi,
    psi_prev,
    clock_rate,
    clock_rate_prev,
    dt,
    omega,
    kappa,
    *,
    drive=None,
    gravity_c_g=1.0,
    gravity_gamma=0.0,
    gravity_mu=0.0,
    gravity_base=1.0,
    gravity_source_strength=1.0,
    gravity_source_kind="density",
    gravity_subtract_mean=False,
    gravity_source=None,
    gravity_clip=(1e-3, 1e3),
):
    """Advance (psi, clock_rate) one step with midpoint coupling.

    - Source is computed from current psi.
    - clock_rate is advanced via wave equation.
    - psi is advanced using clock_rate at the midpoint (N_mid) for symmetry.
    """
    if gravity_source is None:
        src = gravity_source_from_psi(
            psi,
            strength=gravity_source_strength,
            kind=gravity_source_kind,
            subtract_mean=gravity_subtract_mean,
        )
    else:
        src = np.asarray(gravity_source, dtype=float)

    clock_next = step_clock_rate_wave(
        clock_rate,
        clock_rate_prev,
        dt,
        c_g=gravity_c_g,
        gamma=gravity_gamma,
        mu=gravity_mu,
        base=gravity_base,
        source=src,
        clip=gravity_clip,
    )

    clock_mid = 0.5 * (clock_rate + clock_next)
    psi_next, psi_prev_next = step_phase_leapfrog(
        psi,
        psi_prev,
        dt,
        omega,
        kappa,
        drive=drive,
        clock_rate=clock_mid,
    )

    return psi_next, psi_prev_next, clock_next, clock_rate

def hamiltonian_total(psi, *, omega0=0.0, kappa=1.0):
    """Total Hamiltonian proxy matching H = Σ (ω|ψ|^2 + κ Σ |ψ_i-ψ_j|^2)."""
    return float(np.sum(omega0 * energy_density(psi) + kappa * curvature_proxy(psi)))

def weighted_norm(psi, *, clock_rate):
    """Metric-weighted norm for lapse-coupled evolution.

    For dynamics i dψ/dt = N(x) H ψ with H Hermitian and fixed N(x)>0,
    the conserved quadratic form is ∑ |ψ|^2 / N.

    Here `clock_rate` plays the role of N.
    """
    N = np.asarray(clock_rate, dtype=float)
    if np.any(N <= 0):
        raise ValueError("clock_rate must be strictly positive")
    return float(np.sum(energy_density(psi) / N))