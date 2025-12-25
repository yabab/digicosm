import numpy as np
from code.model import curvature_proxy, gravity_source_from_psi, init_clock_rate_wave, step_clock_rate_wave

def test_gravity_field_causal_propagation():
    print("TEST 12: Dynamical clock_rate field propagates causally")

    # We evolve a separate clock_rate field N with a driven wave equation.
    # The drive is a *static* localized curvature source from a frozen psi.
    # Because the update is strictly local, distant points should not change
    # until the wavefront has had time to reach them.

    N = 121
    dt = 0.02

    c_g = 1.0
    gamma = 0.0
    mu = 0.0
    base = 1.0
    clip = (0.05, 2.0)

    # Frozen psi that creates a localized curvature hotspot.
    psi = np.ones((N, N), dtype=np.complex128)
    cy, cx = N // 2, N // 2
    psi[cy - 1 : cy + 2, cx - 1 : cx + 2] *= -1.0

    curv = curvature_proxy(psi)
    if float(np.max(curv)) <= 0.0:
        print("❌ FAIL: curvature source not created")
        return False

    # Important: do NOT subtract mean here; that introduces a uniform drive everywhere
    # which would change the probe immediately (not a propagation effect).
    source = gravity_source_from_psi(psi, strength=0.8, kind="curvature", subtract_mean=False)

    # Initial clock_rate and leapfrog prev.
    clock = np.ones((N, N), dtype=float) * base
    clock_prev = init_clock_rate_wave(
        clock,
        dt,
        c_g=c_g,
        gamma=gamma,
        mu=mu,
        base=base,
        source0=source,
        clip=clip,
    )

    # Probe point well away from the source.
    r_probe = 35
    probe = (cy, cx + r_probe)

    # Physical causality for the wave equation: disturbance speed is ~c_g sites/unit time.
    # Before the wavefront can travel distance r_probe (i.e. c_g * t << r_probe), the
    # probe should remain unchanged up to numerical noise.
    steps_short = 200  # t=4.0 => c_g*t=4.0 << r_probe
    for _ in range(steps_short):
        clock_next = step_clock_rate_wave(
            clock,
            clock_prev,
            dt,
            c_g=c_g,
            gamma=gamma,
            mu=mu,
            base=base,
            source=source,
            clip=clip,
        )
        clock_prev, clock = clock, clock_next

    delta_short = abs(float(clock[probe]) - base)
    print(f"t_short={steps_short*dt:.2f}, |ΔN|(probe)={delta_short:.3e}")

    if delta_short > 1e-8:
        print("❌ FAIL: Probe changed too early (non-causal / too-fast coupling)")
        return False

    # Long time: after c_g * t exceeds r_probe, a response should be detectable.
    steps_long = 2000  # additional t=40.0 => total t=44.0 > r_probe
    for _ in range(steps_long):
        clock_next = step_clock_rate_wave(
            clock,
            clock_prev,
            dt,
            c_g=c_g,
            gamma=gamma,
            mu=mu,
            base=base,
            source=source,
            clip=clip,
        )
        clock_prev, clock = clock, clock_next

    delta_long = abs(float(clock[probe]) - base)
    print(f"t_long={(steps_short+steps_long)*dt:.2f}, |ΔN|(probe)={delta_long:.3e}")

    if delta_long < 1e-8:
        print("❌ FAIL: Probe did not respond after sufficient time")
        return False

    # Sanity: remain within clip and finite.
    if not np.isfinite(clock).all():
        print("❌ FAIL: NaN/Inf in clock_rate field")
        return False
    if float(np.min(clock)) < clip[0] - 1e-12 or float(np.max(clock)) > clip[1] + 1e-12:
        print("❌ FAIL: clock_rate violated clip bounds")
        return False

    print("✅ PASS: clock_rate dynamics are local and causal")
    return True