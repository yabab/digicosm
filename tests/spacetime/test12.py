import numpy as np
from code.model import curvature_proxy, gravity_source_from_psi, init_clock_rate_wave, step_clock_rate_wave

def test_gravity_field_causal_propagation():
    """Test causal propagation of dynamical clock_rate field.
    
    Validates that the clock_rate field evolves according to a local wave equation
    with finite propagation speed. A probe point far from the source should:
    1. Remain unchanged before the wave front arrives (causality)
    2. Show measurable change after sufficient time for propagation
    3. Stay within physical bounds (clip, finite values)
    """
    print("\nTEST 12: Dynamical clock_rate field propagates causally")

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

    # Input validation
    assert N > 0, "Grid size must be positive"
    assert dt > 0 and c_g > 0, "Time step and propagation speed must be positive"
    assert base > 0, "Base clock rate must be positive"
    assert clip[0] > 0 and clip[1] > clip[0], "Invalid clip bounds"

    # Frozen psi that creates a localized curvature hotspot.
    psi = np.ones((N, N), dtype=np.complex128)
    cy, cx = N // 2, N // 2
    psi[cy - 1 : cy + 2, cx - 1 : cx + 2] *= -1.0

    curv = curvature_proxy(psi)
    assert np.isfinite(curv).all(), "Curvature contains non-finite values"
    
    max_curv = float(np.max(curv))
    if max_curv <= 0.0:
        print("❌ FAIL: Curvature source not created (max curvature <= 0)")
        return False
    
    print(f"Info: Grid size={N}, dt={dt}, c_g={c_g}")
    print(f"  Curvature max={max_curv:.6f}")

    # Important: do NOT subtract mean here; that introduces a uniform drive everywhere
    # which would change the probe immediately (not a propagation effect).
    source_strength = 0.8
    source = gravity_source_from_psi(psi, strength=source_strength, kind="curvature", subtract_mean=False)
    
    assert np.isfinite(source).all(), "Source contains non-finite values"
    print(f"  Source strength={source_strength}, max={np.max(np.abs(source)):.6f}")

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
    
    assert np.isfinite(clock_prev).all(), "clock_prev contains non-finite values"

    # Probe point well away from the source.
    r_probe = 35
    probe = (cy, cx + r_probe)
    
    assert 0 <= probe[0] < N and 0 <= probe[1] < N, "Probe position out of bounds"
    print(f"  Probe position: {probe}, distance from center: {r_probe}")

    # Physical causality for the wave equation: disturbance speed is ~c_g sites/unit time.
    # Before the wavefront can travel distance r_probe (i.e. c_g * t << r_probe), the
    # probe should remain unchanged up to numerical noise.
    steps_short = 200  # t=4.0 => c_g*t=4.0 << r_probe=35
    t_short = steps_short * dt
    
    print(f"\nEarly-time test (causality check):")
    print(f"  Running {steps_short} steps (t={t_short:.2f})")
    print(f"  Expected travel distance: c_g * t = {c_g * t_short:.2f} << r_probe = {r_probe}")
    
    for i in range(steps_short):
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
        
        if i % 50 == 0:
            assert np.isfinite(clock_next).all(), f"Non-finite values at step {i}"
        
        clock_prev, clock = clock, clock_next

    delta_short = abs(float(clock[probe]) - base)
    threshold_early = 1e-8
    
    print(f"  Probe change: |ΔN| = {delta_short:.3e} (threshold: {threshold_early:.3e})")

    if delta_short > threshold_early:
        print(f"❌ FAIL: Probe changed too early (non-causal / too-fast coupling)")
        print(f"  ΔN = {delta_short:.3e} > {threshold_early:.3e}")
        return False
    
    print("  ✔ Probe unchanged (causal propagation confirmed)")

    # Long time: after c_g * t exceeds r_probe, a response should be detectable.
    steps_long = 2000  # additional t=40.0 => total t=44.0 > r_probe
    t_total = (steps_short + steps_long) * dt
    
    print(f"\nLate-time test (response check):")
    print(f"  Running {steps_long} additional steps (total t={t_total:.2f})")
    print(f"  Expected travel distance: c_g * t = {c_g * t_total:.2f} > r_probe = {r_probe}")
    
    for i in range(steps_long):
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
        
        if i % 500 == 0:
            assert np.isfinite(clock_next).all(), f"Non-finite values at step {steps_short + i}"
        
        clock_prev, clock = clock, clock_next

    delta_long = abs(float(clock[probe]) - base)
    threshold_late = 1e-8
    
    print(f"  Probe change: |ΔN| = {delta_long:.3e} (threshold: {threshold_late:.3e})")

    if delta_long < threshold_late:
        print(f"❌ FAIL: Probe did not respond after sufficient time")
        print(f"  ΔN = {delta_long:.3e} < {threshold_late:.3e}")
        return False
    
    print("  ✔ Probe responded (propagation detected)")

    # Sanity: remain within clip and finite.
    print(f"\nFinal sanity checks:")
    if not np.isfinite(clock).all():
        print("❌ FAIL: NaN/Inf in clock_rate field")
        return False
    print("  ✔ All values finite")
    
    clock_min = float(np.min(clock))
    clock_max = float(np.max(clock))
    
    if clock_min < clip[0] - 1e-12 or clock_max > clip[1] + 1e-12:
        print(f"❌ FAIL: clock_rate violated clip bounds")
        print(f"  Range: [{clock_min:.6f}, {clock_max:.6f}], Clip: {clip}")
        return False
    print(f"  ✔ Within clip bounds: [{clock_min:.6f}, {clock_max:.6f}]")

    print("\n✅ PASS: clock_rate dynamics are local and causal")
    return True