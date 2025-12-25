import numpy as np
from code.model import run_driven_phase_wave
from code.measurements import detect_from_buffer

# ============================================================
# TEST 2 — Two-Source Interference (CORRECT)
# ============================================================

def test_two_source_interference_relativistic():
    """Test two-source interference pattern formation.
    
    Validates that two coherent sources produce visible interference fringes
    by detecting multiple peaks in the interference pattern.
    """
    print("\nTEST 2: Phase Dynamics Two-Source Interference (Driven)")

    # Increased domain and longer run to produce many fringes
    N = 400
    steps = 1400
    dt = 0.02
    kappa = 1.0
    omega0 = 0.0
    drive_omega = 4.0
    center = N // 2

    # Input validation
    assert N > 0 and steps > 0, "Grid size and steps must be positive"
    assert dt > 0 and kappa > 0, "Time step and coupling must be positive"
    assert drive_omega > 0, "Drive frequency must be positive"

    psi0 = np.zeros((N, N), dtype=np.complex128)

    # Larger vertical separation and sources placed further left for longer propagation
    d = 80
    x_source = center - 160
    
    assert d > 0 and 0 <= x_source < N, "Source separation and position must be valid"
    
    # Use short vertical line sources (extended sources) to increase fringe contrast
    sources = []
    line_half = 20
    amp = 20.0
    
    assert amp > 0 and line_half >= 0, "Amplitude and line half-width must be non-negative"
    
    for dy in range(-line_half, line_half+1):
        y1 = center - d + dy
        y2 = center + d + dy
        assert 0 <= y1 < N and 0 <= y2 < N, f"Source positions out of bounds: ({y1}, {x_source}), ({y2}, {x_source})"
        sources.append((y1, x_source, amp))
        sources.append((y2, x_source, amp))

    assert len(sources) > 0, "No sources were created"

    # Single optimized attempt: run once with the extended sources
    # detection routine has been moved to `model.detect_from_buffer`

    # Run the optimized configuration once
    print(f"Info: N={N}, steps={steps}, dt={dt}, drive_omega={drive_omega}")
    print(f"  Source separation={d}, x_source={x_source}, num_sources={len(sources)}")
    
    psi_avg, buf = run_driven_phase_wave(
        psi0,
        steps,
        dt,
        kappa=kappa,
        omega0=omega0,
        drive_omega=drive_omega,
        sources=sources,
        avg_last=160,
        warmup_frac=0.5,
    )
    
    # Validate outputs
    assert np.isfinite(psi_avg).all(), "Non-finite values in averaged field"
    assert len(buf) > 0, "Buffer is empty after simulation"
    
    best = detect_from_buffer(buf)
    min_fringes = 3
    detected = best['count']
    
    print(f"Results: detected peaks = {detected}, params={best.get('params')}")
    print(f"  Minimum required fringes: {min_fringes}")
    
    if detected >= min_fringes:
        print(f"✅ PASS: Interference fringes detected ({detected} >= {min_fringes})")
        return True

    print(f"❌ FAIL: Insufficient fringes ({detected} < {min_fringes})")
    return False