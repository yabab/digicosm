import numpy as np
from model import step


def detect_front_radial(intensity, center, rmin=5, threshold=1e-6, angles=32):
    N = intensity.shape[0]
    cy, cx = center
    for r in range(rmin, N//2):
        shell = []
        for angle in np.linspace(0, 2*np.pi, angles, endpoint=False):
            y = int(cy + r*np.sin(angle))
            x = int(cx + r*np.cos(angle))
            if 0 <= y < N and 0 <= x < N:
                shell.append(intensity[y, x])
        if len(shell) and np.max(shell) > threshold:
            return r
    return None


def test_c_scaling():
    print("TEST 5: Speed of light scaling")

    speeds = []
    for c in [0.5, 1.0, 2.0]:
        N, steps, dt = 150, 200, 0.05
        center = (N//2, N//2)
        phi = np.zeros((N, N))
        pi = np.zeros_like(phi)
        phi[center[0], center[1]] = 1.0

        fronts = []
        for t in range(steps):
            phi, pi = step(phi, pi, dt, c, 0.0)
            if t % 10 == 0:
                r = detect_front_radial(phi**2, center)
                if r:
                    fronts.append((t*dt, r))

        if len(fronts) < 3:
            print(f"Insufficient fronts for c={c}")
            return False

        times = np.array([t for t, r in fronts])
        radii = np.array([r for t, r in fronts])
        v = np.polyfit(times, radii, 1)[0]
        speeds.append(v)

    ratios = np.array(speeds) / speeds[1]
    print("speed ratios:", ratios)
    ok = np.allclose(ratios, [0.5, 1.0, 2.0], rtol=0.3)
    if ok:
        print("✅ PASS: c-scaling roughly matches")
    else:
        print("❌ FAIL: c-scaling mismatch")
    return ok


if __name__ == '__main__':
    test_c_scaling()
