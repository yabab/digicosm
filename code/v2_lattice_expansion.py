import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_3d_points(center, n, r, jitter_r, jitter_angle):
    """
    Generates n points on a 3D sphere using a Fibonacci Lattice
    plus random jitter and merging logic.
    """
    cx, cy, cz = center
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    new_points = set()
    
    for i in range(n):
        # 1. Fibonacci Lattice (Spherical Golden Mean)
        # Distributes n points evenly over a sphere
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y) # radius at y
        theta = 2 * np.pi * i / phi      # golden angle increment
        
        # 2. Add Jitter
        y += np.random.uniform(-jitter_r, jitter_r)
        theta += np.random.uniform(-jitter_angle, jitter_angle)
        current_r = r + np.random.uniform(-jitter_r, jitter_r)
        
        # 3. Convert Spherical to Cartesian
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        
        px = cx + x * current_r
        py = cy + y * current_r
        pz = cz + z * current_r
        
        # 4. Merge Logic (Rounding to 'Voxel' size)
        # This acts as our Planck Length/Resolution limit
        new_points.add((round(px, 2), round(py, 2), round(pz, 2)))
        
    return new_points

# --- Simulation Parameters ---
STEPS = 5
POINTS_PER_EXPANSION = 8
BASE_RADIUS = 1.0
JITTER_RES = 0.15 # Randomness factor

# Starting point (The Big Bang)
universe_points = {(0.0, 0.0, 0.0)}

print(f"Generating 3D Universe...")

for step in range(STEPS):
    current_step_points = list(universe_points)
    new_generation = set()
    
    for p in current_step_points:
        spawned = generate_3d_points(p, POINTS_PER_EXPANSION, BASE_RADIUS, JITTER_RES, JITTER_RES)
        new_generation.update(spawned)
    
    universe_points.update(new_generation)
    print(f"Step {step+1}: {len(universe_points)} unique points in spacetime.")

# --- 3D Visualization ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Convert set to arrays for plotting
pts = np.array(list(universe_points))
x, y, z = pts[:,0], pts[:,1], pts[:,2]

# Color points by their distance from origin (Time/Age)
dist = np.sqrt(x**2 + y**2 + z**2)
scatter = ax.scatter(x, y, z, c=dist, cmap='magma', s=2, alpha=0.6)

ax.set_title(f"3D Discrete Lattice Expansion ({STEPS} Steps)")
ax.set_axis_off() # Hide grid for better "Space" feel
plt.colorbar(scatter, label='Distance from Origin (Temporal Age)')
plt.show()