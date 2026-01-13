import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class UniverseSimulator:
    def __init__(self, min_dist=0.6):
        self.min_dist = min_dist
        # Spatial Hash: keys are (ix, iy, iz), values are lists of points
        self.spatial_hash = {}
        self.points = []

    def _get_bucket(self, point):
        """Maps a 3D coordinate to a voxel grid coordinate."""
        return tuple((point / self.min_dist).astype(int))

    def is_valid(self, candidate):
        """Checks neighbors in the spatial hash for proximity."""
        bucket = self._get_bucket(candidate)
        
        # Check current bucket and all 26 surrounding neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_bucket = (bucket[0]+dx, bucket[1]+dy, bucket[2]+dz)
                    if neighbor_bucket in self.spatial_hash:
                        for other in self.spatial_hash[neighbor_bucket]:
                            if np.linalg.norm(candidate - other) < self.min_dist:
                                return False
        return True

    def add_point(self, point):
        self.points.append(point)
        bucket = self._get_bucket(point)
        if bucket not in self.spatial_hash:
            self.spatial_hash[bucket] = []
        self.spatial_hash[bucket].append(point)

    def run_expansion(self, steps=10, points_per_step=2000):
        # Initial point (Big Bang)
        self.add_point(np.array([0.0, 0.0, 0.0]))
        phi = (1 + np.sqrt(5)) / 2 # Golden Ratio

        for s in range(1, steps + 1):
            radius = s * 1.2
            # Generate shell points using Fibonacci Sphere
            indices = np.arange(points_per_step)
            phi_coords = np.arccos(1 - 2 * (indices + 0.5) / points_per_step)
            theta_coords = 2 * np.pi * indices / phi
            
            # Shell formation
            xs = radius * np.sin(phi_coords) * np.cos(theta_coords)
            ys = radius * np.sin(phi_coords) * np.sin(theta_coords)
            zs = radius * np.cos(phi_coords)
            
            candidates = np.stack([xs, ys, zs], axis=1)
            # Add thickness jitter so we fill the volume
            candidates += np.random.uniform(-self.min_dist, self.min_dist, candidates.shape)
            
            added_in_step = 0
            for cand in candidates:
                if self.is_valid(cand):
                    self.add_point(cand)
                    added_in_step += 1
            
            print(f"Step {s}: Added {added_in_step} points. Total Universe: {len(self.points)}")

# --- Execution ---
sim = UniverseSimulator(min_dist=0.5)
sim.run_expansion(steps=10, points_per_step=3000)

# --- Visualization ---
pts = np.array(sim.points)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

dists = np.linalg.norm(pts, axis=1)
scatter = ax.scatter(pts[:,0], pts[:,1], pts[:,2], 
                     c=dists, cmap='plasma', s=2, alpha=0.5)

ax.set_title("3D Spatial Hash Universe Expansion")
ax.set_axis_off()
plt.colorbar(scatter, label='Distance from Origin (Time)')
plt.show()