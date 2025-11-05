"""Analyze FK reach map workspace statistics."""
import scipy.io as sio
import numpy as np

mat = sio.loadmat('matlab/exports/reach_surface.mat')
print(f"Available keys: {mat.keys()}")

# Extract surface structure
surface = mat['surface'][0, 0]
print(f"\nSurface fields: {surface.dtype.names}")

# Get vertices
pts = surface['vertices']
print(f"\nVertices shape: {pts.shape}")
radii = np.linalg.norm(pts[:, :2], axis=1)

print(f"FK Reach Map Statistics:")
print(f"========================")
print(f"Total points: {len(pts)}")
print(f"Median radius: {np.median(radii):.3f} m")
print(f"P5/P25/P75/P95: {np.percentile(radii, [5, 25, 75, 95])}")
print(f"\nDistribution:")
print(f"  <0.6m: {100*np.mean(radii<0.6):.1f}%")
print(f"  <0.7m: {100*np.mean(radii<0.7):.1f}%")
print(f"  <0.9m: {100*np.mean(radii<0.9):.1f}%")
