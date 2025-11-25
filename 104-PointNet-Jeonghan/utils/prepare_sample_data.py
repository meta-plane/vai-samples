#!/usr/bin/env python3
"""
Generate sample point cloud data for testing PointNet inference.

Creates various sample point clouds in the format expected by our implementation.
"""

import numpy as np
import argparse
from pathlib import Path


def generate_sphere(num_points=1024, radius=1.0):
    """Generate points uniformly distributed on a sphere."""
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(costheta)
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    return np.stack([x, y, z], axis=1)


def generate_cube(num_points=1024, size=1.0):
    """Generate points uniformly distributed in a cube."""
    return np.random.uniform(-size, size, (num_points, 3))


def generate_plane(num_points=1024, size=1.0):
    """Generate points on a plane (z=0)."""
    points = np.random.uniform(-size, size, (num_points, 3))
    points[:, 2] = 0  # Flatten to z=0
    return points


def generate_two_clusters(num_points=1024):
    """Generate two separate clusters of points."""
    half = num_points // 2
    
    # Cluster 1: centered at (-0.5, 0, 0)
    cluster1 = np.random.randn(half, 3) * 0.3
    cluster1[:, 0] -= 0.5
    
    # Cluster 2: centered at (0.5, 0, 0)
    cluster2 = np.random.randn(num_points - half, 3) * 0.3
    cluster2[:, 0] += 0.5
    
    return np.vstack([cluster1, cluster2])


def generate_chair_like(num_points=1024):
    """Generate a simple chair-like shape."""
    points = []
    
    # Seat (horizontal plane)
    seat_points = int(num_points * 0.3)
    seat = np.random.uniform([-0.3, -0.3, 0], [0.3, 0.3, 0.05], (seat_points, 3))
    points.append(seat)
    
    # Backrest (vertical plane)
    back_points = int(num_points * 0.3)
    back = np.random.uniform([-0.3, -0.3, 0], [0.3, -0.25, 0.5], (back_points, 3))
    points.append(back)
    
    # 4 Legs
    leg_points = int(num_points * 0.1)
    for x in [-0.25, 0.25]:
        for y in [-0.25, 0.25]:
            leg = np.random.uniform([x-0.02, y-0.02, -0.4], [x+0.02, y+0.02, 0], (leg_points, 3))
            points.append(leg)
    
    return np.vstack(points)[:num_points]


def save_point_cloud(points, filename):
    """Save point cloud to text file (x y z per line)."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(filename, points, fmt='%.6f', delimiter=' ')
    print(f"✓ Saved: {filename} ({len(points)} points)")


def main():
    parser = argparse.ArgumentParser(description='Generate sample point cloud data')
    parser.add_argument('--output_dir', type=str, default='assets/data',
                        help='Output directory for sample data')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points per cloud')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating sample point clouds ({args.num_points} points each)...\n")
    
    # Generate various shapes
    samples = {
        'sphere.txt': generate_sphere(args.num_points),
        'cube.txt': generate_cube(args.num_points),
        'plane.txt': generate_plane(args.num_points),
        'clusters.txt': generate_two_clusters(args.num_points),
        'chair.txt': generate_chair_like(args.num_points),
        'sample.txt': generate_sphere(args.num_points),  # Default sample
    }
    
    for filename, points in samples.items():
        save_point_cloud(points, output_dir / filename)
    
    print(f"\n✓ All samples generated in: {output_dir}")
    print("\nYou can visualize these files with MeshLab or CloudCompare.")


if __name__ == '__main__':
    main()

