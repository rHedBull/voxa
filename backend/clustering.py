import numpy as np
from scipy.spatial import KDTree
from typing import List


def region_grow(
    points: np.ndarray,
    seed_index: int,
    normal_threshold_deg: float = 15.0,
    distance_threshold: float = 0.05,
    max_points: int = 50000,
    normals: np.ndarray | None = None,
) -> List[int]:
    """
    Region growing from a seed point based on normal similarity and distance.

    Args:
        points: Nx3 point array
        seed_index: Starting point index
        normal_threshold_deg: Max angle between normals (degrees)
        distance_threshold: Max distance between neighboring points
        max_points: Safety limit
        normals: Nx3 normal array (computed if not provided)

    Returns:
        List of point indices in the grown region
    """
    n_points = len(points)

    # Compute normals if not provided
    if normals is None:
        normals = estimate_normals(points)

    # Build KD-tree for neighbor search
    tree = KDTree(points)

    # Region growing
    normal_threshold = np.cos(np.radians(normal_threshold_deg))

    visited = np.zeros(n_points, dtype=bool)
    region = [seed_index]
    visited[seed_index] = True

    seed_normal = normals[seed_index]

    queue = [seed_index]

    while queue and len(region) < max_points:
        current = queue.pop(0)
        current_point = points[current]

        # Find neighbors within distance threshold
        neighbor_indices = tree.query_ball_point(current_point, distance_threshold)

        for neighbor in neighbor_indices:
            if visited[neighbor]:
                continue

            # Check normal similarity with seed
            neighbor_normal = normals[neighbor]
            dot_product = np.abs(np.dot(seed_normal, neighbor_normal))

            if dot_product >= normal_threshold:
                visited[neighbor] = True
                region.append(neighbor)
                queue.append(neighbor)

    return region


def estimate_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
    """Estimate normals using PCA on local neighborhoods."""
    tree = KDTree(points)
    normals = np.zeros_like(points)

    for i in range(len(points)):
        # Find k nearest neighbors
        _, indices = tree.query(points[i], k=min(k, len(points)))

        # PCA to find normal
        neighbors = points[indices]
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.dot(centered.T, centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Normal is eigenvector with smallest eigenvalue
        normals[i] = eigenvectors[:, 0]

    return normals
