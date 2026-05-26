"""
RANSAC-based primitive fitting for cylinders and boxes.

This module provides functions to fit geometric primitives to point cloud data
using RANSAC (Random Sample Consensus) algorithms.
"""

import numpy as np
from scipy.spatial import KDTree
from typing import List, Dict, Any, Optional, Tuple


def fit_cylinder_ransac(
    points: np.ndarray,
    max_iterations: int = 1000,
    inlier_threshold: float = 0.02,
    min_inliers: int = 100,
) -> Optional[Dict[str, Any]]:
    """
    Fit a single cylinder to points using RANSAC.

    Algorithm:
    1. Sample 3 random points
    2. Use first 2 points to estimate cylinder axis direction
    3. Use 3rd point to estimate radius
    4. Count inliers (points within threshold distance from cylinder surface)
    5. Keep best fit across iterations

    Args:
        points: Nx3 array of point coordinates
        max_iterations: Maximum RANSAC iterations
        inlier_threshold: Distance threshold for inlier classification
        min_inliers: Minimum inliers required for valid fit

    Returns:
        Dict with 'center', 'axis', 'radius', 'height', 'inlier_indices'
        or None if no valid fit found
    """
    n_points = len(points)

    if n_points < 3:
        return None

    best_fit = None
    best_inlier_count = 0

    for _ in range(max_iterations):
        # Sample 3 random points
        sample_indices = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[sample_indices]

        # Estimate axis from p1 and p2
        axis = p2 - p1
        axis_length = np.linalg.norm(axis)

        if axis_length < 1e-6:
            continue

        axis = axis / axis_length

        # Project p3 onto the line through p1 with direction axis
        # to find the closest point on axis to p3
        v = p3 - p1
        t = np.dot(v, axis)
        closest_on_axis = p1 + t * axis

        # Radius is distance from p3 to axis
        radius = np.linalg.norm(p3 - closest_on_axis)

        if radius < 1e-6:
            continue

        # Count inliers: points whose distance to cylinder surface is within threshold
        inlier_mask, _ = _cylinder_inliers(
            points, p1, axis, radius, inlier_threshold
        )
        inlier_count = np.sum(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            inlier_indices = np.where(inlier_mask)[0]

            # Compute cylinder center and height from inliers
            inlier_points = points[inlier_indices]
            projections = np.dot(inlier_points - p1, axis)

            t_min, t_max = projections.min(), projections.max()
            height = t_max - t_min
            center = p1 + ((t_min + t_max) / 2) * axis

            best_fit = {
                'center': center.tolist(),
                'axis': axis.tolist(),
                'radius': float(radius),
                'height': float(height),
                'inlier_indices': inlier_indices.tolist(),
                'inlier_count': int(inlier_count),
            }

    if best_fit is None or best_fit['inlier_count'] < min_inliers:
        return None

    return best_fit


def _cylinder_inliers(
    points: np.ndarray,
    point_on_axis: np.ndarray,
    axis: np.ndarray,
    radius: float,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find points within threshold distance of cylinder surface.

    Returns:
        Tuple of (inlier_mask, distances_to_surface)
    """
    # Vector from axis point to each point
    v = points - point_on_axis

    # Project onto axis
    projections = np.dot(v, axis)

    # Perpendicular distance to axis
    perp = v - np.outer(projections, axis)
    dist_to_axis = np.linalg.norm(perp, axis=1)

    # Distance to cylinder surface
    dist_to_surface = np.abs(dist_to_axis - radius)

    inlier_mask = dist_to_surface <= threshold

    return inlier_mask, dist_to_surface


def fit_cylinders_in_region(
    points: np.ndarray,
    region_center: np.ndarray,
    region_radius: float,
    region_height: float,
    region_axis: np.ndarray = None,
    max_cylinders: int = 5,
    max_iterations: int = 500,
    inlier_threshold: float = 0.02,
    min_inliers: int = 50,
    min_inlier_ratio: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Find all cylinders in a cylindrical search region.

    After finding each cylinder, its inliers are removed and the search continues
    until no more valid cylinders are found or max_cylinders is reached.

    Args:
        points: Nx3 array of all point coordinates
        region_center: Center of cylindrical search region
        region_radius: Radius of search region
        region_height: Height of search region
        region_axis: Axis direction of search region (default: z-axis)
        max_cylinders: Maximum number of cylinders to find
        max_iterations: RANSAC iterations per cylinder
        inlier_threshold: Distance threshold for inlier classification
        min_inliers: Minimum inliers for valid cylinder
        min_inlier_ratio: Minimum ratio of inliers to remaining points

    Returns:
        List of cylinder dicts, each with global point indices
    """
    if region_axis is None:
        region_axis = np.array([0.0, 0.0, 1.0])
    else:
        region_axis = np.array(region_axis)
        region_axis = region_axis / np.linalg.norm(region_axis)

    region_center = np.array(region_center)

    # Find points within the cylindrical search region
    v = points - region_center
    projections = np.dot(v, region_axis)
    perp = v - np.outer(projections, region_axis)
    dist_to_axis = np.linalg.norm(perp, axis=1)

    in_region = (
        (dist_to_axis <= region_radius) &
        (np.abs(projections) <= region_height / 2)
    )

    region_indices = np.where(in_region)[0]

    if len(region_indices) < min_inliers:
        return []

    # Work with regional points
    regional_points = points[region_indices].copy()
    available_mask = np.ones(len(regional_points), dtype=bool)

    cylinders = []

    for _ in range(max_cylinders):
        available_indices = np.where(available_mask)[0]

        if len(available_indices) < min_inliers:
            break

        available_points = regional_points[available_indices]

        # Fit cylinder to available points
        fit = fit_cylinder_ransac(
            available_points,
            max_iterations=max_iterations,
            inlier_threshold=inlier_threshold,
            min_inliers=min_inliers,
        )

        if fit is None:
            break

        # Check inlier ratio
        if fit['inlier_count'] / len(available_points) < min_inlier_ratio:
            break

        # Map inlier indices back to global indices
        local_inliers = np.array(fit['inlier_indices'])
        regional_inliers = available_indices[local_inliers]
        global_inliers = region_indices[regional_inliers]

        fit['inlier_indices'] = global_inliers.tolist()
        cylinders.append(fit)

        # Remove inliers from available points
        available_mask[regional_inliers] = False

    return cylinders


def fit_box_pca(
    points: np.ndarray,
    inlier_threshold: float = 0.02,
) -> Optional[Dict[str, Any]]:
    """
    Fit an oriented bounding box using PCA.

    Args:
        points: Nx3 array of point coordinates
        inlier_threshold: Distance threshold for face inliers

    Returns:
        Dict with 'center', 'rotation' (3x3), 'dimensions', 'inlier_indices'
        or None if insufficient points
    """
    if len(points) < 4:
        return None

    # Compute centroid
    center = points.mean(axis=0)
    centered = points - center

    # PCA to find principal axes
    cov = np.dot(centered.T, centered) / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue descending (largest variance first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Project points onto principal axes
    projected = np.dot(centered, eigenvectors)

    # Find min/max along each axis
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)

    dimensions = maxs - mins

    # Adjust center to be at box center (not centroid)
    center_offset = (mins + maxs) / 2
    center = center + np.dot(center_offset, eigenvectors.T)

    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] = -eigenvectors[:, 2]

    # Find inliers (points near box faces)
    inlier_indices = _box_face_inliers(
        points, center, eigenvectors, dimensions, inlier_threshold
    )

    return {
        'center': center.tolist(),
        'rotation': eigenvectors.T.tolist(),  # 3x3 rotation matrix (rows are axes)
        'dimensions': dimensions.tolist(),  # [width, height, depth]
        'inlier_indices': inlier_indices.tolist(),
        'inlier_count': len(inlier_indices),
    }


def _box_face_inliers(
    points: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,  # 3x3, columns are axes
    dimensions: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Find points near box faces (within threshold of any face).

    Points must be within the box extents (with tolerance) along two axes
    and near a face along the third axis.
    """
    centered = points - center
    projected = np.dot(centered, axes)  # Nx3

    half_dims = dimensions / 2

    inlier_mask = np.zeros(len(points), dtype=bool)

    # Check each face pair (positive and negative along each axis)
    for axis_idx in range(3):
        other_axes = [i for i in range(3) if i != axis_idx]

        # Check if point is within box bounds along other axes
        within_bounds = np.ones(len(points), dtype=bool)
        for other_idx in other_axes:
            within_bounds &= np.abs(projected[:, other_idx]) <= (half_dims[other_idx] + threshold)

        # Check if point is near positive or negative face
        dist_to_pos_face = np.abs(projected[:, axis_idx] - half_dims[axis_idx])
        dist_to_neg_face = np.abs(projected[:, axis_idx] + half_dims[axis_idx])
        near_face = (dist_to_pos_face <= threshold) | (dist_to_neg_face <= threshold)

        inlier_mask |= (within_bounds & near_face)

    return np.where(inlier_mask)[0]


def fit_box_ransac(
    points: np.ndarray,
    max_iterations: int = 500,
    inlier_threshold: float = 0.02,
    min_inliers: int = 50,
) -> Optional[Dict[str, Any]]:
    """
    Fit an oriented bounding box using RANSAC with PCA.

    Samples random subsets of points and uses PCA to fit boxes,
    keeping the one with most face inliers.

    Args:
        points: Nx3 array of point coordinates
        max_iterations: Maximum RANSAC iterations
        inlier_threshold: Distance threshold for face inliers
        min_inliers: Minimum face inliers for valid fit

    Returns:
        Dict with box parameters or None
    """
    n_points = len(points)

    if n_points < 6:
        return None

    best_fit = None
    best_inlier_count = 0

    # Sample size for initial PCA
    sample_size = min(max(20, n_points // 10), n_points)

    for _ in range(max_iterations):
        # Sample random subset
        sample_indices = np.random.choice(n_points, sample_size, replace=False)
        sample_points = points[sample_indices]

        # Fit box to sample using PCA
        fit = fit_box_pca(sample_points, inlier_threshold)

        if fit is None:
            continue

        # Count inliers on full point set
        inlier_indices = _box_face_inliers(
            points,
            np.array(fit['center']),
            np.array(fit['rotation']).T,  # Convert back to columns
            np.array(fit['dimensions']),
            inlier_threshold,
        )

        inlier_count = len(inlier_indices)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count

            # Refit using all inliers for better estimate
            if inlier_count >= min_inliers:
                refined_fit = fit_box_pca(points[inlier_indices], inlier_threshold)
                if refined_fit:
                    # Recount inliers with refined fit
                    refined_inliers = _box_face_inliers(
                        points,
                        np.array(refined_fit['center']),
                        np.array(refined_fit['rotation']).T,
                        np.array(refined_fit['dimensions']),
                        inlier_threshold,
                    )
                    refined_fit['inlier_indices'] = refined_inliers.tolist()
                    refined_fit['inlier_count'] = len(refined_inliers)
                    best_fit = refined_fit
                else:
                    fit['inlier_indices'] = inlier_indices.tolist()
                    fit['inlier_count'] = inlier_count
                    best_fit = fit

    if best_fit is None or best_fit['inlier_count'] < min_inliers:
        return None

    return best_fit


def fit_boxes_in_region(
    points: np.ndarray,
    region_center: np.ndarray,
    region_dimensions: np.ndarray,
    region_rotation: np.ndarray = None,
    max_boxes: int = 5,
    max_iterations: int = 300,
    inlier_threshold: float = 0.02,
    min_inliers: int = 50,
    min_inlier_ratio: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Find all boxes in a box-shaped search region.

    After finding each box, its inliers are removed and the search continues
    until no more valid boxes are found or max_boxes is reached.

    Args:
        points: Nx3 array of all point coordinates
        region_center: Center of box-shaped search region
        region_dimensions: [width, height, depth] of search region
        region_rotation: 3x3 rotation matrix (default: identity)
        max_boxes: Maximum number of boxes to find
        max_iterations: RANSAC iterations per box
        inlier_threshold: Distance threshold for face inliers
        min_inliers: Minimum inliers for valid box
        min_inlier_ratio: Minimum ratio of inliers to remaining points

    Returns:
        List of box dicts, each with global point indices
    """
    region_center = np.array(region_center)
    region_dimensions = np.array(region_dimensions)

    if region_rotation is None:
        region_rotation = np.eye(3)
    else:
        region_rotation = np.array(region_rotation)

    # Find points within the box-shaped search region
    centered = points - region_center
    projected = np.dot(centered, region_rotation.T)  # Project to local coords

    half_dims = region_dimensions / 2
    in_region = np.all(np.abs(projected) <= half_dims, axis=1)

    region_indices = np.where(in_region)[0]

    if len(region_indices) < min_inliers:
        return []

    # Work with regional points
    regional_points = points[region_indices].copy()
    available_mask = np.ones(len(regional_points), dtype=bool)

    boxes = []

    for _ in range(max_boxes):
        available_indices = np.where(available_mask)[0]

        if len(available_indices) < min_inliers:
            break

        available_points = regional_points[available_indices]

        # Fit box to available points
        fit = fit_box_ransac(
            available_points,
            max_iterations=max_iterations,
            inlier_threshold=inlier_threshold,
            min_inliers=min_inliers,
        )

        if fit is None:
            break

        # Check inlier ratio
        if fit['inlier_count'] / len(available_points) < min_inlier_ratio:
            break

        # Map inlier indices back to global indices
        local_inliers = np.array(fit['inlier_indices'])
        regional_inliers = available_indices[local_inliers]
        global_inliers = region_indices[regional_inliers]

        fit['inlier_indices'] = global_inliers.tolist()
        boxes.append(fit)

        # Remove inliers from available points
        available_mask[regional_inliers] = False

    return boxes


def refine_cylinder_fit(
    points: np.ndarray,
    initial_fit: Dict[str, Any],
    iterations: int = 3,
) -> Dict[str, Any]:
    """
    Refine a cylinder fit using iterative least-squares on inliers.

    Args:
        points: Nx3 array of all point coordinates
        initial_fit: Initial cylinder fit dict
        iterations: Number of refinement iterations

    Returns:
        Refined cylinder fit dict
    """
    fit = initial_fit.copy()
    inlier_indices = np.array(fit['inlier_indices'])

    if len(inlier_indices) < 3:
        return fit

    for _ in range(iterations):
        inlier_points = points[inlier_indices]

        # Refit using PCA on inliers to find axis
        center = inlier_points.mean(axis=0)
        centered = inlier_points - center
        cov = np.dot(centered.T, centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Axis is direction of largest variance
        axis = eigenvectors[:, eigenvalues.argmax()]

        # Compute radius as mean distance to axis
        v = inlier_points - center
        projections = np.dot(v, axis)
        perp = v - np.outer(projections, axis)
        distances_to_axis = np.linalg.norm(perp, axis=1)
        radius = np.mean(distances_to_axis)

        # Compute height
        t_min, t_max = projections.min(), projections.max()
        height = t_max - t_min
        center = center + ((t_min + t_max) / 2) * axis

        fit['center'] = center.tolist()
        fit['axis'] = axis.tolist()
        fit['radius'] = float(radius)
        fit['height'] = float(height)

    return fit
