import numpy as np
import trimesh
from plyfile import PlyData, PlyElement
from pathlib import Path
from typing import Optional
import struct

class PointCloud:
    def __init__(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        instance_ids: Optional[np.ndarray] = None,
        face_indices: Optional[np.ndarray] = None,
    ):
        self.points = points.astype(np.float32)
        self.colors = colors.astype(np.uint8) if colors is not None else None
        self.labels = labels.astype(np.int32) if labels is not None else np.zeros(len(points), dtype=np.int32)
        self.instance_ids = instance_ids.astype(np.int32) if instance_ids is not None else np.zeros(len(points), dtype=np.int32)
        self.face_indices = face_indices.astype(np.int32) if face_indices is not None else None

    def __len__(self):
        return len(self.points)


def load_glb(path: Path, num_samples: int = 500000) -> tuple['PointCloud', trimesh.Trimesh]:
    """Load GLB mesh and sample points from surface.

    Returns:
        tuple: (PointCloud with face_indices, original mesh for face extraction)
    """
    mesh = trimesh.load(str(path), force='mesh')

    if isinstance(mesh, trimesh.Scene):
        # Combine all meshes in scene
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError("No valid meshes found in GLB file")

    # Sample points from surface
    points, face_indices = mesh.sample(num_samples, return_index=True)

    # Get colors based on visual type
    colors = None
    if mesh.visual.kind == 'vertex':
        # Interpolate vertex colors
        colors = mesh.visual.vertex_colors[mesh.faces[face_indices]].mean(axis=1)[:, :3]
    elif mesh.visual.kind == 'face':
        colors = mesh.visual.face_colors[face_indices][:, :3]
    elif mesh.visual.kind == 'texture':
        # Sample colors from texture using UV coordinates
        try:
            uv = mesh.visual.uv
            texture = None

            # Try to get texture from PBRMaterial
            if hasattr(mesh.visual.material, 'baseColorTexture'):
                texture = mesh.visual.material.baseColorTexture
            elif hasattr(mesh.visual.material, 'image'):
                texture = mesh.visual.material.image

            if texture is not None and uv is not None:
                tex_array = np.array(texture)
                h, w = tex_array.shape[:2]

                # Get face vertex indices for sampled points
                face_vertices = mesh.faces[face_indices]

                # Compute barycentric coordinates for the sampled points
                triangles = mesh.triangles[face_indices]
                bary = trimesh.triangles.points_to_barycentric(triangles, points)
                bary = np.clip(bary, 0.0, 1.0)

                # Interpolate UV coordinates
                uv_sampled = (
                    uv[face_vertices[:, 0]] * bary[:, 0:1] +
                    uv[face_vertices[:, 1]] * bary[:, 1:2] +
                    uv[face_vertices[:, 2]] * bary[:, 2:3]
                )

                # Convert UV to pixel coordinates (UV is 0-1, y is flipped)
                px = np.clip((uv_sampled[:, 0] * (w - 1)).astype(int), 0, w - 1)
                py = np.clip(((1 - uv_sampled[:, 1]) * (h - 1)).astype(int), 0, h - 1)

                # Sample colors from texture
                colors = tex_array[py, px, :3]
        except Exception as e:
            print(f"Texture sampling failed: {e}")

    if colors is None:
        colors = np.full((len(points), 3), 128, dtype=np.uint8)

    return PointCloud(points=points, colors=colors.astype(np.uint8), face_indices=face_indices), mesh


def load_ply(path: Path) -> tuple['PointCloud', None]:
    """Load PLY file with optional labels and instance_ids.

    Returns:
        tuple: (PointCloud, None) - PLY has no mesh to extract from
    """
    plydata = PlyData.read(str(path))
    vertex = plydata['vertex']

    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

    # Colors
    colors = None
    if 'red' in vertex.data.dtype.names:
        colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T

    # Labels
    labels = None
    if 'label' in vertex.data.dtype.names:
        labels = np.array(vertex['label'])

    # Instance IDs
    instance_ids = None
    if 'instance_id' in vertex.data.dtype.names:
        instance_ids = np.array(vertex['instance_id'])

    return PointCloud(points=points, colors=colors, labels=labels, instance_ids=instance_ids), None


def extract_faces_from_mesh(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    selected_point_indices: np.ndarray,
    padding: float = 0.0
) -> trimesh.Trimesh:
    """Extract original mesh faces within the bounding box of selected points.

    Args:
        mesh: Original source mesh
        points: All sampled points (used to compute bounding box)
        selected_point_indices: Indices of selected points
        padding: Extra padding around bounding box (in mesh units)

    Returns:
        Submesh containing all faces within the bounding box
    """
    selected_points = points[selected_point_indices]

    # Compute bounding box of selected points
    bbox_min = selected_points.min(axis=0) - padding
    bbox_max = selected_points.max(axis=0) + padding

    # Get face centroids and find faces within bounding box
    face_centroids = mesh.triangles_center

    in_bbox = np.all(
        (face_centroids >= bbox_min) & (face_centroids <= bbox_max),
        axis=1
    )

    selected_faces = np.where(in_bbox)[0]

    if len(selected_faces) == 0:
        raise ValueError("No mesh faces found within selected region")

    return mesh.submesh([selected_faces], append=True)


def save_ply(path: Path, pc: PointCloud) -> int:
    """Save point cloud to PLY with labels and instance_ids."""
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('label', 'i4'), ('instance_id', 'i4'),
    ]

    data = np.zeros(len(pc), dtype=dtype)
    data['x'] = pc.points[:, 0]
    data['y'] = pc.points[:, 1]
    data['z'] = pc.points[:, 2]

    if pc.colors is not None:
        data['red'] = pc.colors[:, 0]
        data['green'] = pc.colors[:, 1]
        data['blue'] = pc.colors[:, 2]
    else:
        data['red'] = data['green'] = data['blue'] = 128

    data['label'] = pc.labels
    data['instance_id'] = pc.instance_ids

    vertex = PlyElement.describe(data, 'vertex')
    PlyData([vertex], text=False).write(str(path))

    return len(pc)
