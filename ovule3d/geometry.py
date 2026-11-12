"""Unsmoothed cell meshes with explicit coordinates, units and geometry checks.

Geometry is defined on the union of labelled voxels. Mesh volume and voxel
volume are distinct estimators; mesh sphericity uses mesh volume consistently.
Equivalent-ellipsoid axes come from the covariance of continuous solid voxels,
not the maximum extents of the object. No physical calibration is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes, mesh_surface_area


@dataclass
class CellMesh:
    vertices_xyz: np.ndarray
    faces: np.ndarray
    measures: dict


def boundary_ids(labels, valid):
    """IDs touching an image face or a face-adjacent invalid annotation voxel."""
    valid = np.asarray(valid, dtype=bool)
    if labels.shape != valid.shape or labels.ndim != 3:
        raise ValueError("Equal 3D label and valid-mask shapes required")
    edge = valid & ~ndi.binary_erosion(valid, structure=ndi.generate_binary_structure(3, 1), border_value=0)
    return set(int(x) for x in np.unique(labels[edge]) if x > 0)


def mesh_from_instance(labels, instance_id, *, spacing_zyx=(1, 1, 1), origin_zyx=(0, 0, 0),
                       spatial_unit="voxel", bounds=None):
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer) or instance_id <= 0:
        raise ValueError("3D integer labels and a positive instance ID required")
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    origin = np.asarray(origin_zyx, dtype=np.float64)
    if spacing.shape != (3,) or origin.shape != (3,) or not np.isfinite(spacing).all() or not np.isfinite(origin).all() or (spacing <= 0).any():
        raise ValueError("Finite ZYX origin and positive spacing required")
    if spatial_unit not in ("voxel", "um"):
        raise ValueError("Unit must be explicit: voxel or um")
    if spatial_unit == "voxel" and not np.array_equal(spacing, np.ones(3)):
        raise ValueError("Uncalibrated voxel-space models must use unit spacing")
    if bounds is None:
        where = np.argwhere(labels == instance_id)
        if not len(where):
            raise ValueError("Instance absent")
        bounds = tuple(slice(int(a), int(b) + 1) for a, b in zip(where.min(0), where.max(0)))
    local = (labels[bounds] == instance_id).astype(np.uint8)
    points = np.argwhere(local).astype(np.float64)
    if not len(points):
        raise ValueError("Instance absent from supplied bounds")
    lower = np.array([sl.start or 0 for sl in bounds], dtype=np.float64)
    vertices_zyx, faces, _, _ = marching_cubes(np.pad(local, 1), level=0.5,
                                              spacing=tuple(spacing), allow_degenerate=False)
    vertices_zyx = vertices_zyx.astype(np.float64) + (lower - 1 + origin) * spacing
    # Reversing ZYX into XYZ has determinant -1; restore face winding.
    vertices = vertices_zyx[:, ::-1].copy()
    faces = faces[:, [0, 2, 1]].astype(np.int32)
    centre = vertices.mean(0)
    triangles = vertices[faces] - centre
    signed_volume = float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6)
    if signed_volume < 0:
        faces = faces[:, [0, 2, 1]].copy()
    mesh_volume = abs(signed_volume)
    area = float(mesh_surface_area(vertices, faces))
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, edge_counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    watertight = bool(np.all(edge_counts == 2))
    weighted = points * spacing
    centred = weighted - weighted.mean(0)
    covariance = centred.T @ centred / len(points) + np.diag(spacing**2 / 12)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    full_axes = np.sqrt(20 * np.maximum(eigenvalues, 0))
    _, components = ndi.label(local, structure=ndi.generate_binary_structure(3, 1))
    centroid = (points.mean(0) + lower + origin) * spacing
    measures = {"instance_id": int(instance_id), "spatial_unit": spatial_unit,
                "coordinate_frame": "prepared volume voxel centres, exported XYZ; spacing applied after origin offset",
                "spacing_zyx": spacing.tolist(), "origin_zyx_voxels": origin.tolist(),
                "voxel_count": int(len(points)), "voxel_volume": float(len(points) * np.prod(spacing)),
                "mesh_volume": mesh_volume, "mesh_surface_area": area,
                "mesh_sphericity": float((36 * np.pi * mesh_volume**2)**(1/3) / area) if area else None,
                "centroid_xyz": centroid[::-1].tolist(), "equivalent_ellipsoid_full_axes": full_axes.tolist(),
                "axis_definition": "sqrt(20*eigenvalues(covariance of uniform solid labelled voxels)); not bounding diameters",
                "volume_conventions": "voxel_volume=count*spacing product; mesh_volume=closed oriented triangle integral",
                "vertices": len(vertices), "triangles": len(faces), "watertight": watertight,
                "nonmanifold_or_open_edges": int(np.count_nonzero(edge_counts != 2)),
                "six_connected_components": int(components), "smoothing": "none",
                "surface_method": "Lewiner marching cubes at 0.5; one-voxel zero padding; no degenerate triangles"}
    return CellMesh(vertices, faces, measures)


def export_mesh(mesh: CellMesh, path, *, file_format=None):
    """Export XYZ mesh; a sidecar remains authoritative for units and identity."""
    path = Path(path)
    fmt = (file_format or path.suffix.lstrip(".")).lower()
    if fmt not in ("ply", "obj", "stl"):
        raise ValueError("Supported formats: ply, obj, stl")
    path.parent.mkdir(parents=True, exist_ok=True)
    unit = mesh.measures["spatial_unit"]
    if fmt == "obj":
        with path.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(f"# Ovule3D XYZ; spatial_unit={unit}; unsmoothed; see JSON sidecar\n")
            for vertex in mesh.vertices_xyz:
                stream.write("v " + " ".join(f"{x:.12g}" for x in vertex) + "\n")
            for face in mesh.faces:
                stream.write("f " + " ".join(str(int(x) + 1) for x in face) + "\n")
    elif fmt == "ply":
        header = (f"ply\nformat binary_little_endian 1.0\ncomment Ovule3D unit={unit}; XYZ; unsmoothed\n"
                  f"element vertex {len(mesh.vertices_xyz)}\nproperty double x\nproperty double y\nproperty double z\n"
                  f"element face {len(mesh.faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        packed = np.empty(len(mesh.faces), dtype=np.dtype([("count", "u1"), ("indices", "<i4", (3,))]))
        packed["count"] = 3
        packed["indices"] = mesh.faces
        with path.open("xb") as stream:
            stream.write(header.encode("ascii"))
            stream.write(mesh.vertices_xyz.astype("<f8").tobytes())
            stream.write(packed.tobytes())
    else:
        triangles = mesh.vertices_xyz[mesh.faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        norm = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(norm[:, None], np.finfo(float).tiny)
        packed = np.zeros(len(mesh.faces), dtype=np.dtype([("normal", "<f4", (3,)),
                                                        ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]))
        packed["normal"], packed["vertices"] = normals, triangles
        with path.open("xb") as stream:
            stream.write(f"Ovule3D XYZ unit={unit}; see JSON sidecar".encode("ascii").ljust(80, b" ")[:80])
            stream.write(struct.pack("<I", len(mesh.faces)))
            stream.write(packed.tobytes())
    return path
