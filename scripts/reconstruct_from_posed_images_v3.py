#!/usr/bin/env python3
"""
Reconstruct from Posed Images v3 (ARKit Analytical RDF Cubemap Pipeline)

A robust, modular 3D point cloud reconstruction tool combining:
1. Analytical Cubemap Pose & Slicing Engine (from scripts/extract_cubemap_faces.py):
   - Exact camera-to-world rotation from heading (OpenCV RDF: +X right, +Y down, +Z forward)
   - Analytical 6-face cubemap orientations (front, right, back, left, up, down)
   - Exact 90° FoV pinhole camera intrinsics K
   - Pure bilinear spherical face sampling
   - ARKit originPose heading convention verification
2. Scan Chain Trajectory Ordering:
   - Chronological sorting of panoramas via scanChain.index (Station 1 -> N)
3. MapAnything Model Estimation:
   - Slices perspective cubemap faces per station
   - Supports unposed relative estimation with Optical Nodal Center Lock (default)
     OR directly conditioned posed estimation (--use_pose_inputs)
   - Mixed-precision AMP (bf16/fp16) matching GPU capability
   - Memory-efficient minibatch inference and per-station VRAM cleanup
4. Interactive 3D Camera Pose Trajectory Markers:
   - Colored Fibonacci spheres at station centers
   - Horizontal orientation rings (XZ plane)
   - Sequential optical sight rays color-coded via TRAJECTORY_PALETTE
5. High-Performance Point Cloud Export:
   - Direct binary little-endian PLY export (pure NumPy, zero dependencies, 15 bytes/vertex)
   - Voxel downsampling (Open3D or pure NumPy grid fallback) and full dense mode (--dense)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repo root and scripts dir are on sys.path
repo_root = Path(__file__).resolve().parent.parent
scripts_dir = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import numpy as np
from PIL import Image

# Import cubemap slicing & analytical pose utilities from extract_cubemap_faces
try:
    from scripts.extract_cubemap_faces import (
        FACE_ORDER,
        IMAGE_EXTS,
        LOCAL_ROTATIONS,
        assert_rotation,
        bilinear_sample,
        check_heading_convention,
        column_major,
        face_pose,
        front_rotation,
        index_records,
        intrinsics as compute_intrinsics_dict,
        load_metadata,
        origin_xyz,
        sample_face,
    )
except ModuleNotFoundError:
    from extract_cubemap_faces import (
        FACE_ORDER,
        IMAGE_EXTS,
        LOCAL_ROTATIONS,
        assert_rotation,
        bilinear_sample,
        check_heading_convention,
        column_major,
        face_pose,
        front_rotation,
        index_records,
        intrinsics as compute_intrinsics_dict,
        load_metadata,
        origin_xyz,
        sample_face,
    )


# Color palette for sequentially color-coding camera stations along trajectory
TRAJECTORY_PALETTE = [
    (255, 0, 128),    # 0: Bright Magenta/Pink (Station 0/1)
    (0, 240, 255),    # 1: Bright Cyan
    (255, 220, 0),    # 2: Bright Gold/Yellow
    (0, 255, 128),    # 3: Bright Emerald Green
    (255, 128, 0),    # 4: Bright Orange
    (180, 0, 255),    # 5: Bright Purple
    (0, 160, 255),    # 6: Sky Blue
    (255, 50, 50),    # 7: Bright Coral Red
    (50, 255, 50),    # 8: Lime Green
    (255, 180, 200),  # 9: Soft Pink
    (128, 255, 220),  # 10: Mint Green
    (220, 160, 255),  # 11: Lavender
    (255, 255, 100),  # 12: Light Yellow
    (100, 220, 255),  # 13: Light Cyan
]


def natural_sort_key(s: Union[str, Path]) -> list:
    """Natural numerical sorting key (e.g. frame_2 comes before frame_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def get_dataset_name(input_path: Union[str, Path]) -> str:
    """Extract clean dataset directory name from input path or file."""
    p = Path(str(input_path).rstrip("/\\"))
    if p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() in [".json", ".ply", ".pcd"]:
        name = p.parent.name
    else:
        name = p.name
    return name if name else "reconstruction"


def resolve_output_path(
    output_spec: Union[str, Path],
    input_path: Union[str, Path],
    subfolder: Optional[str] = None,
    default_filename: str = "pointcloud.ply",
) -> Tuple[str, Path]:
    """Resolve final point cloud file path and target directory."""
    dataset_name = get_dataset_name(input_path)
    out_p = Path(str(output_spec).rstrip("/\\"))
    already_contains = dataset_name.lower() in [part.lower() for part in out_p.parts]

    if out_p.suffix.lower() in [".ply", ".pcd"]:
        fname = out_p.with_suffix(".ply").name
        parent_dir = out_p.parent
        target_dir = parent_dir if already_contains else (parent_dir / dataset_name)
        if subfolder:
            target_dir = target_dir / subfolder
        final_file = target_dir / fname
    else:
        target_dir = out_p if already_contains else (out_p / dataset_name)
        if subfolder:
            target_dir = target_dir / subfolder
        final_file = target_dir / default_filename

    return str(final_file), target_dir


def discover_inputs(
    input_path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[dict], Dict[str, Path], Path]:
    """
    Locate metadata records and match with equirectangular panorama images.
    Returns: (records, images_by_jobid, metadata_file)
    """
    in_p = Path(input_path)
    meta_p: Optional[Path] = Path(metadata_path) if metadata_path else None

    # Find metadata file
    if meta_p and meta_p.is_file():
        metadata_file = meta_p
    elif in_p.is_file() and in_p.suffix.lower() == ".json":
        metadata_file = in_p
    else:
        # Search for jobdata.json or any *.json in input directory
        search_dir = in_p if in_p.is_dir() else in_p.parent
        candidates = list(search_dir.glob("jobdata.json")) or list(search_dir.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No metadata JSON found in: {search_dir}")
        metadata_file = candidates[0]

    raw_records = load_metadata(metadata_file)
    by_id = index_records(raw_records)

    # Discover images
    images_dir = in_p if in_p.is_dir() else in_p.parent
    images_by_jobid: Dict[str, Path] = {}

    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            stem = path.stem
            if stem in by_id:
                images_by_jobid[stem] = path

    # If stem didn't match directly, try matching prefixes/substrings
    if len(images_by_jobid) < len(by_id):
        for path in sorted(images_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                for jid in by_id:
                    if jid not in images_by_jobid and (jid in path.name or path.stem in jid):
                        images_by_jobid[jid] = path

    return raw_records, images_by_jobid, metadata_file


def get_scan_chain_index(record: dict) -> int:
    """Extract scanChain.index if available, otherwise return large number for stable sort."""
    chain = record.get("scanChain")
    if isinstance(chain, dict) and "index" in chain:
        try:
            return int(chain["index"])
        except (ValueError, TypeError):
            pass
    return 999999


def extract_heading(record: dict) -> float | None:
    """Extract orientation heading (degrees) from record or nested arKitData."""
    # 1. Primary: magneticHeading
    h = record.get("magneticHeading")
    if h is not None:
        try:
            return float(h)
        except (ValueError, TypeError):
            pass

    # 2. Secondary: arKitData.originHeading
    arkit = record.get("arKitData") or {}
    h = arkit.get("originHeading")
    if h is not None:
        try:
            return float(h)
        except (ValueError, TypeError):
            pass

    # 3. Alternatives: headingAtPlace, heading
    for alt in ["headingAtPlace", "heading"]:
        h = record.get(alt) or arkit.get(alt)
        if h is not None:
            try:
                return float(h)
            except (ValueError, TypeError):
                pass
    return None


def get_intrinsics_matrix(face_size: int, fov_deg: float = 90.0) -> np.ndarray:
    """Compute 3x3 pinhole intrinsics matrix K for arbitrary FoV square cubemap face."""
    half = face_size / 2.0
    tan_half_fov = float(np.tan(np.deg2rad(fov_deg) / 2.0))
    f = half / tan_half_fov
    return np.array([
        [f, 0.0, half],
        [0.0, f, half],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def sample_face_depth(
    equirect_depth: np.ndarray,
    face: str,
    face_size: int,
    fov_deg: float = 90.0,
) -> np.ndarray:
    """
    Bilinear sample of one cubemap perspective depth face from an equirectangular depth map,
    converting spherical radial distance r to camera planar Z-depth in OpenCV RDF frame (+Z forward).

    Args:
        equirect_depth: 2D float32 numpy array (H, W) storing metric radial distance.
        face: Face name in ('front', 'right', 'back', 'left', 'up', 'down').
        face_size: Width and height of square perspective face.
        fov_deg: Field of view in degrees (default 90.0).

    Returns:
        (face_size, face_size) float32 numpy array representing camera-frame Z depth (depth_z).
    """
    height, width = equirect_depth.shape[:2]
    half = face_size / 2.0
    tan_half_fov = float(np.tan(np.deg2rad(fov_deg) / 2.0))
    cols = ((np.arange(face_size, dtype=np.float64) + 0.5 - half) / half) * tan_half_fov
    rows = ((np.arange(face_size, dtype=np.float64) + 0.5 - half) / half) * tan_half_fov
    grid_x, grid_y = np.meshgrid(cols, rows)

    # Cosine projection factor: rho = sqrt(1 + x^2 + y^2) converts radial distance r to planar depth_z = r / rho
    rho = np.sqrt(1.0 + grid_x * grid_x + grid_y * grid_y).astype(np.float32)

    rays = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
    pano = rays @ LOCAL_ROTATIONS[face].T
    x = pano[..., 0]
    y = pano[..., 1]
    z = pano[..., 2]
    lon = np.arctan2(x, z)
    lat_up = np.arctan2(-y, np.hypot(x, z))
    map_x = np.mod((lon / (2.0 * np.pi) + 0.5) * width, width)
    map_y = np.clip((0.5 - lat_up / np.pi) * height, 0.0, height - 1.0)

    # Pure float32 bilinear interpolation (preserves metric precision, avoids uint8 truncation)
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    wx = (map_x - x0).astype(np.float32)
    wy = (map_y - y0).astype(np.float32)
    x0 = np.mod(x0, width)
    x1 = np.mod(x0 + 1, width)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    eq = equirect_depth.astype(np.float32)
    top = eq[y0, x0] * (1.0 - wx) + eq[y0, x1] * wx
    bot = eq[y1, x0] * (1.0 - wx) + eq[y1, x1] * wx
    radial_depth = top * (1.0 - wy) + bot * wy

    # Convert radial distance to camera planar Z depth: depth_z = r / rho
    face_depth_z = radial_depth / rho
    # Clean non-finite or negative depths
    face_depth_z = np.nan_to_num(face_depth_z, nan=0.0, posinf=0.0, neginf=0.0)
    face_depth_z = np.maximum(face_depth_z, 0.0).astype(np.float32)
    return face_depth_z


def load_depth_scales(scale_file: Optional[Union[str, Path]]) -> Dict[str, dict]:
    """Load MoGe depth scales JSON dictionary."""
    if not scale_file:
        return {}
    sf = Path(scale_file)
    if not sf.is_file():
        return {}
    try:
        with open(sf, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Warning: Failed to load depth scales from {sf}: {e}")
        return {}


def discover_depth_maps(
    images_by_jobid: Dict[str, Path],
    depth_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """
    Discover matching .npy depth files for each jobId.
    Searches depth_dir if provided, or adjacent to images.
    """
    depth_by_jobid: Dict[str, Path] = {}
    search_dirs: List[Path] = []
    if depth_dir:
        dp = Path(depth_dir)
        if dp.is_dir():
            search_dirs.append(dp)
    for img_path in images_by_jobid.values():
        if img_path.parent not in search_dirs:
            search_dirs.append(img_path.parent)

    for job_id in images_by_jobid.keys():
        for sdir in search_dirs:
            candidates = [
                sdir / f"{job_id}.npy",
                sdir / f"{job_id}_depth.npy",
            ]
            found = False
            for cand in candidates:
                if cand.is_file():
                    depth_by_jobid[job_id] = cand
                    found = True
                    break
            if found:
                break
            for f in sdir.glob("*.npy"):
                if job_id in f.stem:
                    depth_by_jobid[job_id] = f
                    found = True
                    break
            if found:
                break

    return depth_by_jobid


def generate_camera_markers(
    poses: List[np.ndarray],
    radius: float = 0.25,
    sphere_radius: float = 0.06,
    num_ring_points: int = 72,
    num_sphere_points: int = 120,
    ray_length: float = 0.40,
    marker_style: str = "all_rays",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D point cloud markers for camera stations:
    - A colored solid Fibonacci sphere at each station optical center
    - A circular horizontal ring (XZ plane) around each camera
    - Optical sight rays along camera viewing directions
    """
    marker_pts = []
    marker_rgb = []

    # Cluster poses by optical center (stations)
    centers: List[np.ndarray] = []
    center_to_views: Dict[int, List[int]] = {}

    for idx, p in enumerate(poses):
        c = p[:3, 3]
        matched_idx = None
        for c_idx, existing_c in enumerate(centers):
            if np.linalg.norm(c - existing_c) < 0.05:
                matched_idx = c_idx
                break
        if matched_idx is None:
            matched_idx = len(centers)
            centers.append(c)
            center_to_views[matched_idx] = []
        center_to_views[matched_idx].append(idx)

    for c_idx, c in enumerate(centers):
        color = TRAJECTORY_PALETTE[c_idx % len(TRAJECTORY_PALETTE)]

        # 1. Solid Fibonacci sphere at station optical center
        phi = np.pi * (np.sqrt(5.0) - 1.0)
        sphere_pts = []
        for i in range(num_sphere_points):
            y = 1.0 - (i / float(max(1, num_sphere_points - 1))) * 2.0
            r_y = np.sqrt(max(0.0, 1.0 - y * y))
            theta = phi * i
            x = np.cos(theta) * r_y
            z = np.sin(theta) * r_y
            for s in [0.4, 0.7, 1.0]:
                sphere_pts.append(c + s * sphere_radius * np.array([x, y, z], dtype=np.float32))

        sphere_pts_arr = np.array(sphere_pts, dtype=np.float32)
        marker_pts.append(sphere_pts_arr)
        marker_rgb.append(np.tile(color, (len(sphere_pts_arr), 1)).astype(np.uint8))

        # 2. Horizontal circular ring in XZ plane around station
        angles = np.linspace(0, 2 * np.pi, num_ring_points, endpoint=False)
        ring_pts = [c + np.array([radius * np.cos(a), 0.0, radius * np.sin(a)], dtype=np.float32) for a in angles]
        ring_pts_arr = np.array(ring_pts, dtype=np.float32)
        marker_pts.append(ring_pts_arr)
        marker_rgb.append(np.tile(color, (len(ring_pts_arr), 1)).astype(np.uint8))

        # 3. Sight rays for views originating at this station
        if marker_style != "station_only":
            views_to_draw = center_to_views[c_idx]
            if marker_style == "heading_only":
                views_to_draw = views_to_draw[:1]

            for view_idx in views_to_draw:
                R = poses[view_idx][:3, :3]
                forward_dir = R[:, 2]  # +Z forward in OpenCV RDF
                num_ray_steps = 25
                ray_pts = [c + s * forward_dir for s in np.linspace(sphere_radius, ray_length, num_ray_steps)]
                ray_pts_arr = np.array(ray_pts, dtype=np.float32)
                marker_pts.append(ray_pts_arr)
                ray_color = np.clip(np.array(color, dtype=np.float32) * 1.25, 0, 255).astype(np.uint8)
                marker_rgb.append(np.tile(ray_color, (len(ray_pts_arr), 1)).astype(np.uint8))

    if not marker_pts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    return np.concatenate(marker_pts, axis=0), np.concatenate(marker_rgb, axis=0)


def write_binary_ply(
    filepath: Union[str, Path],
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    """
    Write point cloud directly to binary little-endian PLY format using pure NumPy.
    Zero external dependencies (does not require Open3D or Trimesh).
    Storage size: exactly 15 bytes per point (float32 x, y, z + uint8 r, g, b).
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    colors = np.ascontiguousarray(colors, dtype=np.uint8)
    num_pts = len(points)

    ply_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ])

    vertex_data = np.empty(num_pts, dtype=ply_dtype)
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_pts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    with open(filepath, "wb") as f:
        f.write(header)
        vertex_data.tofile(f)


def export_point_cloud(
    output_path: str,
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: Optional[float] = 0.01,
    camera_markers: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    dense: bool = False,
    verbose: bool = True,
) -> None:
    """
    Save 3D point cloud strictly as binary little-endian .ply format with optional voxel downsampling.
    Default voxel_size is 0.01 (1cm grid). Use dense=True to export all points without downsampling.
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if dense:
        voxel_size = None
        if verbose:
            print(f"Full dense mode: preserving all {len(points):,} points without downsampling.")

    # Apply voxel downsampling
    if voxel_size is not None and voxel_size > 0 and len(points) > 0:
        if verbose:
            print(f"Applying voxel downsampling (voxel_size = {voxel_size:.4f} m)...")
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
            points = np.asarray(pcd_down.points).astype(np.float32)
            colors = (np.asarray(pcd_down.colors) * 255.0).clip(0, 255).astype(np.uint8)
            if verbose:
                print(f"Point count reduced to {len(points):,} points.")
        except Exception:
            # Fast pure NumPy voxel grid fallback
            coords = np.floor(points / voxel_size).astype(np.int32)
            coords_view = np.ascontiguousarray(coords).view(np.dtype((np.void, coords.dtype.itemsize * 3)))
            _, unique_indices = np.unique(coords_view, return_index=True)
            points = points[unique_indices]
            colors = colors[unique_indices]
            if verbose:
                print(f"NumPy voxel grid reduced point count to {len(points):,} points.")

    # Append camera markers after downsampling so markers remain sharp and dense
    if camera_markers is not None:
        m_pts, m_colors = camera_markers
        if len(m_pts) > 0:
            points = np.concatenate([points, m_pts], axis=0)
            colors = np.concatenate([colors, m_colors], axis=0)
            if verbose:
                print(f"Appended {len(m_pts):,} camera marker points to output point cloud.")

    if not output_path.lower().endswith(".ply"):
        output_path = os.path.splitext(output_path)[0] + ".ply"

    write_binary_ply(output_path, points, colors)
    if verbose:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Saved binary PLY point cloud ({size_mb:.2f} MB, {len(points):,} points) to: {output_path}")


def reconstruct_panorama_dataset(
    input_path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
    output_path: Union[str, Path] = "output/reconstruction_v3",
    depth_dir: Optional[Union[str, Path]] = None,
    depth_scale_file: Optional[Union[str, Path]] = None,
    depth_scale_metric: str = "median",
    disable_depth: bool = False,
    face_size: int = 1024,
    fov_deg: float = 90.0,
    voxel_size: Optional[float] = 0.01,
    dense: bool = False,
    max_depth: Optional[float] = 25.0,
    use_pose_inputs: bool = False,
    use_multiview_confidence: bool = False,
    apply_confidence_mask: bool = False,
    confidence_percentile: float = 10.0,
    minibatch_size: int = 1,
    device: Optional[str] = None,
    model: Optional[Any] = None,
    model_name: str = "facebook/map-anything",
    add_camera_markers: bool = True,
    marker_radius: float = 0.25,
    marker_style: str = "all_rays",
    save_cubemap_faces: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Main reconstruction pipeline (v3):
    1. Parse ARKit metadata & images, check heading convention, sort by scanChain.
    2. Extract 6 cubemap perspective faces per panorama using analytical OpenCV RDF poses.
    3. Run MapAnything inference (unposed with optical nodal lock OR posed conditioning).
    4. Unproject depth maps to ARKit world frame.
    5. Render camera markers and export binary PLY point cloud.
    """
    final_ply_file, target_dir = resolve_output_path(output_path, input_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover inputs and parse records
    raw_records, images_by_jobid, metadata_file = discover_inputs(input_path, metadata_path=metadata_path)
    if verbose:
        print(f"\n[Metadata] Source: {metadata_file}")
        print(f"[Metadata] Found {len(raw_records)} records, matched {len(images_by_jobid)} panorama images.")

    # Discover depth maps & scale factors
    depth_scales: Dict[str, dict] = {}
    depth_by_jobid: Dict[str, Path] = {}
    if not disable_depth:
        # If depth_scale_file not explicitly provided, search in depth_dir or adjacent to images
        if not depth_scale_file:
            search_dirs = [Path(depth_dir)] if depth_dir and Path(depth_dir).is_dir() else []
            for p in images_by_jobid.values():
                if p.parent not in search_dirs:
                    search_dirs.append(p.parent)
            for sdir in search_dirs:
                scale_cand = sdir / "moge_depth_scales.json"
                if scale_cand.is_file():
                    depth_scale_file = scale_cand
                    break

        depth_scales = load_depth_scales(depth_scale_file)
        depth_by_jobid = discover_depth_maps(images_by_jobid, depth_dir=depth_dir)
        if verbose:
            if depth_by_jobid:
                print(f"[Depth Prior] Matched {len(depth_by_jobid)}/{len(images_by_jobid)} depth maps.")
                if depth_scale_file:
                    print(f"[Depth Prior] Using scales from: {depth_scale_file} (metric: '{depth_scale_metric}')")
            else:
                print(f"[Depth Prior] No depth maps found. Proceeding with images-only estimation.")

    # Heading convention verification
    if verbose:
        check_heading_convention(raw_records)

    # Match and sort records by scanChain.index (Station 1 -> N)
    station_records = []
    by_id = index_records(raw_records)

    for job_id, img_path in images_by_jobid.items():
        rec = by_id.get(job_id)
        if rec is None:
            continue
        pos = origin_xyz(rec)
        heading = extract_heading(rec)
        if pos is None:
            if verbose:
                print(f"Warning: {job_id} missing arKitData.origin, skipping.")
            continue
        if heading is None:
            if verbose:
                print(f"Warning: {job_id} missing heading, skipping.")
            continue

        chain_idx = get_scan_chain_index(rec)
        station_records.append({
            "jobId": job_id,
            "record": rec,
            "image_path": img_path,
            "position": pos,
            "heading": float(heading),
            "scan_chain_index": chain_idx,
        })

    # Sort sequentially by scan chain index
    station_records.sort(key=lambda s: (s["scan_chain_index"], natural_sort_key(s["jobId"])))

    if not station_records:
        raise RuntimeError("No valid panorama stations found with origin and heading.")

    if verbose:
        print(f"\n[Trajectory] Configured {len(station_records)} stations in sequential scan order:")
        for idx, st in enumerate(station_records):
            p = st["position"]
            chain_str = f"ChainIdx={st['scan_chain_index']}" if st["scan_chain_index"] < 999999 else "NoChainIdx"
            print(f"  Station {idx + 1:<2} ({st['jobId']}): Pos=[{p[0]:6.2f}, {p[1]:6.2f}, {p[2]:6.2f}] Heading={st['heading']:5.1f}° ({chain_str})")

    # Build panorama station poses (OpenCV RDF front face rotation + origin position)
    station_poses = []
    poses_debug_export = []
    for idx, st in enumerate(station_records):
        R_pano = front_rotation(st["heading"])
        T_pano = np.eye(4, dtype=np.float64)
        T_pano[:3, :3] = R_pano
        T_pano[:3, 3] = st["position"]
        station_poses.append(T_pano)

        poses_debug_export.append({
            "station_index": idx + 1,
            "jobId": st["jobId"],
            "scan_chain_index": st["scan_chain_index"],
            "position": st["position"].tolist(),
            "heading_deg": st["heading"],
            "c2w_front_rdf": T_pano.tolist(),
        })

    with open(target_dir / "poses_debug.json", "w", encoding="utf-8") as f:
        json.dump(poses_debug_export, f, indent=2)

    # Fast Dry-Run mode
    if dry_run:
        if verbose:
            print("\n[Dry-Run] Bypassing neural network inference. Generating camera trajectory markers...")
        m_pts, m_colors = generate_camera_markers(
            station_poses,
            radius=marker_radius,
            marker_style=marker_style,
        )
        export_point_cloud(
            output_path=final_ply_file,
            points=m_pts,
            colors=m_colors,
            voxel_size=None,
            camera_markers=None,
            dense=True,
            verbose=verbose,
        )

        if save_cubemap_faces:
            faces_cache_dir = target_dir / "cubemap_faces"
            faces_cache_dir.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"[Dry-Run] Extracting and saving cubemap faces to: {faces_cache_dir}")
            for st in station_records:
                job_id = st["jobId"]
                st_faces_dir = faces_cache_dir / job_id
                st_faces_dir.mkdir(parents=True, exist_ok=True)
                with Image.open(st["image_path"]) as img:
                    equirect = np.asarray(img.convert("RGB"))
                size = face_size if face_size > 0 else equirect.shape[1] // 4

                # Check if equirect depth is available
                st_depth_path = depth_by_jobid.get(job_id)
                eq_depth = None
                if st_depth_path and st_depth_path.is_file():
                    try:
                        raw_d = np.load(st_depth_path).astype(np.float32)
                        if raw_d.ndim == 3 and raw_d.shape[-1] == 1:
                            raw_d = raw_d.squeeze(-1)
                        s_info = depth_scales.get(job_id)
                        d_scale = 1.0
                        if isinstance(s_info, dict):
                            d_scale = float(s_info.get(depth_scale_metric, 1.0))
                        elif isinstance(s_info, (int, float)):
                            d_scale = float(s_info)
                        eq_depth = raw_d * d_scale
                    except Exception as e:
                        print(f"Warning loading dry-run depth {st_depth_path}: {e}")

                for face_name in FACE_ORDER:
                    sampled = sample_face(equirect, face_name, size, fov_deg=fov_deg)
                    Image.fromarray(sampled, mode="RGB").save(st_faces_dir / f"{face_name}.jpg", quality=95)
                    if eq_depth is not None:
                        f_depth_z = sample_face_depth(eq_depth, face_name, size, fov_deg=fov_deg)
                        np.save(st_faces_dir / f"{face_name}_depth.npy", f_depth_z)

        if verbose:
            print(f"[Dry-Run Complete] Saved camera trajectory point cloud to: {final_ply_file}")
        return m_pts, m_colors, station_poses

    # 2. Load MapAnything Model
    try:
        import torch
        from mapanything.models import MapAnything
        from mapanything.utils.device import get_device
        from mapanything.utils.geometry import depthmap_to_world_frame
        from mapanything.utils.image import preprocess_inputs
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Required module for neural inference is missing ({e}). "
            "Please ensure PyTorch and mapanything dependencies are installed in your environment, "
            "or run with --dry_run for trajectory and camera marker point cloud generation."
        ) from e

    if device is None:
        device = get_device()

    if model is None:
        if verbose:
            print(f"\n[Model] Loading MapAnything ({model_name}) on {device}...")
        model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()

    # Determine hardware-matched AMP dtype
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability(device)
            amp_dtype = "bf16" if major >= 8 and torch.cuda.is_bf16_supported() else "fp16"
        except Exception:
            amp_dtype = "fp16"
        use_amp = True
    else:
        amp_dtype = "fp32"
        use_amp = False

    if verbose:
        print(f"[Model] Device: {device}, Mixed Precision AMP: {amp_dtype.upper()}")
        print(f"[Mode] Inference pose conditioning: {'POSED (using analytical RDF poses)' if use_pose_inputs else 'UNPOSED (local estimation with Optical Nodal Lock)'}")

    # Cubemap output cache directory if requested
    faces_cache_dir = target_dir / "cubemap_faces" if save_cubemap_faces else None
    if faces_cache_dir:
        faces_cache_dir.mkdir(parents=True, exist_ok=True)

    all_points = []
    all_colors = []
    all_refined_face_poses = []

    total_stations = len(station_records)
    if verbose:
        print(f"\n[Inference] Slicing and reconstructing {total_stations} panoramas...")

    K_matrix = None

    for s_idx, st in enumerate(station_records):
        job_id = st["jobId"]
        img_path = st["image_path"]
        pos = st["position"]
        heading = st["heading"]
        T_pano = station_poses[s_idx]
        T_pano_torch = torch.from_numpy(T_pano).to(device=device, dtype=torch.float32)

        if verbose:
            print(f"\n--- Station {s_idx + 1}/{total_stations}: '{job_id}' ---")
            print(f"  Anchor World Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m, Heading: {heading:.2f}°")

        # Load equirectangular panorama
        with Image.open(img_path) as img:
            equirect = np.asarray(img.convert("RGB"))

        curr_face_size = face_size if face_size > 0 else equirect.shape[1] // 4
        if K_matrix is None or K_matrix.shape[0] != curr_face_size:
            K_matrix = get_intrinsics_matrix(curr_face_size, fov_deg=fov_deg)

        # Load and scale equirectangular depth prior if available
        equirect_depth = None
        station_depth_path = depth_by_jobid.get(job_id)
        if station_depth_path and station_depth_path.is_file():
            try:
                raw_depth = np.load(station_depth_path).astype(np.float32)
                if raw_depth.ndim == 3 and raw_depth.shape[-1] == 1:
                    raw_depth = raw_depth.squeeze(-1)
                scale_info = depth_scales.get(job_id)
                depth_scale_val = 1.0
                if isinstance(scale_info, dict):
                    depth_scale_val = float(scale_info.get(depth_scale_metric, 1.0))
                elif isinstance(scale_info, (int, float)):
                    depth_scale_val = float(scale_info)
                equirect_depth = raw_depth * depth_scale_val
                if verbose:
                    print(f"  Loaded depth prior: {station_depth_path.name} (scale: {depth_scale_val:.4f}, range: [{np.nanmin(equirect_depth):.2f}, {np.nanmax(equirect_depth):.2f}] m)")
            except Exception as e:
                print(f"  Warning: Failed to load depth from {station_depth_path}: {e}")

        # Slice 6 perspective faces using extract_cubemap_faces sampling
        face_items = []
        station_faces_dir = faces_cache_dir / job_id if faces_cache_dir else None
        if station_faces_dir:
            station_faces_dir.mkdir(parents=True, exist_ok=True)

        for face_name in FACE_ORDER:
            # Sliced face image (curr_face_size, curr_face_size, 3)
            sampled_face = sample_face(equirect, face_name, curr_face_size, fov_deg=fov_deg)
            if station_faces_dir:
                Image.fromarray(sampled_face, mode="RGB").save(station_faces_dir / f"{face_name}.jpg", quality=95)

            # Sliced face depth (curr_face_size, curr_face_size) in camera Z frame
            face_depth_z = None
            if equirect_depth is not None:
                face_depth_z = sample_face_depth(equirect_depth, face_name, curr_face_size, fov_deg=fov_deg)
                if station_faces_dir:
                    np.save(station_faces_dir / f"{face_name}_depth.npy", face_depth_z)

            # Face camera-to-world pose in OpenCV RDF
            R_face = face_pose(pos, heading, face_name)
            T_face = np.eye(4, dtype=np.float32)
            T_face[:3, :3] = R_face.astype(np.float32)
            T_face[:3, 3] = pos.astype(np.float32)

            face_items.append({
                "face_name": face_name,
                "image": sampled_face,
                "depth_z": face_depth_z,
                "c2w": T_face,
            })

        # Build raw views for MapAnything
        pano_raw_views = []
        for face_item in face_items:
            view_dict = {
                "img": torch.from_numpy(face_item["image"]),
                "intrinsics": torch.from_numpy(K_matrix),
                "is_metric_scale": torch.tensor([True]),
            }
            if face_item["depth_z"] is not None:
                # Must be 2D (H, W) float32 for preprocess_inputs
                view_dict["depth_z"] = torch.from_numpy(face_item["depth_z"])
            if use_pose_inputs:
                view_dict["camera_poses"] = torch.from_numpy(face_item["c2w"])
            pano_raw_views.append(view_dict)

        has_depth_input = any("depth_z" in v for v in pano_raw_views)
        pano_proc_views = preprocess_inputs(pano_raw_views, verbose=False)

        # Run inference
        with torch.inference_mode():
            pano_outputs = model.infer(
                pano_proc_views,
                memory_efficient_inference=True,
                minibatch_size=minibatch_size,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                apply_mask=True,
                mask_edges=True,
                use_multiview_confidence=use_multiview_confidence,
                apply_confidence_mask=apply_confidence_mask,
                confidence_percentile=confidence_percentile,
                ignore_calibration_inputs=False,
                ignore_depth_inputs=not has_depth_input,
                ignore_pose_inputs=not use_pose_inputs,
                ignore_pose_scale_inputs=False,
            )

        # Unproject depth maps to world coordinates
        station_pts_count = 0
        for f_idx, face_item in enumerate(face_items):
            pred = pano_outputs[f_idx]
            depth_z = pred["depth_z"][0].squeeze(-1)       # (H, W)
            intrinsics_view = pred["intrinsics"][0]        # (3, 3)
            mask = pred["mask"][0].squeeze(-1).bool()
            img_no_norm = pred["img_no_norm"][0]

            if use_pose_inputs:
                # Use analytical world pose directly
                T_face_world = torch.from_numpy(face_item["c2w"]).to(device=device, dtype=torch.float32)
            else:
                # Optical Nodal Center Lock:
                # Rigidly compose orientation: R_face_world = R_pano @ R_face_local
                T_face_local = pred["camera_poses"][0]     # (4, 4)
                R_face_world = torch.matmul(T_pano_torch[:3, :3], T_face_local[:3, :3])
                T_face_world = torch.eye(4, device=device, dtype=torch.float32)
                T_face_world[:3, :3] = R_face_world
                T_face_world[:3, 3] = T_pano_torch[:3, 3]

            all_refined_face_poses.append(T_face_world.cpu().numpy())

            # Unproject depth map to world 3D coordinates
            pts3d_world, valid_depth = depthmap_to_world_frame(depth_z, intrinsics_view, T_face_world)
            valid_mask = mask & valid_depth

            if max_depth is not None and max_depth > 0:
                valid_mask = valid_mask & (depth_z <= max_depth)

            valid_mask_np = valid_mask.cpu().numpy()
            pts_np = pts3d_world.cpu().numpy()[valid_mask_np]
            rgb_np = (img_no_norm.cpu().numpy()[valid_mask_np] * 255.0).clip(0, 255).astype(np.uint8)

            if len(pts_np) > 0:
                all_points.append(pts_np)
                all_colors.append(rgb_np)
                station_pts_count += len(pts_np)

        if verbose:
            print(f"  -> Generated {station_pts_count:,} world 3D points for '{job_id}'.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_points:
        raise RuntimeError("No valid 3D points produced across all panoramas.")

    combined_points = np.concatenate(all_points, axis=0)
    combined_colors = np.concatenate(all_colors, axis=0)

    if verbose:
        print(f"\n[Complete] Total raw 3D points generated: {len(combined_points):,}")

    # Generate Camera Trajectory Markers & Export Point Cloud
    camera_markers_tuple = None
    if add_camera_markers and station_poses:
        m_pts, m_colors = generate_camera_markers(
            station_poses,
            radius=marker_radius,
            marker_style=marker_style,
        )
        camera_markers_tuple = (m_pts, m_colors)

    export_point_cloud(
        output_path=final_ply_file,
        points=combined_points,
        colors=combined_colors,
        voxel_size=voxel_size,
        camera_markers=camera_markers_tuple,
        dense=dense,
        verbose=verbose,
    )

    # Save all refined face poses array
    if all_refined_face_poses:
        np.save(target_dir / "face_poses_world.npy", np.array(all_refined_face_poses, dtype=np.float32))

    return combined_points, combined_colors, all_refined_face_poses


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct from Posed Images v3: ARKit Analytical RDF Cubemap Pipeline using MapAnything."
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Path to folder containing panorama images and jobdata.json (e.g. 'jobdata').",
    )
    parser.add_argument(
        "--metadata", "--jobdata",
        dest="metadata",
        nargs="+",
        default=None,
        help="Optional explicit path to jobdata.json / metadata JSON if stored separately from images.",
    )
    parser.add_argument(
        "--output", "-o",
        nargs="+",
        default=["output/reconstruction_v3"],
        help="Output directory or point cloud file (.ply). Default: 'output/reconstruction_v3'.",
    )
    parser.add_argument(
        "--face_size",
        type=int,
        default=1024,
        help="Resolution in pixels of cubemap perspective faces (default: 1024, or 0 for equirect width // 4).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="Field of view in degrees (default: 90.0; e.g. 100.0 for 10° visual overlap between adjacent faces).",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Voxel downsampling grid size in meters (default: 0.01 for 1cm grid). Use --dense to disable.",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        default=False,
        help="Export full dense point cloud without voxel downsampling.",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=25.0,
        help="Maximum depth cutoff in meters to eliminate distant noise and sky points (default: 25.0m).",
    )
    parser.add_argument(
        "--use_pose_inputs",
        action="store_true",
        default=False,
        help="Condition MapAnything directly on analytical RDF cubemap poses (default: False for unposed relative estimation with Optical Nodal Lock).",
    )
    parser.add_argument(
        "--use_multiview_confidence",
        action="store_true",
        default=False,
        help="Enable geometric multi-view depth consistency filtering.",
    )
    parser.add_argument(
        "--apply_confidence_mask",
        action="store_true",
        default=False,
        help="Enable confidence percentile filtering.",
    )
    parser.add_argument(
        "--confidence_percentile",
        type=float,
        default=10.0,
        help="Percentile for confidence masking (default: 10.0).",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=1,
        help="Minibatch size for memory-efficient head inference (default: 1).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Fast debugging mode: parses ARKit trajectory, verifies heading convention, and exports camera markers PLY without running neural inference.",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model checkpoint (facebook/map-anything-apache).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device ('cuda', 'cpu', or auto-detect).",
    )
    parser.add_argument(
        "--no_camera_markers",
        dest="add_camera_markers",
        action="store_false",
        default=True,
        help="Do not embed camera markers in the output point cloud.",
    )
    parser.add_argument(
        "--marker_radius",
        type=float,
        default=0.25,
        help="Radius of camera circle markers in meters (default: 0.25m).",
    )
    parser.add_argument(
        "--marker_style",
        type=str,
        default="all_rays",
        choices=["all_rays", "station_only", "heading_only"],
        help="Marker style: 'all_rays' (sphere + ring + rays), 'station_only' (sphere + ring), 'heading_only' (sphere + ring + 1 forward ray).",
    )
    parser.add_argument(
        "--save_cubemap_faces",
        action="store_true",
        default=False,
        help="Export sliced perspective cubemap face images to output directory.",
    )
    parser.add_argument(
        "--depth_dir",
        type=str,
        default=None,
        help="Optional directory containing per-panorama .npy depth files (defaults to searching input images directory).",
    )
    parser.add_argument(
        "--depth_scale_file",
        type=str,
        default=None,
        help="Optional JSON file containing per-panorama depth scale factors (e.g. moge_depth_scales.json).",
    )
    parser.add_argument(
        "--depth_scale_metric",
        type=str,
        default="median",
        choices=["median", "mean"],
        help="Scale metric from depth scale JSON to use for metric depth conversion (default: 'median').",
    )
    parser.add_argument(
        "--disable_depth",
        action="store_true",
        default=False,
        help="Disable input depth conditioning even if depth files are present.",
    )

    args = parser.parse_args()
    model_name = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"

    input_path = " ".join(args.input) if isinstance(args.input, list) else str(args.input)
    metadata_path = " ".join(args.metadata) if isinstance(args.metadata, list) and args.metadata else (args.metadata if args.metadata else None)
    output_path = " ".join(args.output) if isinstance(args.output, list) else str(args.output)

    final_file, target_dir = resolve_output_path(output_path, input_path)
    reconstruct_panorama_dataset(
        input_path=input_path,
        metadata_path=metadata_path,
        output_path=final_file,
        depth_dir=args.depth_dir,
        depth_scale_file=args.depth_scale_file,
        depth_scale_metric=args.depth_scale_metric,
        disable_depth=args.disable_depth,
        face_size=args.face_size,
        fov_deg=args.fov,
        voxel_size=args.voxel_size,
        dense=args.dense,
        max_depth=args.max_depth,
        use_pose_inputs=args.use_pose_inputs,
        use_multiview_confidence=args.use_multiview_confidence,
        apply_confidence_mask=args.apply_confidence_mask,
        confidence_percentile=args.confidence_percentile,
        minibatch_size=args.minibatch_size,
        device=args.device,
        model_name=model_name,
        add_camera_markers=args.add_camera_markers,
        marker_radius=args.marker_radius,
        marker_style=args.marker_style,
        save_cubemap_faces=args.save_cubemap_faces,
        dry_run=args.dry_run,
        verbose=True,
    )


if __name__ == "__main__":
    main()
