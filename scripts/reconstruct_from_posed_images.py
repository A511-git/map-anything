# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Reconstruct a unified 3D point cloud from images and their corresponding camera poses
using MapAnything.

Supports:
- Auto-detecting images in a folder using natural numeric sorting (e.g. img1, img2, ..., img10)
- Poses in OpenCV cam2world (+X Right, +Y Down, +Z Forward) format
- Pose convention conversion (OpenGL/Blender RUB -> OpenCV RDF, COLMAP world2cam -> cam2world)
- Optional camera intrinsics for sharper, planar, distortion-free geometry
- Multi-view depth consistency filtering and max-depth clipping
- Output export to .ply or .pcd with optional voxel downsampling

Usage Examples:
    # Basic reconstruction from image folder and poses.npy:
    python scripts/reconstruct_from_posed_images.py \\
        --images path/to/images/ \\
        --poses path/to/poses.npy \\
        --output output/combined.ply \\
        --voxel_size 0.01

    # Reconstruction with intrinsics, multi-view consistency, and max-depth:
    python scripts/reconstruct_from_posed_images.py \\
        --images path/to/images/ \\
        --poses path/to/poses.npy \\
        --intrinsics path/to/intrinsics.npy \\
        --output output/combined.pcd \\
        --voxel_size 0.005 \\
        --use_multiview_confidence \\
        --max_depth 25.0
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Ensure mapanything and scripts are discoverable on PYTHONPATH
repo_root = Path(__file__).resolve().parent.parent
scripts_dir = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import numpy as np
import torch
from PIL import Image


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def natural_sort_key(s: Union[str, Path]) -> list:
    """
    Key for natural numerical sorting (e.g. frame_2 comes before frame_10).
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def find_images(folder_or_list: Union[str, Path, List[Union[str, Path]]]) -> Tuple[List[str], List[str]]:
    """
    Discover image files and sort them in natural numerical order.

    Args:
        folder_or_list: Directory path or list of image paths

    Returns:
        Tuple of (image_paths, image_filenames)
    """
    image_paths = []

    if isinstance(folder_or_list, (str, Path)):
        p = Path(folder_or_list)
        if p.is_dir():
            files = [f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTS and not f.name.startswith(".")]
            files.sort(key=lambda x: natural_sort_key(x.name))
            image_paths = [str(f) for f in files]
        elif p.is_file():
            image_paths = [str(p)]
        else:
            raise FileNotFoundError(f"Image path not found: {folder_or_list}")
    elif isinstance(folder_or_list, list):
        for item in folder_or_list:
            ip = Path(item)
            if ip.is_file() and ip.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                image_paths.append(str(ip))
            elif ip.is_dir():
                files = [f for f in ip.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTS and not f.name.startswith(".")]
                files.sort(key=lambda x: natural_sort_key(x.name))
                image_paths.extend([str(f) for f in files])
            else:
                image_paths.append(str(ip))
    else:
        raise ValueError(f"Unsupported type for folder_or_list: {type(folder_or_list)}")

    if not image_paths:
        raise ValueError(f"No valid image files found in {folder_or_list}")

    image_filenames = [Path(p).name for p in image_paths]
    return image_paths, image_filenames


def convert_pose_convention(pose: np.ndarray, from_convention: str = "opencv") -> np.ndarray:
    """
    Convert a 4x4 camera pose from an external convention into OpenCV RDF cam2world format
    (+X Right, +Y Down, +Z Forward).

    Supported conventions:
    - "opencv": Already OpenCV cam2world. Returns copy.
    - "opengl" / "blender": RUB (+X Right, +Y Up, -Z Forward). Flips Y and Z axes.
    - "colmap_w2c": COLMAP world2cam matrix. Inverts to cam2world.
    """
    pose = np.array(pose, dtype=np.float32)
    assert pose.shape == (4, 4), f"Expected 4x4 pose matrix, got {pose.shape}"

    conv = from_convention.lower()
    if conv == "opencv":
        return pose.copy()

    elif conv in ["opengl", "blender"]:
        # Transformation: X_opencv = X_opengl, Y_opencv = -Y_opengl, Z_opencv = -Z_opengl
        # Cam2World transform: T_c2w_opencv = T_c2w_opengl @ diag([1, -1, -1, 1])
        c2w = pose.copy()
        c2w[:3, 1] *= -1.0
        c2w[:3, 2] *= -1.0
        return c2w

    elif conv == "colmap_w2c":
        # Invert world2cam to cam2world: [R^T | -R^T * t]
        R = pose[:3, :3]
        t = pose[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_inv
        c2w[:3, 3] = t_inv
        return c2w

    else:
        raise ValueError(
            f"Unsupported pose convention: {from_convention}. Valid options: ['opencv', 'opengl', 'blender', 'colmap_w2c']"
        )


def _parse_jobdata_poses(
    content: list,
    image_filenames: Optional[List[str]] = None,
) -> List[np.ndarray]:
    """
    Parse camera poses from an ARKit scan chain JSON (e.g. jobdata.json).
    - Sorts records by scanChain.index (1 to N)
    - Synthesizes pose for terminal node (where arKitData is None) from previous node's destination
    - Converts ARKit column-major simd_float4x4 to OpenCV RDF cam2world (+X Right, +Y Down, +Z Forward)
    - Matches image filenames by jobId, stem, or sequential index
    """
    # 1. Sort by scanChain.index if available
    records = sorted(
        content,
        key=lambda x: x.get("scanChain", {}).get("index", 999999) if isinstance(x, dict) else 999999
    )

    # 2. Resolve terminal nodes where arKitData is None from previous node's destination
    for i, item in enumerate(records):
        if not isinstance(item, dict):
            continue
        if item.get("arKitData") is None and i > 0 and isinstance(records[i - 1], dict) and records[i - 1].get("arKitData") is not None:
            prev_arkit = records[i - 1]["arKitData"]
            item["arKitData"] = {
                "origin": prev_arkit.get("destination"),
                "originPose": prev_arkit.get("destinationPose"),
                "originHeading": prev_arkit.get("destinationHeading"),
                "destination": None,
                "destinationPose": None,
                "destinationHeading": None,
                "distanceMeters": 0.0,
                "horizontalDistanceMeters": 0.0,
                "isTerminal": True,
            }

    # 3. Transform ARKit coordinates to OpenCV RDF cam2world:
    # M_world = diag([1, -1, -1, 1]), M_cam = diag([1, -1, -1, 1])
    # T_c2w_cv = M_world @ T_c2w_arkit @ M_cam
    M = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    name_to_pose = {}
    ordered_poses = []

    for idx, item in enumerate(records):
        jid = str(item.get("jobId", f"node_{idx}"))
        arkit = item.get("arKitData") if isinstance(item, dict) else None
        c2w_cv = None

        if isinstance(arkit, dict) and "originPose" in arkit:
            raw_p = arkit["originPose"]
            if isinstance(raw_p, (list, tuple)) and len(raw_p) == 16:
                c2w_arkit = np.array(raw_p, dtype=np.float32).reshape((4, 4), order="F")
                c2w_cv = M @ c2w_arkit @ M

        if c2w_cv is None and isinstance(arkit, dict) and "origin" in arkit:
            o = arkit["origin"]
            if isinstance(o, dict):
                ox = float(o.get("x", 0.0))
                oy = -float(o.get("y", 0.0))
                oz = -float(o.get("z", 0.0))
            else:
                ox, oy, oz = float(o[0]), -float(o[1]), -float(o[2])
            h_deg = float(arkit.get("originHeading", 0.0))
            rad = np.radians(h_deg)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            c2w_cv = np.eye(4, dtype=np.float32)
            c2w_cv[0, 0] = cos_a
            c2w_cv[0, 2] = sin_a
            c2w_cv[2, 0] = -sin_a
            c2w_cv[2, 2] = cos_a
            c2w_cv[:3, 3] = [ox, oy, oz]

        if c2w_cv is None:
            c2w_cv = np.eye(4, dtype=np.float32)

        ordered_poses.append(c2w_cv)

        # Register lookup aliases
        aliases = [
            jid,
            f"{jid}.jpg", f"{jid}.jpeg", f"{jid}.png", f"{jid}.webp",
            f"{jid}_blurred.jpg", f"{jid}_blurred.png",
            f"{jid}_pano.jpg", f"{jid}_pano.png",
            f"frame_{idx}", f"frame_{idx+1}",
        ]
        for a in aliases:
            name_to_pose[a] = c2w_cv
            name_to_pose[Path(a).name] = c2w_cv
            name_to_pose[Path(a).stem] = c2w_cv

    if image_filenames:
        matched = []
        all_matched = True
        for fname in image_filenames:
            stem = Path(fname).stem
            cleaned_stem = re.sub(r"(_blurred|_pano|_equirectangular|_360)$", "", stem, flags=re.IGNORECASE)
            if fname in name_to_pose:
                matched.append(name_to_pose[fname])
            elif stem in name_to_pose:
                matched.append(name_to_pose[stem])
            elif cleaned_stem in name_to_pose:
                matched.append(name_to_pose[cleaned_stem])
            else:
                all_matched = False
                break
        if all_matched and len(matched) == len(image_filenames):
            return matched

        if len(ordered_poses) == len(image_filenames):
            return ordered_poses

    return ordered_poses


def load_poses(
    poses_path: str,
    num_expected: Optional[int] = None,
    pose_convention: str = "opencv",
    image_filenames: Optional[List[str]] = None,
) -> List[np.ndarray]:
    """
    Load camera poses from various file formats (.npy, .npz, .json, or folder of matrices).
    Converts poses to OpenCV (+X Right, +Y Down, +Z Forward) cam2world (4x4).

    Args:
        poses_path: Path to pose file or directory
        num_expected: Expected number of poses (optional)
        pose_convention: Source pose convention ('opencv', 'opengl', 'blender', 'colmap_w2c')
        image_filenames: List of image basenames to match poses by filename if available (optional)

    Returns:
        List of 4x4 numpy arrays in OpenCV cam2world format
    """
    path = Path(poses_path)
    if not path.exists():
        raise FileNotFoundError(f"Poses path does not exist: {poses_path}")

    poses = []

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(path)
            if arr.ndim == 3 and arr.shape[1:] == (4, 4):
                poses = [arr[i].astype(np.float32) for i in range(arr.shape[0])]
            elif arr.ndim == 2 and arr.shape == (4, 4):
                poses = [arr.astype(np.float32)]
            else:
                raise ValueError(f"Unexpected shape in .npy pose file: {arr.shape}. Expected (N, 4, 4) or (4, 4).")

        elif suffix == ".npz":
            data = np.load(path)
            key = "poses" if "poses" in data else list(data.keys())[0]
            arr = data[key]
            poses = [arr[i].astype(np.float32) for i in range(arr.shape[0])]

        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
            if isinstance(content, list):
                if len(content) > 0 and isinstance(content[0], dict):
                    # Check if this is an ARKit jobdata scan chain (containing jobId, scanChain, or arKitData)
                    if any("jobId" in item or "scanChain" in item or "arKitData" in item for item in content if isinstance(item, dict)):
                        poses = _parse_jobdata_poses(content, image_filenames=image_filenames)
                        if num_expected is not None and len(poses) != num_expected:
                            raise ValueError(f"Number of loaded poses ({len(poses)}) does not match number of images ({num_expected})")
                        # _parse_jobdata_poses produces OpenCV RDF cam2world directly; if another convention requested, convert:
                        if pose_convention != "opencv":
                            poses = [convert_pose_convention(p, from_convention=pose_convention) for p in poses]
                        return poses

                    name_to_pose = {}
                    for item in content:
                        name = item.get("image_name", "")
                        if "cam2world_pose_4x4" in item:
                            mat = np.array(item["cam2world_pose_4x4"], dtype=np.float32)
                        elif "extrinsic_matrix_3x4" in item:
                            m34 = np.array(item["extrinsic_matrix_3x4"], dtype=np.float32)
                            mat = np.eye(4, dtype=np.float32)
                            mat[:3, :4] = m34
                        elif "pose" in item:
                            mat = np.array(item["pose"], dtype=np.float32)
                        else:
                            mat = np.array(item, dtype=np.float32)
                        if name:
                            name_to_pose[name] = mat
                            name_to_pose[Path(name).name] = mat
                    if image_filenames and any(f in name_to_pose for f in image_filenames):
                        poses = [name_to_pose[f] for f in image_filenames if f in name_to_pose]
                    else:
                        poses = list(name_to_pose.values()) if name_to_pose else [np.array(m, dtype=np.float32) for m in content]
                else:
                    poses = [np.array(m, dtype=np.float32) for m in content]
            elif isinstance(content, dict):
                # Dict mapping image filename to 4x4 pose list
                name_to_pose = {}
                for k, v in content.items():
                    if isinstance(v, dict) and "cam2world_pose_4x4" in v:
                        mat = np.array(v["cam2world_pose_4x4"], dtype=np.float32)
                    elif isinstance(v, dict) and "extrinsic_matrix_3x4" in v:
                        m34 = np.array(v["extrinsic_matrix_3x4"], dtype=np.float32)
                        mat = np.eye(4, dtype=np.float32)
                        mat[:3, :4] = m34
                    else:
                        mat = np.array(v, dtype=np.float32)
                    name_to_pose[k] = mat
                    name_to_pose[Path(k).name] = mat
                if image_filenames and any(f in name_to_pose for f in image_filenames):
                    poses = [name_to_pose[f] for f in image_filenames if f in name_to_pose]
                else:
                    poses = list(name_to_pose.values())
            else:
                raise ValueError(f"Unsupported JSON pose format in: {poses_path}")

        elif suffix in [".txt", ".csv"]:
            arr = np.loadtxt(path)
            if arr.shape == (4, 4):
                poses = [arr.astype(np.float32)]
            elif arr.ndim == 2 and arr.shape[1] == 16:
                poses = [row.reshape(4, 4).astype(np.float32) for row in arr]
            elif arr.ndim == 2 and arr.shape[1] == 12:
                # 3x4 [R | t]
                for row in arr:
                    mat = np.eye(4, dtype=np.float32)
                    mat[:3, :4] = row.reshape(3, 4)
                    poses.append(mat)
            else:
                raise ValueError(f"Could not parse poses from text file: {poses_path} with shape {arr.shape}")
        else:
            raise ValueError(f"Unsupported pose file extension: {suffix}")

    elif path.is_dir():
        # Directory containing individual text or npy files, sorted naturally
        files = [f for f in path.iterdir() if f.suffix.lower() in [".txt", ".npy"]]
        files.sort(key=lambda x: natural_sort_key(x.name))
        for f in files:
            if f.suffix.lower() == ".npy":
                mat = np.load(f)
            else:
                mat = np.loadtxt(f)
            if mat.shape == (3, 4):
                m4 = np.eye(4, dtype=np.float32)
                m4[:3, :4] = mat
                mat = m4
            assert mat.shape == (4, 4), f"Pose in {f} has shape {mat.shape}, expected (4, 4)"
            poses.append(mat.astype(np.float32))

    if num_expected is not None and len(poses) != num_expected:
        raise ValueError(f"Number of loaded poses ({len(poses)}) does not match number of images ({num_expected})")

    # Convert convention if needed
    converted_poses = [convert_pose_convention(p, from_convention=pose_convention) for p in poses]
    return converted_poses


def load_intrinsics(
    intrinsics_input: Optional[Union[str, List[np.ndarray], np.ndarray]],
    num_expected: int,
) -> Optional[List[np.ndarray]]:
    """
    Load camera intrinsics if provided.
    Supports .npy, .json, single 3x3 matrix (broadcast to all views), or list of 3x3 matrices.
    """
    if intrinsics_input is None:
        return None

    if isinstance(intrinsics_input, (str, Path)):
        path = Path(intrinsics_input)
        if not path.exists():
            raise FileNotFoundError(f"Intrinsics path not found: {intrinsics_input}")
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
            if arr.ndim == 2 and arr.shape == (3, 3):
                return [arr.astype(np.float32) for _ in range(num_expected)]
            elif arr.ndim == 3 and arr.shape[1:] == (3, 3):
                return [arr[i].astype(np.float32) for i in range(arr.shape[0])]
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
            if isinstance(content, list):
                if len(content) == 3 and len(content[0]) == 3:
                    # Single 3x3
                    k = np.array(content, dtype=np.float32)
                    return [k for _ in range(num_expected)]
                return [np.array(m, dtype=np.float32) for m in content]

    elif isinstance(intrinsics_input, np.ndarray):
        if intrinsics_input.shape == (3, 3):
            return [intrinsics_input.astype(np.float32) for _ in range(num_expected)]
        elif intrinsics_input.ndim == 3 and intrinsics_input.shape[1:] == (3, 3):
            return [intrinsics_input[i].astype(np.float32) for i in range(intrinsics_input.shape[0])]

    elif isinstance(intrinsics_input, list):
        return [np.array(k, dtype=np.float32) for k in intrinsics_input]

    return None


def resolve_output_pointcloud_path(
    output_spec: Optional[Union[str, Path]],
    images_input: Union[str, Path, List[Union[str, Path]]],
    default_filename: str = "pointcloud.ply",
) -> Tuple[Optional[str], Optional[Path]]:
    """
    Resolve final point cloud file path and target folder.
    
    If output_spec is a directory (e.g. 'kaggle/working/output'), nests the input
    folder name into it:
        'kaggle/working/output' + 'AR_KIT' -> 'kaggle/working/output/AR_KIT/pointcloud.ply'
        
    If output_spec is a file path (e.g. 'output/reconstruction.ply'):
        'output' + 'AR_KIT' -> 'output/AR_KIT/reconstruction.ply'
        
    Returns:
        (resolved_file_path_str, resolved_folder_path)
    """
    if output_spec is None:
        return None, None

    # Determine input folder name
    if isinstance(images_input, (str, Path)):
        inp_path = Path(images_input)
        if inp_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            folder_name = inp_path.parent.name
        else:
            folder_name = inp_path.name
    elif isinstance(images_input, list) and len(images_input) > 0:
        first_p = Path(images_input[0])
        if first_p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            folder_name = first_p.parent.name
        else:
            folder_name = first_p.name
    else:
        folder_name = "reconstruction"

    if not folder_name:
        folder_name = "reconstruction"

    out_p = Path(output_spec)

    if out_p.suffix.lower() in [".ply", ".pcd"]:
        fname = out_p.name
        parent_dir = out_p.parent
        if parent_dir.name == folder_name:
            target_dir = parent_dir
        else:
            target_dir = parent_dir / folder_name
        final_file = target_dir / fname
    else:
        if out_p.name == folder_name:
            target_dir = out_p
        else:
            target_dir = out_p / folder_name
        final_file = target_dir / default_filename

    return str(final_file), target_dir


def generate_camera_markers(
    poses: List[np.ndarray],
    radius: float = 0.25,
    sphere_radius: float = 0.06,
    num_ring_points: int = 72,
    num_sphere_points: int = 150,
    ray_length: float = 0.45,
    marker_style: str = "all_rays",
    color_palette: Optional[List[Tuple[int, int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D point cloud markers for cameras:
    - A colored solid sphere at each optical center
    - A circular ring around each camera in the horizontal plane (radius in meters)
    - Forward-pointing sight rays along camera optical axes

    Args:
        poses: List of 4x4 cam2world camera pose matrices.
        radius: Radius in meters of the horizontal orientation ring.
        sphere_radius: Radius in meters of the center station sphere.
        marker_style: Marker style:
            - 'all_rays' (default): Sphere + ring + sight rays for all views at this center.
            - 'station_only': Sphere + ring at each station center (no rays).
            - 'heading_only': Sphere + ring + single ray along the primary forward heading.
    
    Returns:
        (marker_points, marker_colors) as np.ndarray
    """
    if color_palette is None:
        color_palette = [
            (255, 0, 128),   # Bright Magenta/Pink
            (0, 255, 255),   # Bright Cyan
            (255, 220, 0),   # Bright Gold/Yellow
            (0, 255, 128),   # Bright Emerald Green
            (255, 128, 0),   # Bright Orange
            (180, 0, 255),   # Bright Purple
        ]

    marker_pts = []
    marker_rgb = []

    # Cluster/group poses by camera center (optical center)
    # Using 5 cm threshold ensures all nodal faces of a panorama collapse into 1 station
    centers = []
    center_to_views = {}
    for idx, p in enumerate(poses):
        c = p[:3, 3]
        matched_c_idx = None
        for c_idx, existing_c in enumerate(centers):
            if np.linalg.norm(c - existing_c) < 0.05:
                matched_c_idx = c_idx
                break
        if matched_c_idx is None:
            matched_c_idx = len(centers)
            centers.append(c)
            center_to_views[matched_c_idx] = []
        center_to_views[matched_c_idx].append(idx)

    # For each unique camera center (e.g. each parent pano location):
    for c_idx, c in enumerate(centers):
        color = color_palette[c_idx % len(color_palette)]

        # 1. Solid sphere at optical center (Fibonacci sphere)
        phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle
        sphere_pts = []
        for i in range(num_sphere_points):
            y = 1.0 - (i / float(max(1, num_sphere_points - 1))) * 2.0
            r_y = np.sqrt(max(0.0, 1.0 - y * y))
            theta = phi * i
            x = np.cos(theta) * r_y
            z = np.sin(theta) * r_y
            for s in [0.4, 0.7, 1.0]:
                sphere_pts.append(c + s * sphere_radius * np.array([x, y, z], dtype=np.float32))

        sphere_pts = np.array(sphere_pts, dtype=np.float32)
        marker_pts.append(sphere_pts)
        marker_rgb.append(np.tile(color, (len(sphere_pts), 1)).astype(np.uint8))

        # 2. Horizontal circular ring in XZ plane around camera center
        angles = np.linspace(0, 2 * np.pi, num_ring_points, endpoint=False)
        ring_pts = []
        for a in angles:
            ring_pts.append(c + np.array([radius * np.cos(a), 0.0, radius * np.sin(a)], dtype=np.float32))
        ring_pts = np.array(ring_pts, dtype=np.float32)
        marker_pts.append(ring_pts)
        marker_rgb.append(np.tile(color, (len(ring_pts), 1)).astype(np.uint8))

        # 3. Sight rays for views originating at this center
        if marker_style != "station_only":
            views_to_draw = center_to_views[c_idx]
            if marker_style == "heading_only":
                # Single primary forward ray (first view at this center)
                views_to_draw = views_to_draw[:1]

            for view_idx in views_to_draw:
                R = poses[view_idx][:3, :3]
                forward_dir = R[:, 2]  # +Z forward in OpenCV RDF
                num_ray_steps = 25
                ray_pts = []
                for s in np.linspace(sphere_radius, ray_length, num_ray_steps):
                    ray_pts.append(c + s * forward_dir)
                ray_pts = np.array(ray_pts, dtype=np.float32)
                marker_pts.append(ray_pts)
                ray_color = np.clip(np.array(color, dtype=np.float32) * 1.2, 0, 255).astype(np.uint8)
                marker_rgb.append(np.tile(ray_color, (len(ray_pts), 1)).astype(np.uint8))

    if not marker_pts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    return np.concatenate(marker_pts, axis=0), np.concatenate(marker_rgb, axis=0)


def export_point_cloud(
    output_path: str,
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: Optional[float] = None,
    camera_markers: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    """
    Save point cloud to .ply or .pcd format with optional voxel downsampling.

    Args:
        output_path: Output file path (.ply or .pcd)
        points: (N, 3) float32 coordinates
        colors: (N, 3) uint8 RGB colors [0, 255]
        voxel_size: Optional voxel grid downsample size in meters
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Optional voxel downsampling
    if voxel_size is not None and voxel_size > 0:
        print(f"Applying voxel downsampling with voxel size = {voxel_size:.4f} m...")
        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
            points = np.asarray(pcd_down.points).astype(np.float32)
            colors = (np.asarray(pcd_down.colors) * 255.0).clip(0, 255).astype(np.uint8)
            print(f"Point count reduced from {len(pcd.points):,} to {len(points):,} points.")
        except ImportError:
            # Fallback grid voxel hashing with numpy
            coords = np.floor(points / voxel_size).astype(np.int64)
            _, unique_indices = np.unique(coords, axis=0, return_index=True)
            points = points[unique_indices]
            colors = colors[unique_indices]
            print(f"NumPy voxel grid reduced point count to {len(points):,} points.")

    # Append camera position markers (circles, spheres, sight rays) after downsampling
    if camera_markers is not None:
        marker_pts, marker_colors = camera_markers
        if len(marker_pts) > 0:
            points = np.concatenate([points, marker_pts], axis=0)
            colors = np.concatenate([colors, marker_colors], axis=0)
            print(f"Appended {len(marker_pts):,} camera marker points (spheres + rings) to output point cloud.")

    ext = Path(output_path).suffix.lower()

    if ext == ".pcd":
        # Save as PCD
        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)
            print(f"Saved binary PCD point cloud to: {output_path}")
            return
        except ImportError:
            # Write ASCII PCD header & data as fallback
            with open(output_path, "w", encoding="ascii") as f:
                f.write("# .PCD v0.7 - Point Cloud Data\n")
                f.write("VERSION 0.7\n")
                f.write("FIELDS x y z rgb\n")
                f.write("SIZE 4 4 4 4\n")
                f.write("TYPE F F F U\n")
                f.write("COUNT 1 1 1 1\n")
                f.write(f"WIDTH {len(points)}\n")
                f.write("HEIGHT 1\n")
                f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {len(points)}\n")
                f.write("DATA ascii\n")
                rgb_packed = (
                    (colors[:, 0].astype(np.uint32) << 16)
                    | (colors[:, 1].astype(np.uint32) << 8)
                    | (colors[:, 2].astype(np.uint32))
                )
                for pt, rgb_val in zip(points, rgb_packed):
                    f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {rgb_val}\n")
            print(f"Saved ASCII PCD point cloud to: {output_path}")
            return

    # Default to PLY
    try:
        import trimesh

        pc = trimesh.PointCloud(vertices=points, colors=colors)
        pc.export(output_path)
        print(f"Saved PLY point cloud to: {output_path}")
    except Exception as e:
        # Open3D PLY fallback
        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"Saved PLY point cloud to: {output_path}")
        except Exception:
            raise RuntimeError(f"Failed to export PLY point cloud: {e}")


def _reconstruct_with_model_decided_face_poses(
    model,
    device,
    image_paths: List[str],
    image_filenames: List[str],
    loaded_intrinsics: Optional[List[np.ndarray]],
    loaded_poses: List[np.ndarray],
    pano_res: Optional[dict] = None,
    target_output_dir: Optional[Path] = None,
    is_metric_scale: bool = True,
    resolved_amp_dtype: str = "bf16",
    apply_mask: bool = True,
    mask_edges: bool = True,
    use_multiview_confidence: bool = False,
    apply_confidence_mask: bool = False,
    confidence_percentile: float = 10.0,
    minibatch_size: int = 1,
    max_depth: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Per-panorama hierarchical reconstruction:
    1. Groups perspective face views by parent panorama.
    2. Runs MapAnything inference on each panorama's faces unposed, allowing the model
       to predict the relative face poses, metric depth, and intrinsics.
    3. Rigidly anchors the predicted local cameras and points to the known global pose
       of the panorama as a whole:
           T_face_world = T_pano_world @ T_face_local
           P_world = T_pano_world @ P_local
    4. Merges all world point clouds and saves refined camera poses.
    """
    from mapanything.utils.image import preprocess_inputs
    from mapanything.utils.geometry import depthmap_to_world_frame

    # 1. Group face views by parent panorama
    parent_to_faces: Dict[str, List[Tuple[int, str, str]]] = {}
    pano_global_poses: Dict[str, np.ndarray] = {}

    if pano_res is not None and "extracted_views" in pano_res:
        fname_to_idx = {name: idx for idx, name in enumerate(image_filenames)}
        for idx_ev, ev in enumerate(pano_res["extracted_views"]):
            parent = ev.get("parent_pano", "")
            fname = ev.get("image_name", "")
            actual_idx = fname_to_idx.get(fname, idx_ev)
            if actual_idx < len(image_paths):
                parent_to_faces.setdefault(parent, []).append(
                    (actual_idx, image_paths[actual_idx], image_filenames[actual_idx])
                )

        if "level1_metadata_json" in pano_res and Path(pano_res["level1_metadata_json"]).is_file():
            with open(pano_res["level1_metadata_json"], "r", encoding="utf-8") as f:
                l1_list = json.load(f)
            for item in l1_list:
                mat = np.array(item["cam2world_pose_4x4"], dtype=np.float32)
                pano_global_poses[item["image_name"]] = mat
                pano_global_poses[Path(item["image_name"]).stem] = mat
                if item.get("job_id"):
                    pano_global_poses[str(item["job_id"])] = mat
    else:
        # Regex grouping by stem prefix
        for idx, fname in enumerate(image_filenames):
            stem = Path(fname).stem
            parent_stem = re.sub(
                r"(_front|_right|_back|_left|_top|_up|_bottom|_down|_face_\d+|_yaw[-_]?\d+|_pitch[-_]?\d+).*$",
                "",
                stem,
                flags=re.IGNORECASE,
            )
            parent_to_faces.setdefault(parent_stem, []).append((idx, image_paths[idx], fname))

        # Check candidate locations for normalized_arkit_metadata.json
        candidate_meta_dirs = []
        if target_output_dir is not None:
            candidate_meta_dirs.append(target_output_dir)
        if len(image_paths) > 0 and isinstance(image_paths[0], (str, Path)):
            p0 = Path(image_paths[0])
            candidate_meta_dirs.extend([p0.parent, p0.parent.parent])
        for cdir in candidate_meta_dirs:
            meta_f = cdir / "normalized_arkit_metadata.json"
            if meta_f.is_file():
                try:
                    with open(meta_f, "r", encoding="utf-8") as f:
                        l1_list = json.load(f)
                    for item in l1_list:
                        mat = np.array(item["cam2world_pose_4x4"], dtype=np.float32)
                        pano_global_poses[item["image_name"]] = mat
                        pano_global_poses[Path(item["image_name"]).stem] = mat
                        if item.get("job_id"):
                            pano_global_poses[str(item["job_id"])] = mat
                    break
                except Exception:
                    pass

    if not parent_to_faces:
        raise ValueError("Could not group input perspective faces into parent panoramas.")

    # 2. Process each panorama
    all_points = []
    all_colors = []
    refined_world_poses = [None] * len(image_paths)
    refined_poses_dict = {}

    total_panos = len(parent_to_faces)
    if verbose:
        print(f"\n[Model Face Poses] Hierarchical mode: processing {total_panos} panoramas ({len(image_paths)} total face views)...")

    for pano_idx, (parent_name, face_entries) in enumerate(parent_to_faces.items()):
        # Sort face views so that front / yaw 0 view is first (View 0).
        # In MapAnything unposed inference, View 0 defines the coordinate frame origin.
        # Placing the forward face at index 0 aligns View 0 directly with the panorama's heading.
        def _face_priority(entry):
            fname = entry[2].lower()
            if "_front" in fname or "_yaw_0" in fname or "_yaw0" in fname:
                return 0
            elif "_right" in fname or "_yaw_90" in fname or "_yaw90" in fname:
                return 1
            elif "_back" in fname or "_yaw_180" in fname or "_yaw180" in fname:
                return 2
            elif "_left" in fname or "_yaw_270" in fname or "_yaw270" in fname:
                return 3
            elif "_up" in fname or "_pitch_90" in fname:
                return 4
            elif "_down" in fname or "_pitch_-90" in fname:
                return 5
            return 10

        face_entries.sort(key=_face_priority)

        # Determine global parent panorama pose
        if parent_name in pano_global_poses:
            T_pano = pano_global_poses[parent_name]
        elif Path(parent_name).stem in pano_global_poses:
            T_pano = pano_global_poses[Path(parent_name).stem]
        else:
            first_idx = face_entries[0][0]
            if first_idx < len(loaded_poses):
                T_pano = loaded_poses[first_idx].copy()
            else:
                T_pano = np.eye(4, dtype=np.float32)

        if verbose:
            print(f"\n--- Panorama {pano_idx + 1}/{total_panos}: '{parent_name}' ({len(face_entries)} faces) ---")
            print(f"  Parent Anchor World Position: [{T_pano[0, 3]:.3f}, {T_pano[1, 3]:.3f}, {T_pano[2, 3]:.3f}] m")

        # Construct unposed view dicts
        pano_raw_views = []
        for face_idx, img_item, fname in face_entries:
            if isinstance(img_item, (str, Path)):
                pil_img = Image.open(img_item).convert("RGB")
                img_arr = np.array(pil_img, dtype=np.uint8)
            elif isinstance(img_item, Image.Image):
                img_arr = np.array(img_item.convert("RGB"), dtype=np.uint8)
            elif isinstance(img_item, np.ndarray):
                img_arr = img_item.astype(np.uint8) if img_item.dtype != np.uint8 else img_item
            else:
                raise ValueError(f"Unsupported image type: {type(img_item)}")

            v_dict = {
                "img": torch.from_numpy(img_arr),
                "is_metric_scale": torch.tensor([is_metric_scale]),
            }
            if loaded_intrinsics is not None and face_idx < len(loaded_intrinsics):
                v_dict["intrinsics"] = torch.from_numpy(loaded_intrinsics[face_idx])
            pano_raw_views.append(v_dict)

        pano_proc_views = preprocess_inputs(pano_raw_views, verbose=False)

        # Run unposed inference so MapAnything estimates relative face poses
        pano_outputs = model.infer(
            pano_proc_views,
            memory_efficient_inference=True,
            minibatch_size=minibatch_size,
            use_amp=(resolved_amp_dtype != "fp32"),
            amp_dtype=resolved_amp_dtype,
            apply_mask=apply_mask,
            mask_edges=mask_edges,
            use_multiview_confidence=use_multiview_confidence,
            apply_confidence_mask=apply_confidence_mask,
            confidence_percentile=confidence_percentile,
            ignore_calibration_inputs=False if loaded_intrinsics is not None else True,
            ignore_depth_inputs=True,
            ignore_pose_inputs=True,  # Crucial: let model decide face poses!
            ignore_pose_scale_inputs=not is_metric_scale,
        )

        T_pano_torch = torch.from_numpy(T_pano).to(device=device, dtype=torch.float32)

        # Unproject points and anchor to world pose
        pano_points = 0
        for j, (face_idx, _, fname) in enumerate(face_entries):
            pred = pano_outputs[j]
            depth_z = pred["depth_z"][0].squeeze(-1)  # (H, W)
            intrinsics_view = pred["intrinsics"][0]  # (3, 3)
            mask = pred["mask"][0].squeeze(-1).bool()
            img_no_norm = pred["img_no_norm"][0]

            # Model predicted relative face pose in View 0 frame
            T_face_local = pred["camera_poses"][0]  # (4, 4)

            # Orientations: compose parent panorama orientation with model-predicted local face rotation
            R_face_world = torch.matmul(T_pano_torch[:3, :3], T_face_local[:3, :3])

            # Nodal optical center lock:
            # Panoramic perspective faces physically share the exact same optical nodal center (zero baseline).
            # Any non-zero translation predicted by unconstrained multi-view stereo is synthetic drift.
            # Strictly enforce t_face_world = t_pano to eliminate camera clustering and depth tears.
            T_face_world = torch.eye(4, device=device, dtype=torch.float32)
            T_face_world[:3, :3] = R_face_world
            T_face_world[:3, 3] = T_pano_torch[:3, 3]
            T_face_world_np = T_face_world.cpu().numpy()

            refined_world_poses[face_idx] = T_face_world_np
            refined_poses_dict[fname] = T_face_world_np.tolist()

            # Direct metric world unprojection
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
                pano_points += len(pts_np)

        if verbose:
            print(f"  -> Generated {pano_points:,} world 3D points for '{parent_name}'.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_points:
        raise RuntimeError("No valid 3D points produced. Check inputs or loosen mask/confidence thresholds.")

    combined_points = np.concatenate(all_points, axis=0)
    combined_colors = np.concatenate(all_colors, axis=0)

    # Save refined poses
    if target_output_dir is not None:
        target_output_dir.mkdir(parents=True, exist_ok=True)
        valid_poses = [p if p is not None else np.eye(4, dtype=np.float32) for p in refined_world_poses]
        np.save(target_output_dir / "model_refined_poses.npy", np.array(valid_poses, dtype=np.float32))
        with open(target_output_dir / "model_refined_poses.json", "w", encoding="utf-8") as f:
            json.dump(refined_poses_dict, f, indent=2)
        if verbose:
            print(f"\n[Export] Saved model-refined poses to: {target_output_dir / 'model_refined_poses.npy'}")

    valid_refined_poses = [p if p is not None else np.eye(4, dtype=np.float32) for p in refined_world_poses]
    return combined_points, combined_colors, valid_refined_poses


def reconstruct_from_posed_images(
    images: Union[str, Path, List[Union[str, Path, np.ndarray, Image.Image]]],
    poses: Optional[Union[str, Path, List[np.ndarray], np.ndarray]] = None,
    intrinsics: Optional[Union[str, Path, List[np.ndarray], np.ndarray]] = None,
    pose_convention: str = "opencv",
    is_metric_scale: bool = True,
    use_exact_input_poses: bool = True,
    model_decides_face_poses: bool = False,
    model: Optional[object] = None,
    model_name: str = "facebook/map-anything",
    device: Optional[str] = None,
    output_path: Optional[str] = None,
    voxel_size: Optional[float] = None,
    max_depth: Optional[float] = None,
    apply_mask: bool = True,
    mask_edges: bool = True,
    use_multiview_confidence: bool = False,
    apply_confidence_mask: bool = False,
    confidence_percentile: float = 10.0,
    minibatch_size: int = 1,
    chunk_size: Optional[int] = None,
    amp_dtype: str = "auto",
    verbose: bool = True,
    # Pano processing options:
    split_mode: str = "ring12",
    face_size: int = 1024,
    fov_deg: float = 90.0,
    anchor: Union[int, str] = 0,
    heading_mode: str = "north_aligned",
    cache_dir: Optional[Union[str, Path]] = None,
    # Camera visualization options:
    add_camera_markers: bool = True,
    marker_radius: float = 0.25,
    marker_style: str = "all_rays",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run MapAnything inference using input images and their known camera poses,
    and return/export a combined metric 3D point cloud.

    Supports:
    - Auto-detecting raw ARKit equirectangular panoramas and auto-splitting them into perspective views
    - Auto-detecting poses.npy / poses.json and intrinsics.npy inside the input image folder
    - Exact metric cam2world camera poses in OpenCV RDF format
    """
    # 0. Resolve smart output path (e.g. kaggle/working/output/AR_KIT/pointcloud.ply)
    resolved_output_path, target_output_dir = resolve_output_pointcloud_path(
        output_path, images, default_filename="pointcloud.ply"
    )
    if verbose and resolved_output_path is not None:
        print(f"[Output] Target reconstruction path: {resolved_output_path}")

    # 0.1. Auto-detection & routing for raw ARKit panoramas or pre-existing poses
    pano_res = None
    folder_candidate = Path(images if isinstance(images, (str, Path)) else images[0]) if (isinstance(images, (str, Path)) or (isinstance(images, list) and len(images) == 1)) else None
    if folder_candidate and folder_candidate.is_dir():
        # Check 1: Does the folder contain raw ARKit panoramas (*.jpg + *.json)?
        try:
            from scripts.process_arkit_panos import is_arkit_pano_directory, process_arkit_panoramas
        except ModuleNotFoundError:
            from process_arkit_panos import is_arkit_pano_directory, process_arkit_panoramas

        if is_arkit_pano_directory(folder_candidate):
            if verbose:
                print(f"\n[Auto-Route] Detected raw ARKit panoramas in: {folder_candidate}")
                print(f"[Auto-Route] Splitting panoramas into '{split_mode}' perspective views (FoV: {fov_deg}°) with anchor: {anchor}...")

            if cache_dir is not None:
                proc_out_dir = Path(cache_dir)
            elif target_output_dir is not None:
                proc_out_dir = target_output_dir
            else:
                proc_out_dir = Path("output") / folder_candidate.name

            res = process_arkit_panoramas(
                input_dir=folder_candidate,
                output_dir=proc_out_dir,
                split_mode=split_mode,
                face_size=face_size,
                fov_deg=fov_deg,
                heading_mode=heading_mode,
                anchor=anchor,
                verbose=verbose,
            )
            pano_res = res
            images = res["images_dir"]
            if poses is None:
                poses = res["poses_npy"]
            if intrinsics is None:
                intrinsics = res["intrinsics_npy"]

        elif poses is None:
            # Check 2: Does the folder already contain poses.npy or poses.json?
            if (folder_candidate / "poses.npy").is_file():
                poses = str(folder_candidate / "poses.npy")
                if verbose:
                    print(f"[Auto-Route] Auto-detected poses array: {poses}")
                if intrinsics is None and (folder_candidate / "intrinsics.npy").is_file():
                    intrinsics = str(folder_candidate / "intrinsics.npy")
            elif (folder_candidate / "poses.json").is_file():
                poses = str(folder_candidate / "poses.json")
                if verbose:
                    print(f"[Auto-Route] Auto-detected poses JSON: {poses}")
            elif (folder_candidate / "cubemap_poses.json").is_file():
                poses = str(folder_candidate / "cubemap_poses.json")
                if verbose:
                    print(f"[Auto-Route] Auto-detected cubemap poses JSON: {poses}")
            elif (folder_candidate / "jobdata.json").is_file():
                poses = str(folder_candidate / "jobdata.json")
                if verbose:
                    print(f"[Auto-Route] Auto-detected jobdata JSON: {poses}")
            else:
                raise ValueError(
                    f"No --poses provided, and could not find poses.npy, poses.json, jobdata.json, or ARKit panorama pairs in: {folder_candidate}"
                )

    if poses is None:
        raise ValueError(
            "Camera poses must be provided via --poses, or auto-detected in the --images directory."
        )

    # 1. Discover and naturally sort images
    if isinstance(images, (str, Path)) or (isinstance(images, list) and len(images) > 0 and isinstance(images[0], (str, Path))):
        image_paths, image_filenames = find_images(images)
    else:
        image_paths = images
        image_filenames = [f"image_{i:04d}" for i in range(len(images))]

    num_views = len(image_paths)
    if num_views == 0:
        raise ValueError("No images provided for reconstruction")

    if verbose:
        print(f"Loaded {num_views} images in natural numerical order:")
        for idx, fname in enumerate(image_filenames):
            if idx < 4 or idx >= num_views - 2:
                print(f"  [{idx}] {fname}")
            elif idx == 4:
                print(f"  ... ({num_views - 6} more views) ...")

    # 2. Load and validate camera poses
    if isinstance(poses, (str, Path)):
        loaded_poses = load_poses(
            str(poses),
            num_expected=None if model_decides_face_poses else num_views,
            pose_convention=pose_convention,
            image_filenames=image_filenames,
        )
    elif isinstance(poses, np.ndarray):
        if poses.ndim == 3 and poses.shape[1:] == (4, 4):
            loaded_poses = [convert_pose_convention(poses[i], pose_convention) for i in range(poses.shape[0])]
        elif poses.shape == (4, 4):
            loaded_poses = [convert_pose_convention(poses, pose_convention)]
        else:
            raise ValueError(f"Unexpected array shape for poses: {poses.shape}. Expected (N, 4, 4).")
    elif isinstance(poses, list):
        loaded_poses = [convert_pose_convention(p, pose_convention) for p in poses]
    else:
        raise ValueError(f"Unsupported type for poses: {type(poses)}")

    if not model_decides_face_poses and len(loaded_poses) != num_views:
        raise ValueError(f"Mismatch between number of images ({num_views}) and poses ({len(loaded_poses)})")

    # 3. Gather optional camera intrinsics
    loaded_intrinsics = load_intrinsics(intrinsics, num_expected=num_views)
    if loaded_intrinsics is not None and verbose:
        print(f"Loaded {len(loaded_intrinsics)} camera intrinsics matrices.")

    # 4. Lazy load MapAnything model
    from mapanything.models import MapAnything
    from mapanything.utils.device import get_device
    from mapanything.utils.geometry import depthmap_to_world_frame

    if device is None:
        device = get_device()

    if model is None:
        if verbose:
            print(f"Loading MapAnything checkpoint ({model_name}) onto {device}...")
        model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()

    # Determine AMP dtype with automatic SM75 (Turing / T4) fallback to FP16
    resolved_amp_dtype = amp_dtype
    if resolved_amp_dtype == "auto" or resolved_amp_dtype is None:
        if torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability(device)
                if major >= 8 and torch.cuda.is_bf16_supported():
                    resolved_amp_dtype = "bf16"
                else:
                    resolved_amp_dtype = "fp16"  # SM75 (T4) / SM70 (V100) have FP16 Tensor Cores, not BF16
            except Exception:
                resolved_amp_dtype = "fp16"
        else:
            resolved_amp_dtype = "fp32"

    if verbose:
        print(f"Mixed precision configured: AMP={resolved_amp_dtype.upper()} (hardware-matched).")

    outputs = None
    if model_decides_face_poses:
        combined_points, combined_colors, loaded_poses = _reconstruct_with_model_decided_face_poses(
            model=model,
            device=device,
            image_paths=image_paths,
            image_filenames=image_filenames,
            loaded_intrinsics=loaded_intrinsics,
            loaded_poses=loaded_poses,
            pano_res=pano_res,
            target_output_dir=target_output_dir,
            is_metric_scale=is_metric_scale,
            resolved_amp_dtype=resolved_amp_dtype,
            apply_mask=apply_mask,
            mask_edges=mask_edges,
            use_multiview_confidence=use_multiview_confidence,
            apply_confidence_mask=apply_confidence_mask,
            confidence_percentile=confidence_percentile,
            minibatch_size=minibatch_size,
            max_depth=max_depth,
            verbose=verbose,
        )
    else:
        # 5. Construct view dictionaries
        raw_views = []
        for i in range(num_views):
            item = image_paths[i]
            if isinstance(item, (str, Path)):
                pil_img = Image.open(item).convert("RGB")
                img_arr = np.array(pil_img, dtype=np.uint8)
            elif isinstance(item, Image.Image):
                img_arr = np.array(item.convert("RGB"), dtype=np.uint8)
            elif isinstance(item, np.ndarray):
                img_arr = item.astype(np.uint8) if item.dtype != np.uint8 else item
            else:
                raise ValueError(f"Unsupported image item type at index {i}: {type(item)}")

            view_dict = {
                "img": torch.from_numpy(img_arr),  # (H, W, 3) [0, 255]
                "camera_poses": torch.from_numpy(loaded_poses[i]),  # (4, 4) OpenCV cam2world
                "is_metric_scale": torch.tensor([is_metric_scale]),
            }

            if loaded_intrinsics is not None:
                view_dict["intrinsics"] = torch.from_numpy(loaded_intrinsics[i])

            raw_views.append(view_dict)

        # 6. Preprocess inputs (determines aspect ratio, resizes, scales intrinsics, normalizes)
        if verbose:
            print("Preprocessing input views (aspect ratio mapping & normalization)...")
        from mapanything.utils.image import preprocess_inputs
        processed_views = preprocess_inputs(raw_views, verbose=False)

        # 7. Run inference
        # Auto-detect safe chunk size if not explicitly set and running on limited GPU memory
        effective_chunk_size = chunk_size
        if effective_chunk_size is None and len(processed_views) > 6 and torch.cuda.is_available():
            try:
                total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if total_mem_gb <= 16.5:
                    effective_chunk_size = 4
                    if verbose:
                        print(f"Auto-selected chunk_size={effective_chunk_size} (detected {total_mem_gb:.1f} GB GPU memory and {len(processed_views)} views) to prevent CUDA OOM.")
            except Exception:
                pass

        if effective_chunk_size is not None and effective_chunk_size > 0 and effective_chunk_size < len(processed_views):
            if verbose:
                print(f"Running chunked multi-modal inference ({effective_chunk_size} views/chunk across {len(processed_views)} total views)...")
            outputs = []
            num_chunks = (len(processed_views) + effective_chunk_size - 1) // effective_chunk_size
            for chunk_idx in range(num_chunks):
                start = chunk_idx * effective_chunk_size
                end = min(start + effective_chunk_size, len(processed_views))
                chunk_views = processed_views[start:end]
                if verbose:
                    print(f"  Chunk {chunk_idx + 1}/{num_chunks}: views [{start}..{end - 1}] ({len(chunk_views)} views)...")
                chunk_outputs = model.infer(
                    chunk_views,
                    memory_efficient_inference=True,
                    minibatch_size=minibatch_size,
                    use_amp=True,
                    amp_dtype=resolved_amp_dtype,
                    apply_mask=apply_mask,
                    mask_edges=mask_edges,
                    use_multiview_confidence=use_multiview_confidence,
                    apply_confidence_mask=apply_confidence_mask,
                    confidence_percentile=confidence_percentile,
                    ignore_calibration_inputs=False if loaded_intrinsics is not None else True,
                    ignore_depth_inputs=True,
                    ignore_pose_inputs=False,  # Fuse input poses in latent space
                    ignore_pose_scale_inputs=not is_metric_scale,
                )
                outputs.extend(chunk_outputs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            if verbose:
                print("Running feed-forward multi-modal inference with input camera poses...")
            outputs = model.infer(
                processed_views,
                memory_efficient_inference=True,
                minibatch_size=minibatch_size,
                use_amp=True,
                amp_dtype=resolved_amp_dtype,
                apply_mask=apply_mask,
                mask_edges=mask_edges,
                use_multiview_confidence=use_multiview_confidence,
                apply_confidence_mask=apply_confidence_mask,
                confidence_percentile=confidence_percentile,
                ignore_calibration_inputs=False if loaded_intrinsics is not None else True,
                ignore_depth_inputs=True,
                ignore_pose_inputs=False,  # Fuse input poses in latent space
                ignore_pose_scale_inputs=not is_metric_scale,
            )

        if verbose:
            print("Inference completed! Unprojecting and assembling point clouds...")

        # 8. Unproject depth into world coordinates
        all_points = []
        all_colors = []

        for view_idx, pred in enumerate(outputs):
            depth_z = pred["depth_z"][0].squeeze(-1)  # (H, W) in camera frame
            intrinsics_view = pred["intrinsics"][0]  # (3, 3) recovered or resized intrinsics
            mask = pred["mask"][0].squeeze(-1).bool()  # (H, W)
            img_no_norm = pred["img_no_norm"][0]  # (H, W, 3) [0, 1]

            if use_exact_input_poses:
                # Exact user-provided pose
                exact_pose_torch = torch.from_numpy(loaded_poses[view_idx]).to(device=depth_z.device, dtype=depth_z.dtype)
                pts3d_world, valid_depth = depthmap_to_world_frame(depth_z, intrinsics_view, exact_pose_torch)
                valid_mask = mask & valid_depth
            else:
                # Model's predicted pose (in View 0's reference frame)
                pts3d_world = pred["pts3d"][0]
                valid_depth = depth_z > 0
                valid_mask = mask & valid_depth

            # Optional max-depth clipping (eliminates sky/infinity points)
            if max_depth is not None and max_depth > 0:
                valid_mask = valid_mask & (depth_z <= max_depth)

            valid_mask_np = valid_mask.cpu().numpy()
            pts_np = pts3d_world.cpu().numpy()[valid_mask_np]
            rgb_np = (img_no_norm.cpu().numpy()[valid_mask_np] * 255.0).clip(0, 255).astype(np.uint8)

            if len(pts_np) > 0:
                all_points.append(pts_np)
                all_colors.append(rgb_np)

            if verbose:
                print(f"  View {view_idx + 1}/{num_views} ({image_filenames[view_idx]}): {len(pts_np):,} valid 3D points.")

        if not all_points:
            raise RuntimeError("No valid 3D points produced. Check your inputs or loosen mask/confidence/depth thresholds.")

        combined_points = np.concatenate(all_points, axis=0)
        combined_colors = np.concatenate(all_colors, axis=0)

        if verbose:
            print(f"Total points assembled across {num_views} views: {len(combined_points):,}")

    # 9. Export if output path is requested
    if resolved_output_path is not None:
        camera_markers_tuple = None
        if add_camera_markers and loaded_poses:
            marker_poses = loaded_poses
            if not use_exact_input_poses and outputs and "camera_poses" in outputs[0]:
                try:
                    marker_poses = [p["camera_poses"][0].cpu().numpy() for p in outputs]
                except Exception:
                    marker_poses = loaded_poses
            cam_marker_pts, cam_marker_rgb = generate_camera_markers(
                marker_poses, radius=marker_radius, marker_style=marker_style
            )
            camera_markers_tuple = (cam_marker_pts, cam_marker_rgb)

        export_point_cloud(
            output_path=resolved_output_path,
            points=combined_points,
            colors=combined_colors,
            voxel_size=voxel_size,
            camera_markers=camera_markers_tuple,
        )

    return combined_points, combined_colors


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct a combined metric 3D point cloud from images and camera poses using MapAnything."
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to folder containing images or a comma-separated list of image files",
    )
    parser.add_argument(
        "--poses",
        type=str,
        default=None,
        help="Path to pose file (.npy, .npz, .json, .txt) or directory containing per-view pose files. If omitted, auto-detected from --images folder.",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        default=None,
        help="Optional path to intrinsics file (.npy, .json). Recommended for planar, non-distorted geometry.",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="ring12",
        choices=["cubemap", "ring12", "ring24", "ring8"],
        help="If --images contains raw panos, perspective view splitting mode: 'cubemap' (6 faces), 'ring12' (14 views), or 'ring24' (26 views).",
    )
    parser.add_argument(
        "--face_size",
        type=int,
        default=1024,
        help="If --images contains raw panos, pixel resolution of extracted perspective faces (default: 1024).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=100.0,
        help="If --images contains raw panos, field of view in degrees (default: 100.0 for overlapping cubemap faces).",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="0",
        help="If --images contains raw panos, anchor panorama index or filename (default: 0).",
    )
    parser.add_argument(
        "--heading_mode",
        type=str,
        default="north_aligned",
        choices=["north_aligned", "relative", "raw_compass"],
        help="If --images contains raw panos, heading alignment mode (default: 'north_aligned').",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to save generated perspective views when auto-processing panos (default: output/processed_views/<name>).",
    )
    parser.add_argument(
        "--pose_convention",
        type=str,
        default="opencv",
        choices=["opencv", "opengl", "blender", "colmap_w2c"],
        help="Source pose convention (default: 'opencv'). Options: 'opencv', 'opengl', 'blender', 'colmap_w2c'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory (e.g. 'kaggle/working/output') or file path (.ply / .pcd). If a folder is passed, pointcloud is saved to <output>/<input_folder>/pointcloud.ply.",
    )
    parser.add_argument(
        "--is_metric",
        action="store_true",
        default=True,
        help="Specify if camera poses are in true metric scale (default: True). Use --no_metric if poses are scale-ambiguous.",
    )
    parser.add_argument(
        "--no_metric",
        dest="is_metric",
        action="store_false",
        help="Specify if poses are non-metric (e.g. monocular COLMAP)",
    )
    parser.add_argument(
        "--use_predicted_poses",
        dest="use_exact_input_poses",
        action="store_false",
        default=True,
        help="Use network-predicted poses in View 0 frame instead of exact input poses in world frame.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Optional voxel downsampling grid size in meters (e.g. 0.01 for 1cm grid)",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=None,
        help="Optional maximum depth in meters to clip open sky and distant points (e.g. 30.0)",
    )
    parser.add_argument(
        "--use_multiview_confidence",
        action="store_true",
        help="Enable geometric multi-view depth consistency filtering across overlapping views",
    )
    parser.add_argument(
        "--apply_confidence_mask",
        action="store_true",
        help="Enable confidence percentile filtering",
    )
    parser.add_argument(
        "--confidence_percentile",
        type=float,
        default=10.0,
        help="Percentile for confidence masking if enabled (default: 10.0)",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Mixed precision floating point type ('auto', 'fp16', 'bf16', 'fp32'). 'auto' uses FP16 on SM75 (Turing/T4) / SM70 (Volta) and BF16 on SM80+ (Ampere).",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=1,
        help="Minibatch size for memory-efficient dense prediction heads (default: 1)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Max number of views to process per GPU forward pass to prevent CUDA OOM (e.g. 4 or 6). Defaults to auto-detect (4 on <=16GB GPUs).",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model checkpoint (facebook/map-anything-apache)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device ('cuda', 'cpu', or auto-detect)",
    )
    parser.add_argument(
        "--add_camera_markers",
        action="store_true",
        default=True,
        help="Embed 3D camera position spheres and rings into the output PLY (default: True).",
    )
    parser.add_argument(
        "--no_camera_markers",
        dest="add_camera_markers",
        action="store_false",
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
        help="Camera marker visualization style: 'all_rays' (sphere + ring + rays for all views), 'station_only' (sphere + ring at each station), 'heading_only' (sphere + ring + 1 forward heading ray).",
    )
    parser.add_argument(
        "--model_decides_face_poses",
        action="store_true",
        default=False,
        help="Let the MapAnything model estimate the relative camera pose of each perspective face within a panorama, while anchoring the panorama as a whole using the provided global ARKit/GPS pose.",
    )

    args = parser.parse_args()

    # Parse images argument
    if "," in args.images:
        images_input = [s.strip() for s in args.images.split(",")]
    else:
        images_input = args.images

    model_name = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"

    reconstruct_from_posed_images(
        images=images_input,
        poses=args.poses,
        intrinsics=args.intrinsics,
        pose_convention=args.pose_convention,
        is_metric_scale=args.is_metric,
        use_exact_input_poses=args.use_exact_input_poses,
        model_decides_face_poses=args.model_decides_face_poses,
        model_name=model_name,
        device=args.device,
        output_path=args.output,
        voxel_size=args.voxel_size,
        max_depth=args.max_depth,
        use_multiview_confidence=args.use_multiview_confidence,
        apply_confidence_mask=args.apply_confidence_mask,
        confidence_percentile=args.confidence_percentile,
        minibatch_size=args.minibatch_size,
        chunk_size=args.chunk_size,
        amp_dtype=args.amp_dtype,
        split_mode=args.split_mode,
        face_size=args.face_size,
        fov_deg=args.fov,
        anchor=args.anchor,
        heading_mode=args.heading_mode,
        cache_dir=args.cache_dir,
        add_camera_markers=args.add_camera_markers,
        marker_radius=args.marker_radius,
        marker_style=args.marker_style,
    )


if __name__ == "__main__":
    main()
