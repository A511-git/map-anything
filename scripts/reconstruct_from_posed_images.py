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


def export_point_cloud(
    output_path: str,
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: Optional[float] = None,
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


def reconstruct_from_posed_images(
    images: Union[str, Path, List[Union[str, Path, np.ndarray, Image.Image]]],
    poses: Optional[Union[str, Path, List[np.ndarray], np.ndarray]] = None,
    intrinsics: Optional[Union[str, Path, List[np.ndarray], np.ndarray]] = None,
    pose_convention: str = "opencv",
    is_metric_scale: bool = True,
    use_exact_input_poses: bool = True,
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
    heading_mode: str = "relative",
    cache_dir: Optional[Union[str, Path]] = None,
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
    if poses is None and (isinstance(images, (str, Path)) or (isinstance(images, list) and len(images) == 1)):
        folder_candidate = Path(images if isinstance(images, (str, Path)) else images[0])
        if folder_candidate.is_dir():
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
                images = res["images_dir"]
                poses = res["poses_npy"]
                if intrinsics is None:
                    intrinsics = res["intrinsics_npy"]

            # Check 2: Does the folder already contain poses.npy or poses.json?
            elif (folder_candidate / "poses.npy").is_file():
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
            else:
                raise ValueError(
                    f"No --poses provided, and could not find poses.npy, poses.json, or ARKit panorama pairs in: {folder_candidate}"
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
            num_expected=num_views,
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

    if len(loaded_poses) != num_views:
        raise ValueError(f"Mismatch between number of images ({num_views}) and poses ({len(loaded_poses)})")

    # 3. Gather optional camera intrinsics
    loaded_intrinsics = load_intrinsics(intrinsics, num_expected=num_views)
    if loaded_intrinsics is not None and verbose:
        print(f"Loaded {len(loaded_intrinsics)} camera intrinsics matrices.")

    # 4. Construct view dictionaries
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

    # 5. Preprocess inputs (determines aspect ratio, resizes, scales intrinsics, normalizes)
    if verbose:
        print("Preprocessing input views (aspect ratio mapping & normalization)...")
    from mapanything.utils.image import preprocess_inputs
    processed_views = preprocess_inputs(raw_views, verbose=False)

    # 6. Lazy load MapAnything model
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

    # 7. Run inference
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
        export_point_cloud(
            output_path=resolved_output_path,
            points=combined_points,
            colors=combined_colors,
            voxel_size=voxel_size,
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
        default="relative",
        choices=["relative", "north_aligned", "raw_compass"],
        help="If --images contains raw panos, heading alignment mode (default: 'relative').",
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
    )


if __name__ == "__main__":
    main()
