#!/usr/bin/env python3
"""
Reconstruct from Posed Images v2 (Pure ARKit Debugger Pipeline)

A clean, modular 3D point cloud reconstruction tool built from scratch for ARKit panorama datasets.

Core Architecture:
1. ARKit Pose Handler: Extracts clean trajectory poses using:
   - '3d_distance': Direct distanceMeters along 3D step vector
   - 'horizontal_distance': horizontalDistanceMeters (flattening vertical elevation drift)
   - 'sphere_drift': Incorporating physical ARKit capture sphere lens offsets [dx, dy, dz]
2. Cubemap Slicing: Extracts 6 perspective cubemap faces at 100° FoV (10° overlap).
3. MapAnything Model Estimation:
   - Evaluates perspective faces locally per panorama (ignore_pose_inputs=True)
   - Model determines relative face poses and depth maps
   - Rigidly anchors local predictions to parent panorama's ARKit world pose
   - Enforces optical nodal center lock (t_face = t_pano)
4. Interactive 3D Camera Pose Markers:
   - Colored Fibonacci spheres at station centers
   - Horizontal orientation rings (XZ plane)
   - Sight rays along the optical axes color-coded sequentially along the trajectory
5. Output Export:
   - Exports clean, high-performance binary .ply point clouds with optional voxel downsampling or full dense point cloud (--dense)
   - Supports --all_modes flag to generate comparison point clouds across all 3 distance modes side-by-side
"""

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
import torch
from PIL import Image

try:
    from scripts.arkit_pose_handler_v2 import (
        compute_panorama_poses,
        find_jobdata_and_images,
        get_dataset_name,
        resolve_output_path,
        resolve_scan_chain_records,
    )
    from scripts.split_pano_v2 import (
        compute_face_rotation_matrix,
        compute_pinhole_intrinsics,
        compute_remap_grid,
        get_cubemap_face_definitions,
        split_all_panoramas,
        split_panorama_to_cubemap,
    )
except ModuleNotFoundError:
    from arkit_pose_handler_v2 import (
        compute_panorama_poses,
        find_jobdata_and_images,
        resolve_scan_chain_records,
    )
    from split_pano_v2 import (
        compute_face_rotation_matrix,
        compute_pinhole_intrinsics,
        compute_remap_grid,
        get_cubemap_face_definitions,
        split_all_panoramas,
        split_panorama_to_cubemap,
    )


# Color palette for sequentially color-coding camera stations along trajectory
TRAJECTORY_PALETTE = [
    (255, 0, 128),    # 0: Bright Magenta/Pink (Station 0)
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
    Generate 3D point cloud markers for cameras:
    - A colored solid Fibonacci sphere at each station optical center
    - A circular ring around each camera in the horizontal plane (XZ)
    - Forward-pointing sight rays along camera optical axes

    Args:
        poses: List of 4x4 cam2world camera pose matrices (OpenCV RDF format).
        radius: Radius in meters of the horizontal orientation ring.
        sphere_radius: Radius in meters of the center station sphere.
        marker_style: 'all_rays' | 'station_only' | 'heading_only'

    Returns:
        (marker_points, marker_colors) as np.ndarray
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


def export_point_cloud(
    output_path: str,
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: Optional[float] = None,
    camera_markers: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    dense: bool = False,
    verbose: bool = True,
) -> None:
    """
    Save 3D point cloud to .ply format with optional voxel downsampling.
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if dense:
        voxel_size = None
        if verbose:
            print(f"Full dense mode: preserving all {len(points):,} points without downsampling.")

    # Apply voxel downsampling if requested
    if voxel_size is not None and voxel_size > 0:
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
        except ImportError:
            coords = np.floor(points / voxel_size).astype(np.int64)
            _, unique_indices = np.unique(coords, axis=0, return_index=True)
            points = points[unique_indices]
            colors = colors[unique_indices]
            if verbose:
                print(f"NumPy voxel grid reduced point count to {len(points):,} points.")

    # Append camera markers after downsampling
    if camera_markers is not None:
        m_pts, m_colors = camera_markers
        if len(m_pts) > 0:
            points = np.concatenate([points, m_pts], axis=0)
            colors = np.concatenate([colors, m_colors], axis=0)
            if verbose:
                print(f"Appended {len(m_pts):,} camera marker points to output point cloud.")

    # Ensure .ply extension
    if not output_path.lower().endswith(".ply"):
        output_path = os.path.splitext(output_path)[0] + ".ply"

    # Save to PLY format (Open3D binary or clean ASCII fallback)
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)
        if verbose:
            print(f"Saved binary PLY point cloud to: {output_path}")
    except Exception:
        # Fallback ASCII PLY
        with open(output_path, "w", encoding="ascii") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for pt, col in zip(points, colors):
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {int(col[0])} {int(col[1])} {int(col[2])}\n")
        if verbose:
            print(f"Saved ASCII PLY point cloud to: {output_path}")


def reconstruct_panorama_dataset(
    input_path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
    output_path: Union[str, Path] = "output/reconstruction_v2",
    distance_mode: str = "3d_distance",
    use_direct_origin: bool = False,
    no_normalize_rotation: bool = False,
    face_size: int = 1024,
    fov_deg: float = 100.0,
    voxel_size: Optional[float] = None,
    dense: bool = False,
    max_depth: Optional[float] = 25.0,
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
    dry_run: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Main reconstruction pipeline:
    1. Parse ARKit metadata and compute panorama poses.
    2. Extract 6 cubemap perspective faces per panorama.
    3. Run MapAnything local unposed inference on each panorama's faces.
    4. Rigidly anchor local points and cameras to panorama pose with optical nodal center lock.
    5. Render camera markers and export point cloud.
    """
    final_ply_file, target_dir = resolve_output_path(output_path, input_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse ARKit Metadata & Pano Trajectory
    raw_records, images, jobdata_file = find_jobdata_and_images(input_path, metadata_path=metadata_path)
    if verbose:
        print(f"\n[ARKit] Metadata source: {jobdata_file}")
    resolved_records = resolve_scan_chain_records(raw_records, images)
    pano_poses = compute_panorama_poses(
        resolved_records,
        distance_mode=distance_mode,
        use_direct_origin=use_direct_origin,
        normalize_to_station0=True,
        no_normalize_rotation=no_normalize_rotation,
    )

    if verbose:
        print(f"\n[ARKit] Loaded {len(pano_poses)} panorama stations (Mode: {distance_mode}):")
        for p in pano_poses:
            pos_str = f"[{p['position_xyz'][0]:6.2f}, {p['position_xyz'][1]:6.2f}, {p['position_xyz'][2]:6.2f}]"
            print(f"  Station {p['index']:<2} ({p['jobId']}): Pos={pos_str} Heading={p['heading_deg']:5.1f}° Dist={p['distanceMeters']:.2f}m")

    # Save debug manifest
    json_export = [{k: v for k, v in p.items() if k != "c2w_matrix"} for p in pano_poses]
    with open(target_dir / "poses_debug.json", "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)

    # 2. Extract Cubemap Perspective Faces
    cubemap_cache_dir = target_dir / "cubemap_faces"
    split_res = split_all_panoramas(
        images_or_dir=[p["image_path"] for p in pano_poses if p["image_path"]],
        output_dir=cubemap_cache_dir,
        face_size=face_size,
        fov_deg=fov_deg,
        verbose=verbose,
    )
    parent_to_faces = split_res["parent_to_faces"]

    # If dry_run is enabled, generate camera marker point cloud and skip neural inference
    if dry_run:
        if verbose:
            print("\n[Dry-Run] Bypassing neural network inference. Generating camera trajectory markers...")
        marker_poses = [p["c2w_matrix"] for p in pano_poses]
        m_pts, m_colors = generate_camera_markers(
            marker_poses,
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
        if verbose:
            print(f"[Dry-Run Complete] Saved camera trajectory point cloud to: {final_ply_file}")
        return m_pts, m_colors, marker_poses

    # 3. Initialize MapAnything Model
    from mapanything.models import MapAnything
    from mapanything.utils.device import get_device
    from mapanything.utils.geometry import depthmap_to_world_frame
    from mapanything.utils.image import preprocess_inputs
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

    # 4. Hierarchical Inference & World Point Cloud Assembly
    all_points = []
    all_colors = []
    all_refined_face_poses = []

    total_panos = len(pano_poses)
    if verbose:
        print(f"\n[Inference] Processing {total_panos} panoramas with model-estimated face poses...")

    for p_idx, p_rec in enumerate(pano_poses):
        job_id = p_rec["jobId"]
        img_name = p_rec["image_name"]
        stem = Path(img_name).stem
        T_pano = p_rec["c2w_matrix"]  # (4, 4) in OpenCV RDF

        # Look up 6 extracted faces for this panorama
        faces = parent_to_faces.get(stem) or parent_to_faces.get(job_id)
        if not faces:
            # Fallback: search by partial stem
            for k, v in parent_to_faces.items():
                if stem.startswith(k) or k.startswith(stem):
                    faces = v
                    break

        if not faces or len(faces) != 6:
            if verbose:
                print(f"Warning: Panorama {job_id} does not have exactly 6 faces, skipping.")
            continue

        if verbose:
            pos = p_rec["position_xyz"]
            print(f"\n--- Panorama {p_idx + 1}/{total_panos}: '{job_id}' ---")
            print(f"  Anchor World Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")

        # Construct unposed view inputs
        pano_raw_views = []
        for face in faces:
            pano_raw_views.append({
                "img": torch.from_numpy(face["image"]),
                "intrinsics": torch.from_numpy(face["intrinsics"]),
                "is_metric_scale": torch.tensor([True]),
            })

        pano_proc_views = preprocess_inputs(pano_raw_views, verbose=False)

        # Run unposed inference on this panorama's 6 faces
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
                ignore_depth_inputs=True,
                ignore_pose_inputs=True,  # Crucial: let model decide face poses
                ignore_pose_scale_inputs=False,
            )

        T_pano_torch = torch.from_numpy(T_pano).to(device=device, dtype=torch.float32)

        # Unproject depth and anchor to world frame
        pano_point_count = 0
        for face_idx, face in enumerate(faces):
            pred = pano_outputs[face_idx]
            depth_z = pred["depth_z"][0].squeeze(-1)       # (H, W) in camera frame
            intrinsics_view = pred["intrinsics"][0]        # (3, 3)
            mask = pred["mask"][0].squeeze(-1).bool()
            img_no_norm = pred["img_no_norm"][0]

            # Model predicted relative face pose in View 0 frame
            T_face_local = pred["camera_poses"][0]         # (4, 4)

            # Rigidly compose orientation: R_face_world = R_pano @ R_face_local
            R_face_world = torch.matmul(T_pano_torch[:3, :3], T_face_local[:3, :3])

            # Optical Nodal Center Lock:
            # Sliced cubemap faces physically share the exact same optical center.
            # Enforce t_face_world = t_pano to eliminate synthetic drift.
            T_face_world = torch.eye(4, device=device, dtype=torch.float32)
            T_face_world[:3, :3] = R_face_world
            T_face_world[:3, 3] = T_pano_torch[:3, 3]

            all_refined_face_poses.append(T_face_world.cpu().numpy())

            # Unproject depth map to world frame
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
                pano_point_count += len(pts_np)

        if verbose:
            print(f"  -> Generated {pano_point_count:,} world 3D points for '{job_id}'.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_points:
        raise RuntimeError("No valid 3D points produced across all panoramas.")

    combined_points = np.concatenate(all_points, axis=0)
    combined_colors = np.concatenate(all_colors, axis=0)

    if verbose:
        print(f"\n[Complete] Total 3D points generated: {len(combined_points):,}")

    # 5. Generate Camera Markers & Export
    camera_markers_tuple = None
    if add_camera_markers and pano_poses:
        marker_poses = [p["c2w_matrix"] for p in pano_poses]
        m_pts, m_colors = generate_camera_markers(
            marker_poses,
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
        description="Reconstruct from Posed Images v2: Pure ARKit Debugger Pipeline using MapAnything."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to folder containing panorama images and jobdata.json (e.g. 'jobdata').",
    )
    parser.add_argument(
        "--metadata", "--jobdata",
        dest="metadata",
        type=str,
        default=None,
        help="Optional explicit path to jobdata.json / metadata JSON if stored in a different directory from images.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/reconstruction_v2",
        help="Output directory or point cloud file (.ply). Default: 'output/reconstruction_v2'.",
    )
    parser.add_argument(
        "--distance_mode",
        type=str,
        default="3d_distance",
        choices=["3d_distance", "horizontal_distance", "sphere_drift"],
        help="Distance trajectory mode: '3d_distance', 'horizontal_distance', or 'sphere_drift'.",
    )
    parser.add_argument(
        "--all_modes",
        action="store_true",
        default=False,
        help="Run reconstruction for all 3 distance modes side-by-side for debugger comparison.",
    )
    parser.add_argument(
        "--use_direct_origin",
        action="store_true",
        default=False,
        help="Use raw arKitData.origin points instead of dead-reckoning trajectory.",
    )
    parser.add_argument(
        "--no_normalize_rotation",
        action="store_true",
        default=False,
        help="Keep global ARKit world rotation instead of anchoring Station 0 to Identity.",
    )
    parser.add_argument(
        "--face_size",
        type=int,
        default=1024,
        help="Resolution in pixels of cubemap perspective faces (default: 1024).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=100.0,
        help="Field of view in degrees (default: 100.0 for 10° overlap).",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Optional voxel downsampling grid size in meters (e.g. 0.01 for 1cm grid).",
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
        help="Fast debugging mode: parses ARKit trajectory and exports camera markers PLY without running heavy neural inference.",
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

    args = parser.parse_args()
    model_name = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"

    if args.all_modes:
        modes = ["3d_distance", "horizontal_distance", "sphere_drift"]
        _, base_target_dir = resolve_output_path(args.output, args.input)
        print(f"\n[Debugger] Running all 3 distance modes: {modes} under '{base_target_dir}'...")
        for mode in modes:
            mode_ply_file, mode_target_dir = resolve_output_path(args.output, args.input, subfolder=mode)
            print(f"\n{'='*70}\n[Debugger Mode] Executing mode: {mode.upper()} -> {mode_target_dir}\n{'='*70}")
            reconstruct_panorama_dataset(
                input_path=args.input,
                metadata_path=args.metadata,
                output_path=mode_ply_file,
                distance_mode=mode,
                use_direct_origin=args.use_direct_origin,
                no_normalize_rotation=args.no_normalize_rotation,
                face_size=args.face_size,
                fov_deg=args.fov,
                voxel_size=args.voxel_size,
                dense=args.dense,
                max_depth=args.max_depth,
                use_multiview_confidence=args.use_multiview_confidence,
                apply_confidence_mask=args.apply_confidence_mask,
                confidence_percentile=args.confidence_percentile,
                minibatch_size=args.minibatch_size,
                device=args.device,
                model_name=model_name,
                add_camera_markers=args.add_camera_markers,
                marker_radius=args.marker_radius,
                marker_style=args.marker_style,
                dry_run=args.dry_run,
                verbose=True,
            )
        print(f"\n[Debugger Complete] All 3 modes exported successfully under: {base_target_dir}")

    else:
        final_file, target_dir = resolve_output_path(args.output, args.input)
        reconstruct_panorama_dataset(
            input_path=args.input,
            metadata_path=args.metadata,
            output_path=final_file,
            distance_mode=args.distance_mode,
            use_direct_origin=args.use_direct_origin,
            no_normalize_rotation=args.no_normalize_rotation,
            face_size=args.face_size,
            fov_deg=args.fov,
            voxel_size=args.voxel_size,
            dense=args.dense,
            max_depth=args.max_depth,
            use_multiview_confidence=args.use_multiview_confidence,
            apply_confidence_mask=args.apply_confidence_mask,
            confidence_percentile=args.confidence_percentile,
            minibatch_size=args.minibatch_size,
            device=args.device,
            model_name=model_name,
            add_camera_markers=args.add_camera_markers,
            marker_radius=args.marker_radius,
            marker_style=args.marker_style,
            dry_run=args.dry_run,
            verbose=True,
        )


if __name__ == "__main__":
    main()
