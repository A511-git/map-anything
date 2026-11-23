#!/usr/bin/env python3
"""
ARKit Pose Handler v2
Pure, clean extraction and trajectory computation from ARKit panorama scan data.

Supports:
- Auto-parsing jobdata.json and matching with panorama image files
- Scan chain index sorting (1..N) and automatic terminal node resolution
- Three distance / drift modes:
    1. '3d_distance': Uses direct 3D distanceMeters along vector
    2. 'horizontal_distance': Uses horizontalDistanceMeters (flattening vertical drift)
    3. 'sphere_drift': Incorporates physical ARKit sphere lens offset [dx, dy, dz]
- Direct ARKit rotation from simd_float4x4 column-major matrix
- Conversion to OpenCV RDF (+X Right, +Y Down, +Z Forward)
- Relative Station 0 anchoring (default) or raw global ARKit world coordinates
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def natural_sort_key(s: Union[str, Path]) -> list:
    """Natural numerical sorting key (e.g. frame_2 comes before frame_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def find_jobdata_and_images(
    input_path: Union[str, Path]
) -> Tuple[List[Dict[str, Any]], List[Path], Path]:
    """
    Locate jobdata.json and associated panorama image files in input_path.

    Returns:
        (records, image_paths, jobdata_file_path)
    """
    input_p = Path(input_path)
    if not input_p.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_p.is_file() and input_p.suffix.lower() == ".json":
        jobdata_file = input_p
        search_dir = input_p.parent
    elif input_p.is_dir():
        candidates = [
            input_p / "jobdata.json",
            input_p / "metadata.json",
            input_p / "poses.json",
        ]
        jobdata_file = next((c for c in candidates if c.is_file()), None)
        if jobdata_file is None:
            # Check any json in folder
            json_files = [f for f in input_p.glob("*.json")]
            if json_files:
                jobdata_file = json_files[0]
            else:
                raise FileNotFoundError(f"No jobdata.json or metadata JSON found in: {input_p}")
        search_dir = input_p
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    with open(jobdata_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        if "captures" in raw_data and isinstance(raw_data["captures"], list):
            raw_data = raw_data["captures"]
        elif "poses" in raw_data and isinstance(raw_data["poses"], list):
            raw_data = raw_data["poses"]
        else:
            raw_data = [raw_data]

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected a JSON list of records in {jobdata_file}")

    # Discover images
    images = [
        f for f in search_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS and not f.name.startswith(".")
    ]
    images.sort(key=lambda x: natural_sort_key(x.name))

    return raw_data, images, jobdata_file


def resolve_scan_chain_records(
    raw_records: List[Dict[str, Any]],
    image_paths: List[Path],
) -> List[Dict[str, Any]]:
    """
    Sort records by scanChain.index (1..N) and resolve any terminal node where
    arKitData is null using the previous node's destination data.
    """
    # Build image lookup by stem
    img_by_stem = {img.stem: img for img in image_paths}
    img_by_cleaned_stem = {
        re.sub(r"(_blurred|_pano|_equirectangular|_360)$", "", img.stem, flags=re.IGNORECASE): img
        for img in image_paths
    }

    # Sort records by scan chain index
    sorted_records = sorted(
        raw_records,
        key=lambda x: x.get("scanChain", {}).get("index", 999999) if isinstance(x, dict) else 999999
    )

    resolved: List[Dict[str, Any]] = []

    for i, item in enumerate(sorted_records):
        if not isinstance(item, dict):
            continue

        rec = dict(item)
        jid = str(rec.get("jobId") or f"station_{i}")
        rec["jobId"] = jid

        # Match corresponding image file
        matched_img = None
        if jid in img_by_stem:
            matched_img = img_by_stem[jid]
        elif jid in img_by_cleaned_stem:
            matched_img = img_by_cleaned_stem[jid]
        else:
            # Fallback by index if images and records align
            if i < len(image_paths):
                matched_img = image_paths[i]

        rec["image_path"] = matched_img
        rec["image_name"] = matched_img.name if matched_img else f"{jid}.jpg"

        # Resolve terminal node
        arkit = rec.get("arKitData")
        if arkit is None and i > 0:
            prev_arkit = resolved[i - 1].get("arKitData")
            if prev_arkit:
                rec["arKitData"] = {
                    "origin": prev_arkit.get("destination"),
                    "destination": None,
                    "vector": None,
                    "distanceMeters": 0.0,
                    "horizontalDistanceMeters": 0.0,
                    "originPose": prev_arkit.get("destinationPose"),
                    "destinationPose": None,
                    "originHeading": prev_arkit.get("destinationHeading"),
                    "destinationHeading": None,
                    "isTerminal": True,
                }
                rec["isTerminal"] = True

        resolved.append(rec)

    return resolved


def parse_arkit_rotation_matrix(origin_pose_list: Optional[List[float]], heading_deg: Optional[float] = None) -> np.ndarray:
    """
    Extract 3x3 camera rotation matrix from ARKit simd_float4x4 (column-major)
    and convert to OpenCV RDF camera convention (+X Right, +Y Down, +Z Forward).

    In ARKit:
      Camera frame is OpenGL/Metal convention (+X Right, +Y Up, -Z Forward).
    In OpenCV RDF:
      Camera frame is (+X Right, +Y Down, +Z Forward).
      Transformation: R_opencv = R_arkit @ diag([1, -1, -1])
    """
    if origin_pose_list is not None and len(origin_pose_list) == 16:
        # ARKit stores simd_float4x4 in column-major order ('F')
        c2w_arkit = np.array(origin_pose_list, dtype=np.float32).reshape((4, 4), order="F")
        R_arkit = c2w_arkit[:3, :3]
        # Camera convention conversion: flip Y and Z camera axes
        M_cam = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        R_opencv = R_arkit @ M_cam
        return R_opencv

    # Fallback to heading angle if pose matrix not available
    h = float(heading_deg) if heading_deg is not None else 0.0
    rad = np.radians(h)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    # Rotation around Y axis in OpenCV RDF (+Y Down):
    R = np.array([
        [cos_a, 0.0, sin_a],
        [0.0, 1.0, 0.0],
        [-sin_a, 0.0, cos_a],
    ], dtype=np.float32)
    return R


def compute_panorama_poses(
    records: List[Dict[str, Any]],
    distance_mode: str = "3d_distance",
    use_direct_origin: bool = False,
    normalize_to_station0: bool = True,
    no_normalize_rotation: bool = False,
) -> List[Dict[str, Any]]:
    """
    Compute 4x4 cam2world camera poses for all panoramas in OpenCV RDF format.

    Args:
        records: List of resolved scanChain records
        distance_mode: '3d_distance' | 'horizontal_distance' | 'sphere_drift'
        use_direct_origin: If True, uses arKitData.origin directly instead of dead-reckoning
        normalize_to_station0: If True, translates trajectory so Station 0 is at (0, 0, 0)
        no_normalize_rotation: If True, preserves absolute ARKit world rotation;
                               if False (default), anchors Station 0 to Identity (facing +Z)

    Returns:
        List of dicts with calculated poses, positions, and debug metadata
    """
    valid_modes = {"3d_distance", "horizontal_distance", "sphere_drift"}
    if distance_mode not in valid_modes:
        raise ValueError(f"Unknown distance_mode '{distance_mode}'. Choose from: {valid_modes}")

    num_stations = len(records)
    if num_stations == 0:
        return []

    # 1. Compute raw Rotations for each station
    raw_rotations = []
    for rec in records:
        arkit = rec.get("arKitData") or {}
        pose_list = arkit.get("originPose")
        heading = arkit.get("originHeading")
        R_cv = parse_arkit_rotation_matrix(pose_list, heading_deg=heading)
        raw_rotations.append(R_cv)

    # Base rotation normalization (anchor station 0 to Identity if requested)
    R0 = raw_rotations[0]
    final_rotations = []
    for R in raw_rotations:
        if no_normalize_rotation:
            final_rotations.append(R)
        else:
            # Relative rotation: R_rel = R0^T @ R (so station 0 has R_rel = Identity)
            R_rel = R0.T @ R
            final_rotations.append(R_rel)

    # 2. Compute Station Positions
    positions = []

    if use_direct_origin:
        # Direct Origin Mode: extract arKitData.origin [x, y, z] directly
        for i, rec in enumerate(records):
            arkit = rec.get("arKitData") or {}
            orig = arkit.get("origin")
            if orig is not None:
                if isinstance(orig, dict):
                    pos = np.array([float(orig.get("x", 0.0)), float(orig.get("y", 0.0)), float(orig.get("z", 0.0))], dtype=np.float32)
                else:
                    pos = np.array([float(orig[0]), float(orig[1]), float(orig[2])], dtype=np.float32)
            else:
                pos = positions[-1].copy() if positions else np.zeros(3, dtype=np.float32)

            if distance_mode == "horizontal_distance":
                # Flatten vertical elevation drift relative to station 0
                if i == 0:
                    y0 = pos[1]
                pos[1] = y0

            elif distance_mode == "sphere_drift":
                # Add sphere capture offset [dx, dy, dz]
                sph = (rec.get("arkitCapture") or {}).get("sphere", {}).get("origin")
                if sph and isinstance(sph, (list, tuple)) and len(sph) == 3:
                    pos = pos + np.array([float(sph[0]), float(sph[1]), float(sph[2])], dtype=np.float32)

            positions.append(pos)

    else:
        # Sequential Dead-Reckoning Mode (Default):
        # Station 0 is at (0, 0, 0).
        # Station i+1 is placed at pos_i + direction * distance
        current_pos = np.zeros(3, dtype=np.float32)
        positions.append(current_pos.copy())

        for i in range(num_stations - 1):
            rec = records[i]
            arkit = rec.get("arKitData") or {}

            # Direction vector from origin to destination
            vec_data = arkit.get("vector")
            orig_data = arkit.get("origin")
            dest_data = arkit.get("destination")

            vec = None
            if vec_data is not None:
                if isinstance(vec_data, dict):
                    vec = np.array([float(vec_data.get("x", 0.0)), float(vec_data.get("y", 0.0)), float(vec_data.get("z", 0.0))], dtype=np.float32)
                else:
                    vec = np.array([float(vec_data[0]), float(vec_data[1]), float(vec_data[2])], dtype=np.float32)
            elif orig_data is not None and dest_data is not None:
                ox = float(orig_data.get("x", 0.0) if isinstance(orig_data, dict) else orig_data[0])
                oy = float(orig_data.get("y", 0.0) if isinstance(orig_data, dict) else orig_data[1])
                oz = float(orig_data.get("z", 0.0) if isinstance(orig_data, dict) else orig_data[2])
                dx = float(dest_data.get("x", 0.0) if isinstance(dest_data, dict) else dest_data[0])
                dy = float(dest_data.get("y", 0.0) if isinstance(dest_data, dict) else dest_data[1])
                dz = float(dest_data.get("z", 0.0) if isinstance(dest_data, dict) else dest_data[2])
                vec = np.array([dx - ox, dy - oy, dz - oz], dtype=np.float32)

            dist_3d = float(arkit.get("distanceMeters") or (np.linalg.norm(vec) if vec is not None else 0.0))
            dist_horiz = float(arkit.get("horizontalDistanceMeters") or dist_3d)

            # Determine step vector
            if vec is not None and np.linalg.norm(vec) > 1e-6:
                if distance_mode == "horizontal_distance":
                    # Step purely in horizontal XZ plane
                    vec_xz = np.array([vec[0], 0.0, vec[2]], dtype=np.float32)
                    norm_xz = np.linalg.norm(vec_xz)
                    step = (vec_xz / norm_xz) * dist_horiz if norm_xz > 1e-6 else np.zeros(3, dtype=np.float32)
                else:  # "3d_distance" and base step for "sphere_drift"
                    norm_3d = np.linalg.norm(vec)
                    step = (vec / norm_3d) * dist_3d
            else:
                # Heading-based dead reckoning fallback
                heading = float(arkit.get("originHeading") or 0.0)
                rad = np.radians(heading)
                # Heading clockwise from -Z towards +X
                dir_x = np.sin(rad)
                dir_z = -np.cos(rad)
                d = dist_horiz if distance_mode == "horizontal_distance" else dist_3d
                step = np.array([dir_x * d, 0.0, dir_z * d], dtype=np.float32)

            current_pos = current_pos + step
            positions.append(current_pos.copy())

        # If sphere_drift mode, apply each station's physical capture sphere offset [dx, dy, dz]
        if distance_mode == "sphere_drift":
            for i in range(num_stations):
                sph = (records[i].get("arkitCapture") or {}).get("sphere", {}).get("origin")
                if sph and isinstance(sph, (list, tuple)) and len(sph) == 3:
                    positions[i] = positions[i] + np.array([float(sph[0]), float(sph[1]), float(sph[2])], dtype=np.float32)

    # If normalize_to_station0 and direct origin used, shift origin to 0
    if normalize_to_station0 and use_direct_origin and len(positions) > 0:
        p0 = positions[0].copy()
        positions = [p - p0 for p in positions]

    # Rotate positions into Station 0 reference frame if relative rotation was applied
    if not no_normalize_rotation and len(positions) > 0:
        positions = [R0.T @ p for p in positions]

    # 3. Assemble 4x4 cam2world matrices
    output_records = []
    for idx, (rec, R, pos) in enumerate(zip(records, final_rotations, positions)):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos

        arkit = rec.get("arKitData") or {}
        out_item = {
            "index": idx,
            "jobId": rec.get("jobId"),
            "image_name": rec.get("image_name"),
            "image_path": str(rec.get("image_path")) if rec.get("image_path") else None,
            "scan_chain_index": rec.get("scanChain", {}).get("index", idx + 1),
            "distance_mode": distance_mode,
            "position_xyz": [round(float(v), 5) for v in pos],
            "distanceMeters": float(arkit.get("distanceMeters") or 0.0),
            "horizontalDistanceMeters": float(arkit.get("horizontalDistanceMeters") or 0.0),
            "heading_deg": float(arkit.get("originHeading") or 0.0),
            "isTerminal": rec.get("isTerminal", False),
            "c2w_pose_4x4": c2w.tolist(),
            "c2w_matrix": c2w,
        }
        output_records.append(out_item)

    return output_records


def main():
    parser = argparse.ArgumentParser(
        description="ARKit Pose Handler v2: Extract and compute panorama camera poses from ARKit metadata."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to folder containing jobdata.json and images, or path to jobdata.json itself.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/debug_poses",
        help="Directory to save poses.npy and poses_debug.json (default: output/debug_poses).",
    )
    parser.add_argument(
        "--distance_mode",
        type=str,
        default="3d_distance",
        choices=["3d_distance", "horizontal_distance", "sphere_drift"],
        help="Distance calculation mode: '3d_distance', 'horizontal_distance', or 'sphere_drift'.",
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

    args = parser.parse_args()

    raw_records, images, jobdata_file = find_jobdata_and_images(args.input)
    print(f"Loaded {len(raw_records)} records from {jobdata_file.name} with {len(images)} matched image files.")

    resolved_records = resolve_scan_chain_records(raw_records, images)
    poses = compute_panorama_poses(
        resolved_records,
        distance_mode=args.distance_mode,
        use_direct_origin=args.use_direct_origin,
        normalize_to_station0=True,
        no_normalize_rotation=args.no_normalize_rotation,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save poses.npy (N, 4, 4)
    pose_arr = np.stack([p["c2w_matrix"] for p in poses], axis=0)
    np.save(out_dir / "poses.npy", pose_arr)

    # Save poses_debug.json
    json_export = [{k: v for k, v in p.items() if k != "c2w_matrix"} for p in poses]
    with open(out_dir / "poses_debug.json", "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)

    print(f"\n--- ARKit Trajectory Table (Mode: {args.distance_mode}) ---")
    print(f"{'Idx':<4} {'JobId':<12} {'ScanIdx':<8} {'Position [X, Y, Z] (m)':<28} {'Dist (m)':<10} {'HDist (m)':<10} {'Heading':<8}")
    print("-" * 88)
    for p in poses:
        pos_str = f"[{p['position_xyz'][0]:6.2f}, {p['position_xyz'][1]:6.2f}, {p['position_xyz'][2]:6.2f}]"
        print(f"{p['index']:<4} {p['jobId']:<12} {p['scan_chain_index']:<8} {pos_str:<28} {p['distanceMeters']:<10.2f} {p['horizontalDistanceMeters']:<10.2f} {p['heading_deg']:<8.1f}°")

    print(f"\nSaved {len(poses)} panorama poses to: {out_dir / 'poses.npy'} (shape {pose_arr.shape})")
    print(f"Saved debug manifest to: {out_dir / 'poses_debug.json'}")


if __name__ == "__main__":
    main()
