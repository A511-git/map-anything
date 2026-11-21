# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Process ARKit Panoramas into Posed Perspective Images for MapAnything 3D Reconstruction.

Performs two levels of normalization:
1. Level 1 (Pano-level): Transforms ARKit world coordinates (+X East, +Y Up, +Z South)
   into OpenCV RDF cam2world poses, anchoring the first panorama at origin [0, 0, 0]
   with canonical orientation R = Identity (facing +Z).
2. Level 2 (Perspective splitting): Slices each equirectangular panorama into
   rectilinear perspective views (cubemap or high-overlap ring views) and computes
   the exact 4x4 camera pose (sharing the optical center of the parent pano) and
   pinhole camera intrinsics (K).

Outputs:
- Rectilinear perspective images in <output_dir>/images/
- poses.npy: Array of shape (M, 4, 4) in OpenCV RDF cam2world format
- intrinsics.npy: Array of shape (M, 3, 3) in pinhole format
- poses.json / cubemap_poses.json: Human-readable metadata and camera poses
- normalized_arkit_metadata.json: Level 1 relative poses for all parent panoramas
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure mapanything and scripts are discoverable on PYTHONPATH
repo_root = Path(__file__).resolve().parent.parent
scripts_dir = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


class ARKitCaptureRecord(dict):
    """
    Metadata dictionary representing a single ARKit capture record.
    Provides a .name property and .file_path attribute so it behaves
    consistently alongside Path objects in logging and exports.
    """
    def __init__(self, data: dict, file_path: Optional[Path] = None):
        super().__init__(data)
        self.file_path = file_path

    @property
    def name(self) -> str:
        return self.file_path.name if self.file_path else "metadata.json"


def natural_sort_key(s: Union[str, Path]) -> list:
    """Natural numerical sorting key (e.g. frame_2 comes before frame_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def find_pano_json_pairs(
    input_dir: Union[str, Path],
    image_paths: Optional[List[str]] = None,
    json_paths: Optional[List[str]] = None,
) -> List[Tuple[Path, Union[Path, ARKitCaptureRecord]]]:
    """
    Match equirectangular panorama image files with their companion ARKit JSON files or records.
    
    Supports:
    1. Per-image JSON files:
       - Exact match: aSzi0c8BDE.jpg -> aSzi0c8BDE.json
       - Suffix variations: aSzi0c8BDE_blurred.jpg -> aSzi0c8BDE.json
    2. Single aggregate JSON file (e.g. jobdata.json, metadata.json, poses.json):
       - JSON containing a list of records with "jobId" or "id"
       - JSON containing a dict mapping jobId -> record
       - Preserves scanChain.index ordering if present
    3. Manual lists of image_paths and json_paths
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Discover images
    if image_paths is None:
        raw_images = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS and not f.name.startswith(".")
        ]
        raw_images.sort(key=lambda x: natural_sort_key(x.name))
    else:
        raw_images = [Path(p) for p in image_paths]

    # Discover JSONs
    if json_paths is None:
        raw_jsons = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() == ".json" and not f.name.startswith(".")
        ]
    else:
        raw_jsons = [Path(p) for p in json_paths]

    # First: Check for 1-to-1 matching per-image JSON files
    json_dict = {f.stem: f for f in raw_jsons}
    individual_pairs = []
    for img in raw_images:
        stem = img.stem
        # 1. Exact stem match
        if stem in json_dict:
            individual_pairs.append((img, json_dict[stem]))
            continue

        # 2. Match after removing common suffixes like _blurred, _pano, _equirectangular
        matched_json = None
        for j_stem, j_path in json_dict.items():
            if stem.startswith(j_stem) or j_stem.startswith(stem):
                matched_json = j_path
                break
            cleaned_stem = re.sub(r"(_blurred|_pano|_equirectangular|_360)$", "", stem, flags=re.IGNORECASE)
            if cleaned_stem == j_stem:
                matched_json = j_path
                break

        if matched_json is not None:
            individual_pairs.append((img, matched_json))

    # If individual matching JSONs were found for all or most images, return them
    if len(individual_pairs) > 0 and len(individual_pairs) >= len(raw_images) * 0.5:
        individual_pairs.sort(key=lambda pair: natural_sort_key(pair[0].name))
        return individual_pairs

    # Second: Check for aggregate JSON files (e.g. jobdata.json, metadata.json, or any json with list/dict)
    def json_candidate_priority(p: Path) -> int:
        name_lower = p.name.lower()
        if "jobdata" in name_lower:
            return 0
        if "metadata" in name_lower:
            return 1
        if "poses" in name_lower:
            return 2
        return 3

    sorted_candidate_jsons = sorted(raw_jsons, key=json_candidate_priority)

    for j_path in sorted_candidate_jsons:
        try:
            with open(j_path, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception:
            continue

        record_map = {}
        if isinstance(content, list):
            # Sort content by scanChain.index if available and resolve any terminal node whose arKitData is null
            has_chain = any(isinstance(item, dict) and item.get("scanChain", {}).get("index") is not None for item in content)
            if has_chain:
                content = sorted(
                    content,
                    key=lambda x: x.get("scanChain", {}).get("index", 999999) if isinstance(x, dict) else 999999
                )
                # Resolve terminal node whose arKitData is null from the previous node's destination
                for i, item in enumerate(content):
                    if not isinstance(item, dict):
                        continue
                    if item.get("arKitData") is None and i > 0 and isinstance(content[i - 1], dict) and content[i - 1].get("arKitData") is not None:
                        prev_arkit = content[i - 1]["arKitData"]
                        item["arKitData"] = {
                            "origin": prev_arkit.get("destination"),
                            "destination": None,
                            "originPose": prev_arkit.get("destinationPose"),
                            "destinationPose": None,
                            "originHeading": prev_arkit.get("destinationHeading"),
                            "destinationHeading": None,
                            "distanceMeters": 0.0,
                            "horizontalDistanceMeters": 0.0,
                            "isTerminal": True,
                        }

            for item in content:
                if isinstance(item, dict):
                    jid = item.get("jobId") or item.get("id") or item.get("sessionId") or item.get("image_name")
                    if jid:
                        record_map[str(jid)] = item
                        record_map[Path(str(jid)).stem] = item
        elif isinstance(content, dict):
            if "captures" in content and isinstance(content["captures"], list):
                for item in content["captures"]:
                    if isinstance(item, dict):
                        jid = item.get("jobId") or item.get("id") or item.get("image_name")
                        if jid:
                            record_map[str(jid)] = item
                            record_map[Path(str(jid)).stem] = item
            else:
                for k, v in content.items():
                    if isinstance(v, dict):
                        record_map[str(k)] = v
                        record_map[Path(str(k)).stem] = v

        if not record_map:
            continue

        # Match images against record_map
        matched_agg_pairs = []
        for img in raw_images:
            stem = img.stem
            cleaned_stem = re.sub(r"(_blurred|_pano|_equirectangular|_360)$", "", stem, flags=re.IGNORECASE)
            matched_rec = None
            if stem in record_map:
                matched_rec = record_map[stem]
            elif cleaned_stem in record_map:
                matched_rec = record_map[cleaned_stem]
            else:
                for k, v in record_map.items():
                    if stem.startswith(k) or k.startswith(stem) or cleaned_stem == k:
                        matched_rec = v
                        break

            if matched_rec is not None:
                matched_agg_pairs.append((img, ARKitCaptureRecord(matched_rec, file_path=j_path)))
            else:
                print(f"Warning: No matching record found in {j_path.name} for image {img.name}, skipping.")

        if matched_agg_pairs:
            # Check if records contain scanChain index
            has_chain_index = any(
                isinstance(p[1], dict) and p[1].get("scanChain", {}).get("index") is not None
                for p in matched_agg_pairs
            )
            if has_chain_index:
                # Sort primarily by scan chain index, then by filename
                matched_agg_pairs.sort(
                    key=lambda pair: (
                        pair[1].get("scanChain", {}).get("index", 999999) if isinstance(pair[1], dict) else 999999,
                        natural_sort_key(pair[0].name)
                    )
                )
            else:
                matched_agg_pairs.sort(key=lambda pair: natural_sort_key(pair[0].name))
            return matched_agg_pairs

    if individual_pairs:
        individual_pairs.sort(key=lambda pair: natural_sort_key(pair[0].name))
        return individual_pairs

    raise ValueError(f"No valid panorama image + ARKit JSON pairs found in {input_dir}")


def load_arkit_metadata(json_source: Union[str, Path, dict]) -> dict:
    """
    Load and parse ARKit JSON capture metadata from a file path or a dictionary.
    
    Supports:
    - Legacy individual JSON files (with magneticHeading, origin)
    - New aggregate format (with headingAtPlace in arkitCapture.sphere, scanChain)
    - Pre-parsed dictionary records
    """
    if isinstance(json_source, dict):
        data = json_source
        source_name = getattr(json_source, "name", "record")
    elif isinstance(json_source, (str, Path)):
        source_name = str(json_source)
        with open(json_source, "r", encoding="utf-8") as f:
            raw_content = json.load(f)
        if isinstance(raw_content, list) and len(raw_content) > 0:
            data = raw_content[0]
        elif isinstance(raw_content, dict):
            data = raw_content
        else:
            raise ValueError(f"Unsupported content in {json_source}")
    else:
        raise ValueError(f"Unsupported json_source type: {type(json_source)}")

    # Extract origin: prioritize true global ARKit trajectory origin over local sphere placement
    origin = None
    arkit_data = data.get("arKitData")
    if isinstance(arkit_data, dict):
        if "origin" in arkit_data:
            o = arkit_data["origin"]
            if isinstance(o, dict):
                origin = [float(o.get("x", 0.0)), float(o.get("y", 0.0)), float(o.get("z", 0.0))]
            elif isinstance(o, (list, tuple)) and len(o) == 3:
                origin = [float(v) for v in o]

    if origin is None:
        if "origin_arkit_xyz" in data and isinstance(data["origin_arkit_xyz"], (list, tuple)) and len(data["origin_arkit_xyz"]) == 3:
            origin = [float(v) for v in data["origin_arkit_xyz"]]
        elif "origin" in data and isinstance(data["origin"], (list, tuple)) and len(data["origin"]) == 3:
            origin = [float(v) for v in data["origin"]]
        elif "origin" in data and isinstance(data["origin"], dict):
            o = data["origin"]
            origin = [float(o.get("x", 0.0)), float(o.get("y", 0.0)), float(o.get("z", 0.0))]
        elif "position" in data:
            pos = data["position"]
            if isinstance(pos, dict):
                origin = [float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))]
            elif isinstance(pos, (list, tuple)) and len(pos) == 3:
                origin = [float(v) for v in pos]
        elif "arkitCapture" in data and "sphere" in data["arkitCapture"]:
            origin = data["arkitCapture"]["sphere"].get("origin")

    if origin is None or len(origin) != 3:
        raise ValueError(f"Could not extract 3D origin [x, y, z] from {source_name}")

    # Extract heading: check arKitData originHeading, magneticHeading, headingAtPlace, or heading
    heading = None
    if isinstance(arkit_data, dict):
        heading = arkit_data.get("originHeading") if "originHeading" in arkit_data else None
    if heading is None:
        heading = data.get("magnetic_heading_deg")
    if heading is None:
        heading = data.get("magneticHeading")
    if heading is None and "arkitCapture" in data and "sphere" in data["arkitCapture"]:
        heading = data["arkitCapture"]["sphere"].get("headingAtPlace")
    if heading is None:
        heading = data.get("headingAtPlace")
    if heading is None:
        heading = data.get("heading", 0.0)

    # Extract 4x4 ARKit pose matrix if available (column-major simd_float4x4)
    c2w_arkit_4x4 = None
    if isinstance(arkit_data, dict) and "originPose" in arkit_data:
        raw_p = arkit_data["originPose"]
        if isinstance(raw_p, (list, tuple)) and len(raw_p) == 16:
            c2w_arkit_4x4 = np.array(raw_p, dtype=np.float32).reshape((4, 4), order="F")

    # Extract scanChain and heightAboveFloor if present
    scan_chain = data.get("scanChain", {}) if isinstance(data.get("scanChain"), dict) else {}
    chain_index = scan_chain.get("index")
    chain_id = scan_chain.get("chainId")
    height_above_floor = None
    if "arkitCapture" in data and "sphere" in data["arkitCapture"]:
        height_above_floor = data["arkitCapture"]["sphere"].get("heightAboveFloor")

    return {
        "raw": data,
        "origin_arkit": np.array(origin, dtype=np.float32),  # [X_east, Y_up, Z_south]
        "magnetic_heading_deg": float(heading),
        "c2w_arkit_4x4": c2w_arkit_4x4,
        "job_id": data.get("jobId"),
        "scan_chain_index": chain_index,
        "chain_id": chain_id,
        "height_above_floor": float(height_above_floor) if height_above_floor is not None else None,
    }


def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    """
    3x3 rotation matrix for yaw rotation around OpenCV RDF vertical (+Y Down) axis.
    Clockwise rotation looking down along +Y (matching clockwise compass heading).
    """
    rad = np.radians(angle_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    return np.array([
        [cos_a, 0.0, sin_a],
        [0.0, 1.0, 0.0],
        [-sin_a, 0.0, cos_a],
    ], dtype=np.float32)


def is_arkit_pano_directory(path: Union[str, Path]) -> bool:
    """
    Check if a directory contains equirectangular panoramas and companion ARKit JSON files.
    """
    p = Path(path)
    if not p.is_dir():
        return False
    try:
        pairs = find_pano_json_pairs(p)
        return len(pairs) > 0
    except Exception:
        return False


def resolve_anchor_index(anchor_spec: Union[int, str, None], pano_records: List[dict]) -> int:
    """
    Resolve anchor index from integer or filename string. Defaults to 0 if None.
    """
    if anchor_spec is None:
        return 0
    if isinstance(anchor_spec, int):
        if not (0 <= anchor_spec < len(pano_records)):
            raise ValueError(f"Invalid anchor index {anchor_spec}. Expected 0 <= index < {len(pano_records)}.")
        return anchor_spec
    if isinstance(anchor_spec, str):
        if anchor_spec.isdigit():
            idx = int(anchor_spec)
            if not (0 <= idx < len(pano_records)):
                raise ValueError(f"Invalid anchor index {idx}. Expected 0 <= index < {len(pano_records)}.")
            return idx
        # Match by filename or jobId
        for idx, rec in enumerate(pano_records):
            img_name = rec["image_path"].name
            stem = rec["image_path"].stem
            job_id = rec.get("job_id")
            if anchor_spec in (img_name, stem, job_id):
                return idx
        available = [r["image_path"].name for r in pano_records]
        raise ValueError(f"Anchor '{anchor_spec}' not found in panorama files: {available}")
    return 0


def compute_level1_normalized_poses(
    pano_records: List[dict],
    anchor_index: int = 0,
    pano_yaw_offset: float = 0.0,
    no_normalize: bool = False,
) -> List[dict]:
    """
    Level 1 Pose Computation:
    - If no_normalize=False (default):
      Anchors designated Pano at origin [0, 0, 0] with canonical orientation R_base = diag([1, -1, -1]).
      Couples relative translations and rotations relative to Anchor Station.
    - If no_normalize=True:
      Uses direct ARKit world positions [X_east, Y_up, Z_south] and orientations converted to OpenCV RDF
      via canonical R_base = diag([1, -1, -1]) world shift, preserving the exact global ARKit multi-station frame.
    """
    if not pano_records:
        return []

    assert 0 <= anchor_index < len(pano_records), f"Invalid anchor_index {anchor_index}, total: {len(pano_records)}"

    # Anchor pano
    p0_arkit = pano_records[anchor_index]["origin_arkit"]
    h0_deg = pano_records[anchor_index]["magnetic_heading_deg"]

    # Base orientation for OpenCV RDF camera in Y-Up World (facing North):
    # Camera +X (Right) -> World +X (East)
    # Camera +Y (Down) -> World -Y (Down)
    # Camera +Z (Forward) -> World -Z (North)
    # Proper rotation in SO(3): det(R_base) = +1.0
    R_base = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

    normalized = []
    for idx, rec in enumerate(pano_records):
        p_arkit = rec["origin_arkit"]
        h_deg = rec["magnetic_heading_deg"]

        if no_normalize:
            # DIRECT ARKIT DATA:
            # Use raw ARKit world coordinates without anchoring Station 0 to origin.
            t_pano = p_arkit.copy()

            # Convert camera orientation from ARKit convention to OpenCV RDF:
            c2w_arkit = rec.get("c2w_arkit_4x4")
            if c2w_arkit is not None:
                R_arkit = c2w_arkit[:3, :3]
                R_pano = R_arkit @ R_base
                if pano_yaw_offset != 0.0:
                    rad_off = np.radians(pano_yaw_offset)
                    c_off, s_off = np.cos(rad_off), np.sin(rad_off)
                    R_yaw_off = np.array([
                        [c_off, 0.0, -s_off],
                        [0.0, 1.0, 0.0],
                        [s_off, 0.0, c_off],
                    ], dtype=np.float32)
                    R_pano = R_pano @ R_yaw_off
            else:
                h_eff = (h_deg + pano_yaw_offset) % 360.0
                rad_h = np.radians(h_eff)
                c_h, s_h = np.cos(rad_h), np.sin(rad_h)
                R_yaw = np.array([
                    [c_h, 0.0, -s_h],
                    [0.0, 1.0, 0.0],
                    [s_h, 0.0, c_h],
                ], dtype=np.float32)
                R_pano = R_yaw @ R_base

            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R_pano
            c2w[:3, 3] = t_pano

            rec_copy = dict(rec)
            rec_copy["index"] = idx
            rec_copy["is_anchor"] = (idx == anchor_index)
            rec_copy["c2w_pano_4x4"] = c2w
            rec_copy["R_pano_relative"] = R_pano
            rec_copy["t_pano_relative"] = t_pano
            rec_copy["euclidean_dist_from_anchor"] = float(np.linalg.norm(t_pano - p0_arkit))
            normalized.append(rec_copy)
            continue

        # Normalized SE(3) Anchor Frame:
        delta_arkit = p_arkit - p0_arkit  # [delta_X (east), delta_Y (up), delta_Z (south)]

        delta_world = np.array([
            delta_arkit[0],      # East -> +X
            delta_arkit[1],      # Up -> +Y (elevation preserved as positive!)
            delta_arkit[2],      # South -> +Z
        ], dtype=np.float32)

        # Relative heading delta with optional lens offset
        delta_h = (h_deg - h0_deg + pano_yaw_offset) % 360.0
        rad_rel = np.radians(delta_h)
        cos_rel = np.cos(rad_rel)
        sin_rel = np.sin(rad_rel)
        R_rot = np.array([
            [cos_rel, 0.0, -sin_rel],
            [0.0, 1.0, 0.0],
            [sin_rel, 0.0, cos_rel],
        ], dtype=np.float32)
        R_pano = R_rot @ R_base

        # Rotate horizontal translation into Anchor's reference frame (around +Y Up by -h0_deg)
        R_world_to_anchor = rotation_matrix_y(-h0_deg)
        t_pano = R_world_to_anchor @ delta_world

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_pano
        c2w[:3, 3] = t_pano

        rec_copy = dict(rec)
        rec_copy["index"] = idx
        rec_copy["is_anchor"] = (idx == anchor_index)
        rec_copy["c2w_pano_4x4"] = c2w
        rec_copy["R_pano_relative"] = R_pano
        rec_copy["t_pano_relative"] = t_pano
        rec_copy["euclidean_dist_from_anchor"] = float(np.linalg.norm(t_pano))
        normalized.append(rec_copy)

    return normalized


def get_cubemap_faces_definitions() -> List[Tuple[str, float, float]]:
    """
    Returns list of 6 standard cubemap faces as (face_name, yaw_deg, pitch_deg).
    - Front: yaw 0°, pitch 0°
    - Right: yaw +90°, pitch 0°
    - Back: yaw 180°, pitch 0°
    - Left: yaw -90° (270°), pitch 0°
    - Up: pitch -90° (look up at ceiling)
    - Down: pitch +90° (look down at floor)
    """
    return [
        ("front", 0.0, 0.0),
        ("right", 90.0, 0.0),
        ("back", 180.0, 0.0),
        ("left", 270.0, 0.0),
        ("up", 0.0, -90.0),
        ("down", 0.0, 90.0),
    ]


def get_ring_faces_definitions(num_ring_views: int = 12, include_up_down: bool = True) -> List[Tuple[str, float, float]]:
    """
    Returns high-overlap horizontal ring faces spaced evenly around the equator,
    plus optional top (up) and bottom (down) views.
    
    For num_ring_views=12, views are spaced every 30° (0°, 30°, 60°, ..., 330°),
    giving 67% visual overlap at 90° FoV for superior multi-view depth consistency.
    """
    step = 360.0 / num_ring_views
    faces = []
    for k in range(num_ring_views):
        yaw = k * step
        faces.append((f"yaw_{int(yaw):03d}", yaw, 0.0))

    if include_up_down:
        faces.append(("up", 0.0, -90.0))
        faces.append(("down", 0.0, 90.0))

    return faces


def compute_face_rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """
    Computes 3x3 rotation matrix R_face_to_pano mapping from face camera frame
    (+X right, +Y down, +Z forward) into the panorama local frame (+X right, +Y down, +Z forward).

    yaw_deg: clockwise horizontal rotation around +Y Down (0=front, 90=right, 180=back, 270=left).
    pitch_deg: vertical tilt (0=horizontal, -90=up/ceiling, +90=down/floor).
    """
    # Yaw around +Y (Down)
    R_yaw = rotation_matrix_y(yaw_deg)

    # Pitch: in OpenCV RDF (+X right, +Y down, +Z forward),
    # looking UP (pitch < 0) tilts forward +Z into -Y (ceiling).
    # looking DOWN (pitch > 0) tilts forward +Z into +Y (floor).
    pitch_rad = np.radians(pitch_deg)
    cos_p = np.cos(pitch_rad)
    sin_p = np.sin(pitch_rad)
    R_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_p, sin_p],
        [0.0, -sin_p, cos_p],
    ], dtype=np.float32)

    return R_yaw @ R_pitch


def compute_remap_grid(
    R_face_to_pano: np.ndarray,
    face_size: int = 1024,
    pano_w: int = 8192,
    pano_h: int = 4096,
    fov_deg: float = 90.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dense (map_x, map_y) pixel lookup coordinates for cv2.remap from an
    equirectangular panorama (pano_h, pano_w) to a rectilinear perspective image.
    """
    tan_half_fov = np.tan(np.radians(fov_deg) / 2.0)
    u = np.linspace(-tan_half_fov, tan_half_fov, face_size, dtype=np.float32)
    v = np.linspace(-tan_half_fov, tan_half_fov, face_size, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # Ray in face camera frame (+Z forward)
    rays_face = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
    # Rotate rays into panorama frame
    rays_pano = rays_face @ R_face_to_pano.T

    # Normalize to unit sphere
    norm = np.linalg.norm(rays_pano, axis=-1, keepdims=True)
    rays_pano /= np.maximum(norm, 1e-8)

    xp = rays_pano[..., 0]
    yp = rays_pano[..., 1]
    zp = rays_pano[..., 2]

    phi = np.arctan2(xp, zp)                       # Longitude in [-pi, pi]
    theta = np.arcsin(np.clip(-yp, -1.0, 1.0))     # Latitude in [-pi/2, pi/2]

    # Map to equirectangular pixel coordinates
    map_x = ((phi / np.pi) * 0.5 + 0.5) * (pano_w - 1)
    map_y = (0.5 - (theta / np.pi)) * (pano_h - 1)

    return map_x.astype(np.float32), map_y.astype(np.float32)


def resolve_output_dir(output_dir: Union[str, Path], input_dir: Union[str, Path]) -> Path:
    """
    Ensure the input folder name is nested inside output_dir if not already present.
    e.g. 'kaggle/working/output' + 'AR_KIT' -> 'kaggle/working/output/AR_KIT'
    """
    out_p = Path(output_dir)
    inp_p = Path(input_dir)
    folder_name = inp_p.parent.name if inp_p.suffix.lower() in SUPPORTED_IMAGE_EXTS else inp_p.name
    if not folder_name:
        return out_p
    if out_p.name == folder_name:
        return out_p
    return out_p / folder_name


def process_arkit_panoramas(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    split_mode: str = "ring12",
    face_size: int = 1024,
    fov_deg: float = 90.0,
    anchor: Union[int, str] = 0,
    pano_yaw_offset: float = 0.0,
    no_normalize: bool = False,
    jpeg_quality: int = 95,
    verbose: bool = True,
) -> Dict[str, Union[int, str, List[dict]]]:
    """
    Main processing routine for ARKit panoramas:
    1. Discovers and pairs images and JSONs.
    2. Performs Unified Level 1 normalization in Y-Up right-handed world coordinates
       (anchoring designated Pano at origin [0, 0, 0] with canonical orientation).
    3. Performs Level 2 perspective splitting and pose calculation.
    4. Saves images, poses.npy, intrinsics.npy, and JSON metadata with explicit coordinate documentation.
    """
    input_path = Path(input_dir)
    output_path = resolve_output_dir(output_dir, input_path)
    images_out_dir = output_path / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover pairs
    pairs = find_pano_json_pairs(input_path)
    if verbose:
        print(f"Found {len(pairs)} panorama pairs in {input_path}:")
        for img, jf in pairs:
            print(f"  - {img.name} <-> {jf.name}")

    # Load metadata
    records = []
    for img_path, json_source in pairs:
        meta = load_arkit_metadata(json_source)
        meta["image_path"] = img_path
        if isinstance(json_source, Path):
            meta["json_path"] = json_source
        elif hasattr(json_source, "file_path") and json_source.file_path:
            meta["json_path"] = json_source.file_path
        else:
            meta["json_path"] = Path("metadata.json")
        records.append(meta)

    # Resolve anchor
    anchor_idx = resolve_anchor_index(anchor, records)

    # 2. Level 1 Normalization (or direct ARKit poses if no_normalize is set)
    normalized_panos = compute_level1_normalized_poses(
        records,
        anchor_index=anchor_idx,
        pano_yaw_offset=pano_yaw_offset,
        no_normalize=no_normalize,
    )

    if verbose:
        if no_normalize:
            print(f"\n--- Direct ARKit Poses (Unnormalized, using raw ARKit coordinates) ---")
            for p in normalized_panos:
                t = p["t_pano_relative"]
                h = p["magnetic_heading_deg"]
                print(f"Pano [{p['index']}] {p['image_path'].name}:")
                print(f"  Position (ARKit World): [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
                print(f"  Heading:  {h:.1f}°")
        else:
            print(f"\n--- Level 1 Normalization (Anchor: Pano [{anchor_idx}] {records[anchor_idx]['image_path'].name}) ---")
            for p in normalized_panos:
                t = p["t_pano_relative"]
                h = p["magnetic_heading_deg"]
                dist = p["euclidean_dist_from_anchor"]
                tag = " (ANCHOR ORIGIN)" if p.get("is_anchor") else f" (Dist from anchor: {dist:.2f} m)"
                print(f"Pano [{p['index']}] {p['image_path'].name}:")
                print(f"  Position: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m{tag}")
                print(f"  Heading:  {h:.1f}°")

    coord_system_doc = {
        "world_convention": "Y-Up Right-Handed (Standard Robotics / ARKit / Blender / Three.js)",
        "axes_explanation": {
            "+X": "Horizontal Right / East (transversal axis, units: meters, positive = right/east)",
            "+Y": "Vertical Up / Elevation (gravitational axis, units: meters, positive = up toward ceiling, negative = down toward floor)",
            "+Z": "Horizontal South (longitudinal axis, units: meters, positive = south, negative = north/forward)",
        },
        "camera_convention": "OpenCV RDF (+X Right, +Y Down, +Z Optical Axis Forward in meters)",
        "anchor_station": "Direct ARKit World (Unnormalized)" if no_normalize else f"Station {anchor_idx} ({records[anchor_idx]['image_path'].name}) anchored at origin [0, 0, 0] with canonical orientation",
        "normalization": "none (direct ARKit world coordinates)" if no_normalize else "anchored at origin [0,0,0]",
        "units": "meters",
    }

    # Save Level 1 normalized metadata
    level1_meta_file = output_path / "normalized_arkit_metadata.json"
    level1_export = []
    for p in normalized_panos:
        level1_export.append({
            "image_name": p["image_path"].name,
            "json_name": p["json_path"].name if hasattr(p["json_path"], "name") else str(p["json_path"]),
            "job_id": p.get("job_id"),
            "scan_chain_index": p.get("scan_chain_index"),
            "height_above_floor": p.get("height_above_floor"),
            "is_anchor": p.get("is_anchor", False),
            "coordinate_system": coord_system_doc,
            "origin_arkit_xyz": p["origin_arkit"].tolist(),
            "magnetic_heading_deg": p["magnetic_heading_deg"],
            "relative_position_xyz": p["t_pano_relative"].tolist(),
            "cam2world_pose_4x4": p["c2w_pano_4x4"].tolist(),
            "distance_from_anchor_m": p["euclidean_dist_from_anchor"],
        })
    with open(level1_meta_file, "w", encoding="utf-8") as f:
        json.dump(level1_export, f, indent=2)

    # 3. Determine Face Definitions
    if split_mode == "cubemap":
        face_defs = get_cubemap_faces_definitions()
    elif split_mode == "ring12":
        face_defs = get_ring_faces_definitions(num_ring_views=12, include_up_down=True)
    elif split_mode == "ring24":
        face_defs = get_ring_faces_definitions(num_ring_views=24, include_up_down=True)
    elif split_mode == "ring8":
        face_defs = get_ring_faces_definitions(num_ring_views=8, include_up_down=True)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}. Valid: ['cubemap', 'ring12', 'ring24', 'ring8']")

    # 4. Pinhole Camera Intrinsics
    f_len = (face_size / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    c_xy = face_size / 2.0
    K_pinhole = np.array([
        [f_len, 0.0, c_xy],
        [0.0, f_len, c_xy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    # Precompute remap grids for all face definitions
    # Note: Pano resolution is queried from the first image
    first_pano_img = cv2.imread(str(normalized_panos[0]["image_path"]), cv2.IMREAD_COLOR)
    if first_pano_img is None:
        raise IOError(f"Could not load image: {normalized_panos[0]['image_path']}")
    pano_h, pano_w = first_pano_img.shape[:2]

    face_grids = {}
    for face_name, yaw_deg, pitch_deg in face_defs:
        R_face_to_pano = compute_face_rotation_matrix(yaw_deg, pitch_deg)
        map_x, map_y = compute_remap_grid(
            R_face_to_pano=R_face_to_pano,
            face_size=face_size,
            pano_w=pano_w,
            pano_h=pano_h,
            fov_deg=fov_deg,
        )
        face_grids[face_name] = (R_face_to_pano, map_x, map_y)

    # 5. Level 2 Normalization: Slicing and Pose Composition
    extracted_views = []

    if verbose:
        print(f"\n--- Level 2 Normalization: Extracting {len(face_defs)} perspective views per pano ---")

    for p_idx, p_rec in enumerate(normalized_panos):
        img_path = p_rec["image_path"]
        stem = img_path.stem

        if p_idx == 0:
            pano_bgr = first_pano_img
        else:
            pano_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if pano_bgr is None:
                raise IOError(f"Could not load image: {img_path}")

        R_pano_c2w = p_rec["c2w_pano_4x4"][:3, :3]
        t_pano_c2w = p_rec["c2w_pano_4x4"][:3, 3]

        for face_name, yaw_deg, pitch_deg in face_defs:
            R_face_to_pano, map_x, map_y = face_grids[face_name]

            # Remap perspective image
            face_img = cv2.remap(pano_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR)
            out_filename = f"{stem}_{face_name}.jpg"
            out_filepath = images_out_dir / out_filename
            cv2.imwrite(str(out_filepath), face_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

            # Compose camera pose: R_face = R_pano @ R_face_to_pano, t_face = t_pano
            R_face_c2w = R_pano_c2w @ R_face_to_pano
            t_face_c2w = t_pano_c2w.copy()

            c2w_4x4 = np.eye(4, dtype=np.float32)
            c2w_4x4[:3, :3] = R_face_c2w
            c2w_4x4[:3, 3] = t_face_c2w

            extrinsic_3x4 = c2w_4x4[:3, :4].copy()

            extracted_views.append({
                "image_name": out_filename,
                "parent_pano": img_path.name,
                "face_name": face_name,
                "yaw_deg": yaw_deg,
                "pitch_deg": pitch_deg,
                "camera_position_xyz": t_face_c2w.tolist(),
                "extrinsic_matrix_3x4": extrinsic_3x4.tolist(),
                "cam2world_pose_4x4": c2w_4x4.tolist(),
                "c2w_matrix": c2w_4x4,
                "intrinsics": K_pinhole.copy(),
            })

    # Sort strictly in natural/alphabetical order matching sorted(image_paths)
    extracted_views.sort(key=lambda r: natural_sort_key(r["image_name"]))

    # 6. Save Arrays and Metadata Files
    num_views = len(extracted_views)
    poses_array = np.stack([v["c2w_matrix"] for v in extracted_views], axis=0)          # (M, 4, 4)
    intrinsics_array = np.stack([v["intrinsics"] for v in extracted_views], axis=0)      # (M, 3, 3)

    poses_npy_path = output_path / "poses.npy"
    intrinsics_npy_path = output_path / "intrinsics.npy"
    poses_json_path = output_path / "poses.json"
    cubemap_json_path = output_path / "cubemap_poses.json"

    np.save(poses_npy_path, poses_array)
    np.save(intrinsics_npy_path, intrinsics_array)

    json_export = []
    for v in extracted_views:
        R_mat = np.array(v["cam2world_pose_4x4"])[:3, :3]
        json_export.append({
            "image_name": v["image_name"],
            "parent_pano": v["parent_pano"],
            "face_name": v["face_name"],
            "yaw_deg": v["yaw_deg"],
            "pitch_deg": v["pitch_deg"],
            "coordinate_system": coord_system_doc,
            "camera_position_xyz": v["camera_position_xyz"],
            "axes_directions_in_world": {
                "camera_right_+X_axis": [
                    round(float(R_mat[0, 0]), 6),
                    round(float(R_mat[1, 0]), 6),
                    round(float(R_mat[2, 0]), 6),
                ],
                "camera_down_+Y_axis": [
                    round(float(R_mat[0, 1]), 6),
                    round(float(R_mat[1, 1]), 6),
                    round(float(R_mat[2, 1]), 6),
                ],
                "optical_axis_forward_+Z": [
                    round(float(R_mat[0, 2]), 6),
                    round(float(R_mat[1, 2]), 6),
                    round(float(R_mat[2, 2]), 6),
                ],
            },
            "extrinsic_matrix_3x4": v["extrinsic_matrix_3x4"],
            "cam2world_pose_4x4": v["cam2world_pose_4x4"],
        })

    with open(poses_json_path, "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)
    with open(cubemap_json_path, "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)

    if verbose:
        print(f"\nSuccessfully generated {num_views} perspective views!")
        print(f"  Images:      {images_out_dir}")
        print(f"  Poses:       {poses_npy_path} (shape: {poses_array.shape})")
        print(f"  Intrinsics:  {intrinsics_npy_path} (shape: {intrinsics_array.shape})")
        print(f"  Metadata:    {poses_json_path}")

    return {
        "num_views": num_views,
        "output_dir": str(output_path),
        "images_dir": str(images_out_dir),
        "poses_npy": str(poses_npy_path),
        "intrinsics_npy": str(intrinsics_npy_path),
        "poses_json": str(poses_json_path),
        "level1_metadata_json": str(level1_meta_file),
        "extracted_views": extracted_views,
        "views": extracted_views,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process ARKit Panoramas into Posed Perspective Images for MapAnything 3D Reconstruction."
    )
    parser.add_argument(
        "--input", "--images", "--input_dir",
        dest="input_dir",
        type=str,
        required=True,
        help="Directory containing equirectangular panorama images and companion ARKit JSON files.",
    )
    parser.add_argument(
        "--output_dir", "--output",
        dest="output_dir",
        type=str,
        required=True,
        help="Directory to save generated perspective images, poses, and intrinsics.",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="0",
        help="Anchor panorama index or filename (default: 0). Anchored at origin [0, 0, 0] facing +Z.",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="ring12",
        choices=["cubemap", "ring12", "ring24", "ring8"],
        help="Perspective view splitting mode: 'cubemap' (6 faces), 'ring12' (14 views), 'ring24' (26 views), or 'ring8' (10 views).",
    )
    parser.add_argument(
        "--face_size",
        type=int,
        default=1024,
        help="Resolution (width and height in pixels) of each rectilinear perspective view.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="Field of view in degrees for each perspective view (default: 90.0).",
    )
    parser.add_argument(
        "--pano_yaw_offset",
        type=float,
        default=0.0,
        help="Horizontal yaw offset in degrees to apply to panorama orientations (default: 0.0, use 180.0 if front/back lens is inverted).",
    )
    parser.add_argument(
        "--no_normalize", "--no-normalize",
        dest="no_normalize",
        action="store_true",
        default=False,
        help="Use direct ARKit world coordinates and orientation without anchoring Station 0 to origin [0,0,0].",
    )
    parser.add_argument(
        "--run_reconstruction",
        action="store_true",
        help="Automatically run MapAnything reconstruction after processing.",
    )
    parser.add_argument(
        "--recon_output",
        type=str,
        default=None,
        help="Output point cloud path (.ply or .pcd) if --run_reconstruction is enabled.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Voxel downsample size in meters for reconstruction (default: 0.01).",
    )
    parser.add_argument(
        "--dense",
        "--full_dense",
        dest="dense",
        action="store_true",
        default=False,
        help="Export full dense point cloud without voxel downsampling (bypasses --voxel_size).",
    )

    args = parser.parse_args()

    results = process_arkit_panoramas(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_mode=args.split_mode,
        face_size=args.face_size,
        fov_deg=args.fov,
        pano_yaw_offset=args.pano_yaw_offset,
        anchor=args.anchor,
        no_normalize=args.no_normalize,
    )

    if args.run_reconstruction:
        try:
            from scripts.reconstruct_from_posed_images import reconstruct_from_posed_images
        except ModuleNotFoundError:
            from reconstruct_from_posed_images import reconstruct_from_posed_images

        recon_out = args.recon_output
        if recon_out is None:
            recon_out = str(Path(results["output_dir"]) / "pointcloud.ply")

        print(f"\nLaunching MapAnything Reconstruction -> {recon_out}...")
        reconstruct_from_posed_images(
            images=results["images_dir"],
            poses=results["poses_npy"],
            intrinsics=results["intrinsics_npy"],
            output_path=recon_out,
            voxel_size=None if args.dense else args.voxel_size,
            dense=args.dense,
            use_multiview_confidence=True,
        )


if __name__ == "__main__":
    main()
