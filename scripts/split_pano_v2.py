#!/usr/bin/env python3
"""
Cubemap Splitter v2
Pure, clean equirectangular panorama to 6-face perspective cubemap extractor.

Extracts 6 rectilinear perspective views per panorama:
- front (yaw 0°, pitch 0°)
- right (yaw 90°, pitch 0°)
- back (yaw 180°, pitch 0°)
- left (yaw 270°, pitch 0°)
- up (pitch -90°, looking at ceiling)
- down (pitch +90°, looking at floor)

Features:
- Configurable FoV (default 100.0° for 10° visual overlap between adjacent faces)
- Precomputes cv2.remap grids once per resolution for maximum speed
- Generates exact 3x3 pinhole camera intrinsics K
- Outputs extracted images and intrinsics.npy
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def get_dataset_name(input_path: Union[str, Path]) -> str:
    """
    Extract the clean dataset directory name from an input path or file.
    E.g. '/kaggle/input/custom-mapanything-data/AR_kit' -> 'AR_kit'
         'jobdata/jobdata.json' -> 'jobdata'
    """
    p = Path(str(input_path).rstrip("/\\"))
    if p.suffix.lower() in [".json", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic", ".heif"]:
        name = p.parent.name
    else:
        name = p.name
    return name if name else "reconstruction"


def natural_sort_key(s: Union[str, Path]) -> list:
    """Natural numerical sorting key (e.g. frame_2 comes before frame_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def get_cubemap_face_definitions() -> List[Tuple[str, float, float]]:
    """
    Returns the 6 standard cubemap faces as (face_name, yaw_deg, pitch_deg).
    Order ensures 'front' (yaw 0) is index 0.
    """
    return [
        ("front", 0.0, 0.0),
        ("right", 90.0, 0.0),
        ("back", 180.0, 0.0),
        ("left", 270.0, 0.0),
        ("up", 0.0, -90.0),
        ("down", 0.0, 90.0),
    ]


def compute_pinhole_intrinsics(face_size: int = 1024, fov_deg: float = 100.0) -> np.ndarray:
    """
    Compute 3x3 pinhole camera intrinsics matrix K:
        f = (face_size / 2) / tan(fov / 2)
        c_x = c_y = face_size / 2
    """
    f = (face_size / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    c = face_size / 2.0
    return np.array([
        [f, 0.0, c],
        [0.0, f, c],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def compute_face_rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """
    3x3 rotation matrix R_face_to_pano mapping face camera rays (+Z forward)
    into panorama spherical frame (+X right, +Y down, +Z forward).
    """
    # Yaw around +Y Down (clockwise)
    yaw_rad = np.radians(yaw_deg)
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    R_yaw = np.array([
        [cos_y, 0.0, sin_y],
        [0.0, 1.0, 0.0],
        [-sin_y, 0.0, cos_y],
    ], dtype=np.float32)

    # Pitch around +X Right (+Y Down convention: -pitch tilts up to ceiling, +pitch tilts down to floor)
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
    fov_deg: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute dense pixel coordinate maps (map_x, map_y) for cv2.remap.
    """
    tan_half_fov = np.tan(np.radians(fov_deg) / 2.0)
    u = np.linspace(-tan_half_fov, tan_half_fov, face_size, dtype=np.float32)
    v = np.linspace(-tan_half_fov, tan_half_fov, face_size, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # Unit rays in face camera frame (+Z forward)
    rays_face = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
    # Rotate rays into panorama frame
    rays_pano = rays_face @ R_face_to_pano.T

    # Normalize to unit sphere
    norm = np.linalg.norm(rays_pano, axis=-1, keepdims=True)
    rays_pano /= np.maximum(norm, 1e-8)

    xp = rays_pano[..., 0]
    yp = rays_pano[..., 1]
    zp = rays_pano[..., 2]

    # Spherical coordinates
    phi = np.arctan2(xp, zp)                       # Longitude in [-pi, pi]
    theta = np.arcsin(np.clip(-yp, -1.0, 1.0))     # Latitude in [-pi/2, pi/2]

    # Map to equirectangular pixel coordinates
    map_x = ((phi / np.pi) * 0.5 + 0.5) * (pano_w - 1)
    map_y = (0.5 - (theta / np.pi)) * (pano_h - 1)

    return map_x.astype(np.float32), map_y.astype(np.float32)


def split_panorama_to_cubemap(
    pano_image_or_path: Union[str, Path, np.ndarray],
    output_dir: Optional[Union[str, Path]] = None,
    face_size: int = 1024,
    fov_deg: float = 100.0,
    jpeg_quality: int = 95,
    precomputed_grids: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> List[Dict[str, Any]]:
    """
    Split a single equirectangular panorama into 6 perspective cubemap faces.

    Returns:
        List of dicts per face:
        [{
            "face_name": "front",
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "image": np.ndarray (RGB),
            "image_path": Optional[str],
            "intrinsics": np.ndarray (3, 3),
        }, ...]
    """
    if isinstance(pano_image_or_path, (str, Path)):
        img_p = Path(pano_image_or_path)
        stem = img_p.stem
        pano_bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if pano_bgr is None:
            raise IOError(f"Could not load panorama image: {img_p}")
    elif isinstance(pano_image_or_path, np.ndarray):
        stem = "pano"
        pano_bgr = pano_image_or_path
        if pano_bgr.ndim == 3 and pano_bgr.shape[2] == 3:
            # Assume RGB if numpy, convert to BGR for cv2
            pano_bgr = cv2.cvtColor(pano_bgr, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported pano_image_or_path type: {type(pano_image_or_path)}")

    pano_h, pano_w = pano_bgr.shape[:2]
    face_defs = get_cubemap_face_definitions()
    K = compute_pinhole_intrinsics(face_size=face_size, fov_deg=fov_deg)

    out_p = Path(output_dir) if output_dir else None
    if out_p:
        out_p.mkdir(parents=True, exist_ok=True)

    # Compute remap grids if not precomputed
    if precomputed_grids is None:
        grids = {}
        for face_name, yaw, pitch in face_defs:
            R = compute_face_rotation_matrix(yaw, pitch)
            mx, my = compute_remap_grid(R, face_size=face_size, pano_w=pano_w, pano_h=pano_h, fov_deg=fov_deg)
            grids[face_name] = (mx, my)
    else:
        grids = precomputed_grids

    extracted = []
    for face_name, yaw, pitch in face_defs:
        mx, my = grids[face_name]
        face_bgr = cv2.remap(pano_bgr, mx, my, interpolation=cv2.INTER_LINEAR)
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        saved_path = None
        if out_p:
            fname = f"{stem}_{face_name}.jpg"
            saved_path = str(out_p / fname)
            cv2.imwrite(saved_path, face_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        extracted.append({
            "face_name": face_name,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "image": face_rgb,
            "image_path": saved_path,
            "image_name": f"{stem}_{face_name}.jpg",
            "intrinsics": K.copy(),
        })

    return extracted


def split_all_panoramas(
    images_or_dir: Union[str, Path, List[Union[str, Path]]],
    output_dir: Union[str, Path],
    face_size: int = 1024,
    fov_deg: float = 100.0,
    jpeg_quality: int = 95,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Batch process all panoramas in a directory or list into cubemap faces.
    Saves extracted face images in <output_dir>/images/ and intrinsics in <output_dir>/intrinsics.npy.
    """
    out_base = Path(output_dir)
    images_out_dir = out_base / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    # Discover images
    if isinstance(images_or_dir, (str, Path)):
        p = Path(images_or_dir)
        if p.is_dir():
            pano_files = [f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTS and not f.name.startswith(".")]
            pano_files.sort(key=lambda x: natural_sort_key(x.name))
        else:
            pano_files = [p]
    elif isinstance(images_or_dir, list):
        pano_files = [Path(x) for x in images_or_dir]
    else:
        raise ValueError(f"Invalid images_or_dir type: {type(images_or_dir)}")

    if not pano_files:
        raise ValueError(f"No valid panorama images found in: {images_or_dir}")

    if verbose:
        print(f"Splitting {len(pano_files)} panoramas into 6 cubemap faces (FoV: {fov_deg}°, Face Size: {face_size}px)...")

    # Read first image to obtain resolution and precompute remap grids
    first_img = cv2.imread(str(pano_files[0]), cv2.IMREAD_COLOR)
    if first_img is None:
        raise IOError(f"Could not load first image: {pano_files[0]}")
    pano_h, pano_w = first_img.shape[:2]

    face_defs = get_cubemap_face_definitions()
    grids = {}
    for face_name, yaw, pitch in face_defs:
        R = compute_face_rotation_matrix(yaw, pitch)
        mx, my = compute_remap_grid(R, face_size=face_size, pano_w=pano_w, pano_h=pano_h, fov_deg=fov_deg)
        grids[face_name] = (mx, my)

    all_faces = []
    parent_to_faces: Dict[str, List[Dict[str, Any]]] = {}

    for i, pano_p in enumerate(pano_files):
        if verbose:
            print(f"  [{i+1}/{len(pano_files)}] Splitting {pano_p.name}...")

        faces = split_panorama_to_cubemap(
            pano_image_or_path=pano_p,
            output_dir=images_out_dir,
            face_size=face_size,
            fov_deg=fov_deg,
            jpeg_quality=jpeg_quality,
            precomputed_grids=grids,
        )
        for f in faces:
            f["parent_pano"] = pano_p.name
            f["parent_stem"] = pano_p.stem
        all_faces.extend(faces)
        parent_to_faces[pano_p.stem] = faces

    # Save intrinsics.npy
    intrinsics_array = np.stack([f["intrinsics"] for f in all_faces], axis=0)
    intrinsics_npy_path = out_base / "intrinsics.npy"
    np.save(intrinsics_npy_path, intrinsics_array)

    # Save manifest
    manifest_list = [
        {
            "image_name": f["image_name"],
            "image_path": f["image_path"],
            "parent_pano": f["parent_pano"],
            "face_name": f["face_name"],
            "yaw_deg": f["yaw_deg"],
            "pitch_deg": f["pitch_deg"],
            "intrinsics": f["intrinsics"].tolist(),
        }
        for f in all_faces
    ]
    manifest_path = out_base / "cubemap_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_list, f, indent=2)

    if verbose:
        print(f"\nExtracted {len(all_faces)} total cubemap faces!")
        print(f"  Images saved to:     {images_out_dir}")
        print(f"  Intrinsics saved to: {intrinsics_npy_path} (shape {intrinsics_array.shape})")
        print(f"  Manifest saved to:   {manifest_path}")

    return {
        "num_panos": len(pano_files),
        "total_faces": len(all_faces),
        "output_dir": str(out_base),
        "images_dir": str(images_out_dir),
        "intrinsics_npy": str(intrinsics_npy_path),
        "manifest_json": str(manifest_path),
        "all_faces": all_faces,
        "parent_to_faces": parent_to_faces,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cubemap Splitter v2: Slices 360 equirectangular panoramas into 6 perspective cubemap faces."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to folder containing panorama images or path to a single panorama file.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/cubemap_faces",
        help="Directory to save extracted face images and intrinsics (default: output/cubemap_faces).",
    )
    parser.add_argument(
        "--face_size",
        type=int,
        default=1024,
        help="Resolution in pixels (width and height) of extracted square perspective faces (default: 1024).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=100.0,
        help="Field of view in degrees (default: 100.0 for 10° overlap between adjacent faces).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG output quality (1-100, default: 95).",
    )

    args = parser.parse_args()
    out_dir = Path(str(args.output).rstrip("/\\"))
    dataset_name = get_dataset_name(args.input)
    if dataset_name.lower() not in [part.lower() for part in out_dir.parts]:
        out_dir = out_dir / dataset_name
    split_all_panoramas(
        images_or_dir=args.input,
        output_dir=out_dir,
        face_size=args.face_size,
        fov_deg=args.fov,
        jpeg_quality=args.quality,
        verbose=True,
    )


if __name__ == "__main__":
    main()
