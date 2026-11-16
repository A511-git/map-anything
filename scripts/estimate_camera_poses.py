# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Estimate camera poses (and optional camera intrinsics) from a folder of images
using MapAnything's feed-forward multi-view transformer.

Outputs:
- poses.npy: Array of shape (N, 4, 4) containing OpenCV cam2world (+X Right, +Y Down, +Z Forward) poses relative to View 0
- intrinsics.npy (optional): Array of shape (N, 3, 3) containing recovered pinhole camera intrinsics
- poses.json (optional): Human-readable JSON mapping of filenames to pose matrices

Usage Examples:
    # CLI:
    python scripts/estimate_camera_poses.py \\
        --images path/to/images/ \\
        --output path/to/poses.npy \\
        --save_intrinsics path/to/intrinsics.npy

    # Python:
    from scripts.estimate_camera_poses import estimate_camera_poses
    poses, intrinsics, file_names = estimate_camera_poses(
        images="path/to/images/",
        output_path="output/poses.npy",
        save_intrinsics_path="output/intrinsics.npy",
    )
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Ensure mapanything is discoverable on PYTHONPATH
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch


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


def estimate_camera_poses(
    images: Union[str, Path, List[Union[str, Path]]],
    output_path: Optional[str] = "output/poses.npy",
    save_intrinsics_path: Optional[str] = None,
    save_json: bool = False,
    model: Optional[object] = None,
    model_name: str = "facebook/map-anything",
    device: Optional[str] = None,
    minibatch_size: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Estimate camera poses and intrinsics for a collection of images using MapAnything.

    Args:
        images: Folder containing images or list of image file paths
        output_path: Path to save estimated poses as .npy (shape N, 4, 4)
        save_intrinsics_path: Optional path to save estimated camera intrinsics as .npy (shape N, 3, 3)
        save_json: Whether to also save a JSON file with image-to-pose mapping
        model: Pre-loaded MapAnything model instance (optional)
        model_name: HuggingFace model checkpoint (facebook/map-anything or facebook/map-anything-apache)
        device: Device to use ('cuda', 'cpu', or auto-detect)
        minibatch_size: Minibatch size for dense head execution
        verbose: Print progress logs

    Returns:
        Tuple of (poses_array, intrinsics_array, image_filenames)
            - poses_array: (N, 4, 4) float32 array in OpenCV cam2world convention
            - intrinsics_array: (N, 3, 3) float32 array of recovered pinhole intrinsics
            - image_filenames: List of N image filenames in natural sort order
    """
    # 1. Discover and naturally sort images
    image_paths, image_filenames = find_images(images)
    num_views = len(image_paths)

    if verbose:
        print(f"Discovered {num_views} images in natural numerical order:")
        for idx, fname in enumerate(image_filenames):
            if idx < 5 or idx >= num_views - 2:
                print(f"  [{idx}] {fname}")
            elif idx == 5:
                print(f"  ... ({num_views - 7} more views) ...")

    # 2. Lazy imports to avoid top-level package overhead
    from mapanything.models import MapAnything
    from mapanything.utils.device import get_device
    from mapanything.utils.image import load_images

    if device is None:
        device = get_device()
    if verbose:
        print(f"Using compute device: {device}")

    # 3. Load MapAnything model if not provided
    if model is None:
        if verbose:
            print(f"Loading MapAnything checkpoint: {model_name}...")
        model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()

    # 4. Load and preprocess input views
    if verbose:
        print("Preprocessing input views for model inference...")
    # Pass ordered list of filepaths to load_images
    views = load_images(image_paths, verbose=False)

    # 5. Run inference in image-only mode
    if verbose:
        print(f"Running feed-forward pose estimation on {num_views} views...")
    outputs = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=minibatch_size,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=False,
        mask_edges=False,
    )

    # 6. Extract estimated camera poses and intrinsics
    poses_list = []
    intrinsics_list = []

    for idx, pred in enumerate(outputs):
        # camera_poses: (1, 4, 4) tensor in OpenCV RDF cam2world relative to View 0
        pose_mat = pred["camera_poses"][0].cpu().numpy().astype(np.float32)
        poses_list.append(pose_mat)

        # intrinsics: (1, 3, 3) recovered pinhole camera intrinsics
        if "intrinsics" in pred:
            k_mat = pred["intrinsics"][0].cpu().numpy().astype(np.float32)
            intrinsics_list.append(k_mat)

    poses_arr = np.stack(poses_list, axis=0)  # (N, 4, 4)
    intrinsics_arr = np.stack(intrinsics_list, axis=0) if intrinsics_list else None  # (N, 3, 3)

    if verbose:
        print(f"Successfully estimated camera poses for {num_views} views.")
        print(f"View 0 Pose (Reference Frame Origin):\n{poses_arr[0]}")
        if num_views > 1:
            print(f"View 1 Camera Translation: {poses_arr[1, :3, 3]}")

    # 7. Save outputs
    if output_path is not None:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.save(output_path, poses_arr)
        if verbose:
            print(f"Saved camera poses (shape {poses_arr.shape}) to: {output_path}")

    if save_intrinsics_path is not None and intrinsics_arr is not None:
        intrin_dir = os.path.dirname(save_intrinsics_path)
        if intrin_dir:
            os.makedirs(intrin_dir, exist_ok=True)
        np.save(save_intrinsics_path, intrinsics_arr)
        if verbose:
            print(f"Saved camera intrinsics (shape {intrinsics_arr.shape}) to: {save_intrinsics_path}")

    if save_json and output_path is not None:
        json_path = str(Path(output_path).with_suffix(".json"))
        metadata = {}
        for i, fname in enumerate(image_filenames):
            entry = {"cam2world_pose_4x4": poses_arr[i].tolist()}
            if intrinsics_arr is not None:
                entry["intrinsics_3x3"] = intrinsics_arr[i].tolist()
            metadata[fname] = entry
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        if verbose:
            print(f"Saved pose metadata JSON to: {json_path}")

    return poses_arr, intrinsics_arr, image_filenames


def main():
    parser = argparse.ArgumentParser(
        description="Estimate camera poses from images using MapAnything and save as poses.npy (OpenCV cam2world convention)."
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to folder containing images or comma-separated list of image files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/poses.npy",
        help="Output path for the estimated poses .npy file (default: output/poses.npy)",
    )
    parser.add_argument(
        "--save_intrinsics",
        type=str,
        default=None,
        help="Optional path to save recovered camera intrinsics as .npy file (default: None)",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Also save a companion .json file with image filenames and pose matrices",
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
        help="Compute device ('cuda', 'cpu', or auto-detect)",
    )

    args = parser.parse_args()

    # Parse images argument
    if "," in args.images:
        images_input = [s.strip() for s in args.images.split(",")]
    else:
        images_input = args.images

    model_name = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"

    estimate_camera_poses(
        images=images_input,
        output_path=args.output,
        save_intrinsics_path=args.save_intrinsics,
        save_json=args.save_json,
        model_name=model_name,
        device=args.device,
    )


if __name__ == "__main__":
    main()

