# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Estimate / reconstruct with MapAnything in three modes:

  --mode auto       Images-only. Predicted poses/K (View-0 world) used for the
                    cloud and saved as final cameras.
  --mode poses      GT LDA poses as hard cameras: condition the model, unproject
                    and save with INPUT OpenCV-RDF poses (ARKit world).
  --mode calibrate  GT poses as baseline in latent space; unproject and save
                    with model-refined predicted poses/K (View-0 world).

Every mode writes the final used poses + intrinsics under --output
(<scene>_poses_world_opencv_rdf.json, poses.npy, intrinsics.npy) plus pointcloud.ply.

Input may be a single scene folder or an LDA-style root with scene subfolders:

    LDA/
      9oPhlqgwrM/
        0.jpg … N.jpg
        9oPhlqgwrM_poses_world_opencv_rdf.json
      …

Usage:
    python scripts/estimate_poses_and_pointcloud.py \\
        --mode auto \\
        --input path/to/LDA \\
        --output output/auto_vs_gt \\
        --minibatch_size 1

    python scripts/estimate_poses_and_pointcloud.py \\
        --mode poses \\
        --input path/to/LDA \\
        --output output/posed_reconstruction \\
        --minibatch_size 1

    python scripts/estimate_poses_and_pointcloud.py \\
        --mode calibrate \\
        --input path/to/LDA \\
        --output output/calibrated_reconstruction \\
        --minibatch_size 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repo root and scripts dir are on sys.path
repo_root = Path(__file__).resolve().parent.parent
scripts_dir = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import re

import numpy as np

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def _ensure_torch():
    """Lazy-import torch so --help works without a full ML env."""
    import torch

    if torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return torch


def _v3_io():
    """Lazy-import export helpers from reconstruct_from_posed_images_v3."""
    try:
        from scripts.reconstruct_from_posed_images_v3 import (
            export_point_cloud,
            resolve_output_path,
        )
    except ModuleNotFoundError:
        from reconstruct_from_posed_images_v3 import (
            export_point_cloud,
            resolve_output_path,
        )
    return export_point_cloud, resolve_output_path


def natural_sort_key(s: Union[str, Path]) -> list:
    """Natural numerical sorting key (e.g. frame_2 before frame_10)."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def find_images(
    folder_or_list: Union[str, Path, List[Union[str, Path]]],
) -> Tuple[List[str], List[str]]:
    """Discover image files and sort them in natural numerical order."""
    image_paths: List[str] = []

    if isinstance(folder_or_list, (str, Path)):
        p = Path(folder_or_list)
        if p.is_dir():
            files = [
                f
                for f in p.iterdir()
                if f.suffix.lower() in SUPPORTED_IMAGE_EXTS and not f.name.startswith(".")
            ]
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
                files = [
                    f
                    for f in ip.iterdir()
                    if f.suffix.lower() in SUPPORTED_IMAGE_EXTS
                    and not f.name.startswith(".")
                ]
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


LDA_POSES_GLOB = "*_poses_world_opencv_rdf.json"

LDA_CONVENTION = {
    "frame": "arkitWorld",
    "cameraAxes": "opencvRDF",
    "axes": (
        "world: ARKit world (+Y up, meters); "
        "camera: OpenCV RDF (+X right, +Y down, +Z forward / look +Z)"
    ),
    "units": "meters",
    "matrixOrder": "columnMajor",
    "poses": {
        "cameraToWorld": "worldFromCamera — maps RDF camera coords → ARKit world",
        "worldToCamera": "cameraFromWorld — maps ARKit world coords → RDF camera",
    },
}


def _axis_comment_for_mode(source_mode: Optional[str] = None) -> str:
    """Build a leading // comment describing camera/world axes for exports."""
    if source_mode in ("mapanything_auto", "mapanything_calibrate"):
        world = (
            "worldFrame=View0 (model reference; view0 pose ~ identity); "
            "cameraAxes=OpenCV RDF (+X right, +Y down, +Z forward)"
        )
    else:
        world = (
            "worldFrame=ARKit world (+Y up); "
            "cameraAxes=OpenCV RDF (+X right, +Y down, +Z forward)"
        )
    return (
        f"{world}; poses=4x4 cam2world; units=meters; "
        "matrixOrder=columnMajor (LDA JSON) / row-major nested lists (poses.json)"
    )


def _write_json_with_axis_comment(
    path: Union[str, Path],
    payload: Dict[str, Any],
    axis_comment: str,
) -> None:
    """
    Write JSON with a leading // axis comment (JSONC-style).

    Standard json.load will fail on the comment line; use _load_json_strip_comments
    or strip lines starting with // before parsing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2)
    text = f"// {axis_comment}\n{body}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _load_json_strip_comments(path: Union[str, Path]) -> Any:
    """Load JSON, ignoring leading // comment lines."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    cleaned = [ln for ln in lines if not ln.lstrip().startswith("//")]
    return json.loads("\n".join(cleaned))


def _parse_path_args(values: Optional[List[str]]) -> Optional[Union[str, List[str]]]:
    """Flatten nargs='+' / comma-separated path specs into a path or list of paths."""
    if values is None:
        return None
    parts: List[str] = []
    for v in values:
        if "," in v:
            parts.extend(s.strip() for s in v.split(",") if s.strip())
        else:
            parts.append(v)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return parts


def _dir_has_images(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS and not f.name.startswith("."):
            return True
    return False


def discover_scenes(input_path: Union[str, Path]) -> List[Path]:
    """
    Discover scene folders.

    If input has immediate child dirs that each contain images, treat as a
    multi-scene LDA root. Otherwise treat input itself as a single scene.
    """
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {root}")

    if root.is_file():
        raise ValueError(f"Expected a directory, got file: {root}")

    child_scenes = sorted(
        [p for p in root.iterdir() if p.is_dir() and _dir_has_images(p)],
        key=lambda p: p.name.lower(),
    )
    if child_scenes:
        return child_scenes
    if _dir_has_images(root):
        return [root]
    raise ValueError(f"No scene folders with images found under: {root}")


def find_poses_json(scene_dir: Path, poses_override: Optional[Union[str, Path]] = None) -> Path:
    """Locate LDA poses JSON for a scene (override wins)."""
    if poses_override is not None:
        p = Path(poses_override)
        if not p.is_file():
            raise FileNotFoundError(f"Pose JSON not found: {p}")
        return p

    matches = sorted(scene_dir.glob(LDA_POSES_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No '{LDA_POSES_GLOB}' found in scene folder: {scene_dir}. "
            "Pass --poses explicitly or use --mode auto."
        )
    return matches[0]


def _mat4_to_column_major_list(mat: np.ndarray) -> List[float]:
    m = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    return m.reshape(-1, order="F").tolist()


def _column_major_list_to_mat4(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size != 16:
        raise ValueError(f"Expected 16 pose values, got {arr.size}")
    return arr.reshape(4, 4, order="F")


def _intrinsics_dict_to_matrix(intr: Dict[str, Any]) -> np.ndarray:
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _matrix_to_intrinsics_dict(k: np.ndarray) -> Dict[str, float]:
    k = np.asarray(k, dtype=np.float64).reshape(3, 3)
    return {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }


def load_lda_poses_json(
    path: Union[str, Path],
    image_filenames: List[str],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load OpenCV-RDF cam2world poses (+ optional intrinsics) from LDA JSON.

    Returns:
        poses: (N, 4, 4) float32 aligned to image_filenames
        intrinsics: (N, 3, 3) float32 or None if any frame lacks intrinsicsImage
    """
    path = Path(path)
    data = _load_json_strip_comments(path)

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"LDA pose JSON missing non-empty 'frames': {path}")

    by_basename: Dict[str, dict] = {}
    for fr in frames:
        img_field = fr.get("image") or fr.get("image_name") or ""
        basename = Path(str(img_field)).name
        if not basename:
            idx = fr.get("index")
            if idx is not None:
                basename = f"{idx}.jpg"
        if basename:
            by_basename[basename] = fr

    poses_list: List[np.ndarray] = []
    intrinsics_list: List[np.ndarray] = []
    missing: List[str] = []
    missing_intrinsics = False

    for fname in image_filenames:
        fr = by_basename.get(fname)
        if fr is None:
            missing.append(fname)
            continue
        c2w = fr.get("cameraToWorld") or fr.get("worldFromCamera")
        if c2w is None:
            raise ValueError(f"Frame '{fname}' missing cameraToWorld in {path}")
        poses_list.append(_column_major_list_to_mat4(c2w))

        intr = fr.get("intrinsicsImage")
        if isinstance(intr, dict) and all(k in intr for k in ("fx", "fy", "cx", "cy")):
            intrinsics_list.append(_intrinsics_dict_to_matrix(intr))
        else:
            missing_intrinsics = True

    if missing:
        raise ValueError(
            f"Pose JSON {path} missing frames for {len(missing)} image(s): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if len(poses_list) != len(image_filenames):
        raise ValueError(
            f"Pose/image count mismatch: {len(poses_list)} poses vs "
            f"{len(image_filenames)} images for {path}"
        )

    poses = np.stack(poses_list, axis=0)
    if missing_intrinsics or len(intrinsics_list) != len(image_filenames):
        return poses, None
    return poses, np.stack(intrinsics_list, axis=0)


def write_lda_poses_json(
    path: Union[str, Path],
    scene_name: str,
    image_filenames: List[str],
    poses: np.ndarray,
    intrinsics: np.ndarray,
    source_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Write predicted/used cameras in the LDA OpenCV-RDF JSON schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    source = {
        "mode": "mapanything_auto",
        "imageCount": len(image_filenames),
        **(source_meta or {}),
    }
    source_mode = str(source.get("mode", "mapanything_auto"))
    axis_comment = _axis_comment_for_mode(source_mode)

    convention = dict(LDA_CONVENTION)
    if source_mode in ("mapanything_auto", "mapanything_calibrate"):
        convention["frame"] = "view0World"
        convention["axes"] = (
            "world: MapAnything View-0 frame (view0 pose ~ identity, meters); "
            "camera: OpenCV RDF (+X right, +Y down, +Z forward / look +Z)"
        )
        convention["poses"] = {
            "cameraToWorld": "worldFromCamera — maps RDF camera coords → View-0 world",
            "worldToCamera": "cameraFromWorld — maps View-0 world coords → RDF camera",
        }

    frames = []
    for i, fname in enumerate(image_filenames):
        c2w = np.asarray(poses[i], dtype=np.float64).reshape(4, 4)
        w2c = np.linalg.inv(c2w)
        frames.append(
            {
                "index": i,
                "image": f"{scene_name}/{fname}",
                "intrinsicsImage": _matrix_to_intrinsics_dict(intrinsics[i]),
                "cameraToWorld": _mat4_to_column_major_list(c2w),
                "worldToCamera": _mat4_to_column_major_list(w2c),
                "worldFromCamera": _mat4_to_column_major_list(c2w),
                "cameraFromWorld": _mat4_to_column_major_list(w2c),
            }
        )

    # comment first so it appears at the top of the object as well as the // line
    payload: Dict[str, Any] = {
        "comment": axis_comment,
        "convention": convention,
        "source": source,
        "frames": frames,
    }
    _write_json_with_axis_comment(path, payload, axis_comment)


def _save_npy_json_companions(
    target_dir: Path,
    image_filenames: List[str],
    poses: np.ndarray,
    intrinsics: np.ndarray,
    verbose: bool = True,
    source_mode: Optional[str] = None,
) -> None:
    poses_npy = target_dir / "poses.npy"
    poses_json = target_dir / "poses.json"
    intrinsics_npy = target_dir / "intrinsics.npy"
    intrinsics_json = target_dir / "intrinsics.json"

    np.save(poses_npy, poses.astype(np.float32))
    np.save(intrinsics_npy, intrinsics.astype(np.float32))

    axis_comment = _axis_comment_for_mode(source_mode)
    poses_meta: Dict[str, Any] = {
        "comment": axis_comment,
        "convention": {
            "cameraAxes": "opencvRDF",
            "poseLayout": "cam2world_pose_4x4 row-major nested lists",
            "units": "meters",
        },
        "cameras": {},
    }
    intrinsics_meta: Dict[str, Any] = {
        "comment": (
            "cameraAxes=OpenCV RDF pinhole K; "
            "intrinsics_3x3 = [[fx,0,cx],[0,fy,cy],[0,0,1]]; pixel origin top-left"
        ),
        "convention": {
            "cameraAxes": "opencvRDF",
            "intrinsicsLayout": "3x3 pinhole [[fx,0,cx],[0,fy,cy],[0,0,1]]",
        },
        "cameras": {},
    }
    for i, fname in enumerate(image_filenames):
        poses_meta["cameras"][fname] = {
            "cam2world_pose_4x4": poses[i].astype(np.float32).tolist(),
            "intrinsics_3x3": intrinsics[i].astype(np.float32).tolist(),
        }
        k = intrinsics[i]
        intrinsics_meta["cameras"][fname] = {
            "intrinsics_3x3": k.astype(np.float32).tolist(),
            **_matrix_to_intrinsics_dict(k),
        }

    _write_json_with_axis_comment(poses_json, poses_meta, axis_comment)
    _write_json_with_axis_comment(
        intrinsics_json,
        intrinsics_meta,
        intrinsics_meta["comment"],
    )

    if verbose:
        print(f"Saved poses ({poses.shape}) -> {poses_npy}")
        print(f"Saved poses JSON -> {poses_json}")
        print(f"Saved intrinsics ({intrinsics.shape}) -> {intrinsics_npy}")
        print(f"Saved intrinsics JSON -> {intrinsics_json}")


def _assemble_point_cloud_from_outputs(
    outputs: List[dict],
    image_filenames: List[str],
    unproject_poses: Optional[np.ndarray],
    max_depth: Optional[float],
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unproject depths to world points.

    If unproject_poses is set (N,4,4), use those cam2world matrices (poses mode).
    Otherwise use predicted camera_poses from each pred (auto mode).
    """
    from mapanything.utils.geometry import depthmap_to_world_frame

    torch = _ensure_torch()

    poses_list: List[np.ndarray] = []
    intrinsics_list: List[np.ndarray] = []
    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []
    num_views = len(outputs)

    for idx, pred in enumerate(outputs):
        pred_pose = pred["camera_poses"][0].detach().cpu().numpy().astype(np.float32)
        pred_k = pred["intrinsics"][0].detach().cpu().numpy().astype(np.float32)
        poses_list.append(pred_pose)
        intrinsics_list.append(pred_k)

        depth_z = pred["depth_z"][0].squeeze(-1)
        intrinsics_t = pred["intrinsics"][0]
        mask = pred["mask"][0].squeeze(-1).bool()

        if unproject_poses is not None:
            pose_t = torch.from_numpy(unproject_poses[idx]).to(
                device=depth_z.device, dtype=depth_z.dtype
            )
        else:
            pose_t = pred["camera_poses"][0]

        pts3d, valid_depth = depthmap_to_world_frame(depth_z, intrinsics_t, pose_t)
        mask = mask & valid_depth
        if max_depth is not None and max_depth > 0:
            mask = mask & (depth_z <= max_depth)

        mask_np = mask.detach().cpu().numpy()
        pts_np = pts3d.detach().cpu().numpy()[mask_np]
        rgb_np = (
            pred["img_no_norm"][0].detach().cpu().numpy()[mask_np] * 255.0
        ).clip(0, 255).astype(np.uint8)

        if len(pts_np) > 0:
            all_points.append(pts_np)
            all_colors.append(rgb_np)

        if verbose and (idx < 3 or idx == num_views - 1):
            print(
                f"  View {idx}/{num_views - 1} ({image_filenames[idx]}): "
                f"{len(pts_np):,} points"
            )

    if not all_points:
        raise RuntimeError("No valid 3D points produced from inference outputs.")

    return (
        np.stack(poses_list, axis=0),
        np.stack(intrinsics_list, axis=0),
        np.concatenate(all_points, axis=0),
        np.concatenate(all_colors, axis=0),
    )


def _resolve_scene_output(
    output: Union[str, Path],
    scene_dir: Path,
) -> Tuple[str, Path]:
    _, resolve_output_path = _v3_io()
    ply_file, target_dir = resolve_output_path(
        output,
        scene_dir,
        default_filename="pointcloud.ply",
    )
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return ply_file, target_dir


def run_scene_auto(
    scene_dir: Path,
    output: Union[str, Path],
    model: object,
    device: str,
    minibatch_size: int = 1,
    apply_confidence_mask: bool = False,
    confidence_percentile: float = 10.0,
    use_multiview_confidence: bool = False,
    voxel_size: Optional[float] = 0.01,
    dense: bool = False,
    max_depth: Optional[float] = 25.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Images-only inference; export predicted LDA-schema poses + point cloud."""
    from mapanything.utils.image import load_images

    image_paths, image_filenames = find_images(scene_dir)
    num_views = len(image_paths)
    scene_name = scene_dir.name
    ply_file, target_dir = _resolve_scene_output(output, scene_dir)

    if verbose:
        print(f"\n=== [auto] Scene '{scene_name}' ({num_views} images) ===")
        print(f"Output directory: {target_dir}")

    views = load_images(image_paths, verbose=False)
    if verbose:
        print(f"Running images-only inference on {num_views} views...")

    outputs = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=minibatch_size,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=apply_confidence_mask,
        confidence_percentile=confidence_percentile,
        use_multiview_confidence=use_multiview_confidence,
    )

    pred_poses, pred_intrinsics, points, colors = _assemble_point_cloud_from_outputs(
        outputs,
        image_filenames,
        unproject_poses=None,
        max_depth=max_depth,
        verbose=verbose,
    )

    if verbose:
        print(f"Total raw 3D points: {len(points):,}")
        print(f"View 0 pose (model world origin):\n{pred_poses[0]}")

    lda_json_path = target_dir / f"{scene_name}_poses_world_opencv_rdf.json"
    write_lda_poses_json(
        lda_json_path,
        scene_name=scene_name,
        image_filenames=image_filenames,
        poses=pred_poses,
        intrinsics=pred_intrinsics,
        source_meta={"mode": "mapanything_auto", "scene": scene_name},
    )
    if verbose:
        print(f"Saved predicted LDA poses JSON -> {lda_json_path}")

    _save_npy_json_companions(
        target_dir,
        image_filenames,
        pred_poses,
        pred_intrinsics,
        verbose=verbose,
        source_mode="mapanything_auto",
    )

    export_point_cloud, _ = _v3_io()
    export_point_cloud(
        output_path=ply_file,
        points=points,
        colors=colors,
        voxel_size=voxel_size,
        dense=dense,
        verbose=verbose,
    )
    if verbose:
        print(f"Saved point cloud -> {ply_file}")

    torch = _ensure_torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred_poses, pred_intrinsics, image_filenames, ply_file


def _run_scene_posed(
    scene_dir: Path,
    output: Union[str, Path],
    model: object,
    device: str,
    *,
    mode_label: str,
    use_predicted_poses: bool,
    poses_path: Optional[Union[str, Path]] = None,
    no_intrinsics: bool = False,
    minibatch_size: int = 1,
    apply_confidence_mask: bool = False,
    confidence_percentile: float = 10.0,
    use_multiview_confidence: bool = False,
    voxel_size: Optional[float] = 0.01,
    dense: bool = False,
    max_depth: Optional[float] = 25.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """
    Shared GT-conditioned inference for --mode poses|calibrate.

    use_predicted_poses=False: unproject + save INPUT GT poses (ARKit world).
    use_predicted_poses=True:  unproject + save model-refined predicted poses
                               (View 0 world — soft calibration around GT baseline).
    """
    from PIL import Image

    from mapanything.utils.image import preprocess_inputs

    torch = _ensure_torch()

    image_paths, image_filenames = find_images(scene_dir)
    num_views = len(image_paths)
    scene_name = scene_dir.name
    ply_file, target_dir = _resolve_scene_output(output, scene_dir)

    pose_json = find_poses_json(scene_dir, poses_override=poses_path)
    input_poses, input_intrinsics = load_lda_poses_json(pose_json, image_filenames)
    if no_intrinsics:
        input_intrinsics = None

    if verbose:
        print(f"\n=== [{mode_label}] Scene '{scene_name}' ({num_views} images) ===")
        print(f"Using pose JSON: {pose_json}")
        print(
            f"Intrinsics: "
            f"{'from JSON' if input_intrinsics is not None else 'model-predicted'}"
        )
        if use_predicted_poses:
            print(
                "Cloud/export cameras: model-calibrated predicted poses "
                "(View 0 frame; GT used as latent baseline)"
            )
        else:
            print("Cloud/export cameras: INPUT GT poses (ARKit/OpenCV world)")
        print(f"Output directory: {target_dir}")

    raw_views = []
    for i, img_path in enumerate(image_paths):
        img_arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        view_dict: Dict[str, Any] = {
            "img": torch.from_numpy(img_arr),
            "camera_poses": torch.from_numpy(input_poses[i]),
            "is_metric_scale": torch.tensor([True]),
        }
        if input_intrinsics is not None:
            view_dict["intrinsics"] = torch.from_numpy(input_intrinsics[i])
        raw_views.append(view_dict)

    if verbose:
        print("Preprocessing views (resize + optional K scaling)...")
    processed_views = preprocess_inputs(raw_views, verbose=False)

    if verbose:
        print(f"Running posed inference on {num_views} views...")
    outputs = model.infer(
        processed_views,
        memory_efficient_inference=True,
        minibatch_size=minibatch_size,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=apply_confidence_mask,
        confidence_percentile=confidence_percentile,
        use_multiview_confidence=use_multiview_confidence,
        ignore_calibration_inputs=input_intrinsics is None,
        ignore_depth_inputs=True,
        ignore_pose_inputs=False,
        ignore_pose_scale_inputs=False,
    )

    pred_poses, pred_intrinsics, points, colors = _assemble_point_cloud_from_outputs(
        outputs,
        image_filenames,
        unproject_poses=None if use_predicted_poses else input_poses,
        max_depth=max_depth,
        verbose=verbose,
    )

    if use_predicted_poses:
        # Final used = model-refined cameras (View 0 world).
        export_poses = pred_poses.astype(np.float32)
        export_intrinsics = pred_intrinsics.astype(np.float32)
        source_mode = "mapanything_calibrate"
    else:
        # Final used = INPUT GT poses; K = input when present else predicted.
        export_poses = input_poses.astype(np.float32)
        export_intrinsics = (
            input_intrinsics.astype(np.float32)
            if input_intrinsics is not None
            else pred_intrinsics
        )
        source_mode = "mapanything_poses"

    if verbose:
        print(f"Total raw 3D points: {len(points):,}")

    lda_json_path = target_dir / f"{scene_name}_poses_world_opencv_rdf.json"
    write_lda_poses_json(
        lda_json_path,
        scene_name=scene_name,
        image_filenames=image_filenames,
        poses=export_poses,
        intrinsics=export_intrinsics,
        source_meta={
            "mode": source_mode,
            "scene": scene_name,
            "gt_poses_json": str(pose_json),
            "used_input_intrinsics": input_intrinsics is not None,
            "used_predicted_poses": use_predicted_poses,
        },
    )
    if verbose:
        print(f"Saved final-used LDA poses JSON -> {lda_json_path}")

    _save_npy_json_companions(
        target_dir,
        image_filenames,
        export_poses,
        export_intrinsics,
        verbose=verbose,
        source_mode=source_mode,
    )

    export_point_cloud, _ = _v3_io()
    export_point_cloud(
        output_path=ply_file,
        points=points,
        colors=colors,
        voxel_size=voxel_size,
        dense=dense,
        verbose=verbose,
    )
    if verbose:
        print(f"Saved point cloud -> {ply_file}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return export_poses, export_intrinsics, image_filenames, ply_file


def run_scene_poses(
    scene_dir: Path,
    output: Union[str, Path],
    model: object,
    device: str,
    poses_path: Optional[Union[str, Path]] = None,
    no_intrinsics: bool = False,
    minibatch_size: int = 1,
    apply_confidence_mask: bool = False,
    confidence_percentile: float = 10.0,
    use_multiview_confidence: bool = False,
    voxel_size: Optional[float] = 0.01,
    dense: bool = False,
    max_depth: Optional[float] = 25.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """GT-conditioned inference; unproject and save INPUT OpenCV-RDF poses."""
    return _run_scene_posed(
        scene_dir=scene_dir,
        output=output,
        model=model,
        device=device,
        mode_label="poses",
        use_predicted_poses=False,
        poses_path=poses_path,
        no_intrinsics=no_intrinsics,
        minibatch_size=minibatch_size,
        apply_confidence_mask=apply_confidence_mask,
        confidence_percentile=confidence_percentile,
        use_multiview_confidence=use_multiview_confidence,
        voxel_size=voxel_size,
        dense=dense,
        max_depth=max_depth,
        verbose=verbose,
    )


def run_scene_calibrate(
    scene_dir: Path,
    output: Union[str, Path],
    model: object,
    device: str,
    poses_path: Optional[Union[str, Path]] = None,
    no_intrinsics: bool = False,
    minibatch_size: int = 1,
    apply_confidence_mask: bool = False,
    confidence_percentile: float = 10.0,
    use_multiview_confidence: bool = False,
    voxel_size: Optional[float] = 0.01,
    dense: bool = False,
    max_depth: Optional[float] = 25.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """
    GT poses as latent baseline; unproject and save model-calibrated predicted
    poses/intrinsics in View 0 frame (not raw ARKit world).
    """
    return _run_scene_posed(
        scene_dir=scene_dir,
        output=output,
        model=model,
        device=device,
        mode_label="calibrate",
        use_predicted_poses=True,
        poses_path=poses_path,
        no_intrinsics=no_intrinsics,
        minibatch_size=minibatch_size,
        apply_confidence_mask=apply_confidence_mask,
        confidence_percentile=confidence_percentile,
        use_multiview_confidence=use_multiview_confidence,
        voxel_size=voxel_size,
        dense=dense,
        max_depth=max_depth,
        verbose=verbose,
    )


def estimate_poses_and_pointcloud(
    images: Union[str, Path],
    output: Union[str, Path] = "output/unposed_reconstruction",
    mode: str = "auto",
    poses: Optional[Union[str, Path]] = None,
    no_intrinsics: bool = False,
    model: Optional[object] = None,
    model_name: str = "facebook/map-anything",
    device: Optional[str] = None,
    minibatch_size: int = 1,
    apply_confidence_mask: bool = False,
    confidence_percentile: float = 10.0,
    use_multiview_confidence: bool = False,
    voxel_size: Optional[float] = 0.01,
    dense: bool = False,
    max_depth: Optional[float] = 25.0,
    verbose: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray, List[str], str]]:
    """
    Run --mode auto|poses|calibrate over one scene or an LDA multi-scene root.

    Returns:
        List of (final_used_poses, final_used_intrinsics, filenames, ply_path)
        per scene.
    """
    if mode not in ("auto", "poses", "calibrate"):
        raise ValueError(
            f"mode must be 'auto', 'poses', or 'calibrate', got: {mode!r}"
        )

    # If a list/comma path slipped through, use first directory-like entry.
    if isinstance(images, list):
        images = images[0]

    scenes = discover_scenes(images)
    if verbose:
        print(f"Discovered {len(scenes)} scene(s) under: {images}")
        for s in scenes:
            print(f"  - {s.name}")

    from mapanything.models import MapAnything
    from mapanything.utils.device import get_device

    if device is None:
        device = get_device()
    if verbose:
        print(f"Using compute device: {device}")

    if model is None:
        if verbose:
            print(f"Loading MapAnything checkpoint: {model_name}...")
        model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()

    results = []
    for scene_dir in scenes:
        if mode == "auto":
            result = run_scene_auto(
                scene_dir=scene_dir,
                output=output,
                model=model,
                device=device,
                minibatch_size=minibatch_size,
                apply_confidence_mask=apply_confidence_mask,
                confidence_percentile=confidence_percentile,
                use_multiview_confidence=use_multiview_confidence,
                voxel_size=voxel_size,
                dense=dense,
                max_depth=max_depth,
                verbose=verbose,
            )
        else:
            # --poses override only makes sense for a single scene
            poses_override = poses if len(scenes) == 1 else None
            if poses is not None and len(scenes) > 1 and verbose:
                print(
                    "[warn] --poses ignored for multi-scene batch; "
                    "auto-detecting per-scene JSON."
                )
            posed_kwargs = dict(
                scene_dir=scene_dir,
                output=output,
                model=model,
                device=device,
                poses_path=poses_override,
                no_intrinsics=no_intrinsics,
                minibatch_size=minibatch_size,
                apply_confidence_mask=apply_confidence_mask,
                confidence_percentile=confidence_percentile,
                use_multiview_confidence=use_multiview_confidence,
                voxel_size=voxel_size,
                dense=dense,
                max_depth=max_depth,
                verbose=verbose,
            )
            if mode == "calibrate":
                result = run_scene_calibrate(**posed_kwargs)
            else:
                result = run_scene_poses(**posed_kwargs)
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MapAnything three-mode reconstruction: auto (images-only), "
            "poses (exact GT cameras), calibrate (GT baseline + model-refined cameras). "
            "Always saves the final used poses/intrinsics + pointcloud.ply."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "poses", "calibrate"],
        required=True,
        help=(
            "auto = images-only predicted cameras; "
            "poses = GT cameras for cloud; "
            "calibrate = GT baseline, save model-refined cameras"
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        "--images",
        dest="input",
        nargs="+",
        required=True,
        help="Scene folder or LDA multi-scene root (spaces OK; commas accepted)",
    )
    parser.add_argument(
        "--output",
        "-o",
        nargs="+",
        default=["output/unposed_reconstruction"],
        help=(
            "Output dir or .ply path (default: output/unposed_reconstruction). "
            "Artifacts land in <output>/<scene_name>/."
        ),
    )
    parser.add_argument(
        "--poses",
        type=str,
        default=None,
        help="Optional pose JSON override (poses/calibrate; single-scene only)",
    )
    parser.add_argument(
        "--no_intrinsics",
        action="store_true",
        help=(
            "poses/calibrate: ignore JSON intrinsicsImage; let the model predict K"
        ),
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use facebook/map-anything-apache checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu', or auto-detect)",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=1,
        help="Dense-head minibatch size for memory-efficient inference (default: 1)",
    )
    parser.add_argument(
        "--apply_confidence_mask",
        action="store_true",
        help="Filter low-confidence regions in dense geometry",
    )
    parser.add_argument(
        "--confidence_percentile",
        type=float,
        default=10.0,
        help="Bottom percentile of confidence to drop when --apply_confidence_mask",
    )
    parser.add_argument(
        "--use_multiview_confidence",
        action="store_true",
        help="Use multi-view depth-consistency confidence instead of learned confidence",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Voxel size in meters for PLY downsampling (default: 0.01; ignored with --dense)",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Export full dense cloud without voxel downsampling",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=25.0,
        help="Max depth in meters to keep (<= 0 disables; default: 25.0)",
    )

    args = parser.parse_args()

    images_input = _parse_path_args(args.input)
    if isinstance(images_input, list):
        images_input = " ".join(images_input)

    output_spec = _parse_path_args(args.output)
    if output_spec is None:
        output_spec = "output/unposed_reconstruction"
    elif isinstance(output_spec, list):
        output_spec = " ".join(output_spec) if len(output_spec) > 1 else output_spec[0]

    max_depth = args.max_depth if args.max_depth > 0 else None
    model_name = (
        "facebook/map-anything-apache" if args.apache else "facebook/map-anything"
    )

    estimate_poses_and_pointcloud(
        images=images_input,
        output=output_spec,
        mode=args.mode,
        poses=args.poses,
        no_intrinsics=args.no_intrinsics,
        model_name=model_name,
        device=args.device,
        minibatch_size=args.minibatch_size,
        apply_confidence_mask=args.apply_confidence_mask,
        confidence_percentile=args.confidence_percentile,
        use_multiview_confidence=args.use_multiview_confidence,
        voxel_size=None if args.dense else args.voxel_size,
        dense=args.dense,
        max_depth=max_depth,
    )


if __name__ == "__main__":
    main()
