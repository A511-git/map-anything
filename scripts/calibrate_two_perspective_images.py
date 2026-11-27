#!/usr/bin/env python3
"""
Two-view MapAnything reconstruction with an auto mode and a condition mode.

Both modes pass the two photos and their K. Neither passes depth.

  auto       No pose prior. The cloud and cameras are the model's view-0 outputs.
  condition  JSON cam-to-world poses are injected, like v3 depth_mode=condition
             injects depth. Exported cameras are
             T_world_i = T_gt_anchor @ inv(T_pred_0) @ T_pred_i.

Writes pointcloud.ply, cameras.json, and depths/<stem>_depth.npy (s = Z).

Usage:
    python scripts/calibrate_two_perspective_images.py --mode auto ^
        --input C:\\Users\\A-511\\Downloads\\single_face ^
        --output output/single_face_auto
    python scripts/calibrate_two_perspective_images.py --mode condition ^
        --input C:\\Users\\A-511\\Downloads\\single_face ^
        --output output/single_face_condition
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent
scripts_dir = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import numpy as np
from PIL import Image

POSE_JSON_SUFFIX = "_poses_world_opencv_rdf.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_sort_key(value: str) -> list:
    import re

    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", value)
    ]


def _column_major_list_to_mat4(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size != 16:
        raise ValueError(f"Expected 16 pose values, got {arr.size}")
    return arr.reshape(4, 4, order="F")


def _mat4_to_column_major_list(mat: np.ndarray) -> List[float]:
    matrix = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    return matrix.reshape(-1, order="F").tolist()


def _intrinsics_dict_to_matrix(intr: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            [float(intr["fx"]), 0.0, float(intr["cx"])],
            [0.0, float(intr["fy"]), float(intr["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _matrix_to_intrinsics_dict(k: np.ndarray) -> Dict[str, float]:
    matrix = np.asarray(k, dtype=np.float64).reshape(3, 3)
    return {
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
    }


def _join_path_args(values: Any) -> str:
    if isinstance(values, list):
        return " ".join(values)
    return str(values)


def discover_views(input_dir: Path) -> List[Dict[str, Any]]:
    """Pair each {sceneId}_{index}.jpg with that frame in the scene pose JSON."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    pose_jsons: Dict[str, Path] = {}
    for path in input_dir.glob(f"*{POSE_JSON_SUFFIX}"):
        scene_id = path.name[: -len(POSE_JSON_SUFFIX)]
        if scene_id:
            pose_jsons[scene_id] = path
    if not pose_jsons:
        raise FileNotFoundError(
            f"No *{POSE_JSON_SUFFIX} files in {input_dir}"
        )

    images = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    images.sort(key=lambda path: natural_sort_key(path.name))
    if len(images) != 2:
        raise ValueError(
            f"Expected exactly 2 perspective images in {input_dir}, found {len(images)}: "
            f"{[path.name for path in images]}"
        )

    scene_ids_longest_first = sorted(pose_jsons, key=len, reverse=True)
    views: List[Dict[str, Any]] = []
    for image_path in images:
        scene_id = None
        frame_index = None
        for candidate in scene_ids_longest_first:
            prefix = candidate + "_"
            if not image_path.stem.startswith(prefix):
                continue
            index_text = image_path.stem[len(prefix) :]
            if index_text.isdigit():
                scene_id = candidate
                frame_index = int(index_text)
                break
        if scene_id is None or frame_index is None:
            raise ValueError(
                f"Image '{image_path.name}' does not match any "
                f"{{sceneId}}_{{index}} pose JSON in {input_dir}. "
                f"Pose files: {sorted(path.name for path in pose_jsons.values())}"
            )

        frame = _load_frame(pose_jsons[scene_id], frame_index)
        intrinsics = frame.get("intrinsicsImage")
        if not isinstance(intrinsics, dict) or not all(
            key in intrinsics for key in ("fx", "fy", "cx", "cy")
        ):
            raise ValueError(
                f"Frame {frame_index} in {pose_jsons[scene_id].name} "
                "is missing intrinsicsImage fx, fy, cx, cy."
            )
        c2w_values = frame.get("cameraToWorld") or frame.get("worldFromCamera")
        if not isinstance(c2w_values, list):
            raise ValueError(
                f"Frame {frame_index} in {pose_jsons[scene_id].name} "
                "is missing cameraToWorld."
            )

        views.append(
            {
                "image_path": image_path,
                "scene_id": scene_id,
                "frame_index": frame_index,
                "pose_json": pose_jsons[scene_id],
                "c2w": _column_major_list_to_mat4(c2w_values),
                "intrinsics": _intrinsics_dict_to_matrix(intrinsics),
            }
        )
    return views


def _load_frame(pose_json: Path, frame_index: int) -> dict:
    with pose_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Pose JSON has no frames: {pose_json}")

    basename = f"{frame_index}.jpg"
    for frame in frames:
        image_field = str(frame.get("image") or frame.get("image_name") or "")
        image_name = Path(image_field).name
        if image_name.lower() == basename.lower():
            return frame
    for frame in frames:
        if frame.get("index") == frame_index:
            return frame
    raise ValueError(
        f"Pose JSON {pose_json.name} has no frame whose image is '{basename}' "
        f"or whose index is {frame_index}."
    )


def shared_network_resolution(image_sizes_wh: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Match preprocess_inputs fixed_mapping (resolution set 518) for these photos."""
    from mapanything.utils.image import find_closest_aspect_ratio

    aspects = [width / height for width, height in image_sizes_wh]
    average_aspect = sum(aspects) / len(aspects)
    width, height = find_closest_aspect_ratio(average_aspect, 518)
    return int(width), int(height)


def preprocess_k_geometry(
    original_wh: Tuple[int, int],
    target_wh: Tuple[int, int],
    k_input: np.ndarray,
) -> Tuple[float, np.ndarray, int, int]:
    """
    Scale and center-crop that preprocess_inputs applies to K.

    Returns scale, the rescale principal-point offset, and the integer crop
    origin (left, top) in the rescaled image.
    """
    from mapanything.utils.cropping import (
        bbox_from_intrinsics_in_out,
        camera_matrix_of_crop,
    )

    original = np.asarray(original_wh, dtype=np.float64)
    target = np.asarray(target_wh, dtype=np.float64)
    scale = float(np.max(target / original) + 1e-8)
    rescaled = np.floor(original * scale).astype(int)
    rescale_offset = 0.5 * (original * scale - rescaled)

    k_rescaled = camera_matrix_of_crop(
        np.asarray(k_input, dtype=np.float64),
        original,
        rescaled,
        scaling=scale,
    )
    k_crop_target = camera_matrix_of_crop(
        k_rescaled,
        rescaled,
        target.astype(int),
        offset_factor=0.5,
    )
    left, top, _, _ = bbox_from_intrinsics_in_out(
        k_rescaled,
        k_crop_target,
        (int(target[0]), int(target[1])),
    )
    return scale, rescale_offset, int(left), int(top)


def network_intrinsics_to_original(
    k_network: np.ndarray,
    scale: float,
    rescale_offset: np.ndarray,
    crop_left: int,
    crop_top: int,
) -> np.ndarray:
    """Map a network-resolution K back onto the original photo pixels."""
    k = np.array(k_network, dtype=np.float64, copy=True)
    k[0, 2] += crop_left
    k[1, 2] += crop_top
    # Inverse of camera_matrix_of_crop's rescale (OpenCV -> Colmap -> scale -> offset).
    k[0, 2] += 0.5
    k[1, 2] += 0.5
    k[0, 2] += float(rescale_offset[0])
    k[1, 2] += float(rescale_offset[1])
    k[:2, :] /= scale
    k[0, 2] -= 0.5
    k[1, 2] -= 0.5
    return k


def _input_k_at_network_resolution(
    k_input: np.ndarray,
    original_wh: Tuple[int, int],
    target_wh: Tuple[int, int],
) -> np.ndarray:
    """K after the same rescale and center crop, used to check the mapping."""
    from mapanything.utils.cropping import camera_matrix_of_crop

    scale, _, crop_left, crop_top = preprocess_k_geometry(
        original_wh, target_wh, k_input
    )
    original = np.asarray(original_wh, dtype=np.float64)
    rescaled = np.floor(original * scale).astype(int)
    k = camera_matrix_of_crop(
        np.asarray(k_input, dtype=np.float64),
        original,
        rescaled,
        scaling=scale,
    )
    k = np.array(k, dtype=np.float64, copy=True)
    k[0, 2] -= crop_left
    k[1, 2] -= crop_top
    return k


def _amp_settings(device: str) -> Tuple[bool, str]:
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability(device)
            amp_dtype = "bf16" if major >= 8 and torch.cuda.is_bf16_supported() else "fp16"
        except Exception:
            amp_dtype = "fp16"
        return True, amp_dtype
    return False, "fp32"


def map_predicted_poses_to_world(
    pred_poses: np.ndarray,
    anchor_c2w: np.ndarray,
) -> np.ndarray:
    """Lift latent cam-to-world poses into the anchor camera's world.

    MapAnything returns poses in its own frame, with view 0 near the origin.
    Removing that view-0 pose and applying the anchor keeps the first camera
    on its known world pose and carries the solved relative pose:

        T_world_i = T_gt_anchor @ inv(T_pred_0) @ T_pred_i
    """
    anchor = np.asarray(anchor_c2w, dtype=np.float64)
    pred0 = np.asarray(pred_poses[0], dtype=np.float64)
    world_from_latent = anchor @ np.linalg.inv(pred0)
    mapped = np.empty((len(pred_poses), 4, 4), dtype=np.float64)
    for index, pred in enumerate(pred_poses):
        mapped[index] = world_from_latent @ np.asarray(pred, dtype=np.float64)
    return mapped


def _metric_scale(pred: dict) -> float:
    value = pred["metric_scaling_factor"]
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    scales = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(scales[0])


def _depth_for_export(depth_z, max_depth: Optional[float]) -> np.ndarray:
    depth = depth_z.detach().float().cpu().numpy().astype(np.float32)
    invalid = ~np.isfinite(depth)
    depth[invalid] = 0.0
    if max_depth is not None and max_depth > 0:
        depth[depth > max_depth] = 0.0
    return depth


def run_calibration(
    input_dir: Path,
    output_dir: Path,
    *,
    mode: str,
    voxel_size: Optional[float] = 0.01,
    dense: bool = False,
    max_depth: Optional[float] = 25.0,
    minibatch_size: int = 1,
    model_name: str = "facebook/map-anything",
    device: Optional[str] = None,
) -> None:
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.device import get_device
    from mapanything.utils.geometry import depthmap_to_world_frame
    from mapanything.utils.image import preprocess_inputs

    try:
        from scripts.reconstruct_from_posed_images_v3 import export_point_cloud
    except ModuleNotFoundError:
        from reconstruct_from_posed_images_v3 import export_point_cloud

    if mode not in ("auto", "condition"):
        raise ValueError(f"mode must be 'auto' or 'condition', got {mode!r}")
    use_pose_prior = mode == "condition"

    views = discover_views(input_dir)
    print(f"Input: {input_dir}")
    print(f"Mode: {mode}")
    print(f"Paired {len(views)} perspective images:")
    for view in views:
        center = view["c2w"][:3, 3]
        k = view["intrinsics"]
        pose_note = (
            f"xyz=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
            if use_pose_prior
            else "pose prior off"
        )
        print(
            f"  {view['image_path'].name}  scene={view['scene_id']}  "
            f"frame={view['frame_index']}  {pose_note}  "
            f"K=({k[0, 0]:.1f}, {k[1, 1]:.1f}, {k[0, 2]:.1f}, {k[1, 2]:.1f})"
        )

    if device is None:
        device = get_device()
    print(f"\n[Model] Loading MapAnything ({model_name}) on {device}...")
    model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()
    use_amp, amp_dtype = _amp_settings(str(device))
    print(f"[Model] Device: {device}, Mixed Precision AMP: {amp_dtype.upper()}")
    if use_pose_prior:
        print("[Mode] condition: inject JSON poses, no depth")
        print("[Mode] Export T_gt_anchor @ inv(T_pred_0) @ T_pred")
    else:
        print("[Mode] auto: images and K only, no pose prior")
        print("[Mode] Export the model's view-0 poses, scale, K, and cloud")

    loaded_images: List[np.ndarray] = []
    image_sizes: List[Tuple[int, int]] = []
    for view in views:
        with Image.open(view["image_path"]) as img:
            rgb = np.array(img.convert("RGB"), copy=True)
        loaded_images.append(rgb)
        height, width = rgb.shape[:2]
        image_sizes.append((width, height))
        view["original_wh"] = (width, height)

    target_wh = shared_network_resolution(image_sizes)
    print(f"[Preprocess] Network resolution {target_wh[0]}x{target_wh[1]} (W x H)")

    raw_views = []
    for view, rgb in zip(views, loaded_images):
        view_dict: Dict[str, Any] = {
            "img": torch.from_numpy(rgb),
            "intrinsics": torch.from_numpy(view["intrinsics"].astype(np.float32)),
            "is_metric_scale": torch.tensor([True]),
        }
        if use_pose_prior:
            view_dict["camera_poses"] = torch.from_numpy(view["c2w"].astype(np.float32))
        raw_views.append(view_dict)
    processed_views = preprocess_inputs(raw_views, verbose=False)

    for view, processed in zip(views, processed_views):
        expected = _input_k_at_network_resolution(
            view["intrinsics"], view["original_wh"], target_wh
        )
        got = processed["intrinsics"][0].detach().cpu().numpy()
        if not np.allclose(expected, got, atol=1e-2):
            raise RuntimeError(
                f"Intrinsics resize for {view['image_path'].name} does not match "
                "preprocess_inputs. Refusing to scale fx, fy, cx, cy back to the photo."
            )

    with torch.inference_mode():
        outputs = model.infer(
            processed_views,
            memory_efficient_inference=True,
            minibatch_size=minibatch_size,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            apply_mask=True,
            mask_edges=True,
            ignore_calibration_inputs=False,
            ignore_depth_inputs=True,
            ignore_pose_inputs=not use_pose_prior,
            ignore_pose_scale_inputs=not use_pose_prior,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "depths"
    depth_dir.mkdir(parents=True, exist_ok=True)

    pred_poses = np.stack(
        [
            pred["camera_poses"][0].detach().float().cpu().numpy()
            for pred in outputs
        ],
        axis=0,
    )
    if use_pose_prior:
        world_poses = map_predicted_poses_to_world(pred_poses, views[0]["c2w"])
    else:
        world_poses = pred_poses.astype(np.float64)
    metric_scales = [_metric_scale(pred) for pred in outputs]
    anchor_name = views[0]["image_path"].name
    print(
        f"[Poses] View 0 '{anchor_name}'. "
        f"Predicted translation {pred_poses[0][:3, 3].round(4).tolist()}. "
        f"Metric scale {metric_scales[0]:.6f}"
    )
    for view, pose in zip(views, world_poses):
        center = pose[:3, 3]
        if use_pose_prior:
            delta = center - view["c2w"][:3, 3]
            print(
                f"  calibrated {view['image_path'].name}: "
                f"xyz=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})  "
                f"shift={float(np.linalg.norm(delta)):.3f} m from input"
            )
        else:
            print(
                f"  view-0 {view['image_path'].name}: "
                f"xyz=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
            )

    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []
    camera_records: List[Dict[str, Any]] = []

    for view_index, (view, pred, c2w_np) in enumerate(zip(views, outputs, world_poses)):
        depth_z = pred["depth_z"][0].squeeze(-1)
        intrinsics_view = pred["intrinsics"][0]
        mask = pred["mask"][0].squeeze(-1).bool()
        img_no_norm = pred["img_no_norm"][0]

        c2w = torch.from_numpy(c2w_np.astype(np.float32)).to(
            device=depth_z.device, dtype=depth_z.dtype
        )
        pts3d, valid_depth = depthmap_to_world_frame(depth_z, intrinsics_view, c2w)
        valid_mask = mask & valid_depth
        if max_depth is not None and max_depth > 0:
            valid_mask = valid_mask & (depth_z <= max_depth)

        valid_np = valid_mask.detach().cpu().numpy()
        pts_np = pts3d.detach().cpu().numpy()[valid_np]
        rgb_np = (img_no_norm.detach().cpu().numpy()[valid_np] * 255.0).clip(0, 255).astype(
            np.uint8
        )
        if len(pts_np) > 0:
            all_points.append(pts_np)
            all_colors.append(rgb_np)

        depth_np = _depth_for_export(depth_z, max_depth)
        depth_name = f"{view['image_path'].stem}_depth.npy"
        depth_rel = f"depths/{depth_name}"
        np.save(depth_dir / depth_name, depth_np)

        k_network = intrinsics_view.detach().float().cpu().numpy()
        network_h, network_w = depth_np.shape
        if (network_w, network_h) != target_wh:
            raise RuntimeError(
                f"Predicted depth for {view['image_path'].name} is "
                f"{network_w}x{network_h}, expected network size {target_wh[0]}x{target_wh[1]}."
            )
        scale, rescale_offset, crop_left, crop_top = preprocess_k_geometry(
            view["original_wh"], target_wh, view["intrinsics"]
        )
        k_original = network_intrinsics_to_original(
            k_network, scale, rescale_offset, crop_left, crop_top
        )
        original_w, original_h = view["original_wh"]
        center = c2w_np[:3, 3]
        w2c = np.linalg.inv(c2w_np)
        network_intr = _matrix_to_intrinsics_dict(k_network)
        original_intr = _matrix_to_intrinsics_dict(k_original)
        latent_pose = pred_poses[view_index]
        metric_scale = metric_scales[view_index]

        camera_records.append(
            {
                "image": view["image_path"].name,
                "scene": view["scene_id"],
                "frameIndex": view["frame_index"],
                "metricScale": metric_scale,
                "networkImage": {"width": int(network_w), "height": int(network_h)},
                "originalImage": {"width": int(original_w), "height": int(original_h)},
                "intrinsicsNetwork": {
                    **network_intr,
                    "width": int(network_w),
                    "height": int(network_h),
                    "K": np.asarray(k_network, dtype=np.float64).tolist(),
                },
                "intrinsicsImage": {
                    **original_intr,
                    "width": int(original_w),
                    "height": int(original_h),
                    "K": np.asarray(k_original, dtype=np.float64).tolist(),
                },
                "position": {
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "z": float(center[2]),
                },
                "cameraToWorld": _mat4_to_column_major_list(c2w_np),
                "worldToCamera": _mat4_to_column_major_list(w2c),
                "depth": depth_rel,
            }
        )
        if use_pose_prior:
            camera_records[-1]["latentCameraToWorld"] = _mat4_to_column_major_list(
                latent_pose
            )
            camera_records[-1]["inputCameraToWorld"] = _mat4_to_column_major_list(
                view["c2w"]
            )
        print(
            f"  {view['image_path'].name}: "
            f"{len(pts_np):,} points, "
            f"network K fx={network_intr['fx']:.2f} fy={network_intr['fy']:.2f} "
            f"cx={network_intr['cx']:.2f} cy={network_intr['cy']:.2f}, "
            f"image K fx={original_intr['fx']:.2f} fy={original_intr['fy']:.2f} "
            f"cx={original_intr['cx']:.2f} cy={original_intr['cy']:.2f}, "
            f"scale={metric_scale:.6f}"
        )

    if not all_points:
        raise RuntimeError("No valid 3D points produced for the two images.")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    print(f"\n[Complete] Total raw 3D points: {len(points):,}")

    ply_path = output_dir / "pointcloud.ply"
    export_point_cloud(
        output_path=str(ply_path),
        points=points,
        colors=colors,
        voxel_size=voxel_size,
        dense=dense,
        verbose=True,
    )

    if use_pose_prior:
        comment = (
            "condition: JSON poses were injected and no depth was passed. "
            "cameraToWorld is T_gt_anchor @ inv(T_pred_0) @ T_pred_i. "
            "metricScale and intrinsics are the model outputs. "
            "intrinsicsNetwork is the K used to unproject the depth map (s = Z). "
            "intrinsicsImage is that K mapped back onto the original photo. "
            "inputCameraToWorld is the pose that was conditioned on."
        )
        convention = {
            "frame": "arkitWorld",
            "cameraAxes": "opencvRDF",
            "axes": (
                "world: ARKit world (+Y up, meters); "
                "camera: OpenCV RDF (+X right, +Y down, +Z forward)"
            ),
            "units": "meters",
            "matrixOrder": "columnMajor",
            "projection": "s [u, v, 1]^T = K [X, Y, Z]^T with s = Z",
        }
        source = {
            "mode": "condition",
            "poseMap": "T_gt_anchor @ inv(T_pred_0) @ T_pred_i",
            "anchorImage": anchor_name,
            "input": str(input_dir),
            "maxDepthMeters": max_depth,
        }
    else:
        comment = (
            "auto: images and K only. No pose prior. "
            "cameraToWorld is the model's view-0 pose. "
            "metricScale and intrinsics are the model outputs. "
            "intrinsicsNetwork is the K used to unproject the depth map (s = Z). "
            "intrinsicsImage is that K mapped back onto the original photo."
        )
        convention = {
            "frame": "view0World",
            "cameraAxes": "opencvRDF",
            "axes": (
                "world: MapAnything view-0 frame (first camera near the origin, meters); "
                "camera: OpenCV RDF (+X right, +Y down, +Z forward)"
            ),
            "units": "meters",
            "matrixOrder": "columnMajor",
            "projection": "s [u, v, 1]^T = K [X, Y, Z]^T with s = Z",
        }
        source = {
            "mode": "auto",
            "poseMap": "predicted view-0 camera_poses",
            "input": str(input_dir),
            "maxDepthMeters": max_depth,
        }
    cameras_payload = {
        "comment": comment,
        "convention": convention,
        "source": source,
        "cameras": camera_records,
    }
    cameras_path = output_dir / "cameras.json"
    with cameras_path.open("w", encoding="utf-8") as handle:
        json.dump(cameras_payload, handle, indent=2)
        handle.write("\n")
    print(f"Saved cameras -> {cameras_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct two perspective images. "
            "auto uses the model with no pose prior. "
            "condition injects the JSON poses and saves the refined cameras."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["auto", "condition"],
        help=(
            "auto = images and K only, export the model's view-0 cameras; "
            "condition = inject JSON poses, export T_gt_anchor @ inv(T_pred_0) @ T_pred"
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="Folder with two perspective images and their *_poses_world_opencv_rdf.json files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        nargs="+",
        default=["output/single_face_calibrated"],
        help="Output directory (default: output/single_face_calibrated).",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Voxel downsampling grid size in meters (default: 0.01).",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        default=False,
        help="Export the full dense point cloud without voxel downsampling.",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=25.0,
        help="Maximum depth cutoff in meters (default: 25).",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=1,
        help="Minibatch size for memory-efficient head inference (default: 1).",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use the Apache 2.0 checkpoint (facebook/map-anything-apache).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device ('cuda', 'cpu', or auto-detect).",
    )
    args = parser.parse_args()
    model_name = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"
    run_calibration(
        input_dir=Path(_join_path_args(args.input)),
        output_dir=Path(_join_path_args(args.output)),
        mode=args.mode,
        voxel_size=None if args.dense else args.voxel_size,
        dense=args.dense,
        max_depth=args.max_depth,
        minibatch_size=args.minibatch_size,
        model_name=model_name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
