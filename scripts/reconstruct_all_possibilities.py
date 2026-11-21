#!/usr/bin/env python3
"""
Exhaustive Multi-Possibility Reconstructor for MapAnything.

Runs MapAnything neural inference ONCE, then automatically generates, unprojects,
and saves ALL geometric possibilities (translations, reflections, rotation flips,
yaw inversions, and axis swaps) across coordinate systems.

Possibility Groups Generated:
  1. Standard Axis Reflections: -X, -Y, -Z, -XY, -XZ, -YZ, -XYZ
  2. Axis Reflections with Inverted Yaw Rotation (R -> R^T): -X, -Y, -Z, -XY, -XZ, -YZ, -XYZ
  3. Translation-Only Flips (Rotations untouched): trans_-X, trans_-Y, trans_-Z, trans_-XZ
  4. Rotation-Only Flips (Translations untouched): rot_-X, rot_-Z, rot_yaw_inv
  5. Coordinate Axis Swaps (X <-> Z swap, east vs south)

Outputs:
  - <output_dir>/combos/<combo_name>/pointcloud.ply (Full 3D point cloud for each combo)
  - <output_dir>/all_combos_grid.png (Side-by-side visual top-down comparison grid)
  - <output_dir>/combos_summary.json (Metadata manifest of all combos)

Usage:
  python scripts/reconstruct_all_possibilities.py \
      --images "/kaggle/input/custom-mapanything-data/New folder/2" \
      --output "output" \
      --split_mode cubemap \
      --max_depth 30 \
      --fov 100 \
      --voxel_size 0.01 \
      --chunk_size 50 \
      --no-normalize \
      --add_camera_markers
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.process_arkit_panos import is_arkit_pano_directory, process_arkit_panoramas
from scripts.reconstruct_from_posed_images import (
    apply_pose_flips,
    convert_pose_convention,
    export_point_cloud,
    find_images,
    generate_camera_markers,
    load_intrinsics,
    load_poses,
    resolve_output_pointcloud_path,
)

# Comprehensive list of all candidate geometric possibilities
ALL_POSSIBILITIES = [
    # Group 1: 3-Axis Reflections (Both Translation and Conjugated SO(3) Rotation)
    ("orig", {"tx": False, "ty": False, "tz": False, "rx": False, "ry": False, "rz": False, "inv_rot": False, "swap_xz": False, "desc": "Original input poses"}),
    ("neg_x", {"tx": True, "ty": False, "tz": False, "rx": True, "ry": False, "rz": False, "inv_rot": False, "swap_xz": False, "desc": "Flip X (-X) trans & rot"}),
    ("neg_y", {"tx": False, "ty": True, "tz": False, "rx": False, "ry": True, "rz": False, "inv_rot": False, "swap_xz": False, "desc": "Flip Y (-Y) trans & rot"}),
    ("neg_z", {"tx": False, "ty": False, "tz": True, "rx": False, "ry": False, "rz": True, "inv_rot": False, "swap_xz": False, "desc": "Flip Z (-Z) trans & rot"}),
    ("neg_xy", {"tx": True, "ty": True, "tz": False, "rx": True, "ry": True, "rz": False, "inv_rot": False, "swap_xz": False, "desc": "Flip X & Y trans & rot"}),
    ("neg_xz", {"tx": True, "ty": False, "tz": True, "rx": True, "ry": False, "rz": True, "inv_rot": False, "swap_xz": False, "desc": "Flip X & Z trans & rot"}),
    ("neg_yz", {"tx": False, "ty": True, "tz": True, "rx": False, "ry": True, "rz": True, "inv_rot": False, "swap_xz": False, "desc": "Flip Y & Z trans & rot"}),
    ("neg_xyz", {"tx": True, "ty": True, "tz": True, "rx": True, "ry": True, "rz": True, "inv_rot": False, "swap_xz": False, "desc": "Flip X, Y, Z trans & rot"}),

    # Group 2: Axis Reflections WITH Inverted Yaw Rotation (R -> R^T)
    ("orig_inv_yaw", {"tx": False, "ty": False, "tz": False, "rx": False, "ry": False, "rz": False, "inv_rot": True, "swap_xz": False, "desc": "Original trans, inverted yaw"}),
    ("neg_x_inv_yaw", {"tx": True, "ty": False, "tz": False, "rx": True, "ry": False, "rz": False, "inv_rot": True, "swap_xz": False, "desc": "-X trans, inverted yaw"}),
    ("neg_y_inv_yaw", {"tx": False, "ty": True, "tz": False, "rx": False, "ry": True, "rz": False, "inv_rot": True, "swap_xz": False, "desc": "-Y trans, inverted yaw"}),
    ("neg_z_inv_yaw", {"tx": False, "ty": False, "tz": True, "rx": False, "ry": False, "rz": True, "inv_rot": True, "swap_xz": False, "desc": "-Z trans, inverted yaw"}),
    ("neg_xy_inv_yaw", {"tx": True, "ty": True, "tz": False, "rx": True, "ry": True, "rz": False, "inv_rot": True, "swap_xz": False, "desc": "-X, -Y trans, inverted yaw"}),
    ("neg_xz_inv_yaw", {"tx": True, "ty": False, "tz": True, "rx": True, "ry": False, "rz": True, "inv_rot": True, "swap_xz": False, "desc": "-X, -Z trans, inverted yaw"}),
    ("neg_yz_inv_yaw", {"tx": False, "ty": True, "tz": True, "rx": False, "ry": True, "rz": True, "inv_rot": True, "swap_xz": False, "desc": "-Y, -Z trans, inverted yaw"}),
    ("neg_xyz_inv_yaw", {"tx": True, "ty": True, "tz": True, "rx": True, "ry": True, "rz": True, "inv_rot": True, "swap_xz": False, "desc": "-X, -Y, -Z trans, inverted yaw"}),

    # Group 3: Translation ONLY Flips (keeping original rotation untouched)
    ("trans_only_neg_x", {"tx": True, "ty": False, "tz": False, "rx": False, "ry": False, "rz": False, "inv_rot": False, "swap_xz": False, "desc": "Flip translation X only"}),
    ("trans_only_neg_y", {"tx": False, "ty": True, "tz": False, "rx": False, "ry": False, "rz": False, "inv_rot": False, "swap_xz": False, "desc": "Flip translation Y only"}),
    ("trans_only_neg_z", {"tx": False, "ty": False, "tz": True, "rx": False, "ry": False, "rz": False, "inv_rot": False, "swap_xz": False, "desc": "Flip translation Z only"}),
    ("trans_only_neg_xz", {"tx": True, "ty": False, "tz": True, "rx": False, "ry": False, "rz": False, "inv_rot": False, "swap_xz": False, "desc": "Flip translation X & Z only"}),

    # Group 4: Rotation ONLY Flips (keeping translation untouched)
    ("rot_only_neg_x", {"tx": False, "ty": False, "tz": False, "rx": True, "ry": False, "rz": False, "inv_rot": False, "swap_xz": False, "desc": "Flip rotation X only"}),
    ("rot_only_neg_z", {"tx": False, "ty": False, "tz": False, "rx": False, "ry": False, "rz": True, "inv_rot": False, "swap_xz": False, "desc": "Flip rotation Z only"}),

    # Group 5: Coordinate Axis Swaps (X <-> Z swap, East vs South)
    ("swap_xz", {"tx": False, "ty": False, "tz": False, "rx": False, "ry": False, "rz": False, "inv_rot": False, "swap_xz": True, "desc": "Swap X and Z axes"}),
    ("swap_xz_neg_x", {"tx": True, "ty": False, "tz": False, "rx": True, "ry": False, "rz": False, "inv_rot": False, "swap_xz": True, "desc": "Swap X/Z with -X"}),
    ("swap_xz_neg_z", {"tx": False, "ty": False, "tz": True, "rx": False, "ry": False, "rz": True, "inv_rot": False, "swap_xz": True, "desc": "Swap X/Z with -Z"}),
]


def transform_poses(poses: List[np.ndarray], cfg: dict) -> List[np.ndarray]:
    """
    Transform a list of 4x4 camera poses according to possibility configuration.
    """
    S_trans = np.diag([
        -1.0 if cfg.get("tx") else 1.0,
        -1.0 if cfg.get("ty") else 1.0,
        -1.0 if cfg.get("tz") else 1.0,
    ]).astype(np.float32)

    S_rot = np.diag([
        -1.0 if cfg.get("rx") else 1.0,
        -1.0 if cfg.get("ry") else 1.0,
        -1.0 if cfg.get("rz") else 1.0,
    ]).astype(np.float32)

    swap_xz = cfg.get("swap_xz", False)
    inv_rot = cfg.get("inv_rot", False)

    # Permutation matrix for X <-> Z swap
    P_swap = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)

    transformed = []
    for p in poses:
        p_new = np.array(p, dtype=np.float32).copy()
        t = p_new[:3, 3]
        R = p_new[:3, :3]

        # 1. Optional X <-> Z swap
        if swap_xz:
            t = P_swap @ t
            R = P_swap @ R @ P_swap

        # 2. Translation reflection
        t = S_trans @ t

        # 3. Rotation reflection
        if cfg.get("rx") or cfg.get("ry") or cfg.get("rz"):
            R = S_rot @ R @ S_rot

        # 4. Rotation inversion / yaw transpose
        if inv_rot:
            R = R.T

        p_new[:3, 3] = t
        p_new[:3, :3] = R
        transformed.append(p_new)

    return transformed


def render_all_combos_grid(
    combos_data: List[Tuple[str, str, np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]],
    grid_png_path: str,
):
    """
    Render a clean visual comparison grid (top-down view X vs Z) of all generated combos.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not installed; skipping grid PNG generation.")
        return

    n_combos = len(combos_data)
    ncols = 5
    nrows = int(np.ceil(n_combos / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(25, 5 * nrows))
    axes = axes.flatten()

    for idx, (combo_name, desc, pts, colors, markers) in enumerate(combos_data):
        ax = axes[idx]
        if len(pts) == 0:
            ax.set_title(combo_name)
            continue

        step = max(1, len(pts) // 30000)
        sub_pts = pts[::step]
        sub_colors = colors[::step] / 255.0 if colors.max() > 1.0 else colors[::step]

        ax.scatter(sub_pts[:, 0], sub_pts[:, 2], c=sub_colors, s=0.2, alpha=0.6)

        # Plot camera markers
        if markers is not None and len(markers[0]) > 0:
            m_pts, m_colors = markers
            m_c = m_colors / 255.0 if m_colors.max() > 1.0 else m_colors
            ax.scatter(m_pts[:, 0], m_pts[:, 2], c=m_c, s=10, alpha=0.9, zorder=10)

        ax.set_title(f"{combo_name}\n({desc})", fontsize=10, fontweight="bold")
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Z (m)", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=8)

    for j in range(n_combos, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(grid_png_path), exist_ok=True)
    plt.savefig(grid_png_path, dpi=150)
    plt.close()
    print(f"\n[Visual Grid] Saved all-combos comparison grid to: {grid_png_path}")


def reconstruct_all_possibilities(
    images: Union[str, Path],
    output: str = "output",
    poses: Optional[str] = None,
    intrinsics: Optional[str] = None,
    split_mode: str = "cubemap",
    face_size: int = 1024,
    fov_deg: float = 100.0,
    anchor: str = "0",
    pano_yaw_offset: float = 0.0,
    no_normalize: bool = False,
    include_sphere_offset: bool = False,
    max_depth: Optional[float] = 30.0,
    voxel_size: Optional[float] = 0.01,
    dense: bool = False,
    chunk_size: Optional[int] = 50,
    amp_dtype: str = "auto",
    model_name: str = "facebook/map-anything",
    add_camera_markers: bool = True,
    marker_radius: float = 0.25,
    marker_style: str = "all_rays",
    combos_filter: Optional[str] = None,
):
    """
    Main routine: Runs MapAnything ONCE on GPU, then unprojects and exports ALL possibilities.
    """
    t_start = time.time()
    print("=" * 80)
    print("MAPANYTHING EXHAUSTIVE MULTI-POSSIBILITY RECONSTRUCTION")
    print("=" * 80)

    # 1. Resolve output path
    resolved_output_path, target_output_dir = resolve_output_pointcloud_path(
        output, images, default_filename="pointcloud.ply"
    )
    if target_output_dir is None:
        target_output_dir = Path(output)
    target_output_dir.mkdir(parents=True, exist_ok=True)
    combos_dir = target_output_dir / "combos"
    combos_dir.mkdir(parents=True, exist_ok=True)

    # 2. Check for raw panos and auto-route
    folder_candidate = Path(images) if isinstance(images, (str, Path)) else Path(images[0])
    if folder_candidate.is_dir() and is_arkit_pano_directory(folder_candidate):
        print(f"[Auto-Route] Detected raw ARKit panoramas in: {folder_candidate}")
        res = process_arkit_panoramas(
            input_dir=folder_candidate,
            output_dir=target_output_dir,
            split_mode=split_mode,
            face_size=face_size,
            fov_deg=fov_deg,
            pano_yaw_offset=pano_yaw_offset,
            anchor=anchor,
            no_normalize=no_normalize,
            include_sphere_offset=include_sphere_offset,
            verbose=True,
        )
        image_input = res["images_dir"]
        poses_input = res["poses_npy"]
        intrinsics_input = res["intrinsics_npy"]
    else:
        image_input = images
        poses_input = poses
        intrinsics_input = intrinsics

    # 3. Discover images & load poses
    image_paths, image_filenames = find_images(image_input)
    num_views = len(image_paths)
    loaded_poses = load_poses(str(poses_input), num_expected=num_views, image_filenames=image_filenames)
    loaded_intrinsics = load_intrinsics(intrinsics_input, num_expected=num_views) if intrinsics_input else None

    print(f"\n[Pipeline] Loaded {num_views} perspective views and base camera poses.")

    # 4. Load MapAnything model ONCE
    from mapanything.models import MapAnything
    from mapanything.utils.device import get_device
    from mapanything.utils.geometry import depthmap_to_world_frame
    from mapanything.utils.image import preprocess_inputs

    device = get_device()
    print(f"[Model] Loading MapAnything ({model_name}) onto {device}...")
    model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()

    # Determine hardware-matched AMP dtype
    resolved_amp = amp_dtype
    if resolved_amp == "auto" or resolved_amp is None:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(device)
            resolved_amp = "bf16" if (major >= 8 and torch.cuda.is_bf16_supported()) else "fp16"
        else:
            resolved_amp = "fp32"
    print(f"[Model] Configured Mixed Precision: AMP={resolved_amp.upper()}.")

    # 5. Prepare input views
    raw_views = []
    for i, item in enumerate(image_paths):
        pil_img = Image.open(item).convert("RGB")
        img_arr = np.array(pil_img, dtype=np.uint8)
        v_dict = {
            "img": torch.from_numpy(img_arr),
            "camera_poses": torch.from_numpy(loaded_poses[i]),
            "is_metric_scale": torch.tensor([True]),
        }
        if loaded_intrinsics is not None:
            v_dict["intrinsics"] = torch.from_numpy(loaded_intrinsics[i])
        raw_views.append(v_dict)

    processed_views = preprocess_inputs(raw_views, verbose=False)

    # 6. Single Neural Forward Pass
    print("\n[Inference] Running single feed-forward multi-modal inference pass on GPU...")
    t_inf_start = time.time()
    outputs = model.infer(
        processed_views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype=resolved_amp,
        apply_mask=True,
        mask_edges=True,
        use_multiview_confidence=False,
        ignore_calibration_inputs=False if loaded_intrinsics is not None else True,
        ignore_depth_inputs=True,
        ignore_pose_inputs=False,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[Inference] Completed in {time.time() - t_inf_start:.2f}s!")

    # 7. Filter possibilities if requested
    possibilities = ALL_POSSIBILITIES
    if combos_filter:
        c_filter = [x.strip().lower() for x in combos_filter.split(",")]
        possibilities = [p for p in ALL_POSSIBILITIES if p[0].lower() in c_filter or any(cf in p[0].lower() for cf in c_filter)]
        if not possibilities:
            print(f"Warning: filter '{combos_filter}' did not match any combo. Defaulting to all.")
            possibilities = ALL_POSSIBILITIES

    print(f"\n[Unprojection] Generating {len(possibilities)} geometric possibilities...")
    combos_grid_data = []
    manifest = []

    # Pre-extract predicted view data
    views_cache = []
    for view_idx, pred in enumerate(outputs):
        views_cache.append({
            "depth_z": pred["depth_z"][0].squeeze(-1),
            "intrinsics": pred["intrinsics"][0],
            "mask": pred["mask"][0].squeeze(-1).bool(),
            "img_no_norm": pred["img_no_norm"][0],
        })

    for p_idx, (combo_name, cfg) in enumerate(possibilities):
        desc = cfg.get("desc", combo_name)
        # Transform poses
        transformed_poses = transform_poses(loaded_poses, cfg)

        # Unproject all views with transformed poses
        combo_points = []
        combo_colors = []
        for v_idx, v_data in enumerate(views_cache):
            depth_z = v_data["depth_z"]
            intrinsics_view = v_data["intrinsics"]
            mask = v_data["mask"]
            img_no_norm = v_data["img_no_norm"]

            pose_torch = torch.from_numpy(transformed_poses[v_idx]).to(device=depth_z.device, dtype=depth_z.dtype)
            pts3d, valid_depth = depthmap_to_world_frame(depth_z, intrinsics_view, pose_torch)
            valid_mask = mask & valid_depth
            if max_depth is not None and max_depth > 0:
                valid_mask = valid_mask & (depth_z <= max_depth)

            valid_np = valid_mask.cpu().numpy()
            pts_np = pts3d.cpu().numpy()[valid_np]
            rgb_np = (img_no_norm.cpu().numpy()[valid_np] * 255.0).clip(0, 255).astype(np.uint8)

            if len(pts_np) > 0:
                combo_points.append(pts_np)
                combo_colors.append(rgb_np)

        if not combo_points:
            continue

        comb_pts = np.concatenate(combo_points, axis=0)
        comb_rgb = np.concatenate(combo_colors, axis=0)

        # Camera markers
        cam_markers = None
        if add_camera_markers and transformed_poses:
            m_pts, m_rgb = generate_camera_markers(
                transformed_poses, radius=marker_radius, marker_style=marker_style
            )
            cam_markers = (m_pts, m_rgb)

        # Export PLY
        combo_dir = combos_dir / combo_name
        combo_ply = combo_dir / "pointcloud.ply"
        export_point_cloud(
            output_path=str(combo_ply),
            points=comb_pts,
            colors=comb_rgb,
            voxel_size=None if dense else voxel_size,
            camera_markers=cam_markers,
            dense=dense,
        )

        combos_grid_data.append((combo_name, desc, comb_pts, comb_rgb, cam_markers))
        manifest.append({
            "combo_name": combo_name,
            "description": desc,
            "config": cfg,
            "file_path": str(combo_ply),
            "points_count": int(len(comb_pts)),
        })
        print(f"  [{p_idx + 1:02d}/{len(possibilities):02d}] {combo_name:20s} -> {combo_ply} ({len(comb_pts):,} pts)")

    # 8. Save overall visual grid comparison image
    grid_png = target_output_dir / "all_combos_grid.png"
    render_all_combos_grid(combos_grid_data, str(grid_png))

    # 9. Save JSON manifest
    manifest_file = target_output_dir / "combos_summary.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL {len(possibilities)} POSSIBILITIES GENERATED IN {total_time:.2f} SECONDS!")
    print(f"Grid Comparison Image: {grid_png}")
    print(f"Combos Directory:      {combos_dir}")
    print(f"Summary Manifest:      {manifest_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ALL geometric possibilities (reflections, rotation flips, axis swaps) using MapAnything."
    )
    parser.add_argument("--images", type=str, required=True, help="Input images folder or ARKit panos directory.")
    parser.add_argument("--poses", type=str, default=None, help="Optional poses file/dir.")
    parser.add_argument("--intrinsics", type=str, default=None, help="Optional intrinsics file.")
    parser.add_argument("--output", type=str, default="output", help="Output directory.")
    parser.add_argument("--split_mode", type=str, default="cubemap", choices=["cubemap", "ring12", "ring24", "ring8"])
    parser.add_argument("--face_size", type=int, default=1024, help="Perspective face size in pixels.")
    parser.add_argument("--fov", type=float, default=100.0, help="Field of view in degrees.")
    parser.add_argument("--anchor", type=str, default="0", help="Anchor pano index.")
    parser.add_argument("--pano_yaw_offset", type=float, default=0.0, help="Horizontal yaw offset.")
    parser.add_argument("--no_normalize", "--no-normalize", dest="no_normalize", action="store_true", default=False)
    parser.add_argument("--include_sphere_offset", "--include-sphere-offset", dest="include_sphere_offset", action="store_true", default=False)
    parser.add_argument("--max_depth", type=float, default=30.0, help="Max depth in meters.")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size in meters.")
    parser.add_argument("--dense", "--full_dense", dest="dense", action="store_true", default=False)
    parser.add_argument("--chunk_size", type=int, default=50, help="Max views per inference forward pass.")
    parser.add_argument("--add_camera_markers", action="store_true", default=True)
    parser.add_argument("--no_camera_markers", dest="add_camera_markers", action="store_false")
    parser.add_argument("--marker_radius", type=float, default=0.25)
    parser.add_argument("--marker_style", type=str, default="all_rays", choices=["all_rays", "station_only", "heading_only"])
    parser.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--apache", action="store_true")
    parser.add_argument("--combos", type=str, default=None, help="Optional comma-separated list of specific combo names to generate.")

    args = parser.parse_args()

    model_name = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"

    reconstruct_all_possibilities(
        images=args.images,
        output=args.output,
        poses=args.poses,
        intrinsics=args.intrinsics,
        split_mode=args.split_mode,
        face_size=args.face_size,
        fov_deg=args.fov,
        anchor=args.anchor,
        pano_yaw_offset=args.pano_yaw_offset,
        no_normalize=args.no_normalize,
        include_sphere_offset=args.include_sphere_offset,
        max_depth=args.max_depth,
        voxel_size=args.voxel_size,
        dense=args.dense,
        chunk_size=args.chunk_size,
        amp_dtype=args.amp_dtype,
        model_name=model_name,
        add_camera_markers=args.add_camera_markers,
        marker_radius=args.marker_radius,
        marker_style=args.marker_style,
        combos_filter=args.combos,
    )


if __name__ == "__main__":
    main()
