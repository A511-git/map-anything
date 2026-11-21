#!/usr/bin/env python3
"""
Multi-Axis Flip Reconstructor for MapAnything.

Generates THREE combo reconstructions from the passed final camera poses:
  1. Combo -X (flip_x): Camera poses and unprojected point cloud reflected across X axis.
  2. Combo -Y (flip_y): Camera poses and unprojected point cloud reflected across Y axis.
  3. Combo -Z (flip_z): Camera poses and unprojected point cloud reflected across Z axis.

Outputs generated:
  - <output_dir>/<pano_name>/flip_x/pointcloud.ply
  - <output_dir>/<pano_name>/flip_y/pointcloud.ply
  - <output_dir>/<pano_name>/flip_z/pointcloud.ply
  - <output_dir>/<pano_name>/flip_comparison.png (Top-down visual comparison)

Usage:
  python scripts/reconstruct_multi_axis_flips.py \
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
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.reconstruct_from_posed_images import (
    apply_pose_flips,
    export_point_cloud,
    generate_camera_markers,
    reconstruct_from_posed_images,
    resolve_output_pointcloud_path,
)


def render_top_view_plot(
    points_dict: dict,
    markers_dict: dict,
    output_png_path: str,
):
    """
    Render side-by-side top-down comparison plots (X vs Z) for all flip combos.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping top-view PNG plot generation.")
        return

    combos = list(points_dict.keys())
    n = len(combos)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]

    for idx, name in enumerate(combos):
        ax = axes[idx]
        pts, rgb = points_dict[name]
        if len(pts) == 0:
            continue

        # Subsample for fast plotting
        step = max(1, len(pts) // 100000)
        sub_pts = pts[::step]
        sub_rgb = rgb[::step] / 255.0 if rgb.max() > 1.0 else rgb[::step]

        ax.scatter(sub_pts[:, 0], sub_pts[:, 2], c=sub_rgb, s=0.2, alpha=0.6)

        # Plot camera markers if present
        if name in markers_dict and markers_dict[name] is not None:
            m_pts, m_rgb = markers_dict[name]
            if len(m_pts) > 0:
                ax.scatter(
                    m_pts[:, 0],
                    m_pts[:, 2],
                    c=m_rgb / 255.0 if m_rgb.max() > 1.0 else m_rgb,
                    s=8,
                    alpha=0.9,
                    zorder=10,
                )

        ax.set_title(f"Combo: {name.upper()}", fontsize=14, fontweight="bold")
        ax.set_xlabel("X (East, m)")
        ax.set_ylabel("Z (South, m)")
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved flip comparison plot to: {output_png_path}")


def run_multi_axis_reconstruction(
    images: str,
    output: str = "output",
    poses: Optional[str] = None,
    intrinsics: Optional[str] = None,
    split_mode: str = "cubemap",
    face_size: int = 1024,
    fov: float = 100.0,
    anchor: str = "0",
    pano_yaw_offset: float = 0.0,
    no_normalize: bool = False,
    include_sphere_offset: bool = False,
    flip: Optional[str] = None,
    flip_translation: Optional[str] = None,
    flip_rotation: Optional[str] = None,
    invert_rotation: bool = False,
    max_depth: Optional[float] = 30.0,
    voxel_size: Optional[float] = 0.01,
    dense: bool = False,
    chunk_size: Optional[int] = 50,
    add_camera_markers: bool = True,
    marker_radius: float = 0.25,
    marker_style: str = "all_rays",
    amp_dtype: str = "auto",
    apache: bool = False,
):
    """
    Run multi-axis flip reconstruction producing combo outputs (-x, -y, -z, -xy, -yz, etc.).
    If --flip is passed (e.g. 'x,y'), runs specifically for that combo.
    """
    print("=" * 70)
    print("MAPANYTHING MULTI-AXIS FLIP RECONSTRUCTION")
    if flip or flip_translation or flip_rotation or invert_rotation:
        print(f"  Configuration: flip={flip}, flip_trans={flip_translation}, flip_rot={flip_rotation}, invert_rot={invert_rotation}")
    else:
        print("  Generating all combo outputs (-X, -Y, -Z, -XY, -XZ, -YZ, YAW_INV)")
    print("=" * 70)

    # 1. First run base reconstruction with multi_flip_outputs enabled if no single flip is requested
    model_name = "facebook/map-anything-apache" if apache else "facebook/map-anything"
    is_multi = (flip is None and flip_translation is None and flip_rotation is None and not invert_rotation)

    combined_points, combined_colors = reconstruct_from_posed_images(
        images=images,
        poses=poses,
        intrinsics=intrinsics,
        output_path=output,
        split_mode=split_mode,
        face_size=face_size,
        fov_deg=fov,
        anchor=anchor,
        pano_yaw_offset=pano_yaw_offset,
        no_normalize=no_normalize,
        include_sphere_offset=include_sphere_offset,
        flip=flip,
        flip_translation=flip_translation,
        flip_rotation=flip_rotation,
        invert_rotation=invert_rotation,
        max_depth=max_depth,
        voxel_size=voxel_size,
        dense=dense,
        chunk_size=chunk_size,
        add_camera_markers=add_camera_markers,
        marker_radius=marker_radius,
        marker_style=marker_style,
        amp_dtype=amp_dtype,
        model_name=model_name,
        multi_flip_outputs=is_multi,
    )

    print("\n" + "=" * 70)
    print("MULTI-AXIS FLIP RECONSTRUCTION COMPLETED SUCCESSFULLY!")
    print("Outputs generated:")
    resolved_path, target_dir = resolve_output_pointcloud_path(output, images)
    if target_dir is not None:
        print(f"  - Primary:     {target_dir / 'pointcloud.ply'}")
        if is_multi:
            print(f"  - Combo -X:    {target_dir / 'flip_x' / 'pointcloud.ply'}")
            print(f"  - Combo -Y:    {target_dir / 'flip_y' / 'pointcloud.ply'}")
            print(f"  - Combo -Z:    {target_dir / 'flip_z' / 'pointcloud.ply'}")
            print(f"  - Combo -XY:   {target_dir / 'flip_xy' / 'pointcloud.ply'}")
            print(f"  - Combo -XZ:   {target_dir / 'flip_xz' / 'pointcloud.ply'}")
            print(f"  - Combo -YZ:   {target_dir / 'flip_yz' / 'pointcloud.ply'}")
            print(f"  - Combo Yaw:   {target_dir / 'flip_yaw' / 'pointcloud.ply'}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct combo point clouds (-X, -Y, -Z axis and rotation flips) using MapAnything."
    )
    parser.add_argument("--images", type=str, required=True, help="Input images folder or ARKit panos directory.")
    parser.add_argument("--poses", type=str, default=None, help="Optional poses file/dir (auto-detected if omitted).")
    parser.add_argument("--intrinsics", type=str, default=None, help="Optional intrinsics file.")
    parser.add_argument("--output", type=str, default="output", help="Output directory.")
    parser.add_argument("--split_mode", type=str, default="cubemap", choices=["cubemap", "ring12", "ring24", "ring8"])
    parser.add_argument("--face_size", type=int, default=1024, help="Perspective face size in pixels.")
    parser.add_argument("--fov", type=float, default=100.0, help="Field of view in degrees.")
    parser.add_argument("--anchor", type=str, default="0", help="Anchor pano index.")
    parser.add_argument("--pano_yaw_offset", type=float, default=0.0, help="Horizontal yaw offset.")
    parser.add_argument("--no_normalize", "--no-normalize", dest="no_normalize", action="store_true", default=False)
    parser.add_argument("--include_sphere_offset", "--include-sphere-offset", dest="include_sphere_offset", action="store_true", default=False)
    parser.add_argument("--flip", type=str, default=None, help="Specify axes to flip (e.g. 'x,y', 'y,z', 'x,z', 'x', 'y', 'z').")
    parser.add_argument("--flip_translation", "--flip-translation", "--flip_trans", "--flip-trans", dest="flip_translation", type=str, default=None, help="Axes to flip for translation only (e.g. 'x,y').")
    parser.add_argument("--flip_rotation", "--flip-rotation", "--flip_rot", "--flip-rot", dest="flip_rotation", type=str, default=None, help="Axes to flip for rotation only (e.g. 'x,y' or 'yaw').")
    parser.add_argument("--invert_rotation", "--invert-rotation", "--flip_yaw", "--flip-yaw", dest="invert_rotation", action="store_true", default=False, help="Invert/transpose camera rotation (R -> R^T).")
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

    args = parser.parse_args()

    run_multi_axis_reconstruction(
        images=args.images,
        output=args.output,
        poses=args.poses,
        intrinsics=args.intrinsics,
        split_mode=args.split_mode,
        face_size=args.face_size,
        fov=args.fov,
        anchor=args.anchor,
        pano_yaw_offset=args.pano_yaw_offset,
        no_normalize=args.no_normalize,
        include_sphere_offset=args.include_sphere_offset,
        flip=args.flip,
        flip_translation=args.flip_translation,
        flip_rotation=args.flip_rotation,
        invert_rotation=args.invert_rotation,
        max_depth=args.max_depth,
        voxel_size=None if args.dense else args.voxel_size,
        dense=args.dense,
        chunk_size=args.chunk_size,
        add_camera_markers=args.add_camera_markers,
        marker_radius=args.marker_radius,
        marker_style=args.marker_style,
        amp_dtype=args.amp_dtype,
        apache=args.apache,
    )


if __name__ == "__main__":
    main()
