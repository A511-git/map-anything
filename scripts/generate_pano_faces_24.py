# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Extract horizontal ring perspective views per 360 equirectangular panorama
(e.g., spaced every 30 degrees yaw around the equator plus up/down views)
and compute exact per-view camera positions, orientations, 4x4 cam2world poses
in OpenCV RDF format, and pinhole camera intrinsics.

Outputs:
- Rectilinear perspective images in <output_dir>/images/
- poses.npy: Array of shape (M, 4, 4) in OpenCV cam2world format, sorted by image name
- intrinsics.npy: Array of shape (M, 3, 3) in pinhole format, sorted by image name
- cubemap_poses.json / poses.json: Pose metadata for each perspective image
- normalized_arkit_metadata.json: Level 1 relative poses of parent panoramas
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root and scripts dir are discoverable
repo_root = Path(__file__).resolve().parent.parent
scripts_dir = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

try:
    from scripts.process_arkit_panos import process_arkit_panoramas
except ModuleNotFoundError:
    from process_arkit_panos import process_arkit_panoramas


def main():
    parser = argparse.ArgumentParser(
        description="Extract overlapping horizontal ring perspective views per equirectangular panorama."
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
        default="output/ring_dataset",
        help="Directory to save generated perspective images, poses, and intrinsics (default: output/ring_dataset).",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=12,
        choices=[8, 12, 24],
        help="Number of horizontal equatorial yaw views (8, 12, or 24; default: 12).",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="0",
        help="Anchor panorama index or filename (default: 0). Anchored at origin [0, 0, 0] facing +Z.",
    )
    parser.add_argument(
        "--face_size",
        type=int,
        default=1024,
        help="Resolution (width and height in pixels) of each perspective face (default: 1024).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="Field of view in degrees for each perspective view (default: 90.0).",
    )
    parser.add_argument(
        "--heading_mode",
        type=str,
        default="north_aligned",
        choices=["north_aligned", "relative", "raw_compass"],
        help="Heading alignment mode: 'north_aligned' (default), 'relative', or 'raw_compass'.",
    )
    parser.add_argument(
        "--run_reconstruction",
        action="store_true",
        help="Automatically trigger MapAnything 3D reconstruction after extracting perspective views.",
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

    split_mode = f"ring{args.num_views}"

    results = process_arkit_panoramas(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_mode=split_mode,
        face_size=args.face_size,
        fov_deg=args.fov,
        heading_mode=args.heading_mode,
        anchor=args.anchor,
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
