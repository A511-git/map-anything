# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Extract 6 standard rectilinear perspective cubemap faces per equirectangular panorama
(front, right, back, left, up, down) and compute exact per-view camera positions,
orientations, 4x4 cam2world poses in OpenCV RDF format, and pinhole camera intrinsics.

Outputs:
- 6 perspective images per panorama in <output_dir>/images/
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
        description="Extract 6 standard perspective cubemap faces per equirectangular panorama."
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
        default="output/cubemap_dataset",
        help="Directory to save generated perspective images, poses, and intrinsics (default: output/cubemap_dataset).",
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
        "--include_sphere_offset", "--include-sphere-offset",
        dest="include_sphere_offset",
        action="store_true",
        default=False,
        help="Include physical ARKit capture sphere lens offset [dx, 0, dz] in station origin.",
    )
    parser.add_argument(
        "--flip",
        type=str,
        default=None,
        help="Specify axes to flip for both translation and rotation (e.g. 'x,y', 'y,z', 'x,z', 'x', 'y', 'z').",
    )
    parser.add_argument(
        "--flip_translation", "--flip-translation", "--flip_trans", "--flip-trans",
        dest="flip_translation",
        type=str,
        default=None,
        help="Specify axes to flip ONLY for camera translations (e.g. 'x,y', 'y,z', 'x').",
    )
    parser.add_argument(
        "--flip_rotation", "--flip-rotation", "--flip_rot", "--flip-rot",
        dest="flip_rotation",
        type=str,
        default=None,
        help="Specify axes to flip ONLY for camera rotations (e.g. 'x,y', 'y,z', 'x', or 'yaw').",
    )
    parser.add_argument(
        "--invert_rotation", "--invert-rotation", "--flip_yaw", "--flip-yaw",
        dest="invert_rotation",
        action="store_true",
        default=False,
        help="Invert / transpose camera rotation matrix (R -> R^T, inverts yaw heading).",
    )
    parser.add_argument(
        "--flip_x", "--flip-x",
        dest="flip_x",
        action="store_true",
        default=False,
        help="Multiply X world coordinate by -1 (reflect across X axis).",
    )
    parser.add_argument(
        "--flip_y", "--flip-y",
        dest="flip_y",
        action="store_true",
        default=False,
        help="Multiply Y world coordinate by -1 (reflect across Y axis).",
    )
    parser.add_argument(
        "--flip_z", "--flip-z",
        dest="flip_z",
        action="store_true",
        default=False,
        help="Multiply Z world coordinate by -1 (reflect across Z axis).",
    )
    parser.add_argument(
        "--run_reconstruction",
        action="store_true",
        help="Automatically trigger MapAnything 3D reconstruction after extracting cubemap faces.",
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

    args = parser.parse_args()

    results = process_arkit_panoramas(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_mode="cubemap",
        face_size=args.face_size,
        fov_deg=args.fov,
        pano_yaw_offset=args.pano_yaw_offset,
        anchor=args.anchor,
        no_normalize=args.no_normalize,
        include_sphere_offset=args.include_sphere_offset,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        flip_z=args.flip_z,
        flip=args.flip,
        flip_translation=args.flip_translation,
        flip_rotation=args.flip_rotation,
        invert_rotation=args.invert_rotation,
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
            voxel_size=args.voxel_size,
            use_multiview_confidence=True,
        )


if __name__ == "__main__":
    main()
