# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.reconstruct_from_posed_images import (
    convert_pose_convention,
    export_point_cloud,
    find_images,
    load_intrinsics,
    load_poses,
    natural_sort_key,
)


def test_natural_sort_key_and_find_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files in non-sequential order
        test_names = [
            "frame_10.png",
            "frame_1.png",
            "frame_2.png",
            "frame_100.png",
            "frame_20.png",
            "notes.txt",  # Non-image, should be ignored
        ]
        for name in test_names:
            p = os.path.join(tmpdir, name)
            with open(p, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n")

        image_paths, image_filenames = find_images(tmpdir)

        # Expected natural order: frame_1, frame_2, frame_10, frame_20, frame_100
        expected = ["frame_1.png", "frame_2.png", "frame_10.png", "frame_20.png", "frame_100.png"]
        assert image_filenames == expected
        assert len(image_paths) == 5


def test_convert_pose_convention():
    # 1. OpenCV RDF identity
    pose_in = np.eye(4, dtype=np.float32)
    pose_in[:3, 3] = [1.0, 2.0, 3.0]
    out_opencv = convert_pose_convention(pose_in, "opencv")
    np.testing.assert_allclose(out_opencv, pose_in)

    # 2. OpenGL / Blender RUB (+X Right, +Y Up, -Z Forward) -> OpenCV RDF (+X Right, +Y Down, +Z Forward)
    # Rotation columns: X stays same, Y negated, Z negated
    out_opengl = convert_pose_convention(pose_in, "opengl")
    expected_opengl = pose_in.copy()
    expected_opengl[:3, 1] *= -1.0
    expected_opengl[:3, 2] *= -1.0
    np.testing.assert_allclose(out_opengl, expected_opengl)

    # 3. COLMAP world2cam -> cam2world
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    t = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = t

    c2w = convert_pose_convention(w2c, "colmap_w2c")
    # Verify inverse: c2w @ w2c == Identity
    prod = c2w @ w2c
    np.testing.assert_allclose(prod, np.eye(4), atol=1e-5)


def test_load_poses_npy_with_convention():
    with tempfile.TemporaryDirectory() as tmpdir:
        npy_path = os.path.join(tmpdir, "poses.npy")
        poses_gt = np.eye(4, dtype=np.float32)[None].repeat(3, axis=0)
        poses_gt[1, 0, 3] = 1.0
        poses_gt[2, 1, 3] = 2.0
        np.save(npy_path, poses_gt)

        # Standard OpenCV loading
        loaded = load_poses(npy_path, num_expected=3, pose_convention="opencv")
        assert len(loaded) == 3
        np.testing.assert_allclose(loaded[1], poses_gt[1])

        # OpenGL conversion loading
        loaded_gl = load_poses(npy_path, num_expected=3, pose_convention="opengl")
        assert len(loaded_gl) == 3
        assert loaded_gl[0][1, 1] == -1.0  # Y axis inverted


def test_load_poses_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "poses.json")
        poses_list = [
            np.eye(4, dtype=np.float32).tolist(),
            np.diag([1.0, 1.0, 1.0, 1.0]).tolist(),
        ]
        with open(json_path, "w") as f:
            json.dump(poses_list, f)

        loaded = load_poses(json_path, num_expected=2)
        assert len(loaded) == 2
        np.testing.assert_allclose(loaded[0], np.eye(4))


def test_load_intrinsics():
    with tempfile.TemporaryDirectory() as tmpdir:
        npy_path = os.path.join(tmpdir, "intrinsics.npy")
        k = np.array([[500, 0, 250], [0, 500, 250], [0, 0, 1]], dtype=np.float32)
        np.save(npy_path, k)

        loaded = load_intrinsics(npy_path, num_expected=4)
        assert len(loaded) == 4
        np.testing.assert_allclose(loaded[0], k)
        np.testing.assert_allclose(loaded[3], k)


def test_export_point_cloud():
    with tempfile.TemporaryDirectory() as tmpdir:
        ply_path = os.path.join(tmpdir, "test.ply")
        pcd_path = os.path.join(tmpdir, "test.pcd")

        points = np.random.randn(100, 3).astype(np.float32)
        colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8)

        # Test PLY export with voxel downsampling
        export_point_cloud(ply_path, points, colors, voxel_size=0.1)
        assert os.path.exists(ply_path)
        assert os.path.getsize(ply_path) > 0

        # Test PCD export
        export_point_cloud(pcd_path, points, colors, voxel_size=None)
        assert os.path.exists(pcd_path)
        assert os.path.getsize(pcd_path) > 0

        # Test dense point cloud export (bypassing voxel_size)
        dense_ply_path = os.path.join(tmpdir, "test_dense.ply")
        export_point_cloud(dense_ply_path, points, colors, voxel_size=0.1, dense=True)
        assert os.path.exists(dense_ply_path)
        assert os.path.getsize(dense_ply_path) > 0


def test_dense_flag_cli_parsing():
    import argparse
    from scripts.reconstruct_from_posed_images import main

    # Test parser handling of --dense
    from scripts.reconstruct_from_posed_images import reconstruct_from_posed_images
    import unittest.mock as mock

    test_args = ["reconstruct_from_posed_images.py", "--images", "dummy_dir", "--dense"]
    with mock.patch("sys.argv", test_args):
        with mock.patch("scripts.reconstruct_from_posed_images.reconstruct_from_posed_images") as mock_recon:
            with mock.patch("scripts.reconstruct_from_posed_images.find_images", return_value=(["dummy.jpg"], ["dummy.jpg"])):
                with mock.patch("scripts.reconstruct_from_posed_images.load_poses", return_value=[np.eye(4)]):
                    main()
                    assert mock_recon.called
                    call_kwargs = mock_recon.call_args[1]
                    assert call_kwargs["dense"] is True
                    assert call_kwargs["voxel_size"] is None


if __name__ == "__main__":
    pytest.main([__file__])
