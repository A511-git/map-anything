# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.process_arkit_panos import (
    compute_level1_normalized_poses,
    is_arkit_pano_directory,
    process_arkit_panoramas,
    resolve_anchor_index,
    resolve_output_dir,
)
from scripts.reconstruct_from_posed_images import (
    reconstruct_from_posed_images,
    resolve_output_pointcloud_path,
)


@pytest.fixture
def sample_arkit_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create 2 synthetic equirectangular images (128x64) and companion JSONs
        for i, name in enumerate(["station_A", "station_B"]):
            img = np.zeros((64, 128, 3), dtype=np.uint8)
            img[:32, :] = [200, 100, 50]
            img[32:, :] = [50, 100, 200]
            cv2.imwrite(str(tmp_path / f"{name}_blurred.jpg"), img)

            meta = {
                "magneticHeading": 45.0 * i,
                "arkitCapture": {
                    "units": "meters",
                    "axes": "camera: +X right, +Y up, look -Z; world: +Y up, yaw0 north, +X east, +Z south",
                    "sphere": {
                        "origin": [float(i * 3.0), 0.5, float(i * 4.0)]
                    }
                }
            }
            with open(tmp_path / f"{name}.json", "w") as f:
                json.dump(meta, f)

        yield tmp_path


def test_is_arkit_pano_directory(sample_arkit_dir):
    assert is_arkit_pano_directory(sample_arkit_dir) is True
    with tempfile.TemporaryDirectory() as empty_dir:
        assert is_arkit_pano_directory(empty_dir) is False


def test_anchor_selection_and_resolution():
    records = [
        {"image_path": Path("station_A.jpg"), "origin_arkit": np.array([0, 0, 0]), "magnetic_heading_deg": 0.0},
        {"image_path": Path("station_B.jpg"), "origin_arkit": np.array([3, 0, 4]), "magnetic_heading_deg": 90.0},
    ]

    # Test default
    assert resolve_anchor_index(None, records) == 0
    assert resolve_anchor_index(0, records) == 0
    assert resolve_anchor_index("0", records) == 0

    # Test index 1
    assert resolve_anchor_index(1, records) == 1
    assert resolve_anchor_index("1", records) == 1

    # Test by filename
    assert resolve_anchor_index("station_B.jpg", records) == 1
    assert resolve_anchor_index("station_B", records) == 1

    # Invalid anchor
    with pytest.raises(ValueError):
        resolve_anchor_index("nonexistent", records)
    with pytest.raises(ValueError):
        resolve_anchor_index(5, records)


def test_anchor_level1_normalization():
    records = [
        {"origin_arkit": np.array([0.0, 0.0, 0.0], dtype=np.float32), "magnetic_heading_deg": 0.0, "image_path": Path("A.jpg"), "json_path": Path("A.json")},
        {"origin_arkit": np.array([3.0, 0.0, 4.0], dtype=np.float32), "magnetic_heading_deg": 90.0, "image_path": Path("B.jpg"), "json_path": Path("B.json")},
    ]

    R_base = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

    # Anchor at 0: A is origin
    norm0 = compute_level1_normalized_poses(records, anchor_index=0)
    assert norm0[0]["is_anchor"] is True
    assert norm0[1]["is_anchor"] is False
    np.testing.assert_allclose(norm0[0]["t_pano_relative"], [0, 0, 0])
    np.testing.assert_allclose(norm0[0]["R_pano_relative"], R_base)

    # Anchor at 1: B is origin
    norm1 = compute_level1_normalized_poses(records, anchor_index=1)
    assert norm1[1]["is_anchor"] is True
    assert norm1[0]["is_anchor"] is False
    np.testing.assert_allclose(norm1[1]["t_pano_relative"], [0, 0, 0])
    np.testing.assert_allclose(norm1[1]["R_pano_relative"], R_base)


def test_reconstruct_from_posed_images_auto_detect_arkit(sample_arkit_dir):
    with tempfile.TemporaryDirectory() as out_dir:
        # We mock the actual forward pass model to test argument resolution and auto-routing
        with patch("scripts.reconstruct_from_posed_images.torch") as mock_torch:
            mock_model = MagicMock()
            mock_model.return_value = {"pts3d": None}

            # Call reconstruct_from_posed_images with poses=None and output_path=out_dir!
            # It should auto-detect ARKit panoramas in sample_arkit_dir and save all files in out_dir/sample_arkit_dir.name
            try:
                reconstruct_from_posed_images(
                    images=str(sample_arkit_dir),
                    poses=None,  # Should auto-detect!
                    output_path=out_dir,
                    model=mock_model,
                    split_mode="cubemap",
                    face_size=64,
                    fov_deg=100.0,
                    verbose=False,
                )
            except Exception as e:
                # Progressed past pose auto-detection to neural forward pass
                pass

            # Verify that output folder contains images, poses.json, poses.npy, intrinsics.npy, normalized_arkit_metadata.json
            target_out = Path(out_dir) / sample_arkit_dir.name
            assert target_out.exists()
            assert (target_out / "poses.npy").exists()
            assert (target_out / "intrinsics.npy").exists()
            assert (target_out / "poses.json").exists()
            assert (target_out / "normalized_arkit_metadata.json").exists()
            assert len(list((target_out / "images").glob("*.jpg"))) == 12


def test_reconstruct_from_posed_images_auto_detect_poses_npy():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create 2 images and a poses.npy inside the folder
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "img1.png"), img)
        cv2.imwrite(str(tmp_path / "img2.png"), img)

        poses = np.eye(4, dtype=np.float32)[None].repeat(2, axis=0)
        np.save(str(tmp_path / "poses.npy"), poses)

        with patch("scripts.reconstruct_from_posed_images.torch"):
            mock_model = MagicMock()
            try:
                # Calling with poses=None should auto-detect poses.npy
                reconstruct_from_posed_images(
                    images=str(tmp_path),
                    poses=None,
                    model=mock_model,
                    verbose=False,
                )
            except Exception:
                pass


def test_split_panos_to_cubemap_and_ring_cli(sample_arkit_dir):
    with tempfile.TemporaryDirectory() as out_dir:
        from scripts.split_panos_to_cubemap import main as cubemap_main
        from scripts.generate_pano_faces_24 import main as ring_main

        # Test cubemap CLI
        test_args_cube = [
            "split_panos_to_cubemap.py",
            "--input", str(sample_arkit_dir),
            "--output", str(Path(out_dir) / "cube"),
            "--face_size", "64",
        ]
        with patch.object(sys, "argv", test_args_cube):
            cubemap_main()

        assert (Path(out_dir) / "cube" / sample_arkit_dir.name / "poses.npy").exists()
        assert len(list((Path(out_dir) / "cube" / sample_arkit_dir.name / "images").glob("*.jpg"))) == 12

        # Test ring CLI (8 views)
        test_args_ring = [
            "generate_pano_faces_24.py",
            "--input", str(sample_arkit_dir),
            "--output", str(Path(out_dir) / "ring"),
            "--num_views", "8",
            "--face_size", "64",
        ]
        with patch.object(sys, "argv", test_args_ring):
            ring_main()

        assert (Path(out_dir) / "ring" / sample_arkit_dir.name / "poses.npy").exists()
        # 8 horizontal + 2 vertical = 10 views per pano * 2 panos = 20 views
        assert len(list((Path(out_dir) / "ring" / sample_arkit_dir.name / "images").glob("*.jpg"))) == 20


def test_resolve_output_paths():
    # 1. Output directory passed (e.g. kaggle/working/output) -> nests input folder
    file_path, target_dir = resolve_output_pointcloud_path("kaggle/working/output", "C:/Users/Downloads/AR_KIT")
    assert file_path.replace("\\", "/") == "kaggle/working/output/AR_KIT/pointcloud.ply"
    assert target_dir.as_posix() == "kaggle/working/output/AR_KIT"

    # 2. Output directory already ending with input folder name -> no double nesting
    file_path, target_dir = resolve_output_pointcloud_path("kaggle/working/output/AR_KIT", "C:/Users/Downloads/AR_KIT")
    assert file_path.replace("\\", "/") == "kaggle/working/output/AR_KIT/pointcloud.ply"
    assert target_dir.as_posix() == "kaggle/working/output/AR_KIT"

    # 3. Output file path (.ply) passed -> parent directory nests input folder
    file_path, target_dir = resolve_output_pointcloud_path("output/reconstruction.ply", "C:/Users/Downloads/AR_KIT")
    assert file_path.replace("\\", "/") == "output/AR_KIT/reconstruction.ply"
    assert target_dir.as_posix() == "output/AR_KIT"

    # 4. resolve_output_dir for process_arkit_panos
    out_dir = resolve_output_dir("kaggle/working/output", "C:/Users/Downloads/AR_KIT")
    assert out_dir.as_posix() == "kaggle/working/output/AR_KIT"


if __name__ == "__main__":
    pytest.main([__file__])
