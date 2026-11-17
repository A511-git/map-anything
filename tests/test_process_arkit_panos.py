# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.process_arkit_panos import (
    compute_face_rotation_matrix,
    compute_level1_normalized_poses,
    compute_remap_grid,
    find_pano_json_pairs,
    get_cubemap_faces_definitions,
    get_ring_faces_definitions,
    load_arkit_metadata,
    process_arkit_panoramas,
    rotation_matrix_y,
)


def test_rotation_matrix_y():
    # 1. 0 deg is identity
    r0 = rotation_matrix_y(0.0)
    np.testing.assert_allclose(r0, np.eye(3), atol=1e-6)

    # 2. 90 deg clockwise maps +Z (0, 0, 1) to +X (1, 0, 0)
    r90 = rotation_matrix_y(90.0)
    z_fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    mapped_90 = r90 @ z_fwd
    np.testing.assert_allclose(mapped_90, np.array([1.0, 0.0, 0.0]), atol=1e-5)

    # 3. 180 deg maps +Z to -Z
    r180 = rotation_matrix_y(180.0)
    mapped_180 = r180 @ z_fwd
    np.testing.assert_allclose(mapped_180, np.array([0.0, 0.0, -1.0]), atol=1e-5)

    # 4. Orthogonality and determinant
    for angle in [0.0, 30.0, 45.0, 90.0, 123.45, 270.0]:
        r = rotation_matrix_y(angle)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-6)


def test_level1_normalization_math():
    records = [
        {
            "origin_arkit": np.array([-7.593149, 0.354892, 20.392481], dtype=np.float32),
            "magnetic_heading_deg": 223.79487,
            "image_path": Path("pano_0.jpg"),
            "json_path": Path("pano_0.json"),
        },
        {
            "origin_arkit": np.array([-6.628068, 0.420724, 14.835424], dtype=np.float32),
            "magnetic_heading_deg": 78.48435,
            "image_path": Path("pano_1.jpg"),
            "json_path": Path("pano_1.json"),
        },
    ]

    normalized = compute_level1_normalized_poses(records, heading_mode="relative")
    assert len(normalized) == 2

    # Pano 0 must be identity at origin
    c2w_0 = normalized[0]["c2w_pano_4x4"]
    np.testing.assert_allclose(c2w_0[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(c2w_0[:3, 3], np.zeros(3), atol=1e-6)
    assert normalized[0]["euclidean_dist_from_anchor"] == 0.0

    # Pano 1 must have correct Euclidean distance (approx 5.64m)
    dist_1 = normalized[1]["euclidean_dist_from_anchor"]
    delta_arkit = records[1]["origin_arkit"] - records[0]["origin_arkit"]
    expected_dist = float(np.linalg.norm(delta_arkit))
    assert pytest.approx(dist_1, rel=1e-4) == expected_dist
    assert pytest.approx(dist_1, abs=0.05) == 5.64

    # Relative heading rotation: 78.48 - 223.79 = -145.31 deg = 214.69 deg
    delta_h = (78.48435 - 223.79487) % 360.0
    R1 = normalized[1]["R_pano_relative"]
    np.testing.assert_allclose(R1, rotation_matrix_y(delta_h), atol=1e-5)


def test_heading_modes():
    records = [
        {
            "origin_arkit": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "magnetic_heading_deg": 45.0,
            "image_path": Path("pano_0.jpg"),
            "json_path": Path("pano_0.json"),
        },
        {
            "origin_arkit": np.array([1.0, 0.0, 2.0], dtype=np.float32),
            "magnetic_heading_deg": 90.0,
            "image_path": Path("pano_1.jpg"),
            "json_path": Path("pano_1.json"),
        },
    ]

    # Test north_aligned mode: all rotations identity
    norm_north = compute_level1_normalized_poses(records, heading_mode="north_aligned")
    np.testing.assert_allclose(norm_north[0]["R_pano_relative"], np.eye(3))
    np.testing.assert_allclose(norm_north[1]["R_pano_relative"], np.eye(3))
    # In north aligned: cv coords delta = [1.0, -0.0, -2.0]
    np.testing.assert_allclose(norm_north[1]["t_pano_relative"], [1.0, 0.0, -2.0])

    # Test raw_compass mode
    norm_compass = compute_level1_normalized_poses(records, heading_mode="raw_compass")
    np.testing.assert_allclose(norm_compass[0]["R_pano_relative"], rotation_matrix_y(45.0), atol=1e-5)
    np.testing.assert_allclose(norm_compass[1]["R_pano_relative"], rotation_matrix_y(90.0), atol=1e-5)


def test_level2_perspective_face_definitions():
    cubemap_faces = get_cubemap_faces_definitions()
    assert len(cubemap_faces) == 6
    names = [f[0] for f in cubemap_faces]
    assert names == ["front", "right", "back", "left", "up", "down"]

    ring_faces = get_ring_faces_definitions(num_ring_views=12, include_up_down=True)
    assert len(ring_faces) == 14  # 12 horizontal + up + down

    for name, yaw, pitch in cubemap_faces:
        R = compute_face_rotation_matrix(yaw, pitch)
        # Check orthogonality
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-5)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_find_pano_json_pairs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create test images and jsons
        (tmp_path / "station1_blurred.jpg").touch()
        (tmp_path / "station1.json").touch()
        (tmp_path / "station2.png").touch()
        (tmp_path / "station2.json").touch()
        (tmp_path / "extra_unmatched.jpg").touch()

        pairs = find_pano_json_pairs(tmp_path)
        assert len(pairs) == 2
        assert pairs[0][0].name == "station1_blurred.jpg"
        assert pairs[0][1].name == "station1.json"
        assert pairs[1][0].name == "station2.png"
        assert pairs[1][1].name == "station2.json"


def test_end_to_end_processing_synthetic():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        # Create 2 small synthetic equirectangular images (128x64)
        for i, name in enumerate(["station_A", "station_B"]):
            img = np.zeros((64, 128, 3), dtype=np.uint8)
            img[:32, :] = [255, 100, 50]  # top half
            img[32:, :] = [50, 100, 255]  # bottom half
            cv2.imwrite(str(in_dir / f"{name}_blurred.jpg"), img)

            meta = {
                "magneticHeading": 90.0 * i,
                "arkitCapture": {
                    "units": "meters",
                    "axes": "camera: +X right, +Y up, look -Z; world: +Y up, yaw0 north, +X east, +Z south",
                    "sphere": {
                        "origin": [float(i * 3), 0.5, float(i * 4)]
                    }
                }
            }
            with open(in_dir / f"{name}.json", "w") as f:
                json.dump(meta, f)

        # Run pipeline with cubemap mode
        results = process_arkit_panoramas(
            input_dir=in_dir,
            output_dir=out_dir,
            split_mode="cubemap",
            face_size=64,
            fov_deg=90.0,
            verbose=False,
        )

        assert results["num_views"] == 12  # 2 panos * 6 faces
        poses = np.load(results["poses_npy"])
        intrinsics = np.load(results["intrinsics_npy"])

        assert poses.shape == (12, 4, 4)
        assert intrinsics.shape == (12, 3, 3)

        # Verify all faces of Station A share the origin [0, 0, 0]
        for idx in range(6):
            np.testing.assert_allclose(poses[idx, :3, 3], [0, 0, 0], atol=1e-5)

        # Verify Station B faces share Station B's translated position
        pos_b = poses[6, :3, 3]
        for idx in range(6, 12):
            np.testing.assert_allclose(poses[idx, :3, 3], pos_b, atol=1e-5)

        # Verify distance
        dist_b = np.linalg.norm(pos_b)
        assert pytest.approx(dist_b, abs=1e-4) == 5.0  # sqrt(3^2 + 4^2) = 5.0

        # Verify files on disk
        final_out_dir = Path(results["output_dir"])
        assert (final_out_dir / "poses.json").exists()
        assert (final_out_dir / "normalized_arkit_metadata.json").exists()
        assert len(list((final_out_dir / "images").glob("*.jpg"))) == 12


if __name__ == "__main__":
    pytest.main([__file__])
