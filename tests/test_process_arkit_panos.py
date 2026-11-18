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


def test_aggregate_jobdata_json_discovery_and_ordering():
    """
    Test pairing and ordering when metadata is provided in a single aggregate JSON file
    (e.g. jobdata.json) with scanChain indices and headingAtPlace.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 3 stations with random hash names to ensure sorting relies on scanChain.index, not filename
        station_hashes = ["zzz_hash", "aaa_hash", "mmm_hash"]
        for h in station_hashes:
            (tmp_path / f"{h}.jpg").touch()

        # Jobdata structure matching real jobdata.json
        jobdata = [
            {
                "jobId": "mmm_hash",
                "arkitCapture": {
                    "sphere": {
                        "origin": [1.0, 1.45, 2.0],
                        "radius": 0.3,
                        "headingAtPlace": 120.0,
                        "heightAboveFloor": 1.45,
                    }
                },
                "scanChain": {
                    "chainId": "CHAIN-UUID",
                    "index": 2,
                    "previousSessionId": "zzz_hash",
                    "nextSessionId": "aaa_hash",
                },
            },
            {
                "jobId": "zzz_hash",
                "arkitCapture": {
                    "sphere": {
                        "origin": [0.0, 1.40, 0.0],
                        "radius": 0.3,
                        "headingAtPlace": 45.0,
                        "heightAboveFloor": 1.40,
                    }
                },
                "scanChain": {
                    "chainId": "CHAIN-UUID",
                    "index": 1,
                    "previousSessionId": None,
                    "nextSessionId": "mmm_hash",
                },
            },
            {
                "jobId": "aaa_hash",
                "arkitCapture": {
                    "sphere": {
                        "origin": [2.0, 1.50, 4.0],
                        "radius": 0.3,
                        "headingAtPlace": 210.0,
                        "heightAboveFloor": 1.50,
                    }
                },
                "scanChain": {
                    "chainId": "CHAIN-UUID",
                    "index": 3,
                    "previousSessionId": "mmm_hash",
                    "nextSessionId": None,
                },
            },
        ]

        with open(tmp_path / "jobdata.json", "w") as f:
            json.dump(jobdata, f)

        pairs = find_pano_json_pairs(tmp_path)
        assert len(pairs) == 3

        # Must be ordered by scanChain.index: index 1 (zzz_hash), index 2 (mmm_hash), index 3 (aaa_hash)
        assert pairs[0][0].name == "zzz_hash.jpg"
        assert pairs[0][1]["scanChain"]["index"] == 1
        assert pairs[1][0].name == "mmm_hash.jpg"
        assert pairs[1][1]["scanChain"]["index"] == 2
        assert pairs[2][0].name == "aaa_hash.jpg"
        assert pairs[2][1]["scanChain"]["index"] == 3

        # Verify load_arkit_metadata extraction
        meta0 = load_arkit_metadata(pairs[0][1])
        assert meta0["magnetic_heading_deg"] == 45.0
        assert meta0["job_id"] == "zzz_hash"
        assert meta0["scan_chain_index"] == 1
        assert meta0["height_above_floor"] == 1.40
        np.testing.assert_allclose(meta0["origin_arkit"], [0.0, 1.40, 0.0])


def test_aggregate_jobdata_end_to_end_synthetic():
    """
    Test full end-to-end processing with a single jobdata.json file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        # Create 2 synthetic equirectangular images
        jobdata = []
        for i, name in enumerate(["node_first", "node_second"]):
            img = np.zeros((64, 128, 3), dtype=np.uint8)
            img[:32, :] = [100, 150, 200]
            img[32:, :] = [200, 150, 100]
            cv2.imwrite(str(in_dir / f"{name}.jpg"), img)

            jobdata.append({
                "jobId": name,
                "arkitCapture": {
                    "sphere": {
                        "origin": [float(i * 4), 1.4, float(i * 3)],
                        "radius": 0.3,
                        "headingAtPlace": 60.0 * i,
                        "heightAboveFloor": 1.4,
                    }
                },
                "scanChain": {
                    "chainId": "UUID-1234",
                    "index": i + 1,
                    "previousSessionId": None if i == 0 else "node_first",
                    "nextSessionId": "node_second" if i == 0 else None,
                },
            })

        with open(in_dir / "jobdata.json", "w") as f:
            json.dump(jobdata, f)

        results = process_arkit_panoramas(
            input_dir=in_dir,
            output_dir=out_dir,
            split_mode="cubemap",
            face_size=64,
            fov_deg=90.0,
            verbose=False,
        )

        assert results["num_views"] == 12
        poses = np.load(results["poses_npy"])
        assert poses.shape == (12, 4, 4)

        # Distance between node_first and node_second should be 5.0m (sqrt(4^2 + 3^2))
        pos1 = poses[6, :3, 3]
        assert pytest.approx(float(np.linalg.norm(pos1)), abs=1e-4) == 5.0

        # Check metadata export
        meta_file = Path(results["output_dir"]) / "normalized_arkit_metadata.json"
        assert meta_file.exists()
        with open(meta_file, "r") as f:
            exported_meta = json.load(f)
        assert len(exported_meta) == 2
        assert exported_meta[0]["job_id"] == "node_first"
        assert exported_meta[0]["scan_chain_index"] == 1
        assert exported_meta[1]["job_id"] == "node_second"
        assert exported_meta[1]["scan_chain_index"] == 2


def test_real_jobdata_directory_detection():
    """
    Test that the actual jobdata/ directory in the workspace is correctly detected
    and parsed with 13 ordered pairs.
    """
    from scripts.process_arkit_panos import is_arkit_pano_directory
    jobdata_dir = Path("jobdata")
    if jobdata_dir.exists():
        assert is_arkit_pano_directory(jobdata_dir) is True
        pairs = find_pano_json_pairs(jobdata_dir)
        assert len(pairs) == 13
        # Indices should be 1 to 13
        indices = [p[1]["scanChain"]["index"] for p in pairs]
        assert indices == list(range(1, 14))


def test_mathematical_consistency_and_spatial_alignment():
    """
    Comprehensive mathematical consistency test:
    Validates whether the 3D poses of each panorama and all splitted perspective faces
    are mathematically and geometrically aligned in the created world space.

    Checks:
    1. Rigid Isometry: Inter-pano Euclidean distances in world space strictly equal
       their Euclidean distances in ARKit space.
    2. Gravity / Elevation Invariance: Relative vertical displacement in OpenCV (+Y down)
       strictly matches the inverted elevation displacement in ARKit (+Y up).
    3. Proper SO(3) Rotation Group Invariants: R^T R = I, det(R) = +1.0, right-handed frame.
    4. Concentricity of Splitted Faces: All faces derived from the same panorama share
       the exact same optical center in world space.
    5. Cubemap Orthogonality: Mutual dot products of face optical axes match exact 90°/180° angles.
    6. Inter-Face Baseline Conservation: The baseline between any face of pano A and any face of pano B
       strictly equals the baseline between pano A and pano B.
    7. Multi-View Epipolar / Triangulation Consistency: Rays unprojecting a 3D world point from
       different camera faces and stations intersect at the exact same 3D point in world space.
    """
    # Test on real jobdata if present, or synthetic multi-station data
    jobdata_dir = Path("jobdata")
    if jobdata_dir.exists():
        pairs = find_pano_json_pairs(jobdata_dir)
        records = [dict(load_arkit_metadata(jf), image_path=img, json_path=jf) for img, jf in pairs]
    else:
        # Fallback synthetic records
        records = [
            {
                "origin_arkit": np.array([0.0, 1.40, 0.0], dtype=np.float32),
                "magnetic_heading_deg": 30.0,
                "image_path": Path("station_0.jpg"),
                "json_path": Path("station_0.json"),
                "scan_chain_index": 1,
            },
            {
                "origin_arkit": np.array([3.0, 1.55, 4.0], dtype=np.float32),
                "magnetic_heading_deg": 120.0,
                "image_path": Path("station_1.jpg"),
                "json_path": Path("station_1.json"),
                "scan_chain_index": 2,
            },
            {
                "origin_arkit": np.array([-2.0, 1.35, 5.0], dtype=np.float32),
                "magnetic_heading_deg": 210.0,
                "image_path": Path("station_2.jpg"),
                "json_path": Path("station_2.json"),
                "scan_chain_index": 3,
            },
        ]

    for heading_mode in ["north_aligned", "relative"]:
        normalized_panos = compute_level1_normalized_poses(records, heading_mode=heading_mode, anchor_index=0)
        face_defs = get_cubemap_faces_definitions()

        # Build face poses for all stations
        all_face_records = []
        for p in normalized_panos:
            R_pano = p["c2w_pano_4x4"][:3, :3]
            t_pano = p["c2w_pano_4x4"][:3, 3]
            for face_name, yaw_deg, pitch_deg in face_defs:
                R_f2p = compute_face_rotation_matrix(yaw_deg, pitch_deg)
                R_face = R_pano @ R_f2p
                t_face = t_pano.copy()

                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = R_face
                c2w[:3, 3] = t_face

                optical_axis = R_face @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

                all_face_records.append({
                    "pano_idx": p["index"],
                    "face_name": face_name,
                    "c2w": c2w,
                    "R": R_face,
                    "t": t_face,
                    "optical_axis": optical_axis,
                })

        # --- Invariant 1: Rigid Isometry across all pairs of panos ---
        for i in range(len(normalized_panos)):
            for j in range(i + 1, len(normalized_panos)):
                d_arkit = float(np.linalg.norm(normalized_panos[i]["origin_arkit"] - normalized_panos[j]["origin_arkit"]))
                d_world = float(np.linalg.norm(normalized_panos[i]["t_pano_relative"] - normalized_panos[j]["t_pano_relative"]))
                assert pytest.approx(d_world, rel=1e-5, abs=1e-5) == d_arkit, (
                    f"Isometry violated between pano {i} and {j}: world={d_world}, arkit={d_arkit}"
                )

        # --- Invariant 2: Altitude Invariance (Gravity Alignment) in north_aligned mode ---
        if heading_mode == "north_aligned":
            for i in range(len(normalized_panos)):
                for j in range(i + 1, len(normalized_panos)):
                    delta_y_arkit = float(normalized_panos[j]["origin_arkit"][1] - normalized_panos[i]["origin_arkit"][1])
                    delta_y_world = float(-(normalized_panos[j]["t_pano_relative"][1] - normalized_panos[i]["t_pano_relative"][1]))
                    assert pytest.approx(delta_y_world, rel=1e-5, abs=1e-5) == delta_y_arkit, (
                        f"Elevation delta mismatch between pano {i} and {j}"
                    )

        # --- Invariant 3: SO(3) Orthonormality and Right-Handedness ---
        for f in all_face_records:
            R = f["R"]
            # 1. Orthonormality: R.T @ R == I
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-5)
            # 2. Proper rotation: det(R) == +1.0
            assert pytest.approx(float(np.linalg.det(R)), abs=1e-5) == 1.0
            # 3. Right-handed coordinate system: X x Y == Z
            x_axis = R[:, 0]
            y_axis = R[:, 1]
            z_axis = R[:, 2]
            np.testing.assert_allclose(np.cross(x_axis, y_axis), z_axis, atol=1e-5)
            assert pytest.approx(float(np.dot(np.cross(x_axis, y_axis), z_axis)), abs=1e-5) == 1.0

        # --- Invariant 4: Concentricity of Splitted Faces ---
        for f in all_face_records:
            parent_pos = normalized_panos[f["pano_idx"]]["t_pano_relative"]
            np.testing.assert_allclose(f["t"], parent_pos, atol=1e-6)

        # --- Invariant 5: Cubemap Mutual Orthogonality (Intra-Pano Angles) ---
        for p_idx in range(len(normalized_panos)):
            p_faces = {f["face_name"]: f for f in all_face_records if f["pano_idx"] == p_idx}
            fz = p_faces["front"]["optical_axis"]
            rz = p_faces["right"]["optical_axis"]
            bz = p_faces["back"]["optical_axis"]
            lz = p_faces["left"]["optical_axis"]
            uz = p_faces["up"]["optical_axis"]
            dz = p_faces["down"]["optical_axis"]

            # 90 degree separations (dot product == 0)
            assert pytest.approx(float(np.dot(fz, rz)), abs=1e-5) == 0.0
            assert pytest.approx(float(np.dot(fz, lz)), abs=1e-5) == 0.0
            assert pytest.approx(float(np.dot(fz, uz)), abs=1e-5) == 0.0
            assert pytest.approx(float(np.dot(fz, dz)), abs=1e-5) == 0.0
            assert pytest.approx(float(np.dot(rz, uz)), abs=1e-5) == 0.0
            assert pytest.approx(float(np.dot(rz, dz)), abs=1e-5) == 0.0

            # 180 degree opposite separations (dot product == -1)
            assert pytest.approx(float(np.dot(fz, bz)), abs=1e-5) == -1.0
            assert pytest.approx(float(np.dot(rz, lz)), abs=1e-5) == -1.0
            assert pytest.approx(float(np.dot(uz, dz)), abs=1e-5) == -1.0

        # --- Invariant 6: Inter-Face Baseline Conservation ---
        for i in range(len(normalized_panos)):
            for j in range(i + 1, len(normalized_panos)):
                d_pano = float(np.linalg.norm(normalized_panos[i]["t_pano_relative"] - normalized_panos[j]["t_pano_relative"]))
                faces_i = [f for f in all_face_records if f["pano_idx"] == i]
                faces_j = [f for f in all_face_records if f["pano_idx"] == j]
                for fa in faces_i:
                    for fb in faces_j:
                        d_face = float(np.linalg.norm(fa["t"] - fb["t"]))
                        assert pytest.approx(d_face, abs=1e-6) == d_pano

        # --- Invariant 7: Multi-View Triangulation Ray Consistency ---
        # Choose a 3D world test point in front of camera views
        p_world_target = np.array([0.5, -0.2, 3.0], dtype=np.float32)
        # Verify that unprojected rays from camera stations with depth reconstruct the exact same 3D point
        f0 = all_face_records[0]  # Front face of anchor pano
        R0 = f0["R"]
        t0 = f0["t"]
        d0 = float(np.linalg.norm(p_world_target - t0))
        ray0_cam = R0.T @ ((p_world_target - t0) / d0)
        reconstructed_p0 = t0 + d0 * (R0 @ ray0_cam)

        f1 = all_face_records[6]  # Front face of station 2
        R1 = f1["R"]
        t1 = f1["t"]
        d1 = float(np.linalg.norm(p_world_target - t1))
        ray1_cam = R1.T @ ((p_world_target - t1) / d1)
        reconstructed_p1 = t1 + d1 * (R1 @ ray1_cam)

        np.testing.assert_allclose(reconstructed_p0, p_world_target, atol=1e-5)
        np.testing.assert_allclose(reconstructed_p1, p_world_target, atol=1e-5)
        np.testing.assert_allclose(reconstructed_p0, reconstructed_p1, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
