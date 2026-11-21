# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.reconstruct_all_possibilities import (
    ALL_POSSIBILITIES,
    main,
    transform_poses,
)


def test_all_possibilities_count_and_format():
    assert len(ALL_POSSIBILITIES) >= 20
    for name, cfg in ALL_POSSIBILITIES:
        assert isinstance(name, str)
        assert isinstance(cfg, dict)
        assert "tx" in cfg
        assert "ty" in cfg
        assert "tz" in cfg
        assert "desc" in cfg


def test_transform_poses_so3_and_translations():
    # Canonical camera pose
    R = np.array([
        [-0.68745, 0.0, -0.72623],
        [0.0, 1.0, 0.0],
        [0.72623, 0.0, -0.68745],
    ], dtype=np.float32)
    t = np.array([-5.511, -1.059, 7.212], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = t

    for name, cfg in ALL_POSSIBILITIES:
        transformed = transform_poses([pose], cfg)
        assert len(transformed) == 1
        p_new = transformed[0]
        R_new = p_new[:3, :3]
        t_new = p_new[:3, 3]

        # Invariant 1: det(R_new) MUST BE +1.0 (valid SO(3))
        det = float(np.linalg.det(R_new))
        assert pytest.approx(det, abs=1e-5) == 1.0, f"Failed SO(3) det=1.0 for combo: {name}"

        # Invariant 2: Orthogonality R @ R.T == I
        np.testing.assert_allclose(R_new @ R_new.T, np.eye(3), atol=1e-5)

        # Invariant 3: Translation sign behavior
        if not cfg.get("swap_xz"):
            if cfg["tx"]:
                assert pytest.approx(t_new[0], abs=1e-4) == -t[0]
            else:
                assert pytest.approx(t_new[0], abs=1e-4) == t[0]

            if cfg["ty"]:
                assert pytest.approx(t_new[1], abs=1e-4) == -t[1]
            else:
                assert pytest.approx(t_new[1], abs=1e-4) == t[1]

            if cfg["tz"]:
                assert pytest.approx(t_new[2], abs=1e-4) == -t[2]
            else:
                assert pytest.approx(t_new[2], abs=1e-4) == t[2]


def test_reconstruct_all_possibilities_cli_parsing():
    test_args = [
        "reconstruct_all_possibilities.py",
        "--images", "dummy_dir",
        "--output", "test_output",
        "--combos", "neg_x,neg_z",
    ]
    with mock.patch("sys.argv", test_args):
        with mock.patch("scripts.reconstruct_all_possibilities.reconstruct_all_possibilities") as mock_recon:
            main()
            assert mock_recon.called
            call_kwargs = mock_recon.call_args[1]
            assert call_kwargs["images"] == "dummy_dir"
            assert call_kwargs["output"] == "test_output"
            assert call_kwargs["combos_filter"] == "neg_x,neg_z"


if __name__ == "__main__":
    pytest.main([__file__])
