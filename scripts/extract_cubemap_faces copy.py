#!/usr/bin/env python3
"""Slice equirectangular panoramas into six cubemap faces with OpenCV RDF poses.

World frame is ARKit: meters, +Y up, yaw 0 along world -Z, clockwise from above.
Each face pose is camera-to-world. Columns of R are OpenCV RDF: right, down, forward.

Requires numpy and Pillow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FACE_ORDER = ("front", "right", "back", "left", "up", "down")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Columns are the face camera axes expressed in the front-face RDF frame.
LOCAL_ROTATIONS = {
    "front": np.eye(3),
    "right": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    "back": np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
    "left": np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    "up": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "down": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
}


def front_rotation(heading_deg: float) -> np.ndarray:
    """Camera-to-world rotation for the panorama center (the front face)."""
    theta = np.deg2rad(heading_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    # Clockwise from above: at 0° forward is world -Z, at 90° forward is world +X.
    # Columns are right, down, forward. This matches column-major originPose.
    return np.array(
        [[c, 0.0, s], [0.0, -1.0, 0.0], [s, 0.0, -c]],
        dtype=np.float64,
    )


def assert_rotation(rotation: np.ndarray, label: str) -> None:
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6:
        raise AssertionError(f"{label}: det(R) = {np.linalg.det(rotation)} not +1")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise AssertionError(f"{label}: R is not orthonormal")


def column_major(rotation: np.ndarray, translation: np.ndarray) -> list[float]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix.flatten(order="F").tolist()


def face_pose(position: np.ndarray, heading_deg: float, face: str) -> np.ndarray:
    rotation = front_rotation(heading_deg) @ LOCAL_ROTATIONS[face]
    assert_rotation(rotation, face)
    return rotation


def intrinsics(face_size: int, fov_deg: float = 90.0) -> dict:
    half = face_size / 2.0
    tan_half_fov = float(np.tan(np.deg2rad(fov_deg) / 2.0))
    f = half / tan_half_fov
    return {
        "width": face_size,
        "height": face_size,
        "fx": f,
        "fy": f,
        "cx": half,
        "cy": half,
        "fovDeg": float(fov_deg),
    }


def sample_face(
    equirect: np.ndarray, face: str, face_size: int, fov_deg: float = 90.0
) -> np.ndarray:
    """Bilinear sample of one face at specified FoV. Pixel centers are (u+0.5, v+0.5)."""
    height, width = equirect.shape[:2]
    half = face_size / 2.0
    tan_half_fov = float(np.tan(np.deg2rad(fov_deg) / 2.0))
    cols = ((np.arange(face_size, dtype=np.float64) + 0.5 - half) / half) * tan_half_fov
    rows = ((np.arange(face_size, dtype=np.float64) + 0.5 - half) / half) * tan_half_fov
    grid_x, grid_y = np.meshgrid(cols, rows)
    rays = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
    pano = rays @ LOCAL_ROTATIONS[face].T
    x = pano[..., 0]
    y = pano[..., 1]
    z = pano[..., 2]
    lon = np.arctan2(x, z)
    lat_up = np.arctan2(-y, np.hypot(x, z))
    map_x = np.mod((lon / (2.0 * np.pi) + 0.5) * width, width)
    map_y = np.clip((0.5 - lat_up / np.pi) * height, 0.0, height - 1.0)
    return bilinear_sample(equirect, map_x, map_y)


def bilinear_sample(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    wx = (map_x - x0).astype(np.float32)[..., None]
    wy = (map_y - y0).astype(np.float32)[..., None]
    x0 = np.mod(x0, width)
    x1 = np.mod(x0 + 1, width)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    top = image[y0, x0].astype(np.float32) * (1.0 - wx) + image[y0, x1].astype(np.float32) * wx
    bot = image[y1, x0].astype(np.float32) * (1.0 - wx) + image[y1, x1].astype(np.float32) * wx
    sampled = top * (1.0 - wy) + bot * wy
    return np.clip(np.rint(sampled), 0, 255).astype(np.uint8)


def load_metadata(path: Path) -> list[dict]:
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise SystemExit(f"error: {path} must be a JSON array of panorama records")
    return data


def index_records(records: list[dict]) -> dict[str, dict]:
    indexed = {}
    for record in records:
        job_id = record.get("jobId")
        if not job_id:
            raise SystemExit("error: a metadata record is missing jobId")
        if job_id in indexed:
            raise SystemExit(f"error: duplicate jobId {job_id}")
        indexed[job_id] = record
    return indexed


def origin_xyz(record: dict) -> np.ndarray | None:
    arkit = record.get("arKitData") or {}
    origin = arkit.get("origin")
    if not origin:
        return None
    return np.array([origin["x"], origin["y"], origin["z"]], dtype=np.float64)


def check_heading_convention(records: list[dict]) -> None:
    """Compare originHeading's look direction with the stored ARKit originPose."""
    worst = 0.0
    worst_id = None
    count = 0
    for record in records:
        arkit = record.get("arKitData") or {}
        pose = arkit.get("originPose")
        heading = arkit.get("originHeading")
        if pose is None or heading is None:
            continue
        matrix = np.array(pose, dtype=np.float64).reshape(4, 4, order="F")
        stored = -matrix[:3, 2]
        stored /= np.linalg.norm(stored)
        predicted = front_rotation(heading)[:, 2]
        predicted /= np.linalg.norm(predicted)
        error = float(np.degrees(np.arccos(np.clip(np.dot(stored, predicted), -1.0, 1.0))))
        count += 1
        if error > worst:
            worst = error
            worst_id = record.get("jobId")
    if count == 0:
        print("heading convention check: no originPose/originHeading pairs to compare")
        return
    print(
        f"heading convention check: worst look-direction error {worst:.6f}° "
        f"over {count} poses ({worst_id})"
    )


def discover_images(images_dir: Path | None, overrides: list[tuple[str, Path]]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if images_dir is not None:
        if not images_dir.is_dir():
            raise SystemExit(f"error: images directory not found: {images_dir}")
        for path in sorted(images_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTS and path.is_file():
                found[path.stem] = path
    for job_id, path in overrides:
        if not path.is_file():
            raise SystemExit(f"error: image not found for {job_id}: {path}")
        found[job_id] = path
    return found


def pose_record(
    face: str, rotation: np.ndarray, position: np.ndarray, face_size: int, fov_deg: float = 90.0
) -> dict:
    return {
        "name": face,
        "cameraToWorld": column_major(rotation, position),
        "R": rotation.tolist(),
        "t": position.tolist(),
        "intrinsics": intrinsics(face_size, fov_deg),
    }


def process_job(
    record: dict,
    image_path: Path,
    out_dir: Path,
    face_size: int | None,
    fov_deg: float = 90.0,
) -> dict | None:
    job_id = record["jobId"]
    position = origin_xyz(record)
    if position is None:
        print(f"error: {job_id}: arKitData.origin is missing; skipped", file=sys.stderr)
        return None
    heading = record.get("magneticHeading")
    if heading is None:
        print(f"error: {job_id}: magneticHeading is missing; skipped", file=sys.stderr)
        return None

    with Image.open(image_path) as image:
        equirect = np.asarray(image.convert("RGB"))
    if equirect.ndim != 3 or equirect.shape[2] != 3:
        print(f"error: {job_id}: expected an RGB equirectangular image", file=sys.stderr)
        return None

    size = face_size if face_size else equirect.shape[1] // 4
    if size < 2:
        print(f"error: {job_id}: face size {size} is too small", file=sys.stderr)
        return None

    job_dir = out_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    faces = []
    for face in FACE_ORDER:
        rotation = face_pose(position, float(heading), face)
        sampled = sample_face(equirect, face, size, fov_deg=fov_deg)
        Image.fromarray(sampled, mode="RGB").save(job_dir / f"{face}.jpg", quality=95)
        faces.append(pose_record(face, rotation, position, size, fov_deg=fov_deg))

    payload = {
        "jobId": job_id,
        "image": str(image_path),
        "position": position.tolist(),
        "magneticHeadingDeg": float(heading),
        "convention": (
            "camera-to-world, column-major, OpenCV RDF "
            "(+X right, +Y down, +Z forward), ARKit world +Y up, meters"
        ),
        "faces": faces,
    }
    with (job_dir / "poses.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {job_id}: 6 faces at {size}px, heading {float(heading):.4f}°")
    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="JSON array of panorama records")
    parser.add_argument("--images", type=Path, help="Directory of {jobId}.jpg/.jpeg/.png equirects")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--face-size",
        type=int,
        default=0,
        help="Square face size in pixels. 0 uses equirect width // 4",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="Field of view in degrees (default: 90.0; e.g. 100.0 for 10° visual overlap)",
    )
    parser.add_argument(
        "--image",
        nargs=2,
        action="append",
        default=[],
        metavar=("JOBID", "PATH"),
        help="Equirectangular image for one job id. Repeatable. Overrides --images",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    records = load_metadata(args.metadata)
    by_id = index_records(records)
    check_heading_convention(records)

    overrides = [(job_id, Path(path)) for job_id, path in args.image]
    images = discover_images(args.images, overrides)
    if not images:
        print("error: no equirectangular images given (--images or --image)", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    failed = False
    for job_id in sorted(images):
        record = by_id.get(job_id)
        if record is None:
            print(f"error: {job_id}: no metadata record; skipped", file=sys.stderr)
            failed = True
            continue
        payload = process_job(record, images[job_id], args.out, args.face_size or None, fov_deg=args.fov)
        if payload is None:
            failed = True
            continue
        written.append(payload)

    with (args.out / "poses.json").open("w") as handle:
        json.dump(written, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.out / 'poses.json'} ({len(written)} panoramas)")
    return 1 if failed or not written else 0


if __name__ == "__main__":
    raise SystemExit(main())
