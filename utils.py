"""Small file, annotation, geometry and preview helpers (no model dependencies)."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def digest(value):
    data = (
        value.read_bytes()
        if isinstance(value, Path)
        else json.dumps(value, sort_keys=True).encode()
    )
    return hashlib.sha256(data).hexdigest()


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_image(path):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    path.write_bytes(encoded.tobytes())


def images_in(folder):
    paths = (
        sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if folder.exists()
        else []
    )
    stems = [p.stem.casefold() for p in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate image stems; rename files before processing.")
    return paths


def ordered(points):
    points = np.asarray(points, np.float64)
    center = points.mean(axis=0)
    result = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
    return np.roll(result, -np.lexsort((result[:, 0], result[:, 1]))[0], axis=0)


def polygon_ok(points):
    p = np.asarray(points, np.float32)
    return (
        p.ndim == 2
        and p.shape[1] == 2
        and len(p) >= 3
        and np.isfinite(p).all()
        and cv2.isContourConvex(p)
        and cv2.contourArea(p) > 1e-10
    )


def parse_labels(text, fmt, class_count):
    objects = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            cls = int(fields[0])
            values = np.array([float(v) for v in fields[1:]], np.float64)
            if not 0 <= cls < class_count or not np.isfinite(values).all():
                raise ValueError("class/coordinates invalid")
            if fmt == "detection":
                if len(values) != 4:
                    raise ValueError("expected 5 fields")
                x, y, w, h = values
                if w <= 0 or h <= 0:
                    raise ValueError("empty detection")
                points = np.array(
                    [
                        [x - w / 2, y - h / 2],
                        [x + w / 2, y - h / 2],
                        [x + w / 2, y + h / 2],
                        [x - w / 2, y + h / 2],
                    ]
                )
            else:
                if (fmt == "obb" and len(values) != 8) or len(values) < 6 or len(values) % 2:
                    raise ValueError("invalid vertex count")
                points = values.reshape(-1, 2)
            if not polygon_ok(points) or (points < -1e-7).any() or (points > 1 + 1e-7).any():
                raise ValueError("expected an in-frame convex polygon in perimeter order")
            objects.append(
                {"class_id": cls, "points": np.clip(points, 0, 1).tolist(), "geometry": fmt}
            )
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Label line {number}: {exc}") from exc
    return objects


def fit_box(points, fmt):
    points = np.asarray(points, np.float64)
    if fmt == "detection":
        lo, hi = points.min(axis=0), points.max(axis=0)
        return np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])
    return ordered(cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32))))


def label_text(objects, fmt, size):
    lines = []
    scale = np.array(size, np.float64)  # width, height
    for obj in objects:
        p = np.asarray(obj["points"]) * scale
        box = fit_box(p, fmt) / scale
        if (box < -1e-6).any() or (box > 1 + 1e-6).any():
            raise ValueError("Exported box extends beyond image; cannot clamp an OBB.")
        box = np.clip(box, 0, 1)  # floating-point tolerance only
        if fmt == "detection":
            lo, hi = box.min(axis=0), box.max(axis=0)
            values = np.r_[(lo + hi) / 2, hi - lo]
        else:
            values = box.ravel()
        lines.append(str(obj["class_id"]) + " " + " ".join(f"{v:.8f}" for v in values))
    return "\n".join(lines) + ("\n" if lines else "")


def transformed(points, matrix):
    p = np.c_[points, np.ones(len(points))] @ matrix.T
    if (p[:, 2] <= 1e-8).any():
        raise ValueError("Invalid perspective denominator")
    return p[:, :2] / p[:, 2:3]


def clip_polygon(points, width, height):
    """Sutherland-Hodgman clipping against the image rectangle."""
    result = list(np.asarray(points, np.float64))
    for axis, bound, sign in ((0, 0, 1), (0, width, -1), (1, 0, 1), (1, height, -1)):
        source, result = result, []
        if not source:
            break
        previous = source[-1]
        for current in source:
            inside = sign * (current[axis] - bound) >= 0
            previous_inside = sign * (previous[axis] - bound) >= 0
            if inside != previous_inside:
                t = (bound - previous[axis]) / (current[axis] - previous[axis])
                result.append(previous + t * (current - previous))
            if inside:
                result.append(current)
            previous = current
    return np.asarray(result, np.float64).reshape(-1, 2)


def transform_objects(objects, matrix, input_size, output_size, fmt, policy):
    """Return accurate normalized geometry, or reject the entire candidate."""
    output = []
    size = np.array(output_size)
    for obj in objects:
        if fmt == "obb" and obj["geometry"] == "detection":
            raise ValueError("OBB output requires reviewed OBB corners or polygon outlines.")
        p = transformed(np.asarray(obj["points"]) * input_size, matrix)
        outside = (p < -1e-5).any() or (p > size + 1e-5).any()
        if outside:
            if not policy["allow_crop"] or obj["geometry"] == "detection":
                raise ValueError("Object would be cropped")
            clipped = clip_polygon(p, *output_size)
            if (
                not polygon_ok(clipped)
                or cv2.contourArea(clipped.astype(np.float32))
                / cv2.contourArea(p.astype(np.float32))
                < policy["min_visibility"]
            ):
                raise ValueError("Insufficient visibility")
            p = clipped
        box = fit_box(p, fmt)
        if (box < -1e-4).any() or (box > size + 1e-4).any():
            raise ValueError("Fitted box extends outside frame")
        lengths = np.linalg.norm(box - np.roll(box, 1, axis=0), axis=1)
        if lengths.min() < policy["min_side"]:
            raise ValueError("Object too small")
        if (
            fmt == "obb"
            and cv2.contourArea(box.astype(np.float32)) / cv2.contourArea(p.astype(np.float32))
            > policy["max_obb_area_ratio"] + 1e-6
        ):
            raise ValueError("OBB fit contains excessive background")
        output.append({**obj, "points": (np.clip(p, 0, size) / size).tolist()})
    return output


def geometry_matrix(width, height, cfg, rng, sensitive=False):
    from underwater_effects import enabled

    settings, side = cfg["augmentation"], cfg["image_size"]
    ratio = min(side / width, side / height)
    base = np.array(
        [
            [ratio, 0, (side - width * ratio) / 2],
            [0, ratio, (side - height * ratio) / 2],
            [0, 0, 1.0],
        ]
    )

    def sample(name, default):
        if sensitive and name in ("rotation", "aspect_ratio", "perspective"):
            return default
        if not enabled(settings, name, rng):
            return default
        s = settings[name]
        return (
            rng.uniform(-s["limit"], s["limit"])
            if "limit" in s
            else rng.uniform(s["min"], s["max"])
        )

    angle = np.deg2rad(sample("rotation", 0))
    scale, aspect = sample("scaling", 1), sample("aspect_ratio", 1)
    # Stretch in original image axes, then rotate; one resampling operation.
    linear = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ np.diag(
        [scale * np.sqrt(aspect), scale / np.sqrt(aspect)]
    )
    shift = np.zeros(2)
    if enabled(settings, "translation", rng):
        shift = (
            rng.uniform(-settings["translation"]["limit"], settings["translation"]["limit"], 2)
            * side
        )
    affine = np.eye(3)
    affine[:2, :2] = linear
    affine[:2, 2] = side / 2 - linear @ np.full(2, side / 2) + shift
    matrix = affine @ base
    if not sensitive and enabled(settings, "perspective", rng):
        src = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
        dst = (
            src
            + rng.uniform(-1, 1, (4, 2)).astype(np.float32)
            * side
            * settings["perspective"]["limit"]
        )
        matrix = cv2.getPerspectiveTransform(src, dst) @ matrix
    return matrix, base


def preview(items, destination, classes, fmt=None):
    """Write every annotated preview plus a small contact sheet."""
    tiles = []
    destination.mkdir(parents=True, exist_ok=True)
    for path, objects, caption in items:
        image = read_image(path)
        h, w = image.shape[:2]
        for obj in objects:
            p = np.asarray(obj["points"]) * [w, h]
            if fmt:
                p = fit_box(p, fmt)
            color = (0, 210, 255) if obj.get("low_confidence") else (80, 255, 80)
            cv2.polylines(image, [p.round().astype(np.int32)], True, color, 2, cv2.LINE_AA)
            label = f"{obj['class_id']}: {classes[obj['class_id']]}"
            if obj.get("confidence") is not None:
                label += f" {obj['confidence']:.2f}"
            xy = tuple(np.maximum(p.min(axis=0), [0, 18]).astype(int))
            cv2.putText(image, label, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(
            image,
            caption[:75],
            (8, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        write_image(destination / (path.stem + ".jpg"), image)
        if len(tiles) < 12:
            r = min(320 / w, 260 / h)
            tile = np.zeros((280, 320, 3), np.uint8)
            scaled = cv2.resize(image, (round(w * r), round(h * r)))
            tile[: scaled.shape[0], : scaled.shape[1]] = scaled
            tiles.append(tile)
    if not tiles:
        raise ValueError("No readable images to preview")
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
    write_image(destination / "contact_sheet.jpg", sheet)
    return destination / "contact_sheet.jpg"
