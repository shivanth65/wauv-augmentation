"""WAUV: raw images -> reviewed labels -> conservative augmentation. Python 3.11+."""

import argparse
import copy
import csv
import os
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import cv2
import numpy as np
import yaml

from underwater_effects import apply_effects
from utils import (
    digest,
    fit_box,
    geometry_matrix,
    images_in,
    label_text,
    parse_labels,
    preview,
    read_image,
    read_json,
    transform_objects,
    transformed,
    write_image,
    write_json,
)


def load_config(path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("input_dir", "review_dir", "approved_dir", "output_dir"):
        cfg[key] = (path.parent / cfg[key]).resolve()
    roots = [cfg[k] for k in ("input_dir", "review_dir", "approved_dir", "output_dir")]
    if any(
        a == b or a in b.parents or b in a.parents
        for i, a in enumerate(roots)
        for b in roots[i + 1 :]
    ):
        raise ValueError("Input, review, approved and output directories must not overlap.")
    classes = cfg["classes"]
    if (
        not classes
        or any(not isinstance(x, str) or not x for x in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("classes must be unique nonempty names")
    if cfg["output_format"] not in ("obb", "detection"):
        raise ValueError("output_format must be obb or detection")
    for key in ("image_size", "images_per_input"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not set(cfg["orientation_sensitive_classes"]) <= set(classes):
        raise ValueError("Unknown orientation-sensitive class")
    for name, s in cfg["augmentation"].items():
        if not isinstance(s["enabled"], bool) or not 0 <= s["probability"] <= 1:
            raise ValueError(f"Invalid enable/probability for {name}")
        for key, value in s.items():
            if key != "enabled" and (not isinstance(value, (int, float)) or not np.isfinite(value)):
                raise ValueError(f"Invalid numeric value in {name}")
        if "min" in s and not 0 < s["min"] <= s["max"]:
            raise ValueError(f"Invalid range for {name}")
        if "limit" in s and s["limit"] < 0 or "strength" in s and not 0 <= s["strength"] <= 1:
            raise ValueError(f"Invalid intensity for {name}")
    a = cfg["augmentation"]
    if (
        a["blur"]["max_sigma"] < 0
        or a["motion_blur"]["kernel"] < 3
        or a["motion_blur"]["kernel"] % 2 != 1
    ):
        raise ValueError("Blur sigma must be nonnegative; motion kernel must be odd and >=3")
    if not 1 <= a["jpeg"]["min"] <= a["jpeg"]["max"] <= 100:
        raise ValueError("JPEG quality must be 1..100")
    g = cfg["geometry"]
    if (
        not isinstance(g["attempts"], int)
        or g["attempts"] < 1
        or g["min_side"] <= 0
        or not 0 < g["min_visibility"] <= 1
        or g["max_obb_area_ratio"] < 1
    ):
        raise ValueError("Invalid geometry policy")
    d = cfg["detector"]
    if not 0 <= d["confidence"] <= d["low_confidence"] <= 1 or not 0 < d["iou"] <= 1:
        raise ValueError("Invalid confidence/IoU thresholds")
    if d["weights"]:
        d["weights"] = str((path.parent / d["weights"]).resolve())
    return cfg


def inventory(cfg):
    """Refresh raw images; retain predictions only for unchanged image bytes/classes."""
    paths = images_in(cfg["input_dir"] / "images")
    if not paths:
        raise ValueError(f"Put raw JPG/PNG images in {cfg['input_dir'] / 'images'}")
    manifest_path = cfg["review_dir"] / "manifest.json"
    old = read_json(manifest_path, {})
    records = {}
    for path in paths:
        record = {
            "image_hash": digest(path),
            "class_hash": digest(cfg["classes"]),
            "status": "needs_annotation",
        }
        previous = old.get(path.name, {})
        if all(previous.get(k) == record[k] for k in ("image_hash", "class_hash")):
            record = previous
        try:
            h, w = read_image(path).shape[:2]
            record.update(width=w, height=h)
        except ValueError as exc:
            record.update(status="error", message=str(exc))
            print(str(exc), file=sys.stderr)
        records[path.name] = record
    write_json(manifest_path, records)
    decisions = cfg["review_dir"] / "decisions.csv"
    existing = {}
    if decisions.exists():
        existing = {
            r["image"]: r for r in csv.DictReader(decisions.open(encoding="utf-8", newline=""))
        }
    with decisions.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "decision", "reason"])
        writer.writeheader()
        for name, record in records.items():
            previous = old.get(name, {})
            unchanged = all(previous.get(k) == record[k] for k in ("image_hash", "class_hash"))
            writer.writerow(
                existing.get(name)
                if unchanged and name in existing
                else {"image": name, "decision": "pending", "reason": ""}
            )
    return records


def proposal_objects(result, task, mapping, cfg):
    """Small adapter, also exercised with synthetic result objects in self-test."""
    predictions = result.obb if task == "obb" else result.boxes
    if predictions is None:
        return []
    objects = []
    for i in range(len(predictions)):
        cls = mapping[int(predictions.cls[i].item())]
        confidence = float(predictions.conf[i].item())
        if confidence < cfg["detector"]["confidence"]:
            continue
        if task == "obb":
            values = predictions.xyxyxyxyn[i].cpu().numpy().reshape(-1)
        else:
            values = predictions.xywhn[i].cpu().numpy()
        line = str(cls) + " " + " ".join(map(str, values))
        obj = parse_labels(line, task if task == "obb" else "detection", len(cfg["classes"]))[0]
        obj.update(
            confidence=confidence, low_confidence=confidence < cfg["detector"]["low_confidence"]
        )
        objects.append(obj)
    return objects


def propose(cfg):
    records = inventory(cfg)
    weights = cfg["detector"]["weights"]
    if not weights:
        print(
            "No custom detector configured. No labels were invented. Export to CVAT, draw/correct all signs, import, and explicitly approve. Configure detector.weights later."
        )
        return
    if not Path(weights).is_file():
        raise ValueError(
            "Custom weights do not exist. Set detector.weights to a local file; generic weights are not downloaded."
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ValueError(
            "Optional inference dependency missing: pip install ultralytics (manual workflow needs no model package)."
        ) from exc
    model = YOLO(weights)
    if model.task not in ("detect", "obb"):
        raise ValueError("Only custom YOLO detect/obb models are supported.")
    mapping = {}
    for key, name in model.names.items():
        target = cfg["detector"]["class_map"].get(name, name)
        if target not in cfg["classes"]:
            raise ValueError(
                f"Model class {name!r} has no configured WAUV class mapping. Generic models cannot recognize WAUV classes."
            )
        mapping[key] = cfg["classes"].index(target)
    provenance = {
        "weights_name": Path(weights).name,
        "weights_hash": digest(Path(weights)),
        "task": model.task,
    }
    for name, record in records.items():
        if record["status"] == "error":
            continue
        try:
            result = model.predict(
                source=read_image(cfg["input_dir"] / "images" / name),
                device="cpu",
                conf=cfg["detector"]["confidence"],
                iou=cfg["detector"]["iou"],
                verbose=False,
            )[0]
            objects = proposal_objects(result, model.task, mapping, cfg)
            write_json(
                cfg["review_dir"] / "proposals" / f"{Path(name).stem}.json",
                {
                    "image_hash": record["image_hash"],
                    "class_hash": record["class_hash"],
                    "objects": objects,
                    "model": provenance,
                },
            )
            record["status"] = "proposed" if objects else "no_detections"
            print(f"{name}: {len(objects)} proposed objects; review required")
        except Exception as exc:  # noqa: BLE001 -- keep and report every failed inference image
            record.update(status="error", message=str(exc))
            print(f"{name}: inference failed: {exc}", file=sys.stderr)
    write_json(cfg["review_dir"] / "manifest.json", records)


def staged(cfg, name, record):
    for stage in ("corrected", "proposals"):
        value = read_json(cfg["review_dir"] / stage / f"{Path(name).stem}.json")
        if value and all(value.get(k) == record[k] for k in ("image_hash", "class_hash")):
            for obj in value["objects"]:
                if obj["geometry"] not in ("detection", "obb", "polygon"):
                    raise ValueError(f"{name}: invalid staged geometry type")
                line = (
                    str(obj["class_id"])
                    + " "
                    + " ".join(map(str, np.asarray(obj["points"]).ravel()))
                )
                parse_labels(line, "polygon", len(cfg["classes"]))
                if obj["geometry"] in ("detection", "obb") and len(obj["points"]) != 4:
                    raise ValueError(f"{name}: boxes must have four corners")
            return value["objects"], stage
    return [], "missing"


def export_review(cfg, destination):
    records = inventory(cfg)
    selected = [(n, r, *staged(cfg, n, r)) for n, r in records.items() if r["status"] != "error"]
    # One format per CVAT task. Never disguise axis-aligned proposals as sign corners.
    fmt = (
        "detection"
        if any(o["geometry"] == "detection" for _, _, objs, _ in selected for o in objs)
        else "obb"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "data.yaml",
            yaml.safe_dump(
                {"path": ".", "train": "train.txt", "names": dict(enumerate(cfg["classes"]))}
            ),
        )
        archive.writestr("train.txt", "".join(f"images/train/{n}\n" for n, _, _, _ in selected))
        for name, r, objects, _ in selected:
            archive.write(cfg["input_dir"] / "images" / name, f"images/train/{name}")
            archive.writestr(
                f"labels/train/{Path(name).stem}.txt",
                label_text(objects, fmt, (r["width"], r["height"])),
            )
    print(
        f"CVAT archive: {destination} (Ultralytics YOLO {fmt}). Empty task labels are unreviewed, not approved background."
    )


def import_review(cfg, source, fmt):
    records = inventory(cfg)
    by_stem = {Path(n).stem: n for n in records}
    pending = {}
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive entries")
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe archive path")
        yamls = [n for n in names if PurePosixPath(n).name == "data.yaml"]
        if len(yamls) != 1:
            raise ValueError("Expected one data.yaml in CVAT archive")
        meta = yaml.safe_load(archive.read(yamls[0]))
        classes = meta["names"]
        if isinstance(classes, list):
            classes = dict(enumerate(classes))
        mapping = {
            int(k): cfg["classes"].index(v) for k, v in classes.items() if v in cfg["classes"]
        }
        if (
            len(mapping) != len(classes)
            or set(mapping) != set(range(len(classes)))
            or len(set(mapping.values())) != len(mapping)
        ):
            raise ValueError("Unknown or noncontiguous CVAT classes")
        for name in names:
            p = PurePosixPath(name)
            if p.suffix != ".txt" or "labels" not in p.parts:
                continue
            if p.stem not in by_stem or p.stem in pending:
                raise ValueError(f"Unknown or duplicate label stem: {p.stem}")
            image_name = by_stem[p.stem]
            image_entries = [
                n
                for n in names
                if "images" in PurePosixPath(n).parts and PurePosixPath(n).name == image_name
            ]
            if len(image_entries) != 1:
                raise ValueError(
                    f"{image_name}: export the CVAT dataset WITH images so its source can be verified."
                )
            import hashlib

            if (
                hashlib.sha256(archive.read(image_entries[0])).hexdigest()
                != records[image_name]["image_hash"]
            ):
                raise ValueError(
                    f"{image_name}: CVAT image differs from current raw image. Review the current image again."
                )
            objects = parse_labels(archive.read(name).decode("utf-8-sig"), fmt, len(classes))
            for obj in objects:
                obj["class_id"] = mapping[obj["class_id"]]
            record = records[image_name]
            pending[p.stem] = {
                "image_hash": record["image_hash"],
                "class_hash": record["class_hash"],
                "objects": objects,
                "source": "cvat_review",
            }
        # All parsing succeeds before staging any changes.
        for stem, value in pending.items():
            write_json(cfg["review_dir"] / "corrected" / f"{stem}.json", value)
    # Imported geometry always requires a new explicit decision, even if unchanged.
    decisions_path = cfg["review_dir"] / "decisions.csv"
    with decisions_path.open(encoding="utf-8", newline="") as f:
        decisions = list(csv.DictReader(f))
    approved_path = cfg["approved_dir"] / "manifest.json"
    approved = read_json(approved_path, {})
    for row in decisions:
        if Path(row["image"]).stem in pending:
            row["decision"], row["reason"] = "pending", ""
            approved.pop(row["image"], None)
    write_json(approved_path, approved)
    with decisions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "decision", "reason"])
        writer.writeheader()
        writer.writerows(decisions)
    print(
        f"Imported {len(pending)} corrected files; all require approval. Missing labels stay pending."
    )


def approve(cfg, decisions_path):
    records = inventory(cfg)
    with decisions_path.open(encoding="utf-8-sig", newline="") as f:
        decisions = list(csv.DictReader(f))
    names = [r["image"] for r in decisions]
    if len(set(names)) != len(names) or set(names) != set(records):
        raise ValueError("Decisions must contain every current image exactly once.")
    approved, labels = {}, {}
    for row in decisions:
        name, decision = row["image"], row["decision"].strip()
        if decision not in ("pending", "approved", "background", "excluded"):
            raise ValueError(f"{name}: invalid decision")
        if decision == "excluded" and not row["reason"].strip():
            raise ValueError(f"{name}: exclusion requires a reason")
        if decision in ("pending", "excluded"):
            continue
        record = records[name]
        objects, stage = staged(cfg, name, record)
        if record["status"] == "error" or stage == "missing":
            raise ValueError(
                f"{name}: readable image and current staged annotations required. Import an explicit empty CVAT label for background."
            )
        if (decision == "background") != (len(objects) == 0):
            raise ValueError(
                f"{name}: use background only for explicitly reviewed empty annotations"
            )
        # Validate output eligibility and geometry before any approval is written.
        size = (record["width"], record["height"])
        transform_objects(objects, np.eye(3), size, size, cfg["output_format"], cfg["geometry"])
        text = label_text(objects, cfg["output_format"], size)
        approved[name] = {
            **record,
            "decision": decision,
            "objects": objects,
            "stage": stage,
            "annotation_hash": digest(objects),
            "format": cfg["output_format"],
            "label_hash": digest(text),
        }
        labels[Path(name).stem] = text
    for stem, text in labels.items():
        path = cfg["approved_dir"] / "labels" / f"{stem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    write_json(cfg["approved_dir"] / "manifest.json", approved)
    print(f"Approved {len(approved)} images. Others remain excluded from augmentation.")


def current_approved(cfg):
    records = inventory(cfg)
    approved = read_json(cfg["approved_dir"] / "manifest.json", {})
    valid = []
    for name, record in records.items():
        approval = approved.get(name)
        objects, stage = staged(cfg, name, record)
        label = cfg["approved_dir"] / "labels" / f"{Path(name).stem}.txt"
        if (
            not approval
            or record["status"] == "error"
            or stage == "missing"
            or any(approval.get(k) != record[k] for k in ("image_hash", "class_hash"))
            or approval["annotation_hash"] != digest(objects)
            or not label.exists()
            or approval["label_hash"] != digest(label.read_text(encoding="utf-8"))
            or approval["format"] != cfg["output_format"]
        ):
            print(f"{name}: pending, stale, or excluded; not augmented")
            continue
        valid.append((cfg["input_dir"] / "images" / name, objects))
    return valid


def augment(cfg, count=None):
    inputs = current_approved(cfg)
    if not inputs:
        raise ValueError(
            "No current approved annotations. Complete CVAT review and --approve first."
        )
    count = cfg["images_per_input"] if count is None else count
    if count < 1:
        raise ValueError("--count must be positive")
    out = cfg["output_dir"]
    if (
        (out / "manifest.json").exists()
        or images_in(out / "images")
        or (out / "labels").exists()
        and any((out / "labels").iterdir())
    ):
        raise ValueError(
            "Output already contains a run. Select a new output_dir; files are not overwritten."
        )
    rng = np.random.default_rng(cfg["seed"])
    side = cfg["image_size"]
    fmt = cfg["output_format"]
    generated = []
    rejected = 0
    fallbacks = 0
    for path, objects in inputs:
        image = read_image(path)
        h, w = image.shape[:2]
        sensitive = any(
            cfg["classes"][o["class_id"]] in cfg["orientation_sensitive_classes"] for o in objects
        )
        # Preflight unchanged geometry so invalid labels cannot hide behind retries.
        _, base = geometry_matrix(w, h, cfg, np.random.default_rng(0), sensitive)
        baseline = transform_objects(objects, base, (w, h), (side, side), fmt, cfg["geometry"])
        for index in range(count):
            for attempt in range(cfg["geometry"]["attempts"]):
                matrix, _ = geometry_matrix(w, h, cfg, rng, sensitive)
                try:
                    changed = transform_objects(
                        objects, matrix, (w, h), (side, side), fmt, cfg["geometry"]
                    )
                    break
                except ValueError:
                    rejected += 1
            else:
                matrix, changed = base, baseline
                fallbacks += 1
            warped = cv2.warpPerspective(
                image,
                matrix,
                (side, side),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(40, 45, 40),
            )
            result = apply_effects(warped, cfg["augmentation"], rng)
            stem = f"{path.stem}__aug_{index:04d}"
            text = label_text(changed, fmt, (side, side))
            parse_labels(text, fmt, len(cfg["classes"]))
            label_path = out / "labels" / f"{stem}.txt"
            image_path = out / "images" / f"{stem}.png"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                write_image(image_path, result)
                label_path.write_text(text, encoding="utf-8")
            except Exception:
                image_path.unlink(missing_ok=True)
                label_path.unlink(missing_ok=True)
                raise
            generated.append(
                {"image": image_path.name, "source": path.name, "objects": len(changed)}
            )
    write_json(
        out / "manifest.json",
        {"format": fmt, "classes": cfg["classes"], "seed": cfg["seed"], "images": generated},
    )
    print(
        f"Generated {len(generated)} pairs; {rejected} rejected transforms; {fallbacks} geometry fallbacks. CPU only."
    )


def preview_review(cfg):
    records = inventory(cfg)
    items = []
    for name, record in records.items():
        if record["status"] == "error":
            print(f"{name}: {record.get('message', 'error')}")
            continue
        objects, stage = staged(cfg, name, record)
        items.append(
            (cfg["input_dir"] / "images" / name, objects, f"{name}: {stage}; REVIEW REQUIRED")
        )
    print(preview(items, cfg["review_dir"] / "previews", cfg["classes"]))


def output_items(cfg):
    out = cfg["output_dir"]
    manifest = read_json(out / "manifest.json")
    if not manifest:
        raise ValueError("No completed output run")
    paths = images_in(out / "images")
    if {p.name for p in paths} != {r["image"] for r in manifest["images"]}:
        raise ValueError("Output images do not match run manifest")
    if {p.stem for p in (out / "labels").glob("*.txt")} != {p.stem for p in paths}:
        raise ValueError("Output image/label pairing mismatch")
    expected = {r["image"]: r["objects"] for r in manifest["images"]}
    items = []
    for path in paths:
        h, w = read_image(path).shape[:2]
        objects = parse_labels(
            (out / "labels" / f"{path.stem}.txt").read_text(),
            manifest["format"],
            len(manifest["classes"]),
        )
        if len(objects) != expected[path.name]:
            raise ValueError("Output object count mismatch")
        if manifest["format"] == "obb":
            for obj in objects:
                p = np.asarray(obj["points"]) * [w, h]
                edges = np.roll(p, -1, axis=0) - p
                cosine = np.sum(edges * np.roll(edges, -1, axis=0), axis=1) / (
                    np.linalg.norm(edges, axis=1)
                    * np.linalg.norm(np.roll(edges, -1, axis=0), axis=1)
                )
                if np.max(np.abs(cosine)) > 1e-4:
                    raise ValueError(f"{path.name}: exported OBB is not rectangular")
        items.append((path, objects, path.name))
    return items, manifest


def demo(cfg):
    root = Path(__file__).resolve().parent / "fixtures"
    if root.exists():
        raise ValueError(
            "fixtures already exists. Keep it for inspection; use a fresh checkout to recreate the demo."
        )
    root.mkdir()
    (root / "TEST_FIXTURES.txt").write_text(
        "SYNTHETIC TEST FIXTURES ONLY. Never mix these with real training data. Labels come from known drawing geometry, not a detector.\n"
    )
    cfg = copy.deepcopy(cfg)
    for key, folder in [
        ("input_dir", "input"),
        ("review_dir", "review"),
        ("approved_dir", "approved"),
        ("output_dir", "output"),
    ]:
        cfg[key] = root / folder
    cfg["classes"] = ["gate_sos", "gate_hammer"]
    cfg["detector"]["weights"] = None
    for i in range(3):
        image = np.full((480, 640, 3), (70, 65, 35), np.uint8)
        objects = []
        for cls, center, angle in [(0, (180, 200), i * 12 - 12), (1, (460, 280), -i * 9)]:
            points = cv2.boxPoints((center, (135, 85), angle))
            cv2.fillConvexPoly(image, points.round().astype(np.int32), (180, 220, 235))
            cv2.putText(
                image,
                "SOS" if cls == 0 else "HAMMER",
                (center[0] - 55, center[1] + 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (25, 30, 20),
                2,
            )
            from utils import ordered

            objects.append(
                {
                    "class_id": cls,
                    "points": (ordered(points) / [640, 480]).tolist(),
                    "geometry": "polygon",
                }
            )
        name = f"TEST_ONLY_signs_{i}.png"
        write_image(cfg["input_dir"] / "images" / name, image)
        write_json(
            cfg["review_dir"] / "corrected" / f"TEST_ONLY_signs_{i}.json",
            {
                "image_hash": digest(cfg["input_dir"] / "images" / name),
                "class_hash": digest(cfg["classes"]),
                "objects": objects,
            },
        )
    records = inventory(cfg)
    decisions = cfg["review_dir"] / "decisions.csv"
    with decisions.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "decision", "reason"])
        writer.writeheader()
        writer.writerows(
            {"image": n, "decision": "approved", "reason": "known synthetic drawing geometry"}
            for n in records
        )
    approve(cfg, decisions)
    preview_review(cfg)
    export_review(cfg, root / "cvat_test.zip")
    augment(cfg)
    items, meta = output_items(cfg)
    print(preview(items, cfg["output_dir"] / "previews", meta["classes"], meta["format"]))
    # A standalone fixture config makes every normal CLI command usable on the demo.
    saved = copy.deepcopy(cfg)
    for key in ("input_dir", "review_dir", "approved_dir", "output_dir"):
        saved[key] = str(saved[key].relative_to(root))
    (root / "config.yaml").write_text(yaml.safe_dump(saved, sort_keys=False), encoding="utf-8")
    print("Fixture validated. Use --config fixtures/config.yaml for preview/validation.")


def self_test(cfg):
    """Repeatable numerical and review-integrity tests, isolated in a temporary folder."""
    from utils import clip_polygon

    obj = {
        "class_id": 0,
        "points": [[0.2, 0.3], [0.5, 0.3], [0.5, 0.5], [0.2, 0.5]],
        "geometry": "polygon",
    }
    policy = cfg["geometry"]
    points = np.asarray(obj["points"]) * 100
    matrix = np.array([[0, -1, 100], [1, 0, 0], [0, 0, 1.0]])
    expected = np.c_[100 - points[:, 1], points[:, 0]]
    assert np.allclose(transformed(points, matrix), expected)
    result = transform_objects([obj], matrix, (100, 100), (100, 100), "obb", policy)
    assert np.allclose(result[0]["points"], expected / 100)
    box = fit_box(expected, "obb")
    assert abs(cv2.contourArea(box.astype(np.float32)) - 600) < 0.01
    for fmt in ("obb", "detection"):
        text = label_text(result, fmt, (100, 100))
        assert len(parse_labels(text, fmt, 2)) == 1

    def rejects(function):
        try:
            function()
        except ValueError:
            return
        raise AssertionError("Expected rejection")

    rejects(lambda: parse_labels("0 nan .5 .2 .2", "detection", 2))
    rejects(lambda: parse_labels("2 .5 .5 .2 .2", "detection", 2))
    rejects(
        lambda: transform_objects(
            [{**obj, "geometry": "detection"}], np.eye(3), (100, 100), (100, 100), "obb", policy
        )
    )
    shift = np.array([[1, 0, 70], [0, 1, 0], [0, 0, 1.0]])
    rejects(lambda: transform_objects([obj], shift, (100, 100), (100, 100), "obb", policy))
    assert (
        abs(
            cv2.contourArea(
                clip_polygon(np.array([[-2, 0], [10, 0], [10, 10], [-2, 10]]), 100, 100).astype(
                    np.float32
                )
            )
            - 100
        )
        < 0.01
    )
    settings = copy.deepcopy(cfg["augmentation"])
    for s in settings.values():
        s["enabled"] = False
    sample = np.full((20, 20, 3), 100, np.uint8)
    assert np.array_equal(sample, apply_effects(sample, settings, np.random.default_rng(3)))
    assert np.array_equal(
        apply_effects(sample, cfg["augmentation"], np.random.default_rng(3)),
        apply_effects(sample, cfg["augmentation"], np.random.default_rng(3)),
    )
    # Numerical scale/translation, slight clipping, and perspective expectations.
    affine = np.array([[1.1, 0, 4], [0, 0.9, 5], [0, 0, 1.0]])
    assert np.allclose(transformed(points, affine), points * [1.1, 0.9] + [4, 5])
    perspective = np.array([[1, 0, 0], [0, 1, 0], [0.001, 0, 1.0]])
    assert np.allclose(transformed(points, perspective), points / (1 + 0.001 * points[:, 0:1]))
    crop_policy = {**policy, "allow_crop": True, "min_visibility": 0.9}
    mild_shift = np.array([[1, 0, 51], [0, 1, 0], [0, 0, 1.0]])
    cropped = transform_objects([obj], mild_shift, (100, 100), (100, 100), "detection", crop_policy)
    assert np.isclose(np.asarray(cropped[0]["points"])[:, 0].max(), 1)
    rejects(
        lambda: transform_objects([obj], shift, (100, 100), (100, 100), "detection", crop_policy)
    )
    sensitive_cfg = copy.deepcopy(cfg)
    for name, s in sensitive_cfg["augmentation"].items():
        s["enabled"] = name in ("rotation", "aspect_ratio", "perspective")
        s["probability"] = 1
    matrix, base = geometry_matrix(100, 100, sensitive_cfg, np.random.default_rng(4), True)
    assert np.allclose(matrix, base)
    # Probability zero works even when enabled. Exercise optional effects on CPU.
    for s in settings.values():
        s["enabled"] = True
        s["probability"] = 0
    assert np.array_equal(sample, apply_effects(sample, settings, np.random.default_rng(4)))
    for name in ("jpeg", "motion_blur"):
        settings[name]["probability"] = 1
    assert apply_effects(sample, settings, np.random.default_rng(4)).shape == sample.shape
    # Test the inference adapter without installing torch or fabricating real labels.
    from types import SimpleNamespace

    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def __getitem__(self, index):
            return FakeTensor(self.value[index])

        def item(self):
            return self.value.item()

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakePredictions:
        def __len__(self):
            return 2

    predictions = FakePredictions()
    predictions.cls = FakeTensor([0, 1])
    predictions.conf = FakeTensor([0.9, 0.4])
    predictions.xywhn = FakeTensor([[0.4, 0.4, 0.2, 0.2], [0.7, 0.6, 0.1, 0.2]])
    predictions.xyxyxyxyn = FakeTensor([obj["points"], obj["points"]])
    adapter_cfg = copy.deepcopy(cfg)
    adapter_cfg["classes"] = ["gate_sos", "gate_hammer"]
    for task in ("detect", "obb"):
        adapted = proposal_objects(
            SimpleNamespace(boxes=predictions, obb=predictions), task, {0: 1, 1: 0}, adapter_cfg
        )
        assert [o["class_id"] for o in adapted] == [1, 0]
        assert adapted[1]["low_confidence"] and adapted[0]["confidence"] == 0.9
    assert proposal_objects(SimpleNamespace(boxes=None), "detect", {}, adapter_cfg) == []
    with tempfile.TemporaryDirectory() as directory:
        local = copy.deepcopy(cfg)
        root = Path(directory)
        for key in ("input_dir", "review_dir", "approved_dir", "output_dir"):
            local[key] = root / key
        local["detector"]["weights"] = None
        local["classes"] = ["gate_sos", "gate_hammer"]
        local["image_size"] = 100
        path = local["input_dir"] / "images" / "test.png"
        write_image(path, np.full((100, 100, 3), 100, np.uint8))
        propose(local)
        assert not current_approved(local)
        record = inventory(local)["test.png"]
        staged_path = local["review_dir"] / "corrected" / "test.json"
        write_json(
            staged_path,
            {
                "image_hash": record["image_hash"],
                "class_hash": record["class_hash"],
                "objects": [obj],
            },
        )
        decisions = local["review_dir"] / "decisions.csv"
        decisions.write_text("image,decision,reason\ntest.png,approved,test\n")
        approve(local, decisions)
        assert len(current_approved(local)) == 1
        export_review(local, root / "review.zip")
        import_review(local, root / "review.zip", "obb")
        assert not current_approved(local)  # imports revoke approval
        decisions.write_text("image,decision,reason\ntest.png,approved,test\n")
        approve(local, decisions)
        augment(local, 2)
        assert len(output_items(local)[0]) == 2
        first_hashes = [digest(p) for p in sorted((local["output_dir"] / "images").glob("*.png"))]
        local["output_dir"] = root / "second_output"
        augment(local, 2)
        assert first_hashes == [
            digest(p) for p in sorted((local["output_dir"] / "images").glob("*.png"))
        ]
        # Blank annotations need a specific background decision.
        corrected = read_json(staged_path)
        write_json(staged_path, {**corrected, "objects": []})
        rejects(lambda: approve(local, decisions))
        decisions.write_text("image,decision,reason\ntest.png,background,checked empty\n")
        approve(local, decisions)
        assert len(current_approved(local)) == 1
        write_json(staged_path, corrected)
        decisions.write_text("image,decision,reason\ntest.png,approved,test\n")
        approve(local, decisions)
        original_image = path.read_bytes()
        write_image(path, np.full((100, 100, 3), 101, np.uint8))
        assert not current_approved(local)
        path.write_bytes(original_image)
        changed = read_json(staged_path)
        changed["objects"][0]["class_id"] = 1
        write_json(staged_path, changed)
        assert not current_approved(local)
        rejects(lambda: augment(local, 1))
        # A changed raw image cannot receive annotations from an older export.
        write_image(path, np.full((100, 100, 3), 102, np.uint8))
        rejects(lambda: import_review(local, root / "review.zip", "obb"))
    print(
        "Self-test passed: geometry, cropping, perspective, orientation protection, detector adapter, optional effects, reproducibility, CVAT exchange, background approval, stale annotations and CPU output pairing."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--count", type=int, help="Variants per approved input image")
    actions = parser.add_mutually_exclusive_group()
    for flag in ("propose", "preview", "preview-review", "validate", "demo", "self-test"):
        actions.add_argument("--" + flag, action="store_true")
    for flag in ("export-review", "import-review", "approve"):
        actions.add_argument("--" + flag, type=Path)
    parser.add_argument("--review-format", choices=["obb", "detection", "polygon"])
    args = parser.parse_args()
    try:
        cfg = load_config(args.config.resolve())
        if args.propose:
            propose(cfg)
        elif args.export_review:
            export_review(cfg, args.export_review)
        elif args.import_review:
            if not args.review_format:
                raise ValueError("--import-review requires --review-format")
            import_review(cfg, args.import_review, args.review_format)
        elif args.approve:
            approve(cfg, args.approve)
        elif args.preview_review:
            preview_review(cfg)
        elif args.preview or args.validate:
            items, meta = output_items(cfg)
            if args.preview:
                print(
                    preview(items, cfg["output_dir"] / "previews", meta["classes"], meta["format"])
                )
            print(f"Validated {len(items)} output image/label pairs.")
        elif args.demo:
            demo(cfg)
        elif args.self_test:
            self_test(cfg)
        else:
            augment(cfg, args.count)
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile, yaml.YAMLError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
