# WAUV sign labeling and augmentation

A small CPU-only pipeline for **raw underwater images → proposed/manual annotations → explicit review → augmented YOLO data**. Three Python files; no training, external dataset downloads, automatic segmentation, or synthetic-scene engine. This implementation is independent of the reference training repository.

## Install and try it

Python 3.11+ (tested locally on Python 3.13):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python augment.py --self-test
python augment.py --demo
python augment.py --config fixtures/config.yaml --preview
python augment.py --config fixtures/config.yaml --validate
```

The demo creates three **TEST_ONLY** sign images with exact known drawing outlines and 30 augmented images under `fixtures/`. Its approvals are only for known synthetic test geometry, not model predictions. See `fixtures/output/previews/contact_sheet.jpg`. Fixtures are excluded from Git and never read by the real configuration. The demo refuses to overwrite an existing fixture directory.

## Label your raw images

1. Put JPG/PNG files in `input/images/`. Use unique stems; no labels are required. Keep raw images unchanged during review.
2. Set the ordered `classes` list in `config.yaml`; initially `gate_sos` and `gate_hammer`. Draw the physical sign panel, not the whole gate. Preserve class order once labeling begins.
3. Prepare proposals and a CVAT task:

```bash
python augment.py --propose
python augment.py --preview-review
python augment.py --export-review review/cvat.zip
```

No suitable model is bundled. With no weights configured, the pipeline explains the manual bootstrap workflow and creates **no predictions**. Empty task labels mean **unreviewed**, not background. Images with no detections remain in the task; errors are reported in `review/manifest.json`.

4. In [CVAT](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/), create a project/task with the configured classes and upload the images from the ZIP. Import its annotations using the Ultralytics YOLO format printed by the export command. Draw/correct rotated boxes around every sign; fix classes, add missed signs, and remove false positives. A detection-model proposal is only an upright box: manually correct the sign corners before using OBB output. No CVAT account connection or upload is performed by this script.
5. Export the corrected dataset **with images**, in Ultralytics YOLO OBB format, then:

```bash
python augment.py --import-review corrected.zip --review-format obb
python augment.py --preview-review
```

Detection and convex polygon exports are also accepted with `--review-format detection` or `polygon` (CVAT's segmentation format). Polygon vertices must follow the perimeter; masks and nonconvex outlines are outside this lightweight version. Images inside the export must match the raw image bytes; mismatches are rejected so old annotations cannot be attached to changed images.

6. Inspect `review/previews/`, including the individual full-size images. Edit `review/decisions.csv`: choose `approved` only after checking **all** objects, `background` for a checked image with an explicit empty annotation file, `excluded` with a reason, or leave `pending`. Missing annotation files never imply background.

```bash
python augment.py --approve review/decisions.csv
```

Approval is explicit even for high-confidence proposals. Importing corrections revokes approval for those images. Image, geometry, class-list, and approved-label changes invalidate approval. A pending/excluded decision takes effect when you rerun `--approve`; do this after every CSV change. Approved labels live in `approved/labels`; augmentation also checks the approval manifest, rather than trusting files copied into that folder.

## Optional custom model

The base installation does not include a detector or PyTorch. To use local **custom Ultralytics detect/OBB `.pt` weights**, install `ultralytics` in a separate inference environment with this project's dependencies. Ultralytics may also install graphical OpenCV; keep only one OpenCV package in that environment to avoid overlapping `cv2` installations.

Set `detector.weights` to the local file. Names must match configured WAUV classes, or set `detector.class_map`, e.g. `{sos_panel: gate_sos}`. Unknown model classes are rejected. Generic pretrained weights are not assumed to recognize WAUV signs, and no weights are automatically downloaded. Only load model files you trust.

`--propose` runs on CPU, stores per-object classes/confidences and model provenance in `review/proposals`, and flags scores below `low_confidence`. The default proposal floor is 0.25; scores below 0.5 are highlighted. Predictions are estimates, not pixel-perfect boundaries. Real recognition accuracy cannot be tested until suitable WAUV weights are supplied. No model training is included.

## Augment approved images

```bash
python augment.py --config config.yaml
python augment.py --config config.yaml --count 100
python augment.py --preview
python augment.py --validate
```

`--count` is variants **per approved image**. Output PNGs and matching TXT labels go to `output/images` and `output/labels`; previews go to `output/previews`. Choose a new `output_dir` for another run: existing runs are never overwritten. Pending, stale, excluded, and erroneous inputs are reported and skipped. If none are eligible the command fails clearly.

All paths in the config are relative to the config file. All effects have independent enable/probability settings. Edit the documented YAML ranges for rotation (±20°), translation (±10%), scale (0.6–1.4), aspect (0.9–1.1), lighting, shadows, blue/green tint, haze, blur, and noise. Optional motion blur, JPEG artifacts, and perspective start disabled. `seed` makes results repeatable within the same software environment.

`output_format: obb` writes `class_id x1 y1 x2 y2 x3 y3 x4 y4`. `detection` writes `class_id center_x center_y width height`. Coordinates are normalized. Changing output format requires reapproval. OBB output requires reviewed corners/outlines; ordinary boxes cannot reveal true sign orientation. An OBB's corner ordering describes its rectangle, not an arrow's semantic direction.

Geometry transforms the whole image and its original vertices together, then fits the smallest enclosing rectangle; it does **not** independently reposition objects. Images are letterboxed to 640 without stretching. No flips. Configure `orientation_sensitive_classes` to suppress rotation, aspect changes, and perspective on images containing those classes. Annotation geometry is never changed by photometric effects.

Cropping is disabled by default. Unsafe geometry is retried, then falls back to unchanged geometry plus photometric effects. Optional cropping needs reviewed OBB/polygon geometry and 90% retained area. The minimum box side is two pixels. OBB area may exceed polygon area by at most 15%; both settings are configurable. Perspective can make a rectangle trapezoidal, so some enclosing-box background is unavoidable. There is no blind contour detection, arbitrary padding, or promise of pixel-perfect boundaries.

## Files and verification

- `augment.py`: CLI, optional inference, review/approval, augmentation and self-checks.
- `underwater_effects.py`: image-only effects.
- `utils.py`: files, geometry, labels and preview drawing.

`--self-test` checks known-corner geometry, invalid labels, review approval/staleness, CVAT exchange, reproducibility, and CPU output pairing in temporary directories. `--validate` checks every generated image/label pair, coordinate/class validity, object counts, and OBB rectangularity. Visual review remains essential for judging real sign boundaries and detector misses. Add real images only under `input/images`; never copy fixture images into that folder.
