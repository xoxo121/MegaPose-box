# Dynamic Bounding Box Pipeline for MegaPose

This repository adds a **dynamic, per-frame bounding box** pipeline for running MegaPose on videos.
Instead of a single fixed box, it uses a **lightweight OpenCV tracker (KCF)** with **periodic YOLO (on a small crop)** to keep the box locked to the moving object—even under moderate scale changes or brief occlusions—then feeds those boxes into MegaPose.

---

## What’s included

* **`run_pipeline_dynamic.py`** — end-to-end runner (extract → track → infer → overlay → video).
* **`moving_bbox_lite.py`** — manual ROI on frame 1 → KCF tracker every frame + **YOLO on a small crop** periodically → writes `dets/bboxes.json`.
* **`batch_infer_dynamic_bbox.py`** — MegaPose batch inference that **reads** `dets/bboxes.json`.

Existing scripts reused as-is: `overlay_mesh_bbox_from_obj.py`, `make_video.py`.

---

## Overview & data flow

1. **Frame Extraction**
   Split the input video into frames:

   ```
   $MEGAPOSE_DATA_DIR/examples/<name>/images/IMG-000001.jpg …
   ```
2. **Dynamic Bounding Boxes** (`moving_bbox_lite.py`)

   * You draw the ROI on the first frame.
   * KCF tracks it each frame; YOLO (e.g., `yolov8n.pt`) runs on a **padded crop** every *N* frames or on tracker failure.
   * Outputs:

     ```
     $MEGAPOSE_DATA_DIR/examples/<name>/dets/bboxes.json
     ```

     where each entry is `[x1, y1, x2, y2]` in pixels.
3. **MegaPose Inference** (`batch_infer_dynamic_bbox.py`)

   * Loads per-frame boxes from `bboxes.json`.
   * Writes:

     ```
     $MEGAPOSE_DATA_DIR/examples/<name>/batch_outputs_offscreen/<frame>/outputs/object_data.json
     ```
4. **Overlay & Video**

   * `overlay_mesh_bbox_from_obj.py` renders visualizations.
   * `make_video.py` builds:

     ```
     $MEGAPOSE_DATA_DIR/examples/<name>/overlay_bbox.mp4
     ```

**Mesh/label convention**
`<name>` is used as the MegaPose **label**. Place the mesh at:

```
$MEGAPOSE_DATA_DIR/examples/<name>/meshes/<name>/model.obj|ply
```

---

## Quick start

```bash
export MEGAPOSE_DATA_DIR=/path/to/megapose_data

python3 run_pipeline_dynamic.py \
  --video /path/to/video.mp4 \
  --megapose_root /path/to/megapose6d \
  --megapose_data_dir $MEGAPOSE_DATA_DIR \
  --example_name <name> \
  --num_frames 900 --mesh_units mm --fps 12 \
  --yolo_model yolov8n.pt --imgsz 320 --conf 0.25 \
  --detect_every 10 --device 0 --show
```

### Minimal, practical knobs

* `--imgsz` (YOLO crop size): **256–416** recommended (default 320).
* `--detect_every`: run YOLO more/less often (lower = more robust; higher = faster).
* `--device` (bbox stage): `0` for GPU or `cpu` (safe & VRAM-free).

---

## Requirements

* Python **3.9+**
  *If on 3.9, use `Optional[int]` in type hints (not `int | None`).*
* Packages:

  ```bash
  pip install ultralytics opencv-python
  ```
* **ffmpeg** recommended for faster extraction (OpenCV fallback is built in).
* A working MegaPose setup (CUDA recommended for inference).

---

## Outputs (where to find results)

* Frames: `$MEGAPOSE_DATA_DIR/examples/<name>/images/`
* BBoxes: `$MEGAPOSE_DATA_DIR/examples/<name>/dets/bboxes.json`
* Poses: `$MEGAPOSE_DATA_DIR/examples/<name>/batch_outputs_offscreen/<frame>/outputs/object_data.json`
* Video: `$MEGAPOSE_DATA_DIR/examples/<name>/overlay_bbox.mp4`

---

## Common pitfalls

* **Wrong video/first frame shows up**
  Old frames were reused. Clean before re-running or use a new `--example_name`:

  ```bash
  EX=<name>; ROOT=$MEGAPOSE_DATA_DIR/examples/$EX
  rm -rf "$ROOT/images/"* "$ROOT/dets/"* "$ROOT/batch_outputs_offscreen/"* "$ROOT/overlay_bbox.mp4"
  ```
* **GPU OOM in bbox stage**
  Lower `--imgsz`, increase `--detect_every`, or use `--device cpu`. (KCF is CPU-fast; YOLO runs on a small crop.)
* **Occlusions / drift**
  Decrease `--detect_every` (e.g., 5) for more frequent crop-YOLO; optionally switch KCF → CSRT in `moving_bbox_lite.py` for more stability.
* **Non-COCO objects**
  Default `yolov8n.pt` may not include your class name (e.g., “robot”). The lite script doesn’t filter by class; it uses your ROI and IoU. For best results on novel objects, use a custom YOLO model.

---