#!/usr/bin/env python3
"""
Batch inference for MegaPose with per-frame visualizations (off-screen).
- Uses one fixed bbox for all frames.
- Rescales camera intrinsics per frame if resolution differs.
- Forces Panda3D off-screen rendering to avoid X/GL crashes.
- Keeps mesh database on CPU to avoid GPU OOM.
"""

import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import logging
import torch, gc

# ---- Off-screen rendering before any MegaPose imports ----
os.environ.setdefault("PANDA_OFFSCREEN", "1")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# ---- MegaPose Imports ----
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import ObservationTensor
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import load_named_model, NAMED_MODELS
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay
from bokeh.io import export_png
from bokeh.plotting import gridplot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("megapose-batch")

# ============== USER SETTINGS ==============
EXAMPLE = "bot"                      # folder under $MEGAPOSE_DATA_DIR/examples
IMAGES_SUB = "images"                # subfolder with frames
MODEL = "megapose-1.0-RGB-multi-hypothesis"           # lighter single-hypothesis model
FIXED_BBOX = [0, 0, 960, 443]      # [xmin, ymin, xmax, ymax] for ALL frames
SKIP_VISUALS = False                 # set True to skip overlay PNGs
LABEL = "bot"                        # must match meshes/<LABEL>
MAX_WIDTH = None                   
# ===========================================

def scale_K(K, old_hw, new_hw):
    """Rescale intrinsics when resolution changes."""
    oh, ow = old_hw
    nh, nw = new_hw
    sx, sy = nw / ow, nh / oh
    K = np.array(K, dtype=float)
    return np.array([
        [K[0][0] * sx, 0.0, K[0][2] * sx],
        [0.0, K[1][1] * sy, K[1][2] * sy],
        [0.0, 0.0, 1.0]
    ])

def maybe_downscale(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale image if wider than MAX_WIDTH. Returns (rgb_new, scale)."""
    if not MAX_WIDTH:
        return rgb, 1.0
    h, w = rgb.shape[:2]
    if w <= MAX_WIDTH:
        return rgb, 1.0
    scale = MAX_WIDTH / float(w)
    new_w = MAX_WIDTH
    new_h = max(1, int(round(h * scale)))
    rgb_small = np.array(Image.fromarray(rgb).resize((new_w, new_h), resample=Image.BILINEAR))
    return rgb_small, scale

def main():
    if "MEGAPOSE_DATA_DIR" not in os.environ:
        raise RuntimeError("Please set MEGAPOSE_DATA_DIR")

    data_dir = Path(os.environ["MEGAPOSE_DATA_DIR"])
    ex_dir   = (data_dir / "examples" / EXAMPLE).resolve()
    frames   = ex_dir / IMAGES_SUB
    mesh_dir = ex_dir / "meshes"
    cam_path = ex_dir / "camera_data.json"

    assert ex_dir.is_dir(),  f"Missing example dir: {ex_dir}"
    assert frames.is_dir(),  f"Missing images dir:  {frames}"
    assert mesh_dir.is_dir(),f"Missing meshes dir:  {mesh_dir}"
    assert cam_path.exists(),f"Missing camera_data.json at {cam_path}"

    base_cam = json.loads(cam_path.read_text())
    K_base   = np.array(base_cam["K"], dtype=float)
    base_res = base_cam["resolution"]  # [H, W]

    out_root = ex_dir / "batch_outputs_offscreen"
    out_root.mkdir(exist_ok=True)

    images = sorted([p for p in frames.iterdir()
                     if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}])
    if not images:
        raise RuntimeError(f"No images found in {frames}")

    # ---- Initialize model and renderer once ----
    logger.info(f"Loading mesh from: {mesh_dir}")
    obj_path = mesh_dir / LABEL / "model.obj"
    if not obj_path.exists():
        # allow PLY fallback
        obj_path = mesh_dir / LABEL / "model.ply"
        assert obj_path.exists(), f"Mesh not found: {mesh_dir / LABEL / 'model.obj'} or .ply"

    object_dataset = RigidObjectDataset([
        RigidObject(label=LABEL, mesh_path=obj_path, mesh_units="mm")
    ])

    logger.info(f"Loading model: {MODEL}")
    model_info = NAMED_MODELS[MODEL]
    # Keep mesh DB on CPU to avoid GPU OOM
    pose_estimator = load_named_model(
        MODEL,
        object_dataset
    ).cuda()

    renderer = None
    if not SKIP_VISUALS:
        renderer = Panda3dSceneRenderer(object_dataset)

    logger.info(f"Running inference on {len(images)} images with fixed bbox {FIXED_BBOX}\n")

    for i, img_path in enumerate(images, 1):
        stem = img_path.stem
        logger.info(f"[{i}/{len(images)}] {img_path.name}")

        # Load & maybe downscale RGB
        rgb_full = np.array(Image.open(img_path).convert("RGB"))
        rgb, _ = maybe_downscale(rgb_full)
        H, W = rgb.shape[:2]

        # Per-frame intrinsics
        K_frame = scale_K(K_base, base_res, [H, W]) if [H, W] != base_res else K_base

        # Build observation & detections
        observation = ObservationTensor.from_numpy(rgb, None, K_frame).cuda()
        object_data = [ObjectData(label=LABEL, bbox_modal=FIXED_BBOX)]
        detections  = make_detections_from_object_data(object_data).cuda()

        # Run inference
        output, _ = pose_estimator.run_inference_pipeline(
            observation, detections=detections, **model_info["inference_parameters"]
        )

        # Save predictions
        poses = output.poses.cpu().numpy()
        pred_data = [ObjectData(label=LABEL, TWO=Transform(p)) for p in poses]
        out_dir = out_root / stem / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "object_data.json").write_text(json.dumps([d.to_json() for d in pred_data]))
        logger.info(f"Saved predictions to {out_dir}")

        # Visualization (safe)
        if not SKIP_VISUALS and renderer:
            try:
                vis_dir = out_root / stem / "visualizations"
                vis_dir.mkdir(parents=True, exist_ok=True)

                cam_data = CameraData(K=K_frame, resolution=(H, W), TWC=Transform(np.eye(4)))
                cam_data, pred_pd = convert_scene_observation_to_panda3d(cam_data, pred_data)

                light_datas = [Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1))]
                renderings = renderer.render_scene(
                    pred_pd, [cam_data], light_datas,
                    render_depth=False, render_binary_mask=False, render_normals=False, copy_arrays=True
                )[0]

                plotter = BokehPlotter()
                fig_rgb = plotter.plot_image(rgb)
                fig_overlay = plotter.plot_overlay(rgb, renderings.rgb)
                contour = make_contour_overlay(rgb, renderings.rgb,
                                               dilate_iterations=1, color=(0, 255, 0))["img"]
                fig_contour = plotter.plot_image(contour)

                export_png(fig_overlay,  filename=vis_dir / "mesh_overlay.png")
                export_png(fig_contour,  filename=vis_dir / "contour_overlay.png")
                export_png(
                    gridplot([[fig_rgb, fig_contour, fig_overlay]], toolbar_location=None),
                    filename=vis_dir / "all_results.png"
                )
                logger.info(f"Saved visualizations to {vis_dir}")
            except Exception as e:
                logger.warning(f"Visualization failed on {img_path.name}: {e}")

        # Free GPU memory between frames
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n✓ Done. Results in: {out_root}")
    print("Each frame folder contains outputs/object_data.json and visualizations/*")

if __name__ == "__main__":
    main()
