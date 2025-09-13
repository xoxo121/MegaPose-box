#!/usr/bin/env python3

import os, json, logging, gc
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import torch

os.environ.setdefault("PANDA_OFFSCREEN","1")
os.environ.setdefault("PYOPENGL_PLATFORM","egl")

from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import ObservationTensor
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import load_named_model, NAMED_MODELS

from bokeh.io import export_png
from bokeh.plotting import gridplot
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger("megapose-dynamic-bbox")

EXAMPLE="bot"
IMAGES_SUB="images"
LABEL="bot"
MODEL="megapose-1.0-RGB-multi-hypothesis"
MAX_WIDTH=None

def scale_K(K, old_hw, new_hw):
    oh,ow=old_hw; nh,nw=new_hw
    sx,sy=nw/ow, nh/oh; K=np.array(K,dtype=float)
    return np.array([[K[0][0]*sx,0.0,K[0][2]*sx],[0.0,K[1][1]*sy,K[1][2]*sy],[0.0,0.0,1.0]])

def maybe_downscale(rgb: np.ndarray):
    if not MAX_WIDTH: return rgb,1.0
    h,w=rgb.shape[:2]
    if w<=MAX_WIDTH: return rgb,1.0
    scale=MAX_WIDTH/float(w); new_w=MAX_WIDTH; new_h=max(1,int(round(h*scale)))
    small=np.array(Image.fromarray(rgb).resize((new_w,new_h),resample=Image.BILINEAR))
    return small,scale

def _load_per_frame_bboxes(ex_dir: Path)->Dict[str,List[float]]:
    det_json=ex_dir/"dets"/"bboxes.json"
    if not det_json.exists():
        raise RuntimeError(f"Missing {det_json}. Run moving_bbox_lite.py first.")
    data=json.loads(det_json.read_text())
    return {str(k):[float(x) for x in v] for k,v in data.items()}

def main():
    if "MEGAPOSE_DATA_DIR" not in os.environ:
        raise RuntimeError("Please set MEGAPOSE_DATA_DIR")
    data_dir=Path(os.environ["MEGAPOSE_DATA_DIR"]).resolve()
    ex_dir=(data_dir/"examples"/EXAMPLE).resolve()
    frames=ex_dir/IMAGES_SUB
    mesh_dir=ex_dir/"meshes"
    cam_path=ex_dir/"camera_data.json"

    assert ex_dir.is_dir()
    assert frames.is_dir()
    assert mesh_dir.is_dir()
    assert cam_path.exists()

    base_cam=json.loads(cam_path.read_text())
    K_base=np.array(base_cam["K"],dtype=float)
    base_res=base_cam["resolution"]  # [H,W]

    per_frame_bbox=_load_per_frame_bboxes(ex_dir)
    last_bbox: Optional[List[float]]=None

    out_root=ex_dir/"batch_outputs_offscreen"; out_root.mkdir(exist_ok=True)
    images=sorted([p for p in frames.iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}])
    if not images: raise RuntimeError(f"No images found in {frames}")

    obj_path=(mesh_dir/LABEL/"model.obj")
    if not obj_path.exists():
        obj_path=(mesh_dir/LABEL/"model.ply"); assert obj_path.exists()
    object_dataset=RigidObjectDataset([RigidObject(label=LABEL, mesh_path=obj_path, mesh_units="mm")])

    logger.info(f"Loading model: {MODEL}")
    model_info=NAMED_MODELS[MODEL]
    pose_estimator=load_named_model(MODEL, object_dataset).cuda()
    renderer=Panda3dSceneRenderer(object_dataset)

    logger.info(f"Running inference on {len(images)} images with dynamic bboxes")
    for i,img_path in enumerate(images,1):
        stem=img_path.stem; logger.info(f"[{i}/{len(images)}] {img_path.name}")
        rgb_full=np.array(Image.open(img_path).convert("RGB"))
        rgb,_=maybe_downscale(rgb_full); H,W=rgb.shape[:2]
        K_frame=scale_K(K_base, base_res, [H,W]) if [H,W]!=base_res else K_base

        bbox=per_frame_bbox.get(stem,None)
        if bbox is None and last_bbox is not None: bbox=last_bbox
        if bbox is None:
            logger.warning(f"No bbox for {stem}, skipping."); continue
        last_bbox=bbox

        observation=ObservationTensor.from_numpy(rgb, None, K_frame).cuda()
        detections=make_detections_from_object_data([ObjectData(label=LABEL, bbox_modal=[float(x) for x in bbox])]).cuda()

        output,_=pose_estimator.run_inference_pipeline(observation, detections=detections, **model_info["inference_parameters"])
        poses=output.poses.cpu().numpy()
        pred_data=[ObjectData(label=LABEL, TWO=Transform(p)) for p in poses]

        out_dir=out_root/stem/"outputs"; out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir/"object_data.json").write_text(json.dumps([d.to_json() for d in pred_data]))
        logger.info(f"Saved predictions to {out_dir}")

        try:
            vis_dir=out_root/stem/"visualizations"; vis_dir.mkdir(parents=True, exist_ok=True)
            cam_data=CameraData(K=K_frame, resolution=(H,W), TWC=Transform(np.eye(4)))
            cam_data,pred_pd=convert_scene_observation_to_panda3d(cam_data, pred_data)
            light_datas=[Panda3dLightData(light_type="ambient", color=(1,1,1,1))]
            renderings=renderer.render_scene(pred_pd,[cam_data],light_datas,render_depth=False,render_binary_mask=False,render_normals=False,copy_arrays=True)[0]
            from megapose.visualization.bokeh_plotter import BokehPlotter
            from megapose.visualization.utils import make_contour_overlay
            plotter=BokehPlotter()
            fig_rgb=plotter.plot_image(rgb)
            fig_overlay=plotter.plot_overlay(rgb, renderings.rgb)
            contour=make_contour_overlay(rgb, renderings.rgb, dilate_iterations=1, color=(0,255,0))["img"]
            fig_contour=plotter.plot_image(contour)
            export_png(fig_overlay,  filename=vis_dir/"mesh_overlay.png")
            export_png(fig_contour,  filename=vis_dir/"contour_overlay.png")
            export_png(gridplot([[fig_rgb, fig_contour, fig_overlay]], toolbar_location=None), filename=vis_dir/"all_results.png")
        except Exception as e:
            logger.warning(f"Visualization skipped on {img_path.name}: {e}")

        torch.cuda.empty_cache(); gc.collect()
    print(f"\n Done. Results in: {out_root}")

if __name__=="__main__": main()
