#!/usr/bin/env python3
"""
Lite tracker for per-frame bboxes:
- Manual ROI on first frame
- OpenCV KCF tracker every frame (CPU, fast)
- YOLO on a *cropped region* around last box every N frames or on tracker failure
- Outputs MegaPose-compatible dets/bboxes.json

Usage:
  python3 moving_bbox_lite.py \
    --megapose_data_dir /home/nahar3/megapose_data \
    --example_name bot \
    --yolo_model yolov8n.pt \
    --imgsz 320 --detect_every 10 \
    --device 0 --show

If GPU OOM persists, add: --device cpu  (or lower --imgsz)
"""
import os, json, argparse
from pathlib import Path
from typing import List, Dict
import cv2, numpy as np

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    _YOLO_OK = False

def list_images(d: Path):
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts])

def xywh_to_xyxy(x, y, w, h): return [float(x), float(y), float(x+w), float(y+h)]
def xyxy_to_xywh(b): x1,y1,x2,y2 = map(float,b); return [x1,y1,max(1.0,x2-x1),max(1.0,y2-y1)]
def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1); inter=iw*ih
    if inter<=0: return 0.0
    area_a=max(0.0,ax2-ax1)*max(0.0,ay2-ay1); area_b=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    return inter/max(1e-6,(area_a+area_b-inter))
def clamp(v,lo,hi): return max(lo,min(hi,v))

def crop_from_box(img, box_xyxy, pad=0.6):
    H,W=img.shape[:2]; x1,y1,x2,y2=box_xyxy
    w=max(1.0,x2-x1); h=max(1.0,y2-y1); cx,cy=x1+0.5*w,y1+0.5*h
    w2,h2=w*(1+pad),h*(1+pad)
    nx1=int(clamp(cx-0.5*w2,0,W-1)); ny1=int(clamp(cy-0.5*h2,0,H-1))
    nx2=int(clamp(cx+0.5*w2,0,W-1)); ny2=int(clamp(cy+0.5*h2,0,H-1))
    return img[ny1:ny2, nx1:nx2].copy(), nx1, ny1

def select_roi(img_bgr):
    print("Draw initial ROI (drag rectangle), press ENTER/SPACE. Press C to cancel.")
    r=cv2.selectROI("Select ROI", img_bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    if r==(0,0,0,0): raise SystemExit("No ROI selected.")
    return xywh_to_xyxy(*r)

def yolo_refine_on_crop(model, frame_bgr, last_box_xyxy, conf=0.25, imgsz=320, device=None):
    if not _YOLO_OK or model is None or last_box_xyxy is None: return None
    crop,x0,y0=crop_from_box(frame_bgr,last_box_xyxy,pad=0.6)
    if crop.size==0: return None
    rlist=model.predict(crop[...,::-1], conf=conf, imgsz=imgsz, verbose=False,
                        device=device, half=False, max_det=30)
    if not rlist: return None
    r=rlist[0]
    if r.boxes is None or len(r.boxes)==0: return None
    boxes=r.boxes.xyxy.cpu().numpy()
    lb=[last_box_xyxy[0]-x0,last_box_xyxy[1]-y0,last_box_xyxy[2]-x0,last_box_xyxy[3]-y0]
    ious=[iou_xyxy(lb,b.tolist()) for b in boxes]; best=int(np.argmax(ious)); bx=boxes[best].tolist()
    return [float(bx[0]+x0),float(bx[1]+y0),float(bx[2]+x0),float(bx[3]+y0)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--megapose_data_dir", required=True)
    ap.add_argument("--example_name", default="bot")
    ap.add_argument("--yolo_model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)   # "0" or "cpu"
    ap.add_argument("--detect_every", type=int, default=10)
    ap.add_argument("--show", action="store_true")
    args=ap.parse_args()

    data_dir=Path(args.megapose_data_dir).resolve()
    ex_dir=data_dir/"examples"/args.example_name
    images_d=ex_dir/"images"
    det_dir=ex_dir/"dets"; det_dir.mkdir(parents=True, exist_ok=True)
    out_json=det_dir/"bboxes.json"

    imlist=list_images(images_d)
    if not imlist: raise SystemExit(f"No frames in {images_d}")

    f0=cv2.imread(str(imlist[0]))
    if f0 is None: raise SystemExit(f"Failed to read first frame: {imlist[0]}")
    init_xyxy=select_roi(f0)

    tracker=(cv2.legacy.TrackerKCF_create() if hasattr(cv2,"legacy") else cv2.TrackerKCF_create())
    if not tracker.init(f0, tuple(xyxy_to_xywh(init_xyxy))):
        raise SystemExit("Tracker failed to initialize.")
    last_box=init_xyxy

    yolo=YOLO(args.yolo_model) if (_YOLO_OK and args.yolo_model) else None
    per_frame={imlist[0].stem:[float(x) for x in last_box]}

    for i,p in enumerate(imlist[1:], start=1):
        img=cv2.imread(str(p))
        if img is None: print(f"[warn] read fail: {p.name}"); continue
        ok, rect=tracker.update(img); cur=xywh_to_xyxy(*rect) if ok else None

        if (i % args.detect_every == 0) or (cur is None):
            if yolo is not None:
                refined=yolo_refine_on_crop(yolo, img, last_box, conf=args.conf, imgsz=args.imgsz, device=args.device)
                if refined is not None:
                    cur=refined
                    tracker=(cv2.legacy.TrackerKCF_create() if hasattr(cv2,"legacy") else cv2.TrackerKCF_create())
                    tracker.init(img, tuple(xyxy_to_xywh(cur)))
        if cur is None: cur=last_box

        per_frame[p.stem]=[float(x) for x in cur]
        last_box=cur

        if args.show:
            vis=img.copy()
            x1,y1,x2,y2=map(int,cur)
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(vis,f"{p.name} (i={i})",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            cv2.imshow("Lite tracking (KCF + crop YOLO)", vis)
            if cv2.waitKey(1)&0xFF==27: break

    if args.show: cv2.destroyAllWindows()
    out_json.write_text(json.dumps(per_frame, indent=2))
    print(f"[lite] Saved per-frame bboxes: {out_json}  ({len(per_frame)}/{len(imlist)} frames)")
    print("Next: batch_infer_dynamic_bbox.py")
if __name__=="__main__": main()
