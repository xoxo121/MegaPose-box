#!/usr/bin/env python3

import os, sys, argparse, subprocess
from pathlib import Path
import shutil
from typing import Optional

def run(cmd, env=None):
    print(">>", " ".join(map(str, cmd)))
    subprocess.check_call(list(map(str, cmd)), env=env)

def must_find(name: str, *candidates: Path) -> Path:
    for c in candidates:
        if c and c.exists():
            return c.resolve()
    msg = f"Could not find required script '{name}'. Tried:\n" + "\n".join([f" - {c}" for c in candidates if c])
    raise FileNotFoundError(msg)

def extract_frames(video: Path, out_dir: Path, num_frames: int, fps: Optional[int], force: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    src_file = out_dir.parent / "video_source.txt"
    current_sig = f"video={video}\nfps={fps}\nnum_frames={num_frames}\n"

    # Decide whether to (re)extract
    do_extract = force
    if not do_extract:
        has_any = any(out_dir.glob("IMG-*.jpg")) or any(out_dir.glob("IMG-*.png"))
        if not has_any:
            do_extract = True
        else:
            # If frames exist, check if they belong to the same video
            if not src_file.exists() or src_file.read_text() != current_sig:
                for p in out_dir.glob("*"):
                    p.unlink()
                do_extract = True

    if not do_extract:
        print(f"[extract] Frames already match the requested video/settings in {out_dir}. Skipping.")
        return

    # Try ffmpeg
    try:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            pattern = str(out_dir / "IMG-%06d.jpg")
            cmd = [ffmpeg, "-y", "-i", str(video)]
            if fps:
                cmd += ["-vf", f"fps={fps}"]
            cmd += ["-vframes", str(num_frames), pattern]
            print(f"[extract] ffmpeg -> {pattern}")
            run(cmd)
            return
        else:
            print("[extract] ffmpeg not found; using OpenCV fallback.")
    except Exception as e:
        print(f"[extract] ffmpeg failed: {e}; falling back to OpenCV.")

    # OpenCV fallback (no deps)
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    total = 0
    frame_id = 1
    step = 1
    
    if fps:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if src_fps and src_fps > 0:
            step = max(1, int(round(src_fps / fps)))
    while total < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_id - 1) % step == 0:
            out = out_dir / f"IMG-{total+1:06d}.jpg"
            cv2.imwrite(str(out), frame)
            total += 1
        frame_id += 1
    cap.release()
    print(f"[extract] Wrote {total} frames to {out_dir}")
    src_file.write_text(current_sig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--megapose_root", default="/home/nahar3/megapose6d")
    ap.add_argument("--megapose_data_dir", default="/home/nahar3/megapose_data")
    ap.add_argument("--example_name", default="bot")
    ap.add_argument("--num_frames", type=int, default=900)
    ap.add_argument("--mesh_units", choices=["mm","m"], default="mm")
    ap.add_argument("--fps", type=int, default=12, help="Target FPS for extraction/preview")
    ap.add_argument("--size", default=None, help="WxH for final video (optional)")
    ap.add_argument("--force_extract", action="store_true", help="Always re-extract frames (clears images/)")


    # moving bbox knobs
    ap.add_argument("--yolo_model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)  # e.g., "0" or "cpu"
    ap.add_argument("--detect_every", type=int, default=10)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    megapose_root = Path(args.megapose_root).resolve()
    work_dir      = Path.cwd().resolve()
    data_root     = Path(args.megapose_data_dir).resolve()
    ex_dir        = data_root / "examples" / args.example_name
    images_dir    = ex_dir / "images"
    outputs_root  = ex_dir / "batch_outputs_offscreen"
    final_video   = ex_dir / "overlay_bbox.mp4"

    env = dict(os.environ)
    env["MEGAPOSE_DATA_DIR"] = str(data_root)

    moving_bbox = must_find(
        "moving_bbox_lite.py",
        work_dir / "moving_bbox_lite.py",
        megapose_root / "moving_bbox_lite.py",
        Path("/mnt/data/moving_bbox.py"),     
    )
    dyn_infer = must_find(
        "batch_infer_dynamic_bbox.py",
        work_dir / "batch_infer_dynamic_bbox.py",
        megapose_root / "batch_infer_dynamic_bbox.py",
    )
    overlay_script = must_find(
        "overlay_mesh_bbox_from_obj.py",
        work_dir / "overlay_mesh_bbox_from_obj.py",
        megapose_root / "overlay_mesh_bbox_from_obj.py",
        Path("/mnt/data/overlay_mesh_bbox_from_obj.py"),
    )
    makevid_script = must_find(
        "make_video.py",
        work_dir / "make_video.py",
        megapose_root / "make_video.py",
        Path("/mnt/data/make_video.py"),
    )

    print(f" Extracting up to {args.num_frames} frames from {video} -> {images_dir}")
    extract_frames(video, images_dir, num_frames=args.num_frames, fps=args.fps, force=args.force_extract)

    print(f" Tracking bboxes with KCF + crop-YOLO")
    bbox_cmd = [
        sys.executable, str(moving_bbox),
        "--megapose_data_dir", str(data_root),
        "--example_name", args.example_name,
        "--yolo_model", args.yolo_model,
        "--imgsz", str(args.imgsz),
        "--conf", str(args.conf),
        "--detect_every", str(args.detect_every),
    ]
    if args.device:
        bbox_cmd += ["--device", args.device]
    if args.show:
        bbox_cmd += ["--show"]
    run(bbox_cmd, env=env)

    print(f" MegaPose batch inference (dynamic bboxes)")
    run([sys.executable, str(dyn_infer)], env=env)

    print(f" Overlaying results")
    run([
        sys.executable, str(overlay_script),
        "--examples_dir", str(ex_dir),
        "--label", args.example_name,
        "--mesh_units", args.mesh_units
    ], env=env)

    print(f" Building final video -> {final_video}")
    make_cmd = [
        sys.executable, str(makevid_script),
        "--root", str(outputs_root),
        "--out", str(final_video),
        "--fps", str(args.fps),
    ]
    if args.size:
        make_cmd += ["--size", args.size]
    run(make_cmd, env=env)

    print("\nPipeline complete.")
    print(" Frames:", images_dir)
    print(" BBoxes:", ex_dir / "dets" / "bboxes.json")
    print(" Poses:",  outputs_root / "<frame>" / "outputs" / "object_data.json")
    print(" Video:",  final_video)

if __name__ == "__main__":
    main()
