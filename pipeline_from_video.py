#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from pathlib import Path
import cv2
from typing import Optional


def _run_module_or_subprocess(module_name: str, script_path: Path, argv: list[str]):
    try:
        if script_path and str(script_path.parent) not in sys.path:
            sys.path.insert(0, str(script_path.parent))
        mod = __import__(module_name, fromlist=['*'])
        if hasattr(mod, "main"):
            _old_argv = sys.argv
            try:
                sys.argv = [str(script_path)] + argv
                return mod.main()
            finally:
                sys.argv = _old_argv
        else:
            raise ImportError(f"{module_name} has no main()")
    except Exception as e:
        # Fallback: subprocess
        cmd = [sys.executable, str(script_path)] + argv
        print(f"[info] Falling back to subprocess for {module_name}: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        return None


def extract_frames_ffmpeg_or_cv(video_path: Path, images_dir: Path, num_frames: int = 900):
    images_dir.mkdir(parents=True, exist_ok=True)
    # Prefer ffmpeg (faster, preserves quality)
    try:
        subprocess.check_call([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vframes", str(num_frames),
            "-qscale:v", "2",
            str(images_dir / "IMG-%06d.jpg")
        ])
        return
    except Exception:
        print("[warn] ffmpeg not available; using OpenCV fallback")
    # OpenCV fallback
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = 0
    while total < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = images_dir / f"IMG-{total+1:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        total += 1
    cap.release()
    if total < num_frames:
        print(f"[warn] Only extracted {total}/{num_frames} frames (video too short).")

def run_megapose_batch_infer(megapose_root: Path):
    os.chdir(str(megapose_root))
    os.environ.setdefault("PANDA_OFFSCREEN", "1")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        del os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

    script = megapose_root / "batch_infer.py"
    _run_module_or_subprocess("batch_infer", script, [])

def run_overlay(examples_dir: Path, label: str = "bot", mesh_units: str = "mm", megapose_root: Optional[Path] = None):
    search_dir = Path.cwd()
    script = search_dir / "overlay_mesh_bbox_from_obj.py"
    module_name = "overlay_mesh_bbox_from_obj"
    if not script.exists() and megapose_root is not None:
        script = megapose_root / "overlay_mesh_bbox_from_obj.py"
    argv = ["--examples_dir", str(examples_dir), "--label", label, "--mesh_units", mesh_units]
    _run_module_or_subprocess(module_name, script, argv)

def run_make_video(outputs_root: Path, out_video: Path, fps: int = 12, size: Optional[str] = None, megapose_root: Optional[Path] = None):
    search_dir = Path.cwd()
    script = search_dir / "make_video.py"
    module_name = "make_video"
    if not script.exists() and megapose_root is not None:
        script = megapose_root / "make_video.py"
    argv = ["--root", str(outputs_root), "--out", str(out_video), "--fps", str(fps)]
    if size:
        argv += ["--size", size]
    _run_module_or_subprocess(module_name, script, argv)


def main():
    ap = argparse.ArgumentParser(description="End-to-end: video -> 900 frames -> MegaPose poses -> 3D bbox overlays -> video")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--megapose_root", default="/home/nahar3/megapose6d",
                    help="Folder containing batch_infer.py, overlay_mesh_bbox_from_obj.py, make_video.py")
    ap.add_argument("--megapose_data_dir", default="/home/nahar3/megapose_data",
                    help="MEGAPOSE_DATA_DIR root")
    ap.add_argument("--example_name", default="bot", help="Example folder name under MEGAPOSE_DATA_DIR/examples/")
    ap.add_argument("--num_frames", type=int, default=900, help="Number of frames to extract")
    ap.add_argument("--mesh_units", choices=["mm","m"], default="mm", help="Units of your mesh used by MegaPose")
    ap.add_argument("--fps", type=int, default=12, help="Output video FPS")
    ap.add_argument("--size", default=None, help="Output WxH, e.g., 1920x1080 (optional)")
    args = ap.parse_args()

    megapose_root = Path(args.megapose_root).resolve()
    data_root = Path(args.megapose_data_dir).resolve()
    examples_dir = data_root / "examples" / args.example_name
    images_dir = examples_dir / "images"
    outputs_root = examples_dir / "batch_outputs_offscreen"
    final_video = examples_dir / "overlay_bbox.mp4"

    os.environ["MEGAPOSE_DATA_DIR"] = str(data_root)

    print(f" Extracting {args.num_frames} frames from {args.video} -> {images_dir}")
    extract_frames_ffmpeg_or_cv(Path(args.video), images_dir, num_frames=args.num_frames)

    print(f" Running MegaPose batch inference from {megapose_root}")
    run_megapose_batch_infer(megapose_root)

    print(f" Overlaying 3D bbox on frames under {outputs_root}")
    run_overlay(examples_dir, label=args.example_name, mesh_units=args.mesh_units, megapose_root=megapose_root)

    print(f" Building final video -> {final_video}")
    size = args.size
    run_make_video(outputs_root, final_video, fps=args.fps, size=size, megapose_root=megapose_root)

    print(f"\n Pipeline complete.\n  Frames: {images_dir}\n  Poses:  {outputs_root}/*/outputs/object_data.json\n  Video:  {final_video}")

if __name__ == "__main__":
    main()
