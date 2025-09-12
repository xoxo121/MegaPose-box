#!/usr/bin/env python3
import argparse, cv2, os
from pathlib import Path
from natsort import natsorted  # pip install natsort

def pad_to_size(img, size):
    H, W = size
    h, w = img.shape[:2]
    # scale to fit inside target while keeping aspect
    scale = min(W / w, H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    # pad to center
    top = (H - nh) // 2
    left = (W - nw) // 2
    canvas = cv2.copyMakeBorder(resized, top, H - nh - top, left, W - nw - left,
                                borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    return canvas

def main():
    ap = argparse.ArgumentParser(description="Make a video from overlay images")
    ap.add_argument("--root", required=True,
                    help="Path to batch_outputs_offscreen (frames as subfolders).")
    ap.add_argument("--pattern", default="overlay_3dbox.png",
                    help="Image filename in each frame folder.")
    ap.add_argument("--out", default="overlay_bbox.mp4", help="Output video path")
    ap.add_argument("--fps", type=int, default=12, help="Frames per second")
    ap.add_argument("--size", default=None,
                    help="Target WxH (e.g., 1280x720). Default = size of first frame")
    ap.add_argument("--fourcc", default="mp4v", help="Codec fourcc (mp4v, avc1, XVID...)")
    args = ap.parse_args()

    root = Path(args.root)
    frames = natsorted([p / args.pattern for p in root.iterdir() if (p / args.pattern).exists()],
                       key=lambda p: p.parent.name)  # sort by folder name

    if not frames:
        raise SystemExit(f"No frames found under {root} matching {args.pattern}")

    # determine output size
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise SystemExit(f"Failed to read {frames[0]}")
    if args.size:
        W, H = map(int, args.size.lower().split("x"))
        out_size = (W, H)
    else:
        H, W = first.shape[:2]
        out_size = (W, H)

    print(f"Writing {args.out} at {args.fps} fps, size={out_size[0]}x{out_size[1]} ({len(frames)} frames)")

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    vw = cv2.VideoWriter(args.out, fourcc, args.fps, out_size)

    for i, f in enumerate(frames, 1):
        img = cv2.imread(str(f))
        if img is None:
            print(f"[WARN] skip unreadable: {f}")
            continue
        if img.shape[1] != out_size[0] or img.shape[0] != out_size[1]:
            img = pad_to_size(img, (out_size[1], out_size[0]))  # (H,W)
        vw.write(img)
        if i % 25 == 0:
            print(f"  {i}/{len(frames)}")

    vw.release()
    print("✓ Done:", args.out)

if __name__ == "__main__":
    main()
