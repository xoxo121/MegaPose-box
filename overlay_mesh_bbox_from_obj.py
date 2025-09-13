#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
import numpy as np
import cv2
try:
    import open3d as o3d
except Exception:
    o3d = None
try:
    import trimesh
except Exception:
    trimesh = None

def load_K_and_base_res(cam_json):
    data = json.loads(Path(cam_json).read_text())
    K = np.array(data["K"], dtype=float)
    base_res = data["resolution"]  # [H, W]
    return K, base_res

def scale_intrinsics(K, old_hw, new_hw):
    oh, ow = old_hw
    nh, nw = new_hw
    sx, sy = nw / ow, nh / oh
    K = np.array(K, dtype=float)
    return np.array([
        [K[0,0]*sx, 0.0,       K[0,2]*sx],
        [0.0,       K[1,1]*sy, K[1,2]*sy],
        [0.0,       0.0,       1.0]
    ])

def find_image_by_stem(images_dir: Path, stem: str):
    for ext in (".png",".jpg",".jpeg",".bmp",".tif",".tiff"):
        p = images_dir / f"{stem}{ext}"
        if p.exists(): return p
    for p in images_dir.iterdir():
        if p.is_file() and p.stem == stem: return p
    return None

def quat_to_R(q, fmt="xyzw"):
    q = np.asarray(q, dtype=float).reshape(-1)
    if fmt == "wxyz":
        qw, qx, qy, qz = q
    else: 
        qx, qy, qz, qw = q
    n = np.linalg.norm([qx,qy,qz,qw]) or 1.0
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    x2, y2, z2 = 2*qx, 2*qy, 2*qz
    xx, yy, zz = qx*x2, qy*y2, qz*z2
    xy, xz, yz = qx*y2, qx*z2, qy*z2
    wx, wy, wz = qw*x2, qw*y2, qw*z2
    return np.array([
        [1-(yy+zz),   xy-wz,       xz+wy],
        [xy+wz,       1-(xx+zz),   yz-wx],
        [xz-wy,       yz+wx,       1-(xx+yy)]
    ], dtype=float)

def _Rt_from_entry(d, quat_fmt):
    if "TWO" in d:
        two = d["TWO"]
        if isinstance(two, dict):
            if "matrix" in two:
                T = np.array(two["matrix"], dtype=float)
                return T[:3,:4]
            if "mat" in two:
                T = np.array(two["mat"], dtype=float)
                return T[:3,:4] if T.shape==(3,4) else T[:3,:4]
            if {"rotation","translation"} <= set(two.keys()):
                R = np.array(two["rotation"], dtype=float).reshape(3,3)
                t = np.array(two["translation"], dtype=float).reshape(3,1)
                return np.concatenate([R, t], axis=1)
        if isinstance(two, list) and len(two) == 2:
            q = two[0]
            t = np.array(two[1], dtype=float).reshape(3,1)
            R = quat_to_R(q, fmt=quat_fmt)
            return np.concatenate([R, t], axis=1)
    # fallback: flat arrays
    for k, v in d.items():
        if "matrix" in k and isinstance(v, list) and len(v) in (12,16):
            arr = np.array(v, dtype=float)
            return arr.reshape(4,4)[:3,:4] if arr.size==16 else arr.reshape(3,4)
    return None

def load_all_Rts(obj_json_path: Path, quat_fmt="xyzw"):
    data = json.loads(obj_json_path.read_text())
    if not data:
        return []
    Rts = []
    for d in data:
        Rt = _Rt_from_entry(d, quat_fmt)
        if Rt is not None:
            Rts.append(Rt)
    return Rts

def _unit_scale_from_flag(units: str) -> float:
    u = units.lower()
    if u in ("m", "meter", "meters"):
        return 1.0
    if u in ("cm", "centimeter", "centimeters"):
        return 1e-2
    if u in ("mm", "millimeter", "millimeters"):
        return 1e-3
    if u in ("0.1mm", "0.1-mm", "0_1mm"):
        return 1e-4
    raise ValueError(f"Unsupported mesh_units: {units}")

def get_bbox_corners_from_mesh(mesh_path: Path, mesh_units="mm", use_obb=False, mesh_scale=1.0):
    unit_scale = _unit_scale_from_flag(mesh_units) * float(mesh_scale)

    verts = None
    if o3d is not None:
        try:
            geo = o3d.io.read_triangle_mesh(str(mesh_path))
            if geo.has_vertices():
                verts = np.asarray(geo.vertices, dtype=np.float64)
        except Exception:
            pass
    if verts is None and trimesh is not None:
        m = trimesh.load(str(mesh_path), force='mesh')
        verts = np.asarray(m.vertices, dtype=np.float64)
    if verts is None:
        raise RuntimeError("Need open3d or trimesh to read mesh vertices.")

    verts = verts * unit_scale

    if use_obb:
        if o3d is not None:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verts))
            obb = pcd.get_oriented_bounding_box()
            corners = np.asarray(obb.get_box_points(), dtype=np.float64)
        elif trimesh is not None:
            corners = np.asarray(trimesh.points.PointCloud(verts).bounding_box_oriented.vertices,
                                 dtype=np.float64)
        else:
            raise RuntimeError("No geometry backend for OBB.")
    else:
        vmin = verts.min(axis=0); vmax = verts.max(axis=0)
        x0,y0,z0 = vmin; x1,y1,z1 = vmax
        corners = np.array([
            [x0,y0,z0],[x0,y1,z0],[x1,y1,z0],[x1,y0,z0],
            [x0,y0,z1],[x0,y1,z1],[x1,y1,z1],[x1,y0,z1]
        ], dtype=np.float64)

    size = corners.max(axis=0) - corners.min(axis=0)
    print(f"[bbox] size (m): {size[0]:.5f} x {size[1]:.5f} x {size[2]:.5f}  (units={mesh_units}, scale={mesh_scale}, OBB={use_obb})")
    return corners

def project_points(K, Rt, X_obj):
    R = Rt[:,:3]; t = Rt[:,3:4]
    Xc = (R @ X_obj.T) + t  # 3xN
    z = Xc[2,:].copy(); z[z == 0] = 1e-9
    x = (K @ Xc).T
    u = x[:,0] / z
    v = x[:,1] / z
    return np.stack([u, v], axis=1), Xc.T 

def draw_box(img, pts2d, color=(0,255,0), thickness=2):
    pts = np.round(pts2d).astype(int)
    E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    vis = img.copy()
    for a,b in E:
        cv2.line(vis, tuple(pts[a]), tuple(pts[b]), color, thickness)
    for p in pts:
        cv2.circle(vis, tuple(p), 2, (0,0,255), -1)
    return vis

def _box_score(pts2d, Xc, W, H):
    in_front = (Xc[:,2] > 0).sum() / 8.0
    inside = ((pts2d[:,0] >= 0) & (pts2d[:,0] < W) & (pts2d[:,1] >= 0) & (pts2d[:,1] < H)).sum() / 8.0
    xmin, ymin = pts2d[:,0].min(), pts2d[:,1].min()
    xmax, ymax = pts2d[:,0].max(), pts2d[:,1].max()
    area = max(0.0, (xmax - xmin)) * max(0.0, (ymax - ymin))
    area_norm = area / float(W * H + 1e-9)
    penalty = 0.0
    if area_norm < 0.005: penalty += 0.5  
    if area_norm > 0.70: penalty += 0.5   
    return in_front + inside - penalty

def pick_best_Rt(Rts, K, corners_obj, W, H):
    best = None; best_score = -1e9
    for Rt in Rts:
        pts2d, Xc = project_points(K, Rt, corners_obj)
        s = _box_score(pts2d, Xc, W, H)
        if s > best_score:
            best_score, best = s, (Rt, pts2d, Xc)
    return best 

def main():
    ap = argparse.ArgumentParser("Overlay 3D mesh bbox per MegaPose pose (multi-hyp robust)")
    ap.add_argument("--examples_dir", required=True,
                    help="e.g. /home/nahar3/megapose_data/examples/bot")
    ap.add_argument("--images_sub", default="images")
    ap.add_argument("--outputs_root", default="batch_outputs_offscreen")
    ap.add_argument("--label", default="bot")
    ap.add_argument("--mesh_path", default=None,
                    help="override mesh; else uses examples_dir/meshes/<label>/model.(obj|ply)")
    ap.add_argument("--mesh_units", default="mm", choices=["m","cm","mm","0.1mm"])
    ap.add_argument("--mesh_scale", type=float, default=1.0, help="uniform scale in object coords")
    ap.add_argument("--quat_fmt", default="xyzw", choices=["xyzw","wxyz"], help="quaternion order in JSON")
    ap.add_argument("--use_obb", action="store_true", help="use oriented bbox in object space")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ex_dir = Path(args.examples_dir).resolve()
    images_dir = ex_dir / args.images_sub
    outputs_root = ex_dir / args.outputs_root
    cam_json = ex_dir / "camera_data.json"

    assert images_dir.is_dir(), f"Missing images dir: {images_dir}"
    assert outputs_root.is_dir(), f"Missing outputs root: {outputs_root}"
    assert cam_json.exists(), f"Missing camera_data.json at {ex_dir}"

    K_base, base_res = load_K_and_base_res(cam_json)

    if args.mesh_path:
        mesh_path = Path(args.mesh_path)
    else:
        mesh_dir = ex_dir / "meshes" / args.label
        mesh_path = mesh_dir / "model.obj"
        if not mesh_path.exists():
            mesh_path = mesh_dir / "model.ply"
    assert mesh_path.exists(), f"Mesh not found: {mesh_path}"
    corners_obj = get_bbox_corners_from_mesh(
        mesh_path, mesh_units=args.mesh_units, use_obb=args.use_obb, mesh_scale=args.mesh_scale
    )

    # Iterate frames
    frame_dirs = sorted([p for p in outputs_root.iterdir() if p.is_dir()])
    if not frame_dirs:
        raise RuntimeError(f"No frame folders under {outputs_root}")

    for fd in frame_dirs:
        obj_json = fd / "outputs" / "object_data.json"
        if not obj_json.exists():
            continue
        stem = fd.name
        img_path = find_image_by_stem(images_dir, stem)
        if img_path is None:
            print(f"[WARN] no image for {stem}")
            continue

        save_path = fd / "overlay_3dbox.png"
        if save_path.exists() and not args.overwrite:
            continue

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] cannot read {img_path}")
            continue
        H, W = bgr.shape[:2]
        K = scale_intrinsics(K_base, base_res, [H, W]) if [H,W] != base_res else K_base
        Rts = load_all_Rts(obj_json, quat_fmt=args.quat_fmt)

        if not Rts:
            other = "wxyz" if args.quat_fmt == "xyzw" else "xyzw"
            Rts = load_all_Rts(obj_json, quat_fmt=other)

        if not Rts:
            print(f"[WARN] no valid pose parsed in {obj_json}")
            vis = bgr
        else:
            Rt, pts2d, Xc = pick_best_Rt(Rts, K, corners_obj, W, H)
            vis = draw_box(bgr, pts2d, (0,255,0), 2)

        cv2.imwrite(str(save_path), vis)
        print("Saved", save_path)

    print("Done")

if __name__ == "__main__":
    main()
