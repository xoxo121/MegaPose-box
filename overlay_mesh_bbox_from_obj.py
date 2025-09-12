#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import trimesh

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

def load_object_pose_from_json(obj_json_path: Path):
    import numpy as np
    def quat_xyzw_to_R(q):
        qx, qy, qz, qw = q
        n = np.linalg.norm([qx,qy,qz,qw]); 
        if n == 0: n = 1.0
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
        x2, y2, z2 = 2*qx, 2*qy, 2*qz
        xx, yy, zz = qx*x2, qy*y2, qz*z2
        xy, xz, yz = qx*y2, qx*z2, qy*z2
        wx, wy, wz = qw*x2, qw*y2, qw*z2
        return np.array([
            [1-(yy+zz),   xy-wz,       xz+wy   ],
            [xy+wz,       1-(xx+zz),   yz-wx   ],
            [xz-wy,       yz+wx,       1-(xx+yy)]
        ], dtype=float)

    data = json.loads(obj_json_path.read_text())
    if not data:
        raise RuntimeError(f"No objects in {obj_json_path}")
    d0 = data[0]

    if "TWO" in d0:
        two = d0["TWO"]
        # TWO is dict with matrix/mat or rotation/translation
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
                Rt = np.concatenate([R, t], axis=1)
                return Rt
            
        # TWO is list: [quat_xyzw, t]
        if isinstance(two, list) and len(two) == 2:
            q = np.array(two[0], dtype=float).reshape(-1)
            t = np.array(two[1], dtype=float).reshape(3,1)
            if q.size == 4:  # assume [qx,qy,qz,qw]
                R = quat_xyzw_to_R(q)
                Rt = np.concatenate([R, t], axis=1)  # 3x4
                return Rt

    for k, v in d0.items():
        if "matrix" in k and isinstance(v, list) and len(v) in (12,16):
            arr = np.array(v, dtype=float)
            if arr.size == 16:
                T = arr.reshape(4,4)
                return T[:3,:4]
            else:
                Rt = arr.reshape(3,4)
                return Rt

    raise RuntimeError(f"Unrecognized pose format in {obj_json_path}")


def get_bbox_corners_from_mesh(mesh_path: Path, mesh_units="m", use_obb=False):
    scale = 0.001 if mesh_units.lower() in ("mm","millimeter","millimeters") else 1.0
    verts = None
    try:
        m = o3d.io.read_triangle_mesh(str(mesh_path))
        if not m.has_vertices(): raise RuntimeError("Mesh has no vertices")
        if use_obb:
            obb = m.get_oriented_bounding_box()
            corners = np.asarray(obb.get_box_points(), dtype=np.float64) * scale
            return corners
        else:
            aabb = m.get_axis_aligned_bounding_box()
            minv = np.asarray(aabb.get_min_bound(), dtype=np.float64) * scale
            maxv = np.asarray(aabb.get_max_bound(), dtype=np.float64) * scale
    except Exception:
        # fallback: trimesh
        m = trimesh.load(str(mesh_path), force='mesh')
        if use_obb:
            obb = m.bounding_box_oriented
            corners = np.asarray(obb.vertices, dtype=np.float64) * scale
            return corners
        else:
            minv = np.asarray(m.bounds[0], dtype=np.float64) * scale
            maxv = np.asarray(m.bounds[1], dtype=np.float64) * scale

    # AABB corners
    x0,y0,z0 = minv
    x1,y1,z1 = maxv
    return np.array([
        [x0,y0,z0],[x0,y1,z0],[x1,y1,z0],[x1,y0,z0],
        [x0,y0,z1],[x0,y1,z1],[x1,y1,z1],[x1,y0,z1]
    ], dtype=np.float64)

def project_points(K, Rt, X_obj):
    R = Rt[:,:3]; t = Rt[:,3:4]
    Xc = (R @ X_obj.T) + t 
    z = Xc[2,:].copy(); z[z==0]=1e-9
    x = (K @ Xc).T
    u = x[:,0]/z; v = x[:,1]/z
    return np.stack([u,v], axis=1), Xc.T  

def draw_box(img, pts2d, color=(0,255,0), thickness=2):
    pts = np.round(pts2d).astype(int)
    E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    vis = img.copy()
    # draw edges
    for a,b in E:
        cv2.line(vis, tuple(pts[a]), tuple(pts[b]), color, thickness)
    # draw corners
    for p in pts:
        cv2.circle(vis, tuple(p), 2, (0,0,255), -1)
    return vis

def main():
    ap = argparse.ArgumentParser("Overlay 3D mesh bbox per MegaPose pose")
    ap.add_argument("--examples_dir", required=True,
                    help="e.g. /home/nahar3/megapose_data/examples/bot")
    ap.add_argument("--images_sub", default="images")
    ap.add_argument("--outputs_root", default="batch_outputs_offscreen")
    ap.add_argument("--label", default="bot")
    ap.add_argument("--mesh_path", default=None,
                    help="override mesh; else uses examples_dir/meshes/<label>/model.(obj|ply)")
    ap.add_argument("--mesh_units", default="m", choices=["m","mm"])
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

    # camera
    K_base, base_res = load_K_and_base_res(cam_json)

    # mesh - bbox corners (object coords)
    if args.mesh_path:
        mesh_path = Path(args.mesh_path)
    else:
        mesh_dir = ex_dir / "meshes" / args.label
        mesh_path = mesh_dir / "model.obj"
        if not mesh_path.exists():
            mesh_path = mesh_dir / "model.ply"
    assert mesh_path.exists(), f"Mesh not found: {mesh_path}"
    corners_obj = get_bbox_corners_from_mesh(mesh_path, args.mesh_units, use_obb=args.use_obb)

    # iterate frames
    frame_dirs = sorted([p for p in outputs_root.iterdir() if p.is_dir()])
    if not frame_dirs:
        raise RuntimeError(f"No frame folders under {outputs_root}")

    for fd in frame_dirs:
        obj_json = fd / "outputs" / "object_data.json"
        if not obj_json.exists(): continue
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

        Rt = load_object_pose_from_json(obj_json)  # 3x4
        pts2d, Xc = project_points(K, Rt, corners_obj)
        mask_front = (Xc[:,2] > 0).astype(bool)
        vis = draw_box(bgr, pts2d, (0,255,0), 2)

        cv2.imwrite(str(save_path), vis)
        print("Saved", save_path)

    print("Done")

if __name__ == "__main__":
    main()
