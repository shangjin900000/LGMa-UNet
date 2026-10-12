"""Enhanced visualization for precomputed ABC GT Poisson fields.

Features
--------
1) Visualize three orthogonal mid-slices (x/y/z) of the PSR volume.
2) Overlay the zero level-set contour on each slice.
3) Extract the zero isosurface with marching cubes.
4) Transform the MC mesh from SAP coordinates back to ABC coordinates.
5) Compare GT mesh and reconstructed mesh using uniformly sampled surface points.
6) Save a preview PNG, reconstructed mesh PLY, and a CSV summary.

Example
-------
python vis_abc_psr_enhanced.py \
    --data_root D:/Datasets/2024_abc_train \
    --psr_dir psr \
    --mesh_dir 03_meshes \
    --out_dir D:/Datasets/2024_abc_train/psr_vis \
    --n 4 \
    --scale 2.0
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Prefer project mc_from_psr when running inside LGMa-Recon tree.
try:
    from SAP.src.utils import mc_from_psr  # type: ignore
except Exception:
    # Fallback: local implementation compatible with the project version.
    def mc_from_psr(psr_grid, pytorchify=False, real_scale=False, zero_level=0):
        from skimage import measure
        import numpy as _np
        if isinstance(psr_grid, torch.Tensor):
            arr = psr_grid.squeeze().detach().cpu().numpy()
            device = psr_grid.device
        else:
            arr = np.asarray(psr_grid).squeeze()
            device = None
        s = arr.shape[-1]
        try:
            verts, faces, normals, values = measure.marching_cubes(arr, level=zero_level)
        except Exception:
            verts, faces, normals, values = measure.marching_cubes(arr)
        if real_scale:
            verts = verts / (s - 1)
        else:
            verts = verts / s
        if pytorchify:
            verts = torch.as_tensor(verts, dtype=torch.float32, device=device)
            faces = torch.as_tensor(faces.astype(np.int64), dtype=torch.long, device=device)
            normals = torch.as_tensor(normals, dtype=torch.float32, device=device)
        return verts, faces, normals


def sap_to_abc(points: np.ndarray, sap_scale: float) -> np.ndarray:
    return sap_scale * (points - 0.5)


def _mesh_path(mesh_dir: str, name: str) -> str:
    for ext in (".ply", ".obj", ".off", ".stl"):
        path = os.path.join(mesh_dir, name + ext)
        if os.path.isfile(path):
            return path
    return os.path.join(mesh_dir, name + ".ply")


def _sample_mesh_points(mesh: o3d.geometry.TriangleMesh, n: int, seed: int = 0) -> np.ndarray:
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if hasattr(o3d.utility, "random") and hasattr(o3d.utility.random, "seed"):
        o3d.utility.random.seed(seed)
    pcd = mesh.sample_points_uniformly(number_of_points=int(n))
    return np.asarray(pcd.points, dtype=np.float64)


def _maybe_subsample(pts: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if pts.shape[0] <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], max_points, replace=False)
    return pts[idx]


def _scatter_pair(ax, gt_pts: np.ndarray, rec_pts: np.ndarray, i: int, j: int, title: str):
    ax.scatter(gt_pts[:, i], gt_pts[:, j], s=0.35, c="#8a8a8a", alpha=0.28, linewidths=0, label="GT surface")
    ax.scatter(rec_pts[:, i], rec_pts[:, j], s=0.35, c="#d04a2b", alpha=0.40, linewidths=0, label="MC surface")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)


def _plot_slice(ax, data2d: np.ndarray, axis_name: str, idx: int, vmax: float):
    im = ax.imshow(data2d.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    try:
        ax.contour(data2d.T, levels=[0.0], colors="k", linewidths=0.9)
    except Exception:
        pass
    ax.set_title(f"PSR {axis_name}={idx}", fontsize=10)
    ax.tick_params(labelsize=7)
    return im


def _chamfer_l2_mean(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Symmetric Chamfer using Open3D KD-tree. Returns (cd_mean, a2b_mean, b2a_mean)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")

    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(b)
    tree_b = o3d.geometry.KDTreeFlann(pcd_b)
    d_a2b = np.empty((a.shape[0],), dtype=np.float64)
    for idx, p in enumerate(a):
        _, _, dist2 = tree_b.search_knn_vector_3d(p, 1)
        d_a2b[idx] = math.sqrt(dist2[0]) if len(dist2) else np.nan

    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(a)
    tree_a = o3d.geometry.KDTreeFlann(pcd_a)
    d_b2a = np.empty((b.shape[0],), dtype=np.float64)
    for idx, p in enumerate(b):
        _, _, dist2 = tree_a.search_knn_vector_3d(p, 1)
        d_b2a[idx] = math.sqrt(dist2[0]) if len(dist2) else np.nan

    a2b = float(np.nanmean(d_a2b))
    b2a = float(np.nanmean(d_b2a))
    cd = a2b + b2a
    return cd, a2b, b2a


def vis_one(
    npz_path: str,
    mesh_path: str,
    out_png: str,
    out_ply: str,
    sap_scale: float,
    surf_n: int,
    draw_n: int,
) -> Dict[str, float]:
    pack = np.load(npz_path)
    psr = np.asarray(pack["psr"], dtype=np.float32)
    saved_scale = float(np.asarray(pack["scale"]).reshape(-1)[0]) if "scale" in pack.files else sap_scale
    if abs(saved_scale - sap_scale) > 1e-6:
        raise RuntimeError(f"{npz_path} scale {saved_scale:.4f} != {sap_scale:.4f}")

    name = os.path.splitext(os.path.basename(npz_path))[0]
    psr_min = float(psr.min())
    psr_max = float(psr.max())
    zero_cross = bool(psr_min < 0.0 and psr_max > 0.0)

    grid = torch.from_numpy(psr)[None]
    verts, faces, _ = mc_from_psr(grid, zero_level=0.0)
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    rec = sap_to_abc(verts, sap_scale)

    rec_mesh = o3d.geometry.TriangleMesh()
    rec_mesh.vertices = o3d.utility.Vector3dVector(rec)
    rec_mesh.triangles = o3d.utility.Vector3iVector(faces)
    rec_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_ply, rec_mesh)

    gt_mesh = o3d.io.read_triangle_mesh(mesh_path)
    gt_mesh.compute_vertex_normals()

    gt_surf = _sample_mesh_points(gt_mesh, surf_n, seed=0)
    rec_surf = _sample_mesh_points(rec_mesh, surf_n, seed=1)

    cd, gt2rec, rec2gt = _chamfer_l2_mean(gt_surf, rec_surf)

    gt_draw = _maybe_subsample(gt_surf, draw_n, seed=10)
    rec_draw = _maybe_subsample(rec_surf, draw_n, seed=11)

    sx, sy, sz = psr.shape
    idx_x, idx_y, idx_z = sx // 2, sy // 2, sz // 2
    slice_x = psr[idx_x, :, :]
    slice_y = psr[:, idx_y, :]
    slice_z = psr[:, :, idx_z]
    vmax = float(np.percentile(np.abs(psr), 99))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.2))
    im = _plot_slice(axes[0, 0], slice_x, "x", idx_x, vmax)
    _plot_slice(axes[0, 1], slice_y, "y", idx_y, vmax)
    _plot_slice(axes[0, 2], slice_z, "z", idx_z, vmax)
    fig.colorbar(im, ax=axes[0, :], fraction=0.025, pad=0.02)

    _scatter_pair(axes[1, 0], gt_draw, rec_draw, 0, 1, "projection: x-y")
    _scatter_pair(axes[1, 1], gt_draw, rec_draw, 0, 2, "projection: x-z")
    _scatter_pair(axes[1, 2], gt_draw, rec_draw, 1, 2, "projection: y-z")
    axes[1, 2].legend(markerscale=8, fontsize=8, loc="upper right")

    sap_min = float(pack["sap_min"]) if "sap_min" in pack.files else float("nan")
    sap_max = float(pack["sap_max"]) if "sap_max" in pack.files else float("nan")
    fig.suptitle(
        (
            f"{name}  S={sap_scale:.2f}  zero_cross={zero_cross}  "
            f"SAP[{sap_min:.3f}, {sap_max:.3f}]  PSR[{psr_min:.3f}, {psr_max:.3f}]\n"
            f"MC verts={rec.shape[0]} faces={faces.shape[0]}  "
            f"surfCD={cd:.6f}  gt→mc={gt2rec:.6f}  mc→gt={rec2gt:.6f}"
        ),
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return {
        "name": name,
        "psr_min": psr_min,
        "psr_max": psr_max,
        "sap_min": sap_min,
        "sap_max": sap_max,
        "zero_cross": int(zero_cross),
        "mc_verts": int(rec.shape[0]),
        "mc_faces": int(faces.shape[0]),
        "surf_cd": float(cd),
        "gt2mc": float(gt2rec),
        "mc2gt": float(rec2gt),
    }


def main():
    parser = argparse.ArgumentParser(description="Enhanced visualization for ABC GT PSR")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--psr_dir", default="psr")
    parser.add_argument("--mesh_dir", default="03_meshes")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--names", default="", help="comma-separated shape names")
    parser.add_argument("--surface_points", type=int, default=20000)
    parser.add_argument("--draw_points", type=int, default=20000)
    args = parser.parse_args()

    data_root = os.path.expanduser(args.data_root)
    psr_dir = args.psr_dir if os.path.isabs(args.psr_dir) else os.path.join(data_root, args.psr_dir)
    mesh_dir = args.mesh_dir if os.path.isabs(args.mesh_dir) else os.path.join(data_root, args.mesh_dir)
    out_dir = args.out_dir or os.path.join(psr_dir, "vis_enhanced")
    os.makedirs(out_dir, exist_ok=True)

    if args.names:
        names = [x.strip() for x in args.names.split(",") if x.strip()]
        files = [os.path.join(psr_dir, n if n.endswith(".npz") else n + ".npz") for n in names]
    else:
        files = sorted(os.path.join(psr_dir, f) for f in os.listdir(psr_dir) if f.endswith(".npz"))[: max(args.n, 0)]
    if not files:
        raise RuntimeError(f"no npz in {psr_dir}")

    rows: List[Dict[str, float]] = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        stats = vis_one(
            path,
            _mesh_path(mesh_dir, name),
            os.path.join(out_dir, name + ".png"),
            os.path.join(out_dir, name + "_mc.ply"),
            args.scale,
            surf_n=args.surface_points,
            draw_n=args.draw_points,
        )
        rows.append(stats)
        print(
            "%s  zero_cross=%d  PSR[%.3f, %.3f]  SAP[%.3f, %.3f]  verts=%d faces=%d  surfCD=%.6f"
            % (
                stats["name"],
                stats["zero_cross"],
                stats["psr_min"],
                stats["psr_max"],
                stats["sap_min"],
                stats["sap_max"],
                stats["mc_verts"],
                stats["mc_faces"],
                stats["surf_cd"],
            ),
            flush=True,
        )

    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name", "zero_cross", "psr_min", "psr_max", "sap_min", "sap_max",
                "mc_verts", "mc_faces", "surf_cd", "gt2mc", "mc2gt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {len(rows)} previews -> {out_dir}")
    print(f"summary csv -> {csv_path}")


if __name__ == "__main__":
    main()
