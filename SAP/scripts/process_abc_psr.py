"""Precompute ABC GT Poisson fields with a dataset-wide SAP scale.

P_SAP = P_ABC / S + 0.5, S=2.0 (from ABC max|P|=0.862).
Mesh -> uniform surface samples with triangle normals -> SAP box -> DPSR(sig=0).

Example:
  python SAP/scripts/process_abc_psr.py \
    --data_root D:/Datasets/2024_abc_train \
    --mesh_dir 03_meshes \
    --out_dir psr \
    --scale 2.0 \
    --num_points 100000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from SAP.src.dpsr import DPSR

DEFAULT_SCALE = 2.0


def abc_to_sap(points, sap_scale):
    return points / sap_scale + 0.5


def _shape_names(data_root, list_file, mesh_dir):
    if list_file:
        path = list_file if os.path.isfile(list_file) else os.path.join(data_root, list_file)
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    names = []
    for fname in sorted(os.listdir(mesh_dir)):
        if fname.lower().endswith((".ply", ".obj", ".off", ".stl")):
            names.append(os.path.splitext(fname)[0])
    return names


def _mesh_path(mesh_dir, name):
    direct = os.path.join(mesh_dir, name)
    if os.path.isfile(direct):
        return direct
    for ext in (".ply", ".obj", ".off", ".stl"):
        path = os.path.join(mesh_dir, name + ext)
        if os.path.isfile(path):
            return path
    return os.path.join(mesh_dir, name + ".ply")


def sample_oriented_points(mesh, num_points):
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    pcd = mesh.sample_points_uniformly(
        number_of_points=int(num_points),
        use_triangle_normal=True,
    )
    points = np.asarray(pcd.points, dtype=np.float64)
    normals = np.asarray(pcd.normals, dtype=np.float64)
    if normals.size == 0:
        raise RuntimeError("sampled point cloud has no normals")
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(norms, 1e-8, None)
    return points, normals


def process_one(mesh_path, out_path, dpsr, num_points, device, sap_scale):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError("empty mesh: %s" % mesh_path)
    points, normals = sample_oriented_points(mesh, num_points)
    points_unit = abc_to_sap(points, sap_scale)
    lo = float(points_unit.min())
    hi = float(points_unit.max())
    if lo < 0.0 or hi > 0.99:
        raise RuntimeError("SAP coords out of [0, 0.99]: [%.4f, %.4f] file=%s" % (lo, hi, mesh_path))
    with torch.no_grad():
        psr = dpsr(
            torch.from_numpy(points_unit.astype(np.float32))[None].to(device),
            torch.from_numpy(normals.astype(np.float32))[None].to(device),
        )
    psr_np = psr.squeeze().detach().cpu().numpy().astype(np.float16)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        psr=psr_np,
        scale=np.float32(sap_scale),
        sap_min=np.float32(lo),
        sap_max=np.float32(hi),
    )
    return lo, hi


def main():
    parser = argparse.ArgumentParser(description="ABC mesh -> GT Poisson field")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--mesh_dir", default="03_meshes")
    parser.add_argument("--out_dir", default="psr")
    parser.add_argument("--list_file", default="", help="optional trainset.txt / valset.txt")
    parser.add_argument("--num_points", type=int, default=100000)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=0.0, help="GT DPSR uses sig=0")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="dataset-wide S_ABC")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    if args.scale <= 0:
        raise ValueError("--scale must be > 0")

    data_root = os.path.expanduser(args.data_root)
    mesh_dir = args.mesh_dir if os.path.isabs(args.mesh_dir) else os.path.join(data_root, args.mesh_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(data_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "scale.json"), "w") as f:
        json.dump({"sap_scale": float(args.scale), "formula": "P_SAP = P_ABC / S + 0.5"}, f, indent=2)

    names = _shape_names(data_root, args.list_file, mesh_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dpsr = DPSR(res=(args.resolution, args.resolution, args.resolution), sig=args.sigma).to(device)
    dpsr.eval()

    failed = []
    sap_lo, sap_hi = 1.0, 0.0
    n_ok = 0
    for name in tqdm(names, desc="ABC PSR"):
        out_path = os.path.join(out_dir, name + ".npz")
        if args.skip_existing and os.path.isfile(out_path):
            continue
        mesh_path = _mesh_path(mesh_dir, name)
        try:
            lo, hi = process_one(mesh_path, out_path, dpsr, args.num_points, device, args.scale)
            sap_lo = min(sap_lo, lo)
            sap_hi = max(sap_hi, hi)
            n_ok += 1
        except Exception as exc:
            failed.append((name, str(exc)))
            print("[skip] %s: %s" % (name, exc), flush=True)

    print(
        "done: %d ok, %d failed, S=%.3f, sampled SAP range [%.4f, %.4f], out=%s"
        % (n_ok, len(failed), args.scale, sap_lo, sap_hi, out_dir)
    )
    if failed:
        fail_log = os.path.join(out_dir, "failed.txt")
        with open(fail_log, "w") as f:
            for name, msg in failed:
                f.write("%s\t%s\n" % (name, msg))
        print("failed list: %s" % fail_log)


if __name__ == "__main__":
    main()
