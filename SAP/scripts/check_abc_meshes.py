"""Check ABC triangle meshes for holes, non-manifold edges, winding, and normal sign.

Reports:
  - open boundary edges (used by 1 face)
  - non-manifold edges (used by 3+ faces)
  - inconsistent face orientation (2 faces share an edge in the same direction)
  - inward vs outward (signed volume; only for closed, consistently oriented meshes)

Example:
  python SAP/scripts/check_abc_meshes.py --mesh_dir D:/Datasets/2024_abc_train/03_meshes
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, total=None, desc="", **kwargs):
        n = 0
        for x in it:
            n += 1
            if n % 200 == 0 or (total is not None and n == total):
                print("%s %d/%s" % (desc, n, total if total is not None else "?"), flush=True)
            yield x


def _parse_header(f):
    header = []
    while True:
        line = f.readline()
        if not line:
            raise RuntimeError("truncated ply header")
        text = line.decode("ascii", errors="replace").strip()
        header.append(text)
        if text == "end_header":
            break
    fmt = None
    n_vert = n_face = 0
    vert_props = []
    face_count_ty = "u1"
    face_idx_ty = "i4"
    section = None
    dtype_map = {
        "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
        "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
        "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
        "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    }
    for text in header:
        if text.startswith("format "):
            fmt = text.split()[1]
        elif text.startswith("element vertex "):
            n_vert = int(text.split()[-1])
            section = "vertex"
            vert_props = []
        elif text.startswith("element face "):
            n_face = int(text.split()[-1])
            section = "face"
        elif text.startswith("element "):
            section = None
        elif text.startswith("property ") and section == "vertex":
            parts = text.split()
            vert_props.append((parts[-1], dtype_map[parts[1]]))
        elif text.startswith("property list ") and section == "face":
            parts = text.split()
            face_count_ty = dtype_map[parts[2]]
            face_idx_ty = dtype_map[parts[3]]
    return fmt, n_vert, n_face, vert_props, face_count_ty, face_idx_ty


def _read_ply_mesh(path):
    with open(path, "rb") as f:
        fmt, n_vert, n_face, vert_props, count_ty, idx_ty = _parse_header(f)
        endian = "<" if fmt != "binary_big_endian" else ">"
        if fmt == "ascii":
            verts = np.empty((n_vert, 3), dtype=np.float64)
            names = [n for n, _ in vert_props]
            ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
            for i in range(n_vert):
                parts = f.readline().decode("ascii", errors="replace").split()
                verts[i] = (float(parts[ix]), float(parts[iy]), float(parts[iz]))
            faces = []
            for _ in range(n_face):
                parts = f.readline().decode("ascii", errors="replace").split()
                n = int(parts[0])
                faces.append(tuple(int(x) for x in parts[1:1 + n]))
            return verts, faces

        dt = np.dtype([(name, endian + ty) for name, ty in vert_props])
        raw = np.fromfile(f, dtype=dt, count=n_vert)
        verts = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64, copy=False)
        count_size = np.dtype(count_ty).itemsize
        idx_dt = np.dtype(endian + idx_ty)
        faces = []
        blob = f.read()
        off = 0
        for _ in range(n_face):
            n = int(np.frombuffer(blob, dtype=endian + count_ty, count=1, offset=off)[0])
            off += count_size
            idx = np.frombuffer(blob, dtype=idx_dt, count=n, offset=off)
            off += n * idx_dt.itemsize
            faces.append(tuple(int(x) for x in idx))
        return verts, faces


def inspect_mesh(path):
    verts, faces = _read_ply_mesh(path)
    n_vert = int(verts.shape[0])
    n_face = len(faces)
    n_non_tri = 0
    n_degen = 0
    edge_count = defaultdict(int)
    edge_wind = defaultdict(int)

    vol6 = 0.0
    for face in faces:
        if len(face) != 3:
            n_non_tri += 1
            continue
        i, j, k = face
        if i == j or j == k or k == i:
            n_degen += 1
            continue
        if min(i, j, k) < 0 or max(i, j, k) >= n_vert:
            n_degen += 1
            continue
        v0, v1, v2 = verts[i], verts[j], verts[k]
        vol6 += float(np.dot(v0, np.cross(v1, v2)))
        for a, b in ((i, j), (j, k), (k, i)):
            u, v = (a, b) if a < b else (b, a)
            wind = 1 if a < b else -1
            edge_count[(u, v)] += 1
            edge_wind[(u, v)] += wind

    n_open = 0
    n_nonman = 0
    n_inconsist = 0
    for e, c in edge_count.items():
        if c == 1:
            n_open += 1
        elif c >= 3:
            n_nonman += 1
        elif c == 2 and edge_wind[e] != 0:
            n_inconsist += 1

    closed = n_open == 0 and n_nonman == 0 and n_non_tri == 0
    consistent = n_inconsist == 0
    volume = vol6 / 6.0
    if closed and consistent:
        orient = "outward" if volume >= 0 else "inward"
    elif not consistent:
        orient = "mixed_winding"
    else:
        orient = "open_or_nonmanifold"

    return {
        "file": os.path.splitext(os.path.basename(path))[0],
        "n_vert": n_vert,
        "n_face": n_face,
        "n_non_tri": n_non_tri,
        "n_degen": n_degen,
        "n_open": n_open,
        "n_nonman": n_nonman,
        "n_inconsist": n_inconsist,
        "closed": int(closed),
        "consistent": int(consistent),
        "volume": volume,
        "orient": orient,
    }


def _scan_one(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        rec = inspect_mesh(path)
        return ("ok", rec)
    except Exception as exc:
        return ("fail", stem, str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", default=r"D:\Datasets\2024_abc_train\03_meshes")
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    mesh_dir = os.path.expanduser(args.mesh_dir)
    files = [
        os.path.join(mesh_dir, name)
        for name in sorted(os.listdir(mesh_dir))
        if name.lower().endswith((".ply", ".obj", ".off", ".stl"))
    ]
    out_csv = args.out_csv or os.path.join(os.path.dirname(mesh_dir), "mesh_quality.csv")

    rows = []
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_scan_one, p) for p in files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="ABC mesh check", unit="file"):
            rec = fut.result()
            if rec[0] == "ok":
                rows.append(rec[1])
            else:
                failed.append((rec[1], rec[2]))

    n = len(rows)
    n_open = sum(1 for r in rows if r["n_open"] > 0)
    n_nonman = sum(1 for r in rows if r["n_nonman"] > 0)
    n_inconsist = sum(1 for r in rows if r["n_inconsist"] > 0)
    n_closed = sum(1 for r in rows if r["closed"])
    n_out = sum(1 for r in rows if r["orient"] == "outward")
    n_in = sum(1 for r in rows if r["orient"] == "inward")
    n_mix = sum(1 for r in rows if r["orient"] == "mixed_winding")
    n_other = sum(1 for r in rows if r["orient"] == "open_or_nonmanifold")

    print("=" * 64)
    print("mesh_dir=%s  n=%d  failed=%d" % (mesh_dir, n, len(failed)))
    print("open boundary          : %d / %d  (%.2f%%)" % (n_open, n, 100.0 * n_open / max(n, 1)))
    print("non-manifold edges     : %d / %d  (%.2f%%)" % (n_nonman, n, 100.0 * n_nonman / max(n, 1)))
    print("inconsistent winding   : %d / %d  (%.2f%%)" % (n_inconsist, n, 100.0 * n_inconsist / max(n, 1)))
    print("closed manifold        : %d / %d  (%.2f%%)" % (n_closed, n, 100.0 * n_closed / max(n, 1)))
    print("normals outward        : %d" % n_out)
    print("normals inward         : %d" % n_in)
    print("mixed winding          : %d" % n_mix)
    print("open/nonmanifold (no sign): %d" % n_other)

    def _top(key, k=8):
        bad = [r for r in rows if r[key] > 0]
        bad.sort(key=lambda r: r[key], reverse=True)
        return bad[:k]

    print("-" * 64)
    print("worst open-boundary:")
    for r in _top("n_open"):
        print("  %6d open  %s" % (r["n_open"], r["file"]))
    print("worst non-manifold:")
    for r in _top("n_nonman"):
        print("  %6d nonman %s" % (r["n_nonman"], r["file"]))
    print("worst inconsistent winding:")
    for r in _top("n_inconsist"):
        print("  %6d wind   %s" % (r["n_inconsist"], r["file"]))
    print("inward-normal closed meshes:")
    inward = [r for r in rows if r["orient"] == "inward"]
    inward.sort(key=lambda r: r["volume"])
    for r in inward[:8]:
        print("  vol=%.6f  %s" % (r["volume"], r["file"]))
    if not inward:
        print("  (none)")

    fields = [
        "file", "n_vert", "n_face", "n_non_tri", "n_degen",
        "n_open", "n_nonman", "n_inconsist", "closed", "consistent",
        "volume", "orient",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["file"]):
            w.writerow(r)
    print("csv: %s" % out_csv)
    if failed[:5]:
        print("first fails: %s" % failed[:5])


if __name__ == "__main__":
    main()
