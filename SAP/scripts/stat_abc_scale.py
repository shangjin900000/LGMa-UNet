"""Scan ABC meshes/complete clouds to choose a shared SAP scale S.

P_SAP = P_ABC / S + 0.5
Want mapped coords in [0.05, 0.95] => S >= A / 0.45, A = max |P|.
"""

from __future__ import annotations

import argparse
import os
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


def _read_ply_xyz(path):
    """Read x,y,z from a PLY vertex list (ascii or binary). Ignores faces."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("truncated ply header: %s" % path)
            text = line.decode("ascii", errors="replace").strip()
            header.append(text)
            if text == "end_header":
                break

        fmt = None
        n_vert = 0
        props = []
        in_vertex = False
        for text in header:
            if text.startswith("format "):
                fmt = text.split()[1]
            elif text.startswith("element vertex "):
                n_vert = int(text.split()[-1])
                in_vertex = True
                props = []
            elif text.startswith("element "):
                in_vertex = False
            elif in_vertex and text.startswith("property "):
                parts = text.split()
                props.append(parts[-1] if parts[1] != "list" else None)

        if n_vert <= 0:
            return None
        names = [p for p in props if p is not None]
        try:
            ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
        except ValueError:
            raise RuntimeError("ply has no x/y/z: %s" % path)

        if fmt == "ascii":
            pts = np.empty((n_vert, 3), dtype=np.float64)
            for i in range(n_vert):
                parts = f.readline().decode("ascii", errors="replace").split()
                pts[i, 0] = float(parts[ix])
                pts[i, 1] = float(parts[iy])
                pts[i, 2] = float(parts[iz])
            return pts

        dtype_map = {
            "char": "i1", "uchar": "u1",
            "short": "i2", "ushort": "u2",
            "int": "i4", "uint": "u4",
            "float": "f4", "double": "f8",
        }
        fields = []
        in_vertex = False
        for text in header:
            if text.startswith("element vertex "):
                in_vertex = True
            elif text.startswith("element "):
                in_vertex = False
            elif in_vertex and text.startswith("property "):
                parts = text.split()
                if parts[1] == "list":
                    fields.append(("list", None))
                else:
                    fields.append((parts[-1], dtype_map[parts[1]]))

        endian = "<" if fmt == "binary_little_endian" else ">"
        dt = np.dtype([(name, endian + ty) for name, ty in fields if ty is not None])
        raw = np.fromfile(f, dtype=dt, count=n_vert)
        if raw.size != n_vert:
            raise RuntimeError("expected %d vertices, got %d: %s" % (n_vert, raw.size, path))
        pts = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64, copy=False)
        return pts


def _absmax_xyz(path):
    if path.lower().endswith(".npy"):
        pts = np.load(path)
    else:
        pts = _read_ply_xyz(path)
    if pts is None or pts.size == 0:
        return None
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    return float(np.max(np.abs(pts))), int(pts.shape[0]), pts.min(axis=0), pts.max(axis=0)


def _iter_files(folder, exts):
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(exts):
            yield os.path.join(folder, name), os.path.splitext(name)[0]


def _scan_one(item):
    path, stem, kind = item
    try:
        got = _absmax_xyz(path)
    except Exception as exc:
        return ("fail", stem, str(exc))
    if got is None:
        return ("fail", stem, "empty")
    a, n, mn, mx = got
    return ("ok", stem, a, n, mn, mx)


def scan_folder(folder, kind, exts, workers):
    files = [(path, stem, kind) for path, stem in _iter_files(folder, exts)]
    rows = []
    failed = []
    if workers <= 1:
        iterator = (_scan_one(it) for it in files)
        n_total = len(files)
        for rec in tqdm(iterator, total=n_total, desc=os.path.basename(folder), unit="file"):
            if rec[0] == "ok":
                _, stem, a, n, mn, mx = rec
                rows.append((stem, a, n, mn, mx))
            else:
                failed.append((rec[1], rec[2]))
        return rows, failed

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_scan_one, it) for it in files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=os.path.basename(folder), unit="file"):
            rec = fut.result()
            if rec[0] == "ok":
                _, stem, a, n, mn, mx = rec
                rows.append((stem, a, n, mn, mx))
            else:
                failed.append((rec[1], rec[2]))
    return rows, failed


def summarize(tag, rows):
    if not rows:
        print("[%s] empty" % tag)
        return 0.0
    absmax = np.array([r[1] for r in rows], dtype=np.float64)
    worst = rows[int(np.argmax(absmax))]
    print("=" * 60)
    print("[%s] n=%d" % (tag, len(rows)))
    print("  A=max|P|=%.6f  file=%s  n_pts=%d" % (worst[1], worst[0], worst[2]))
    print("  percentile |P|_max  50/90/95/99 = %.4f / %.4f / %.4f / %.4f" % tuple(
        np.percentile(absmax, [50, 90, 95, 99])
    ))
    for thr in (0.5, 0.6, 0.7, 0.8, 1.0):
        n = int((absmax > thr).sum())
        print("  |P|_max > %.1f : %d (%.2f%%)" % (thr, n, 100.0 * n / len(rows)))
    mn = np.stack([r[3] for r in rows], axis=0)
    mx = np.stack([r[4] for r in rows], axis=0)
    print("  global bbox min xyz = %s" % np.min(mn, axis=0))
    print("  global bbox max xyz = %s" % np.max(mx, axis=0))
    return float(absmax.max())


def pick_s(a, margin=0.05):
    # map [-A,A] -> [0.5 - A/S, 0.5 + A/S] subset of [margin, 1-margin]
    # A/S <= 0.5 - margin  => S >= A / (0.5 - margin)
    need = a / (0.5 - margin) if a > 0 else 0.0
    candidates = [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]
    chosen = None
    for s in candidates:
        if s + 1e-9 >= need:
            chosen = s
            break
    if chosen is None:
        chosen = float(np.ceil(need * 10.0) / 10.0)
    lo = 0.5 - a / chosen
    hi = 0.5 + a / chosen
    return need, chosen, lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=r"D:\Datasets\2024_abc_train")
    parser.add_argument("--mesh_dir", default="03_meshes")
    parser.add_argument("--complete_dirs", nargs="*", default=["complete", "complete_8192"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = os.path.expanduser(args.data_root)
    mesh_dir = args.mesh_dir if os.path.isabs(args.mesh_dir) else os.path.join(root, args.mesh_dir)

    a_all = 0.0
    per = {}

    mesh_rows, mesh_fail = scan_folder(mesh_dir, "mesh", (".ply", ".obj", ".off", ".stl"), args.workers)
    per["mesh"] = summarize("03_meshes", mesh_rows)
    a_all = max(a_all, per["mesh"])
    print("  failed=%d" % len(mesh_fail))
    if mesh_fail[:5]:
        print("  first fails: %s" % mesh_fail[:5])

    for sub in args.complete_dirs:
        d = sub if os.path.isabs(sub) else os.path.join(root, sub)
        if not os.path.isdir(d):
            print("[skip] missing %s" % d)
            continue
        rows, failed = scan_folder(d, "pcd", (".ply", ".npy"), args.workers)
        per[sub] = summarize(sub, rows)
        a_all = max(a_all, per[sub])
        print("  failed=%d" % len(failed))

    print("=" * 60)
    print("A (dataset) = %.6f" % a_all)
    for margin in (0.05, 0.10):
        need, chosen, lo, hi = pick_s(a_all, margin=margin)
        print(
            "target [%.2f, %.2f] => S >= %.4f ; pick S=%.2f "
            "=> mapped A to [%.4f, %.4f]"
            % (margin, 1.0 - margin, need, chosen, lo, hi)
        )


if __name__ == "__main__":
    main()
