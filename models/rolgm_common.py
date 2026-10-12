# RoLGM shared utilities: coordinate serialization and kNN indexing.

from __future__ import annotations

import torch

# axis index: 0=x, 1=y, 2=z — lexicographic primary → secondary → tertiary
SERIAL_AXIS_ORDERS = {
    "xyz": (0, 1, 2),
    "xzy": (0, 2, 1),
    "yxz": (1, 0, 2),
    "yzx": (1, 2, 0),
    "zxy": (2, 0, 1),
    "zyx": (2, 1, 0),
}
DEFAULT_SERIAL_ORDERS = ("xyz", "yzx", "zxy")


def knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Euclidean kNN indices. xyz/new_xyz: (B, N, 3) -> (B, S, nsample)."""
    dist = torch.cdist(new_xyz.float(), xyz.float())
    _, idx = torch.topk(dist, nsample, dim=-1, largest=False, sorted=True)
    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index. points: (B, N, C), idx: (B, S) or (B, S, K)."""
    device = points.device
    b = points.shape[0]
    if idx.dim() == 2:
        batch = torch.arange(b, device=device).view(b, 1).expand_as(idx)
        return points[batch, idx, :]
    batch = torch.arange(b, device=device).view(b, 1, 1).expand_as(idx)
    return points[batch, idx, :]


def _serial_order_names(names):
    if names is None:
        return list(DEFAULT_SERIAL_ORDERS)
    return [str(n).lower() for n in names]


def _parse_serial_orders(names) -> tuple[tuple[int, ...], ...]:
    if names is None:
        return tuple(SERIAL_AXIS_ORDERS[n] for n in DEFAULT_SERIAL_ORDERS)
    orders = []
    for name in names:
        key = str(name).lower()
        if key not in SERIAL_AXIS_ORDERS:
            raise ValueError(f"Unknown serialization order '{name}', choose from {list(SERIAL_AXIS_ORDERS)}")
        orders.append(SERIAL_AXIS_ORDERS[key])
    return tuple(orders)


def _lexsort_indices(xyz: torch.Tensor, axis_order: tuple[int, ...]) -> torch.Tensor:
    """Lexicographic sort indices for point tokens."""
    b, n, _ = xyz.shape
    device = xyz.device
    sort_idx = torch.arange(n, device=device, dtype=torch.long).unsqueeze(0).expand(b, n)
    batch = torch.arange(b, device=device).unsqueeze(1)
    for axis in reversed(axis_order):
        key = xyz[batch, sort_idx, axis]
        rel = key.argsort(dim=1, stable=True)
        sort_idx = sort_idx.gather(1, rel)
    return sort_idx


def _gather_sorted(xyz: torch.Tensor, feat: torch.Tensor, axis_order: tuple[int, ...]):
    """Sort tokens by lexicographic coordinate order."""
    sort_idx = _lexsort_indices(xyz, axis_order)
    idx_xyz = sort_idx.unsqueeze(-1).expand(-1, -1, 3)
    xyz_s = torch.gather(xyz, 1, idx_xyz)
    idx_c = sort_idx.unsqueeze(-1).expand(-1, -1, feat.size(-1))
    feat_s = torch.gather(feat, 1, idx_c)
    return xyz_s, feat_s, sort_idx


def _scatter_sorted(feat_s: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
    inv = sort_idx.argsort(dim=1)
    idx_c = inv.unsqueeze(-1).expand(-1, -1, feat_s.size(-1))
    return torch.gather(feat_s, 1, idx_c)
