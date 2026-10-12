# RoLGM: Reordered Local-Global Mamba (Vim BiMamba v2) (mamba-1p1p1 fork: bimamba_type / if_divide_out on Mamba).
# Layer stack follows Vim models_mamba.Block (RMSNorm/LN + BiMamba mixer + residual).
# Requires: pip install -e ./extensions/mamba-1p1p1 (see README)

from __future__ import annotations

import inspect
import random
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from timm.models.layers import DropPath
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from models.rolgm_common import (
    _gather_sorted,
    _parse_serial_orders,
    _scatter_sorted,
    _serial_order_names,
    index_points,
    knn_point,
)


def _check_vim_mamba_installed() -> None:
    params = inspect.signature(Mamba.__init__).parameters
    if "bimamba_type" not in params:
        raise ImportError(
            "Vim BiMamba requires extensions/mamba-1p1p1 (editable install). "
            "See README Environment section. "
            "Stock pip mamba-ssm does not support bimamba_type."
        )


def _mamba_kwargs_from_cfg(items: dict, ssm_cfg=None) -> dict:
    """Build kwargs for Vim Mamba; strip RoLGM-only keys."""
    kw = {}
    if ssm_cfg is not None and hasattr(ssm_cfg, "items"):
        kw.update(dict(ssm_cfg))
    elif isinstance(ssm_cfg, dict):
        kw.update(ssm_cfg)
    for key in (
        "bimamba_type",
        "if_divide_out",
        "d_state",
        "d_conv",
        "expand",
        "dt_rank",
        "use_fast_path",
        "init_layer_scale",
    ):
        if key in items and key not in kw:
            kw[key] = items[key]
    kw.setdefault("bimamba_type", "v2")
    kw.setdefault("if_divide_out", True)
    return kw


def _norm_cls(rms_norm: bool):
    if rms_norm and RMSNorm is not None:
        return partial(RMSNorm, eps=1e-5)
    return partial(nn.LayerNorm, eps=1e-5)


class VimBiMambaBlock(nn.Module):
    """Vim models_mamba.Block: Add→Norm→BiMamba mixer (returns mixer out + residual branch)."""

    def __init__(
        self,
        dim: int,
        mamba_kwargs=None,
        drop_path: float = 0.0,
        rms_norm: bool = True,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
    ):
        super().__init__()
        _check_vim_mamba_installed()
        if mamba_kwargs is None:
            mamba_kwargs = {}
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = Mamba(d_model=dim, **mamba_kwargs)
        self.norm = _norm_cls(rms_norm)(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.bimamba_type = str(mamba_kwargs.get("bimamba_type", "v2"))
        if fused_add_norm:
            if RMSNorm is None or rms_norm_fn is None:
                raise ImportError("fused_add_norm requires mamba_ssm Triton RMSNorm kernels.")
            if not isinstance(self.norm, (nn.LayerNorm, RMSNorm)):
                raise TypeError("VimBiMambaBlock fused_add_norm supports LayerNorm/RMSNorm only.")

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual


class VimBiMambaStack(nn.Module):
    """Stack of Vim blocks + final residual fusion (VisionMamba norm_f when enabled)."""

    def __init__(
        self,
        dim: int,
        depth: int = 1,
        mamba_kwargs=None,
        drop_path_rate=0.0,
        rms_norm: bool = True,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        final_norm: bool = True,
    ):
        super().__init__()
        if isinstance(drop_path_rate, float):
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            dpr = list(drop_path_rate)
        self.blocks = nn.ModuleList(
            [
                VimBiMambaBlock(
                    dim,
                    mamba_kwargs=mamba_kwargs,
                    drop_path=dpr[i],
                    rms_norm=rms_norm,
                    fused_add_norm=fused_add_norm,
                    residual_in_fp32=residual_in_fp32,
                )
                for i in range(depth)
            ]
        )
        self.drop_path = DropPath(dpr[-1]) if dpr[-1] > 0.0 else nn.Identity()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if final_norm:
            self.norm_f = _norm_cls(rms_norm)(dim)
        else:
            self.norm_f = None

    def forward(self, x: Tensor, inference_params=None) -> Tensor:
        hidden_states = x
        residual = None
        for layer in self.blocks:
            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            if self.norm_f is None:
                return residual.to(dtype=x.dtype)
            return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

        if self.norm_f is None:
            return hidden_states
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        return fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
            eps=self.norm_f.eps,
        )


class RoLGMLocalBranch(nn.Module):
    def __init__(
        self,
        dim: int,
        k: int,
        depth: int = 1,
        mamba_kwargs=None,
        drop_path_rate: float = 0.0,
        local_agg: str = "max",
        rms_norm: bool = True,
        fused_add_norm: bool = False,
        final_norm: bool = True,
    ):
        super().__init__()
        self.k = k
        self.local_agg = local_agg
        self.pos_mlp = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.vim_stack = VimBiMambaStack(
            dim,
            depth=depth,
            mamba_kwargs=mamba_kwargs,
            drop_path_rate=drop_path_rate,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            final_norm=final_norm,
        )
        if local_agg == "both":
            self.agg_fuse = nn.Linear(dim * 2, dim)
        else:
            self.agg_fuse = None
        if local_agg == "attn":
            hidden = max(dim // 4, 8)
            self.local_score = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.local_score = None

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        b, m, c = feat.shape
        idx = knn_point(self.k, xyz, xyz)
        neigh = index_points(feat, idx)
        neigh_xyz = index_points(xyz, idx)
        rel = neigh_xyz - xyz.unsqueeze(2)
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rel_embed = self.pos_mlp(torch.cat([rel, dist], dim=-1))
        neigh = neigh + rel_embed

        seq = neigh.reshape(b * m, self.k, c)
        seq = self.vim_stack(seq)
        if self.local_agg == "max":
            out = seq.max(dim=1)[0]
        elif self.local_agg == "mean":
            out = seq.mean(dim=1)
        elif self.local_agg == "both":
            out = self.agg_fuse(torch.cat([seq.mean(dim=1), seq.max(dim=1)[0]], dim=-1))
        elif self.local_agg == "attn":
            score = self.local_score(seq)
            weight = torch.softmax(score, dim=1)
            out = (seq * weight).sum(dim=1)
        else:
            raise ValueError(f"local_agg must be mean|max|both|attn, got {self.local_agg}")
        return out.view(b, m, c)


class RoLGMGlobalBranch(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 1,
        mamba_kwargs=None,
        drop_path_rate: float = 0.0,
        mode: str = "prepend_token",
        rms_norm: bool = True,
        fused_add_norm: bool = False,
        final_norm: bool = True,
    ):
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.fuse = nn.Linear(dim * 2, dim)
        else:
            self.fuse = None
        self.vim_stack = VimBiMambaStack(
            dim,
            depth=depth,
            mamba_kwargs=mamba_kwargs,
            drop_path_rate=drop_path_rate,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            final_norm=final_norm,
        )

    def forward(self, gf: torch.Tensor, feat_lm: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            m = feat_lm.shape[1]
            g = gf.unsqueeze(1).expand(-1, m, -1)
            h = self.fuse(torch.cat([g, feat_lm], dim=-1))
            return self.vim_stack(h)

        seq = torch.cat([gf.unsqueeze(1), feat_lm], dim=1)
        seq = self.vim_stack(seq)
        return seq[:, 1:, :]


class RoLGMBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        knn_k: int,
        lm_depth: int = 1,
        sm_depth: int = 1,
        mamba_kwargs=None,
        drop_path: float = 0.0,
        residual_scale: float = 1.0,
        shape_branch_mode: str = "prepend_token",
        local_agg: str = "max",
        use_local_branch: bool = True,
        rms_norm: bool = True,
        fused_add_norm: bool = False,
        vim_final_norm: bool = True,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.use_local_branch = use_local_branch and lm_depth > 0
        vim_kw = dict(
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            final_norm=vim_final_norm,
        )
        if self.use_local_branch:
            self.local_branch = RoLGMLocalBranch(
                dim,
                k=knn_k,
                depth=lm_depth,
                mamba_kwargs=mamba_kwargs,
                drop_path_rate=drop_path,
                local_agg=local_agg,
                **vim_kw,
            )
        else:
            self.local_branch = None
        self.shape_branch = RoLGMGlobalBranch(
            dim,
            depth=sm_depth,
            mamba_kwargs=mamba_kwargs,
            drop_path_rate=drop_path,
            mode=shape_branch_mode,
            **vim_kw,
        )

    def forward(self, xyz_sorted: torch.Tensor, feat_sorted: torch.Tensor, gf: torch.Tensor) -> torch.Tensor:
        if self.local_branch is not None:
            flm = self.local_branch(xyz_sorted, feat_sorted)
        else:
            flm = feat_sorted
        out = self.shape_branch(gf, flm)
        return feat_sorted + self.residual_scale * out


class RoLGM(nn.Module):
    """
    Paper RoLGM module (Vim BiMamba v2) v2 inside each layer (requires mamba-1p1p1).
    Same serialization API as RoLGMClassic in rolgm_common.
    """

    def __init__(
        self,
        in_channel: int,
        dim: int = 64,
        knn_k: int = 12,
        ls_mamba_cycles: int = 1,
        ls_mamba_per_axis: int = 1,
        serialization_orders=None,
        lm_depth: int = 1,
        sm_depth: int = 1,
        drop_path: float = 0.0,
        residual_scale: float = 1.0,
        shape_branch_mode: str = "prepend_token",
        local_agg: str = "max",
        mamba_kwargs=None,
        share_serial_blocks: bool = True,
        use_local_branch: bool = True,
        random_serial_one: bool = False,
        eval_serial_one: bool = False,
        eval_serial_order: str = "xyz",
        bimamba_type: str = "v2",
        if_divide_out: bool = True,
        rms_norm: bool = True,
        fused_add_norm: bool = False,
        vim_final_norm: bool = True,
    ):
        super().__init__()
        _check_vim_mamba_installed()
        if mamba_kwargs is None:
            mamba_kwargs = {}
        mk = dict(mamba_kwargs)
        mk.setdefault("bimamba_type", bimamba_type)
        mk.setdefault("if_divide_out", if_divide_out)

        self.dim = dim
        self.bimamba_type = str(mk.get("bimamba_type", "v2"))
        self.if_divide_out = bool(mk.get("if_divide_out", True))
        self.rms_norm = bool(rms_norm)
        self.use_local_branch = use_local_branch and lm_depth > 0
        self.random_serial_one = bool(random_serial_one)
        self.eval_serial_one = bool(eval_serial_one)
        self.eval_serial_order = str(eval_serial_order).lower()
        self.serial_order_names = _serial_order_names(serialization_orders)
        self.ls_mamba_cycles = ls_mamba_cycles
        self.serial_axis_orders = _parse_serial_orders(serialization_orders)
        self.share_serial_blocks = share_serial_blocks
        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

        def _make_block_stack():
            return nn.ModuleList(
                [
                    RoLGMBlock(
                        dim,
                        knn_k=knn_k,
                        lm_depth=lm_depth,
                        sm_depth=sm_depth,
                        mamba_kwargs=mk,
                        drop_path=drop_path,
                        residual_scale=residual_scale,
                        shape_branch_mode=shape_branch_mode,
                        local_agg=local_agg,
                        use_local_branch=self.use_local_branch,
                        rms_norm=rms_norm,
                        fused_add_norm=fused_add_norm,
                        vim_final_norm=vim_final_norm,
                    )
                    for _ in range(ls_mamba_per_axis)
                ]
            )

        if share_serial_blocks:
            self.serial_block_stacks = _make_block_stack()
        else:
            self.serial_block_stacks = nn.ModuleList(
                [_make_block_stack() for _ in self.serial_axis_orders]
            )

    def _blocks_for_pass(self, pass_idx: int) -> nn.ModuleList:
        if self.share_serial_blocks:
            return self.serial_block_stacks
        return self.serial_block_stacks[pass_idx]

    def _iter_serial_passes(self):
        n = len(self.serial_axis_orders)
        if self.training and self.random_serial_one and n > 1:
            idx = random.randint(0, n - 1)
            yield idx, self.serial_axis_orders[idx]
            return
        if not self.training and self.eval_serial_one and n > 0:
            key = self.eval_serial_order
            idx = self.serial_order_names.index(key) if key in self.serial_order_names else 0
            yield idx, self.serial_axis_orders[idx]
            return
        for pass_idx, axis_order in enumerate(self.serial_axis_orders):
            yield pass_idx, axis_order

    def _run_ls_mamba_serial(self, xyz: torch.Tensor, feat: torch.Tensor, gf: torch.Tensor) -> torch.Tensor:
        h = feat
        for _ in range(self.ls_mamba_cycles):
            for pass_idx, axis_order in self._iter_serial_passes():
                blocks = self._blocks_for_pass(pass_idx)
                xyz_s, h_s, sort_idx = _gather_sorted(xyz, h, axis_order)
                for blk in blocks:
                    h_s = blk(xyz_s, h_s, gf)
                h = _scatter_sorted(h_s, sort_idx)
        return h

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        identity = x
        h = self.linear_start(x).permute(0, 2, 1).contiguous()
        xyz = pos.permute(0, 2, 1).contiguous()
        gf = h.max(dim=1)[0]
        h = self._run_ls_mamba_serial(xyz, h, gf)
        y = self.linear_end(h.permute(0, 2, 1).contiguous())
        return y + identity


def build_rolgm_from_cfg(
    in_channel: int,
    dim: int = 64,
    rolgm_cfg=None,
    lsm_cfg=None,
) -> RoLGM:
    cfg = rolgm_cfg if rolgm_cfg is not None else lsm_cfg
    if cfg is None:
        items = {}
    elif hasattr(cfg, "items"):
        items = dict(cfg)
    else:
        items = {}
    ssm_cfg = items.get("ssm_cfg", None)
    if ssm_cfg is not None and hasattr(ssm_cfg, "items"):
        ssm_cfg = dict(ssm_cfg)
    lm_depth = int(items.get("lm_depth", 1))
    if "use_local_branch" in items:
        use_local_branch = bool(items.get("use_local_branch"))
    else:
        use_local_branch = lm_depth > 0
    mamba_kwargs = _mamba_kwargs_from_cfg(items, ssm_cfg)
    return RoLGM(
        in_channel=in_channel,
        dim=dim,
        knn_k=int(items.get("knn_k", 12)),
        ls_mamba_cycles=int(items.get("rolgm_cycles", items.get("ls_mamba_cycles", 1))),
        ls_mamba_per_axis=int(items.get("rolgm_per_axis", items.get("ls_mamba_per_axis", 1))),
        serialization_orders=items.get("serialization_orders", None),
        lm_depth=lm_depth,
        sm_depth=int(items.get("sm_depth", 1)),
        drop_path=float(items.get("drop_path", 0.0)),
        residual_scale=float(items.get("residual_scale", 1.0)),
        shape_branch_mode=str(items.get("shape_branch_mode", "prepend_token")),
        local_agg=str(items.get("local_agg", "max")),
        mamba_kwargs=mamba_kwargs,
        share_serial_blocks=bool(items.get("share_serial_blocks", True)),
        use_local_branch=use_local_branch,
        random_serial_one=bool(items.get("random_serial_one", False)),
        eval_serial_one=bool(items.get("eval_serial_one", False)),
        eval_serial_order=str(items.get("eval_serial_order", "xyz")),
        bimamba_type=str(items.get("bimamba_type", mamba_kwargs.get("bimamba_type", "v2"))),
        if_divide_out=bool(items.get("if_divide_out", mamba_kwargs.get("if_divide_out", True))),
        rms_norm=bool(items.get("rms_norm", True)),
        fused_add_norm=bool(items.get("fused_add_norm", False)),
        vim_final_norm=bool(items.get("vim_final_norm", True)),
    )

