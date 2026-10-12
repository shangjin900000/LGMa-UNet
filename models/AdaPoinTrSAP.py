"""AdaPoinTr completion + SAP reconstruction.

ABC: P_SAP = P_ABC / S + 0.5 with a dataset-wide S (sap_scale=2.0).
GT Poisson field is loaded from mesh-precomputed psr.npz.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from models.AdaPoinTr import AdaPoinTr
from models.build import MODELS
from SAP.src.dpsr import DPSR
from SAP.src.model import Encode2Points
from SAP.src.utils import export_mesh, mc_from_psr
from utils import misc
from utils.logger import print_log


def abc_to_sap(points, sap_scale):
    """P_SAP = P_ABC / S + 0.5."""
    return points / sap_scale + 0.5


def sap_to_abc(points, sap_scale):
    """P_ABC = S * (P_SAP - 0.5)."""
    return sap_scale * (points - 0.5)


def dmc_unit_box(points, min_v, max_v):
    """DMC: (p - min) / (max + 1 - min)."""
    scale = (max_v + 1.0 - min_v).clamp_min(1e-6)
    return (points - min_v) / scale


def dmc_box_from_ref(ref_points):
    """Per-sample scalar min/max over all xyz, matching DMC shell min/max."""
    min_v = ref_points.amin(dim=(1, 2), keepdim=True)
    max_v = ref_points.amax(dim=(1, 2), keepdim=True)
    return min_v, max_v


def _reshape_box(value, like):
    if value is None:
        return None
    value = value.to(device=like.device, dtype=like.dtype)
    if value.dim() == 0:
        value = value.view(1, 1, 1).expand(like.size(0), 1, 1)
    elif value.dim() == 1:
        value = value.view(-1, 1, 1)
    return value


@MODELS.register_module()
class AdaPoinTrSAP(nn.Module):
    """PoinTr-in-DMC replaced by AdaPoinTr; SAP head follows DMC Encode2Points + DPSR."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.completion = AdaPoinTr(config, **kwargs)
        self.w_psr = float(getattr(config, "w_psr", 1.0))
        self.psr_tanh = bool(getattr(config, "psr_tanh", True))
        # 0 = DMC: send the full AdaPoinTr dense cloud into Encode2Points/DPSR
        self.sap_n_points = int(getattr(config, "sap_n_points", 0))
        self.export_mesh = bool(getattr(config, "export_sap_mesh", True))
        self.sap_scale = float(getattr(config, "sap_scale", 0.0) or 0.0)

        sap_cfg = str(getattr(config, "sap_config", "SAP/configs/learning_based/noise_small/ours.yaml"))
        grid_res = int(getattr(config, "psr_grid_res", 128))
        psr_sigma = float(getattr(config, "psr_sigma", 2))

        self.model_sap = Encode2Points(sap_cfg)
        self.dpsr = DPSR(res=(grid_res, grid_res, grid_res), sig=psr_sigma)
        box_msg = f"sap_scale={self.sap_scale}" if self.sap_scale > 0 else "DMC min/max box"
        print_log(
            f"AdaPoinTrSAP: sap_cfg={sap_cfg} grid={grid_res}^3 sigma={psr_sigma} "
            f"sap_n={'full' if self.sap_n_points <= 0 else self.sap_n_points} "
            f"w_psr={self.w_psr} {box_msg}",
            logger="AdaPoinTrSAP",
        )

        sap_ckpt = getattr(config, "sap_ckpt", None)
        if sap_ckpt:
            self._load_sap_ckpt(str(sap_ckpt))

        comp_ckpt = getattr(config, "completion_ckpt", None)
        if comp_ckpt:
            self._load_completion_ckpt(str(comp_ckpt))

        if bool(getattr(config, "freeze_sap", False)):
            for p in self.model_sap.parameters():
                p.requires_grad = False
            print_log("AdaPoinTrSAP: Encode2Points frozen", logger="AdaPoinTrSAP")

        self.psr_crit = nn.MSELoss()

    def _load_sap_ckpt(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"sap_ckpt not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        msg = self.model_sap.load_state_dict(state, strict=False)
        print_log(
            f"AdaPoinTrSAP loaded sap_ckpt missing={len(msg.missing_keys)} "
            f"unexpected={len(msg.unexpected_keys)}",
            logger="AdaPoinTrSAP",
        )

    def _load_completion_ckpt(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"completion_ckpt not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            if "base_model" in state:
                state = state["base_model"]
            elif "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        if any(k.startswith("completion.") for k in state):
            msg = self.load_state_dict(state, strict=False)
        else:
            msg = self.completion.load_state_dict(state, strict=False)
        print_log(
            f"AdaPoinTrSAP loaded completion_ckpt missing={len(msg.missing_keys)} "
            f"unexpected={len(msg.unexpected_keys)}",
            logger="AdaPoinTrSAP",
        )

    def _maybe_fps(self, pts):
        n = self.sap_n_points
        if n <= 0 or pts.size(1) <= n:
            return pts
        return misc.fps(pts, n)

    def _sap_reconstruct(self, dense_points, min_v=None, max_v=None, ref_points=None):
        if self.sap_scale > 0:
            pred_unit = abc_to_sap(dense_points, self.sap_scale)
            min_v = dense_points.new_tensor(self.sap_scale)
            max_v = dense_points.new_tensor(self.sap_scale)
        else:
            if min_v is None or max_v is None:
                ref = ref_points if ref_points is not None else dense_points
                min_v, max_v = dmc_box_from_ref(ref)
            else:
                min_v = _reshape_box(min_v, dense_points)
                max_v = _reshape_box(max_v, dense_points)
            pred_unit = dmc_unit_box(dense_points, min_v, max_v)
        pred_unit = self._maybe_fps(pred_unit)
        sap_points, sap_normals = self.model_sap(pred_unit)
        psr_grid = self.dpsr(sap_points, sap_normals)
        return sap_points, sap_normals, psr_grid, min_v, max_v

    def get_loss(self, ret, gt, epoch=1):
        pred_coarse, denoised_coarse, denoised_fine, pred_fine, psr_grid, gt_psr = ret[:6]
        loss_denoised, loss_recon = self.completion.get_loss(
            (pred_coarse, denoised_coarse, denoised_fine, pred_fine), gt, epoch
        )
        if self.w_psr != 0 and gt_psr is None:
            raise RuntimeError(
                "AdaPoinTrSAP w_psr!=0 but gt_psr is missing. "
                "Load a mesh-precomputed Poisson field (psr.npz); do not estimate it from GT points."
            )
        loss_psr = pred_fine.new_zeros(())
        if gt_psr is not None and self.w_psr != 0:
            pred_f = torch.tanh(psr_grid) if self.psr_tanh else psr_grid
            gt_f = torch.tanh(gt_psr) if self.psr_tanh else gt_psr
            loss_psr = self.w_psr * self.psr_crit(pred_f, gt_f)
        loss_all = loss_denoised + loss_recon + loss_psr
        return loss_all, loss_recon, loss_psr

    def forward(self, xyz, gt=None, gt_psr=None, min_v=None, max_v=None):
        if self.training and self.w_psr != 0 and gt_psr is None:
            raise RuntimeError(
                "AdaPoinTrSAP training needs precomputed gt_psr from the mesh DPSR script."
            )

        comp_ret = self.completion(xyz)
        if self.training:
            pred_coarse, denoised_coarse, denoised_fine, pred_fine = comp_ret
            dense = pred_fine
        else:
            pred_coarse, pred_fine = comp_ret
            dense = pred_fine

        sap_points, sap_normals, psr_grid, min_v, max_v = self._sap_reconstruct(
            dense, min_v=min_v, max_v=max_v, ref_points=gt,
        )

        if self.training:
            return (
                pred_coarse,
                denoised_coarse,
                denoised_fine,
                pred_fine,
                psr_grid,
                gt_psr,
                sap_points,
                min_v,
                max_v,
            )
        return pred_coarse, pred_fine, sap_points, sap_normals, psr_grid, min_v, max_v


def sap_mesh_from_psr(psr_grid, min_v=None, max_v=None, sap_scale=None, zero_level=0.0):
    verts, faces, _ = mc_from_psr(psr_grid, zero_level=zero_level)
    if sap_scale is not None and float(sap_scale) > 0:
        verts = sap_to_abc(verts, float(sap_scale))
        return verts, faces
    min_np = min_v.detach().cpu().numpy()
    max_np = max_v.detach().cpu().numpy()
    while min_np.ndim > 0 and min_np.size == 1:
        min_np = min_np.reshape(-1)[0]
        max_np = max_np.reshape(-1)[0]
    verts = verts * (max_np + 1.0 - min_np) + min_np
    return verts, faces


def save_sap_mesh(path, psr_grid, min_v=None, max_v=None, sap_scale=None, zero_level=0.0):
    verts, faces = sap_mesh_from_psr(
        psr_grid[:1], min_v=None if min_v is None else min_v[:1],
        max_v=None if max_v is None else max_v[:1],
        sap_scale=sap_scale, zero_level=zero_level,
    )
    export_mesh(path, verts, faces)
