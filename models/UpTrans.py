#! /usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from extensions.chamfer_dist import ChamferDistanceL2, ChamferDistanceL1

from models.adapointr_dense import AdaPoinTrFCDecoder
from models.pointnet import build_pointnet_sa_module, MLP_Res, UpLayer
from models.rolgm import build_rolgm_from_cfg
from models.unet_config import (
    parse_encoder_stages,
    parse_decoder_stages,
    summarize_stages,
    _sa_mlp_dims,
    parse_sa_version,
    parse_sa_geo_cfg,
    parse_sa_pool_mode,
    _cfg_bool,
    _cfg_int,
    _cfg_block,
)

from utils.logger import print_log

from .build import MODELS


class UNet(nn.Module):
    """Config-driven encoder (Geo-SA + RoLGM) and decoder (upconv + DecRoLGM + skip)."""

    def __init__(
        self,
        config=None,
        dim_feat=384,
        num_seeds=1024,
        seed_fea=128,
        rolgm_cfg=None,
        rolgm_dim=64,
        encoder_stages=None,
        decoder_stages=None,
    ):
        super(UNet, self).__init__()
        if config is not None:
            dim_feat = int(getattr(config, "dim_feat", dim_feat))
            num_seeds = int(getattr(config, "num_seeds", num_seeds))
            seed_fea = int(getattr(config, "seed_fea", seed_fea))
            rolgm_cfg = _cfg_block(config, "rolgm", "lsm", rolgm_cfg)
            rolgm_dim = _cfg_int(config, "rolgm_dim", "lsm_dim", rolgm_dim)
            self.enc_serial_type = str(getattr(config, "enc_serial_type", "rolgm")).lower()
            self.use_rolgm = _cfg_bool(config, "use_rolgm", "use_lsm", True)
            self.rolgm_cfg = rolgm_cfg
            self.rolgm_dim = rolgm_dim
        else:

            class _LegacyCfg:
                pass

            config = _LegacyCfg()
            config.num_seeds = num_seeds
            config.seed_fea = seed_fea
            config.dim_feat = dim_feat
            self.enc_serial_type = "rolgm"
            self.use_rolgm = True
            self.rolgm_cfg = None
            self.rolgm_dim = 64

        if encoder_stages is None:
            encoder_stages = parse_encoder_stages(config)
        else:
            encoder_stages = list(encoder_stages)

        if decoder_stages is None:
            decoder_stages = parse_decoder_stages(config, encoder_stages)
        else:
            decoder_stages = list(decoder_stages)

        self.encoder_stages = encoder_stages
        self.decoder_stages = decoder_stages
        self.seed_fea = seed_fea
        self.num_seeds = num_seeds
        self.dec_fea_refine_type = str(getattr(config, "dec_fea_refine_type", "dec_rolgm")).lower()
        self.use_dec_rolgm = self.dec_fea_refine_type in (
            "dec_rolgm", "rolgm", "lsm", "v5", "v6", "v6-4"
        )
        if not self.use_dec_rolgm:
            raise ValueError(
                f"dec_fea_refine_type must be 'dec_rolgm' (paper setup), got {self.dec_fea_refine_type!r}."
            )
        dec_rolgm_raw = _cfg_block(config, "dec_rolgm", "dec_lsm", None)
        if dec_rolgm_raw is not None and hasattr(dec_rolgm_raw, "items"):
            self.dec_rolgm_cfg = dict(dec_rolgm_raw)
        elif self.rolgm_cfg is not None and hasattr(self.rolgm_cfg, "items"):
            self.dec_rolgm_cfg = dict(self.rolgm_cfg)
        else:
            self.dec_rolgm_cfg = {}
        self.dec_rolgm_dim = _cfg_int(config, "dec_rolgm_dim", "dec_lsm_dim", self.rolgm_dim)
        self.dec_pos_hidden = int(getattr(config, "dec_pos_hidden", 32))
        dec_rolgm_stages_raw = _cfg_block(config, "dec_rolgm_stages", "dec_lsm_stages", None)
        if dec_rolgm_stages_raw is None:
            self.dec_rolgm_stage_ids = None
        else:
            self.dec_rolgm_stage_ids = {int(x) for x in dec_rolgm_stages_raw}

        self.enc_sa = nn.ModuleList()
        self.enc_rolgm = nn.ModuleList()
        prev_ch = 3
        for i, st in enumerate(encoder_stages):
            feat_ch = int(st["feat_ch"])
            group_all = bool(st.get("group_all", False))
            mlp = st.get("mlp")
            if mlp is None:
                mlp = _sa_mlp_dims(
                    prev_ch, feat_ch, first_from_xyz=(i == 0), group_all=group_all
                )
            else:
                mlp = list(mlp)

            sa_ver = parse_sa_version(config, st)
            sa_geo = parse_sa_geo_cfg(config, st)
            sa_pool = parse_sa_pool_mode(config, st)

            if group_all:
                sa = build_pointnet_sa_module(
                    None,
                    None,
                    prev_ch,
                    mlp,
                    group_all=True,
                    if_bn=False,
                    if_idx=False,
                    sa_version=sa_ver,
                    sa_geo_cfg=sa_geo,
                    pool_mode=sa_pool,
                )
                rolgm = None
            else:
                sa = build_pointnet_sa_module(
                    int(st["npoints"]),
                    int(st.get("knn", 16)),
                    prev_ch,
                    mlp,
                    group_all=False,
                    if_bn=False,
                    if_idx=True,
                    sa_version=sa_ver,
                    sa_geo_cfg=sa_geo,
                    pool_mode=sa_pool,
                )
                rolgm = self._build_serial_block(feat_ch, st)
            self.enc_sa.append(sa)
            self.enc_rolgm.append(rolgm)
            prev_ch = feat_ch

        self.bottom_ch = prev_ch
        self.dec_up = nn.ModuleList()
        self.dec_fuse = nn.ModuleList()
        self.dec_pos_heads = nn.ModuleList()
        self.dec_rolgm = nn.ModuleList()
        dec_in_ch = self.bottom_ch
        n_dec = len(decoder_stages)
        for i, st in enumerate(decoder_stages):
            out_ch = int(st["out_ch"])
            up_factor = int(st["up_factor"])
            is_last = i == n_dec - 1
            defer_skip = is_last and bool(st.get("xyz_head", False))
            skip_enc = st.get("skip_encoder", None)
            if st.get("fuse_bottom", False):
                up_stride = 1
            else:
                up_stride = up_factor
            self.dec_up.append(
                nn.ConvTranspose1d(dec_in_ch, out_ch, up_factor, stride=up_stride, bias=True)
            )
            if st.get("fuse_bottom", False):
                self.dec_fuse.append(MLP_Res(in_dim=dec_in_ch + out_ch, hidden_dim=out_ch, out_dim=out_ch))
            else:
                self.dec_fuse.append(None)
            if self.use_dec_rolgm:
                if skip_enc is not None:
                    if defer_skip:
                        self.dec_pos_heads.append(None)
                        if self._dec_rolgm_at_stage(i, True):
                            self.dec_rolgm.append(
                                self._build_rolgm_module(self.seed_fea, self.dec_rolgm_dim, self.dec_rolgm_cfg)
                            )
                        else:
                            self.dec_rolgm.append(None)
                    elif self._dec_rolgm_at_stage(i, True):
                        self.dec_pos_heads.append(
                            MLP_Res(in_dim=out_ch, hidden_dim=self.dec_pos_hidden, out_dim=3)
                        )
                        self.dec_rolgm.append(
                            self._build_rolgm_module(out_ch, self.dec_rolgm_dim, self.dec_rolgm_cfg)
                        )
                    else:
                        self.dec_pos_heads.append(None)
                        self.dec_rolgm.append(None)
                else:
                    self.dec_pos_heads.append(None)
                    self.dec_rolgm.append(None)
            dec_in_ch = out_ch

        last_st = decoder_stages[-1]
        if last_st.get("xyz_head", False):
            self.xyz_head = MLP_Res(in_dim=dec_in_ch, hidden_dim=32, out_dim=3)
        else:
            self.xyz_head = None

        self._stage_summary = summarize_stages(encoder_stages, decoder_stages)

    def _dec_rolgm_at_stage(self, stage_idx, has_skip):
        if not self.use_dec_rolgm or not has_skip:
            return False
        if self.dec_rolgm_stage_ids is None:
            return True
        return stage_idx in self.dec_rolgm_stage_ids

    def _build_rolgm_module(self, in_channel, dim, rolgm_cfg):
        return build_rolgm_from_cfg(in_channel, dim=dim, rolgm_cfg=rolgm_cfg)

    def _build_serial_block(self, feat_ch, stage):
        use_serial = bool(stage.get("use_rolgm", stage.get("use_lsm", self.use_rolgm)))
        if not use_serial:
            raise ValueError("Paper setup requires use_rolgm=True on every encoder stage.")

        serial_type = str(stage.get("enc_serial_type", self.enc_serial_type)).lower()
        if serial_type not in ("bimamba", "lsm_bimamba", "vim_bimamba", "rolgm"):
            raise ValueError(
                f"enc_serial_type must be 'rolgm', got {serial_type!r}."
            )
        stage_rolgm_cfg = stage.get("rolgm", stage.get("lsm", self.rolgm_cfg))
        stage_dim = int(stage.get("rolgm_dim", stage.get("lsm_dim", self.rolgm_dim)))
        return build_rolgm_from_cfg(feat_ch, dim=stage_dim, rolgm_cfg=stage_rolgm_cfg)

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, 3, N)
        Returns:
            bottom_fea, seed_xyz, seed_fea
        """
        xyz = point_cloud
        fea = point_cloud
        enc_xyz, enc_fea = [], []

        for i, (sa, rolgm) in enumerate(zip(self.enc_sa, self.enc_rolgm)):
            if sa.if_idx:
                xyz, fea, _idx = sa(xyz, fea)
            else:
                xyz, fea = sa(xyz, fea)
            if rolgm is not None:
                fea = rolgm(fea, xyz)
            enc_xyz.append(xyz)
            enc_fea.append(fea)

        bottom_fea = fea
        h = bottom_fea
        seed_xyz = None
        seed_fea = None

        for i, (up, fuse) in enumerate(zip(self.dec_up, self.dec_fuse)):
            st = self.decoder_stages[i]
            is_last = i == len(self.dec_up) - 1
            bottom_snapshot = bottom_fea
            h = up(h)
            if fuse is not None:
                b = bottom_snapshot.repeat(1, 1, h.size(2))
                h = fuse(torch.cat([h, b], dim=1))
            skip = st.get("skip_encoder", None)
            defer_skip_to_xyz = is_last and bool(st.get("xyz_head", False))

            if self.use_dec_rolgm and skip is not None:
                skip_idx = int(skip)
                dec_rolgm = self.dec_rolgm[i]
                if defer_skip_to_xyz:
                    pos_part = self.xyz_head(h)
                    seed_xyz = torch.cat([enc_xyz[skip_idx], pos_part], dim=2)
                    seed_fea = torch.cat([enc_fea[skip_idx], h], dim=2)
                    if dec_rolgm is not None:
                        seed_fea = dec_rolgm(seed_fea, seed_xyz)
                elif dec_rolgm is not None:
                    pos_part = self.dec_pos_heads[i](h)
                    xyz_cat = torch.cat([enc_xyz[skip_idx], pos_part], dim=2)
                    h = torch.cat([enc_fea[skip_idx], h], dim=2)
                    h = dec_rolgm(h, xyz_cat)
                else:
                    h = torch.cat([enc_fea[skip_idx], h], dim=2)

        if seed_fea is None or seed_xyz is None:
            raise RuntimeError("DecRoLGM decoder did not produce seed outputs.")

        return bottom_fea, seed_xyz, seed_fea


class PostProcess(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        upscale=(1, 4, 4),
        up_block_type="uptransformer",
        up_attn_dim=64,
        n_knn=20,
    ):
        super(PostProcess, self).__init__()
        self.up_block_type = up_block_type
        up_layers = []
        for i, factor in enumerate(upscale):
            up_layers.append(
                UpLayer(
                    dim=embed_dim,
                    seed_dim=embed_dim,
                    up_factor=factor,
                    i=i,
                    n_knn=n_knn,
                    radius=1,
                    interpolate="three",
                    attn_channel=True,
                    up_block_type=up_block_type,
                    up_attn_dim=up_attn_dim,
                )
            )
        self.up_layers = nn.ModuleList(up_layers)

    def forward(self, seed, fea):
        pred_pcds = [seed.permute(0, 2, 1).contiguous()]
        K_prev = None
        pcd = seed
        for layer in self.up_layers:
            pcd, K_prev = layer(pcd, seed, fea, K_prev)
            pred_pcds.append(pcd.permute(0, 2, 1).contiguous())
        return pred_pcds


@MODELS.register_module()
class UpTrans(nn.Module):
    def __init__(self, config, **kwargs):
        super(UpTrans, self).__init__()
        self.config = config
        self.num_pred = config.num_pred
        self.num_seeds = int(getattr(config, "num_seeds", 1024))
        self.use_L2 = config.l2_loss
        self.dense_decoder_type = str(getattr(config, "dense_decoder_type", "uplayer")).lower()
        if self.dense_decoder_type not in ("uplayer", "fc"):
            raise ValueError(
                f"dense_decoder_type must be 'uplayer' or 'fc', got {self.dense_decoder_type!r}."
            )

        self.feat_extractor = UNet(config=config)
        if self.dense_decoder_type == "fc":
            if self.num_pred % self.num_seeds != 0:
                raise ValueError(
                    f"FC dense decoder requires num_pred ({self.num_pred}) "
                    f"divisible by num_seeds ({self.num_seeds})."
                )
            self.decoder = AdaPoinTrFCDecoder(
                feat_dim=int(config.seed_fea),
                num_coarse=self.num_seeds,
                num_dense=int(self.num_pred),
                global_dim=int(getattr(config, "dense_global_dim", 1024)),
                fc_hidden=int(getattr(config, "dense_fc_hidden", 512)),
            )
            print_log(
                f"UpTrans dense_decoder_type=fc (AdaPoinTr): "
                f"coarse={self.num_seeds} -> dense={self.num_pred} "
                f"(×{self.num_pred // self.num_seeds})",
                logger="UpTrans",
            )
        else:
            dense_up_block_type = str(getattr(config, "dense_up_block_type", "uptransformer")).lower()
            up_attn_dim = int(getattr(config, "up_attn_dim", 64))
            up_n_knn = int(getattr(config, "up_n_knn", 20))
            self.decoder = PostProcess(
                embed_dim=config.seed_fea,
                upscale=config.upscales,
                up_block_type=dense_up_block_type,
                up_attn_dim=up_attn_dim,
                n_knn=up_n_knn,
            )
            dec_rolgm_dim = _cfg_int(config, "dec_rolgm_dim", "dec_lsm_dim", _cfg_int(config, "rolgm_dim", "lsm_dim", 64))
            dec_rolgm_stages = _cfg_block(config, "dec_rolgm_stages", "dec_lsm_stages", None)
            dec_rolgm_raw = _cfg_block(config, "dec_rolgm", "dec_lsm", None)
            sm_only = False
            random_serial = False
            eval_serial_one = False
            eval_serial_order = "xyz"
            if dec_rolgm_raw is not None:
                if hasattr(dec_rolgm_raw, "items"):
                    dlc = dict(dec_rolgm_raw)
                else:
                    dlc = {}
                if "use_local_branch" in dlc:
                    sm_only = not bool(dlc["use_local_branch"])
                else:
                    sm_only = int(dlc.get("lm_depth", 1)) <= 0
                random_serial = bool(dlc.get("random_serial_one", False))
                eval_serial_one = bool(dlc.get("eval_serial_one", False))
                eval_serial_order = str(dlc.get("eval_serial_order", "xyz"))
            rolgm_raw = _cfg_block(config, "rolgm", "lsm", None)
            if rolgm_raw is not None and hasattr(rolgm_raw, "items"):
                rolgm_d = dict(rolgm_raw)
                random_serial = random_serial or bool(rolgm_d.get("random_serial_one", False))
                if rolgm_d.get("eval_serial_one", False):
                    eval_serial_one = True
                if "eval_serial_order" in rolgm_d:
                    eval_serial_order = str(rolgm_d["eval_serial_order"])
            stages_msg = (
                f"stages={list(dec_rolgm_stages)}"
                if dec_rolgm_stages is not None
                else "stages=all"
            )
            branch_msg = "Global-only" if sm_only else "RoLGM"
            if random_serial and eval_serial_one:
                serial_msg = f"train=random1/eval={eval_serial_order}"
            elif random_serial:
                serial_msg = "random1/train,all/eval"
            else:
                serial_msg = "all"
            print_log(
                f"UpTrans dense_decoder_type=uplayer (UpLayer upscales={list(config.upscales)}, "
                f"up_block={dense_up_block_type}, dec_refine=Dec{branch_msg} dim={dec_rolgm_dim}, "
                f"{stages_msg}, serial={serial_msg})",
                logger="UpTrans",
            )
        self.build_loss_func()

    def build_loss_func(self):
        if self.use_L2 == 0:
            self.loss_func = ChamferDistanceL1()
        else:
            self.loss_func = ChamferDistanceL2()

    def get_loss(self, ret, gt):
        Pc, P3, P1, P2 = ret
        if self.dense_decoder_type == "fc":
            cd_coarse = self.loss_func(Pc, gt)
            cd_dense = self.loss_func(P3, gt)
            loss_all = (cd_coarse + cd_dense) * 1e3
            return loss_all, cd_coarse, cd_dense
        cd0 = self.loss_func(Pc, gt)
        cd1 = self.loss_func(P1, gt)
        cd2 = self.loss_func(P2, gt)
        cd3 = self.loss_func(P3, gt)
        loss_all = (1 * cd0 + 1 * cd1 + 1 * cd2 + 1 * cd3) * 1e3
        return loss_all, cd0, cd3

    def forward(self, partial_point_cloud):
        in_pcd = partial_point_cloud.permute(0, 2, 1).contiguous()
        _bottom_fea, xyz, fea = self.feat_extractor(in_pcd)
        if self.dense_decoder_type == "fc":
            coarse_n = xyz.size(2)
            if coarse_n != self.num_seeds:
                raise RuntimeError(
                    f"FC decoder expects coarse N={self.num_seeds}, got {coarse_n}. "
                    "Check decoder_stages / ConvTranspose stride / skip logic."
                )
            coarse, dense = self.decoder(xyz, fea)
            return coarse, dense, coarse, coarse
        pred_list = self.decoder(xyz, fea)
        pcd, p1, p2, p3 = pred_list[0], pred_list[1], pred_list[2], pred_list[3]
        return pcd, p3, p1, p2
