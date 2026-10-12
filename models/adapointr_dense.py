"""AdaPoinTr FC dense decoder for UpTrans.

Math matches PoinTr ``decoder_type: fc`` (SimpleRebuildFCLayer + coarse-fixed offsets).
Wiring matches ODGNet UpTrans: UNet seed (B,3,M)/(B,C,M) -> coarse / dense.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.AdaPoinTr import SimpleRebuildFCLayer


class AdaPoinTrFCDecoder(nn.Module):
    """
    Dense = coarse + FC(refined_feat), coarse xyz fixed (AdaPoinTr ``decoder_type: fc``).

    Inputs from UNet:
        seed_xyz: (B, 3, M)
        seed_fea: (B, C, M)
    Outputs:
        coarse: (B, M, 3)
        dense:  (B, M * upsample_ratio, 3)
    """

    def __init__(
        self,
        feat_dim: int = 128,
        num_coarse: int = 1024,
        num_dense: int = 8192,
        global_dim: int = 1024,
        fc_hidden: int = 512,
    ):
        super().__init__()
        if num_dense % num_coarse != 0:
            raise ValueError(
                f"num_dense ({num_dense}) must be divisible by num_coarse ({num_coarse}) for FC decoder."
            )
        self.feat_dim = int(feat_dim)
        self.num_coarse = int(num_coarse)
        self.num_dense = int(num_dense)
        self.upsample_ratio = self.num_dense // self.num_coarse

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.feat_dim, global_dim, 1),
            nn.BatchNorm1d(global_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(global_dim, global_dim, 1),
        )
        self.reduce_map = nn.Linear(global_dim + self.feat_dim + 3, self.feat_dim)
        self.decode_head = SimpleRebuildFCLayer(
            input_dims=self.feat_dim * 2,
            step=self.upsample_ratio,
            hidden_dim=fc_hidden,
        )

    def forward(self, seed_xyz: torch.Tensor, seed_fea: torch.Tensor):
        coarse = seed_xyz.permute(0, 2, 1).contiguous()
        q = seed_fea.permute(0, 2, 1).contiguous()
        b, m, _ = coarse.shape

        global_feature = self.increase_dim(seed_fea)
        global_feature = global_feature.permute(0, 2, 1).contiguous()
        global_feature = torch.max(global_feature, dim=1)[0]

        rebuild_feature = torch.cat(
            [
                global_feature.unsqueeze(1).expand(-1, m, -1),
                q,
                coarse,
            ],
            dim=-1,
        )
        rebuild_feature = self.reduce_map(rebuild_feature)
        relative_xyz = self.decode_head(rebuild_feature)
        dense = (relative_xyz + coarse.unsqueeze(-2)).reshape(b, -1, 3).contiguous()
        return coarse, dense
