#! /usr/bin/python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, three_nn, three_interpolate, grouping_operation

class Conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1,  if_bn=True, activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    sorted_dist, indices = torch.sort(sqrdists, dim=-1, descending=False)
    idx = indices[:, :, pad: nsample+pad]
    return idx.int()


def _broadcast_center_scalar(x, nsample):
    """(B, npoint) -> (B, 1, npoint, nsample)"""
    return x.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, nsample)


def compute_sa_geo_features(grouped_xyz, nsample, sa_geo_cfg, eps=1e-6):
    """
    Build optional local geometry channels from relative neighbor coords.

    grouped_xyz: (B, 3, npoint, nsample), already center-relative.
    Returns (B, C_geo, npoint, nsample) or None if all flags are off.
    """
    if sa_geo_cfg is None:
        return None

    parts = []
    if sa_geo_cfg.get("use_distance"):
        parts.append(torch.norm(grouped_xyz, dim=1, keepdim=True))

    need_pca = (
        sa_geo_cfg.get("use_curvature")
        or sa_geo_cfg.get("use_linearity")
        or sa_geo_cfg.get("use_planarity")
    )
    if need_pca:
        q = grouped_xyz.permute(0, 2, 3, 1).contiguous()
        cov = torch.matmul(q.transpose(-1, -2), q) / q.size(2)
        eigvals = torch.linalg.eigvalsh(cov)
        lambda3 = eigvals[..., 0]
        lambda2 = eigvals[..., 1]
        lambda1 = eigvals[..., 2]

        if sa_geo_cfg.get("use_curvature"):
            curvature = lambda3 / (lambda1 + lambda2 + lambda3 + eps)
            parts.append(_broadcast_center_scalar(curvature, nsample))
        if sa_geo_cfg.get("use_linearity"):
            linearity = (lambda1 - lambda2) / (lambda1 + eps)
            parts.append(_broadcast_center_scalar(linearity, nsample))
        if sa_geo_cfg.get("use_planarity"):
            planarity = (lambda2 - lambda3) / (lambda1 + eps)
            parts.append(_broadcast_center_scalar(planarity, nsample))

    if not parts:
        return None
    return torch.cat(parts, dim=1)


def append_sa_geo_features(new_points, grouped_xyz, nsample, sa_geo_cfg):
    geo = compute_sa_geo_features(grouped_xyz, nsample, sa_geo_cfg)
    if geo is None:
        return new_points
    return torch.cat([new_points, geo], dim=1)


def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        k: int (knn count)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, k)
        idx: Tensor, (B, npoint, k)
        grouped_xyz: Tensor, (B, 3, npoint, k)
    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous() # (B, N, 3)
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint)) # (B, 3, npoint)
    if idx is None:
        idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx) # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = grouping_operation(points, idx) # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def count_sa_geo_channels(sa_geo_cfg):
    """Number of extra patch channels from local geometry encoding."""
    if sa_geo_cfg is None:
        return 0
    n = 0
    if sa_geo_cfg.get("use_distance"):
        n += 1
    if sa_geo_cfg.get("use_curvature"):
        n += 1
    if sa_geo_cfg.get("use_linearity"):
        n += 1
    if sa_geo_cfg.get("use_planarity"):
        n += 1
    return n


class PointNet_SA_Module_KNN_v1(nn.Module):
    """
    Geo-SA: geometry-aware set abstraction (distance / PCA descriptors + max+avg pool).

    Toggle via sa_geo_cfg (per-stage or global) and pool_mode ('max' | 'max_avg').
    Local geometry descriptors are skipped for group_all stages.
    """

    def __init__(
        self,
        npoint,
        nsample,
        in_channel,
        mlp,
        if_bn=True,
        group_all=False,
        use_xyz=True,
        if_idx=False,
        sa_geo_cfg=None,
        pool_mode="max",
    ):
        super(PointNet_SA_Module_KNN_v1, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        self.sa_geo_cfg = None if group_all else sa_geo_cfg
        if group_all:
            self.pool_mode = "max"
        else:
            self.pool_mode = pool_mode if pool_mode in ("max", "max_avg") else "max"

        mlp_in = in_channel
        if use_xyz:
            mlp_in += 3
        mlp_in += count_sa_geo_channels(self.sa_geo_cfg)

        last_channel = mlp_in
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

        if self.pool_mode == "max_avg":
            self.pool_fuse = nn.Conv1d(mlp[-1] * 2, mlp[-1], 1)
        else:
            self.pool_fuse = None

    def _aggregate_neighbors(self, new_points):
        if self.pool_mode == "max_avg":
            feat_max = torch.max(new_points, dim=3)[0]
            feat_avg = torch.mean(new_points, dim=3)
            return self.pool_fuse(torch.cat([feat_max, feat_avg], dim=1))
        return torch.max(new_points, dim=3)[0]

    def forward(self, xyz, points, idx=None):
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(
                xyz, points, self.use_xyz
            )
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(
                xyz, points, self.npoint, self.nsample, self.use_xyz, idx=idx
            )
            new_points = append_sa_geo_features(
                new_points, grouped_xyz, self.nsample, self.sa_geo_cfg
            )

        new_points = self.mlp_conv(new_points)
        new_points = self._aggregate_neighbors(new_points)

        if self.if_idx:
            return new_xyz, new_points, idx
        return new_xyz, new_points


GeoSA = PointNet_SA_Module_KNN_v1


def build_pointnet_sa_module(
    npoint,
    nsample,
    in_channel,
    mlp,
    group_all=False,
    if_bn=False,
    if_idx=False,
    use_xyz=True,
    sa_version="v1",
    sa_geo_cfg=None,
    pool_mode="max",
):
    """Build Geo-SA module (sa_version must be 'v1')."""
    sa_version = str(sa_version).lower()
    if sa_version not in ("v1", "geo", "geosa"):
        raise ValueError(f"sa_version must be 'v1', got {sa_version!r}.")
    return GeoSA(
        npoint,
        nsample,
        in_channel,
        mlp,
        if_bn=if_bn,
        group_all=group_all,
        use_xyz=use_xyz,
        if_idx=if_idx,
        sa_geo_cfg=sa_geo_cfg,
        pool_mode=pool_mode,
    )


def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd


def get_nearest_index(target, source, k=1, return_dis=False):
    """
    Args:
        target: (bs, 3, v1)
        source: (bs, 3, v2)
    Return:
        nearest_index: (bs, v1, 1)
    """
    inner = torch.bmm(target.transpose(1, 2), source)  # (bs, v1, v2)
    s_norm_2 = torch.sum(source**2, dim=1)  # (bs, v2)
    t_norm_2 = torch.sum(target**2, dim=1)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(
        2) - 2 * inner  # (bs, v1, v2)
    nearest_dis, nearest_index = torch.topk(d_norm_2,
                                            k=k,
                                            dim=-1,
                                            largest=False)
    if not return_dis:
        return nearest_index
    else:
        return nearest_index, nearest_dis


def indexing_neighbor(x, index):
    """
    Args:
        x: (bs, dim, num_points0)
        index: (bs, num_points, k)
    Return:
        feature: (bs, dim, num_points, k)
    """
    if index.dtype != torch.long:
        index = index.long()
    max_idx = x.size(2) - 1
    index = index.clamp(min=0, max=max_idx)

    idx_expand = index.unsqueeze(1).expand(-1, x.size(1), -1, -1)  # (B, C, N, K)
    x_expand = x.unsqueeze(2).expand(-1, -1, index.size(1), -1)    # (B, C, N, N0)
    feature = torch.gather(x_expand, dim=3, index=idx_expand)       # (B, C, N, K)
    return feature

class UpTransformer(nn.Module):
    def __init__(self, in_channel, out_channel, dim, n_knn=20, up_factor=2, use_upfeat=True, 
                 pos_hidden_dim=64, attn_hidden_multiplier=4, scale_layer=nn.Softmax, attn_channel=True):
        super(UpTransformer, self).__init__()
        self.n_knn = n_knn
        self.up_factor = up_factor
        self.use_upfeat = use_upfeat
        attn_out_channel = dim if attn_channel else 1

        self.mlp_v = MLP_Res(in_dim=in_channel*2, hidden_dim=in_channel, out_dim=in_channel)
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        if use_upfeat:
            self.conv_upfeat = nn.Conv1d(in_channel, dim, 1)

        self.scale = scale_layer(dim=-1) if scale_layer is not None else nn.Identity()

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = [nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
                         nn.BatchNorm2d(dim * attn_hidden_multiplier),
                         nn.ReLU()]
        if up_factor:
            self.attn_mlp.append(nn.ConvTranspose2d(dim * attn_hidden_multiplier, attn_out_channel, (up_factor,1), (up_factor,1)))
        else:
            self.attn_mlp.append(nn.Conv2d(dim * attn_hidden_multiplier, attn_out_channel, 1))
        self.attn_mlp = nn.Sequential(*self.attn_mlp)

        self.upsample1 = nn.Upsample(scale_factor=(up_factor,1)) if up_factor else nn.Identity()
        self.upsample2 = nn.Upsample(scale_factor=up_factor) if up_factor else nn.Identity()

        self.conv_end = nn.Conv1d(dim, out_channel, 1)
        if in_channel != out_channel:
            self.residual_layer = nn.Conv1d(in_channel, out_channel, 1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, pos, key, query, upfeat):
        """
        Inputs:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
        """

        value = self.mlp_v(torch.cat([key, query], 1)) # (B, dim, N)
        identity = value
        key = self.conv_key(key) # (B, dim, N)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)

        key = grouping_operation(key, idx_knn)  # (B, dim, N, k)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # (B, 3, N, k)
        pos_embedding = self.pos_mlp(pos_rel)  # (B, dim, N, k)

        if self.use_upfeat:
            upfeat = self.conv_upfeat(upfeat) # (B, dim, N)
            upfeat_rel = upfeat.reshape((b, -1, n, 1)) - grouping_operation(upfeat, idx_knn) # (B, dim, N, k)
        else:
            upfeat_rel = torch.zeros_like(qk_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding + upfeat_rel) # (B, dim, N*up_factor, k)
        attention = self.scale(attention)

        value = grouping_operation(value, idx_knn) + pos_embedding + upfeat_rel # (B, dim, N, k)
        value = self.upsample1(value) # (B, dim, N*up_factor, k)

        agg = torch.einsum('b c i j, b c i j -> b c i', attention, value)  # (B, dim, N*up_factor)
        y = self.conv_end(agg) # (B, out_dim, N*up_factor)

        identity = self.residual_layer(identity) # (B, out_dim, N)
        identity = self.upsample2(identity) # (B, out_dim, N*up_factor)

        return y+identity

def _build_up_refiner(
    up_block_type,
    in_channel,
    out_channel,
    n_knn,
    up_factor,
    attn_channel=True,
    up_attn_dim=64,
):
    block_type = str(up_block_type).lower()
    if block_type not in ("uptransformer", "transformer"):
        raise ValueError(
            f"dense_up_block_type must be 'uptransformer' (paper setup), got {block_type!r}."
        )
    return UpTransformer(
        in_channel,
        out_channel,
        dim=up_attn_dim,
        n_knn=n_knn,
        use_upfeat=True,
        attn_channel=attn_channel,
        up_factor=up_factor,
    )


class UpLayer(nn.Module):
    """Upsample layer with UpTransformer refinement."""

    def __init__(
        self,
        dim,
        seed_dim,
        up_factor=2,
        i=0,
        radius=1,
        n_knn=20,
        interpolate='three',
        attn_channel=True,
        up_block_type='uptransformer',
        up_attn_dim=64,
    ):
        super(UpLayer, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.n_knn = n_knn
        self.interpolate = interpolate
        self.up_block_type = up_block_type

        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + seed_dim, layer_dims=[256, dim])

        self.uptrans1 = _build_up_refiner(
            up_block_type,
            dim,
            dim,
            n_knn=self.n_knn,
            up_factor=None,
            up_attn_dim=up_attn_dim,
        )
        self.uptrans2 = _build_up_refiner(
            up_block_type,
            dim,
            dim,
            n_knn=self.n_knn,
            up_factor=self.up_factor,
            attn_channel=attn_channel,
            up_attn_dim=up_attn_dim,
        )

        self.upsample = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=dim*2, hidden_dim=dim, out_dim=dim)

        self.mlp_delta = MLP_CONV(in_channel=dim, layer_dims=[64, 3])

    def forward(self, pcd_prev, seed, seed_feat, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            seed: Tensor, (B, 3, N_seed)
            seed_feat: Tensor, (B, seed_dim, N_seed)
            K_prev: Tensor, (B, dim, N_prev)

        Returns:
            pcd_new: Tensor, (B, 3, N_prev * up_factor)
            K_curr: Tensor, (B, dim, N_prev * up_factor)
        """
        if self.interpolate == 'nearest':
            idx = get_nearest_index(pcd_prev, seed)
            feat_upsample = indexing_neighbor(seed_feat, idx).squeeze(3) # (B, seed_dim, N_prev)
        elif self.interpolate == 'three':
            idx, dis = get_nearest_index(pcd_prev, seed, k=3, return_dis=True) # (B, N_prev, 3), (B, N_prev, 3)
            idx = idx.clamp(min=0, max=seed_feat.size(2) - 1)
            dist_recip = 1.0 / (dis + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True) # (B, N_prev, 1)
            weight = dist_recip / norm # (B, N_prev, 3)
            feat_upsample = torch.sum(indexing_neighbor(seed_feat, idx) * weight.unsqueeze(1), dim=-1) # (B, seed_dim, N_prev)
        else:
            raise ValueError('Unknown Interpolation: {}'.format(self.interpolate))

        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_upsample], 1)
        Q = self.mlp_2(feat_1)

        H = self.uptrans1(pcd_prev, K_prev if K_prev is not None else Q, Q, upfeat=feat_upsample) # (B, 128, N_prev)
        feat_child = self.uptrans2(pcd_prev, K_prev if K_prev is not None else H, H, upfeat=feat_upsample) # (B, 128, N_prev*up_factor)

        H_up = self.upsample(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)
        pcd_new = self.upsample(pcd_prev)
        pcd_new = pcd_new + delta

        return pcd_new, K_curr
