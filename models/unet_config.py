"""Parse encoder/decoder stage specs from UpTrans yaml config."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "items"):
        return dict(obj)
    raise TypeError(f"Expected dict-like stage spec, got {type(obj)}")


def parse_sa_geo_cfg(config, stage: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """Local geometry flags for PointNet_SA_Module_KNN_v1."""
    raw = None
    if stage is not None:
        raw = stage.get("sa_geo", None)
    if raw is None:
        raw = getattr(config, "sa_geo", None)
    if raw is None:
        return {
            "use_distance": False,
            "use_curvature": False,
            "use_linearity": False,
            "use_planarity": False,
        }
    d = _to_dict(raw)
    return {
        "use_distance": bool(d.get("use_distance", False)),
        "use_curvature": bool(d.get("use_curvature", False)),
        "use_linearity": bool(d.get("use_linearity", False)),
        "use_planarity": bool(d.get("use_planarity", False)),
    }


def parse_sa_pool_mode(config, stage: Optional[Dict[str, Any]] = None) -> str:
    if stage is not None and stage.get("sa_pool_mode") is not None:
        mode = str(stage["sa_pool_mode"]).lower()
    else:
        mode = str(getattr(config, "sa_pool_mode", "max")).lower()
    if mode not in ("max", "max_avg"):
        raise ValueError(f"sa_pool_mode must be 'max' or 'max_avg', got {mode!r}")
    return mode


def parse_sa_version(config, stage: Optional[Dict[str, Any]] = None) -> str:
    if stage is not None and stage.get("sa_version") is not None:
        return str(stage["sa_version"]).lower()
    return str(getattr(config, "sa_version", "v1")).lower()


def _sa_mlp_dims(
    in_ch: int, feat_ch: int, first_from_xyz: bool, group_all: bool = False
) -> List[int]:
    """SA MLP widths; group_all matches legacy DictFold sa_module_4: [384, dim_feat]."""
    if group_all:
        return [384, feat_ch]
    if first_from_xyz:
        hidden = max(feat_ch // 4, 32)
        return [hidden, feat_ch]
    hidden = min(max(in_ch, 64), 128)
    return [hidden, feat_ch]


def _cfg_bool(config, new_key: str, legacy_key: str, default=True) -> bool:
    if hasattr(config, new_key):
        return bool(getattr(config, new_key))
    if hasattr(config, legacy_key):
        return bool(getattr(config, legacy_key))
    return default


def _cfg_int(config, new_key: str, legacy_key: str, default: int) -> int:
    if hasattr(config, new_key):
        return int(getattr(config, new_key))
    if hasattr(config, legacy_key):
        return int(getattr(config, legacy_key))
    return default


def _cfg_block(config, new_key: str, legacy_key: str, default=None):
    raw = getattr(config, new_key, None)
    if raw is None:
        raw = getattr(config, legacy_key, default)
    return raw


def default_encoder_stages(config) -> List[Dict[str, Any]]:
    """Legacy ODGNet UpTrans: 512-128 + global pool."""
    num_seeds = int(config.num_seeds)
    seed_fea = int(config.seed_fea)
    dim_feat = int(config.dim_feat)
    use_rolgm = _cfg_bool(config, "use_rolgm", "use_lsm", True)
    return [
        {"npoints": num_seeds // 2, "feat_ch": seed_fea, "knn": 16, "use_rolgm": use_rolgm},
        {"npoints": num_seeds // 8, "feat_ch": 256, "knn": 16, "use_rolgm": use_rolgm},
        {
            "group_all": True,
            "feat_ch": dim_feat,
            "use_rolgm": False,
            "mlp": [384, dim_feat],
        },
    ]


def default_decoder_stages(config) -> List[Dict[str, Any]]:
    num_seeds = int(config.num_seeds)
    seed_fea = int(config.seed_fea)
    return [
        {
            "out_ch": 256,
            "up_factor": num_seeds // 16,
            "fuse_bottom": True,
        },
        {
            "out_ch": 256,
            "up_factor": 2,
            "skip_encoder": 1,
        },
        {
            "out_ch": seed_fea,
            "up_factor": 2,
            "skip_encoder": 0,
            "xyz_head": True,
        },
    ]


def parse_encoder_stages(config) -> List[Dict[str, Any]]:
    raw = getattr(config, "encoder_stages", None)
    if raw is None:
        return default_encoder_stages(config)
    stages = [_to_dict(s) for s in raw]
    if not stages:
        raise ValueError("encoder_stages must contain at least one stage")
    default_use_rolgm = _cfg_bool(config, "use_rolgm", "use_lsm", True)
    for i, st in enumerate(stages):
        if "feat_ch" not in st:
            raise ValueError(f"encoder_stages[{i}] missing feat_ch")
        group_all = bool(st.get("group_all", False))
        if group_all:
            st.setdefault("use_rolgm", st.get("use_lsm", False))
            if "mlp" not in st:
                st["mlp"] = [384, int(st["feat_ch"])]
        else:
            if "npoints" not in st:
                raise ValueError(f"encoder_stages[{i}] missing npoints (or set group_all: true)")
            st.setdefault("knn", 16)
            if "use_rolgm" not in st and "use_lsm" not in st:
                st.setdefault("use_rolgm", default_use_rolgm)
            elif "use_rolgm" not in st and "use_lsm" in st:
                st.setdefault("use_rolgm", bool(st["use_lsm"]))
    return stages


def parse_decoder_stages(config, encoder_stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw = getattr(config, "decoder_stages", None)
    if raw is None:
        return default_decoder_stages(config)
    stages = [_to_dict(s) for s in raw]
    if not stages:
        raise ValueError("decoder_stages must contain at least one stage")
    n_enc = len(encoder_stages)
    for i, st in enumerate(stages):
        if "out_ch" not in st:
            raise ValueError(f"decoder_stages[{i}] missing out_ch")
        if "up_factor" not in st:
            raise ValueError(f"decoder_stages[{i}] missing up_factor")
        skip = st.get("skip_encoder", None)
        if skip is not None and not (0 <= int(skip) < n_enc):
            raise ValueError(f"decoder_stages[{i}] skip_encoder={skip} out of range [0, {n_enc})")
        st.setdefault("fuse_bottom", False)
        st.setdefault("xyz_head", False)
    return stages


def summarize_stages(encoder_stages, decoder_stages) -> str:
    enc = []
    for st in encoder_stages:
        if st.get("group_all"):
            enc.append(f"global({st['feat_ch']})")
        else:
            enc.append(f"{st['npoints']}x{st['feat_ch']}")
    dec = [f"up{x['up_factor']}->{x['out_ch']}" for x in decoder_stages]
    return f"encoder [{' -> '.join(enc)}] | decoder [{' -> '.join(dec)}]"
