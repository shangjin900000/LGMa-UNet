"""Log UpTrans architecture and parameter stats at training start."""

from __future__ import annotations

from typing import Any, Tuple

import torch.nn as nn

from utils.logger import print_log


def count_params(module: nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, total_params) for a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def log_model_startup_info(model: nn.Module, model_config: Any, logger=None) -> Tuple[int, int]:
    """Print config, architecture, module breakdown, and verified param counts."""
    trainable, total = count_params(model)

    print_log("=" * 60, logger=logger)
    print_log("[ModelInfo] training startup summary", logger=logger)
    print_log("=" * 60, logger=logger)

    name = getattr(model_config, "NAME", type(model).__name__)
    print_log(f"[ModelInfo] NAME={name}", logger=logger)
    for key in (
        "num_pred",
        "num_seeds",
        "seed_fea",
        "dim_feat",
        "dense_decoder_type",
        "dense_global_dim",
        "dense_fc_hidden",
        "num_query",
        "num_points",
        "decoder_type",
        "encoder_type",
        "upscales",
        "use_rolgm",
        "enc_serial_type",
        "dense_up_block_type",
        "dec_rolgm_dim",
        "up_n_knn",
        "dec_fea_refine_type",
        "dec_rolgm_stages",
        "dec_rolgm",
        "dec_pos_hidden",
        "rolgm_dim",
        "l2_loss",
        "rolgm",
        "w_psr",
        "psr_grid_res",
        "sap_n_points",
        "sap_config",
        "sap_scale",
        "freeze_sap",
    ):
        if hasattr(model_config, key):
            print_log(f"[ModelInfo] {key}={getattr(model_config, key)}", logger=logger)

    fe = getattr(model, "feat_extractor", None)
    if fe is not None:
        if hasattr(fe, "_stage_summary"):
            print_log(f"[ModelInfo] stages: {fe._stage_summary}", logger=logger)
        use_rolgm = getattr(fe, "use_rolgm", None)
        enc_serial_type = getattr(fe, "enc_serial_type", None)
        if enc_serial_type is not None:
            print_log(f"[ModelInfo] enc_serial_type={enc_serial_type}", logger=logger)
        if use_rolgm is not None:
            if not use_rolgm:
                backend = "disabled"
            else:
                backend = "RoLGM(Vim BiMamba)"
            print_log(f"[ModelInfo] encoder backend: {backend}", logger=logger)
        dec_rolgm = getattr(fe, "dec_rolgm", None)
        if dec_rolgm is not None:
            for i, block in enumerate(dec_rolgm):
                if block is not None:
                    rs = getattr(block, "random_serial_one", False)
                    es = getattr(block, "eval_serial_one", False)
                    eo = getattr(block, "eval_serial_order", "xyz")
                    n_ord = len(getattr(block, "serial_axis_orders", ()))
                    bi = getattr(block, "bidirectional", False)
                    bim = getattr(block, "bimamba_type", None)
                    if bim is not None:
                        rms = getattr(block, "rms_norm", None)
                        rms_tag = f" rms_norm={rms}" if rms is not None else ""
                        bi_tag = f" vim_block bimamba={bim} divide={getattr(block, 'if_divide_out', True)}{rms_tag}"
                    elif bi:
                        bi_tag = " bi_ssm=True"
                    else:
                        bi_tag = ""
                    print_log(
                        f"[ModelInfo] dec[{i}] block: DecRoLGM({type(block).__name__}) "
                        f"orders={n_ord} train_random1={rs} eval_one={es}({eo}){bi_tag}",
                        logger=logger,
                    )
        enc_rolgm_list = getattr(fe, "enc_rolgm", None)
        if enc_rolgm_list is not None:
            for i, block in enumerate(enc_rolgm_list):
                if block is not None:
                    rs = getattr(block, "random_serial_one", False)
                    es = getattr(block, "eval_serial_one", False)
                    eo = getattr(block, "eval_serial_order", "xyz")
                    n_ord = len(getattr(block, "serial_axis_orders", ()))
                    bi = getattr(block, "bidirectional", False)
                    bim = getattr(block, "bimamba_type", None)
                    if bim is not None:
                        rms = getattr(block, "rms_norm", None)
                        rms_tag = f" rms_norm={rms}" if rms is not None else ""
                        bi_tag = f" vim_block bimamba={bim} divide={getattr(block, 'if_divide_out', True)}{rms_tag}"
                    elif bi:
                        bi_tag = " bi_ssm=True"
                    else:
                        bi_tag = ""
                    print_log(
                        f"[ModelInfo] enc[{i}] block: {type(block).__name__} "
                        f"orders={n_ord} train_random1={rs} eval_one={es}({eo}){bi_tag}",
                        logger=logger,
                    )
                elif block is None:
                    print_log(f"[ModelInfo] enc[{i}] block: (none, group_all)", logger=logger)

    print_log("[ModelInfo] model structure:", logger=logger)
    print_log(str(model), logger=logger)

    print_log("-" * 60, logger=logger)
    print_log("[ParamStats] module breakdown (trainable / total):", logger=logger)

    child_total = 0
    for child_name, child in model.named_children():
        child_train, child_all = count_params(child)
        child_total += child_all
        print_log(
            f"  {child_name:20s} {child_train:>12,} / {child_all:>12,}  ({child_all / 1e6:.3f}M total)",
            logger=logger,
        )

    if fe is not None:
        print_log("  [UNet detail]", logger=logger)
        for child_name, child in fe.named_children():
            _, child_all = count_params(child)
            if child_all == 0:
                continue
            child_train, _ = count_params(child)
            print_log(f"    {child_name:18s} {child_train:>12,} / {child_all:>12,}", logger=logger)

        enc_sa = getattr(fe, "enc_sa", None)
        enc_rolgm = getattr(fe, "enc_rolgm", None)
        if enc_sa is not None and enc_rolgm is not None:
            for i, (sa, rolgm) in enumerate(zip(enc_sa, enc_rolgm)):
                sa_n = sum(p.numel() for p in sa.parameters())
                rolgm_n = sum(p.numel() for p in rolgm.parameters()) if rolgm is not None else 0
                rolgm_name = type(rolgm).__name__ if rolgm is not None else "none"
                print_log(f"    enc[{i}] GeoSA={sa_n:,} {rolgm_name}={rolgm_n:,}", logger=logger)

    decoder = getattr(model, "decoder", None)
    if decoder is not None and hasattr(decoder, "up_layers"):
        up_block = getattr(decoder, "up_block_type", "uptransformer")
        print_log(f"  [PostProcess UpLayer] up_block={up_block}", logger=logger)
        upscales = getattr(model_config, "upscales", [])
        for i, layer in enumerate(decoder.up_layers):
            _, layer_all = count_params(layer)
            up_factor = upscales[i] if i < len(upscales) else "?"
            print_log(f"    [{i}] upscale={up_factor}  {layer_all:>12,}", logger=logger)
    elif decoder is not None and type(decoder).__name__ == "AdaPoinTrFCDecoder":
        print_log(
            f"  [AdaPoinTrFCDecoder] coarse={getattr(decoder, 'num_coarse', '?')} "
            f"dense={getattr(decoder, 'num_dense', '?')} "
            f"ratio={getattr(decoder, 'upsample_ratio', '?')}",
            logger=logger,
        )

    if type(model).__name__ == "AdaPoinTrSAP":
        print_log(
            f"  [AdaPoinTrSAP] sap_n={getattr(model, 'sap_n_points', '?')} "
            f"w_psr={getattr(model, 'w_psr', '?')} "
            f"psr_tanh={getattr(model, 'psr_tanh', '?')}",
            logger=logger,
        )

    non_trainable = total - trainable
    print_log("-" * 60, logger=logger)
    print_log(
        f"[ParamStats] model trainable/all: {trainable:,} / {total:,} "
        f"({trainable / 1e6:.3f}M / {total / 1e6:.3f}M)",
        logger=logger,
    )
    if non_trainable > 0:
        print_log(
            f"[ParamStats] non-trainable: {non_trainable:,} ({non_trainable / 1e6:.3f}M)",
            logger=logger,
        )
    if total > 0:
        print_log(f"[ParamStats] tuning ratio: {100.0 * trainable / total:.2f}%", logger=logger)

    named_total = sum(p.numel() for p in model.parameters())
    if named_total != total:
        print_log(
            f"[ParamStats] WARN: .parameters() sum {named_total:,} != total {total:,}",
            logger=logger,
        )
    else:
        print_log("[ParamStats] verified: sum(.parameters()) == total", logger=logger)

    if child_total != total:
        print_log(
            f"[ParamStats] WARN: top-level children total {child_total:,} != model total {total:,}",
            logger=logger,
        )
    else:
        print_log("[ParamStats] verified: sum(top-level children) == total", logger=logger)

    print_log("=" * 60, logger=logger)
    return trainable, total
