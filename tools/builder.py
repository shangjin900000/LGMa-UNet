import os, sys
# online package
import torch
# optimizer
import torch.optim as optim
# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
# utils
from utils.logger import *
from utils.misc import *

def dataset_builder(args, config, distributed=None):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    shuffle = config.others.subset == 'train'
    use_distributed = args.distributed if distributed is None else distributed
    if use_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs if shuffle else 1,
                                            num_workers = int(args.num_workers),
                                            pin_memory=True,
                                            drop_last = config.others.subset == 'train',
                                            worker_init_fn = worker_init_fn,
                                            sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs if shuffle else 1,
                                                shuffle = shuffle, 
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                pin_memory=True,
                                                worker_init_fn=worker_init_fn)
    return sampler, dataloader

def model_builder(config):
    model = build_model_from_cfg(config)
    return model

def build_opti_sche(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        optimizer = optim.AdamW(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError()

    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'WarmUpCosLR':
        assert sche_config.kwargs.lr_max == opti_config.kwargs.lr
        scheduler = build_warm_cos_sche(optimizer, sche_config.kwargs)  # misc.py
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]
    
    return optimizer, scheduler

def resume_model(base_model, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    # print(best_metrics)

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger = logger)
    return start_epoch, best_metrics

def resume_optimizer(optimizer, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger = None):
    if args.local_rank == 0:
        torch.save({
                    'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    'metrics' : metrics.state_dict() if metrics is not None else dict(),
                    'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)

_LEGACY_UPTRANS_PREFIX_MAP = (
    ("feat_extractor.sa_module_1.", "feat_extractor.enc_sa.0."),
    ("feat_extractor.sa_module_2.", "feat_extractor.enc_sa.1."),
    ("feat_extractor.sa_module_4.", "feat_extractor.enc_sa.2."),
    ("feat_extractor.transformer_1.", "feat_extractor.enc_rolgm.0."),
    ("feat_extractor.transformer_2.", "feat_extractor.enc_rolgm.1."),
    ("feat_extractor.enc_lsm.", "feat_extractor.enc_rolgm."),
    ("feat_extractor.dec_lsm.", "feat_extractor.dec_rolgm."),
    ("feat_extractor.ps_0.", "feat_extractor.dec_up.0."),
    ("feat_extractor.mlp_1.", "feat_extractor.dec_fuse.0."),
    ("feat_extractor.ps_1.", "feat_extractor.dec_up.1."),
    ("feat_extractor.ps_2.", "feat_extractor.dec_up.2."),
    ("feat_extractor.refine_1.", "feat_extractor.dec_refine.1."),
    ("feat_extractor.refine_2.", "feat_extractor.dec_refine.2."),
    ("feat_extractor.mlp_3.", "feat_extractor.xyz_head."),
)


def _remap_legacy_uptrans_keys(base_ckpt):
    remapped = {}
    legacy_hits = 0
    for k, v in base_ckpt.items():
        new_k = k
        for old_prefix, new_prefix in _LEGACY_UPTRANS_PREFIX_MAP:
            if k.startswith(old_prefix):
                new_k = new_prefix + k[len(old_prefix):]
                legacy_hits += 1
                break
        remapped[new_k] = v
    return remapped, legacy_hits


def load_model(base_model, ckpt_path, logger=None, ignore_prefix=()):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger=logger)

    state_dict = torch.load(ckpt_path, map_location='cpu')
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')

    filtered = {}
    ignored = []
    for k, v in base_ckpt.items():
        if any(k.startswith(p) for p in ignore_prefix):
            ignored.append(k)
            continue
        filtered[k] = v

    remapped, legacy_hits = _remap_legacy_uptrans_keys(filtered)
    model_sd = base_model.state_dict()
    loadable = {}
    shape_mismatch = []
    for k, v in remapped.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            loadable[k] = v
        else:
            shape_mismatch.append(k)

    msg = base_model.load_state_dict(loadable, strict=False)

    print_log(f'[load_model] ignored prefix keys: {len(ignored)}', logger=logger)
    print_log(f'[load_model] legacy UpTrans key remap hits: {legacy_hits}', logger=logger)
    print_log(f'[load_model] loaded (shape match): {len(loadable)}', logger=logger)
    print_log(f'[load_model] shape-mismatch skipped: {len(shape_mismatch)}', logger=logger)
    print_log(
        f'[load_model] missing keys: {len(msg.missing_keys)} unexpected keys: {len(msg.unexpected_keys)}',
        logger=logger,
    )

    epoch = state_dict.get('epoch', -1)
    metrics = state_dict.get('metrics', 'No Metrics')
    if not isinstance(metrics, dict) and hasattr(metrics, 'state_dict'):
        metrics = metrics.state_dict()
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger=logger)
    return 