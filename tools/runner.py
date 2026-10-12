import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import os
import json
from . import builder
from .model_info import log_model_startup_info
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

_CUDNN_FALLBACK_LOGGED = False
GRAD_NORM_CLIP = 1.0
MAX_CONSECUTIVE_NAN_GRAD = 10


def _sync_any_rank_flag(local_flag, device):
    """True if any rank reported local_flag=True (DDP-safe skip/step decision)."""
    if not dist.is_initialized():
        return bool(local_flag)
    flag = torch.tensor([1 if local_flag else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    return flag.item() > 0

_SHAPENET_VIEW_CHOICES = [
    torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]), torch.Tensor([1, -1, 1]), torch.Tensor([-1, 1, 1]),
    torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]), torch.Tensor([1, -1, -1]), torch.Tensor([-1, -1, -1]),
]

CROP_RATIO = {
    'easy': 1 / 4,
    'median': 1 / 2,
    'hard': 3 / 4,
}


def _apply_cudnn_config(config, logger=None):
    force_disable_cudnn = bool(getattr(config, 'FORCE_DISABLE_CUDNN', False))
    if force_disable_cudnn:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        print_log('[INFO] FORCE_DISABLE_CUDNN=True, disable cuDNN for stability.', logger=logger)


def _resolve_eval_mode(args, config):
    if getattr(args, 'mode', None) is not None:
        return args.mode
    return getattr(config, 'eval_mode', 'easy')


def _aggregate_category_metrics(category_metrics):
    test_metrics = AverageMeter(Metrics.names())
    for _, meter in category_metrics.items():
        test_metrics.update(meter.avg())
    return test_metrics


def _print_eval_results(category_metrics, test_metrics, logger, title='TEST RESULTS'):
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log(f'============================ {title} ============================', logger=logger)
    header = 'Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items) + '\t#ModelName\t'
    print_log(header, logger=logger)

    for taxonomy_id in category_metrics:
        msg = f'{taxonomy_id}\t{category_metrics[taxonomy_id].count(0)}\t'
        msg += '\t'.join(['%.3f ' % value for value in category_metrics[taxonomy_id].avg()])
        if taxonomy_id in shapenet_dict:
            msg += f'\t{shapenet_dict[taxonomy_id]}\t'
        else:
            msg += f'\t{taxonomy_id}\t'
        print_log(msg, logger=logger)

    msg = 'Overall\t\t' + '\t'.join(['%.3f ' % value for value in test_metrics.avg()])
    print_log(msg, logger=logger)


def _split_sap_data(data):
    """PCN/ABC batch: (partial, gt) or (partial, gt, gt_psr[, min, max])."""
    partial = data[0].cuda()
    gt = data[1].cuda()
    gt_psr = min_v = max_v = None
    if len(data) >= 3:
        gt_psr = data[2].cuda()
    if len(data) >= 5:
        min_v = data[3].cuda()
        max_v = data[4].cuda()
    return partial, gt, gt_psr, min_v, max_v


def evaluate(base_model, eval_dataloader, dataset_cfg, ChamferDisL1, ChamferDisL2,
             args, config, logger=None, eval_mode='easy', shapenet_protocol='benchmark',
             progress_writer=None, progress_interval=50, epoch=None):
    """Shared eval loop. PCN/ABC always use test split as-is.

    ShapeNet:
      - shapenet_protocol='train': random partial crop (same as training val, fast)
      - shapenet_protocol='benchmark': 8 fixed views + fps (--test paper protocol)
    """
    base_model.eval()

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    category_metrics = dict()
    n_samples = len(eval_dataloader)
    npoints = dataset_cfg._base_.N_POINTS
    dataset_name = dataset_cfg._base_.NAME
    _metrics = None

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(eval_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            if dataset_name in ['PCN', 'ABC']:
                partial, gt, gt_psr, min_v, max_v = _split_sap_data(data)
                ret = _safe_model_forward(
                    base_model, partial, logger=logger, gt=gt, gt_psr=gt_psr,
                    min_v=min_v, max_v=max_v, use_gt=_is_adapointr_sap(config),
                )
                coarse_points = ret[0]
                dense_points = ret[1]

                sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
                dense_loss_l1 = ChamferDisL1(dense_points, gt)
                dense_loss_l2 = ChamferDisL2(dense_points, gt)
                test_losses.update([
                    sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000,
                    dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000,
                ])
                _metrics = Metrics.get(dense_points, gt)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)
                _maybe_export_sap_mesh(ret, args, config, model_id, idx, logger=logger)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                if shapenet_protocol == 'train':
                    partial, _ = misc.seprate_point_cloud(
                        gt, npoints, [int(npoints * 1 / 4), int(npoints * 3 / 4)], fixed_points=None,
                    )
                    partial = partial.cuda()
                    partial_batches = [(partial,)]
                else:
                    num_crop = int(npoints * CROP_RATIO[eval_mode])
                    partial_batches = []
                    for item in _SHAPENET_VIEW_CHOICES:
                        partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points=item)
                        partial = misc.fps(partial, 2048)
                        partial_batches.append((partial,))

                for view_i, (partial,) in enumerate(partial_batches):
                    ret = _safe_model_forward(
                        base_model, partial, logger=logger, gt=gt, use_gt=_is_adapointr_sap(config),
                    )
                    coarse_points = ret[0]
                    dense_points = ret[1]

                    sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 = ChamferDisL1(dense_points, gt)
                    dense_loss_l2 = ChamferDisL2(dense_points, gt)
                    test_losses.update([
                        sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000,
                        dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000,
                    ])
                    _metrics = Metrics.get(dense_points, gt)
                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
                    _maybe_export_sap_mesh(
                        ret, args, config, model_id, idx, logger=logger, view_i=view_i,
                    )

            elif dataset_name == 'MVP':
                gt = data[1].cuda()
                partial = data[0].cuda()
                ret = _safe_model_forward(
                    base_model, partial, logger=logger, gt=gt, use_gt=_is_adapointr_sap(config),
                )
                coarse_points = ret[0]
                dense_points = ret[1]

                sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
                dense_loss_l1 = ChamferDisL1(dense_points, gt)
                dense_loss_l2 = ChamferDisL2(dense_points, gt)
                test_losses.update([
                    sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000,
                    dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000,
                ])
                _metrics = Metrics.get(dense_points, gt)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)
                _maybe_export_sap_mesh(ret, args, config, model_id, idx, logger=logger)

            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = _safe_model_forward(base_model, partial, logger=logger)
                dense_points = ret[1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                os.makedirs(target_path, exist_ok=True)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()],
                )
                _maybe_export_sap_mesh(ret, args, config, model_id, idx, logger=logger)
                continue
            else:
                raise NotImplementedError(f'Eval phase do not support {dataset_name}')

            if progress_writer is not None and idx % progress_interval == 0 and _metrics is not None:
                epoch_tag = f' epoch={epoch}' if epoch is not None else ''
                print_log(
                    f'Eval[{idx}/{n_samples}]{epoch_tag} Taxonomy={taxonomy_id} Sample={model_id} '
                    f'Losses={["%.4f" % l for l in test_losses.val()]} '
                    f'Metrics={["%.4f" % m for m in _metrics]}',
                    logger=logger,
                )

    test_metrics = _aggregate_category_metrics(category_metrics)
    return test_losses, test_metrics, category_metrics, dataset_name


def _train_loss_display(sparse_loss, dense_loss):
    """Build labeled loss string aligned with backward scale."""
    cd_coarse = sparse_loss.item() * 1000
    cd_dense = dense_loss.item() * 1000
    return (
        cd_coarse,
        cd_dense,
        f"CD(coarse={cd_coarse:.4f}, dense={cd_dense:.4f})",
    )


def _format_train_loss_avgs(avg_values):
    cd_coarse, cd_dense = avg_values[0], avg_values[1]
    return f"CD(coarse={cd_coarse:.4f}, dense={cd_dense:.4f})"

def _is_cudnn_internal_error(exc):
    msg = str(exc)
    return ('CUDNN_STATUS_INTERNAL_ERROR' in msg) or ('cuDNN error' in msg)

def _is_adapointr_sap(config):
    return str(getattr(getattr(config, "model", config), "NAME", "")) == "AdaPoinTrSAP"


def _maybe_export_sap_mesh(ret, args, config, model_id, idx, logger=None, view_i=0):
    if not _is_adapointr_sap(config):
        return
    if not bool(getattr(config.model, "export_sap_mesh", True)):
        return
    if not getattr(args, "test", False):
        return
    if len(ret) < 7:
        return
    from models.AdaPoinTrSAP import save_sap_mesh
    psr_grid = ret[4]
    sap_scale = float(getattr(config.model, "sap_scale", 0.0) or 0.0)
    mesh_dir = os.path.join(args.experiment_path, "vis_mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    out_path = os.path.join(mesh_dir, f"{model_id}_{idx:03d}_v{view_i:02d}.ply")
    try:
        if sap_scale > 0:
            save_sap_mesh(out_path, psr_grid, sap_scale=sap_scale)
        else:
            save_sap_mesh(out_path, psr_grid, min_v=ret[5], max_v=ret[6])
        print_log(f"[SAP mesh] {out_path}", logger=logger)
    except Exception as exc:
        print_log(f"[SAP mesh] skip {model_id}: {exc}", logger=logger)


def _safe_model_forward(
    base_model, partial, logger=None, gt=None, use_gt=False,
    gt_psr=None, min_v=None, max_v=None,
):
    global _CUDNN_FALLBACK_LOGGED
    if not torch.isfinite(partial).all():
        raise RuntimeError('Input partial contains NaN or Inf values.')

    def _call():
        if use_gt:
            return base_model(partial, gt=gt, gt_psr=gt_psr, min_v=min_v, max_v=max_v)
        return base_model(partial)

    try:
        return _call()
    except RuntimeError as exc:
        if not _is_cudnn_internal_error(exc):
            raise
        if not _CUDNN_FALLBACK_LOGGED:
            print_log('[WARN] cuDNN internal error detected, fallback to cudnn disabled for forward.', logger=logger)
            _CUDNN_FALLBACK_LOGGED = True
        with torch.backends.cudnn.flags(enabled=False):
            return _call()

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    _apply_cudnn_config(config, logger=logger)
    # Build Dataset
    (train_sampler, train_dataloader) = builder.dataset_builder(args, config.dataset.train)
    val_dataset_name = config.dataset.val._base_.NAME
    if val_dataset_name == 'ShapeNet':
        # ShapeNet34/55: keep distributed val loader (original training validation behavior)
        _, val_dataloader = builder.dataset_builder(args, config.dataset.val)
        print_log(
            '[INFO] ShapeNet validation keeps original distributed loader and train protocol.',
            logger=logger,
        )
    else:
        # PCN / ABC / MVP: rank0 full val loader so ckpt performance matches single-GPU --test
        if (not args.distributed) or args.local_rank == 0:
            _, val_dataloader = builder.dataset_builder(args, config.dataset.val, distributed=False)
            print_log(
                '[INFO] Validation uses full test split on rank 0 (non-distributed loader).',
                logger=logger,
            )
        else:
            val_dataloader = None
    # Build Model
    base_model = builder.model_builder(config.model)

    if args.use_gpu:
        base_model.to(args.local_rank)
        
    # Parameter Setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # Resume Ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    log_model_startup_info(base_model, config.model, logger=logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, \
                                                         device_ids=[args.local_rank % torch.cuda.device_count()], \
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        if torch.cuda.device_count() > 1:
            print_log('Using Data parallel ...' , logger = logger)
            base_model = nn.DataParallel(base_model).cuda()
        else:
            print_log('Using single GPU ...' , logger = logger)
            base_model = base_model.cuda()

    model_core = base_model.module if hasattr(base_model, 'module') else base_model

    # Optimizer & Scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # Training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = None

        num_iter = 0
        consecutive_nan_grad = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        dataset_name = config.dataset.train._base_.NAME
        npoints = config.dataset.train._base_.N_POINTS
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            
            
            gt_psr = min_v = max_v = None
            if dataset_name in ['PCN', 'ABC']:
                partial, gt, gt_psr, min_v, max_v = _split_sap_data(data)
                if dataset_name == 'PCN' and config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()

            elif dataset_name == 'MVP':
                gt = data[1].cuda()
                partial = data[0].cuda()
                
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
           
            model_name = str(getattr(config.model, "NAME", type(model_core).__name__))
            ret = _safe_model_forward(
                base_model, partial, logger=logger, gt=gt, use_gt=(model_name == "AdaPoinTrSAP"),
                gt_psr=gt_psr, min_v=min_v, max_v=max_v,
            )
            if model_name == "AdaPoinTr":
                sparse_loss, dense_loss = model_core.get_loss(ret, gt, epoch)
                loss_pcd = sparse_loss + dense_loss
            elif model_name == "AdaPoinTrSAP":
                loss_pcd, sparse_loss, dense_loss = model_core.get_loss(ret, gt, epoch)
            else:
                loss_pcd, sparse_loss, dense_loss = model_core.get_loss(ret, gt)
            _loss = loss_pcd

            bad_loss = _sync_any_rank_flag(not torch.isfinite(_loss), partial.device)
            if bad_loss:
                if (not args.distributed) or args.local_rank == 0:
                    print_log(
                        f'[NaN ERROR] non-finite loss at epoch={epoch}, batch={idx}, '
                        f'taxonomy={taxonomy_ids}, model={model_ids}, '
                        f'loss_pcd={loss_pcd}. '
                        f'Stop training to protect checkpoint.',
                        logger=logger,
                    )
                base_model.zero_grad()
                raise RuntimeError('Non-finite loss detected.')

            _loss.backward()

            if num_iter == config.step_per_update:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(),
                    GRAD_NORM_CLIP,
                    norm_type=2,
                )

                bad_grad = _sync_any_rank_flag(not torch.isfinite(total_norm), partial.device)
                if bad_grad:
                    consecutive_nan_grad += 1
                    if (not args.distributed) or args.local_rank == 0:
                        print_log(
                            f'[NaN WARN] non-finite grad norm at epoch={epoch}, batch={idx}, '
                            f'taxonomy={taxonomy_ids}, model={model_ids}, grad_norm={total_norm}, '
                            f'skip optimizer.step() ({consecutive_nan_grad}/{MAX_CONSECUTIVE_NAN_GRAD})',
                            logger=logger,
                        )
                    base_model.zero_grad()
                    num_iter = 0
                    if consecutive_nan_grad >= MAX_CONSECUTIVE_NAN_GRAD:
                        raise RuntimeError(
                            f'Non-finite grad norm for {consecutive_nan_grad} consecutive steps; '
                            'stop training to protect checkpoint.'
                        )
                    continue

                consecutive_nan_grad = 0
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)

            if model_name == "AdaPoinTrSAP":
                cd_coarse = sparse_loss.item() * 1000
                cd_dense = dense_loss.item()
                loss_msg = f"CD(recon={cd_coarse:.4f}) PSR={cd_dense:.6f}"
            else:
                cd_coarse, cd_dense, loss_msg = _train_loss_display(sparse_loss, dense_loss)
            if losses is None:
                losses = AverageMeter(['CD_coarse', 'CD_dense'])
            losses.update([cd_coarse, cd_dense])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            # if train_writer is not None:
            #     train_writer.add_scalar('Loss/Batch/CD_coarse', cd_coarse, n_itr)
            #     train_writer.add_scalar('Loss/Batch/CD_dense', cd_dense, n_itr)
            #     train_writer.add_scalar('LR/training', optimizer.param_groups[0]['lr'], n_itr)
            #     train_writer.add_scalar('Penalty/Batch/Orth_dict512', orth1, n_itr)
            #     train_writer.add_scalar('Penalty/Batch/Orth_dict256', orth2, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                print_log('[Memory: %f GB Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) %s lr = %.6f' %
                            (mem, epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            loss_msg, optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None and losses is not None:
            train_writer.add_scalar('Loss/Epoch/CD_coarse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/CD_dense', losses.avg(1), epoch)
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
        print_log('[Training] Memory: %f GB EPOCH: %d EpochTime = %.3f (s) %s' %
            (mem, epoch, epoch_end_time - epoch_start_time,
             _format_train_loss_avgs(losses.avg())), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            val_metrics = validate(
                base_model, val_dataloader, epoch, ChamferDisL1, ChamferDisL2,
                val_writer, args, config, logger=logger,
            )
            if val_metrics is not None:
                metrics = val_metrics
            if val_metrics is not None and val_metrics.better_than(best_metrics):
                best_metrics = val_metrics
                builder.save_checkpoint(
                    base_model, optimizer, epoch, val_metrics, best_metrics,
                    'ckpt-best', args, logger=logger,
                )
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        # Per-epoch numbered ckpts disabled to save disk; re-enable or set save_epoch_ckpt_freq if needed
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
        torch.cuda.empty_cache()
            
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

    
def validate(base_model, val_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=None):
    dataset_name_cfg = config.dataset.val._base_.NAME
    eval_mode = _resolve_eval_mode(args, config)
    use_full_rank0_val = dataset_name_cfg != 'ShapeNet'

    if use_full_rank0_val:
        # PCN / ABC / MVP: rank0 runs full test split on bare module; other ranks wait at barriers.
        if args.distributed:
            dist.barrier()

        if args.local_rank != 0:
            if args.distributed:
                dist.barrier()
            return None

        if val_dataloader is None:
            raise RuntimeError('Validation dataloader is missing on rank 0.')

        eval_model = base_model.module if hasattr(base_model, 'module') else base_model
        print_log(f'[VALIDATION] Start validating epoch {epoch}', logger=logger)

        test_losses, test_metrics, category_metrics, dataset_name = evaluate(
            eval_model, val_dataloader, config.dataset.val,
            ChamferDisL1, ChamferDisL2, args, config, logger=logger,
            eval_mode=eval_mode, shapenet_protocol='train',
            progress_writer=val_writer, progress_interval=args.val_interval, epoch=epoch,
        )

        if dataset_name == 'KITTI':
            if args.distributed:
                dist.barrier()
            return None

        print_log(
            '[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]),
            logger=logger,
        )
        _print_eval_results(category_metrics, test_metrics, logger, title='TEST RESULTS')

        if val_writer is not None:
            val_writer.add_scalar('Test/Epoch/Sparse', test_losses.avg(0), epoch)
            val_writer.add_scalar('Test/Epoch/Dense', test_losses.avg(2), epoch)
            for i, metric in enumerate(test_metrics.items):
                val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

        if args.distributed:
            dist.barrier()

        return Metrics(config.consider_metric, test_metrics.avg())

    # ShapeNet: all ranks validate with distributed loader + DDP forward (unchanged protocol).
    if val_dataloader is None:
        raise RuntimeError('ShapeNet validation dataloader is missing.')

    print_log(
        f'[VALIDATION] Start validating epoch {epoch} (ShapeNet train protocol)',
        logger=logger,
    )

    test_losses, test_metrics, category_metrics, dataset_name = evaluate(
        base_model, val_dataloader, config.dataset.val,
        ChamferDisL1, ChamferDisL2, args, config, logger=logger,
        eval_mode=eval_mode, shapenet_protocol='train',
        progress_writer=val_writer, progress_interval=args.val_interval, epoch=epoch,
    )

    if dataset_name == 'KITTI':
        return None

    print_log(
        '[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]),
        logger=logger,
    )
    _print_eval_results(category_metrics, test_metrics, logger, title='TEST RESULTS')

    if val_writer is not None and args.local_rank == 0:
        val_writer.add_scalar('Test/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Test/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


def test_net(args, config, test_writer=None):
    logger = get_logger(args.log_name)
    _apply_cudnn_config(config, logger=logger)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test, distributed=False)

    base_model = builder.model_builder(config.model)
    print_log(base_model, logger=logger)

    builder.load_model(base_model, args.ckpts, logger=logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        raise NotImplementedError()

    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, test_writer, args, config, logger=logger)


def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, test_writer, args, config, logger=None):
    eval_mode = _resolve_eval_mode(args, config)
    print_log(f'[TEST] eval_mode={eval_mode}', logger=logger)

    test_losses, test_metrics, category_metrics, dataset_name = evaluate(
        base_model, test_dataloader, config.dataset.test,
        ChamferDisL1, ChamferDisL2, args, config, logger=logger,
        eval_mode=eval_mode, shapenet_protocol='benchmark',
        progress_writer=test_writer, progress_interval=args.test_interval,
    )

    if dataset_name == 'KITTI':
        return

    print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)
    _print_eval_results(category_metrics, test_metrics, logger, title='TEST RESULTS')

