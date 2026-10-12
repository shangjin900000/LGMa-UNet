# LGMa-UNet

**Local–Global Mamba U-Net for Point Cloud Completion**

LGMa-UNet completes partial point clouds with a U-Net backbone that combines:

- **Geo-SA** — set abstraction with local geometric descriptors
- **RoLGM** — randomized-order Local–Global Mamba (Vim BiMamba) for serial modeling
- **U-Net decoder** with skip connections and Dec-RoLGM refinement

---

## Environment

- Ubuntu 20.04 LTS
- Python 3.9.25
- PyTorch 1.13.1
- CUDA 11.7

```bash
pip install -r requirements.txt
pip install ./extensions/chamfer_dist
pip install ./extensions/pointnet2_ops_lib
pip install -e ./extensions/mamba-1p1p1
```

---

## Datasets

**PCN:** [PCN-PyTorch Dataset](https://github.com/qinglew/PCN-PyTorch#dataset) — Please reference `render` and `sample` to create your own dataset. Also, we decompressed all `.lmdb` data from PCN data into `.ply` data which has smaller volume 8.1G and upload it into Google Drive. Here is the shared link: [Google Drive](https://drive.google.com/file/d/1OvvRyx02-C_DkzYiJ5stpin0mnXydHQ7/view?usp=sharing).

**ShapeNet / KITTI:** Download datasets following [PoinTr DATASET.md](https://github.com/yuxumin/PoinTr/blob/master/DATASET.md).

---

## Pretrained Models

Pretrained checkpoints: [Google Drive](TBD)

---

## Training

All commands from the `LGMa-UNet` root (2× GPU example):

```bash
# PCN
bash scripts/train_pcn.sh

# ShapeNet-55 / ShapeNet-34
bash scripts/train_shapenet55.sh
bash scripts/train_shapenet34.sh

# KITTI (train on PCN Cars; optional --start_ckpts for PCN init)
bash scripts/train_kitti.sh
```

Or call `main.py` directly (see scripts for full flags).

---

## Testing

```bash
# PCN / ShapeNet / KITTI
bash scripts/test.sh ./cfgs/PCN_models/UpTrans_PCN.yaml ./pretrained/UpTrans_PCN.pth my_test

# ShapeNet difficulty modes: easy | median | hard
bash scripts/test.sh ./cfgs/ShapeNet55_models/UpTrans_ShapeNet55.yaml ./pretrained/UpTrans_ShapeNet55.pth sn55 hard
```

### KITTI paper metrics (Fidelity / MMD)

```bash
# 1) Dump predictions
bash scripts/test.sh ./cfgs/KITTI_models/UpTrans_KITTI.yaml ./pretrained/UpTrans_KITTI.pth KITTI_eval

# 2) Metric script (vis_result under experiments/...)
CUDA_VISIBLE_DEVICES=0 python KITTI_metric.py \
  --vis_path ./experiments/UpTrans_KITTI/KITTI_models/test_KITTI_eval/vis_result
```

---

## Acknowledgement

This codebase builds on [ODGNet](https://github.com/corecai163/ODGNet), [PoinTr](https://github.com/yuxumin/PoinTr), and [Vim](https://github.com/hustvl/Vim). We thank the authors for their open-source contributions.

---

## Citation

If you use this work, please cite (update with your final paper):

```bibtex
@inproceedings{LGMaUNet,
  title={LGMa-UNet: Local-Global Mamba U-Net for Point Cloud Completion},
  author={...},
  booktitle={...},
  year={2026}
}
```
