"""ABC completion dataset with optional mesh-precomputed Poisson fields (DMC-style)."""

import os
import random

import numpy as np
import torch
import torch.utils.data as data

import data_transforms
from .io import IO
from .build import DATASETS
from utils.logger import print_log


@DATASETS.register_module()
class ABC(data.Dataset):
    def __init__(self, config):
        self.data_root = os.path.expanduser(config.DATA_ROOT)
        self.partial_dir = config.PARTIAL_DIR
        self.complete_dir = config.COMPLETE_DIR
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.n_renderings = config.N_RENDERINGS
        self.psr_dir = getattr(config, "PSR_DIR", None)
        self.sap_scale = float(getattr(config, "SAP_SCALE", 0.0) or 0.0)
        if self.psr_dir:
            if not os.path.isabs(self.psr_dir):
                self.psr_dir = os.path.join(self.data_root, self.psr_dir)

        list_file = getattr(config, "LIST_FILE", None)
        if list_file is None:
            if self.subset == "train":
                list_file = "trainset.txt"
            elif self.subset == "test":
                list_file = "testset.txt"
                if not os.path.exists(os.path.join(self.data_root, list_file)):
                    list_file = "valset.txt"
            else:
                list_file = "valset.txt"
        list_path = os.path.join(self.data_root, list_file)

        with open(list_path, "r") as f:
            shape_names = [line.strip() for line in f if line.strip()]

        self.file_list = []
        for name in shape_names:
            partial_paths = [
                os.path.join(self.data_root, self.partial_dir, "%s.ply_%d.ply" % (name, i))
                for i in range(self.n_renderings)
            ]
            gt_path = os.path.join(self.data_root, self.complete_dir, "%s.ply" % name)
            sample = {
                "taxonomy_id": "abc",
                "model_id": name,
                "partial_path": partial_paths,
                "gt_path": gt_path,
            }
            if self.psr_dir:
                sample["psr_path"] = os.path.join(self.psr_dir, "%s.npz" % name)
            self.file_list.append(sample)

        print_log(
            "ABC %s split loaded: %d samples (%s) psr=%s sap_scale=%s"
            % (self.subset, len(self.file_list), self.data_root, bool(self.psr_dir), self.sap_scale or "off"),
            logger="ABCDATASET",
        )
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        # PSR fields are axis-aligned grids; do not mirror points without warping the field.
        ops = [{
            "callback": "RandomSamplePoints",
            "parameters": {"n_points": 2048},
            "objects": ["partial"],
        }]
        if subset == "train" and not self.psr_dir:
            ops.append({
                "callback": "RandomMirrorPoints",
                "objects": ["partial", "gt"],
            })
        ops.append({"callback": "ToTensor", "objects": ["partial", "gt"]})
        return data_transforms.Compose(ops)

    def _resample_to_npoints(self, pts):
        if pts.shape[0] == self.npoints:
            return pts
        if pts.shape[0] >= self.npoints:
            choice = np.random.choice(pts.shape[0], self.npoints, replace=False)
            return pts[choice]
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        return pts[choice]

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset == "train" else 0
        partial_path = sample["partial_path"][rand_idx]
        gt_path = sample["gt_path"]

        partial = IO.get(partial_path).astype(np.float32)[:, :3]
        gt = IO.get(gt_path).astype(np.float32)[:, :3]
        gt = self._resample_to_npoints(gt)

        data = {"partial": partial, "gt": gt}
        if self.transforms is not None:
            data = self.transforms(data)

        packed = [data["partial"], data["gt"]]
        if self.psr_dir:
            psr_path = sample["psr_path"]
            if not os.path.isfile(psr_path):
                raise FileNotFoundError(
                    "Missing ABC Poisson field %s. Run SAP/scripts/process_abc_psr.py first."
                    % os.path.abspath(psr_path)
                )
            psr_pack = np.load(psr_path)
            gt_psr = torch.from_numpy(np.asarray(psr_pack["psr"], dtype=np.float32))
            if "scale" in psr_pack.files and self.sap_scale > 0:
                saved = float(np.asarray(psr_pack["scale"]).reshape(-1)[0])
                if abs(saved - self.sap_scale) > 1e-6:
                    raise RuntimeError(
                        "PSR scale %.4f in %s != dataset SAP_SCALE %.4f"
                        % (saved, psr_path, self.sap_scale)
                    )
            packed.append(gt_psr)

        return sample["taxonomy_id"], sample["model_id"], tuple(packed)

    def __len__(self):
        return len(self.file_list)
