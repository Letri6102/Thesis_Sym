"""Train Stage 1 y hệt train_Sym_Stage_1.py, nhưng:
  - train trên split `fit`  (thay vì toàn bộ `train`)
  - chọn best checkpoint trên split `inner` (thay vì outer test)
  - trọng số WAL lấy từ <DS>_fit_weight_fold{k}.txt
Không sửa file gốc: script này import train_Sym_Stage_1 rồi thay get_dataloader.

  python train_Sym_Stage_1_fit.py --dataset DISFA --arc resnet50 --exp-name resnet50_stage1_fit ^
      -b 64 -lr 0.0001 --fold 2
(Có thể truyền mọi tham số khác của train_Sym_Stage_1.py; thêm --train-split / --val-split nếu cần.)
"""
import argparse
import os
import random
import sys

import numpy as _np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import train_Sym_Stage_1 as t1
from conf import get_config, set_env, set_logger, set_outdir
from stage2_rules.common import read_split
from utils import image_test, image_train

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--train-split", default="fit")
_ap.add_argument("--val-split", default="inner")
SPLIT, _ = _ap.parse_known_args()


class SplitDataset(Dataset):
    """Giống DISFA/BP4D trong dataset.py (stage=1) nhưng đọc split tuỳ ý."""

    def __init__(self, root, ds, split, fold, train, crop_size):
        self.paths, labels, _ = read_split(root, ds, split, fold)
        self.labels = labels.astype(_np.float64)
        self.img_root = os.path.join(root, "img")
        self.train, self.crop_size = train, crop_size
        self.tf = image_train(crop_size=crop_size) if train else image_test(crop_size=crop_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, k):
        p = os.path.normpath(os.path.join(self.img_root, self.paths[k]))
        with open(p, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.train:
            w, h = img.size
            oy, ox = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
            img = self.tf(img, random.randint(0, 1), ox, oy)
        else:
            img = self.tf(img)
        return img, self.labels[k]


def get_dataloader(conf):
    print(f"==> Preparing data (train={SPLIT.train_split}, val={SPLIT.val_split})...")
    tr = SplitDataset(conf.dataset_path, conf.dataset, SPLIT.train_split, conf.fold, True, conf.crop_size)
    va = SplitDataset(conf.dataset_path, conf.dataset, SPLIT.val_split, conf.fold, False, conf.crop_size)
    return (DataLoader(tr, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers),
            DataLoader(va, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers),
            len(tr), len(va))


class _NpProxy:
    """train_Sym_Stage_1.main đọc cứng <DS>_train_weight_fold{k}.txt → chuyển sang split fit."""

    def __getattr__(self, name):
        return getattr(_np, name)

    def loadtxt(self, fname, *a, **k):
        fname = str(fname).replace("_train_weight_fold", f"_{SPLIT.train_split}_weight_fold")
        return _np.loadtxt(fname, *a, **k)


t1.get_dataloader = get_dataloader
t1.np = _NpProxy()

if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    set_outdir(conf)
    set_logger(conf)
    t1.main(conf)
