"""Chạy detector Stage 1 (đóng băng) trên các split và lưu xác suất AU ra .npz.

Decoder ở bước sau chỉ làm việc với file .npz này, nên mỗi split chỉ cần chạy model 1 lần.

  python -m stage2_rules.dump_probs --dataset DISFA --arc resnet50 --fold 2 ^
      --ckpt results/<exp_fit>/bs_64_seed_0_lr_0.0001/best_model_fold2.pth --splits inner test

Đầu ra: results/stage2_rules/<DS>/fold{k}/probs_<split>.npz  với
  probs (N, A) float32, labels (N, A) int8, subjects (N,), paths (N,)
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.SymStage1 import MEFARGStage1
from utils import image_test, load_state_dict

from .common import read_split


class ListDataset(Dataset):
    def __init__(self, img_root, paths, transform):
        self.img_root, self.paths, self.transform = img_root, paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, k):
        p = os.path.normpath(os.path.join(self.img_root, self.paths[k]))
        with open(p, "rb") as f:
            img = Image.open(f).convert("RGB")
        return self.transform(img), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="DISFA")
    ap.add_argument("--root", default=None, help="mặc định đọc dataset_path từ config/<DS>_config.yaml")
    ap.add_argument("--arc", default="resnet50")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", nargs="+", default=["inner", "test"])
    ap.add_argument("--crop-size", type=int, default=224)
    ap.add_argument("-b", "--batch-size", type=int, default=64)
    ap.add_argument("-j", "--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", default=os.path.join("results", "stage2_rules"))
    args = ap.parse_args()

    ds = args.dataset
    if args.root is None:
        with open(os.path.join("config", f"{ds}_config.yaml")) as f:
            cfg = yaml.safe_load(f)
        root, num_aus = cfg["dataset_path"], cfg["num_classes"]
    else:
        root, num_aus = args.root, (8 if ds == "DISFA" else 12)

    net = MEFARGStage1(num_aus=num_aus, backbone=args.arc, num_expr=7, use_contrastive=False)
    net = load_state_dict(net, args.ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device).eval()
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    out = os.path.join(args.out_dir, ds, f"fold{args.fold}")
    os.makedirs(out, exist_ok=True)

    for split in args.splits:
        paths, labels, subjects = read_split(root, ds, split, args.fold)
        loader = DataLoader(ListDataset(os.path.join(root, "img"), paths, image_test(crop_size=args.crop_size)),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        probs = np.zeros(labels.shape, np.float32)
        with torch.no_grad():
            for x, idx in tqdm(loader, desc=f"{split} fold{args.fold}"):
                _, _, p_a, _ = net(x.to(device))          # p_a đã là xác suất (sigmoid trong AUHead)
                probs[idx.numpy()] = p_a.float().cpu().numpy()
        f = os.path.join(out, f"probs_{split}.npz")
        np.savez_compressed(f, probs=probs, labels=labels, subjects=subjects,
                            paths=np.array(paths), ckpt=args.ckpt)
        print(f"[SAVED] {f}  ({len(paths)} frame)")


if __name__ == "__main__":
    main()
