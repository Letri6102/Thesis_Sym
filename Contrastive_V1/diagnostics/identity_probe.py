"""Probe danh tính: một lớp tuyến tính đoán subject từ đặc trưng đóng băng của detector đã train.

  python -m diagnostics.identity_probe --checkpoint <run>/best.pt --manifest <Sym>/splits_v2/fold2.json \
      --data-root /content/DISFA [--output identity_probe.json]

Chỉ dùng frame của FIT (các subject train) với transform của test (Resize 256 + CenterCrop 224, không augmentation).
Trong mỗi subject: lấy mỗi `stride` frame theo số thứ tự frame; 70% đầu để train probe, 30% cuối để đo, bỏ một
đoạn 5% ở giữa để hai phía không có frame gần như trùng nhau (hai frame liền nhau cách 50 ms).

Ba bộ đặc trưng:
- `trunk`: token đầu ra của global_linear lấy trung bình, tức đầu vào chung của mọi head;
- `pooled`: V_a.mean(1), vector đi vào proj_global (nút Global);
- `au_concat`: V_a của 8 AU nối lại.
Độ chính xác càng gần mức ngẫu nhiên 1/S thì đặc trưng càng ít mang danh tính. NLL trên phần đo bão hoà chậm hơn
độ chính xác, nên dùng được cả khi độ chính xác chạm 100%.

Probe cố định từ trước, không tune: chuẩn hoá theo phần train, hồi quy logistic đa lớp, L2 = 1e-3, L-BFGS 300 vòng.
Không đọc val hay test: probe cần nhiều subject, và subject test không được đụng tới.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

TRAIN_FRAC, GAP_FRAC, L2, ITERS, STRIDE = .7, .05, 1e-3, 300, 4


def frame_number(path):
    return int(Path(path).stem)


def fit_frames(root, manifest, stride=STRIDE):
    """(paths, subjects, frames) của các frame fit được chọn, sắp theo subject rồi theo số frame."""
    from experiment_data import read_partition
    if stride < 1:
        raise ValueError("stride >= 1")
    paths, _, _ = read_partition(root, manifest["fold"], "train")
    by = {}
    for i in manifest["fit_indices"]:
        by.setdefault(paths[i].split("/")[0], []).append(i)
    keep = [i for s in sorted(by) for i in sorted(by[s], key=lambda i: frame_number(paths[i]))[::stride]]
    chosen = [paths[i] for i in keep]
    return chosen, np.array([p.split("/")[0] for p in chosen]), np.array([frame_number(p) for p in chosen])


def chrono_split(subjects, frames, train_frac=TRAIN_FRAC, gap_frac=GAP_FRAC):
    train, test = [], []
    for s in sorted(set(subjects.tolist())):
        idx = np.flatnonzero(subjects == s)
        idx = idx[np.argsort(frames[idx], kind="stable")]
        n = len(idx)
        a, g = int(round(train_frac * n)), int(round(gap_frac * n))
        train += idx[:a].tolist()
        test += idx[min(n, a + g):].tolist()
    return np.array(train, dtype=np.int64), np.array(test, dtype=np.int64)


def linear_probe(X, y, train, test, l2=L2, iters=ITERS):
    classes, yi = np.unique(np.asarray(y).astype(str), return_inverse=True)
    if set(yi[test].tolist()) - set(yi[train].tolist()):
        raise ValueError("Có subject chỉ nằm ở phần đo")
    X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    yt = torch.as_tensor(yi, dtype=torch.long)
    tr, te = torch.as_tensor(train), torch.as_tensor(test)
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
    Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    C = len(classes)
    W = torch.zeros(X.shape[1], C, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(C, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Ztr @ W + b, yt[tr]) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        logits, yte = Zte @ W + b, yt[te]
        pred = logits.argmax(1)
        per = {str(classes[c]): float((pred[yte == c] == c).double().mean()) for c in range(C) if bool((yte == c).any())}
        return dict(acc=float((pred == yte).double().mean()), balanced_acc=float(np.mean(list(per.values()))),
                    nll=float(F.cross_entropy(logits, yte)),
                    train_acc=float(((Ztr @ W + b).argmax(1) == yt[tr]).double().mean()),
                    chance=1 / C, nll_chance=float(np.log(C)), n_classes=C, n_train=len(train), n_test=len(test),
                    per_subject_acc=per)


class Frames(Dataset):
    """Ảnh theo danh sách đường dẫn, transform của test (không augmentation)."""

    def __init__(self, root, paths, crop_size=224):
        from utils import image_test
        self.root, self.paths, self.transform = Path(root), list(paths), image_test(crop_size=crop_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / "img" / self.paths[i]) as img:
            return self.transform(img.convert("RGB"))


def extract(model, root, paths, batch_size, workers, device):
    model.eval().to(device)
    feats = dict(trunk=[], pooled=[], au_concat=[])
    hook = model.global_linear.register_forward_hook(lambda m, i, o: feats["trunk"].append(o.mean(1).float().cpu()))
    try:
        with torch.no_grad():
            for x in DataLoader(Frames(root, paths), batch_size=batch_size, shuffle=False, num_workers=workers):
                va = model(x.to(device))[0]
                feats["pooled"].append(va.mean(1).float().cpu())
                feats["au_concat"].append(va.flatten(1).float().cpu())
    finally:
        hook.remove()
    return {k: torch.cat(v).numpy() for k, v in feats.items()}


def run(checkpoint, manifest, data_root, stride=STRIDE, batch_size=64, workers=4, device=None):
    from experiment_data import load_manifest
    from .common import check_manifest, load_model

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, meta = load_model(checkpoint, "eval")
    check_manifest(cfg, manifest)
    m = load_manifest(manifest, data_root)
    paths, subjects, frames = fit_frames(data_root, m, stride)
    feats = extract(model, data_root, paths, batch_size, workers, device)
    del model
    train, test = chrono_split(subjects, frames)
    return dict(checkpoint=Path(checkpoint).name, epoch=meta["epoch"], variant=cfg["variant"], fold=m["fold"],
                seed=cfg["seed"], split="fit", stride=stride, train_frac=TRAIN_FRAC, gap_frac=GAP_FRAC, l2=L2,
                iters=ITERS, subjects=sorted(set(subjects.tolist())), n_frames=len(paths),
                probes={k: linear_probe(v, subjects, train, test) for k, v in feats.items()})


def main(argv=None):
    from .common import write_json
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device")
    ap.add_argument("--output")
    args = ap.parse_args(argv)
    r = run(args.checkpoint, args.manifest, args.data_root, args.stride, args.batch_size, args.workers, args.device)
    for k, p in r["probes"].items():
        print(f"{k:<10} acc {100 * p['acc']:5.1f}%  (ngẫu nhiên {100 * p['chance']:.1f}%)  "
              f"NLL {p['nll']:.3f} (ngẫu nhiên {p['nll_chance']:.3f})")
    if args.output:
        write_json(args.output, r)
    return r


if __name__ == "__main__":
    main()
