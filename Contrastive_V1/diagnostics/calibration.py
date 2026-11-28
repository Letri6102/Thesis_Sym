"""Hiệu chỉnh trên inner val, chỉ numpy: AUC theo AU và ngưỡng tối ưu riêng của từng subject val.

  python -m diagnostics.calibration --run <Sym>/runs_aug1/full/fold2/seed0 [--output calibration.json]

Đọc <run>/evaluation_val/val_predictions.npz (do cell L ghi). Không đọc thư mục test.
- auc_pooled[a]: AUC của AU a trên toàn bộ val, không phụ thuộc ngưỡng.
- Với mỗi subject val và mỗi AU có ≥ MIN_POS frame dương và ≥ MIN_POS frame âm: ngưỡng t* cho F1 cao nhất trên
  lưới của C1 (0.05–0.95, bước 0.01; hoà thì lấy ngưỡng nhỏ nhất, như stage2_decoder.core.tune_thresholds),
  cùng F1 ở 0.5 và F1 ở t*.
- thr_sd[a]: độ lệch chuẩn (ddof = 0) của t* giữa các subject đủ điều kiện; cần ≥ 2 subject.
- oracle_gain[a]: trung bình theo subject của F1(t*) − F1(0.5), tức phần F1 mất chỉ vì hiệu chỉnh lệch theo người.
Val chỉ có 3 subject, nên các số này dùng để so cặp baseline/full trên cùng frame, không phải ước lượng tổng thể.
"""
import argparse
from pathlib import Path

import numpy as np

from stage2_decoder.core import AU_NAMES, counts, f1, load_predictions, prediction_dir

GRID = np.round(np.arange(.05, .95 + 1e-9, .01), 2)
MIN_POS = 20


def auc(y, p):
    """AUC = xác suất một frame dương có điểm cao hơn một frame âm, hoà tính 1/2. NaN khi thiếu một lớp."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=np.float64)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(cnt) - (cnt - 1) / 2               # hạng trung bình (từ 1) của mỗi giá trị
    r = avg_rank[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def f1_curve(y, p, grid=GRID):
    y = np.asarray(y).astype(bool)[:, None]
    pred = np.asarray(p)[:, None] >= grid[None, :]
    tp, fp, fn = (pred & y).sum(0), (pred & ~y).sum(0), (~pred & y).sum(0)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def best_threshold(y, p, grid=GRID):
    c = f1_curve(y, p, grid)
    i = int(np.argmax(c))                                    # hoà → phần tử đầu = ngưỡng nhỏ nhất
    return float(grid[i]), float(c[i])


def f1_at(y, p, t=.5):
    return float(f1(counts((np.asarray(p) >= t)[:, None], np.asarray(y)[:, None]))[0])


def _nanmean(v):
    v = np.asarray(v, dtype=np.float64)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def calibration(probs, labels, subjects, min_pos=MIN_POS, grid=GRID):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int8)
    subjects = np.asarray(subjects).astype(str)
    n_au = probs.shape[1]
    subs = sorted(set(subjects.tolist()))
    per = {}
    for s in subs:
        m = subjects == s
        ys, ps = labels[m], probs[m]
        row = dict(n_frames=int(m.sum()), n_pos=[], thr=[], f1_at_05=[], f1_at_best=[], auc=[])
        for a in range(n_au):
            n1 = int(ys[:, a].sum())
            row["n_pos"].append(n1)
            row["auc"].append(auc(ys[:, a], ps[:, a]))
            if n1 >= min_pos and len(ys) - n1 >= min_pos:
                t, fb = best_threshold(ys[:, a], ps[:, a], grid)
                row["thr"].append(t)
                row["f1_at_best"].append(fb)
                row["f1_at_05"].append(f1_at(ys[:, a], ps[:, a]))
            else:
                row["thr"].append(None)
                row["f1_at_best"].append(None)
                row["f1_at_05"].append(None)
        per[s] = row

    thr_sd, thr_range, gain, n_elig = [], [], [], []
    for a in range(n_au):
        ok = [s for s in subs if per[s]["thr"][a] is not None]
        ts = [per[s]["thr"][a] for s in ok]
        n_elig.append(len(ok))
        thr_sd.append(float(np.std(ts)) if len(ts) >= 2 else float("nan"))
        thr_range.append(float(max(ts) - min(ts)) if len(ts) >= 2 else float("nan"))
        gain.append(_nanmean([per[s]["f1_at_best"][a] - per[s]["f1_at_05"][a] for s in ok]) if ok else float("nan"))
    auc_pooled = [auc(labels[:, a], probs[:, a]) for a in range(n_au)]
    pooled_f1 = f1(counts(probs >= .5, labels))             # >= như train_contrastive.metrics
    return dict(
        au_names=AU_NAMES if n_au == len(AU_NAMES) else [f"AU{i}" for i in range(n_au)],
        subjects=subs, min_pos=min_pos, grid=[float(grid[0]), float(grid[-1]), float(round(grid[1] - grid[0], 6))],
        macro_f1_at_05=float(pooled_f1.mean()), f1_at_05=pooled_f1.tolist(),
        auc_pooled=auc_pooled, auc_subject_mean=[_nanmean([per[s]["auc"][a] for s in subs]) for a in range(n_au)],
        thr_sd=thr_sd, thr_range=thr_range, oracle_gain=gain, n_subjects_eligible=n_elig, per_subject=per,
        summary=dict(macro_auc=_nanmean(auc_pooled), mean_thr_sd=_nanmean(thr_sd), mean_oracle_gain=_nanmean(gain),
                     n_au_with_thr_sd=int(np.isfinite(thr_sd).sum())))


def run_calibration(run):
    npz = prediction_dir(run, "val") / "val_predictions.npz"
    d = load_predictions(npz)
    out = calibration(d["probs"], d["labels"], d["subjects"])
    out["source"] = str(Path(npz).relative_to(run)).replace("\\", "/")
    return out


def main(argv=None):
    from .common import write_json
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--output")
    args = ap.parse_args(argv)
    r = run_calibration(Path(args.run))
    s = r["summary"]
    print(f"AUC {100 * s['macro_auc']:.1f} | SD ngưỡng t* giữa subject {s['mean_thr_sd']:.3f} "
          f"({s['n_au_with_thr_sd']} AU) | F1(t*) − F1(0.5) {100 * s['mean_oracle_gain']:+.1f}")
    if args.output:
        write_json(args.output, r)
    return r


if __name__ == "__main__":
    main()
