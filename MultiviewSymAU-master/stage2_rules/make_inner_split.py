"""Chia tập train của mỗi fold thành `fit` / `inner` theo subject (không chạm outer test).

Ghi ra data/<DS>/list/:
  <DS>_fit_img_path_fold{k}.txt,   <DS>_fit_label_fold{k}.txt,   <DS>_fit_weight_fold{k}.txt
  <DS>_inner_img_path_fold{k}.txt, <DS>_inner_label_fold{k}.txt
và  <DS>_inner_split.json (subject của từng phần, tỉ lệ dương mỗi AU).

Chọn nhóm inner trong n_draws lần bốc ngẫu nhiên sao cho tỉ lệ dương của từng AU
ở inner gần với fit nhất (và mỗi AU có đủ frame dương ở inner).

Chạy từ thư mục gốc repo:
  python -m stage2_rules.make_inner_split --dataset DISFA --n-inner 6
"""
import argparse
import json
import os

import numpy as np

from .common import AU_NAMES, list_file, read_split


def write_split(root, ds, split, fold, paths, labels):
    with open(list_file(root, ds, split, "img_path", fold), "w", encoding="utf-8") as f:
        f.write("\n".join(paths) + "\n")
    np.savetxt(list_file(root, ds, split, "label", fold), labels, fmt="%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="DISFA")
    ap.add_argument("--root", default=None, help="mặc định data/<dataset>")
    ap.add_argument("--n-inner", type=int, default=6, help="số subject đưa vào inner (DISFA: 6/18)")
    ap.add_argument("--n-draws", type=int, default=2000)
    ap.add_argument("--min-inner-pos", type=int, default=100,
                    help="mỗi AU phải có ít nhất chừng này frame dương ở inner")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = args.dataset
    root = args.root or os.path.join("data", ds)
    names = AU_NAMES[ds]
    summary = {}

    for fold in (1, 2, 3):
        paths, labels, subjects = read_split(root, ds, "train", fold)
        subj = sorted(str(s) for s in set(subjects))
        pos = np.stack([labels[subjects == s].sum(0) for s in subj])      # (S, A)
        nfr = np.array([(subjects == s).sum() for s in subj])             # (S,)

        rng = np.random.default_rng(args.seed + fold)
        best, best_score = None, np.inf
        for _ in range(args.n_draws):
            pick = rng.choice(len(subj), size=args.n_inner, replace=False)
            m = np.zeros(len(subj), bool)
            m[pick] = True
            pos_in, pos_fit = pos[m].sum(0), pos[~m].sum(0)
            if (pos_in < args.min_inner_pos).any() or (pos_fit < args.min_inner_pos).any():
                continue
            r_in, r_fit = pos_in / nfr[m].sum(), pos_fit / nfr[~m].sum()
            score = np.abs(np.log((r_in + 1e-4) / (r_fit + 1e-4))).mean()
            if score < best_score:
                best, best_score = m, score
        if best is None:
            raise RuntimeError(f"fold{fold}: không tìm được cách chia thoả min-inner-pos; giảm ngưỡng.")

        inner_subj = [s for s, b in zip(subj, best) if b]
        is_inner = np.isin(subjects, inner_subj)
        fit_paths = [p for p, b in zip(paths, is_inner) if not b]
        inn_paths = [p for p, b in zip(paths, is_inner) if b]
        fit_lab, inn_lab = labels[~is_inner], labels[is_inner]

        write_split(root, ds, "fit", fold, fit_paths, fit_lab)
        write_split(root, ds, "inner", fold, inn_paths, inn_lab)

        # Trọng số WAL cùng công thức tool/<DS>_calculate_AU_class_weights.py, nhưng chỉ từ fit
        rate = (fit_lab > 0).mean(0)
        w = 1.0 / rate
        w = w / w.sum() * w.shape[0]
        np.savetxt(list_file(root, ds, "fit", "weight", fold), w[None], fmt="%f", delimiter="\t")

        summary[f"fold{fold}"] = {
            "fit_subjects": [s for s in subj if s not in inner_subj],
            "inner_subjects": inner_subj,
            "n_fit": int(len(fit_paths)), "n_inner": int(len(inn_paths)),
            "pos_rate_fit": dict(zip(names, np.round(fit_lab.mean(0), 4).tolist())),
            "pos_rate_inner": dict(zip(names, np.round(inn_lab.mean(0), 4).tolist())),
            "balance_score": float(best_score),
        }
        print(f"[fold{fold}] inner = {inner_subj} | fit {len(fit_paths)} frame, inner {len(inn_paths)} frame")
        print("   AU      fit%   inner%")
        for a, rf, ri in zip(names, fit_lab.mean(0), inn_lab.mean(0)):
            print(f"   {a:<6}{100*rf:6.2f}  {100*ri:6.2f}")

    out = os.path.join(root, "list", f"{ds}_inner_split.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
