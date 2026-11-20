"""Gộp kết quả decoder theo protocol: mean macro-F1 qua 3 fold cho từng seed, rồi mean ± SD qua seed.

  python -m stage2_decoder.summarize --results <out>/results --variants baseline full --output <out>/summary.json

So sánh (bootstrap subject trong từng fold, ghép cặp, rồi lấy trung bình fold):
  chính  R0 − C0
  phụ    R0 − R_FACS (Q2a: nguồn luật), R_FACS − C0, R1 − C1, C1 − C0
Tiêu chí (đề cương §6): mean Δ(R0 − C0) qua seed > 0 VÀ CI95 không chứa 0 ở ít nhất 2/3 số seed.
Chỉ đọc decode_summary.json (đã mở test); decode_val.json (--val-only) không được gộp ở đây.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .core import AU_NAMES, f1
from .decode import COMPARISONS, PRIMARY, TEST_FILE

METHODS = ("C0", "C1", "R0", "R1", "R_FACS")


def load(results, variants, folds, seeds):
    S = {}
    for v in variants:
        for k in folds:
            for s in seeds:
                p = Path(results) / v / f"fold{k}" / f"seed{s}"
                if not (p / TEST_FILE).exists():
                    continue
                d = json.loads((p / TEST_FILE).read_text(encoding="utf-8"))
                missing = [m for m in METHODS if m not in d["methods"]]
                if missing:
                    raise ValueError(f"{p / TEST_FILE}: thiếu nhánh {missing} — chạy lại decode bằng bản mới")
                c = np.load(p / "test_subject_counts.npz")
                d["counts"] = {m: c[m] for m in METHODS}
                S[(v, k, s)] = d
    return S


def fold_mean_boot(S, v, seed, folds, a, b, n_boot, rng):
    """Δ = mean_fold F1(a) − mean_fold F1(b); resample subject trong từng fold, ghép cặp."""
    out = np.zeros(n_boot)
    for k in folds:
        ca, cb = S[(v, k, seed)]["counts"][a], S[(v, k, seed)]["counts"][b]
        n = len(ca)
        for i in range(n_boot):
            w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
            out[i] += (f1(np.tensordot(w, ca, 1)).mean() - f1(np.tensordot(w, cb, 1)).mean()) / len(folds)
    return out


def verdict(per_seed):
    """Đề cương §6: mean qua seed > 0 và CI không chứa 0 ở ≥ 2/3 số seed."""
    mean = float(np.mean([x["delta_pp"] for x in per_seed]))
    n_ci = sum(x["ci_lo"] > 0 for x in per_seed)
    need = math.ceil(2 * len(per_seed) / 3)
    if mean > 0 and n_ci >= need:
        return "ĐẠT"
    return "CHƯA CHẮC" if mean > 0 else "KHÔNG ĐẠT"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--variants", nargs="+", default=["baseline", "full"])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    S = load(args.results, args.variants, args.folds, args.seeds)
    if not S:
        raise SystemExit(f"Chưa có {TEST_FILE} nào.")
    rng = np.random.default_rng(0)
    report = {"per_variant": {}, "missing": [], "cross_variant": {}, "primary": f"{PRIMARY[0]}-{PRIMARY[1]}"}
    for (v, k, s) in [(v, k, s) for v in args.variants for k in args.folds for s in args.seeds]:
        if (v, k, s) not in S:
            report["missing"].append(f"{v}/fold{k}/seed{s}")

    print("TEST MACRO-F1 (%) — mean 3 fold theo seed; cột cuối = mean ± SD qua seed hoàn chỉnh")
    for v in args.variants:
        full_seeds = [s for s in args.seeds if all((v, k, s) in S for k in args.folds)]
        if not full_seeds:
            print(f"{v}: chưa đủ 3 fold cho seed nào")
            continue
        rep = {"seeds": full_seeds, "methods": {}, "deltas": {}, "lambda": {}}
        print(f"\n[{v}] seeds đủ 3 fold: {full_seeds}")
        for m in METHODS:
            per_seed = [100 * np.mean([S[(v, k, s)]["methods"][m]["test_f1"] for k in args.folds]) for s in full_seeds]
            per_au = 100 * np.mean([S[(v, k, s)]["methods"][m]["test_f1_au"] for k in args.folds for s in full_seeds], 0)
            sd = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None
            rep["methods"][m] = dict(per_seed=per_seed, mean=float(np.mean(per_seed)), sd=sd, per_au=per_au.tolist())
            print(f"  {m:7}" + "".join(f"{x:>8.2f}" for x in per_seed)
                  + f"   {np.mean(per_seed):6.2f}" + (f" ± {sd:.2f}" if sd is not None else "") + "   |"
                  + "".join(f"{x:>6.1f}" for x in per_au))
        print("  " + " " * (7 + 8 * len(full_seeds)) + "              |" + "".join(f"{a:>6}" for a in AU_NAMES))
        for a, b in COMPARISONS:
            per_seed = []
            for s in full_seeds:
                d = fold_mean_boot(S, v, s, args.folds, a, b, args.n_boot, rng)
                obs = np.mean([S[(v, k, s)]["methods"][a]["test_f1"] - S[(v, k, s)]["methods"][b]["test_f1"]
                               for k in args.folds])
                per_seed.append(dict(seed=s, delta_pp=100 * obs, ci_lo=100 * np.percentile(d, 2.5),
                                     ci_hi=100 * np.percentile(d, 97.5), p_gt0=float((d > 0).mean())))
            key = f"{a}-{b}"
            rep["deltas"][key] = per_seed
            print(f"  Δ {key:<10}{'(chính)' if (a, b) == PRIMARY else '':8}" + " | ".join(
                f"seed{x['seed']} {x['delta_pp']:+.2f} [{x['ci_lo']:+.2f}, {x['ci_hi']:+.2f}] P>0={x['p_gt0']:.2f}"
                for x in per_seed))
        for arm in ("R0", "R1", "R_FACS"):
            rep["lambda"][arm] = {f"fold{k}_seed{s}": S[(v, k, s)]["methods"][arm]["lambda_"]
                                  for k in args.folds for s in full_seeds}
            print(f"  λ {arm} chọn trên val: " + ", ".join(f"{key}={l:g}" for key, l in rep["lambda"][arm].items()))
        rep["verdict"] = verdict(rep["deltas"][f"{PRIMARY[0]}-{PRIMARY[1]}"])
        print(f"  → {PRIMARY[0]} − {PRIMARY[1]}: {rep['verdict']}"
              + (f"  (mới {len(full_seeds)} seed — sơ bộ)" if len(full_seeds) < 3 else ""))
        report["per_variant"][v] = rep

    base = args.variants[0]
    for v in args.variants[1:]:
        if base in report["per_variant"] and v in report["per_variant"]:
            seeds = sorted(set(report["per_variant"][base]["seeds"]) & set(report["per_variant"][v]["seeds"]))
            row = {}
            for m in ("C0", "R0", "R_FACS"):
                row[m] = [100 * np.mean([S[(v, k, s)]["methods"][m]["test_f1"] - S[(base, k, s)]["methods"][m]["test_f1"]
                                         for k in args.folds]) for s in seeds]
            report["cross_variant"][f"{v}-{base}"] = dict(seeds=seeds, **row)
            print(f"\n{v} − {base} (điểm %, theo seed {seeds}): "
                  + " | ".join(f"{m}: " + ", ".join(f"{x:+.2f}" for x in row[m]) for m in row))
    if report["missing"]:
        print(f"\nThiếu {len(report['missing'])} run: {', '.join(report['missing'])}")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SAVED] {args.output}")
    return report


if __name__ == "__main__":
    main()
