"""Gộp kết quả decoder theo protocol: mean macro-F1 qua 3 fold cho từng seed, rồi mean ± SD qua seed.

  python -m stage2_decoder.summarize --results downstream/stage2_decoder/results \
      --variants baseline current full --output downstream/stage2_decoder/summary.json

Báo thêm Δ(R* − C0), Δ(R* − C1) với bootstrap subject trong từng fold rồi lấy trung bình fold,
và so sánh giữa variant (vd. full − baseline) trước/sau decoder.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import AU_NAMES, f1

METHODS = ("C0", "C1", "R0", "R1", "R*")


def load(results, variants, folds, seeds):
    S = {}
    for v in variants:
        for k in folds:
            for s in seeds:
                p = Path(results) / v / f"fold{k}" / f"seed{s}"
                if (p / "decode_summary.json").exists():
                    d = json.loads((p / "decode_summary.json").read_text(encoding="utf-8"))
                    c = np.load(p / "test_subject_counts.npz")
                    d["counts"] = {m: c[m] for m in ("C0", "C1", "R0", "R1")}
                    d["counts"]["R*"] = d["counts"][d["r_star"]]
                    d["methods"]["R*"] = d["methods"][d["r_star"]]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--variants", nargs="+", default=["baseline", "current", "full"])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    S = load(args.results, args.variants, args.folds, args.seeds)
    if not S:
        raise SystemExit("Chưa có decode_summary.json nào.")
    rng = np.random.default_rng(0)
    report = {"per_variant": {}, "missing": [], "cross_variant": {}}
    for (v, k, s) in [(v, k, s) for v in args.variants for k in args.folds for s in args.seeds]:
        if (v, k, s) not in S:
            report["missing"].append(f"{v}/fold{k}/seed{s}")

    print("TEST MACRO-F1 (%) — mean 3 fold theo seed; cột cuối = mean ± SD qua seed hoàn chỉnh")
    for v in args.variants:
        full_seeds = [s for s in args.seeds if all((v, k, s) in S for k in args.folds)]
        if not full_seeds:
            print(f"{v}: chưa đủ 3 fold cho seed nào")
            continue
        rep = {"seeds": full_seeds, "methods": {}, "deltas": {}, "lambda": {}, "r_star_choice": {}}
        print(f"\n[{v}] seeds đủ 3 fold: {full_seeds}")
        for m in METHODS:
            per_seed = [100 * np.mean([S[(v, k, s)]["methods"][m]["test_f1"] for k in args.folds]) for s in full_seeds]
            per_au = 100 * np.mean([S[(v, k, s)]["methods"][m]["test_f1_au"] for k in args.folds for s in full_seeds], 0)
            sd = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None
            rep["methods"][m] = dict(per_seed=per_seed, mean=float(np.mean(per_seed)), sd=sd, per_au=per_au.tolist())
            print(f"  {m:3}" + "".join(f"{x:>8.2f}" for x in per_seed)
                  + f"   {np.mean(per_seed):6.2f}" + (f" ± {sd:.2f}" if sd is not None else "") + "   |"
                  + "".join(f"{x:>6.1f}" for x in per_au))
        print("      " + " " * (8 * len(full_seeds)) + "              |" + "".join(f"{a:>6}" for a in AU_NAMES))
        for c in ("C0", "C1"):
            per_seed = []
            for s in full_seeds:
                d = fold_mean_boot(S, v, s, args.folds, "R*", c, args.n_boot, rng)
                obs = np.mean([S[(v, k, s)]["methods"]["R*"]["test_f1"] - S[(v, k, s)]["methods"][c]["test_f1"]
                               for k in args.folds])
                per_seed.append(dict(seed=s, delta_pp=100 * obs, ci_lo=100 * np.percentile(d, 2.5),
                                     ci_hi=100 * np.percentile(d, 97.5), p_gt0=float((d > 0).mean())))
            rep["deltas"][f"R*-{c}"] = per_seed
            print(f"  Δ R* − {c}: " + " | ".join(
                f"seed{x['seed']} {x['delta_pp']:+.2f} [{x['ci_lo']:+.2f}, {x['ci_hi']:+.2f}] P>0={x['p_gt0']:.2f}"
                for x in per_seed))
        for k in args.folds:
            for s in full_seeds:
                d = S[(v, k, s)]
                rep["lambda"][f"fold{k}_seed{s}"] = d["methods"][d["r_star"]]["lambda_"]
                rep["r_star_choice"][f"fold{k}_seed{s}"] = d["r_star"]
        print("  R*/λ chọn trên val:", ", ".join(f"{key}={rep['r_star_choice'][key]}/{rep['lambda'][key]:g}"
                                            for key in rep["lambda"]))
        wins = all(x["delta_pp"] > 0 for c in ("C0", "C1") for x in rep["deltas"][f"R*-{c}"])
        strong = wins and all(x["p_gt0"] >= .9 for c in ("C0", "C1") for x in rep["deltas"][f"R*-{c}"])
        rep["verdict"] = ("TÍN HIỆU DƯƠNG" if strong else "CHƯA CHẮC" if wins else "KHÔNG CÓ TÍN HIỆU")
        print(f"  → {rep['verdict']} (R* thắng cả C0 và C1 ở mọi seed"
              + (" với P(Δ>0) ≥ 0.9)" if strong else ")" if wins else " — KHÔNG)"))
        report["per_variant"][v] = rep

    base = args.variants[0]
    for v in args.variants[1:]:
        if base in report["per_variant"] and v in report["per_variant"]:
            seeds = sorted(set(report["per_variant"][base]["seeds"]) & set(report["per_variant"][v]["seeds"]))
            row = {}
            for m in ("C0", "C1", "R*"):
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


if __name__ == "__main__":
    main()
