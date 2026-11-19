"""Gộp kết quả decode_eval của các fold.

  python -m stage2_rules.summarize --dataset DISFA
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="DISFA")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--base", default=os.path.join("results", "stage2_rules"))
    args = ap.parse_args()

    S = {}
    for k in args.folds:
        f = os.path.join(args.base, args.dataset, f"fold{k}", "decode_summary.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                S[k] = json.load(fh)
        else:
            print(f"[thiếu] {f}")
    if not S:
        return
    names = next(iter(S.values()))["au_names"]
    print(f"{'':4}" + "".join(f"{'fold'+str(k):>9}" for k in S) + f"{'mean':>9}  " + "".join(f"{a:>6}" for a in names))
    for m in ("C0", "C1", "R0", "R1"):
        f1 = [100 * S[k]["methods"][m]["test_f1"] for k in S]
        au = np.mean([S[k]["methods"][m]["test_f1_au"] for k in S], 0) * 100
        print(f"{m:4}" + "".join(f"{x:>9.2f}" for x in f1) + f"{np.mean(f1):>9.2f}  " + "".join(f"{x:>6.1f}" for x in au))
    print()
    for k in S:
        r = S[k]["r_star"]
        dd = [S[k]["deltas"][f"{r}-{c}"] for c in ("C0", "C1")]
        print(f"fold{k}: {S[k]['n_rules']} luật, chọn {r} (λ={S[k]['methods'][r]['lambda']:g}) | "
              f"Δ vs C0 {100*dd[0]['delta']:+.2f} (P>0 {dd[0]['p_gt0']:.2f}) | "
              f"Δ vs C1 {100*dd[1]['delta']:+.2f} (P>0 {dd[1]['p_gt0']:.2f})")
    wins = sum(all(S[k]["deltas"][f"{S[k]['r_star']}-{c}"]["delta"] > 0 for c in ("C0", "C1")) for k in S)
    print(f"\nSố fold luật thắng cả hai đối chứng: {wins}/{len(S)}")


if __name__ == "__main__":
    main()
