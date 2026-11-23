"""Bảng tổng hợp CHỈ VAL: Tầng 1 (run đã train, đúng khoá) → Tầng 2 (C0/C1/R0/R1/R_FACS) cho mọi run đã có kết quả.

  python -m stage2_decoder.val_overview --runs <Sym>/runs_aug1 --results <Sym>/stage2_v2/results \
      [--output <Sym>/stage2_v2/val_overview.json]

Chạy được khi mới có một phần trong 18 run; run chưa xong, lệch khoá hoặc chưa có Tầng 2 được liệt kê riêng.
λ và ngưỡng C1 được chọn trên chính val, nên số ở đây chỉ để xem chiều hướng: KHÔNG báo cáo, KHÔNG so với 63.0
(63.0 là test 3 fold). Không đọc thư mục evaluation/ hay decode_summary.json, tức là không đụng test.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .contrast_ci import BASELINE, CANDIDATE, LOCK, jobs, run_dir, run_problems
from .decode import ARMS, VAL_FILE
from .run_all import is_current

ARM_NAMES = [a for a, *_ in ARMS]


def collect(runs, results, lock=LOCK):
    rows, notes = [], []
    for v, k, s in jobs():
        run, tag = run_dir(runs, v, k, s), f"{v}/fold{k}/seed{s}"
        res = Path(results) / v / f"fold{k}" / f"seed{s}" / VAL_FILE
        if not (run / "config.json").exists():
            continue
        bad = run_problems(run, v, k, s, lock)
        if bad:
            notes.append(f"{tag}: chưa xong hoặc lệch khoá ({'; '.join(bad)})")
        elif not res.exists():
            notes.append(f"{tag}: chưa có Tầng 2 (chạy run_all --val-only)")
        elif not is_current(res, run):
            notes.append(f"{tag}: kết quả Tầng 2 làm từ checkpoint khác (chạy lại run_all --val-only)")
        else:
            d = json.loads(res.read_text(encoding="utf-8"))
            env = run / "environment.json"
            rows.append(dict(variant=v, fold=k, seed=s, epoch=d.get("selected_epoch"), n_rules=d["n_rules"],
                             f1={a: d["methods"][a]["val_f1"] for a in ARM_NAMES},
                             lambda_r0=d["methods"]["R0"]["lambda_"], au_c0=d["methods"]["C0"]["val_f1_au"],
                             au_names=d["au_names"],
                             gpu=json.loads(env.read_text(encoding="utf-8")).get("gpu") if env.exists() else None))
    return rows, notes


def summarize(rows):
    by = {(r["variant"], r["fold"], r["seed"]): r for r in rows}
    means = {}
    for v in (BASELINE, CANDIDATE):
        rs = [r for r in rows if r["variant"] == v]
        if rs:
            means[v] = dict(n=len(rs), **{a: float(np.mean([r["f1"][a] for r in rs])) for a in ARM_NAMES},
                            **{f"{a}-C0": float(np.mean([r["f1"][a] - r["f1"]["C0"] for r in rs]))
                               for a in ARM_NAMES if a != "C0"},
                            au_c0=np.mean([r["au_c0"] for r in rs], axis=0).tolist())
    pairs = [(k, s) for _, k, s in jobs() if (BASELINE, k, s) in by and (CANDIDATE, k, s) in by]
    pairs = sorted(set(pairs))
    paired = dict(n=len(pairs), per_pair=[dict(fold=k, seed=s, **{a: by[CANDIDATE, k, s]["f1"][a] - by[BASELINE, k, s]["f1"][a]
                                                                    for a in ("C0", "R0")}) for k, s in pairs])
    if pairs:
        paired.update({f"mean_{a}": float(np.mean([p[a] for p in paired["per_pair"]])) for a in ("C0", "R0")})
    return dict(n_runs=len(rows), of=len(jobs()), means=means, full_minus_baseline=paired,
                gpus=sorted({str(r["gpu"]) for r in rows}))


def report(rows, notes, summ):
    p = lambda x: f"{100 * x:6.1f}"  # noqa: E731
    print("=" * 100)
    print(f"TỔNG HỢP CHỈ VAL — {summ['n_runs']}/{summ['of']} run có đủ Tầng 1 + Tầng 2 | GPU train: {', '.join(summ['gpus']) or '-'}")
    print("λ và ngưỡng C1 chọn trên chính val: chỉ xem chiều hướng. KHÔNG báo cáo, KHÔNG so với 63.0 (test 3 fold).")
    print("=" * 100)
    print(f"{'run':<22}{'epoch':>6}{'luật':>6}" + "".join(f"{a:>8}" for a in ARM_NAMES) + f"{'R0−C0':>8}{'λ R0':>7}")
    for r in sorted(rows, key=lambda r: (r["seed"], r["fold"], r["variant"])):
        print(f"{r['variant'] + '/f' + str(r['fold']) + '/s' + str(r['seed']):<22}{r['epoch'] or '-':>6}{r['n_rules']:>6}"
              + "".join(f"{p(r['f1'][a]):>8}" for a in ARM_NAMES)
              + f"{100 * (r['f1']['R0'] - r['f1']['C0']):>+8.2f}{r['lambda_r0']:>7g}")
    if summ["means"]:
        print("\nTrung bình theo variant (trên các run đã có; chưa đủ 3 fold × 3 seed thì chưa cân bằng):")
        for v, m in summ["means"].items():
            print(f"  {v:<9} n={m['n']:<2}" + "".join(f"  {a} {p(m[a])}" for a in ARM_NAMES)
                  + f"  | R0−C0 {100 * m['R0-C0']:+.2f}  C1−C0 {100 * m['C1-C0']:+.2f}")
        names = rows[0]["au_names"]
        print("  F1 từng AU của C0:" + "".join(f"{a:>7}" for a in names))
        for v, m in summ["means"].items():
            print(f"  {v:<18}" + "".join(f"{p(x):>7}" for x in m["au_c0"]))
    fb = summ["full_minus_baseline"]
    if fb["n"]:
        print(f"\nfull − baseline theo cặp cùng fold/seed (n={fb['n']}):  C0 {100 * fb['mean_C0']:+.2f}  R0 {100 * fb['mean_R0']:+.2f}")
        for q in fb["per_pair"]:
            print(f"  fold{q['fold']}/seed{q['seed']}:  C0 {100 * q['C0']:+.2f}  R0 {100 * q['R0']:+.2f}")
        print("  Câu trả lời chính thức cho full − baseline là CI trên test 3 fold × 3 seed (contrast_ci), không phải bảng này.")
    if notes:
        print("\nChưa tính:\n  " + "\n  ".join(notes))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--results", required=True, help="<output của run_all>/results")
    ap.add_argument("--output", help="ghi bảng ra JSON")
    args = ap.parse_args(argv)
    rows, notes = collect(args.runs, args.results)
    summ = summarize(rows)
    report(rows, notes, summ)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(dict(summary=summ, rows=rows, notes=notes, val_only=True), indent=2,
                                                ensure_ascii=False), encoding="utf-8")
    return summ


if __name__ == "__main__":
    main()
