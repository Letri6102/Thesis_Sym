"""Stage 2 v2 — bước 2.1 + 2.2: sinh luật ứng viên AU–AU và kiểm định bằng bootstrap theo subject.

Chỉ đọc NHÃN của tập `fit` (không đọc ảnh, không đọc inner/test).

Hai loại luật, cùng một thước đo "lift của mẫu vi phạm":
  exclusion   ¬AUi ∨ ¬AUj : vi phạm = (i=1, j=1),  E = n_i·n_j / N
  implication  AUi → AUj  : vi phạm = (i=1, j=0),  E = n_i·(N−n_j) / N
  lift_v = (O + 0.5) / (E + 0.5),  O = số frame vi phạm thực tế
lift_v < 1  ⇔ mẫu vi phạm hiếm hơn mức độc lập. Trọng số w0 = −log UCB95(lift_v)
(PMI bảo thủ của mẫu vi phạm), chặn trên bởi --w-max.

Nhận luật khi đồng thời:
  E_total ≥ --min-expected                      (đủ lực kiểm định — tránh nhầm "hiếm" với "loại trừ")
  UCB95(lift_v) < --ucb-max                     (bằng chứng vững qua bootstrap subject)
  tỉ lệ subject có lift_v < 1 ≥ --min-subject-frac (chỉ xét subject có E_s ≥ --min-subject-expected)
  implication: LCB95(P(j|i)) ≥ --conf-min

Đầu ra (--out-dir/fold{k}/):
  rules_all.csv       mọi ứng viên + thống kê (mở được bằng Excel)
  rules.json          luật được nhận: type, i, j, w0 (+ cấu hình chạy)
  V.npy               ma trận vi phạm (2^A, R) theo thứ tự cấu hình của common.all_configs

  python -m stage2_rules.build_rules --dataset DISFA --fold 2
"""
import argparse
import csv
import json
import os

import numpy as np

from .common import AU_NAMES, OLD_EXCLUSION_DISFA, all_configs, read_split, violation_matrix


def candidate_rules(A):
    rules = []
    for i in range(A):
        for j in range(i + 1, A):
            rules.append({"type": "exclusion", "i": i, "j": j})
    for i in range(A):
        for j in range(A):
            if i != j:
                rules.append({"type": "implication", "i": i, "j": j})
    return rules


def rule_stats(n, c, co, ri, rj, is_excl):
    """Vector hoá cho mọi luật. n: (...,), c: (..., A), co: (..., A, A).
    Trả O, E, conf (mỗi cái (..., R))."""
    ci, cj = c[..., ri], c[..., rj]
    cij = co[..., ri, rj]
    nn = n[..., None].astype(np.float64)
    O = np.where(is_excl, cij, ci - cij).astype(np.float64)
    E = np.where(is_excl, ci * cj / np.maximum(nn, 1), ci * (nn - cj) / np.maximum(nn, 1))
    conf = cij / np.maximum(ci, 1)                       # P(j|i), chỉ dùng cho implication
    return O, E, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="DISFA")
    ap.add_argument("--root", default=None)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--split", default="fit", help="tập dùng để ước lượng luật (mặc định fit)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ucb-max", type=float, default=0.8)
    ap.add_argument("--min-expected", type=float, default=50.0)
    ap.add_argument("--min-subject-frac", type=float, default=0.7)
    ap.add_argument("--min-subject-expected", type=float, default=5.0)
    ap.add_argument("--conf-min", type=float, default=0.6)
    ap.add_argument("--w-max", type=float, default=5.0)
    ap.add_argument("--no-implication", action="store_true", help="chỉ dùng luật loại trừ")
    ap.add_argument("--out-dir", default=os.path.join("results", "stage2_rules"))
    args = ap.parse_args()

    if args.split in ("test", "inner"):
        raise SystemExit("Không ước lượng luật trên inner/test — dùng fit (hoặc train khi chỉ xem thử).")

    ds = args.dataset
    root = args.root or os.path.join("data", ds)
    names = AU_NAMES[ds]
    _, Y, subjects = read_split(root, ds, args.split, args.fold)
    A = Y.shape[1]
    assert A == len(names), f"số cột nhãn {A} ≠ {len(names)}"

    subj = sorted(set(subjects))
    S = len(subj)
    n_s = np.array([(subjects == s).sum() for s in subj])                            # (S,)
    c_s = np.stack([Y[subjects == s].sum(0) for s in subj]).astype(np.float64)        # (S, A)
    co_s = np.stack([Y[subjects == s].T.astype(np.int64) @ Y[subjects == s]
                     for s in subj]).astype(np.float64)                                # (S, A, A)

    rules = candidate_rules(A)
    if args.no_implication:
        rules = [r for r in rules if r["type"] == "exclusion"]
    ri = np.array([r["i"] for r in rules])
    rj = np.array([r["j"] for r in rules])
    is_excl = np.array([r["type"] == "exclusion" for r in rules])

    # ---- toàn bộ fit ----
    O, E, conf = rule_stats(n_s.sum(), c_s.sum(0), co_s.sum(0), ri, rj, is_excl)
    lift = (O + 0.5) / (E + 0.5)

    # ---- bootstrap theo subject ----
    rng = np.random.default_rng(args.seed + args.fold)
    M = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(args.n_boot)]).astype(np.float64)
    Ob, Eb, confb = rule_stats(M @ n_s, M @ c_s, np.einsum("bs,sij->bij", M, co_s), ri, rj, is_excl)
    lift_b = (Ob + 0.5) / (Eb + 0.5)
    lift_lo, lift_hi = np.percentile(lift_b, [2.5, 97.5], axis=0)
    conf_lo = np.percentile(confb, 2.5, axis=0)

    # ---- nhất quán giữa các subject ----
    Os, Es, _ = rule_stats(n_s, c_s, co_s, ri, rj, is_excl)                           # (S, R)
    eligible = Es >= args.min_subject_expected
    below = ((Os + 0.5) / (Es + 0.5) < 1.0) & eligible
    n_eval = eligible.sum(0)
    subj_frac = np.where(n_eval > 0, below.sum(0) / np.maximum(n_eval, 1), 0.0)

    accepted = ((E >= args.min_expected) & (lift_hi < args.ucb_max)
                & (subj_frac >= args.min_subject_frac)
                & (is_excl | (conf_lo >= args.conf_min)))
    w0 = np.where(accepted, np.minimum(args.w_max, -np.log(np.maximum(lift_hi, 1e-12))), 0.0)

    # ---- ghi file ----
    out = os.path.join(args.out_dir, ds, f"fold{args.fold}")
    os.makedirs(out, exist_ok=True)
    c_all, co_all = c_s.sum(0), co_s.sum(0)
    rows = []
    for k, r in enumerate(rules):
        a, b = names[r["i"]], names[r["j"]]
        if r["type"] == "exclusion":
            text, support = f"¬{a} ∨ ¬{b}", c_all[r["i"]] + c_all[r["j"]] - co_all[r["i"], r["j"]]
            old = ds == "DISFA" and ((a, b) in OLD_EXCLUSION_DISFA or (b, a) in OLD_EXCLUSION_DISFA)
        else:
            text, support, old = f"{a} → {b}", c_all[r["i"]], False
        rows.append({
            "rule": text, "type": r["type"], "au_i": a, "au_j": b,
            "n_i": int(c_all[r["i"]]), "n_j": int(c_all[r["j"]]), "n_both": int(co_all[r["i"], r["j"]]),
            "support": int(support), "exceptions": int(O[k]), "expected": round(float(E[k]), 1),
            "lift": round(float(lift[k]), 4), "lift_ci_lo": round(float(lift_lo[k]), 4),
            "lift_ci_hi": round(float(lift_hi[k]), 4),
            "conf_j_given_i": round(float(conf[k]), 4) if r["type"] == "implication" else "",
            "conf_ci_lo": round(float(conf_lo[k]), 4) if r["type"] == "implication" else "",
            "subj_frac": round(float(subj_frac[k]), 3), "n_subj_eval": int(n_eval[k]),
            "accepted": int(accepted[k]), "w0": round(float(w0[k]), 4), "old_rule": int(old),
        })
    rows.sort(key=lambda d: (-d["accepted"], d["lift_ci_hi"]))
    with open(os.path.join(out, "rules_all.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    acc_rules = []
    for k in np.flatnonzero(accepted):
        r = rules[k]
        a, b = names[r["i"]], names[r["j"]]
        acc_rules.append({
            "type": r["type"], "i": int(r["i"]), "j": int(r["j"]), "w0": float(w0[k]),
            "rule": f"¬{a} ∨ ¬{b}" if r["type"] == "exclusion" else f"{a} → {b}",
        })
    V = violation_matrix(all_configs(A), acc_rules) if acc_rules else np.zeros((2 ** A, 0), np.int8)
    np.save(os.path.join(out, "V.npy"), V)
    with open(os.path.join(out, "rules.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": ds, "fold": args.fold, "split": args.split, "au_names": names,
                   "n_frames": int(n_s.sum()), "n_subjects": S, "config": vars(args),
                   "rules": acc_rules}, f, indent=2, ensure_ascii=False)

    print(f"[fold{args.fold}] {args.split}: {n_s.sum()} frame, {S} subject | "
          f"{len(rules)} ứng viên → nhận {len(acc_rules)} luật")
    print(f"  {'luật':<16}{'ngoại lệ':>9}{'kỳ vọng':>9}{'lift':>7}{'CI95':>16}{'%subj':>7}{'w0':>7}")
    for d in rows:
        if d["accepted"]:
            print(f"  {d['rule']:<16}{d['exceptions']:>9}{d['expected']:>9.0f}{d['lift']:>7.2f}"
                  f"   [{d['lift_ci_lo']:.2f}, {d['lift_ci_hi']:.2f}]{d['subj_frac']:>7.2f}{d['w0']:>7.2f}")
    if ds == "DISFA":
        rej_old = [d["rule"] for d in rows if d["old_rule"] and not d["accepted"]]
        if rej_old:
            print(f"  Luật tay cũ bị loại: {', '.join(rej_old)}")
    print(f"[SAVED] {out}/rules_all.csv, rules.json, V.npy")


if __name__ == "__main__":
    main()
