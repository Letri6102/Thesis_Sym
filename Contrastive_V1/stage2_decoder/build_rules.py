"""Bước 2.1 + 2.2: luật AU–AU kiểm định bằng bootstrap theo subject, CHỈ từ nhãn của fit.

Bảng luật chỉ phụ thuộc nhãn fit của fold → dùng chung cho mọi variant/seed trong fold đó.

  python -m stage2_decoder.build_rules --data-root /data/DISFA --manifest splits/fold1.json \
      --output downstream/stage2_decoder/rules

Thước đo chung cho hai loại luật — lift của mẫu vi phạm:
  exclusion   ¬AUi ∨ ¬AUj : vi phạm (1,1)  E = n_i n_j / N
  implication  AUi → AUj  : vi phạm (1,0)  E = n_i (N − n_j) / N
  lift = (O + 0.5) / (E + 0.5);  w0 = min(w_max, −log UCB95(lift))
Nhận luật khi: E ≥ min_expected, UCB95 < ucb_max, ≥ min_subject_frac subject cùng chiều,
và (implication) LCB95 P(j|i) ≥ conf_min.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .core import AU_NAMES, OLD_EXCLUSION, all_configs, candidate_rules, digest, load_split, rule_stats, \
    rule_text, violation_matrix


def build(fit, args, seed):
    Y, subjects = fit["labels"], fit["subjects"]
    subj = sorted(set(subjects))
    S = len(subj)
    n_s = np.array([(subjects == s).sum() for s in subj])
    c_s = np.stack([Y[subjects == s].sum(0) for s in subj]).astype(np.float64)
    co_s = np.stack([Y[subjects == s].T.astype(np.int64) @ Y[subjects == s] for s in subj]).astype(np.float64)

    rules = candidate_rules(not args.no_implication)
    ri, rj = np.array([r["i"] for r in rules]), np.array([r["j"] for r in rules])
    is_excl = np.array([r["type"] == "exclusion" for r in rules])

    O, E, conf = rule_stats(n_s.sum(), c_s.sum(0), co_s.sum(0), ri, rj, is_excl)
    lift = (O + .5) / (E + .5)

    rng = np.random.default_rng(seed)
    M = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(args.n_boot)]).astype(np.float64)
    Ob, Eb, cb = rule_stats(M @ n_s, M @ c_s, np.einsum("bs,sij->bij", M, co_s), ri, rj, is_excl)
    lo, hi = np.percentile((Ob + .5) / (Eb + .5), [2.5, 97.5], axis=0)
    conf_lo = np.percentile(cb, 2.5, axis=0)

    Os, Es, _ = rule_stats(n_s, c_s, co_s, ri, rj, is_excl)
    elig = Es >= args.min_subject_expected
    n_eval = elig.sum(0)
    frac = np.where(n_eval > 0, (((Os + .5) / (Es + .5) < 1) & elig).sum(0) / np.maximum(n_eval, 1), 0.)

    acc = (E >= args.min_expected) & (hi < args.ucb_max) & (frac >= args.min_subject_frac) \
        & (is_excl | (conf_lo >= args.conf_min))
    w0 = np.where(acc, np.minimum(args.w_max, -np.log(np.maximum(hi, 1e-12))), 0.)

    c_all, co_all = c_s.sum(0), co_s.sum(0)
    rows = []
    for k, r in enumerate(rules):
        a, b = AU_NAMES[r["i"]], AU_NAMES[r["j"]]
        excl = r["type"] == "exclusion"
        rows.append(dict(
            rule=rule_text(r), type=r["type"], au_i=a, au_j=b,
            n_i=int(c_all[r["i"]]), n_j=int(c_all[r["j"]]), n_both=int(co_all[r["i"], r["j"]]),
            support=int(c_all[r["i"]] + c_all[r["j"]] - co_all[r["i"], r["j"]] if excl else c_all[r["i"]]),
            exceptions=int(O[k]), expected=round(float(E[k]), 1), lift=round(float(lift[k]), 4),
            lift_ci_lo=round(float(lo[k]), 4), lift_ci_hi=round(float(hi[k]), 4),
            conf_j_given_i="" if excl else round(float(conf[k]), 4),
            conf_ci_lo="" if excl else round(float(conf_lo[k]), 4),
            subj_frac=round(float(frac[k]), 3), n_subj_eval=int(n_eval[k]),
            accepted=int(acc[k]), w0=round(float(w0[k]), 4),
            old_rule=int(excl and ((a, b) in OLD_EXCLUSION or (b, a) in OLD_EXCLUSION))))
    rows.sort(key=lambda d: (-d["accepted"], d["lift_ci_hi"]))
    accepted = [dict(type=rules[k]["type"], i=int(rules[k]["i"]), j=int(rules[k]["j"]),
                     w0=float(w0[k]), rule=rule_text(rules[k])) for k in np.flatnonzero(acc)]
    return rows, accepted, dict(n_frames=int(n_s.sum()), n_subjects=S, subjects=subj)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True, help="thư mục gốc; ghi vào <output>/fold{k}/")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ucb-max", type=float, default=.8)
    ap.add_argument("--min-expected", type=float, default=50.)
    ap.add_argument("--min-subject-frac", type=float, default=.7)
    ap.add_argument("--min-subject-expected", type=float, default=5.)
    ap.add_argument("--conf-min", type=float, default=.6)
    ap.add_argument("--w-max", type=float, default=5.)
    ap.add_argument("--no-implication", action="store_true", help="chỉ luật loại trừ")
    ap.add_argument("--force", action="store_true", help="ghi đè bảng luật đã có")
    args = ap.parse_args(argv)

    sp = load_split(args.manifest, args.data_root)
    fold = sp["fold"]
    out = Path(args.output) / f"fold{fold}"
    if (out / "rules.json").exists() and not args.force:
        raise SystemExit(f"Đã có {out/'rules.json'} — bảng luật phải cố định cho mọi variant; dùng --force nếu chủ đích tạo lại.")
    out.mkdir(parents=True, exist_ok=True)

    rows, accepted, info = build(sp["fit"], args, args.seed + fold)
    with (out / "rules_all.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.save(out / "V.npy", violation_matrix(all_configs(), accepted))
    meta = dict(fold=fold, source="fit labels only", manifest=str(args.manifest), manifest_hash=digest(args.manifest),
                config={k: v for k, v in vars(args).items() if k not in ("data_root", "output", "force")},
                au_names=AU_NAMES, **info, rules=accepted)
    (out / "rules.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[fold{fold}] fit: {info['n_frames']} frame, {info['n_subjects']} subject → nhận {len(accepted)} luật")
    print(f"  {'luật':<16}{'ngoại lệ':>9}{'kỳ vọng':>9}{'lift':>7}{'CI95':>17}{'%subj':>7}{'w0':>7}")
    for d in rows:
        if d["accepted"]:
            print(f"  {d['rule']:<16}{d['exceptions']:>9}{d['expected']:>9.0f}{d['lift']:>7.2f}"
                  f"   [{d['lift_ci_lo']:.2f}, {d['lift_ci_hi']:.2f}]{d['subj_frac']:>7.2f}{d['w0']:>7.2f}")
    old_rej = [d["rule"] for d in rows if d["old_rule"] and not d["accepted"]]
    if old_rej:
        print("  Luật tay cũ bị loại:", ", ".join(old_rej))
    print(f"[SAVED] {out}/rules_all.csv, rules.json, V.npy")


if __name__ == "__main__":
    main()
