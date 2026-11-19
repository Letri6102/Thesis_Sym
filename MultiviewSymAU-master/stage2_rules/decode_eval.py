"""Decoder cố định dùng bảng luật — so với hai đối chứng không luật, trên outer test.

Mọi lựa chọn (ngưỡng, λ) chỉ dựa trên inner. Test chỉ dùng để báo cáo.

Phương pháp
  C0  ngưỡng 0.5                              (không luật)
  C1  ngưỡng riêng từng AU tối ưu trên inner   (không luật)
  R0  MAP có luật, bias 0,  λ chọn trên inner
  R1  MAP có luật, bias = ngưỡng của C1, λ chọn trên inner
MAP chính xác trên toàn bộ 2^A cấu hình:
  y* = argmax_y  Σ_i y_i·(logit p_i + b_i)  −  λ·Σ_r w_r·V[y, r]
λ = 0 nằm trong lưới ⇒ R0 ≡ C0, R1 ≡ C1 khi luật không giúp được gì trên inner.

  python -m stage2_rules.decode_eval --dataset DISFA --fold 2
"""
import argparse
import csv
import json
import os

import numpy as np

from .common import AU_NAMES, all_configs, counts, f1_from_counts, macro_f1, violation_matrix

EPS = 1e-6


def logit(p):
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


class Decoder:
    def __init__(self, A, rules):
        self.Y = all_configs(A).astype(np.float64)                     # (C, A)
        self.V = violation_matrix(all_configs(A), rules).astype(np.float64)  # (C, R)
        self.w = np.array([r["w0"] for r in rules], np.float64)
        self.pen = self.V @ self.w if len(rules) else np.zeros(len(self.Y))

    def decode(self, L, lam, chunk=8192):
        """L: (N, A) logit + bias. Trả (N, A) int8 và chỉ số cấu hình (N,)."""
        idx = np.empty(len(L), np.int64)
        for s in range(0, len(L), chunk):
            sc = L[s:s + chunk] @ self.Y.T - lam * self.pen[None]
            idx[s:s + chunk] = sc.argmax(1)
        return self.Y[idx].astype(np.int8), idx

    def config_index(self, y):
        """(N, A) 0/1 -> chỉ số cấu hình (A bit, AU đầu là bit cao nhất, khớp all_configs)."""
        A = y.shape[1]
        return (y.astype(np.int64) * (1 << np.arange(A - 1, -1, -1))).sum(1)


def tune_thresholds(p, y, grid):
    thr = np.zeros(p.shape[1])
    for a in range(p.shape[1]):
        best, bt = -1.0, 0.5
        for t in grid:
            tp, fp, fn = counts((p[:, a] >= t)[:, None], y[:, a:a + 1])
            f = f1_from_counts(tp, fp, fn)[0]
            if f > best + 1e-12:
                best, bt = f, t
        thr[a] = bt
    return thr


def per_subject_counts(pred, y, subjects, subj_list):
    out = np.zeros((len(subj_list), 3, y.shape[1]))
    for k, s in enumerate(subj_list):
        m = subjects == s
        out[k] = np.stack(counts(pred[m], y[m]))
    return out


def boot_delta(cnt_a, cnt_b, n_boot, rng):
    """Bootstrap theo subject cho Δ macro-F1 = F1(a) − F1(b), ghép cặp."""
    S = cnt_a.shape[0]
    d = np.empty(n_boot)
    for b in range(n_boot):
        m = np.bincount(rng.integers(0, S, S), minlength=S).astype(np.float64)
        ca, cb = np.tensordot(m, cnt_a, 1), np.tensordot(m, cnt_b, 1)
        d[b] = f1_from_counts(*ca).mean() - f1_from_counts(*cb).mean()
    return d


def rule_diagnostics(dec, pred_idx, y):
    """help: dự đoán vi phạm luật mà nhãn thật thoả; hurt: cả hai cùng vi phạm."""
    if dec.V.shape[1] == 0:
        return {"pred_violation_rate": 0.0, "gt_violation_rate": 0.0, "help": 0, "hurt": 0}
    vp = dec.V[pred_idx] > 0                     # (N, R)
    vg = dec.V[dec.config_index(y)] > 0
    return {
        "pred_violation_rate": float(vp.any(1).mean()),
        "gt_violation_rate": float(vg.any(1).mean()),
        "help": int((vp & ~vg).sum()),
        "hurt": int((vp & vg).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="DISFA")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--dir", default=None, help="mặc định results/stage2_rules/<DS>/fold{k}")
    ap.add_argument("--rules", default=None, help="mặc định <dir>/rules.json")
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8])
    ap.add_argument("--thr-grid", type=float, nargs=3, default=[0.05, 0.95, 0.01], metavar=("LO", "HI", "STEP"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = args.dataset
    names = AU_NAMES[ds]
    d = args.dir or os.path.join("results", "stage2_rules", ds, f"fold{args.fold}")
    with open(args.rules or os.path.join(d, "rules.json"), encoding="utf-8") as f:
        rules = json.load(f)["rules"]
    inner = np.load(os.path.join(d, "probs_inner.npz"), allow_pickle=True)
    test = np.load(os.path.join(d, "probs_test.npz"), allow_pickle=True)
    A = inner["probs"].shape[1]
    if len(set(inner["subjects"]) & set(test["subjects"])):
        raise SystemExit("inner và test trùng subject — kiểm tra lại split.")

    dec = Decoder(A, rules)
    Li, Lt = logit(inner["probs"]), logit(test["probs"])
    yi, yt = inner["labels"], test["labels"]
    grid = np.arange(args.thr_grid[0], args.thr_grid[1] + 1e-9, args.thr_grid[2])

    thr_c1 = tune_thresholds(inner["probs"], yi, grid)
    bias = {"C0": np.zeros(A), "C1": -logit(thr_c1)}

    methods, curves = {}, {}
    for name, base in (("C0", "C0"), ("C1", "C1"), ("R0", "C0"), ("R1", "C1")):
        b = bias[base]
        if name.startswith("C"):
            lam = 0.0
        else:
            curve = [(lam, macro_f1(dec.decode(Li + b, lam)[0], yi)[0]) for lam in args.lambdas]
            curves[name] = curve
            best = max(f for _, f in curve)
            lam = min(l for l, f in curve if f >= best - 1e-12)   # hoà ⇒ λ nhỏ nhất
        pi, idx_i = dec.decode(Li + b, lam)
        pt, idx_t = dec.decode(Lt + b, lam)
        fi, _ = macro_f1(pi, yi)
        ft, ft_au = macro_f1(pt, yt)
        methods[name] = {"lambda": lam, "inner_f1": fi, "test_f1": ft, "test_f1_au": ft_au.tolist(),
                         "diag_inner": rule_diagnostics(dec, idx_i, yi),
                         "diag_test": rule_diagnostics(dec, idx_t, yt), "_pred": pt}

    # ---- bootstrap Δ trên test, theo subject ----
    subj = sorted(set(test["subjects"]))
    cnt = {k: per_subject_counts(v["_pred"], yt, test["subjects"], subj) for k, v in methods.items()}
    rng = np.random.default_rng(args.seed + args.fold)
    deltas = {}
    for r in ("R0", "R1"):
        for c in ("C0", "C1"):
            dd = boot_delta(cnt[r], cnt[c], args.n_boot, rng)
            deltas[f"{r}-{c}"] = {"delta": methods[r]["test_f1"] - methods[c]["test_f1"],
                                  "ci_lo": float(np.percentile(dd, 2.5)),
                                  "ci_hi": float(np.percentile(dd, 97.5)),
                                  "p_gt0": float((dd > 0).mean())}

    # Biến thể luật "chính thức" chọn theo INNER, không theo test
    r_star = max(("R0", "R1"), key=lambda k: methods[k]["inner_f1"])
    dv = [deltas[f"{r_star}-C0"], deltas[f"{r_star}-C1"]]
    if all(x["delta"] > 0 and x["p_gt0"] >= 0.9 for x in dv):
        verdict = "TÍN HIỆU DƯƠNG: luật thắng cả hai đối chứng → đáng đầu tư neural gate / Stage 3"
    elif all(x["delta"] > 0 for x in dv):
        verdict = "CHƯA CHẮC: thắng cả hai nhưng CI còn chạm 0 → xem kết quả 3 fold"
    else:
        verdict = "KHÔNG CÓ TÍN HIỆU: luật không thắng cả hai đối chứng"

    # ---- in ----
    gap = methods["C0"]["inner_f1"] - methods["C0"]["test_f1"]
    print(f"\n=== {ds} fold{args.fold} | {len(rules)} luật | inner {len(yi)} frame, test {len(yt)} frame ===")
    if gap > 0.10:
        print(f"[CẢNH BÁO] F1@0.5 inner cao hơn test {100*gap:.1f} điểm — có thể detector đã train trên "
              "subject của inner (xác suất in-sample). Khi đó λ và ngưỡng chọn trên inner không đáng tin; "
              "train lại Stage 1 trên tập fit (train_Sym_Stage_1_fit.py).")
    print("Ngưỡng C1:", " ".join(f"{a}:{t:.2f}" for a, t in zip(names, thr_c1)))
    for k in ("R0", "R1"):
        print(f"Đường λ ({k}, inner F1):", " ".join(f"{l:g}:{100*f:.2f}" for l, f in curves[k]))
    hdr = f"{'':4}{'λ':>6}{'inner':>8}{'test':>8}  " + "".join(f"{a:>6}" for a in names) + \
          f"{'vi phạm':>9}{'help':>7}{'hurt':>7}"
    print(hdr)
    for k, v in methods.items():
        dg = v["diag_test"]
        print(f"{k:4}{v['lambda']:>6g}{100*v['inner_f1']:>8.2f}{100*v['test_f1']:>8.2f}  "
              + "".join(f"{100*x:>6.1f}" for x in v["test_f1_au"])
              + f"{100*dg['pred_violation_rate']:>8.1f}%{dg['help']:>7}{dg['hurt']:>7}")
    print(f"(nhãn thật test vi phạm ≥1 luật ở {100*methods['C0']['diag_test']['gt_violation_rate']:.1f}% frame)")
    for k, v in deltas.items():
        print(f"Δ {k}: {100*v['delta']:+.2f}  CI95 [{100*v['ci_lo']:+.2f}, {100*v['ci_hi']:+.2f}]  "
              f"P(Δ>0)={v['p_gt0']:.2f}")
    print(f"Biến thể luật chọn theo inner: {r_star} → {verdict}")

    # ---- lưu ----
    for v in methods.values():
        v.pop("_pred")
    summary = {"dataset": ds, "fold": args.fold, "n_rules": len(rules), "thr_c1": thr_c1.tolist(),
               "methods": methods, "curves": curves, "deltas": deltas, "r_star": r_star,
               "verdict": verdict, "au_names": names}
    with open(os.path.join(d, "decode_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(d, "decode_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "lambda", "inner_f1", "test_f1"] + names + ["pred_violation_rate", "help", "hurt"])
        for k, v in methods.items():
            dg = v["diag_test"]
            wr.writerow([k, v["lambda"], round(100 * v["inner_f1"], 2), round(100 * v["test_f1"], 2)]
                        + [round(100 * x, 2) for x in v["test_f1_au"]]
                        + [round(dg["pred_violation_rate"], 4), dg["help"], dg["hurt"]])
    print(f"[SAVED] {d}/decode_summary.json, decode_summary.csv")


if __name__ == "__main__":
    main()
