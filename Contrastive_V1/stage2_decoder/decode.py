"""Decoder cố định dùng bảng luật cho MỘT run Stage 1 (variant/fold/seed).

Đầu vào có sẵn từ train_contrastive.py evaluate:
  <run>/evaluation/val_predictions.npz   (inner val — chọn ngưỡng và λ)
  <run>/evaluation/test_predictions.npz  (outer test — chỉ báo cáo)
Phương pháp:
  C0 ngưỡng 0.5 | C1 ngưỡng từng AU tối ưu trên val       (không luật)
  R0 MAP+luật, bias 0 | R1 MAP+luật, bias = ngưỡng C1    (λ chọn trên val; λ=0 ∈ lưới)
Biến thể luật dùng để kết luận (R*) được chọn theo F1 trên val, không theo test.

  python -m stage2_decoder.decode --run runs_retry1/full/fold1/seed0 --rules downstream/stage2_decoder/rules/fold1 \
      --manifest splits/fold1.json --data-root /data/DISFA --output downstream/stage2_decoder/results
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import AU_NAMES, Decoder, boot_delta, digest, load_predictions, load_split, logit, macro_f1, \
    per_subject_counts, tune_thresholds

LAMBDAS = [0, .1, .25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8]


def check_pair(run, pred, split_data, name):
    ev = run / "evaluation"
    if list(pred["paths"]) != list(split_data["paths"]):
        raise ValueError(f"{name}_predictions.npz không khớp thứ tự/đường dẫn của manifest")
    if not np.array_equal(pred["labels"], split_data["labels"]):
        raise ValueError(f"{name}_predictions.npz: nhãn khác list hiện tại")
    meta = json.loads((ev / f"{name}_metrics.json").read_text(encoding="utf-8"))
    return meta


def rule_diag(dec, idx, y):
    if dec.V.shape[1] == 0:
        return dict(pred_violation_rate=0., gt_violation_rate=0., help=0, hurt=0)
    vp, vg = dec.V[idx] > 0, dec.V[Decoder.config_index(y)] > 0
    return dict(pred_violation_rate=float(vp.any(1).mean()), gt_violation_rate=float(vg.any(1).mean()),
                help=int((vp & ~vg).sum()), hurt=int((vp & vg).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="thư mục run Stage 1: .../<variant>/fold{k}/seed{s}")
    ap.add_argument("--rules", required=True, help="thư mục .../rules/fold{k}")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True, help="gốc kết quả; ghi vào <output>/<variant>/fold{k}/seed{s}/")
    ap.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    ap.add_argument("--thr-grid", type=float, nargs=3, default=[.05, .95, .01], metavar=("LO", "HI", "STEP"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    variant, fold, seed = cfg["variant"], cfg["fold"], cfg["seed"]
    rules_meta = json.loads((Path(args.rules) / "rules.json").read_text(encoding="utf-8"))
    if rules_meta["fold"] != fold or rules_meta["manifest_hash"] != digest(args.manifest):
        raise ValueError("Bảng luật không thuộc fold/manifest của run này")
    if cfg["manifest_hash"] != digest(args.manifest):
        raise ValueError("Run Stage 1 được train với manifest khác")
    rules = rules_meta["rules"]

    sp = load_split(args.manifest, args.data_root)
    val = load_predictions(run / "evaluation" / "val_predictions.npz")
    test = load_predictions(run / "evaluation" / "test_predictions.npz")
    mv = check_pair(run, val, sp["val"], "val")
    mt = check_pair(run, test, sp["test"], "test")
    if mv["checkpoint_hash"] != mt["checkpoint_hash"]:
        raise ValueError("val và test được dự đoán từ hai checkpoint khác nhau — chạy lại evaluate --split val với best.pt")

    dec = Decoder(rules)
    yv, yt = val["labels"], test["labels"]
    Lv, Lt = logit(val["probs"]), logit(test["probs"])
    grid = np.arange(args.thr_grid[0], args.thr_grid[1] + 1e-9, args.thr_grid[2])
    thr = tune_thresholds(val["probs"], yv, grid)
    bias = {"C0": np.zeros(len(AU_NAMES)), "C1": -logit(thr)}

    methods, curves, preds = {}, {}, {}
    for name, base in (("C0", "C0"), ("C1", "C1"), ("R0", "C0"), ("R1", "C1")):
        b = bias[base]
        lam = 0.
        if name.startswith("R"):
            curves[name] = [(l, macro_f1(dec.decode(Lv + b, l)[0], yv)[0]) for l in args.lambdas]
            top = max(f for _, f in curves[name])
            lam = min(l for l, f in curves[name] if f >= top - 1e-12)
        pv, iv = dec.decode(Lv + b, lam)
        pt, it = dec.decode(Lt + b, lam)
        fv, _ = macro_f1(pv, yv)
        ft, ft_au = macro_f1(pt, yt)
        preds[name] = pt
        methods[name] = dict(lambda_=lam, val_f1=fv, test_f1=ft, test_f1_au=ft_au.tolist(),
                             diag_val=rule_diag(dec, iv, yv), diag_test=rule_diag(dec, it, yt))
    if abs(methods["C0"]["test_f1"] - mt["macro_f1"]) > 1e-4:   # chỉ lệch khi có p đúng bằng 0.5
        raise ValueError(f"C0 test F1 {methods['C0']['test_f1']:.6f} ≠ test_metrics.json {mt['macro_f1']:.6f}")

    subj = sorted(set(test["subjects"]))
    cnt = {k: per_subject_counts(v, yt, test["subjects"], subj) for k, v in preds.items()}
    rng = np.random.default_rng(1000 * fold + seed)
    deltas = {}
    for r in ("R0", "R1"):
        for c in ("C0", "C1"):
            d = boot_delta(cnt[r], cnt[c], args.n_boot, rng)
            deltas[f"{r}-{c}"] = dict(delta=methods[r]["test_f1"] - methods[c]["test_f1"],
                                      ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)),
                                      p_gt0=float((d > 0).mean()))
    r_star = max(("R0", "R1"), key=lambda k: (methods[k]["val_f1"], k == "R0"))

    out = Path(args.output) / variant / f"fold{fold}" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(variant=variant, fold=fold, seed=seed, n_rules=len(rules), thresholds_c1=thr.tolist(),
                   methods=methods, curves=curves, deltas=deltas, r_star=r_star, au_names=AU_NAMES,
                   checkpoint_hash=mt["checkpoint_hash"], selected_epoch=mt.get("selected_epoch"),
                   rules_file=str(Path(args.rules) / "rules.json"))
    (out / "decode_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / "test_subject_counts.npz", subjects=np.array(subj),
                        **{k: v for k, v in cnt.items()})

    print(f"\n=== {variant} fold{fold} seed{seed} | {len(rules)} luật | val {len(yv)} frame, test {len(yt)} frame "
          f"| best epoch {mt.get('selected_epoch')}")
    print("Ngưỡng C1:", " ".join(f"{a}:{t:.2f}" for a, t in zip(AU_NAMES, thr)))
    for k in ("R0", "R1"):
        print(f"λ ({k}, val F1):", " ".join(f"{l:g}:{100*f:.2f}" for l, f in curves[k]))
    print(f"{'':4}{'λ':>6}{'val':>8}{'test':>8}  " + "".join(f"{a:>6}" for a in AU_NAMES) + f"{'vi phạm':>9}{'help':>7}{'hurt':>7}")
    for k, v in methods.items():
        dg = v["diag_test"]
        print(f"{k:4}{v['lambda_']:>6g}{100*v['val_f1']:>8.2f}{100*v['test_f1']:>8.2f}  "
              + "".join(f"{100*x:>6.1f}" for x in v["test_f1_au"])
              + f"{100*dg['pred_violation_rate']:>8.1f}%{dg['help']:>7}{dg['hurt']:>7}")
    for k, v in deltas.items():
        print(f"Δ {k}: {100*v['delta']:+.2f}  CI95 [{100*v['ci_lo']:+.2f}, {100*v['ci_hi']:+.2f}]  P(Δ>0)={v['p_gt0']:.2f}")
    print(f"R* (chọn theo val) = {r_star}   [SAVED] {out}")


if __name__ == "__main__":
    main()
