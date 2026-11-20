"""Decoder cố định dùng bảng luật cho MỘT run Stage 1 (variant/fold/seed).

Đầu vào (train_contrastive.py evaluate, từ CÙNG best.pt):
  <run>/evaluation_val/val_predictions.npz   inner val — chọn ngưỡng C1 và λ
  <run>/evaluation/test_predictions.npz      outer test — chỉ báo cáo; không đọc khi --val-only
  (bản cũ để val ở <run>/evaluation/ vẫn đọc được)

Các nhánh, cùng một decoder MAP chính xác trên 2^8 cấu hình:
  C0      ngưỡng 0.5, không luật
  C1      ngưỡng từng AU tối ưu trên val, không luật            phụ — dễ quá khớp val 3 subject
  R0      luật rút từ dữ liệu (rules.json), bias 0               ← NHÁNH CHÍNH
  R1      luật rút từ dữ liệu, bias = ngưỡng C1                  phụ
  R_FACS  5 luật loại trừ viết tay (SymGraphAU), w = 1, bias 0   Q2a: cùng decoder, khác nguồn luật
λ của mọi nhánh R chọn trên val (λ = 0 có trong lưới; hoà → λ nhỏ nhất).
So sánh chính: R0 − C0. Phụ: R0 − R_FACS (Q2a), R_FACS − C0, R1 − C1, C1 − C0.

  python -m stage2_decoder.decode --run <runs>/full/fold1/seed0 --rules <out>/rules/fold1 \
      --manifest <splits>/fold1.json --data-root <DISFA> --output <out>/results [--val-only]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import AU_NAMES, Decoder, boot_delta, digest, facs_rules, load_predictions, load_split, logit, \
    macro_f1, per_subject_counts, prediction_dir, tune_thresholds

LAMBDAS = [0, .1, .25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8]
# (tên nhánh, bias, nguồn luật); nguồn None = không luật (λ = 0)
ARMS = [("C0", "C0", None), ("C1", "C1", None), ("R0", "C0", "mined"), ("R1", "C1", "mined"),
        ("R_FACS", "C0", "facs")]
PRIMARY = ("R0", "C0")
COMPARISONS = [PRIMARY, ("R0", "R_FACS"), ("R_FACS", "C0"), ("R1", "C1"), ("C1", "C0")]
VAL_FILE, TEST_FILE = "decode_val.json", "decode_summary.json"


def check_pair(pred, split_data, metrics_path, name):
    if list(pred["paths"]) != list(split_data["paths"]):
        raise ValueError(f"{name}_predictions.npz không khớp thứ tự/đường dẫn của manifest")
    if not np.array_equal(pred["labels"], split_data["labels"]):
        raise ValueError(f"{name}_predictions.npz: nhãn khác list hiện tại")
    return json.loads(Path(metrics_path).read_text(encoding="utf-8"))


def rule_diag(dec, idx, y):
    if dec.V.shape[1] == 0:
        return dict(pred_violation_rate=0., gt_violation_rate=0., help=0, hurt=0)
    vp, vg = dec.V[idx] > 0, dec.V[Decoder.config_index(y)] > 0
    return dict(pred_violation_rate=float(vp.any(1).mean()), gt_violation_rate=float(vg.any(1).mean()),
                help=int((vp & ~vg).sum()), hurt=int((vp & vg).sum()))


def choose_lambda(curve):
    top = max(f for _, f in curve)
    return min(l for l, f in curve if f >= top - 1e-12)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="thư mục run Stage 1: .../<variant>/fold{k}/seed{s}")
    ap.add_argument("--rules", required=True, help="thư mục .../rules/fold{k}")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True, help="gốc kết quả; ghi vào <output>/<variant>/fold{k}/seed{s}/")
    ap.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    ap.add_argument("--thr-grid", type=float, nargs=3, default=[.05, .95, .01], metavar=("LO", "HI", "STEP"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--val-only", action="store_true",
                    help="chỉ dùng inner val (giai đoạn thiết kế): không đọc test, ghi decode_val.json")
    args = ap.parse_args(argv)

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
    vdir = prediction_dir(run, "val")
    val = load_predictions(vdir / "val_predictions.npz")
    mv = check_pair(val, sp["val"], vdir / "val_metrics.json", "val")
    test = mt = None
    if not args.val_only:
        tdir = prediction_dir(run, "test")
        test = load_predictions(tdir / "test_predictions.npz")
        mt = check_pair(test, sp["test"], tdir / "test_metrics.json", "test")
        if mv["checkpoint_hash"] != mt["checkpoint_hash"]:
            raise ValueError("val và test được dự đoán từ hai checkpoint khác nhau — chạy lại evaluate --split val với best.pt")

    decoders = {"mined": Decoder(rules), "facs": Decoder(facs_rules())}
    yv, Lv = val["labels"], logit(val["probs"])
    grid = np.arange(args.thr_grid[0], args.thr_grid[1] + 1e-9, args.thr_grid[2])
    thr = tune_thresholds(val["probs"], yv, grid)
    bias = {"C0": np.zeros(len(AU_NAMES)), "C1": -logit(thr)}
    if test is not None:
        yt, Lt = test["labels"], logit(test["probs"])

    methods, curves, preds = {}, {}, {}
    for name, base, src in ARMS:
        dec, b, lam = decoders[src or "mined"], bias[base], 0.
        if src:
            curves[name] = [(l, macro_f1(dec.decode(Lv + b, l)[0], yv)[0]) for l in args.lambdas]
            lam = choose_lambda(curves[name])
        pv, iv = dec.decode(Lv + b, lam)
        fv, fv_au = macro_f1(pv, yv)
        m = dict(lambda_=lam, bias=base, rules=src, n_rules=dec.V.shape[1] if src else 0,
                 val_f1=fv, val_f1_au=fv_au.tolist(), diag_val=rule_diag(dec, iv, yv))
        if test is not None:
            pt, it = dec.decode(Lt + b, lam)
            ft, ft_au = macro_f1(pt, yt)
            preds[name] = pt
            m.update(test_f1=ft, test_f1_au=ft_au.tolist(), diag_test=rule_diag(dec, it, yt))
        methods[name] = m
    for split, meta in (("val", mv), ("test", mt)):   # C0 phải trùng metric của evaluate (chỉ lệch khi p == 0.5)
        if meta is not None and abs(methods["C0"][f"{split}_f1"] - meta["macro_f1"]) > 1e-4:
            raise ValueError(f"C0 {split} F1 {methods['C0'][f'{split}_f1']:.6f} ≠ {split}_metrics.json {meta['macro_f1']:.6f}")

    deltas, cnt = {}, {}
    if test is not None:
        subj = sorted(set(test["subjects"]))
        cnt = {k: per_subject_counts(v, yt, test["subjects"], subj) for k, v in preds.items()}
        rng = np.random.default_rng(1000 * fold + seed)
        for a, b in COMPARISONS:
            d = boot_delta(cnt[a], cnt[b], args.n_boot, rng)
            deltas[f"{a}-{b}"] = dict(delta=methods[a]["test_f1"] - methods[b]["test_f1"],
                                      ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)),
                                      p_gt0=float((d > 0).mean()))

    out = Path(args.output) / variant / f"fold{fold}" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(variant=variant, fold=fold, seed=seed, val_only=test is None, n_rules=len(rules),
                   rules_mined=[r.get("rule", "") for r in rules], rules_facs=[r["rule"] for r in facs_rules()],
                   thresholds_c1=thr.tolist(), methods=methods, curves=curves, deltas=deltas,
                   primary=f"{PRIMARY[0]}-{PRIMARY[1]}", au_names=AU_NAMES,
                   checkpoint_hash=mv["checkpoint_hash"], selected_epoch=mv.get("selected_epoch"),
                   rules_file=str(Path(args.rules) / "rules.json"))
    target = out / (VAL_FILE if test is None else TEST_FILE)
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if test is not None:
        np.savez_compressed(out / "test_subject_counts.npz", subjects=np.array(subj), **cnt)

    split = "val" if test is None else "test"
    print(f"\n=== {variant} fold{fold} seed{seed} | {len(rules)} luật dữ liệu, {len(facs_rules())} luật FACS "
          f"| val {len(yv)} frame" + ("" if test is None else f", test {len(yt)} frame")
          + f" | best epoch {mv.get('selected_epoch')}" + ("  [CHỈ VAL — test chưa mở]" if test is None else ""))
    print("Ngưỡng C1:", " ".join(f"{a}:{t:.2f}" for a, t in zip(AU_NAMES, thr)))
    for k, c in curves.items():
        print(f"λ ({k}, val F1):", " ".join(f"{l:g}:{100*f:.2f}" for l, f in c))
    print(f"{'':7}{'λ':>6}{'val':>8}{'test':>8}  " + "".join(f"{a:>6}" for a in AU_NAMES)
          + f"   ({split})" + f"{'vi phạm':>9}{'help':>7}{'hurt':>7}")
    for k, v in methods.items():
        dg = v[f"diag_{split}"]
        print(f"{k:7}{v['lambda_']:>6g}{100*v['val_f1']:>8.2f}"
              + (f"{100*v['test_f1']:>8.2f}" if test is not None else f"{'-':>8}") + "  "
              + "".join(f"{100*x:>6.1f}" for x in v[f"{split}_f1_au"]) + " " * 9
              + f"{100*dg['pred_violation_rate']:>8.1f}%{dg['help']:>7}{dg['hurt']:>7}")
    for k, v in deltas.items():
        print(f"Δ {k:<10}{'(chính)' if k == summary['primary'] else '':8}{100*v['delta']:+.2f}  "
              f"CI95 [{100*v['ci_lo']:+.2f}, {100*v['ci_hi']:+.2f}]  P(Δ>0)={v['p_gt0']:.2f}")
    print(f"[SAVED] {target}")


if __name__ == "__main__":
    main()
