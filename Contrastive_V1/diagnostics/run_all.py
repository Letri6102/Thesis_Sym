"""Chạy chẩn đoán cho mọi run Tầng 1 đã xong và đúng khoá (Colab, cell N, sau cell L). CHỈ fit/val, không đụng test.

  python -m diagnostics.run_all --runs <Sym>/runs_aug1 --splits <Sym>/splits_v2 --data-root /content/DISFA \
      --output <Sym>/diagnostics [--checks calibration identity grad] [--stage-dir /content/diag_ckpt]

Với mỗi run <variant>/fold{k}/seed{s} trong 18 run khoá (baseline/full × 3 fold × 3 seed):
  calibration.json     từ evaluation_val/val_predictions.npz: AUC, SD ngưỡng t* giữa subject val (CPU, vài giây)
  identity_probe.json  best.pt, frame fit: probe tuyến tính đoán subject (GPU, khoảng 1–2 phút)
  grad_norms.json      best.pt và last.pt, 20 batch fit: chuẩn gradient từng số hạng loss (GPU, khoảng 1 phút)
Cần cell L chạy trước: evaluation_val/ là nơi lấy dự đoán val và checkpoint_hash của best.pt.
Kết quả đã có thì bỏ qua, trừ khi best.pt của run đã đổi (so checkpoint_hash); chạy lại an toàn khi có thêm run.
Cuối cùng ghi <output>/summary.json: từng run, trung bình theo variant, full − baseline theo cặp cùng fold/seed.
Mọi số ở đây là chẩn đoán mô tả trên fit/val: không báo cáo thay cho F1 test, không so với 63.0.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from stage2_decoder.contrast_ci import BASELINE, CANDIDATE, LOCK, jobs, run_dir, run_problems
from stage2_decoder.core import digest, prediction_dir

from . import calibration
from .common import staged_copy, write_json

CHECKS = ("calibration", "identity", "grad")
FILES = dict(calibration="calibration.json", identity="identity_probe.json", grad="grad_norms.json")
FEATS = ("trunk", "pooled", "au_concat")


def val_key(run):
    """(checkpoint_hash, val_metrics) của lần evaluate val, hoặc (None, None) nếu chưa evaluate."""
    d = prediction_dir(run, "val")
    if not ((d / "val_metrics.json").exists() and (d / "val_predictions.npz").exists()):
        return None, None
    m = json.loads((d / "val_metrics.json").read_text(encoding="utf-8"))
    return m["checkpoint_hash"], m


def is_current(path, key):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("val_checkpoint_hash") == key
    except (OSError, ValueError):
        return False


def run_one(run, tag, manifest, todo, dest, key, vm, args):
    """Chạy các chẩn đoán còn thiếu của một run. Trả danh sách lỗi (mỗi chẩn đoán độc lập)."""
    errors = []
    base = dict(run=tag, val_checkpoint_hash=key, val_only=True)
    if "calibration" in todo:
        try:
            r = calibration.run_calibration(run)
            if abs(r["macro_f1_at_05"] - vm["macro_f1"]) > 1e-9:
                raise ValueError("val_predictions.npz không khớp val_metrics.json")
            write_json(dest / FILES["calibration"], dict(base, **r))
        except Exception:
            errors.append(f"{tag} calibration:\n{traceback.format_exc()}")
    if not {"identity", "grad"} & set(todo):
        return errors

    import torch
    from . import grad_norms, identity_probe
    common = dict(workers=args.workers, device=args.device)
    best = None
    try:
        best = staged_copy(run / "best.pt", args.stage_dir)
        if digest(best) != key:
            raise ValueError("best.pt khác checkpoint đã evaluate val (chạy lại cell L)")
        if "identity" in todo:
            try:
                r = identity_probe.run(best, manifest, args.data_root, args.stride, args.batch_size, **common)
                write_json(dest / FILES["identity"], dict(base, **r))
            except Exception:
                errors.append(f"{tag} identity:\n{traceback.format_exc()}")
        if "grad" in todo:
            try:
                out = dict(base, best=grad_norms.measure(best, manifest, args.data_root, args.grad_batches, **common),
                           last=None)
                if args.stage_dir:
                    best.unlink(missing_ok=True)
                    best = None
                if (run / "last.pt").exists():
                    last = staged_copy(run / "last.pt", args.stage_dir)
                    try:
                        out["last"] = grad_norms.measure(last, manifest, args.data_root, args.grad_batches, **common)
                    finally:
                        if args.stage_dir:
                            last.unlink(missing_ok=True)
                else:
                    out["note"] = "không có last.pt: chỉ đo best.pt"
                write_json(dest / FILES["grad"], out)
            except Exception:
                errors.append(f"{tag} grad:\n{traceback.format_exc()}")
    except Exception:
        errors.append(f"{tag} best.pt:\n{traceback.format_exc()}")
    finally:
        if args.stage_dir and best is not None:
            best.unlink(missing_ok=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return errors


def load_row(dest, key, v, k, s):
    row = dict(variant=v, fold=k, seed=s)
    files = {c: dest / FILES[c] for c in CHECKS}
    got = {c: json.loads(p.read_text(encoding="utf-8")) for c, p in files.items() if is_current(p, key)}
    if "calibration" in got:
        row.update(got["calibration"]["summary"])
        row["auc_au"] = got["calibration"]["auc_pooled"]
        row["thr_sd_au"] = got["calibration"]["thr_sd"]
    if "identity" in got:
        pr = got["identity"]["probes"]
        row.update({f"id_acc_{f}": pr[f]["acc"] for f in FEATS})
        row.update({f"id_nll_{f}": pr[f]["nll"] for f in FEATS})
        row.update(id_chance=pr["pooled"]["chance"], id_nll_chance=pr["pooled"]["nll_chance"])
    if "grad" in got:
        for which in ("best", "last"):
            g = got["grad"].get(which)
            if g:
                row[f"grad_{which}_epoch"] = g["epoch"]
                row[f"grad_{which}_ratio"] = g["ratio_contrastive_to_wa"]["backbone"]
                row[f"grad_{which}_cos"] = g["cos_wa_contrastive"]["backbone"]
                row[f"grad_{which}_wa"] = g["grad_norm_raw"]["wa"]["backbone"]
    return row


METRICS = ("macro_auc", "mean_thr_sd", "mean_oracle_gain", *(f"id_acc_{f}" for f in FEATS),
           *(f"id_nll_{f}" for f in FEATS), "grad_best_wa", "grad_last_wa")


def summarize(rows):
    def mean(vals):
        vals = [x for x in vals if x is not None and np.isfinite(x)]
        return float(np.mean(vals)) if vals else None

    means = {v: dict(n=sum(r["variant"] == v for r in rows),
                     **{m: mean([r.get(m) for r in rows if r["variant"] == v]) for m in METRICS
                        + ("grad_best_ratio", "grad_last_ratio", "grad_best_cos", "grad_last_cos")})
             for v in (BASELINE, CANDIDATE) if any(r["variant"] == v for r in rows)}
    by = {(r["variant"], r["fold"], r["seed"]): r for r in rows}
    pairs = []
    for k, s in sorted({(k, s) for _, k, s in jobs()}):
        a, b = by.get((BASELINE, k, s)), by.get((CANDIDATE, k, s))
        if a and b:
            pairs.append(dict(fold=k, seed=s, **{m: (b[m] - a[m]) if a.get(m) is not None and b.get(m) is not None
                                                 else None for m in METRICS}))
    paired = dict(n=len(pairs), per_pair=pairs, mean={m: mean([p[m] for p in pairs]) for m in METRICS})
    return dict(n_runs=len(rows), of=len(jobs()), means=means, full_minus_baseline=paired)


def report(rows, summ, notes):
    f = lambda x, d=1, s=100: "-" if x is None else f"{s * x:.{d}f}"  # noqa: E731
    print("=" * 108)
    print(f"CHẨN ĐOÁN CHỈ FIT/VAL — {summ['n_runs']}/{summ['of']} run. Mô tả, không báo cáo, không so với 63.0.")
    print("AUC, SD(t*), F1(t*)−F1(.5): val 3 subject | probe danh tính: frame fit (ngẫu nhiên = 1/số subject)"
          " | grad: ‖∇contrastive‖/‖∇L_wa‖ trên backbone")
    print("=" * 108)
    print(f"{'run':<20}{'AUC':>6}{'SD t*':>7}{'gain':>6}{'id trunk':>9}{'id pool':>8}{'id AU':>7}{'ngẫu nh.':>9}"
          f"{'ratio best':>11}{'ratio last':>11}{'cos last':>9}")
    for r in sorted(rows, key=lambda r: (r["seed"], r["fold"], r["variant"])):
        print(f"{r['variant'] + '/f' + str(r['fold']) + '/s' + str(r['seed']):<20}{f(r.get('macro_auc')):>6}"
              f"{f(r.get('mean_thr_sd'), 3, 1):>7}{f(r.get('mean_oracle_gain')):>6}"
              + "".join(f"{f(r.get(f'id_acc_{x}')):>{w}}" for x, w in zip(FEATS, (9, 8, 7)))
              + f"{f(r.get('id_chance')):>9}{f(r.get('grad_best_ratio'), 3, 1):>11}{f(r.get('grad_last_ratio'), 3, 1):>11}"
              f"{f(r.get('grad_last_cos'), 3, 1):>9}")
    fb = summ["full_minus_baseline"]
    if fb["n"]:
        m = fb["mean"]
        print(f"\nfull − baseline theo cặp cùng fold/seed (n={fb['n']}): AUC {f(m['macro_auc'], 2)} | SD t* "
              f"{f(m['mean_thr_sd'], 3, 1)} | probe danh tính trunk/pooled/AU {f(m['id_acc_trunk'])}/"
              f"{f(m['id_acc_pooled'])}/{f(m['id_acc_au_concat'])} điểm %")
        print("  Probe danh tính thấp hơn và SD t* nhỏ hơn ở full thì contrastive có làm đặc trưng bớt phụ thuộc người.")
    if notes:
        print("\nChưa tính:\n  " + "\n  ".join(notes))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--splits", required=True, help="thư mục chứa fold{k}.json (manifest mà run đã train)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checks", nargs="+", choices=CHECKS, default=list(CHECKS))
    ap.add_argument("--stage-dir", help="chép checkpoint về đây trước khi nạp (Colab: /content/diag_ckpt)")
    ap.add_argument("--stride", type=int, default=4, help="probe danh tính: lấy mỗi N frame fit")
    ap.add_argument("--grad-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64, help="batch khi trích đặc trưng cho probe")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    out = Path(args.output)

    notes, errors, rows = [], [], []
    for v, k, s in jobs():
        run, tag = run_dir(args.runs, v, k, s), f"{v}/fold{k}/seed{s}"
        if not (run / "config.json").exists():
            continue
        bad = run_problems(run, v, k, s, LOCK)
        if bad:
            notes.append(f"{tag}: chưa xong hoặc lệch khoá ({'; '.join(bad)})")
            continue
        key, vm = val_key(run)
        if key is None:
            notes.append(f"{tag}: chưa có evaluation_val (chạy cell L trước)")
            continue
        dest = out / v / f"fold{k}" / f"seed{s}"
        todo = [c for c in args.checks if not is_current(dest / FILES[c], key)]
        if todo:
            print(f"\n>> {tag}: {', '.join(todo)}", flush=True)
            errors += run_one(run, tag, Path(args.splits) / f"fold{k}.json", todo, dest, key, vm, args)
        rows.append(load_row(dest, key, v, k, s))

    summ = summarize(rows)
    report(rows, summ, notes)
    write_json(out / "summary.json", dict(summary=summ, rows=rows, notes=notes, val_only=True))
    if errors:
        print(f"\n{len(errors)} chẩn đoán lỗi:\n" + "\n".join(errors), flush=True)
        sys.exit(1)
    return summ


if __name__ == "__main__":
    main()
