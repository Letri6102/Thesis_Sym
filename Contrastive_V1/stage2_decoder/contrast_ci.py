"""CI cho so sánh chính của Tầng 1: full − baseline trên outer test (đề cương §6).

  python -m stage2_decoder.contrast_ci --runs <Sym>/runs_aug1 --check-only                     # KHÔNG đọc test
  python -m stage2_decoder.contrast_ci --runs <Sym>/runs_aug1 --output <Sym>/contrast_ci/summary.json

Phép đo cố định trước khi mở test, không có tuỳ chọn để đổi fold/seed/số lần bootstrap:
  Δ_seed = mean 3 fold của [macro-F1(full) − macro-F1(baseline)], ngưỡng 0.5 như train_contrastive.metrics;
  CI95   = percentile của 2000 lần resample subject trong từng fold, hai variant dùng chung trọng số
           (dùng lại summarize.fold_mean_boot, cùng phép tính với R0 − C0);
  ĐẠT    = mean Δ qua 3 seed > 0 và CI95 nằm trên 0 ở ≥ 2/3 seed.

Không evaluate và không train. Chỉ đọc file test khi đủ 18 run đúng cấu hình khoá VÀ đủ 18 bộ dự đoán test.
File nằm trong stage2_decoder/ nên không thuộc danh sách file bị hash của train_contrastive.py.
CI chỉ phản ánh dao động giữa subject test, không gồm dao động giữa các lần train.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .check_stage1 import best_point
from .core import AU_NAMES, digest, f1, load_predictions, per_subject_counts
from .summarize import fold_mean_boot, verdict

BASELINE, CANDIDATE = "baseline", "full"
FOLDS, SEEDS = (1, 2, 3), (0, 1, 2)
N_BOOT, BOOT_SEED = 2000, 2026

# Cấu hình khoá ngày 22/11 (CLAUDE.md §6). Cùng giá trị với cell G; tests/test_contrast_ci.py kiểm hai nơi khớp nhau.
LOCK = dict(
    common=dict(arc="resnet50", epochs=20, batch_size=32, workers=4, lr=1e-4, weight_decay=5e-4, temperature=.1,
                gamma=.05, proj_dim=128, precision="float32", threshold=.5, expr_loss="legacy_bce", pretrained=True,
                ema_decay=0., val_per_epoch=1, backbone_lr_mult=1.),
    variant_settings={
        "baseline": dict(global_w=0., node_w=0., cross_w=0., bidirectional=True, queue=0, warmup=3),
        "full": dict(global_w=.05, node_w=.05, cross_w=.05, bidirectional=True, queue=0, warmup=3)},
    env=dict(torch="2.14.1+cu130", torchvision="0.29.1+cu130", numpy="2.4.6", timm="1.0.22", pillow="11.3.0"),
    manifests={1: "00670123caf67f0661a0fc8cdafabadfc6965a488f987591c25e3a341dd04ba2",
               2: "19bd03dafc852f0e259cdff52ca666a62b3de16dd209d32a4ea2e7416b637cf2",
               3: "248bb62b6357642f8cf3b6011eb1c37d5cf301e0377220abab47c78889e673d7"},
    source="94fcd27ef11c40cd30052f8d3dfeb8fb01b178b29655c24a4e8f46dc1604670e",
    test_subjects=9)                                     # số subject test mỗi fold


def source_digest(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def jobs():
    return [(v, k, s) for s in SEEDS for k in FOLDS for v in (BASELINE, CANDIDATE)]


def run_dir(runs, v, k, s):
    return Path(runs) / v / f"fold{k}" / f"seed{s}"


def expected_config(v, k, s, lock):
    return dict(lock["common"], variant=v, fold=k, seed=s, variant_settings=lock["variant_settings"][v],
                environment_versions=lock["env"], manifest_hash=lock["manifests"][k])


# ---------------------------------------------------------------- bước 1: không đọc test
def run_problems(run, v, k, s, lock):
    """Chỉ đọc config.json và history.json, và kiểm best.pt có tồn tại."""
    try:
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        hist = json.loads((run / "history.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["thiếu hoặc hỏng config.json / history.json"]
    out = []
    want = expected_config(v, k, s, lock)
    got = {key: val for key, val in cfg.items() if key != "source_hashes"}
    bad = sorted(key for key in set(want) | set(got) if key not in want or key not in got or want[key] != got[key])
    if bad:
        out.append(f"config lệch khoá ở {bad}")
    if source_digest(cfg.get("source_hashes", {})) != lock["source"]:
        out.append("hash code lệch khoá")
    n = lock["common"]["epochs"]
    if [r.get("epoch") for r in hist] != list(range(1, n + 1)):
        out.append(f"history chưa đủ {n} epoch")
    if not (run / "best.pt").exists():
        out.append("thiếu best.pt")
    return out


def check_runs(runs, lock):
    return [f"{v}/fold{k}/seed{s}: {p}" for v, k, s in jobs() for p in run_problems(run_dir(runs, v, k, s), v, k, s, lock)]


def missing_test(runs):
    """Chỉ kiểm file có tồn tại, không mở."""
    return [f"{v}/fold{k}/seed{s}/evaluation/{name}" for v, k, s in jobs()
            for name in ("test_metrics.json", "test_predictions.npz")
            if not (run_dir(runs, v, k, s) / "evaluation" / name).exists()]


# ---------------------------------------------------------------- bước 2: đọc test
def load_run(run, v, k, s):
    """Thông báo lỗi ở đây không được chứa số test."""
    tag = f"{v}/fold{k}/seed{s}"
    m = json.loads((run / "evaluation" / "test_metrics.json").read_text(encoding="utf-8"))
    if (m["variant"], m["fold"], m["seed"], m["split"], m["stage"], m["threshold"]) != (v, k, s, "test", 1, .5):
        raise ValueError(f"{tag}: test_metrics không phải của run này")
    if m["config"] != json.loads((run / "config.json").read_text(encoding="utf-8")):
        raise ValueError(f"{tag}: test được evaluate từ config khác")
    selected, _ = best_point(json.loads((run / "history.json").read_text(encoding="utf-8")))
    if m.get("selected_progress", m["selected_epoch"]) != selected:
        raise ValueError(f"{tag}: test không dùng checkpoint tốt nhất trên inner val")

    d = load_predictions(run / "evaluation" / "test_predictions.npz")
    p, y = d["probs"], d["labels"]
    if p.shape != y.shape or p.shape[1:] != (len(AU_NAMES),) or not np.isfinite(p).all() or not np.isin(y, (0, 1)).all():
        raise ValueError(f"{tag}: test_predictions sai dạng")
    subjects = sorted(set(d["subjects"].tolist()))
    cnt = per_subject_counts(p >= .5, y, d["subjects"], subjects)     # (S, 3, A); >= như metrics() của Tầng 1
    per_au = f1(cnt.sum(0))
    if not np.allclose(per_au, m["per_au_f1"], rtol=0, atol=1e-9) or abs(per_au.mean() - m["macro_f1"]) > 1e-9:
        raise ValueError(f"{tag}: dự đoán không khớp test_metrics")
    return dict(counts=cnt, subjects=subjects, labels=y, paths=d["paths"], frame_subjects=d["subjects"],
                per_au=per_au, init=m["initial_detector_hash"], checkpoint=m["checkpoint_hash"], selected=selected)


def check_pairing(data, n_subjects):
    for k in FOLDS:
        ref = data[(BASELINE, k, SEEDS[0])]
        if len(ref["subjects"]) != n_subjects:
            raise ValueError(f"fold{k}: cần {n_subjects} subject test")
        for s in SEEDS:
            for v in (BASELINE, CANDIDATE):
                for key in ("labels", "paths", "frame_subjects"):
                    if not np.array_equal(data[(v, k, s)][key], ref[key]):
                        raise ValueError(f"{v}/fold{k}/seed{s}: {key} không ghép cặp được")
            if data[(BASELINE, k, s)]["init"] != data[(CANDIDATE, k, s)]["init"]:
                raise ValueError(f"fold{k}/seed{s}: hai variant khác trọng số khởi tạo")
    every = [u for k in FOLDS for u in data[(BASELINE, k, SEEDS[0])]["subjects"]]
    if len(every) != len(set(every)):
        raise ValueError("Một subject nằm trong test của hơn một fold")


def compare(data, n_boot=N_BOOT):
    per_seed = []
    for s in SEEDS:
        by_fold = {v: np.array([data[(v, k, s)]["per_au"] for k in FOLDS]) for v in (BASELINE, CANDIDATE)}
        pair = {("pair", k, s): dict(counts={v: data[(v, k, s)]["counts"] for v in (BASELINE, CANDIDATE)}) for k in FOLDS}
        boot = 100 * fold_mean_boot(pair, "pair", s, FOLDS, CANDIDATE, BASELINE, n_boot,
                                    np.random.default_rng([BOOT_SEED, s]))
        diff = 100 * (by_fold[CANDIDATE] - by_fold[BASELINE])
        per_seed.append(dict(seed=s, baseline=float(100 * by_fold[BASELINE].mean()),
                             full=float(100 * by_fold[CANDIDATE].mean()), delta_pp=float(diff.mean()),
                             ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                             p_gt0=float((boot > 0).mean()), delta_by_fold=diff.mean(1).tolist(),
                             delta_by_au=diff.mean(0).tolist()))

    def mean_sd(key):
        x = [r[key] for r in per_seed]
        return dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)))

    return dict(comparison=f"{CANDIDATE}-{BASELINE}", split="test", folds=list(FOLDS), seeds=list(SEEDS),
                n_boot=n_boot, boot_seed=BOOT_SEED, au_names=AU_NAMES,
                rule="mean Δ > 0 và CI95 nằm trên 0 ở >= 2/3 seed",
                note="CI chỉ gồm dao động giữa subject test, không gồm dao động giữa các lần train",
                per_seed=per_seed, baseline=mean_sd("baseline"), full=mean_sd("full"), delta=mean_sd("delta_pp"),
                seeds_ci_above_0=sum(r["ci_lo"] > 0 for r in per_seed), verdict=verdict(per_seed),
                checkpoints={f"{v}/fold{k}/seed{s}": dict(sha256=d["checkpoint"], selected=d["selected"])
                             for (v, k, s), d in data.items()},
                code_sha256={name: digest(Path(__file__).with_name(name))
                             for name in ("contrast_ci.py", "core.py", "summarize.py")})


def show(rep):
    # "TEST MACRO-F1" cũng là dấu hiệu để notebook báo cáo (cell F) ẩn đuôi log này.
    print("TEST MACRO-F1 (%) full − baseline, mean 3 fold; CI95 bootstrap theo subject")
    print(f"{'seed':>4}{'baseline':>10}{'full':>9}{'Δ':>8}{'CI95':>20}{'P(Δ>0)':>9}")
    for r in rep["per_seed"]:
        ci = f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
        print(f"{r['seed']:>4}{r['baseline']:>10.2f}{r['full']:>9.2f}{r['delta_pp']:>+8.2f}{ci:>20}{r['p_gt0']:>9.2f}")
    b, c, d = rep["baseline"], rep["full"], rep["delta"]
    print(f"mean ± SD qua seed: baseline {b['mean']:.2f} ± {b['sd']:.2f} | full {c['mean']:.2f} ± {c['sd']:.2f}"
          f" | Δ {d['mean']:+.2f} ± {d['sd']:.2f}")
    print(f"CI trên 0 ở {rep['seeds_ci_above_0']}/{len(SEEDS)} seed → {rep['verdict']}")


def main(argv=None, lock=LOCK):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="kiểm đủ 18 run đúng khoá; không đọc file test")
    mode.add_argument("--output", help="file JSON kết quả; không bao giờ ghi đè")
    args = ap.parse_args(argv)

    if args.output and Path(args.output).exists():
        raise SystemExit(f"Đã có {args.output}: so sánh chính chỉ chạy một lần, không ghi đè.")
    problems = check_runs(args.runs, lock)
    if problems:
        raise SystemExit(f"CHƯA sẵn sàng ({len(problems)} lỗi), không đọc file test nào:\n  " + "\n  ".join(problems))
    total = len(jobs())
    if args.check_only:
        print(f"OK: đủ {total} run đúng cấu hình khoá. Không đọc file test nào.")
        return None
    missing = missing_test(args.runs)
    if missing:
        raise SystemExit(f"Thiếu {len(missing)} file dự đoán test, không đọc file test nào:\n  " + "\n  ".join(missing))

    data = {(v, k, s): load_run(run_dir(args.runs, v, k, s), v, k, s) for v, k, s in jobs()}
    check_pairing(data, lock["test_subjects"])
    rep = compare(data)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    show(rep)
    print(f"[SAVED] {out}")
    return rep


if __name__ == "__main__":
    main()
