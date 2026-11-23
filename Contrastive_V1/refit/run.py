"""Dòng (b) refit (chủ dự án chốt 23/11): train lại mỗi run trên ĐỦ 18 subject outer-train (fit + inner val),
cùng công thức đã khoá, dừng ở đúng epoch E mà run (a) tương ứng chọn trên inner val. Chỉ dòng này đặt cạnh 63.0.

  Công thức: train_contrastive.py đã khoá, không đổi byte nào; lịch cosine 20 epoch giữ nguyên và dừng sau epoch E
  (--stop-after E). Model refit là last.pt (KHÔNG phải best.pt). Trọng số lớp của L_wa tính lại trên 18 subject.
  Inner val đã nằm trong tập train nên không còn val: vòng train vẫn đo trên 64 frame "mồi" của val cũ cho chạy được,
  số đó vô nghĩa và không dùng để chọn gì.

Gọi từ notebook (cell K) hoặc pc/train_18_run.py refit:
  jobs(...)      → danh sách run refit còn phải làm, với E lấy từ history của run (a)
  one_cmd(...)   → lệnh chạy một run: python -m refit.run one <tham số của train_contrastive.py train>
  finish(...)    → kiểm run vừa xong (đủ E epoch, config đúng khoá) và ghi refit.json
File nằm trong refit/ nên không thuộc danh sách file bị hash.
"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from stage2_decoder.check_stage1 import best_point  # noqa: E402
from stage2_decoder.contrast_ci import LOCK, config_problems, jobs as all_jobs, run_problems  # noqa: E402

PROBE = 64


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive(orig, labels):
    """Manifest refit suy ra từ manifest gốc: fit = fit ∪ val (18 subject), trọng số lớp tính lại trên đó."""
    fit = sorted(orig["fit_indices"] + orig["val_indices"])
    rates = labels[fit].mean(0)
    if (rates <= 0).any():
        raise ValueError("Refit thiếu mẫu dương của một AU")
    w = 1 / rates
    w = w / w.sum() * len(rates)                         # cùng công thức với experiment_data
    m = dict(orig)
    m.update(fit_indices=fit, val_indices=sorted(orig["val_indices"])[:PROBE],
             fit_subjects=sorted(orig["fit_subjects"] + orig["val_subjects"]), train_weights=w.tolist(),
             refit=True, refit_note="val_indices chỉ là mồi nằm trong tập train, không phải validation")
    return m


def refit_loader(original_load):
    """Thay load_manifest trong tiến trình train: manifest refit phải đúng bằng derive(manifest gốc nằm cạnh nó)."""
    def load(path, root):
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        if not m.get("refit"):
            return original_load(path, root)
        from experiment_data import read_partition
        orig = original_load(Path(path).with_name(Path(path).stem + ".orig.json"), root)
        _, labels, _ = read_partition(root, orig["fold"], "train")
        if m != derive(orig, labels):
            raise ValueError(f"{path} không khớp manifest gốc")
        return m
    return load


def write_manifests(splits, data, root):
    """Ghi <root>/manifests/fold{k}.json (refit) + fold{k}.orig.json (bản gốc đã khoá). Ghi lại thì phải trùng."""
    from experiment_data import load_manifest, read_partition
    out = Path(root) / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for k in (1, 2, 3):
        src = Path(splits) / f"fold{k}.json"
        orig = load_manifest(src, data)
        _, labels, _ = read_partition(data, k, "train")
        content = {f"fold{k}.orig.json": src.read_bytes(),
                   f"fold{k}.json": json.dumps(derive(orig, labels), indent=2).encode("utf-8")}
        for name, b in content.items():
            f = out / name
            if f.exists() and f.read_bytes() != b:
                raise SystemExit(f"{f} đã có và khác: không ghi đè")
            f.write_bytes(b)
        hashes[k] = sha256(out / f"fold{k}.json")
    return hashes


def selected_epoch(run_a):
    """E = epoch mà run (a) chọn trên inner val (cấu hình khoá đánh giá val 1 lần/epoch nên E là số nguyên)."""
    e, _ = best_point(json.loads((Path(run_a) / "history.json").read_text(encoding="utf-8")))
    if int(e) != e:
        raise ValueError(f"{run_a}: epoch chọn {e} không nguyên")
    return int(e)


def n_epochs(run):
    try:
        return len(json.loads((Path(run) / "history.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return 0


def jobs(runs_a, root, splits, data, seeds=(0, 1, 2), folds=(1, 2, 3), lock=LOCK):
    """Run refit còn phải làm. Run (a) chưa xong hoặc lệch khoá thì bỏ qua, ghi lý do."""
    hashes = write_manifests(splits, data, root)
    todo, skipped = [], []
    for v, k, s in all_jobs():
        if s not in seeds or k not in folds:
            continue
        run_a = Path(runs_a) / v / f"fold{k}" / f"seed{s}"
        bad = run_problems(run_a, v, k, s, lock)
        if bad:
            skipped.append(f"{v}/fold{k}/seed{s}: run (a) chưa dùng được ({bad[0]})")
            continue
        E = selected_epoch(run_a)
        out = Path(root) / v / f"fold{k}" / f"seed{s}"
        if (out / "refit.json").exists():
            continue
        todo.append(dict(variant=v, fold=k, seed=s, E=E, run_a=str(run_a), out=str(out),
                         manifest=str(Path(root) / "manifests" / f"fold{k}.json"), manifest_hash=hashes[k]))
    return todo, skipped


def clear_stale(job, failed_dir, busy_min=15, old_session_dead=False):
    """Trước khi chạy: chặn nếu run vừa được ghi (có thể nơi khác đang train); cất run bị ngắt trong epoch 1."""
    out = Path(job["out"])
    seen = [out / n for n in ("config.json", "history.json", "last.pt", "best.pt") if (out / n).exists()]
    if seen and not old_session_dead:
        age = (time.time() - max(p.stat().st_mtime for p in seen)) / 60
        if age < busy_min:
            raise SystemExit(f"{out} vừa được ghi {age:.0f} phút trước: có thể nơi khác đang train. "
                             f"Đợi {busy_min} phút hoặc xác nhận phiên cũ đã tắt")
    if (out / "config.json").exists() and not (out / "last.pt").exists():
        if (out / "history.json").exists():
            raise SystemExit(f"{out} có history nhưng không có last.pt: gửi Claude xem, đừng xoá")
        dest = Path(failed_dir) / (f"refit1_{job['variant']}_fold{job['fold']}_seed{job['seed']}_"
                                   + time.strftime("%Y%m%d_%H%M%S"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(dest))
        return dest
    return None


def one_cmd(job, data, python=sys.executable, lock=LOCK):
    c = lock["common"]
    cmd = [python, "-u", "-m", "refit.run", "one", "--variant", job["variant"], "--seed", str(job["seed"]),
           "--arc", c["arc"], "--data-root", str(data), "--manifest", job["manifest"], "--output", job["out"],
           "--epochs", str(c["epochs"]), "--batch-size", str(c["batch_size"]), "--workers", str(c["workers"]),
           "--stop-after", str(job["E"])]
    if (Path(job["out"]) / "last.pt").exists():
        cmd += ["--resume", str(Path(job["out"]) / "last.pt")]
    return cmd


def finish(job, lock=LOCK):
    out = Path(job["out"])
    if n_epochs(out) != job["E"] or not (out / "last.pt").exists():
        raise SystemExit(f"{out}: refit chưa đủ {job['E']} epoch")
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    refit_lock = dict(lock, manifests={**lock["manifests"], job["fold"]: job["manifest_hash"]})
    bad = config_problems(cfg, job["variant"], job["fold"], job["seed"], refit_lock)
    if bad:
        raise SystemExit(f"{out}: config refit lệch khoá {bad}")
    rec = dict(job, model="last.pt", source_history_sha256=sha256(Path(job["run_a"]) / "history.json"),
               wrapper_sha256=sha256(Path(__file__)))
    (out / "refit.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] != ["one"]:
        raise SystemExit(__doc__)
    import train_contrastive as TC
    TC.load_manifest = refit_loader(TC.load_manifest)
    sys.argv = ["train_contrastive.py", "train", *argv[1:]]
    TC.main()


if __name__ == "__main__":
    main()
