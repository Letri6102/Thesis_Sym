"""Kiểm tra checkpoint Stage 1 (runs của train_contrastive.py) trước khi chạy decoder.

  python -m stage2_decoder.check_stage1 --runs <Sym>/runs_aug1 [--hash]

Mỗi run <variant>/fold{k}/seed{s} cần: config.json, history.json đủ epoch, best.pt
(đúng epoch tốt nhất trên val), evaluation_val/val_* và evaluation/test_* từ CÙNG best.pt
(bản cũ để val ở evaluation/ vẫn được nhận).
--hash: tính sha256 của best.pt để đối chiếu checkpoint_hash trong *_metrics.json (chậm, ~0.5 GB/file).
"""
import argparse
import json
from pathlib import Path

from .core import digest, prediction_dir


def status(run, do_hash):
    r = dict(run=str(run), actions=[], notes=[])
    cfgp, hp, best = run / "config.json", run / "history.json", run / "best.pt"
    if not cfgp.exists():
        r["state"] = "chưa chạy"
        r["actions"].append("train")
        return r
    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
    hist = json.loads(hp.read_text(encoding="utf-8")) if hp.exists() else []
    done, total = (hist[-1]["epoch"] if hist else 0), cfg["epochs"]
    r.update(epochs=f"{done}/{total}")
    if done < total:
        r["state"] = "đang train dở"
        r["actions"].append("resume" if (run / "last.pt").exists() else "train lại (thiếu last.pt)")
        return r
    sel = max(hist, key=lambda x: x["val"]["macro_f1"])        # max() giữ epoch sớm nhất khi hoà
    r.update(best_epoch=sel["epoch"], val_f1=round(100 * sel["val"]["macro_f1"], 2))
    if sel["epoch"] <= 2:
        r["notes"].append(f"best ở epoch {sel['epoch']}/{total}")
    if not best.exists():
        r["state"] = "THIẾU best.pt"
        r["actions"].append("tải best.pt về (đã train xong nhưng file không có ở đây)")
        return r
    hashes = {}
    for split in ("val", "test"):
        ev = prediction_dir(run, split)
        mj, nz = ev / f"{split}_metrics.json", ev / f"{split}_predictions.npz"
        if mj.exists() and nz.exists():
            m = json.loads(mj.read_text(encoding="utf-8"))
            hashes[split] = m["checkpoint_hash"]
            if m.get("selected_epoch") != sel["epoch"]:
                r["notes"].append(f"{split}_metrics từ epoch {m.get('selected_epoch')} ≠ best {sel['epoch']}")
            if split == "test":
                r["test_f1"] = round(100 * m["macro_f1"], 2)
        else:
            r["actions"].append(f"evaluate --split {split}")
    if len(set(hashes.values())) > 1:
        r["notes"].append("val và test từ hai checkpoint khác nhau → xoá file val_* rồi evaluate lại")
    if do_hash and hashes:
        h = digest(best)
        if any(v != h for v in hashes.values()):
            r["notes"].append("best.pt hiện tại KHÁC checkpoint đã evaluate → evaluate lại")
    r["state"] = "SẴN SÀNG" if not r["actions"] and not any("≠" in n or "KHÁC" in n or "khác" in n for n in r["notes"]) else "cần xử lý"
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--variants", nargs="+", default=["baseline", "full"])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hash", action="store_true")
    args = ap.parse_args()

    rows = [status(Path(args.runs) / v / f"fold{k}" / f"seed{s}", args.hash) | dict(v=v, k=k, s=s)
            for v in args.variants for k in args.folds for s in args.seeds]
    print(f"{'variant':<10}{'fold':>5}{'seed':>5}{'epoch':>8}{'best':>6}{'val F1':>8}{'test F1':>9}  trạng thái / việc cần làm")
    for r in rows:
        print(f"{r['v']:<10}{r['k']:>5}{r['s']:>5}{r.get('epochs', '-'):>8}{r.get('best_epoch', '-'):>6}"
              f"{r.get('val_f1', '-'):>8}{r.get('test_f1', '-'):>9}  {r['state']}"
              + (f" | {', '.join(r['actions'])}" if r["actions"] else "")
              + (f" | {'; '.join(r['notes'])}" if r["notes"] else ""))
    ready = sum(r["state"] == "SẴN SÀNG" for r in rows)
    print(f"\nSẵn sàng cho decoder: {ready}/{len(rows)}")
    for s in args.seeds:
        ok = [v for v in args.variants
              if all(r["state"] == "SẴN SÀNG" for r in rows if r["v"] == v and r["s"] == s)]
        print(f"  seed{s}: variant đủ 3 fold = {ok if ok else 'chưa có'}")


if __name__ == "__main__":
    main()
