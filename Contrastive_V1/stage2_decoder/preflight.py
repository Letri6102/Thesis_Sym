"""Kiểm tra máy remote trước khi chạy Stage 2 decoder (và train Stage 1 còn thiếu nếu có).

  python -m stage2_decoder.preflight --sym ~/Sym
Bố cục mong đợi (giống Google Drive MyDrive/Sym):
  <sym>/DISFA/img, <sym>/DISFA/list        (hoặc --data-root nếu ảnh đã chép về /content)
  <sym>/splits_v2/fold{1,2,3}.json
  <sym>/runs_aug1/<variant>/fold{k}/seed{s}/{config.json,history.json,best.pt,evaluation_val/,evaluation/}
Code lấy từ git (thư mục chứa gói stage2_decoder), không đọc từ <sym>.
"""
import argparse
import random
import sys
from pathlib import Path

from .core import load_split

fails = []


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'FAIL':4}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sym", required=True)
    ap.add_argument("--runs", default="runs_aug1")
    ap.add_argument("--splits", default="splits_v2")
    ap.add_argument("--data-root", default=None, help="mặc định <sym>/DISFA")
    ap.add_argument("--n-sample", type=int, default=500)
    ap.add_argument("--need-pretrained", action="store_true", help="kiểm tra cả checkpoints/resnet50 (chỉ cần khi train Stage 1)")
    args = ap.parse_args()
    sym = Path(args.sym).expanduser()
    code = Path(__file__).resolve().parent.parent
    data = Path(args.data_root) if args.data_root else sym / "DISFA"
    splits, runs = sym / args.splits, sym / args.runs

    check("Python ≥ 3.9", sys.version_info >= (3, 9), sys.version.split()[0])
    try:
        import torch
        ok = torch.cuda.is_available()
        check("torch + CUDA", ok, f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
              + (torch.cuda.get_device_name(0) if ok else "KHÔNG có GPU (evaluate sẽ rất chậm)"))
    except ImportError as e:
        check("torch", False, str(e))
    for mod in ("torchvision", "timm", "PIL", "numpy"):
        try:
            __import__(mod)
            check(f"import {mod}", True)
        except ImportError as e:
            check(f"import {mod}", False, str(e))

    check("code Contrastive_V1/train_contrastive.py", (code / "train_contrastive.py").exists(), str(code))
    check("matrixMAE/M_AE_DISFA.npy", (code / "matrixMAE" / "M_AE_DISFA.npy").exists())
    if args.need_pretrained:
        check("checkpoints/resnet50-19c8e357.pth", (code / "checkpoints" / "resnet50-19c8e357.pth").exists(),
              "chỉ cần khi train Stage 1 mới")

    for k in (1, 2, 3):
        mf = splits / f"fold{k}.json"
        if not mf.exists():
            check(f"{args.splits}/fold{k}.json", False, f"không có trong {splits}")
            continue
        try:
            sp = load_split(mf, data)            # kiểm hash list ↔ manifest
            n = {s: len(sp[s]["paths"]) for s in ("fit", "val", "test")}
            sample = random.sample(sp["val"]["paths"] + sp["test"]["paths"], args.n_sample)
            miss = [p for p in sample if not (data / "img" / p).is_file()]
            check(f"fold{k}: manifest khớp list (hash) — fit {n['fit']}, val {n['val']}, test {n['test']} frame; "
                  f"val subjects {sorted(map(str, set(sp['val']['subjects'])))}", True)
            check(f"fold{k}: ảnh tồn tại ({args.n_sample} mẫu)", not miss, f"thiếu {len(miss)}, vd. {miss[:2]}" if miss else "")
        except Exception as e:
            check(f"fold{k}: manifest/list", False, f"{type(e).__name__}: {e}")

    check(f"thư mục runs {runs}", runs.is_dir())
    print("\n" + ("SẴN SÀNG." if not fails else f"{len(fails)} mục FAIL: {fails}"))
    print(f"Trạng thái từng run: python -m stage2_decoder.check_stage1 --runs {runs}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
