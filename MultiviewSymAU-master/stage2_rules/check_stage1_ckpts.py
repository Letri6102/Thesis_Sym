"""Kiểm tra checkpoint Stage 1: đủ fold chưa, train hết epoch chưa, chọn best trên split nào.

  python -m stage2_rules.check_stage1_ckpts --results results --dataset DISFA

Với mỗi thư mục run (chứa train.log và/hoặc *.pth) in ra:
  - fold, số epoch đã chạy / tổng, epoch + F1 của best
  - val_data_num trong log → suy ra best được chọn trên outer test hay inner
  - best_model_fold{k}.pth: nạp được không, epoch bên trong, đủ nhóm tham số chưa
Cuối cùng: fold nào đã có checkpoint hợp lệ cho thí nghiệm decoder (train trên fit, chọn trên inner).
"""
import argparse
import os
import re

from .common import read_split

GROUPS = ["backbone.", "global_linear.", "au_head.", "expr_head."]
RE_FOLD = re.compile(r"Fold: \[(\d+) \| \d+\s+val_data_num: (\d+)")
RE_EPOCH = re.compile(r"Epoch: \[(\d+) \| (\d+) LR")
RE_BEST = re.compile(r"\[BEST\] Epoch (\d+): F1=([\d.]+)%")


def parse_log(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    starts = [m.start() for m in RE_FOLD.finditer(text)]
    if not starts:
        return None
    seg = text[starts[-1]:]                 # chỉ lần chạy cuối (log được ghi nối tiếp)
    fold, nval = map(int, RE_FOLD.search(seg).groups())
    ep = [tuple(map(int, m.groups())) for m in RE_EPOCH.finditer(seg)]
    best = [(int(a), float(b)) for a, b in RE_BEST.findall(seg)]
    return {"fold": fold, "val_data_num": nval,
            "epoch_done": ep[-1][0] if ep else 0, "epoch_total": ep[-1][1] if ep else 0,
            "best": best[-1] if best else None, "n_runs_in_log": len(starts)}


def inspect_ckpt(path):
    try:
        import torch
    except ImportError:
        return {"ok": None, "msg": "chưa cài torch — bỏ qua bước nạp file"}
    try:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(path, map_location="cpu")
    except Exception as e:  # file hỏng / ghi dở
        return {"ok": False, "msg": f"KHÔNG nạp được: {type(e).__name__}: {e}"}
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else {}
    keys = [k[7:] if k.startswith("module.") else k for k in sd.keys()]
    missing = [g for g in GROUPS if not any(k.startswith(g) for k in keys)]
    return {"ok": not missing, "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
            "n_tensors": len(keys), "missing_groups": missing,
            "msg": "đủ nhóm tham số" if not missing else f"THIẾU nhóm: {missing}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--dataset", default="DISFA")
    ap.add_argument("--root", default=None, help="mặc định data/<DS>; dùng để đối chiếu val_data_num")
    args = ap.parse_args()
    root = args.root or os.path.join("data", args.dataset)

    sizes = {}
    for fold in (1, 2, 3):
        for split in ("test", "inner"):
            try:
                sizes[(fold, split)] = len(read_split(root, args.dataset, split, fold)[0])
            except (OSError, ValueError):
                pass

    run_dirs = sorted({dp for dp, _, fs in os.walk(args.results)
                       if "train.log" in fs or any(f.endswith(".pth") for f in fs)})
    if not run_dirs:
        print(f"Không thấy run nào trong {args.results}")
        return

    usable = {}
    for d in run_dirs:
        info = parse_log(os.path.join(d, "train.log")) if os.path.exists(os.path.join(d, "train.log")) else None
        pths = sorted(f for f in os.listdir(d) if f.endswith(".pth"))
        if not any("model_fold" in f for f in pths) and not (info and info["epoch_total"]):
            continue                                   # không phải run Stage 1 (vd. phase2)
        print(f"\n== {d}")
        val_on, finished = "?", False
        if info:
            k = info["fold"]
            if sizes.get((k, "test")) == info["val_data_num"]:
                val_on = "OUTER TEST"
            elif sizes.get((k, "inner")) == info["val_data_num"]:
                val_on = "inner"
            finished = bool(info["epoch_done"]) and info["epoch_done"] == info["epoch_total"]
            done = "XONG" if finished else "CHƯA XONG"
            b = f"epoch {info['best'][0]}, F1={info['best'][1]:.2f}%" if info["best"] else "chưa có"
            print(f"   log: fold{k} | epoch {info['epoch_done']}/{info['epoch_total']} ({done}) | best {b} | "
                  f"val_data_num={info['val_data_num']} → chọn best trên: {val_on}"
                  + (f" | log chứa {info['n_runs_in_log']} lần chạy, chỉ xét lần cuối" if info["n_runs_in_log"] > 1 else ""))
        else:
            print("   (không có train.log)")
        print(f"   file: {', '.join(pths) if pths else 'không có .pth'}")
        for f in pths:
            if f.startswith("best_model_fold"):
                r = inspect_ckpt(os.path.join(d, f))
                size_mb = os.path.getsize(os.path.join(d, f)) / 2 ** 20
                print(f"   {f}: {size_mb:.0f} MB | epoch trong file={r.get('epoch')} | {r['msg']}")
                fold = int(re.search(r"fold(\d)", f).group(1))
                if r["ok"] is not False and val_on == "inner" and finished:
                    usable.setdefault(fold, []).append(os.path.join(d, f))

    print("\n=== Checkpoint dùng được cho thí nghiệm decoder (train trên fit, chọn best trên inner) ===")
    for fold in (1, 2, 3):
        print(f"   fold{fold}: " + (", ".join(usable[fold]) if fold in usable else "CHƯA CÓ → chạy train_Sym_Stage_1_fit.py"))
    if not sizes:
        print(f"(Không đọc được list trong {root} nên không đối chiếu được val_data_num — truyền --root)")


if __name__ == "__main__":
    main()
