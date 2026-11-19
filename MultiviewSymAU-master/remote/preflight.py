"""Kiểm tra máy remote đã đủ mọi thứ để chạy thí nghiệm decoder chưa.

  python remote/preflight.py --arc resnet50 --dataset DISFA
In OK / FAIL cho từng mục; thoát mã 1 nếu có FAIL.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PRETRAINED = {
    "resnet18": "resnet18-5c106cde.pth", "resnet50": "resnet50-19c8e357.pth",
    "resnet101": "resnet101-5d3b4d8f.pth",
    "swin_transformer_tiny": "swin_tiny_patch4_window7_224.pth",
    "swin_transformer_small": "swin_small_patch4_window7_224.pth",
    "swin_transformer_base": "swin_base_patch4_window7_224.pth",
}
fails = []


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'FAIL':4}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc", default="resnet50")
    ap.add_argument("--dataset", default="DISFA")
    ap.add_argument("--n-sample", type=int, default=300, help="số ảnh ngẫu nhiên kiểm tra tồn tại mỗi list")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()

    check("Python", sys.version_info >= (3, 9), sys.version.split()[0])
    try:
        import torch
        cuda = torch.cuda.is_available()
        check("torch + CUDA", cuda, f"torch {torch.__version__}, CUDA build {torch.version.cuda}, "
              + (", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) if cuda else "không có GPU"))
    except ImportError as e:
        check("torch", False, str(e))
        torch = None
    for mod in ("torchvision", "timm", "easydict", "yaml", "pandas", "tqdm", "PIL"):
        try:
            __import__(mod)
            check(f"import {mod}", True)
        except ImportError as e:
            check(f"import {mod}", False, str(e))

    for f in ("train_Sym_Stage_1.py", "train_Sym_Stage_2.py"):
        if os.path.exists(f):
            txt = open(f, encoding="utf-8").read()
            check(f"{f}: đường dẫn matrixMAE dạng Linux", 'r"matrixMAE\\M_AE' not in txt,
                  "còn dấu \\ — chạy lệnh sed trong setup_env.sh")
    check("matrixMAE/M_AE_DISFA.npy", os.path.exists("matrixMAE/M_AE_DISFA.npy"))
    ck = os.path.join("checkpoints", PRETRAINED[args.arc])
    check(f"backbone pretrained {ck}", os.path.exists(ck), "tải về thư mục checkpoints/ (xem README_remote.md)")

    import yaml
    cfg = yaml.safe_load(open(os.path.join("config", f"{args.dataset}_config.yaml")))
    root = cfg["dataset_path"]
    check(f"dataset_path = {root}", os.path.isdir(root), "sửa config/<DS>_config.yaml nếu data để chỗ khác")
    check(f"{root}/img", os.path.isdir(os.path.join(root, "img")))

    from stage2_rules.common import list_file, read_split
    for fold in (1, 2, 3):
        for split in ("train", "test"):
            try:
                paths, labels, subj = read_split(root, args.dataset, split, fold)
                miss = [p for p in random.sample(paths, min(args.n_sample, len(paths)))
                        if not os.path.exists(os.path.join(root, "img", p))]
                check(f"fold{fold} {split}: {len(paths)} frame, {len(set(subj))} subject, ảnh mẫu tồn tại",
                      not miss, f"thiếu {len(miss)}/{args.n_sample}, vd. {miss[:2]}" if miss else "")
            except Exception as e:
                check(f"fold{fold} {split} list", False, str(e))
        for f in (os.path.join(root, "list", f"{args.dataset}_weight_fold{fold}.txt"),
                  list_file(root, args.dataset, "train", "weight", fold)):
            check(os.path.basename(f), os.path.exists(f), "Stage 1 đọc file này")
        fit = list_file(root, args.dataset, "fit", "img_path", fold)
        print(f"[info] fold{fold} fit/inner split: {'đã có' if os.path.exists(fit) else 'chưa có (run_trial.sh sẽ tạo)'}")

    if torch is not None and not args.skip_model:
        try:
            from model.SymStage1 import MEFARGStage1
            net = MEFARGStage1(num_aus=cfg["num_classes"], backbone=args.arc, num_expr=7, use_contrastive=True)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            net = net.to(dev).eval()
            with torch.no_grad():
                out = net(torch.randn(2, 3, 224, 224, device=dev))
            check("dựng MEFARGStage1 + forward 1 batch", out[2].shape == (2, cfg["num_classes"]),
                  f"p_a {tuple(out[2].shape)} trên {dev}")
        except Exception as e:
            check("dựng MEFARGStage1 + forward 1 batch", False, f"{type(e).__name__}: {e}")

    print("\n" + ("SẴN SÀNG." if not fails else f"{len(fails)} mục FAIL: {fails}"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
