"""Chạy trọn Stage 2 decoder cho mọi run Stage 1 đã train xong.

  python -m stage2_decoder.run_all --runs runs_retry1 --splits splits --data-root /data/DISFA \
      --output downstream/stage2_decoder --variants baseline current full --seeds 0

Các bước (bỏ qua bước đã có kết quả, nên chạy lại an toàn):
  1. bảng luật cho từng fold (chỉ nhãn fit — dùng chung mọi variant/seed)
  2. với mỗi run có best.pt và history đủ epoch:
       evaluate --split val  (GPU, ~3 subject)  nếu chưa có val_predictions.npz
       evaluate --split test (GPU)              nếu chưa có test_predictions.npz
       decode (CPU, vài giây)
  3. summarize
Run chưa train xong / thiếu best.pt chỉ được liệt kê, không bị train ở đây.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # thư mục Contrastive_V1


def sh(cmd):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--splits", required=True, help="thư mục chứa fold{k}.json")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--variants", nargs="+", default=["baseline", "current", "full"])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-implication", action="store_true")
    ap.add_argument("--no-test-eval", action="store_true", help="không tự chạy evaluate --split test")
    args = ap.parse_args()

    py = sys.executable
    runs, splits, out = Path(args.runs).resolve(), Path(args.splits).resolve(), Path(args.output).resolve()
    data = Path(args.data_root).resolve()
    rules_root, res_root = out / "rules", out / "results"

    for k in args.folds:
        if not (rules_root / f"fold{k}" / "rules.json").exists():
            sh([py, "-m", "stage2_decoder.build_rules", "--data-root", data, "--manifest", splits / f"fold{k}.json",
                "--output", rules_root] + (["--no-implication"] if args.no_implication else []))

    skipped = []
    for v in args.variants:
        for k in args.folds:
            for s in args.seeds:
                run = runs / v / f"fold{k}" / f"seed{s}"
                tag = f"{v}/fold{k}/seed{s}"
                cfg, hist, best = run / "config.json", run / "history.json", run / "best.pt"
                if not (cfg.exists() and hist.exists() and best.exists()):
                    skipped.append(f"{tag}: thiếu config/history/best.pt")
                    continue
                h = json.loads(hist.read_text(encoding="utf-8"))
                if not h or h[-1]["epoch"] != json.loads(cfg.read_text(encoding="utf-8"))["epochs"]:
                    skipped.append(f"{tag}: chưa train đủ epoch")
                    continue
                ev = run / "evaluation"
                for split in ("val", "test"):
                    if (ev / f"{split}_predictions.npz").exists():
                        continue
                    if split == "test" and args.no_test_eval:
                        break
                    sh([py, "train_contrastive.py", "evaluate", "--split", split, "--checkpoint", best,
                        "--data-root", data, "--manifest", splits / f"fold{k}.json", "--output", ev,
                        "--batch-size", args.batch_size, "--workers", args.workers])
                if not (ev / "test_predictions.npz").exists():
                    skipped.append(f"{tag}: chưa có test_predictions")
                    continue
                if (res_root / v / f"fold{k}" / f"seed{s}" / "decode_summary.json").exists():
                    continue
                sh([py, "-m", "stage2_decoder.decode", "--run", run, "--rules", rules_root / f"fold{k}",
                    "--manifest", splits / f"fold{k}.json", "--data-root", data, "--output", res_root])

    sh([py, "-m", "stage2_decoder.summarize", "--results", res_root, "--variants", *args.variants,
        "--folds", *args.folds, "--seeds", *args.seeds, "--output", out / "summary.json"])
    if skipped:
        print(f"\nBỏ qua {len(skipped)} run:\n  " + "\n  ".join(skipped))


if __name__ == "__main__":
    main()
