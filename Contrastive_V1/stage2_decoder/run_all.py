"""Chạy trọn Stage 2 decoder cho mọi run Stage 1 đã train xong (trên Colab, xem COLAB_CELLS.md).

  python -m stage2_decoder.run_all --runs <Sym>/runs_aug1 --splits <Sym>/splits_v2 --data-root <DISFA> \
      --output <Sym>/stage2_v2 --variants baseline full --seeds 0 [--val-only]

Các bước (bỏ qua bước đã có kết quả, nên chạy lại an toàn):
  1. bảng luật cho từng fold (chỉ nhãn fit — dùng chung mọi variant/seed)
  2. với mỗi run có best.pt và history đủ epoch:
       evaluate --split val  → <run>/evaluation_val/   (GPU, 3 subject)  nếu chưa có
       evaluate --split test → <run>/evaluation/       (GPU)             nếu chưa có — BỎ QUA khi --val-only
       decode (CPU, vài giây)
  3. summarize (chỉ khi đã mở test)
--val-only: giai đoạn thiết kế — mọi lựa chọn chỉ dựa trên val, outer test không bị đụng tới.
Run chưa train xong / thiếu best.pt chỉ được liệt kê, không bị train ở đây.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .core import EVAL_DIRS, prediction_dir
from .decode import TEST_FILE, VAL_FILE

ROOT = Path(__file__).resolve().parent.parent          # thư mục Contrastive_V1


def sh(cmd):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--splits", required=True, help="thư mục chứa fold{k}.json (manifest mà run đã train)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--variants", nargs="+", default=["baseline", "full"])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-implication", action="store_true")
    ap.add_argument("--val-only", action="store_true", help="không evaluate/đọc outer test")
    args = ap.parse_args(argv)

    py = sys.executable
    runs, splits, out = Path(args.runs).resolve(), Path(args.splits).resolve(), Path(args.output).resolve()
    data = Path(args.data_root).resolve()
    rules_root, res_root = out / "rules", out / "results"
    result_file = VAL_FILE if args.val_only else TEST_FILE

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
                for split in ("val",) if args.val_only else ("val", "test"):
                    if (prediction_dir(run, split) / f"{split}_predictions.npz").exists():
                        continue
                    sh([py, "train_contrastive.py", "evaluate", "--split", split, "--checkpoint", best,
                        "--data-root", data, "--manifest", splits / f"fold{k}.json",
                        "--output", run / EVAL_DIRS[split][0],
                        "--batch-size", args.batch_size, "--workers", args.workers])
                if (res_root / v / f"fold{k}" / f"seed{s}" / result_file).exists():
                    continue
                sh([py, "-m", "stage2_decoder.decode", "--run", run, "--rules", rules_root / f"fold{k}",
                    "--manifest", splits / f"fold{k}.json", "--data-root", data, "--output", res_root]
                   + (["--val-only"] if args.val_only else []))

    if not args.val_only:
        sh([py, "-m", "stage2_decoder.summarize", "--results", res_root, "--variants", *args.variants,
            "--folds", *args.folds, "--seeds", *args.seeds, "--output", out / "summary.json"])
    if skipped:
        print(f"\nBỏ qua {len(skipped)} run:\n  " + "\n  ".join(skipped))


if __name__ == "__main__":
    main()
