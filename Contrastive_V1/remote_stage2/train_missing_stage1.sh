#!/usr/bin/env bash
# (Tuỳ chọn) Train các run Stage 1 còn thiếu theo đúng EXPERIMENT_PROTOCOL / COLAB_PRO:
# ResNet50, 20 epoch, batch 32, lr 1e-4, cùng manifest. Có last.pt thì resume, có best+đủ epoch thì bỏ qua.
#   SYM=~/Sym SEEDS="1 2" nohup bash Contrastive_V1/remote_stage2/train_missing_stage1.sh > ~/Sym/logs/stage1_train.log 2>&1 &
# Yêu cầu: $SYM/Contrastive_V1/checkpoints/resnet50-19c8e357.pth
set -euo pipefail
SYM=${SYM:-$HOME/Sym}
RUNS=${RUNS:-runs_retry1}
VARIANTS=${VARIANTS:-"baseline current full"}
FOLDS=${FOLDS:-"1 2 3"}
SEEDS=${SEEDS:-"1 2"}
WORKERS=${WORKERS:-8}
cd "$SYM/Contrastive_V1"
source "$SYM/.venv/bin/activate"
[ -f checkpoints/resnet50-19c8e357.pth ] || {
  mkdir -p checkpoints
  wget -q -O checkpoints/resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth; }

for V in $VARIANTS; do for F in $FOLDS; do for S in $SEEDS; do
  OUT="$SYM/$RUNS/$V/fold$F/seed$S"
  if [ -f "$OUT/history.json" ] && python - "$OUT" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]); h = json.loads((p/'history.json').read_text())
sys.exit(0 if h and h[-1]['epoch'] == json.loads((p/'config.json').read_text())['epochs'] and (p/'best.pt').exists() else 1)
EOF
  then echo "== $V fold$F seed$S: đã xong"; continue; fi
  CMD=(python train_contrastive.py train --variant "$V" --seed "$S" --arc resnet50
       --data-root "$SYM/DISFA" --manifest "$SYM/splits/fold$F.json" --output "$OUT"
       --epochs 20 --batch-size 32 --workers "$WORKERS")
  [ -f "$OUT/last.pt" ] && CMD+=(--resume "$OUT/last.pt")
  echo "== $V fold$F seed$S: ${CMD[*]}"
  "${CMD[@]}"
done; done; done
