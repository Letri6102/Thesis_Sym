#!/usr/bin/env bash
# Stage 2 decoder cho mọi run Stage 1 đã train xong (bỏ qua bước đã có → chạy lại an toàn).
#   SYM=~/Sym nohup bash Contrastive_V1/remote_stage2/run_decoder.sh > ~/Sym/logs/stage2_decoder.log 2>&1 &
# Tuỳ chọn: SEEDS="0" VARIANTS="baseline full" RUNS=runs_retry1 WORKERS=8 NO_IMPL=1
set -euo pipefail
SYM=${SYM:-$HOME/Sym}
RUNS=${RUNS:-runs_retry1}
VARIANTS=${VARIANTS:-"baseline current full"}
SEEDS=${SEEDS:-"0 1 2"}
WORKERS=${WORKERS:-8}
BS=${BS:-32}
cd "$SYM/Contrastive_V1"
source "$SYM/.venv/bin/activate"
mkdir -p "$SYM/logs"

python -m stage2_decoder.check_stage1 --runs "$SYM/$RUNS" --variants $VARIANTS --seeds $SEEDS
python -m stage2_decoder.run_all --runs "$SYM/$RUNS" --splits "$SYM/splits" --data-root "$SYM/DISFA" \
    --output "$SYM/downstream/stage2_decoder" --variants $VARIANTS --seeds $SEEDS \
    --batch-size "$BS" --workers "$WORKERS" ${NO_IMPL:+--no-implication}
echo "Xong. Kết quả: $SYM/downstream/stage2_decoder/summary.json"
