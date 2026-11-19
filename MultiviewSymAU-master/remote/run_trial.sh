#!/usr/bin/env bash
# Chạy trọn thí nghiệm decoder cố định cho các fold:
#   split fit/inner → Stage 1 trên fit → luật → xác suất → decoder → tổng hợp
# Chạy nền, tắt SSH vẫn chạy tiếp:
#   nohup bash remote/run_trial.sh > logs/trial.log 2>&1 &
#   tail -f logs/trial.log
# Tuỳ chỉnh: ARC=swin_transformer_base BS=32 FOLDS="2" WORKERS=8 GPU=0 bash remote/run_trial.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

DS=${DS:-DISFA}
ARC=${ARC:-resnet50}
BS=${BS:-64}
LR=${LR:-0.0001}
EPOCHS=${EPOCHS:-20}
FOLDS=${FOLDS:-"1 2 3"}
WORKERS=${WORKERS:-8}
GPU=${GPU:-0}
EXP=${EXP:-${ARC}_stage1_fit}
mkdir -p logs

# thư mục output giống conf.set_outdir: results/<exp>/bs_<bs>_seed_0_lr_<float(lr)>
OUT="results/${EXP}/$(python -c "print(f'bs_{int(\"$BS\")}_seed_0_lr_{float(\"$LR\")}')")"

if [ ! -f "data/${DS}/list/${DS}_fit_img_path_fold1.txt" ]; then
  python -m stage2_rules.make_inner_split --dataset "$DS" --n-inner 6 | tee "logs/inner_split.log"
fi

for F in $FOLDS; do
  echo "================ fold $F ================"
  if [ ! -f "${OUT}/.done_fold${F}" ]; then
    python train_Sym_Stage_1_fit.py --dataset "$DS" --arc "$ARC" --exp-name "$EXP" \
        -b "$BS" -lr "$LR" -e "$EPOCHS" --fold "$F" -j "$WORKERS" --gpu_ids "$GPU" \
        2>&1 | tee "logs/stage1_fit_fold${F}.log"
    touch "${OUT}/.done_fold${F}"          # chỉ đánh dấu khi train chạy hết (set -e)
  else
    echo "Stage 1 fold $F đã xong — bỏ qua"
  fi
  python -m stage2_rules.build_rules --dataset "$DS" --fold "$F" | tee "logs/rules_fold${F}.log"
  CUDA_VISIBLE_DEVICES="$GPU" python -m stage2_rules.dump_probs --dataset "$DS" --arc "$ARC" --fold "$F" \
      --ckpt "${OUT}/best_model_fold${F}.pth" --splits inner test -b "$BS" -j "$WORKERS"
  python -m stage2_rules.decode_eval --dataset "$DS" --fold "$F" | tee "logs/decode_fold${F}.log"
done

python -m stage2_rules.summarize --dataset "$DS" | tee "logs/summary.log"
python -m stage2_rules.check_stage1_ckpts --results results --dataset "$DS" | tee "logs/check_ckpts.log"
