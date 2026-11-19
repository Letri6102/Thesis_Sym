#!/usr/bin/env bash
# Cài môi trường trên máy Linux có GPU. Chạy từ bất kỳ đâu:
#   bash remote/setup_env.sh                 # mặc định CUDA 12.8 (torch 2.7.1 như máy Windows)
#   CUDA_TAG=cu126 bash remote/setup_env.sh  # nếu driver chỉ hỗ trợ CUDA 12.6
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
CUDA_TAG=${CUDA_TAG:-cu128}

echo "== GPU / driver"
nvidia-smi || { echo "Không thấy nvidia-smi — máy chưa có driver GPU?"; exit 1; }

echo "== Tạo venv (.venv) bằng $($PY --version)"
$PY -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install torch==2.7.1 torchvision==0.22.1 --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install -r remote/requirements.txt

echo "== Sửa đường dẫn kiểu Windows (matrixMAE\\M_AE_DISFA.npy) — dấu / chạy được cả Windows lẫn Linux"
sed -i 's#r"matrixMAE\\M_AE_DISFA\.npy"#"matrixMAE/M_AE_DISFA.npy"#' train_Sym_Stage_1.py train_Sym_Stage_2.py

echo "== Kiểm tra trước khi train"
python remote/preflight.py "$@"
