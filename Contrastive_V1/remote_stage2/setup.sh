#!/usr/bin/env bash
# Cài môi trường trên máy Linux + GPU.  Dùng:
#   SYM=~/Sym bash Contrastive_V1/remote_stage2/setup.sh
#   CUDA_TAG=cu126 SYM=~/Sym bash ...      # nếu driver chưa hỗ trợ CUDA 12.8
set -euo pipefail
SYM=${SYM:-$HOME/Sym}
CUDA_TAG=${CUDA_TAG:-cu128}
PY=${PY:-python3}
cd "$SYM/Contrastive_V1"

nvidia-smi || echo "!! Không thấy nvidia-smi — evaluate sẽ chạy CPU (chậm)"
$PY -m venv "$SYM/.venv"
source "$SYM/.venv/bin/activate"
pip install -U pip wheel
pip install torch==2.7.1 torchvision==0.22.1 --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install -r requirements-colab.txt "numpy<2.3" pillow
pip freeze > "$SYM/remote-environment.txt"

python -m pytest tests -q || echo "!! tests lỗi — xem lại trước khi chạy"
python -m stage2_decoder.preflight --sym "$SYM"
python -m stage2_decoder.check_stage1 --runs "$SYM/runs_retry1"
