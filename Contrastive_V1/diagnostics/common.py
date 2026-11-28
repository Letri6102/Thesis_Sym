"""Phần dùng chung: nạp checkpoint của train_contrastive.py, kiểm manifest, chép checkpoint khỏi Drive, ghi JSON."""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def load_model(checkpoint, weights="eval"):
    """Dựng MEFARGStage1 đúng cấu hình của run và nạp đủ trọng số, kể cả projection head.

    weights="eval": `state_dict` (trọng số dùng để evaluate; là EMA nếu run có EMA).
    weights="train": `train_state_dict` nếu có (trọng số đang train), không thì `state_dict`.
    Chỉ nạp checkpoint do chính dự án tạo ra: torch.load(weights_only=False) dùng pickle.
    """
    import torch
    from model.SymStage1 import MEFARGStage1

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise ValueError(f"{checkpoint}: thiếu config, không phải checkpoint Tầng 1 của train_contrastive.py")
    cfg = ckpt["config"]
    model = MEFARGStage1(num_aus=8, num_expr=7, backbone=cfg["arc"], use_contrastive=True,
                         proj_dim=cfg.get("proj_dim", 128), pretrained=False)
    state = ckpt["state_dict"] if weights == "eval" else ckpt.get("train_state_dict", ckpt["state_dict"])
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    meta = dict(epoch=int(ckpt["epoch"]) + 1, progress=float(ckpt.get("progress", ckpt["epoch"] + 1)),
                initial_detector_hash=ckpt.get("initial_detector_hash"))
    return model, cfg, meta


def check_manifest(cfg, manifest):
    from experiment_data import digest
    if digest(manifest) != cfg["manifest_hash"]:
        raise ValueError(f"{manifest}: manifest khác manifest lúc train run này")


def staged_copy(src, stage_dir, timeout=1800):
    """Chép file về đĩa cục bộ trước khi torch.load. Đọc Drive có thể treo không báo lỗi (CLAUDE.md §5), nên
    trên Linux dùng `cp` có timeout. stage_dir=None: dùng thẳng file gốc."""
    src = Path(src)
    if stage_dir is None:
        return src
    dst = Path(stage_dir) / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        shutil.copy2(src, dst)
    else:
        subprocess.run(["cp", str(src), str(dst)], check=True, timeout=timeout)
    return dst


def clean(x):
    """Đổi sang kiểu JSON chuẩn; NaN/inf → None (json.dumps mặc định ghi NaN, không phải JSON hợp lệ)."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
