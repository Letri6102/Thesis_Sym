"""Tạo .venv-pc ở gốc repo với đúng phiên bản thư viện đã khoá ngày 22/11. Windows hoặc Linux.

  python pc/setup_env.py            # cần Python 3.10–3.14 và driver NVIDIA chạy được CUDA 13.0
  python pc/setup_env.py --dry-run  # chỉ in các lệnh

Phiên bản phải khớp LOCK["env"] trong Contrastive_V1/stage2_decoder/contrast_ci.py (pytest kiểm hai nơi khớp nhau).
Sai một phiên bản là config của run lệch khoá, và cell nhập trên Colab sẽ từ chối run đó.
"""
import argparse
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv-pc"
LOCK_ENV = {"torch": "2.14.1+cu130", "torchvision": "0.29.1+cu130", "numpy": "2.4.6", "timm": "1.0.22",
            "pillow": "11.3.0"}
STEPS = [
    [f"torch=={LOCK_ENV['torch']}", f"torchvision=={LOCK_ENV['torchvision']}",
     "--index-url", "https://download.pytorch.org/whl/cu130"],
    # cài sau torch để ghim lại numpy/pillow mà torchvision có thể đã kéo về theo bản khác
    [f"numpy=={LOCK_ENV['numpy']}", f"pillow=={LOCK_ENV['pillow']}", f"timm=={LOCK_ENV['timm']}", "pytest==8.4.2"],
]
CHECK = ("import json, importlib.metadata as M; "
         "print(json.dumps({k: M.version(k) for k in ('torch', 'torchvision', 'numpy', 'timm', 'pillow')}))")


def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not (3, 10) <= sys.version_info[:2] <= (3, 14):
        raise SystemExit(f"Cần Python 3.10–3.14 (đang là {sys.version.split()[0]}); torch 2.14.1+cu130 chỉ có bản cho các phiên bản này.")
    py = venv_python()
    commands = [[str(py), "-m", "pip", "install", "--upgrade", "pip"]] + [[str(py), "-m", "pip", "install", *s] for s in STEPS]
    if args.dry_run:
        print(f"python -m venv {VENV}")
        print("\n".join(" ".join(c) for c in commands))
        return commands
    if not py.exists():
        print(f"Tạo môi trường {VENV} ...", flush=True)
        venv.create(VENV, with_pip=True)
    for c in commands:
        print("\n$ " + " ".join(c), flush=True)
        subprocess.run(c, check=True)
    got = json.loads(subprocess.run([str(py), "-c", CHECK], check=True, capture_output=True, text=True).stdout)
    if got != LOCK_ENV:
        raise SystemExit(f"Phiên bản sau khi cài KHÁC khoá:\n  có  {got}\n  cần {LOCK_ENV}")
    print(f"\nOK: môi trường khớp khoá. Từ giờ dùng python này:\n  {py}")
    return commands


if __name__ == "__main__":
    main()
