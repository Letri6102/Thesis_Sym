"""Gom một thư mục để chép nguyên sang máy GPU riêng (ví dụ D:\\LVTS_TriLe qua UltraViewer). Chạy ở máy cá nhân.

  <py> pc/make_bundle.py --splits <thư mục chứa splits_v2/fold*.json> --out C:\\CODE\\Symbolic\\LVTS_TriLe

Kết quả:
  <out>/Thesis_Sym/                      code đúng commit HEAD (git archive, giữ nguyên byte)
  <out>/sym_download/DISFA_img/          27 tar dựng từ img/ ở máy này, đúng định dạng cell A
  <out>/sym_download/DISFA/list/         15 file list
  <out>/sym_download/splits_v2/          3 manifest đã khoá
  <out>/1_cai_dat.bat, 2_train.bat       chạy trên máy GPU
  <out>/tar_sha256.json                  để cell J trên Colab đối chiếu với DISFA_img trên Drive
Không có trọng số ImageNet: 1_cai_dat.bat tải bản chính thức từ download.pytorch.org (prepare kiểm SHA-256).
Ảnh ở máy này chỉ được dùng để train khi cell J báo KHỚP đủ 27/27 (CLAUDE.md §1).
"""
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Contrastive_V1"))
from stage2_decoder.contrast_ci import LOCK, source_digest  # noqa: E402

N_IMAGES = 130814


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_digest(code_root):
    """Cùng phép tính với train_contrastive.source_files(), nhưng không cần torch."""
    c = Path(code_root)
    paths = list(c.glob("*.py")) + list((c / "model").glob("*.py")) + list((c / "config").glob("*.yaml"))
    paths += [c / "matrixMAE" / "M_AE_DISFA.npy", c / "EXPERIMENT_PROTOCOL.md"]
    return source_digest({p.relative_to(c).as_posix(): sha256(p) for p in sorted(paths)})


def image_list(lists):
    return sorted({p.strip() for k in (1, 2, 3)
                   for p in (Path(lists) / f"DISFA_test_img_path_fold{k}.txt").read_text().split() if p.strip()})


def write_tar(path, img_root, names):
    """Giống cell A: GNU format, tên 'img/<SNxxx>/<n>.png', mtime 0, thứ tự theo list đã sort."""
    with tarfile.open(path, "w", format=tarfile.GNU_FORMAT) as tar:
        for p in names:
            data = (Path(img_root) / p).read_bytes()
            info = tarfile.TarInfo("img/" + p)
            info.size, info.mtime = len(data), 0
            tar.addfile(info, io.BytesIO(data))


def build(img, lists, splits, out, lock=LOCK, n_images=N_IMAGES, ref="HEAD"):
    out, lists = Path(out), Path(lists)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} đã có nội dung: chọn thư mục mới hoặc xoá thư mục cũ trước")
    for k in (1, 2, 3):
        f = Path(splits) / f"fold{k}.json"
        if sha256(f) != lock["manifests"][k]:
            raise SystemExit(f"{f} khác bản đã khoá")
        for name, h in json.loads(f.read_text(encoding="utf-8"))["input_hashes"].items():
            if sha256(lists / name) != h:
                raise SystemExit(f"{lists / name} khác bản manifest đã ghi")
    rel = image_list(lists)
    if len(rel) != n_images:
        raise SystemExit(f"List có {len(rel)} ảnh, cần {n_images}")
    missing = [p for p in rel if not (Path(img) / p).is_file()]
    if missing:
        raise SystemExit(f"Thiếu {len(missing)} ảnh trong {img}, ví dụ {missing[:3]}")

    code = out / "Thesis_Sym"
    code.mkdir(parents=True)
    blob = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", ref], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as t:
        t.extractall(code, filter="data")
    if code_digest(code / "Contrastive_V1") != lock["source"]:
        raise SystemExit(f"Code ở {ref} khác bản đã khoá: không đóng gói")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", ref], check=True,
                            capture_output=True, text=True).stdout.strip()

    dl = out / "sym_download"
    shutil.copytree(lists, dl / "DISFA" / "list", ignore=lambda d, names: [n for n in names if not n.endswith(".txt")])
    (dl / "splits_v2").mkdir(parents=True)
    for k in (1, 2, 3):
        shutil.copyfile(Path(splits) / f"fold{k}.json", dl / "splits_v2" / f"fold{k}.json")

    tars, subjects = dl / "DISFA_img", sorted({p.split("/")[0] for p in rel})
    tars.mkdir()
    digests, total = {}, 0
    for i, s in enumerate(subjects, 1):
        path = tars / f"{s}.tar"
        names = [p for p in rel if p.split("/")[0] == s]
        write_tar(path, img, names)
        size = path.stat().st_size
        (tars / f"{s}.json").write_text(json.dumps(dict(n=len(names), size=size)))
        digests[s] = dict(size=size, sha256=sha256(path))
        total += size
        print(f"  {i:>2}/{len(subjects)} {s}: {len(names)} ảnh, {size / 1e6:.0f} MB", flush=True)
    (tars / "DONE.json").write_text(json.dumps(dict(subjects=subjects, n_images=len(rel), bytes=total)))
    (out / "tar_sha256.json").write_text(json.dumps(dict(commit=commit, tars=digests), indent=1))
    for bat in sorted((ROOT / "pc" / "windows").glob("*.bat")):
        shutil.copyfile(bat, out / bat.name)
    print(f"XONG: {out} | code {commit} | {len(rel)} ảnh trong {len(subjects)} tar, {total / 1e9:.2f} GB", flush=True)
    return digests


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img", default=str(ROOT / "img"))
    ap.add_argument("--lists", default=str(ROOT / "DISFA" / "list"))
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    build(a.img, a.lists, a.splits, a.out)


if __name__ == "__main__":
    main()
