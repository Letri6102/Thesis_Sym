"""Train các run của bộ 18 run (cấu hình khoá 22/11) trên máy GPU riêng, Windows hoặc Linux. Hướng dẫn: pc/README.md.

  <py> pc/train_18_run.py prepare --downloads <thư mục giải nén từ Drive> --data <DATA>
  <py> pc/train_18_run.py check   --data <DATA>
  <py> pc/train_18_run.py smoke   --data <DATA>                # 1 epoch thử mỗi variant, KHÔNG thuộc 18 run
  <py> pc/train_18_run.py train   --data <DATA> --seeds 0 1 2  # chạy lại là tiếp tục (từ 23/11: cả 18 run ở máy này)
  <py> pc/train_18_run.py status

<py> là python của .venv-pc (pc/setup_env.py). Kết quả nằm ở pc_runs/runs_aug1/; mỗi run xong được nén thành
pc_runs/upload/aug1_<variant>_fold<k>_seed<s>.zip để tải lên MyDrive/Sym/upload/ rồi nhập bằng cell I trên Colab.
Không evaluate, không đụng test. File nằm ngoài Contrastive_V1/ nên không đổi hash của code đã khoá.
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "Contrastive_V1"
sys.path.insert(0, str(CODE))
from stage2_decoder.contrast_ci import LOCK, config_problems, jobs, run_problems, source_digest  # noqa: E402  (chỉ cần numpy)

OUT = ROOT / "pc_runs"
WEIGHT = CODE / "checkpoints" / "resnet50-19c8e357.pth"
WEIGHT_PREFIX = "19c8e357"
N_IMAGES = 130814
PILOT_INIT = "a6121dc9b5a8f905ef056e6263f54260d3e2b429562da63331db16551a0195f3"   # seed 0, in trong log pilot Colab
PILOT_FULL_EP1 = .749                                                                # val F1 epoch 1 của pilot full fold 2 seed 0
PILOT_BASE_EP1 = .768                                                                # như trên, baseline
MIN_FREE_GB, BUSY_MIN = 15, 15
LIST_PATTERN = re.compile(r"DISFA_(train|test)_(img_path|label|weight)_fold[123]\.txt")
REFERENCE = json.loads((ROOT / "pc" / "data_reference.json").read_text(encoding="utf-8"))          # mã băm 12 list
TARS = json.loads((ROOT / "pc" / "bundle_tar_sha256.json").read_text(encoding="utf-8"))["tars"]     # mã băm 27 tar
WEIGHT_URL = "https://download.pytorch.org/models/resnet50-19c8e357.pth"


def say(*a):
    print(*a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_one(root, name, parent=None):
    """Tìm đúng một file (bỏ qua bản trùng nội dung) — Drive hay chia folder tải về thành nhiều zip lồng nhau."""
    hits = [p for p in Path(root).rglob(name) if p.is_file() and (parent is None or p.parent.name == parent)]
    if not hits:
        raise SystemExit(f"Không thấy {parent + '/' if parent else ''}{name} trong {root}")
    if len({sha256(p) for p in hits}) > 1:
        raise SystemExit(f"Có nhiều bản {name} khác nội dung: {[str(p) for p in hits]}")
    return hits[0]


# ---------------------------------------------------------------- prepare
def prepare(downloads, data, lock=LOCK, weight=WEIGHT, weight_prefix=WEIGHT_PREFIX, n_images=N_IMAGES):
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    meta = json.loads(find_one(downloads, "DONE.json", "DISFA_img").read_text())
    tars = []
    for s in meta["subjects"]:
        tar = find_one(downloads, f"{s}.tar")
        size = json.loads(find_one(downloads, f"{s}.json", "DISFA_img").read_text())["size"]
        if tar.stat().st_size != size:
            raise SystemExit(f"{tar.name} có {tar.stat().st_size} byte, Drive ghi {size}: tải lại file này")
        tars.append(tar)
    marker = data / ".img_ok"
    if not marker.exists():
        for i, tar in enumerate(tars, 1):
            with tarfile.open(tar) as t:
                try:
                    t.extractall(data, filter="data")
                except TypeError:                       # Python cũ chưa có filter; tar do chính ta dựng
                    t.extractall(data)
            say(f"  giải nén {i:>2}/{len(tars)} {tar.name}")
        n = sum(len(f) for _, _, f in os.walk(data / "img"))
        if n != meta["n_images"] or n != n_images:
            raise SystemExit(f"Giải nén được {n} ảnh, cần {n_images}")
        marker.write_text(json.dumps(dict(n_images=n, subjects=meta["subjects"])))
    say(f"Ảnh: {json.loads(marker.read_text())['n_images']} ảnh trong {data / 'img'}")

    lists = [p for p in Path(downloads).rglob("DISFA_*.txt") if LIST_PATTERN.fullmatch(p.name) and p.parent.name == "list"]
    names = {p.name for p in lists}
    if len(names) != 15:
        raise SystemExit(f"Cần đủ 15 file trong DISFA/list, thấy {len(names)}")
    (data / "list").mkdir(exist_ok=True)
    for name in sorted(names):
        shutil.copyfile(find_one(downloads, name, "list"), data / "list" / name)

    (data / "splits_v2").mkdir(exist_ok=True)
    for k in (1, 2, 3):
        src = find_one(downloads, f"fold{k}.json", "splits_v2")
        if sha256(src) != lock["manifests"][k]:
            raise SystemExit(f"splits_v2/fold{k}.json khác bản đã khoá")
        shutil.copyfile(src, data / "splits_v2" / f"fold{k}.json")

    src = find_one(downloads, weight.name)
    if not sha256(src).startswith(weight_prefix):
        raise SystemExit(f"{weight.name} sai nội dung")
    weight.parent.mkdir(exist_ok=True)
    shutil.copyfile(src, weight)
    say(f"OK: dữ liệu ở {data}, trọng số ở {weight}. Bước tiếp: check")


# ---------------------------------------------------------------- from-folder: dữ liệu có sẵn trên máy (chỉ đọc)
class _HashSink:
    """Nhận byte của tar đang ghi và chỉ băm, không ghi ra đĩa."""
    def __init__(self):
        self.h, self.n = hashlib.sha256(), 0

    def write(self, b):
        self.h.update(b)
        self.n += len(b)
        return len(b)

    def tell(self):
        return self.n


def tar_digest(img_root, names):
    """(kích thước, SHA-256) của tar mà cell A sẽ dựng từ các ảnh này — trùng nhau khi và chỉ khi ảnh trùng."""
    sink = _HashSink()
    with tarfile.open(fileobj=sink, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for p in names:
            data = (Path(img_root) / p).read_bytes()
            info = tarfile.TarInfo("img/" + p)
            info.size, info.mtime = len(data), 0
            tar.addfile(info, io.BytesIO(data))
    return sink.n, sink.h.hexdigest()


# Chia fold gốc của ME-GraphAU (Contrastive_V1/tool/DISFA_image_label_process.py), cũng là cách paper dùng.
PARTS = (["SN002", "SN010", "SN001", "SN026", "SN027", "SN032", "SN030", "SN009", "SN016"],
         ["SN013", "SN018", "SN011", "SN028", "SN012", "SN006", "SN031", "SN021", "SN024"],
         ["SN003", "SN029", "SN023", "SN025", "SN008", "SN005", "SN007", "SN017", "SN004"])
LIST_AUS = (1, 2, 4, 6, 9, 12, 25, 26)


def lists_from_labels(labels_root, parts=PARTS):
    """Sinh 12 file list từ ActionUnit_Labels gốc đúng như script gốc: intensity >= 2 là dương;
    fold1 test = part3, fold2 test = part2, fold3 test = part1; train = hai part còn lại theo đúng thứ tự gốc."""
    import numpy as np

    def part(subjects):
        names, arrays = [], []
        for fr in subjects:
            d = Path(labels_root) / fr
            total = len((d / f"{fr}_au1.txt").read_text().splitlines())
            a = np.zeros((total, 8), dtype=int)
            for ai, au in enumerate(LIST_AUS):
                f = d / f"{fr}_au{au}.txt"
                if f.is_file():
                    for t, line in enumerate(f.read_text().splitlines()):
                        a[t, ai] = 1 if int(line.split(",")[1]) >= 2 else 0
            names += [f"{fr}/{i}.png" for i in range(total)]
            arrays.append(a)
        return names, np.concatenate(arrays, axis=0)

    def paths(names):
        return "".join(n + "\n" for n in names).encode()

    def labels(arr):
        buf = io.BytesIO()
        np.savetxt(buf, arr, fmt="%d", delimiter=" ")
        return buf.getvalue()

    p1, p2, p3 = (part(p) for p in parts)
    out = {}
    for k, test, train in ((1, p3, (p1, p2)), (2, p2, (p1, p3)), (3, p1, (p2, p3))):
        out[f"DISFA_test_img_path_fold{k}.txt"] = paths(test[0])
        out[f"DISFA_test_label_fold{k}.txt"] = labels(test[1])
        out[f"DISFA_train_img_path_fold{k}.txt"] = paths(train[0][0] + train[1][0])
        out[f"DISFA_train_label_fold{k}.txt"] = labels(np.concatenate([train[0][1], train[1][1]]))
    return out


def locked_splits(data):
    """Tạo lại splits_v2 đúng tham số đã dùng ngày 20/11 (kết quả trùng từng byte nếu list trùng)."""
    from experiment_data import make_manifest
    return {k: make_manifest(data, k, 3, 2026, check_images=False, min_val_positive=100) for k in (1, 2, 3)}


def download(url, dest):
    import urllib.request
    part = Path(str(dest) + ".part")
    urllib.request.urlretrieve(url, part)
    os.replace(part, dest)


def from_folder(src, data, lists_dir=None, lock=LOCK, ref=REFERENCE, tars=TARS, n_images=N_IMAGES,
                make_splits=locked_splits, weight=WEIGHT, weight_prefix=WEIGHT_PREFIX, fetch=download, parts=PARTS):
    """Dùng ảnh có sẵn trên máy (ví dụ D:\\KLTN_Khang\\DISFA_Data, sắp xếp kiểu gì cũng được): chỉ ĐỌC thư mục đó,
    chép vào <data>, rồi kiểm từng subject bằng SHA-256 tar so với bản chuẩn. Lệch là dừng, không đánh dấu .img_ok."""
    src, data = Path(src), Path(data)
    if not src.is_dir():
        raise SystemExit(f"Không thấy thư mục {src}")
    data.mkdir(parents=True, exist_ok=True)
    if (data / ".img_ok").exists():
        say(f"{data} đã chuẩn bị xong trước đó, bỏ qua")
        return

    say("== 1/4 List: tìm và đối chiếu mã băm")
    look = Path(lists_dir) if lists_dir else src
    (data / "list").mkdir(exist_ok=True)
    absent, differ = [], []
    for name, want in ref["lists"].items():
        cands = [p for p in look.rglob(name) if p.is_file()]
        good = [p for p in cands if sha256(p) == want]
        if good:
            shutil.copyfile(good[0], data / "list" / name)
        else:
            (differ if cands else absent).append(name)
    au1 = [p.parent.parent for p in src.rglob("SN001_au1.txt")] if not lists_dir and len(absent) == len(ref["lists"]) else []
    if au1:                                        # không có list: sinh từ nhãn gốc bằng đúng logic script gốc
        say(f"  không có file list; sinh list từ nhãn gốc {au1[0]} theo fold gốc (part1/2/3)")
        made = lists_from_labels(au1[0], parts)
        absent = []
        for name, want in ref["lists"].items():
            if hashlib.sha256(made[name]).hexdigest() == want:
                (data / "list" / name).write_bytes(made[name])
            else:
                differ.append(name)
        if not differ:
            say("  12/12 list sinh từ nhãn gốc TRÙNG từng byte với list đang dùng trên Colab")
    if absent or differ:
        raise SystemExit(f"List trong {look}: không thấy {len(absent)} file, {len(differ)} file có nhưng KHÁC bản chuẩn.\n"
                         f"  không thấy: {absent}\n  khác: {differ}\n"
                         "Chép thư mục DISFA\\list của máy cá nhân sang máy này: train_data_co_san.bat tự dùng D:\\LVTS_TriLe\\list; "
                         "chạy tay thì thêm --lists <thư mục đó>")

    say("== 2/4 splits_v2: tạo lại từ list và đối chiếu với bản khoá")
    (data / "splits_v2").mkdir(exist_ok=True)
    for k, m in make_splits(data).items():
        f = data / "splits_v2" / f"fold{k}.json"
        f.write_bytes(json.dumps(m, indent=2).encode("utf-8"))      # LF như bản ghi trên Colab
        if sha256(f) != lock["manifests"][k]:
            raise SystemExit(f"splits_v2/fold{k}.json tạo lại KHÁC bản khoá: gửi Claude xem")

    say(f"== 3/4 Ảnh: quét {src} (chỉ đọc), chép vào {data / 'img'}, kiểm từng subject")
    names = sorted({p.strip() for k in (1, 2, 3)
                    for p in (data / "list" / f"DISFA_test_img_path_fold{k}.txt").read_text().split() if p.strip()})
    if len(names) != n_images:
        raise SystemExit(f"List có {len(names)} ảnh, cần {n_images}")
    found = {}
    for p in src.rglob("*.png"):
        if re.fullmatch(r"SN\d{3}", p.parent.name) and p.stem.isdigit():
            found.setdefault(f"{p.parent.name}/{p.name}", p)
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f"Thiếu {len(missing)} ảnh dạng SNxxx\\<số>.png trong {src}, ví dụ {missing[:3]}")
    free = shutil.disk_usage(data).free / 1e9
    if free < 10:
        raise SystemExit(f"Ổ chứa {data} còn {free:.0f} GB, cần khoảng 10 GB để chép ảnh")
    subjects = sorted({n.split("/")[0] for n in names})
    bad = []
    for i, s in enumerate(subjects, 1):
        mine = [n for n in names if n.split("/")[0] == s]
        (data / "img" / s).mkdir(parents=True, exist_ok=True)
        for n in mine:
            dst = data / "img" / n
            if not dst.exists() or dst.stat().st_size != found[n].stat().st_size:
                shutil.copyfile(found[n], dst)
        size, digest = tar_digest(data / "img", mine)
        ok = (size, digest) == (tars[s]["size"], tars[s]["sha256"])
        say(f"  {i:>2}/{len(subjects)} {s}: {'KHỚP' if ok else 'KHÁC'}")
        if not ok:
            bad.append(s)
    if bad:
        raise SystemExit(f"Ảnh KHÁC bản chuẩn ở {bad}: KHÔNG train bằng dữ liệu này (seed sẽ không so được với Colab). "
                         "Dùng gói LVTS_TriLe thay thế.")

    say("== 4/4 Trọng số ImageNet")
    have = [p for p in src.rglob(weight.name) if p.is_file() and sha256(p).startswith(weight_prefix)]
    weight.parent.mkdir(parents=True, exist_ok=True)
    if have:
        shutil.copyfile(have[0], weight)
    elif not (weight.exists() and sha256(weight).startswith(weight_prefix)):
        say(f"  tải {WEIGHT_URL}")
        fetch(WEIGHT_URL, weight)
    if not sha256(weight).startswith(weight_prefix):
        raise SystemExit(f"{weight} sai nội dung")
    (data / ".img_ok").write_text(json.dumps(dict(n_images=len(names), subjects=subjects, source=str(src))))
    say(f"OK: {len(names)} ảnh trùng bản chuẩn, dữ liệu ở {data}. Bước tiếp: check, smoke")


# ---------------------------------------------------------------- check
PROBE = r"""
import sys, json, importlib.metadata as M
import torch
from train_contrastive import source_files
from experiment_data import load_manifest
for k in (1, 2, 3):
    load_manifest(f"{sys.argv[1]}/fold{k}.json", sys.argv[2])
gpu = None
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    x = torch.randn(4, 3, 64, 64, device="cuda")
    y = torch.nn.Conv2d(3, 8, 3).cuda()(x).relu().mean()
    y.backward()
    torch.cuda.synchronize()
    p = torch.cuda.get_device_properties(0)
    gpu = dict(name=p.name, gb=round(p.total_memory / 1e9, 1), capability=list(torch.cuda.get_device_capability(0)))
env = {k: M.version(k) for k in ("torch", "torchvision", "numpy", "timm", "pillow")}
print("PROBE" + json.dumps(dict(src=source_files(), env=env, gpu=gpu)))
"""


def check(data, out=OUT, lock=LOCK, weight=WEIGHT, weight_prefix=WEIGHT_PREFIX, need_gpu=True):
    data = Path(data)
    for k in (1, 2, 3):
        f = data / "splits_v2" / f"fold{k}.json"
        if not f.exists() or sha256(f) != lock["manifests"][k]:
            raise SystemExit(f"{f} thiếu hoặc khác bản khoá: chạy prepare")
    if not (data / ".img_ok").exists():
        raise SystemExit(f"Chưa có ảnh trong {data}: chạy prepare")
    if not weight.exists() or not sha256(weight).startswith(weight_prefix):
        raise SystemExit(f"Thiếu hoặc sai {weight}: chạy prepare")
    r = subprocess.run([sys.executable, "-c", PROBE, str(data / "splits_v2"), str(data)], cwd=CODE,
                       capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1"))
    if r.returncode:
        raise SystemExit("Kiểm code/manifest/GPU lỗi:\n" + r.stderr[-3000:])
    got = json.loads(next(x for x in r.stdout.splitlines() if x.startswith("PROBE"))[5:])
    if source_digest(got["src"]) != lock["source"]:
        raise SystemExit("Code bị hash KHÁC bản đã khoá: kéo đúng commit trên main, không sửa Contrastive_V1/")
    if got["env"] != lock["env"]:
        raise SystemExit(f"Thư viện khác khoá: {got['env']}\nChạy lại: python pc/setup_env.py")
    if need_gpu and not got["gpu"]:
        raise SystemExit("torch không thấy GPU (driver phải chạy được CUDA 13.0)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / 1e9
    if free < MIN_FREE_GB:
        raise SystemExit(f"Ổ chứa {out} còn {free:.0f} GB, cần {MIN_FREE_GB} GB")
    say(f"Khoá OK | GPU {got['gpu']} | ổ trống {free:.0f} GB")
    return got


# ---------------------------------------------------------------- chạy một lần train
def train_cmd(data, v, k, s, out_dir, lock=LOCK, extra=()):
    c = lock["common"]
    return [sys.executable, "-u", "train_contrastive.py", "train", "--variant", v, "--seed", str(s),
            "--arc", c["arc"], "--data-root", str(data), "--manifest", str(Path(data) / "splits_v2" / f"fold{k}.json"),
            "--output", str(out_dir), "--epochs", str(c["epochs"]), "--batch-size", str(c["batch_size"]),
            "--workers", str(c["workers"]), *extra]


def launch(cmd, log):
    """Chạy train_contrastive.py, ghi đủ log, in dòng đáng xem; trả về toàn bộ output."""
    log.parent.mkdir(parents=True, exist_ok=True)
    say(f">> log: {log}")
    lines = []
    out = log.parent.parent                          # <OUT>/logs/<log>: tiến độ tính trên <OUT>
    live, inline, shown = "smoke" not in log.name, False, 0.0

    def newline():
        nonlocal inline
        if inline:
            print(flush=True)
            inline = False
    with open(log, "w", encoding="utf-8") as f, subprocess.Popen(
            cmd, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1, env=dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")) as p:
        try:
            for line in p.stdout:
                f.write(line)
                f.flush()
                lines.append(line)
                step = STEP_RE.match(line)
                if line.startswith('{"epoch"'):
                    r = json.loads(line)
                    newline()
                    say(f"   ep{r['epoch']:>2}  wa={r['train']['wa']:.4f}  valF1={r['val']['macro_f1']:.3f}  "
                        f"{r['seconds'] / 60:.1f} phút" + (f"  | {progress_line(out)}" if live else ""))
                elif step:
                    if live and time.time() - shown > 20:      # dòng tiến độ tại chỗ, làm mới mỗi 20 giây
                        shown = time.time()
                        e, n_e, s, n = map(int, step.groups())
                        text = f"   epoch {e}/{n_e} {100 * s / n:3.0f}% | {progress_line(out)}"
                        print("\r" + text[:110].ljust(110), end="", flush=True)
                        inline = True
                elif "Warning" not in line and "warnings.warn" not in line:
                    newline()
                    say("   " + line.rstrip())
            newline()
        except KeyboardInterrupt:
            p.terminate()
            p.wait()
            raise SystemExit("Đã dừng. Chạy lại cùng lệnh để tiếp tục (run dở sẽ resume).")
    if p.returncode:
        raise SystemExit(f"train dừng với mã {p.returncode}, xem {log}")
    return "".join(lines)


def n_epochs(run):
    try:
        return len(json.loads((run / "history.json").read_text()))
    except (OSError, ValueError):
        return 0


def is_done(run, epochs):
    return n_epochs(run) >= epochs and all((run / n).exists() for n in ("config.json", "best.pt", "last.pt"))


def init_hash(run):
    """Hash khởi tạo train_contrastive ghi vào last.pt lúc tạo run (giữ nguyên qua resume)."""
    import torch
    try:
        return torch.load(run / "last.pt", map_location="cpu", weights_only=False).get("initial_detector_hash")
    except Exception:                                   # noqa: BLE001  (file hỏng cũng là lỗi cặp)
        return None


def pair_problem(runs, k, s, epochs):
    """full − baseline chỉ là so cặp khi hai variant cùng fold/seed khởi tạo trùng nhau. Chưa đủ cặp thì chưa kiểm."""
    rds = [Path(runs) / v / f"fold{k}" / f"seed{s}" for v in ("baseline", "full")]
    if not all(is_done(rd, epochs) for rd in rds):
        return None
    a, b = (init_hash(rd) for rd in rds)
    if a is None or a != b:
        return f"fold{k}/seed{s}: baseline {str(a)[:12]}…, full {str(b)[:12]}…"
    return None


def pack(run, v, k, s, out, prefix="aug1"):
    """Nén một run xong thành zip để tải lên MyDrive/Sym/upload/ (cell I trên Colab giải nén và kiểm)."""
    up = Path(out) / "upload"
    up.mkdir(parents=True, exist_ok=True)
    target = up / f"{prefix}_{v}_fold{k}_seed{s}.zip"
    tmp = target.with_suffix(".zip.partial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(run.rglob("*")):
            if p.is_file() and not p.name.endswith(".tmp"):
                z.write(p, f"{v}/fold{k}/seed{s}/{p.relative_to(run).as_posix()}")
    os.replace(tmp, target)
    return target


def train(data, seeds, folds=(1, 2, 3), out=OUT, lock=LOCK, old_session_dead=False, run=launch):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = out / "runs_aug1"
    epochs = lock["common"]["epochs"]
    for v, k, s in jobs():
        cfg = runs / v / f"fold{k}" / f"seed{s}" / "config.json"
        if cfg.exists():
            bad = config_problems(json.loads(cfg.read_text(encoding="utf-8")), v, k, s, lock)
            if bad:
                raise SystemExit(f"{v}/fold{k}/seed{s} có sẵn nhưng lệch khoá: {bad}")
    todo = [(v, k, s) for v, k, s in jobs() if s in seeds and k in folds
            and not is_done(runs / v / f"fold{k}" / f"seed{s}", epochs)]
    say(f"Máy này: {len(todo)} run cần train ({', '.join(f'{v}/f{k}/s{s}' for v, k, s in todo) or 'không có'})")
    for i, (v, k, s) in enumerate(todo, 1):
        rd, name = runs / v / f"fold{k}" / f"seed{s}", f"{v}_fold{k}_seed{s}"
        free = shutil.disk_usage(out).free / 1e9
        say(f"\n[{i}/{len(todo)}] {name} | ổ trống {free:.0f} GB")
        if free < MIN_FREE_GB:
            raise SystemExit(f"Ổ còn {free:.0f} GB, cần {MIN_FREE_GB} GB: dọn pc_runs/upload sau khi đã tải lên Drive")
        seen = [p for p in (rd / "config.json", rd / "history.json", rd / "last.pt", rd / "best.pt") if p.exists()]
        if seen and not old_session_dead:
            age = (time.time() - max(p.stat().st_mtime for p in seen)) / 60
            if age < BUSY_MIN:
                raise SystemExit(f"{name} vừa được ghi {age:.0f} phút trước: có thể một tiến trình khác đang train. "
                                 f"Đợi {BUSY_MIN} phút hoặc thêm --old-session-dead nếu chắc chắn đã tắt")
        extra = []
        if (rd / "last.pt").exists():
            extra = ["--resume", str(rd / "last.pt")]
        elif (rd / "config.json").exists():
            if (rd / "history.json").exists():
                raise SystemExit(f"{name} có history nhưng không có last.pt: gửi Claude xem, đừng xoá")
            dest = out / "failed_runs" / f"aug1_{name}_{time.strftime('%Y%m%d_%H%M%S')}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(rd), str(dest))
            say(f"  lần trước bị ngắt trong epoch 1 → cất sang {dest}, train lại")
        run(train_cmd(data, v, k, s, rd, lock, extra), out / "logs" / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.log")
        if n_epochs(rd) < epochs and (rd / "last.pt").exists():     # bị ngắt giữa ghi last.pt và history.json
            import torch
            hist = torch.load(rd / "last.pt", map_location="cpu", weights_only=False)["history"]
            (rd / "history.json").write_text(json.dumps(hist, indent=2))
        bad = run_problems(rd, v, k, s, lock)
        if bad or not is_done(rd, epochs):
            raise SystemExit(f"{name} chưa xong hoặc lệch khoá: {bad}")
        bad = pair_problem(runs, k, s, epochs)
        if bad:
            raise SystemExit(f"Hai variant cùng seed khởi tạo KHÁC nhau ({bad}): full − baseline không còn là so cặp. "
                             "Dừng lại, gửi Claude log.")
        say(f"  xong → {pack(rd, v, k, s, out)}  (tải lên MyDrive/Sym/upload/)")
    say(f"\nXONG phần của máy này. Zip chờ tải lên: {sorted(p.name for p in (out / 'upload').glob('*.zip'))}")


# ---------------------------------------------------------------- refit (dòng b)
def refit(data, seeds, folds=(1, 2, 3), out=OUT, lock=LOCK, old_session_dead=False, run=launch):
    """Refit các run (a) đã xong trên máy này: train lại trên đủ 18 subject, dừng ở epoch E (Contrastive_V1/refit)."""
    from refit import run as R
    out = Path(out)
    todo, skipped = R.jobs(out / "runs_aug1", out / "runs_refit1", Path(data) / "splits_v2", data, seeds, folds, lock)
    for x in skipped:
        say(f"  bỏ qua {x}")
    names = [f"{j['variant']}/f{j['fold']}/s{j['seed']} E={j['E']}" for j in todo]
    say(f"Refit: {len(todo)} run ({', '.join(names) or 'không có'})")
    for i, j in enumerate(todo, 1):
        name = f"refit1_{j['variant']}_fold{j['fold']}_seed{j['seed']}"
        say(f"\n[{i}/{len(todo)}] {name} | dừng sau epoch {j['E']}")
        moved = R.clear_stale(j, out / "failed_runs", BUSY_MIN, old_session_dead)
        if moved:
            say(f"  lần trước bị ngắt trong epoch 1 → cất sang {moved}")
        run(R.one_cmd(j, data, lock=lock), out / "logs" / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.log")
        R.finish(j, lock)
        say(f"  xong → {pack(Path(j['out']), j['variant'], j['fold'], j['seed'], out, prefix='refit1')}")
    say(f"\nXONG refit. Zip chờ tải lên: {sorted(p.name for p in (out / 'upload').glob('refit1_*.zip'))}")


# ---------------------------------------------------------------- smoke
def smoke(data, out=OUT, lock=LOCK, keep=False, run=launch):
    """1 epoch baseline + 1 epoch full của fold2/seed0 (đúng cặp pilot trên Colab); thư mục smoke không thuộc 18 run
    và bị xoá sau đó. Từ 23/11 cả 18 run train trên máy này, nên điều kiện chặn là: config đúng khoá, và hai variant
    cùng seed khởi tạo trùng nhau trên máy này (so cặp). Hash khởi tạo khác pilot Colab chỉ được ghi lại: trọng số
    ImageNet đã kiểm cùng file (19c8e357), code và thư viện đã khoá; phần còn lại là head khởi tạo ngẫu nhiên trên CPU,
    lệch bit theo CPU và hệ điều hành (smoke 23/11 trên 5090: a0b67247…, Colab: a6121dc9…)."""
    stamp, res = time.strftime("%Y%m%d_%H%M%S"), {}
    for v in ("baseline", "full"):
        rd = Path(out) / "smoke" / f"{stamp}_{v}"
        t0 = time.time()
        text = run(train_cmd(data, v, 2, 0, rd, lock, ["--stop-after", "1"]), Path(out) / "logs" / f"smoke_{rd.name}.log")
        init = re.search(r'"initial_detector_hash": "([0-9a-f]{64})"', text)
        h = json.loads((rd / "history.json").read_text())
        problems = config_problems(json.loads((rd / "config.json").read_text(encoding="utf-8")), v, 2, 0, lock)
        if not keep:
            shutil.rmtree(rd)
        if problems:
            raise SystemExit(f"Config của smoke {v} lệch khoá: {problems}")
        if not init:
            raise SystemExit(f"Log smoke {v} không có initial_detector_hash: gửi Claude log smoke")
        res[v] = dict(init=init[1], val=h[0]["val"]["macro_f1"], seconds=h[0]["seconds"], minutes=(time.time() - t0) / 60)
    init = res["full"]["init"]
    if res["baseline"]["init"] != init:
        raise SystemExit("baseline và full cùng seed khởi tạo KHÁC nhau trên máy này: full − baseline không còn là so cặp. "
                         "Gửi Claude log smoke.")
    say(f"\nSMOKE OK | baseline và full cùng seed khởi tạo trùng nhau ({init[:12]}…)")
    say(f"  khởi tạo {'TRÙNG' if init == PILOT_INIT else 'khác'} pilot Colab"
        + ("" if init == PILOT_INIT else " — bình thường khi khác CPU; cả 18 run train trên máy này nên vẫn so cặp được"))
    for v, ref in (("baseline", PILOT_BASE_EP1), ("full", PILOT_FULL_EP1)):
        r = res[v]
        say(f"  {v:>8}: val F1 epoch 1 = {r['val']:.3f} (pilot Colab {ref:.3f}; khác GPU nên lệch một chút là bình thường)"
            f" | {r['minutes']:.1f} phút, kể cả khởi động")
        if abs(r["val"] - ref) > .05:
            say(f"  CẢNH BÁO: {v} lệch pilot hơn 0.05 — có thể ảnh/list khác Drive. Gửi Claude log smoke trước khi train.")
    hours = 20 * max(r["seconds"] for r in res.values()) / 3600
    say(f"Ước tính 1 run 20 epoch ≈ {hours:.1f} giờ; 18 run ≈ {18 * hours:.0f} giờ, chưa kể refit")
    return dict(init=init, same_as_colab=init == PILOT_INIT, runs=res)


# ---------------------------------------------------------------- tiến độ (chỉ đọc, chạy song song với train được)
SEC_DEFAULT = {"baseline": 170.0, "full": 190.0}   # giây/epoch đo trên 5090 ngày 23/11 (gồm val), khi chưa có số đo
REFIT_SCALE = 1.2                                    # refit train trên 18 subject thay vì 15: ước lượng khi chưa đo
RUN_OVERHEAD = 60.0                                  # khởi động + nén zip mỗi run, giây (ước lượng)
LIVE_MIN = 15                                        # log được ghi trong 15 phút gần nhất thì coi là đang chạy
STEP_RE = re.compile(r"epoch (\d+)/(\d+) step (\d+)/(\d+)")


def _hist(rd):
    try:
        h = json.loads((Path(rd) / "history.json").read_text(encoding="utf-8"))
        return h if isinstance(h, list) else []
    except (OSError, ValueError):
        return []


def _latest_log(out, name):
    pat = re.compile(r"\d{8}_\d{6}_" + re.escape(name) + r"\.log")
    logs = [p for p in (Path(out) / "logs").glob(f"*{name}.log") if pat.fullmatch(p.name)]
    return max(logs, key=lambda p: p.stat().st_mtime, default=None)


def _in_epoch(log, done):
    """Phần đã chạy của epoch thứ done+1, theo dòng 'epoch e/E step s/n' cuối cùng của log."""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return 0.0
    m = STEP_RE.findall(tail)
    if not m:
        return 0.0
    e, _, s, n = map(int, m[-1])
    return min(s / n, 1.0) if e == done + 1 and n else 0.0


def _mean(xs, default):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else default


def progress(out=OUT, lock=LOCK, now=None):
    """Tiến độ (a) và refit trên máy này, tính theo epoch có trọng số thời gian; thời gian còn lại dựa trên tốc độ đo
    được trên chính máy này. E của refit chỉ biết khi run (a) xong; trước đó lấy trung bình các E đã biết (chưa có thì 5)."""
    from stage2_decoder.check_stage1 import best_point
    out, now = Path(out), time.time() if now is None else now
    n_ep = lock["common"]["epochs"]
    roots = {"a": out / "runs_aug1", "refit": out / "runs_refit1"}

    def rd(ph, v, k, s):
        return roots[ph] / v / f"fold{k}" / f"seed{s}"

    hist = {(ph, v, k, s): _hist(rd(ph, v, k, s)) for ph in roots for v, k, s in jobs()}
    sec = {("a", v): _mean([r.get("seconds") for (ph, vv, _, _), h in hist.items() if ph == "a" and vv == v for r in h],
                           SEC_DEFAULT[v]) for v in ("baseline", "full")}
    for v in ("baseline", "full"):
        sec["refit", v] = _mean([r.get("seconds") for (ph, vv, _, _), h in hist.items() if ph == "refit" and vv == v
                                 for r in h], REFIT_SCALE * sec["a", v])
    known_e = {}
    for v, k, s in jobs():
        h = hist["a", v, k, s]
        if len(h) >= n_ep:
            try:
                known_e[v, k, s] = int(best_point(h)[0])
            except (KeyError, ValueError, TypeError):
                pass
    e_guess = _mean(list(known_e.values()), 5.0)
    items = []
    for ph, prefix in (("a", ""), ("refit", "refit1_")):
        for v, k, s in jobs():
            total = n_ep if ph == "a" else known_e.get((v, k, s), e_guess)
            done = min(len(hist[ph, v, k, s]), total)
            name = f"{prefix}{v}_fold{k}_seed{s}"
            log = _latest_log(out, name) if done < total else None
            live = log is not None and now - log.stat().st_mtime < LIVE_MIN * 60
            got = done + (_in_epoch(log, int(done)) if live else 0.0)
            items.append(dict(phase=ph, variant=v, fold=k, seed=s, total=total, done=got, sec=sec[ph, v],
                              finished=done >= total, live=live, epoch=int(done) + 1,
                              left=(total - got) * sec[ph, v] + (RUN_OVERHEAD if done < total else 0.0)))

    def agg(its):
        full = sum(i["total"] * i["sec"] for i in its)
        return dict(pct=100 * sum(i["done"] * i["sec"] for i in its) / full if full else 100.0,
                    runs_done=sum(i["finished"] for i in its), n=len(its), left=sum(i["left"] for i in its))
    live = [i for i in items if i["live"]]
    return dict(now=now, a=agg([i for i in items if i["phase"] == "a"]), refit=agg([i for i in items if i["phase"] == "refit"]),
                total=agg(items), current=live[0] if live else None, known_e=len(known_e), e_guess=e_guess,
                sec={f"{ph}/{v}": x for (ph, v), x in sec.items()}, epochs=n_ep)


def _clock(t):
    return time.strftime("%H:%M %d/%m", time.localtime(t))


def _bar(pct, width=30):
    n = int(round(width * min(max(pct, 0), 100) / 100))
    return "[" + "#" * n + "." * (width - n) + "]"


def progress_text(p):
    now, a, r, t = p["now"], p["a"], p["refit"], p["total"]
    lines = [f"TIẾN ĐỘ TRAIN TRÊN MÁY NÀY — cập nhật {_clock(now)}",
             f"  (a)    {_bar(a['pct'])} {a['pct']:5.1f}%  {a['runs_done']:>2}/{a['n']} run xong  còn ~{a['left'] / 3600:.1f} giờ",
             f"  refit  {_bar(r['pct'])} {r['pct']:5.1f}%  {r['runs_done']:>2}/{r['n']} run xong  còn ~{r['left'] / 3600:.1f} giờ"
             f"  (E đã biết {p['known_e']}/18, TB {p['e_guess']:.1f} epoch)",
             f"  TỔNG   {_bar(t['pct'])} {t['pct']:5.1f}%  xong (a) ~{_clock(now + a['left'])}, xong cả refit ~{_clock(now + t['left'])}"]
    c = p["current"]
    if c:
        tot = p["epochs"] if c["phase"] == "a" else int(round(c["total"]))
        frac = c["done"] - int(c["done"])
        lines.append(f"  Đang chạy: {'refit ' if c['phase'] == 'refit' else ''}{c['variant']}/fold{c['fold']}/seed{c['seed']}"
                     f" — epoch {c['epoch']}/{tot}, {100 * frac:.0f}% epoch này — run xong ~{_clock(now + c['left'])}")
    elif t["runs_done"] == t["n"]:
        lines.append("  XONG HẾT 18 run (a) và 18 run refit. Zip nằm ở pc_runs\\upload.")
    else:
        lines.append(f"  Không thấy run nào ghi log trong {LIVE_MIN} phút: train có thể đang dừng. "
                     "Chạy lại train_data_co_san.bat để tiếp tục (run dở sẽ resume).")
    s = p["sec"]
    lines.append(f"  Tốc độ đo trên máy này: baseline {s['a/baseline'] / 60:.1f} / full {s['a/full'] / 60:.1f} phút/epoch; "
                 f"refit {s['refit/baseline'] / 60:.1f} / {s['refit/full'] / 60:.1f}")
    return "\n".join(lines)


def progress_line(out):
    """Một dòng ngắn cho màn hình train; lỗi ở đây không bao giờ được làm dừng train."""
    try:
        p = progress(out)
        return f"(a) {p['a']['pct']:.1f}% · tổng {p['total']['pct']:.1f}% · xong dự kiến {_clock(p['now'] + p['total']['left'])}"
    except Exception:                                   # noqa: BLE001
        return ""


def watch(out=OUT, every=60, once=False):
    try:
        while True:
            text = progress_text(progress(out))
            if not once:
                os.system("cls" if os.name == "nt" else "clear")
            say(text)
            if once:
                return text
            say(f"\n(làm mới mỗi {every} giây; đóng cửa sổ này không ảnh hưởng train)")
            time.sleep(every)
    except KeyboardInterrupt:
        return None


def status(out=OUT, lock=LOCK):
    runs = Path(out) / "runs_aug1"
    for v, k, s in jobs():
        rd = runs / v / f"fold{k}" / f"seed{s}"
        zp = Path(out) / "upload" / f"aug1_{v}_fold{k}_seed{s}.zip"
        state = "xong" if is_done(rd, lock["common"]["epochs"]) else (f"{n_epochs(rd)} epoch" if rd.exists() else "-")
        say(f"{v:>8}/fold{k}/seed{s}: {state:>9}" + ("  | zip chờ tải lên" if zp.exists() else ""))
    say("Chỉ là trạng thái máy này; run do Colab train không hiện ở đây.")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):        # thông báo tiếng Việt vẫn đọc được khi ghi ra file/pipe
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--downloads", required=True)
    p.add_argument("--data", required=True)
    f = sub.add_parser("from-folder", help="dùng ảnh có sẵn trên máy (chỉ đọc), kiểm từng subject")
    f.add_argument("--src", required=True)
    f.add_argument("--data", required=True)
    f.add_argument("--lists", help="thư mục chứa 12 file list chuẩn, nếu list trong --src khác bản chuẩn")
    for name in ("check", "smoke", "train", "refit"):
        q = sub.add_parser(name)
        q.add_argument("--data", required=True)
    sub.choices["smoke"].add_argument("--keep", action="store_true")
    for name in ("train", "refit"):
        sub.choices[name].add_argument("--seeds", type=int, nargs="+", required=True, choices=(0, 1, 2),
                                       help="chỉ các seed giao cho máy này; Colab không được chạy trùng")
        sub.choices[name].add_argument("--folds", type=int, nargs="+", default=[1, 2, 3], choices=(1, 2, 3))
        sub.choices[name].add_argument("--old-session-dead", action="store_true")
    sub.add_parser("status")
    w = sub.add_parser("progress", help="%% hoàn thành và giờ xong dự kiến (chỉ đọc, chạy song song với train được)")
    w.add_argument("--watch", type=int, default=0, metavar="GIÂY", help="làm mới sau mỗi GIÂY giây; 0 = in một lần")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(a.downloads, a.data)
    elif a.cmd == "from-folder":
        from_folder(a.src, a.data, a.lists)
    elif a.cmd == "check":
        check(a.data)
    elif a.cmd == "smoke":
        check(a.data)
        smoke(a.data, keep=a.keep)
    elif a.cmd == "train":
        check(a.data)
        train(a.data, a.seeds, a.folds, old_session_dead=a.old_session_dead)
    elif a.cmd == "refit":
        check(a.data)
        refit(a.data, a.seeds, a.folds, old_session_dead=a.old_session_dead)
    elif a.cmd == "progress":
        watch(every=a.watch, once=a.watch <= 0)
    else:
        status()


if __name__ == "__main__":
    main()
