"""Train các run của bộ 18 run (cấu hình khoá 22/11) trên máy GPU riêng, Windows hoặc Linux. Hướng dẫn: pc/README.md.

  <py> pc/train_18_run.py prepare --downloads <thư mục giải nén từ Drive> --data <DATA>
  <py> pc/train_18_run.py check   --data <DATA>
  <py> pc/train_18_run.py smoke   --data <DATA>            # 1 epoch thử, KHÔNG thuộc 18 run
  <py> pc/train_18_run.py train   --data <DATA> --seeds 2  # chạy lại là tiếp tục
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
                make_splits=locked_splits, weight=WEIGHT, weight_prefix=WEIGHT_PREFIX, fetch=download):
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
    bad = []
    for name, want in ref["lists"].items():
        good = [p for p in look.rglob(name) if p.is_file() and sha256(p) == want]
        if good:
            shutil.copyfile(good[0], data / "list" / name)
        else:
            bad.append(name)
    if bad:
        raise SystemExit(f"{len(bad)} file list thiếu hoặc khác bản chuẩn trong {look}: {bad}\n"
                         "Chép thư mục DISFA\\list từ máy cá nhân (12 file nhỏ) sang máy này rồi thêm --lists <thư mục đó>")

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
    with open(log, "w", encoding="utf-8") as f, subprocess.Popen(
            cmd, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1, env=dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")) as p:
        try:
            for line in p.stdout:
                f.write(line)
                f.flush()
                lines.append(line)
                if line.startswith('{"epoch"'):
                    r = json.loads(line)
                    say(f"   ep{r['epoch']:>2}  wa={r['train']['wa']:.4f}  valF1={r['val']['macro_f1']:.3f}  "
                        f"{r['seconds'] / 60:.1f} phút")
                elif not re.match(r"epoch \d+/\d+ step \d+/", line) and "Warning" not in line:
                    say("   " + line.rstrip())
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
    """1 epoch của full/fold2/seed0 (đúng run pilot đã có trên Colab) để so: hash khởi tạo phải trùng hệt,
    val F1 epoch 1 phải gần pilot, và đo phút/epoch. Thư mục smoke không thuộc 18 run và bị xoá sau đó."""
    rd = Path(out) / "smoke" / time.strftime("%Y%m%d_%H%M%S")
    t0 = time.time()
    text = run(train_cmd(data, "full", 2, 0, rd, lock, ["--stop-after", "1"]), Path(out) / "logs" / f"smoke_{rd.name}.log")
    minutes = (time.time() - t0) / 60
    init = re.search(r'"initial_detector_hash": "([0-9a-f]{64})"', text)
    h = json.loads((rd / "history.json").read_text())
    val = h[0]["val"]["macro_f1"]
    problems = config_problems(json.loads((rd / "config.json").read_text(encoding="utf-8")), "full", 2, 0, lock)
    if not keep:
        shutil.rmtree(rd)
    if problems:
        raise SystemExit(f"Config của smoke lệch khoá: {problems}")
    if not init or init[1] != PILOT_INIT:
        raise SystemExit("Hash khởi tạo KHÁC pilot trên Colab: trọng số hoặc torch không khớp. Gửi Claude log smoke.")
    say(f"\nSMOKE OK | khởi tạo trùng pilot | val F1 epoch 1 = {val:.3f} (pilot Colab {PILOT_FULL_EP1:.3f}, "
        f"khác GPU nên lệch một chút là bình thường) | {minutes:.1f} phút, kể cả khởi động")
    if abs(val - PILOT_FULL_EP1) > .05:
        say("CẢNH BÁO: lệch pilot hơn 0.05 — có thể ảnh/list khác Drive. Gửi Claude log smoke trước khi train.")
    say(f"Ước tính 1 run 20 epoch ≈ {20 * h[0]['seconds'] / 3600:.1f} giờ")
    return dict(val=val, init=init[1], minutes=minutes)


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
    else:
        status()


if __name__ == "__main__":
    main()
