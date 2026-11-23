"""Test pc/train_18_run.py, pc/setup_env.py và cell I (nhập zip vào Drive) — không train, không cần GPU."""
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import shutil
import subprocess
import tarfile
import time
import types
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import train_contrastive as TC
from stage2_decoder import contrast_ci as CI

REPO = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "pc" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PC, SETUP = load("train_18_run"), load("setup_env")
DS = {}                                     # bộ list + splits nhỏ cho cell I, mỗi tmp_path một bộ
HIST = [dict(epoch=e, val=dict(macro_f1=.75), seconds=60.) for e in range(1, 21)]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def toy_lock(splits=None):
    lock = dict(CI.LOCK, source=CI.source_digest({"toy.py": "0"}))
    if splits:
        lock["manifests"] = {k: sha((splits / f"fold{k}.json").read_bytes()) for k in (1, 2, 3)}
    return lock


def write_run(rd, v, k, s, lock, epochs=20, last=True, best=True, age_min=60, manifest_hash=None):
    rd.mkdir(parents=True, exist_ok=True)
    cfg = dict(CI.expected_config(v, k, s, lock), source_hashes={"toy.py": "0"})
    if manifest_hash:
        cfg["manifest_hash"] = manifest_hash
    (rd / "config.json").write_text(json.dumps(cfg))
    if epochs:
        (rd / "history.json").write_text(json.dumps(HIST[:epochs]))
    if last:
        torch.save(dict(history=HIST), rd / "last.pt")
    if best:
        (rd / "best.pt").write_bytes(b"x")
    (rd / "environment.json").write_text(json.dumps(dict(gpu="NVIDIA GeForce RTX 5090")))
    (rd / "source").mkdir(exist_ok=True)
    (rd / "source" / "toy.py").write_text("0")
    t = time.time() - 60 * age_min
    for p in rd.rglob("*"):
        if p.is_file():
            import os
            os.utime(p, (t, t))


def fake_train(lock, calls, noop_resume=False):
    def run(cmd, log):
        got = {}
        with patch("sys.argv", cmd[2:]), patch.object(TC, "train", lambda a: got.update(vars(a))):
            TC.main()                                  # argparse thật của train_contrastive
        calls.append(got)
        out, k = Path(got["output"]), int(Path(got["manifest"]).stem[-1])
        if got["resume"]:
            assert Path(got["resume"]).exists()
            if noop_resume:
                return ""
        else:
            assert not (out / "config.json").exists(), "train thật sẽ FileExistsError"
        write_run(out, got["variant"], k, got["seed"], lock, age_min=0)
        return '"initial_detector_hash": "' + PC.PILOT_INIT + '"'
    return run


# ---------------------------------------------------------------- môi trường
def test_setup_env_pins_exactly_the_lock():
    assert SETUP.LOCK_ENV == CI.LOCK["env"]
    cmds = " ".join(" ".join(c) for c in SETUP.main(["--dry-run"]))
    for pkg, ver in CI.LOCK["env"].items():
        assert f"{pkg}=={ver}" in cmds
    assert "--index-url https://download.pytorch.org/whl/cu130" in cmds


# ---------------------------------------------------------------- prepare
def drive_download(tmp, subjects=("SN001", "SN002"), per=3):
    """Giống thư mục giải nén từ zip Drive: lồng nhau, có file trùng nội dung."""
    root = tmp / "downloads" / "drive-download-1"
    img_dir = root / "DISFA_img"
    img_dir.mkdir(parents=True)
    for s in subjects:
        tar = img_dir / f"{s}.tar"
        with tarfile.open(tar, "w") as t:
            for j in range(per):
                data = f"{s}-{j}".encode()
                info = tarfile.TarInfo(f"img/{s}/{j}.png")
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
        (img_dir / f"{s}.json").write_text(json.dumps(dict(n=per, size=tar.stat().st_size)))
    (img_dir / "DONE.json").write_text(json.dumps(dict(subjects=list(subjects), n_images=per * len(subjects))))
    shutil.copytree(REPO / "DISFA" / "list", root / "DISFA" / "list")
    for k in (1, 2, 3):
        (root / "splits_v2").mkdir(exist_ok=True)
        (root / "splits_v2" / f"fold{k}.json").write_text(json.dumps(dict(fold=k)))
    (root / "resnet50-19c8e357.pth").write_bytes(b"weights")
    dup = tmp / "downloads" / "drive-download-2" / "splits_v2"         # zip thứ hai có bản trùng
    dup.mkdir(parents=True)
    shutil.copy(root / "splits_v2" / "fold1.json", dup / "fold1.json")
    return tmp / "downloads", root


def run_prepare(tmp, downloads, root):
    weight = tmp / "ckpt" / "resnet50-19c8e357.pth"
    PC.prepare(downloads, tmp / "data", toy_lock(root / "splits_v2"), weight=weight,
               weight_prefix=sha(b"weights")[:8], n_images=6)
    return weight


def test_prepare_extracts_and_verifies(tmp_path):
    downloads, root = drive_download(tmp_path)
    weight = run_prepare(tmp_path, downloads, root)
    data = tmp_path / "data"
    assert (data / "img" / "SN002" / "2.png").read_bytes() == b"SN002-2"
    assert len(list((data / "list").glob("*.txt"))) == 15 and weight.read_bytes() == b"weights"
    assert (data / "splits_v2" / "fold3.json").exists() and json.loads((data / ".img_ok").read_text())["n_images"] == 6
    run_prepare(tmp_path, downloads, root)                              # chạy lại: không giải nén lại


@pytest.mark.parametrize("how,msg", [
    ("tar", "tải lại"), ("split", "khác nội dung"), ("weight", "sai nội dung"), ("list", "15 file")])
def test_prepare_rejects_bad_downloads(tmp_path, how, msg):
    downloads, root = drive_download(tmp_path)
    if how == "tar":
        with open(root / "DISFA_img" / "SN001.tar", "ab") as f:
            f.write(b"0")
    elif how == "split":
        (downloads / "drive-download-2" / "splits_v2" / "fold1.json").write_text("other")
    elif how == "weight":
        (root / "resnet50-19c8e357.pth").write_bytes(b"other")
    else:
        next((root / "DISFA" / "list").glob("*.txt")).unlink()
    with pytest.raises(SystemExit, match=msg):
        weight = tmp_path / "ckpt" / "resnet50-19c8e357.pth"
        PC.prepare(downloads, tmp_path / "data", toy_lock(root / "splits_v2") if how != "split" else toy_lock(),
                   weight=weight, weight_prefix=sha(b"weights")[:8], n_images=6)


def test_check_probe_reads_real_code_and_lists(tmp_path):
    """Probe thật (subprocess): hash code repo phải ra đúng LOCK["source"], 3 manifest phải nạp được với list thật."""
    from experiment_data import make_manifest
    data = tmp_path / "data"
    shutil.copytree(REPO / "DISFA" / "list", data / "list")
    (data / "splits_v2").mkdir()
    for k in (1, 2, 3):
        m = make_manifest(data, k, check_images=False)
        (data / "splits_v2" / f"fold{k}.json").write_text(json.dumps(m))
    (data / ".img_ok").write_text("{}")
    weight = tmp_path / "w.pth"
    weight.write_bytes(b"weights")
    lock = dict(toy_lock(data / "splits_v2"), source=CI.LOCK["source"],
                env={k: importlib.metadata.version(k) for k in CI.LOCK["env"]})
    got = PC.check(data, out=tmp_path / "out", lock=lock, weight=weight, weight_prefix=sha(b"weights")[:8], need_gpu=False)
    assert CI.source_digest(got["src"]) == CI.LOCK["source"]
    with pytest.raises(SystemExit, match="Thư viện khác khoá"):
        PC.check(data, out=tmp_path / "out", lock=dict(lock, env=CI.LOCK["env"] | {"numpy": "0"}), weight=weight,
                 weight_prefix=sha(b"weights")[:8], need_gpu=False)


# ---------------------------------------------------------------- train
def queue(tmp_path, seeds=(2,), prepare_runs=None, noop_resume=False, lock=None, **kw):
    lock, calls, out = lock or toy_lock(), [], tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    if prepare_runs:
        prepare_runs(out / "runs_aug1", lock)
    with redirect_stdout(io.StringIO()):
        PC.train(tmp_path / "data", list(seeds), out=out, lock=lock, run=fake_train(lock, calls, noop_resume), **kw)
    return calls, out


def test_train_seed_two_then_rerun(tmp_path):
    calls, out = queue(tmp_path)
    assert [(c["variant"], Path(c["manifest"]).name, c["seed"]) for c in calls] == [
        (v, f"fold{k}.json", 2) for k in (1, 2, 3) for v in ("baseline", "full")]
    assert all(c["epochs"] == 20 and c["batch_size"] == 32 and c["workers"] == 4 and c["arc"] == "resnet50"
               and c["ema_decay"] == 0 and c["val_per_epoch"] == 1 and c["backbone_lr_mult"] == 1 for c in calls)
    zips = sorted(p.name for p in (out / "upload").glob("*.zip"))
    assert zips == sorted(f"aug1_{v}_fold{k}_seed2.zip" for v in ("baseline", "full") for k in (1, 2, 3))
    names = zipfile.ZipFile(out / "upload" / "aug1_full_fold3_seed2.zip").namelist()
    assert "full/fold3/seed2/last.pt" in names and "full/fold3/seed2/source/toy.py" in names
    again, _ = queue(tmp_path)
    assert again == []


def target(runs):
    return runs / "baseline" / "fold1" / "seed2"


@pytest.mark.parametrize("state,expect_resume,moved", [
    (dict(epochs=0, last=False, best=False), False, True),             # ngắt trong epoch 1
    (dict(epochs=7), True, False),                                     # ngắt giữa chừng
])
def test_train_recovers_interrupted_runs(tmp_path, state, expect_resume, moved):
    calls, out = queue(tmp_path, prepare_runs=lambda r, lock: write_run(target(r), "baseline", 1, 2, lock, **state))
    assert bool(calls[0]["resume"]) == expect_resume and len(calls) == 6
    assert len(list((out / "failed_runs").glob("aug1_baseline_fold1_seed2_*"))) == (1 if moved else 0)


def test_train_repairs_history_after_last_epoch(tmp_path):
    calls, out = queue(tmp_path, noop_resume=True,
                       prepare_runs=lambda r, lock: write_run(target(r), "baseline", 1, 2, lock, epochs=19))
    assert len(json.loads((target(out / "runs_aug1") / "history.json").read_text())) == 20


@pytest.mark.parametrize("prep,msg", [
    (lambda r, lock: write_run(target(r), "baseline", 1, 2, lock, epochs=3, age_min=1), "vừa được ghi"),
    (lambda r, lock: write_run(target(r), "baseline", 1, 2, dict(lock, common=dict(lock["common"], lr=1.))), "lệch khoá"),
    (lambda r, lock: write_run(target(r), "baseline", 1, 2, lock, epochs=3, last=False), "không có last.pt"),
])
def test_train_stops_before_damage(tmp_path, prep, msg):
    with pytest.raises(SystemExit, match=msg):
        queue(tmp_path, prepare_runs=prep)


def test_smoke_checks_init_hash_against_pilot(tmp_path):
    lock, calls = toy_lock(), []
    res = PC.smoke(tmp_path / "data", out=tmp_path / "out", lock=lock, run=fake_train(lock, calls))
    assert res["init"] == PC.PILOT_INIT and calls[0]["stop_after"] == 1 and calls[0]["variant"] == "full"
    assert not list((tmp_path / "out" / "smoke").iterdir())          # thư mục thử đã xoá

    def wrong(cmd, log):
        fake_train(lock, [])(cmd, log)
        return '"initial_detector_hash": "' + "0" * 64 + '"'
    with pytest.raises(SystemExit, match="KHÁC pilot"):
        PC.smoke(tmp_path / "data", out=tmp_path / "out", lock=lock, run=wrong)


# ---------------------------------------------------------------- cell I trên Colab
def cell_i():
    nb = json.loads((REPO / "notebooks" / "Nhap_run_PC.ipynb").read_text(encoding="utf-8"))
    code = [''.join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert "IMAGES   = False" in code[0] and code[1] in (REPO / "COLAB_CELLS.md").read_text(encoding="utf-8")
    return code[1]


def import_zips(tmp_path, lock, prepare_drive=None):
    ds_root, ds_splits = DS.get(tmp_path) or DS.setdefault(tmp_path, small_dataset(tmp_path / "ds"))
    sym = tmp_path / "Sym"
    runs = sym / "runs_aug1"
    runs.mkdir(parents=True, exist_ok=True)
    if prepare_drive:
        prepare_drive(runs)

    def run(cmd, check=True, timeout=None):
        if cmd[0] == "cp" and cmd[1] == "-r":
            shutil.copytree(cmd[2], cmd[3])
        elif cmd[0] == "cp":
            shutil.copyfile(cmd[1], cmd[2])
        else:
            raise AssertionError(cmd)
    code = cell_i().replace("Path('/content/pc_import')", f"Path(r'{tmp_path / 'work'}')")
    ns = dict(SYM=sym, RUNS=runs, SPLITS=ds_splits, DATA=ds_root, CODE=REPO / "Contrastive_V1", json=json, sys=__import__("sys"), time=time,
              shutil=shutil, Path=Path, subprocess=types.SimpleNamespace(run=run),
              drive=types.SimpleNamespace(flush_and_unmount=lambda: None))
    with patch.object(CI, "LOCK", lock), redirect_stdout(io.StringIO()) as buf:
        exec(code, ns)
    return sym, buf.getvalue()


def made_zips(tmp_path, lock):
    _, out = queue(tmp_path, lock=lock)
    up = tmp_path / "Sym" / "upload"
    up.mkdir(parents=True, exist_ok=True)
    for z in (out / "upload").glob("*.zip"):
        shutil.copy(z, up / z.name)
    return up


def test_cell_i_imports_and_never_overwrites(tmp_path):
    lock = toy_lock()
    up = made_zips(tmp_path, lock)
    keep = lambda runs: write_run(runs / "full" / "fold2" / "seed2", "full", 2, 2, lock)              # Drive đã xong
    stale = lambda runs: write_run(runs / "full" / "fold3" / "seed2", "full", 3, 2, lock, epochs=4)   # dở, cũ
    busy = lambda runs: write_run(runs / "baseline" / "fold3" / "seed2", "baseline", 3, 2, lock, epochs=4, age_min=2)
    sym, log = import_zips(tmp_path, lock, lambda r: (keep(r), stale(r), busy(r)))
    runs = sym / "runs_aug1"
    assert sorted(p.name for p in (up / "done").glob("*.zip")) == [
        "aug1_baseline_fold1_seed2.zip", "aug1_baseline_fold2_seed2.zip", "aug1_full_fold1_seed2.zip", "aug1_full_fold3_seed2.zip"]
    assert [p.name for p in (up / "duplicate").glob("*")] == ["aug1_full_fold2_seed2.zip"]
    assert (up / "aug1_baseline_fold3_seed2.zip").exists() and "vừa được ghi" in log         # để lại, chạy lại sau
    assert len(json.loads((runs / "full" / "fold3" / "seed2" / "history.json").read_text())) == 20
    assert len(list((sym / "failed_runs").glob("aug1_full_fold3_seed2_*_truoc_nhap"))) == 1
    assert CI.run_problems(runs / "baseline" / "fold1" / "seed2", "baseline", 1, 2, lock) == []
    assert not list(runs.rglob("*.partial"))


def test_cell_i_rejects_drifted_or_unsafe_zips(tmp_path):
    lock = toy_lock()
    up = made_zips(tmp_path, lock)
    bad = up / "aug1_baseline_fold1_seed2.zip"
    with zipfile.ZipFile(bad, "a") as z:
        z.writestr("baseline/fold1/seed2/../../../evil.txt", "x")
    other = dict(lock, env=dict(lock["env"], torch="2.11.0+cu130"))
    sym, log = import_zips(tmp_path, lock)
    assert (up / "rejected" / bad.name).exists() and not (tmp_path / "evil.txt").exists()
    up2 = made_zips(tmp_path / "second", other)                         # train bằng thư viện khác khoá
    shutil.rmtree(tmp_path / "Sym")
    (tmp_path / "Sym" / "upload").mkdir(parents=True)
    for z in up2.glob("*.zip"):
        shutil.copy(z, tmp_path / "Sym" / "upload" / z.name)
    sym, log = import_zips(tmp_path, lock)
    assert len(list((sym / "upload" / "rejected").glob("*.zip"))) == 6 and "lệch khoá" in log


# ---------------------------------------------------------------- gói chép sang máy GPU riêng
MB = load("make_bundle")


def test_bundle_code_digest_matches_training_code():
    assert MB.code_digest(REPO / "Contrastive_V1") == CI.source_digest(TC.source_files()) == CI.LOCK["source"]


def test_bat_files_are_crlf_ascii_and_use_the_runner():
    for name in ("1_cai_dat.bat", "2_train.bat"):
        b = (REPO / "pc" / "windows" / name).read_bytes()
        assert b.isascii() and b.count(b"\n") == b.count(b"\r\n") > 10, name
    setup = (REPO / "pc" / "windows" / "1_cai_dat.bat").read_text()
    for step in ("setup_env.py", "prepare --downloads", "check --data", "smoke --data", "resnet50-19c8e357.pth"):
        assert step in setup
    assert 'train --data "%DATA%" --seeds %SEEDS%' in (REPO / "pc" / "windows" / "2_train.bat").read_text()


def small_dataset(tmp):
    """3 fold × 2 subject test; ảnh nhỏ; manifest sinh bằng make_manifest thật."""
    from experiment_data import make_manifest
    root, subjects = tmp / "local", [f"SN{i:03d}" for i in range(1, 7)]
    (root / "list").mkdir(parents=True)
    for s in subjects:
        (root / "img" / s).mkdir(parents=True)
        for j in range(3):
            (root / "img" / s / f"{j}.png").write_bytes(f"{s}/{j}".encode() * 10)
    for k in (1, 2, 3):
        test = subjects[2 * (k - 1): 2 * k]
        for part, subj in (("train", [s for s in subjects if s not in test]), ("test", test)):
            paths = [f"{s}/{j}.png" for s in subj for j in range(3)]
            (root / "list" / f"DISFA_{part}_img_path_fold{k}.txt").write_text("\n".join(paths))
            lab = [[(j + i) % 2 for i in range(8)] for j in range(len(paths))]
            (root / "list" / f"DISFA_{part}_label_fold{k}.txt").write_text("\n".join(" ".join(map(str, r)) for r in lab))
            if part == "train":                                   # list thật chỉ có weight cho train
                (root / "list" / f"DISFA_{part}_weight_fold{k}.txt").write_text("1")
    splits = tmp / "splits_v2"
    splits.mkdir()
    for k in (1, 2, 3):
        (splits / f"fold{k}.json").write_text(json.dumps(make_manifest(root, k, val_subject_count=1, check_images=False)))
    return root, splits


def test_bundle_then_prepare_gives_identical_images(tmp_path):
    root, splits = small_dataset(tmp_path)
    lock = dict(CI.LOCK, manifests={k: sha((splits / f"fold{k}.json").read_bytes()) for k in (1, 2, 3)})
    out = tmp_path / "LVTS_TriLe"
    with redirect_stdout(io.StringIO()):
        digests = MB.build(root / "img", root / "list", splits, out, lock=lock, n_images=18)
    assert sorted(digests) == [f"SN{i:03d}" for i in range(1, 7)]
    assert (out / "1_cai_dat.bat").read_bytes() == (REPO / "pc" / "windows" / "1_cai_dat.bat").read_bytes()
    assert MB.code_digest(out / "Thesis_Sym" / "Contrastive_V1") == CI.LOCK["source"]
    record = json.loads((out / "tar_sha256.json").read_text())
    assert record["tars"] == digests and all(sha((out / "sym_download" / "DISFA_img" / f"{s}.tar").read_bytes())
                                             == d["sha256"] for s, d in digests.items())
    (out / "sym_download" / "resnet50-19c8e357.pth").write_bytes(b"weights")      # 1_cai_dat.bat tải file này
    with redirect_stdout(io.StringIO()):
        PC.prepare(out / "sym_download", tmp_path / "DISFA", lock, weight=tmp_path / "ckpt" / "resnet50-19c8e357.pth",
                   weight_prefix=sha(b"weights")[:8], n_images=18)
    for p in (root / "img").rglob("*.png"):
        assert (tmp_path / "DISFA" / "img" / p.relative_to(root / "img")).read_bytes() == p.read_bytes()
    with pytest.raises(SystemExit, match="đã có nội dung"):
        MB.build(root / "img", root / "list", splits, out, lock=lock, n_images=18)


def test_bundle_tar_layout_is_cell_a(tmp_path):
    """Cùng ảnh → cùng byte với tar mà cell A dựng trên Drive, nên so SHA-256 là so được ảnh."""
    root, _ = small_dataset(tmp_path)
    cell_a = tmp_path / "a.tar"
    with tarfile.open(cell_a, "w", format=tarfile.GNU_FORMAT) as tar:       # đúng mấy dòng của cell A
        for p in ["SN001/0.png", "SN001/1.png", "SN001/2.png"]:
            data = (root / "img" / p).read_bytes()
            info = tarfile.TarInfo("img/" + p)
            info.size, info.mtime = len(data), 0
            tar.addfile(info, io.BytesIO(data))
    MB.write_tar(tmp_path / "b.tar", root / "img", ["SN001/0.png", "SN001/1.png", "SN001/2.png"])
    assert cell_a.read_bytes() == (tmp_path / "b.tar").read_bytes()


@pytest.mark.parametrize("how,msg", [("split", "khác bản đã khoá"), ("list", "khác bản manifest"), ("img", "Thiếu 1 ảnh")])
def test_bundle_refuses_inconsistent_inputs(tmp_path, how, msg):
    root, splits = small_dataset(tmp_path)
    lock = dict(CI.LOCK, manifests={k: sha((splits / f"fold{k}.json").read_bytes()) for k in (1, 2, 3)})
    if how == "split":
        lock["manifests"][2] = "0" * 64
    elif how == "list":
        (root / "list" / "DISFA_train_label_fold1.txt").write_text("0")
    else:
        (root / "img" / "SN003" / "1.png").unlink()
    with pytest.raises(SystemExit, match=msg):
        MB.build(root / "img", root / "list", splits, tmp_path / "out", lock=lock, n_images=18)


def test_cell_j_checks_exactly_the_recorded_bundle():
    rec = json.loads((REPO / "pc" / "bundle_tar_sha256.json").read_text(encoding="utf-8"))
    nb = json.loads((REPO / "notebooks" / "Nhap_run_PC.ipynb").read_text(encoding="utf-8"))
    cell = "".join(nb["cells"][-1]["source"])
    assert cell.startswith("# ==== Cell J") and cell in (REPO / "COLAB_CELLS.md").read_text(encoding="utf-8")
    ns = {}
    exec("EXPECT = " + cell.split("EXPECT = ", 1)[1].split("\n}\n", 1)[0] + "\n}", ns)
    assert ns["EXPECT"] == {s: (d["size"], d["sha256"]) for s, d in rec["tars"].items()} and len(rec["tars"]) == 27


# ---------------------------------------------------------------- dữ liệu có sẵn trên máy GPU (thư mục của Khang)
def khang_layout(tmp, root):
    """Thư mục 'chia fold sẵn': ảnh lặp lại trong từng fold, list để ở chỗ khác, có trọng số."""
    src = tmp / "KLTN_Khang" / "DISFA_Data"
    for k in (1, 2, 3):
        for part in ("train", "test"):
            for name in (root / "list" / f"DISFA_{part}_img_path_fold{k}.txt").read_text().split():
                dst = src / f"fold{k}" / part / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / "img" / name, dst)
    shutil.copytree(root / "list", src / "labels" / "list_ok")
    (src / "weights").mkdir()
    (src / "weights" / "resnet50-19c8e357.pth").write_bytes(b"weights")
    return src


def folder_args(tmp, root):
    from experiment_data import make_manifest
    splits = lambda data: {k: make_manifest(data, k, val_subject_count=1, check_images=False) for k in (1, 2, 3)}
    lock = dict(CI.LOCK, manifests={k: sha(json.dumps(m, indent=2).encode()) for k, m in splits(root).items()})
    names = [f"DISFA_{p}_{t}_fold{k}.txt" for p in ("train", "test") for t in ("img_path", "label") for k in (1, 2, 3)]
    ref = dict(lists={n: sha((root / "list" / n).read_bytes()) for n in names})
    subj = sorted(p.name for p in (root / "img").iterdir())
    tars = {}
    for s in subj:
        size, digest = PC.tar_digest(root / "img", sorted(f"{s}/{p.name}" for p in (root / "img" / s).iterdir()))
        tars[s] = dict(size=size, sha256=digest)
    return dict(lock=lock, ref=ref, tars=tars, n_images=18, make_splits=splits,
                weight=tmp / "ckpt" / "resnet50-19c8e357.pth", weight_prefix=sha(b"weights")[:8])


def test_reference_files_shipped_with_repo():
    assert len(PC.REFERENCE["lists"]) == 12 and PC.REFERENCE["n_images"] == 130814 and len(PC.TARS) == 27


def test_tar_digest_equals_the_bundle_tar(tmp_path):
    root, _ = small_dataset(tmp_path)
    names = ["SN002/0.png", "SN002/1.png", "SN002/2.png"]
    MB.write_tar(tmp_path / "x.tar", root / "img", names)
    assert PC.tar_digest(root / "img", names) == ((tmp_path / "x.tar").stat().st_size, sha((tmp_path / "x.tar").read_bytes()))


def test_from_folder_uses_fold_split_data_read_only(tmp_path):
    root, _ = small_dataset(tmp_path)
    src = khang_layout(tmp_path, root)
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in src.rglob("*") if p.is_file()}
    kw = folder_args(tmp_path, root)
    with redirect_stdout(io.StringIO()) as out:
        PC.from_folder(src, tmp_path / "DISFA", **kw)
    data = tmp_path / "DISFA"
    for p in (root / "img").rglob("*.png"):
        assert (data / "img" / p.relative_to(root / "img")).read_bytes() == p.read_bytes()
    assert out.getvalue().count("KHỚP") == 6 and (data / ".img_ok").exists()
    assert all(sha((data / "splits_v2" / f"fold{k}.json").read_bytes()) == kw["lock"]["manifests"][k] for k in (1, 2, 3))
    assert b"\r" not in (data / "splits_v2" / "fold1.json").read_bytes()
    assert kw["weight"].read_bytes() == b"weights"
    assert {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in src.rglob("*") if p.is_file()} == before   # chỉ đọc
    with redirect_stdout(io.StringIO()) as again:
        PC.from_folder(src, data, **kw)
    assert "bỏ qua" in again.getvalue()


def test_from_folder_stops_on_different_images(tmp_path):
    root, _ = small_dataset(tmp_path)
    src = khang_layout(tmp_path, root)
    for p in src.rglob("SN004/1.png"):
        p.write_bytes(b"crop khac")
    with pytest.raises(SystemExit, match=r"KHÁC bản chuẩn ở \['SN004'\]"), redirect_stdout(io.StringIO()):
        PC.from_folder(src, tmp_path / "DISFA", **folder_args(tmp_path, root))
    assert not (tmp_path / "DISFA" / ".img_ok").exists()


def test_from_folder_lists_fallback_and_weight_download(tmp_path):
    root, _ = small_dataset(tmp_path)
    src = khang_layout(tmp_path, root)
    shutil.rmtree(src / "labels")
    (src / "weights" / "resnet50-19c8e357.pth").unlink()
    for k in (1, 2, 3):                                    # list "chia fold" riêng của Khang, khác bản chuẩn
        (src / f"fold{k}" / f"DISFA_train_label_fold{k}.txt").write_text("0 0 0 0 0 0 0 0")
    kw = folder_args(tmp_path, root)
    with pytest.raises(SystemExit, match="--lists"), redirect_stdout(io.StringIO()):
        PC.from_folder(src, tmp_path / "DISFA", **kw)
    got = []
    kw["fetch"] = lambda url, dest: (got.append(url), Path(dest).write_bytes(b"weights"))
    with redirect_stdout(io.StringIO()):
        PC.from_folder(src, tmp_path / "DISFA", lists_dir=root / "list", **kw)
    assert got == [PC.WEIGHT_URL] and (tmp_path / "DISFA" / ".img_ok").exists()


def test_one_file_runner_bat():
    b = (REPO / "pc" / "windows" / "train_data_co_san.bat").read_bytes()
    assert b.isascii() and b.count(b"\n") == b.count(b"\r\n") > 30
    t = b.decode()
    for step in (r'set "SRC=D:\KLTN_Khang\DISFA_Data"', "setup_env.py", "from-folder --src", "check --data",
                 "smoke --data", "train --data", r'"%REPO%\..")'):
        assert step in t, step


# ---------------------------------------------------------------- refit (dòng b) trên máy riêng → cell I
def refit_setup(tmp_path):
    """Máy riêng: list + splits nhỏ trong <data>, run (a) seed 2 đã xong, lock khớp splits nhỏ."""
    root, splits = small_dataset(tmp_path / "pcdata")
    data = tmp_path / "pcdata" / "local"
    shutil.copytree(splits, data / "splits_v2")
    lock = toy_lock(data / "splits_v2")
    out = tmp_path / "out"
    with redirect_stdout(io.StringIO()):
        PC.train(data, [2], out=out, lock=lock, run=fake_train(lock, []))
    return data, lock, out


def fake_refit(lock, calls):
    def run(cmd, log):
        assert cmd[1:5] == ["-u", "-m", "refit.run", "one"]
        got = {}
        with patch("sys.argv", ["x", "train", *cmd[5:]]), patch.object(TC, "train", lambda a: got.update(vars(a))):
            TC.main()
        calls.append(got)
        k = int(Path(got["manifest"]).stem[-1])
        write_run(Path(got["output"]), got["variant"], k, got["seed"], lock, epochs=got["stop_after"], age_min=0,
                  manifest_hash=sha(Path(got["manifest"]).read_bytes()))
        return ""
    return run


def test_pc_refit_runs_after_train_and_packs(tmp_path):
    data, lock, out = refit_setup(tmp_path)
    calls = []
    with redirect_stdout(io.StringIO()):
        PC.refit(data, [2], out=out, lock=lock, run=fake_refit(lock, calls))
    assert len(calls) == 6 and all(c["stop_after"] == 1 and c["epochs"] == 20 for c in calls)   # E = epoch (a) chọn
    assert all(Path(c["manifest"]).parent.name == "manifests" for c in calls)
    zips = sorted(p.name for p in (out / "upload").glob("refit1_*.zip"))
    assert len(zips) == 6
    assert "full/fold2/seed2/refit.json" in zipfile.ZipFile(out / "upload" / "refit1_full_fold2_seed2.zip").namelist()
    with redirect_stdout(io.StringIO()):
        PC.refit(data, [2], out=out, lock=lock, run=fake_refit(lock, calls))                   # chạy lại: không làm lại
    assert len(calls) == 6


def test_cell_i_imports_refit_only_with_its_run_a(tmp_path):
    data, lock, out = refit_setup(tmp_path)
    with redirect_stdout(io.StringIO()):
        PC.refit(data, [2], out=out, lock=lock, run=fake_refit(lock, []))
    up = tmp_path / "Sym" / "upload"
    up.mkdir(parents=True)
    for z in (out / "upload").glob("refit1_*_fold1_*.zip"):                 # chỉ refit fold 1, thiếu run (a) trên Drive
        shutil.copy(z, up / z.name)
    DS[tmp_path] = (data, data / "splits_v2")
    sym, log = import_zips(tmp_path, lock)
    assert len(list((up / "rejected").glob("refit1_*.zip"))) == 2 and "thiếu run (a)" in log
    for z in (out / "upload").glob("*_fold2_*.zip"):                        # đủ (a) + refit của fold 2
        shutil.copy(z, up / z.name)
    sym, log = import_zips(tmp_path, lock)
    done = sorted(p.name for p in (up / "done").glob("*.zip"))
    assert done == ["aug1_baseline_fold2_seed2.zip", "aug1_full_fold2_seed2.zip",
                    "refit1_baseline_fold2_seed2.zip", "refit1_full_fold2_seed2.zip"]
    rec = json.loads((sym / "runs_refit1" / "full" / "fold2" / "seed2" / "refit.json").read_text())
    assert rec["E"] == 1 and rec["model"] == "last.pt"
    assert "runs_refit1: 2/18" in log and "runs_aug1: 2/18" in log
