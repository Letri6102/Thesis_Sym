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
HIST = [dict(epoch=e, val=dict(macro_f1=.75), seconds=60.) for e in range(1, 21)]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def toy_lock(splits=None):
    lock = dict(CI.LOCK, source=CI.source_digest({"toy.py": "0"}))
    if splits:
        lock["manifests"] = {k: sha((splits / f"fold{k}.json").read_bytes()) for k in (1, 2, 3)}
    return lock


def write_run(rd, v, k, s, lock, epochs=20, last=True, best=True, age_min=60):
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps(dict(CI.expected_config(v, k, s, lock), source_hashes={"toy.py": "0"})))
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
    ns = dict(SYM=sym, RUNS=runs, CODE=REPO / "Contrastive_V1", json=json, sys=__import__("sys"), time=time,
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
