"""Test dòng (b) refit: chạy train_contrastive.train THẬT (mô hình tí hon) cho run (a) rồi refit."""
import importlib.metadata
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

import train_contrastive as TC
from experiment_data import load_manifest, make_manifest, read_partition
from refit import run as R
from stage2_decoder import contrast_ci as CI
from test_contrastive import TinyStage

SUBJECTS = [f"SN{i:03d}" for i in range(1, 7)]


def dataset(tmp):
    root = tmp / "data"
    (root / "list").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for s in SUBJECTS:
        (root / "img" / s).mkdir(parents=True)
        for j in range(4):
            Image.new("RGB", (64, 64), (j * 50, 80, 120)).save(root / "img" / s / f"{j}.png")
    for k in (1, 2, 3):
        test = SUBJECTS[2 * (k - 1): 2 * k]
        for part, subj in (("train", [s for s in SUBJECTS if s not in test]), ("test", test)):
            paths = [f"{s}/{j}.png" for s in subj for j in range(4)]
            (root / "list" / f"DISFA_{part}_img_path_fold{k}.txt").write_text("\n".join(paths))
            lab = (rng.random((len(paths), 8)) < .5).astype(int)
            lab[0], lab[1] = 1, 0
            np.savetxt(root / "list" / f"DISFA_{part}_label_fold{k}.txt", lab, fmt="%d")
    splits = tmp / "splits_v2"
    splits.mkdir()
    for k in (1, 2, 3):
        (splits / f"fold{k}.json").write_text(json.dumps(make_manifest(root, k, val_subject_count=1), indent=2))
    return root, splits


def toy_lock(splits):
    return dict(CI.LOCK, common=dict(CI.LOCK["common"], arc="resnet18", epochs=3, batch_size=4, workers=0),
                env={k: importlib.metadata.version(k) for k in CI.LOCK["env"]},
                manifests={k: R.sha256(splits / f"fold{k}.json") for k in (1, 2, 3)})


def train_cli(argv):
    with patch("train_contrastive.MEFARGStage1", TinyStage), patch.object(sys, "argv", ["train_contrastive.py", *argv]):
        TC.main()


def run_a(tmp, root, splits, lock, v="baseline", k=1, s=0):
    out = tmp / "runs_aug1" / v / f"fold{k}" / f"seed{s}"
    c = lock["common"]
    train_cli(["train", "--variant", v, "--seed", str(s), "--arc", c["arc"], "--data-root", str(root),
               "--manifest", str(splits / f"fold{k}.json"), "--output", str(out), "--epochs", str(c["epochs"]),
               "--batch-size", str(c["batch_size"]), "--workers", str(c["workers"])])
    (out / "best.pt").exists()
    return out


def refit_one(job, root, lock):
    cmd = R.one_cmd(job, root, lock=lock)
    assert cmd[1:5] == ["-u", "-m", "refit.run", "one"]
    with patch("train_contrastive.MEFARGStage1", TinyStage), patch.object(TC, "load_manifest", TC.load_manifest):
        R.main(cmd[4:])


def test_derive_merges_fit_and_val_and_recomputes_weights(tmp_path):
    root, splits = dataset(tmp_path)
    orig = load_manifest(splits / "fold1.json", root)
    _, labels, _ = read_partition(root, 1, "train")
    m = R.derive(orig, labels)
    assert m["fit_indices"] == list(range(len(labels))) and set(m["val_indices"]) <= set(orig["val_indices"])
    assert m["fit_subjects"] == sorted(orig["fit_subjects"] + orig["val_subjects"])
    w = 1 / labels.mean(0)
    assert np.allclose(m["train_weights"], w / w.sum() * 8) and not np.allclose(m["train_weights"], orig["train_weights"])
    assert {k: v for k, v in m.items() if k not in ("fit_indices", "val_indices", "fit_subjects", "train_weights",
                                                     "refit", "refit_note")} == \
           {k: v for k, v in orig.items() if k not in ("fit_indices", "val_indices", "fit_subjects", "train_weights")}


def test_refit_trains_on_all_outer_train_and_stops_at_E(tmp_path):
    root, splits = dataset(tmp_path)
    lock = toy_lock(splits)
    a = run_a(tmp_path, root, splits, lock)
    todo, skipped = R.jobs(tmp_path / "runs_aug1", tmp_path / "runs_refit1", splits, root, lock=lock)
    assert len(todo) == 1 and len(skipped) == 17                         # chỉ (baseline, fold1, seed0) có run (a)
    job = todo[0]
    assert job["E"] == R.selected_epoch(a) == TC.best_point(json.loads((a / "history.json").read_text()))[0]
    job["E"] = 2                                                            # cố định để thử cả resume
    first = dict(job, E=1)
    refit_one(first, root, lock)                                            # bị "ngắt" sau epoch 1
    refit_one(job, root, lock)                                              # chạy lại: resume tới epoch 2
    rec = R.finish(job, lock=lock)
    out = Path(job["out"])
    last = torch.load(out / "last.pt", weights_only=False)
    _, labels, _ = read_partition(root, 1, "train")
    assert last["manifest"]["fit_indices"] == list(range(len(labels))) and last["epoch"] == 1
    assert len(json.loads((out / "history.json").read_text())) == 2 and rec["model"] == "last.pt"
    cfg_a, cfg_b = (json.loads((d / "config.json").read_text()) for d in (a, out))
    assert {k for k in cfg_a if cfg_a[k] != cfg_b[k]} == {"manifest_hash"}           # cùng công thức, khác dữ liệu train
    assert R.jobs(tmp_path / "runs_aug1", tmp_path / "runs_refit1", splits, root, lock=lock)[0] == []


def test_refit_loader_rejects_tampered_manifest(tmp_path):
    root, splits = dataset(tmp_path)
    R.write_manifests(splits, root, tmp_path / "refit")
    f = tmp_path / "refit" / "manifests" / "fold2.json"
    loader = R.refit_loader(load_manifest)
    assert loader(f, root)["refit"] is True
    m = json.loads(f.read_text())
    m["fit_indices"] = m["fit_indices"][:-1]
    f.write_text(json.dumps(m, indent=2))
    with pytest.raises(ValueError, match="không khớp manifest gốc"):
        loader(f, root)
    with pytest.raises(SystemExit, match="không ghi đè"):
        R.write_manifests(splits, root, tmp_path / "refit")
    assert loader(splits / "fold1.json", root)["fold"] == 1                  # manifest thường đi qua bình thường


def test_refit_skips_runs_that_are_not_finished(tmp_path):
    root, splits = dataset(tmp_path)
    lock = toy_lock(splits)
    a = run_a(tmp_path, root, splits, lock)
    hist = json.loads((a / "history.json").read_text())
    (a / "history.json").write_text(json.dumps(hist[:-1]))                    # (a) chưa đủ epoch
    todo, skipped = R.jobs(tmp_path / "runs_aug1", tmp_path / "runs_refit1", splits, root, lock=lock)
    assert todo == [] and any("baseline/fold1/seed0" in s and "chưa đủ" in s for s in skipped)
