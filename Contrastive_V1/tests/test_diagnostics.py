"""Chẩn đoán Tầng 1 (diagnostics/): chỉ fit/val, không đọc test, đúng công thức, chạy lại thì bỏ qua phần đã có."""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics import calibration as CAL  # noqa: E402
from diagnostics import grad_norms as GR  # noqa: E402
from diagnostics import identity_probe as IP  # noqa: E402
from diagnostics import run_all as RA  # noqa: E402
from experiment_data import digest, load_manifest, make_manifest  # noqa: E402
from stage2_decoder import contrast_ci as CI  # noqa: E402

LOCK = dict(CI.LOCK, source=CI.source_digest({"toy.py": "0"}))
DIAG = Path(__file__).resolve().parents[1] / "diagnostics"


# ---------------------------------------------------------------- hiệu chỉnh (numpy)
def test_auc_matches_pairwise_count_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = rng.integers(0, 10, 300) / 10                       # nhiều giá trị hoà
    pos, neg = p[y == 1], p[y == 0]
    brute = ((pos[:, None] > neg).sum() + .5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg))
    assert CAL.auc(y, p) == pytest.approx(brute)
    assert CAL.auc([0, 1], [.2, .9]) == 1 and CAL.auc([1, 0], [.2, .9]) == 0
    assert np.isnan(CAL.auc(np.zeros(5), np.ones(5)))


def shifted_val(n=200):
    """SN_A hiệu chỉnh đúng (dương 0.75, âm 0.25); SN_B lệch xuống 0.3 (dương 0.45, âm 0). AU cuối: SN_B chỉ 5 dương."""
    y = np.zeros((2 * n, 8), dtype=np.int8)
    y[:n:2] = 1
    y[n::2] = 1
    y[n:, 7] = 0
    y[n:n + 5, 7] = 1
    p = np.where(y == 1, .75, .25)
    p[n:] = np.clip(p[n:] - .3, 0, 1)
    return p, y, np.array(["SN_A"] * n + ["SN_B"] * n)


def test_calibration_detects_per_subject_shift():
    p, y, s = shifted_val()
    r = CAL.calibration(p, y, s)
    a, b = r["per_subject"]["SN_A"], r["per_subject"]["SN_B"]
    assert a["thr"][0] == pytest.approx(.26) and b["thr"][0] == pytest.approx(.05)    # ngưỡng nhỏ nhất đạt F1 = 1
    assert a["f1_at_05"][0] == 1 and b["f1_at_05"][0] == 0 and b["f1_at_best"][0] == 1
    assert r["thr_sd"][0] == pytest.approx(.105) and r["oracle_gain"][0] == pytest.approx(.5)
    assert b["thr"][7] is None and r["n_subjects_eligible"][7] == 1 and np.isnan(r["thr_sd"][7])
    assert r["summary"]["n_au_with_thr_sd"] == 7
    assert r["auc_pooled"][0] == pytest.approx(CAL.auc(y[:, 0], p[:, 0]))
    same = CAL.calibration(p[:200].repeat(2, 0), y[:200].repeat(2, 0), np.array(["SN_A", "SN_C"] * 200))
    assert same["thr_sd"][0] == 0 and same["oracle_gain"][0] == 0


# ---------------------------------------------------------------- probe danh tính
def test_chrono_split_keeps_order_and_gap():
    subjects = np.repeat(["SN001", "SN002", "SN003"], 100)
    frames = np.concatenate([np.arange(100)[::-1], np.arange(100), np.arange(100)])
    tr, te = IP.chrono_split(subjects, frames)
    for s in ("SN001", "SN002", "SN003"):
        ftr, fte = frames[tr][subjects[tr] == s], frames[te][subjects[te] == s]
        assert len(ftr) == 70 and len(fte) == 25 and ftr.max() + 5 < fte.min()


def test_probe_separable_vs_noise():
    rng = np.random.default_rng(0)
    subjects = np.repeat(["SN001", "SN002", "SN003"], 100)
    frames = np.tile(np.arange(100), 3)
    tr, te = IP.chrono_split(subjects, frames)
    onehot = (subjects[:, None] == np.array(["SN001", "SN002", "SN003"])).astype(float)
    sep = IP.linear_probe(5 * onehot + rng.normal(size=(300, 3)), subjects, tr, te)
    noise = IP.linear_probe(rng.normal(size=(300, 10)), subjects, tr, te)
    assert sep["acc"] == 1 and sep["chance"] == pytest.approx(1 / 3) and sep["n_test"] == 75
    assert noise["acc"] < .6 and noise["nll"] > sep["nll"]
    assert set(sep["per_subject_acc"]) == {"SN001", "SN002", "SN003"}


def synthetic(tmp_path, frames=12):
    """4 subject train (1 làm inner val) + 1 subject test; màu ảnh theo subject; nhãn đổi theo frame."""
    root = tmp_path / "data"
    (root / "list").mkdir(parents=True)
    for part, subjects in (("train", ["SN001", "SN002", "SN003", "SN004"]), ("test", ["SN009"])):
        paths, labels = [], []
        for si, sub in enumerate(subjects):
            (root / "img" / sub).mkdir(parents=True)
            for j in range(frames):
                Image.new("RGB", (256, 256), (40 + 50 * si, 90 + 5 * j, 110)).save(root / "img" / sub / f"{j}.png")
                paths.append(f"{sub}/{j}.png")
                labels.append([(j + a) % 2 for a in range(8)])
        (root / "list" / f"DISFA_{part}_img_path_fold1.txt").write_text("\n".join(paths))
        np.savetxt(root / "list" / f"DISFA_{part}_label_fold1.txt", labels)
    manifest = tmp_path / "fold1.json"
    manifest.write_text(json.dumps(make_manifest(root, 1, val_subject_count=1)))
    return root, manifest


def test_fit_frames_only_fit_subjects_numeric_order_and_stride(tmp_path):
    root, manifest = synthetic(tmp_path)
    m = load_manifest(manifest, root)
    paths, subjects, frames = IP.fit_frames(root, m, stride=2)
    assert set(subjects) == set(m["fit_subjects"]) and len(paths) == 3 * 6
    assert not set(subjects) & (set(m["val_subjects"]) | set(m["test_subjects"]))
    first = [f for f, s in zip(frames, subjects) if s == m["fit_subjects"][0]]
    assert first == [0, 2, 4, 6, 8, 10]                      # thứ tự số, không phải thứ tự chữ ("10" < "2")


def tiny_checkpoint(tmp_path, manifest, variant, epoch=19):
    from model.SymStage1 import MEFARGStage1
    from train_contrastive import VARIANTS
    torch.manual_seed(0)
    model = MEFARGStage1(backbone="resnet18", pretrained=False, use_contrastive=True)
    cfg = dict(variant=variant, seed=0, arc="resnet18", batch_size=4, temperature=.1, gamma=.05, proj_dim=128,
               variant_settings=VARIANTS[variant], manifest_hash=digest(manifest))
    path = tmp_path / f"{variant}_{epoch}.pt"
    torch.save(dict(state_dict=model.state_dict(), epoch=epoch, progress=epoch + 1, config=cfg,
                    initial_detector_hash="x"), path)
    return path


def test_identity_probe_and_grad_norms_on_real_model(tmp_path):
    root, manifest = synthetic(tmp_path)
    full, base = tiny_checkpoint(tmp_path, manifest, "full"), tiny_checkpoint(tmp_path, manifest, "baseline")
    r = IP.run(full, manifest, root, stride=1, batch_size=8, workers=0, device="cpu")
    assert r["split"] == "fit" and r["n_frames"] == 36 and set(r["probes"]) == {"trunk", "pooled", "au_concat"}
    assert all(0 <= p["acc"] <= 1 and p["n_classes"] == 3 for p in r["probes"].values())

    g = GR.measure(full, manifest, root, n_batches=2, batch_size=4, workers=0, device="cpu")
    assert g["n_batches"] == 2 and g["ramp"] == 1 and g["coefficients"]["global"] == pytest.approx(.05)
    assert g["contrastive_norm"]["backbone"] > 0 and np.isfinite(g["ratio_contrastive_to_wa"]["backbone"])
    assert g["grad_norm_raw"]["wa"]["backbone"] > 0
    assert g["grad_norm_raw"]["wa"]["backbone_last"] <= g["grad_norm_raw"]["wa"]["backbone"] + 1e-12
    assert g["grad_norm_weighted"]["we"]["head"] == pytest.approx(.05 * g["grad_norm_raw"]["we"]["head"])

    b = GR.measure(base, manifest, root, n_batches=1, batch_size=4, workers=0, device="cpu")
    assert b["ratio_contrastive_to_wa"]["backbone"] == 0 and np.isnan(b["cos_wa_contrastive"]["backbone"])
    early = GR.measure(tiny_checkpoint(tmp_path, manifest, "full", epoch=0), manifest, root, n_batches=1,
                       batch_size=4, workers=0, device="cpu")
    assert early["ramp"] == pytest.approx(1 / 3) and early["coefficients"]["node"] == pytest.approx(.05 / 3)

    other = tmp_path / "other.json"
    other.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="manifest"):
        IP.run(full, other, root, workers=0, device="cpu")


# ---------------------------------------------------------------- run_all
def make_run(root, v, k, s, best=b"x", epochs=20, last=True):
    run = root / "runs" / v / f"fold{k}" / f"seed{s}"
    (run / "evaluation_val").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(dict(CI.expected_config(v, k, s, LOCK), source_hashes={"toy.py": "0"})))
    (run / "history.json").write_text(json.dumps([dict(epoch=e) for e in range(1, epochs + 1)]))
    (run / "best.pt").write_bytes(best)
    if last:
        (run / "last.pt").write_bytes(b"last")
    p, y, subj = shifted_val(50)
    np.savez_compressed(run / "evaluation_val" / "val_predictions.npz", labels=y, probabilities=p,
                        paths=np.array([f"{q}/{i}.png" for i, q in enumerate(subj)]), subjects=subj)
    macro = CAL.calibration(p, y, subj)["macro_f1_at_05"]
    (run / "evaluation_val" / "val_metrics.json").write_text(json.dumps(dict(checkpoint_hash=digest(run / "best.pt"),
                                                                            macro_f1=macro)))
    (run / "evaluation").mkdir()                            # nếu bị đọc thì hỏng: chẩn đoán không được đụng test
    (run / "evaluation" / "test_metrics.json").write_text("not json: must never be read")
    (run / "evaluation" / "test_predictions.npz").write_bytes(b"garbage")
    return run


def test_run_all_val_only_skips_done_and_redoes_changed(tmp_path, monkeypatch):
    make_run(tmp_path, "baseline", 1, 0)
    full = make_run(tmp_path, "full", 1, 0, best=b"y", last=False)
    make_run(tmp_path, "baseline", 2, 0, epochs=7)           # đang train: chỉ ghi chú
    (tmp_path / "splits").mkdir()
    calls = []

    def fake_probe(ckpt, manifest, data_root, stride, batch_size, workers, device):
        calls.append(("identity", Path(ckpt).name, Path(manifest).name))
        return dict(probes={f: dict(acc=.5 if "baseline" in str(ckpt) else .4, nll=1., chance=.25, nll_chance=1.4)
                            for f in RA.FEATS})

    def fake_grad(ckpt, manifest, data_root, n_batches, workers, device):
        calls.append(("grad", Path(ckpt).name))
        return dict(epoch=20, ratio_contrastive_to_wa=dict(backbone=2.), cos_wa_contrastive=dict(backbone=-.1),
                    grad_norm_raw=dict(wa=dict(backbone=.01)))

    monkeypatch.setattr(RA, "LOCK", LOCK)
    monkeypatch.setattr(IP, "run", fake_probe)
    monkeypatch.setattr(GR, "measure", fake_grad)
    out = tmp_path / "diag"
    argv = ["--runs", str(tmp_path / "runs"), "--splits", str(tmp_path / "splits"), "--data-root", "unused",
            "--output", str(out)]
    with redirect_stdout(io.StringIO()) as buf:
        summ = RA.main(argv)
    assert summ["n_runs"] == 2 and summ["full_minus_baseline"]["n"] == 1
    assert summ["full_minus_baseline"]["mean"]["id_acc_pooled"] == pytest.approx(-.1)
    assert summ["full_minus_baseline"]["mean"]["mean_thr_sd"] == pytest.approx(0)
    assert ("identity", "best.pt", "fold1.json") in calls and calls.count(("grad", "last.pt")) == 1
    g = json.loads((out / "full" / "fold1" / "seed0" / "grad_norms.json").read_text(encoding="utf-8"))
    assert g["last"] is None and "last.pt" in g["note"] and g["val_only"]
    assert "baseline/fold2/seed0" in buf.getvalue() and "chưa xong" in buf.getvalue()
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["val_only"]

    calls.clear()
    with redirect_stdout(io.StringIO()):
        RA.main(argv)
    assert calls == []                                       # đã có, cùng checkpoint: bỏ qua

    (full / "best.pt").write_bytes(b"z")                     # run được thay (ví dụ bản 5090 thay bản Colab)
    vm = full / "evaluation_val" / "val_metrics.json"
    vm.write_text(json.dumps(dict(json.loads(vm.read_text()), checkpoint_hash=digest(full / "best.pt"))))
    with redirect_stdout(io.StringIO()):
        RA.main(argv)
    assert ("identity", "best.pt", "fold1.json") in calls and len([c for c in calls if c[0] == "identity"]) == 1


def test_run_all_refuses_best_that_was_not_evaluated(tmp_path, monkeypatch):
    run = make_run(tmp_path, "full", 1, 0)
    (run / "best.pt").write_bytes(b"changed after evaluate")
    (tmp_path / "splits").mkdir()
    monkeypatch.setattr(RA, "LOCK", LOCK)
    monkeypatch.setattr(IP, "run", lambda *a, **k: pytest.fail("không được chạy probe trên best.pt chưa evaluate"))
    with redirect_stdout(io.StringIO()) as buf, pytest.raises(SystemExit):
        RA.main(["--runs", str(tmp_path / "runs"), "--splits", str(tmp_path / "splits"), "--data-root", "x",
                 "--output", str(tmp_path / "diag"), "--checks", "identity"])
    assert "best.pt khác checkpoint" in buf.getvalue()


def test_diagnostics_never_name_the_test_split():
    for f in DIAG.glob("*.py"):
        src = f.read_text(encoding="utf-8")
        assert '"test"' not in src and "'test'" not in src and "test_predictions" not in src, f.name
