"""Test CI full − baseline (stage2_decoder/contrast_ci.py) bằng dữ liệu tổng hợp."""
import importlib.metadata
import json
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import train_contrastive as TC
from stage2_decoder import contrast_ci as CI
from test_contrastive import TinyStage, args_for, synthetic_data

REPO = Path(__file__).resolve().parents[2]
N_FRAMES, N_SUBJECTS, BEST_EPOCH = 30, 4, 2


def toy_lock():
    return dict(CI.LOCK, common=dict(CI.LOCK["common"], epochs=3), test_subjects=N_SUBJECTS,
                source=CI.source_digest({"toy.py": "0"}))


def make_runs(tmp_path, lock, signal=(0., 0.), evaluated=True):
    """18 run giả đúng bố cục train_contrastive; signal = (baseline, full) là độ tách thêm của mỗi variant."""
    runs, frames = tmp_path / "runs", {}
    for v, k, s in CI.jobs():
        run = CI.run_dir(runs, v, k, s)
        run.mkdir(parents=True)
        cfg = dict(CI.expected_config(v, k, s, lock), source_hashes={"toy.py": "0"})
        (run / "config.json").write_text(json.dumps(cfg))
        (run / "history.json").write_text(json.dumps(
            [dict(epoch=e, val=dict(macro_f1=f)) for e, f in ((1, .5), (BEST_EPOCH, .7), (3, .6))]))
        (run / "best.pt").write_bytes(b"x")
        if not evaluated:
            continue
        subj = np.repeat([f"SN{10 * k + i:03d}" for i in range(N_SUBJECTS)], N_FRAMES)
        y = (np.random.default_rng(k).random((len(subj), 8)) < .3).astype(np.float32)
        # cùng fold/seed thì hai variant chung nhiễu: khác nhau đúng bằng signal
        noise = np.random.default_rng([k, s]).normal(0, .25, y.shape)
        extra = signal[v == CI.CANDIDATE]
        p = np.clip(.3 + .3 * y + extra * (y - .5) + noise, 0, 1).astype(np.float32)
        p[0, 0] = .5                                     # đúng 0.5 phải tính là dương, như metrics()
        ev = run / "evaluation"
        ev.mkdir()
        np.savez_compressed(ev / "test_predictions.npz", labels=y, probabilities=p, subjects=subj,
                            paths=np.array([f"{u}/{j}.png" for j, u in enumerate(subj)]))
        (ev / "test_metrics.json").write_text(json.dumps(dict(
            TC.metrics(y, p), variant=v, fold=k, seed=s, split="test", stage=1, config=cfg,
            selected_epoch=BEST_EPOCH, selected_progress=BEST_EPOCH, initial_detector_hash=f"init{s}",
            checkpoint_hash=f"{v}{k}{s}")))
        frames[v, k, s] = (y, p, subj)
    return runs, frames


def edit_json(path, fn):
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


def test_bootstrap_equals_duplicating_whole_subjects(tmp_path):
    """Tính lại độc lập bằng cách lặp nguyên frame của subject được rút, rồi gọi metrics() của Tầng 1."""
    runs, frames = make_runs(tmp_path, toy_lock(), signal=(0., .1))
    data = {j: CI.load_run(CI.run_dir(runs, *j), *j) for j in CI.jobs()}
    n_boot = 40
    rep = CI.compare(data, n_boot=n_boot)
    for s, got in zip(CI.SEEDS, rep["per_seed"]):
        rng, boot, obs = np.random.default_rng([CI.BOOT_SEED, s]), np.zeros(n_boot), 0.
        for k in CI.FOLDS:
            (y, pb, subj), (_, pf, _) = frames[CI.BASELINE, k, s], frames[CI.CANDIDATE, k, s]
            names = np.unique(subj)
            obs += (TC.metrics(y, pf)["macro_f1"] - TC.metrics(y, pb)["macro_f1"]) * 100 / 3
            for i in range(n_boot):
                idx = np.concatenate([np.flatnonzero(subj == names[j]) for j in rng.integers(0, len(names), len(names))])
                boot[i] += (TC.metrics(y[idx], pf[idx])["macro_f1"] - TC.metrics(y[idx], pb[idx])["macro_f1"]) * 100 / 3
        assert got["delta_pp"] == pytest.approx(obs, abs=1e-9)
        assert got["ci_lo"] == pytest.approx(np.percentile(boot, 2.5), abs=1e-9)
        assert got["ci_hi"] == pytest.approx(np.percentile(boot, 97.5), abs=1e-9)
        assert np.mean(got["delta_by_au"]) == pytest.approx(np.mean(got["delta_by_fold"]), abs=1e-9)


@pytest.mark.parametrize("signal,expected", [((0., 0.), "KHÔNG ĐẠT"), ((0., .4), "ĐẠT"), ((.4, 0.), "KHÔNG ĐẠT")])
def test_end_to_end_verdict(tmp_path, signal, expected):
    lock = toy_lock()
    runs, _ = make_runs(tmp_path, lock, signal)
    out = tmp_path / "out" / "summary.json"
    rep = CI.main(["--runs", str(runs), "--output", str(out)], lock=lock)
    assert rep["verdict"] == expected and json.loads(out.read_text(encoding="utf-8")) == rep
    assert len(rep["per_seed"]) == 3 and len(rep["checkpoints"]) == 18
    if signal[0] == signal[1]:                           # dự đoán giống hệt: ghép cặp phải triệt tiêu hoàn toàn
        assert all(r["delta_pp"] == r["ci_lo"] == r["ci_hi"] == 0 for r in rep["per_seed"])


def test_output_is_reproducible(tmp_path):
    lock = toy_lock()
    runs, _ = make_runs(tmp_path, lock, (0., .1))
    a = CI.main(["--runs", str(runs), "--output", str(tmp_path / "a.json")], lock=lock)
    b = CI.main(["--runs", str(runs), "--output", str(tmp_path / "b.json")], lock=lock)
    assert a == b


BREAK = {
    "thiếu run": lambda r: (r / "history.json").unlink(),
    "chưa đủ epoch": lambda r: edit_json(r / "history.json", lambda h: h.pop()),
    "thiếu best.pt": lambda r: (r / "best.pt").unlink(),
    "bật EMA": lambda r: edit_json(r / "config.json", lambda c: c.update(ema_decay=.999)),
    "thêm tham số lạ": lambda r: edit_json(r / "config.json", lambda c: c.update(extra=None)),
    "khác code": lambda r: edit_json(r / "config.json", lambda c: c.update(source_hashes={"toy.py": "1"})),
    "khác manifest": lambda r: edit_json(r / "config.json", lambda c: c.update(manifest_hash="x")),
    "khác thư viện": lambda r: edit_json(r / "config.json", lambda c: c.update(environment_versions={})),
    "thiếu dự đoán test": lambda r: (r / "evaluation" / "test_predictions.npz").unlink(),
}


@pytest.mark.parametrize("how", list(BREAK))
def test_refuses_before_reading_any_test_file(tmp_path, how):
    lock = toy_lock()
    runs, _ = make_runs(tmp_path, lock)
    BREAK[how](CI.run_dir(runs, "full", 3, 2))           # hỏng run CUỐI: không được đọc test của 17 run trước
    with patch.object(CI, "load_run", side_effect=AssertionError("đã đọc test")), pytest.raises(SystemExit):
        CI.main(["--runs", str(runs), "--output", str(tmp_path / "o.json")], lock=lock)
    assert not (tmp_path / "o.json").exists()


def test_check_only_never_opens_evaluation(tmp_path, capsys):
    lock = toy_lock()
    runs, _ = make_runs(tmp_path, lock)
    real = Path.read_text

    def guarded(self, *a, **k):
        assert "evaluation" not in self.parts, f"đã đọc {self}"
        return real(self, *a, **k)

    with patch.object(Path, "read_text", guarded), patch.object(CI, "load_predictions", side_effect=AssertionError):
        assert CI.main(["--runs", str(runs), "--check-only"], lock=lock) is None
    assert "OK" in capsys.readouterr().out
    bare, _ = make_runs(tmp_path / "bare", lock, evaluated=False)
    CI.main(["--runs", str(bare), "--check-only"], lock=lock)      # chưa evaluate test vẫn kiểm được


def test_never_overwrites(tmp_path):
    lock = toy_lock()
    runs, _ = make_runs(tmp_path, lock)
    out = tmp_path / "o.json"
    out.write_text("old")
    with patch.object(CI, "check_runs", side_effect=AssertionError("phải dừng trước")), pytest.raises(SystemExit):
        CI.main(["--runs", str(runs), "--output", str(out)], lock=lock)
    assert out.read_text() == "old"


def rewrite_npz(run, fn):
    path = run / "evaluation" / "test_predictions.npz"
    d = dict(np.load(path))
    fn(d)
    np.savez_compressed(path, **d)


CORRUPT = {
    "khác checkpoint": (lambda r: edit_json(r / "evaluation/test_metrics.json", lambda m: m.update(selected_progress=3)),
                        "checkpoint tốt nhất"),
    "khác config": (lambda r: edit_json(r / "evaluation/test_metrics.json", lambda m: m["config"].update(lr=1.)),
                    "config khác"),
    "nhầm run": (lambda r: edit_json(r / "evaluation/test_metrics.json", lambda m: m.update(seed=0)),
                 "không phải của run này"),
    "metric lệch": (lambda r: edit_json(r / "evaluation/test_metrics.json", lambda m: m.update(macro_f1=.123456)),
                    "không khớp"),
    "khác khởi tạo": (lambda r: edit_json(r / "evaluation/test_metrics.json", lambda m: m.update(initial_detector_hash="x")),
                      "khởi tạo"),
    "khác thứ tự frame": (lambda r: rewrite_npz(r, lambda d: d.update(paths=d["paths"][::-1])), "ghép cặp"),
}


@pytest.mark.parametrize("how", list(CORRUPT))
def test_rejects_unpaired_or_stale_without_printing_numbers(tmp_path, capsys, how):
    lock = toy_lock()
    runs, _ = make_runs(tmp_path, lock)
    fn, message = CORRUPT[how]
    fn(CI.run_dir(runs, "full", 2, 1))
    with pytest.raises(ValueError, match=message) as e:
        CI.main(["--runs", str(runs), "--output", str(tmp_path / "o.json")], lock=lock)
    assert not re.search(r"\d\.\d\d", str(e.value) + capsys.readouterr().out)
    assert not (tmp_path / "o.json").exists()


def test_rejects_subject_in_two_test_folds(tmp_path):
    runs, _ = make_runs(tmp_path, toy_lock())
    data = {j: CI.load_run(CI.run_dir(runs, *j), *j) for j in CI.jobs()}
    for v, k, s in CI.jobs():
        if k == 2:
            data[v, k, s]["subjects"] = data[CI.BASELINE, 1, 0]["subjects"]
    with pytest.raises(ValueError, match="hơn một fold"):
        CI.check_pairing(data, N_SUBJECTS)


def test_reads_real_train_and_evaluate_output(tmp_path):
    """Bố cục config/metrics/npz lấy từ train() và evaluate() thật, không phải file giả."""
    root, manifest = synthetic_data(tmp_path)
    loaded = {}
    lock = dict(CI.LOCK, common=dict(CI.LOCK["common"], arc="resnet18", epochs=2, batch_size=4, workers=0, pretrained=False),
                env={k: importlib.metadata.version(k) for k in CI.LOCK["env"]}, manifests={1: TC.digest(manifest)},
                source=CI.source_digest(TC.source_files()))
    for v in (CI.BASELINE, CI.CANDIDATE):
        run = CI.run_dir(tmp_path / "runs", v, 1, 0)
        with patch("train_contrastive.MEFARGStage1", TinyStage):
            a = args_for(root, manifest, run, v)
            TC.train(a)
            vars(a).update(checkpoint=str(run / "best.pt"), output=str(run / "evaluation"), split="test")
            TC.evaluate(a)
        assert CI.run_problems(run, v, 1, 0, lock) == []
        loaded[v] = CI.load_run(run, v, 1, 0)
    assert loaded[CI.BASELINE]["init"] == loaded[CI.CANDIDATE]["init"]
    assert np.array_equal(loaded[CI.BASELINE]["labels"], loaded[CI.CANDIDATE]["labels"])


def test_lock_matches_code_and_cell_g():
    """Đỏ nghĩa là file bị hash đã đổi: cấm push cho tới khi đủ 18 run (CLAUDE.md §6)."""
    assert CI.source_digest(TC.source_files()) == CI.LOCK["source"]
    assert CI.LOCK["variant_settings"] == {v: TC.VARIANTS[v] for v in (CI.BASELINE, CI.CANDIDATE)}
    nb = json.loads((REPO / "notebooks" / "Train_18_run.ipynb").read_text(encoding="utf-8"))
    cell = "".join(nb["cells"][-1]["source"])
    assert cell in (REPO / "COLAB_CELLS.md").read_text(encoding="utf-8")
    ns = {}
    head = cell.split("LOCKED = ", 1)[1].split("MIN_FREE_GB", 1)[0]
    exec("LOCKED = " + head, ns)
    assert ns["LOCK_SRC"] == CI.LOCK["source"] and ns["LOCK_MANIFEST"] == CI.LOCK["manifests"]
    assert ns["LOCK_ENV"] == CI.LOCK["env"] and all(CI.LOCK["common"][k] == v for k, v in ns["LOCKED"].items())


def test_report_notebook_hides_this_log(tmp_path, capsys):
    lock = toy_lock()
    runs, _ = make_runs(tmp_path, lock, (0., .1))
    CI.main(["--runs", str(runs), "--output", str(tmp_path / "o.json")], lock=lock)
    nb = json.loads((REPO / "notebooks" / "Bao_cao_log.ipynb").read_text(encoding="utf-8"))
    cell = "".join(next(c for c in nb["cells"] if c["cell_type"] == "code")["source"])
    pattern = re.search(r"TEST_LOG = re\.compile\(r'(.*)'\)", cell).group(1)
    assert re.search(pattern, capsys.readouterr().out)


@pytest.mark.parametrize("deltas,lows,old,new", [
    ((1., 1., 1.), (.1, .1, -.1), "ĐẠT", "ĐẠT"),
    ((2., 2., -.5), (.1, .1, -2.), "ĐẠT", "CHƯA CHẮC"),        # đúng ca mà điều kiện 22/11 chặn
    ((2., 2., 0.), (.1, .1, -1.), "ĐẠT", "CHƯA CHẮC"),         # Δ = 0 không tính là dương
    ((1., 1., 1.), (.1, -.1, -.1), "CHƯA CHẮC", "CHƯA CHẮC"),  # cả 3 dương không cứu được khi thiếu CI
    ((-1., -1., .5), (-2., -2., -1.), "KHÔNG ĐẠT", "KHÔNG ĐẠT"),
])
def test_rule_requires_every_seed_positive(deltas, lows, old, new):
    from stage2_decoder.summarize import verdict
    per_seed = [dict(delta_pp=d, ci_lo=lo) for d, lo in zip(deltas, lows)]
    assert verdict(per_seed) == old                            # R0 − C0 vẫn theo quy tắc cũ
    assert CI.contrast_verdict(per_seed) == new


def test_one_negative_seed_blocks_pass_end_to_end(tmp_path, capsys):
    runs, _ = make_runs(tmp_path, toy_lock(), signal=(0., .4))
    data = {j: CI.load_run(CI.run_dir(runs, *j), *j) for j in CI.jobs()}
    for k in CI.FOLDS:
        data["baseline", k, 2], data["full", k, 2] = data["full", k, 2], data["baseline", k, 2]
    rep = CI.compare(data, n_boot=200)
    assert rep["delta"]["mean"] > 0 and rep["seeds_ci_above_0"] == 2 and rep["seeds_delta_above_0"] == 2
    assert rep["verdict"] == "CHƯA CHẮC"
    CI.show(rep)
    assert "Δ > 0 ở 2/3 seed (cần đủ 3)" in capsys.readouterr().out
