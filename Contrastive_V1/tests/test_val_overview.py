"""Bảng tổng hợp CHỈ VAL: chỉ tính run đúng khoá có Tầng 2 làm từ đúng checkpoint; không đọc thư mục test."""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_decoder import contrast_ci as CI  # noqa: E402
from stage2_decoder import val_overview as VO  # noqa: E402

LOCK = dict(CI.LOCK, source=CI.source_digest({"toy.py": "0"}))
AUS = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]


def make(root, v, k, s, c0, r0, ckpt="h", result_ckpt=None, epochs=20, test_dir=False):
    run = root / "runs" / v / f"fold{k}" / f"seed{s}"
    (run / "evaluation_val").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(dict(CI.expected_config(v, k, s, LOCK), source_hashes={"toy.py": "0"})))
    (run / "history.json").write_text(json.dumps([dict(epoch=e) for e in range(1, epochs + 1)]))
    (run / "best.pt").write_bytes(b"x")
    (run / "environment.json").write_text(json.dumps(dict(gpu="NVIDIA GeForce RTX 5090")))
    (run / "evaluation_val" / "val_metrics.json").write_text(json.dumps(dict(checkpoint_hash=ckpt)))
    (run / "evaluation_val" / "val_predictions.npz").write_bytes(b"x")
    if test_dir:                                            # nếu bảng đọc test thì sẽ hỏng vì file rác này
        (run / "evaluation").mkdir()
        (run / "evaluation" / "test_metrics.json").write_text("not json: must never be read")
    res = root / "results" / v / f"fold{k}" / f"seed{s}"
    res.mkdir(parents=True)
    f1 = dict(C0=c0, C1=c0 + .02, R0=r0, R1=r0 + .02, R_FACS=c0)
    (res / VO.VAL_FILE).write_text(json.dumps(dict(
        selected_epoch=4, n_rules=6, checkpoint_hash=result_ckpt or ckpt, au_names=AUS,
        methods={a: dict(val_f1=f, lambda_=.5, val_f1_au=[f] * 8) for a, f in f1.items()})))


def test_overview_pairs_and_skips(tmp_path):
    make(tmp_path, "baseline", 1, 0, .70, .71, test_dir=True)
    make(tmp_path, "full", 1, 0, .73, .735)
    make(tmp_path, "baseline", 2, 0, .70, .70)
    make(tmp_path, "full", 2, 0, .71, .72, ckpt="new", result_ckpt="pilot")     # Tầng 2 cũ của pilot: không tính
    make(tmp_path, "baseline", 3, 0, .60, .60, epochs=7)                         # đang train: không tính
    rows, notes = VO.collect(tmp_path / "runs", tmp_path / "results", LOCK)
    summ = VO.summarize(rows)
    assert summ["n_runs"] == 3 and summ["of"] == 18
    assert any("full/fold2/seed0" in n and "checkpoint khác" in n for n in notes)
    assert any("baseline/fold3/seed0" in n and "chưa xong" in n for n in notes)
    fb = summ["full_minus_baseline"]
    assert fb["n"] == 1 and fb["mean_C0"] == pytest.approx(.03) and fb["mean_R0"] == pytest.approx(.025)
    assert summ["means"]["baseline"]["n"] == 2 and summ["means"]["baseline"]["R0-C0"] == pytest.approx(.005)
    assert summ["gpus"] == ["NVIDIA GeForce RTX 5090"]
    with redirect_stdout(io.StringIO()) as buf:
        VO.report(rows, notes, summ)
    assert "KHÔNG so với 63.0" in buf.getvalue() and "fold1/seed0" in buf.getvalue()


def test_cli_writes_json(tmp_path, monkeypatch):
    make(tmp_path, "baseline", 1, 0, .70, .71)
    monkeypatch.setattr(VO, "LOCK", LOCK)
    monkeypatch.setattr(VO.collect, "__defaults__", (LOCK,))
    out = tmp_path / "overview.json"
    with redirect_stdout(io.StringIO()):
        VO.main(["--runs", str(tmp_path / "runs"), "--results", str(tmp_path / "results"), "--output", str(out)])
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["val_only"] and d["summary"]["n_runs"] == 1


def test_one_file_notebook_matches_colab_cells():
    repo = Path(__file__).resolve().parents[2]
    nb = json.loads((repo / "notebooks" / "Tong_hop_val.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    md = (repo / "COLAB_CELLS.md").read_text(encoding="utf-8")
    assert len(code) == 3 and "IMAGES   = True" in code[0]
    assert code[1].startswith("# ==== Cell I:") and code[1] in md
    assert code[2].startswith("# ==== Cell L:") and code[2] in md
    assert "--val-only" in code[2] and "stage2_decoder.val_overview" in code[2]
