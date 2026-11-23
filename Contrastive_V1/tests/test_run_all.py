"""run_all không được giữ kết quả decode làm từ checkpoint khác (23/11: runs_aug1 thay pilot Colab bằng run 5090)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_decoder import run_all as RA  # noqa: E402


def make_run(run, ckpt):
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(dict(epochs=20)))
    (run / "history.json").write_text(json.dumps([dict(epoch=20)]))
    (run / "best.pt").write_bytes(b"x")
    ev = run / "evaluation_val"
    ev.mkdir()
    (ev / "val_predictions.npz").write_bytes(b"x")
    (ev / "val_metrics.json").write_text(json.dumps(dict(checkpoint_hash=ckpt)))


@pytest.mark.parametrize("old,redo", [("ckpt-5090", False), ("ckpt-pilot-colab", True)])
def test_decode_from_another_checkpoint_is_redone(tmp_path, monkeypatch, old, redo):
    runs, out = tmp_path / "runs", tmp_path / "out"
    make_run(runs / "full" / "fold2" / "seed0", "ckpt-5090")
    (out / "rules" / "fold2").mkdir(parents=True)
    (out / "rules" / "fold2" / "rules.json").write_text("{}")
    res = out / "results" / "full" / "fold2" / "seed0"
    res.mkdir(parents=True)
    (res / RA.VAL_FILE).write_text(json.dumps(dict(checkpoint_hash=old)))
    calls = []
    monkeypatch.setattr(RA, "sh", lambda cmd: calls.append(" ".join(map(str, cmd))))
    RA.main(["--runs", str(runs), "--splits", str(tmp_path / "splits"), "--data-root", str(tmp_path),
             "--output", str(out), "--variants", "full", "--folds", "2", "--seeds", "0", "--val-only"])
    assert any("stage2_decoder.decode" in c for c in calls) == redo
    assert not any("evaluate" in c for c in calls)                 # dự đoán val của run hiện tại đã có
    stale = list((out / "stale").rglob(RA.VAL_FILE)) if (out / "stale").exists() else []
    assert len(stale) == (1 if redo else 0)                         # bản cũ được cất, không xoá
    if redo:
        assert json.loads(stale[0].read_text())["checkpoint_hash"] == old
