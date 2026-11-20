"""Test Tầng 2 (stage2_decoder) bằng dữ liệu tổng hợp — không cần ảnh, GPU hay Drive."""
import json

import numpy as np
import pytest

from experiment_data import make_manifest
from stage2_decoder import build_rules, decode, summarize
from stage2_decoder.core import AU_NAMES, OLD_EXCLUSION, Decoder, all_configs, digest, facs_rules, load_split, \
    logit, macro_f1, prediction_dir, violation_matrix

N_FRAMES = 150


def test_violation_matrix_semantics():
    Y = np.array([[1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]], np.int8)
    rules = [dict(type="exclusion", i=0, j=1), dict(type="implication", i=0, j=1)]
    assert violation_matrix(Y, rules).tolist() == [[1, 0], [0, 1], [0, 0]]


def test_decoder_is_exact_map():
    rng = np.random.default_rng(0)
    rules = [dict(type="exclusion", i=0, j=4, w0=1.3), dict(type="implication", i=3, j=5, w0=.7),
             dict(type="exclusion", i=1, j=2, w0=2.)]
    w = np.array([r["w0"] for r in rules])
    dec = Decoder(rules)
    L = rng.normal(0, 2, (50, 8))
    Y = all_configs()
    for lam in (0, .5, 3):
        pred, _ = dec.decode(L, lam)
        best = (L @ Y.T - lam * (violation_matrix(Y, rules) @ w)).max(1)
        got = (L * pred).sum(1) - lam * (violation_matrix(pred, rules) @ w)
        assert np.allclose(got, best)


def test_lambda_zero_is_plain_threshold():
    p = np.random.default_rng(1).uniform(.01, .99, (200, 8))
    dec = Decoder(facs_rules())
    assert np.array_equal(dec.decode(logit(p), 0)[0], (p > .5).astype(np.int8))
    t = np.full(8, .3)
    assert np.array_equal(dec.decode(logit(p) - logit(t), 0)[0], (p > .3).astype(np.int8))


def test_facs_rules_are_the_old_handwritten_exclusions():
    r = facs_rules()
    assert len(r) == len(OLD_EXCLUSION) == 5
    assert {frozenset((AU_NAMES[x["i"]], AU_NAMES[x["j"]])) for x in r} == {frozenset(e) for e in OLD_EXCLUSION}
    assert all(x["type"] == "exclusion" and x["w0"] == 1. and x["i"] < x["j"] for x in r)


def test_prediction_dir_prefers_evaluation_val(tmp_path):
    run = tmp_path / "run"
    assert prediction_dir(run, "val") == run / "evaluation_val"        # chưa có gì → chỗ ghi chuẩn
    (run / "evaluation").mkdir(parents=True)
    (run / "evaluation" / "val_predictions.npz").write_bytes(b"x")
    assert prediction_dir(run, "val") == run / "evaluation"            # bố cục cũ vẫn đọc được
    (run / "evaluation_val").mkdir()
    (run / "evaluation_val" / "val_predictions.npz").write_bytes(b"x")
    assert prediction_dir(run, "val") == run / "evaluation_val"
    assert prediction_dir(run, "test") == run / "evaluation"


# ---------------------------------------------------------------- end-to-end trên dữ liệu tổng hợp
def synthetic_disfa(tmp_path):
    rng = np.random.default_rng(7)
    subjects = [f"SN{i:03d}" for i in range(1, 10)]
    lab = {}
    for s in subjects:
        y = (rng.random((N_FRAMES, 8)) < .3).astype(int)
        y[:, 5] |= y[:, 3]                                   # AU6 → AU12 luôn đúng
        y[:, 1] = y[:, 0] & (rng.random(N_FRAMES) < .8)      # AU2 chỉ xuất hiện cùng AU1
        lab[s] = y
    root = tmp_path / "DISFA"
    (root / "list").mkdir(parents=True)
    folds = {1: subjects[:3], 2: subjects[3:6], 3: subjects[6:]}
    for k, test in folds.items():
        for part, subj in (("train", [s for s in subjects if s not in test]), ("test", test)):
            paths = [f"{s}/{j}.png" for s in subj for j in range(N_FRAMES)]
            (root / "list" / f"DISFA_{part}_img_path_fold{k}.txt").write_text("\n".join(paths))
            np.savetxt(root / "list" / f"DISFA_{part}_label_fold{k}.txt",
                       np.concatenate([lab[s] for s in subj]), fmt="%d")
    splits = tmp_path / "splits"
    splits.mkdir()
    for k in folds:
        m = make_manifest(root, k, val_subject_count=2, check_images=False)
        (splits / f"fold{k}.json").write_text(json.dumps(m))
    return root, splits


def fake_run(tmp_path, root, splits, k, parts, variant="baseline", seed=0):
    """Giả lập đầu ra của train_contrastive.py (config + evaluate) cho một run."""
    sp = load_split(splits / f"fold{k}.json", root)
    run = tmp_path / "runs" / variant / f"fold{k}" / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(dict(variant=variant, fold=k, seed=seed, epochs=1,
                                                     manifest_hash=digest(splits / f"fold{k}.json"))))
    rng = np.random.default_rng(100 * k + seed)
    for split in parts:
        d = sp[split]
        y = d["labels"]
        p = np.clip(.3 + .4 * y + rng.normal(0, .2, y.shape), .01, .99)
        ev = prediction_dir(run, split)
        ev.mkdir(exist_ok=True)
        np.savez_compressed(ev / f"{split}_predictions.npz", labels=y.astype(np.float32), probabilities=p,
                            paths=np.array(d["paths"]), subjects=d["subjects"])
        (ev / f"{split}_metrics.json").write_text(json.dumps(dict(
            macro_f1=macro_f1((p > .5).astype(np.int8), y)[0], checkpoint_hash="h", selected_epoch=1)))
    return run


def test_end_to_end_val_only_then_test(tmp_path):
    root, splits = synthetic_disfa(tmp_path)
    out = tmp_path / "stage2"
    for k in (1, 2, 3):
        build_rules.main(["--data-root", str(root), "--manifest", str(splits / f"fold{k}.json"),
                          "--output", str(out / "rules"), "--n-boot", "100"])
    meta = json.loads((out / "rules" / "fold1" / "rules.json").read_text(encoding="utf-8"))
    assert any(r["rule"] == "AU6 → AU12" for r in meta["rules"])          # luật cài sẵn phải được nhận

    def run_decode(k, *extra):
        run = tmp_path / "runs" / "baseline" / f"fold{k}" / "seed0"
        decode.main(["--run", str(run), "--rules", str(out / "rules" / f"fold{k}"),
                     "--manifest", str(splits / f"fold{k}.json"), "--data-root", str(root),
                     "--output", str(out / "results"), "--n-boot", "100", *extra])
        return out / "results" / "baseline" / f"fold{k}" / "seed0"

    # Giai đoạn thiết kế: chỉ có val, test chưa evaluate → --val-only không được đụng tới test.
    fake_run(tmp_path, root, splits, 1, ("val",))
    res = run_decode(1, "--val-only")
    d = json.loads((res / decode.VAL_FILE).read_text(encoding="utf-8"))
    assert d["val_only"] and not (res / decode.TEST_FILE).exists()
    assert set(d["methods"]) == {a for a, _, _ in decode.ARMS}
    assert all("test_f1" not in m for m in d["methods"].values())
    assert d["methods"]["R_FACS"]["n_rules"] == 5 and d["methods"]["R0"]["n_rules"] == len(meta["rules"])
    with pytest.raises(FileNotFoundError):                                # không --val-only thì cần test
        run_decode(1)

    # Sau khi chốt thiết kế: mở test cho cả 3 fold rồi gộp.
    for k in (1, 2, 3):
        fake_run(tmp_path, root, splits, k, ("val", "test"))
        res = run_decode(k)
        d = json.loads((res / decode.TEST_FILE).read_text(encoding="utf-8"))
        assert d["primary"] == "R0-C0"
        assert set(d["deltas"]) == {f"{a}-{b}" for a, b in decode.COMPARISONS}
        for m in d["methods"].values():
            assert 0 <= m["test_f1"] <= 1 and m["lambda_"] in decode.LAMBDAS
    report = summarize.main(["--results", str(out / "results"), "--variants", "baseline",
                             "--seeds", "0", "--n-boot", "100"])
    rep = report["per_variant"]["baseline"]
    assert rep["verdict"] in {"ĐẠT", "CHƯA CHẮC", "KHÔNG ĐẠT"}
    assert set(rep["methods"]) == set(summarize.METHODS)
    assert set(rep["lambda"]) == {"R0", "R1", "R_FACS"}


def test_verdict_rule():
    mk = lambda *xs: [dict(delta_pp=d, ci_lo=lo) for d, lo in xs]
    assert summarize.verdict(mk((.5, .1), (.4, .05), (.3, -.1))) == "ĐẠT"          # 2/3 seed CI > 0
    assert summarize.verdict(mk((.5, .1), (.4, -.05), (.3, -.1))) == "CHƯA CHẮC"
    assert summarize.verdict(mk((-.5, -1), (.1, -.2), (.1, -.3))) == "KHÔNG ĐẠT"
