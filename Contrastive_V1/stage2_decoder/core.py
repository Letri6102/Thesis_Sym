"""Lõi Stage 2 decoder (chỉ numpy): đọc manifest, thống kê luật, MAP decoder, metric, bootstrap.

Quy ước dữ liệu giống experiment_data.py của Contrastive_V1:
  <data-root>/list/DISFA_{train,test}_{img_path,label}_fold{k}.txt
  splits/fold{k}.json : fit_indices / val_indices (chỉ số trong outer train), input_hashes
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

AU_NAMES = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]
A = len(AU_NAMES)
EPS = 1e-6

# Luật loại trừ viết tay trong train_Sym_Stage_2.py cũ (tri thức FACS của SymGraphAU).
OLD_EXCLUSION = {("AU2", "AU6"), ("AU2", "AU9"), ("AU1", "AU9"), ("AU9", "AU12"), ("AU4", "AU12")}

# Nơi train_contrastive.py evaluate ghi dự đoán. Bản đầu (notebook Tầng 2, 13/11) để val ở
# evaluation_val/ và test ở evaluation/; bản cũ hơn để cả hai ở evaluation/ — vẫn đọc được.
EVAL_DIRS = {"val": ("evaluation_val", "evaluation"), "test": ("evaluation",)}


def prediction_dir(run, split):
    """Thư mục chứa <split>_predictions.npz của run; chưa có thì trả thư mục chuẩn (để ghi)."""
    dirs = [Path(run) / d for d in EVAL_DIRS[split]]
    return next((d for d in dirs if (d / f"{split}_predictions.npz").exists()), dirs[0])


def facs_rules(weight=1.0):
    """Nhánh R_FACS: cùng decoder MAP nhưng dùng luật viết tay thay cho luật rút từ dữ liệu.
    Không có trọng số từ dữ liệu nên mọi luật cùng w0 = weight; độ mạnh do λ (chọn trên val) quyết định."""
    rules = []
    for a, b in sorted(OLD_EXCLUSION, key=lambda r: (AU_NAMES.index(r[0]), AU_NAMES.index(r[1]))):
        i, j = sorted((AU_NAMES.index(a), AU_NAMES.index(b)))
        rules.append(dict(type="exclusion", i=i, j=j, w0=float(weight), rule=f"¬{AU_NAMES[i]} ∨ ¬{AU_NAMES[j]}"))
    return rules


# ---------------------------------------------------------------- dữ liệu
def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_partition(root, fold, part):
    base = Path(root) / "list"
    paths = [p.strip().replace("\\", "/")
             for p in (base / f"DISFA_{part}_img_path_fold{fold}.txt").read_text().splitlines() if p.strip()]
    labels = np.loadtxt(base / f"DISFA_{part}_label_fold{fold}.txt", ndmin=2)
    if labels.shape != (len(paths), A) or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"fold{fold} {part}: cần {len(paths)}x{A} nhãn nhị phân")
    return paths, labels.astype(np.int8)


def load_split(manifest_path, data_root, check_hash=True):
    """Trả manifest + nhãn/subject của fit, val (inner) và test."""
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if check_hash:
        for name, expected in m["input_hashes"].items():
            if digest(Path(data_root) / "list" / name) != expected:
                raise ValueError(f"List đã thay đổi so với manifest: {name}")
    fold = m["fold"]
    tr_paths, tr_lab = read_partition(data_root, fold, "train")
    te_paths, te_lab = read_partition(data_root, fold, "test")
    out = {"manifest": m, "fold": fold}
    for split, idx in (("fit", m["fit_indices"]), ("val", m["val_indices"])):
        p = [tr_paths[i] for i in idx]
        out[split] = dict(paths=p, labels=tr_lab[idx], subjects=np.array([q.split("/")[0] for q in p]))
    out["test"] = dict(paths=te_paths, labels=te_lab, subjects=np.array([q.split("/")[0] for q in te_paths]))
    fs, vs, ts = (set(out[s]["subjects"]) for s in ("fit", "val", "test"))
    if fs & vs or fs & ts or vs & ts:
        raise ValueError("Subject bị trùng giữa fit / val / test")
    return out


def load_predictions(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    return dict(probs=z["probabilities"].astype(np.float64), labels=z["labels"].astype(np.int8),
                subjects=z["subjects"].astype(str), paths=z["paths"].astype(str))


# ---------------------------------------------------------------- luật
def candidate_rules(with_implication=True):
    rules = [{"type": "exclusion", "i": i, "j": j} for i in range(A) for j in range(i + 1, A)]
    if with_implication:
        rules += [{"type": "implication", "i": i, "j": j} for i in range(A) for j in range(A) if i != j]
    return rules


def rule_text(r):
    a, b = AU_NAMES[r["i"]], AU_NAMES[r["j"]]
    return f"¬{a} ∨ ¬{b}" if r["type"] == "exclusion" else f"{a} → {b}"


def rule_stats(n, c, co, ri, rj, is_excl):
    """Số frame vi phạm O, kỳ vọng nếu độc lập E, và P(j|i). Vector hoá trên mọi luật."""
    ci, cj, cij = c[..., ri], c[..., rj], co[..., ri, rj]
    nn = np.maximum(n[..., None].astype(np.float64), 1)
    O = np.where(is_excl, cij, ci - cij).astype(np.float64)
    E = np.where(is_excl, ci * cj / nn, ci * (nn - cj) / nn)
    return O, E, cij / np.maximum(ci, 1)


def all_configs():
    return np.array(list(itertools.product([0, 1], repeat=A)), dtype=np.int8)   # AU1 = bit cao nhất


def violation_matrix(Y, rules):
    V = np.zeros((len(Y), len(rules)), np.int8)
    for k, r in enumerate(rules):
        yi, yj = Y[:, r["i"]], Y[:, r["j"]]
        V[:, k] = (yi == 1) & (yj == 1) if r["type"] == "exclusion" else (yi == 1) & (yj == 0)
    return V


# ---------------------------------------------------------------- decoder
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


class Decoder:
    """MAP chính xác trên 2^8 cấu hình: argmax_y Σ y_i(logit p_i + b_i) − λ Σ_r w_r V[y,r]."""

    def __init__(self, rules):
        self.Yi = all_configs()
        self.Y = self.Yi.astype(np.float64)
        self.V = violation_matrix(self.Yi, rules).astype(np.float64)
        self.pen = self.V @ np.array([r["w0"] for r in rules]) if rules else np.zeros(len(self.Y))

    def decode(self, L, lam, chunk=16384):
        idx = np.empty(len(L), np.int64)
        for s in range(0, len(L), chunk):
            idx[s:s + chunk] = (L[s:s + chunk] @ self.Y.T - lam * self.pen).argmax(1)
        return self.Yi[idx], idx

    @staticmethod
    def config_index(y):
        return (y.astype(np.int64) << np.arange(A - 1, -1, -1)).sum(1)


# ---------------------------------------------------------------- metric
def counts(pred, y):
    pred, y = pred.astype(bool), y.astype(bool)
    return (np.stack([(pred & y).sum(0), (pred & ~y).sum(0), (~pred & y).sum(0)]).astype(np.float64))


def f1(cnt):
    """cnt: (3, A) tp, fp, fn. Cùng công thức train_contrastive.metrics: 2tp / max(2tp+fp+fn, 1)."""
    tp, fp, fn = cnt
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def macro_f1(pred, y):
    v = f1(counts(pred, y))
    return float(v.mean()), v


def tune_thresholds(p, y, grid):
    thr = np.empty(A)
    for a in range(A):
        scores = [f1(counts((p[:, a] >= t)[:, None], y[:, a:a + 1]))[0] for t in grid]
        thr[a] = grid[int(np.argmax(scores))]          # hoà ⇒ ngưỡng nhỏ nhất trong lưới
    return thr


def per_subject_counts(pred, y, subjects, subj_list):
    return np.stack([counts(pred[subjects == s], y[subjects == s]) for s in subj_list])   # (S, 3, A)


def boot_delta(cnt_a, cnt_b, n_boot, rng):
    S = len(cnt_a)
    out = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, S, S), minlength=S).astype(np.float64)
        out[b] = f1(np.tensordot(w, cnt_a, 1)).mean() - f1(np.tensordot(w, cnt_b, 1)).mean()
    return out
