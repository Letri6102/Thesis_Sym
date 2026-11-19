"""Tiện ích dùng chung cho Stage 2 v2 (chỉ numpy, không cần torch)."""
import os
import itertools
import numpy as np

AU_NAMES = {
    "DISFA": ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"],
    "BP4D": ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU14",
             "AU15", "AU17", "AU23", "AU24"],
}

# Luật loại trừ viết tay trong train_Sym_Stage_2.py cũ — chỉ để đối chiếu (cột old_rule).
OLD_EXCLUSION_DISFA = {("AU2", "AU6"), ("AU2", "AU9"), ("AU1", "AU9"),
                       ("AU9", "AU12"), ("AU4", "AU12")}


def list_file(root, dataset, split, kind, fold):
    """kind: 'img_path' | 'label' | 'weight'."""
    return os.path.join(root, "list", f"{dataset}_{split}_{kind}_fold{fold}.txt")


def subject_of(path):
    """'SN001/0.png' -> 'SN001' ; 'F001/T1/2440.jpg' -> 'F001'."""
    return path.strip().replace("\\", "/").split("/")[0]


def read_split(root, dataset, split, fold):
    """Trả (paths, labels[N,A] int8 0/1, subjects[N] str)."""
    with open(list_file(root, dataset, split, "img_path", fold), encoding="utf-8") as f:
        paths = [l.strip() for l in f if l.strip()]
    labels = np.loadtxt(list_file(root, dataset, split, "label", fold))
    labels = np.atleast_2d(labels)
    if len(paths) != labels.shape[0]:
        raise ValueError(f"{split} fold{fold}: {len(paths)} ảnh nhưng {labels.shape[0]} dòng nhãn")
    labels = (labels > 0).astype(np.int8)
    subjects = np.array([subject_of(p) for p in paths])
    return paths, labels, subjects


def all_configs(num_aus):
    """(2^A, A) — mọi cấu hình nhị phân của A AU, thứ tự cố định."""
    return np.array(list(itertools.product([0, 1], repeat=num_aus)), dtype=np.int8)


def violation_matrix(Y, rules):
    """Y: (C, A) cấu hình; rules: list dict {type, i, j}. Trả V (C, R) 0/1.
    exclusion  ¬AUi ∨ ¬AUj : vi phạm khi y_i=1, y_j=1
    implication AUi → AUj  : vi phạm khi y_i=1, y_j=0
    """
    V = np.zeros((Y.shape[0], len(rules)), dtype=np.int8)
    for r, rule in enumerate(rules):
        yi, yj = Y[:, rule["i"]], Y[:, rule["j"]]
        if rule["type"] == "exclusion":
            V[:, r] = (yi == 1) & (yj == 1)
        elif rule["type"] == "implication":
            V[:, r] = (yi == 1) & (yj == 0)
        else:
            raise ValueError(rule["type"])
    return V


def f1_from_counts(tp, fp, fn):
    """Cùng công thức với utils.calc_f1_score (F1=0 khi không có TP)."""
    prec = tp / (tp + fp + 1e-20)
    rec = tp / (tp + fn + 1e-20)
    return 2 * prec * rec / (prec + rec + 1e-20)


def counts(pred, y):
    """pred, y: (N, A) 0/1 -> tp, fp, fn mỗi AU (A,)."""
    pred = pred.astype(bool)
    y = y.astype(bool)
    tp = (pred & y).sum(0).astype(np.float64)
    fp = (pred & ~y).sum(0).astype(np.float64)
    fn = (~pred & y).sum(0).astype(np.float64)
    return tp, fp, fn


def macro_f1(pred, y):
    f1 = f1_from_counts(*counts(pred, y))
    return float(f1.mean()), f1
