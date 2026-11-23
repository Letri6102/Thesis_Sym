"""Mô phỏng sinh bảng ở đề cương §6: vì sao thêm "Δ > 0 ở cả 3 seed" (22/11). Không đọc data hay dự đoán.

  python -m stage2_decoder.sim_contrast_rule        # khoảng 3 phút CPU, seed cố định

Mọi số là tổng hợp. Giả định:
  - 3 fold × 9 subject test × 4845 frame; tần suất AU lấy từ fit fold 2, mức F1 từng AU theo bảng của paper;
  - run_sd: nhiễu chung của một lần train (theo AU, thang logit) — chính là phần mà CI theo subject bỏ sót;
  - thêm nhiễu subject × lần train (0.3) và subject × variant (0.15);
  - Δ thật = 0: hai variant hoán đổi được.
Quy tắc cũ = summarize.verdict; quy tắc mới = contrast_ci.contrast_verdict.
"""
import numpy as np

from . import contrast_ci as CI
from .core import f1
from .summarize import verdict

PREV = np.array([4.8, 5.2, 11.9, 7.8, 4.4, 13.1, 25.3, 6.6]) / 100
QUAL = np.array([.533, .521, .776, .513, .485, .746, .917, .544])
N_FRAMES, REPS, N_BOOT = 4845, 2000, 1000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def draw(rng, run_sd, effect):
    """tp/fp/fn kỳ vọng: (variant, fold, seed, subject, 3, AU)."""
    rate = np.clip(PREV * rng.lognormal(-.5, 1., (3, 9, 8)), 0, .9) * (rng.random((3, 9, 8)) > .1)
    pos = rate * N_FRAMES
    ability = rng.normal(0, .7, (2, 1, 3, 1, 9, 8)).reshape(2, 3, 1, 9, 8)
    q = (rng.normal(0, run_sd, (2, 3, 3, 1, 8))
         + rng.normal(0, .3, (2, 3, 3, 9, 8))
         + rng.normal(0, .15, (2, 3, 1, 9, 8)))
    q[1] += effect
    recall = sigmoid(logit(QUAL) + ability[:1] + q)
    fpr = sigmoid(logit(PREV * (1 - QUAL) / (1 - PREV)) + ability[1:] - q)
    pos = pos[None, :, None]
    tp = pos * recall
    return np.stack([tp, (N_FRAMES - pos) * fpr, pos - tp], axis=-2)


def macro(cnt):
    return f1(np.moveaxis(cnt, -2, 0)).mean(-1)


def analyse(cnt, n_boot):
    """Như contrast_ci.compare, nhưng vector hoá cho nhanh."""
    point = macro(cnt.sum(3))
    delta = 100 * (point[1] - point[0]).mean(0)
    flat = cnt.reshape(cnt.shape[:4] + (24,))
    boot = np.zeros((3, n_boot))
    for s in range(3):
        rng = np.random.default_rng([CI.BOOT_SEED, s])
        for k in range(3):
            idx = rng.integers(0, 9, (n_boot, 9))
            weight = np.zeros((n_boot, 9))
            np.add.at(weight, (np.arange(n_boot)[:, None], idx), 1)
            m = [macro((weight @ flat[v, k, s]).reshape(n_boot, 3, 8)) for v in (0, 1)]
            boot[s] += 100 * (m[1] - m[0]) / 3
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=1)
    return delta, lo, hi


def main():
    print(f"{REPS} lần lặp mỗi dòng, {N_BOOT} bootstrap")
    print(f"{'run_sd':>6}{'effect':>7}{'mean Δ':>9}{'SD seed':>9}{'cũ':>9}{'mới':>9}   (% ĐẠT; effect 0 = ĐẠT nhầm)")
    for effect in (0., .04, .08):
        for run_sd in (0., .05, .1, .2, .3):
            rng = np.random.default_rng([int(effect * 100), int(run_sd * 100)])
            passed, means, variances = np.zeros(2), [], []
            for _ in range(REPS):
                delta, lo, _ = analyse(draw(rng, run_sd, effect), N_BOOT)
                per_seed = [dict(delta_pp=d, ci_lo=x) for d, x in zip(delta, lo)]
                passed += verdict(per_seed) == "ĐẠT", CI.contrast_verdict(per_seed) == "ĐẠT"
                means.append(delta.mean())
                variances.append(delta.var(ddof=1))
            old, new = 100 * passed / REPS
            print(f"{run_sd:>6.2f}{effect:>7.2f}{np.mean(means):>+9.2f}{np.sqrt(np.mean(variances)):>9.2f}"
                  f"{old:>8.1f}%{new:>8.1f}%", flush=True)


if __name__ == "__main__":
    main()
