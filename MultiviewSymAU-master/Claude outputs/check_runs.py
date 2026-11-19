# ============================================================
# Kiểm tra tiến độ: run nào đã train xong, run nào đã evaluate xong
# ============================================================
import json
from pathlib import Path

expected, missing_ckpt, missing_eval, stale = [], [], [], []

for variant in VARIANTS:
    for fold in FOLDS:
        for seed in SEEDS:
            out  = RUNS / variant / f'fold{fold}' / f'seed{seed}'
            ckpt = out / 'best.pt'
            mj   = out / 'evaluation' / 'test_metrics.json'
            has_c, has_m = ckpt.exists(), mj.exists()
            expected.append((variant, fold, seed, has_c, has_m))
            if not has_c:
                missing_ckpt.append((variant, fold, seed))
            elif not has_m:
                missing_eval.append((variant, fold, seed))
            elif ckpt.stat().st_mtime > mj.stat().st_mtime:
                stale.append((variant, fold, seed))   # metrics chấm trên checkpoint cũ

n   = len(expected)
n_c = sum(r[3] for r in expected)
n_m = sum(r[4] for r in expected)

print(f"RUNS = {RUNS}")
print(f"Kỳ vọng      : {n} run  ({len(VARIANTS)} variant × {len(FOLDS)} fold × {len(SEEDS)} seed)")
print(f"Có best.pt   : {n_c}/{n}")
print(f"Có metrics   : {n_m}/{n}\n")

# ---- bảng variant × fold, mỗi cột là một seed ----
w = max([len(str(v)) for v in VARIANTS] + [7]) + 2
hdr = f"{'variant':<{w}}{'fold':>5}   " + "  ".join(f"{'s'+str(s):>4}" for s in SEEDS)
print(hdr); print("-" * len(hdr))
for variant in VARIANTS:
    for fold in FOLDS:
        cells = []
        for seed in SEEDS:
            out = RUNS / variant / f'fold{fold}' / f'seed{seed}'
            if   (out/'evaluation'/'test_metrics.json').exists(): cells.append(f"{'OK':>4}")
            elif (out/'best.pt').exists():                       cells.append(f"{'--':>4}")
            else:                                                cells.append(f"{'..':>4}")
        print(f"{str(variant):<{w}}{fold:>5}   " + "  ".join(cells))
print("\nOK = đã evaluate    -- = train xong, CHƯA evaluate    .. = chưa có checkpoint")

# ---- thư mục run có trên đĩa nhưng không nằm trong VARIANTS/FOLDS/SEEDS ----
exp_paths = {RUNS / v / f'fold{f}' / f'seed{s}' for v in VARIANTS for f in FOLDS for s in SEEDS}
found = {p for p in RUNS.glob('*/fold*/seed*') if p.is_dir()} if RUNS.exists() else set()
extra = sorted(found - exp_paths)

# ---- kết luận ----
if missing_ckpt:
    print(f"\n[CẦN TRAIN LẠI] {len(missing_ckpt)} run thiếu best.pt:")
    for v, f, s in missing_ckpt: print(f"    {v}  fold{f}  seed{s}")
if missing_eval:
    print(f"\n[CẦN CHẠY CELL EVALUATE] {len(missing_eval)} run đã train nhưng chưa chấm điểm:")
    for v, f, s in missing_eval: print(f"    {v}  fold{f}  seed{s}")
if stale:
    print(f"\n[CẢNH BÁO] {len(stale)} run có best.pt MỚI HƠN test_metrics.json")
    print("           → metrics đang là của checkpoint cũ. Xoá file json rồi chạy lại cell evaluate:")
    for v, f, s in stale: print(f"    {v}  fold{f}  seed{s}")
if extra:
    print(f"\n[THỪA] {len(extra)} thư mục run không nằm trong VARIANTS/FOLDS/SEEDS:")
    for p in extra: print(f"    {p.relative_to(RUNS)}")
if not (missing_ckpt or missing_eval or stale):
    print("\n>>> ĐỦ HẾT. Sẵn sàng gộp kết quả.")

# ---- in cấu trúc một file metrics, để viết bước gộp ----
sample = next((RUNS / v / f'fold{f}' / f'seed{s}' / 'evaluation' / 'test_metrics.json'
               for v, f, s, _, hm in expected if hm), None)
if sample is not None:
    print(f"\n--- cấu trúc {sample.relative_to(RUNS)} ---")
    print(json.dumps(json.loads(sample.read_text(encoding='utf-8')),
                     indent=2, ensure_ascii=False)[:1200])
