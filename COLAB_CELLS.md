# Các cell Colab chuẩn

Quy tắc chung nằm ở [CLAUDE.md](CLAUDE.md). Code luôn lấy từ GitHub; data và kết quả nằm trên Drive `MyDrive/Sym`.
Thứ tự một phiên làm việc: **0. Setup** → cell của việc cần làm.

---

## 0. Setup — chạy đầu mỗi phiên

Cần secret `GH_TOKEN` trong 🔑 Secrets, bật Notebook access. Token phải có:
- Repository access: Only select → `Letri6102/Thesis_Sym`;
- Contents: Read-only.

Hàm `git()` cố ý **không** dùng `check=True`: `CalledProcessError` in nguyên lệnh, tức là in cả header chứa token.

```python
import os, sys, json, time, base64, shutil, subprocess
from pathlib import Path
from google.colab import drive, userdata

if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive')

REPO_URL = 'https://github.com/Letri6102/Thesis_Sym.git'
BRANCH   = 'main'
REPO     = Path('/content/Thesis_Sym')
CODE     = REPO / 'Contrastive_V1'
SYM      = Path('/content/drive/MyDrive/Sym')
SPLITS   = SYM / 'splits_v2'           # manifest mới; splits/ cũ chỉ cho runs_retry1
RUNS     = SYM / 'runs_aug1'           # run mới với strong_train_aug
LOGS     = SYM / 'logs'

tok  = userdata.get('GH_TOKEN')
auth = 'http.extraHeader=Authorization: Basic ' + base64.b64encode(f'x-access-token:{tok}'.encode()).decode()
def git(*args, cwd=None):
    r = subprocess.run(['git', '-c', auth, *args], cwd=cwd, capture_output=True, text=True,
                       env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
    if r.returncode:
        raise RuntimeError(f'git {args[0]} lỗi (exit {r.returncode}): {r.stderr.strip()}')
try:
    if (REPO / '.git').exists():
        git('fetch', '-q', 'origin', BRANCH, cwd=REPO)
        git('reset', '-q', '--hard', f'origin/{BRANCH}', cwd=REPO)
    else:
        git('clone', '-q', '--depth', '1', '-b', BRANCH, REPO_URL, str(REPO))
finally:
    del tok, auth
print('commit:', subprocess.run(['git', 'log', '-1', '--format=%h %s'], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())

IMG_TAR = SYM / 'DISFA_img.tar'
if IMG_TAR.exists():
    DATA = Path('/content/DISFA')
    if not (DATA / 'img').exists():
        t0 = time.time()
        DATA.mkdir(parents=True, exist_ok=True)
        shutil.copy(IMG_TAR, '/content/DISFA_img.tar')
        subprocess.run(['tar', '-xf', '/content/DISFA_img.tar', '-C', str(DATA)], check=True)
        os.remove('/content/DISFA_img.tar')
        shutil.copytree(SYM / 'DISFA' / 'list', DATA / 'list', dirs_exist_ok=True)
        print(f'Đã chép ảnh về /content trong {time.time() - t0:.0f}s')
else:
    DATA = SYM / 'DISFA'                 # đọc thẳng Drive: chậm và hay lỗi FileNotFoundError
print('DATA =', DATA)

os.chdir(CODE)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-colab.txt'], check=True)
(CODE / 'checkpoints').mkdir(exist_ok=True)
w_src = SYM / 'Contrastive_V1' / 'checkpoints' / 'resnet50-19c8e357.pth'
w_dst = CODE / 'checkpoints' / 'resnet50-19c8e357.pth'
if w_src.exists() and not w_dst.exists():
    shutil.copy(w_src, w_dst)
import torch, torchvision
print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda', torch.cuda.is_available())

import re
def run_logged(cmd, name):
    """Chạy lệnh con, ghi toàn bộ output vào MyDrive/Sym/logs/ và in các dòng đáng xem.
    (Colab không hiện stdout của tiến trình con nếu chỉ dùng subprocess.run.)"""
    LOGS.mkdir(exist_ok=True)
    log = LOGS / f'{time.strftime("%Y%m%d_%H%M%S")}_{name}.log'
    print(f'>> {name} | log: {log.name}', flush=True)
    with open(log, 'w') as f, subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for line in p.stdout:
            f.write(line); f.flush()
            if line.startswith('{"epoch"'):
                r = json.loads(line)
                print(f"   ep{r['epoch']:>2}  wa={r['train']['wa']:.4f}  valF1={r['val']['macro_f1']:.3f}"
                      f"  {r['seconds'] / 60:.0f} phút", flush=True)
            elif line.startswith('epoch ') and ' step ' in line:
                if re.search(r'step (1000|2000)/', line):
                    print('   ' + line.rstrip(), flush=True)
            elif 'FutureWarning' not in line and 'warnings.warn' not in line:
                print('   ' + line.rstrip(), flush=True)
    if p.returncode:
        raise RuntimeError(f'{name} dừng với mã {p.returncode}, xem log {log}')
```

---

## A. Đóng gói ảnh thành `DISFA_img.tar` — chạy MỘT lần, không cần GPU

Đọc 130.814 ảnh từ Drive (16 luồng, mỗi ảnh thử lại tối đa 5 lần), ghi tar không nén ra `/content`, kiểm đủ số ảnh, rồi mới chép lên Drive. File chép lên mang tên `.partial` và chỉ được đổi tên khi chép xong, nên cell Setup không bao giờ dùng nhầm một file dở dang.
Nên chọn runtime **CPU** để khỏi tốn GPU. Mất khoảng 1–3 giờ. Nếu phiên bị ngắt thì chạy lại từ đầu.

```python
import os, io, time, shutil, tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from google.colab import drive
if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive')

SYM   = Path('/content/drive/MyDrive/Sym')
SRC   = SYM / 'DISFA' / 'img'
FINAL = SYM / 'DISFA_img.tar'
LOCAL = Path('/content/DISFA_img.tar')
assert not FINAL.exists(), f'{FINAL} đã có — không cần chạy lại'

rel = sorted({p.strip() for k in (1, 2, 3)
              for p in (SYM / 'DISFA' / 'list' / f'DISFA_test_img_path_fold{k}.txt').read_text().split() if p.strip()})
print(len(rel), 'ảnh cần đóng gói')        # phải là 130814

def read(p, tries=5):
    for t in range(tries):
        try:
            return (SRC / p).read_bytes()
        except OSError:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))

t0 = time.time()
with tarfile.open(LOCAL, 'w', format=tarfile.GNU_FORMAT) as tar, ThreadPoolExecutor(16) as ex:
    for i in range(0, len(rel), 2000):
        chunk = rel[i:i + 2000]
        for p, data in zip(chunk, ex.map(read, chunk)):
            info = tarfile.TarInfo('img/' + p)
            info.size, info.mtime = len(data), 0
            tar.addfile(info, io.BytesIO(data))
        print(f'{i + len(chunk):>6}/{len(rel)}  {(time.time() - t0) / 60:.1f} phút', flush=True)

with tarfile.open(LOCAL) as tar:
    n = sum(1 for m in tar if m.isfile())
assert n == len(rel), f'tar có {n} ảnh, cần {len(rel)}'
print(f'tar OK: {n} ảnh, {LOCAL.stat().st_size / 1e9:.2f} GB — đang chép lên Drive...')
tmp = SYM / 'DISFA_img.tar.partial'
shutil.copy(LOCAL, tmp)
tmp.rename(FINAL)
print('Xong:', FINAL, f'| tổng {(time.time() - t0) / 60:.0f} phút')
```

---

## B. Pilot — `baseline` + `full`, fold 2, seed 0 (cần GPU, chạy sau Setup)

Chỉ dùng inner val, **không** đánh giá test. Cell ghi log vào `MyDrive/Sym/logs/` và in tóm tắt mỗi epoch, qua hàm `run_logged` định nghĩa trong cell Setup.
`EPOCHS`, `BATCH`, `WORKERS` nằm trong config của run. Đổi chúng giữa chừng thì `--resume` sẽ từ chối.
Bị ngắt phiên: chạy lại Setup rồi chạy lại cell này. Cell tự resume từ `last.pt` và bỏ qua run đã xong.

```python
VARIANTS, FOLD, SEED = ['baseline', 'full'], 2, 0
EPOCHS, BATCH, WORKERS = 20, 32, 4
assert str(DATA).startswith('/content/'), 'Chưa có DISFA_img.tar: chạy cell A trước (đọc thẳng Drive hay lỗi)'

for v in VARIANTS:
    out = RUNS / v / f'fold{FOLD}' / f'seed{SEED}'
    hist = out / 'history.json'
    if hist.exists() and len(json.loads(hist.read_text())) >= EPOCHS:
        print(f'{v}: đã xong {EPOCHS} epoch, bỏ qua'); continue
    cmd = [sys.executable, '-u', 'train_contrastive.py', 'train', '--variant', v, '--seed', SEED,
           '--arc', 'resnet50', '--data-root', DATA, '--manifest', SPLITS / f'fold{FOLD}.json',
           '--output', out, '--epochs', EPOCHS, '--batch-size', BATCH, '--workers', WORKERS]
    if (out / 'last.pt').exists():
        cmd += ['--resume', out / 'last.pt']
    run_logged(cmd, f'train_aug1_{v}_fold{FOLD}_seed{SEED}')
```

---

## C. Kiểm pilot — so với run cũ cùng fold 2 (val fold 2 không đổi nên so được)

Pilot đạt khi (đề cương §5, §7):
- `wa` cuối **không** về khoảng 0;
- val F1 đạt đỉnh **sau** epoch 2;
- val F1 tốt nhất **không thấp hơn** run cũ (0.715 cho baseline, 0.705 cho full).

Nếu F1 của các AU nhỏ (AU1/2/9/26) tụt thì thử bỏ `RandomErasing`.

```python
import json
AU = [1, 2, 4, 6, 9, 12, 25, 26]
for v in ['baseline', 'full']:
    p_new = SYM / 'runs_aug1' / v / 'fold2' / 'seed0' / 'history.json'
    if not p_new.exists():
        print(v, ': chưa có'); continue
    new = json.loads(p_new.read_text())
    old = json.loads((SYM / 'runs_retry1' / v / 'fold2' / 'seed0' / 'history.json').read_text())
    bn = max(new, key=lambda e: e['val']['macro_f1'])
    bo = max(old, key=lambda e: e['val']['macro_f1'])
    print(f"{v}: {len(new)} epoch | val F1 tốt nhất mới {bn['val']['macro_f1']:.3f} @ep{bn['epoch']}"
          f"  vs cũ {bo['val']['macro_f1']:.3f} @ep{bo['epoch']}"
          f" | wa cuối mới {new[-1]['train']['wa']:.4f}  cũ {old[-1]['train']['wa']:.5f}")
    print('   F1/AU mới:', ' '.join(f'AU{a}:{x:.2f}' for a, x in zip(AU, bn['val']['per_au_f1'])))
    print('   F1/AU cũ :', ' '.join(f'AU{a}:{x:.2f}' for a, x in zip(AU, bo['val']['per_au_f1'])))
    print('   val F1 theo epoch:', ' '.join(f"{e['val']['macro_f1']:.3f}" for e in new))
```

---

## D. Tầng 2 — giai đoạn thiết kế, CHỈ VAL (cần GPU cho bước evaluate val, chạy sau Setup)

Dùng module `stage2_decoder`, không dùng notebook `Tang2_Decoder.ipynb` nữa. Cell này:
1. rút luật từ nhãn fit của `splits_v2`;
2. chạy `evaluate --split val` với `best.pt`, ghi vào `<run>/evaluation_val/`;
3. decode các nhánh C0/C1/R0/R1/R_FACS **chỉ trên val**.

**Không đụng outer test.** Kết quả ghi vào `MyDrive/Sym/stage2_v2/`. Chạy lại an toàn: bước nào đã có kết quả thì bỏ qua.
Lưu ý: điểm val ở đây có λ và ngưỡng được chọn trên chính val, nên chỉ dùng để xem chiều hướng, **không** dùng để báo cáo.

```python
VARIANTS, FOLDS, SEEDS = ['baseline', 'full'], [2], [0]     # pilot: fold 2
STAGE2 = SYM / 'stage2_v2'
run_logged([sys.executable, '-u', '-m', 'stage2_decoder.run_all', '--runs', RUNS, '--splits', SPLITS,
            '--data-root', DATA, '--output', STAGE2, '--variants', *VARIANTS,
            '--folds', *FOLDS, '--seeds', *SEEDS, '--val-only'], 'stage2_valonly')
```

**Mở test** (chỉ làm **một lần**, sau khi đã chốt Tầng 1 và đủ 3 fold): dùng đúng lệnh trên nhưng **bỏ** `'--val-only'` và để `FOLDS = [1, 2, 3]`. Lệnh sẽ evaluate test, decode và gộp bằng `summarize` thành `stage2_v2/summary.json`. So sánh chính là `R0 − C0`.

---

## Phụ lục

**Kéo log và kết quả về máy để phân tích** (bỏ ảnh, nhãn gốc, trọng số):

```python
import os, zipfile
from google.colab import files
ROOT, OUT = '/content/drive/MyDrive/Sym', '/content/Sym_source.zip'
SKIP_DIRS = {'img', 'ActionUnit_Labels', 'source', '__pycache__', '.pytest_cache', 'tmp', 'checkpoints', '.ipynb_checkpoints'}
SKIP_EXT  = {'.pt', '.pth', '.ckpt', '.zip', '.tar', '.gz', '.npz', '.npy', '.partial'}
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for f in fns:
            if os.path.splitext(f)[1].lower() not in SKIP_EXT and os.path.getsize(os.path.join(dp, f)) < 20e6:
                p = os.path.join(dp, f); z.write(p, os.path.relpath(p, ROOT))
print(f'{os.path.getsize(OUT) / 1e6:.1f} MB'); files.download(OUT)
```

**Sinh `splits_v2`**: đã chạy ngày 20/11. Fold 1 có val mới là SN002, SN016, SN032; fold 2/3 không đổi. **Không chạy lại.**
