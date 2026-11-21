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
BRANCH   = 'main'                      # 'stage1-ema-valchecks': chỉ ở phiên chạy cell B2 khi nhánh chưa merge
REPO     = Path('/content/Thesis_Sym')
CODE     = REPO / 'Contrastive_V1'
SYM      = Path('/content/drive/MyDrive/Sym')
SPLITS   = SYM / 'splits_v2'           # manifest mới; splits/ cũ chỉ cho runs_retry1
RUNS     = SYM / 'runs_aug1'           # run mới với strong_train_aug
LOGS     = SYM / 'logs'
IMAGES   = True                        # False: không giải nén ảnh (phiên CPU chỉ chạy Tầng 2, cell E)

tok  = userdata.get('GH_TOKEN')
auth = 'http.extraHeader=Authorization: Basic ' + base64.b64encode(f'x-access-token:{tok}'.encode()).decode()
def git(*args, cwd=None):
    r = subprocess.run(['git', '-c', auth, *args], cwd=cwd, capture_output=True, text=True,
                       env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
    if r.returncode:
        raise RuntimeError(f'git {args[0]} lỗi (exit {r.returncode}): {r.stderr.strip()}')
try:
    if (REPO / '.git').exists():
        # refspec đầy đủ: bản clone 1 nhánh vẫn đổi được sang nhánh khác (fetch tên trơn chỉ cập nhật FETCH_HEAD)
        git('fetch', '-q', '--depth', '1', 'origin', f'+refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}', cwd=REPO)
        git('reset', '-q', '--hard', f'origin/{BRANCH}', cwd=REPO)
    else:
        git('clone', '-q', '--depth', '1', '-b', BRANCH, REPO_URL, str(REPO))
finally:
    del tok, auth
print('commit:', subprocess.run(['git', 'log', '-1', '--format=%h %s'], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())

SHARDS = SYM / 'DISFA_img'                     # 27 tar theo subject, do cell A tạo
if IMAGES and (SHARDS / 'DONE.json').exists():
    DATA = Path('/content/DISFA')
    if not (DATA / '.img_ok').exists():        # chưa giải nén, hoặc lần trước giải nén dở
        t0, meta = time.time(), json.loads((SHARDS / 'DONE.json').read_text())
        DATA.mkdir(parents=True, exist_ok=True)
        print(f'Giải nén {len(meta["subjects"])} tar ({meta["bytes"] / 1e9:.1f} GB) về /content:', flush=True)
        for i, s in enumerate(meta['subjects'], 1):
            subprocess.run(['tar', '-xf', str(SHARDS / f'{s}.tar'), '-C', str(DATA)], check=True, timeout=1800)
            print(f'  {i:>2}/{len(meta["subjects"])} {s}  {(time.time() - t0) / 60:.1f} phút', flush=True)
        n = sum(len(f) for _, _, f in os.walk(DATA / 'img'))
        assert n == meta['n_images'], f'Giải nén được {n} ảnh, cần {meta["n_images"]}'
        shutil.copytree(SYM / 'DISFA' / 'list', DATA / 'list', dirs_exist_ok=True)
        (DATA / '.img_ok').touch()
        print(f'Đã giải nén {n} ảnh về /content trong {time.time() - t0:.0f}s')
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
print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| CPU', os.cpu_count(),
      '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'KHÔNG CÓ')

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

## A. Đóng gói ảnh thành 27 tar theo subject — chạy lại được, không cần GPU

Cell này không cần cell Setup và không cần token GitHub. Kết quả nằm trong `MyDrive/Sym/DISFA_img/`:
- `SNxxx.tar`: ảnh của một subject;
- `SNxxx.json`: dấu đã xong, ghi số ảnh và số byte của tar;
- `DONE.json`: chỉ có khi đủ 27 subject. Cell Setup chỉ dùng ảnh khi thấy file này.

Với mỗi subject chưa xong, cell:
1. đọc khoảng 4.845 ảnh từ Drive bằng 16 luồng, mỗi ảnh thử lại tối đa 5 lần;
2. ghi tar ra `/content/shards` rồi kiểm đủ ảnh;
3. chép lên Drive dưới tên `.partial`, đổi tên khi chép xong, rồi mới ghi `SNxxx.json`.

**Resume:** subject nào có `SNxxx.json` khớp với `SNxxx.tar` thì bỏ qua. Bị ngắt phiên chỉ mất subject đang làm dở.

**Chống treo:** bản một file đầu tiên (21/11) đứng ở 40.000/130.814 ảnh suốt hơn 12 giờ mà không báo lỗi, vì đọc file trên Drive không có timeout. Giờ mỗi ảnh chỉ được đọc tối đa 5 phút. Quá hạn thì cell dừng và in `DRIVE TREO`. Khi đó vào **Runtime → Restart session** (không được thì **Disconnect and delete runtime**) rồi chạy lại cell A.

Nên chọn runtime **CPU**. Nếu Drive không treo, toàn bộ mất khoảng 40–60 phút. Cuối cell gọi `flush_and_unmount` để chờ Drive đồng bộ xong: **đừng tắt tab** trước khi cell in dòng `XONG`.

```python
# ==== Cell A: đóng gói MyDrive/Sym/DISFA/img thành MyDrive/Sym/DISFA_img/SNxxx.tar (chạy lại được) ====
import os, io, json, time, shutil, tarfile
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from google.colab import drive

if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive')

SYM   = Path('/content/drive/MyDrive/Sym')
SRC   = SYM / 'DISFA' / 'img'
OUT   = SYM / 'DISFA_img'                    # SNxxx.tar + SNxxx.json (dấu đã xong) + DONE.json
LOCAL = Path('/content/shards')
THREADS, READ_TIMEOUT, COPY_TIMEOUT = 16, 300, 1800   # giây; đọc 1 ảnh bình thường < 1 s
OUT.mkdir(exist_ok=True)
LOCAL.mkdir(exist_ok=True)
assert not (OUT / 'DONE.json').exists(), 'Đã đóng gói đủ 27 subject — không cần chạy lại'
(SYM / 'DISFA_img.tar.partial').unlink(missing_ok=True)    # rác của cell A bản một file (nếu có)

rel = sorted({p.strip() for k in (1, 2, 3)
              for p in (SYM / 'DISFA' / 'list' / f'DISFA_test_img_path_fold{k}.txt').read_text().split() if p.strip()})
assert len(rel) == 130814, f'List có {len(rel)} ảnh, cần 130814 — dừng lại kiểm tra'
by_subj = defaultdict(list)
for p in rel:
    by_subj[p.split('/')[0]].append(p)

def is_done(s):
    mark, tar = OUT / f'{s}.json', OUT / f'{s}.tar'
    if not (mark.exists() and tar.exists()):
        return False
    m = json.loads(mark.read_text())
    return m['n'] == len(by_subj[s]) and m['size'] == tar.stat().st_size

todo = [s for s in sorted(by_subj) if not is_done(s)]
free = shutil.disk_usage('/content/drive').free
print(f'{len(rel)} ảnh, {len(by_subj)} subject | đã xong {len(by_subj) - len(todo)}, còn {len(todo)}'
      f' | Drive trống {free / 1e9:.0f} GB', flush=True)
assert free > 0.4e9 * len(todo), 'Google Drive không đủ chỗ (~0.35 GB mỗi subject) — dọn bớt'

def read(p, tries=5):
    for t in range(tries):
        try:
            return (SRC / p).read_bytes()
        except OSError:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))

class DriveHang(Exception):
    pass

def wait(fut, timeout, what):
    try:
        return fut.result(timeout=timeout)
    except FutTimeout:
        raise DriveHang(f'{what} quá {timeout} s') from None

ex = ThreadPoolExecutor(THREADS)
t0 = time.time()
try:
    for i, s in enumerate(todo, 1):
        ts, paths, loc = time.time(), by_subj[s], LOCAL / f'{s}.tar'
        futs = [ex.submit(read, p) for p in paths]
        with tarfile.open(loc, 'w', format=tarfile.GNU_FORMAT) as tar:
            for p, f in zip(paths, futs):
                data = wait(f, READ_TIMEOUT, f'Đọc {p}')
                info = tarfile.TarInfo('img/' + p)
                info.size, info.mtime = len(data), 0
                tar.addfile(info, io.BytesIO(data))
        with tarfile.open(loc) as tar:
            names = sorted(m.name for m in tar if m.isfile())
        assert names == ['img/' + p for p in paths], f'{s}: tar thiếu/thừa ảnh'
        size, tmp = loc.stat().st_size, OUT / f'{s}.tar.partial'
        wait(ex.submit(shutil.copy, loc, tmp), COPY_TIMEOUT, f'Chép {s}.tar lên Drive')
        assert tmp.stat().st_size == size, f'{s}: bản chép lên Drive bị thiếu byte'
        tmp.replace(OUT / f'{s}.tar')
        (OUT / f'{s}.json').write_text(json.dumps({'n': len(paths), 'size': size}))
        loc.unlink()
        el = time.time() - t0
        print(f'[{i:>2}/{len(todo)}] {s}: {len(paths)} ảnh, {size / 1e6:.0f} MB, {(time.time() - ts) / 60:.1f} phút'
              f' | tổng {el / 60:.0f} phút, còn ~{el / i * (len(todo) - i) / 60:.0f} phút', flush=True)
except DriveHang as e:
    raise RuntimeError(f'DRIVE TREO ({e}). Các subject đã xong vẫn nằm trên Drive. '
                       'Runtime → Restart session (không được thì Disconnect and delete runtime), '
                       'rồi chạy lại cell A.') from None
finally:
    ex.shutdown(wait=False, cancel_futures=True)      # không chờ luồng đang treo

assert all(is_done(s) for s in by_subj), 'Còn subject chưa xong — chạy lại cell A'
total = sum((OUT / f'{s}.tar').stat().st_size for s in by_subj)
(OUT / 'DONE.json').write_text(json.dumps({'subjects': sorted(by_subj), 'n_images': len(rel), 'bytes': total}))
print('Đang chờ Drive đồng bộ (có thể vài phút)...', flush=True)
drive.flush_and_unmount()
print(f'XONG: {len(by_subj)} tar trong {OUT} ({total / 1e9:.2f} GB) | {(time.time() - t0) / 60:.0f} phút')
```

---

## B. Pilot — `baseline` + `full`, fold 2, seed 0 (cần GPU, chạy sau Setup)

Chỉ dùng inner val, **không** đánh giá test. Cell ghi log vào `MyDrive/Sym/logs/` và in tóm tắt mỗi epoch, qua hàm `run_logged` định nghĩa trong cell Setup.
`EPOCHS`, `BATCH`, `WORKERS` nằm trong config của run. Đổi chúng giữa chừng thì `--resume` sẽ từ chối.
Bị ngắt phiên: chạy lại Setup rồi chạy lại cell này. Cell tự resume từ `last.pt` và bỏ qua run đã xong.

```python
VARIANTS, FOLD, SEED = ['baseline', 'full'], 2, 0
EPOCHS, BATCH, WORKERS = 20, 32, 4
assert str(DATA).startswith('/content/'), 'Chưa có DISFA_img/DONE.json: chạy cell A cho xong trước (đọc thẳng Drive hay lỗi)'

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

## B2. Thử EMA + val dày + LR backbone — fold 2, seed 0 → `runs_aug2` (cần GPU, chạy sau Setup)

So với pilot (cell B), cell này giữ nguyên augmentation, split, số epoch, batch và variant. Chỉ thêm 3 thay đổi:
- `--ema-decay 0.999`: đánh giá, chọn và lưu trọng số EMA;
- `--val-per-epoch 4`: đánh giá val 4 lần mỗi epoch, chọn checkpoint trên mọi lần đánh giá;
- `--backbone-lr-mult 0.1`: backbone dùng lr 1e-5, head giữ 1e-4.

Ba thay đổi này đi chung một gói vệ sinh huấn luyện, không phải đóng góp của luận văn, nên thử gộp một lần.

**Code nằm ở nhánh `stage1-ema-valchecks` cho tới khi pilot xong.** Nhánh này sửa các file bị hash; merge vào `main` lúc pilot đang chạy thì pilot sẽ không resume được nữa.
- Muốn chạy song song với pilot: mở một phiên GPU **khác**, trong Setup đặt `BRANCH = 'stage1-ema-valchecks'`. Phiên pilot vẫn để `BRANCH = 'main'`.
- Sau khi pilot xong và nhánh đã merge: dùng `BRANCH = 'main'` như thường.

`EPOCHS`, `BATCH`, `WORKERS`, `EMA`, `VAL_PER_EPOCH`, `BB_LR` nằm trong config của run, phải giữ nguyên qua mọi lần resume.

```python
VARIANTS, FOLD, SEED = ['baseline', 'full'], 2, 0
EPOCHS, BATCH, WORKERS = 20, 32, 4
EMA, VAL_PER_EPOCH, BB_LR = 0.999, 4, 0.1
RUNS2 = SYM / 'runs_aug2'                  # aug + EMA + val dày + LR backbone; tách khỏi pilot runs_aug1
assert str(DATA).startswith('/content/'), 'Chưa có DISFA_img/DONE.json: chạy cell A cho xong trước'
assert '--ema-decay' in (CODE / 'train_contrastive.py').read_text(), \
    "Code chưa có EMA: trong Setup đặt BRANCH = 'stage1-ema-valchecks' (hoặc chờ nhánh được merge vào main)"

for v in VARIANTS:
    out = RUNS2 / v / f'fold{FOLD}' / f'seed{SEED}'
    hist = out / 'history.json'
    if hist.exists() and len(json.loads(hist.read_text())) >= EPOCHS:
        print(f'{v}: đã xong {EPOCHS} epoch, bỏ qua'); continue
    cmd = [sys.executable, '-u', 'train_contrastive.py', 'train', '--variant', v, '--seed', SEED,
           '--arc', 'resnet50', '--data-root', DATA, '--manifest', SPLITS / f'fold{FOLD}.json',
           '--output', out, '--epochs', EPOCHS, '--batch-size', BATCH, '--workers', WORKERS,
           '--ema-decay', EMA, '--val-per-epoch', VAL_PER_EPOCH, '--backbone-lr-mult', BB_LR]
    if (out / 'last.pt').exists():
        cmd += ['--resume', out / 'last.pt']
    run_logged(cmd, f'train_aug2_{v}_fold{FOLD}_seed{SEED}')
```

---

## C2. So `runs_aug2` với pilot `runs_aug1` và run cũ (fold 2, seed 0)

`runs_aug2` được chọn checkpoint trên 80 lần đánh giá, còn pilot chỉ có 20. Chọn max trên nhiều điểm hơn thì số val tự cao lên một chút. Vì vậy cell in **hai** con số cho `runs_aug2`:
- tốt nhất trên mọi lần đánh giá: đây là checkpoint thật sự được dùng;
- tốt nhất chỉ trên các lần đánh giá cuối epoch: dùng để so ngang hàng với pilot.

Quy tắc giữ gói thay đổi:
- Giữ nếu con số "cuối epoch" của `runs_aug2` **không thấp hơn** pilot.
- Chỉ gọi là "tốt hơn" khi hơn rõ, cỡ 0.5 điểm trở lên. Một fold và một seed không đủ để phân biệt mức chênh nhỏ hơn.

Cell này chỉ đọc file JSON, nên chạy được ở bất kỳ phiên nào, kể cả phiên CPU và không cần Setup. Pilot chưa xong cũng chạy được: cell in số epoch đã có và điểm tốt nhất tính tới lúc đó.

```python
import os, json
from pathlib import Path
from google.colab import drive
if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive')
SYM = Path('/content/drive/MyDrive/Sym')
AU = [1, 2, 4, 6, 9, 12, 25, 26]
def points(h, end_only=False):
    return [(c['progress'], c['val']) for r in h
            for c in ([dict(progress=r['epoch'], val=r['val'])] if end_only or 'val_checks' not in r else r['val_checks'])]
for v in ['baseline', 'full']:
    print(f'== {v} ==')
    for name in ('runs_retry1', 'runs_aug1', 'runs_aug2'):
        p = SYM / name / v / 'fold2' / 'seed0' / 'history.json'
        if not p.exists():
            print(f'  {name:<12} chưa có'); continue
        h = json.loads(p.read_text())
        pa, va = max(points(h), key=lambda x: x[1]['macro_f1'])
        pe, ve = max(points(h, end_only=True), key=lambda x: x[1]['macro_f1'])
        print(f"  {name:<12} {len(h):>2} epoch | tốt nhất {va['macro_f1']:.3f} @ {pa:g}"
              f" | chỉ cuối epoch {ve['macro_f1']:.3f} @ {pe:g} | wa cuối {h[-1]['train']['wa']:.4f}")
        print('               F1/AU: ' + ' '.join(f'AU{a}:{x:.2f}' for a, x in zip(AU, va['per_au_f1'])))
        if name == 'runs_aug2':
            print('               val theo lần đánh giá: ' + ' '.join(f"{x['macro_f1']:.3f}" for _, x in points(h)))
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

## E. Chạy thử Tầng 2 trên 6 run cũ — CHỈ VAL, CPU, chạy song song với pilot được

Mục đích: kiểm module `stage2_decoder` chạy đúng trên dữ liệu thật (đường dẫn, file dự đoán, manifest) mà **không** phải chờ pilot. Đây **không** phải số báo cáo:
- detector cũ quá khớp, và test của nó đã bị mở;
- λ và ngưỡng được chọn trên chính val này, nên `R0 ≥ C0` trên val gần như do cách chọn (λ = 0 nằm trong lưới), không phải bằng chứng;
- val fold 1 của split cũ chỉ có 4 frame AU2, nên F1 của AU2 và ngưỡng C1 ở fold 1 không có nghĩa.

Cách chạy: mở một phiên Colab **CPU** riêng. Chạy cell Setup với `IMAGES = False` để khỏi giải nén ảnh, rồi chạy cell này.
- Cell chỉ **đọc** `runs_retry1` (đang đóng băng) và manifest cũ `splits/`. Kết quả ghi vào `MyDrive/Sym/stage2_smoke_retry1/`.
- Cell không bao giờ gọi `evaluate`. Nếu run nào thiếu `val_predictions.npz`, cell dừng trước khi chạy bất cứ gì.
- Chạy lại an toàn: bước nào đã có kết quả thì bỏ qua. Dự kiến vài phút.

```python
# ==== Cell E: chạy thử Tầng 2 trên 6 run cũ (runs_retry1), CHỈ VAL, CPU — kiểm module, không lấy số báo cáo ====
import hashlib
OLD_RUNS   = SYM / 'runs_retry1'            # ĐÓNG BĂNG: cell này chỉ đọc, không evaluate lại
OLD_SPLITS = SYM / 'splits'                 # manifest cũ mà các run này đã train
LISTS      = SYM / 'DISFA'                  # Tầng 2 chỉ đọc list + nhãn, không đọc ảnh
SMOKE      = SYM / 'stage2_smoke_retry1'    # thư mục riêng, không lẫn với stage2_v2
VARIANTS, FOLDS, SEED = ['baseline', 'full'], [1, 2, 3], 0
ARMS = ['C0', 'C1', 'R0', 'R1', 'R_FACS']

# 1) Kiểm trước: mỗi run phải có sẵn dự đoán val và đúng manifest cũ. Thiếu thì dừng, KHÔNG evaluate.
problems = []
for v in VARIANTS:
    for k in FOLDS:
        run = OLD_RUNS / v / f'fold{k}' / f'seed{SEED}'
        pdir = next((run / d for d in ('evaluation_val', 'evaluation') if (run / d / 'val_predictions.npz').exists()), None)
        if pdir is None or not (pdir / 'val_metrics.json').exists():
            problems.append(f'{v}/fold{k}: thiếu val_predictions.npz hoặc val_metrics.json')
        elif json.loads((run / 'config.json').read_text())['manifest_hash'] != \
                hashlib.sha256((OLD_SPLITS / f'fold{k}.json').read_bytes()).hexdigest():
            problems.append(f'{v}/fold{k}: run được train với manifest khác splits/fold{k}.json')
assert not problems, 'Dừng, chưa chạy gì:\n  ' + '\n  '.join(problems)
print(f'Đủ dự đoán val cho {len(VARIANTS) * len(FOLDS)} run cũ.', flush=True)

# 2) Rút luật từ nhãn fit của splits cũ (vài giây mỗi fold)
for k in FOLDS:
    if not (SMOKE / 'rules' / f'fold{k}' / 'rules.json').exists():
        run_logged([sys.executable, '-u', '-m', 'stage2_decoder.build_rules', '--data-root', LISTS,
                    '--manifest', OLD_SPLITS / f'fold{k}.json', '--output', SMOKE / 'rules'], f'smoke_rules_fold{k}')

# 3) Giải mã chỉ trên val (decode chỉ đọc run, ghi vào SMOKE)
for v in VARIANTS:
    for k in FOLDS:
        if (SMOKE / 'results' / v / f'fold{k}' / f'seed{SEED}' / 'decode_val.json').exists():
            continue
        run_logged([sys.executable, '-u', '-m', 'stage2_decoder.decode',
                    '--run', OLD_RUNS / v / f'fold{k}' / f'seed{SEED}', '--rules', SMOKE / 'rules' / f'fold{k}',
                    '--manifest', OLD_SPLITS / f'fold{k}.json', '--data-root', LISTS,
                    '--output', SMOKE / 'results', '--val-only'], f'smoke_decode_{v}_fold{k}')

# 4) Tóm tắt
print('\nVAL macro-F1 (%). λ và ngưỡng chọn trên CHÍNH val này → số lạc quan, chỉ để kiểm module.')
print(f"{'run':<15}{'ep':>4}{'luật':>6}" + ''.join(f'{m:>8}' for m in ARMS) + f"{'λ R0':>7}{'λ FACS':>8}{'R0−C0':>8}")
for v in VARIANTS:
    for k in FOLDS:
        d = json.loads((SMOKE / 'results' / v / f'fold{k}' / f'seed{SEED}' / 'decode_val.json').read_text())
        M = d['methods']
        print(f'{v}/fold{k}'.ljust(15) + f"{str(d['selected_epoch']):>4}{d['n_rules']:>6}"
              + ''.join(f"{100 * M[m]['val_f1']:>8.2f}" for m in ARMS)
              + f"{M['R0']['lambda_']:>7g}{M['R_FACS']['lambda_']:>8g}"
              + f"{100 * (M['R0']['val_f1'] - M['C0']['val_f1']):>+8.2f}")
print(f'Chi tiết: {SMOKE}/results/<variant>/fold<k>/seed0/decode_val.json | luật: {SMOKE}/rules/fold<k>/rules_all.csv')
```

---

## F. Báo cáo log — gom trạng thái thành 1 file để gửi Claude (CPU, không cần Setup/token)

Có sẵn dạng notebook: [notebooks/Bao_cao_log.ipynb](notebooks/Bao_cao_log.ipynb).
- Ghi `MyDrive/Sym/reports/status_<ngày>_<giờ>.md` và tự tải về máy.
- Nội dung: tóm tắt mọi run, bảng theo epoch của `runs_aug*`, Tầng 2 chỉ val, đuôi 8 log mới nhất.
- Không in số test. Nếu log có token, cell tự che đi.

```python
# ==== Cell F: gom trạng thái run + Tầng 2 + đuôi log thành 1 file .md để gửi Claude (chỉ đọc, không cần GPU/token) ====
import os, re, json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from google.colab import drive, files

if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive')
SYM    = Path('/content/drive/MyDrive/Sym')
AU     = [1, 2, 4, 6, 9, 12, 25, 26]
VN     = timezone(timedelta(hours=7))
NOW    = datetime.now(VN)
DETAIL = ('runs_aug1', 'runs_aug2', 'runs_aug3')   # in bảng theo epoch; thư mục khác chỉ in tóm tắt
N_LOGS, TAIL = 8, 40                               # số log mới nhất được in đuôi, số dòng cuối mỗi log
FENCE = '`' * 3                                    # viết gián tiếp để cell nhúng được trong khối code markdown
out = []
w = out.append

def rd(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None

def age(p):
    m = datetime.fromtimestamp(p.stat().st_mtime, VN)
    return f'{m:%d/%m %H:%M} ({(NOW - m).total_seconds() / 60:.0f} phút trước)'

def points(h, end_only=False):
    return [(c['progress'], c['val']) for r in h
            for c in ([dict(progress=r['epoch'], val=r['val'])] if end_only or 'val_checks' not in r else r['val_checks'])]

def short_line(line):
    """Dòng JSON cuối epoch rất dài (tp/fp/fn, val_checks) → rút gọn; bỏ dòng tiến độ step."""
    if line.startswith('{"epoch"'):
        try:
            r = json.loads(line)
            return f"ep{r['epoch']} wa={r['train']['wa']:.4f} valF1={r['val']['macro_f1']:.4f} {r['seconds'] / 60:.1f} phút"
        except Exception:
            return line[:200]
    if re.match(r'epoch \d+/\d+ step \d+/\d+', line):
        return None
    return line[:300]

w(f'# Báo cáo trạng thái Thesis_Sym — {NOW:%d/%m/%Y %H:%M} (giờ VN)\n')
w('## Dữ liệu')
w(f"- DISFA_img/DONE.json: {'có' if (SYM / 'DISFA_img' / 'DONE.json').exists() else 'CHƯA'}")
w(f"- splits_v2: {sorted(p.name for p in (SYM / 'splits_v2').glob('*.json'))}")

w('\n## Run Tầng 1')
for root in sorted(p for p in SYM.glob('runs_*') if p.is_dir()):
    runs = sorted(c.parent for c in root.glob('*/fold*/seed*/config.json'))
    if not runs:
        continue
    w(f'\n### {root.name} ({len(runs)} run)')
    for run in runs:
        cfg, h = rd(run / 'config.json') or {}, rd(run / 'history.json') or []
        src = cfg.get('source_hashes', {})
        w(f"\n**{'/'.join(run.parts[-3:])}** — {len(h)}/{cfg.get('epochs')} epoch"
          + (f" | last.pt sửa {age(run / 'last.pt')}" if (run / 'last.pt').exists() else ' | chưa có last.pt'))
        w(f"- config: batch {cfg.get('batch_size')}, workers {cfg.get('workers')}, lr {cfg.get('lr')}, "
          f"ema {cfg.get('ema_decay', 0)}, val/epoch {cfg.get('val_per_epoch', 1)}, bb_lr×{cfg.get('backbone_lr_mult', 1)}, "
          f"torch {cfg.get('environment_versions', {}).get('torch')}, "
          f"hash train_contrastive {src.get('train_contrastive.py', '')[:8]} / experiment_data {src.get('experiment_data.py', '')[:8]}")
        if not h:
            continue
        pa, va = max(points(h), key=lambda x: x[1]['macro_f1'])
        pe, ve = max(points(h, True), key=lambda x: x[1]['macro_f1'])
        w(f"- val tốt nhất {va['macro_f1']:.4f} @ {pa:g} | chỉ cuối epoch {ve['macro_f1']:.4f} @ {pe:g} | "
          f"epoch cuối {h[-1]['val']['macro_f1']:.4f} | wa cuối {h[-1]['train']['wa']:.4f}")
        w('- F1/AU tại best: ' + ' '.join(f'AU{a}:{x:.3f}' for a, x in zip(AU, va['per_au_f1'])))
        ev = [f'{d}/{f}' for d in ('evaluation_val', 'evaluation') for f in ('val_metrics.json', 'test_metrics.json')
              if (run / d / f).exists()]
        if ev:
            vm = next((rd(run / d / 'val_metrics.json') for d in ('evaluation_val', 'evaluation')
                       if (run / d / 'val_metrics.json').exists()), None)
            w(f"- evaluate có: {', '.join(ev)}"
              + (f" | val F1 {vm['macro_f1']:.4f} @ {vm.get('selected_progress', vm.get('selected_epoch'))}" if vm else '')
              + ' (không in số test)')
        if root.name in DETAIL:
            w('\n| ep | wa | we | global | node | cross | val cuối ep | val giữa ep | phút |')
            w('|---|---|---|---|---|---|---|---|---|')
            for r in h:
                t = r['train']
                mid = ' '.join(f"{c['val']['macro_f1']:.3f}" for c in r.get('val_checks', [])[:-1])
                w(f"| {r['epoch']} | {t['wa']:.4f} | {t['we']:.4f} | {t['global_con']:.3f} | {t['node_con']:.3f} | "
                  f"{t['cross_con']:.3f} | {r['val']['macro_f1']:.4f} | {mid} | {r['seconds'] / 60:.1f} |")

w('\n## Tầng 2 — chỉ val (decode_val.json)')
for name in sorted(p.name for p in SYM.glob('stage2*') if p.is_dir()):
    res = sorted((SYM / name / 'results').glob('*/fold*/seed*/decode_val.json'))
    n_test = len(list((SYM / name / 'results').glob('*/fold*/seed*/decode_summary.json')))
    if not res and not n_test:
        continue
    w(f'\n### {name}' + (f' — có {n_test} decode_summary.json (đã mở test, không in số)' if n_test else ''))
    if res:
        w('| run | luật | C0 | C1 | R0 | R1 | R_FACS | λ R0 | λ FACS |')
        w('|---|---|---|---|---|---|---|---|---|')
    for p in res:
        d = rd(p)
        if not d:
            continue
        M = d['methods']
        w(f"| {'/'.join(p.parts[-4:-1])} | {d['n_rules']} | "
          + ' | '.join(f"{100 * M[m]['val_f1']:.2f}" for m in ('C0', 'C1', 'R0', 'R1', 'R_FACS'))
          + f" | {M['R0']['lambda_']:g} | {M['R_FACS']['lambda_']:g} |")

w('\n## Log mới nhất (MyDrive/Sym/logs)')
logs = sorted((SYM / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
w('\n'.join(f'- {p.name} — {p.stat().st_size / 1e3:.0f} KB, sửa {age(p)}' for p in logs[:2 * N_LOGS]))
for p in logs[:N_LOGS]:
    kept = [s for s in map(short_line, p.read_text(errors='replace').splitlines()) if s]
    w(f'\n### {p.name} — {TAIL} dòng cuối (đã bỏ dòng tiến độ step)\n{FENCE}text')
    w('\n'.join(kept[-TAIL:]))
    w(FENCE)

text = '\n'.join(out) + '\n'
# Chặn rò token nếu lỡ có trong log (đã từng lộ một lần ngày 20/11)
text = re.sub(r'Authorization:\s*\S+(\s+\S+)?|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}', '[ĐÃ ẨN]', text)
name = f'status_{NOW:%Y%m%d_%H%M}.md'
(SYM / 'reports').mkdir(exist_ok=True)
(SYM / 'reports' / name).write_text(text)
Path('/content', name).write_text(text)
for line in out:
    if line.startswith(('\n**', '- val tốt nhất', '\n### runs')):
        print(line.strip())
print(f'\nĐã ghi {len(text) / 1e3:.0f} KB → MyDrive/Sym/reports/{name}; đang tải về máy...')
files.download(f'/content/{name}')
```

---

## Phụ lục

**Kéo log và kết quả về máy để phân tích** (bỏ ảnh, nhãn gốc, trọng số):

```python
import os, zipfile
from google.colab import files
ROOT, OUT = '/content/drive/MyDrive/Sym', '/content/Sym_source.zip'
SKIP_DIRS = {'img', 'DISFA_img', 'ActionUnit_Labels', 'source', '__pycache__', '.pytest_cache', 'tmp', 'checkpoints', '.ipynb_checkpoints'}
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
