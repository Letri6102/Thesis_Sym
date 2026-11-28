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
BRANCH   = 'main'                      # nhánh stage1-ema-valchecks đã merge vào main (21/11)
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

**Kết quả 22/11: trượt tiêu chí thứ hai** (đỉnh ở epoch 1). Không chỉnh augmentation theo val fold 2 rồi áp cho mọi fold: 3 subject này là test của fold 1/3 (CLAUDE.md §7).

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

**Code đã có trên `main` (merge ngày 21/11, `586b221`).** Dùng Setup bình thường với `BRANCH = 'main'`. `runs_aug2` đã chạy xong ngày 21/11 (kết quả ở CLAUDE.md §7).

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

**Chỉ để mô tả, không dùng để chọn cấu hình** (đính chính 22/11):
- mốc "hơn 0.5 điểm" trước đây là sai: riêng pilot `baseline` nhảy tới 10 điểm giữa hai epoch liền nhau;
- EMA làm mượt đường val, nên so max với run không EMA là thiệt cho EMA;
- val fold 2 gồm subject test của fold 1/3.

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

**Không đụng outer test.** Kết quả ghi vào `MyDrive/Sym/stage2_v2/`. Chạy lại an toàn: bước nào đã có kết quả thì bỏ qua. Riêng kết quả decode làm từ checkpoint khác thì không được giữ, chẳng hạn kết quả của pilot Colab trước khi `runs_aug1` được thay bằng run 5090 (23/11). Kết quả đó bị chuyển sang `stage2_v2/stale/` rồi decode lại.
Lưu ý: điểm val ở đây có λ và ngưỡng được chọn trên chính val, nên chỉ dùng để xem chiều hướng, **không** dùng để báo cáo.

```python
VARIANTS, FOLDS, SEEDS = ['baseline', 'full'], [2], [0]     # pilot: fold 2
STAGE2 = SYM / 'stage2_v2'
run_logged([sys.executable, '-u', '-m', 'stage2_decoder.run_all', '--runs', RUNS, '--splits', SPLITS,
            '--data-root', DATA, '--output', STAGE2, '--variants', *VARIANTS,
            '--folds', *FOLDS, '--seeds', *SEEDS, '--val-only'], 'stage2_valonly')
```

Đủ 18 run thì chạy cell này với `FOLDS = [1, 2, 3]`, `SEEDS = [0, 1, 2]`, **vẫn giữ** `--val-only`.

**Không mở test bằng cách sửa cell này** (đính chính 22/11: hướng dẫn cũ để `SEEDS = [0]`, sẽ mở test khi mới có 1 seed). Cell mở test sẽ được viết riêng, sau khi đủ ba điều kiện ở CLAUDE.md §7 mục 8.

---

## D2. Chẩn đoán `runs_aug2`: AU6/AU26 tụt do ngưỡng hay do mất đặc trưng? (GPU khoảng 10 phút, chạy sau Setup)

Ở B2, val tốt nhất mất khoảng 3 điểm so với run cũ, và phần lớn đến từ một số AU:

| AU (`baseline` / `full`) | B2 | run cũ | Chênh |
|---|---|---|---|
| AU6 | 0.42 / 0.42 | 0.64 / 0.62 | khoảng −20 |
| AU26 | 0.43 / 0.47 | 0.55 / 0.50 | −11 / −3 |
| AU4 | 0.76 / 0.74 | 0.68 / 0.66 | +8 |
| AU12 | 0.83 / 0.78 | 0.73 / 0.73 | +10 / +5 |

Cell này `evaluate` val cho 2 run, rồi decode **chỉ trên val**. C1 là F1 sau khi chỉnh ngưỡng riêng cho từng AU.
**Kết quả 22/11:** AU6 ở C1 là 42.2 / 45.5, tức không cứu được bằng một ngưỡng chung. Điều này **chưa** chứng minh mất đặc trưng do augmentation: cũng có thể do lệch giữa subject. Cột "run cũ" ở bảng trên là epoch đẹp nhất, nên mức chênh bị phóng đại.

Mốc so sánh từ run cũ `baseline` fold 2 (cell E): C0 71.5 → C1 73.0; AU6 63.8 → 66.9 (ngưỡng 0.38); AU26 54.6 → 60.6.

Kết quả ghi vào `MyDrive/Sym/stage2_aug2/`, tách khỏi `stage2_v2`: hai nơi dùng chung đường dẫn `results/<variant>/fold2/seed0/`, để chung sẽ đè nhau. Cell không đụng outer test.

```python
RUNS2, OUT2 = SYM / 'runs_aug2', SYM / 'stage2_aug2'
assert str(DATA).startswith('/content/'), 'evaluate cần ảnh trong /content: chạy Setup với IMAGES = True'
run_logged([sys.executable, '-u', '-m', 'stage2_decoder.run_all', '--runs', RUNS2, '--splits', SPLITS,
            '--data-root', DATA, '--output', OUT2, '--variants', 'baseline', 'full',
            '--folds', 2, '--seeds', 0, '--val-only'], 'stage2_valonly_aug2')
AU = [1, 2, 4, 6, 9, 12, 25, 26]
for v in ('baseline', 'full'):
    d = json.loads((OUT2 / 'results' / v / 'fold2' / 'seed0' / 'decode_val.json').read_text())
    M = d['methods']
    print(f"\n{v}: C0 {100 * M['C0']['val_f1']:.2f} → C1 {100 * M['C1']['val_f1']:.2f} (ngưỡng chỉnh trên val)"
          f" | R0 {100 * M['R0']['val_f1']:.2f} (λ {M['R0']['lambda_']:g})")
    print('  ngưỡng C1: ' + ' '.join(f'AU{a}:{t:.2f}' for a, t in zip(AU, d['thresholds_c1'])))
    print('  F1 C0    : ' + ' '.join(f'AU{a}:{100 * x:.1f}' for a, x in zip(AU, M['C0']['val_f1_au'])))
    print('  F1 C1    : ' + ' '.join(f'AU{a}:{100 * x:.1f}' for a, x in zip(AU, M['C1']['val_f1_au'])))
```

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
- Nội dung: tóm tắt mọi run, bảng theo epoch của `runs_aug*`, Tầng 2 chỉ val (kèm F1 và ngưỡng từng AU, đủ để đọc D2), đuôi 8 log mới nhất.
- Không in số test: log có kết quả test thì không in đuôi. Nếu log có token, cell tự che đi.

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
# log có kết quả test thì không in đuôi (evaluate / decode / summarize / check_stage1)
TEST_LOG = re.compile(r'"split":\s*"test"|--split\W+test|TEST MACRO-F1|, test \d+ frame|test F1')
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
    if res:
        w('\n| run (val, theo AU) | hàng | ' + ' | '.join(f'AU{a}' for a in AU) + ' |')
        w('|---' * (len(AU) + 2) + '|')
    for p in res:
        d = rd(p)
        if not d:
            continue
        tag = '/'.join(p.parts[-4:-1])
        for m in ('C0', 'C1', 'R0'):
            w(f'| {tag} | F1 {m} | ' + ' | '.join(f'{100 * x:.1f}' for x in d['methods'][m]['val_f1_au']) + ' |')
        w(f'| {tag} | ngưỡng C1 | ' + ' | '.join(f'{t:.2f}' for t in d['thresholds_c1']) + ' |')

w('\n## Log mới nhất (MyDrive/Sym/logs)')
logs = sorted((SYM / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
w('\n'.join(f'- {p.name} — {p.stat().st_size / 1e3:.0f} KB, sửa {age(p)}' for p in logs[:2 * N_LOGS]))
for p in logs[:N_LOGS]:
    raw = p.read_text(errors='replace')
    if TEST_LOG.search(raw):
        w(f'\n### {p.name} — không in đuôi (log có kết quả test)')
        continue
    kept = [s for s in map(short_line, raw.splitlines()) if s]
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

## G2. Train trên Colab bằng runner chung với máy 5090 — GPU, chạy sau Setup (`IMAGES = True`)

Chủ dự án đổi 26/11: Colab và máy 5090 **train xen kẽ**, bàn giao cả run dở. Có sẵn dạng notebook: [notebooks/Train_Colab.ipynb](notebooks/Train_Colab.ipynb) (Setup → I nhận từ 5090 → G2 train → M bàn giao cho 5090). Cell G cũ không dùng nữa.
- Chạy `pc/train_18_run.py` với `--out MyDrive/Sym`: cùng khoá, cùng cách resume, cùng kiểm cặp như máy 5090. Run dở do máy 5090 train được resume ở đây.
- **Quy tắc cặp:** run mới chỉ được tạo khi run cùng cặp (nếu đã có) có cùng hash khởi tạo với máy này (runner tính thử trước, khoảng 10 giây). Không khớp thì run đó để lại cho máy kia.
- Run đã bàn giao (`HANDED_OFF.json`) không train tiếp, trừ khi `RECLAIM = True` (chỉ khi máy 5090 chưa train tiếp run đó).
- Mỗi run khoảng 2,5 giờ trên L4. Phiên bị ngắt thì chạy lại cell: run dở resume. Log vừa được ghi dưới 15 phút thì runner dừng (có thể phiên khác đang train), trừ khi `OLD_SESSION_DEAD = True`.

```python
# ==== Cell G2: train bộ 18 run trên Colab bằng đúng runner của máy 5090 (26/11). Drive là bản chính. KHÔNG đụng test ====
if not os.path.isdir('/content/drive/MyDrive'):            # cell I/M tháo Drive ở cuối: mount lại
    drive.mount('/content/drive')
assert str(DATA).startswith('/content/'), 'Ảnh chưa nằm trong /content: chạy Setup với IMAGES = True'
shutil.copytree(SPLITS, DATA / 'splits_v2', dirs_exist_ok=True)       # runner đọc manifest ở <DATA>/splits_v2 (có kiểm hash khoá)
SEEDS, FOLDS = [0, 1, 2], [1, 2, 3]
REFIT = True                     # xong (a) thì refit luôn những run (a) đã xong
OLD_SESSION_DEAD = False         # chỉ đặt True khi chắc chắn phiên Colab trước đã tắt hẳn
RECLAIM = False                  # chỉ đặt True khi lấy lại run đã bàn giao mà máy 5090 CHƯA train tiếp
args = ['--data', DATA, '--out', SYM, '--seeds', *SEEDS, '--folds', *FOLDS, '--no-pack']
args += ['--old-session-dead'] * OLD_SESSION_DEAD + ['--reclaim'] * RECLAIM
runner = [sys.executable, '-u', REPO / 'pc' / 'train_18_run.py']
run_logged(runner + ['train', *args], 'colab_train')
if REFIT:
    run_logged(runner + ['refit', *args], 'colab_refit')
run_logged(runner + ['progress', '--out', SYM], 'colab_progress')
```

## M. Bàn giao cho máy 5090 — CPU hoặc GPU, chạy sau Setup

Dừng cell G2 trước. Cell nén vào `MyDrive/Sym/handoff/`:
- `partial_aug1_*.zip` / `partial_refit1_*.zip`: run dở, đủ file để máy 5090 train tiếp (khoảng 0,7 GB mỗi file, dưới giới hạn 1 GB của UltraViewer). Run này được đánh dấu `HANDED_OFF` trên Drive.
- `stubs_*.zip`: bản tóm tắt của run đã xong (config, history, init, không có trọng số), để máy 5090 bỏ qua chúng và biết E khi refit.

Tải các file đó về máy cá nhân, chép vào `D:\LVTS_TriLe\nhan` trên máy 5090, rồi chạy `train_data_co_san.bat` (tự nhận trước khi train).

```python
# ==== Cell M: bàn giao cho máy 5090 — zip run dở (partial_...) + bản tóm tắt run đã xong (stubs_...) vào MyDrive/Sym/handoff ====
if not os.path.isdir('/content/drive/MyDrive'):            # cell I/M tháo Drive ở cuối: mount lại
    drive.mount('/content/drive')
# Dừng cell G2 trước. Run dở được đánh dấu HANDED_OFF trên Drive: Colab không train tiếp cho tới khi nhận bản mới về (cell I).
STOPPED_HERE = False             # True: vừa tự dừng cell G2 trong phiên này, không cần chờ 15 phút
sys.path.insert(0, str(REPO / 'pc'))
from stage2_decoder.contrast_ci import LOCK
import train_18_run as PC
OUT_DIR = SYM / 'handoff'
for old in OUT_DIR.glob('*.zip'):                         # bản bàn giao cũ chưa dùng: cất đi, tránh nhận nhầm
    (OUT_DIR / 'cu').mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(OUT_DIR / 'cu' / old.name))
PC.send(out=SYM, dest=OUT_DIR, complete='stub', lock=LOCK, work=Path('/content/handoff_work'),
        old_session_dead=STOPPED_HERE, to='máy 5090')
print('Tải các file trên trong MyDrive/Sym/handoff về máy cá nhân, chép vào D:\\LVTS_TriLe\\nhan trên máy 5090,'
      ' rồi chạy train_data_co_san.bat (tự nhận trước khi train).', flush=True)
drive.flush_and_unmount()
```

---

## H. Kiểm đủ 18 run đúng khoá — CPU, KHÔNG đọc test

Chạy Setup với `IMAGES = False`, rồi chạy cell này. Cell chỉ đọc `config.json`, `history.json` và kiểm `best.pt` có tồn tại. Chưa đủ thì cell liệt kê run thiếu hoặc lệch rồi báo lỗi.

Lệnh tính CI thật (`--output`) **không** nằm ở đây: nó thuộc cell mở test, viết sau khi xong các điều kiện ở CLAUDE.md §7 mục 8.

```python
run_logged([sys.executable, '-u', '-m', 'stage2_decoder.contrast_ci', '--runs', RUNS, '--check-only'], 'contrast_ci_check')
```

---

## I. Nhận run từ máy 5090 — CPU, chạy sau Setup (`IMAGES = False` là đủ)

Có sẵn dạng notebook kèm Setup: [notebooks/Nhap_run_PC.ipynb](notebooks/Nhap_run_PC.ipynb); cũng nằm trong `Train_Colab.ipynb` và `Tong_hop_val.ipynb`. Logic ở `PC.receive` (`pc/train_18_run.py`), chung với lệnh `receive` của máy 5090.
- Nhận `aug1_*` / `refit1_*` (run xong), `partial_aug1_*` / `partial_refit1_*` (run dở, 26/11) trong `MyDrive/Sym/upload/`.
- Kiểm khoá (config, hash code, thư viện, manifest). Bản nhận **chỉ thay** bản trên Drive khi nó là phần nối tiếp: lịch sử epoch của bản trên Drive là phần đầu của bản nhận. Bản bị thay được cất vào `failed_runs/…_truoc_nhap`, không xoá.
- **Không nhập:** bản lệch nhánh (hai nơi cùng train một run), bản có hash khởi tạo khác run cùng cặp, bản lệch khoá. Run mà log vừa được ghi dưới 15 phút (Colab đang train) thì để lại, chạy lại sau. Refit chỉ nhập khi run (a) tương ứng đã có và E trùng.
- Zip đã xử lý chuyển sang `upload/done/`, `upload/duplicate/` hoặc `upload/rejected/`. Chạy lại cell là xử lý tiếp phần còn lại.
- **Không còn chặn trộn máy** (chủ dự án đổi 26/11): run từ L4 và 5090 cùng nằm trong `runs_aug1`; máy train từng epoch ghi ở `machines.json` của run.

```python
# ==== Cell I: nhận run từ máy 5090 (MyDrive/Sym/upload/*.zip): run xong và run dở — kiểm khoá, dòng lịch sử, cặp ====
sys.path.insert(0, str(REPO / 'pc'))
from stage2_decoder.contrast_ci import LOCK
import train_18_run as PC
PC.receive(SYM / 'upload', out=SYM, splits=SPLITS, data=DATA, lock=LOCK, work=Path('/content/pc_import'))
for kind, root in PC.KIND_DIR.items():
    n = sum(PC._done(kind, SYM / root / v / f'fold{k}' / f'seed{s}', LOCK['common']['epochs']) for v, k, s in PC.jobs())
    print(f'{root}: {n}/18 run xong', flush=True)
print('Đang chờ Drive đồng bộ...', flush=True)
drive.flush_and_unmount()
```

---

## J. Đối chiếu ảnh của gói `LVTS_TriLe` với Drive — CPU, khoảng 10 phút, không cần Setup

Gói chép sang máy GPU riêng (`pc/make_bundle.py`) lấy ảnh từ `img/` ở máy cá nhân, không lấy từ Drive. Tar trong gói được dựng **đúng định dạng cell A**, nên ảnh giống hệt thì tar cũng trùng từng byte. Cell này so SHA-256 của 27 tar trên Drive với số của gói.
**Chỉ được train trên máy GPU khi cell báo `KHỚP 27/27`.** Nếu khác: tải `DISFA_img/` từ Drive, thay thư mục `sym_download/DISFA_img` trong gói, rồi chạy lại `1_cai_dat.bat`.

```python
# ==== Cell J: đối chiếu 27 tar của gói LVTS_TriLe (dựng từ img/ ở máy cá nhân) với DISFA_img trên Drive — CPU, không cần Setup ====
import os, json, hashlib, subprocess
from pathlib import Path
from google.colab import drive

if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive')
SRC, TMP = Path('/content/drive/MyDrive/Sym/DISFA_img'), Path('/content/tar_check')
TMP.mkdir(exist_ok=True)
EXPECT = {                                  # pc/bundle_tar_sha256.json, gói dựng từ code 4e610e2
    'SN001': (338176000, '393a2000c5f22dcb42d78d9b25a2e585f03ed1a13c676a287426331f0f353c93'),
    'SN002': (369029120, 'e8a01a3d749473be3ff95b55b05a88ec697bb4be651ef6ebd96ee2bdbf40fb16'),
    'SN003': (340551680, 'b8c52330619371cdf378090667309791b792aa7260580c77b3d2853c27a82f3a'),
    'SN004': (347494400, 'cda7a8a30e0860a288f6512ea2f116c14f71337479e8bc14369b53f562bf3191'),
    'SN005': (310538240, 'e523ed014ea8d5571f5598b1848f2f54f3c13823cd3e96f9ae1a04c97daaddbd'),
    'SN006': (313845760, 'e5b98bc6dc59eb80f259cc29bf4ac4a1189f9de52a92a168cba03eb701905ff2'),
    'SN007': (302182400, '45b100926ab2bf42850cc8846b2d32e3fafb9446bb543caad6ef6cb2bd1de7c4'),
    'SN008': (316416000, '9ce9036866b3a1699d912b99fb2527db616e52dfc4b280468623a89852cb4168'),
    'SN009': (323205120, 'aaca676e856301459ddc438c67ba215e621df72d16044d04e5a13ea6a2863d72'),
    'SN010': (320337920, '7d6489c3e27abf2af1892062a708d46571a16290bd740e5097be4d3a641d32a3'),
    'SN011': (330147840, '7fd49d6fed1cbba84d366d78dd16bbabe2da29085963f33decdfb50c521fd019'),
    'SN012': (370083840, '4995de042323479ea59863c53536c18c6d60e8706b897ce02ace7aefc8ab4072'),
    'SN013': (306667520, '3f879b0edd53c6a1597595d46f96c9a90c5206cf67908dca06eb43411c01bd0e'),
    'SN016': (379463680, 'd5137b37fbf4697f3311282e133571edc227bd8eaddb1898034b3d405df8e0c2'),
    'SN017': (324300800, '46a946bbe3a79ac35cee0993a96e2815f54050d10f413c69151d1b8edc35dbfa'),
    'SN018': (358594560, '21cbc64024545cfa25481d40adb4081481de230926e23d3df27d31dd0b998a0a'),
    'SN021': (296212480, 'b6aefa68d5aebf2358dad6838221b4bb7f12e27f9b097813ec1a44aa0b518484'),
    'SN023': (292945920, '13403558a95a7233506f80acd9f758010493a2b0b727fc629d57d640a29e2027'),
    'SN024': (315494400, '773b8c9390d7c58b00f94c650e895061f7e5ced2649d2f5c797c0172e99dc3fb'),
    'SN025': (307333120, 'b62af11e89c14e4cea1a8a7a4d3b32987fa347f9b5428e3719e39d9820f91e7f'),
    'SN026': (301824000, 'b8ad68fa146c0292f80ccfb15952c0bb41b494a3043329be3ed59f19b2d3a18a'),
    'SN027': (290570240, '270d34064cdae28a227755f81f5fbb72f95049218e680429d5c119d13e65c70e'),
    'SN028': (309800960, 'b049cae37e948e47d8aa47dadb9f664c6a08ccd72cc8ae7d1877967c9a372beb'),
    'SN029': (304179200, '5db9d64ff0eb49ae84e2adb9cba3a5764a76fce053c9f86389e894e27c52cb18'),
    'SN030': (266752000, '31617046c885f52da5a4d73f88665c5ecf106417002dc8abb8408ce24fce92ec'),
    'SN031': (313640960, '796902bef88cee76bcfd20e428d740882dea267ca6823794cf1a7375ed4a07dc'),
    'SN032': (302254080, '7743fe12496dcf9a7a18f1a5005fb954ce7855fd4811789262fac41ddaaee215'),
}

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

assert sorted(json.loads((SRC / 'DONE.json').read_text())['subjects']) == sorted(EXPECT), 'Danh sách subject khác'
bad = []
for i, (s, (size, want)) in enumerate(sorted(EXPECT.items()), 1):
    if json.loads((SRC / f'{s}.json').read_text())['size'] != size:
        bad.append(s); print(f'{i:>2}/27 {s}: KHÁC (kích thước)', flush=True); continue
    local = TMP / f'{s}.tar'
    subprocess.run(['cp', str(SRC / f'{s}.tar'), str(local)], check=True, timeout=1800)   # đọc Drive có timeout
    got = sha256(local)
    local.unlink()
    print(f'{i:>2}/27 {s}: ' + ('KHỚP' if got == want else 'KHÁC'), flush=True)
    if got != want:
        bad.append(s)
print('\nKẾT QUẢ: ' + ('KHỚP 27/27: ảnh trong gói đúng bản trên Drive, máy GPU được train.' if not bad
                       else f'KHÁC ở {bad}: KHÔNG train bằng gói này, báo Claude.'))
```

---

## K. Dòng (b) refit — GPU, chạy sau Setup (`IMAGES = True`)

Chủ dự án chốt ngày 23/11. Mỗi run (a) đã xong được train lại trên **đủ 18 subject** outer-train (fit + inner val), với đúng công thức đã khoá, và dừng sau epoch E mà run (a) chọn trên inner val. Model refit là `last.pt`. Kết quả ghi vào `MyDrive/Sym/runs_refit1/`. Code ở `Contrastive_V1/refit/`, không đụng file bị hash.
- Run (a) chưa xong thì bỏ qua, nên chạy lúc nào cũng được. Chạy lại cell là làm tiếp.
- Mỗi run mất khoảng E × 10 phút trên L4. **Không chạy nữa (23/11):** máy 5090 tự refit cả 18 run (`train_data_co_san.bat` chạy refit ngay sau train).
- Không evaluate test. Test của dòng (b) mở cùng lần với dòng (a).

```python
# ==== Cell K: dòng (b) refit — train lại trên đủ 18 subject, dừng ở epoch E của run (a). Model refit = last.pt ====
SEEDS = [0, 1]                 # cùng phân công với cell G; máy GPU riêng tự refit seed của nó
OLD_SESSION_DEAD = False       # chỉ đặt True khi đã tự kiểm phiên cũ tắt hẳn

assert torch.cuda.is_available(), 'Cần runtime GPU'
assert str(DATA).startswith('/content/'), 'Ảnh chưa nằm trong /content: chạy Setup với IMAGES = True'
assert RUNS.name == 'runs_aug1', f'RUNS phải là runs_aug1, đang là {RUNS}'
sys.path.insert(0, str(CODE))
from refit import run as R

REFIT = SYM / 'runs_refit1'
todo, skipped = R.jobs(RUNS, REFIT, SPLITS, DATA, seeds=SEEDS)
print(f'Refit: {len(todo)} run cần làm, khoảng {sum(j["E"] for j in todo) * 10 / 60:.1f} giờ L4', flush=True)
for x in skipped:
    print('  bỏ qua', x)
for i, j in enumerate(todo, 1):
    name = f"refit1_{j['variant']}_fold{j['fold']}_seed{j['seed']}"
    print(f'\n[{i}/{len(todo)}] {name} | dừng sau epoch {j["E"]}', flush=True)
    moved = R.clear_stale(j, SYM / 'failed_runs', 15, OLD_SESSION_DEAD)
    if moved:
        print(f'  lần trước bị ngắt trong epoch 1 → cất sang failed_runs/{moved.name}')
    run_logged(R.one_cmd(j, DATA), name)
    R.finish(j)
print('\nXONG refit của phiên này. Đang chờ Drive đồng bộ...', flush=True)
drive.flush_and_unmount()
```

---

## L. Chạy tổng hợp CHỈ VAL — Tầng 1 → Tầng 2 → một bảng (GPU, sau Setup với `IMAGES = True`)

Có sẵn dạng notebook một file: [notebooks/Tong_hop_val.ipynb](notebooks/Tong_hop_val.ipynb) (Setup → cell I nhập zip mới → cell L). Chủ dự án chốt 23/11: kết quả tổng thể đo **chỉ trên inner val**; test vẫn mở một lần sau khi đủ 18 run.
- Không train thêm: Tầng 1 là các run đã có trong `runs_aug1`, train trên máy 5090.
- Với mỗi run đủ 20 epoch: `evaluate --split val` bằng `best.pt` (bỏ qua nếu đã có), rồi Tầng 2 chỉ val (C0, C1, R0, R1, R_FACS). Run chưa có thì bỏ qua, nên chạy lại cell khi có thêm run.
- Bảng in ra và ghi vào `MyDrive/Sym/stage2_v2/val_overview.json` gồm: từng run, trung bình theo variant, `R0 − C0`, `full − baseline` theo cặp cùng fold và seed, F1 từng AU.
- λ và ngưỡng C1 được chọn trên chính val, nên các số này chỉ để xem chiều hướng: **không báo cáo, không so với 63.0**.
- Run từ L4 và 5090 được trộn (26/11); bảng in GPU của các run.
- Mỗi run mất khoảng 2–3 phút (evaluate val) cộng 30 giây (Tầng 2).

```python
# ==== Cell L: CHẠY TỔNG HỢP — run Tầng 1 đã có → evaluate val → Tầng 2 → một bảng. CHỈ VAL, không đụng test ====
if not os.path.isdir('/content/drive/MyDrive'):            # cell I tháo Drive ở cuối
    drive.mount('/content/drive')
assert str(DATA).startswith('/content/'), 'evaluate val cần ảnh trong /content: chạy Setup với IMAGES = True'
assert RUNS.name == 'runs_aug1', f'RUNS phải là runs_aug1, đang là {RUNS}'
STAGE2 = SYM / 'stage2_v2'
run_logged([sys.executable, '-u', '-m', 'stage2_decoder.run_all', '--runs', RUNS, '--splits', SPLITS,
            '--data-root', DATA, '--output', STAGE2, '--variants', 'baseline', 'full',
            '--folds', 1, 2, 3, '--seeds', 0, 1, 2, '--val-only'], 'tong_hop_stage2_val')
run_logged([sys.executable, '-u', '-m', 'stage2_decoder.val_overview', '--runs', RUNS,
            '--results', STAGE2 / 'results', '--output', STAGE2 / 'val_overview.json'], 'tong_hop_bang_val')
print('Bảng đã ghi vào', STAGE2 / 'val_overview.json', flush=True)
drive.flush_and_unmount()
```

---

## N. Chẩn đoán contrastive — CHỈ fit/val (GPU, chạy sau cell L, Setup với `IMAGES = True`)

Có trong [notebooks/Tong_hop_val.ipynb](notebooks/Tong_hop_val.ipynb), ngay sau cell L. Module `diagnostics/` (ngoài danh sách file bị hash) đo ba thứ cho mỗi run đã có `evaluation_val/` ([Huong_SOTA_Contrastive.md](Huong_SOTA_Contrastive.md) §5.3):
- `calibration.json`: AUC theo AU; SD của ngưỡng tối ưu giữa 3 subject val; F1(t*) − F1(0.5). Chỉ đọc `val_predictions.npz`.
- `identity_probe.json`: lớp tuyến tính đoán subject từ đặc trưng đóng băng của `best.pt`, trên frame **fit** (lấy mỗi 4 frame; 70% đầu train probe, 30% cuối đo). Càng gần mức ngẫu nhiên thì đặc trưng càng ít mang danh tính.
- `grad_norms.json`: chuẩn gradient của từng số hạng loss trên 20 batch fit, cho `best.pt` và `last.pt`; tỉ số ‖∇contrastive‖ / ‖∇L_wa‖ và cos giữa hai gradient trên backbone. Dùng để kiểm giả thuyết "về cuối train, phần contrastive lái backbone".

Không train, không đụng test. Checkpoint được chép về `/content/diag_ckpt` (có timeout) rồi mới nạp, xong thì xoá. Kết quả ở `MyDrive/Sym/diagnostics/<variant>/fold{k}/seed{s}/`, bảng ở `diagnostics/summary.json`. Chạy lại an toàn: phần đã có được bỏ qua, trừ khi `best.pt` đã đổi. Ước lượng 2–4 phút mỗi run, chưa đo; phần lớn là chép checkpoint từ Drive.
Đây là số mô tả trên fit/val: không báo cáo thay cho F1, không so với 63.0.

```python
# ==== Cell N: CHẨN ĐOÁN — hiệu chỉnh theo subject val, probe danh tính, chuẩn gradient. CHỈ fit/val, không đụng test ====
if not os.path.isdir('/content/drive/MyDrive'):            # cell L tháo Drive ở cuối
    drive.mount('/content/drive')
assert str(DATA).startswith('/content/'), 'probe danh tính và chuẩn gradient cần ảnh trong /content: chạy Setup với IMAGES = True'
assert RUNS.name == 'runs_aug1', f'RUNS phải là runs_aug1, đang là {RUNS}'
DIAG = SYM / 'diagnostics'
run_logged([sys.executable, '-u', '-m', 'diagnostics.run_all', '--runs', RUNS, '--splits', SPLITS,
            '--data-root', DATA, '--output', DIAG, '--stage-dir', '/content/diag_ckpt'], 'chan_doan')
print('Bảng đã ghi vào', DIAG / 'summary.json', flush=True)
drive.flush_and_unmount()
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
