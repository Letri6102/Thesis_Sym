# Thesis_Sym — quy tắc làm việc

Luận văn: nhận diện AU trên DISFA. Tầng 1 là detector contrastive, Tầng 2 là luật rút từ dữ liệu cộng giải mã MAP chính xác.
Kế hoạch nghiên cứu và trạng thái hiện tại nằm ở [De_cuong_nghien_cuu.md](De_cuong_nghien_cuu.md). Đọc file đó trước khi đề xuất thí nghiệm.

## 1. Quy tắc bắt buộc (quyết định của chủ dự án, 20/11/2026)

- **Không train, evaluate hay chạy Tầng 2 ở máy local.** Mọi thứ đụng tới dataset hoặc GPU đều chạy trên **Google Colab** qua notebook.
- **Mọi code đều phải commit và push lên GitHub** (`Letri6102/Thesis_Sym`, nhánh `main`). Notebook Colab lấy code từ git, **không** đọc code từ Drive.
- **Dataset và kết quả chạy vẫn nằm trên Google Drive** (`MyDrive/Sym`) như cũ. Không đưa data, trọng số hay kết quả lên git.
- Ở local chỉ được: sửa code, chạy unit test nhẹ (`Contrastive_V1/.venv-tests`, `pytest tests`), đọc log và kết quả đã kéo từ Drive về để phân tích.
- Thư mục `img/` và `ActionUnit_Labels/` ở local chỉ là bản sao để tra cứu, không dùng để train.

## 2. Luồng làm việc

1. Sửa code ở local, chạy `pytest` nếu sửa Tầng 1.
2. `git commit` rồi `git push origin main`.
3. Trên Colab, chạy cell **Setup** (mục 5). Cell này clone repo, hoặc `fetch` + `reset --hard` về `origin/main`.
4. Chạy các cell train/evaluate/Tầng 2. Mọi đầu ra ghi vào `MyDrive/Sym/...`.
5. Khi cần phân tích ở local, kéo log và kết quả về bằng cell nén zip: bỏ `img`, `ActionUnit_Labels`, `source/`, các file `.pt`/`.pth`/`.npz`. Giải nén đè vào thư mục repo; `.gitignore` đã chặn để chúng không lọt vào git.

**Không sửa code trực tiếp trong `/content/Thesis_Sym`:** lần chạy cell Setup sau sẽ xoá hết.

## 3. Ràng buộc resume — dễ làm hỏng run

`train_contrastive.py` lưu vào `config` của mỗi run:
- hash SHA-256 của **mọi** `Contrastive_V1/*.py` ở cấp gốc, `model/*.py`, `config/*.yaml`, `matrixMAE/M_AE_DISFA.npy` và `EXPERIMENT_PROTOCOL.md`;
- phiên bản `torch`, `torchvision`, `numpy`, `timm`, `pillow`;
- cùng các tham số `batch_size`, `workers`, ...

`--resume` sẽ từ chối nếu bất kỳ thứ nào ở trên khác đi. Hệ quả:

- **Không push thay đổi vào các file bị hash khi còn run đang chạy dở**, kể cả sửa một dấu cách. Muốn sửa thì để run xong, hoặc chấp nhận train lại từ đầu vào thư mục mới.
- File nằm ngoài danh sách trên (ví dụ `stage2_decoder/`, `remote_stage2/`, notebook, `.md` ở gốc repo) thì sửa thoải mái.
- `.gitattributes` đặt `* -text` có chủ đích: git không tự đổi line ending, nên byte của code clone về trùng với byte lúc commit (code hiện dùng LF, giống bản trên Drive). **Không bật `core.autocrlf`** và không thêm quy tắc `eol=` khi còn run dở cần resume.
- `torch` không được pin trong `requirements-colab.txt`. Cell Setup in phiên bản torch ra; nếu Colab tự nâng torch giữa hai phiên thì run dở sẽ không resume được.

## 4. Cấu trúc

**Google Drive `MyDrive/Sym`** (nguồn chuẩn cho data và kết quả):

```text
DISFA/img/SNxxx/N.png        130.814 frame, ảnh 224×224 đã crop
DISFA/list/                  DISFA_{train,test}_{img_path,label,weight}_fold{1,2,3}.txt
DISFA/ActionUnit_Labels/     nhãn gốc
splits/fold{1,2,3}.json      manifest CŨ (seed 2026, 3 subject val) — chỉ cho runs_retry1, không sửa
splits_v2/fold{1,2,3}.json   manifest MỚI: như cũ + mỗi AU >= 100 frame dương trong val (chỉ fold 1 đổi)
runs_retry1/                 9 run cũ, KHÔNG augmentation, ĐÓNG BĂNG — chỉ đọc, không ghi thêm
runs_aug1/                   run mới với strong_train_aug
runs_swin_probe/             dò Swin-B (đang rỗng)
failed_runs/, logs/, reports/, stage2_out/
Contrastive_V1/checkpoints/resnet50-19c8e357.pth   trọng số ImageNet
```

**Repo GitHub** (chỉ có code và tài liệu):

```text
Contrastive_V1/              Tầng 1 (train_contrastive.py là entry point) + stage2_decoder/ (Tầng 2 chuẩn)
MultiviewSymAU-master/       bản cài đặt lại SymGraphAU (Stage 2/3 cũ) để đối chiếu
Tang2_Decoder.ipynb          Tầng 2 bản notebook đã chạy 13/11 — lệch đề cương, sẽ thay bằng stage2_decoder
De_cuong_nghien_cuu.md       đề cương bản 2
```

`Contrastive_V1/COLAB_PRO.md` và `notebooks/Colab_Contrastive_V1.ipynb` đã **lỗi thời**: hai file này giả định code nằm ở `MyDrive/SymAU` trên Drive. Dùng mục 5 thay thế.

## 5. Cell Setup chuẩn trên Colab

Cần secret `GH_TOKEN` trong Colab Secrets: fine-grained token, chỉ cấp cho repo này, quyền Contents read-only. Không bao giờ dán token vào code hay notebook.

```python
import os, sys, base64, shutil, subprocess
from pathlib import Path
from google.colab import drive, userdata

if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive')

REPO_URL = 'https://github.com/Letri6102/Thesis_Sym.git'
BRANCH   = 'main'
REPO     = Path('/content/Thesis_Sym')
CODE     = REPO / 'Contrastive_V1'
SYM      = Path('/content/drive/MyDrive/Sym')
SPLITS   = SYM / 'splits_v2'           # manifest mới (val fold 1 rút lại); splits/ cũ chỉ cho runs_retry1
RUNS     = SYM / 'runs_aug1'

tok  = userdata.get('GH_TOKEN')
auth = 'http.extraHeader=Authorization: Basic ' + base64.b64encode(f'x-access-token:{tok}'.encode()).decode()
def git(*args, cwd=None):
    subprocess.run(['git', '-c', auth, *args], cwd=cwd, check=True)
if (REPO / '.git').exists():
    git('fetch', '-q', 'origin', BRANCH, cwd=REPO)
    git('reset', '-q', '--hard', f'origin/{BRANCH}', cwd=REPO)
else:
    git('clone', '-q', '--depth', '1', '-b', BRANCH, REPO_URL, str(REPO))
del tok, auth
print('commit:', subprocess.run(['git', 'log', '-1', '--format=%h %s'], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())

IMG_TAR = SYM / 'DISFA_img.tar'          # ảnh đóng gói một file; có thì chép về /content cho nhanh
if IMG_TAR.exists():
    DATA = Path('/content/DISFA')
    if not (DATA / 'img').exists():
        DATA.mkdir(parents=True, exist_ok=True)
        shutil.copy(IMG_TAR, '/content/DISFA_img.tar')
        subprocess.run(['tar', '-xf', '/content/DISFA_img.tar', '-C', str(DATA)], check=True)
        os.remove('/content/DISFA_img.tar')
        shutil.copytree(SYM / 'DISFA' / 'list', DATA / 'list', dirs_exist_ok=True)
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
```

Lệnh train mẫu (chạy sau Setup):

```python
out = RUNS / 'baseline' / 'fold2' / 'seed0'
cmd = [sys.executable, 'train_contrastive.py', 'train', '--variant', 'baseline', '--seed', '0',
       '--arc', 'resnet50', '--data-root', str(DATA), '--manifest', str(SPLITS / 'fold2.json'),
       '--output', str(out), '--epochs', '20', '--batch-size', '32', '--workers', '2']
if (out / 'last.pt').exists():
    cmd += ['--resume', str(out / 'last.pt')]
subprocess.run(cmd, check=True)
```

## 6. Quyết định giao thức đã chốt

- Thứ tự 8 AU: `1, 2, 4, 6, 9, 12, 25, 26`; intensity ≥ 2 là dương. Chia 3 fold theo subject, mỗi fold 3 subject làm val bên trong.
- Inner val (20/11): `splits_v2`, sinh bằng `experiment_data.py --split-seed 2026 --val-subjects 3 --min-val-positive 100`. Fold 2/3 trùng bản cũ, fold 1 rút lại, vì val cũ có AU2 = 4 frame và AU1 = 73 frame, tất cả từ một subject.
- Bỏ variant `current` (12/11). Các variant chính: `baseline` và `full`.
- Mọi lựa chọn (checkpoint, ngưỡng, λ, luật) **chỉ** dựa trên inner val. Outer test chỉ mở **một lần**, sau khi đã chốt thiết kế.
- Test đã bị mở trên detector cũ (29/10 cho Tầng 1, 13/11 cho Tầng 2). Các số đó chỉ là **thăm dò**; không so trực tiếp với run mới.
- Cả 9 run trong `runs_retry1` chạy **không augmentation** (hash khớp `experiment_data.py.bak_20261113_1402`). `strong_train_aug` có trong `experiment_data.py` từ 13/11; mọi run mới ghi vào `runs_aug1`.
- Khi theo dõi loss, đọc các cột thô `global_con`/`node_con`/`cross_con` trong `history.json`. Không đọc `total`, vì cột này có nhân hệ số ramp.

## 7. Việc đang mở (cập nhật 20/11/2026)

- Đóng gói `DISFA/img` thành `MyDrive/Sym/DISFA_img.tar` (làm trên Colab, từ Drive). Hiện 10/26 lần train bị dừng vì Drive báo `FileNotFoundError` với ảnh vẫn có thật.
- Sinh `splits_v2` trên Colab (code đã có, xem đề cương §3). Phải xong trước run đầu tiên dùng `splits_v2`.
- Thống nhất Tầng 2 về `stage2_decoder`: sửa đường dẫn `evaluation/` → `evaluation_val/` cho val, thêm nhánh `R_FACS`, lấy R0 làm nhánh chính. C1 (chỉnh ngưỡng trên 3 subject val) làm điểm test giảm tới 4 điểm.
- Pilot `baseline` + `full`, fold 2, `runs_aug1`.
