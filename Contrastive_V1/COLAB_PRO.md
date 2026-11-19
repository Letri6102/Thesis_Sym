# Chạy Contrastive V1 trên Google Colab Pro

Notebook để upload lên Colab: `notebooks/Colab_Contrastive_V1.ipynb`.

Đọc `EXPERIMENT_PROTOCOL.md` trước. Entry point thực nghiệm là
**train_contrastive.py**, không phải `train_Sym_Stage_1.py`/`test.py` cũ.
Các script cũ vẫn tồn tại để đối chiếu. Phần dưới dùng DISFA + ResNet50.

## 1. Chuẩn bị

Trong Colab chọn Runtime → Change runtime type → GPU. Kiểm tra GPU thật bằng
`!nvidia-smi`. Pro không bảo đảm A100/L4/T4 hay thời gian chạy cố định; VM có thể
bị ngắt. Đọc [Colab FAQ](https://research.google.com/colaboratory/faq.html).

Upload code lên Drive thành `MyDrive/SymAU/Contrastive_V1/`. **Không upload**
`.venv-tests`, `__pycache__`, `tmp`, `.pytest_cache`. Lưu data đã crop/aligned trong
`MyDrive/SymAU/DISFA.zip`, với cấu trúc zip chứa thư mục `DISFA/`:

```text
DISFA/
  img/SN001/0.png ...         # ảnh crop chuẩn, không dùng raw full-video frames
  ActionUnit_Labels/SN001/SN001_au1.txt ...   # nếu cần tạo lists
  list/DISFA_train_img_path_fold1.txt
  list/DISFA_train_label_fold1.txt
  list/DISFA_test_img_path_fold1.txt
  list/DISFA_test_label_fold1.txt
  ... fold2, fold3
```

Đường dẫn trong list phải là `SN001/0.png`, tương đối so với `img/`, không phải
đường dẫn Windows tuyệt đối. Thứ tự 8 AU: `1,2,4,6,9,12,25,26`; intensity >=2
là active. Cần đủ 27 subjects để chạy full CV. Không tự bỏ frame thiếu.

Chạy cell Python:

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import shutil, subprocess, zipfile

PROJECT = Path('/content/Contrastive_V1')
if not PROJECT.exists():
    shutil.copytree('/content/drive/MyDrive/SymAU/Contrastive_V1', PROJECT,
                    ignore=shutil.ignore_patterns('.venv-tests', '__pycache__', '.pytest_cache', 'tmp'))
if not Path('/content/DISFA').exists():
    with zipfile.ZipFile('/content/drive/MyDrive/SymAU/DISFA.zip') as z:
        z.extractall('/content')  # archive tự chuẩn bị/tin cậy
%cd /content/Contrastive_V1
!pip -q install -r requirements-colab.txt
!python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
!nvidia-smi
```

Giữ cặp torch/torchvision CUDA Colab cung cấp; không cài bản CPU và không ép
CUDA 12.8 theo `Set up.txt` cũ. Gói SAT đúng là **python-sat**, không phải `pysat`.
Các dependency phụ được pin trong requirements; lưu môi trường thực tế mỗi run:

```python
!pip freeze > /content/drive/MyDrive/SymAU/colab-environment.txt
```

Ảnh đặt trên SSD `/content`, checkpoint trên Drive. Cách này giảm hàng trăm nghìn
lần đọc file nhỏ qua Drive; Google cũng khuyến nghị copy archive rồi giải nén local.

## 2. ImageNet weights và preflight

Download đúng ResNet50 ImageNet V1 mà clone sử dụng (nguồn PyTorch):

```python
import torch
from pathlib import Path
Path('checkpoints').mkdir(exist_ok=True)
torch.hub.download_url_to_file(
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'checkpoints/resnet50-19c8e357.pth')
!python -m pytest tests -q
```

Tests dùng CPU + synthetic data; không download weight và không phải benchmark F1.
Nếu thay backbone, cung cấp weight đúng tên trong `model/resnet.py` hoặc
`model/swin_transformer.py`; chạy lại toàn bộ đối chứng với cùng backbone.

Nếu chưa có lists, xác nhận số frame của ảnh: ví dụ annotation frame `1` tương
ứng ảnh `0.png` thì offset=-1; nếu tương ứng `1.png` thì offset=0. **Xem trực tiếp
quy trình crop trước**, không tự đoán từ số lượng file. Sau đó chạy một lựa chọn:

```python
!python prepare_disfa_lists.py --data-root /content/DISFA --frame-offset -1
```

Không chạy bước tạo lists nếu đã có lists chuẩn; tool từ chối ghi đè lists khác.
Tool kiểm tra frame ID giữa các AU, nhãn, toàn bộ đường dẫn ảnh và không padding
nhãn thiếu. Không chạy script preprocess cũ vốn có thể tự pad/skip nhãn.

Tạo manifest một lần; output trên Drive để dùng lại giữa sessions:

```python
!python experiment_data.py --data-root /content/DISFA \
  --output /content/drive/MyDrive/SymAU/splits --split-seed 2026 --val-subjects 3
```

Nếu preflight báo AU không có positive trong fit, phải thiết kế lại split trước
thí nghiệm cho **mọi** variant và lưu seed mới; không đổi split sau khi xem test.

## 3. Train smoke/pilot và xác định batch

Runner mặc định FP32, batch 32; mỗi batch chỉ một view. Thử một epoch baseline
và full trên inner split trước, vẫn giữ cosine schedule tổng 20 epochs:

```python
!python train_contrastive.py train --variant baseline --seed 0 --arc resnet50 \
  --data-root /content/DISFA --manifest /content/drive/MyDrive/SymAU/splits/fold1.json \
  --output /content/drive/MyDrive/SymAU/runs/baseline/fold1/seed0 \
  --epochs 20 --batch-size 32 --workers 2 --stop-after 1
!python train_contrastive.py train --variant full --seed 0 --arc resnet50 \
  --data-root /content/DISFA --manifest /content/drive/MyDrive/SymAU/splits/fold1.json \
  --output /content/drive/MyDrive/SymAU/runs/full/fold1/seed0 \
  --epochs 20 --batch-size 32 --workers 2 --stop-after 1
```

Quan sát `history.json`: thời gian, peak VRAM, loss và
`node_eligible_batch_fraction` theo từng AU. Coverage thấp nghĩa là batch không
có đủ positive/negative cho AU đó; cân nhắc sampling/queue ở ablation tiếp theo,
không mặc định coi loss=0 là thành công. Nếu OOM, bắt đầu các run mới với batch
16 cho cả baseline/full/current; gradient accumulation không thay thế batch
contrastive lớn. Không dùng kết quả test để chọn batch/temperature.

Runner lưu `best.pt`, `last.pt` ở mỗi epoch. `last.pt` có model, optimizer,
queue, RNG, lịch sử; resume từ ranh giới epoch, batch dở bị chạy lại. Vì checkpoint
chứa pickle RNG/optimizer, chỉ load checkpoint tự tạo/tin cậy.

## 4. Các run chính và resume

Chạy mỗi run trong một session hoặc một nhóm đủ ngắn theo thời gian pilot.
Cell Python sau chạy cả run mới lẫn resume nhưng có thể cần nhiều sessions:

```python
import subprocess, sys
from pathlib import Path
RUNS = Path('/content/drive/MyDrive/SymAU/runs')
SPLITS = Path('/content/drive/MyDrive/SymAU/splits')
# Chốt danh sách này TRƯỚC khi xem outer test.
variants = ['baseline', 'current', 'full']
# Ablation mở rộng: thêm ['global', 'node', 'cross', 'global_cross']
for variant in variants:
    for fold in [1, 2, 3]:
        for seed in [0, 1, 2]:
            out = RUNS / variant / f'fold{fold}' / f'seed{seed}'
            cmd = [sys.executable, 'train_contrastive.py', 'train',
                   '--variant', variant, '--seed', str(seed), '--arc', 'resnet50',
                   '--data-root', '/content/DISFA', '--manifest', str(SPLITS/f'fold{fold}.json'),
                   '--output', str(out), '--epochs', '20', '--batch-size', '32', '--workers', '2']
            if (out/'last.pt').exists():
                cmd += ['--resume', str(out/'last.pt')]
            subprocess.run(cmd, check=True)
```

3 variants × 3 folds × 3 seeds = **27 Stage-1 runs**. Cả 7 variants = 63 runs.
Đây là ngân sách lớn; pilot 1 fold/1 seed chỉ dùng đánh giá khả năng chạy và chi
phí, chưa đủ kết luận hiệu quả. Dự toán bằng thời gian đo được, không hứa số giờ.

Resume phải giữ code/source hashes, manifest và config y hệt. Sau khi Colab ngắt,
mount Drive, copy cùng snapshot code và data local, cài lại môi trường rồi chạy
cell trên. Không tăng `--epochs` khi resume vì sẽ đổi lịch learning rate. Nếu muốn
thí nghiệm mới, tạo output root mới; runner từ chối âm thầm ghi đè run cũ.
Resume cần cấu trúc GPU tương thích để tái lập RNG; GPU/version khác có thể làm
thay đổi số học. V1 chưa bật AMP; thêm AMP là thay đổi áp dụng cho cả nhóm.

## 5. Đánh giá test sau khi chốt thiết kế

```python
for variant in variants:
    for fold in [1, 2, 3]:
        for seed in [0, 1, 2]:
            out = RUNS / variant / f'fold{fold}' / f'seed{seed}'
            ev = out / 'evaluation'
            if (ev/'test_metrics.json').exists():
                continue
            subprocess.run([sys.executable, 'train_contrastive.py', 'evaluate',
                '--checkpoint', str(out/'best.pt'), '--data-root', '/content/DISFA',
                '--manifest', str(SPLITS/f'fold{fold}.json'), '--output', str(ev)], check=True)
!python summarize_experiments.py --root /content/drive/MyDrive/SymAU/runs \
  --baseline baseline --candidate full --output /content/drive/MyDrive/SymAU/full_vs_baseline.json
!python summarize_experiments.py --root /content/drive/MyDrive/SymAU/runs \
  --baseline current --candidate full --output /content/drive/MyDrive/SymAU/full_vs_current.json
```

Metrics F1 trong file evaluation là [0,1]; summary đổi sang phần trăm và delta
percentage points. Summary kiểm tra đủ 3 folds/seed và config/khởi tạo ghép cặp,
báo mean ± SD theo seeds; không tự chọn seed tốt nhất. Kết quả một seed được
đánh dấu preliminary. `test_predictions.npz` có nhãn, xác suất, path và subject.

## 6. Kiểm định downstream Stage 2–3 (tùy chọn sau Stage 1)

Chạy từ working directory `/content/Contrastive_V1`. Ví dụ **full/fold1/seed0**;
lặp giống nhau cho baseline/current, các folds và seeds đã chốt:

```python
!python train_Sym_Stage_2.py --dataset DISFA --arc resnet50 --fold 1 --seed 0 \
  --data-root /content/DISFA --split-manifest /content/drive/MyDrive/SymAU/splits/fold1.json \
  --resume /content/drive/MyDrive/SymAU/runs/full/fold1/seed0/best.pt \
  --output-root /content/drive/MyDrive/SymAU/downstream --exp-name full_f1_s0_stage2 \
  -b 32 -lr .01 --epochs 5 -j 2
!python train_Sym_Stage_3.py --dataset DISFA --arc resnet50 --fold 1 --seed 0 \
  --data-root /content/DISFA --split-manifest /content/drive/MyDrive/SymAU/splits/fold1.json \
  --resume /content/drive/MyDrive/SymAU/runs/full/fold1/seed0/best.pt \
  --resume-phase2 /content/drive/MyDrive/SymAU/downstream/full_f1_s0_stage2/bs_32_seed_0_lr_0.01/best_model_phase2_fold1.pth \
  --output-root /content/drive/MyDrive/SymAU/downstream --exp-name full_f1_s0_stage3 \
  -b 32 -lr 1e-6 --epochs 20 -j 2
!python train_contrastive.py evaluate \
  --checkpoint /content/drive/MyDrive/SymAU/downstream/full_f1_s0_stage3/bs_32_seed_0_lr_1e-06/best_model_phase3_fold1.pth \
  --data-root /content/DISFA --manifest /content/drive/MyDrive/SymAU/splits/fold1.json \
  --output /content/drive/MyDrive/SymAU/downstream/full_f1_s0_stage3/evaluation
```

Stage 2/3 adapter dùng fit/inner val từ manifest; fit-only class weights ở Stage 3;
kiểm tra Stage-2 checkpoint sinh từ đúng Stage-1 checkpoint. Không tái sử dụng
logic GCN giữa variants. Dùng `summarize_experiments.py --stage 3` sau đủ runs.

**Giới hạn downstream:** Stage 2/3 vẫn là trainer legacy, chưa hỗ trợ resume đầy
đủ optimizer/RNG như runner Stage 1. Nếu bị ngắt, chạy lại stage đó từ đầu trong
thư mục mới; không gọi `--resume` là tiếp tục Stage 3 (đó là khởi tạo từ Stage 1).
Known issues symbolic graph (polarity/SAT/topology) chưa được tái thiết kế trong
V1; luôn báo kết quả là so sánh trên downstream hiện có, chưa phải reproduction
chính xác SymGraphAU. Không dùng cải thiện Stage 1 để suy ra cải thiện Stage 3.
