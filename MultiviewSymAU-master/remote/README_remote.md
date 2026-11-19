# Chuyển sang máy remote (Linux + GPU) để train

## 1. Cần mang những gì

| Thứ | Ở đâu trên máy remote | Ghi chú |
|---|---|---|
| Source code | `MultiviewSymAU-master/` | Bỏ qua `.venv*`, `__pycache__`, `results/` |
| Ảnh DISFA đã crop | `data/DISFA/img/SN001/0.png` … | Phần nặng nhất, nên copy riêng bằng rsync |
| List + trọng số | `data/DISFA/list/` | `DISFA_{train,test}_{img_path,label}_fold{1,2,3}.txt`, `DISFA_weight_fold*.txt`, **`DISFA_train_weight_fold*.txt`** (Stage 1 đọc file này) |
| Backbone ImageNet | `checkpoints/resnet50-19c8e357.pth` | Hoặc `swin_base_patch4_window7_224.pth` nếu dùng Swin |
| `matrixMAE/M_AE_DISFA.npy` | đã nằm trong source | |

Không cần mang checkpoint Stage 1 cũ: thí nghiệm decoder cần checkpoint train trên `fit`, nên sẽ train lại trên remote.

## 2. Đóng gói trên Windows (PowerShell)

```powershell
cd C:\CODE\Symbolic\SymAU-master
tar -czf symau_code.tgz --exclude=.venv --exclude=.venv39 --exclude=__pycache__ `
    --exclude=results --exclude=data --exclude=checkpoints --exclude=*.png MultiviewSymAU-master
# list + weight (nhỏ)
tar -czf disfa_list.tgz -C <thư_mục_chứa_data> data/DISFA/list
```

`--exclude=*.png` bỏ cả file `logic_multipanel_fold2.png` ở thư mục gốc. Ảnh DISFA nằm trong `data/`, không nằm trong source.

## 3. Đẩy lên remote

```powershell
scp symau_code.tgz disfa_list.tgz user@REMOTE:~/
# Ảnh: dùng rsync (trong WSL hoặc Git Bash) để tiếp tục được khi đứt mạng
rsync -avP /mnt/<ổ>/<đường_dẫn>/data/DISFA/img/ user@REMOTE:~/MultiviewSymAU-master/data/DISFA/img/
```

Nếu không có rsync: nén `img` bằng `tar -cf disfa_img.tar ...` rồi `scp`. WinSCP cũng được.

## 4. Trên remote

```bash
tar -xzf symau_code.tgz && tar -xzf disfa_list.tgz -C MultiviewSymAU-master
cd MultiviewSymAU-master
mkdir -p checkpoints
wget -P checkpoints https://download.pytorch.org/models/resnet50-19c8e357.pth   # URL có sẵn trong model/resnet.py

bash remote/setup_env.sh --arc resnet50     # venv + torch cu128 + sửa path Windows + preflight
#   driver cũ hơn CUDA 12.8:  CUDA_TAG=cu126 bash remote/setup_env.sh
```

Chỉ train khi `preflight` in **SẴN SÀNG**.

## 5. Chạy thí nghiệm

```bash
# Chạy nền, thoát SSH vẫn chạy tiếp
nohup bash remote/run_trial.sh > logs/trial.log 2>&1 &
tail -f logs/trial.log

# Tuỳ chọn
FOLDS="2" bash remote/run_trial.sh                         # thử 1 fold trước
ARC=swin_transformer_base BS=32 bash remote/run_trial.sh   # Swin (VRAM lớn hơn)
GPU=1 WORKERS=12 bash remote/run_trial.sh
```

- Mỗi fold chạy lần lượt: Stage 1 trên fit (20 epoch, chọn best trên inner) → luật → xác suất → decoder.
- Stage 1 của fold nào xong thì có file đánh dấu `.done_fold{k}`. Chạy lại script sẽ bỏ qua fold đó.
- Kết quả cuối nằm ở `logs/summary.log` và `results/stage2_rules/DISFA/`.

Theo dõi GPU: `watch -n 2 nvidia-smi`.

## 6. Lấy kết quả về

```powershell
scp -r user@REMOTE:~/MultiviewSymAU-master/results/stage2_rules .\results\
scp -r user@REMOTE:~/MultiviewSymAU-master/logs .\logs_remote\
```

Checkpoint (`results/<exp>/.../best_model_fold*.pth`) nặng, chỉ tải về khi cần.
