# Stage 2 decoder trên máy remote (Linux + GPU)

Stage 2 ở đây là bản cải tiến: **luật AU–AU kiểm định từ nhãn fit + decoder MAP cố định**. Không dùng SAT, GCN hay class center.

- Không train lại Stage 1. Decoder dùng nguyên `best.pt` của `train_contrastive.py`, đã chọn trên inner val theo `EXPERIMENT_PROTOCOL.md`.
- GPU chỉ cần cho `train_contrastive.py evaluate --split val` (3 subject mỗi run), và `--split test` nếu run chưa có.
- Phần luật và decoder chạy CPU, vài giây mỗi run.
- Bảng luật chỉ phụ thuộc nhãn fit của fold, nên `baseline`, `current`, `full` dùng **chung** một bảng trong cùng fold.

## 1. Bố cục trên remote (giống `MyDrive/Sym`)

```text
~/Sym/
  Contrastive_V1/                 code (có stage2_decoder/, remote_stage2/)
  DISFA/img/SN001/0.png ...       ảnh crop
  DISFA/list/DISFA_{train,test}_{img_path,label}_fold{1,2,3}.txt
  splits/fold1.json fold2.json fold3.json
  runs_retry1/<variant>/fold{k}/seed{s}/{config.json, history.json, best.pt, evaluation/}
```

**Quan trọng:**
- `splits/*.json` lưu hash của các file list. Phải dùng **đúng** bộ list đã tạo manifest (trong DISFA trên Drive), nếu không preflight sẽ báo lỗi.
- Với `runs_retry1`, chỉ cần `best.pt`, `config.json`, `history.json` và `evaluation/`. **Bỏ `last.pt`** (~0,5 GB/run), trừ khi cần resume.

## 2. Đưa dữ liệu lên

**Cách A — kéo thẳng từ Google Drive bằng rclone** (khuyên dùng, không cần qua máy Windows):

```bash
curl https://rclone.org/install.sh | sudo bash      # hoặc tải binary nếu không có sudo
rclone config                                        # tạo remote tên "gdrive" (Google Drive)
mkdir -p ~/Sym && cd ~/Sym
rclone copy gdrive:Sym/splits splits -P
rclone copy gdrive:Sym/runs_retry1 runs_retry1 -P --exclude "last.pt" --exclude "source/**"
rclone copy gdrive:Sym/DISFA.zip . -P && unzip -q DISFA.zip      # nếu Drive có DISFA.zip
# nếu chỉ có thư mục: rclone copy gdrive:Sym/DISFA DISFA -P --transfers 32   (nhiều file nhỏ, chậm)
```

**Cách B — từ Windows** (code luôn đi đường này; data dùng khi không kéo được từ Drive):

```powershell
cd C:\CODE\Symbolic\SymAU-master
tar -czf contrastive_v1.tgz --exclude=.venv-tests --exclude=__pycache__ --exclude=.pytest_cache --exclude=tmp Contrastive_V1
scp contrastive_v1.tgz user@REMOTE:~/Sym/
```

Trên remote: `cd ~/Sym && tar -xzf contrastive_v1.tgz`.

## 3. Cài môi trường

```bash
SYM=~/Sym bash ~/Sym/Contrastive_V1/remote_stage2/setup.sh
#   driver cũ:  CUDA_TAG=cu126 SYM=~/Sym bash ...
```

Script này tạo venv, cài torch 2.7.1 cùng `requirements-colab.txt`, rồi chạy `pytest tests`, `preflight` và `check_stage1`. Chỉ chạy tiếp khi preflight in **SẴN SÀNG**.

## 4. Chạy decoder

```bash
mkdir -p ~/Sym/logs
SYM=~/Sym nohup bash ~/Sym/Contrastive_V1/remote_stage2/run_decoder.sh > ~/Sym/logs/stage2_decoder.log 2>&1 &
tail -f ~/Sym/logs/stage2_decoder.log
```

Mặc định chạy mọi run đã xong của `baseline current full` × 3 fold × seed `0 1 2`. Run nào chưa có sẽ được liệt kê và bỏ qua.

Một số biến thể:

```bash
SEEDS="0" SYM=~/Sym bash .../run_decoder.sh           # chỉ seed 0 (hiện đã đủ 9 run)
NO_IMPL=1 SYM=~/Sym bash .../run_decoder.sh           # chỉ luật loại trừ (bảng luật mới → xoá rules/ trước)
```

Kết quả nằm ở `~/Sym/downstream/stage2_decoder/`:
- `rules/fold{k}/rules_all.csv`: mọi ứng viên, kèm lift, CI95, %subject, w0
- `results/<variant>/fold{k}/seed{s}/decode_summary.json`
- `summary.json`: bảng C0/C1/R0/R1/R* theo seed (mean ± SD), Δ R*−C0 và R*−C1, cùng full−baseline trước và sau decoder

## 5. (Tuỳ chọn) Train Stage 1 còn thiếu (seed 1, 2)

Protocol cần 3 seed. Theo báo cáo ngày 29/10, mới có seed 0.

```bash
SYM=~/Sym SEEDS="1 2" nohup bash ~/Sym/Contrastive_V1/remote_stage2/train_missing_stage1.sh > ~/Sym/logs/stage1_train.log 2>&1 &
```

- Mỗi run chạy 20 epoch, ResNet50, batch 32, đúng như `COLAB_PRO.md`. Trên Colab mất khoảng 7–20 giờ/run.
- **Train mọi variant của cùng một seed trên cùng một máy/môi trường.** `summarize_experiments.py` yêu cầu `environment_versions` trùng nhau giữa baseline và full ở cùng fold/seed.
- Run Colab đang dở không resume được nếu phiên bản thư viện khác, vì config phải trùng y hệt.

Sau khi train xong, chạy lại `run_decoder.sh`. Script tự evaluate val/test cho các run mới.

## 6. Lấy kết quả về

```powershell
scp -r user@REMOTE:~/Sym/downstream/stage2_decoder C:\CODE\Symbolic\SymAU-master\downstream\
scp user@REMOTE:~/Sym/logs/stage2_decoder.log C:\CODE\Symbolic\SymAU-master\logs\
```

Hoặc trên remote: `rclone copy ~/Sym/downstream gdrive:Sym/downstream -P`.
