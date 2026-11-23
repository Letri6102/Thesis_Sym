# Train trên máy GPU riêng (RTX 5090, remote)

Máy này chỉ **train** một phần của bộ 18 run, với đúng cấu hình đã khoá ngày 22/11 (CLAUDE.md §6). Evaluate, Tầng 2 và mở test vẫn làm trên Colab. Mọi bước đều kiểm khoá: code, phiên bản thư viện, manifest, trọng số, ảnh. Lệch một thứ là dừng, không train.

## 0. Phân công — làm trước, tránh hai máy train trùng một run

Mặc định: **máy 5090 lấy seed 2** (6 run). **Colab lấy seed 0 và 1**: khi chạy lại cell G, sửa thành `SEEDS = [0, 1]`.

- Nếu máy 5090 xong sớm, có thể nhận thêm run của seed 1 mà Colab **chưa bắt đầu**. Dùng `--seeds 1 --folds 3`, và sửa Colab tương ứng.
- Hai máy không thấy tiến độ của nhau, nên phân công phải ghi rõ trước khi chạy.

## 1. Tải từ Google Drive (`MyDrive/Sym`) về máy 5090

Chọn cả 4 mục rồi bấm Download. Drive sẽ nén thành một hoặc vài file zip.

| Mục trên Drive | Dung lượng |
|---|---|
| `DISFA_img/` (27 file `.tar`, 27 file `.json`, `DONE.json`) | khoảng 8,7 GB |
| `DISFA/list/` | nhỏ |
| `splits_v2/` | nhỏ |
| `Contrastive_V1/checkpoints/resnet50-19c8e357.pth` | khoảng 100 MB |

Giải nén **tất cả** file zip vào cùng một thư mục, ví dụ `D:\sym_download` hoặc `~/sym_download`. Cấu trúc bên trong không quan trọng: lệnh `prepare` tự tìm file.

**Máy 5090 không kết nối Drive (cách đang dùng, 23/11):**
1. Tải 4 mục trên về máy cá nhân và giải nén vào `C:\sym_download`.
2. Chép sang máy 5090: `scp -r -P <port> C:\sym_download <user>@<host>:~/`.
3. Khi train xong, lấy zip kết quả về máy cá nhân: `scp -P <port> "<user>@<host>:~/Thesis_Sym/pc_runs/upload/*.zip" C:\pc_upload\`.
4. Kéo các zip đó vào `MyDrive/Sym/upload/`, rồi làm bước 7.

## 2. Lấy code

```bash
git clone https://github.com/Letri6102/Thesis_Sym.git
cd Thesis_Sym
```

Repo là private. Dùng token GitHub chỉ đọc làm mật khẩu khi git hỏi, hoặc tải ZIP của nhánh `main` trên GitHub.
**Không sửa file nào trong `Contrastive_V1/`**: chỉ cần lệch một byte là khoá báo lỗi.

## 3. Môi trường Python

Cần Python 3.10–3.14 (khuyến nghị 3.12 hoặc 3.13) và driver NVIDIA chạy được CUDA 13.0. Lệnh tải khoảng 3 GB.

```bash
python pc/setup_env.py
```

Từ đây `<py>` là python của môi trường vừa tạo:
- Windows: `.venv-pc\Scripts\python`
- Linux: `.venv-pc/bin/python`

## 4. Chuẩn bị dữ liệu, rồi kiểm

```bash
<py> pc/train_18_run.py prepare --downloads D:\sym_download --data D:\DISFA
<py> pc/train_18_run.py check   --data D:\DISFA
```

`prepare`:
- kiểm kích thước từng file `.tar` so với Drive;
- giải nén đủ 130.814 ảnh;
- chép list và `splits_v2` (đối chiếu hash đã khoá);
- chép trọng số ImageNet vào `Contrastive_V1/checkpoints/`.

`check` kiểm hash code, phiên bản thư viện, 3 manifest, GPU và dung lượng ổ trống.

## 5. Chạy thử 1 epoch (bắt buộc trước khi train thật)

```bash
<py> pc/train_18_run.py smoke --data D:\DISFA
```

Lệnh chạy 1 epoch của `full/fold2/seed0`, tức đúng run pilot đã có trên Colab, rồi xoá thư mục thử. Đạt khi:
- **hash khởi tạo trùng hệt pilot.** Nếu khác, dừng ngay và gửi Claude file log trong `pc_runs/logs/`;
- **val F1 epoch 1 gần 0.749.** Khác GPU nên lệch một chút là bình thường; lệch hơn 0.05 sẽ có cảnh báo;
- in ra số giờ ước tính cho một run.

## 6. Train

```bash
<py> pc/train_18_run.py train --data D:\DISFA --seeds 2
<py> pc/train_18_run.py status
```

- **Bị ngắt (mất mạng, tắt máy, Ctrl+C):** chạy lại đúng lệnh trên. Run dở sẽ resume; run bị ngắt ngay trong epoch 1 được cất sang `pc_runs/failed_runs/` rồi train lại.
- **Trên Windows remote:** tắt chế độ ngủ của máy, và chạy trong cửa sổ không bị đóng khi ngắt Remote Desktop. Trên Linux thì chạy trong `tmux`.
- **Đầu ra:** run nằm ở `pc_runs/runs_aug1/`, log ở `pc_runs/logs/`. Mỗi run xong được nén thành `pc_runs/upload/aug1_<variant>_fold<k>_seed<s>.zip` (khoảng 0,7 GB).

## 7. Đưa kết quả về Drive

1. Tải các file zip trong `pc_runs/upload/` lên `MyDrive/Sym/upload/` (tạo thư mục nếu chưa có).
2. Trên Colab, mở `notebooks/Nhap_run_PC.ipynb` (runtime CPU là đủ), chạy Setup rồi cell I. Cell I:
   - kiểm khoá cho từng zip, rồi chép vào `runs_aug1/`;
   - **không bao giờ ghi đè** run đã đủ 20 epoch trên Drive;
   - chuyển zip đã xử lý sang `upload/done/` (đã nhập), `upload/duplicate/` (Drive đã có) hoặc `upload/rejected/` (lệch khoá).
3. Chỉ xoá zip trên máy 5090 sau khi cell I đã báo `ĐÃ NHẬP`.

**Ghi trong luận văn:** run seed 2 train trên RTX 5090, seed 0/1 trên L4. GPU đã được ghi trong `environment.json` của từng run. `full − baseline` được so trong cùng seed, nên hai variant của cùng một seed phải train trên cùng một máy: phân công theo seed đảm bảo điều này.
