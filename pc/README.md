# Train trên máy GPU riêng (RTX 5090, remote)

Máy này chỉ **train** một phần của bộ 18 run, với đúng cấu hình đã khoá ngày 22/11 (CLAUDE.md §6). Evaluate, Tầng 2 và mở test vẫn làm trên Colab. Mọi bước đều kiểm khoá: code, phiên bản thư viện, manifest, trọng số, ảnh. Lệch một thứ là dừng, không train.

## Máy GPU đã có sẵn dữ liệu DISFA (ví dụ `D:\KLTN_Khang\DISFA_Data`) — cách nhanh nhất, 23/11

Không cần chép gói 8 GB. Dữ liệu có sẵn chỉ được **đọc**, và phải trùng từng byte với bản chuẩn thì mới train.

```bat
cd /d D:\LVTS_TriLe
git clone https://github.com/Letri6102/Thesis_Sym.git
D:\LVTS_TriLe\Thesis_Sym\pc\windows\train_data_co_san.bat
```

- **Lần chạy đầu:** dò Python và cài môi trường. Sau đó lệnh `from-folder` làm các việc sau:
  - tìm 12 file list và đối chiếu với mã băm chuẩn ở `pc/data_reference.json`;
  - tạo lại `splits_v2`, trùng từng byte với bản đã khoá;
  - chép ảnh vào `D:\LVTS_TriLe\DISFA`;
  - kiểm từng subject bằng SHA-256 tar trong `pc/bundle_tar_sha256.json`.

  Cuối cùng chạy `check` và chạy thử 1 epoch.
- **Lần chạy thứ hai trở đi:** train cả 18 run (seed 0 1 2), rồi refit. Bị ngắt thì chạy lại file này để tiếp tục.
- **Nguồn chỉ có `img` + `ActionUnit_Labels` (data gốc, như thư mục của Khang):** `from-folder` tự sinh 12 list từ nhãn gốc bằng đúng logic script gốc (`Contrastive_V1/tool/DISFA_image_label_process.py`, fold part1/2/3) rồi đối chiếu mã băm. Không cần chép list.
- **List của nguồn khác bản chuẩn** (đã gặp 23/11 với thư mục của Khang): chép thư mục `DISFA\list` của máy cá nhân (hoặc `LVTS_TriLe\sym_download\DISFA\list` trong gói) thành `D:\LVTS_TriLe\list`; file `.bat` tự nhận.
- **Ảnh khác bản chuẩn:** file dừng ở subject đầu tiên bị lệch và không đánh dấu dữ liệu, nên không train được. Khi đó dùng gói `LVTS_TriLe` ở mục dưới.
- **Mã băm chuẩn lấy từ ảnh ở máy cá nhân.** Muốn chắc chắn trùng bản trên Drive thì chạy thêm cell J (`Nhap_run_PC.ipynb`).

## Máy Windows chỉ được dùng một thư mục (D:\LVTS_TriLe, UltraViewer) — cách đang dùng 23/11

1. **Ở máy cá nhân**, dựng gói (đã dựng sẵn tại `C:\CODE\Symbolic\LVTS_TriLe`):
   `<py> pc/make_bundle.py --splits <thư mục splits_v2> --out C:\CODE\Symbolic\LVTS_TriLe`
2. **Trên Colab**, chạy cell J (notebook `Nhap_run_PC.ipynb`, cell cuối): phải báo `KHỚP 27/27`.
3. **Chép nguyên thư mục** `LVTS_TriLe` sang máy GPU thành `D:\LVTS_TriLe` bằng tính năng truyền file của UltraViewer (khoảng 8,7 GB).
4. **Trên máy GPU**, chạy `D:\LVTS_TriLe\1_cai_dat.bat`. File này dò Python, tạo môi trường, tải trọng số ImageNet chính thức, chạy `prepare`, `check`, `smoke`. Cache pip, file tạm và cache CUDA đều nằm trong `D:\LVTS_TriLe`.
5. Smoke báo OK thì chạy `D:\LVTS_TriLe\2_train.bat` (mặc định seed 0 1 2, tức cả 18 run).
6. Zip kết quả nằm ở `D:\LVTS_TriLe\Thesis_Sym\pc_runs\upload\`: chép về máy cá nhân, kéo vào `MyDrive/Sym/upload/`, rồi chạy cell I.

Cần khoảng 35 GB trống trên D: (gói 8,7 GB + ảnh giải nén 8,3 GB + môi trường khoảng 6 GB + run và zip). Sau khi `prepare` xong có thể xoá `D:\LVTS_TriLe\sym_download\DISFA_img` để lấy lại 8,7 GB.

## 0. Phân công — làm trước, tránh hai máy train trùng một run

**Từ 23/11: máy 5090 train cả 18 run (a) và refit**, Colab không train Tầng 1 nữa (dừng cell G, K). Trước lần nhập đầu tiên bằng cell I: đổi tên `MyDrive/Sym/runs_aug1` → `runs_aug1_colab` (và `runs_refit1` → `runs_refit1_colab` nếu có). Cell I không nhập zip train trên GPU khác GPU của run đã có.

**Smoke** chạy 1 epoch baseline và 1 epoch full của fold 2/seed 0. Chặn khi config lệch khoá hoặc hai variant cùng seed khởi tạo khác nhau. Hash khởi tạo khác pilot Colab thì chỉ ghi lại: trọng số, code, thư viện đã kiểm trùng; phần khác là head khởi tạo ngẫu nhiên trên CPU. `train` cũng kiểm cặp này sau mỗi run full.

- Phân công cũ (trước 23/11: máy 5090 seed 2, Colab seed 0/1) không dùng nữa.

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

Lệnh chạy 1 epoch của `baseline` và 1 epoch của `full` (fold 2/seed 0, đúng cặp pilot trên Colab), rồi xoá thư mục thử. Đạt khi:
- **config đúng khoá, và hai variant khởi tạo trùng nhau trên máy này.** Nếu khác, dừng ngay và gửi Claude file log trong `pc_runs/logs/`. Hash khác pilot Colab chỉ được in ra (khác CPU), không chặn;
- **val F1 epoch 1 gần pilot** (baseline 0.768, full 0.749). Khác GPU nên lệch một chút là bình thường; lệch hơn 0.05 sẽ có cảnh báo;
- in ra số giờ ước tính cho một run.

## 6. Train

```bash
<py> pc/train_18_run.py train --data D:\DISFA --seeds 0 1 2
<py> pc/train_18_run.py status
```

- **Bị ngắt (mất mạng, tắt máy, Ctrl+C):** chạy lại đúng lệnh trên. Run dở sẽ resume; run bị ngắt ngay trong epoch 1 được cất sang `pc_runs/failed_runs/` rồi train lại.
- **Trên Windows remote:** tắt chế độ ngủ của máy, và chạy trong cửa sổ không bị đóng khi ngắt Remote Desktop. Trên Linux thì chạy trong `tmux`.
- **Refit (dòng b):** `train_data_co_san.bat` và `2_train.bat` tự chạy `refit` ngay sau `train`. Lệnh này train lại mỗi run trên đủ 18 subject và dừng ở epoch E mà run đó chọn, cho ra zip `refit1_*.zip`. Chạy tay: `<py> pc/train_18_run.py refit --data <DATA> --seeds 0 1 2`.
- **Đầu ra:** run nằm ở `pc_runs/runs_aug1/`, log ở `pc_runs/logs/`. Mỗi run xong được nén thành `pc_runs/upload/aug1_<variant>_fold<k>_seed<s>.zip` (khoảng 0,7 GB).

## 6a. Xen kẽ với Colab: bàn giao cả run dở (26/11)

Drive (`MyDrive/Sym`) là bản chính. Mỗi lúc chỉ một nơi train một run.

- **Colab → máy 5090** (khi máy 5090 rảnh):
  1. Trên Colab (`notebooks/Train_Colab.ipynb`): dừng cell G2, chạy cell M. Cell tạo trong `MyDrive/Sym/handoff/`: `partial_aug1_*.zip` (run dở, khoảng 0,7 GB mỗi file) và `stubs_*.zip` (bản tóm tắt run đã xong).
  2. Tải các file đó về máy cá nhân, chép bằng UltraViewer vào `D:\LVTS_TriLe\nhan`.
  3. Chạy `train_data_co_san.bat`: file tự nhận (`receive`) rồi train tiếp. Run dở của Colab được resume; run Colab đã xong được bỏ qua.
- **Máy 5090 → Colab** (khi cần trả máy):
  1. Đóng cửa sổ train, chạy `pc\windows\ban_giao.bat`. File nén run dở thành `partial_aug1_*.zip` và đánh dấu đã bàn giao; zip run xong đã nằm sẵn trong `pc_runs\upload`.
  2. Chép mọi file trong `pc_runs\upload` về máy cá nhân, kéo vào `MyDrive/Sym/upload/`.
  3. Trên Colab chạy cell I rồi cell G2.
- **Quy tắc tự động:**
  - Bản nhận chỉ thay bản đang có khi là phần nối tiếp (cùng hash khởi tạo, lịch sử cũ là phần đầu). Lệch nhánh thì không nhập, zip vào `rejected/`, gửi Claude.
  - Run đã bàn giao không được train tiếp ở nơi gửi, trừ khi chạy với `--reclaim` (Colab: `RECLAIM = True`) khi chắc chắn nơi kia chưa train tiếp run đó.
  - Run mới chỉ được tạo khi run cùng cặp (nếu đã có) có cùng hash khởi tạo với máy này. Không khớp thì run đó để lại cho máy kia.
  - Máy train từng đoạn epoch được ghi ở `machines.json` trong run.
- **Chạy song song cho nhanh:** chia theo seed. Ví dụ Colab giữ seed đang làm (`SEEDS` trong cell G2), máy 5090 nhận các seed còn lại (sửa dòng `set "SEEDS=..."` trong `train_data_co_san.bat` khi file **không** chạy). Cuối cùng gom zip của máy 5090 về Drive bằng cell I.

## 6b. Xem tiến độ: % hoàn thành và giờ xong dự kiến

- **Bấm là xem:** nháy đúp `D:\LVTS_TriLe\Thesis_Sym\pc\windows\theo_doi.bat`. Cửa sổ làm mới mỗi 60 giây, chỉ đọc, đóng lúc nào cũng được; không ảnh hưởng train đang chạy.
  - Hiện ba thanh tiến độ: (a), refit, tổng. Kèm số run đã xong, run đang chạy (epoch, % của epoch), giờ xong run đó, giờ xong (a) và giờ xong cả refit.
  - Thời gian còn lại tính theo tốc độ đo được trên chính máy này. E của refit chỉ biết khi run (a) tương ứng xong; trước đó lấy trung bình các E đã biết.
  - Không run nào ghi log trong 15 phút thì báo train có thể đang dừng.
- Từ bản code 24/11, cửa sổ train cũng in một dòng tiến độ tại chỗ (làm mới mỗi 20 giây) và in % tổng sau mỗi epoch.
- Lệnh tay: `<py> pc/train_18_run.py progress` (in một lần) hoặc `progress --watch 60`.
- **Kéo code mới khi đang train:** được, miễn là thay đổi không chạm vào file `.bat` đang chạy. cmd đọc file `.bat` từng đoạn trong lúc chạy, nên sửa file đó giữa chừng sẽ làm cửa sổ train chạy sai lệnh. Tiến trình Python đang chạy không bị ảnh hưởng; bước kế tiếp (ví dụ refit) mới dùng code mới.

## 7. Đưa kết quả về Drive

1. Tải các file zip trong `pc_runs/upload/` lên `MyDrive/Sym/upload/` (tạo thư mục nếu chưa có).
2. Trên Colab, mở `notebooks/Nhap_run_PC.ipynb` (runtime CPU là đủ), chạy Setup rồi cell I. Cell I:
   - kiểm khoá cho từng zip, rồi chép vào `runs_aug1/`;
   - **không bao giờ ghi đè** run đã đủ 20 epoch trên Drive;
   - chuyển zip đã xử lý sang `upload/done/` (đã nhập), `upload/duplicate/` (Drive đã có) hoặc `upload/rejected/` (lệch khoá).
3. Chỉ xoá zip trên máy 5090 sau khi cell I đã báo `ĐÃ NHẬP`.

**Ghi trong luận văn:** cả 18 run (a) và refit train trên RTX 5090 (Windows); pilot dùng để khoá cấu hình train trên L4 (Colab) và có khởi tạo khác. GPU đã được ghi trong `environment.json` của từng run. `full − baseline` được so trong cùng seed; mọi run train trên một máy nên hai variant cùng seed có cùng khởi tạo (smoke và `train` kiểm điều này).
