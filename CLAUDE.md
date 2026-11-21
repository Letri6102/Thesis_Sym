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

**Claude được phép tự commit và push lên `main`** mỗi khi một thay đổi cần có trên Colab, không phải hỏi trước (chủ dự án cho phép ngày 20/11). Vẫn phải chạy `pytest` trước khi push, không force-push, và cảnh báo trước nếu thay đổi chạm vào file bị hash (mục 3).

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
DISFA_img/                   SNxxx.tar (27 tar ảnh theo subject) + SNxxx.json (dấu đã xong) + DONE.json — cell A tạo, Setup giải nén vào /content
splits/fold{1,2,3}.json      manifest CŨ (seed 2026, 3 subject val) — chỉ cho runs_retry1, không sửa
splits_v2/fold{1,2,3}.json   manifest MỚI: như cũ + mỗi AU >= 100 frame dương trong val (chỉ fold 1 đổi)
runs_retry1/                 9 run cũ, KHÔNG augmentation, ĐÓNG BĂNG — chỉ đọc, không ghi thêm
runs_aug1/                   run mới với strong_train_aug
runs_swin_probe/             dò Swin-B (đang rỗng)
stage2_v2/                   Tầng 2 chuẩn (stage2_decoder): rules/fold{k}/, results/<variant>/fold{k}/seed{s}/, summary.json
stage2_out/                  Tầng 2 bản notebook 13/11 trên detector cũ — chỉ là thăm dò
stage2_smoke_retry1/         cell E: chạy thử stage2_decoder trên runs_retry1 + splits cũ, CHỈ VAL — kiểm module, không báo cáo
failed_runs/, logs/, reports/
Contrastive_V1/checkpoints/resnet50-19c8e357.pth   trọng số ImageNet
```

**Repo GitHub** (chỉ có code và tài liệu):

```text
Contrastive_V1/              Tầng 1 (train_contrastive.py là entry point) + stage2_decoder/ (Tầng 2 chuẩn, test: tests/test_stage2.py)
Contrastive_V1/remote_stage2/ script cho máy remote (runs_retry1, splits cũ) — LỖI THỜI, trái quy tắc chỉ chạy Colab
MultiviewSymAU-master/       bản cài đặt lại SymGraphAU (Stage 2/3 cũ) để đối chiếu
Tang2_Decoder.ipynb          Tầng 2 bản notebook 13/11 — LỖI THỜI, lệch đề cương; dùng stage2_decoder
De_cuong_nghien_cuu.md       đề cương bản 2
```

`Contrastive_V1/COLAB_PRO.md` và `notebooks/Colab_Contrastive_V1.ipynb` đã **lỗi thời**: hai file này giả định code nằm ở `MyDrive/SymAU` trên Drive. Dùng mục 5 thay thế.

## 5. Cell Colab chuẩn → [COLAB_CELLS.md](COLAB_CELLS.md)

Mọi cell Colab (Setup, đóng gói ảnh, pilot, kiểm kết quả, kéo log về) nằm trong `COLAB_CELLS.md`. Khi đưa cell mới cho chủ dự án, cập nhật luôn file đó.

Quy tắc cho mọi cell:
- Token chỉ đọc qua `userdata.get('GH_TOKEN')`, không bao giờ in ra. Hàm `git()` **không** dùng `check=True`, vì `CalledProcessError` in nguyên lệnh, kể cả header chứa token (đã từng lộ token một lần ngày 20/11, phải thu hồi). Nếu output nào hiện `Authorization: Basic ...`, thu hồi token ngay.
- Colab **không hiện** stdout của tiến trình con (`subprocess.run` không capture). Muốn thấy tiến trình thì dùng `Popen` và đọc từng dòng, đồng thời ghi log vào `MyDrive/Sym/logs/`.
- Train chỉ chạy khi `DATA` nằm trong `/content` (Setup đã giải nén `DISFA_img/`). Đọc ảnh thẳng từ Drive đã làm 10/26 lần train bị dừng.
- Mọi lần đọc/chép file trên Drive trong vòng lặp dài phải có timeout và chạy lại được (resume). Ngày 21/11, cell A bản một file đứng ở 40.000/130.814 ảnh suốt hơn 12 giờ mà không báo lỗi: một lần đọc Drive bị treo, không có timeout nên chờ mãi.
- `EPOCHS`, `BATCH`, `WORKERS` của một run phải giữ nguyên qua các lần resume.

## 6. Quyết định giao thức đã chốt

- Thứ tự 8 AU: `1, 2, 4, 6, 9, 12, 25, 26`; intensity ≥ 2 là dương. Chia 3 fold theo subject, mỗi fold 3 subject làm val bên trong.
- Inner val (20/11): `splits_v2`, sinh bằng `experiment_data.py --split-seed 2026 --val-subjects 3 --min-val-positive 100`. Fold 2/3 trùng bản cũ, fold 1 rút lại, vì val cũ có AU2 = 4 frame và AU1 = 73 frame, tất cả từ một subject. Val fold 1 mới là SN002, SN016, SN032. **Chủ dự án đã chọn giữ `splits_v2`** dù fit fold 1 mất khoảng 45% số frame AU1/AU2/AU26 (AU2 còn 1098); đã cân nhắc mức trần "val ≤ 1/3" và không chọn. Không tự sinh lại split.
- Bỏ variant `current` (12/11). Các variant chính: `baseline` và `full`.
- Mọi lựa chọn (checkpoint, ngưỡng, λ, luật) **chỉ** dựa trên inner val. Outer test chỉ mở **một lần**, sau khi đã chốt thiết kế.
- Test đã bị mở trên detector cũ (29/10 cho Tầng 1, 13/11 cho Tầng 2). Các số đó chỉ là **thăm dò**; không so trực tiếp với run mới.
- Cả 9 run trong `runs_retry1` chạy **không augmentation** (hash khớp `experiment_data.py.bak_20261113_1402`). `strong_train_aug` có trong `experiment_data.py` từ 13/11; mọi run mới ghi vào `runs_aug1`.
- Tầng 2: nhánh chính là **R0** (luật dữ liệu, bias 0, λ chọn trên val); so sánh chính là `R0 − C0`. Bỏ R\* (20/11). C1/R1 (ngưỡng chỉnh trên 3 subject val) chỉ là nhánh phụ. `R_FACS` (5 luật viết tay) trả lời Q2a. Giai đoạn thiết kế chạy `--val-only`.
- Khi theo dõi loss, đọc các cột thô `global_con`/`node_con`/`cross_con` trong `history.json`. Không đọc `total`, vì cột này có nhân hệ số ramp.

## 7. Việc đang mở (cập nhật 20/11/2026)

`splits_v2` đã sinh xong ngày 20/11.

1. ~~Đóng gói ảnh~~ **XONG 21/11**: 27 tar trong `MyDrive/Sym/DISFA_img/` cùng `DONE.json`, tổng 8,65 GB, chạy hết 100 phút. Không chạy lại cell A.
2. Pilot `baseline` + `full`, fold 2, `runs_aug1`: cell B, kiểm bằng cell C. Phiên đầu (21/11) chạy trên torch 2.14.1+cu130, torchvision 0.29.1, GPU L4, 12 CPU; Setup giải nén ảnh mất 5,6 phút. Nếu resume bị từ chối vì torch đổi phiên bản, cài lại đúng bản này trước khi resume.
3. Sau pilot: chạy Tầng 2 chỉ trên val (cell D) cho fold 2. Phần code Tầng 2 đã thống nhất xong ngày 20/11.
4. Song song với pilot: cell E chạy thử `stage2_decoder` trên 6 run cũ (baseline/full × 3 fold), chỉ val, trên phiên CPU với Setup `IMAGES = False`. Mục đích là kiểm module chạy được trên dữ liệu thật; số liệu thu được không dùng để báo cáo.
5. Tuỳ chọn, cho Q2b: Stage 2/3 của SymGraphAU trong `MultiviewSymAU-master/` (đồ thị CNF + GCN) còn lỗi sigmoid hai lần và mất dấu phủ định. Đây không phải phương pháp chính; nếu không kịp thì Q2b trả lời bằng lập luận (đề cương §1).
