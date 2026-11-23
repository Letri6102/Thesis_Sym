# Thesis_Sym — quy tắc làm việc

Luận văn: nhận diện AU trên DISFA. Tầng 1 là detector contrastive, Tầng 2 là luật rút từ dữ liệu cộng giải mã MAP chính xác.
Kế hoạch nghiên cứu và trạng thái hiện tại nằm ở [De_cuong_nghien_cuu.md](De_cuong_nghien_cuu.md). Đọc file đó trước khi đề xuất thí nghiệm.
Baseline SymGraphAU, các số cần vượt và các hướng cải tiến đã đề xuất nằm ở [CONTEXT.md](CONTEXT.md).
Mốc chính trên DISFA:
- ResNet-50: **63.0** (62.94). Paper: JFL 62.1, Baseline 61.2.
- Swin-B: **64.5**.

Mỗi backbone phải so với số của chính backbone đó.

## 1. Quy tắc bắt buộc (quyết định của chủ dự án, 20/11/2026)

- **Không train, evaluate hay chạy Tầng 2 ở máy local.** Mọi thứ đụng tới dataset hoặc GPU đều chạy trên **Google Colab** qua notebook.
  - **Ngoại lệ (chủ dự án mở 23/11):** được **train** run của bộ 18 run trên máy GPU riêng (RTX 5090, remote), chỉ bằng `pc/train_18_run.py` (hướng dẫn `pc/README.md`), cùng khoá. Kết quả nhập vào Drive bằng cell I. Evaluate, Tầng 2 và mở test vẫn chỉ trên Colab. Máy dev hiện tại (RTX 5060 Ti) không phải máy train.
  - Máy 5090 chỉ được dùng thư mục `D:\LVTS_TriLe`, truy cập qua UltraViewer, không đăng nhập Drive. Gói chép sang dựng bằng `pc/make_bundle.py`, ảnh lấy từ `img/` ở máy cá nhân. **Chỉ train khi cell J báo `KHỚP 27/27`** (SHA-256 tar so với `DISFA_img` trên Drive; số ghi ở `pc/bundle_tar_sha256.json`).
  - Cách đang dùng (23/11): máy 5090 có sẵn dữ liệu ở `D:\KLTN_Khang\DISFA_Data` (thư mục của người khác, **chỉ đọc**). `pc/windows/train_data_co_san.bat` → `from-folder`: kiểm list (`pc/data_reference.json`), tạo lại `splits_v2` (trùng byte), chép ảnh vào `D:\LVTS_TriLe\DISFA` và kiểm SHA-256 từng subject; lệch là dừng.
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
runs_aug2/                   B2: aug + EMA + val dày + LR backbone, chỉ fold 2 seed 0 — thử nghiệm, không thuộc 18 run
stage2_aug2/                 cell D2: Tầng 2 chỉ val trên runs_aug2
runs_refit1/                 dòng (b) refit: <variant>/fold{k}/seed{s}/ (last.pt = model, refit.json = E và nguồn) + manifests/
runs_swin_probe/             dò Swin-B (đang rỗng)
stage2_v2/                   Tầng 2 chuẩn (stage2_decoder): rules/fold{k}/, results/<variant>/fold{k}/seed{s}/, summary.json
contrast_ci/summary.json     CI full − baseline trên test (stage2_decoder/contrast_ci.py) — chỉ tạo một lần, khi mở test
stage2_out/                  Tầng 2 bản notebook 13/11 trên detector cũ — chỉ là thăm dò
stage2_smoke_retry1/         cell E: chạy thử stage2_decoder trên runs_retry1 + splits cũ, CHỈ VAL — kiểm module, không báo cáo
upload/                      zip run train trên máy riêng chờ cell I nhập; đã xử lý → upload/done|duplicate|rejected/
failed_runs/, logs/, reports/
Contrastive_V1/checkpoints/resnet50-19c8e357.pth   trọng số ImageNet
```

**Repo GitHub** (chỉ có code và tài liệu):

```text
Contrastive_V1/              Tầng 1 (train_contrastive.py là entry point) + stage2_decoder/ (Tầng 2 chuẩn, test: tests/test_stage2.py)
Contrastive_V1/refit/        dòng (b) refit: gọi train_contrastive đã khoá với manifest gộp fit + val (test: tests/test_refit.py)
Contrastive_V1/remote_stage2/ script cho máy remote (runs_retry1, splits cũ) — LỖI THỜI, trái quy tắc chỉ chạy Colab
MultiviewSymAU-master/       bản cài đặt lại SymGraphAU (Stage 2/3 cũ) để đối chiếu
Tang2_Decoder.ipynb          Tầng 2 bản notebook 13/11 — LỖI THỜI, lệch đề cương; dùng stage2_decoder
De_cuong_nghien_cuu.md       đề cương bản 2
notebooks/                   Train_18_run.ipynb (Setup + cell G), Nhap_run_PC.ipynb (Setup + cell I), Bao_cao_log.ipynb (cell F) — bản đang dùng
pc/                          train trên máy GPU riêng: setup_env.py, train_18_run.py (prepare/check/smoke/train/status), make_bundle.py + windows/*.bat (gói một thư mục), README.md
CONTEXT.md                   baseline SymGraphAU (paper), số cần vượt, đối chiếu pipeline, hướng cải tiến
```

`Contrastive_V1/COLAB_PRO.md` và `Contrastive_V1/notebooks/Colab_Contrastive_V1.ipynb` đã **lỗi thời**: hai file này giả định code nằm ở `MyDrive/SymAU` trên Drive. Dùng mục 5 thay thế.

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
- **3 fold ngoài là fold GỐC** (kiểm 23/11): test fold 1 = part3, fold 2 = part2, fold 3 = part1 của code gốc `Contrastive_V1/tool/DISFA_image_label_process.py` (ME-GraphAU, cùng cách chia paper dùng). Ta chỉ tự chia **inner val** (3 subject lấy từ 18 subject train của mỗi fold) để chọn checkpoint mà không nhìn test; tập test không đổi. Train trên đủ 18 subject như paper là dòng (b) refit.
- Inner val (20/11): `splits_v2`, sinh bằng `experiment_data.py --split-seed 2026 --val-subjects 3 --min-val-positive 100`. Fold 2/3 trùng bản cũ, fold 1 rút lại, vì val cũ có AU2 = 4 frame và AU1 = 73 frame, tất cả từ một subject. Val fold 1 mới là SN002, SN016, SN032. **Chủ dự án đã chọn giữ `splits_v2`** dù fit fold 1 mất khoảng 45% số frame AU1/AU2/AU26 (AU2 còn 1098); đã cân nhắc mức trần "val ≤ 1/3" và không chọn. Không tự sinh lại split.
- Bỏ variant `current` (12/11). Các variant chính: `baseline` và `full`.
- Mọi lựa chọn (checkpoint, ngưỡng, λ, luật) **chỉ** dựa trên inner val. Outer test chỉ mở **một lần**, sau khi đã chốt thiết kế.
- Test đã bị mở trên detector cũ (29/10 cho Tầng 1, 13/11 cho Tầng 2). Các số đó chỉ là **thăm dò**; không so trực tiếp với run mới.
- Cả 9 run trong `runs_retry1` chạy **không augmentation** (hash khớp `experiment_data.py.bak_20261113_1402`). `strong_train_aug` có trong `experiment_data.py` từ 13/11; mọi run mới ghi vào `runs_aug1`.
- Tầng 2: nhánh chính là **R0** (luật dữ liệu, bias 0, λ chọn trên val); so sánh chính là `R0 − C0`. Bỏ R\* (20/11). C1/R1 (ngưỡng chỉnh trên 3 subject val) chỉ là nhánh phụ. `R_FACS` (5 luật viết tay) trả lời Q2a. Giai đoạn thiết kế chạy `--val-only`.
- **KHOÁ cấu hình Tầng 1 (chủ dự án chốt 22/11):** đúng cấu hình pilot `runs_aug1`.
  - ResNet-50, aug mạnh, lr 1e-4, wd 5e-4, 20 epoch, batch 32, workers 4; không EMA, val 1 lần/epoch, LR backbone ×1.
  - Chỉ `baseline` và `full`; 3 fold × seed 0/1/2 = 18 run, train bằng cell G (cell tự kiểm khoá code, manifest, thư viện).
  - **Không sửa hay push file bị hash, không tune, không thêm variant cho tới khi đủ 18 run.**
  - Phân công 23/11: máy RTX 5090 train **seed 2**, Colab train seed 0 và 1 (cell G với `SEEDS = [0, 1]`). Hai variant của cùng một seed phải train trên cùng một máy.
  - Phải công bố trong luận văn: seed 2 train trên RTX 5090, seed 0/1 trên L4; pilot trượt tiêu chí "đỉnh sau epoch 2"; việc khoá diễn ra sau khi đã nhìn val fold 2 (chứa subject test của fold 1/3); aug được chọn 13/11, sau lần mở test thăm dò 29/10.
- **Dòng (b) refit (chủ dự án chốt 23/11):** train lại mỗi run (a) đã xong trên **đủ 18 subject** outer-train (fit + inner val), cùng code và cấu hình đã khoá; lịch cosine 20 epoch giữ nguyên và dừng sau epoch E mà run (a) chọn trên inner val; model là `last.pt`; trọng số lớp của L_wa tính lại trên 18 subject. Code: `Contrastive_V1/refit/` (ngoài file bị hash), chạy bằng cell K (Colab) hoặc `pc/train_18_run.py refit`; kết quả ở `runs_refit1/`. Chỉ dòng (b) đặt cạnh 63.0; dòng (a) vẫn là kết quả chính.
- Khi theo dõi loss, đọc các cột thô `global_con`/`node_con`/`cross_con` trong `history.json`. Không đọc `total`, vì cột này có nhân hệ số ramp.

## 7. Việc đang mở (cập nhật 22/11/2026)

`splits_v2` đã sinh xong ngày 20/11.

1. ~~Đóng gói ảnh~~ **XONG 21/11**: 27 tar trong `MyDrive/Sym/DISFA_img/` cùng `DONE.json`, tổng 8,65 GB, chạy hết 100 phút. Không chạy lại cell A.
2. ~~Pilot~~ **XONG 22/11** (`runs_aug1`, fold 2, seed 0; torch 2.14.1+cu130, torchvision 0.29.1, numpy 2.4.6 — resume bị từ chối vì phiên bản thì cài lại đúng các bản này; L4, 2,5 giờ/run): **đạt 2/3 tiêu chí, tức là KHÔNG đạt** theo tiêu chí đã đặt.
   - Đạt: `wa` cuối 0.0059 / 0.0072; val tốt nhất không thấp hơn run cũ. Trượt: đỉnh val ở **epoch 1** ở cả hai variant.
   - Số val fold 2, cuối epoch (baseline / full):

     | | tốt nhất | TB epoch 11–20 | epoch 20 |
     |---|---|---|---|
     | run cũ (không aug) | 71.5 / 70.5 | 68.1 / 66.2 | 68.5 / 67.0 |
     | pilot (aug mạnh) | 76.8 / 74.9 | 62.4 / 70.1 | 60.5 / 69.5 |
     | B2 (aug + EMA + LR) | 68.1 / 67.4 | 64.6 / 61.6 | 64.0 / 61.3 |

   - Dấu của `pilot − cũ`, `B2 − pilot`, `full − baseline` **đổi theo cách đo và theo variant**. 1 fold + 1 seed không xếp hạng được cấu hình; không gọi chênh vài điểm là cải thiện.
   - Val fold 2 (3 subject) **không so được** với 63.0 của paper (test, 3 fold).
3. **ĐANG LÀM: train nốt 16 run.** Colab chạy seed 0/1 (cell G, `notebooks/Train_18_run.ipynb`); máy 5090 chạy seed 2 (`pc/train_18_run.py`; chưa chạy smoke trên máy đó), nhập về Drive bằng cell I. Gói `C:\CODE\Symbolic\LVTS_TriLe` dựng 23/11 từ code 4e610e2; chờ cell J. Đủ 18 run thì chạy Tầng 2 chỉ val (cell D, 3 fold × 3 seed).
4. ~~Cell E~~ **XONG 21/11**: `stage2_decoder` chạy đúng trên 6 run cũ, chỉ val, mất khoảng 30 giây. Kết quả tóm tắt nằm ở đề cương §6 và không dùng để báo cáo. Trên val, `R0 − C0` chỉ từ +0.0 đến +0.9, trong khi C1 − C0 từ +1.5 đến +10.6.
5. **Gói EMA + val 4 lần/epoch + LR backbone 0.1×** (code trên `main` từ `586b221`, cờ mặc định tắt): **không có bằng chứng có lợi, đề xuất không đưa vào cấu hình chính.** EMA và LR chưa tách được.
   - D2 (22/11): C0 khớp lúc train (68.53 / 68.16); C1 72.27 / 72.00 (tune trên chính val); `R0 − C0` = +0.26 / +0.04.
   - AU6 không cứu được bằng ngưỡng chung (42.1 → 42.2 / 41.7 → 45.5), và bị dự đoán **thừa** 1.5–1.65 lần. Chưa chứng minh "mất đặc trưng do aug".
   - Đính chính 21/11: mức "AU6 tụt 20 điểm" là so với epoch đẹp nhất của run cũ (AU6 run cũ dao động 32.5–63.8 theo epoch); so trung vị thì 10.6 / 11.9 điểm.
6. Tuỳ chọn, cho Q2b: Stage 2/3 của SymGraphAU trong `MultiviewSymAU-master/` (đồ thị CNF + GCN) còn lỗi sigmoid hai lần và mất dấu phủ định. Đây không phải phương pháp chính; nếu không kịp thì Q2b trả lời bằng lập luận (đề cương §1).
7. **Rò rỉ khi chọn cấu hình chung trên val (phát hiện 22/11): khoá cấu hình để không rò thêm; phần đã nhìn phải công bố (§6).**
   - Val fold 2 = SN008 (test fold 1), SN016 và SN032 (test fold 3). Gộp val 3 fold thì chạm 2/9, 2/9, 3/9 subject test của fold 1, 2, 3.
   - Không chọn cấu hình trên val fold 2 hay val gộp rồi áp cho mọi fold. Chọn checkpoint, ngưỡng, λ riêng trong từng fold thì sạch.
8. **CHẶN mở test** cho tới khi xong các mục sau:
   - đủ 18 run;
   - ~~code CI cho `full − baseline`~~ **XONG 22/11**: `stage2_decoder/contrast_ci.py` (test: `tests/test_contrast_ci.py`). Không đọc file test nào nếu chưa đủ 18 run đúng khoá; `--check-only` (cell H) không bao giờ đọc test. Test `test_lock_matches_code_and_cell_g` đỏ nghĩa là file bị hash đã đổi: không push.
   - ~~chốt dòng (b) refit~~ **ĐÃ CHỐT + CODE 23/11** (cell K, `pc/train_18_run.py refit`). Còn thiếu trước khi mở test: (1) evaluate test cho `last.pt` của refit, viết cùng cell mở test; (2) Tầng 2 trên refit dùng λ và ngưỡng C1 lấy từ run (a) tương ứng (`stage2_decoder` chưa hỗ trợ λ cố định).
   - ~~quy tắc 3 seed~~ **ĐÃ CHỐT 22/11 (chủ dự án), trước khi mở test:** `full − baseline` chỉ ĐẠT khi thêm "Δ > 0 ở cả 3 seed" (`contrast_ci.contrast_verdict`; lý do: `stage2_decoder/sim_contrast_rule.py`, đề cương §6). `R0 − C0` giữ quy tắc cũ.
