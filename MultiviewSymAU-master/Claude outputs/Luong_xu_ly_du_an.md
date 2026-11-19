# Luồng xử lý dự án — Tầng 1 Contrastive → Tầng 2 Luật

> **Tài liệu sống.** Cập nhật mỗi khi có kết quả mới.
> Cập nhật lần cuối: **13/11/2026** · Trạng thái: *tuần 1, đã có đủ val npz, đang chẩn đoán quá khớp*

---

## 1. Định vị và mục tiêu (chốt 12/11/2026)

**Hướng:** cải tiến để **vượt điểm paper**. **Thời hạn:** 6 tuần. **Mục tiêu:** vượt **62.94** macro-F1 DISFA (SymGraphAU, Ours-CNN).

**Mâu thuẫn phải giải:** Swin-B là backbone của chính paper, khớp giao thức là vệ sinh, sửa chọn checkpoint là sửa lỗi — không cái nào là đóng góp. **Đóng góp thật chỉ còn contrastive (Tầng 1) và decoder (Tầng 2).** Mà hiện tại contrastive đang **âm** và đang **phân kỳ** (mục 7).

---

## 2. Điểm xuất phát

Seed 0, ResNet-50, 3 fold: baseline **60.53** · current 59.06 · full 60.23. Cần bù **2.41 điểm**.

Biên độ giữa fold 12.84 so với giữa variant 1.47 — gấp 8.7 lần. Với 1 seed chưa kết luận được gì về contrastive.

---

## 3. CHẨN ĐOÁN CHÍNH (13/11) — quá khớp là gốc của mọi vấn đề

### 3.1 Loss contrastive đang phân kỳ

| variant | train ep1 | train ep20 | |
|---|---|---|---|
| baseline | 0.0413 | 2.10e-05 | giảm **1964×** |
| current | 0.4690 | 0.826 | **TĂNG 1.8×** |
| full | 0.0849 | 0.223 | **TĂNG 2.6×** |

Với weight 0.1, SupCon (~4–8 ở τ=0.1) đóng góp 0.4–0.8 vào tổng trong khi `wa` chỉ ~0.03 — **contrastive áp đảo tổng gấp 10–25 lần mà vẫn tăng**. Gradient của detector loss thắng trên backbone, ép học thuộc; học thuộc phá cấu trúc cụm mà SupCon cần. Hai mục tiêu đánh nhau.

### 3.2 Quá khớp hoàn toàn ở baseline

`wa` đi từ 0.0343 xuống **1.64e-05** — học thuộc 72.674 frame. Val macro-F1 đỉnh ở epoch 1–2 rồi tụt suốt 19 epoch, và cách test 16.3 điểm.

**Kết luận: `best_epoch ≤ 2` và `full − baseline = −0.31` là CÙNG MỘT vấn đề.** Tune η/ζ/τ trước khi chặn quá khớp là đốt GPU.

### 3.3 Đính chính — inner val chỉ hỏng ở fold 1

| fold | AU hiếm nhất | đánh giá |
|---|---|---|
| 1 | AU1=73, **AU2=4**, AU9=139 | hỏng |
| 2 | tối thiểu 952 | ổn |
| 3 | tối thiểu 293 | ổn |

Đổi tiêu chí chọn chỉ đổi **3/9 run, đều ở fold 1**. Không đủ lý do train lại tất cả vì lý do này.

### 3.4 Lỗ hổng cấu hình: không variant nào có node_w CÙNG queue

| variant | global_w | node_w | cross_w | queue | warmup |
|---|---|---|---|---|---|
| baseline | 0 | 0 | 0 | 0 | 3 |
| current | 0.1 | **0** | 0.1 | **2048** | 1 |
| full | 0.05 | **0.05** | 0.05 | **0** | 3 |

Memory queue sinh ra để bù positive cho AU hiếm ở contrastive mức node — mà `full` (có node_w) lại tắt queue. Log của `full` xác nhận: `node_eligible_batch_fraction` cho AU2 chỉ **0.218**, AU9 **0.278**. Tổ hợp `full + queue` chưa ai thử.

---

## 4. Ngân sách GPU — mở khoá lớn

**33% thời gian bị đốt vào epoch 1 đọc ảnh từ Drive:**

| | epoch 1 | epoch 2–20 | tổng |
|---|---|---|---|
| fold1 | 2.36h | 31 phút | 12.2h |
| fold2 | **10.47h** | 20 phút | 16.7h |
| fold3 | 3.73h | 53 phút | 20.6h |

16.6h / 49.5h chỉ để đọc ảnh lần đầu.

Tài nguyên bỏ không: VRAM đỉnh **3.3 / 15 GB** (batch 32 phí 78%); `/content` còn **236 GB** trong khi ảnh DISFA chỉ ~3–5 GB.

| Cấu hình | giờ/run | run trong 315h |
|---|---|---|
| hiện tại (Drive, batch 32, workers 2) | 16.5 | 19 |
| ảnh ở đĩa cục bộ | 11.5 | 27 |
| + workers 8 + batch 128 | **~4** | **~80** |

---

## 5. Lịch — thứ tự ĐÃ ĐỔI (Swin-B tụt xuống thứ 5)

### Tuần 1

- [x] Xác nhận `--split {val,test}` hợp lệ
- [x] Sửa `DATA` → `/content/drive/MyDrive/Sym/DISFA`
- [x] **Sinh `val_predictions.npz` cho 9 run** — xong, Tầng 2 hết bị chặn
- [x] Chẩn đoán `best_epoch` → trả lời: quá khớp, không phải lỗi code
- [ ] Chẩn đoán A/B/C/D (notebook `Chan_doan_khong_can_GPU.ipynb`)
- [ ] **Tăng tốc**: chép ảnh về `/content`, `--workers 8`, `--batch-size 128`
- [ ] **Chặn quá khớp**: augmentation + scheduler + weight decay

### Tuần 2
- [ ] Xác nhận contrastive hết phân kỳ (1 fold, variant `full`, sau khi sửa)
- [ ] Sweep contrastive — ưu tiên tổ hợp `node_w > 0` **cùng** `queue > 0`

### Tuần 3–4 — ablation R50 + headline Swin-B
### Tuần 5 — Stage 2 decoder, bảng kết quả, bootstrap CI
### Tuần 6 — viết và dự phòng

**Stage 2 decoder chạy được ngay bây giờ**, không cần chờ gì — val npz đã đủ.

---

## 6. Quyết định đã chốt

| Ngày | Quyết định |
|---|---|
| 02/11 | Tầng 2 chỉ chạy 2.1 + 2.2 + decoder MAP. Hoãn neural gate, `L_rule`, ablation A–E |
| 02/11 | Một thang trọng số chung: `w0 = −log UCB95(lift_v)`, chặn trên 5 |
| 03/11 | Pipeline chuẩn `Contrastive_V1`. Không cần train lại Stage 1 cho decoder |
| 12/11 | Định vị: vượt điểm paper. Bỏ variant `current` khỏi kế hoạch chạy |
| 12/11 | Hoãn trục AU–Biểu cảm, chỉ giữ chứng minh tính vòng tròn |
| **13/11** | **Ưu tiên lại: chặn quá khớp TRƯỚC, Swin-B tụt xuống thứ 5** |
| **13/11** | **Không sweep η/ζ/τ cho tới khi contrastive hết phân kỳ** |

---

## 7. Phát hiện kỹ thuật

### Về paper SymGraphAU

- **Trục AU–Biểu cảm rỗng thông tin**: `H(biểu cảm | AU) = 0 bit`, kiểm trên cả 256 cấu hình. Áp dụng cho cả DISFA lẫn BP4D vì cả hai dùng nhãn giả từ `M_AE` viết tay.
- **Dấu phủ định không có trong đồ thị**: Fig. 4 chỉ có 4 loại nút, không có NOT, trong khi LENSR mà họ trích dẫn thì có.
- **Luật FACS sai chiều**: `¬AU1 ∨ ¬AU9` dùng làm loại trừ trong khi lift = 1.44.
- `G^us` vẫn thoả CNF 11–19% frame neutral; ~2% frame CNF rỗng.
- Báo cáo 1 lần chạy, không seed, không CI. EKR đóng góp +0.9 điểm.
- Swin-B: DISFA +1.5 nhưng BP4D **−0.5** — hiệu ứng riêng tập dữ liệu.

### Về code — đã vá

| Lỗi | |
|---|---|
| `gcn_forward_double` đặt bias sai (lệch 21% tại nút Global) | vá 24/08 |
| `build_cnf_batch` ThreadPool phá RNG determinism | vá 24/08 |
| Cell `RUN_OUTER_TEST` thiếu `--split` | phát hiện 12/11 |
| Notebook v1: `DATA` thừa cấp `data/` | vá, bản v2 |
| Notebook v1: `help %` sai ~8 lần | vá |
| Notebook v1: cộng `net` qua các cặp là đếm trùng | vá — dùng frame phân biệt |
| Notebook v1: phán NO-GO khi 0 dữ liệu | vá |

### Về code — còn (pipeline cũ `MultiviewSymAU-master`, không dùng nữa)

`sigmoid` áp hai lần ở Stage 2/3 · `test.py` import sai model

### Điểm tốt của `Contrastive_V1`

- `load_manifest` cưỡng chế: hash từng file `list/`, hash `M_AE`, `fit`/`val` phân hoạch rời rạc, không chồng subject, WAL chỉ tính từ `fit`.
- Mỗi run lưu kèm `source/` — đúng phiên bản code đã chạy, truy vết được.
- Có `--resume` và `--stop-after N` (dừng sau epoch N không đổi schedule, resume từ `last.pt`) — giải quyết hẳn vấn đề ngắt phiên Colab.
- `--arc` hỗ trợ sẵn `swin_transformer_base/small/tiny`.
- `--variant` có `{baseline, global, node, cross, global_cross, full, current}` — cấu trúc ablation tách từng số hạng đã có sẵn.

---

## 8. Đường dẫn chuẩn

```
/content/drive/MyDrive/Sym/
    Contrastive_V1/   train_contrastive.py · experiment_data.py · summarize_experiments.py
    DISFA/            list/ (15 file) · img/ (27 subject × 4845 frame)    <-- data-root
    splits/           fold1.json fold2.json fold3.json
    runs_retry1/<variant>/fold<k>/seed<s>/
        best.pt · last.pt (486 MB mỗi cái) · history.json · config.json
        manifest.json · environment.json · source/ (bản sao code đã chạy)
        evaluation/      test_metrics.json · test_predictions.npz
        evaluation_val/  val_metrics.json  · val_predictions.npz
    runs_swin_probe/ · failed_runs/ · logs/ · reports/
```

Cấu hình đã chạy: `resnet50 · 20 epoch · batch 32 · workers 2 · lr 1e-4 · wd 5e-4 · τ=0.1 · γ=0.05 · proj_dim 128 · pretrained`

**Bắt buộc bật GPU.** CPU mất 4–8 giờ cho bước sinh val thay vì 15–30 phút.

---

## 9. Câu hỏi còn mở

1. ~~`best_epoch ≤ 2` do đâu?~~ → **đã trả lời: quá khớp**
2. Số hạng contrastive nào phân kỳ — `global_con`, `node_con`, hay `cross_con`? → mục A notebook chẩn đoán
3. Augmentation và scheduler hiện có gì? → mục B
4. `help`/`hurt` bằng bao nhiêu? → mục D
5. ~~`--arc` có tồn tại không?~~ → **có, hỗ trợ swin_transformer_base**
6. Swin-B có cho +1.5 trên setup này không? → cần run dò
7. Sau khi chặn quá khớp, contrastive có hết phân kỳ không? → quyết định có đóng góp hay không

---

## 10. Cách dùng tài liệu này

Sau mỗi mốc, cập nhật mục 5 (tick), 6 (quyết định mới), 7 (phát hiện mới), 9 (đóng/mở câu hỏi). Mở phiên mới thì đọc mục 1, 3, 5, 9 là đủ ngữ cảnh.

Tài liệu liên quan: `Kien_truc_pipeline_Stage1_Stage2.md` · `Stage2_decoder_remote_status.md` · `SymGraphAU_code_vs_paper_review.md` · `SymGraphAU_stage2_logic_check.md`
