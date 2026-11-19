# Đề cương nghiên cứu — Nhận diện AU bằng luật rút từ dữ liệu

**Dữ liệu:** DISFA · 8 AU (1, 2, 4, 6, 9, 12, 25, 26) · 27 subject · 130.815 frame
**Giao thức:** 3-fold chia theo subject; mỗi fold giữ 3 subject làm inner val; outer test không chạm khi chọn tham số
**Đối chứng:** SymGraphAU (Pattern Recognition 165, 2025) — **62.94** macro-F1 trên DISFA, backbone ResNet-50
**Ngày lập:** 13/11/2026

---

## 1. Câu hỏi nghiên cứu

SymGraphAU đề xuất đưa tri thức ký hiệu (luật FACS về quan hệ AU–AU và AU–Biểu cảm) vào mạng nơ-ron qua một đồ thị CNF được nhúng bằng GCN. Nghiên cứu này đặt ba câu hỏi:

**Q1.** Cơ chế đó có hoạt động đúng như mô tả không?
**Q2.** Nếu thay tri thức viết tay bằng **luật rút từ dữ liệu có kiểm định thống kê**, và thay nhúng xấp xỉ bằng **suy luận chính xác**, kết quả có tốt hơn không?
**Q3.** Ràng buộc logic đóng góp bao nhiêu khi dữ liệu có nhãn đầy đủ?

---

## 2. Những gì đã xác định được

Ba phát hiện dưới đây **đã đo xong**, không còn là giả thuyết.

### 2.1 Trục AU–Biểu cảm của paper rỗng thông tin

DISFA và BP4D đều không có nhãn biểu cảm. Paper sinh nhãn giả bằng `k_e = argmax(Y_a · M_AE)` với `M_AE` là ma trận viết tay. Kiểm trên cả 256 cấu hình AU: mỗi cấu hình cho đúng một biểu cảm, tức **`H(biểu cảm | AU) = 0 bit`**.

Hệ quả: mọi luật "AU → Biểu cảm" rút từ nhãn giả chỉ là viết lại `M_AE`. **Đóng góp được tuyên bố là trung tâm của paper không mang thông tin nào ngoài giả định ban đầu**, ở cả hai tập dữ liệu họ báo cáo.

### 2.2 Dấu phủ định không được mã hoá trong đồ thị

Fig. 4 của paper liệt kê đúng bốn loại nút — Global, OR, AND, Leaf — không có NOT, trong khi LENSR mà họ trích dẫn thì có. Nếu đúng vậy, GCN không phân biệt được `[¬AU2, ¬AU6]` với `[AU2, AU6]`, và về nguyên tắc không thể suy ra một mệnh đề thoả hay không.

Trong bản cài đặt lại còn phát hiện: `G^us` vẫn thoả CNF ở 11–19% frame neutral, và ~2% frame có CNF rỗng — tức nhãn "không thoả" bị gán sai.

### 2.3 Luật FACS viết tay có cái sai chiều

`¬AU1 ∨ ¬AU9` được dùng làm luật loại trừ, trong khi dữ liệu cho **lift = 1.44** — hai AU này liên hệ **dương**. Đây là thứ mà bất kỳ bước kiểm định nào cũng loại ngay.

---

## 3. Pipeline

Hai nhánh độc lập, gặp nhau ở bước cuối.

### Tầng 1 — Bộ nhận diện AU (có huấn luyện)

| Khối | Vào | Ra |
|---|---|---|
| Backbone ResNet-50 / Swin-B | ảnh `3×224×224` | `(B, 49, 2048)` → `global_linear` → `(B, 49, 512)` |
| AU head — 8 nhánh Conv1D độc lập | `(B, 49, 512)` | `V_a (B, 8, 256)` · `p_a (B, 8)` |
| Expr head — 7 nhánh | `(B, 49, 512)` | `V_e (B, 7, 256)` · `p_e (B, 7)` |
| Projection heads (chỉ lúc train) | `V_a, V_e` | `z_g (B,128)` · `z_a (B,8,128)` · `z_e (B,7,128)` |

**Hàm mất mát:** `L = L_wa + 0.05·L_we + η·L_global + ν·L_node + ζ·L_cross`, hệ số contrastive ramp tuyến tính trong `warmup` epoch đầu. LR cosine giảm về 0 qua `epochs`.

**Variant:** `baseline` (η=ν=ζ=0) · `global` · `node` · `cross` · `global_cross` · `current` · `full`.

### Tầng 2a — Rút luật (từ nhãn `fit`, không qua mạng)

Ứng viên: 28 cặp loại trừ `¬AU_i ∨ ¬AU_j` + 56 kéo theo có hướng `AU_i → AU_j`.

Bootstrap **1000 lần theo subject** — không theo frame, vì DISFA chỉ có 27 subject và frame liên tiếp cách nhau 50 ms. Cả hai loại luật dùng chung một thước đo: **lift của mẫu vi phạm** (loại trừ → `(i=1,j=1)`, kéo theo → `(i=1,j=0)`).

Nhận khi `UCB95(lift_v) < 0.8`, `E ≥ 50`, ≥70% subject cùng chiều; kéo theo thêm `LCB95(P(j|i)) ≥ 0.6`. Trọng số `w0 = −log UCB95(lift_v)`, chặn trên 5.

Lọc tiếp bằng `help − hurt` trên inner val: bỏ luật mà việc áp dụng phá nhiều hơn sửa.

### Tầng 2b — Giải mã MAP chính xác

```
y* = argmax_y  Σ_i [ y_i·log p_i + (1−y_i)·log(1−p_i) ]  −  λ·Σ_r w_r·V[y,r]
```

Duyệt hết `2⁸ = 256` cấu hình mỗi frame. Không SAT solver, không GCN, không xấp xỉ.

Bốn nhánh: **C0** ngưỡng 0.5 · **C1** ngưỡng riêng từng AU chọn trên val · **R0** có luật, bias 0 · **R1** có luật, bias = ngưỡng C1. `λ` chọn trên val với **λ=0 nằm trong lưới**, nên luật tự tắt nếu vô ích.

---

## 4. Đầu ra

### Tệp sinh ra

| Tệp | Nội dung |
|---|---|
| `runs/<variant>/fold<k>/seed<s>/best.pt`, `last.pt` | checkpoint |
| `history.json` | loss tách theo số hạng + val metrics mỗi epoch |
| `config.json`, `manifest.json`, `environment.json`, `source/` | truy vết đầy đủ |
| `evaluation/test_metrics.json`, `test_predictions.npz` | kết quả outer test |
| `evaluation_val/val_metrics.json`, `val_predictions.npz` | dùng chọn ngưỡng và λ |
| `stage2_out/rules_fold{k}.csv` | `luật \| loại \| lift \| CI95 \| E \| %subject \| help \| hurt \| w0 \| facs_prior` |
| `stage2_out/V_fold{k}.npy` | ma trận vi phạm `(256 × R)` |
| `stage2_out/stage2_results.json` | kết quả 4 nhánh + CI |

### Bảng và hình cho luận văn

1. **Bảng chính** — macro-F1 và F1 từng AU, mean 3 fold theo seed → mean ± SD qua seed, cho từng variant
2. **Bảng ablation Tầng 1** — baseline / global / node / cross / full
3. **Bảng Tầng 2** — C0, C1, R0, R1, R*, kèm Δ và CI bootstrap theo subject
4. **Bảng luật** — luật nào qua kiểm định ở fold nào, đối chiếu cột `facs_prior`
5. **Hình pipeline** — hai nhánh và điểm hợp lưu
6. **Hình chẩn đoán** — quỹ đạo `train.wa` và val macro-F1 theo epoch, trước và sau khi sửa augmentation

---

## 5. Outcome mong muốn

### Mức tối thiểu — phần lớn đã đạt

- Chứng minh được ba lỗi ở §2, có số liệu
- Thay thế bằng phương pháp có kiểm định thống kê và suy luận chính xác
- Báo cáo đạt chuẩn: nhiều seed, CI bootstrap theo subject, không rò rỉ outer test, mọi siêu tham số chọn trên inner val

Phần này **không phụ thuộc vào việc có vượt điểm paper hay không**, và tự nó đã là đóng góp đứng được.

### Mức kỳ vọng

| Mục tiêu | Trạng thái |
|---|---|
| Contrastive cải thiện so với baseline của chính mình | **chưa đạt** — hiện `full − baseline = −0.31` |
| Tầng 2 cho +0.2 đến +0.5 điểm, CI không chứa 0 | **chưa đạt** — hiện R0−C0 = +0.32 nhưng CI chứa 0 |
| Loại bỏ quá khớp: `train.wa` không về 1e-05, val đỉnh dịch khỏi epoch 1 | **đang kiểm** |

### Mức tốt nhất

Vượt **62.94** macro-F1 trên DISFA. Khoảng cách hiện tại là **2.41 điểm** (baseline tốt nhất đang ở 60.53).

| Nguồn bù điểm | Thấp | Cao |
|---|---|---|
| Sửa quá khớp (augmentation) | +0.5 | +2.0 |
| Khớp giao thức train (refit trên `fit + inner val`) | +0.5 | +1.5 |
| Đổi sang Swin-B | +1.0 | +1.5 |
| Tune contrastive sau khi hết quá khớp | 0.0 | +0.5 |
| Tầng 2 decoder | +0.25 | +0.5 |

Kịch bản xấu đưa tới 62.78 (thiếu 0.16), kịch bản tốt tới 66.5.

**Một cảnh báo phải nêu trong luận văn:** đòn bẩy chắc nhất (Swin-B) là backbone của chính paper; khớp giao thức là vệ sinh nghiên cứu; sửa quá khớp là sửa lỗi. Đóng góp thật chỉ còn **contrastive ở Tầng 1** và **decoder ở Tầng 2**. Nếu vượt được 62.94 mà toàn bộ mức vượt đến từ Swin-B, hội đồng sẽ hỏi đúng câu đó.

---

## 6. Tiêu chí thành công và tiêu chí dừng

**Tầng 2 được đầu tư tiếp** chỉ khi `R*` thắng **cả `C0` lẫn `C1`**, với CI bootstrap theo subject không chứa 0.

> Đo lần đầu (13/11, trên Tầng 1 chưa sửa): **TRƯỢT**. R* thua C0 ở 2/3 variant, mọi CI của `R*−C0` đều chứa 0. So sánh sạch `R0 vs C0` cho +0.32 điểm trung bình, không tách được khỏi nhiễu.
>
> Phép đo này đặt trên một detector đang quá khớp nặng, nên phải **đo lại sau khi sửa Tầng 1** trước khi kết luận.

**Contrastive được coi là có đóng góp** khi `full − baseline > 0` với CI không chứa 0, trên ít nhất 2 seed.

**Kết quả âm vẫn là kết quả.** Nếu sau khi sửa Tầng 1 mà luật vẫn không giúp, kết luận đúng là: *trên tập nhãn đầy đủ, ràng buộc logic gần như dư thừa vì `w_r = 0` ở mọi frame mà nhãn thật vi phạm — loss về bản chất chỉ là trọng số hoá lại false positive*. Khi đó giá trị thật của luật nằm ở **chế độ ít nhãn**, và đó là thí nghiệm nên làm tiếp nếu còn thời gian.

---

## 7. Rủi ro

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Augmentation không chặn được quá khớp | cao | chạy thử 1 fold trước khi train toàn bộ |
| Swin-B không cho +1.5 trên setup này | cao — paper đo BP4D **−0.5** | dò 1 fold trước khi đặt cược |
| Biên độ giữa fold (12.84) gấp 8.7 lần giữa variant (1.47) | cao | so sánh theo cặp trong từng fold, nhiều seed |
| Inner val fold 1 quá thưa (AU2 có 4 frame dương / 14.535) | trung bình | loại AU dưới ngưỡng mẫu khỏi tiêu chí chọn checkpoint |
| Ngân sách GPU Colab | trung bình | chép ảnh về `/content`; dùng `--stop-after` + `--resume` |

---

*Tài liệu liên quan trong project: `Luong_xu_ly_du_an.md` (theo dõi tiến độ), `Kien_truc_pipeline_Stage1_Stage2.md` (kiến trúc chi tiết), `SymGraphAU_code_vs_paper_review.md` và `SymGraphAU_stage2_logic_check.md` (đối chiếu với paper).*
