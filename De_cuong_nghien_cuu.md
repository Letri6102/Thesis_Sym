# Đề cương nghiên cứu — Nhận diện AU bằng luật rút từ dữ liệu

**Dữ liệu:** DISFA · 8 AU (1, 2, 4, 6, 9, 12, 25, 26) · 27 subject · 130.815 frame (27 × 4.845)
**Giao thức:** 3-fold chia theo subject, mỗi subject nằm trong outer test đúng một lần. Trong outer train, 3 subject làm inner val (split seed 2026). Outer test chỉ mở một lần, sau khi đã chốt thiết kế.
**Đối chứng:** SymGraphAU (Pattern Recognition 165, 2025) — **62.94** macro-F1 trên DISFA, backbone ResNet-50.
> [cần kiểm] `paper.txt` trong thư mục dự án là bài FoolsGold, không phải SymGraphAU. Cần bản PDF để xác nhận con số 62.94, backbone, và cách paper chọn epoch/ngưỡng (xem §5, so sánh cùng giao thức).

**Ngày lập:** 13/11/2026 · **Bản 2:** 13/11/2026, sửa sau khi đối chiếu với code. **Cập nhật 19/11:** thêm §2.4 từ `history.json` của 9 run trên Drive; ghi bản sửa augmentation đã có trên Drive. Bản gốc: `De_cuong_nghien_cuu.v1.md`. Danh sách thay đổi ở §8.

---

## 1. Câu hỏi nghiên cứu

SymGraphAU đề xuất đưa tri thức ký hiệu (luật FACS về quan hệ AU–AU và AU–Biểu cảm) vào mạng nơ-ron qua một đồ thị CNF được nhúng bằng GCN. Nghiên cứu này đặt ba câu hỏi:

**Q1.** Cơ chế đó có hoạt động đúng như mô tả không?

**Q2.** Nếu thay tri thức viết tay bằng **luật rút từ dữ liệu có kiểm định thống kê**, và thay nhúng xấp xỉ bằng **suy luận chính xác**, kết quả có tốt hơn không? Q2 gộp hai thay đổi, nên tách để quy được nguyên nhân:
- **Q2a — nguồn luật:** cùng decoder MAP, luật FACS viết tay so với luật rút từ dữ liệu (nhánh `R_FACS` ở §3).
- **Q2b — cách suy luận:** cùng bộ luật, nhúng GCN so với MAP chính xác. Q2b cần chạy lại Stage 2/3 cũ, vốn đang có lỗi sigmoid hai lần và mất dấu phủ định (§2.2). Nếu không kịp, luận văn phải ghi rõ Q2b chỉ được trả lời bằng lập luận, không bằng số.

**Q3.** Ràng buộc logic đóng góp bao nhiêu khi dữ liệu có nhãn đầy đủ?

---

## 2. Những gì đã xác định được

Mỗi phát hiện ghi rõ mức chắc chắn:
- **đã chứng minh**: suy ra trực tiếp từ định nghĩa hoặc từ code;
- **đã đo**: có số liệu kèm CI;
- **suy luận**: đọc từ paper, chưa đối chiếu với code chính thức.

### 2.1 Nhãn biểu cảm giả là hàm tất định của nhãn AU — *đã chứng minh*

DISFA và BP4D không có nhãn biểu cảm. Nhãn giả được sinh bằng `k_e = argmax(Y_a · M_AE)`, riêng cấu hình toàn 0 được gán neutral. Trong code `train_contrastive.py`: `k = (labels @ matrix).argmax(1)` rồi `k[labels.sum(1) == 0] = 6`. Vì `k_e` là hàm tất định của `Y_a`, **`H(biểu cảm | AU) = 0` đúng theo định nghĩa**. Phép kiểm trên 256 cấu hình không chứng minh thêm điều gì, nó chỉ xác nhận argmax không có trường hợp hoà.

Những điều được khẳng định:
- Mọi luật "AU → Biểu cảm" rút từ nhãn giả chỉ là viết lại `M_AE`. Chúng vòng tròn và không kiểm chứng được bằng dữ liệu.
- Nhánh biểu cảm không mang thêm thông tin nhãn nào ngoài `Y_a` và `M_AE`.

Điều **không** được khẳng định là "trục AU–Biểu cảm vô dụng". `M_AE` là tri thức tiên nghiệm từ bên ngoài, và việc đưa nó vào mạng có giúp như một regularizer hay không là câu hỏi thực nghiệm. Chính Tầng 1 của nghiên cứu này cũng dùng `M_AE`: `L_we` có mặt ở mọi variant, `L_cross` ở `cross`/`full`. Vì vậy ablation `cross` so với `baseline` chính là phép đo câu hỏi đó.

Cách phát biểu trong luận văn: *trục AU–Biểu cảm không kiểm chứng được và không mang thông tin nhãn mới; lợi ích của nó, nếu có, đến từ tiên nghiệm `M_AE`*.

### 2.2 Dấu phủ định không được mã hoá trong đồ thị

- **Bản cài đặt lại — đã chứng minh từ code.** Xem `build_logic_graph_for_sample` trong `MultiviewSymAU-master/train_Sym_Stage_2.py`:
  - literal được tách thành `(name, neg)` nhưng `neg` không được dùng khi nối cạnh;
  - mỗi AU chỉ có một nút lá, dùng chung cho mọi mệnh đề;
  - ma trận kề là 0/1, không mang dấu;
  - đặc trưng của nút lá bằng giá trị gán × class center.

  Hệ quả là `¬AU2 ∨ ¬AU6` và `AU2 ∨ AU6` cho ra **cùng một đồ thị**.
- **Paper — suy luận.** Fig. 4 chỉ có bốn loại nút (Global, OR, AND, Leaf) và không có NOT, trong khi LENSR mà paper trích dẫn thì có. Cần đối chiếu với code chính thức của tác giả, nếu có, trước khi viết "paper bỏ phủ định". Nếu chưa đối chiếu được thì viết "Fig. 4 không thể hiện nút phủ định".
- **Bản cài đặt lại — đã đo** [ghi rõ fold và tập dữ liệu]. `G^us` (đồ thị "không thoả", sinh bằng cách lật một nghiệm thoả do Minisat tìm ra) vẫn thoả CNF ở 11–19% frame neutral, và ~2% frame có CNF rỗng. Ở những frame này, nhãn "không thoả" mà discriminator học là sai.

### 2.3 Luật viết tay `¬AU1 ∨ ¬AU9` không được dữ liệu ủng hộ

Số liệu trên outer train fold 2 (87.208 frame), lấy từ `au_pair_stats_DISFA_fold2.csv`: AU1 có 5.037 frame, AU9 có 4.186. Số frame đồng xuất hiện là O = 347, trong khi kỳ vọng nếu hai AU độc lập là E ≈ 242, nên **lift ≈ 1.44**. Đây là ước lượng điểm trên một fold và chưa có CI.

- **Phát biểu được ngay:** luật này **không qua** tiêu chí nhận ở §3, vốn đòi UCB95(lift) < 0.8.
- **Muốn nói "hai AU liên hệ dương / luật sai chiều"** thì cần LCB95(lift) > 1 theo bootstrap subject trên cả 3 fold. Lấy số này từ cột `lift_ci_lo` của `rules_all.csv`.

### 2.4 Tầng 1 hiện có quá khớp nặng; contrastive không phân kỳ — *đã đo*

Nguồn: `runs_retry1/{baseline,current,full}/fold{1,2,3}/seed0/history.json` (Drive, kiểm 19/11), 20 epoch, 1 seed. Cả 9 run dùng `experiment_data.py` **trước** bản sửa augmentation: hash trong `config.json → source_hashes` khớp với `experiment_data.py.bak_20261113_1402`, không khớp bản hiện tại.

**Contrastive không phân kỳ.** Các cột thô của `full` giảm đều, epoch 1 → 20:

| fold | `global_con` | `node_con` | `cross_con` |
|---|---|---|---|
| 1 | 2.70 → 2.41 | 1.52 → 1.08 | 1.16 → 1.08 |
| 2 | 2.76 → 2.38 | 1.46 → 0.93 | 1.16 → 1.07 |
| 3 | 2.65 → 2.28 | 1.57 → 1.09 | 1.14 → 1.06 |

Chẩn đoán "contrastive phân kỳ" trong `Luong_xu_ly_du_an.md` là ảo giác của cột `total` (nhân `ramp`). Quyết định "không sweep η/ζ/τ" vì lý do này **không còn căn cứ**.

Ở `current`, số hạng contrastive gần như không học: `global_con` chỉ từ 6.9 → 6.6 (gần mức ngẫu nhiên), `cross_con` đứng ở 1.67–1.68 từ epoch 1. Cùng với quyết định 12/11, đây là thêm lý do bỏ `current`.

**Quá khớp nặng.** `wa` trên train về 0.0000 ở epoch 20 (từ ~0.03–0.04 ở epoch 1); val macro-F1 (ngưỡng 0.5) đạt đỉnh sớm rồi giảm:

| variant | val F1 tốt nhất, fold 1 / 2 / 3 (epoch) | TB tốt nhất | TB epoch 20 |
|---|---|---|---|
| `baseline` | 0.412 (2) / 0.715 (1) / 0.537 (1) | **0.555** | 0.510 |
| `current` | 0.397 (5) / 0.689 (7) / 0.549 (7) | 0.545 | 0.511 |
| `full` | 0.389 (5) / 0.705 (1) / 0.527 (8) | 0.540 | 0.507 |

- Chênh lệch giữa variant (≤ 1.5 điểm) nhỏ hơn nhiều chênh lệch giữa fold (≥ 30 điểm), nên với 1 seed **không kết luận được** variant nào hơn.
- Fold 1 yếu bất thường: ở epoch 1 của `full`, F1 AU1 = 0.006, AU2 = 0.004. Khớp với vấn đề inner val fold 1 quá thưa (§3).
- Mọi kết quả Tầng 1 và Tầng 2 hiện có (kể cả phép đo 13/11 ở §6) đều đặt trên detector này.

---

## 3. Pipeline

Hai nhánh độc lập, gặp nhau ở bước cuối.

### Tầng 1 — Bộ nhận diện AU (có huấn luyện)

| Khối | Vào | Ra |
|---|---|---|
| Backbone ResNet-50 | ảnh `3×224×224` | `(B, 49, 2048)` → `global_linear` → `(B, 49, 512)` |
| Backbone Swin-B | ảnh `3×224×224` | `(B, 49, 1024)` → `global_linear` → `(B, 49, 512)` |
| AU head — 8 nhánh Conv1D độc lập | `(B, 49, 512)` | `V_a (B, 8, 256)` · `p_a (B, 8)` |
| Expr head — 7 nhánh | `(B, 49, 512)` | `V_e (B, 7, 256)` · `p_e (B, 7)` |
| Projection heads (chỉ lúc train) | `V_a, V_e` | `z_g (B,128)` · `z_a (B,8,128)` · `z_e (B,7,128)` |

**Hàm mất mát:** `L = L_wa + 0.05·L_we + ramp(t)·(η·L_global + ν·L_node + ζ·L_cross)`
- `ramp(t) = min(1, t / warmup)`, với t là số epoch tính liên tục.
- `L_we` là BCE trên nhãn biểu cảm giả (§2.1).
- LR theo cosine, giảm từ 1e-4 về 0 trong `epochs` epoch.

**Theo dõi loss:** đọc các cột thô `global_con`, `node_con`, `cross_con` trong `history.json`, không đọc cột `total`. `total` có nhân `ramp`, mà ramp trung bình ở epoch 1 chỉ là 1/6 khi warmup = 3 (1/2 khi warmup = 1). Vì thế `total` tăng dần ngay cả khi số hạng contrastive đứng yên (xem §7).

**Augmentation — đã sửa trên Drive (13/11 14:02), chưa có run nào dùng.**
- Trước sửa: `image_train` trong `Contrastive_V1/utils.py` đã comment `PlaceCrop` và `SetFlip`, còn `experiment_data.py` gọi transform với `flip=0, offset=(0,0)`. Ảnh train chỉ khác ảnh test ở ColorJitter ±0.1. Đây là cấu hình của cả 9 run hiện có (§2.4).
- Bản sửa nằm trong `experiment_data.py` (class `strong_train_aug`, thay `image_train` cho split `fit`; `utils.py` không đổi): `Resize(256)` → `RandomResizedCrop(224, scale 0.70–1.0, ratio 0.85–1.18)` → lật ngang 0.5 → `ColorJitter(0.4, 0.4, 0.4, 0.08)` → grayscale 0.2 → Gaussian blur 0.3 → normalize → `RandomErasing(0.25, scale 0.02–0.15)`. Test giữ `Resize(256) + CenterCrop(224)`.
- Bản sửa **mạnh hơn** đề xuất ban đầu (chỉ crop + lật). Lật ngang không đổi nhãn vì cả 8 AU của DISFA đối xứng hai bên. `RandomErasing` có thể che đúng vùng của AU nhỏ (mắt cho AU1/2/4, miệng cho AU25/26) trong khi nhãn giữ nguyên, tức là thêm nhiễu nhãn (xem §7).
- Hash mã nguồn đã đổi, nên **mọi variant, kể cả baseline, phải train lại**. Không so run mới với run cũ.

**Chọn checkpoint:** theo macro-F1 trên inner val, ngưỡng 0.5; khi hoà thì giữ epoch sớm hơn.
- Inner val cũ của fold 1 (SN001, SN006, SN021) có AU1 = 73 và AU2 = **4** frame dương trên 14.535 frame. Toàn bộ số frame này đến từ SN021; SN001 và SN006 không có frame AU1/AU2 nào. Với 4 frame dương, F1 của AU2 trên val nhảy 0 → 0.26 → 0 giữa hai epoch liền nhau (run `baseline`). Mức nhảy đó bằng 0.033 macro-F1, lớn hơn chênh lệch giữa các epoch.
- **Đã quyết định (20/11): rút lại inner val, chung một quy tắc cho cả 3 fold** (`splits_v2`, `--min-val-positive 100`): dùng seed 2026, rút trên cùng luồng RNG như bản cũ, chỉ nhận bộ 3 subject mà mọi AU có ≥ 100 frame dương. Lần rút đầu giống hệt bản cũ, nên fold 2/3 giữ nguyên, chỉ fold 1 đổi.
  - Ở fold 1 có 561/816 bộ 3 subject thoả quy tắc. Lấy bộ thoả đầu tiên trên luồng RNG, tương đương rút ngẫu nhiên đều trong các bộ thoả, nên không cố ý chọn bộ "giàu" AU1/AU2, vốn sẽ làm fit mất mẫu hiếm.
  - Quy tắc chỉ dùng nhãn của outer train và được chốt trước mọi run mới. Outer test của cả 3 fold không đổi.
  - Mọi run mới dùng `splits_v2`. `splits/` cũ giữ nguyên, vì 9 run trong `runs_retry1` tham chiếu hash của nó.

**Variant:**
- Đang có: `baseline` (η=ν=ζ=0) · `global` · `node` · `cross` · `global_cross` · `full`.
- `current` bỏ khỏi kế hoạch chạy (quyết định 12/11); số cũ chỉ giữ làm tham khảo.
- Lưu ý: queue hiện có **chỉ cấp key cho `L_global`**; node loss chưa có queue (`EXPERIMENT_PROTOCOL.md`: "Chưa dùng node queue"). Vì vậy `full + queue` **không** tăng được coverage của AU2/AU9 ở node loss. Muốn kiểm giả thuyết này phải cài thêm node-level queue, và đó là một variant mới chưa có trong code.

### Tầng 2a — Rút luật (chỉ từ nhãn `fit`, không qua mạng)

**Ứng viên:** 28 cặp loại trừ `¬AU_i ∨ ¬AU_j` và 56 cặp kéo theo có hướng `AU_i → AU_j`, tổng 84.

**Thước đo chung** cho cả hai loại là lift của mẫu vi phạm: `lift = (O + 0.5) / (E + 0.5)`.
- Loại trừ: mẫu vi phạm là `(1,1)`, với `E = n_i·n_j / N`.
- Kéo theo: mẫu vi phạm là `(1,0)`, với `E = n_i·(N − n_j) / N`.

**Bootstrap 2000 lần theo subject** của tập fit (mặc định `--n-boot 2000` trong code). Resample theo subject chứ không theo frame, vì DISFA chỉ có 27 subject và các frame liên tiếp cách nhau 50 ms.

**Tiêu chí nhận luật:**
- `E ≥ 50`;
- `UCB95(lift) < 0.8`;
- ≥ 70% subject cùng chiều (`lift_s < 1`), chỉ tính các subject có `E_s ≥ 5`;
- riêng luật kéo theo: thêm `LCB95(P(j|i)) ≥ 0.6`.

Trọng số `w0 = min(5, −log UCB95(lift))`. Mỗi fold có một bảng luật cố định, dùng chung cho mọi variant và seed.

**Bỏ bước lọc `help − hurt` khỏi tiêu chí nhận luật;** chỉ giữ nó làm số chẩn đoán. Lý do:
1. "help" (dự đoán vi phạm trong khi nhãn thật không vi phạm) không có nghĩa là decoder sửa đúng. Với luật loại trừ, decoder có thể tắt nhầm AU kia.
2. Inner val chỉ có 3 subject và đã được dùng để tune ngưỡng C1, chọn λ và chọn R*. Chọn thêm từng luật trên đó làm tăng quá khớp vào val.
3. λ được chọn trên val với λ = 0 nằm trong lưới, và đó đã là một cổng tắt luật toàn cục.
4. Code chuẩn (`stage2_decoder/`) không cài bước này, chỉ notebook `Tang2_Decoder.ipynb` có. Hai nơi phải thống nhất.

Nếu vẫn muốn lọc từng luật, hãy dùng ΔF1 trên val khi bỏ riêng từng luật (leave-one-rule-out) thay cho `help − hurt`.

### Tầng 2b — Giải mã MAP chính xác

```
y* = argmax_y  Σ_i y_i·(logit p_i + b_i)  −  λ·Σ_r w_r·V[y,r]
```

Khi b = 0, công thức này tương đương với `Σ[y log p + (1−y) log(1−p)]`, chỉ sai một hằng số. Decoder duyệt hết 256 cấu hình cho mỗi frame, không dùng SAT solver, GCN hay xấp xỉ.

| Nhánh | Luật | `b_i` | `λ` |
|---|---|---|---|
| **C0** | không | 0 | — |
| **C1** | không | `−logit(t_i)`; `t_i` tune trên val, lưới 0.05–0.95, bước 0.01 | — |
| **R0** | luật dữ liệu | 0 | chọn trên val |
| **R1** | luật dữ liệu | như C1 | chọn trên val |
| **R_FACS** *(mới, chưa có trong code)* | 5 luật loại trừ viết tay cũ, `w_r = 1` | như R* | chọn trên val |

- Lưới λ: `{0, .1, .25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8}`. Khi hoà thì chọn λ nhỏ nhất.
- R* là R0 hoặc R1, chọn theo F1 trên val; khi hoà thì ưu tiên R0. Test chỉ dùng để báo cáo.
- So sánh `R* − R_FACS` trả lời Q2a. Code đã có sẵn tập luật cũ trong `core.OLD_EXCLUSION`.

---

## 4. Đầu ra

### Tệp sinh ra — theo cấu trúc thật trên Drive (`MyDrive/Sym`, kiểm 19/11)

| Tệp | Nội dung |
|---|---|
| `runs_retry1/<variant>/fold<k>/seed<s>/best.pt`, `last.pt` | checkpoint (~0,49 GB mỗi file, `last.pt` chỉ cần khi resume) |
| `history.json` | loss tách theo số hạng (`total`, `wa`, `we`, `global_con`, `node_con`, `cross_con`) và val metrics mỗi epoch |
| `config.json`, `manifest.json`, `environment.json`, `source/` | truy vết đầy đủ |
| `evaluation/test_metrics.json`, `test_predictions.npz` | kết quả outer test |
| `evaluation_val/val_metrics.json`, `val_predictions.npz` | dùng để chọn ngưỡng và λ |
| `stage2_out/rules_fold{k}.csv` | bảng luật mỗi fold (sinh bởi notebook `Tang2_Decoder.ipynb`) |
| `stage2_out/stage2_results.json` | kết quả các nhánh kèm CI |

**Hai bản cài Tầng 2 chưa thống nhất.**
- Bản **đã chạy** là notebook `Tang2_Decoder.ipynb`. Notebook này đọc `evaluation_val/` và ghi ra `stage2_out/`.
- Module `Contrastive_V1/stage2_decoder/` (cùng `remote_stage2/`) **chỉ có ở bản local, không có trên Drive**. Module này đọc `evaluation/val_predictions.npz` và ghi ra `downstream/stage2_decoder/`.

Phải chọn một bản làm chuẩn trước khi đo lại Tầng 2. Nếu chọn module thì phải sửa đường dẫn val cho khớp với `evaluation_val/`.

### Bảng và hình cho luận văn

1. **Bảng chính** — macro-F1 và F1 từng AU: lấy mean 3 fold cho mỗi seed, rồi báo mean ± SD qua seed, cho từng variant.
2. **Bảng ablation Tầng 1** — baseline / global / node / cross / full.
3. **Bảng Tầng 2** — C0, C1, R0, R1, R*, R_FACS, kèm Δ và CI bootstrap theo subject.
4. **Bảng luật** — luật nào qua kiểm định ở fold nào, đối chiếu với cột `old_rule`.
5. **Hình pipeline** — hai nhánh và điểm hợp lưu.
6. **Hình chẩn đoán** — quỹ đạo `wa`, các cột contrastive thô và val macro-F1 theo epoch, trước và sau khi sửa augmentation.
7. **Bảng giao thức** — đối chiếu giao thức của nghiên cứu này với paper (chọn epoch, ngưỡng, số lần chạy).

---

## 5. Outcome mong muốn

### Mức tối thiểu

| Hạng mục | Trạng thái |
|---|---|
| Ba phát hiện ở §2 có bằng chứng | **một phần**: 2.1 đã chứng minh; 2.2 đã chứng minh cho bản cài đặt lại, phần về paper còn là suy luận; 2.3 cần CI trên 3 fold |
| Phương pháp thay thế có kiểm định và suy luận chính xác | **code xong**, chưa có kết quả trên Tầng 1 đã sửa |
| Báo cáo đạt chuẩn: nhiều seed, CI theo subject, không rò rỉ outer test, siêu tham số chọn trên inner val | **chưa đạt**: mới có 1/3 seed |

Phần này **không phụ thuộc vào việc có vượt điểm paper hay không**, và tự nó đã là một đóng góp đứng được.

### Mức kỳ vọng

| Mục tiêu | Trạng thái |
|---|---|
| Contrastive cải thiện so với baseline của chính mình | **chưa đạt**: hiện `full − baseline = −0.31` (1 seed) |
| Tầng 2: `R* − C1 > 0` và `R* − C0 > 0`, CI không chứa 0, mức +0.2 đến +0.5 | **chưa đạt**: lần đo 13/11 có R0 − C0 = +0.32 nhưng CI chứa 0 |
| Hết quá khớp: `wa` không về 1e-05, val đạt đỉnh sau epoch 2 | **chưa đạt ở 9 run cũ** (`wa` → 0.0000, đỉnh val ở epoch 1–8, §2.4); chờ run đầu tiên có `strong_train_aug` |

### Mức tốt nhất

Vượt **62.94** macro-F1 trên DISFA. Khoảng cách hiện tại là **2.41 điểm**, tính so với baseline seed 0 = 60.53 (mới có 1 seed, chưa có SD).

Bảng dưới là **giả định để lập kế hoạch, không phải kết quả:**

| Nguồn bù điểm | Thấp | Cao | Loại |
|---|---|---|---|
| Sửa quá khớp (augmentation) | +0.5 | +2.0 | sửa lỗi |
| Khớp giao thức train (refit trên `fit + inner val`) | +0.5 | +1.5 | vệ sinh |
| Đổi sang Swin-B | +1.0 | +1.5 | backbone của paper |
| Tune ngưỡng từng AU (C1) | chưa đo | chưa đo | vệ sinh |
| Tune contrastive sau khi hết quá khớp | 0.0 | +0.5 | **đóng góp** |
| Tầng 2 decoder (R* so với C1) | +0.25 | +0.5 | **đóng góp** |

Cộng các mức thấp được 62.78 và các mức cao được 66.5, nhưng **các khoản không cộng thẳng được**:
1. Augmentation và refit cùng chữa một bệnh (quá khớp/thiếu dữ liệu), nên phần lợi chồng lên nhau.
2. Detector càng tốt thì decoder càng còn ít lỗi để sửa.
3. "Mức thấp" chưa phải kịch bản xấu: §7 nêu Swin-B có thể cho ≤ 0 (paper đo trên BP4D được −0.5), và augmentation cũng có thể không giúp. Kịch bản xấu thật là không vượt được paper. Khi đó luận văn đứng trên mức tối thiểu.

**So sánh với 62.94 phải cùng giao thức.**
- 62.94 là kết quả của một lần chạy, không seed, không CI [cần kiểm cách paper chọn epoch và ngưỡng].
- Số của nghiên cứu này là mean qua seed, với mô hình chọn trên inner val, tức giao thức chặt hơn.
- Báo cáo hai dòng:
  - **(a) giao thức chặt** — kết quả chính;
  - **(b) giao thức khớp paper** — refit trên `fit + inner val` với số epoch cố định lấy từ (a). Dòng này chỉ dùng để so với 62.94.

**Xung đột giữa refit và Tầng 2.** Sau khi refit trên `fit + inner val` thì không còn val để tune ngưỡng C1, chọn λ và chọn R*. Cách xử lý: lấy `t_i`, λ và R* từ mô hình (a), áp nguyên cho mô hình (b), và ghi rõ đây là chuyển giao tham số. Không tune lại trên test.

**Cảnh báo phải nêu trong luận văn:**
- Đòn bẩy chắc nhất (Swin-B) là backbone của chính paper.
- Khớp giao thức và tune ngưỡng là vệ sinh nghiên cứu; sửa quá khớp là sửa lỗi.
- Đóng góp thật chỉ còn **contrastive ở Tầng 1** và **decoder ở Tầng 2**. Nếu vượt 62.94 mà toàn bộ phần vượt đến từ Swin-B thì hội đồng sẽ hỏi đúng câu đó.
- Vì vậy bảng chính phải tách đóng góp theo từng khoản, đo trên cùng backbone.

---

## 6. Tiêu chí thành công và tiêu chí dừng

**CI dùng chung cho mọi tiêu chí:** bootstrap 2000 lần, resample subject **trong từng fold**, rồi lấy trung bình 3 fold (`stage2_decoder/summarize.py`). Phép so sánh ghép cặp trên cùng fold/seed. Mỗi seed cho một CI.

**Tầng 2 được đầu tư tiếp** chỉ khi `R* − C1 > 0` **và** `R* − C0 > 0`: mean qua seed dương, và CI không chứa 0 ở ít nhất 2/3 seed.

> Đo lần đầu (13/11, trên Tầng 1 chưa sửa): **TRƯỢT**. R* thua C0 ở 2/3 variant, mọi CI của `R*−C0` đều chứa 0. So sánh sạch `R0 vs C0` cho +0.32 điểm trung bình, chưa tách được khỏi nhiễu.
>
> Phép đo này đặt trên một detector đang quá khớp nặng, nên phải **đo lại sau khi sửa Tầng 1** rồi mới kết luận.

**Contrastive được coi là có đóng góp** khi thoả cả hai bước:
1. Chọn **một** cấu hình contrastive trên inner val **trước khi mở test**: `full` hoặc cấu hình sweep tốt nhất trên val.
2. Cấu hình đó so với `baseline` có mean Δ > 0 qua 3 seed, và CI không chứa 0 ở ít nhất 2/3 seed.

Chỉ một so sánh chính này được mở test. Các variant còn lại được báo cáo như ablation và không dùng làm tiêu chí, để tránh so sánh nhiều lần.

**Kết quả âm vẫn là kết quả.** Nếu sau khi sửa Tầng 1 mà luật vẫn không giúp, kết luận đúng là:
- Trên tập nhãn đầy đủ, detector đã tự học được quan hệ đồng xuất hiện giữa các AU từ nhãn.
- Decoder chỉ can thiệp ở các frame mà dự đoán vi phạm luật, và mỗi lần can thiệp có thể sửa đúng hoặc tắt nhầm AU, nên lợi ích ròng nhỏ hơn nhiễu.
- Giá trị thật của luật khi đó nằm ở **chế độ ít nhãn**: rút luật từ phần có nhãn, rồi áp lên detector được train với ít nhãn. Đó là thí nghiệm nên làm tiếp nếu còn thời gian.

---

## 7. Rủi ro

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Augmentation không chặn được quá khớp | cao | chạy thử 1 fold trước khi train toàn bộ |
| Swin-B không cho +1.5 trên setup này | cao — paper đo BP4D **−0.5** | dò 1 fold trước khi đặt cược |
| Biên độ giữa fold (12.84) gấp 8.7 lần giữa variant (1.47) | cao | so sánh ghép cặp trong từng fold, nhiều seed |
| ~~Chẩn đoán "contrastive phân kỳ" có thể sai~~ | **đã xác nhận sai (19/11, §2.4)**: các cột thô giảm đều | mở lại sweep η/ζ/τ, nhưng chỉ **sau** khi có run hết quá khớp |
| `strong_train_aug` quá mạnh cho AU tinh: `RandomErasing` che vùng AU mà nhãn giữ nguyên; ColorJitter 0.4 + grayscale làm mờ dấu hiệu nhỏ | trung bình | run thử 1 fold so với bản chỉ crop + lật; nếu F1 các AU nhỏ (AU1/2/9/26) tụt thì bỏ `RandomErasing` trước |
| `full + queue` không giúp AU hiếm vì queue chỉ cấp cho `L_global` | trung bình | cài node-level queue, hoặc bỏ giả thuyết này |
| Quá khớp vào inner val: ngưỡng C1, λ, R* và checkpoint đều chọn trên 3 subject | trung bình | không thêm bước chọn trên val (bỏ lọc help−hurt); với AU thưa trong val thì dùng ngưỡng cố định 0.5 |
| Inner val fold 1 quá thưa (AU2 chỉ 4 frame dương / 14.535) | trung bình | **đã xử lý ở code (20/11)**: `splits_v2` với `--min-val-positive 100`; cần sinh manifest trên Colab trước run fold 1 đầu tiên |
| Hai bản cài Tầng 2 (notebook đã chạy và module `stage2_decoder` chỉ có ở local) đọc val ở hai đường dẫn khác nhau | trung bình | chọn một bản làm chuẩn, sửa đường dẫn, đo lại trên cùng một bản |
| Chưa xác minh được con số và giao thức của paper | trung bình | lấy PDF SymGraphAU (`paper.txt` hiện tại là bài khác) |
| Ngân sách GPU Colab | trung bình | chép ảnh về `/content`; dùng `--stop-after` + `--resume` |

---

## 8. Thay đổi so với bản 1

1. **§1:** tách Q2 thành Q2a (nguồn luật) và Q2b (cách suy luận), vì Q2 gộp hai thay đổi nên không quy được nguyên nhân. Thêm nhánh `R_FACS`.
2. **§2:** ghi mức chắc chắn cho từng phát hiện.
   - 2.1 là hệ quả của định nghĩa, không phải phép đo; bỏ câu "đóng góp trung tâm không mang thông tin" vì quá mạnh, và Tầng 1 của chính nghiên cứu cũng dùng `M_AE`.
   - 2.2 tách phần đã chứng minh từ code (bản cài đặt lại) khỏi phần suy luận từ Fig. 4.
   - 2.3 ghi rõ lift 1.44 là ước lượng điểm trên một fold, chưa có CI.
3. **§3 Tầng 1:** thêm shape của Swin-B; ghi augmentation đang bị tắt trong code và hệ quả phải train lại toàn bộ; đề xuất rút lại inner val fold 1; bỏ `current`; ghi rõ queue chỉ áp cho global loss.
4. **§3 Tầng 2a:** sửa số bootstrap 1000 → 2000 cho khớp code; ghi đủ công thức lift và E; bỏ lọc `help − hurt` khỏi tiêu chí nhận luật.
5. **§3 Tầng 2b:** sửa "bias = ngưỡng C1" thành `b_i = −logit(t_i)`; ghi lưới λ, quy tắc hoà, cách chọn R*.
6. **§4:** đối chiếu với cấu trúc thật trên Drive (19/11). Giữ `stage2_out/` và `evaluation_val/` vì đây là đường dẫn đã thực sự chạy; ghi rõ có hai bản cài Tầng 2 chưa thống nhất.
7. **§5:**
   - sửa trạng thái mức tối thiểu ("phần lớn đã đạt" → từng mục cụ thể);
   - đánh dấu bảng bù điểm là giả định và nêu lý do không cộng thẳng;
   - thêm dòng tune ngưỡng C1 (vệ sinh);
   - thêm yêu cầu so sánh cùng giao thức và cách xử lý xung đột giữa refit và Tầng 2.
8. **§6:** định nghĩa CI chung; yêu cầu Tầng 2 thắng cả C1; chỉ một so sánh contrastive chính, chọn trên val trước khi mở test; viết lại đoạn kết quả âm (câu "`w_r = 0` ở mọi frame mà nhãn thật vi phạm" lẫn giữa trọng số luật và loss lúc train — `L_rule` đã hoãn từ 02/11).
9. **§7:** thêm 5 rủi ro mới tìm thấy khi đối chiếu code.
10. **Cập nhật 19/11** (sau khi kéo mã nguồn và log từ Drive về):
    - thêm §2.4: contrastive không phân kỳ, Tầng 1 cũ quá khớp nặng, kèm số liệu 9 run;
    - §3 Tầng 1: augmentation đã được sửa trên Drive bằng `strong_train_aug` trong `experiment_data.py`; ghi cấu hình cụ thể và xác nhận bằng hash rằng 9 run cũ chưa dùng nó;
    - §5: cập nhật trạng thái mục "hết quá khớp";
    - §7: đóng rủi ro chẩn đoán phân kỳ, thêm rủi ro augmentation quá mạnh.
11. **Cập nhật 20/11:** chốt việc rút lại inner val (`splits_v2`, ≥ 100 frame dương mỗi AU, cùng seed 2026); chỉ fold 1 đổi. Code ở `experiment_data.py` (`min_val_positive`), test ở `tests/test_contrastive.py`.
