# Context — baseline SymGraphAU và kết quả cần đạt

**Nguồn:** Xie et al., *SymGraphAU: Prior knowledge based symbolic graph for action unit recognition*, Pattern Recognition 165 (2025) 111640, doi:10.1016/j.patcog.2025.111640.
- PDF nằm ở máy chủ dự án: `C:\CODE\Symbolic\Paper\`. Không đưa lên git.
- Đọc và ghi lại ngày 21/11/2026.
- Phần supplementary (danh sách quan hệ AU–biểu cảm, cặp AU, `p_uct`, thuật toán) **chưa có trong tay**.

Mọi con số dưới đây lấy từ bảng của paper. Chỗ nào là suy luận thì ghi rõ.

---

## 1. Mô hình của paper

### Stage 1 — JFL (học đặc trưng nút bằng hai tác vụ)
- **Backbone:** ResNet-50 hoặc Swin-B, pretrain ImageNet-1k.
- **Hai head tách rời:**
  - `N_a` bộ trích AU → `V_a`, `p_a`;
  - `N_e` bộ trích biểu cảm → `V_e`, `p_e`.
  
  Mỗi bộ trích gồm FC, GAP và sigmoid.
- **Loss:** `L_jf = L_wa + γ·L_we`, với `γ = 0.05`.
  - `L_wa`: weighted asymmetric loss (lấy từ ME-GraphAU). Trọng số mỗi AU tỉ lệ nghịch với tần suất xuất hiện trong tập train.
  - `L_we`: loss trên nhãn biểu cảm giả `k_e = argmax(Y_a · M_AE)`. Mẫu không có AU nào bật được gán Neutral.
  - `M_AE`: 0.9 (liên quan chính), 0.5 (liên quan phụ), 0.1 (không liên quan); `β = 0.1`.
- **Train:** 20 epoch, lr 1e-4.

### Stage 2 — GEK (bộ nhúng tri thức bằng GCN)
- **CNF riêng cho từng mẫu:**
  - mệnh đề AU–biểu cảm dạng `¬AU_i ∨ ¬AU_j ∨ Expr`, chỉ được thêm với một xác suất bất định `p_uct`;
  - đồng xuất hiện AU–AU: lấy từ nhãn thật của mẫu;
  - loại trừ AU–AU: chỉ thêm khi có cả trong nhãn mẫu lẫn trong danh sách tri thức định sẵn.
- **CNF → đồ thị.** Có 4 loại nút: lá, AND, OR, global, mỗi loại đánh dấu bằng tiền tố 4 bit.
  - Nút lá: tâm lớp EMA (`α = 0.1`) của `V_a`/`V_e`, tính sau Stage 1 rồi giữ cố định.
  - Đồ thị **không có nút NOT** (xem đề cương §2.2).
- **GCN 2 lớp, train bằng `L_le = L_h + λ·L_c`**, với `λ = 0.1`:
  - `L_h`: triplet loss, kéo nhúng CNF lại gần phép gán thoả và đẩy xa phép gán không thoả (cả hai sinh bằng PySAT);
  - `L_c`: bộ phân loại thoả / không thoả.
- **Train:** 5 epoch, lr 1e-2.

### Stage 3 — EKR (tinh chỉnh detector bằng tri thức)
- **Loss:** `L = L_wa + μ·L_l`, với `μ = 0.1`. `L_l` là khoảng cách L2 giữa nút global của đồ thị CNF và nút global của đồ thị dự đoán (dựng từ `p_a`, `p_e`). `L_we` không còn trong stage này.
- **Train:** 20 epoch, lr **1e-6**.
- **Suy diễn:** chỉ dùng detector, không chạy GCN.

### Chi tiết train chung
- AdamW, β = (0.9, 0.999), weight decay 5e-4, batch 32. Tổng cộng 45 epoch (20 + 5 + 20). Chạy trên GPU P100.
- Tiền xử lý: MTCNN căn chỉnh rồi crop 224×224.
- Giao thức: 3 fold độc lập theo subject, "giống [6, 8]" (HMP-PS, ME-GraphAU). Metric là F1 theo frame cho từng AU, rồi lấy trung bình 8 AU.

### Paper **không** nói rõ
- **Có tập validation không, và chọn epoch thế nào.** [cần kiểm] Code ME-GraphAU đánh giá trên fold test sau mỗi epoch. Nếu SymGraphAU làm theo, số của paper là số đã chọn epoch trên test, tức lạc quan.
- **Ngưỡng quyết định.** Không ghi; nhiều khả năng là 0.5.
- **Số lần chạy, seed, độ lệch chuẩn, CI.** Không có. Mỗi con số là một lần chạy.
- **Có khởi tạo DISFA từ mô hình train trên BP4D không.** Không nhắc tới. Ta coi là không, và **không** dùng cách này.

---

## 2. Kết quả của paper trên DISFA (8 AU, F1 %)

| Phương pháp | AU1 | AU2 | AU4 | AU6 | AU9 | AU12 | AU25 | AU26 | **TB** |
|---|---|---|---|---|---|---|---|---|---|
| **SymGraphAU, ResNet-50** | 53.3 | 52.1 | 77.6 | 51.3 | 48.5 | 74.6 | 91.7 | 54.4 | **63.0** (62.94) |
| **SymGraphAU, Swin-B** | 62.1 | 61.6 | 69.2 | 47.2 | 52.6 | 78.1 | 92.3 | 53.2 | **64.5** (64.54) |
| ME-GraphAU (IJCAI'22), ResNet-50 | 54.6 | 47.1 | 72.9 | 54.0 | 55.7 | 76.7 | 91.1 | 53.0 | 63.1 |
| FAN-Trans (WACV'23) | 56.4 | 50.2 | 68.6 | 49.2 | 57.6 | 75.6 | 93.6 | 58.8 | 63.8 |

Số trong ngoặc là trung bình tính lại từ 8 cột. Paper làm tròn thành 63.0 và 64.5.

Với backbone CNN trên DISFA, SymGraphAU (63.0) **không** cao hơn ME-GraphAU (63.1). Paper tự nhận điểm này và giải thích là do nhãn DISFA thưa.

**Ablation (Table 3, DISFA):**

| Backbone | Baseline (chỉ `L_wa`) | + JFL | + JFL + EKR |
|---|---|---|---|
| ResNet-50 | 61.2 | 62.1 (+0.9) | **63.0** (+0.9) |
| Swin-B | 63.4 | 64.2 (+0.8) | **64.5** (+0.3) |

Trên BP4D (ngoài phạm vi luận văn): 65.3 với ResNet-50, 64.8 với Swin-B.

---

## 3. Đối chiếu với pipeline của luận văn

| Paper | Luận văn | Ghi chú |
|---|---|---|
| Baseline (chỉ `L_wa`), 61.2 | chưa có variant tương ứng | |
| Stage 1 JFL, 62.1 | **Tầng 1 `baseline`**: `L_wa + 0.05·L_we` | Cùng lr 1e-4, wd 5e-4, AdamW, batch 32, 20 epoch, ImageNet-1k |
| Stage 2 + 3 (GCN + EKR, +0.9), 63.0 | **contrastive trong Tầng 1** (`full`) + **Tầng 2** (luật rút từ dữ liệu + MAP chính xác) | đây là hai đóng góp của luận văn |
| Head: FC + GAP + sigmoid | Conv1D (k = 3) trên 49 token → BN → ReLU → GAP → Linear → sigmoid, embedding 256 | khác kiến trúc head |
| `L_we`: "multi-label softmax" (Eq. 4) | BCE (`expr_loss = legacy_bce`) | khác loss |
| Không nói về augmentation | `strong_train_aug` (từ 21/11, `runs_aug1`) | |
| Không thấy val | inner val 3 subject (`splits_v2`), chọn checkpoint trên val | giao thức chặt hơn |

---

## 4. Kết quả mong muốn

**Mục tiêu chính:** vượt SymGraphAU **trên cùng backbone**, macro-F1 DISFA, trung bình 3 fold.
- **ResNet-50: vượt 63.0 (62.94).** Đây là mốc chính.
- **Swin-B: vượt 64.5.** Nếu đổi sang Swin-B thì phải so với **64.5**, không so với 62.94. Dùng Swin-B để vượt mốc ResNet-50 là so lệch backbone.

**Hai dòng báo cáo** (đề cương §5):
- **(a) Giao thức chặt:** chọn checkpoint trên inner val, mean ± SD qua 3 seed. Đây là kết quả chính.
- **(b) Giao thức khớp paper:** refit trên fit + val với số epoch lấy từ (a). Chỉ dòng này được đặt cạnh 63.0.

**Mốc trung gian:**
1. Tầng 1 `baseline` đạt khoảng **62.1**, tức tái lập được Stage 1 của paper. Nếu thấp hơn nhiều, lỗi nằm ở Tầng 1, không ở phần đóng góp.
2. Đóng góp tách riêng, trên cùng backbone:
   - `full − baseline > 0`. Mức tương ứng trong paper là EKR: +0.9 với ResNet-50, +0.3 với Swin-B.
   - `R0 − C0 > 0`, mục tiêu +0.25 đến +0.5.
   
   Tiêu chí: CI không chứa 0 ở ít nhất 2/3 số seed (đề cương §6). Riêng `full − baseline` còn cần Δ > 0 ở cả 3 seed (chốt 22/11).

**Hiện trạng (22/11):**
- Detector cũ, không augmentation, test thăm dò ngày 29/10: `baseline` seed 0 = **60.53**, kém 62.94 là 2.41 điểm. Thấp hơn cả JFL của paper (62.1). Số này chỉ là thăm dò.
- Tầng 2 chạy thử trên run cũ, chỉ val, λ chọn trên chính val: `R0 − C0` từ +0.0 đến +0.9.
- Pilot `runs_aug1` fold 2, seed 0 xong: val tốt nhất 76.8 / 74.9 (`baseline` / `full`), đều ở epoch 1. Đây là val của 3 subject, **không so được** với 63.0. Chi tiết ở CLAUDE.md §7.

---

## 5. Điểm của paper dùng được trong phần lập luận

- **Nhãn DISFA thưa.** Paper thừa nhận: tri thức sinh ra cho DISFA thưa, các đồ thị CNF dồn cụm trong không gian nhúng, và bản CNN không đạt tốt nhất trên DISFA. Khớp với phát hiện của ta: luật rút từ dữ liệu ít và đổi theo fold (6/2/4 luật, đề cương §6).
- **Stage 3 có lr rất nhỏ.** EKR chạy 20 epoch với lr 1e-6. [suy luận] Ở lr này trọng số đổi rất ít, nên một phần mức +0.9 có thể đến từ việc có thêm 20 lần chọn epoch. Chưa kết luận được khi chưa có code của tác giả.
- **Không có seed, không có CI.** Một lần chạy duy nhất, trong khi chênh lệch giữa các phương pháp trong bảng chỉ cỡ 0.1–1 điểm.
- **Đồ thị không có nút NOT** (Fig. 4), xem đề cương §2.2.

---

## 6. Hướng cải tiến — đề xuất 21/11, ĐÓNG BĂNG từ 22/11

**Cấu hình đã khoá ngày 22/11** (CLAUDE.md §6): chỉ mục 1. Mục 2–7 không đưa vào 18 run này, chỉ là hướng phát triển sau.

Ràng buộc chung:
- Mọi thay đổi chỉ được chọn dựa trên inner val.
- Không sửa file bị hash cho tới khi đủ 18 run (CLAUDE.md §3, §6).
- ~~Mỗi thay đổi thử trước trên fold 2, seed 0~~ **Sai, rút lại 22/11:** cả 3 subject val fold 2 là subject test của fold 1/3, nên chọn cấu hình ở đó rồi áp cho mọi fold là rò rỉ. Hoặc định trước cấu hình, hoặc tune lồng riêng từng fold.

| # | Tầng | Thay đổi | Lý do | Chi phí | Trạng thái |
|---|---|---|---|---|---|
| 1 | 1 | Augmentation mạnh | run cũ quá khớp (`wa` → 0, val đạt đỉnh ở epoch 1–2) | đã có | **pilot xong 22/11:** đỉnh cao hơn run cũ nhưng vẫn ở epoch 1; TB epoch cuối không nhất quán giữa variant |
| 2 | 1 | Đánh giá val mỗi ¼ epoch + EMA trọng số | đỉnh val rơi vào epoch 1, nên checkpoint tốt có thể nằm giữa epoch; EMA làm mượt | sửa code nhỏ | **B2 + D2 xong 22/11:** không có bằng chứng có lợi (val tốt nhất kém pilot 8.7 / 7.5 điểm; TB epoch 11–20 thì +2.2 / −8.4). Đề xuất không dùng |
| 3 | 1 | LR backbone nhỏ hơn head (1e-5 / 1e-4) | fine-tune trên chỉ 15 subject dễ quá khớp. Lưu ý: head + projection có 19.1M tham số, gần bằng backbone ResNet-50 (23.5M), và vẫn giữ lr 1e-4 | sửa code nhỏ | **đã thử trong B2**, chưa tách được khỏi EMA; đề xuất không dùng |
| 4 | 1 | Contrastive **khác subject**: cặp dương phải đến từ hai subject khác nhau; thêm queue cho node loss | lỗi lớn nhất là lệch giữa subject (ngưỡng C1 lệch xa 0.5, chênh giữa fold ≥ 30 điểm); AU2/AU9 ít mẫu dương trong mỗi batch | variant mới | đóng góp chính, chờ pilot |
| 5 | 1 | Swin-B | backbone tốt nhất của paper | gấp khoảng 2 lần GPU | chỉ so với 64.5 |
| 6 | 2 | Ngưỡng C1 có giới hạn (lưới hẹp, hoặc co về 0.5) | C1 thắng lớn trên val nhưng dễ quá khớp 3 subject | CPU | đo lại sau pilot |
| 7 | 3 (tuỳ chọn) | Tinh chỉnh bằng loss luật chính xác: `w·p_i·p_j` cho loại trừ, `w·p_i·(1 − p_j)` cho kéo theo, dùng luật rút từ dữ liệu | thay EKR/GCN bằng ràng buộc chính xác, khả vi; cho Q2b một con số đo được mà không phải sửa GCN cũ | 1 lần chạy ngắn mỗi fold | đề xuất |
| ✗ | — | Làm mượt theo thời gian, chuẩn hoá theo từng người, khởi tạo từ BP4D | paper chỉ so với phương pháp ảnh tĩnh, không dùng dữ liệu thêm | — | **không làm** |
