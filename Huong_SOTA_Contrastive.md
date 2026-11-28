# Nhận xét pipeline và hướng đạt SOTA, bắt buộc có contrastive

**Ngày:** 28/11/2026. **Đầu vào:** survey "Lộ trình đạt SOTA nhận diện AU" (bản 11/2026), đề cương, [CONTEXT.md](CONTEXT.md), code `Contrastive_V1/` ở commit `3518500`, nhãn `DISFA/list` (bản tra cứu ở local).

**Ký hiệu mức chắc chắn**
- **[đo]**: tôi tính từ nhãn hoặc code ở local. Script: `batch_stats.py` (nằm ngoài repo, chỉ đọc file nhãn).
- **[bài]**: đọc nguyên văn từ bài gốc (HTML arXiv, README).
- **[bài†]**: lấy từ phần sau của trang HTML, nơi công cụ đọc chỉ trả về bản tóm tắt. Cần đối chiếu PDF trước khi trích.
- **[ước lượng]**: phán đoán của tôi, chưa có số đo.

Toàn bộ đề xuất ở đây **chỉ bắt đầu sau khi đủ 18 run** (CLAUDE.md §6). Tài liệu này không sửa file nào bị hash.

---

## 0. Tóm tắt

1. **Survey nhìn chung đúng hướng nhưng có 6 chỗ cần sửa** (mục 1). Hai chỗ quan trọng nhất:
   - UAU-Net (66.6 DISFA) và SSM (71.9) đều dùng **chuỗi frame**, nên không cùng sân "ảnh tĩnh" với SymGraphAU.
   - **AUNCE** (Pattern Recognition 2026, 66.4 DISFA, có code) là một phương pháp **contrastive**, nhưng survey không ghi nhận điều này. Đây là tiền lệ gần nhất với luận văn.
2. **Không phương pháp dẫn đầu nào có backbone khuôn mặt dùng contrastive.** FMAE-IAT, MAE-Face, Norface và DIME đều là MAE, kèm đối kháng danh tính hoặc chuẩn hoá danh tính bằng mô hình sinh [bài]. Ngược lại, các phương pháp contrastive cho AU (AUNCE, CLP, CLEF, D2CA) đều chạy trên backbone ImageNet, FaRL hoặc VoxCeleb. Chỗ trống này là nơi luận văn có thể đóng góp.
3. **Contrastive V1 có 4 điểm yếu đo được từ nhãn và code** (mục 3.3):
   - Cụm trung tính chiếm 59–78% trọng số positive của `L_global`.
   - `L_node` chỉ chạy được ở 42–57% số batch cho AU1, AU2 và AU9.
   - Giá trị số hạng contrastive lớn gấp 4–40 lần `L_wa`.
   - Không số hạng nào dùng thông tin subject, trong khi lỗi lớn nhất của detector là lệch giữa subject.
4. **Đề xuất chính: SA-Con**, contrastive đa nhãn có giám sát và nhận biết subject.
   - Positive phải đến từ **subject khác**.
   - Frame cùng subject nhưng khác tập AU là **negative khó**.
   - Mỗi AU có hàng đợi riêng kèm momentum encoder, để AU hiếm luôn có positive.
   - Sampler lấy theo subject, dùng chung cho mọi variant, kể cả baseline.

   SA-Con thay vai trò của đối kháng danh tính (IAT) bằng contrastive, và được so trực tiếp với IAT.
5. **Hai track backbone:**
   - **R50 ImageNet**, so với 63.0 của SymGraphAU (kết quả chính của luận văn).
   - **FMAE ViT-B Face9M** (checkpoint công khai), so với FMAE và FMAE-IAT chạy lại trên fold của luận văn.

   Chỉ track thứ hai mới chạm được vùng 68–71 [ước lượng].
6. **Đổi kỳ vọng về "SOTA" cho trung thực.**
   - Với R50 ImageNet ảnh tĩnh, mức trần đã công bố là khoảng 63 (ME-GraphAU 63.1, SymGraphAU 63.0). Vượt mức này với giao thức chặt hơn đã là kết quả mạnh.
   - SOTA tuyệt đối trên DISFA (Norface 72.7 đã bình duyệt, DIME-L 74.7 chưa bình duyệt) cần backbone khuôn mặt cỡ lớn, và không đạt được trong ngân sách hiện tại.

---

## 1. Đối chiếu survey với bài gốc

| Mục trong survey | Bài gốc nói gì | Hệ quả |
|---|---|---|
| UAU-Net 66.8 / 66.6, "đã đối chiếu", xếp cùng sân A | Đầu vào là cửa sổ T frame liên tiếp, dự đoán frame giữa; ResNet không ghi độ sâu [bài]. Số liệu [bài†] | **Không phải ảnh tĩnh.** Không đặt cạnh SymGraphAU như phương pháp cùng loại |
| SSM là "preprint chưa bình duyệt" | Bản v3 (08/2026): "Accepted by IEEE TAC" [bài]. Dùng CLIP ViT-B/16, clip 16 frame, DFEW, MoE [bài] | Đã bình duyệt; vừa có dữ liệu ngoài vừa dùng chuỗi frame |
| HiVA, DISFA "chỉ qua trích dẫn" | Bài có số DISFA 67.4 [bài†]. Backbone Swin + BERT-large. Ablation vision-only: 60.4 [bài†] | Không có loss contrastive. `L_diff` chỉ là loss khử tương quan |
| AUNCE 66.1 / 66.4 "theo trích dẫn" | Là **contrastive đa nhãn**, backbone Swin cộng stage 1 của MEGraph; encoder chỉ học bằng AUNCE, sau đó train head tuyến tính [bài†]. Code: `github.com/Ziqiao-Shang/AUNCE` [bài] | **Tiền lệ trực tiếp cho luận văn**, phải nằm trong related work và bảng so sánh |
| FLCM 67.7 / 67.8 "cần đối chiếu" | Tìm không ra bài tên FLCM. Có thể là Huang et al., TAC, "feature-based AU **L**ocalization and **C**onfidence-based relation **M**ining" (FL + CM). Chưa kiểm số | Không dùng làm mục tiêu cho tới khi đọc được bài |
| FMAE-IAT 67.1 / 70.1, có mã | Đúng [bài], ViT-L. Checkpoint Face9M ViT-S/B/L công khai trên Hugging Face, kèm bản fine-tune DISFA cho từng fold. Giấy phép CC BY-NC 4.0. Recipe DISFA: blr 5e-4, 20 epoch, batch 64, BCE thường, λ_IAT = 0.5. **Không nêu cách chọn checkpoint** | Dùng được cho luận văn (mục đích phi thương mại). Phải chạy lại trên fold của luận văn |
| MAE-Face 67.4 / 70.8 | Khởi tạo từ MAE ImageNet rồi train tiếp trên 2.17M ảnh khuôn mặt. Ablation từ đầu / MAE-IN1k / MAE-Face trên DISFA là 49.1 / 67.6 / 70.8 [bài†]. **Không có code hay checkpoint** | Bằng chứng cho luận điểm "dữ liệu khuôn mặt mới là đòn bẩy", nhưng không dùng lại được |
| Norface 69.3 / 72.7 | Số đúng [bài]. Bỏ khâu chuẩn hoá danh tính thì DISFA còn 70.2 (−2.5) [bài†]. Repo chỉ phát hành **ảnh DISFA đã chuẩn hoá**, không phát hành trọng số | Bằng chứng mạnh nhất rằng khử danh tính đáng khoảng 2 điểm trên DISFA |
| DIME 73.2 / 74.7 | Preprint 12/10/2026, MAE kiểu "differential" trên Face-20M, **không contrastive**. Code ghi ở trang dự án, chưa kiểm. Số liệu [bài†] | Đúng như survey |
| Beyond the Fold ±6.5 điểm | 4 lần chia subject ngẫu nhiên trên **BP4D+**, R50. ±0.065 là trung bình biên 95% của các AU, bị AU hiếm chi phối (AU24 ±0.156, AU6 ±0.021) [bài] | Không áp thẳng con số này cho DISFA với fold cố định. Bài khuyến nghị báo thêm AUC |
| AUFormer "có thể thay Swin-B" | ViT-B/16 ImageNet-1K đóng băng + MoKE + MDWA-loss. **Chưa phát hành code** [bài] | Chỉ lấy được ý tưởng loss MDWA |

Một điểm survey bỏ sót: **gần như không bài nào nêu cách chọn checkpoint** (FMAE, MDHR, Norface, DIME, SSM, AUNCE, CLEF). MDHR ghi "average of the three fold validation results" [bài], theo đúng cách làm của code ME-GraphAU là đánh giá trên fold test. Số của các bài này vì thế nhiều khả năng lạc quan hơn giao thức chọn trên inner val của luận văn. Đây là lý do bắt buộc phải có dòng (b).

---

## 2. Bản đồ lại theo đúng sân của luận văn (DISFA, 8 AU)

| Phương pháp | Backbone / dữ liệu ngoài | Tĩnh? | Contrastive? | DISFA | Code |
|---|---|---|---|---|---|
| ME-GraphAU | R50 ImageNet | tĩnh | không | 63.1 | có |
| **SymGraphAU** | R50 ImageNet | tĩnh | không | **63.0** | không |
| MDHR | R50 / Swin-B ImageNet | **chuỗi** | không | 64.9 / 66.2 | có (MIT) |
| CLEF | ViT-B FaRL, DISFA fine-tune **từ BP4D** | tĩnh | ảnh–ảnh + ảnh–văn bản | 64.8 | không |
| AUNCE | Swin + MEGraph stage 1 | tĩnh | **có** (AUNCE) | 66.4 [bài†] | có |
| AUFormer | ViT-B IN1k đóng băng + PEFT | tĩnh | không | 66.4 | không |
| VL-FAU | Swin-B | tĩnh | không (sinh mô tả) | 66.5 | có |
| UAU-Net | ResNet | **chuỗi** | không | 66.6 [bài†] | không |
| HiVA | Swin + BERT-L + GPT-4 | tĩnh | không | 67.4 [bài†] | chưa |
| CLP | ResNet-34 tự giám sát trên VoxCeleb2, **probe tuyến tính** | chuỗi khi pretrain | **có** (thời gian + tái tạo chéo danh tính) | 57.4 [bài†] | không |
| FMAE / FMAE-IAT | ViT-L Face9M | tĩnh | không | 68.7 / 70.1 | có |
| MAE-Face | ViT-B, 2.17M ảnh khuôn mặt | tĩnh | không | 70.8 [bài†] | không |
| SSM | CLIP ViT-B + DFEW | **chuỗi** | không (prototype văn bản) | 71.9 [bài†] | chưa |
| Norface | MAE khuôn mặt + mạng chuẩn hoá danh tính | tĩnh | không | 72.7 | chỉ có ảnh |
| DIME-L | ViT-L Face-20M | tĩnh | không | 74.7 [bài†] | chưa kiểm |

Đọc bảng:
- Ở đúng sân của luận văn (R50 ImageNet, ảnh tĩnh, chỉ nhãn DISFA), mốc đã công bố là khoảng **63**.
- Contrastive có giám sát tốt nhất đã công bố là AUNCE (66.4), nhưng đứng trên Swin. Trong bảng loss của chính AUNCE, cùng encoder MEGraph mà dùng WCE thì DISFA chỉ đạt 60.8 [bài†].

---

## 3. Nhận xét pipeline hiện tại

### 3.1 Điểm mạnh, cần giữ

- **Giao thức chặt hơn mọi bài đã đọc:** chọn checkpoint trên inner val, 3 seed, CI theo subject, cặp baseline/full cùng khởi tạo, khoá cấu hình, test chỉ mở một lần. Đây là đóng góp phương pháp luận đứng độc lập.
- **Tầng 2 giải mã MAP chính xác** trên 256 cấu hình, luật rút từ dữ liệu có bootstrap theo subject. Kèm phát hiện "mất dấu phủ định" và "nhãn biểu cảm giả là hàm tất định của nhãn AU". Các phát hiện này có giá trị dù không vượt điểm.
- **Đã nhận ra nhiễu của val 3 subject** (val nhảy khoảng 9 điểm giữa hai epoch, lệch khoảng 5 điểm giữa hai máy) và đã chặn việc gọi các chênh lệch nhỏ là cải thiện.

### 3.2 Hai điểm nghẽn ngoài phần contrastive

**(a) Backbone ImageNet R50 là trần cứng.** Ablation của MAE-Face cho thấy chỉ riêng việc đổi từ MAE-ImageNet sang MAE-Face đã thêm 3.2 điểm DISFA [bài†]. Các encoder đa dụng cỡ lớn (MAE-L, DINOv3-L) chỉ đạt khoảng 63–65 DISFA [bài†]. Trên R50 ImageNet, mục tiêu hợp lý là vượt 63.0, không phải 70.

**(b) Lỗi lớn nhất là lệch giữa subject.** Bằng chứng đã có trong dự án:
- `C1 − C0` trên val đạt +1.5 đến +10.6, và ngưỡng AU1/AU2 ở fold 3 chạm biên 0.95 (đề cương §6). Detector lệch hiệu chỉnh giữa các người.
- Chênh lệch giữa fold lớn gấp khoảng 8.7 lần chênh lệch giữa variant (đề cương §7).
- Trong tài liệu, đúng hai phương pháp mạnh nhất đã bình duyệt đều nhắm vào danh tính: IAT thêm +1.4 DISFA [bài], chuẩn hoá danh tính thêm +2.5 [bài†].

Contrastive V1 **không dùng nhãn subject ở bất kỳ đâu**.

### 3.3 Contrastive V1: bốn vấn đề cụ thể

Số liệu dưới đây là thống kê nhãn trên tập fit của fold 2 và fold 3 (`splits` cũ, trùng `splits_v2` ở hai fold này), với 3000 batch ngẫu nhiên cỡ 32, đúng cách `loader_for` xáo dữ liệu [đo].

| | fold 2 | fold 3 |
|---|---|---|
| Frame trung tính (không AU nào bật) | 57.1% | 46.0% |
| Tỷ trọng positive của `L_global` đến từ cặp **cả hai trung tính** | **78.1%** | **58.8%** |
| Tỷ lệ cặp cùng subject trong batch | 6.7% | 6.7% |
| Tỷ trọng positive (bỏ cặp trung tính) đến từ cặp cùng subject | 15.4% | 13.2% |
| Số mẫu dương kỳ vọng mỗi batch: AU1 / AU2 / AU9 / AU26 | 1.5 / 1.7 / 1.4 / 2.1 | 1.9 / 1.8 / 1.7 / 2.2 |
| Tỷ lệ batch mà `L_node` chạy được cho AU đó (≥ 2 mẫu dương): AU1 / AU2 / AU9 / AU26 | 0.47 / 0.51 / 0.42 / 0.64 | 0.57 / 0.56 / 0.51 / 0.67 |

**V1-a. `L_global` chủ yếu kéo các mặt trung tính lại với nhau.** `jaccard_weights` gán 0.5 cho cặp cả hai rỗng ([contrastive.py:131-146](Contrastive_V1/model/contrastive.py:131)). Vì hơn một nửa số frame là trung tính, 59–78% "lực kéo" của loss rơi vào một cụm mà `L_wa` vốn đã tách rất tốt. Phần thật sự học cấu trúc tổ hợp AU chỉ còn khoảng 22–41%.

**V1-b. Không có tín hiệu bất biến danh tính.** Giả thuyết ban đầu của tôi là cặp positive cùng subject (hai frame cách nhau 50 ms gần như trùng ảnh) chiếm phần lớn. **Số đo không ủng hộ giả thuyết này:**
- Cặp cùng subject chỉ chiếm 13–15% trọng số positive không trung tính.
- Mức đó gấp khoảng 2 lần tỷ lệ ngẫu nhiên, tức có thiên lệch nhưng không áp đảo.

Vấn đề thật là chiều ngược lại. Loss **không có negative cùng subject có chủ đích**: cùng một người, AU bật so với AU tắt, đúng loại cặp dạy mạng phân biệt AU mà không dựa vào danh tính. Với batch ngẫu nhiên, loại cặp này hiếm và bị đối xử như mọi negative khác.

**V1-c. `L_node` thiếu positive đúng ở các AU yếu.**
- AU1, AU2 và AU9 chỉ có positive ở khoảng một nửa số batch; khi có thì thường chỉ có đúng 1 cặp ([contrastive.py:302-319](Contrastive_V1/model/contrastive.py:302)).
- AU9 và AU26 cũng là hai AU mà SymGraphAU kém các phương pháp tốt nhất 9–13 điểm (survey §2.1).
- Node loss chưa có queue (EXPERIMENT_PROTOCOL §1), nên tăng `queue` của variant không giúp được.

**V1-d. Về độ lớn, số hạng contrastive áp đảo `L_wa`.**
- Cột thô cuối train của `full`: `global_con` ≈ 2.4, `node_con` ≈ 1.0, `cross_con` ≈ 1.07 (đề cương §2.4). Với trọng số 0.05, tổng đóng góp ≈ 0.22.
- `L_wa` giảm từ 0.059 ở epoch 1 xuống 0.006 ở epoch 20 (CLAUDE.md §7).
- Như vậy, về giá trị, phần contrastive lớn gấp 4 lần `L_wa` ở epoch 1 và khoảng 40 lần ở epoch 20.
- SupCon có cận dưới khác 0 (nhiều positive) và nhân 1/τ = 10 vào gradient, nên càng về cuối, backbone càng bị contrastive lái nhiều hơn tín hiệu AU.

Đây là **giả thuyết** [ước lượng], vì độ lớn của loss không phải độ lớn của gradient. Nó có thể giải thích vì sao `full − baseline` đổi dấu theo cách đo (+1.7 ở điểm tốt nhất nhưng −3.9 ở TB epoch 11–20, fold 1). **Cần đo chuẩn gradient của từng số hạng** trên checkpoint đã có (mục 5.3).

**Hai điểm phụ:**
- `L_cross` dùng nhãn biểu cảm giả, mà nhãn này là hàm tất định của `Y_a` (đề cương §2.1). Thông tin mới duy nhất là tiên nghiệm `M_AE`.
- V1 chỉ có một view mỗi ảnh. Nó thiếu positive đáng tin nhất là cùng ảnh qua hai augmentation (AUNCE dùng loại này [bài†]), nên augmentation mạnh chỉ đóng vai nhiễu, chưa thành tín hiệu bất biến.

### 3.4 Các điểm nhỏ khác

- **Head:** Conv1D (k = 3) quét 49 token theo thứ tự hàng, coi lưới 7×7 như một chuỗi 1D; ở cuối mỗi hàng, kernel dính sang hàng kế. Đây là điểm nhỏ, nhưng các head 19.1M tham số đang ngang backbone (23.5M). Với backbone ViT (196 token), nên đổi sang attention pooling theo AU (một query cho mỗi AU).
- **Loss:** `WeightedAsymmetricLoss` chỉ có focusing `p·log(1−p)` cho negative ([utils.py:235-254](Contrastive_V1/utils.py:235)). MDWA của AUFormer thêm margin cắt và γ theo độ khó, được khoảng +0.9 trên BP4D [bài, cách đọc dòng ablation chưa chắc]. Thay đổi rẻ, nhưng phải áp cho mọi variant.
- **Tầng 2:** `R0 − C0` trên val chỉ +0.0 đến +0.9, và detector càng mạnh thì phần sửa được càng nhỏ. Giữ Tầng 2 làm đóng góp riêng, không trông vào nó để lên SOTA. Chỗ nó có cơ hội rõ nhất là chế độ ít nhãn (đề cương §6).

---

## 4. Chỗ trống mà contrastive có thể lấp

| Nhóm | Cách khử danh tính | Contrastive |
|---|---|---|
| FMAE-IAT | gradient reversal trên head ID tuyến tính | không |
| Norface | sinh lại mặt với danh tính đích | không |
| MAE-Face, DIME | không có | không |
| CLP | tái tạo chéo danh tính, tự giám sát | có, nhưng chỉ probe tuyến tính, 57.4 DISFA |
| AUNCE | không có (xử lý lệch lớp và nhãn nhiễu) | có |
| D2CA | triplet theo danh tính trên đặc trưng miền | có, chỉ cho chéo miền 5 AU |
| CL-ILE (NTU) | nhúng nhãn danh tính | có; **chưa đọc được bài** (link repository bị chuyển hướng) |

**Câu chuyện đề xuất cho luận văn:** *contrastive có giám sát, nhận biết subject, dùng làm cơ chế bất biến danh tính, so trực tiếp với cơ chế đối kháng (IAT), trên cả backbone ImageNet lẫn backbone khuôn mặt; Tầng 2 giữ nguyên.*

Hai lý do câu chuyện này đứng được:
1. Nó nhắm đúng lỗi lớn nhất đã đo ở mục 3.2(b).
2. Nó có thước đo phụ ít nhiễu hơn F1 (mục 5.3): độ chính xác probe danh tính và độ phân tán ngưỡng giữa các subject.

---

## 5. Đề xuất: Contrastive V2 (SA-Con)

### 5.1 Thành phần

Ký hiệu: `s_i` là subject của frame i, `J(·,·)` là Jaccard tập AU, `z` là embedding đã chiếu và chuẩn hoá L2, `τ = 0.1` (giữ như V1).

**C0. Sampler P×K theo subject, dùng cho mọi variant kể cả baseline.**
- Mỗi batch 32 = 4 subject × 8 frame. Khi đó cặp cùng subject chiếm 7/31 ≈ 23% số cặp, so với 6.7% hiện nay.
- Trong mỗi subject, lấy mẫu có trọng số nghịch tần suất AU, để AU1, AU2 và AU9 xuất hiện đều hơn.
- Baseline cũng phải dùng sampler này, nếu không phần lợi do sampler sẽ bị tính nhầm cho contrastive.

**C1. `L_global-SA`** thay cho `L_global`:
- Trọng số positive: `w_ij = J(y_i, y_j) · 1[s_i ≠ s_j]`. Cặp cùng subject không bao giờ là positive.
- Negative khó: mẫu số `Σ_k ω_ik exp(z_i·z_k/τ)`, với `ω_ik = 1 + h` khi `s_i = s_k` và `J(y_i, y_k) < 0.5`; còn lại `ω_ik = 1`. Đặt `h = 1` cố định.
- Anchor trung tính: bỏ. Frame trung tính vẫn có mặt làm key và negative. Lý do ở V1-a.

**C2. `L_node-SA`** có hàng đợi riêng cho từng AU:
- Momentum encoder kiểu MoCo (m = 0.999), chỉ cần cho backbone, `global_linear`, AU head và `proj_au`.
- Mỗi AU c có một queue "bật" (1024 key) và một queue "tắt" (4096 key), lưu embedding `z_a[:, c]` cùng subject.
- Anchor: các mẫu có AU c bật trong batch. Positive: key bật của **subject khác**. Negative: key tắt; key tắt cùng subject có trọng số `1 + h`.
- Lấy trung bình đều qua các AU, như V1.
- Bộ nhớ khoảng 8 × 5120 × 128 × 4 byte ≈ 21 MB. Thời gian mỗi bước tăng khoảng 30–40% vì thêm forward của momentum encoder [ước lượng].
- Thành phần này gần với positive kiểu "trung bình M mẫu cùng lớp" của AUNCE (tỷ trọng tốt nhất 0.7 [bài†]), nhưng lấy theo từng AU và qua subject.

**C3. (Ablation) positive hai view.** Hai augmentation của cùng một frame, thêm một số hạng SimCLR trọng số nhỏ. Thành phần này làm chi phí forward tăng gấp đôi, nên chỉ chạy khi C1+C2 đã có kết quả.

**C4. Bỏ `L_cross` khỏi cấu hình chính.**
- Lý do: thành phần này không mang thông tin nhãn mới (§2.1), và là một nguồn nữa làm phần contrastive áp đảo `L_wa`.
- Nếu vẫn muốn có trục tri thức, thay bằng **contrastive AU–văn bản**: prototype bật/tắt của từng AU lấy từ mô tả FACS, mã hoá bằng text encoder CLIP đóng băng, rồi so khớp `z_a[:, c]` với hai prototype. Hướng này có tiền lệ ở CLEF, SSM (TSP) và HiVA, nhưng mức tăng trên DISFA không nhất quán: SSM ghi TSP chỉ +0.2 [bài†]. Xếp ở mức thứ yếu.

**C5. Quy tắc trọng số, chốt trước và không tune trên val.** Ở 200 bước đầu trên **fit**, đo chuẩn gradient lên lớp cuối của backbone cho `L_wa` và cho phần contrastive. Chọn hệ số sao cho phần contrastive bằng 0.5 lần `L_wa`, rồi giữ hệ số đó cố định cho cả run. Quy tắc chỉ nhìn fit, nên không rò rỉ qua val.

### 5.2 Đối chứng (cùng sampler, cùng aug, cùng backbone, cùng khởi tạo)

| Variant | Nội dung | Trả lời câu hỏi |
|---|---|---|
| `base2` | `L_wa` + `L_we` + sampler C0 | baseline mới |
| `sacon` | `base2` + C1 + C2 | **so sánh chính:** `sacon − base2` |
| `iat` | `base2` + head subject tuyến tính + GRL, λ = 0.5 (theo FMAE, DISFA) | contrastive so với đối kháng |
| `full` (V1) | đã có trong 18 run | V2 so với V1 (chỉ so mô tả, vì sampler khác) |
| `sacon+iat` | tuỳ chọn | hai cơ chế có cộng dồn được không |

Theo mô phỏng ở đề cương §6, hiệu ứng +1 điểm chỉ phát hiện được khoảng 16–22% số lần với 27 subject × 3 seed. Hai cách xử lý:
- tăng lên 5 seed cho **riêng** so sánh chính, nếu ngân sách cho phép;
- dựa thêm vào các thước đo phụ ở mục 5.3 để có bằng chứng ít nhiễu hơn.

### 5.3 Chẩn đoán đi kèm, không cần mở test

**Đã viết (28/11):** module `Contrastive_V1/diagnostics/` (ngoài danh sách file bị hash, test ở `tests/test_diagnostics.py`), chạy bằng cell N trong [notebooks/Tong_hop_val.ipynb](notebooks/Tong_hop_val.ipynb) sau cell L. Kết quả ghi ở `MyDrive/Sym/diagnostics/`. Mục 1–4 dưới đây đã có code; mục 5 đọc thẳng từ `history.json`. Mỗi thước đo kèm một câu trả lời:

1. **Probe danh tính.** Train một lớp tuyến tính đoán subject từ `V_a.mean(1)` (đặc trưng đóng băng), trên frame của fit chia 70/30 theo thời gian trong từng subject. FMAE đo được 83% trên BP4D [bài].
   → Contrastive có làm đặc trưng bớt mang danh tính không? Thước đo này ít nhiễu hơn F1 nhiều.
2. **Độ phân tán ngưỡng tối ưu giữa 3 subject val.** Trong từng fold, tìm ngưỡng tối ưu riêng cho từng subject, rồi lấy độ lệch chuẩn.
   → Hiệu chỉnh có đồng đều hơn giữa các người không?
3. **AUC theo AU**, bên cạnh F1 ở ngưỡng 0.5 (theo khuyến nghị của Beyond the Fold [bài]).
4. **Chuẩn gradient của từng số hạng** `L_wa`, `L_global`, `L_node`, `L_cross` trên vài batch fit, với `last.pt`.
   → Kiểm giả thuyết V1-d.
5. **`node_eligible_batch_fraction`**, đã có trong `history.json`.
   → So với bảng ở 3.3.

### 5.4 Hai track backbone

**Track R50 (kết quả chính, cùng backbone với SymGraphAU).**
- Mục tiêu: dòng (b) vượt 63.0 và `sacon − base2 > 0` theo tiêu chí §6.
- Mức hợp lý là 63–65 [ước lượng]. Nếu `base2` vẫn ở khoảng 60.5 như detector cũ thì phải tìm lỗi ở baseline trước.
- Chi phí: `base2`, `sacon` và `iat` là 27 run, khoảng 1.2–1.4 giờ/run trên 5090 [ước lượng], tổng khoảng 35 giờ, chưa kể refit.

**Track khuôn mặt (FMAE ViT-B Face9M, checkpoint công khai).**
- Thay backbone trong `MEFARGStage1`. Head nhận `(B, 196, 768)`. Giữ recipe gốc của FMAE cho baseline: blr 5e-4, 20 epoch, AutoAugment, RandomErasing 0.25, BCE.
- Variant: `fmae` (head tuyến tính + BCE), `fmae+iat` và `fmae+sacon`.
- Phải chạy lại trên fold và inner val của luận văn: README của FMAE dùng cách chia subject của MSCL/CLEF, chưa chắc trùng ME-GraphAU.
- Mốc tham khảo FMAE-IAT 70.1 là của ViT-L và không rõ cách chọn checkpoint, nên chỉ so mô tả. Dự kiến khoảng 68–71 [ước lượng].
- Chi phí: ViT-B tốn khoảng 4 lần FLOPs của R50, nên ước lượng 2–4 giờ/run với bf16 [ước lượng]. 18–27 run tức 36–100 giờ. Chạy smoke để đo trước khi cam kết.
- ViT-L (khoảng 15 lần R50) chỉ nên chạy cho một bộ 3 fold × 1 seed để có một dòng "headline", ghi rõ là một seed.

---

## 6. Thứ tự làm, khớp với quy tắc dự án

1. **Bây giờ:** không đụng file bị hash; để 18 run chạy xong. Chạy cell N (chẩn đoán ở 5.3, đã có code) trên các run đã xong, sau cell L.
2. **Chốt trước thiết kế V2 trong đề cương, trước khi mở test của 18 run:** variant, `h`, kích thước queue, quy tắc C5, sampler, số seed, quy tắc chọn checkpoint.
   - Lý do: nếu mở test cho V1 rồi mới thiết kế V2, thì V2 là lần nhìn test thứ hai.
   - Phương án sạch: mở test **một lần cho tất cả** so sánh đã đăng ký.
   - Nếu chủ dự án không muốn chờ V2 thì phải công bố V2 được thiết kế sau khi đã thấy test V1.
3. **Code V2** trong thư mục mới `Contrastive_V2/`, import từ V1 nhưng không sửa V1, nên hash của run V1 không đổi. Cần một lớp dataset trả thêm subject id; không sửa `experiment_data.py`.
4. Train track R50 (`base2`, `sacon`, `iat`) bằng runner hiện có (có thể mở rộng `pc/train_18_run.py` thành bản V2 riêng).
5. Smoke đo thời gian ViT-B, rồi train track khuôn mặt.
6. Chạy Tầng 2 chỉ trên val cho mọi variant, rồi mở test một lần, rồi làm dòng (b).
7. Tuỳ chọn, nếu còn thời gian:
   - **Chế độ ít nhãn** (10% / 1% nhãn fit): nơi cả contrastive lẫn luật có cơ hội rõ nhất.
   - Contrastive AU–văn bản (C4).
   - Positive chéo danh tính bằng ảnh DISFA đã chuẩn hoá của Norface. **Rủi ro:** mạng chuẩn hoá được fine-tune trên DISFA, có thể đã thấy subject test (dù không thấy nhãn). Phải kiểm trước khi dùng.

---

## 7. Cách tuyên bố kết quả

| Sân | Mốc | Cách nói được phép |
|---|---|---|
| R50 ImageNet, ảnh tĩnh, chỉ nhãn DISFA | SymGraphAU 63.0, ME-GraphAU 63.1 | "vượt SymGraphAU cùng backbone, giao thức chặt hơn"; chỉ dòng (b) đặt cạnh 63.0 |
| Backbone khuôn mặt công khai | FMAE / FMAE-IAT chạy lại trên fold của luận văn | "contrastive nhận biết subject hơn / ngang IAT", kèm Δ từng seed |
| Bảng tham khảo | Norface 72.7, MAE-Face 70.8, DIME 74.7 (preprint), SSM 71.9 (chuỗi + DFEW) | tách bảng; ghi backbone, dữ liệu ngoài, tĩnh hay chuỗi; không tuyên bố SOTA tuyệt đối |

---

## 8. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| `sacon − base2` nằm trong nhiễu | Báo Δ từng seed cùng probe danh tính và độ phân tán ngưỡng; nếu được thì dùng 5 seed cho so sánh chính |
| Sampler P×K làm giảm đa dạng mỗi batch, khiến baseline tụt | Sampler dùng chung cho mọi variant; báo `base2` so với baseline V1 như một dòng mô tả |
| Momentum encoder cùng queue làm run chậm hơn 30–40% | Chạy smoke để đo; nếu quá chậm thì thay queue bằng prototype EMA theo AU (rẻ hơn, kém đa dạng hơn) |
| ViT-B vượt ngân sách | Ưu tiên track R50; track khuôn mặt chỉ chạy 1 seed, ghi rõ trong bài |
| Giấy phép CC BY-NC của FMAE | Luận văn phi thương mại thì dùng được; trích dẫn và ghi giấy phép |
| Survey hoặc số [bài†] sai | Đối chiếu PDF trước khi trích, nhất là AUNCE, UAU-Net, HiVA, DIME, SSM, MAE-Face |

---

## Nguồn đã đọc (28/11/2026)

- FMAE / FMAE-IAT: [arXiv 2407.11243v2](https://arxiv.org/html/2407.11243v2), [GitHub](https://github.com/forever208/FMAE-IAT) (nhánh `mae_pretraining`, `DISFA_finetune_or_ID_adversarial`), checkpoint ở [Hugging Face](https://huggingface.co/forever208/FMAE-IAT) (link lấy từ README, chưa mở)
- MAE-Face: [arXiv 2210.15878](https://arxiv.org/html/2210.15878) · Norface: [arXiv 2407.15617](https://arxiv.org/html/2407.15617v1), [trang dự án](https://norface-fea.github.io), [GitHub](https://github.com/liuhw01/Norface)
- DIME: [arXiv 2610.17015](https://arxiv.org/html/2610.17015v1) · SSM: [arXiv 2604.10541](https://arxiv.org/html/2604.10541v3)
- UAU-Net: [arXiv 2604.21227](https://arxiv.org/html/2604.21227) · HiVA: [arXiv 2602.14425](https://arxiv.org/html/2602.14425) · VL-FAU: [arXiv 2408.00644](https://arxiv.org/html/2408.00644v1)
- AUFormer: [arXiv 2403.04697v2](https://arxiv.org/html/2403.04697v2), [GitHub](https://github.com/yuankaishen2001/AUFormer) · MDHR: [arXiv 2404.06443](https://arxiv.org/html/2404.06443v1), [GitHub](https://github.com/CVI-SZU/MDHR)
- AUNCE: [arXiv 2402.06165v9](https://arxiv.org/html/2402.06165v9), [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320325014098) · CLP: [arXiv 2403.03400](https://arxiv.org/html/2403.03400v1) · D2CA: [arXiv 2503.08977](https://arxiv.org/html/2503.08977) · CLEF: [arXiv 2304.00058](https://arxiv.org/html/2304.00058v2) · CL-ILE: [NTU IRep](https://irep.ntu.ac.uk/id/eprint/55121/) (không mở được)
- Beyond the Fold: [arXiv 2604.02162](https://arxiv.org/html/2604.02162v1) · AUDD: [arXiv 2503.08974](https://arxiv.org/html/2503.08974v1) · UDCT: [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/37037) (trang open access trả 403; chỉ đọc được tóm tắt)
- SymGraphAU: PDF ở `C:\CODE\Symbolic\Paper\`, tóm tắt ở [CONTEXT.md](CONTEXT.md). FLCM: không tìm thấy.
