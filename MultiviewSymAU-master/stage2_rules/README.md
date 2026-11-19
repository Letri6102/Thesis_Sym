# Stage 2 v2: luật AU–AU kiểm định + decoder cố định

Đây là phiên bản thử đầu tiên: bước 2.1 + 2.2 + decoder MAP cố định. Không train thêm gì ngoài detector Stage 1.
Mục tiêu: xem luật có tăng **macro-F1 thật** trên outer test so với **cả hai** đối chứng không luật hay không.

| | Mô tả |
|---|---|
| C0 | detector Stage 1, ngưỡng 0.5 |
| C1 | detector Stage 1, ngưỡng riêng từng AU tối ưu trên inner |
| R0 | MAP có luật, bias 0, λ chọn trên inner |
| R1 | MAP có luật, bias = ngưỡng của C1, λ chọn trên inner |

Nguyên tắc dữ liệu:
- Luật chỉ được ước lượng từ **nhãn** của tập `fit`.
- Ngưỡng và λ chỉ được chọn trên `inner`.
- Outer `test` chỉ dùng để báo cáo.
- Biến thể luật dùng để kết luận (R0 hay R1) cũng được chọn theo inner.

## Chạy (PowerShell, từ thư mục gốc repo)

```powershell
# 0) Chia train → fit / inner theo subject (1 lần, cho cả 3 fold)
python -m stage2_rules.make_inner_split --dataset DISFA --n-inner 6

# 1) Train lại Stage 1 trên fit, chọn best trên inner (mỗi fold)
python .\train_Sym_Stage_1_fit.py --dataset DISFA --arc resnet50 --exp-name resnet50_stage1_fit -b 64 -lr 0.0001 --fold 2

# 2) Luật (2.1 + 2.2) — chỉ đọc nhãn fit, chạy vài giây
python -m stage2_rules.build_rules --dataset DISFA --fold 2

# 3) Xác suất của detector trên inner và test
python -m stage2_rules.dump_probs --dataset DISFA --arc resnet50 --fold 2 `
    --ckpt results/resnet50_stage1_fit/bs_64_seed_0_lr_0.0001/best_model_fold2.pth --splits inner test

# 4) Decoder + so sánh + bootstrap Δ theo subject
python -m stage2_rules.decode_eval --dataset DISFA --fold 2

# 5) Sau khi chạy đủ 3 fold
python -m stage2_rules.summarize --dataset DISFA
```

Mọi đầu ra nằm ở `results/stage2_rules/DISFA/fold{k}/`:
- `rules_all.csv`: mọi ứng viên, gồm `luật | support | ngoại lệ | kỳ vọng | lift | CI95 | %subject | accepted | w0`
- `rules.json`, `V.npy`: luật được nhận và ma trận vi phạm trên 2^8 cấu hình
- `probs_inner.npz`, `probs_test.npz`
- `decode_summary.json/.csv`

## Vì sao phải train lại Stage 1 (bước 1)

`train_Sym_Stage_1.py` hiện tại:
- train trên **toàn bộ** train split, nên detector đã thấy các subject của inner;
- chọn `best_model` trên **outer test**.

Nếu dùng checkpoint này thì:
- xác suất trên inner là in-sample (quá tự tin), nên λ và ngưỡng chọn trên inner bị lệch;
- con số trên test đã bị lạc quan sẵn.

`decode_eval` sẽ in **[CẢNH BÁO]** nếu F1@0.5 trên inner cao hơn test quá 10 điểm.

Muốn xem thử nhanh bằng checkpoint cũ thì vẫn chạy được từ bước 2, nhưng **không được báo cáo** kết quả đó.

## Tham số kiểm định (build_rules)

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--ucb-max` | 0.8 | Nhận luật nếu cận trên CI95 của lift vi phạm < 0.8 |
| `--min-expected` | 50 | Số frame vi phạm kỳ vọng nếu độc lập phải ≥ 50, để tránh nhầm "hai AU hiếm" với "loại trừ" |
| `--min-subject-frac` | 0.7 | ≥ 70% subject cùng chiều; chỉ xét subject có kỳ vọng ≥ `--min-subject-expected` (5) |
| `--conf-min` | 0.6 | Riêng luật kéo theo: CI95 dưới của P(j\|i) ≥ 0.6 |
| `--no-implication` | | Chỉ dùng luật loại trừ, đúng với đề xuất ban đầu |

Trọng số: `w0 = −log UCB95(lift_vi_phạm)`, chặn trên bởi `--w-max` (5).

## Đọc kết quả

- Các cột `help` / `hurt` (trên test):
  - `help`: số cặp (frame, luật) dự đoán vi phạm luật trong khi nhãn thật thoả luật, tức là chỗ luật có thể sửa.
  - `hurt`: số cặp mà nhãn thật cũng vi phạm, tức là ngoại lệ thật.
- Nếu `help` của C1 nhỏ thì decoder không có nhiều chỗ để thắng.
- Tiêu chí kết luận: biến thể luật (chọn theo inner) có Δ > 0 so với **cả C0 và C1**, với P(Δ>0) ≥ 0.9 qua bootstrap theo subject, và lặp lại ở nhiều fold.
