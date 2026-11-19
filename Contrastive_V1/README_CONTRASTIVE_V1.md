# Contrastive V1

1. [Quy tắc thực nghiệm, pipeline và đối chứng](EXPERIMENT_PROTOCOL.md)
2. [Hướng dẫn Colab Pro, train/resume/evaluate](COLAB_PRO.md)
3. [Notebook tải lên Colab](notebooks/Colab_Contrastive_V1.ipynb)

Entrypoint chuẩn: `train_contrastive.py train|evaluate`.
`experiment_data.py` tạo manifest 3 folds; `summarize_experiments.py` so sánh
ghép cặp; `prepare_disfa_lists.py` tạo lists nếu chưa có và đã xác nhận frame IDs.

Đã triển khai: global Jaccard SupCon, node-level AU SupCon (AU-on anchors),
cross-modal soft-target hai chiều, warm-up, 7 cấu hình ablation, logging coverage,
checkpoint epoch-boundary có optimizer/queue/RNG, test tách khỏi train.
`model/SymStage1.py` vẫn trả 4 tensors mặc định; projection chỉ tham gia loss
train. Forward test/Stage 2/3 không cần projection.

Phạm vi hiện tại: DISFA. Không thay đổi `../MultiviewSymAU-master`.
`train_Sym_Stage_1.py` và `test.py` là legacy; entrypoint cũ dừng với hướng dẫn
runner mới để tránh dùng outer test làm validation hoặc tạo sai model.

Thay đổi chung ngoài CL trong clone: đường dẫn matrix portable Linux, strict
checkpoint loading ngoài projection, WAL không bật grad ngoài ý muốn, Stage 2/3
bỏ sigmoid lần hai và hỗ trợ inner split. BCE expression giữ nguyên để không
lẫn thay đổi objective vào ablation. Source snapshot/hash lưu cùng từng run.

Unit/integration tests: `python -m pytest tests -q`.
Tests không phải kết quả train DISFA; chưa có cơ sở tuyên bố F1 tăng/giảm.
