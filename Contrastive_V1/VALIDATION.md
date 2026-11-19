# Kiểm tra triển khai — 2026-09-30

Môi trường local: Windows, Python 3.12, PyTorch 2.7.1+cpu,
torchvision 0.22.1+cpu, dependencies theo requirements-colab.txt.

`python -m pytest tests -q`: **17 passed**. Một FutureWarning từ import
`timm.models.layers` của backbone Swin cũ; không có failure.

Phạm vi kiểm tra:

- Node SupCon ưu tiên AU-on gần nhau và tách AU-off; gradient hữu hạn.
- Batch toàn on/off, AU không đủ positive, singleton global, không có positive.
- Cross-modal hai chiều truyền gradient vào cả AU và expression; neutral bỏ qua.
- Queue rollover, không làm thay đổi trạng thái no_grad trong WAL.
- Manifest phát hiện giao nhau/chỉnh sửa split; metric F1 và summary ghép cặp.
- Train/resume/evaluate với synthetic images và model fixture nhẹ cho ba variants
  baseline/current/full: trọng số sau resume trùng bit-for-bit với chạy liên tục
  trên CPU trong test, gồm khôi phục queue của current.
- Model thật MEFARGStage1 + ResNet18: forward/backward, shape, gradient raw heads,
  giao diện 4 đầu ra khi inference và projection 128 chiều khi train.
- Batched GCN khớp hai forward riêng; Stage-3 GCN đóng băng vẫn truyền gradient
  qua logic loss đến probabilities của detector.
- CLI train/evaluate/Stage 2/Stage 3: kiểm tra help/parse.
- Notebook: JSON hợp lệ và mọi code cell parse được bằng Python AST.

Chưa chạy full DISFA, ImageNet-pretrained ResNet50 train, full Stage 2/3 hoặc
GPU Colab session. Chưa có kết quả F1 để xác nhận cải thiện. Notebook cần dataset,
GPU runtime và người dùng mount Drive. Tests CPU không chứng minh tái lập bit-level
giữa GPU khác nhau hoặc performance thực tế.

Các hạn chế symbolic graph legacy được ghi trong EXPERIMENT_PROTOCOL.md;
V1 chưa sửa toàn bộ CNF/polarity/SAT và không được gọi là reproduction paper.
