"""
Contrastive learning cho Tầng 1 (JFL) của SymGraphAU.

Động cơ
-------
Toàn bộ phần ký hiệu của mô hình đứng trên 15 vector tâm lớp (8 AU + 7 biểu cảm)
sinh ra ở Tầng 1 rồi đóng băng. Chúng là nút lá của mọi đồ thị logic ở Tầng 2-3.
Nhưng ở bản gốc, V^a_i chỉ được giám sát bởi một Linear(256 -> 1) phía trên, nên
không có lực nào ép các mẫu cùng bật một AU tụm lại thành cụm chặt — tâm lớp EMA
vì thế là trung bình của một đám mây tán loạn.

Chính paper thừa nhận điều này cho DISFA (mục A3):
    "the logic graphs may be clustered together ... due to the lack of accurate
     semantics of the leaf nodes ... which may be the reason that our method does
     not achieve the best result for DISFA"

Module này cung cấp hai hàm mất mát bổ sung cho Tầng 1:

1. `MultiLabelSupConLoss` — supervised contrastive đa nhãn trên vector gộp
   V^a.mean(dim=1). Positive không còn nhị phân mà là trọng số Jaccard giữa hai
   tập nhãn AU, nên hai khuôn mặt cùng bật {AU6, AU12} là positive mạnh, còn
   {AU6, AU12} với {AU6, AU25} là positive yếu. Vector gộp này cũng chính là
   vector nút Global trong đồ thị logic, nên siết nó là siết cả nút lá lẫn nút
   Global.

2. `CrossModalAUExprLoss` — căn chỉnh không gian AU với không gian biểu cảm theo
   ma trận tri thức M_{A-E}. Vá một lỗ hổng thật của thiết kế gốc: V^a và V^e
   sinh ra từ hai head hoàn toàn tách biệt, không có gì làm chúng cùng thang đo,
   thế nhưng đồ thị logic lại trộn chúng làm cùng một loại nút (cùng mã 4 bit
   1000, cùng ma trận W^l của GCN).

Kèm `EmbeddingQueue` kiểu MoCo để bù số positive quá ít của các AU hiếm
(AU9 chỉ có ~4.2k khung hình dương trên ~87k; batch 64 trung bình chỉ ~3 mẫu).

CẢNH BÁO về queue: ở đây KHÔNG dùng momentum encoder, nên embedding trong queue
là của mạng ở các bước trước (queue 2048 / batch 64 = trễ 32 bước). Với lr 1e-4
độ trôi thường chấp nhận được, nhưng nếu thấy loss dao động mạnh thì đặt
`--con-queue 0` và tăng batch size thay vì tăng queue.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Projection head
# ============================================================

class ProjectionHead(nn.Module):
    """MLP 2 lớp, đầu ra chuẩn hoá L2.

    Nhận đầu vào (..., in_dim) với số chiều đầu tuỳ ý — (B, D) cho vector gộp,
    (B, N, D) cho tập embedding AU/biểu cảm. Mọi chiều đầu được gộp về chiều
    batch trước khi qua BatchNorm, nghĩa là 8 nhánh AU DÙNG CHUNG thống kê BN và
    dùng chung trọng số. Đây là chủ ý: mục tiêu là một không gian chung cho mọi
    AU, không phải 8 không gian rời rạc.
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        lead_shape = x.shape[:-1]
        z = self.net(x.reshape(-1, x.shape[-1]))
        z = F.normalize(z, dim=-1)
        return z.reshape(*lead_shape, self.out_dim)


# ============================================================
# 2. Memory queue
# ============================================================

class EmbeddingQueue(nn.Module):
    """Hàng đợi vòng giữ embedding + nhãn AU của các batch trước.

    Các buffer đặt `persistent=False` nên KHÔNG đi vào checkpoint — queue là
    trạng thái tạm của quá trình huấn luyện, không phải tham số mô hình.
    """

    def __init__(self, dim: int, num_aus: int, size: int = 2048):
        super().__init__()
        self.size = int(size)
        self.register_buffer("feats", torch.zeros(self.size, dim), persistent=False)
        self.register_buffer("labels", torch.zeros(self.size, num_aus), persistent=False)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor, labels: torch.Tensor):
        feats = feats.detach()
        labels = labels.detach()
        b = feats.size(0)
        if b > self.size:                      # batch lớn hơn cả queue
            feats, labels, b = feats[-self.size:], labels[-self.size:], self.size
        p = int(self.ptr.item())
        idx = (torch.arange(b, device=feats.device) + p) % self.size
        self.feats.index_copy_(0, idx, feats.to(self.feats.dtype))
        self.labels.index_copy_(0, idx, labels.to(self.labels.dtype))
        self.ptr[0] = (p + b) % self.size
        self.filled[0] = min(int(self.filled.item()) + b, self.size)

    def get(self):
        """Trả (feats, labels) của phần đã ghi, hoặc (None, None) nếu còn rỗng."""
        n = int(self.filled.item())
        if n == 0:
            return None, None
        return self.feats[:n], self.labels[:n]


# ============================================================
# 3. Trọng số Jaccard giữa hai tập nhãn AU
# ============================================================

def jaccard_weights(y_a: torch.Tensor,
                    y_b: torch.Tensor,
                    neutral_weight: float = 0.5) -> torch.Tensor:
    """Ma trận trọng số positive mềm.

    y_a: (Na, C) nhị phân 0/1
    y_b: (Nb, C) nhị phân 0/1
    ->   (Na, Nb) với w_ij = |A ∩ B| / |A ∪ B|

    Quy ước cho khung hình không bật AU nào (rất nhiều trong DISFA):
      - cả hai đều rỗng  -> `neutral_weight` (mặc định 0.5, KHÔNG phải 1.0, để
        cụm Neutral vốn chiếm đa số không nuốt mất toàn bộ hàm mất mát)
      - chỉ một bên rỗng -> 0.0
    """
    y_a = (y_a > 0.5).float()
    y_b = (y_b > 0.5).float()

    inter = y_a @ y_b.t()                                  # (Na, Nb)
    sa = y_a.sum(dim=1, keepdim=True)                      # (Na, 1)
    sb = y_b.sum(dim=1, keepdim=True).t()                  # (1, Nb)
    union = sa + sb - inter

    w = torch.where(union > 0, inter / union.clamp(min=1e-8), torch.zeros_like(inter))
    both_empty = (sa < 0.5) & (sb < 0.5)                   # broadcast -> (Na, Nb)
    w = torch.where(both_empty, torch.full_like(w, float(neutral_weight)), w)
    return w


# ============================================================
# 4. Multi-label supervised contrastive
# ============================================================

class MultiLabelSupConLoss(nn.Module):
    """SupCon với positive mềm theo Jaccard.

    Với mỗi anchor i:

        L_i = - (1 / Σ_j w_ij) · Σ_j w_ij · log( exp(z_i·z_j/τ) / Σ_k exp(z_i·z_k/τ) )

    trong đó k chạy trên toàn bộ key (batch hiện tại + queue), trừ chính i.
    Anchor không có positive nào (Σ_j w_ij = 0) bị bỏ qua thay vì đóng góp 0 —
    nếu không, mean sẽ bị pha loãng bởi các anchor vô nghĩa.
    """

    def __init__(self,
                 temperature: float = 0.1,
                 neutral_weight: float = 0.5,
                 pos_threshold: float = 0.0):
        super().__init__()
        self.temperature = float(temperature)
        self.neutral_weight = float(neutral_weight)
        self.pos_threshold = float(pos_threshold)

    def forward(self, z, y, z_queue=None, y_queue=None):
        """
        z:       (B, D) đã chuẩn hoá L2, CÓ gradient
        y:       (B, C) nhãn AU nhị phân
        z_queue: (K, D) key bổ sung, KHÔNG gradient (hoặc None)
        y_queue: (K, C) nhãn tương ứng (hoặc None)
        """
        B = z.size(0)
        if z_queue is not None and z_queue.numel() > 0:
            keys = torch.cat([z, z_queue.detach()], dim=0)
            keys_y = torch.cat([y, y_queue.detach()], dim=0)
        else:
            keys, keys_y = z, y
        K = keys.size(0)

        # mặt nạ cặp (i, i) — chỉ tồn tại trong khối batch đầu tiên
        self_mask = torch.zeros(B, K, dtype=torch.bool, device=z.device)
        self_mask[:, :B] = torch.eye(B, dtype=torch.bool, device=z.device)

        logits = (z @ keys.t()) / self.temperature                  # (B, K)
        logits = logits.masked_fill(self_mask, float("-inf"))
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        # QUAN TRỌNG: vị trí bị mask có log_prob = -inf; phải đưa về 0 trước khi
        # nhân với w, nếu không 0 * (-inf) = NaN.
        log_prob = log_prob.masked_fill(self_mask, 0.0)

        w = jaccard_weights(y, keys_y, self.neutral_weight)          # (B, K)
        w = w.masked_fill(self_mask, 0.0)
        if self.pos_threshold > 0:
            w = torch.where(w >= self.pos_threshold, w, torch.zeros_like(w))

        w_sum = w.sum(dim=1)                                         # (B,)
        valid = w_sum > 1e-8
        if int(valid.sum().item()) == 0:
            return z.new_zeros(())

        per_anchor = -(w * log_prob).sum(dim=1) / w_sum.clamp(min=1e-8)
        return per_anchor[valid].mean()


# ============================================================
# 5. Cross-modal AU <-> Expression
# ============================================================

class CrossModalAUExprLoss(nn.Module):
    """Kéo embedding của AU đang bật về phía các biểu cảm mà M_{A-E} cho là liên quan.

    Với mỗi mẫu b và mỗi AU i đang BẬT:
        p_ij = softmax_j( cos(ẑ^a_bi, ẑ^e_bj) / τ )
        q_i  = phân phối mục tiêu suy từ hàng i của M_{A-E}
        L    = trung bình cross-entropy( q_i, p_i ) trên các (b, i) có y_bi = 1

    So sánh diễn ra giữa AU và biểu cảm CỦA CÙNG MỘT ẢNH, nên hàm này căn hai
    head về cùng một thang đo theo từng mẫu, chứ không kéo về một tâm toàn cục.

    AU đang TẮT không tham gia — một AU không xuất hiện thì không nên bị ép gần
    biểu cảm nào cả.
    """

    def __init__(self, temperature: float = 0.1, target_temperature=None):
        super().__init__()
        self.temperature = float(temperature)
        self.target_temperature = target_temperature

    def forward(self, z_au, z_expr, y_a, M_AE):
        """
        z_au:   (B, Na, D) đã chuẩn hoá L2
        z_expr: (B, Ne, D) đã chuẩn hoá L2
        y_a:    (B, Na) nhãn AU nhị phân
        M_AE:   (Na, Ne) ma trận tri thức AU-Biểu cảm (0.9 / 0.5 / 0.1)
        """
        mask = (y_a > 0.5).float()                                   # (B, Na)
        denom = mask.sum()
        if float(denom.item()) < 1.0:
            return z_au.new_zeros(())

        logits = torch.einsum("bid,bjd->bij", z_au, z_expr) / self.temperature
        log_p = F.log_softmax(logits, dim=2)                          # (B, Na, Ne)

        M = M_AE.to(z_au.dtype).to(z_au.device)
        if self.target_temperature is not None:
            q = F.softmax(M / float(self.target_temperature), dim=1)  # (Na, Ne)
        else:
            q = M / M.sum(dim=1, keepdim=True).clamp(min=1e-8)

        ce = -(q.unsqueeze(0) * log_p).sum(dim=2)                     # (B, Na)
        return (ce * mask).sum() / denom.clamp(min=1.0)
