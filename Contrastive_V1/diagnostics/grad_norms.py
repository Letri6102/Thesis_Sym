"""Chuẩn gradient của từng số hạng loss Tầng 1 trên vài batch fit (giả thuyết V1-d, Huong_SOTA_Contrastive.md §3.3).

  python -m diagnostics.grad_norms --checkpoint <run>/last.pt --manifest <Sym>/splits_v2/fold2.json \
      --data-root /content/DISFA [--batches 20] [--output grad_norms.json]

Dựng lại đúng loss của train_contrastive.train tại trạng thái của checkpoint: model.train(), strong_train_aug, nhãn
biểu cảm giả, trọng số của variant, ramp theo epoch của checkpoint (queue của variant `current` không được dựng lại).
Không có optimizer.step(): trọng số không đổi; chỉ BN running stats trong bộ nhớ thay đổi và không được lưu.

Với mỗi số hạng: chuẩn gradient trên `backbone`, `backbone_last` (layer4 của ResNet, stage cuối của Swin) và
`head` (global_linear + AU head), cả bản thô lẫn bản đã nhân hệ số. Thêm hai số cho phần contrastive
(tổng global + node + cross đã nhân hệ số): tỉ số ‖∇contrastive‖ / ‖∇L_wa‖, và cos(∇L_wa, ∇contrastive).
Cùng `--seed` thì baseline và full của một fold đo trên cùng các batch.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TERMS = ("wa", "we", "global", "node", "cross")
CON = ("global", "node", "cross")
GROUPS = ("backbone", "backbone_last", "head")
N_BATCHES, SEED = 20, 2026


def param_groups(model):
    bb = model.backbone
    last = bb.layer4 if hasattr(bb, "layer4") else bb.layers[-1]
    head = list(model.global_linear.parameters()) + list(model.au_head.parameters())
    return dict(backbone=list(bb.parameters()), backbone_last=list(last.parameters()), head=head)


def coefficients(cfg, progress):
    vs = cfg["variant_settings"]
    ramp = min(1., progress / max(vs["warmup"], 1))
    return ramp, dict(wa=1., we=float(cfg["gamma"]), **{t: float(vs[f"{t}_w"]) * ramp for t in CON})


def measure(checkpoint, manifest, data_root, n_batches=N_BATCHES, batch_size=None, workers=4, device=None, seed=SEED):
    from experiment_data import MATRIX_PATH, ExperimentFrames, load_manifest, loader_for
    from model.contrastive import CrossModalAUExprLoss, MultiLabelSupConLoss, NodeAUSupConLoss
    from utils import ExpressionBCELoss, WeightedAsymmetricLoss
    from .common import check_manifest, load_model

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, meta = load_model(checkpoint, "train")
    check_manifest(cfg, manifest)
    m = load_manifest(manifest, data_root)
    vs = cfg["variant_settings"]
    ramp, coef = coefficients(cfg, meta["progress"])
    model.to(device).train()
    groups = param_groups(model)
    params, seen = [], set()
    for ps in groups.values():
        for p in ps:
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)

    matrix = torch.tensor(np.load(MATRIX_PATH), device=device, dtype=torch.float32)
    wal = WeightedAsymmetricLoss(weight=torch.tensor(m["train_weights"], device=device))
    expr = ExpressionBCELoss()
    global_loss, node_loss = MultiLabelSupConLoss(cfg["temperature"]), NodeAUSupConLoss(cfg["temperature"])
    cross_loss = CrossModalAUExprLoss(cfg["temperature"], bidirectional=vs["bidirectional"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader = loader_for(ExperimentFrames(data_root, m, "fit"), batch_size or cfg["batch_size"], workers, seed, True)
    value = {t: [] for t in TERMS}
    raw_norm = {t: {g: [] for g in GROUPS} for t in TERMS}
    con_norm, ratio, cos = ({g: [] for g in GROUPS} for _ in range(3))
    used = 0
    for images, labels in loader:
        if used >= n_batches:
            break
        if images.size(0) < 2:                         # như lúc train: batch 1 ảnh không chạy contrastive
            continue
        images, labels = images.to(device), labels.to(device)
        model.zero_grad(set_to_none=True)
        _, _, pa, pe, zg, za, ze = model(images, return_proj=True)
        k = (labels @ matrix).argmax(1)
        k[labels.sum(1) == 0] = 6
        raw = {"wa": wal(pa, labels), "we": expr(pe, F.one_hot(k, 7).float()), "global": global_loss(zg, labels),
               "node": node_loss(za, labels), "cross": cross_loss(za, ze, labels, matrix)}
        grads = {}
        for t in TERMS:
            value[t].append(float(raw[t].detach()))
            g = torch.autograd.grad(raw[t], params, retain_graph=True, allow_unused=True)
            grads[t] = {id(p): (gi if gi is not None else torch.zeros_like(p)) for p, gi in zip(params, g)}
        for name, ps in groups.items():
            vec = {t: torch.cat([grads[t][id(p)].flatten() for p in ps]) for t in TERMS}
            for t in TERMS:
                raw_norm[t][name].append(float(vec[t].norm()))
            con = sum(coef[t] * vec[t] for t in CON)
            wa_n, con_n = float(vec["wa"].norm()), float(con.norm())
            con_norm[name].append(con_n)
            ratio[name].append(con_n / wa_n if wa_n > 0 else float("nan"))
            cos[name].append(float(F.cosine_similarity(vec["wa"], con, dim=0)) if wa_n > 0 and con_n > 0 else float("nan"))
        used += 1
    del model

    mean = lambda v: float(np.mean(v)) if v and np.isfinite(v).all() else float("nan")  # noqa: E731
    return dict(checkpoint=Path(checkpoint).name, epoch=meta["epoch"], progress=meta["progress"],
                variant=cfg["variant"], fold=m["fold"], seed=cfg["seed"], ramp=ramp, coefficients=coef,
                queue_ignored=bool(vs.get("queue")), n_batches=used, batch_size=batch_size or cfg["batch_size"],
                batch_seed=seed,
                loss={t: mean(value[t]) for t in TERMS},
                loss_weighted={t: coef[t] * mean(value[t]) for t in TERMS},
                grad_norm_raw={t: {g: mean(raw_norm[t][g]) for g in GROUPS} for t in TERMS},
                grad_norm_weighted={t: {g: coef[t] * mean(raw_norm[t][g]) for g in GROUPS} for t in TERMS},
                contrastive_norm={g: mean(con_norm[g]) for g in GROUPS},
                ratio_contrastive_to_wa={g: mean(ratio[g]) for g in GROUPS},
                cos_wa_contrastive={g: mean(cos[g]) for g in GROUPS})


def main(argv=None):
    from .common import write_json
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--batches", type=int, default=N_BATCHES)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device")
    ap.add_argument("--output")
    args = ap.parse_args(argv)
    r = measure(args.checkpoint, args.manifest, args.data_root, args.batches, args.batch_size, args.workers,
                args.device, args.seed)
    print(f"{r['variant']} epoch {r['epoch']}: ‖∇contrastive‖/‖∇L_wa‖ trên backbone {r['ratio_contrastive_to_wa']['backbone']:.3g}, "
          f"cos {r['cos_wa_contrastive']['backbone']:.3f}")
    if args.output:
        write_json(args.output, r)
    return r


if __name__ == "__main__":
    main()
