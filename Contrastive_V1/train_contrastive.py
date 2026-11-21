"""Controlled Stage-1 ablations. See EXPERIMENT_PROTOCOL.md before running."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F

from experiment_data import ROOT, MATRIX_PATH, AU_NAMES, digest, load_manifest, ExperimentFrames, loader_for
from model.SymStage1 import MEFARGStage1
from model.contrastive import MultiLabelSupConLoss, NodeAUSupConLoss, CrossModalAUExprLoss, EmbeddingQueue
from utils import WeightedAsymmetricLoss, ExpressionBCELoss

# Fixed designs; changing these changes the source hash and disallows resume.
VARIANTS = {
    'baseline': dict(global_w=0., node_w=0., cross_w=0., bidirectional=True, queue=0, warmup=3),
    'global': dict(global_w=.05, node_w=0., cross_w=0., bidirectional=True, queue=0, warmup=3),
    'node': dict(global_w=0., node_w=.05, cross_w=0., bidirectional=True, queue=0, warmup=3),
    'cross': dict(global_w=0., node_w=0., cross_w=.05, bidirectional=True, queue=0, warmup=3),
    'global_cross': dict(global_w=.05, node_w=0., cross_w=.05, bidirectional=True, queue=0, warmup=3),
    'full': dict(global_w=.05, node_w=.05, cross_w=.05, bidirectional=True, queue=0, warmup=3),
    'current': dict(global_w=.1, node_w=0., cross_w=.1, bidirectional=False, queue=2048, warmup=1),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def source_files():
    paths = list(ROOT.glob('*.py')) + list((ROOT / 'model').glob('*.py'))
    paths += list((ROOT / 'config').glob('*.yaml'))
    paths += [MATRIX_PATH, ROOT / 'EXPERIMENT_PROTOCOL.md']
    return {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in sorted(paths)}


def state_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith('proj_'):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def metrics(labels, probs):
    positive = probs >= .5
    truth = labels > .5
    tp = (positive & truth).sum(0)
    fp = (positive & ~truth).sum(0)
    fn = (~positive & truth).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return dict(macro_f1=float(f1.mean()), per_au_f1=f1.tolist(),
                tp=tp.tolist(), fp=fp.tolist(), fn=fn.tolist(), threshold=.5)


@torch.no_grad()
def predict(model, data, batch, workers, device):
    model.eval()
    targets, probabilities = [], []
    for images, labels in loader_for(data, batch, workers, 0):
        _, _, pa, _ = model(images.to(device))
        if not torch.isfinite(pa).all() or (pa < 0).any() or (pa > 1).any():
            raise FloatingPointError('Detector must return finite probabilities in [0,1]')
        targets.append(labels.numpy())
        probabilities.append(pa.cpu().numpy())
    y, p = np.concatenate(targets), np.concatenate(probabilities)
    return metrics(y, p), y, p


class ModelEMA:
    """Trung bình mũ của trọng số (kể cả BN running stats); tensor không phải số thực thì chép thẳng.
    decay khởi động theo (1 + t) / (10 + t) để giai đoạn đầu không bị kéo về trọng số khởi tạo."""
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        current = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(d).add_(current[name].detach(), alpha=1 - d)
            else:
                value.copy_(current[name])


def check_steps(steps_per_epoch, checks_per_epoch):
    """Chỉ số step (0-based) có đánh giá val giữa epoch; lần cuối epoch đánh giá riêng."""
    marks = {math.ceil(k * steps_per_epoch / checks_per_epoch) - 1 for k in range(1, checks_per_epoch)}
    return sorted(s for s in marks if 0 <= s < steps_per_epoch - 1)


def selection_points(history):
    """Mọi lần đánh giá val theo thứ tự thời gian: (progress, metrics).
    progress = số epoch đã train (cuối epoch e là e). Run chỉ đánh giá cuối epoch: mỗi epoch một điểm."""
    return [(c['progress'], c['val']) for r in history
            for c in (r.get('val_checks') or [dict(progress=r['epoch'], val=r['val'])])]


def best_point(history):
    """Điểm có macro-F1 val cao nhất; hoà thì lấy điểm sớm hơn (max giữ phần tử đầu)."""
    return max(selection_points(history), key=lambda pv: pv[1]['macro_f1'])


def restore_detector(model, checkpoint):
    state = {k.removeprefix('module.'): v for k, v in checkpoint['state_dict'].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in list(missing) + list(unexpected) if not k.startswith('proj_')]
    if bad:
        raise ValueError(f'Detector checkpoint mismatch: {bad}')


def train(args):
    if args.epochs <= 0 or args.batch_size < 2 or args.temperature <= 0 or args.lr <= 0 or args.weight_decay < 0 or args.workers < 0:
        raise ValueError('Positive epochs/lr/temperature, batch-size>=2 and nonnegative wd/workers required')
    if args.stop_after is not None and args.stop_after <= 0:
        raise ValueError('stop-after must be positive')
    if not 0 <= args.ema_decay < 1 or args.val_per_epoch < 1 or args.backbone_lr_mult <= 0:
        raise ValueError('ema-decay in [0,1), val-per-epoch >= 1, backbone-lr-mult > 0 required')
    m = load_manifest(args.manifest, args.data_root)
    device = torch.device(args.device)
    variant = VARIANTS[args.variant]
    config = dict(variant=args.variant, seed=args.seed, fold=m['fold'], arc=args.arc,
                  epochs=args.epochs, batch_size=args.batch_size, workers=args.workers,
                  lr=args.lr, weight_decay=args.weight_decay, temperature=args.temperature,
                  gamma=.05, proj_dim=128, precision='float32', threshold=.5,
                  expr_loss='legacy_bce', variant_settings=variant,
                  pretrained=not args.no_pretrained,
                  ema_decay=args.ema_decay, val_per_epoch=args.val_per_epoch,
                  backbone_lr_mult=args.backbone_lr_mult,
                  environment_versions={k: importlib.metadata.version(k) for k in ('torch', 'torchvision', 'numpy', 'timm', 'pillow')},
                  manifest_hash=digest(args.manifest), source_hashes=source_files())
    seed_all(args.seed)
    # Same initialization and number/order of created heads for every variant.
    # Resume/evaluate never need ImageNet download.
    model = MEFARGStage1(num_aus=8, num_expr=7, backbone=args.arc,
                        use_contrastive=True, proj_dim=128,
                        pretrained=not args.no_pretrained and not args.resume)
    init_hash = state_digest(model)
    model.to(device)
    # Hai nhóm tham số: backbone (lr × backbone_lr_mult) và phần còn lại (lr). Mult = 1 cho cùng kết quả
    # như một nhóm duy nhất: AdamW cập nhật từng tham số độc lập.
    backbone = getattr(model, 'backbone', None)
    bb_params = list(backbone.parameters()) if backbone is not None else []
    if args.backbone_lr_mult != 1 and not bb_params:
        raise ValueError('backbone-lr-mult needs a model with a parameterised .backbone')
    bb_ids = {id(p) for p in bb_params}
    groups = [dict(params=bb_params, lr_mult=args.backbone_lr_mult),
              dict(params=[p for p in model.parameters() if id(p) not in bb_ids], lr_mult=1.)]
    optimizer = torch.optim.AdamW([g for g in groups if g['params']], lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(.9, .999))
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    fit = ExperimentFrames(args.data_root, m, 'fit')
    val = ExperimentFrames(args.data_root, m, 'val')
    matrix = torch.tensor(np.load(MATRIX_PATH), device=device, dtype=torch.float32)
    weights = torch.tensor(m['train_weights'], device=device)
    wal, expr = WeightedAsymmetricLoss(weight=weights), ExpressionBCELoss()
    global_loss = MultiLabelSupConLoss(args.temperature)
    node_loss = NodeAUSupConLoss(args.temperature)
    cross_loss = CrossModalAUExprLoss(args.temperature, bidirectional=variant['bidirectional'])
    queue = EmbeddingQueue(128, 8, variant['queue']).to(device) if variant['queue'] else None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start, best, history = 0, -1., []
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if ckpt['config'] != config:
            raise ValueError('Resume config/source/split differs; use the original run configuration')
        if Path(args.resume).resolve().parent != output.resolve():
            raise ValueError('Resume into the original output directory (preserve best.pt)')
        # Có EMA: 'state_dict' là trọng số EMA (dùng để đánh giá), 'train_state_dict' là trọng số đang train.
        model.load_state_dict(ckpt.get('train_state_dict', ckpt['state_dict']), strict=True)
        if ema is not None:
            ema.module.load_state_dict(ckpt['state_dict'], strict=True)
            ema.updates = ckpt['ema_updates']
        optimizer.load_state_dict(ckpt['optimizer'])
        start, best, history = ckpt['epoch'] + 1, ckpt['best_val_f1'], ckpt['history']
        init_hash = ckpt['initial_detector_hash']
        if queue is not None:
            for name, buf in queue.named_buffers():
                buf.copy_(ckpt['queue'][name].to(device))
        random.setstate(ckpt['rng_python'])
        np.random.set_state(ckpt['rng_numpy'])
        torch.set_rng_state(ckpt['rng_torch'])
        if device.type == 'cuda':
            if ckpt['rng_cuda'] is not None:
                torch.cuda.set_rng_state_all(ckpt['rng_cuda'])
    else:
        if (output / 'config.json').exists() or (output / 'last.pt').exists():
            raise FileExistsError('Run already exists; use --resume or a new output directory')
        for name in config['source_hashes']:
            dest = output / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        shutil.copy2(args.manifest, output / 'manifest.json')
        environment = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                           cuda=torch.version.cuda, device=str(device),
                           gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                           packages={p.metadata['Name']: p.version for p in importlib.metadata.distributions() if p.metadata['Name']})
        (output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
        (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    print(json.dumps(dict(config=config, initial_detector_hash=init_hash), indent=2))
    if args.stop_after is not None and start >= args.stop_after:
        print('Requested pilot stop epoch already reached; no additional training.')
        return
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    evaluated = ema.module if ema is not None else model

    def make_checkpoint(epoch, progress, full):
        # best.pt chỉ cần trọng số để evaluate; last.pt thêm optimizer/RNG/EMA để resume.
        ck = dict(state_dict=evaluated.state_dict(), epoch=epoch, progress=progress, config=config,
                  initial_detector_hash=init_hash, manifest=m, history=history, best_val_f1=best)
        if full:
            ck.update(optimizer=optimizer.state_dict(),
                      queue={n: t.detach().cpu().clone() for n, t in queue.named_buffers()} if queue else None,
                      rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                      rng_torch=torch.get_rng_state(),
                      rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
            if ema is not None:
                ck.update(train_state_dict=model.state_dict(), ema_updates=ema.updates)
        return ck

    def val_check(epoch, step, progress):
        nonlocal best
        validation, _, _ = predict(evaluated, val, args.batch_size, args.workers, device)
        model.train()                                    # predict() chuyển sang eval khi không có EMA
        improved = validation['macro_f1'] > best         # hoà → giữ lần đánh giá sớm hơn
        best = max(best, validation['macro_f1'])
        if improved:
            atomic_save(make_checkpoint(epoch, progress, full=False), output / 'best.pt')
        if args.val_per_epoch > 1:
            print(f'val_check {progress:g} macro_f1={validation["macro_f1"]:.4f} best={best:.4f}'
                  + (' *' if improved else ''), flush=True)
        return dict(progress=progress, step=step, val=validation)

    for epoch in range(start, args.epochs):
        tick = time.perf_counter()
        # Independent epoch RNG makes augmentation order comparable even with queue/heads.
        seed_all(args.seed * 10000 + epoch + 1000)
        data = loader_for(fit, args.batch_size, args.workers, args.seed * 10000 + epoch, True)
        model.train()
        sums = dict(total=0., wa=0., we=0., global_con=0., node_con=0., cross_con=0.)
        eligible = np.zeros(8)
        seen = 0
        mid_checks, checks = set(check_steps(len(data), args.val_per_epoch)), []
        for step, (images, labels) in enumerate(data):
            lr = args.lr * (1 + np.cos(np.pi * (epoch + step / len(data)) / args.epochs)) / 2
            for group in optimizer.param_groups:
                group['lr'] = lr * group['lr_mult']
            images, labels = images.to(device), labels.to(device)
            # BatchNorm in global projection requires >1 image. Keep singleton image
            # for detection training; skip all contrastive paths on that last batch.
            use_con = args.variant != 'baseline' and images.size(0) > 1
            optimizer.zero_grad(set_to_none=True)
            values = model(images, return_proj=use_con)
            va, ve, pa, pe = values[:4]
            k = (labels @ matrix).argmax(1)
            k[labels.sum(1) == 0] = 6
            pseudo = F.one_hot(k, 7).float()
            lwa, lwe = wal(pa, labels), expr(pe, pseudo)
            lg = ln = lx = pa.sum() * 0.
            if use_con:
                zg, za, ze = values[4:]
                if variant['global_w']:
                    qf, ql = queue.get() if queue is not None else (None, None)
                    lg = global_loss(zg, labels, qf, ql)
                if variant['node_w']:
                    ln = node_loss(za, labels)
                    eligible += node_loss.last_eligible.cpu().numpy()
                if variant['cross_w']:
                    lx = cross_loss(za, ze, labels, matrix)
            ramp = min(1., (epoch + step / len(data)) / max(variant['warmup'], 1))
            loss = lwa + .05 * lwe + ramp * (variant['global_w'] * lg + variant['node_w'] * ln + variant['cross_w'] * lx)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch={epoch}, step={step}')
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            if queue is not None and use_con:
                queue.enqueue(zg, labels)
            bs = len(labels)
            seen += bs
            for name, value in zip(sums, [loss, lwa, lwe, lg, ln, lx]):
                sums[name] += value.detach().item() * bs
            if step % 100 == 0:
                print(f'epoch {epoch+1}/{args.epochs} step {step}/{len(data)} loss={loss.item():.5f}', flush=True)
            if step in mid_checks:
                checks.append(val_check(epoch, step, round(epoch + (step + 1) / len(data), 6)))
        checks.append(val_check(epoch, len(data) - 1, epoch + 1))
        record = dict(epoch=epoch + 1, train={k: v / seen for k, v in sums.items()},
                      val=checks[-1]['val'], seconds=time.perf_counter() - tick,
                      node_eligible_batch_fraction=(eligible / len(data)).tolist(),
                      peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
        if len(checks) > 1:
            record['val_checks'] = checks
        history.append(record)
        atomic_save(make_checkpoint(epoch, epoch + 1, full=True), output / 'last.pt')
        (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)
        if args.stop_after is not None and epoch + 1 >= args.stop_after:
            break


def evaluate(args):
    # Only load trusted, locally produced checkpoints: optimizer/RNG use pickle.
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    reference = ckpt if 'config' in ckpt else ckpt.get('experiment')
    if reference is None:
        raise ValueError('Checkpoint lacks controlled experiment metadata')
    cfg = reference['config']
    if args.split == 'test' and 'config' in ckpt:
        history_path = Path(args.checkpoint).parent / 'history.json'
        completed = json.loads(history_path.read_text()) if history_path.exists() else []
        if not completed or completed[-1]['epoch'] != cfg['epochs']:
            raise ValueError('Finish the planned Stage-1 epochs before outer-test evaluation; use --split val for pilot')
        selected, _ = best_point(completed)
        if ckpt.get('progress', ckpt['epoch'] + 1) != selected:
            raise ValueError('Test requires the best inner-validation checkpoint (earliest check on ties)')
    m = load_manifest(args.manifest, args.data_root)
    if digest(args.manifest) != cfg['manifest_hash']:
        raise ValueError('Evaluation manifest differs from training')
    model = MEFARGStage1(num_aus=8, num_expr=7, backbone=cfg['arc'],
                        use_contrastive=False, pretrained=False)
    restore_detector(model, ckpt)
    device = torch.device(args.device)
    model.to(device)
    data = ExperimentFrames(args.data_root, m, args.split)
    result, labels, probs = predict(model, data, args.batch_size, args.workers, device)
    result.update(variant=cfg['variant'], seed=cfg['seed'], fold=m['fold'], split=args.split,
                  stage=3 if 'experiment' in ckpt else 1,
                  config=cfg, initial_detector_hash=reference['initial_detector_hash'],
                  checkpoint_hash=digest(args.checkpoint),
                  selected_epoch=ckpt['epoch'] + 1, selected_progress=ckpt.get('progress', ckpt['epoch'] + 1),
                  au_names=AU_NAMES)
    if 'experiment' in ckpt:
        result['downstream'] = {k: reference[k] for k in ('stage2', 'stage3')}
    else:
        hp = Path(args.checkpoint).parent / 'history.json'
        run_history = json.loads(hp.read_text()) if hp.exists() else ckpt['history']
        result['training'] = dict(
            total_train_val_seconds=sum(r['seconds'] for r in run_history),
            peak_vram_bytes=max(r['peak_vram_bytes'] for r in run_history),
            mean_node_eligible_batch_fraction=np.mean(
                [r['node_eligible_batch_fraction'] for r in run_history], axis=0).tolist())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'{args.split}_metrics.json'
    if target.exists():
        raise FileExistsError(f'Evaluation already exists: {target}')
    np.savez_compressed(out / f'{args.split}_predictions.npz', labels=labels, probabilities=probs,
                        paths=np.asarray(data.paths), subjects=np.asarray([p.split('/')[0] for p in data.paths]))
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'config'}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    tr = sub.add_parser('train')
    ev = sub.add_parser('evaluate')
    for s in (tr, ev):
        s.add_argument('--data-root', required=True)
        s.add_argument('--manifest', required=True)
        s.add_argument('--output', required=True)
        s.add_argument('--batch-size', type=int, default=32)
        s.add_argument('--workers', type=int, default=2)
        s.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    tr.add_argument('--variant', choices=VARIANTS, required=True)
    tr.add_argument('--arc', choices=['resnet18', 'resnet50', 'resnet101', 'swin_transformer_base', 'swin_transformer_small', 'swin_transformer_tiny'], default='resnet50')
    tr.add_argument('--seed', type=int, default=0)
    tr.add_argument('--epochs', type=int, default=20)
    tr.add_argument('--lr', type=float, default=1e-4)
    tr.add_argument('--weight-decay', type=float, default=5e-4)
    tr.add_argument('--temperature', type=float, default=.1)
    tr.add_argument('--no-pretrained', action='store_true', help='Only for smoke tests or a separate random-init study')
    tr.add_argument('--resume', default='')
    tr.add_argument('--stop-after', type=int, help='Stop after this epoch without changing schedule; resume from last.pt')
    tr.add_argument('--ema-decay', type=float, default=0.,
                    help='0 = off. >0: evaluate/select/save EMA weights (warm-up (1+t)/(10+t))')
    tr.add_argument('--val-per-epoch', type=int, default=1,
                    help='inner-val evaluations per epoch; checkpoint chosen over all of them (earliest on ties)')
    tr.add_argument('--backbone-lr-mult', type=float, default=1.,
                    help='backbone lr = lr × this; heads/projections keep lr')
    ev.add_argument('--checkpoint', required=True)
    ev.add_argument('--split', choices=['val', 'test'], default='test')
    args = p.parse_args()
    train(args) if args.command == 'train' else evaluate(args)


if __name__ == '__main__':
    main()
