"""Paired fold/seed comparison. Never select a winner from one best run."""
import argparse
import json
from pathlib import Path
import numpy as np


def collect(root, variant, stage):
    found = {}
    for path in Path(root).rglob('test_metrics.json'):
        r = json.loads(path.read_text())
        if r['variant'] != variant or r['stage'] != stage:
            continue
        key = (r['fold'], r['seed'])
        if key in found:
            raise ValueError(f'Duplicate fold/seed for {variant}: {key}')
        found[key] = r
    return found


def summarize(root, baseline, candidate, stage=1):
    a, b = collect(root, baseline, stage), collect(root, candidate, stage)
    if not a or set(a) != set(b):
        raise ValueError('Need nonempty, matching fold/seed sets for both variants')
    seeds = sorted({s for _, s in a})
    for s in seeds:
        if {f for f, seed in a if seed == s} != {1, 2, 3}:
            raise ValueError(f'Seed {s} needs all three folds; no best-fold summary')
    shared = ('seed', 'fold', 'arc', 'epochs', 'batch_size', 'workers', 'lr', 'weight_decay',
              'temperature', 'gamma', 'proj_dim', 'precision', 'threshold', 'expr_loss',
              'pretrained', 'environment_versions', 'manifest_hash', 'source_hashes')
    deltas = []
    for key in sorted(a):
        if stage == 3 and a[key].get('downstream') != b[key].get('downstream'):
            raise ValueError(f'Downstream config/source differs: {key}')
        for field in shared:
            if a[key]['config'][field] != b[key]['config'][field]:
                raise ValueError(f'Unpaired config {key}: {field}')
        if a[key]['initial_detector_hash'] != b[key]['initial_detector_hash']:
            raise ValueError(f'Unpaired initial detector weights: {key}')
        deltas.append(dict(fold=key[0], seed=key[1],
                           delta_pp=100 * (b[key]['macro_f1'] - a[key]['macro_f1'])))
    per_seed = []
    for seed in seeds:
        av = np.mean([a[(f, seed)]['macro_f1'] for f in (1, 2, 3)]) * 100
        bv = np.mean([b[(f, seed)]['macro_f1'] for f in (1, 2, 3)]) * 100
        per_seed.append(dict(seed=seed, baseline_f1_percent=float(av),
                             candidate_f1_percent=float(bv), delta_pp=float(bv - av)))
    delta = [r['delta_pp'] for r in per_seed]
    avs = [r['baseline_f1_percent'] for r in per_seed]
    bvs = [r['candidate_f1_percent'] for r in per_seed]
    costs = {}
    for name, group in [(baseline, a), (candidate, b)]:
        if all('training' in r for r in group.values()):
            costs[name] = dict(
                mean_train_val_seconds=float(np.mean([r['training']['total_train_val_seconds'] for r in group.values()])),
                peak_vram_bytes=max(r['training']['peak_vram_bytes'] for r in group.values()),
                mean_node_eligible_batch_fraction=np.mean([r['training']['mean_node_eligible_batch_fraction'] for r in group.values()], axis=0).tolist())
    return dict(baseline=baseline, candidate=candidate, stage=stage, seeds=seeds,
                training_costs=costs,
                baseline_mean=float(np.mean(avs)), candidate_mean=float(np.mean(bvs)),
                baseline_sd=float(np.std(avs, ddof=1)) if len(seeds) > 1 else None,
                candidate_sd=float(np.std(bvs, ddof=1)) if len(seeds) > 1 else None,
                mean_delta_pp=float(np.mean(delta)),
                delta_seed_sd=float(np.std(delta, ddof=1)) if len(seeds) > 1 else None,
                per_seed=per_seed, paired_runs=deltas,
                per_au_delta_pp=(100 * np.mean([np.array(b[k]['per_au_f1']) - np.array(a[k]['per_au_f1']) for k in a], axis=0)).tolist(),
                au_names=a[next(iter(a))]['au_names'],
                status='complete planned seeds' if seeds == [0, 1, 2] else 'preliminary: planned seeds 0,1,2 incomplete',
                interpretation='Observed paired differences only; no frame-independent significance claim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--baseline', default='baseline')
    p.add_argument('--candidate', default='full')
    p.add_argument('--stage', type=int, choices=[1, 3], default=1)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = summarize(a.root, a.baseline, a.candidate, a.stage)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
