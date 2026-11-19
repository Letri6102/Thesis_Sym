"""Subject-disjoint manifests shared by all ablations and downstream stages."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as _T
class strong_train_aug(object):
    """Augmentation chống ghi nhớ danh tính. Chữ ký (img, *_) để thay image_train."""
    def __init__(self, img_size=256, crop_size=224):
        self.t = _T.Compose([
            _T.Resize(img_size),
            _T.RandomResizedCrop(crop_size, scale=(0.70, 1.0), ratio=(0.85, 1.18)),
            _T.RandomHorizontalFlip(0.5),
            _T.ColorJitter(0.4, 0.4, 0.4, 0.08),
            _T.RandomGrayscale(0.2),
            _T.RandomApply([_T.GaussianBlur(5, (0.1, 2.0))], 0.3),
            _T.ToTensor(),
            _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            _T.RandomErasing(0.25, scale=(0.02, 0.15)),
        ])
    def __call__(self, img, *a):
        return self.t(img)


AU_NAMES = [1, 2, 4, 6, 9, 12, 25, 26]
ROOT = Path(__file__).resolve().parent
MATRIX_PATH = ROOT / 'matrixMAE' / 'M_AE_DISFA.npy'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_partition(root, fold, part):
    base = Path(root) / 'list'
    paths_file = base / f'DISFA_{part}_img_path_fold{fold}.txt'
    labels_file = base / f'DISFA_{part}_label_fold{fold}.txt'
    paths = [p.strip().replace('\\', '/') for p in paths_file.read_text().splitlines() if p.strip()]
    labels = np.loadtxt(labels_file, ndmin=2, dtype=np.float32)
    if labels.shape != (len(paths), 8) or not np.isin(labels, [0, 1]).all():
        raise ValueError(f'Expected {len(paths)} x 8 binary labels: {labels_file}')
    if len(paths) != len(set(paths)):
        raise ValueError(f'Duplicate image paths: {paths_file}')
    for p in paths:
        pp = PurePosixPath(p)
        if pp.is_absolute() or '..' in pp.parts or ':' in p or len(pp.parts) != 2:
            raise ValueError(f'Expected relative subject/frame path: {p}')
        if not pp.parts[0].startswith('SN'):
            raise ValueError(f'Unrecognized DISFA subject: {p}')
    return paths, labels, {paths_file.name: digest(paths_file), labels_file.name: digest(labels_file)}


def make_manifest(root, fold, val_subject_count=3, split_seed=2026, check_images=True):
    train, labels, hashes = read_partition(root, fold, 'train')
    test, _, test_hashes = read_partition(root, fold, 'test')
    hashes.update(test_hashes)
    subjects = sorted({p.split('/')[0] for p in train})
    test_subjects = sorted({p.split('/')[0] for p in test})
    if set(subjects) & set(test_subjects):
        raise ValueError('Outer train/test subjects overlap')
    if not 0 < val_subject_count < len(subjects):
        raise ValueError('Invalid number of inner-validation subjects')
    rng = np.random.default_rng(split_seed + fold)
    val_subjects = sorted(rng.choice(subjects, val_subject_count, replace=False).tolist())
    fit = [i for i, p in enumerate(train) if p.split('/')[0] not in val_subjects]
    val = [i for i, p in enumerate(train) if p.split('/')[0] in val_subjects]
    rates = labels[fit].mean(0)
    if np.any(rates == 0):
        raise ValueError(f'No positive fit samples for AU(s) {np.array(AU_NAMES)[rates == 0]}')
    weights = 1.0 / rates
    weights = weights / weights.sum() * 8
    if check_images:
        for p in train + test:
            if not (Path(root) / 'img' / p).is_file():
                raise FileNotFoundError(Path(root) / 'img' / p)
    return dict(schema=1, dataset='DISFA', fold=fold, split_seed=split_seed,
                fit_indices=fit, val_indices=val,
                fit_subjects=sorted(set(subjects) - set(val_subjects)),
                val_subjects=val_subjects, test_subjects=test_subjects,
                train_weights=weights.tolist(), input_hashes=hashes,
                matrix_hash=digest(MATRIX_PATH), au_names=AU_NAMES)


def load_manifest(path, root):
    m = json.loads(Path(path).read_text(encoding='utf-8'))
    if m['schema'] != 1 or m['dataset'] != 'DISFA':
        raise ValueError('Only DISFA manifest v1 is supported')
    for name, expected in m['input_hashes'].items():
        if digest(Path(root) / 'list' / name) != expected:
            raise ValueError(f'Data changed since manifest creation: {name}')
    if digest(MATRIX_PATH) != m['matrix_hash']:
        raise ValueError('M_AE changed since manifest creation')
    train, labels, _ = read_partition(root, m['fold'], 'train')
    fit, val = m['fit_indices'], m['val_indices']
    if sorted(fit + val) != list(range(len(train))):
        raise ValueError('Fit and val must be a disjoint partition of outer train')
    fs = {train[i].split('/')[0] for i in fit}
    vs = {train[i].split('/')[0] for i in val}
    test, _, _ = read_partition(root, m['fold'], 'test')
    ts = {p.split('/')[0] for p in test}
    if fs & vs or fs & ts or vs & ts or fs != set(m['fit_subjects']) or vs != set(m['val_subjects']) or ts != set(m['test_subjects']):
        raise ValueError('Subject leakage or inconsistent manifest')
    rates = labels[fit].mean(0)
    if (rates <= 0).any():
        raise ValueError('Fit has missing positive classes')
    weights = 1 / rates
    weights = weights / weights.sum() * 8
    if not np.allclose(weights, m['train_weights']):
        raise ValueError('Manifest weights must be calculated from fit only')
    return m


class ExperimentFrames(Dataset):
    def __init__(self, root, manifest, split, crop_size=224):
        from utils import image_train, image_test
        if split not in ('fit', 'val', 'test'):
            raise ValueError(split)
        self.root = Path(root)
        self.split = split
        paths, labels, _ = read_partition(root, manifest['fold'], 'test' if split == 'test' else 'train')
        ids = range(len(paths)) if split == 'test' else manifest[f'{split}_indices']
        self.paths = [paths[i] for i in ids]
        self.labels = labels[list(ids)]
        self.transform = strong_train_aug(crop_size=crop_size) if split == 'fit' else image_test(crop_size=crop_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / 'img' / self.paths[i]) as img:
            img = img.convert('RGB')
            x = self.transform(img, 0, 0, 0) if self.split == 'fit' else self.transform(img)
        return x, torch.from_numpy(self.labels[i].copy())


def loader_for(dataset, batch_size, workers, seed, train=False):
    # Sampler and worker/augmentation seeds are independent of model init/RNG.
    g = torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                      num_workers=workers, generator=g, pin_memory=torch.cuda.is_available(),
                      drop_last=False, persistent_workers=False)


def legacy_loaders(conf):
    m = load_manifest(conf.split_manifest, conf.dataset_path)
    if conf.dataset != 'DISFA' or conf.fold != m['fold']:
        raise ValueError('Dataset/fold must match manifest')
    fit = ExperimentFrames(conf.dataset_path, m, 'fit', conf.crop_size)
    val = ExperimentFrames(conf.dataset_path, m, 'val', conf.crop_size)
    return (loader_for(fit, conf.batch_size, conf.num_workers, conf.seed, True),
            loader_for(val, conf.batch_size, conf.num_workers, conf.seed), len(fit), len(val))


def downstream_metadata(conf):
    return dict(epochs=conf.epochs, lr=conf.learning_rate, weight_decay=conf.weight_decay,
                batch_size=conf.batch_size, arc=conf.arc, seed=conf.seed, fold=conf.fold,
                source_hashes={str(p.relative_to(ROOT)).replace('\\', '/'): digest(p)
                               for p in sorted(list(ROOT.glob('*.py')) + list((ROOT / 'model').glob('*.py')))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split-seed', type=int, default=2026)
    parser.add_argument('--val-subjects', type=int, default=3)
    args = parser.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    manifests = [make_manifest(args.data_root, f, args.val_subjects, args.split_seed) for f in (1, 2, 3)]
    universes = [set(m['fit_subjects'] + m['val_subjects'] + m['test_subjects']) for m in manifests]
    tests = [s for m in manifests for s in m['test_subjects']]
    if any(u != universes[0] for u in universes) or len(universes[0]) != 27 or len(tests) != len(set(tests)) or set(tests) != universes[0]:
        raise ValueError('DISFA CV requires the same 27 subjects in all folds, each tested exactly once')
    for fold, manifest in zip((1, 2, 3), manifests):
        out = dest / f'fold{fold}.json'
        content = json.dumps(manifest, indent=2)
        if out.exists() and json.loads(out.read_text()) != manifest:
            raise FileExistsError(f'Refusing to replace a different split: {out}')
        out.write_text(content, encoding='utf-8')
        print(f'{out}: fit={manifest["fit_subjects"]}, val={manifest["val_subjects"]}, test={manifest["test_subjects"]}')
