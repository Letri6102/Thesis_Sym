"""Create existing DISFA fold lists from label frame IDs, never silently pad labels."""
import argparse
from pathlib import Path
import numpy as np
from experiment_data import AU_NAMES

PARTS = [
    ['SN002','SN010','SN001','SN026','SN027','SN032','SN030','SN009','SN016'],
    ['SN013','SN018','SN011','SN028','SN012','SN006','SN031','SN021','SN024'],
    ['SN003','SN029','SN023','SN025','SN008','SN005','SN007','SN017','SN004'],
]


def prepare(root, frame_offset):
    root = Path(root)
    records = {}
    for subject in sum(PARTS, []):
        tables = [np.loadtxt(root / 'ActionUnit_Labels' / subject / f'{subject}_au{au}.txt', delimiter=',', ndmin=2) for au in AU_NAMES]
        ids = tables[0][:, 0]
        if not np.equal(ids, ids.astype(int)).all() or len(set(ids)) != len(ids):
            raise ValueError(f'Invalid frame IDs: {subject}')
        if any(not np.array_equal(t[:, 0], ids) for t in tables):
            raise ValueError(f'Unaligned AU labels: {subject}')
        if any(not np.isfinite(t).all() or not np.isin(t[:, 1], range(6)).all() for t in tables):
            raise ValueError(f'Invalid DISFA intensity: {subject}')
        paths = [f'{subject}/{int(i) + frame_offset}.png' for i in ids]
        for path in paths:
            if not (root / 'img' / path).is_file():
                raise FileNotFoundError(f'{path}: check frame offset/cropping; refusing to skip frames')
        records[subject] = paths, np.stack([t[:, 1] >= 2 for t in tables], axis=1).astype(int)
    out = root / 'list'
    out.mkdir(exist_ok=True)
    planned = {}
    # Matches the cloned code: fold1 tests part3, fold2 part2, fold3 part1.
    for fold, test_subjects in zip((1, 2, 3), reversed(PARTS)):
        for part in ('train', 'test'):
            subjects = test_subjects if part == 'test' else [s for s in sum(PARTS, []) if s not in test_subjects]
            paths = sum([records[s][0] for s in subjects], [])
            labels = np.concatenate([records[s][1] for s in subjects])
            planned[f'DISFA_{part}_img_path_fold{fold}.txt'] = '\n'.join(paths) + '\n'
            planned[f'DISFA_{part}_label_fold{fold}.txt'] = '\n'.join(' '.join(map(str, row)) for row in labels) + '\n'
            if part == 'train':
                rates = labels.mean(0)
                if (rates == 0).any():
                    raise ValueError('Missing train AU class')
                weights = 1 / rates
                planned[f'DISFA_train_weight_fold{fold}.txt'] = ' '.join(map(str, weights / weights.sum() * 8)) + '\n'
    for name, content in planned.items():
        path = out / name
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f'Existing list differs; inspect it before generating new lists: {path}')
    for name, content in planned.items():
        (out / name).write_text(content, encoding='utf-8')
    print(f'Validated all 27 subjects and wrote lists to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--frame-offset', type=int, required=True,
                   help='Image number = annotation frame ID + offset; confirm your crop numbering first')
    args = p.parse_args()
    prepare(args.data_root, args.frame_offset)
