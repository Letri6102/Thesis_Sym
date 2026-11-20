import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

from model.contrastive import NodeAUSupConLoss, MultiLabelSupConLoss, CrossModalAUExprLoss, EmbeddingQueue
from experiment_data import make_manifest, load_manifest
from train_contrastive import train, evaluate, metrics
from utils import WeightedAsymmetricLoss

torch.set_num_threads(1)


def test_node_loss_prefers_separated_active_inactive():
    y = torch.tensor([[1.], [1.], [0.], [0.]])
    good = torch.tensor([[[1., 0.]], [[1., 0.]], [[-1., 0.]], [[-1., 0.]]], requires_grad=True)
    bad = torch.tensor([[[1., 0.]], [[-1., 0.]], [[1., 0.]], [[-1., 0.]]])
    loss = NodeAUSupConLoss()
    assert loss(good, y) < loss(bad, y)
    loss(good, y).backward()
    assert good.grad is not None and torch.isfinite(good.grad).all()


@pytest.mark.parametrize('labels', [torch.zeros(4, 8), torch.ones(4, 8), torch.eye(8)[:4]])
def test_node_no_valid_pairs_is_differentiable_zero(labels):
    z = torch.randn(4, 8, 5, requires_grad=True)
    loss = NodeAUSupConLoss()(z, labels)
    assert loss.item() == 0
    loss.backward()
    assert z.grad is not None and torch.count_nonzero(z.grad) == 0


def test_global_singleton_and_no_positive_finite():
    for b in (1, 4):
        z = torch.randn(b, 4, requires_grad=True)
        loss = MultiLabelSupConLoss()(F.normalize(z, dim=-1), torch.eye(4)[:b])
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(z.grad).all()


def test_cross_both_directions_have_gradients_and_ignore_neutral():
    a, e = torch.randn(4, 8, 5, requires_grad=True), torch.randn(4, 7, 5, requires_grad=True)
    matrix = torch.rand(8, 7) + .1
    y = torch.zeros(4, 8)
    y[:2, :3] = 1
    loss = CrossModalAUExprLoss(bidirectional=True)(F.normalize(a, dim=-1), F.normalize(e, dim=-1), y, matrix)
    loss.backward()
    assert a.grad[:2].abs().sum() > 0 and e.grad[:2].abs().sum() > 0
    assert a.grad[2:].abs().sum() == 0 and e.grad[2:].abs().sum() == 0
    zero = CrossModalAUExprLoss()(a, e, torch.zeros_like(y), matrix)
    assert zero.item() == 0 and zero.requires_grad


def test_queue_rollover():
    q = EmbeddingQueue(2, 1, 3)
    q.enqueue(torch.arange(10.).reshape(5, 2), torch.arange(5.).reshape(5, 1))
    feats, labels = q.get()
    assert set(labels.flatten().tolist()) == {2., 3., 4.}
    assert not feats.requires_grad
    q.enqueue(torch.tensor([[10., 11.]]), torch.tensor([[5.]]))
    assert set(q.get()[1].flatten().tolist()) == {3., 4., 5.}


def test_wal_does_not_enable_grad_inside_no_grad():
    with torch.no_grad():
        WeightedAsymmetricLoss()(torch.full((2, 8), .3), torch.zeros(2, 8))
        assert not torch.is_grad_enabled()


def synthetic_data(tmp_path):
    root = tmp_path / 'data'
    lists = root / 'list'
    lists.mkdir(parents=True)
    for part, subjects in [('train', ['SN001', 'SN002', 'SN003']), ('test', ['SN004'])]:
        paths, labels = [], []
        for subject in subjects:
            dest = root / 'img' / subject
            dest.mkdir(parents=True)
            for j in range(4):
                Image.new('RGB', (256, 256), (j * 60, 90, 110)).save(dest / f'{j}.png')
                paths.append(f'{subject}/{j}.png')
                labels.append([j % 2] * 8)
        (lists / f'DISFA_{part}_img_path_fold1.txt').write_text('\n'.join(paths))
        np.savetxt(lists / f'DISFA_{part}_label_fold1.txt', labels)
    m = make_manifest(root, 1, val_subject_count=1)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(m))
    return root, manifest


def label_lists(tmp_path, train_spec):
    """Chỉ ghi list/nhãn (không ảnh) cho fold 1; test luôn là SN009."""
    root = tmp_path / 'lists_only'
    lists = root / 'list'
    lists.mkdir(parents=True)
    for part, spec in [('train', train_spec), ('test', {'SN009': [[1] * 8, [0] * 8]})]:
        paths, labels = [], []
        for subject, rows in spec.items():
            for j, row in enumerate(rows):
                paths.append(f'{subject}/{j}.png')
                labels.append(row)
        (lists / f'DISFA_{part}_img_path_fold1.txt').write_text('\n'.join(paths))
        np.savetxt(lists / f'DISFA_{part}_label_fold1.txt', labels)
    return root


def test_val_draw_min_positive(tmp_path):
    rich = [[1] * 8, [1] * 8, [0] * 8]
    no_au2 = [[1, 0, 1, 1, 1, 1, 1, 1], [0] * 8]
    spec = {'SN001': rich, 'SN002': rich, 'SN003': no_au2, 'SN004': no_au2, 'SN005': no_au2, 'SN006': no_au2}
    root = label_lists(tmp_path, spec)

    # Mặc định: đúng thuật toán cũ, không thêm trường mới (manifest cũ tái tạo được y hệt).
    plain = make_manifest(root, 1, val_subject_count=1, check_images=False)
    expected = sorted(np.random.default_rng(2026 + 1).choice(sorted(spec), 1, replace=False).tolist())
    assert plain['val_subjects'] == expected and 'val_min_positive' not in plain

    m = make_manifest(root, 1, val_subject_count=1, min_val_positive=2, check_images=False)
    assert m['val_subjects'][0] in ('SN001', 'SN002')
    assert m['val_min_positive'] == 2 and m['val_draws'] >= 1
    path = tmp_path / 'm.json'
    path.write_text(json.dumps(m))
    load_manifest(path, root)

    m['val_min_positive'] = 3
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='val_min_positive'):
        load_manifest(path, root)
    with pytest.raises(ValueError, match='min_val_positive=3'):
        make_manifest(root, 1, val_subject_count=1, min_val_positive=3, check_images=False, max_draws=50)


def test_manifest_leakage_and_tamper(tmp_path):
    root, manifest = synthetic_data(tmp_path)
    m = load_manifest(manifest, root)
    assert not set(m['fit_subjects']) & set(m['val_subjects'])
    m['val_indices'] += [m['fit_indices'][0]]
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='disjoint'):
        load_manifest(manifest, root)


class TinyStage(nn.Module):
    """Fast integration fixture; actual production heads get a separate forward test."""
    def __init__(self, **kwargs):
        super().__init__()
        self.fc = nn.Linear(3, 15 * 8)
        self.au_classifier = nn.Linear(8, 1)
        self.expr_classifier = nn.Linear(8, 1)
        self.proj_global = nn.Linear(8, 128)
        self.proj_au = nn.Linear(8, 128)
        self.proj_expr = nn.Linear(8, 128)

    def forward(self, x, return_proj=False):
        f = self.fc(x.mean((-1, -2))).reshape(-1, 15, 8)
        a, e = f[:, :8], f[:, 8:]
        result = (a, e, self.au_classifier(a).squeeze(-1).sigmoid(), self.expr_classifier(e).squeeze(-1).sigmoid())
        if return_proj:
            return result + (F.normalize(self.proj_global(a.mean(1)), dim=-1),
                             F.normalize(self.proj_au(a), dim=-1), F.normalize(self.proj_expr(e), dim=-1))
        return result


def args_for(root, manifest, output, variant):
    return argparse.Namespace(data_root=str(root), manifest=str(manifest), output=str(output),
                              variant=variant, epochs=2, batch_size=4, temperature=.1, seed=0,
                              arc='resnet18', lr=1e-4, weight_decay=5e-4, device='cpu', workers=0,
                              no_pretrained=True, resume='', stop_after=None)


@pytest.mark.parametrize('variant', ['baseline', 'full', 'current'])
def test_train_resume_eval_reproduces_uninterrupted(tmp_path, variant):
    root, manifest = synthetic_data(tmp_path)
    continuous = args_for(root, manifest, tmp_path / 'continuous', variant)
    resumed = args_for(root, manifest, tmp_path / 'resumed', variant)
    with patch('train_contrastive.MEFARGStage1', TinyStage):
        train(continuous)
        resumed.stop_after = 1
        train(resumed)
        resumed.resume = str(Path(resumed.output) / 'last.pt')
        resumed.stop_after = None
        train(resumed)
        ev = argparse.Namespace(data_root=str(root), manifest=str(manifest),
                                checkpoint=str(Path(resumed.output) / 'best.pt'), output=str(tmp_path / 'eval'),
                                batch_size=4, workers=0, device='cpu', split='test')
        evaluate(ev)
    ca = torch.load(Path(continuous.output) / 'last.pt', weights_only=False)
    cb = torch.load(Path(resumed.output) / 'last.pt', weights_only=False)
    for key in ca['state_dict']:
        torch.testing.assert_close(ca['state_dict'][key], cb['state_dict'][key], rtol=0, atol=0)
    assert ca['initial_detector_hash'] == cb['initial_detector_hash']
    assert ca['best_val_f1'] == cb['best_val_f1']
    assert (tmp_path / 'eval' / 'test_predictions.npz').is_file()


def test_real_model_forward_backward_and_interface():
    from model.SymStage1 import MEFARGStage1
    model = MEFARGStage1(backbone='resnet18', pretrained=False, use_contrastive=True)
    outputs = model(torch.randn(2, 3, 64, 64), return_proj=True)
    va, ve, pa, pe, zg, za, ze = outputs
    assert va.shape == (2, 8, 256) and ve.shape == (2, 7, 256)
    assert za.shape == (2, 8, 128) and zg.shape == (2, 128)
    assert ((pa >= 0) & (pa <= 1)).all()
    y = torch.ones(2, 8)
    loss = CrossModalAUExprLoss(bidirectional=True)(za, ze, y, torch.ones(8, 7))
    loss.backward()
    assert model.au_head.extractors[0].conv1.weight.grad.abs().sum() > 0
    assert model.expr_head.extractors[0].conv1.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        assert len(model(torch.randn(1, 3, 64, 64))) == 4


def test_gcn_batch_matches_two_single_calls():
    from train_Sym_Stage_2 import LogicGCN
    from train_Sym_Stage_3 import gcn_forward_double
    g = LogicGCN(8, 6, 4)
    a = torch.tensor([[0., 1., 1.], [1., 0., 0.], [1., 0., 0.]])
    x, y = torch.randn(3, 8), torch.randn(3, 8)
    u, v = gcn_forward_double(g, x, y, a)
    torch.testing.assert_close(u, g(x, a))
    torch.testing.assert_close(v, g(y, a))


def test_metric_macro_not_accuracy():
    result = metrics(np.array([[1, 0], [0, 1]]), np.array([[.9, .7], [.2, .8]]))
    assert result['macro_f1'] == pytest.approx((1 + 2/3) / 2)


def test_summary_is_paired_and_requires_three_folds(tmp_path):
    from summarize_experiments import summarize
    shared = ('seed', 'fold', 'arc', 'epochs', 'batch_size', 'workers', 'lr', 'weight_decay',
              'temperature', 'gamma', 'proj_dim', 'precision', 'threshold', 'expr_loss',
              'pretrained', 'environment_versions', 'manifest_hash', 'source_hashes')
    for variant in ['baseline', 'full']:
        for fold in (1, 2, 3):
            folder = tmp_path / variant / f'fold{fold}'
            folder.mkdir(parents=True)
            value = .5 if variant == 'baseline' else .51
            r = dict(variant=variant, stage=1, seed=0, fold=fold, macro_f1=value,
                     per_au_f1=[value]*8, au_names=[1,2,4,6,9,12,25,26],
                     config={key: 0 for key in shared}, initial_detector_hash='same')
            (folder / 'test_metrics.json').write_text(json.dumps(r))
    r = summarize(tmp_path, 'baseline', 'full')
    assert r['mean_delta_pp'] == pytest.approx(1.)
    assert r['delta_seed_sd'] is None
    path = tmp_path / 'full' / 'fold1' / 'test_metrics.json'
    data = json.loads(path.read_text())
    data['config']['lr'] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Unpaired config'):
        summarize(tmp_path, 'baseline', 'full')


def test_frozen_stage3_logic_backpropagates_to_probabilities():
    from train_Sym_Stage_2 import LogicGCN, LogicOperatorEmbeddings, LogicRuleBasePySAT
    from train_Sym_Stage_3 import compute_logic_loss_for_batch
    g = LogicGCN(8, 4, 4)
    for p in g.parameters():
        p.data.fill_(.1)
        p.requires_grad_(False)
    ops = LogicOperatorEmbeddings(4)
    for p in ops.parameters():
        p.requires_grad_(False)
    pa = torch.full((2, 8), .3, requires_grad=True)
    pe = torch.full((2, 7), .3, requires_grad=True)
    loss, raw = compute_logic_loss_for_batch(torch.ones(2, 8, 4), pa, pe,
        torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0.], [0.] * 8]),
        g, ops, torch.ones(8, 4), torch.ones(7, 4), 4, LogicRuleBasePySAT(8, 7))
    assert raw > 0
    loss.backward()
    assert pa.grad.abs().sum() > 0 and pe.grad.abs().sum() > 0
    assert all(p.grad is None for p in g.parameters())
