"""Calibrate standalone heatmaps on development val, then evaluate fresh pages.

The checkpoint and selected parameters are locked before fresh data generation.
Existing test images/labels are never loaded; manifests are used only to audit
source/seed/content overlap. No network parameter or deployed output is changed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoredewarp.eval.auxiliary import structure_count_metrics
from scoredewarp.learn.dataset import build_targets
from scoredewarp.learn.infer import load
from scoredewarp.learn.structure_postprocess import ridge_scores, structure_masks, threshold_counts
from prepare_structure_suite import KINDS, PROFILES, generate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


@torch.no_grad()
def predict(net, folder, size, device):
    target = build_targets(str(folder), tuple(size))
    if not target['struct_valid']:
        raise ValueError(f'Missing structure labels: {folder}')
    if not all(np.isfinite(value).all() for value in target.values()):
        raise ValueError(f'Nonfinite target: {folder}')
    page = target['page'][0] > 0
    if page.mean() < .01 or (target['fwd'][:, page] < 0).any() or (target['fwd'][:, page] > 1).any():
        raise ValueError(f'Invalid coordinate target: {folder}')
    x = torch.from_numpy(target['image'])[None].to(device)
    output = net(x)
    return output['struct'][0].sigmoid().cpu().numpy(), target


def choose_thresholds(counts, grid):
    # Fixed objective: per-channel micro F1 with 1px localization tolerance.
    # First maximum wins: ties favor the smaller threshold / greater recall.
    scores = np.array([[r['f1'] or 0. for r in structure_count_metrics(c)['tolerant']['channels']]
                       for c in counts])
    indices = scores.argmax(0)
    return [float(grid[i]) for i in indices], counts[indices, np.arange(3)]


def summarize(records, method):
    counts = np.stack([r['counts'][method] for r in records]).sum(0)
    return {'pages': len(records), **structure_count_metrics(counts)}


def bootstrap(records, method, repeats=1000):
    """Paired page bootstrap stratified by kind x profile, fixed seed."""
    rng = np.random.default_rng(731)
    cells = [[i for i, r in enumerate(records) if r['kind'] == k and r['profile'] == p]
             for k in KINDS for p in PROFILES]
    baseline = np.array([r['counts']['baseline'] for r in records])
    candidate = np.array([r['counts'][method] for r in records])
    differences = []
    for _ in range(repeats):
        indices = np.concatenate([rng.choice(cell, len(cell), replace=True) for cell in cells])
        differences.append(structure_count_metrics(candidate[indices].sum(0))['tolerant']['macro_f1'] -
                           structure_count_metrics(baseline[indices].sum(0))['tolerant']['macro_f1'])
    return {'replicates': repeats, 'seed': 731, 'stratification': 'kind x profile',
            'delta_tolerant_macro_f1_ci95': np.quantile(differences, [.025, .975]).tolist()}


def draw_preview(folder, target, probabilities, selected):
    photo = (target['image'].transpose(1, 2, 0)[..., ::-1]*255).astype(np.uint8)
    colors = np.array([[30, 210, 30], [240, 90, 20], [30, 30, 240]], np.uint8)
    tiles = []
    masks = [('input', None), ('GT', target['struct'] > 0),
             ('baseline 0.5', structure_masks(probabilities)),
             ('threshold + ridge', structure_masks(probabilities, selected['thresholds'], selected['radius']))]
    for label, mask in masks:
        tile = photo.copy()
        if mask is not None:
            for channel in range(3):
                tile[mask[channel]] = colors[channel]
        tile = cv2.copyMakeBorder(tile, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        tiles.append(tile)
    if not cv2.imwrite(str(folder), np.concatenate(tiles, axis=1)):
        raise ValueError(f'Cannot save {folder}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, default=Path('outputs/auxiliary_20261115/original_spatial/validated.pt'))
    ap.add_argument('--calibration-data', type=Path, default=Path('data/prepared/structure_suite_20261115_v2/val'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=2026111502)
    ap.add_argument('--per-cell', type=int, default=6)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    args = ap.parse_args()
    if min(args.per_cell, args.workers) < 1:
        ap.error('Counts must be positive')
    if args.calibration_data.name != 'val':
        ap.error('Calibration must use an explicit val directory')
    if args.out.exists() and any(args.out.iterdir()):
        ap.error('Use a fresh empty output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    grid = np.r_[np.arange(5, 100)/100, .995, .999].astype(np.float32)
    source_paths = ['scripts/calibrate_structure.py', 'scripts/prepare_structure_suite.py',
                    'scoredewarp/learn/structure_postprocess.py', 'scoredewarp/eval/auxiliary.py',
                    'scoredewarp/learn/dataset.py', 'scoredewarp/learn/model.py',
                    'scoredewarp/synth/render.py', 'scoredewarp/synth/distort.py', 'scoredewarp/synth/fonts.py']
    protocol = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha(args.checkpoint),
                'calibration_data': str(args.calibration_data), 'grid': grid.tolist(),
                'objective': 'per-channel micro F1 with 1 model-input-pixel Chebyshev tolerance',
                'nms_radius': 2, 'seed': args.seed, 'per_cell': args.per_cell,
                'profiles': PROFILES, 'fresh_pages': args.per_cell*15,
                'existing_test_images_or_labels_accessed': False,
                'geometry_and_kind_changed': False, 'deployment_ready': False,
                'source_sha256': {p: sha(ROOT/p) for p in source_paths}}
    save(args.out/'protocol.json', protocol)
    start = time.time()
    net, size = load(args.checkpoint, args.device)
    net.requires_grad_(False)
    protocol['size'] = list(size)
    counts = {0: np.zeros((len(grid), 3, 5), np.int64), 2: np.zeros((len(grid), 3, 5), np.int64)}
    calibration_rows = []
    paths = sorted(p for p in args.calibration_data.iterdir() if (p/'gt.json').exists())
    if not paths:
        raise ValueError('Empty calibration set')
    for i, path in enumerate(paths):
        probabilities, target = predict(net, path, size, args.device)
        for radius in counts:
            counts[radius] += threshold_counts(ridge_scores(probabilities, radius), target['struct']>0, grid)
        calibration_rows.append({'sample': path.name, 'sha256': {p: sha(path/p) for p in ('photo.jpg', 'gt.json', 'gt_bmap_s2.npy')}})
        if (i+1) % 15 == 0:
            print(f'calibration {i+1}/{len(paths)} ({time.time()-start:.0f}s)', flush=True)
    selected = {}
    for name, radius in (('threshold_only', 0), ('threshold_ridge', 2)):
        thresholds, selected_counts = choose_thresholds(counts[radius], grid)
        selected[name] = {'thresholds': thresholds, 'radius': radius,
                          'calibration_metrics': structure_count_metrics(selected_counts)}
    selected['baseline'] = {'thresholds': [.5]*3, 'radius': 0}
    save(args.out/'calibration.json', {'pages': len(paths), 'inputs': calibration_rows,
                                      'selected': selected, 'threshold_counts': {str(k): v.tolist() for k,v in counts.items()}})
    save(args.out/'selected.json', {'checkpoint_sha256': protocol['checkpoint_sha256'], 'methods': selected,
                                   'deployment_ready': False, 'coordinate_space': 'model input', 'size': list(size)})
    selection_sha = sha(args.out/'selected.json')
    print('LOCKED parameters', {k:v['thresholds'] for k,v in selected.items()}, flush=True)

    # Read only old manifests for cross-experiment de-duplication, not test data.
    prior_paths = sorted((ROOT/'data/prepared').glob('*/manifest.json'))
    prior_rows = [row for path in prior_paths for row in json.loads(path.read_text())['samples']]
    prior_groups = {r['source_group'] for r in prior_rows}
    prior_seeds = {r.get('seed') for r in prior_rows}
    prior_hashes = {filename: {r.get('sha256', {}).get(filename) for r in prior_rows}
                    for filename in ('photo.jpg', 'flat.png')}
    fresh = args.out/'fresh'; fresh.mkdir()
    jobs = []
    for ki, kind in enumerate(KINDS):
        for pi, profile in enumerate(PROFILES):
            for index in range(args.per_cell):
                seed = int(np.random.SeedSequence([args.seed, 77, ki, pi, index]).generate_state(1, dtype=np.uint64)[0])
                if seed in prior_seeds or f'structure-suite-{seed}' in prior_groups:
                    raise ValueError('Fresh seed/group overlaps prior experiment')
                jobs.append((str(fresh), 'val', kind, profile, index, seed, (768, 1024)))
    save(fresh/'protocol.json', {**protocol, 'selected_sha256': selection_sha,
                               'prior_manifests': {str(p.relative_to(ROOT)): sha(p) for p in prior_paths},
                               'jobs': [list(j[1:]) for j in jobs]})
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(generate, jobs):
            rows.append(row)
            if len(rows) % 15 == 0:
                print(f'fresh generation {len(rows)}/{len(jobs)} ({time.time()-start:.0f}s)', flush=True)
    for filename in prior_hashes:
        hashes = [r['sha256'][filename] for r in rows]
        if len(set(hashes)) != len(hashes) or set(hashes) & prior_hashes[filename]:
            raise ValueError(f'Content overlap: {filename}')
    if len({r['source_group'] for r in rows}) != len(rows):
        raise ValueError('Repeated fresh group')
    save(fresh/'manifest.json', {'counts': {'train': 0, 'val': len(rows), 'test': 0}, 'samples': rows})
    save(args.out/'overlap_audit.json', {'prior_manifests': len(prior_paths), 'fresh_pages': len(rows),
                                       'source_group_overlap': 0, 'seed_overlap': 0,
                                       'flat_or_photo_duplicates': 0, 'sampled_projection_filter': True})
    # This is the first and only evaluation of fresh labels. No parameter update.
    records = []
    preview_rows = []
    for i, row in enumerate(rows):
        probabilities, target = predict(net, fresh/row['path'], size, args.device)
        result = {'sample': row['path'], 'kind': row['kind'], 'profile': row['profile'], 'counts': {}}
        for name, config in selected.items():
            thresholds = config['thresholds']
            unique = sorted(set(thresholds))
            c = threshold_counts(ridge_scores(probabilities, config['radius']), target['struct']>0, unique)
            result['counts'][name] = c[[unique.index(t) for t in thresholds], np.arange(3)].tolist()
        records.append(result)
        if row['profile'] == 'blur' and row['path'].endswith('_000'):
            preview = args.out/(row['kind']+'.preview.jpg')
            draw_preview(preview, target, probabilities, selected['threshold_ridge'])
            preview_rows.append(cv2.imread(str(preview)))
        if (i+1) % 15 == 0:
            print(f'fresh evaluation {i+1}/{len(rows)} ({time.time()-start:.0f}s)', flush=True)
    summary = {name: summarize(records, name) for name in selected}
    groups = {key: {name: summarize([r for r in records if r[field] == key], name) for name in selected}
              for field, keys in (('kind', KINDS), ('profile', PROFILES)) for key in keys}
    comparisons = {name: bootstrap(records, name) for name in selected if name != 'baseline'}
    if sha(args.checkpoint) != protocol['checkpoint_sha256'] or sha(args.out/'selected.json') != selection_sha:
        raise ValueError('Frozen model/selection changed during evaluation')
    save(args.out/'evaluation.json', {'summary': summary, 'groups': groups, 'comparisons': comparisons,
                                      'per_sample': records, 'elapsed_s': time.time()-start,
                                      'checkpoint_unchanged': True, 'selection_unchanged': True})
    cv2.imwrite(str(args.out/'preview.jpg'), np.concatenate(preview_rows, axis=0))
    print(json.dumps(summary, indent=2), flush=True)
    print(args.out/'evaluation.json', flush=True)


if __name__ == '__main__':
    main()
