"""Evaluate final ABC+mesh using locked maps from a completed CPMS regression.

Other methods retain frozen traces/masks/maps. Original verified photos, not
cached JPEG previews, are used for the added refinement. All common supports
and paired statistics are recomputed. This is exposed development data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_cpms import save, summarize, paired_comparison
from scoredewarp.core.pipeline import Config, refine_dewarp
from scoredewarp.core.transforms import BackwardMap
from scoredewarp.core.structures import draw_structures
from scoredewarp.eval.geometry import map_points
from scoredewarp.eval.real_score import trace_metrics, sampled_map_folds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--download-root', type=Path, default=Path('data/external/cpms'))
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        ap.error('Use an empty output directory')
    if not (args.experiment/'evaluation.json').exists():
        ap.error('Wait for the source evaluation to complete')
    protocol = json.loads((args.experiment/'protocol.json').read_text())
    if 'traditional' not in protocol['methods'] or 'document' not in protocol['methods']:
        ap.error('Source evaluation must include traditional and document')
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    manifest = json.loads((args.download_root/'download_manifest.json').read_text())
    source = Path(manifest['source'])
    methods = protocol['methods']
    updated = {**protocol, 'cached_baseline_experiment': str(args.experiment.resolve()),
               'cached_protocol_sha256': hashlib.sha256((args.experiment/'protocol.json').read_bytes()).hexdigest(),
               'document': 'original ABC, then bounded staff mesh only when coarse and final evidence are staff',
               'final_source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [Path(__file__), ROOT/'scoredewarp/core/pipeline.py', ROOT/'scoredewarp/core/line_mesh.py',
                    ROOT/'scoredewarp/core/transforms.py']}}
    save(args.out/'protocol.json', updated)
    snapshot = args.out/'source_snapshot'
    snapshot.mkdir()
    for p in updated['final_source_sha256']:
        (snapshot/p.replace('/', '__')).write_bytes((ROOT/p).read_bytes())
    records = []
    start_all = time.perf_counter()
    for index, info in enumerate(protocol['photos']):
        sample = info['sample']
        before, after = args.experiment/sample, args.out/sample
        after.mkdir()
        for path in before.iterdir():
            if path.is_file() and not path.name.startswith('document') and path.name != 'metrics.json':
                os.link(path, after/path.name)
        rows = json.loads((before/'metrics.json').read_text())
        baseline = next(r for r in rows if r['method'] == 'traditional')
        photo = source/'data'/(sample+'.jpeg')
        if hashlib.sha256(photo.read_bytes()).hexdigest() != info['photo_sha256']:
            raise ValueError(f'Photo changed: {sample}')
        image = cv2.imread(str(photo))
        width = protocol['working_width']
        image = cv2.resize(image, (width, round(image.shape[0]*width/image.shape[1])), interpolation=cv2.INTER_AREA)
        with np.load(before/'traditional_bmap.npz') as z:
            initial = BackwardMap(z['grid'], int(z['stride']), tuple(z['out_size']))
        coarse_kind = baseline['stages'].get('structures_coarse', {}).get('page_type', baseline['page_type'])
        eligible = coarse_kind == 'staff' and baseline['page_type'] == 'staff'
        t = time.perf_counter()
        result = refine_dewarp(image, initial, Config(use_field=False, use_line_mesh=eligible))
        refinement_time = time.perf_counter() - t
        result.stages = {**baseline['stages'], **result.stages, 'initialization': 'structure_document',
                         'cached_baseline': 'traditional', 'coarse_staff_gate': eligible}
        staves = [np.asarray(s) for s in json.loads((before/'source_traces.json').read_text())]
        coords, masks = [], []
        for staff in staves:
            p, valid = map_points(result.bmap, staff)
            coords.append(p); masks.append(valid)
        content = np.asarray(json.loads((before/'margin_content_points.json').read_text())).reshape(-1, 2)
        row = {'sample': sample, 'content_group': info['content_group'], 'method': 'document', 'status': 'ok',
               'time_s': baseline['time_s'] + refinement_time, 'page_type': result.page_type,
               'detected_staves': len(result.structures.staves), 'output_size': list(result.bmap.out_size),
               'own': trace_metrics(coords, masks), 'grid': sampled_map_folds(result.bmap), 'stages': result.stages,
               'margin_content_points': len(content),
               'margin_content_coverage': float(map_points(result.bmap, content)[1].mean()) if len(content) else None}
        rows = [row if r['method'] == 'document' else r for r in rows]
        mapped, validities = {'input': staves, 'document': coords}, {'document': masks}
        validities['input'] = [np.ones(s.shape[:2], bool) for s in staves]
        for method in methods:
            if method in ('input', 'document'):
                continue
            stored = json.loads((before/(method+'_mapped_traces.json')).read_text())
            mapped[method] = [np.asarray(s, float) for s in stored['points']]
            validities[method] = [np.asarray(s, bool) for s in stored['valid']]
        common = [np.logical_and.reduce([validities[m][i] for m in methods]) for i in range(len(staves))]
        for r in rows:
            r['common'] = trace_metrics(mapped[r['method']], common)
        cv2.imwrite(str(after/'document.jpg'), result.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(after/'document_debug.jpg'), draw_structures(result.image, result.structures))
        np.savez_compressed(after/'document_bmap.npz', grid=result.bmap.grid,
                            stride=result.bmap.stride, out_size=result.bmap.out_size)
        save(after/'document_mapped_traces.json', {'points': coords, 'valid': masks})
        save(after/'metrics.json', rows)
        records.extend(rows)
        save(args.out/'per_sample.json', records)
        print(f'{index+1}/{len(protocol["photos"])} {sample} gate={eligible} '
              f'traditional={baseline["own"]["staff_straight"]:.4f} document={row["own"]["staff_straight"]:.4f}', flush=True)
    save(args.out/'evaluation.json', {'summary': summarize(records, methods),
         'paired_vs_input': paired_comparison(records, methods),
         'paired_vs_traditional': paired_comparison(records, methods, 'traditional'),
         'elapsed_s': time.perf_counter()-start_all, 'reference_is_ground_truth': False,
         'omr_accuracy_measured': False, 'checkpoint_unchanged': True})
    print(args.out, flush=True)


if __name__ == '__main__':
    main()
