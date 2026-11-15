"""Inspect cached rectified PNGs; optionally compare a saved detector baseline.

Example:
  python scripts/evaluate_structure_detector.py --images outputs/structure_validation_20261115 \
    --before outputs/structure_validation_20261115/before.json --out outputs/detector_check
PNG basenames must be unique sample IDs. This measures page-type agreement,
not staff-instance precision: use annotated data with evaluate_materials.py for that.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.core.pipeline import _detect
from scoredewarp.core.structures import draw_structures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--before', type=Path)
    ap.add_argument('--expected-kind', choices=('staff','jianpu','mixed'), default='staff')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    paths = sorted(args.images.glob('*.png'))
    if not paths:
        ap.error('No cached PNGs found')
    if args.out.exists() and any(args.out.iterdir()):
        ap.error('Use a fresh empty output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    before = {}
    if args.before:
        before = {r['sample']:r for r in json.loads(args.before.read_text())}
        if set(before) != {p.stem for p in paths}:
            ap.error('Baseline and current image sets must match exactly')
    rows = []
    for i,p in enumerate(paths):
        image = cv2.imread(str(p))
        if image is None:
            raise ValueError(f'Unreadable image: {p}')
        _,_,structures = _detect(image)
        row = {'sample':p.stem, **structures.summary(), 'thickness':structures.thickness,
               'spacing':structures.spacing_est}
        if before:
            row['before'] = before[p.stem]
        rows.append(row)
        (args.out/'per_sample.json').write_text(json.dumps(rows, indent=2))
        if i < 12 or structures.page_type != args.expected_kind:
            debug = draw_structures(image,structures)
            cv2.imwrite(str(args.out/(p.stem+'_debug.jpg')),
                        cv2.resize(debug,(800,round(800*image.shape[0]/image.shape[1]))))
        if (i+1)%20 == 0:
            print(i+1,len(paths),flush=True)
    after_counts = dict(Counter(r['page_type'] for r in rows))
    lines = ['# 缓存校正图的结构检测验证', '',
             f'样本数：{len(rows)}；预期谱种：{args.expected_kind}。同一批固定图像比较检测器，不改变几何映射。',
             '这是谱种输出检查；没有谱线真值时不能据此宣称谱表或关键点准确率。', '',
             '| 版本 | 谱种输出分布 | 谱种一致率 |', '|---|---|---:|']
    counts = [('after',after_counts)]
    if before:
        counts.insert(0,('before',dict(Counter(r['page_type'] for r in before.values()))))
    for name,dist in counts:
        lines.append(f"| {name} | {json.dumps(dist,ensure_ascii=False)} | {dist.get(args.expected_kind,0)/len(rows):.2%} |")
    if before:
        improved = [r['sample'] for r in rows if before[r['sample']]['page_type'] != args.expected_kind and r['page_type'] == args.expected_kind]
        regressed = [r['sample'] for r in rows if before[r['sample']]['page_type'] == args.expected_kind and r['page_type'] != args.expected_kind]
        lines += ['',f'改善 {len(improved)} 页，退化 {len(regressed)} 页。',f'退化样本：{regressed}', '']
    remaining = [r['sample'] for r in rows if r['page_type'] != args.expected_kind]
    lines += ['仍未正确输出谱种：', '', *[f'- {s}' for s in remaining]]
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    (args.out/'summary.json').write_text(json.dumps({'counts':dict(counts),'remaining':remaining},indent=2))
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
