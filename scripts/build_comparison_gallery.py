"""Build a local, two-method comparison viewer from locked CPMS outputs.

No model inference or additional image enhancement. Crop anchors are selected
in the source photo once, mapped using saved trace coordinates, then displayed
with uniform scaling to 24 pixels per staff gap. No deskew or anisotropic resize.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoredewarp.synth.fonts import cjk_font_paths

NAMES = {'input': '原图', 'traditional': '传统 ABC', 'document': 'ABC + 结构网格',
         'network_robust': '本轮网络原始输出', 'network_mesh': '本轮网络 + 谱线网格',
         'uvdoc': '官方 UVDoc', 'uvdoc_mesh': 'UVDoc + 谱线网格'}


def crop_strip(image, traces, masks, anchors, font):
    width, height, gap = 960, 248, 24
    canvas = Image.new('RGB', (width, height * len(anchors)), 'white')
    draw = ImageDraw.Draw(canvas)
    audit = []
    for row, (staff, column, label) in enumerate(anchors):
        p, valid = traces[staff], masks[staff]
        spacings = np.diff(p[..., 1], axis=1)
        good = valid[:, 1:] & valid[:, :-1] & np.isfinite(spacings) & (spacings > 0)
        spacing = float(np.median(spacings[good])) if good.any() else np.nan
        usable = bool(valid[column, 2] and np.isfinite(p[column, 2]).all() and spacing > 0)
        entry = {'source_staff_index': staff, 'source_column_index': column,
                 'anchor_valid': usable, 'mapped_staff_spacing': spacing if np.isfinite(spacing) else None}
        if usable:
            center = p[column, 2]
            scale = gap / spacing
            matrix = np.array([[scale, 0, width / 2 - scale * center[0]],
                               [0, scale, height / 2 + 12 - scale * center[1]]])
            crop = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            canvas.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (0, row * height))
            entry.update(scale=scale, center_output=center.tolist())
            # Only short ticks at the edge: they indicate equal line spacing
            # without painting over the music being inspected.
            for k in range(-2, 3):
                y = row * height + height / 2 + 12 + k * gap
                draw.line([(0, y), (14, y)], fill=(30, 135, 180), width=2)
        else:
            draw.text((50, row * height + 100), '该源位置没有有效映射；请检查整页是否裁切', font=font, fill=(160, 35, 35))
        draw.rectangle((0, row * height, width, row * height + 32), fill='white')
        draw.text((18, row * height + 2), label, font=font, fill=(30, 30, 30))
        draw.line([(0, (row + 1) * height - 1), (width, (row + 1) * height - 1)], fill=(205, 210, 215))
        audit.append(entry)
    return canvas, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', type=Path, default=Path('outputs/document_final_cpms_20261117'))
    ap.add_argument('--out', type=Path, default=Path('reports/comparisons_20261117'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'crops').mkdir(exist_ok=True)
    (args.out/'sheets').mkdir(exist_ok=True)
    protocol = json.loads((args.experiment/'protocol.json').read_text())
    rows = json.loads((args.experiment/'per_sample.json').read_text())
    qualities = json.loads((args.experiment/'mapping_quality.json').read_text())['records']
    quality = {(r['sample'], r['method']): r for r in qualities}
    lookup = {(r['sample'], r['method']): r for r in rows}
    font_paths = cjk_font_paths()
    font = ImageFont.truetype(font_paths[0], 22) if font_paths else ImageFont.load_default()
    entries = []
    for info in protocol['photos']:
        sample = info['sample']
        source = args.experiment/sample
        reference = [np.asarray(s, float) for s in json.loads((source/'source_traces.json').read_text())]
        anchors = []
        if reference:
            order = sorted(range(len(reference)), key=lambda i: np.nanmedian(reference[i][..., 1]))
            for i, fraction, label in [(order[0], .22, '上部谱表 · 左段'),
                                       (order[len(order)//2], .5, '中部谱表 · 中段'),
                                       (order[-1], .78, '下部谱表 · 右段')]:
                anchors.append((i, round((len(reference[i])-1)*fraction), label))
        entry = {'sample': sample, 'content_group': info['content_group'], 'methods': {}}
        for method in protocol['methods']:
            row = lookup[sample, method]
            if method == 'input':
                traces = reference
                masks = [np.ones(s.shape[:2], bool) for s in reference]
            else:
                saved = json.loads((source/(method+'_mapped_traces.json')).read_text())
                traces = [np.asarray(s, float) for s in saved['points']]
                masks = [np.asarray(s, bool) for s in saved['valid']]
            image = cv2.imread(str(source/(method+'.jpg')))
            if image is None:
                raise ValueError(f'Missing image: {source}/{method}.jpg')
            strip, crop_audit = crop_strip(image, traces, masks, anchors, font)
            crop_path = args.out/'crops'/f'{sample}_{method}.jpg'
            strip.save(crop_path, quality=94, subsampling=0)
            own = row.get('own', {})
            entry['methods'][method] = {
                'full': os.path.relpath(source/(method+'.jpg'), args.out),
                'crop': os.path.relpath(crop_path, args.out), 'size': [image.shape[1], image.shape[0]],
                'straight': own.get('staff_straight'), 'coverage': own.get('point_coverage'),
                'margin': row.get('margin_content_coverage'),
                'fold': quality.get((sample, method), {}).get('folded_cell_fraction'),
                'crop_anchors': crop_audit}
        entry['delta'] = (lookup[sample, 'document']['common']['staff_straight'] -
                          lookup[sample, 'traditional']['common']['staff_straight'])
        entries.append(entry)
        print(sample, flush=True)

    # Deterministic representatives from different musical content groups,
    # plus explicitly named failure and border cases, not only successful pages.
    groups = {}
    for e in entries:
        groups.setdefault(e['content_group'], e['sample'])
    candidates = sorted(groups.values())
    selected = [candidates[i] for i in np.linspace(0, len(candidates)-1, 8).astype(int)]
    for sample in ('IMG_1709', 'IMG_1650', 'IMG_1615', 'IMG_1643'):
        if sample not in selected:
            selected.append(sample)
    methods = ['traditional', 'document', 'network_mesh']
    for sample in selected:
        sheet = Image.new('RGB', (960*3, 800), 'white')
        draw = ImageDraw.Draw(sheet)
        for col, method in enumerate(methods):
            crop = Image.open(args.out/'crops'/f'{sample}_{method}.jpg')
            sheet.paste(crop, (col*960, 56))
            draw.text((col*960+16, 12), f'{sample} · {NAMES[method]}', font=font, fill='black')
        sheet.save(args.out/'sheets'/f'{sample}.jpg', quality=94, subsampling=0)
    data = {'methods': NAMES, 'pages': entries, 'selected': selected,
            'notes': 'Same frozen source anchors; uniform scale to 24px per staff gap; no extra rotation or anisotropic resize.',
            'experiment': str(args.experiment),
            'protocol_sha256': hashlib.sha256((args.experiment/'protocol.json').read_bytes()).hexdigest()}
    (args.out/'manifest.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    template = (ROOT/'scripts/comparison_viewer.html').read_text()
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    (args.out/'index.html').write_text(template.replace('__COMPARISON_DATA__', payload))
    lines = ['# 乐谱校正对照图：先看局部，再看整页', '',
             '[打开全部 60 页交互对照](index.html)（在浏览器打开本地 HTML；任意两个方案切换、缩放和并排滚动）', '',
             '建议先比较「传统 ABC / ABC + 结构网格」，再比较「本轮网络原始输出 / 本轮网络 + 谱线网格」。'
             '整页检查标题、页码、谱行有无丢失；局部检查线条起伏、音符挤压、符干倾斜。', '',
             '局部图使用同一张原图中的固定位置，上部左段、中部中段、下部右段各一处。'
             '按线距等比放大到 24px，没有额外纠斜或横纵单独拉伸。左侧蓝色短刻度表示水平等距参考。'
             '这些局部只展示选中区域，内容保留必须同时检查整页；缺失映射明确标红。', '',
             '以下样本包含按内容组均匀抽样、已知退化页和边缘裁切页。每张图从左至右：传统 ABC、ABC + 结构网格、本轮网络 + 谱线网格。'
             '网络分支展示原始实验结果，未启用 CLI 的翻折回退；线条变直不代表映射已经可靠。', '']
    for sample in selected:
        entry = next(e for e in entries if e['sample'] == sample)
        reason = '固定内容组抽样'
        if sample == 'IMG_1709': reason = '结构网格退化页'
        elif sample == 'IMG_1650': reason = '结构方案几乎无变化'
        elif sample == 'IMG_1615': reason = '标题和首行谱表裁切检查'
        elif sample == 'IMG_1643': reason = '边缘有效像素检查'
        lines += [f'## {sample} · {reason}', '',
                  f'[整页与更多方案](index.html#{sample}) · [打开原尺寸局部对照](sheets/{sample}.jpg)', '',
                  f'![{sample} 同位置局部对照](sheets/{sample}.jpg)', '']
    (args.out/'README.md').write_text('\n'.join(lines))
    print(f'{len(entries)} pages, {len(entries)*len(NAMES)} crop strips, {len(selected)} comparison sheets')


if __name__ == '__main__':
    main()
