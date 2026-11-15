"""Compare actual classical/learned outputs with geometry and music metrics.

Example:
  python scripts/evaluate_materials.py --data data/synth --checkpoint MODEL --out outputs/materials
Use --data more than once for multiple datasets. Tune on validation data;
previously inspected test sets are regression checks, not independent evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.core.pipeline import Config, dewarp, refine_dewarp, result_json
from scoredewarp.core.structures import draw_structures
from scoredewarp.eval.geometry import (sample_geometry, geometry_metrics, map_points,
                                       network_points, paired_geometry_metrics, orientation_error_rate)
from scoredewarp.eval.metrics import evaluate
from scoredewarp.eval.structures import staff_detection_metrics
from scoredewarp.learn.infer import load, predict

STRUCTURE_KEYS = ("coverage", "staff_straight", "staff_spacing_cv", "barline_tilt_deg", "jp_baseline",
                  "jp_height_cv", "jp_tilt_deg", "jp_dot_err", "jp_lyric_err", "jp_rowgap_cv")


def measure(folder, net, size, width, save_dir=None, *, networks=None):
    image, meta, points, target, valid = sample_geometry(folder)
    gt = json.loads((folder / "gt.json").read_text())
    networks = networks if networks is not None else {"network_only": (net, size)}
    rows, maps, predictions = [], {}, {}
    for name in ("input", "traditional", *networks):
        row = {"sample": folder.name, "kind": meta["kind"], "method": name,
               "profile": meta.get("profile", "unspecified")}
        start = time.perf_counter()
        try:
            result, raw = None, None
            if name == "input":
                pred, keep = points, np.isfinite(points).all(-1)
                mapper = lambda p: np.asarray(p, float).copy()
                validity = lambda p: np.isfinite(p).all(-1)
            else:
                if name == "traditional":
                    result = dewarp(image, Config(coarse_width=width))
                else:
                    model, model_size = networks[name]
                    bm, _, raw = predict(model, image, model_size, out_size=(width, round(width * 2263 / 1600)))
                    result = refine_dewarp(image, bm, Config(use_field=False))
                row["time_s"] = time.perf_counter() - start
                bm = result.bmap
                # Capture bmap/raw in defaults: maps remain valid after this loop.
                def validity(p, bm=bm, raw=raw):
                    page = None
                    if raw is not None:
                        _, page = network_points(raw, p, (image.shape[1], image.shape[0]),
                                                 meta["size"][1] / meta["size"][0])
                    return map_points(bm, p, page_valid=page)[1]
                mapper = bm.forward_points
                pred = mapper(points.reshape(-1, 2)).reshape(points.shape)
                keep = validity(points)
                row.update(page_type=result.page_type, type_correct=result.page_type == meta["kind"],
                           structures=result.structures.summary())
            # Inference timing excludes scattered forward inversion, metrics
            # and file export, matching the CLI's image-production timing.
            row.setdefault("time_s", time.perf_counter() - start)
            row.update(geometry_metrics(target, pred, valid, keep), status="ok")
            row["orientation_error_rate"] = orientation_error_rate(target, pred, valid & keep)
            music = evaluate(gt, mapper, point_valid=validity)
            row["music"] = music
            if result is not None:
                row["detection"] = staff_detection_metrics(
                    gt, lambda p: np.where(validity(p)[:, None], mapper(p), np.nan), result.structures)
            maps[name] = (mapper, validity)
            predictions[name] = (pred, keep)
            if save_dir is not None and result is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_dir / f"{name}_rectified.png"), result.image)
                cv2.imwrite(str(save_dir / f"{name}_debug.jpg"), draw_structures(result.image, result.structures))
                (save_dir / f"{name}_result.json").write_text(json.dumps(result_json(result), indent=2))
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    # Only compare the two actual correction methods. The original photo is
    # reported separately, so it cannot change their common support.
    names = ("traditional", *networks)
    if all(name in maps for name in names):
        paired = paired_geometry_metrics(target, {name: predictions[name] for name in names}, valid)
        def common_valid(p):
            return np.logical_and.reduce([maps[name][1](p) for name in names])
        for row in rows:
            name = row["method"]
            if name in paired:
                row["common_geometry"] = paired[name]
                row["common_music"] = evaluate(gt, maps[name][0], point_valid=common_valid)
        # Preserve pairwise support as well: a third method with low coverage
        # must not hide errors when comparing the two training ablations.
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                pair = paired_geometry_metrics(target, {n: predictions[n] for n in (left, right)}, valid)
                def pair_valid(p, left=left, right=right):
                    return maps[left][1](p) & maps[right][1](p)
                for row in rows:
                    name = row["method"]
                    if name in pair:
                        row.setdefault("paired", {})[right if name == left else left] = {
                            "geometry": pair[name],
                            "music": evaluate(gt, maps[name][0], point_valid=pair_valid)}
    return rows


def median(rows, key, section=None):
    vals = [(r.get(section, {}) if section else r).get(key) for r in rows]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else None


def report(rows, out):
    lines = ["# 素材评测：几何、共同有效点、乐谱结构与检测质量", "",
             "traditional = A+B+C；其余校正方法为网络反演后的实际输出映射，不加 C。",
             "校正方法均检查画布边界与 2px 往返误差；common 列使用所有校正方法的有效点交集。",
             "结构误差由真值关键点计算，非检测器自评分；DeepScores 无结构真值，相关项留空。",
             "误差与覆盖率列取逐页中位数；谱种正确率取均值。谱表检测按实例一对一匹配汇总。",
             "未定义指标显示 —，不计为零。coverage 为采样网格覆盖率，music 内另有结构关键点覆盖率。", ""]
    summary = []
    methods = list(dict.fromkeys(r["method"] for r in rows))
    corrected = [m for m in methods if m != "input"]
    def fmt(v, mult=1):
        return "—" if v is None else f"{v * mult:.3f}"
    for dataset, kind in sorted({(r['dataset'], r['kind']) for r in rows}):
        lines += [f"## {dataset} / {kind}", "",
                  "| 方法 | 页数/失败 | AD% | 局部畸变% | 覆盖率% | common AD% | common 局部畸变% | common 覆盖率% | 谱种正确率% |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        groups = {}
        for method in methods:
            group = [r for r in rows if (r['dataset'], r['kind'], r['method']) == (dataset, kind, method)]
            groups[method] = group
            entry = {"dataset": dataset, "kind": kind, "method": method, "samples": len(group),
                     "failed": sum(r['status'] != 'ok' for r in group)}
            for key in ("ad_pct", "local_distortion_pct", "coverage"):
                entry[key] = median(group, key)
                entry['common_' + key] = median(group, key, 'common_geometry')
            tc = [r['type_correct'] for r in group if 'type_correct' in r]
            entry['type_accuracy'] = float(np.mean(tc)) if tc else None
            entry['music'] = {key: median(group, key, 'music') for key in STRUCTURE_KEYS}
            entry['common_music'] = {key: median(group, key, 'common_music') for key in STRUCTURE_KEYS}
            entry['detection'] = {key: median(group, key, 'detection') for key in ('staff_precision','staff_recall')}
            entry['detection_counts'] = {key: sum(r.get('detection', {}).get(key, 0) for r in group)
                                         for key in ('staff_matched','staff_pred_count','staff_gt_count')}
            summary.append(entry)
            lines.append(f"| {method} | {len(group)}/{entry['failed']} | " + ' | '.join([
                fmt(entry['ad_pct']), fmt(entry['local_distortion_pct']), fmt(entry['coverage'], 100),
                fmt(entry['common_ad_pct']), fmt(entry['common_local_distortion_pct']),
                fmt(entry['common_coverage'], 100), fmt(entry['type_accuracy'], 100)]) + ' |')
        keys = [key for key in STRUCTURE_KEYS if any(median(g, key, 'music') is not None for g in groups.values())]
        if keys:
            lines += ['', '结构指标（逐页中位数；coverage 是结构关键点覆盖率 0–1，平直度按线距/字高归一，倾角单位 °）：', '',
                      '| 方法 / 有效点 | ' + ' | '.join(keys) + ' |', '|---|' + '---:|' * len(keys)]
            for method, group in groups.items():
                for section in (('music',) if method == 'input' else ('music', 'common_music')):
                    lines.append('| ' + method + (' / common' if section == 'common_music' else ' / own') + ' | ' +
                                 ' | '.join(fmt(median(group, key, section)) for key in keys) + ' |')
            lines += ['', '| 方法 | 匹配/预测/真值谱表数 | 谱表 precision | 谱表 recall |', '|---|---:|---:|---:|']
            for method in corrected:
                g = groups[method]
                counts = [sum(r.get('detection', {}).get(key, 0) for r in g)
                          for key in ('staff_matched', 'staff_pred_count', 'staff_gt_count')]
                matched, predicted, truth = counts
                precision = matched / predicted if predicted else None
                recall = matched / truth if truth else None
                lines.append(f"| {method} | {matched}/{predicted}/{truth} | {fmt(precision)} | {fmt(recall)} |")
    failures = [r for r in rows if r['status'] != 'ok']
    lines += ['', '## 失败与困难样本', '']
    for r in failures:
        lines.append(f"- {r['dataset']}/{r['sample']} / {r['method']}: {r['error']}")
    if not failures:
        lines.append('无运行异常。以下列出各方法 AD 最高的 5 页，低覆盖率也需单独检查：')
    for method in corrected:
        worst = sorted([r for r in rows if r['method'] == method and r.get('ad_pct') is not None], key=lambda r: r['ad_pct'], reverse=True)[:5]
        for r in worst:
            lines.append(f"- {method}: {r['dataset']}/{r['sample']}, AD={r['ad_pct']:.3f}%, coverage={r['coverage']:.1%}")
        low = sorted([r for r in rows if r['method'] == method and r.get('coverage') is not None],
                     key=lambda r: r['coverage'])[:5]
        lines += ['', f'{method} 覆盖率最低的 5 页：', '']
        for r in low:
            lines.append(f"- {r['dataset']}/{r['sample']}: {r['coverage']:.1%}")
        for key in ('staff_straight', 'jp_baseline'):
            difficult = sorted([r for r in rows if r['method'] == method and
                                 r.get('music', {}).get(key) is not None],
                                key=lambda r: r['music'][key], reverse=True)[:3]
            if difficult:
                lines += ['', f'{method} / {key} 最高的 3 页：', '']
                for r in difficult:
                    lines.append(f"- {r['dataset']}/{r['sample']}: {r['music'][key]:.3f}")
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, action='append', required=True)
    ap.add_argument('--checkpoint', type=Path, action='append', required=True,
                    help='repeat to compare models; multi-model names use checkpoint parent directory names')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0, help='per dataset, sorted by sample name')
    ap.add_argument('--save-images', action='store_true')
    ap.add_argument('--width', type=int, default=1600)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        ap.error('Use a new empty output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    names = ['network_only'] if len(args.checkpoint) == 1 else [p.parent.name for p in args.checkpoint]
    if len(set(names)) != len(names) or set(names) & {'input', 'traditional'}:
        ap.error('Checkpoint parent directory names must be distinct and not input/traditional')
    networks = {name: load(path) for name, path in zip(names, args.checkpoint)}
    config = {key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()}
    config['data'] = [str(p.resolve()) for p in args.data]
    config['checkpoint'] = {name: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                            for name, path in zip(names, args.checkpoint)}
    # Include code diff because checks are often run before committing a fix.
    config['git_head'] = subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    diff = subprocess.run(['git','diff'],capture_output=True,text=True).stdout
    (args.out / 'code.diff').write_text(diff)
    (args.out / 'config.json').write_text(json.dumps(config, indent=2))
    rows = []
    for dataset in args.data:
        folders = sorted(p for p in dataset.iterdir() if (p / 'gt.json').exists())
        if args.limit:
            folders = folders[:args.limit]
        if not folders:
            ap.error(f'No samples: {dataset}')
        label = str(dataset)
        for i, folder in enumerate(folders):
            dest = args.out / dataset.parent.name / dataset.name / folder.name if args.save_images else None
            sample_rows = measure(folder, None, None, args.width, dest, networks=networks)
            for row in sample_rows:
                row['dataset'] = label
            rows.extend(sample_rows)
            (args.out / 'per_sample.json').write_text(json.dumps(rows, indent=2))
            print(f'{label}: {i+1}/{len(folders)} {folder.name}', flush=True)
    report(rows, args.out)
    for profile in sorted({r['profile'] for r in rows} - {'unspecified'}):
        dest = args.out / 'profiles' / profile
        dest.mkdir(parents=True)
        report([r for r in rows if r['profile'] == profile], dest)
    print(args.out / 'report.md', flush=True)
    if any(r['status'] != 'ok' for r in rows):
        raise SystemExit('Some method runs failed; see per_sample.json')


if __name__ == '__main__':
    main()
