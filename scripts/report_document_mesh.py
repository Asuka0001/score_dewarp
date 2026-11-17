"""Report fixed CPMS regression, independent synthetic GT, and full-page previews."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoredewarp.core.transforms import BackwardMap, mapping_quality
from scoredewarp.synth.fonts import cjk_font_paths


NAMES = {'input': '原图', 'traditional': '传统 A+B+C', 'document': '结构网格方案',
         'network_robust': '本轮网络', 'uvdoc': '官方 UVDoc',
         'network_mesh': '本轮网络 + 谱线网格', 'uvdoc_mesh': 'UVDoc + 谱线网格'}


def fmt(v, scale=1):
    return '—' if v is None else f'{v * scale:.3f}'


def contact_sheet(experiment, samples, methods, output):
    tw, th = 470, 720
    sheet = Image.new('RGB', (tw * len(methods), (th + 68) * len(samples)), 'white')
    draw = ImageDraw.Draw(sheet)
    paths = cjk_font_paths()
    font = ImageFont.truetype(paths[0], 20) if paths else ImageFont.load_default()
    for row, sample in enumerate(samples):
        metrics = {r['method']: r for r in json.loads((experiment/sample/'metrics.json').read_text())}
        for col, method in enumerate(methods):
            path = experiment/sample/(method+'.jpg')
            if not path.exists():
                continue
            im = Image.open(path).convert('RGB')
            im.thumbnail((tw-12, th-8))
            x, y = col*tw, row*(th+68)
            draw.text((x+8, y+4), NAMES[method], font=font, fill='black')
            value = metrics[method].get('own', {}).get('staff_straight')
            draw.text((x+8, y+32), f'{sample}  平直度 {fmt(value)}', font=font, fill='black')
            sheet.paste(im, (x+(tw-im.width)//2, y+68+(th-im.height)//2))
    sheet.save(output, quality=92)


def diagnose_maps(experiment, protocol):
    records = []
    for sample in protocol['photos']:
        folder = experiment/sample['sample']
        with Image.open(folder/'input.jpg') as im:
            size = im.size
        for method in protocol['methods'][1:]:
            path = folder/(method+'_bmap.npz')
            if not path.exists():
                continue
            with np.load(path) as z:
                bm = BackwardMap(z['grid'], int(z['stride']), tuple(z['out_size']))
            records.append({'sample': sample['sample'], 'method': method, **mapping_quality(bm, size)})
    result = {'reference': 'all four bilinear Jacobian corners, cells with all vertices inside source photo',
              'proof_of_global_injectivity': False,
              'source_sha256': hashlib.sha256((ROOT/'scoredewarp/core/transforms.py').read_bytes()).hexdigest(),
              'records': records}
    (experiment/'mapping_quality.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpms', type=Path, required=True)
    ap.add_argument('--synthetic', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    protocol = json.loads((args.cpms/'protocol.json').read_text())
    evaluation = json.loads((args.cpms/'evaluation.json').read_text())
    rows = json.loads((args.cpms/'per_sample.json').read_text())
    synthetic = json.loads((args.synthetic/'summary.json').read_text())
    diagnostic_path = args.cpms/'mapping_quality.json'
    diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
    current_hash = hashlib.sha256((ROOT/'scoredewarp/core/transforms.py').read_bytes()).hexdigest()
    if diagnostic is None or diagnostic.get('source_sha256') != current_hash:
        diagnostic = diagnose_maps(args.cpms, protocol)
    history = json.loads((ROOT/'outputs/diverse_training_20261115/model/history.json').read_text())[-1]
    contribution = 5 * history['train_parts']['cons'] / history['train_loss'] * 100
    summary = evaluation['summary']
    classical = summary['traditional']['common']['staff_straight']
    improved = summary['document']['common']['staff_straight']
    paired = evaluation['paired_vs_traditional']['document']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# 文档矫正方法复核与谱线网格改造（2026-11-17）', '',
        f'本轮 CPMS 开发回归：结构网格方案的平直度中位数从传统 {classical:.3f} 降至 {improved:.3f}，'
        f'下降约 {(1-improved/classical)*100:.1f}%，逐页 {paired["improved_pages"]}/{paired["paired_pages"]} 胜传统。'
        '这是固定检测谱线上的诊断收益，不是未见真实数据的识谱准确率结论。', '',
        '运行入口已接入 CLI，但保留传统方案为无 checkpoint 时的默认：新残差虽然不裁切矩形画布，'
        '组合上游含 NaN 的映射时，有限域边缘仍可能损失有效像素；边缘墨迹代理指标也有退化。'
        '不以单项平直度收益自动替换默认。', '',
        '## 原方案的问题', '',
        '1. 当前网络在照片像素上自由预测二维正向 UV，再用页面掩码选点、Delaunay 散点插值反演。'
        '其结构约束只在训练时软惩罚，推理时没有利用当前照片的谱线证据；正向场中的局部异常会进入反演。',
        f'2. 实际第 120 轮：cons={history["train_parts"]["cons"]:.8f}，乘权重 5 后仅占总损失约 **{contribution:.3f}%**。'
        '这是损失数值贡献，不等同于梯度贡献。选模为 EPE + 边界 EPE + 折叠率，没有直接选择谱线平直度。'
        '因此“加了结构损失”不能推断模型已充分学会谱线约束。',
        '3. 正 Jacobian 仍允许不合理的拉伸/剪切；仅在左上角检查一次也可能漏掉另一个三角形的翻折。'
        '白页背景上的平均坐标误差不能替代音符与谱线的局部几何质量。',
        '4. 水平谱线能强约束纵向展开，但不能单独确定横向真实长度。'
        '不能为了把线拉直而任意调整横向尺度；仍需相机/曲面、竖线或其他形状证据。', '',
        '## 参考成熟方法与实际实现', '',
        '- [RDGR / Grid Regularization (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Revisiting_Document_Image_Dewarping_by_Grid_Regularization_CVPR_2022_paper.pdf)：'
        '借鉴显式行几何与弯曲能量共同求解网格；本项目将文本行先验换成五线谱整线水平、同谱表等距约束。不是论文复现。',
        '- [UVDoc 官方实现](https://github.com/tanguymagne/UVDoc)：'
        '接入原作者 2D/3D 双任务预训练模型，使用直接 backward 控制网格；'
        '固定版本 `4c9b82b537057aff2526e6dd118a847cdd072e82`，Git blob 与 SHA-256 校验；MIT 许可随下载保存。没有在本项目重新训练它。',
        '- [DocTr 官方推理](https://github.com/fh2019ustc/DocTr/blob/main/inference.py)、'
        '[DewarpNet 官方实现](https://github.com/cvlab-stonybrook/DewarpNet)：'
        '直接 backward map 与从原图采样是成熟路径。本次接入 UVDoc 进行实测，没有声称运行全部这些模型。', '',
        '结构方案：保持 A/B 在原图相机坐标系的全局拟合，并保留 C 的横向尺度、竖线与字符约束，'
        '之后加入受约束的谱线网格。仅粗校正和最终结果均检测为 staff 时启用；简谱/混排或证据不确定时保留原 ABC。', '',
        '网格拟合整条线的绝对目标，使用二阶弯曲正则与 Huber 重加权；'
        '纵向局部尺度限制为 0.5–2，纵向剪切上限 0.5，必要时缩小位移。'
        'X 保持不变，上下边界固定，按单调列反演，避免再次使用散点反演。'
        '不改善结构或损伤其他已检出谱表的解被拒绝。上述保证仅针对新残差，不能修复上游已发生的翻折。', '',
        'CLI 另检查学习方案最终映射的四个 Jacobian 角；发现局部无效方向时回退结构方案，并在 JSON 明确记录。'
        '`--network-only` 保留未经回退的研究基线。下表评测原始各分支，不启用 CLI 回退，以免掩盖失败。', '',
        '## CPMS 60 页开发回归', '',
        '这批 60 张照片 / 30 个内容组已用于多次开发判断，不是未见测试集。'
        '输入检测谱线固定后映射到各输出；平直度为线距倍数，越低越好。'
        '检测器与优化共用证据，因此该指标有偏好，不能代替人工 GT、音符保留率或 OMR。'
        'common 是本轮全部方法有效点交集，不能与旧报告的绝对值直接横比。', '',
        '| 方法 | common 平直度 | P90 | own 谱线点覆盖率% | 边缘墨迹代理 <95% 页数 | 时间中位数 s |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for method in protocol['methods']:
        s = summary[method]
        lines.append(f'| {NAMES[method]} | {fmt(s["common"]["staff_straight"])} | '
                     f'{fmt(s["common"]["straightness_p90"])} | {fmt(s["own"]["point_coverage"],100)} | '
                     f'{s["margin_content"]["pages_below_95pct"]} | {fmt(s["time_s_median"])} |')
    lines += ['', '逐页配对传统方案（共同有效点，按内容组 bootstrap）：', '',
              '| 方法 | 胜传统页数 | 组均值差 | 95% 区间 |', '|---|---:|---:|---:|']
    for method, s in evaluation['paired_vs_traditional'].items():
        lo, hi = s['group_bootstrap_mean_delta_ci95']
        lines.append(f'| {NAMES[method]} | {s["improved_pages"]}/{s["paired_pages"]} | '
                     f'{fmt(s["group_mean_delta"])} | [{fmt(lo)}, {fmt(hi)}] |')
    lines += ['', '最终映射诊断（所有四角落在原照片内的单元；不使用模型页掩码过滤坏区域）：', '',
              '| 方法 | 检出局部翻折的页数 | 翻折单元比例中位数% | 各向异性 P99 的逐页中位数 |',
              '|---|---:|---:|---:|']
    for method in protocol['methods'][1:]:
        r = [v for v in diagnostic['records'] if v['method'] == method]
        v = [v['folded_cell_fraction'] for v in r if v['folded_cell_fraction'] is not None]
        stretch = [v['anisotropy_p99'] for v in r if 'anisotropy_p99' in v]
        lines.append(f'| {NAMES[method]} | {sum(not v["safe_local_orientation"] for v in r)}/{len(r)} | '
                     f'{fmt(float(np.median(v)) if v else None,100)} | {fmt(float(np.median(stretch)) if stretch else None)} |')
    lines += ['', '## 合成几何真值回归', '',
              '固定三谱种验证集，用真实几何标注检查整页 AD、局部畸变和谱线/字符；'
              '不使用检测器自身的线作为真值。coverage 为页内采样点覆盖率，非音符保留率。'
              '传统 B 按内容裁切导致整页覆盖率偏低，这个问题本次未彻底解决。', '',
              '| 谱种 | 方法 | 页数/失败 | AD% | common AD% | common 局部畸变% | 页点覆盖率% | 谱线平直度 | 简谱基线 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in synthetic:
        if r['method'] not in ('traditional', 'document'):
            continue
        lines.append(f'| {r["kind"]} | {NAMES[r["method"]]} | {r["samples"]}/{r["failed"]} | '
                     f'{fmt(r["ad_pct"])} | {fmt(r["common_ad_pct"])} | {fmt(r["common_local_distortion_pct"])} | '
                     f'{fmt(r["coverage"],100)} | {fmt(r["music"]["staff_straight"])} | {fmt(r["music"]["jp_baseline"])} |')
    lines += ['', '不能只因谱线指标下降就宣称整体恢复更准。AD、线距 CV、字符倾角、覆盖率和困难页须一起检查；'
              '完整结构表见合成实验的 `report.md`。', '', '## 固定样本预览', '',
              '[查看同位置局部放大对照](comparisons_20261117/README.md) · '
              '[打开全部 60 页、7 个方案的并排浏览器](comparisons_20261117/index.html)。', '',
              '浏览器可切换任意两个方案、整页/局部、缩放、同步滚动；局部图按线距等比显示。'
              '包括固定内容组抽样、退化页和边缘裁切页。早期的三页缩略拼图已由此取代。', '']
    lookup = {(r['sample'], r['method']): r for r in rows}
    def difference(sample):
        a = lookup[sample, 'document'].get('common', {}).get('staff_straight')
        b = lookup[sample, 'traditional'].get('common', {}).get('staff_straight')
        return a-b if a is not None and b is not None else -np.inf
    worst = sorted([p['sample'] for p in protocol['photos']], key=difference, reverse=True)[:3]
    lines += ['结构方案相对传统方案差值最大的三页（含退化页）：', '',
              *[f'- [{s}](comparisons_20261117/index.html#{s})：平直度差 {difference(s):+.4f}。' for s in worst], '',
              '## 复现', '', '```bash',
              '.venv/bin/python scripts/download_uvdoc.py',
              '.venv/bin/python scripts/dewarp.py photo.jpg --method document --out outputs/document_demo',
              '.venv/bin/python scripts/dewarp.py photo.jpg --method uvdoc --out outputs/uvdoc_demo',
              '.venv/bin/python scripts/dewarp.py photo.jpg --method traditional --out outputs/abc_demo',
              '.venv/bin/python scripts/evaluate_cpms.py --out outputs/document_cpms_repeat --document \\',
              '  --uvdoc-root data/external/uvdoc --variant network_mesh=network_robust:M --variant uvdoc_mesh=uvdoc:M',
              '.venv/bin/python scripts/evaluate_materials.py --data data/prepared/structure_suite_20261115_v2/val \\',
              '  --checkpoint outputs/diverse_training_20261115/model/best.pt --document --width 1240 \\',
              '  --out outputs/document_synth_repeat', '```', '',
              f'本次完整 CPMS 产物：`{args.cpms}`；合成真值产物：`{args.synthetic}`。', '',
              '## 限制与后续', '',
              '- 低维网格与显式线约束比自由坐标回归更符合该任务，但通用预训练模型也不保证适配乐谱；以本次实测为准。',
              '- 残差网格不恢复被上游 B 裁掉的标题，也不能修复上游网络已有的横向形变或翻折。'
              'CLI 回退是拒绝不可靠候选，不是证明回退结果完全正确。',
              '- 还需新的真实照片、人工点标注及 OMR 音高/时值错误率，尤其覆盖简谱与混排。'
              '后续若训练自有模型，应优先 backward 控制网格、按谱线尺度归一的结构误差，以及结构/局部畸变参与选模；不要沿原目标盲目增加轮数。',
              '- 未改动旧权重、训练数据或当前完成的训练记录。', '']
    args.out.write_text('\n'.join(lines))
    print(args.out)


if __name__ == '__main__':
    main()
