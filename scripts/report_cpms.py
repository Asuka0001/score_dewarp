"""Build a Chinese CPMS report and a fixed, outcome-independent contact sheet."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def f(value, percent=False):
    return '—' if value is None else f'{value*(100 if percent else 1):.3f}'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();exp=args.experiment
    evaluation=json.loads((exp/'evaluation.json').read_text())
    diagnostic=json.loads((exp/'field_diagnostic.json').read_text())
    protocol=json.loads((exp/'protocol.json').read_text())
    audit=json.loads((exp/'reference_audit.json').read_text())
    records=json.loads((exp/'per_sample.json').read_text())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    summary=evaluation['summary'];methods=protocol['methods']
    groups={r['content_group'] for r in audit}
    names={'input':'原图','traditional':'传统 A+B+C','network_original':'原始全量网络 + 已验证 kind',
           'network_adapted':'三谱种 data_only 微调网络'}
    for method in methods:names.setdefault(method,method)
    lines=['# CPMS 真实拍照五线谱零样本评测（2026-11-15）','',
           '**结论：两版网络在本批真实照片上的泛化不足，出现额外弯曲和局部撕裂；当前不宜默认用于 CPMS 这类输入。传统方案在本轮固定谱线诊断上更好，但仍需人工检查。**','',
           '## 数据与实验边界','',
           f"- 官方来源：[itec-hust/CPMS](https://github.com/itec-hust/CPMS)，固定版本 `{protocol['dataset_revision']}`。下载完整仓库，逐一核对 2,647 个 Git blob 的大小与 SHA-1，另保存 SHA-256。原始文件与压缩包位于被 Git 忽略的 `data/external/cpms/`。",
           f"- 评测全部 **{len(audit)} 张完整照片**，{sum(r['annotated_rows'] for r in audit)} 个标注谱行、{sum(r['annotated_notes'] for r in audit)} 个标注音符。按逐行音高/时值序列得到 **{len(groups)} 个内容组**，每组两张拍摄图，统计不将它们当作完全独立曲谱。",
           '- `semantic/training` 和 `semantic/test` 的裁图与语义文件也已完整下载；本轮不训练、不改阈值，也不把裁图数量当成独立完整照片数。',
           '- 所有输入先等比例缩到宽 1240；原始照片为 3024×4032。网络内部输入 384×512，输出 1240×1754；传统方案使用宽度 1240 的既有 A+B+C。两个网络均不加 C，不改权重。',
           '- JSON 的 x/y 与原始照片坐标不直接对应，仓库未说明各图的缩放/坐标变换。检查叠加图发现不能直接使用原始坐标，也未确认所有照片共享同一个缩放。因此本轮不使用这些坐标计算音符保留率、识谱准确率或几何真值误差。',
           '- CPMS 没有本项目合成数据形式的稠密形变真值；本轮 **不报告 AD、UV EPE 或结构检测 GT F1**。',
           '', '## 固定谱线诊断的方法与局限','',
           '先在原图上运行既有稳健谱表检测器，冻结每个谱表的采样列（每列 5 个点），再通过各方法的实际输出映射变换同一批点。平直度为各线 Y 标准差除以映射后的谱线间距；越低越好。每个谱表至少保留 8 列且不少于源列数的一半才参与平直度统计。',
           '',
           '有效点需通过画布边界、有限值和 ≤2px 往返误差检查，网络还需通过原输入坐标上的预测页面掩码。共同点取三种校正方法交集，并对原图使用同一交集；同时报告各方法自己的覆盖率，避免丢失内容后显得更平直。',
           '',
           f"原图检测器共提供 {sum(r['detected_source_staves'] for r in audit)} 个候选谱表、{sum(r['source_trace_points'] for r in audit)} 个采样点。它可能漏检、误分组，且与传统拟合共享检测器，因此这是**固定伪参考上的诊断，不能当作人工 GT，也不是整页或音符的内容保留率**。未检测到的区域不在覆盖率分母中。输出谱表数也不是一对一实例 precision/recall。",
           '', '## 完整照片结果','',
           '误差与覆盖率取逐页中位数，P90 是较差尾部；所有平直度均为无单位的线距倍数。','',
           '| 方法 | 异常/总页 | own 点覆盖率% | common 平直度 | common 平直度 P90 | common 可测页数 | own 覆盖率<95%页数 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for method,s in summary.items():
        lines.append(f"| {names[method]} | {s['exceptions']}/{s['pages']} | {f(s['own']['point_coverage'],True)} | {f(s['common']['staff_straight'])} | {f(s['common']['straightness_p90'])} | {s['common']['measurable_pages']} | {s['coverage_below_95pct_pages']} |")
    lines += ['', 'own 与 common 的覆盖率含义不同，common 是所有方法相同的有效源点交集；指标缺失不计为零。', '',
              '| 方法 | own 平直度 | common 线距 CV | 谱种输出 staff 比例% | 输出检测谱表数中位数 | 图像生成耗时中位数 s |',
              '|---|---:|---:|---:|---:|---:|']
    for method,s in summary.items():
        lines.append(f"| {names[method]} | {f(s['own']['staff_straight'])} | {f(s['common']['staff_spacing_cv'])} | {f(s['type_staff_rate'],True)} | {f(s['detected_staves_median'])} | {f(s['time_s_median'])} |")
    lines += ['', '耗时为本机 CPU 图像生成耗时，不含指标计算与文件导出；不是跨设备性能基准。谱种输出正确不代表校正图正确。','',
              '## 同一页面共同点上的配对变化','',
              '差值 = 校正后 − 原图，负值为改善。Bootstrap 按内容组重采样 1,000 次，seed=731，组内两张拍摄图先取平均；区间是组均值差的区间。','',
              '| 方法 | 平直度改善/有效页 | 逐页差值中位数 | 内容组均值差 | 组 bootstrap 95%区间 |',
              '|---|---:|---:|---:|---:|']
    for method,r in evaluation['paired_vs_input'].items():
        low,high=r['group_bootstrap_mean_delta_ci95']
        lines.append(f"| {names[method]} | {r['improved_pages']}/{r['paired_pages']} | {f(r['page_delta_median'])} | {f(r['group_mean_delta'])} | [{f(low)}, {f(high)}] |")
    lines += ['', '## 反演前坐标场诊断','',
              '固定权重和输入预处理，额外检查网络直接预测的正向坐标场。下面的折叠率为预测页内四角都有效的采样单元中，Jacobian ≤0 的比例；没有 GT 页面掩码，不代表严格的全局可逆性证明。它可帮助判断问题是否在反演前已经存在。','',
              '| 网络 | 直接正向场谱线平直度中位数 | 预测页内非正 Jacobian 比例% 中位数 |',
              '|---|---:|---:|']
    for method,r in diagnostic['summary'].items():
        lines.append(f"| {names[method]} | {f(r['direct_forward_traces'].get('staff_straight'))} | {f(r['field_folds'].get('predicted_page_nonpositive_fraction'),True)} |")
    lines += ['', '正向场上已出现弯曲和非正 Jacobian，支持问题不只来自坐标反演；反演和插值仍可能进一步放大局部撕裂。这一诊断不能仅凭单个汇总值确定训练失败的具体原因。','',
              '## 典型失败与复查','']
    for method in methods[1:]:
        worst=sorted([r for r in records if r['method']==method and r.get('common',{}).get('staff_straight') is not None],
                     key=lambda r:r['common']['staff_straight'],reverse=True)[:3]
        low=sorted([r for r in records if r['method']==method and r.get('own',{}).get('point_coverage') is not None],
                   key=lambda r:r['own']['point_coverage'])[:3]
        lines.append(f"- {names[method]}：共同点平直度最差为 "+'、'.join(f"{r['sample']} ({f(r['common']['staff_straight'])})" for r in worst)+'。')
        lines.append('  源谱线点覆盖率最低为 '+'、'.join(f"{r['sample']} ({f(r['own']['point_coverage'],True)}%)" for r in low)+'。')
    lines += ['', '## 对当前可用性的判断与下一步','',
              '- 合成分布内的好成绩没有迁移到这批真实拍照五线谱。应将网络定位为仍需改进的实验模型，不能据此宣称通用照片校正已经可用。传统方案可作为本场景当前的对照和优先人工试用方案。',
              '- 下一轮训练优先加入贴边裁切、缺少可见纸边、轻形变/恒等映射和真实乐谱纹理的分布，加入页内方向/折叠约束，并检查高频坐标噪声；这属于基于失败现象的改进假设，本轮未训练验证。',
              '- 先解决 CPMS 标注坐标变换，或单独人工标注少量谱线/音符点，才能给出可靠的 GT 定量评测；之后再接入 OMR 比较音高与时值错误率。',
              '- CPMS 主要覆盖印刷五线谱，不回答简谱、混排、手写谱泛化问题。本次 60 页已暴露，后续调参后不能继续把它们称为未见测试集；应按内容组保留新素材，避免两张同谱照片跨训练和测试。',
              '', '## 复现与本地产物','',
              '```bash',
              '.venv/bin/python scripts/download_cpms.py',
              '.venv/bin/python scripts/evaluate_cpms.py --out outputs/cpms_repeat',
              '.venv/bin/python scripts/diagnose_cpms_fields.py --experiment outputs/cpms_repeat',
              '.venv/bin/python scripts/report_cpms.py --experiment outputs/cpms_repeat --out reports/cpms_repeat.md',
              '```','',
              '本地完整结果：`outputs/cpms_20261115/`。包含锁定协议/权重指纹、原图冻结谱线、逐页共同点与覆盖率、四方法对照图、三种校正映射，以及反演前坐标场诊断。下载脚本可复用并验证已有文件，评测脚本要求新空输出目录。',
              '', '本轮新增 4 项诊断指标测试，全套 53 项测试通过。覆盖尺度不变性、弯曲、低覆盖率、塌缩、共同点裁剪偏差及映射折叠。','',
              '## 固定样本可视化','',
              '按文件名排序固定取首张、中间张和末张，不依据改善程度挑选；每行从左到右为原图、传统方案、原始网络、微调网络。图片等比例缩放并留白，便于同时检查形变与内容缺失。','']
    chosen=[protocol['photos'][i]['sample'] for i in sorted(set((0,len(audit)//2,len(audit)-1)))];contact=[]
    for sample in chosen:
        tiles=[]
        for method in methods:
            image=cv2.imread(str(exp/sample/(method+'.jpg')))
            if image is None:image=np.full((600,450,3),255,np.uint8)
            h,w=image.shape[:2];scale=min(360/w,520/h)
            image=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
            tile=np.full((552,360,3),255,np.uint8)
            x=(360-image.shape[1])//2;tile[32:32+image.shape[0],x:x+image.shape[1]]=image
            cv2.putText(tile,f'{sample} {method}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1)
            tiles.append(tile)
        contact.append(np.concatenate(tiles,axis=1))
    preview=args.out.with_suffix('.preview.jpg')
    if not cv2.imwrite(str(preview),np.concatenate(contact,axis=0)):raise ValueError('Preview write failed')
    lines.append(f'![CPMS 固定样本对照]({preview.name})')
    args.out.write_text('\n'.join(lines)+'\n')
    print(args.out)


if __name__=='__main__':main()
