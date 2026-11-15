"""Summarize dual-domain validation, staged head fitting and guarded inference."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scoredewarp.learn.infer import load


def fmt(value, scale=1):
    return '—' if value is None else f'{value*scale:.3f}'


def preview(exp,destination):
    model,_=load(exp/'original_spatial/validated.pt')
    colors=np.array([[25,25,25],[230,230,230],[230,130,40],[50,160,240],[160,60,180]],np.uint8)
    rows=[]
    for sample in ('staff_blur_000','jianpu_blur_000','mixed_shadow_000'):
        photo=cv2.imread(str(ROOT/'data/prepared/structure_suite_20261115_v2/val'/sample/'photo.jpg'))
        feature=np.load(exp/'cache_original/dense_val'/(sample+'.npy')).astype(np.float32)
        target=np.load(exp/'cache_original/dense_val'/(sample+'.targets.npz'))
        photo=cv2.resize(photo,(feature.shape[2],feature.shape[1]),interpolation=cv2.INTER_AREA)
        rgb=photo[...,::-1].astype(np.float32).transpose(2,0,1)/255
        x=torch.from_numpy(np.concatenate([feature,rgb]))[None]
        with torch.no_grad():
            pred=model.head_seg(x)[0].argmax(0).numpy()
            prob=model.head_struct(x)[0].sigmoid().numpy()
        gt=np.stack([cv2.dilate(t,np.ones((3,3),np.uint8)) for t in target['struct']])
        def heat(prob):
            strength=prob.max(0)[...,None]
            colored=np.stack([prob[1],prob[0],prob[2]],-1)*255
            return np.clip(photo*(1-.7*strength)+colored*.7,0,255).astype(np.uint8)
        tiles=[]
        for title,image in [('input',photo),('GT regions',colors[target['seg']]),('predicted regions',colors[pred]),
                            ('GT structures (+1px view)',heat(gt)),('predicted structures',heat(prob))]:
            tile=cv2.resize(image,(270,360),interpolation=cv2.INTER_NEAREST if 'regions' in title else cv2.INTER_AREA)
            tile=cv2.copyMakeBorder(tile,30,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
            cv2.putText(tile,title,(5,21),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1)
            tiles.append(tile)
        strip=cv2.hconcat(tiles)
        strip=cv2.copyMakeBorder(strip,26,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(strip,sample,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
        rows.append(strip)
    if not cv2.imwrite(str(destination),cv2.vconcat(rows),[cv2.IMWRITE_JPEG_QUALITY,92]):
        raise ValueError(destination)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',type=Path,default=Path('outputs/auxiliary_20261115'))
    ap.add_argument('--out',type=Path,default=Path('reports/auxiliary_20261115.md'))
    args=ap.parse_args();exp=args.experiment.resolve()
    data=json.loads((exp/'evaluation.json').read_text())
    if (exp/'cli_audit.json').exists():
        data['cli_audit']=json.loads((exp/'cli_audit.json').read_text())
    for name,path in [('training_protocol',exp/'protocol.json'),('spatial_protocol',exp/'original_spatial/protocol.json'),
                      ('integrity_audit',exp/'integrity_audit.json')]:
        data[name]=json.loads(path.read_text())
    original=data['anchors']['original'];adapted=data['anchors']['adapted']
    final=original['models']['pixels'];kind=original['models']['kind']
    lines=['# 双域选模与辅助头训练（2026-11-15）','','本轮完成双域准入规则、谱种头训练与输出接入，以及版面/结构头的独立训练和检测质量评测。仅使用训练集与开发验证集；没有读取测试集，也没有新独立测试成绩。','','## 结论','',
           f"- 原模型起点的谱种头：新合成验证集原始分类准确率 {fmt(kind['kind']['synthetic']['kind']['accuracy'],100)}%，固定置信度融合后 {fmt(kind['kind']['synthetic']['fused']['accuracy'],100)}%；手工检测器 {fmt(kind['kind']['synthetic']['manual']['accuracy'],100)}%。DeepScores 融合准确率 {fmt(kind['kind']['deepscores']['fused']['accuracy'],100)}%。",
           f"- 空间辅助分支：完整版面 mIoU {fmt(final['pixels']['summary']['all']['seg']['miou'])}，结构宏 F1 {fmt(final['pixels']['summary']['all']['struct']['macro_f1'])}。结构召回提升，但存在较多假阳性，尚不能直接作为 B/C 的几何观测。",
           '- 两个起点的校正参数及 BatchNorm 缓冲区均保持逐张量不变；空间分支也保持已训练谱种头不变。新辅助头没有改变校正映射。',
           '- 原模型起点通过绝对双域几何门槛，谱种头按固定置信度接入；数据微调起点保留上轮已有原域退化，未通过门槛，其谱种输出继续回退。','',
           '## 双域规则','',
           '- 验证：新合成 val 90 张（各谱种 30）+ DeepScores val 172 张；两个域分别检查，新合成集还逐谱种检查。',
           '- 手工检测器基线使用宽度 1240 的校正输出；kind 输入为 384×512。实际 CLI 回归另检查默认宽度 1600。',
           '- 参照：原始全量模型。各域/谱种的 UV EPE、AD、局部畸变、谱线/基线误差允许相对增加不超过 5%（仅加 1e-6 数值余量）；页面覆盖率最多下降 0.5 个百分点，采样方向错误率最多增加 0.05 个百分点。缺失结构标注留空，不计零误差。',
           '- 辅助权重选择与输出准入分开：kind 按两个域等权的 balanced accuracy 保存最优；DeepScores 只有 staff，那里等价于 staff 召回。像素头按新合成集 0.5×mIoU + 0.5×结构宏 F1 选模，DeepScores 无像素标注不参与该分数。',
           '- 谱种准入：固定置信度 0.8，低置信度回退手工检测；新合成融合 balanced accuracy 至少比检测器高 5 个百分点，DeepScores 融合准确率最多下降 1 个百分点，并通过绝对几何门槛。未经验证的旧 kind 头不会启用。',
           '- 几何统计为训练分辨率 384×512 的原始坐标场：UV EPE 对所有 GT 页内像素计算，AD/局部畸变在 stride-8 网格计算；下表取逐页均值，不与上轮原始分辨率/实际反演映射的中位数直接横比。结构误差来自 GT 关键点，覆盖率为 GT 页内像素的预测页面覆盖。新权重还通过冻结参数指纹核对，因此实际反演输出与相应起点相同。','',
           '| 起点 | 验证域/谱种 | UV EPE% 均值 | AD% 均值 | 覆盖率% | 谱线误差 | 简谱基线 |','|---|---|---:|---:|---:|---:|---:|']
    for anchor,entry in data['anchors'].items():
        for domain,metrics in entry['geometry'].items():
            lines.append(f"| {anchor} | {domain} | {fmt(metrics['coord_epe_pct'])} | {fmt(metrics['ad_pct'])} | {fmt(metrics['coverage'],100)} | {fmt(metrics['staff_straight'])} | {fmt(metrics['jp_baseline'])} |")
    lines+=['','## 分阶段训练','',
            '- 相同训练素材：300 张三谱种合成页 + 固定 100 张 DeepScores train 回放页；来源分组与两个验证集隔离。',
            '- 两个起点：原始全量权重、上轮 data_only 权重。先冻结骨干和几何头（eval 模式固定 BatchNorm），用真实 bottleneck 的池化特征训练 kind MLP：120 轮、LR=0.003、batch=64、按训练类频率平衡 CE，seed=7。',
            '- 然后训练原有 1×1 seg/struct 头：每张训练页均匀无放回采样 8,192 像素，每张验证页固定 16,384 像素；20 轮、LR=0.005，seg 类频率平方根平衡 CE，struct 正样本权重 20。缺少像素标注的 DeepScores 回放不作为负样本。',
            '- 线性头几乎不检出简谱基线/竖线，因此在原模型起点追加空间分支：RGB + 冻结 decoder 的 32 通道特征，16 隐藏通道；seg 卷积 dilation=1,2，struct=1,2,4，GroupNorm。20 轮、LR=0.001、batch=4，seg CE + 0.5×按训练频率平衡的 BCE（权重上限 200）+ 0.5×Dice。骨干和 kind 仍冻结。',
            '- 空间分支比较同时改变了头结构、RGB 输入与损失；不能把全部收益单独归因于卷积结构。每两轮（另含第 1 轮）做完整验证选模。所有像素特征缓存为 float16，头训练/推理为 float32。','',
            '## 谱种验证','',
            '| 起点/阶段 | 合成 kind 准确率% | 合成融合准确率% | DS kind 准确率% | DS 融合准确率% | 双域几何通过 | kind 启用 |','|---|---:|---:|---:|---:|---|---|']
    for anchor,entry in data['anchors'].items():
        for stage in ('kind','pixels'):
            r=entry['models'][stage];k=r['kind'];ready=r['kind_readiness']
            lines.append(f"| {anchor}/{stage} | {fmt(k['synthetic']['kind']['accuracy'],100)} | {fmt(k['synthetic']['fused']['accuracy'],100)} | {fmt(k['deepscores']['kind']['accuracy'],100)} | {fmt(k['deepscores']['fused']['accuracy'],100)} | {ready['geometry_passed']} | {ready['ready']} |")
    lines+=['','融合结果是假设按固定置信度使用该头的验证结果；实际运行还要求 kind 启用=True。adapted 行未通过门槛，所以不会实际启用其融合输出。所有完整混淆矩阵见随附 JSON。','','## 独立像素与结构验证','',
            '在 90 张新合成验证页的完整像素上计算；阈值固定为 0.5。版面类别为页外/纸/谱表区/简谱区/文本歌词。下表结构通道为谱线/简谱基线/竖线。','','| 起点/阶段 | mIoU | 宏 F1 | 谱线 F1 | 基线 F1 | 竖线 F1 |','|---|---:|---:|---:|---:|---:|']
    for anchor,entry in data['anchors'].items():
        for stage,r in entry['models'].items():
            if not r['pixels']:continue
            p=r['pixels']['summary']['all'];f=p['struct']
            lines.append(f"| {anchor}/{stage} | {fmt(p['seg']['miou'])} | {fmt(f['macro_f1'])} | "+' | '.join(fmt(c['f1']) for c in f['channels'])+' |')
    lines+=['','空间分支各通道质量：']
    p=final['pixels']['summary']['all']
    lines += ['', '空间版面各类 IoU：' + '、'.join(name+'='+fmt(value) for name,value in
              zip(('页外','纸','谱表区','简谱区','文本歌词'),p['seg']['iou'])) + '。文本/歌词区域仍是明显短板。', '',
              '| 通道 | precision% | recall% | F1 | 1px 容差 F1 | 固定采样 AP |','|---|---:|---:|---:|---:|---:|']
    for i,name in enumerate(('staff','jianpu baseline','vertical')):
        c=p['struct']['channels'][i]
        lines.append(f"| {name} | {fmt(c['precision'],100)} | {fmt(c['recall'],100)} | {fmt(c['f1'])} | {fmt(p['tolerant_1px'][i]['f1'])} | {fmt(p['sampled_average_precision'][i])} |")
    lines+=['','1px 容差是在模型输入分辨率上计算，precision 和 recall 分别按预测/真值邻域匹配。AP 取固定均匀验证像素，阈值无关。另记录 GT 结构实例上至少 50% 像素被 1px 邻域覆盖的比例；这是 GT 行覆盖诊断，不是预测实例的一对一 precision/recall，高行覆盖不能抵消假阳性。','',
            '## 输出与后续','',
            '- 实际 CLI 回归：固定 `jianpu_blur_000` 从 unknown 改判 jianpu（置信度约 0.824）；backward map、校正图和增强图逐像素一致。',
            '- `original_kind/validated.pt` 可用于带谱种置信度回退的校正；`original_spatial/validated.pt` 另带已训练空间分支，seg/struct 当前仅独立评测，不进入 B/C，也不替换区域或关键点导出。',
            '- CLI 的 `page_type` 采用准入后的分类决策，`page_type_source` 标明来源，`stages.kind_decision` 保留置信度及检测器原标签；`summary` 仍描述手工检测到的结构。旧权重没有验证元数据时保持检测器分类。',
            '- 下一步应先解决简谱基线与竖线的假阳性、文本/歌词区域的分割不足，建立实例解码与置信度校准，再研究用于几何精修。本轮没有独立测试成绩，不能把开发验证准确率当成泛化精度。','',
            '```bash',
            '.venv/bin/python scripts/train_auxiliary.py --out outputs/auxiliary_repeat --device mps',
            '.venv/bin/python scripts/train_spatial_auxiliary.py --experiment outputs/auxiliary_repeat --device mps',
            '.venv/bin/python scripts/evaluate_auxiliary.py --experiment outputs/auxiliary_repeat --device mps',
            '.venv/bin/python scripts/report_auxiliary.py --experiment outputs/auxiliary_repeat --out reports/auxiliary_repeat.md',
            '```','',f'本地完整权重、特征缓存、配置与历史：`{exp.relative_to(ROOT)}`。训练脚本要求新输出目录；源码改变时不静默复用旧缓存。']
    args.out.parent.mkdir(parents=True,exist_ok=True)
    image_path=args.out.with_suffix('.preview.jpg');preview(exp,image_path)
    lines+=['','## 固定验证样本可视化','','三个样本固定为 staff_blur_000、jianpu_blur_000、mixed_shadow_000，不按结果挑选。结构颜色：绿=谱线，蓝=简谱基线，红=竖线。真值结构仅在图中加粗 1px 便于观看，评测仍用原真值。','',f'![辅助头验证]({image_path.name})']
    args.out.write_text('\n'.join(lines)+'\n')
    args.out.with_suffix('.json').write_text(json.dumps(data,indent=2))
    print(args.out)


if __name__=='__main__':main()
