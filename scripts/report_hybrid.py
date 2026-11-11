"""Summarize network/C validation, including paired common-support metrics."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    summary = json.loads((args.eval / "summary.json").read_text())
    config = json.loads((args.eval / "config.json").read_text())
    rows = json.loads((args.eval / "per_sample.json").read_text())
    lines = ["# 网络与 C 局部精修验证", "", f"数据：`{config['data']}`，共 {len(config['samples'])} 页。",
             f"权重：`{config['checkpoint']}`。", "",
             "仅使用验证集。network 是原始预测坐标；network_bmap 是反演后的实际输出映射；network+C 再做两轮局部精修。",
             "反演映射额外要求输出点映回原图的误差不超过 2 px；页面掩码、画布边界和缺失映射均计入覆盖率损失。", "",
             "| 方法 | AD % 中位数 | 局部畸变 % 中位数 | 平均覆盖率 | 失败页数 |",
             "|---|---:|---:|---:|---:|"]
    def fmt(v):
        return "—" if v is None else f"{v:.3f}"
    for name, s in summary.items():
        lines.append(f"| {name} | {fmt(s['ad_pct_median'])} | {fmt(s['local_distortion_pct_median'])} | "
                     f"{s['coverage_mean'] * 100:.2f}% | {s['failed']} |")
    lines += ["", "## 相同有效点上的比较", "",
              "每页使用 network_bmap 与 network+C 都有效的点，分别重新做相似变换对齐。差值为 C 后减 C 前，负值表示改善。", "",
              "| 指标 | C 前中位数 | C 后中位数 | 逐页差值中位数 | 改善页数 |",
              "|---|---:|---:|---:|---:|"]
    by_method = {name: {r['sample']: r for r in rows if r['method'] == name} for name in summary}
    before, after = by_method['network_bmap'], by_method['network+C']
    for key in ('common_ad_pct', 'common_local_distortion_pct'):
        pairs = [(before[k][key], after[k][key]) for k in before.keys() & after.keys()
                 if before[k].get(key) is not None and after[k].get(key) is not None]
        if pairs:
            a, b = np.array(pairs).T
            lines.append(f"| {key} | {np.median(a):.3f} | {np.median(b):.3f} | {np.median(b-a):+.3f} | {int((b<a).sum())}/{len(a)} |")
    coverage_drops = sum(after[k]['coverage'] < before[k]['coverage'] - .05 for k in before.keys() & after.keys())
    lines += ["", f"C 后覆盖率下降超过 5 个百分点的页面：{coverage_drops}。",
              "本阶段仅接入 C，尚未将网络坐标作为 B 的 3D 曲面初值；结果不代表完整的网络+B+C 方案。",
              "全量训练使用 172 页验证集，pilot 使用 64 页；不同验证集的汇总数值不能直接用来判断增加训练数据的收益。"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
