# 网络与 C 局部精修验证

数据：`/Users/haolan/Downloads/MusicAI/score_dewarp/data/prepared/deepscores_pilot/val`，共 64 页。
权重：`/Users/haolan/Downloads/MusicAI/score_dewarp/outputs/deepscores_finetune/ft_ca_lr1e-3_s7/best.pt`。

仅使用验证集。network 是原始预测坐标；network_bmap 是反演后的实际输出映射；network+C 再做两轮局部精修。
反演映射额外要求输出点映回原图的误差不超过 2 px；页面掩码、画布边界和缺失映射均计入覆盖率损失。

| 方法 | AD % 中位数 | 局部畸变 % 中位数 | 平均覆盖率 | 失败页数 |
|---|---:|---:|---:|---:|
| network | 2.004 | 13.303 | 99.75% | 0 |
| network_bmap | 1.998 | 13.150 | 98.96% | 0 |
| network+C | 1.874 | 12.843 | 98.13% | 0 |

## 相同有效点上的比较

每页使用 network_bmap 与 network+C 都有效的点，分别重新做相似变换对齐。差值为 C 后减 C 前，负值表示改善。

| 指标 | C 前中位数 | C 后中位数 | 逐页差值中位数 | 改善页数 |
|---|---:|---:|---:|---:|
| common_ad_pct | 1.996 | 1.874 | -0.016 | 43/64 |
| common_local_distortion_pct | 13.114 | 12.843 | -0.061 | 60/64 |

C 后覆盖率下降超过 5 个百分点的页面：0。
本阶段仅接入 C，尚未将网络坐标作为 B 的 3D 曲面初值；结果不代表完整的网络+B+C 方案。
全量训练使用 172 页验证集，pilot 使用 64 页；不同验证集的汇总数值不能直接用来判断增加训练数据的收益。
