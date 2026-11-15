# 固定分层合成基准（2026-11-15）

本阶段完成合成素材扩展、固定划分与原模型基线评测；不涉及真实照片。

## 数据协议

- train/val/test = 300/90/90；三种谱种（五线谱、简谱、混排）× 五档条件（轻度、标准、强形变、投射阴影、模糊）均衡。
- train 每格 20 张；val/test 每格 6 张，每类各 30 张。照片 768×1024，独立逐页 seed。
- mild/standard/strong 的 strength 为 0.5/1.0/1.3；shadow 强制生成投射阴影，blur 固定 sigma=1.8，后两档 strength=1.0。其他降质仍随机，各档不是互斥因素。
- 固定使用 v2：97×129 投影网格的每格两个三角形均检查方向，最小有符号面积/中位面积需大于 0.02。480 张均通过，11 张生成时重采样形变；这是采样几何检查，不是连续可逆性的证明。
- 无重复平整页或照片；来源分组跨集合重叠为 0，所有坐标检查通过。保存字体、素材和标注的 SHA-256。
- 首版 train/val 审计发现 9 张局部投影翻折，已停用；修正不依据保留 test 的模型表现。
- 以下只评测 90 张 val；保留 test 在两组训练全部完成后才评测。

manifest SHA-256：`a8d286b9e2497cf66929c97c707b28d74ba730216ceacdc2f7c19b2fda07fb8e`。

## 原模型与传统方案（90 张 val）

原模型为 `outputs/deepscores_finetune/full_ca_lr1e-3_s7/best.pt`；传统方案为 A+B+C。统一输出宽度 1240。网络采用实际反演输出映射，不加 C；均检查画布边界、网络页面掩码（网络）和 2px 往返误差。

下表误差和覆盖率为逐页中位数；common AD 采用两种校正方法共同有效点。不同覆盖率下的 own AD 不单独用于判优。

| 谱种 | 方法 | 页数/异常 | own AD% | 覆盖率% | common AD% | 谱种正确率% |
|---|---|---:|---:|---:|---:|---:|
| jianpu | input | 30/0 | 3.152 | 100.000 | — | — |
| jianpu | traditional | 30/0 | 0.868 | 77.472 | 0.862 | 66.667 |
| jianpu | network_only | 30/0 | 0.851 | 98.444 | 0.791 | 63.333 |
| mixed | input | 30/0 | 2.994 | 100.000 | — | — |
| mixed | traditional | 30/0 | 0.954 | 75.567 | 0.954 | 53.333 |
| mixed | network_only | 30/0 | 0.889 | 98.559 | 0.870 | 60.000 |
| staff | input | 30/0 | 2.777 | 100.000 | — | — |
| staff | traditional | 30/0 | 1.113 | 70.911 | 1.113 | 93.333 |
| staff | network_only | 30/0 | 1.075 | 98.522 | 0.997 | 96.667 |

## 共同关键点上的结构指标

缺失标注留空，不计为零；谱线平直度以线距归一，简谱基线以字高归一，越低越好。

| 谱种 | 方法 | 谱线平直度 | 线距 CV | 简谱基线 | 字高 CV | 字符倾角° |
|---|---|---:|---:|---:|---:|---:|
| jianpu | traditional | — | — | 0.060 | 0.031 | 0.669 |
| jianpu | network_only | — | — | 0.167 | 0.037 | 1.082 |
| mixed | traditional | 0.095 | 0.024 | 0.044 | 0.023 | 1.020 |
| mixed | network_only | 0.255 | 0.035 | 0.135 | 0.032 | 1.062 |
| staff | traditional | 0.077 | 0.021 | — | — | — |
| staff | network_only | 0.263 | 0.034 | — | — | — |

整页 AD 接近不能代表结构同样平直。此基准支持后续分别检验增加简谱/混排数据和增加结构一致性损失的作用。谱种来自手工结构检测器，不能解释为未训练分类头的效果，也不等于谱表实例或关键点精度。

## 复现与产物

```bash
.venv/bin/python scripts/prepare_structure_suite.py --out data/prepared/structure_suite_20261115_v2
.venv/bin/python scripts/check_external_data.py data/prepared/structure_suite_20261115_v2
.venv/bin/python scripts/evaluate_materials.py \
  --data data/prepared/structure_suite_20261115_v2/val \
  --checkpoint outputs/deepscores_finetune/full_ca_lr1e-3_s7/best.pt \
  --width 1240 --out outputs/structure_benchmark_repeat --save-images
```

生成与评测要求新的空目录。完整逐页结果、困难样本、五档分层报告及校正图在 `outputs/structure_ablation_20261115_v2/baseline_val/`。此前宽度 1600 的检测指标不与本轮直接横比。
