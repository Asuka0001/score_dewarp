# score_dewarp — 五线谱 / 简谱扭曲照片的平直化校正（v0.1）

输入一张手机拍摄的乐谱照片（透视、卷曲、旋转、阴影、遮挡；五线谱 / 简谱 / 混排），输出：

- **校正后的彩色图像** + **光照归一化增强图**（便于 OMR / 阅读）
- **谱种标签**：`staff` / `jianpu` / `mixed` / `unknown`
- **版面区域**：五线谱谱表区、简谱音符行、文本 / 歌词行（多边形）
- **结构关键点**：谱线折线（含谱表内线号）、简谱数字框（含识别出的 0–7）、文字行框
- **变换场**：输出像素 → 原图像素的 backward map（`*_bmap.npy`）

v0.1 提供**以结构先验为核心的几何优化方案**，可以直接运行、不依赖训练数据；同时提供已完成几何预训练与乐谱微调的多任务网络，以及合成数据、辅助头训练和双域验证工具。

> **当前主线（2026-11-17 起）**：只保留一条矫正管线。页面单应 + 纠斜 → 针孔相机 + 一般柱面（可展曲面）拟合 → 按方向平滑的残差场，见 `scoredewarp/core/metric_pipeline.py` 和 `gcs.py`。它是 `dewarp.py` 的默认方法。
> 评测改用 Doc3D 真实纸张形变重新贴图的乐谱基准（`data/bench/doc3d_score`，dev/test 各 120 页），指标全部来自真值。
> 在测试集（只评一次）上，相比旧 ABC：内容区 AD 1.76 → 0.97，LD 13.1 → 7.0，谱线平直度 0.47 → 0.063，竖线倾角 3.4° → 1.7°，简谱基线 0.136 → 0.052，页面覆盖 0.90 → 0.998。
> 网络 UV、UVDoc 和混合分支已从 CLI 移除，代码保留。详见 [主线报告](reports/metric_route_20261117.md)。下文第 2–19 节为历史记录。

---

## 1. 快速开始

```bash
cd score_dewarp
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 校正自己的照片（默认：主线 metric 管线）
.venv/bin/python scripts/dewarp.py my_score.jpg --out outputs/run
# 旧 ABC，仅作对照
.venv/bin/python scripts/dewarp.py my_score.jpg --method traditional --out outputs/abc

# 主线评测：构建 Doc3D 重贴图基准，在 dev 上比较（test 只在最终验收时跑一次）
.venv/bin/python scripts/make_doc3d_bench.py --out data/bench/doc3d_score --per-split 120
.venv/bin/python scripts/bench.py --data data/bench/doc3d_score/dev --methods input abc metric --out outputs/bench/dev

# 生成合成基准（带真值），评测所有方法和消融
.venv/bin/python scripts/make_synth.py --n 10 --out data/synth
.venv/bin/python scripts/evaluate.py --data data/synth --out outputs/eval --save_images

# 学习模型（几何头已完成 Doc3D 预训练，正式配置与结果见第 9 节）
.venv/bin/python scripts/make_synth.py --n 300 --out data/train --seed 7
.venv/bin/python scripts/train.py --data data/train --epochs 50
```

## 2. 目录

```
scoredewarp/
  synth/      render.py   合成平整乐谱页（五线谱 / 简谱 / 混排）+ 逐元素真值标注
              distort.py  物理仿真拍照：可展曲面卷曲 + 3D 位姿 + 针孔相机 + 阴影/遮挡/模糊/JPEG
  core/       preprocess.py  光照归一化、二值化
              page.py        A：页面四边形 → 单应；投影轮廓法纠斜
              structures.py  结构检测：谱线/谱表、简谱数字行/文字行、竖线线索、谱种分类
              surface3d.py   B：3D 页面曲面 + 相机，结构先验约束的鲁棒非线性最小二乘
              field.py       C：非刚性残差形变场（稀疏线性最小二乘 + IRLS）
              transforms.py  BackwardMap：映射的组合、正/反向点映射
              pipeline.py    串联各阶段，输出结果
  eval/       metrics.py     结构感知评测指标
  learn/      dataset.py / model.py / infer.py   多任务网络（坐标场 + 版面 + 结构热图 + 谱种）
scripts/      dewarp.py  make_synth.py  evaluate.py  train.py
```

## 3. 方案总览

```
 照片 ──► A 粗校正 ──► 结构检测 ──► B 3D曲面+相机拟合 ──► 重新检测 ──► C 非刚性残差场 ──► 输出
         页面四边形单应   谱线/谱表          (全局透视+卷曲,            (局部鼓包/折痕,     校正图/增强图
         + 投影纠斜       简谱数字行         弧长重参数化)               逐点线性约束)       谱种/区域/关键点
                          文字/歌词行                                                       变换场
                          小节线/符干
```

**核心思想**：五线谱、简谱、文本，这几类结构在平整纸面上都满足一些简单的几何关系，比如“同一条线上的点 Y 相同”“相邻谱线等距”“同一行字高相同”“小节线竖直”。把检测到的结构当作观测，各谱种的几何关系当作约束，去反求一个**物理合理**的页面形变。所有约束都在同一个优化框架里统一表达，混排页面上五线谱区和简谱区的约束会同时生效、互相补充。

- **A 粗校正**：页面四边形检测（亮度/饱和度分割 + 凸包多边形逼近，再检查角度、贴边情况和与分割区域的 IoU），透视矫正到正视图，画布四周留 15% 边距；在 ±20° 范围内用投影轮廓方差估计倾斜。检测失败（纸面贴边、背景颜色接近纸面）时退化为只做纠斜，透视交给 B 处理。
- **B 3D 曲面 + 相机**：曲面 `z(X,Y)` 用 8 项低阶多项式表示（X 方向卷曲到 4 次，Y 方向弯曲，加交叉项），相机为针孔模型，位姿用 solvePnP 从 A 的单应初始化。未知量包括全局参数（位姿 6 维 + 曲面 8 维）和每个结构自己的平面坐标。残差是重投影误差（soft-L1 鲁棒核）加上各谱种的先验（见第 4 节）。输出时沿曲面做**弧长重参数化**，避免卷曲处被压扁。
- **C 非刚性残差场**：在 B 的输出上重新检测结构，求解定义在 48px 节点网格上的前向场 `F=(U,V)`。所有先验都写成**关于 U 或 V 的线性差分约束**，再加二阶平滑项和弱恒等先验，用稀疏最小二乘求解，IRLS-Huber 迭代 4 轮。“检测 → 求解”默认重复 2 轮。B 的低阶模型表达不了的局部鼓包、折痕和检测误差，由 C 来修正。
- 各阶段统一用 `BackwardMap` 表示，可以组合，**只对原图重采样一次**；正向点映射用于导出关键点和做评测。

## 4. 五线谱与简谱的区别（题目要求重点）

| 维度 | 五线谱 | 简谱 |
|---|---|---|
| **校正目标** | 谱线水平；谱线间距均匀（局部和全页都要求）；谱表矩形化（上下线等长、左右对齐）；符头与线/间的相对关系稳定，便于 OMR 判断音高 | 数字行基线**和顶线**都水平；字符竖直、尺寸一致；行距均匀；小节线竖直；高低音点、减时线、增时线与数字的相对位置不变；歌词与对应音符纵向对齐 |
| **主要几何线索** | 贯穿全宽的长细线，**5 线等距**这一强周期结构；竖直的小节线和符干 | **没有长线**。线索来自离散的字形：数字 0–7（字高统一、无下伸部）、短下划线（减时线）、短横（增时线）、点、小节线、汉字歌词行 |
| **检测方法** | 游程统计估计线粗和线距 → 多方向形态学开运算 → **逐列游程跟踪**（谱线因符头粘连时依然稳健）→ 每列找 5 根等距线，用并查集聚成谱表 | 去掉长横线 → 连通域 → 面积加权估计主字高 → 左右“互为最近邻”链接成行 → **数字模板匹配**（0–7，多种字体）判断是简谱行还是文本行 |
| **3D 拟合中的先验** | 每个谱表只设一个 `Y0` 和间距 `s`：第 i 线 `Y=Y0+i·s`（等距是硬结构），各谱表的 `s` 再软约束趋于一致 | 每行设 `Y_top` 和 `h`：所有完整字高的数字顶点落在 `Y_top`、底点落在 `Y_top+h`（**双线约束**，同时约束了倾斜和局部缩放）；各简谱行的 `h` 软约束趋于一致；歌词行单独一组 |
| **残差场中的约束** | V：沿线相等，相邻线差 = s；U：沿线弧长 × (s_目标/s_局部)，即用谱线间距做局部尺度归一 | V：顶线相等、底线相等、底−顶 = h；U：相邻字的间距 × (h_目标/h_局部)，即用字高做局部尺度归一 |
| **竖直方向的信息来源** | 小节线、符干（长、直，很可靠） | 小节线、数字 “1/4” 和汉字里的竖笔画；**不使用外接框中心**（外接框中心不携带剪切信息，用它反而会把照片的剪切固化下来，实验已验证） |
| **评测** | 谱线平直度（y 标准差 / 线距）、线距变异系数、小节线倾角 | 基线平直度（y 标准差 / 字高）、字高变异系数、字符倾角、高低音点相对偏移误差、歌词—音符对齐误差、行距变异系数 |

**混排页面**：同一次优化中，谱表区的约束（等距 5 线）和简谱区的约束（字高双线）同时作用于同一个曲面或形变场，二者在各自覆盖的区域内提供证据，空白区域由平滑项衔接。谱种标签由区域统计得到：至少有 1 个谱表且至少有 1 个简谱行，判为 `mixed`。

**为什么不能只用五线谱线先验**：消融实验（见第 6 节）中，在简谱页上只用谱线先验（加竖线），基线平直度只从 0.90 降到 0.79，基本没校正；加上简谱先验后降到 0.02 左右。

## 5. 学习模型

`learn/model.py` 是一个共享 U-Net，带 4 个输出头：

- **fwd**：逐像素的“照片 → 平面页坐标”（DocUNet 风格的稠密坐标场）+ 页面掩码。推理时用散点反演得到 BackwardMap，与几何方案共用同一套接口。
- **seg**：版面 5 类（页外 / 纸 / 五线谱区 / 简谱区 / 文本歌词区）。
- **struct**：谱线、简谱基线、竖线三类热图。已完成独立训练与评测；实例解码和置信度尚未校准，当前不替换手工检测器，也不用于 B/C（见第 14 节）。
- **kind**：谱种分类。

损失：坐标 L1 + 梯度一致性（局部畸变）+ CE/BCE，外加按谱种设计的**结构一致性损失**：同一条谱线或简谱基线实例上，预测的平面 Y 方差应为 0；同一条小节线/符干实例上，预测的平面 X 方差应为 0。这样第 4 节的先验以可微的形式进入了训练。

当前状态：数据集、模型、损失、训练（MPS/CUDA/CPU）、推理都已打通。
已用 Doc3D 第 8 分片完成 **30 轮几何预训练**，并建立两个验证集上的传统几何基线；配置和结果见第 9 节。
全量几何微调与三谱种训练对照已完成（见第 11、13 节）。谱种头已完成开发验证，可通过准入元数据启用；版面与结构头已训练，但检测精度不足，当前仅独立评测（见第 14 节）。真实手机照片尚未评测。

可选结构改进（默认关闭，旧权重与旧命令不受影响，见第 10 节）：
`--coord-input` 在输入端拼接归一化 (x, y) 坐标通道（CoordConv）；
`--context attn` 在 stride-16 瓶颈处加入 2 层带 2D 位置编码的自注意力（门控残差，门初始化为 0）。
两者在初始化时均为恒等映射，可直接用旧权重（如 Doc3D 预训练）初始化；网络结构写入 checkpoint 的 `arch` 字段，推理时自动重建。

## 6. 评测（合成基准，每类 10 页，共 30 页）

真值来自合成器：把每个元素的平面坐标经仿真形变映射到照片上，再经待测方法的正向映射映射到输出图，在输出图里测量。指标都用输出图自身的线距或字高归一化，与输出分辨率无关；`ad_pct` 是做最优相似变换对齐后的残差，以页宽的百分比表示。

下表为**中位数**（每类 10 页；均值会被少数极端失败样本拉高，完整的均值表见 `outputs/eval/summary.md`，逐样本结果见 `per_sample.json`）。

**五线谱页**

| 方法 | ad_pct | 谱线平直度 | 线距 CV | 小节线倾角° |
|---|---|---|---|---|
| 原始照片 | 2.164 | 2.627 | 0.064 | 5.04 |
| A 粗校正（四边形 + 纠斜） | 1.688 | 0.583 | 0.027 | **0.55** |
| A + B 3D 曲面 | **0.517** | **0.064** | 0.018 | 0.62 |
| A + B + C 完整（全部先验） | 0.599 | 0.074 | **0.012** | 1.09 |

**简谱页**

| 方法 | ad_pct | 基线平直度 | 字高 CV | 字符倾角° | 高低音点误差 | 歌词对齐误差 |
|---|---|---|---|---|---|---|
| 原始照片 | 2.873 | 1.163 | 0.078 | 5.48 | 0.084 | 0.235 |
| A 粗校正 | 2.592 | 0.525 | 0.046 | **0.56** | **0.009** | **0.026** |
| A + B 3D 曲面 | 0.981 | 0.086 | 0.039 | 1.12 | 0.017 | 0.050 |
| A + B + C 完整（全部先验） | **0.928** | **0.082** | **0.037** | 1.09 | 0.017 | 0.049 |
| 完整，但**只用五线谱先验** | 4.061 | 1.253 | 0.095 | 1.56 | 0.025 | 0.063 |

**混排页**

| 方法 | ad_pct | 谱线平直度 | 简谱基线平直度 | 字符倾角° | 小节线倾角° |
|---|---|---|---|---|---|
| 原始照片 | 2.460 | 1.880 | 1.000 | 5.19 | 4.94 |
| A 粗校正 | 1.287 | 0.681 | 0.402 | 0.77 | 0.81 |
| A + B + C 完整 | 0.746 | **0.073** | **0.042** | 1.22 | 1.27 |

谱种分类：完整方法 30/30 全部正确。每页耗时约 3 秒（CPU，Apple Silicon）。

**结论**
- 绝大部分收益来自 B（结构约束的 3D 拟合）：谱线和简谱基线的平直度提升约 30 倍。C 主要改善间距均匀性（线距 CV 0.018 → 0.012），对平直度的提升有限。
- **简谱先验不可或缺**：只用谱线先验时，简谱页的基线平直度为 1.25，和原图差不多，有时甚至比原图更差，因为没有约束的区域被平滑项随意拉扯；加上简谱先验后是 0.08。
- **目前的主要短板是竖直方向**：B/C 会引入约 0.5° 的剪切（小节线、字符倾角从粗校正后的约 0.55° 升到约 1.1°），高低音点和歌词的相对位置误差也随之略有上升。原因可能是竖直线索（小节线、符干）的权重和数量都远小于水平约束。下一步优先处理：提高竖线权重、加入谱表左右端对齐约束、在 B 中显式地给剪切加正则。
- 典型失败样本：局部陡度超过 20° 的 S 形强卷曲（staff_002，谱线平直度约 1.1）；页面在照片中很小且透视很强（jianpu_005）。


各列含义见 [scoredewarp/eval/metrics.py](scoredewarp/eval/metrics.py)，所有指标都是越低越好（`coverage` 和 `type_acc` 越高越好）。

## 7. 已知局限 & 下一步

0. **竖直剪切**（见第 6 节结论）：这是当前最值得先做的改进。
1. **只在合成数据上验证过**。下一步最重要的是收集一批真实手机拍摄的乐谱（五线谱、简谱、混排各若干张），人工标注少量关键点，在真实数据上评测，并据此调整阈值和合成器分布。
2. **俯拍时卷曲引起的横向压缩不可观测**：正对着拍卷曲页面时，线条在图像里几乎不弯，3D 模型推不出曲率。可以利用已知形状元素的宽高比（符头椭圆、简谱数字的宽度），或者让网络从外观（明暗）上去估计。
3. **90° / 180° 方向**：90° 检测默认关闭（`--rot90` 开启，基于投影轮廓，对简谱的列对齐容易误判）；180° 还未处理。可以用谱号位置、数字朝向，或网络的 kind 头加一个方向分类来解决。
4. **页面检测**：遇到浅色背景会失败（会退回到只纠斜，结果仍然可用）。可以改成边缘 + 直线拟合，或者用网络的页面掩码。
5. **遮挡**：手指遮挡区域没有结构证据，靠平滑项插值，目前可以接受；后续可以把网络的遮挡分割作为置信度掩码。
6. **混合方案**：用网络的 struct/seg 头替换手工检测器，用网络的坐标场初始化 B，再让 B/C 做结构约束精修（“学习给初值，优化保精度”）。
7. **下游指标**：接入 OMR（如 oemer / Audiveris）和简谱 OCR，报告识别准确率的提升。

## 8. 外部训练数据：DeepScoresV2 / Doc3D

新增三个脚本：`download_datasets.py` 下载、校验并解压官方数据；
`prepare_datasets.py` 转成当前训练器能读取的目录；`check_external_data.py` 检查坐标和数据划分。
所有原始数据、压缩包、生成样本均位于被 Git 忽略的 `data/` 中。

本次已准备：DeepScoresV2 Dense 全部 1,714 张原始页，以及 `data/prepared/deepscores_pilot/`
中的 train/val/test = **128/64/64** 张照片（114 个曲谱分组，无跨集合重叠）。
全部样本坐标检查通过；完成 2 步训练、1 步权重加载微调及原有合成数据兼容性冒烟测试，尚未正式训练。
可查看[数据对照图](data/prepared/deepscores_pilot/preview.jpg)和[检查报告](data/prepared/deepscores_pilot/validation.json)。
原始数据与压缩包约占 1.7 GiB，起步训练数据约占 445 MiB。
Doc3D 第 8 分片的 img + uv 已下载并完成大小检查与 ZIP 解压 CRC 校验，共 **2,066** 组配对数据。
`data/prepared/doc3d_pilot/` 已转换该分片全部样本，train/val/test = **1,683/190/193**，
共 1,870 个父网格分组，无跨集合重叠；全部样本坐标检查通过。
可查看 [Doc3D 对照图](data/prepared/doc3d_pilot/preview.jpg)和[检查报告](data/prepared/doc3d_pilot/validation.json)。
数据接入时完成 Doc3D 训练及跨数据集微调冒烟测试；后续已完成的 30 轮正式几何预训练见第 9 节。
Doc3D 解压数据约占 1.6 GiB，转换后的数据约占 903 MiB，下载压缩包另占约 1.63 GiB。

### DeepScoresV2 Dense：五线谱内容 + 合成拍照形变

来源：[官方 Zenodo](https://zenodo.org/records/4012193)，CC BY 4.0。
Dense 版共 1,714 页，压缩包 741,814,529 字节，MD5 为
`7237318e381e6e0848ec30eb82decb83`；保留原始框、语义分割和实例分割供后续结构训练使用。

```bash
.venv/bin/python scripts/download_datasets.py deepscores

# 起步数据：每个划分最多选 64 个平整页面，训练页各生成 2 个拍照变体。
# 输出目录必须为空；重复实验请使用新目录，避免旧样本混入。
.venv/bin/python scripts/prepare_datasets.py deepscores \
  --source data/external/deepscores --out data/prepared/deepscores_pilot \
  --limit 64 --variants 2 --seed 7
.venv/bin/python scripts/check_external_data.py data/prepared/deepscores_pilot --preview

# 正式训练命令；本次准备数据时只运行少量步骤来验证可训练性。
.venv/bin/python scripts/train.py --data data/prepared/deepscores_pilot/train \
  --epochs 50 --out outputs/deepscores_pilot.pt

# 扩展到全部页面（80/10/10 按曲谱分组，训练页各生成 5 个变体）。
.venv/bin/python scripts/prepare_datasets.py deepscores \
  --source data/external/deepscores --out data/prepared/deepscores_full \
  --limit 0 --variants 5 --seed 7
```

官方 Dense train/test 分别有 1,362/352 页，但根据文件名的曲谱 ID，二者存在同曲不同字体或页面的交叉。
这里合并原始索引后，按 `-aug-` 前的曲谱 ID 和 seed 做稳定哈希，重新分为 train/val/test。
每首曲子的所有页面、字体和形变变体固定在同一划分中；`manifest.json` 保留原始划分和来源。
`--limit` 在划分之后按页面采样，不是每类乐谱的数量。

平整页最长边默认缩到 1,200 像素，合成照片默认 768×1,024；可用
`--flat-long-edge`、`--photo-size W H` 和 `--strength` 调整。
当前转换只训练**几何坐标、页面掩码和 staff 谱种**；没有把符号分割强行当成版面或谱线实例真值，
因此 seg/struct 损失关闭，结构一致性项也没有实例参与。完整结构训练仍应结合现有合成谱数据。

### Doc3D：通用几何预训练

来源：[官方 Hugging Face](https://huggingface.co/datasets/StonyBrook-CVLab/doc3D-dataset)
和[项目说明](https://github.com/cvlab-stonybrook/doc3D-dataset)。
官方仓库需要登录并接受共享 HF 账户信息的访问条件，匿名下载会返回 HTTP 401。
需由用户在网页提交访问申请，并确认已获得授权，然后在本机执行 `hf auth login`，或在本机环境设置 `HF_TOKEN`。
仅完成本机登录不会自动获得数据集权限；HTTP 403 且提示不在 authorized list 中时，需等待授权，
并确保本机登录账户与网页获批账户一致。
不要将 Token 发到聊天、写进项目或提交到 Git。

```bash
# 访问开通后运行；仅取最小的第 8 分片 img + uv，共约 1.75 GB。
.venv/bin/python scripts/download_datasets.py doc3d --shards 8
# 转换此分片全部配对数据；如仅需每个集合 64 张，可改为 --limit 64 并使用新输出目录。
.venv/bin/python scripts/prepare_datasets.py doc3d \
  --source data/external/doc3d --out data/prepared/doc3d_pilot --limit 0
.venv/bin/python scripts/check_external_data.py data/prepared/doc3d_pilot --preview
.venv/bin/python scripts/train.py --data data/prepared/doc3d_pilot/train \
  --epochs 50 --out outputs/doc3d_pretrain.pt

# 几何预训练后，在乐谱数据上微调；--init 只加载模型权重。
.venv/bin/python scripts/train.py --data data/prepared/deepscores_pilot/train \
  --init outputs/doc3d_pretrain.pt --lr 1e-4 --out outputs/deepscores_finetune.pt
```

下载器支持 `.part` 断点续传，优先从 `HF_TOKEN` 或本机 HF token 文件读取凭据，不输出 Token。
扩展时可指定其他分片，如 `--shards 8 1 2`。不会下载整套约 549 GB 数据，也不需要体积很大的 bm/3D 标注。
Doc3D 的 UV 已是照片→平面的正向坐标场：按官方 `fm2bm.m` 约定，OpenCV 通道为
`validity,V,U`，转换为 `X=U, Y=1-V`，再乘 448，保存为当前 loader 的像素坐标格式。
按[官方父子网格命名说明](https://github.com/cvlab-stonybrook/doc3D-dataset/issues/8)，
从首个 `-` 之前的 mesh ID 提取末尾两段 `parentmeshA_parentmeshB` 分组，
使 `A_B` 父网格与 `child_A_B` 子网格固定在同一集合，避免形变泄漏。
Doc3D 没有项目所需的乐谱标签，seg/struct/kind 损失均关闭，只训练几何与页面掩码。
UV 轴序、缺失标注损失掩码、父子网格分组和下载授权错误提示均有单元测试；
真实 Doc3D 样本已完成加载、全量坐标检查、真值校正可视化以及训练/跨数据集微调验证。
对照图左列是 UV 坐标可视化，中列是输入照片，右列是真值坐标反演后的校正结果；
不代表当前模型的预测效果。

准备结果中的 `validation.json` 记录坐标范围、有效页面比例和曲谱/mesh 分组检查。
验证集和测试集不参与训练；当前 `train.py` 仍不自动执行验证或早停。
训练器默认关闭样本内存缓存，避免扩大数据集后持续占用内存。

## 9. 验证集基线与 Doc3D 几何预训练

`scripts/evaluate_geometry.py` 不依赖谱线、字符等结构标注，直接从真值坐标场中采样固定的输入照片网格。
当前实验固定使用 Doc3D val 的 190 张和 DeepScores val 的 64 张，test 集不参与训练、选模或本轮评测。

```bash
# 原图、A、A+B、A+B+C 四组基线，保持几何方案的默认 coarse_width=1600。
.venv/bin/python scripts/evaluate_geometry.py --data data/prepared/deepscores_pilot/val \
  --out outputs/baselines/deepscores_val --workers 2
.venv/bin/python scripts/evaluate_geometry.py --data data/prepared/doc3d_pilot/val \
  --out outputs/baselines/doc3d_val --workers 2
# 中断后使用完全相同的参数附加 --resume，可跳过已完成的样本。

# 几何预训练：方形文档用 256×256 输入，固定 seed=7，最多 30 轮。
# 自动选择 CUDA/MPS/CPU；macOS 沙箱内可能无法发现 MPS，实际运行需允许 GPU 访问。
.venv/bin/python scripts/train.py \
  --data data/prepared/doc3d_pilot/train --val-data data/prepared/doc3d_pilot/val \
  --geometry-only --epochs 30 --patience 8 --bs 8 --size 256 256 --lr 0.001 \
  --seed 7 --out outputs/doc3d_pretrain/best.pt

# 最佳模型的同域评测，以及未做乐谱微调的跨域验证。
.venv/bin/python scripts/evaluate_geometry.py --data data/prepared/doc3d_pilot/val \
  --checkpoint outputs/doc3d_pretrain/best.pt --out outputs/doc3d_pretrain/eval_doc3d_val
.venv/bin/python scripts/evaluate_geometry.py --data data/prepared/deepscores_pilot/val \
  --checkpoint outputs/doc3d_pretrain/best.pt --out outputs/doc3d_pretrain/eval_deepscores_val
```

评测指标：

- `ad_pct`：预测平面坐标经过最佳保方向相似变换对齐真值后，点误差 RMS / 平整页宽 × 100。
  消除整体平移、旋转、等比例缩放，保留剪切、非均匀缩放、卷曲误差。
- `local_distortion_pct`：对齐后的相邻网格边向量误差 RMS / 真值边长 RMS × 100。
- `coverage`：真值页面采样点中能得到有效输出的比例。误差仅在有效预测点上计算，必须结合覆盖率、失败数和未定义误差数一起读。
- 网络额外记录所有 GT 页面点的坐标 EPE，不因预测页面掩码而剔除难点。
- 网络还记录 `orientation_error_rate`：在 GT 页内的采样网格单元上，预测与真值局部方向不一致的比例，
  用于暴露折叠或坐标塌缩。它只检查采样网格，不保证整张映射可逆。

每页最多采样 4,096 个网格点；按页计算再报告均值和中位数。
本节 AD 的拟合方向为预测→真值，不能直接与第 6 节旧结构关键点评测的数值混用。
计时包括映射反演等评测开销，且基线并发执行，仅作本轮运行耗时参考。
逐样本结果实时保存到 `samples/*.json`，总表为 `summary.md` / `summary.json`，配置保存在 `config.json`。
网络评测配置还保存权重文件 SHA-256，避免复用同一路径下另一版模型的旧结果。
基线中发现并修复了没有结构观测时的空数组异常；这类页面现在回退到粗校正。
修复后用 `--resume --retry-failed` 重跑失败样本，原始错误记录保留于 `failures_before_retry/`。

训练器每轮评测验证集，以**所有 GT 页内像素上的逐页平均 UV 坐标 EPE**选择最佳模型，
两个坐标分别归一化到 `[0,1]`，乘 100 后报告百分数。它与相似对齐后的 AD 是不同指标。
同时记录页面 IoU，避免空掩码隐藏误差。连续 8 轮没有单轮至少 0.02 个百分点的改善时早停。
Doc3D 仅训练共享网络、坐标场和页面掩码，谱种、版面和结构头被冻结，不能据此期待谱种或结构检测结果。

权重与日志：`best.pt` 是最低验证 EPE 的权重，`best.last.pt` 是最后一轮，
`best.config.json` / `best.history.json` / `best.curves.png` / `best.report.json` 保存配置、曲线和完成报告。
`--init` 仅加载权重用于新一轮微调，不是优化器断点续训。已有权重路径会被拒绝覆盖。

本轮已完成 **30 轮 / 6,330 步** MPS 预训练，耗时约 **34.6 分钟**。
最佳模型来自第 **28** 轮：验证 UV 坐标 EPE **37.80% → 2.19%**，页面 IoU **0.9901**。
完整结果见[实验报告](outputs/report_archive_20261117/reports/geometry_pretrain.md)，训练曲线见[best.curves.png](outputs/doc3d_pretrain/best.curves.png)。

| 验证集 | 原图 AD % 中位数 | A+B AD % 中位数 | 预训练模型 AD % 中位数 | 模型平均覆盖率 |
|---|---:|---:|---:|---:|
| Doc3D（190 页） | 5.98 | 6.00 | **1.73** | 99.18% |
| DeepScores（64 页，未微调） | 3.22 | **1.95** | 8.48 | 98.60% |

模型在 Doc3D 上学到了几何校正，但未微调时在五线谱上出现明显退化。
其采样单元方向错误率在 Doc3D 上约为 0.031%，在 DeepScores 上约为 3.738%，
可查看[Doc3D 模型预览](outputs/doc3d_pretrain/preview_doc3d_val.jpg)和[五线谱模型预览](outputs/doc3d_pretrain/preview_deepscores_val.jpg)。
当前权重用于后续微调初始化；实际乐谱校正先沿用传统方案。
预训练对乐谱微调的收益，需要下一步与从零训练的 DeepScores 模型做对照，不能由本轮同域结果直接推断。
本节这轮预训练实验只评测验证集，未使用测试集；后续五线谱微调及所选模型的测试集结果见第 10 节。真实手机照片仍未评测。

```bash
# 生成定性预览和汇总报告。
.venv/bin/python scripts/preview_geometry_model.py --data data/prepared/doc3d_pilot/val \
  --checkpoint outputs/doc3d_pretrain/best.pt --out outputs/doc3d_pretrain/preview_doc3d_val.jpg
.venv/bin/python scripts/preview_geometry_model.py --data data/prepared/deepscores_pilot/val \
  --checkpoint outputs/doc3d_pretrain/best.pt --out outputs/doc3d_pretrain/preview_deepscores_val.jpg
.venv/bin/python scripts/report_pretrain.py
```

## 10. DeepScores 从零训练的坐标塌缩与网络改进

在准备“Doc3D 预训练 vs 从零训练”对照实验（`scripts/run_finetune_ablation.sh`）时发现：
原网络在 DeepScores 上以 384×512 **从零训练会塌缩**——所有像素都输出约 0.5 的常数坐标，
验证 EPE 38.2%，恰好等于“全部预测 0.5”的常数基线 38.3%，19 轮无进展。
原因：U-Net 感受野约 200 px，在 512 高的输入上看不到页边；而乐谱页大部分是白纸，
竖直方向上谱表重复出现，局部外观无法区分自己在哪一个谱表。
这样的对照只能说明“从零训练跑不起来”，无法衡量预训练的收益。

诊断（`outputs/deepscores_finetune/diag/`，pilot 128 张训练、64 张验证，seed 7，单次运行）：

| 设置（从零训练，lr 1e-3） | 分辨率 | 轮数 | 验证 EPE % | U / V 与真值相关系数 |
|---|---|---:|---:|---|
| 原网络 | 384×512 | 19（中断） | 38.15，塌缩 | — |
| 原网络 | 192×256 | 30 | **3.93** | — |
| + 自注意力 | 384×512 | 20 | 25.40 | 0.99 / **0.37**（V 仍塌缩） |
| + CoordConv | 384×512 | 20 | 8.74 | 0.98 / 0.99 |
| + CoordConv + 自注意力 | 384×512 | 20 | **7.27** | 0.99 / 0.99 |

| 设置（Doc3D 初始化，lr 1e-4） | 分辨率 | 轮数 | 初始 → 验证 EPE % |
|---|---|---:|---|
| 原网络 | 384×512 | 5 | 18.75 → 9.71 |
| + 自注意力 | 384×512 | 5 | 18.75 → 9.59 |
| + CoordConv + 自注意力 | 384×512 | 5 | 18.75 → 9.10 |

结论：
- 自注意力单独只解决了水平坐标 U，竖直坐标 V 依然塌缩；**CoordConv 是解决 V 塌缩的关键**，二者结合最好。
- 新模块初始化为恒等映射：从 Doc3D 权重初始化时初始 EPE 与原网络完全相同（18.75%）。
- 192×256 的原网络从零训练也能收敛，但分辨率减半会损失细节，后续正式实验建议用 384×512 + `--coord-input --context attn`。
- 以上为单 seed、短训练的诊断，各行轮数和学习率不完全相同，不能作为“预训练 vs 从零训练”的结论。

```bash
# 推荐的对照实验写法（每组相同数据、步数、分辨率和网络结构）
COORD=1 CONTEXT=attn scripts/run_finetune_ablation.sh scratch_ca_lr1e-3_s7 data/prepared/deepscores_pilot/train 60 1e-3 7
COORD=1 CONTEXT=attn scripts/run_finetune_ablation.sh ft_ca_lr1e-4_s7 data/prepared/deepscores_pilot/train 60 1e-4 7 \
  outputs/doc3d_pretrain/best.pt
```

全量 DeepScores 数据已重新生成到 `data/prepared/deepscores_full_v3/`（`--limit 0 --variants 2 --seed 7`）：
train/val/test = **2,810/172/137**，363 个曲谱分组，无跨集合重叠，坐标全部有效，约 5.3 GiB。
它与 `deepscores_pilot` 使用相同的分组哈希，pilot 的 train/val/test 分组分别只落在 full 的对应集合中，可混用而不泄漏。
此前中断的半成品 `deepscores_full_v2` 已删除。

### 正式对照：Doc3D 预训练 vs 从零训练（新结构）

`scripts/run_phase1_ca.sh`：384×512、`--coord-input --context attn`、pilot 128 张训练、每组 60 轮 / 960 步、seed 7，
以验证集 UV EPE 选模。test 集只评测按验证集选出的两组（各自 arm 的最佳）。完整表格见
[outputs/report_archive_20261117/reports/deepscores_finetune.md](outputs/report_archive_20261117/reports/deepscores_finetune.md)（由 `scripts/report_finetune.py` 生成）。

| 运行 | 初始化 | LR | 验证 EPE % | 验证 AD % 中位数 | 验证局部畸变 % 中位数 | 覆盖率 |
|---|---|---:|---:|---:|---:|---:|
| scratch_ca_lr1e-3_s7 | 随机 | 1e-3 | 4.61 | 3.52 | 22.85 | 99.26% |
| scratch_ca_lr3e-4_s7 | 随机 | 3e-4 | 3.71 | 2.78 | 20.61 | 98.85% |
| ft_ca_lr1e-4_s7 | Doc3D | 1e-4 | 3.24 | 2.78 | 21.75 | 99.68% |
| ft_ca_lr3e-4_s7 | Doc3D | 3e-4 | 2.69 | 2.12 | 15.38 | 99.75% |
| **ft_ca_lr1e-3_s7** | Doc3D | 1e-3 | **2.61** | **2.00** | **13.30** | 99.75% |

| test（64 页） | AD % 中位数 | 局部畸变 % 中位数 | 覆盖率 |
|---|---:|---:|---:|
| 原图 | 3.11 | 13.50 | 100% |
| A+B（传统） | 1.82 | **10.28** | 90.57% |
| ft_ca_lr1e-3_s7 | **1.70** | 12.74 | **99.74%** |
| scratch_ca_lr3e-4_s7 | 2.65 | 19.97 | 99.10% |

结论：
- **Doc3D 预训练有效**：同学习率下逐页配对比较，预训练在 AD 上均显著更好
  （lr 3e-4：55/64 页更好，p=5e-6；lr 1e-3：63/64，p=3e-11）；
  test 上两组选出的模型：59/64 页更好，AD 中位数差 −0.77，p=2e-10。预训练模型收敛也明显更快。
- **与传统几何方案相比**：最佳网络与 A+B 的 AD 逐页无显著差异（test p=0.98，val p=0.54），
  但局部畸变更差（test 上 18/64 页更好，p=0.003）；优势是覆盖率——A+B 在 test 有 9 页、val 有 11 页覆盖率低于 80%。
  因此网络更适合作为 B/C 优化的初值或兜底，而非直接替代（与第 7 节“学习给初值，优化保精度”的规划一致）。
- 局限：单 seed、仅 128 张训练页、仅合成形变；预训练组的最佳轮次接近第 60 轮，可能尚未收敛。
  下一步：多 seed 复现、在 `deepscores_full_v3`（2,810 张）上训练、把网络输出接入 B/C 精修。

## 11. 第二阶段：多 seed、全量训练与网络+C

2026-11-11 启动本机 MPS 训练队列，2026-11-12 全部完成（结果见本节末尾）。`scripts/run_next_stage.py` 顺序运行：

1. pilot 的 seed=17、27，各做从零训练 / Doc3D 初始化两组，统一 LR=1e-3、60 轮、384×512、CoordConv + attention。与已有 seed=7 的同 LR 结果组成三 seed 对照。
2. Doc3D 初始化的全量训练：train=2,810、val=172，seed=7、LR=1e-3、最多 60 轮，patience=12。按验证 UV EPE 选模。
3. 在全量验证集上比较原始网络坐标、反演后的实际映射、网络+C 局部精修。另已用现有 seed=7 pilot 最佳微调权重完成 64 页混合验证，结果见下方。

队列本身不做 test 评测。队列状态在 `outputs/next_stage/status.json`，各任务日志在同目录；当前 epoch 查看对应训练目录的 `best.history.json`。
多 seed 报告见 [outputs/report_archive_20261117/reports/deepscores_multiseed.md](outputs/report_archive_20261117/reports/deepscores_multiseed.md)。
全量权重目录为 `outputs/deepscores_finetune/full_ca_lr1e-3_s7/`。全量 val（172 页）的网络结果在 `eval_full_val/`（原 `eval_val/`，已改名），
混合结果在 `outputs/hybrid/full_val/`。`eval_val/` 和 `eval_test/` 现在是 64 页 pilot val/test 上的结果，与第 10 节及 `report_finetune.py` 的约定保持一致。

混合版本先使用 **网络 → C**，跳过 A/B：B 当前接收单应初值，尚未支持非线性网络场初始化。
`refine_dewarp` 把 C 的残差映射组合回原照片，避免对中间图像重复重采样；最终谱种仍由结构检测器产生，不采用未训练的 kind 头。
不保证 C 一定改善结果，需结合局部畸变和覆盖率检查。

```bash
# 新启动队列；已经有队列运行时不要重复启动。
.venv/bin/python scripts/run_next_stage.py --device mps

# 使用现有权重校正并做 C 精修；加 --network-only 可只看网络结果。
.venv/bin/python scripts/dewarp.py my_score.jpg \
  --checkpoint outputs/deepscores_finetune/ft_ca_lr1e-3_s7/best.pt --out outputs/hybrid/run

# 混合流程评测与报告（完成后可使用 --resume 复用相同配置的结果）。
.venv/bin/python scripts/evaluate_geometry.py --data data/prepared/deepscores_pilot/val \
  --checkpoint outputs/deepscores_finetune/ft_ca_lr1e-3_s7/best.pt \
  --hybrid --out outputs/hybrid/pilot_val --resume
.venv/bin/python scripts/report_hybrid.py --eval outputs/hybrid/pilot_val --out outputs/hybrid/pilot_val/report.md
```

混合评测使用预测页面掩码，并检查输出画布边界及 2 px 原图往返误差，避免映射反演跨越空洞时虚增覆盖率。
`summary.json` 额外记录 network_bmap / network+C 在每页共同有效点上的 `common_ad_pct`、`common_local_distortion_pct` 和 `common_coverage`，
用于区分真实精修收益与有效点减少造成的误差下降。四页冒烟结果仅用于确认流程可运行，不作为效果结论。

### 首轮网络+C 验证结果（pilot val，64 页）

该历史试验的原始指标位于 `outputs/hybrid/pilot_val/`，早期独立报告已清理。
当前结果与对照见 [文档方案报告](reports/document_mesh_20261117.md)。本轮 64 页均完成，无异常失败；19 项单元测试通过。

| 方法 | AD % 中位数 | 局部畸变 % 中位数 | 平均覆盖率 |
|---|---:|---:|---:|
| 原始网络坐标 | 2.004 | 13.303 | 99.75% |
| 网络反演映射 | 1.998 | 13.150 | 98.96% |
| 网络+C | 1.874 | 12.843 | 98.13% |

在 C 前后每页共同有效的点上，43/64 页的 AD 改善，60/64 页的局部畸变改善；
逐页差值中位数分别为 −0.016 和 −0.061 个百分点。改善幅度较小，仍未达到传统 A+B 的局部畸变水平。
C 后没有页面的覆盖率下降超过 5 个百分点，但整体平均覆盖率有所降低，后续需处理边界缺失和检测不稳定。

### 多 seed 复现（pilot，128 个训练样本）

三个 seed（7 / 17 / 27）下，Doc3D 初始化在 pilot val 上的 AD 都显著优于从零训练：每个 seed 都是 63/64 页更好，p < 1e-10；
验证 EPE 为 2.59–2.61%（预训练）对比 4.61–5.85%（从零）。预训练组 seed 之间的波动也小得多。
完整表格见 [outputs/report_archive_20261117/reports/deepscores_multiseed.md](outputs/report_archive_20261117/reports/deepscores_multiseed.md)。

### 全量训练结果与 test 评测（2026-11-12）

全量模型 `full_ca_lr1e-3_s7`：全量 val EPE 19.05% → **0.54%**（第 53 轮，约 7 h）；pilot 模型是 2.61%。
为了和基线及 pilot 模型逐页配对，全量模型另外在 64 页 pilot val/test 上评测（照片与全量数据中的对应页逐字节相同，
与全量 train 无曲谱分组重叠）。此外，在 137 页全量 test 上补跑了传统基线和 pilot 模型。
test 只评测一次，模型和 C 的参数都已按验证集固定。完整表格和检验见 [outputs/report_archive_20261117/reports/deepscores_full.md](outputs/report_archive_20261117/reports/deepscores_full.md)
（`scripts/report_full.py` 生成）。

| 全量 test（137 页） | AD % 中位数 | 局部畸变 % 中位数 | 平均覆盖率 | 覆盖率 <80% 页数 |
|---|---:|---:|---:|---:|
| 原图 | 3.131 | 13.38 | 100.00% | 0 |
| A+B（传统） | 1.929 | 11.57 | 90.60% | 15 |
| pilot 网络 | 1.794 | 12.97 | 99.74% | 0 |
| **全量网络** | **0.367** | **3.11** | **99.92%** | 0 |
| 全量网络+C | 0.386 | 3.05 | 97.99% | 0 |

结论：
- **训练数据规模是目前最大的收益来源**。全量网络在 137/137 页上 AD 和局部畸变都优于 pilot 网络，AD 降到约 1/5，局部畸变降到约 1/4。
  pilot test（64 页）和 pilot val 的结果与此一致（AD 0.347% / 0.375%）。
- **网络已全面超过传统 A+B**：AD 133/137 页更好，局部畸变 130/137 页更好（p < 1e-22）。
  之前网络局部畸变落后于 A+B 的问题已经消失，覆盖率仍接近 100%。第 10 节“网络只适合作为初值或兜底”的结论需要更新。
- **C 精修对全量网络已无收益**：在共同有效点上，AD 和局部畸变的变化都不显著（p ≥ 0.1，改善页数不到一半），
  反而平均覆盖率降低约 1 个百分点。对全量模型，默认应直接输出网络反演映射；C 只保留为可选项。
- 训练在第 53 轮早停，但最佳轮正是最后一轮：`min_delta=0.02` 是绝对阈值，EPE 降到约 0.55% 后每轮约 0.01 的改进不再计数。
  之后重新训练时，应改用相对阈值或关闭早停，跑满余弦学习率周期。

局限与下一步：
1. 所有形变都是合成的。0.37% 的 AD 只代表合成分布内的表现，**下一步最重要的是真实手机照片评测**（第 7 节第 1 项）。
2. DeepScores 只有五线谱；简谱和混排页面仍依赖传统方案，需要相应的训练数据。
3. 单 seed 的全量训练；未跑满学习率周期。

## 12. 结构评测与谱线检测修复（2026-11-15）

新增统一评测 `scripts/evaluate_materials.py`：几何误差、共同有效点比较、谱线平直度/线距、小节线倾角、简谱基线/字高/字符倾角/点与歌词误差，以及有标注时的谱表实例 precision / recall。缺失标注不计零分，结构关键点覆盖率单独记录。

修复浅色谱线丢失、断裂谱线分组及标题缺字方框误识别为 0。固定的 172 张 DeepScores val 校正图上，谱种正确输出从 57/172 提高到 160/172，无正确页面退化；仍有 12 张 unknown。该结果衡量谱种输出，不代表所有谱表和关键点准确。

改进检测用于最终谱种、区域、关键点导出；B/C 保持原有观测规则，避免未校准的恢复线改变几何拟合。32 项单元测试通过。详细设计、局限、复现命令及最终结果目录见 [结构评测报告](outputs/report_archive_20261117/reports/structure_evaluation.md)。

```bash
.venv/bin/python scripts/evaluate_materials.py --data data/synth \
  --checkpoint outputs/deepscores_finetune/full_ca_lr1e-3_s7/best.pt \
  --out outputs/materials_new --save-images
```

## 13. 固定合成基准与简谱/混排训练（2026-11-15）

`prepare_structure_suite.py` 固定生成三种谱种 × 五档条件（轻度、标准、强形变、阴影、模糊），
train/val/test = **300/90/90**。每页独立 seed、来源分组与 SHA-256；保存字体校验和，
检查重复平整页与照片，验证坐标范围及跨集合来源隔离。标准档也可能随机含阴影/模糊，各档不是互斥因素。
最终使用 v2：生成时检查 97×129 网格中每格两个三角形的投影方向，拒绝翻折与近切向形变，
避免给需要一一对应坐标场的模型提供不可靠真值；检查结果与重采样次数写入 manifest。

`run_structure_ablation.py` 先评测既有全量模型，再运行两组受控微调：
相同初始化、300 张合成训练页 + 固定 100 张 DeepScores train 回放页、seed=7、24 轮、LR=1e-4。
`data_only` 只训练几何；`structure_consistency` 额外约束水平/竖直结构实例的坐标方差（权重 50）。
两组均冻结 seg/struct/kind 头，按新 val 的 UV EPE 选模，训练全部完成后才评测保留 test。
另在 172 张 DeepScores val 检查原域保持。它验证结构约束对几何头的效果，尚未训练结构检测热图头。

统一评测现支持重复 `--checkpoint`，一次比较多个网络与传统方案；保留全体共同点及逐对共同点结果，
并按条件输出分层报告。本轮输出宽度统一为 1240，与此前宽度 1600 的检测指标不直接横比。

```bash
# 要求新的空数据目录；默认产生 480 张素材
.venv/bin/python scripts/prepare_structure_suite.py --out data/prepared/structure_suite_20261115_v2
.venv/bin/python scripts/check_external_data.py data/prepared/structure_suite_20261115_v2
# 一个 GPU 任务依次执行，状态和日志写入 outputs/structure_ablation_20261115_v2/
.venv/bin/python scripts/run_structure_ablation.py --device mps
```

训练器新增 `--geometry-only --consistency-weight 50`，保留页内结构实例，同时关闭辅助头监督；
`--consistency-weight 0` 为相同训练数据下的几何对照。未提供参数时保持原训练行为。
损失函数支持部分权重覆盖，并跳过采样点不足的实例，避免空均值导致 NaN。

固定基准及原模型结果见 [分层合成基准报告](outputs/report_archive_20261117/reports/structure_benchmark_20261115.md)。

两组均完成 24 轮 / 1,200 步，最佳权重都来自第 20 轮：验证 UV EPE 从 1.485% 降到
data_only **0.6823%**、structure_consistency **0.6799%**。
保留 test 上，增加数据后简谱/混排 AD 中位数分别从 0.745%/0.866% 降至 0.435%/0.422%，
网络覆盖率中位数约 98.5%–98.7%。额外结构约束有小幅平直度收益，但混排谱种识别下降，
且两组在 DeepScores val 的 AD 均从 0.383% 升至约 0.448%；保留两组权重，不切换通用默认权重。
完整配对比较、局限与困难页对照见 [训练对照报告](outputs/report_archive_20261117/reports/structure_ablation_20261115.md)。

## 14. 双域准入与辅助头（2026-11-15）

新增 `learn/selection.py`：两个域与新合成集各谱种分别检查几何/结构误差（最多相对增加 5%）、
覆盖率（最多下降 0.5 个百分点）、方向错误率（最多增加 0.05 个百分点）。缺失标注留空。
辅助任务选模按域等权，避免 DeepScores 页数更多而掩盖简谱/混排退化。

使用原始全量模型与上轮 data_only 两个起点，冻结几何骨干及 BatchNorm，依次训练 kind、seg/struct。
原模型起点的 kind 验证准确率为 93.3%；固定置信度 0.8、低置信度回退手工检测后，
新合成 val 从 73.3% 提高到 **90.0%**，DeepScores val 从 94.2% 提高到 **98.8%**。
该起点通过几何门槛，`original_kind/validated.pt` 可接入输出；data_only 起点未通过原域门槛，保持回退。

原有 1×1 结构头检出不足，另训练独立空间辅助分支（RGB + 冻结特征，校正和 kind 均不变）。
20 轮后，完整 val 的版面 mIoU 为 **0.471**，结构宏 F1 为 **0.306**；
简谱基线和竖线仍有较多假阳性，因此 seg/struct 只独立评测，不用于 B/C 或区域/关键点替换。
本轮仅使用开发验证集，没有独立测试成绩。

```bash
.venv/bin/python scripts/dewarp.py \
  data/prepared/structure_suite_20261115_v2/val/jianpu_blur_000/photo.jpg \
  --checkpoint outputs/auxiliary_20261115/original_kind/validated.pt \
  --network-only --out outputs/kind_demo
```

该固定样本从 unknown 改判 jianpu，映射、校正图和增强图逐像素不变。
JSON 的 `page_type_source` 与 `stages.kind_decision` 记录来源和置信度；`summary` 仍描述手工检测结构。
旧权重缺少验证元数据时不启用 kind 头。
完整双域规则、训练复现、分通道 precision/recall/AP、局限与可视化见 [辅助头报告](outputs/report_archive_20261117/reports/auxiliary_20261115.md)。

## 15. 结构热图阈值校准与独立验证（2026-11-15）

在已有 90 页开发验证集上分别校准三个结构通道的阈值，锁定后生成另 90 页独立合成素材。
谱线/简谱基线/竖线阈值为 **0.86/0.95/0.98**；新素材严格像素宏 F1 从 **0.307 → 0.458**，
1px 容差宏 F1 从 **0.523 → 0.688**，五档条件均改善。方向压细对照不如仅阈值方案，保持可选。
简谱基线严格像素召回从 93.9% 降到 36.0%，因此仍仅供独立结构诊断，不接入几何精修。

```bash
# 新空目录；校准参数锁定后才生成独立验证素材，不读取旧测试图像/标签。
.venv/bin/python scripts/calibrate_structure.py --out outputs/structure_calibration_repeat

# 导出三通道掩码、概率及叠加图，坐标为模型输入像素（384×512）。
.venv/bin/python scripts/predict_structures.py \
  outputs/structure_calibration_20261115/fresh/val/jianpu_blur_000/photo.jpg \
  --checkpoint outputs/auxiliary_20261115/original_spatial/validated.pt \
  --calibration outputs/structure_calibration_20261115/selected.json \
  --out outputs/structure_masks_demo
```

导出工具校验权重哈希与校准分辨率，不改动既有校正流程或权重。
本轮新增 4 项后处理测试，合计 49 项测试通过。
完整结果、召回代价、复现协议及固定样本对照见 [结构校准报告](outputs/report_archive_20261117/reports/structure_calibration_20261115.md)。

## 16. CPMS 真实拍照五线谱测试（2026-11-15）

新增固定版本 CPMS 下载与逐文件 Git blob 校验，包含 60 张完整照片及其标注、谱行裁图与语义文件。
本轮对完整照片比较原图、传统 A+B+C、原始全量网络、三谱种微调网络，不在 CPMS 上训练或调参。

**真实照片暴露了当前网络的泛化不足**：出现额外弯曲、局部撕裂，传统方案在固定源谱线诊断上更好。
合成测试成绩不能直接推断真实照片可用性；当前不建议默认将这些网络用于 CPMS 这类输入。

官方 JSON 的坐标缩放规则未明确，不能直接对应原始照片。本轮冻结原图检测的谱线作为伪参考，
在各方法共同有效点上比较平直度，并单独报告覆盖率与映射折叠；这些不是人工 GT、音符保留率或 OMR 准确率。

```bash
.venv/bin/python scripts/download_cpms.py
.venv/bin/python scripts/evaluate_cpms.py --out outputs/cpms_repeat
.venv/bin/python scripts/diagnose_cpms_fields.py --experiment outputs/cpms_repeat
.venv/bin/python scripts/report_cpms.py --experiment outputs/cpms_repeat --out reports/cpms_repeat.md
```

原始数据保存在被 Git 忽略的 `data/external/cpms/`；完整对照、逐页诊断与映射在 `outputs/cpms_20261115/`。
该轮原始结果保存在 `outputs/cpms_20261115/`；早期报告与三页缩略图已由
[当前完整报告](reports/document_mesh_20261117.md)和[全部 60 页对照](reports/comparisons_20261117/README.md)替代。

## 17. 多来源充分训练（2026-11-15 启动，2026-11-17 已完成）

全面审计 `data/` 后，混合使用 DeepScores 全量训练素材、Doc3D train、三谱种合成页，以及
CPMS 官方 training 谱行经质量筛选/校直后生成的配对纹理页；不使用 CPMS 已评测的 60 张照片训练。
补充恒等映射、轻形变、贴边裁切、阴影与模糊，并筛除粗网格上方向异常的监督映射。
最终 train/val = **11,236/1,073**，来源组跨集合重叠为 0。

本轮固定 **120 轮、关闭早停**，MPS、batch=8，并加入多尺度防折叠与页边界监督。
每个来源独立验证，再对四类来源等权选模；保留 last/best/里程碑、源码快照和断点恢复状态。
既有通用训练器也修复了小幅改善不能累计、可能提前早停的问题。

```bash
# 查看后台任务进度与训练日志
cat outputs/diverse_training_20261115/model/status.json
tail -n 20 outputs/diverse_training_20261115/train.log

# 更新当前进度报告
.venv/bin/python scripts/report_diverse_training.py

# 中断后恢复（勿在现有队列仍运行时重复启动；需要可用的 MPS）
.venv/bin/python scripts/run_diverse_training.py --resume
```

训练后队列自动运行 DeepScores/Doc3D 验证保持检查及 CPMS 固定回归。训练未完成前不宣称真实照片效果改善，
也不自动替换默认权重。完整来源、质量审计和配置见 [多来源训练报告](reports/diverse_training_20261115.md)。

目前 120/120 轮与既定回归均已完成。本轮网络的真实照片效果明显改善，但仍未追平传统 ABC；
其权重没有被设为默认。后续文档方法改造见第 19 节。

## 18. 下一步待验证：受约束网格与局部形变检查（2026-11-16）

**状态更新（2026-11-17）：第 17 节已完成；第 19 节实现了谱线引导、尺度/剪切受限的纵向残差网格，
并接入官方 UVDoc backward 网格模型。下面更完整的二维网格拟合、训练消融和新真实测试集仍属计划，不能视为均已完成。**

目标是解决轻微变形或近乎平直的输入被网络额外弯曲、局部拉伸甚至撕裂的问题。
普通纸张弯曲通常具有空间连续性，但真实折痕可能存在，因此采用整体平滑、允许有依据的局部折痕的先验。
当前网络逐像素预测二维正向坐标场；已有梯度匹配、结构一致性和防折叠损失，仍不能排除局部异常。
特别是正 Jacobian 不代表形变合理：横向拉长、纵向压缩可以不翻折，却严重破坏音符。

### 第一阶段：固定权重，验证受约束网格后处理

- [ ] 冻结本轮选出的 checkpoint，保存原始预测场，增加独立分支：
  **预测正向坐标场 → 受约束控制网格拟合 → 坐标反演 → 重采样输出**。
  从约 16×24 个控制点起步，比较不同网格密度；用连续插值生成稠密场。
  粗网格与光滑插值本身不保证无翻折，需同时检查网格单元和插值后的稠密场。
- [ ] 网格拟合同时考虑原预测拟合误差、二阶弯曲能量、单元方向及局部拉伸/剪切约束。
  Jacobian 奇异值及其比例用于发现异常压缩、拉长和剪切；统一坐标尺度，允许正常透视带来的尺度变化，
  避免将所有局部变换强迫为等距。对真实折痕使用稳健或自适应平滑，检查过度平滑的代价。
- [ ] 有可靠谱线检测时，加入同一谱表五线平直、近似等距、上下顺序不变，以及沿线采样点左右顺序不变的约束。
  按检测置信度加权，检测不足时跳过相应约束；不依赖当前未准入的辅助头。
- [ ] 在独立开发验证集上选择网格密度和约束强度，锁定后再做固定回归。
  保留原图、传统 A+B+C、旧网络、本轮网络、本轮网络+网格约束的同页对照；
  分别记录坐标反演前与最终重采样映射的指标、失败页及耗时。输出写入独立实验目录。

### 验证与验收重点

同时保留已知几何真值的合成恢复测试和真实照片测试；按内容/来源组划分，避免同谱泄漏。
恒等、轻形变、贴边、常规弯曲及真实折痕分别汇总，并检查五线谱、简谱和混排是否退化。
CPMS 已评测的 60 张照片仅作固定回归，不用于选择后处理参数；真实泛化结论需要新保留素材。
CPMS 教师校直纹理不等于原始照片的人工几何真值。

| 检查项 | 要发现的失败 |
|---|---|
| 原有坐标/边界 EPE、页掩码覆盖率和折叠率 | 整体几何退化；掩码覆盖率不等于最终内容保留率 |
| 分块坐标误差的 P95、最差块误差及逐页尾部 | 平均误差掩盖局部严重错误；分块尺度预先固定 |
| 极端拉伸、压缩、剪切比例及最大异常连通区域 | 无翻折但音符已变形；阈值在开发集上确定 |
| 谱线局部曲率、五线间距、上下/左右顺序 | 谱线变直但挤压、错序或局部起伏 |
| 恒等/轻形变样本的输出退化率 | 好输入被改坏；结合真值、固定判据及人工复查 |
| 最终输出的音符/谱行保留与可读性 | 反演、裁切和插值造成的内容损伤；有条件时加入 OMR 错误率 |

验收不能只看平均 EPE 或折叠率下降：必须检查最差页面和局部区域，确认严重撕裂、翻折、谱行丢失减少，
同时没有削弱正确恢复能力。无 GT 的几何检查只能发现部分异常，不能单独证明校正正确。

### 第二阶段：后处理有效后再做训练消融

- [ ] 将控制网格作为网络输出表示，与现有逐像素坐标头在相同数据和训练预算下比较。
- [ ] 分别消融二阶平滑、局部拉伸/剪切、五线谱间距与顺序约束，区分各项收益及过度约束造成的退化。
- [ ] 如二维约束仍不足，再考虑三维纸面形状与二维展开网格联合监督；需要相应三维标签和数据支持。

参考：[UVDoc：三维/二维网格联合预测](https://arxiv.org/abs/2302.02887)、
[Grid Regularization：边界与文本行约束优化](https://arxiv.org/abs/2203.16850)、
[DocGeoNet：全局三维形状与局部文本行几何](https://arxiv.org/abs/2210.08161)。
以上是借鉴方向，尚无本项目效果结论。

## 19. 成熟文档方法接入与谱线网格（2026-11-17）

> 2026-11-17 晚：本节的 `--method document / uvdoc / network` 已从 CLI 移除，相关代码模块保留。当前做法见文首"当前主线"和 [主线报告](reports/metric_route_20261117.md)。

原网络的结构约束只在训练时软惩罚；第 120 轮该项乘权重后仅占总损失约 0.16%，
选模也没有直接考核谱线平直度。新流程在推理时显式利用当前照片的谱线证据。

- `--method document`：保留 A/B 在原图相机坐标系的全局拟合，以及 C 的横向尺度、竖线与字符约束；
  再使用整线水平、同谱表等距的绝对目标网格校正残余起伏。
  加入二阶弯曲正则、Huber 重加权、局部尺度与剪切界限；不改善或损伤已检测谱表的解被拒绝。
  仅粗校正和最终检测均判为五线谱时启用。简谱/混排或证据不确定时保留原 ABC。
  分类与检测仍可能误判，详见报告的跨谱种真值结果。
- 网格只拟合水平线能直接约束的纵向残差，X 不变、上下边界固定，按单调列反演；
  不再为这个残差使用 Delaunay 散点反演。边界保证仅针对残差，不能恢复上游 B 已裁掉的标题。
- `--method uvdoc`：官方 MIT 许可预训练模型直接输出 backward 控制网格，再做相同谱线精修。
  固定官方 revision，下载时校验 Git blob 与 SHA-256；源码和许可在 `data/external/uvdoc/`。
  输入 488×712，输出沿用输入宽高比；它是独立基线，不是本项目已经训练好的新网络。
- `--method network --checkpoint ...`：保留旧权重作为初始化，增加谱线精修。
  学习分支的最终映射检查每个有效网格单元的四个 Jacobian 角，发现翻折时回退结构方案，JSON 记录原因。
  这只是局部方向检查，不能证明校正准确或全局可逆。`--network-only` 保留不回退、不精修的研究基线。
- `--method traditional` 复现原 ABC，并保留为无 checkpoint 时的默认；`--no-field` 同时关闭局部网格/C。
  旧权重与训练记录不变。Python API 的 `dewarp()` 保留传统行为，新入口为 `dewarp_document()`。

```bash
.venv/bin/python scripts/download_uvdoc.py
.venv/bin/python scripts/dewarp.py my_score.jpg --method uvdoc --out outputs/uvdoc_demo

# 同页比较：传统、新结构流程、本轮网络、官方 UVDoc，以及两者加谱线网格
.venv/bin/python scripts/evaluate_cpms.py --out outputs/document_cpms_repeat --document \
  --uvdoc-root data/external/uvdoc \
  --variant network_mesh=network_robust:M --variant uvdoc_mesh=uvdoc:M

# 三谱种验证集，使用独立几何真值；--per-kind-limit 5 可先跑固定均匀选样的 15 页
.venv/bin/python scripts/evaluate_materials.py \
  --data data/prepared/structure_suite_20261115_v2/val \
  --checkpoint outputs/diverse_training_20261115/model/best.pt \
  --document --width 1240 --out outputs/document_synth_repeat
```

参考 [RDGR / Grid Regularization](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Revisiting_Document_Image_Dewarping_by_Grid_Regularization_CVPR_2022_paper.pdf)
的显式行几何与平滑网格思想，以及 [UVDoc 官方模型](https://github.com/tanguymagne/UVDoc)。
本项目的纵向残差网格不是论文复现，也不是完整的 3D/2D 联合训练。
实测与局限见 [文档方案改造报告](reports/document_mesh_20261117.md)。
60 页 CPMS 开发回归中，传统/结构网格平直度中位数为 0.158/0.123（58/60 页更直）；
90 页带几何真值的合成回归中，五线谱平直度为 0.077/0.056，整体 AD 基本持平。
新方案的边缘墨迹代理指标有部分退化，旧网络加网格后仍检出局部翻折，因此未切换默认。
新增几何恢复、非谱线采样点、边界、异常证据和四角 Jacobian 检查，合计 71 项测试通过。
CPMS 已暴露，属于开发回归；不能以谱线自评分替代新真实照片的人工标注与 OMR 验收。

### 查看更多对照图

[同位置局部放大图](reports/comparisons_20261117/README.md) ·
[全部 60 页并排浏览器](reports/comparisons_20261117/index.html)（用浏览器打开本地 HTML）。
可自由切换 7 个方案、整页/局部、缩放及同步滚动，也可记录逐页判断并导出 CSV。
局部按谱线间距等比放大，不额外纠斜或拉伸；判断内容保留时需要同时检查整页。
报告目录入口见 [reports/README.md](reports/README.md)。
