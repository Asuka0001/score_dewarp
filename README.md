# score_dewarp — 五线谱 / 简谱扭曲照片的平直化校正（v0.1）

输入一张手机拍摄的乐谱照片（透视、卷曲、旋转、阴影、遮挡；五线谱 / 简谱 / 混排），输出：

- **校正后的彩色图像** + **光照归一化增强图**（便于 OMR / 阅读）
- **谱种标签**：`staff` / `jianpu` / `mixed` / `unknown`
- **版面区域**：五线谱谱表区、简谱音符行、文本 / 歌词行（多边形）
- **结构关键点**：谱线折线（含谱表内线号）、简谱数字框（含识别出的 0–7）、文字行框
- **变换场**：输出像素 → 原图像素的 backward map（`*_bmap.npy`）

v0.1 是一套**以结构先验为核心的几何优化方案**，可以直接运行、不依赖训练数据。另外附了一个多任务网络骨架，以及配套的合成数据和训练脚本，作为下一步“学习 + 优化”混合方案的起点。

---

## 1. 快速开始

```bash
cd score_dewarp
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 校正自己的照片
.venv/bin/python scripts/dewarp.py my_score.jpg --out outputs/run

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

## 5. 学习模型（v0 骨架）

`learn/model.py` 是一个共享 U-Net，带 4 个输出头：

- **fwd**：逐像素的“照片 → 平面页坐标”（DocUNet 风格的稠密坐标场）+ 页面掩码。推理时用散点反演得到 BackwardMap，与几何方案共用同一套接口。
- **seg**：版面 5 类（页外 / 纸 / 五线谱区 / 简谱区 / 文本歌词区）。
- **struct**：谱线、简谱基线、竖线三类热图。它们可以直接替换 `structures.py` 里手工设计的检测器，喂给 B/C 优化（计划中的混合方案）。
- **kind**：谱种分类。

损失：坐标 L1 + 梯度一致性（局部畸变）+ CE/BCE，外加按谱种设计的**结构一致性损失**：同一条谱线或简谱基线实例上，预测的平面 Y 方差应为 0；同一条小节线/符干实例上，预测的平面 X 方差应为 0。这样第 4 节的先验以可微的形式进入了训练。

当前状态：数据集、模型、损失、训练（MPS/CUDA/CPU）、推理都已打通。
已用 Doc3D 第 8 分片完成 **30 轮几何预训练**，并建立两个验证集上的传统几何基线；配置和结果见第 9 节。
谱种、版面与结构头尚未正式训练。五线谱小规模微调及预训练对照已完成（见第 10 节）；真实手机照片尚未评测。

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
完整结果见[实验报告](reports/geometry_pretrain.md)，训练曲线见[best.curves.png](outputs/doc3d_pretrain/best.curves.png)。

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
[reports/deepscores_finetune.md](reports/deepscores_finetune.md)（由 `scripts/report_finetune.py` 生成）。

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
多 seed 报告见 [reports/deepscores_multiseed.md](reports/deepscores_multiseed.md)。
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
.venv/bin/python scripts/report_hybrid.py --eval outputs/hybrid/pilot_val --out reports/hybrid_pilot.md
```

混合评测使用预测页面掩码，并检查输出画布边界及 2 px 原图往返误差，避免映射反演跨越空洞时虚增覆盖率。
`summary.json` 额外记录 network_bmap / network+C 在每页共同有效点上的 `common_ad_pct`、`common_local_distortion_pct` 和 `common_coverage`，
用于区分真实精修收益与有效点减少造成的误差下降。四页冒烟结果仅用于确认流程可运行，不作为效果结论。

### 首轮网络+C 验证结果（pilot val，64 页）

完整结果见 [reports/hybrid_pilot.md](reports/hybrid_pilot.md)。本轮 64 页均完成，无异常失败；19 项单元测试通过。

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
完整表格见 [reports/deepscores_multiseed.md](reports/deepscores_multiseed.md)。

### 全量训练结果与 test 评测（2026-11-12）

全量模型 `full_ca_lr1e-3_s7`：全量 val EPE 19.05% → **0.54%**（第 53 轮，约 7 h）；pilot 模型是 2.61%。
为了和基线及 pilot 模型逐页配对，全量模型另外在 64 页 pilot val/test 上评测（照片与全量数据中的对应页逐字节相同，
与全量 train 无曲谱分组重叠）。此外，在 137 页全量 test 上补跑了传统基线和 pilot 模型。
test 只评测一次，模型和 C 的参数都已按验证集固定。完整表格和检验见 [reports/deepscores_full.md](reports/deepscores_full.md)
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
