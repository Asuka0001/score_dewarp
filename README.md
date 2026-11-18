# score_dewarp — 乐谱照片平直化校正（五线谱 / 简谱 / 混排）

输入一张手机拍摄的乐谱照片（透视、卷曲、褶皱、阴影），输出：

- **校正后的彩色图**（`_rectified.png`）和**阅读副本**（`_enhanced.png`：去阴影、纸面置白、纸外置白；`--binary` 另存黑白版）
- **谱种标签**：`staff` / `jianpu` / `mixed` / `unknown`
- **结构**：谱线折线（含线号）、简谱数字框（含 0–7 识别）、文字行框、版面区域（`_result.json`）
- **变换场**：输出像素 → 原图像素的 backward map（`_bmap.npy`）

不需要训练数据：页面形变由检测到的乐谱结构（谱线、简谱数字行、竖线、谱表端点）反求，全部是几何优化。

## 快速开始

```bash
cd score_dewarp
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 校正照片（可一次传多张）
.venv/bin/python scripts/dewarp.py my_score.jpg --out outputs/run --binary
```

iPhone 的 HEIC 照片需先转成 JPEG（OpenCV 不读 HEIC），macOS 上：`sips -s format jpeg IMG_0001.HEIC --out IMG_0001.jpg`。

其他选项：`--no-field` 只用曲面模型、不跑残差场；`--crop-content` 只输出内容框；`--method traditional` 是旧 ABC 方案，仅作对照。单张 12 MP 照片约 10 秒（Apple Silicon，单进程）。

## 方法

```
照片 ─► A 页面单应 + 纠斜 ─► 结构检测 ─► B 针孔相机 + 可展柱面拟合 ─► 重新检测 ─► C 残差场 ─► 输出范围 / 页边距 / 阅读副本
```

- **A**（`core/page.py`）：检测页面四边形做透视矫正，再用投影轮廓纠斜；检测不到四边形时只纠斜。
- **结构检测**（`core/structures.py`）：游程统计估计线粗和线距，逐列跟踪谱线并聚成五线谱表；简谱按字高把连通域链接成行，用模板识别 0–7；竖线（小节线、符干）只取细的部分，去掉符头 / 符尾连接处，否则符干整体偏约 1°。
- **B**（`core/gcs.py`）：纸张弯曲但不拉伸，页面用一般柱面（可展曲面）表示，加针孔相机。未知量是相机位姿、母线方向、截面斜率曲线和每个结构的平面坐标；约束是重投影误差加各谱种的几何关系（见下表）。曲面是等距展开的，所以输出的比例是度量正确的，不需要再估计长宽比。
- **C**（`core/field.py`）：在 B 的输出上重新检测，解一个网格上的前向场 `(U, V)`。所有先验都写成关于 U 或 V 的线性约束：同一谱线 V 相等、相邻谱线差一个线距、竖线 U 相等、各行谱表的左右端 U 相等（页边对齐），加平滑项，用 IRLS 求稀疏最小二乘。全局比例不变（`keep_scale`）。
- **输出**（`core/metric_pipeline.py`）：纸张边缘（GrabCut 掩码 + 纸面直线检验）限定范围，范围再扩到装下所有印刷内容，四周补 3% 白边；阅读副本只保留纸面。各阶段都表示成 `BackwardMap`，只对原图重采样一次。

| | 五线谱 | 简谱 |
|---|---|---|
| 主要线索 | 贯穿全宽、5 线等距的长细线；小节线、符干；谱表左右端对齐 | 没有长线：数字 0–7 字高一致（顶线 + 底线双约束）；小节线、数字和汉字的竖笔画 |
| B 中的先验 | 每个谱表一组 `Y0, s`：第 i 线 `Y = Y0 + i·s`，各谱表的 `s` 软约束一致；各行谱表的左右端共用一个 X | 每行一组 `Y_top, h`：数字顶点在 `Y_top`、底点在 `Y_top + h`；各行 `h` 软约束一致 |
| C 中的先验 | V：沿线相等、相邻线差 = 线距；U：沿线弧长按局部线距归一；谱表端点 U 相等 | V：顶线、底线各自相等，底 − 顶 = 字高；U：相邻字间距按局部字高归一 |

混排页面上两类约束同时作用于同一个曲面和残差场。

## 评测

基准用 Doc3D 的真实纸张形变（UV 图）给乐谱页重新贴图（`data/bench/doc3d_score`，dev / test 各 120 页，五线谱、简谱、混排，轻 / 中 / 重三档），所有指标都来自真值，不经过我们的检测器。

```bash
.venv/bin/python scripts/download_datasets.py deepscores   # 以及 doc3d；CPMS 真实照片用 download_cpms.py
.venv/bin/python scripts/make_doc3d_bench.py --out data/bench/doc3d_score --per-split 120
.venv/bin/python scripts/bench.py --data data/bench/doc3d_score/dev --methods input abc metric --out outputs/bench/dev
.venv/bin/python scripts/bench.py --data data/bench/doc3d_score/dev --methods metric "gcs:c.ends_w=4" --out outputs/bench/x   # 参数变体
.venv/bin/python scripts/make_compare_figures.py --out reports/compare_xxx                                                  # 对比图
```

dev 中位数（2026-11-18，当前指标定义）：

| 指标 | 原图 | 旧 ABC | 主线 |
|---|---:|---:|---:|
| 内容区 AD（页宽 %） | 2.44 | 1.77 | **0.88** |
| 内容区 LD（局部畸变 %） | 13.7 | 11.9 | **6.9** |
| 可见页面保留比例 | 1.000 | 0.881 | 0.998 |
| 谱线平直度（y 标准差 / 线距） | 1.30 | 0.71 | **0.15** |
| 竖线倾角（°） | 2.74 | 2.50 | **0.75** |
| 简谱基线平直度（/ 字高） | 0.51 | 0.24 | **0.084** |
| 页边倾斜（°） | 2.26 | 1.95 | **0.54** |
| 首末行宽度差（%） | 7.6 | 5.0 | **1.6** |

结构指标在整页对齐之后测量（Doc3D 的页面长宽比是任意的，对齐分别拟合 X、Y 尺度），因此一个把页面压窄的方法不能靠此降低角度指标。test 集只在 2026-11-17 评过一次，当时用的是旧指标定义，结果见报告。CPMS 真实照片（无真值）只用于目视检查。

## 目录

```
scoredewarp/
  core/   page.py (A)  structures.py (检测)  gcs.py (B)  field.py (C)  metric_pipeline.py (主线)
          pipeline.py (旧 ABC 与共用的收尾)  surface3d.py / line_mesh.py (旧 ABC 用)
          preprocess.py (光照、二值化、纸面、阅读副本)  transforms.py (BackwardMap)
  eval/   geometry.py  metrics.py  structures.py  real_score.py      评测指标
  synth/  render.py  fonts.py  doc3d_retexture.py                   渲染平整乐谱页、Doc3D 重贴图
scripts/  dewarp.py (CLI)  bench.py  bench_report.py  make_doc3d_bench.py
          make_compare_figures.py  make_route_figures.py  make_align_figures.py
          download_datasets.py  download_cpms.py  clean_cpms_mosaics.py
tests/    python -m unittest discover tests
reports/  metric_route_20261117.md（主线报告，补充一至五）与对比图
```

## 已知局限

1. **两页摊开、背景杂乱的照片**（2026-11-18 的手持拍书测试）：主页面能展平，但输出里会留下对面那页、桌面和手，阅读副本把背景变成灰色纹理。需要只保留主页面。
2. **对面那页不会被展平**：曲面模型一次只表示一张纸。
3. **严重褶皱页上曲面拟合会陷入局部解**（如 dev 0004）：输入稍有变化结果就可能差很多，需要多起点拟合。
4. **方向**：90° 检测默认关闭，180° 未处理。

## 历史

- 主线的设计、调参和每一轮改动的评测：[reports/metric_route_20261117.md](reports/metric_route_20261117.md)。
- 2026-11-17 之前尝试过的方案（多任务网络 UV、UVDoc、网络 + 优化混合、合成数据训练、辅助头）已放弃，代码于 2026-11-18 从仓库删除，需要时用 git 标签取回：`git checkout archive/pre-cleanup-20261118`；它们的报告在提交 `9acb325` 中（`git checkout 9acb325 -- reports/`）。
